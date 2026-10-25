"""
Confidence-triggered room learning, mixed into the coordinator.

Each update cycle, for every tracked device, this scores how well its room is
being recognised (see ``room_learning``). Devices recognised well are left
alone. When a device's confidence stays poor for a while, a Repairs prompt asks
the user whether to train it; training captures per-scanner RSSI while the
device sits in a room the user names. From then on, whenever that device's
confidence is poor, its live readings are compared with the trained rooms and a
clear match overrides the distance-picked room.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import voluptuous as vol
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import slugify

from .const import (
    _LOGGER,
    CONF_LEARNING_ENABLED,
    CONF_LEARNING_SUSTAINED,
    CONF_LEARNING_THRESHOLD,
    DEFAULT_LEARNING_ENABLED,
    DEFAULT_LEARNING_SUSTAINED,
    DEFAULT_LEARNING_THRESHOLD,
    DOMAIN,
    LEARN_INTERVAL,
    LEARN_MAX_SECONDS,
    LEARN_MIN_SAMPLES,
    LEARN_RECOVERY_BONUS,
    LEARN_SAMPLES,
    REPAIR_TRAIN_PREFIX,
)
from .device_kind import device_kind, ha_device_name
from .room_learning import ConfidenceResult, ConfidenceTracker, ProfileStore, RoomProfile, match_rooms

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
    from homeassistant.helpers.area_registry import AreaRegistry

    from . import BermudaConfigEntry
    from .bermuda_device import BermudaDevice


@dataclass
class _TrainingRun:
    """One in-progress training session for a device."""

    area_id: str
    started: float
    profile: RoomProfile
    last_sample: float
    last_stamp: float = 0.0


class BermudaLearningMixin:
    """Score room recognition, prompt for training when it is poor, and apply trained rooms."""

    # Class-level default so a coordinator that skipped __init__ (as some tests build) stays inert.
    _learning_ready: bool = False

    if TYPE_CHECKING:
        # Provided by BermudaDataUpdateCoordinator / BermudaMicrolocationMixin.
        hass: HomeAssistant
        ar: AreaRegistry
        options: dict[str, Any]
        devices: dict[str, BermudaDevice]

        def resolve_area_name(self, area_id: str) -> str | None:
            """Declared for mypy; implemented by BermudaDataUpdateCoordinator."""
            ...

        def _require_device(self, value: str) -> BermudaDevice:
            """Declared for mypy; implemented by BermudaMicrolocationMixin."""
            ...

        def _resolve_area_id(self, value: str | None) -> str | None:
            """Declared for mypy; implemented by BermudaMicrolocationMixin."""
            ...

    def _learning_init(self, hass: HomeAssistant, entry: BermudaConfigEntry) -> None:
        """Set up the profile store (loaded in the background), bookkeeping and services."""
        self.room_profiles = ProfileStore(hass)
        self._confidence: dict[str, ConfidenceTracker] = {}
        self._low_since: dict[str, float] = {}
        self._train_prompts: set[str] = set()  # device keys with an open "train it?" prompt
        self._training: dict[str, _TrainingRun] = {}
        self.training_room: dict[str, str] = {}  # device address -> area id picked on the select entity
        entry.async_create_background_task(
            hass, self.room_profiles.async_load(), "Load Bermuda room profiles", eager_start=True
        )
        self._register_learning_services(hass, entry)
        self._learning_ready = True

    # --- options ------------------------------------------------------------

    @property
    def learning_enabled(self) -> bool:
        """Whether confidence monitoring, prompts and trained-room matching are on."""
        return bool(self.options.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED))

    def _learning_threshold(self) -> float:
        return float(self.options.get(CONF_LEARNING_THRESHOLD, DEFAULT_LEARNING_THRESHOLD))

    def _learning_sustained(self) -> float:
        return float(self.options.get(CONF_LEARNING_SUSTAINED, DEFAULT_LEARNING_SUSTAINED))

    # --- per-cycle ----------------------------------------------------------

    def _refresh_learning(self) -> None:
        """Run after area selection each cycle: monitor, train, and apply trained rooms."""
        if not self._learning_ready or not self.room_profiles.loaded or not self.learning_enabled:
            return
        now = monotonic_time_coarse()
        threshold = self._learning_threshold()
        for device in self.devices.values():
            if not device.create_sensor or device.is_scanner:
                continue
            key = device.address
            live = self._learning_live_rssi(device)
            self._learning_advance_training(device, live, now)

            tracker = self._confidence.setdefault(key, ConfidenceTracker())
            tracker.add(now, device.area_id, device.floor_id, self._learning_margin(device))
            result = tracker.result(now)

            if profiles := self.room_profiles.get(key):
                # Trained devices don't need prompting; trained rooms only step in when
                # plain distance is doubtful, so a well-recognised device is never overridden.
                self._withdraw_train_prompt(key)
                if result.score < threshold:
                    self._apply_trained_room(device, live, profiles)
            else:
                self._monitor_confidence(device, result, threshold, now)

    @staticmethod
    def _learning_live_rssi(device: BermudaDevice) -> dict[str, float]:
        """Strongest current RSSI from each scanner that can hear the device right now."""
        live: dict[str, float] = {}
        for advert in device.adverts.values():
            if advert.rssi_distance is None or advert.rssi is None:
                continue
            if advert.rssi > live.get(advert.scanner_address, float("-inf")):
                live[advert.scanner_address] = advert.rssi
        return live

    @staticmethod
    def _learning_margin(device: BermudaDevice) -> float | None:
        """How much farther the nearest scanner in a *different* area is, as a fraction of the winner's distance."""
        heard = sorted(
            (a.rssi_distance, a.area_id) for a in device.adverts.values() if a.rssi_distance is not None and a.area_id
        )
        if not heard:
            return None
        best_distance, best_area = heard[0]
        for distance, area_id in heard[1:]:
            if area_id != best_area:
                return (distance - best_distance) / max(best_distance, 0.1)
        return None

    # --- prompting ----------------------------------------------------------

    def _monitor_confidence(
        self, device: BermudaDevice, result: ConfidenceResult, threshold: float, now: float
    ) -> None:
        """Raise a "train it?" prompt once confidence has been poor long enough; withdraw it on recovery."""
        key = device.address
        if result.score < threshold:
            since = self._low_since.setdefault(key, now)
            if now - since >= self._learning_sustained() and key not in self._train_prompts:
                self._raise_train_prompt(device, result)
        else:
            self._low_since.pop(key, None)
            if result.score >= threshold + LEARN_RECOVERY_BONUS:
                self._withdraw_train_prompt(key)

    def _raise_train_prompt(self, device: BermudaDevice, result: ConfidenceResult) -> None:
        """Create the Repairs issue (a dismissed issue stays dismissed, so this can't nag)."""
        names = [self.resolve_area_name(a) or a for a in result.top_areas]
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            f"{REPAIR_TRAIN_PREFIX}{slugify(device.address)}",
            is_fixable=True,
            is_persistent=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="room_confidence_low",
            translation_placeholders={
                "device": ha_device_name(self.hass, device),
                "kind": device_kind(device),
                "areas": " and ".join(names) if names else "several rooms",
            },
            data={"device_key": device.address},
        )
        self._train_prompts.add(device.address)
        _LOGGER.debug("Room confidence for %s is poor (%s); offered training", device.name, result)

    def _withdraw_train_prompt(self, key: str, *, force: bool = False) -> None:
        """
        Remove a device's prompt if one is open.

        A prompt the user dismissed is left in the issue registry (marked dismissed) so that it
        stays dismissed if confidence drops again; ``force`` (the user acted on the device) deletes it.
        """
        if key not in self._train_prompts:
            return
        self._train_prompts.discard(key)
        issue_id = f"{REPAIR_TRAIN_PREFIX}{slugify(key)}"
        issue = ir.async_get(self.hass).async_get_issue(DOMAIN, issue_id)
        if force or issue is None or issue.dismissed_version is None:
            ir.async_delete_issue(self.hass, DOMAIN, issue_id)

    # --- applying trained rooms ---------------------------------------------

    @staticmethod
    def _apply_trained_room(
        device: BermudaDevice, live: Mapping[str, float], profiles: Mapping[str, RoomProfile]
    ) -> None:
        """Override the distance-picked room when live readings clearly match a different trained room."""
        match = match_rooms(live, profiles)
        if match.area_id is None or match.area_id == device.area_id:
            return
        in_room = [a.rssi_distance for a in device.adverts.values() if a.area_id == match.area_id and a.rssi_distance]
        distance = min(in_room) if in_room else (device.area_distance or 0.0)
        _LOGGER.debug("Trained room match moved %s %s -> %s", device.name, device.area_name, match.area_id)
        device.apply_area_override(match.area_id, distance)

    # --- training -----------------------------------------------------------

    def start_training(self, device: BermudaDevice, area_id: str) -> dict[str, Any]:
        """
        Begin capturing a training run: the device must stay in ``area_id`` until it completes.

        Samples are taken every LEARN_INTERVAL seconds from fresh adverts, up to
        LEARN_SAMPLES or LEARN_MAX_SECONDS. Re-training a room replaces it.
        """
        if self.ar.async_get_area(area_id) is None:
            msg = f"No Home Assistant area with id '{area_id}'."
            raise ServiceValidationError(msg)
        if not device.create_sensor or device.is_scanner:
            msg = f"{device.name} isn't a tracked device, so it can't be trained."
            raise ServiceValidationError(msg)
        now = monotonic_time_coarse()
        self._training[device.address] = _TrainingRun(
            area_id, now, RoomProfile(area_id), last_sample=now - LEARN_INTERVAL
        )
        self._withdraw_train_prompt(device.address, force=True)
        return self.training_status(device.address)

    def training_status(self, address: str) -> dict[str, Any]:
        """Describe a device's training state, for service responses and diagnostics."""
        run = self._training.get(address)
        return {
            "device_address": address,
            "training": run is not None,
            "area_id": run.area_id if run else None,
            "samples": run.profile.samples if run else 0,
            "samples_wanted": LEARN_SAMPLES,
            "trained_rooms": sorted(self.room_profiles.get(address)),
        }

    def _learning_advance_training(self, device: BermudaDevice, live: Mapping[str, float], now: float) -> None:
        """Take a sample if one is due, and finish the run when it has enough or has run out of time."""
        run = self._training.get(device.address)
        if run is None:
            return
        if live and now - run.last_sample >= LEARN_INTERVAL:
            freshest = max((a.stamp or 0.0) for a in device.adverts.values() if a.rssi_distance is not None)
            if freshest > run.last_stamp:  # only genuinely new adverts, never the same one twice
                run.profile.add_sample(live)
                run.last_sample, run.last_stamp = now, freshest
        if run.profile.samples >= LEARN_SAMPLES or now - run.started > LEARN_MAX_SECONDS:
            self._finish_training(device, run)

    def _finish_training(self, device: BermudaDevice, run: _TrainingRun) -> None:
        """Keep the run if it gathered enough samples, otherwise discard it."""
        del self._training[device.address]
        if run.profile.samples < LEARN_MIN_SAMPLES:
            _LOGGER.warning(
                "Training %s for %s gathered only %d samples; discarded. Keep it in the room and try again.",
                device.name,
                self.resolve_area_name(run.area_id) or run.area_id,
                run.profile.samples,
            )
            return
        self.room_profiles.put(device.address, run.profile)
        self._confidence.pop(device.address, None)
        self._low_since.pop(device.address, None)
        _LOGGER.info(
            "Trained %s for %s from %d samples",
            device.name,
            self.resolve_area_name(run.area_id) or run.area_id,
            run.profile.samples,
        )

    def reset_learning(self, device: BermudaDevice) -> bool:
        """Forget everything trained for a device (and any run in progress). Returns whether anything was trained."""
        self._training.pop(device.address, None)
        self._confidence.pop(device.address, None)
        self._low_since.pop(device.address, None)
        self._withdraw_train_prompt(device.address, force=True)
        return self.room_profiles.reset(device.address)

    def learning_diagnostics(self) -> dict[str, Any]:
        """Learning state for the diagnostics download."""
        return {
            "enabled": self.learning_enabled,
            "trained_devices": len(self.room_profiles.device_keys()),
            "open_prompts": len(self._train_prompts),
            "training": len(self._training),
        }

    # --- services -----------------------------------------------------------

    def _register_learning_services(self, hass: HomeAssistant, entry: BermudaConfigEntry) -> None:
        """Register learn_room / reset_learning (the same actions the buttons and Repairs flow use)."""
        text = cv.string
        services = [
            (
                "learn_room",
                self.service_learn_room,
                vol.Schema({vol.Required("device"): text, vol.Required("area"): text}),
            ),
            ("reset_learning", self.service_reset_learning, vol.Schema({vol.Required("device"): text})),
        ]
        for name, handler, schema in services:
            hass.services.async_register(DOMAIN, name, handler, schema, SupportsResponse.OPTIONAL)

        def _unregister() -> None:
            for name, _handler, _schema in services:
                hass.services.async_remove(DOMAIN, name)

        entry.async_on_unload(_unregister)

    async def service_learn_room(self, call: ServiceCall) -> ServiceResponse:
        """Start training a device in the named room (it must stay there for a few minutes)."""
        device = self._require_device(call.data["device"])
        area_id = self._resolve_area_id(call.data["area"])
        if area_id is None:  # pragma: no cover - area is a required, non-empty field
            msg = "Please say which room the device is in."
            raise ServiceValidationError(msg)
        return cast("ServiceResponse", self.start_training(device, area_id))

    async def service_reset_learning(self, call: ServiceCall) -> ServiceResponse:
        """Forget everything trained for a device."""
        device = self._require_device(call.data["device"])
        return cast("ServiceResponse", {"device": device.name, "reset": self.reset_learning(device)})
