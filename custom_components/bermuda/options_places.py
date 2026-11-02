"""
"Places" (micro-locations) as a guided screen in the options flow.

A place is a named spot such as "Kitchen table" or "Sofa". Put any device (a phone is fine) where
the spot is, keep it still while the progress screen counts down (so the reading comes from a
settled device, not a stray advert), and Bermuda saves the signal pattern every proxy hears there.
Places belong to the house: from then on EVERY tracked device is matched against them, whatever its
transmit power, and its *micro-location* sensor says which place it is at (or just its room).
Matching uses raw signal strength, so it does not depend on the distance calibration.

The saving itself is the coordinator's ``calibrate_location`` (the same thing the
``bermuda.calibrate_location`` service does); this screen only waits for a settled reading and reports.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.config_entries import OptionsFlow
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from .calibration import CHUNK_SECONDS
from .device_kind import device_label, ha_device_name
from .room_learning import device_live_rssi

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowResult

    from .bermuda_device import BermudaDevice
    from .coordinator import BermudaDataUpdateCoordinator

PLACE_SETTLE_SECONDS = 15  # how long the device must sit still before its place is recorded


class PlacesFlowMixin(OptionsFlow):
    """Places steps; the options-flow handler subclasses this and provides the helpers used here."""

    if TYPE_CHECKING:

        def _candidates(self, *, include_tracked: bool) -> list[BermudaDevice]:
            """Provided by the handler."""
            ...

        @staticmethod
        def _list_selector(options: list[SelectOptionDict], *, multiple: bool = True) -> SelectSelector:
            """Provided by the handler."""
            ...

        async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
            """Provided by the handler (the main menu)."""
            ...

    _place: dict[str, Any] | None = None
    _place_task: asyncio.Task[None] | None = None
    _place_started: float = 0.0

    @property
    def _place_coordinator(self) -> BermudaDataUpdateCoordinator:
        return cast("BermudaDataUpdateCoordinator", self.config_entry.runtime_data.coordinator)

    def _place_device(self) -> BermudaDevice | None:
        assert self._place is not None  # noqa: S101
        return self._place_coordinator.devices.get(self._place["device"])

    def _place_device_name(self) -> str:
        device = self._place_device()
        return ha_device_name(self.hass, device) if device else (self._place or {}).get("device", "")

    def _place_room_name(self) -> str:
        assert self._place is not None  # noqa: S101
        coordinator = self._place_coordinator
        area_id = self._place["area_id"]
        if not area_id:
            device = self._place_device()
            return (device.area_name if device else None) or "the room it is in"
        return coordinator.resolve_area_name(area_id) or area_id

    def _place_existing_text(self) -> str:
        """One line per place, for the menu description."""
        coordinator = self._place_coordinator
        places = sorted(coordinator.fingerprints.list(), key=lambda f: f.name.casefold())
        if not places:
            return "No places are marked yet."
        lines = []
        for fingerprint in places:
            room = coordinator.resolve_area_name(fingerprint.area_id) if fingerprint.area_id else None
            lines.append(f"- **{fingerprint.name}**" + (f" ({room})" if room else ""))
        return "Marked so far\n" + "\n".join(lines)

    # --- menu ---------------------------------------------------------------

    async def async_step_places(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Places menu: mark one, or forget some."""
        return self.async_show_menu(
            step_id="places",
            menu_options=["place_mark", "place_forget"],
            description_placeholders={"existing": self._place_existing_text()},
        )

    # --- mark ---------------------------------------------------------------

    async def async_step_place_mark(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Mark a place, step 1: which device, what to call the place, and which room."""
        coordinator = self._place_coordinator
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input.get("name") or "").strip()
            device = coordinator.devices.get(str(user_input["device"]).lower())
            if device is None or not name:
                errors["base"] = "place_cannot_start"
            else:
                self._place = {
                    "device": device.address,
                    "name": name,
                    "area_id": user_input.get("room") or None,
                    "result": None,
                    "error": None,
                }
                self._place_started = time.monotonic()
                self._place_task = None
                return await self.async_step_place_run()
        devices = [d for d in self._candidates(include_tracked=True) if d.create_sensor and not d.is_scanner]
        if not devices:
            return self.async_abort(reason="no_tracked_devices")
        areas = sorted(coordinator.ar.async_list_areas(), key=lambda a: a.name)
        return self.async_show_form(
            step_id="place_mark",
            data_schema=vol.Schema(
                {
                    vol.Required("device"): self._list_selector(
                        [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices],
                        multiple=False,
                    ),
                    vol.Required("name"): TextSelector(),
                    vol.Optional("room"): SelectSelector(
                        SelectSelectorConfig(
                            options=[SelectOptionDict(value=a.id, label=a.name) for a in areas],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
            description_placeholders={"existing": self._place_existing_text()},
            errors=errors or None,
        )

    def _place_progress_text(self) -> dict[str, str]:
        assert self._place is not None  # noqa: S101
        device = self._place_device()
        heard = len(device_live_rssi(device)) if device else 0
        left = max(0, round(PLACE_SETTLE_SECONDS - (time.monotonic() - self._place_started)))
        return {
            "device": self._place_device_name(),
            "place": self._place["name"],
            "heard": str(heard),
            "left": str(left),
        }

    async def async_step_place_run(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Wait for the device to settle, refreshing every few seconds."""
        return await self._place_chunk("place_run", "place_listen")

    async def async_step_place_listen(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The other half of the loop: Home Assistant only refreshes the screen when the next step differs."""
        return await self._place_chunk("place_listen", "place_run")

    async def _place_chunk(self, this_step: str, other_step: str) -> ConfigFlowResult:
        assert self._place is not None  # noqa: S101
        if self._place_task is None:
            self._place_task = self.hass.async_create_task(asyncio.sleep(CHUNK_SECONDS))
        if not self._place_task.done():
            return self.async_show_progress(
                step_id=this_step,
                progress_action="placing",
                progress_task=self._place_task,
                description_placeholders=self._place_progress_text(),
            )
        self._place_task = None
        if time.monotonic() - self._place_started >= PLACE_SETTLE_SECONDS:
            return self.async_show_progress_done(next_step_id="place_done")
        return self.async_show_progress_done(next_step_id=other_step)

    async def async_step_place_done(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Record the place now that the device has settled, and say how it went."""
        assert self._place is not None  # noqa: S101
        if user_input is not None:
            self._place = None
            return await self.async_step_init()
        if self._place["result"] is None and self._place["error"] is None:
            device = self._place_device()
            try:
                if device is None:
                    msg = "That device is no longer visible."
                    raise ServiceValidationError(msg)  # noqa: TRY301
                self._place["result"] = self._place_coordinator.calibrate_location(
                    device, self._place["name"], self._place["area_id"]
                )
            except ServiceValidationError as err:
                self._place["error"] = str(err)
        result, error = self._place["result"], self._place["error"]
        name, who = self._place["name"], self._place_device_name()
        if error:
            outcome = f"Nothing was saved. {error}"
        else:
            room = (result or {}).get("area_name") or self._place_room_name()
            count = (result or {}).get("scanner_count", 0)
            verb = "Updated" if (result or {}).get("replaced") else "Saved"
            readings = (result or {}).get("samples", 0)
            outcome = (
                f"{verb} **{name}** in **{room}**, using **{who}**: heard by {count} proxies ({readings} readings). "
                "Every tracked device is matched against it."
            )
            for key in ("warning", "readings_warning"):
                if warning := (result or {}).get(key):
                    outcome += f"\n\n{warning}"
        return self.async_show_form(
            step_id="place_done", data_schema=vol.Schema({}), description_placeholders={"outcome": outcome}
        )

    # --- forget -------------------------------------------------------------

    async def async_step_place_forget(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Tick the places to forget."""
        coordinator = self._place_coordinator
        fingerprints = coordinator.fingerprints.list()
        if not fingerprints:
            return self.async_abort(reason="no_places")
        if user_input is not None:
            for fingerprint_id in user_input.get("places", []):
                coordinator.forget_location(fingerprint_id)
            return await self.async_step_init()
        options = []
        for fingerprint in sorted(fingerprints, key=lambda f: f.name.casefold()):
            room = coordinator.resolve_area_name(fingerprint.area_id) if fingerprint.area_id else None
            label = fingerprint.name + (f" ({room})" if room else "")
            options.append(SelectOptionDict(value=fingerprint.id, label=label))
        return self.async_show_form(
            step_id="place_forget",
            data_schema=vol.Schema({vol.Optional("places"): self._list_selector(options)}),
        )
