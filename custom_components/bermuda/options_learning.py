"""
"Learn a room" as a guided screen in the options flow, with a live countdown like Calibrate.

Pick the phone and the room it is in, then keep it there while the progress screen shows the
readings collected and the time left. What is learned belongs to the whole house (every tracked
device is matched against it), so any phone can be used to teach a room, and a room can be taught
from several spots. The training itself is the coordinator's (the same one
``bermuda.learn_room`` and the Repairs prompt start); this only watches it and reports the result.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.config_entries import OptionsFlow
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from .calibration import CHUNK_SECONDS
from .const import LEARN_INTERVAL, LEARN_MAX_SECONDS
from .device_kind import device_label, ha_device_name

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowResult

    from .bermuda_device import BermudaDevice
    from .coordinator import BermudaDataUpdateCoordinator


class LearnFlowMixin(OptionsFlow):
    """Learn-a-room steps; the options-flow handler subclasses this and provides the helpers used here."""

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

    _learn: dict[str, Any] | None = None
    _learn_task: asyncio.Task[None] | None = None
    _learn_started: float = 0.0

    @property
    def _learn_coordinator(self) -> BermudaDataUpdateCoordinator:
        return cast("BermudaDataUpdateCoordinator", self.config_entry.runtime_data.coordinator)

    def _learn_room_name(self) -> str:
        assert self._learn is not None  # noqa: S101
        if not self._learn["area_id"]:
            return ""
        return self._learn_coordinator.resolve_area_name(self._learn["area_id"]) or self._learn["area_id"]

    def _learn_device_name(self) -> str:
        assert self._learn is not None  # noqa: S101
        device = self._learn_coordinator.devices.get(self._learn["device"])
        return ha_device_name(self.hass, device) if device else self._learn["device"]

    async def async_step_learn(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Learn a room, step 1: which phone, and which room is it in."""
        coordinator = self._learn_coordinator
        errors: dict[str, str] = {}
        if user_input is not None:
            device = coordinator.devices.get(user_input["device"].lower())
            room = user_input.get("room")
            forget_all = bool(user_input.get("forget_all"))
            forget = bool(user_input.get("forget")) or forget_all
            label = str(user_input.get("label") or "").strip()
            self._learn = {
                "device": user_input["device"].lower(),
                "area_id": room,
                "forget": forget,
                "forget_all": forget_all,
                "label": label,
            }
            try:
                if forget:
                    if not forget_all and not room:
                        msg = "Pick the room to forget, or tick forget everything."
                        raise ServiceValidationError(msg)  # noqa: TRY301
                    area = None if forget_all else room
                    self._learn["forgot"] = coordinator.reset_learning(area, None if forget_all else (label or None))
                    return await self.async_step_learn_done()
                if device is None or not room:
                    msg = "Pick a visible device and a room."
                    raise ServiceValidationError(msg)  # noqa: TRY301
                coordinator.start_training(device, room, replace=bool(user_input.get("replace")), label=label)
            except ServiceValidationError:
                errors["base"] = "learn_cannot_start"
            else:
                self._learn_started = time.monotonic()
                return await self.async_step_learn_run()
        devices = [d for d in self._candidates(include_tracked=True) if d.create_sensor and not d.is_scanner]
        if not devices:
            return self.async_abort(reason="no_tracked_devices")
        areas = sorted(coordinator.ar.async_list_areas(), key=lambda a: a.name)
        return self.async_show_form(
            step_id="learn",
            data_schema=vol.Schema(
                {
                    vol.Required("device"): self._list_selector(
                        [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices],
                        multiple=False,
                    ),
                    vol.Optional("room"): SelectSelector(
                        SelectSelectorConfig(
                            options=[SelectOptionDict(value=a.id, label=a.name) for a in areas],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("label"): TextSelector(),
                    vol.Optional("replace", default=False): BooleanSelector(),
                    vol.Optional("forget", default=False): BooleanSelector(),
                    vol.Optional("forget_all", default=False): BooleanSelector(),
                }
            ),
            description_placeholders={"trained": self._learn_trained_summary()},
            errors=errors or None,
        )

    def _learn_trained_summary(self) -> str:
        """What is already learned, one line per room, for the form's description."""
        coordinator = self._learn_coordinator
        rooms = coordinator.room_profiles.labels()
        if not rooms:
            return "Nothing is learned yet."
        lines = [
            f"- **{coordinator.resolve_area_name(area_id) or area_id}**: {', '.join(labels)}"
            for area_id, labels in sorted(rooms.items(), key=lambda kv: coordinator.resolve_area_name(kv[0]) or kv[0])
        ]
        return "Already learned (room: spots)\n" + "\n".join(lines)

    def _learn_spot_text(self) -> str:
        assert self._learn is not None  # noqa: S101
        label = self._learn.get("label") or ""
        return f" ({label})" if label else ""

    def _learn_progress_text(self) -> dict[str, str]:
        assert self._learn is not None  # noqa: S101
        status = self._learn_coordinator.training_status(self._learn["device"])
        samples, wanted = status["samples"], status["samples_wanted"]
        elapsed = time.monotonic() - self._learn_started
        left = max(0, round(min((wanted - samples) * LEARN_INTERVAL, LEARN_MAX_SECONDS - elapsed)))
        return {
            "device": self._learn_device_name(),
            "room": self._learn_room_name() + self._learn_spot_text(),
            "samples": str(samples),
            "wanted": str(wanted),
            "left": f"{left // 60}:{left % 60:02d}",
        }

    async def async_step_learn_run(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Watch the training, refreshing every few seconds."""
        return await self._learn_chunk("learn_run", "learn_listen")

    async def async_step_learn_listen(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The other half of the loop: Home Assistant only refreshes the screen when the next step differs."""
        return await self._learn_chunk("learn_listen", "learn_run")

    async def _learn_chunk(self, this_step: str, other_step: str) -> ConfigFlowResult:
        assert self._learn is not None  # noqa: S101
        if self._learn_task is None:
            self._learn_task = self.hass.async_create_task(asyncio.sleep(CHUNK_SECONDS))
        if not self._learn_task.done():
            return self.async_show_progress(
                step_id=this_step,
                progress_action="learning",
                progress_task=self._learn_task,
                description_placeholders=self._learn_progress_text(),
            )
        self._learn_task = None
        if self._learn_coordinator.training_status(self._learn["device"])["training"]:
            return self.async_show_progress_done(next_step_id=other_step)
        return self.async_show_progress_done(next_step_id="learn_done")

    async def async_step_learn_done(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Say how it went, then go back to the menu."""
        assert self._learn is not None  # noqa: S101
        if user_input is not None:
            self._learn = None
            return await self.async_step_init()
        if self._learn.get("forget"):
            room = "" if self._learn.get("forget_all") else self._learn_room_name()
            if not self._learn.get("forgot"):
                outcome = "Nothing matched, so there was nothing to forget."
            elif room and self._learn.get("label"):
                outcome = f"Forgot the **{self._learn['label']}** spot in **{room}**."
            elif room:
                outcome = f"Forgot what was learned for **{room}**."
            else:
                outcome = "Forgot every room that was learned."
            return self.async_show_form(
                step_id="learn_done", data_schema=vol.Schema({}), description_placeholders={"outcome": outcome}
            )
        result = self._learn_coordinator.last_training.get(self._learn["device"]) or {}
        name, samples = self._learn_device_name(), result.get("samples", 0)
        room = self._learn_room_name()
        if result.get("label"):
            room = f"{room} ({result['label']})"
        if result.get("kept"):
            outcome = (
                f"Done. **{room}** is trained, taught by **{name}** from {samples} readings. "
                "Every tracked device is matched against it."
            )
            if result.get("dropped"):
                outcome += f" That room was full, so its oldest spot (**{result['dropped']}**) was dropped."
        else:
            outcome = (
                f"Only {samples} readings were heard, so nothing was saved. "
                f"Keep **{name}** awake and in the {room}, and try again."
            )
        return self.async_show_form(
            step_id="learn_done", data_schema=vol.Schema({}), description_placeholders={"outcome": outcome}
        )
