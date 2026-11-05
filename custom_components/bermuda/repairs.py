"""
Repairs flows for Bermuda.

The scanner_without_area issue is raised by the scanner mixin whenever one or
more Bluetooth proxies have no Area assigned (Bermuda cannot place devices
without it). The fix flow doesn't assign the area itself - only the user can
decide where a proxy lives - but once they have done so, confirming the flow
forces an immediate scanner re-check instead of waiting for the next natural
refresh, so the issue clears (or reappears with the remaining proxies) right
away.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.components.repairs import RepairsFlow
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.selector import AreaSelector

from .calibration import CHUNK_SECONDS
from .const import DOMAIN, LEARN_INTERVAL, LEARN_MAX_SECONDS, REPAIR_SCANNER_WITHOUT_AREA, REPAIR_TRAIN_PREFIX
from .device_kind import device_kind, ha_device_name

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.data_entry_flow import FlowResult

    from . import BermudaConfigEntry


class ScannerWithoutAreaRepairFlow(RepairsFlow):
    """Handler for the scanner_without_area issue: re-check once the user has assigned areas."""

    def __init__(self, hass: HomeAssistant, scannerlist: str) -> None:
        """Store hass (to reach the coordinator) and the offending-scanner list text."""
        self._hass = hass
        self._scannerlist = scannerlist

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the first step of a fix flow."""
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Wait for the user to confirm they have assigned the missing areas, then re-check."""
        if user_input is not None:
            for entry in self._hass.config_entries.async_entries(DOMAIN):
                if entry.state is ConfigEntryState.LOADED:
                    coordinator = cast("BermudaConfigEntry", entry).runtime_data.coordinator
                    # Re-inspect the scanner roster now; this re-raises the issue
                    # (with an updated proxy list) if any are still missing an area.
                    coordinator.refresh_scanners(force=True)
            return self.async_create_entry(title="", data={})
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={"scannerlist": self._scannerlist},
        )


class TrainRoomRepairFlow(RepairsFlow):
    """
    Handler for the "train a room?" issue: teach a room now (with a live countdown), or skip it for good.

    The screen stays open while Bermuda listens and says at the end whether the room was saved, so the
    user knows to keep the phone still and whether it worked.
    """

    def __init__(self, hass: HomeAssistant, device_key: str) -> None:
        """Store hass (to reach the coordinator) and the address of the device that raised the prompt."""
        self._hass = hass
        self._device_key = device_key
        self._area_id: str = ""
        self._started: float = 0.0
        self._task: asyncio.Task[None] | None = None

    def _coordinator_and_device(self) -> tuple[Any, Any]:
        coordinator = next(
            (
                cast("BermudaConfigEntry", entry).runtime_data.coordinator
                for entry in self._hass.config_entries.async_entries(DOMAIN)
                if entry.state is ConfigEntryState.LOADED
            ),
            None,
        )
        device = coordinator.devices.get(self._device_key) if coordinator else None
        return coordinator, device

    def _issue_areas(self) -> str:
        """The rooms named when the issue was raised (kept in the issue's placeholders)."""
        issue = ir.async_get(self._hass).async_get_issue(DOMAIN, self.issue_id)
        placeholders = (issue.translation_placeholders if issue else None) or {}
        return str(placeholders.get("areas") or "several rooms")

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Offer to teach a room now, or to skip this prompt."""
        coordinator, device = self._coordinator_and_device()
        if coordinator is None or device is None:
            return self.async_abort(reason="device_gone")
        return self.async_show_menu(
            step_id="init",
            menu_options=["train", "skip"],
            description_placeholders={
                "device": ha_device_name(self._hass, device),
                "kind": device_kind(device),
                "areas": self._issue_areas(),
            },
        )

    async def async_step_train(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Ask for the room, then start a training run using the device and watch it."""
        coordinator, device = self._coordinator_and_device()
        if coordinator is None or device is None:
            return self.async_abort(reason="device_gone")
        if user_input is not None and (area := user_input.get("area")):
            coordinator.start_training(device, area)
            self._area_id = area
            self._started = time.monotonic()
            return await self.async_step_train_run()
        return self.async_show_form(
            step_id="train",
            data_schema=vol.Schema({vol.Required("area"): AreaSelector()}),
            description_placeholders={"device": ha_device_name(self._hass, device), "kind": device_kind(device)},
        )

    def _progress_text(self, coordinator: Any, device: Any) -> dict[str, str]:
        status = coordinator.training_status(self._device_key)
        samples, wanted = status["samples"], status["samples_wanted"]
        elapsed = time.monotonic() - self._started
        left = max(0, round(min((wanted - samples) * LEARN_INTERVAL, LEARN_MAX_SECONDS - elapsed)))
        return {
            "device": ha_device_name(self._hass, device) if device else self._device_key,
            "room": coordinator.resolve_area_name(self._area_id) or self._area_id,
            "samples": str(samples),
            "wanted": str(wanted),
            "left": f"{left // 60}:{left % 60:02d}",
        }

    async def async_step_train_run(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Watch the training, refreshing every few seconds."""
        return await self._chunk("train_run", "train_listen")

    async def async_step_train_listen(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """The other half of the loop: Home Assistant only refreshes the screen when the next step differs."""
        return await self._chunk("train_listen", "train_run")

    async def _chunk(self, this_step: str, other_step: str) -> FlowResult:
        coordinator, device = self._coordinator_and_device()
        if coordinator is None:
            return self.async_abort(reason="device_gone")
        if self._task is None:
            self._task = self._hass.async_create_task(asyncio.sleep(CHUNK_SECONDS))
        if not self._task.done():
            return self.async_show_progress(
                step_id=this_step,
                progress_action="learning",
                progress_task=self._task,
                description_placeholders=self._progress_text(coordinator, device),
            )
        self._task = None
        if coordinator.training_status(self._device_key)["training"]:
            return self.async_show_progress_done(next_step_id=other_step)
        return self.async_show_progress_done(next_step_id="train_done")

    async def async_step_train_done(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Say whether the room was saved; submitting closes the notice."""
        if user_input is not None:
            return self.async_create_entry(title="", data={})
        coordinator, device = self._coordinator_and_device()
        if coordinator is None:
            return self.async_abort(reason="device_gone")
        result = coordinator.last_training.get(self._device_key) or {}
        name = ha_device_name(self._hass, device) if device else self._device_key
        room = coordinator.resolve_area_name(self._area_id) or self._area_id
        samples = result.get("samples", 0)
        if result.get("kept"):
            outcome = (
                f"Done. **{room}** is trained, taught by **{name}** from {samples} readings. "
                "Every tracked device is matched against it. If Bermuda still mixes up other rooms, "
                "teach those too under Settings > Bermuda > Configure > Calibrate > Learn a room."
            )
        else:
            outcome = (
                f"Only {samples} readings were heard, so nothing was saved. "
                f"Keep **{name}** awake and in the {room}, and try again."
            )
        return self.async_show_form(
            step_id="train_done", data_schema=vol.Schema({}), description_placeholders={"outcome": outcome}
        )

    async def async_step_skip(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Skip: dismiss the prompt, and it stays dismissed (Bermuda will not raise it again for this device)."""
        ir.async_ignore_issue(self._hass, DOMAIN, self.issue_id, True)
        return self.async_create_entry(title="", data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a flow to fix an issue raised by this integration."""
    if issue_id == REPAIR_SCANNER_WITHOUT_AREA:
        scannerlist = str((data or {}).get("scannerlist", ""))
        return ScannerWithoutAreaRepairFlow(hass, scannerlist)
    if issue_id.startswith(REPAIR_TRAIN_PREFIX):
        return TrainRoomRepairFlow(hass, str((data or {}).get("device_key", "")))
    msg = f"unknown repair issue: {issue_id}"
    raise ValueError(msg)
