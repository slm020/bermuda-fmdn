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

from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.components.repairs import RepairsFlow
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.selector import AreaSelector

from .const import DOMAIN, REPAIR_SCANNER_WITHOUT_AREA, REPAIR_TRAIN_PREFIX
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
    """Handler for the "train a room?" issue: teach a room now, or skip it for good."""

    def __init__(self, hass: HomeAssistant, device_key: str) -> None:
        """Store hass (to reach the coordinator) and the address of the device that raised the prompt."""
        self._hass = hass
        self._device_key = device_key

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
                "areas": "several rooms",
            },
        )

    async def async_step_train(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Ask for the room, then start a training run using the device."""
        coordinator, device = self._coordinator_and_device()
        if coordinator is None or device is None:
            return self.async_abort(reason="device_gone")
        if user_input is not None and (area := user_input.get("area")):
            coordinator.start_training(device, area)
            return self.async_create_entry(title="", data={})
        return self.async_show_form(
            step_id="train",
            data_schema=vol.Schema({vol.Required("area"): AreaSelector()}),
            description_placeholders={"device": ha_device_name(self._hass, device), "kind": device_kind(device)},
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
