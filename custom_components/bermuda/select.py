"""Select platform for Bermuda: a per-device mobility-mode selector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.select import SelectEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.restore_state import RestoreEntity

from .const import MOBILITY_OPTIONS, SIGNAL_DEVICE_NEW
from .entity import BermudaEntity

PARALLEL_UPDATES = 0

if TYPE_CHECKING:
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from . import BermudaConfigEntry
    from .coordinator import BermudaDataUpdateCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BermudaConfigEntry,
    async_add_devices: AddEntitiesCallback,
) -> None:
    """Set up the select platform."""
    coordinator: BermudaDataUpdateCoordinator = entry.runtime_data.coordinator
    created_devices: list[str] = []

    @callback
    def device_new(address: str) -> None:
        """Create the mobility select for a newly tracked device."""
        if address not in created_devices:
            async_add_devices(
                [
                    BermudaMobilityTypeSelect(coordinator, entry, address),
                    BermudaTrainingRoomSelect(coordinator, entry, address),
                ],
                update_before_add=False,
            )
            created_devices.append(address)
        coordinator.select_created(address)

    entry.async_on_unload(async_dispatcher_connect(hass, SIGNAL_DEVICE_NEW, device_new))


class BermudaMobilityTypeSelect(BermudaEntity, SelectEntity, RestoreEntity):
    """Per-device mobility mode selector that tunes Bermuda's RSSI filtering and hysteresis."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "mobility_type"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_options = MOBILITY_OPTIONS

    async def async_added_to_hass(self) -> None:
        """Restore the saved mobility mode (falling back to the default)."""
        await super().async_added_to_hass()
        old_state = await self.async_get_last_state()
        self._device.set_mobility_type(old_state.state if old_state is not None else None)

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this mobility select entity."""
        return f"{self._device.unique_id}_mobility"

    @property
    def current_option(self) -> str:
        """Return the currently selected mobility mode."""
        return self._device.get_mobility_type()

    async def async_select_option(self, option: str) -> None:
        """Set the selected mobility mode."""
        self._device.set_mobility_type(option)
        self.async_write_ha_state()


class BermudaTrainingRoomSelect(BermudaEntity, SelectEntity):
    """Which room to train this device in; used by the "Learn this room" button."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "training_room"
    _attr_entity_category = EntityCategory.CONFIG

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this training-room select entity."""
        return f"{self._device.unique_id}_training_room"

    @property
    def options(self) -> list[str]:
        """Every Home Assistant area, by name."""
        return sorted(area.name for area in self.ar.async_list_areas())

    @property
    def current_option(self) -> str | None:
        """The chosen training room, if it still exists."""
        area_id = self.coordinator.training_room.get(self.address)
        area = self.ar.async_get_area(area_id) if area_id else None
        return area.name if area else None

    async def async_select_option(self, option: str) -> None:
        """Remember the chosen room for the Learn button."""
        if (area := self.ar.async_get_area_by_name(option)) is not None:
            self.coordinator.training_room[self.address] = area.id
        self.async_write_ha_state()
