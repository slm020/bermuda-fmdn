"""Button platform for Bermuda: per-device room-learning controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import SIGNAL_DEVICE_NEW
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
    """Set up the button platform."""
    coordinator: BermudaDataUpdateCoordinator = entry.runtime_data.coordinator
    created_devices: list[str] = []

    @callback
    def device_new(address: str) -> None:
        """Create the room-learning buttons for a newly tracked device."""
        if address not in created_devices:
            async_add_devices(
                [
                    BermudaLearnRoomButton(coordinator, entry, address),
                    BermudaResetLearningButton(coordinator, entry, address),
                ],
                update_before_add=False,
            )
            created_devices.append(address)

    entry.async_on_unload(async_dispatcher_connect(hass, SIGNAL_DEVICE_NEW, device_new))


class BermudaLearnRoomButton(BermudaEntity, ButtonEntity):
    """Start training this device in the room chosen on its "Training room" select."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "learn_room"
    _attr_entity_category = EntityCategory.CONFIG

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this button."""
        return f"{self._device.unique_id}_learn_room"

    async def async_press(self) -> None:
        """Begin a training run (keep the device in the chosen room for a few minutes)."""
        area_id = self.coordinator.training_room.get(self.address)
        if not area_id:
            msg = "Pick a training room for this device first."
            raise ServiceValidationError(msg)
        self.coordinator.start_training(self._device, area_id)


class BermudaResetLearningButton(BermudaEntity, ButtonEntity):
    """Forget every room trained for this device."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "reset_learning"
    _attr_entity_category = EntityCategory.CONFIG

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this button."""
        return f"{self._device.unique_id}_reset_learning"

    async def async_press(self) -> None:
        """Forget trained rooms."""
        self.coordinator.reset_learning(self._device)
