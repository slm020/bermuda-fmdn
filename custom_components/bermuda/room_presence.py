"""
Which rooms are empty right now, according to the house's own presence sensors.

Signal strength cannot tell the two sides of a thin wall apart (a toilet one plasterboard sheet
behind the sofa looks the same to every proxy). A room's occupancy sensor can: when every
occupancy/presence sensor in a room reports nobody there, nothing tracked is in that room either,
and the room filter treats that as strong evidence against it.

Only binary sensors with device class ``occupancy`` or ``presence`` count (radar-style sensors
that see people sitting still); plain ``motion`` sensors go off when someone sits still, so they
would wrongly empty a room. A sensor that is unavailable or unknown says nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

PRESENCE_CLASSES = frozenset({"occupancy", "presence"})
REDISCOVER_SECONDS = 300.0


class RoomPresence:
    """Finds each room's presence sensors (re-checked every few minutes) and reports empty rooms."""

    def __init__(self) -> None:
        """Start with nothing discovered."""
        self._sensors: dict[str, list[str]] = {}  # area id -> presence entity ids
        self._next_discovery = 0.0

    def discover(self, hass: HomeAssistant) -> None:
        """Map every enabled occupancy/presence binary sensor to its room (its own or its device's)."""
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        found: dict[str, list[str]] = {}
        for entry in ent_reg.entities.values():
            if entry.domain != "binary_sensor" or entry.disabled_by is not None:
                continue
            device_class = entry.device_class or entry.original_device_class
            if device_class not in PRESENCE_CLASSES:
                continue
            area_id = entry.area_id
            if area_id is None and entry.device_id and (device := dev_reg.async_get(entry.device_id)):
                area_id = device.area_id
            if area_id:
                found.setdefault(area_id, []).append(entry.entity_id)
        self._sensors = found

    def empty_areas(self, hass: HomeAssistant, now: float) -> set[str]:
        """Rooms whose presence sensors all say nobody is there (rooms without sensors never count)."""
        if now >= self._next_discovery:
            self._next_discovery = now + REDISCOVER_SECONDS
            self.discover(hass)
        empty: set[str] = set()
        for area_id, entity_ids in self._sensors.items():
            states = [s.state for e in entity_ids if (s := hass.states.get(e)) is not None]
            known = [s for s in states if s in (STATE_ON, STATE_OFF)]
            if known and all(s == STATE_OFF for s in known):
                empty.add(area_id)
        return empty
