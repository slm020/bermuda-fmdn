"""Room presence: rooms whose occupancy sensors say nobody is there."""

from __future__ import annotations

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.bermuda.room_filter import RoomFilter, room_evidence
from custom_components.bermuda.room_presence import RoomPresence


async def test_empty_rooms_from_presence_sensors(hass):
    areas = ar.async_get(hass)
    bathroom = areas.async_get_or_create("Bathroom")
    lounge = areas.async_get_or_create("Lounge")
    hall = areas.async_get_or_create("Hall")
    entry = MockConfigEntry(domain="test")
    entry.add_to_hass(hass)
    ents = er.async_get(hass)
    devs = dr.async_get(hass)
    radar = devs.async_get_or_create(config_entry_id=entry.entry_id, identifiers={("test", "radar")})
    devs.async_update_device(radar.id, area_id=bathroom.id)
    bath = ents.async_get_or_create(
        "binary_sensor", "test", "bath", device_id=radar.id, original_device_class="occupancy"
    )
    lounge_e = ents.async_get_or_create("binary_sensor", "test", "lounge", original_device_class="presence")
    ents.async_update_entity(lounge_e.entity_id, area_id=lounge.id)
    pir = ents.async_get_or_create("binary_sensor", "test", "pir", original_device_class="motion")
    ents.async_update_entity(pir.entity_id, area_id=hall.id)
    off = ents.async_get_or_create("binary_sensor", "test", "off", original_device_class="occupancy")
    ents.async_update_entity(off.entity_id, area_id=hall.id, disabled_by=er.RegistryEntryDisabler.USER)
    ents.async_get_or_create("sensor", "test", "notbinary", original_device_class="occupancy")
    ents.async_get_or_create("binary_sensor", "test", "noarea", original_device_class="occupancy")

    presence = RoomPresence()
    hass.states.async_set(bath.entity_id, "off")
    hass.states.async_set(lounge_e.entity_id, "on")
    hass.states.async_set(pir.entity_id, "off")
    assert presence.empty_areas(hass, 0.0) == {bathroom.id}  # motion-only hall never counts
    hass.states.async_set(bath.entity_id, "unavailable")
    assert presence.empty_areas(hass, 1.0) == set()  # an unavailable sensor says nothing
    hass.states.async_set(lounge_e.entity_id, "off")
    assert presence.empty_areas(hass, 2.0) == {lounge.id}


def test_an_empty_room_loses_a_near_tie_through_the_wall():
    areas = {"lr": "living_room", "bath": "bathroom"}
    f = RoomFilter()
    live = {"lr": -80.0, "bath": -81.0}  # sofa against the bathroom wall
    for _ in range(30):
        shown = f.update(
            room_evidence(live, areas, {"living_room": 1.5, "bathroom": 1.0}, set(areas.values()), {"living_room"})
        )
    assert shown == "bathroom"  # living room is empty, so it is the toilet side
    for _ in range(30):
        shown = f.update(
            room_evidence(live, areas, {"living_room": 1.5, "bathroom": 1.0}, set(areas.values()), {"bathroom"})
        )
    assert shown == "living_room"


def test_a_dog_the_radar_misses_still_shows_where_the_radio_says():
    areas = {"lr": "living_room", "kit": "kitchen"}
    f = RoomFilter()
    live = {"lr": -90.0, "kit": -85.0}  # clearly nearer the kitchen proxy, but the kitchen radar sees nobody
    for _ in range(30):
        shown = f.update(room_evidence(live, areas, {"living_room": 3.0}, set(areas.values()), {"kitchen"}))
    assert shown == "kitchen"
