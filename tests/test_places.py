"""Tests for the Places (micro-location) screens under Configure > Calibrate."""

from __future__ import annotations

import pytest
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import area_registry as ar

from custom_components.bermuda import options_places
from tests.test_services import _set_distance, _setup_keys_in_kitchen


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    """Shrink the 15 s settle time and the 5 s refresh so the flow can be driven in a test."""
    monkeypatch.setattr(options_places, "CHUNK_SECONDS", 0.02)
    monkeypatch.setattr(options_places, "PLACE_SETTLE_SECONDS", 0.06)


async def _open_places(hass, entry):
    flows = hass.config_entries.options
    result = await flows.async_init(entry.entry_id)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
    assert "places" in result["menu_options"]
    return flows, await flows.async_configure(result["flow_id"], user_input={"next_step_id": "places"})


async def _follow(hass, flows, result):
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        await hass.async_block_till_done()
        result = await flows.async_configure(result["flow_id"])
    return result


async def test_mark_a_place_with_a_countdown_then_report(hass, setup_bermuda_entry):
    coordinator, keys, _s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    assert result["description_placeholders"]["existing"] == "No places are marked yet."

    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    assert result["step_id"] == "place_mark"
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "name": "  Kitchen table "}
    )
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "placing"
    shown = result["description_placeholders"]
    assert shown["device"] == "Keys"
    assert shown["place"] == "Kitchen table"
    assert shown["heard"] == "2"

    result = await _follow(hass, flows, result)
    assert result["step_id"] == "place_done"
    outcome = result["description_placeholders"]["outcome"]
    assert "Saved **Kitchen table** in **Kitchen**, using **Keys**: heard by 2 proxies (5 readings)." in outcome
    spots = coordinator.fingerprints.list(keys.address)
    assert [f.name for f in spots] == ["Kitchen table"]

    result = await flows.async_configure(result["flow_id"], user_input={})
    assert result["type"] is FlowResultType.MENU  # back at the main menu


async def test_marking_the_same_name_again_updates_it_and_the_menu_lists_places(hass, setup_bermuda_entry):
    coordinator, keys, s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    coordinator.calibrate_location(keys, "Sofa")
    flows, result = await _open_places(hass, setup_bermuda_entry)
    assert "- **Sofa**" in result["description_placeholders"]["existing"]

    _set_distance(keys, s1, 2.0)
    kitchen = ar.async_get(hass).async_get_or_create("Kitchen")
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "name": "sofa", "room": kitchen.id}
    )
    result = await _follow(hass, flows, result)
    assert "Updated **sofa**" in result["description_placeholders"]["outcome"]
    assert len(coordinator.fingerprints.list(keys.address)) == 1


async def test_marking_needs_a_device_and_a_name(hass, setup_bermuda_entry):
    _coordinator, keys, _s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper(), "name": "   "})
    assert result["step_id"] == "place_mark"
    assert result["errors"] == {"base": "place_cannot_start"}


async def test_a_device_nobody_can_hear_reports_nothing_saved(hass, setup_bermuda_entry):
    coordinator, keys, _s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper(), "name": "Desk"})
    keys.adverts.clear()  # it goes quiet while we wait
    result = await _follow(hass, flows, result)
    assert "Nothing was saved." in result["description_placeholders"]["outcome"]
    assert coordinator.fingerprints.list(keys.address) == []


async def test_a_device_that_vanishes_reports_nothing_saved(hass, setup_bermuda_entry):
    coordinator, keys, _s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper(), "name": "Desk"})
    del coordinator.devices[keys.address]
    result = await _follow(hass, flows, result)
    assert "no longer visible" in result["description_placeholders"]["outcome"]


async def test_forget_places(hass, setup_bermuda_entry):
    coordinator, keys, s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_forget"})
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_places"

    coordinator.calibrate_location(keys, "Sofa")
    _set_distance(keys, s1, 4.0)
    coordinator.calibrate_location(keys, "Desk", "kitchen")
    keys.micro_location_id = coordinator.fingerprints.find_by_name("Sofa").id
    keys.micro_location_name = "Sofa"
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_forget"})
    assert result["step_id"] == "place_forget"
    sofa = coordinator.fingerprints.find_by_name("Sofa")
    result = await flows.async_configure(result["flow_id"], user_input={"places": [sofa.id]})
    assert result["type"] is FlowResultType.MENU
    assert [f.name for f in coordinator.fingerprints.list(keys.address)] == ["Desk"]
    assert keys.micro_location_name is None  # no longer reported as being on the sofa

    assert coordinator.forget_location("no-such-id") is False


async def test_no_tracked_devices_aborts_marking(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    for device in coordinator.devices.values():
        device.create_sensor = False
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_tracked_devices"


async def test_the_progress_screen_always_shows_before_moving_on(hass, setup_bermuda_entry):
    """The frontend needs a progress screen between progress-done steps; never done -> done."""
    _coordinator, keys, _s1, _s2 = await _setup_keys_in_kitchen(hass, setup_bermuda_entry)
    flows, result = await _open_places(hass, setup_bermuda_entry)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "place_mark"})
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper(), "name": "Desk"})
    kinds = []
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        kinds.append(result["type"])
        await hass.async_block_till_done()
        result = await flows.async_configure(result["flow_id"])
    assert kinds[0] is FlowResultType.SHOW_PROGRESS
    assert all(a is not b or a is FlowResultType.SHOW_PROGRESS for a, b in zip(kinds, kinds[1:], strict=False))
    assert result["step_id"] == "place_done"
