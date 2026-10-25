"""
Tests for the friendlier device pickers: kind labels, strongest-first order, and the hide list.

``describe_kind`` is pure and tested directly. The pickers are driven through the real options
flow on a loaded entry with hand-built devices, since the labels depend on real adverts.
"""

from __future__ import annotations

import pytest
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.data_entry_flow import FlowResultType

from custom_components.bermuda.const import (
    BDADDR_TYPE_OTHER,
    BDADDR_TYPE_RANDOM_RESOLVABLE,
    CONF_BLOCKED_DEVICES,
    CONF_BLOCKED_KINDS,
    CONF_DEVICES,
    CONF_TRACK_CATEGORIES,
)
from custom_components.bermuda.const import ADDR_TYPE_IBEACON, ADDR_TYPE_PRIVATE_BLE_DEVICE
from custom_components.bermuda.device_kind import (
    describe_kind,
    device_facts,
    device_kind,
    device_label,
    device_room,
    identity_hint,
    signal_phrase,
    strongest_rssi,
)
from custom_components.bermuda.helpers import is_blocked
from tests.test_services import _make_scanner, _mock_ad

APPLE = 0x004C


def _kind(manufacturer_data=None, services=(), name=None, tracker=None, manufacturer=None) -> str:
    return describe_kind(
        manufacturer_data=manufacturer_data or {},
        service_uuids=set(services),
        name=name,
        tracker_type=tracker,
        manufacturer=manufacturer,
    )


# ---------------------------------------------------------------------------
# describe_kind
# ---------------------------------------------------------------------------


def test_recognised_trackers_win():
    assert _kind({APPLE: bytes([0x12, 0x19])}, tracker="Apple AirTag") == "Apple AirTag"


@pytest.mark.parametrize(
    ("first_byte", "expected"),
    [
        (0x10, "Apple device (iPhone, iPad, Mac or Watch)"),
        (0x0C, "Apple device (iPhone, iPad, Mac or Watch)"),
        (0x07, "AirPods / Beats earbuds"),
        (0x09, "Apple TV / HomePod / AirPlay speaker"),
        (0x06, "HomeKit accessory"),
        (0x02, "iBeacon"),
        (0x77, "Apple device"),
    ],
)
def test_apple_continuity_types(first_byte, expected):
    assert _kind({APPLE: bytes([first_byte, 5, 1, 2, 3])}) == expected


def test_apple_with_empty_payload_is_still_apple():
    assert _kind({APPLE: b""}) == "Apple device"


@pytest.mark.parametrize(
    ("company", "expected"),
    [
        (0x0006, "Windows PC / Microsoft device"),
        (0x0075, "Samsung device (phone, TV, watch or earbuds)"),
        (0x00E0, "Google / Android device"),
        (0x0171, "Amazon device (Echo, Fire, Ring...)"),
    ],
)
def test_vendor_payloads(company, expected):
    assert _kind({company: b"\x01"}) == expected


def test_well_known_services_name_the_device_type():
    assert _kind(services={"180D"}) == "Heart-rate monitor"
    assert _kind(services={"FE2C"}).startswith("Android / Fast Pair")
    assert _kind(services={"0000", "1812"}) == "Keyboard / mouse / controller (HID)"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Steve's iPhone", "Apple device"),
        ("Galaxy S24", "Android phone / watch"),
        ("[TV] Samsung 7 Series", "TV"),
        ("JBL Flip 6", "Audio device"),
        ("Fitbit Charge", "Watch / fitness band"),
    ],
)
def test_name_hints(name, expected):
    assert _kind(name=name) == expected


def test_falls_back_to_the_company_then_to_unidentified():
    assert _kind(manufacturer="Garmin") == "Garmin device"
    assert _kind() == "Unidentified device"


# ---------------------------------------------------------------------------
# signal, room, hint, label, facts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rssi", "with_room", "without_room"),
    [
        (-50, "right by Study", "very close"),
        (-65, "close to Study", "close"),
        (-75, "close to Study", "close"),
        (-80, "far from everywhere", "far from everywhere"),
        (-95, "far from everywhere", "far from everywhere"),
    ],
)
def test_signal_phrase_says_close_to_a_room_or_far_from_everywhere(rssi, with_room, without_room):
    assert signal_phrase(rssi, "Study") == with_room
    assert signal_phrase(rssi, None) == without_room
    assert signal_phrase(None, "Study") == "no signal"


def _hear(coordinator, address, *, rssi, scanner, manufacturer_data=None, service_uuids=None, name=None):
    """Create a device and give it one advert from a proxy."""
    device = coordinator._get_or_create_device(address)
    ad = _mock_ad(rssi)
    ad.manufacturer_data = manufacturer_data or {}
    ad.service_uuids = service_uuids or []
    ad.local_name = name
    device.process_advertisement(scanner, ad)
    return device


def _scanner(coordinator, address, room_id, room_name):
    scanner = _make_scanner(coordinator, address, room_name, room_id)
    scanner._is_scanner = True  # so the pickers skip it, as they skip real proxies
    return scanner


async def _neighbourhood(entry):
    """
    A study PC of mine, an unidentifiable gadget in the bedroom, and three neighbours' Apple devices.

    Rebuilt on the live coordinator each time a step is opened, because saving options reloads the
    integration and drops hand-built devices.
    """
    coordinator = entry.runtime_data.coordinator
    study = _scanner(coordinator, "11:11:11:11:11:11", "study", "Study")
    living = _scanner(coordinator, "22:22:22:22:22:22", "living", "Living room")
    bedroom = _scanner(coordinator, "33:33:33:33:33:33", "bedroom", "Bedroom")
    pc = _hear(
        coordinator, "AA:BB:CC:DD:EE:11", rssi=-45, scanner=study, manufacturer_data={0x0006: b"\x01"}, name="My PC"
    )
    pc.address_type = BDADDR_TYPE_OTHER
    gadget = _hear(coordinator, "AA:BB:CC:DD:EE:22", rssi=-78, scanner=bedroom, manufacturer_data={0x04F2: b"\x01\x02"})
    gadget.address_type = BDADDR_TYPE_OTHER
    apples = []
    for i, rssi in enumerate((-88, -75, -82)):
        apple = _hear(
            coordinator,
            f"5A:00:00:00:00:{i:02X}",
            rssi=rssi,
            scanner=living,
            manufacturer_data={APPLE: bytes([0x10, 5, 1])},
        )
        apple.address_type = BDADDR_TYPE_RANDOM_RESOLVABLE
        apples.append(apple)
    return coordinator, pc, gadget, apples


async def test_label_says_what_it_is_its_name_and_how_close_it_is(hass, setup_bermuda_entry):
    _coordinator, pc, _gadget, apples = await _neighbourhood(setup_bermuda_entry)
    assert device_label(pc) == 'Windows PC / Microsoft device · "My PC" · right by Study · …DD:EE:11'
    assert device_room(pc) == ("study", "Study")
    assert strongest_rssi(pc) == -45
    assert device_label(apples[1]) == "Apple device (iPhone, iPad, Mac or Watch) · close to Living room · …00:00:01"
    assert device_label(apples[0]).endswith("far from everywhere · …00:00:00")


async def test_unidentified_devices_carry_raw_clues_in_their_label(hass, setup_bermuda_entry):
    _coordinator, _pc, gadget, _apples = await _neighbourhood(setup_bermuda_entry)
    assert device_kind(gadget) == "Unidentified device"
    assert identity_hint(gadget) == "company 0x04F2"
    assert device_label(gadget) == "Unidentified device · company 0x04F2 · far from everywhere · …DD:EE:22"


async def test_hint_uses_a_service_id_when_there_is_no_company_id(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    scanner = _scanner(coordinator, "11:11:11:11:11:11", "study", "Study")
    odd = _hear(
        coordinator,
        "AA:BB:CC:DD:EE:33",
        rssi=-60,
        scanner=scanner,
        service_uuids=["0000fe9f-0000-1000-8000-00805f9b34fb"],
    )
    assert identity_hint(odd) == "service 0xFE9F"
    known = _hear(
        coordinator,
        "AA:BB:CC:DD:EE:34",
        rssi=-60,
        scanner=scanner,
        service_uuids=["0000180d-0000-1000-8000-00805f9b34fb"],
    )
    assert identity_hint(known) is None  # already identified, so no clue needed
    silent = coordinator._get_or_create_device("AA:BB:CC:DD:EE:35")
    assert identity_hint(silent) is None
    assert device_room(silent) is None
    assert device_label(silent) == "Unidentified device · no signal · …DD:EE:35"


async def test_facts_list_everything_known_about_a_device(hass, setup_bermuda_entry):
    _coordinator, _pc, gadget, apples = await _neighbourhood(setup_bermuda_entry)
    gadget.last_seen = 100.0
    facts = dict(device_facts(gadget, now=130.0))
    assert facts["Looks like"] == "Unidentified device"
    assert facts["Name"] == "none broadcast"
    assert facts["Address"].startswith("AA:BB:CC:DD:EE:22 · public")
    assert facts["Manufacturer data 0x04F2"] == "01 02"
    assert "Bedroom" in facts["Heard by"]
    assert "-78 dBm" in facts["Heard by"]
    assert facts["Last heard"] == "30 s ago"
    assert "resolvable" in dict(device_facts(apples[0]))["Address"]


async def test_facts_truncate_long_payloads_and_cover_service_data(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    scanner = _scanner(coordinator, "11:11:11:11:11:11", "study", "Study")
    device = _hear(
        coordinator, "AA:BB:CC:DD:EE:36", rssi=-60, scanner=scanner, manufacturer_data={0x1234: bytes(range(40))}
    )
    advert = next(iter(device.adverts.values()))
    advert.service_data = [{"0000fe2c-0000-1000-8000-00805f9b34fb": b"\x01\x02"}]
    device.manufacturer = "Acme"
    device.manufacturer_id = 0x1234
    facts = dict(device_facts(device))
    assert facts["Manufacturer data 0x1234"].endswith("…")
    assert facts["Service data 0xFE2C"] == "01 02"
    assert facts["Company"] == "Acme 0x1234"


# ---------------------------------------------------------------------------
# is_blocked
# ---------------------------------------------------------------------------


async def test_is_blocked_by_address_and_by_kind(hass, setup_bermuda_entry):
    _coordinator, pc, _gadget, apples = await _neighbourhood(setup_bermuda_entry)
    apple, kind = apples[0], device_kind(apples[0])

    assert not is_blocked(apple, {})
    assert is_blocked(apple, {CONF_BLOCKED_DEVICES: [apple.address.lower()]})  # case-insensitive
    assert is_blocked(apple, {CONF_BLOCKED_KINDS: [kind]})
    assert not is_blocked(pc, {CONF_BLOCKED_KINDS: [kind]})  # a different kind
    # a kind hides every address type, but never something the user tracks by hand
    apple.address_type = BDADDR_TYPE_OTHER
    assert is_blocked(apple, {CONF_BLOCKED_KINDS: [kind]})
    assert not is_blocked(apple, {CONF_BLOCKED_KINDS: [kind], CONF_DEVICES: [apple.address.upper()]})
    assert is_blocked(pc, {CONF_BLOCKED_DEVICES: [pc.address], CONF_DEVICES: [pc.address]})


async def test_blocked_devices_are_not_auto_tracked_by_category(hass, setup_bermuda_entry):
    coordinator, _pc, _gadget, apples = await _neighbourhood(setup_bermuda_entry)
    neighbour = apples[0]
    coordinator.options[CONF_TRACK_CATEGORIES] = [neighbour.category]

    coordinator.options[CONF_BLOCKED_DEVICES] = [neighbour.address.upper()]
    neighbour.create_sensor = False
    neighbour.calculate_data()
    assert neighbour.create_sensor is False

    coordinator.options[CONF_BLOCKED_DEVICES] = []
    neighbour.calculate_data()
    assert neighbour.create_sensor is True


# ---------------------------------------------------------------------------
# the options-flow screens
# ---------------------------------------------------------------------------


def _field(result, name: str):
    for key, validator in result["data_schema"].schema.items():
        if str(key.schema) == name:
            return key, validator
    msg = f"no field {name}"
    raise AssertionError(msg)


def _options(result, name: str) -> list[tuple[str, str]]:
    """(value, label) pairs offered by one form field."""
    return [(o["value"], o["label"]) for o in _field(result, name)[1].config["options"]]


def _values(result, name: str) -> list[str]:
    return [value for value, _label in _options(result, name)]


async def _menu(hass, entry, *path: str, populate: bool = True):
    """Open the options flow and pick menu entries in turn."""
    if populate:
        await _neighbourhood(entry)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    for step in path:
        result = await hass.config_entries.options.async_configure(result["flow_id"], user_input={"next_step_id": step})
    return result


async def _submit(hass, result, **fields):
    return await hass.config_entries.options.async_configure(result["flow_id"], user_input=fields)


async def test_main_menu_is_short_and_settings_hold_the_rest(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry)
    assert result["menu_options"] == ["add", "hide", "identify", "tracked", "hidden", "google", "settings"]
    result = await _menu(hass, setup_bermuda_entry, "settings")
    assert result["menu_options"] == ["globalopts", "categories", "area_entities", "enrol_private"]


async def test_add_lists_strongest_first_with_tick_boxes_and_adds_what_is_ticked(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "add")
    assert result["step_id"] == "add"  # straight to the list: no room question
    assert _values(result, "devices") == [  # strongest signal first
        "AA:BB:CC:DD:EE:11",
        "5A:00:00:00:00:01",
        "AA:BB:CC:DD:EE:22",
        "5A:00:00:00:00:02",
        "5A:00:00:00:00:00",
    ]
    assert _field(result, "devices")[1].config["mode"] == "list"  # tick boxes, not a dropdown
    assert result["description_placeholders"] == {"count": "5"}
    labels = dict(_options(result, "devices"))
    assert "right by Study" in labels["AA:BB:CC:DD:EE:11"]
    assert "close to Living room" in labels["5A:00:00:00:00:01"]
    assert "far from everywhere" in labels["5A:00:00:00:00:00"]

    result = await _submit(hass, result, devices=["AA:BB:CC:DD:EE:11"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_bermuda_entry.options[CONF_DEVICES] == ["AA:BB:CC:DD:EE:11"]

    # tracked devices are no longer offered for adding, and adding never drops what was tracked
    result = await _menu(hass, setup_bermuda_entry, "add")
    assert "AA:BB:CC:DD:EE:11" not in _values(result, "devices")
    result = await _submit(hass, result, devices=["AA:BB:CC:DD:EE:22"])
    assert setup_bermuda_entry.options[CONF_DEVICES] == ["AA:BB:CC:DD:EE:11", "AA:BB:CC:DD:EE:22"]


async def test_pickers_skip_proxies_private_devices_and_long_gone_random_macs(hass, setup_bermuda_entry):
    coordinator, _pc, _gadget, _apples = await _neighbourhood(setup_bermuda_entry)
    private = coordinator._get_or_create_device("AA:BB:CC:DD:EE:40")
    private.address_type = ADDR_TYPE_PRIVATE_BLE_DEVICE
    stale = coordinator._get_or_create_device("5A:00:00:00:00:77")
    stale.address_type = BDADDR_TYPE_RANDOM_RESOLVABLE
    stale.last_seen = monotonic_time_coarse() - 3 * 60 * 60
    beacon = coordinator._get_or_create_device("AA:BB:CC:DD:EE:41")
    beacon.address_type = ADDR_TYPE_IBEACON
    result = await _menu(hass, setup_bermuda_entry, "add", populate=False)
    values = set(_values(result, "devices"))
    assert "AA:BB:CC:DD:EE:41" in values  # iBeacon metadevices are offered
    assert not values & {"AA:BB:CC:DD:EE:40", "5A:00:00:00:00:77", "11:11:11:11:11:11"}


async def test_hide_by_kind_hides_every_device_of_that_kind_at_once(hass, setup_bermuda_entry):
    kind = "Apple device (iPhone, iPad, Mac or Watch)"
    result = await _menu(hass, setup_bermuda_entry, "hide")
    assert result["step_id"] == "hide"
    assert (kind, f"{kind} (3)") in _options(result, "kinds")
    result = await _submit(hass, result, kinds=[kind], individually=False)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_bermuda_entry.options[CONF_BLOCKED_KINDS] == [kind]

    result = await _menu(hass, setup_bermuda_entry, "add")
    assert result["description_placeholders"] == {"count": "2"}  # the three Apples are gone
    result = await _menu(hass, setup_bermuda_entry, "hide")
    assert kind not in _values(result, "kinds")


async def test_hide_individually_goes_to_a_tick_list(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "hide")
    result = await _submit(hass, result, kinds=[], individually=True)
    assert result["step_id"] == "hide_pick"
    assert _values(result, "devices")[:2] == ["AA:BB:CC:DD:EE:11", "5A:00:00:00:00:01"]
    result = await _submit(hass, result, devices=["5A:00:00:00:00:01"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_bermuda_entry.options[CONF_BLOCKED_DEVICES] == ["5A:00:00:00:00:01"]
    result = await _menu(hass, setup_bermuda_entry, "add")
    assert "5A:00:00:00:00:01" not in _values(result, "devices")


async def test_hidden_screen_unhides_what_is_unticked(hass, setup_bermuda_entry):
    kind = "Apple device (iPhone, iPad, Mac or Watch)"
    hass.config_entries.async_update_entry(
        setup_bermuda_entry,
        options={
            **setup_bermuda_entry.options,
            CONF_BLOCKED_KINDS: [kind],
            CONF_BLOCKED_DEVICES: ["AA:BB:CC:DD:EE:22", "AA:AA:AA:AA:AA:AA"],
        },
    )
    await hass.async_block_till_done()
    result = await _menu(hass, setup_bermuda_entry, "hidden")
    assert _options(result, "kinds") == [(kind, kind)]
    labels = dict(_options(result, "devices"))
    assert labels["AA:AA:AA:AA:AA:AA"] == "AA:AA:AA:AA:AA:AA (not seen now)"
    assert "Unidentified device" in labels["AA:BB:CC:DD:EE:22"]
    assert _field(result, "devices")[0].default() == ["AA:BB:CC:DD:EE:22", "AA:AA:AA:AA:AA:AA"]  # all ticked

    result = await _submit(hass, result, kinds=[], devices=["AA:AA:AA:AA:AA:AA"])
    assert setup_bermuda_entry.options[CONF_BLOCKED_KINDS] == []
    assert setup_bermuda_entry.options[CONF_BLOCKED_DEVICES] == ["AA:AA:AA:AA:AA:AA"]


async def test_tracked_screen_untick_stops_tracking(hass, setup_bermuda_entry):
    hass.config_entries.async_update_entry(
        setup_bermuda_entry,
        options={**setup_bermuda_entry.options, CONF_DEVICES: ["AA:BB:CC:DD:EE:11", "AA:AA:AA:AA:AA:AA"]},
    )
    await hass.async_block_till_done()
    result = await _menu(hass, setup_bermuda_entry, "tracked")
    labels = dict(_options(result, "devices"))
    assert "Windows PC" in labels["AA:BB:CC:DD:EE:11"]
    assert labels["AA:AA:AA:AA:AA:AA"].endswith("(not seen now)")
    assert _field(result, "devices")[0].default() == ["AA:BB:CC:DD:EE:11", "AA:AA:AA:AA:AA:AA"]
    result = await _submit(hass, result, devices=["AA:BB:CC:DD:EE:11"])
    assert setup_bermuda_entry.options[CONF_DEVICES] == ["AA:BB:CC:DD:EE:11"]


async def test_identify_shows_everything_known_about_one_device(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "identify")
    assert result["step_id"] == "identify"
    assert _field(result, "device")[1].config["multiple"] is False  # pick exactly one
    result = await _submit(hass, result, device="AA:BB:CC:DD:EE:22")
    assert result["step_id"] == "identify_detail"
    facts = result["description_placeholders"]["facts"]
    assert "| Looks like | Unidentified device |" in facts
    assert "| Manufacturer data 0x04F2 | 01 02 |" in facts
    assert "Bedroom" in facts

    back = await _submit(hass, result)  # "Submit" returns to the main menu
    assert back["type"] is FlowResultType.MENU
    assert back["step_id"] == "init"


async def test_identify_copes_with_a_device_that_has_gone(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "identify")
    result = await _submit(hass, result, device="AA:BB:CC:DD:EE:22")
    flow = hass.config_entries.options._progress[result["flow_id"]]  # noqa: SLF001
    flow._identify = "FF:FF:FF:FF:FF:FF"  # noqa: SLF001
    result = await flow.async_step_identify_detail()
    assert "no longer seen" in result["description_placeholders"]["facts"]


async def test_google_tags_screen_explains_how_to_connect_when_not_connected(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "google")
    assert result["step_id"] == "google"
    assert "Not connected yet" in result["description_placeholders"]["status"]
    back = await _submit(hass, result)
    assert back["step_id"] == "init"


async def test_google_tags_screen_lists_known_tags_when_connected(hass, setup_bermuda_entry):
    from custom_components.bermuda.fmdn_resolver import FmdnTag  # noqa: PLC0415

    coordinator = setup_bermuda_entry.runtime_data.coordinator
    hass.data["googlefindmy"] = {"eid_resolver": type("R", (), {"resolve_eid": lambda self, eid: None})()}
    result = await _menu(hass, setup_bermuda_entry, "google")
    assert "knows of no tags yet" in result["description_placeholders"]["status"]

    tag = coordinator._fmdn_get_metadevice(FmdnTag("dev1", "canon", "entry"))  # noqa: SLF001
    tag.name = "Keys tag"
    tag.last_seen = monotonic_time_coarse() - 30
    other = coordinator._fmdn_get_metadevice(FmdnTag("dev2", "canon2", "entry"))  # noqa: SLF001
    other.name = "Wallet tag"
    other.last_seen = 0
    result = await _menu(hass, setup_bermuda_entry, "google", populate=False)
    status = result["description_placeholders"]["status"]
    assert "Connected." in status
    assert "- Keys tag: heard 30 s ago" in status
    assert "- Wallet tag: not heard yet" in status


async def test_categories_screen_persists_choices(hass, setup_bermuda_entry):
    result = await _menu(hass, setup_bermuda_entry, "settings", "categories")
    assert result["step_id"] == "categories"
    result = await _submit(hass, result, track_categories=["ibeacon"])
    assert setup_bermuda_entry.options[CONF_TRACK_CATEGORIES] == ["ibeacon"]


def test_hiding_twice_does_not_duplicate_and_drops_tracking():
    from custom_components.bermuda.options_flow import BermudaOptionsFlowHandler  # noqa: PLC0415

    flow = BermudaOptionsFlowHandler()
    flow._options = {CONF_DEVICES: ["AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"]}
    flow._hide(["AA:BB:CC:DD:EE:01"], ["Some kind"])
    flow._hide(["AA:BB:CC:DD:EE:01"], ["Some kind"])
    assert flow._options[CONF_DEVICES] == ["AA:BB:CC:DD:EE:02"]
    assert flow._options[CONF_BLOCKED_DEVICES] == ["AA:BB:CC:DD:EE:01"]
    assert flow._options[CONF_BLOCKED_KINDS] == ["Some kind"]
