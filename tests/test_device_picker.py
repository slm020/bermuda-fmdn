"""
Tests for the friendlier device pickers: kind labels, strongest-first order, and the hide list.

``describe_kind`` is pure and tested directly. The pickers are driven through the real options
flow on a loaded entry with hand-built devices, since the labels depend on real adverts.
"""

from __future__ import annotations

import pytest
from homeassistant.data_entry_flow import FlowResultType

from custom_components.bermuda.const import (
    BDADDR_TYPE_OTHER,
    BDADDR_TYPE_RANDOM_RESOLVABLE,
    CATEGORY_RANDOM,
    CONF_BLOCKED_DEVICES,
    CONF_BLOCKED_KINDS,
    CONF_DEVICES,
    CONF_TRACK_CATEGORIES,
)
from custom_components.bermuda.device_kind import describe_kind, device_kind, device_label, strongest_rssi
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
# labels on real devices
# ---------------------------------------------------------------------------


def _hear(coordinator, address, *, rssi, manufacturer_data=None, service_uuids=None, name=None, scanner=None):
    """Create a device and give it one advert from a proxy."""
    scanner = scanner or _make_scanner(coordinator, "11:11:11:11:11:11", "Proxy", "kitchen")
    device = coordinator._get_or_create_device(address)
    ad = _mock_ad(rssi)
    ad.manufacturer_data = manufacturer_data or {}
    ad.service_uuids = service_uuids or []
    ad.local_name = name
    device.process_advertisement(scanner, ad)
    return device


async def test_label_shows_kind_name_signal_age_and_address(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    phone = _hear(
        coordinator, "AA:BB:CC:DD:EE:01", rssi=-48, manufacturer_data={APPLE: bytes([0x10, 5, 1])}, name="Steve's phone"
    )
    phone.last_seen = 1000.0
    label = device_label(phone, now=1003.0)
    assert (
        label
        == 'Apple device (iPhone, iPad, Mac or Watch) · "Steve\'s phone" · -48 dBm, very close · 3s ago · …DD:EE:01'
    )
    assert device_kind(phone).startswith("Apple device")
    assert strongest_rssi(phone) == -48


async def test_label_without_a_real_name_or_signal_stays_tidy(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    ghost = coordinator._get_or_create_device("AA:BB:CC:DD:EE:02")
    ghost.last_seen = 0
    label = device_label(ghost, now=50.0)
    assert label == "Unidentified device · no signal · …DD:EE:02"
    assert strongest_rssi(ghost) is None


@pytest.mark.parametrize(("rssi", "hint"), [(-50, "very close"), (-65, "near"), (-80, "far"), (-95, "very weak")])
async def test_signal_hints(hass, setup_bermuda_entry, rssi, hint):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    device = _hear(coordinator, "AA:BB:CC:DD:EE:03", rssi=rssi)
    assert hint in device_label(device, now=0.0)


# ---------------------------------------------------------------------------
# is_blocked
# ---------------------------------------------------------------------------


async def test_is_blocked_by_address_and_by_kind(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    apple = {APPLE: bytes([0x10, 5, 1])}
    rotating = _hear(coordinator, "5A:00:00:00:00:01", rssi=-80, manufacturer_data=apple)  # resolvable random
    rotating.address_type = BDADDR_TYPE_RANDOM_RESOLVABLE
    public = _hear(coordinator, "AA:BB:CC:DD:EE:04", rssi=-80, manufacturer_data=apple)
    public.address_type = BDADDR_TYPE_OTHER
    kind = device_kind(rotating)

    assert not is_blocked(rotating, {})
    assert is_blocked(rotating, {CONF_BLOCKED_DEVICES: ["5a:00:00:00:00:01"]})  # case-insensitive
    assert is_blocked(rotating, {CONF_BLOCKED_KINDS: [kind]})
    assert not is_blocked(public, {CONF_BLOCKED_KINDS: [kind]})  # kinds only hide random addresses
    assert not is_blocked(
        rotating, {CONF_BLOCKED_KINDS: [kind], CONF_DEVICES: ["5A:00:00:00:00:01"]}
    )  # tracked by hand
    assert is_blocked(public, {CONF_BLOCKED_DEVICES: ["AA:BB:CC:DD:EE:04"], CONF_DEVICES: ["AA:BB:CC:DD:EE:04"]})


async def test_blocked_devices_are_not_auto_tracked_by_category(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    neighbour = _hear(coordinator, "5A:00:00:00:00:02", rssi=-85)
    neighbour.address_type = BDADDR_TYPE_RANDOM_RESOLVABLE
    coordinator.options[CONF_TRACK_CATEGORIES] = [neighbour.category]
    assert neighbour.category == CATEGORY_RANDOM

    coordinator.options[CONF_BLOCKED_DEVICES] = [neighbour.address.upper()]
    neighbour.create_sensor = False
    neighbour.calculate_data()
    assert neighbour.create_sensor is False

    coordinator.options[CONF_BLOCKED_DEVICES] = []
    neighbour.calculate_data()
    assert neighbour.create_sensor is True


# ---------------------------------------------------------------------------
# the options-flow pickers
# ---------------------------------------------------------------------------


def _field_options(result, field: str) -> list[tuple[str, str]]:
    """(value, label) pairs offered by one form field."""
    for key, validator in result["data_schema"].schema.items():
        if str(key.schema) == field:
            return [(opt["value"], opt["label"]) for opt in validator.config["options"]]
    msg = f"no field {field}"
    raise AssertionError(msg)


async def _open(hass, entry, step: str, *, populate: bool = True):
    """
    Open an options step.

    Saving options reloads the integration (dropping the hand-built devices), so by default the
    neighbourhood is re-created on the live coordinator each time a step is opened.
    """
    if populate:
        await _neighbourhood(hass, entry)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    return await hass.config_entries.options.async_configure(result["flow_id"], user_input={"next_step_id": step})


async def _neighbourhood(hass, entry):
    """Mine (public MAC, close) plus three of the neighbours' random-address Apple devices (faint)."""
    coordinator = entry.runtime_data.coordinator
    scanner = _make_scanner(coordinator, "11:11:11:11:11:11", "Proxy", "kitchen")
    scanner._is_scanner = True  # so the pickers skip it, as they skip real proxies
    apple = {APPLE: bytes([0x10, 5, 1])}
    mine = _hear(coordinator, "AA:BB:CC:DD:EE:11", rssi=-45, manufacturer_data=apple, name="Mine", scanner=scanner)
    mine.address_type = BDADDR_TYPE_OTHER
    neighbours = []
    for i, rssi in enumerate((-88, -75, -82)):
        device = _hear(coordinator, f"5A:00:00:00:00:{i:02X}", rssi=rssi, manufacturer_data=apple, scanner=scanner)
        device.address_type = BDADDR_TYPE_RANDOM_RESOLVABLE
        neighbours.append(device)
    return coordinator, mine, neighbours


async def test_scan_lists_strongest_first_with_friendly_labels(hass, setup_bermuda_entry):
    _coordinator, mine, _neighbours = await _neighbourhood(hass, setup_bermuda_entry)
    result = await _open(hass, setup_bermuda_entry, "scan")
    add = _field_options(result, "add")
    assert [value for value, _label in add] == [
        "AA:BB:CC:DD:EE:11",
        "5A:00:00:00:00:01",
        "5A:00:00:00:00:02",
        "5A:00:00:00:00:00",
    ]
    assert "Apple device (iPhone, iPad, Mac or Watch)" in add[0][1]
    assert '"Mine"' in add[0][1]
    assert "very close" in add[0][1]
    assert "very weak" in add[-1][1]
    assert _field_options(result, "hide") == add
    assert _field_options(result, "hide_kinds") == [
        ("Apple device (iPhone, iPad, Mac or Watch)", "Apple device (iPhone, iPad, Mac or Watch) (3 nearby)")
    ]


async def test_hiding_a_device_removes_it_everywhere_and_it_can_be_unhidden(hass, setup_bermuda_entry):
    _coordinator, _mine, neighbours = await _neighbourhood(hass, setup_bermuda_entry)
    target = neighbours[1].address.upper()
    result = await _open(hass, setup_bermuda_entry, "scan")
    result = await hass.config_entries.options.async_configure(result["flow_id"], user_input={"hide": [target]})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_bermuda_entry.options[CONF_BLOCKED_DEVICES] == [target]

    result = await _open(hass, setup_bermuda_entry, "scan")
    assert target not in {value for value, _ in _field_options(result, "add")}
    result = await _open(hass, setup_bermuda_entry, "selectdevices")
    assert target not in {value for value, _ in _field_options(result, "devices")}

    result = await _open(hass, setup_bermuda_entry, "blocklist")
    assert [value for value, _ in _field_options(result, "devices")] == [target]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"devices": [], "kinds": []}
    )
    assert setup_bermuda_entry.options[CONF_BLOCKED_DEVICES] == []
    result = await _open(hass, setup_bermuda_entry, "scan")
    assert target in {value for value, _ in _field_options(result, "add")}


async def test_hiding_a_kind_hides_the_rotating_ones_but_not_mine(hass, setup_bermuda_entry):
    _coordinator, _mine, neighbours = await _neighbourhood(hass, setup_bermuda_entry)
    kind = "Apple device (iPhone, iPad, Mac or Watch)"
    result = await _open(hass, setup_bermuda_entry, "scan")
    await hass.config_entries.options.async_configure(result["flow_id"], user_input={"hide_kinds": [kind]})
    assert setup_bermuda_entry.options[CONF_BLOCKED_KINDS] == [kind]

    result = await _open(hass, setup_bermuda_entry, "scan")
    assert [value for value, _ in _field_options(result, "add")] == ["AA:BB:CC:DD:EE:11"]
    assert _field_options(result, "hide_kinds") == []

    result = await _open(hass, setup_bermuda_entry, "blocklist")
    assert _field_options(result, "kinds") == [(kind, kind)]
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"devices": [], "kinds": []}
    )
    assert setup_bermuda_entry.options[CONF_BLOCKED_KINDS] == []


async def test_hide_wins_when_a_device_is_both_added_and_hidden(hass, setup_bermuda_entry):
    _coordinator, _mine, neighbours = await _neighbourhood(hass, setup_bermuda_entry)
    target = neighbours[0].address.upper()
    result = await _open(hass, setup_bermuda_entry, "scan")
    await hass.config_entries.options.async_configure(result["flow_id"], user_input={"add": [target], "hide": [target]})
    assert target not in setup_bermuda_entry.options.get(CONF_DEVICES, [])
    assert setup_bermuda_entry.options[CONF_BLOCKED_DEVICES] == [target]


def test_hiding_a_tracked_address_also_stops_tracking_it():
    from custom_components.bermuda.options_flow import BermudaOptionsFlowHandler  # noqa: PLC0415

    flow = BermudaOptionsFlowHandler()
    flow._options = {CONF_DEVICES: ["AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"]}
    flow._hide(["AA:BB:CC:DD:EE:01"], ["Some kind"])
    assert flow._options[CONF_DEVICES] == ["AA:BB:CC:DD:EE:02"]
    assert flow._options[CONF_BLOCKED_DEVICES] == ["AA:BB:CC:DD:EE:01"]
    assert flow._options[CONF_BLOCKED_KINDS] == ["Some kind"]
    flow._hide(["AA:BB:CC:DD:EE:01"], ["Some kind"])  # hiding twice doesn't duplicate
    assert flow._options[CONF_BLOCKED_DEVICES] == ["AA:BB:CC:DD:EE:01"]
    assert flow._options[CONF_BLOCKED_KINDS] == ["Some kind"]


async def test_blocklist_labels_hidden_devices_that_are_gone(hass, setup_bermuda_entry):
    hass.config_entries.async_update_entry(
        setup_bermuda_entry, options={**setup_bermuda_entry.options, CONF_BLOCKED_DEVICES: ["AA:AA:AA:AA:AA:AA"]}
    )
    await hass.async_block_till_done()
    flow_result = await _open(hass, setup_bermuda_entry, "blocklist", populate=False)
    assert _field_options(flow_result, "devices") == [("AA:AA:AA:AA:AA:AA", "AA:AA:AA:AA:AA:AA (not currently seen)")]
