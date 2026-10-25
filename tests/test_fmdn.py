"""
Tests for Google Find Hub (FMDN) tag recognition.

Covers payload extraction, the GoogleFindMy-HA resolver adapter (against fake
resolver objects, since no Google account is available), the coordinator mixin
that turns rotating MACs into one stable metadevice, and the end-to-end
guarantee that a micro-location spot survives the tag's MAC rotating.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.bermuda.address_type import classify_address
from custom_components.bermuda.bermuda_device import BermudaDevice
from custom_components.bermuda.const import (
    ADDR_TYPE_FMDN_DEVICE,
    DOMAIN,
    DOMAIN_GOOGLEFINDMY,
    FMDN_ADDRESS_PREFIX,
    FMDN_RETRY_INTERVAL,
    METADEVICE_FMDN_DEVICE,
    METADEVICE_TYPE_FMDN_SOURCE,
    MICROLOC_HYSTERESIS_CYCLES,
)
from custom_components.bermuda.fmdn_extraction import extract_eid_candidates, extract_raw_payloads
from custom_components.bermuda.fmdn_resolver import FmdnTag, GoogleFindMyResolver, ResolutionStatus
from tests.test_services import _make_scanner, _mock_ad, _set_distance

FMDN_UUID_LONG = "0000feaa-0000-1000-8000-00805f9b34fb"
EID20 = bytes(range(20))
EID32 = bytes(range(32))
PAYLOAD_20 = b"\x40" + EID20 + b"\x7f"  # frame type, EID, hashed flags
PAYLOAD_32 = b"\x41" + EID32


def _match(device_id: str = "dev1", canonical: str = "canon1", entry: str = "entry1") -> SimpleNamespace:
    """Shape of GoogleFindMy-HA's EIDMatch."""
    return SimpleNamespace(
        device_id=device_id, config_entry_id=entry, canonical_id=canonical, time_offset=0, is_reversed=False
    )


class FakeEidResolver:
    """Stands in for GoogleFindMy-HA's resolver; maps exact byte strings to matches."""

    def __init__(self, table: dict[bytes, list[SimpleNamespace]]) -> None:
        self.table = table
        self.calls: list[bytes] = []

    def resolve_eid_all(self, eid: bytes) -> list[SimpleNamespace]:
        self.calls.append(eid)
        return self.table.get(eid, [])

    def resolve_eid(self, eid: bytes) -> SimpleNamespace | None:  # pragma: no cover - present for interface realism
        matches = self.resolve_eid_all(eid)
        return matches[0] if matches else None


class SingleOnlyResolver:
    """An older resolver without resolve_eid_all."""

    def __init__(self, match: SimpleNamespace | None) -> None:
        self.match = match

    def resolve_eid(self, eid: bytes) -> SimpleNamespace | None:
        return self.match


class FakeTagResolver:
    """Replaces the coordinator's FmdnResolver wholesale (bypasses byte handling)."""

    available = True

    def __init__(self, tags: list[FmdnTag], status: ResolutionStatus = ResolutionStatus.RESOLVED) -> None:
        self.tags = tags
        self.status = status
        self.calls = 0

    def resolve(self, payloads):
        self.calls += 1
        return list(self.tags), self.status


# ---------------------------------------------------------------------------
# extraction + address classification
# ---------------------------------------------------------------------------


def test_extract_raw_payloads_finds_fmdn_by_long_or_short_uuid():
    found = extract_raw_payloads({FMDN_UUID_LONG: PAYLOAD_20, "FEAA": PAYLOAD_32})
    assert found == [PAYLOAD_20, PAYLOAD_32]


def test_extract_raw_payloads_ignores_other_services_and_junk():
    data = {"0000fd5a-0000-1000-8000-00805f9b34fb": PAYLOAD_20, FMDN_UUID_LONG: b"", "feaa": "not bytes"}
    assert extract_raw_payloads(data) == []


def test_extract_eid_candidates_strips_frame_and_flags():
    assert extract_eid_candidates(PAYLOAD_20) == [EID20]
    assert extract_eid_candidates(PAYLOAD_32) == [EID20, EID32]  # 32-byte EIDs also yield the 20-byte prefix
    assert extract_eid_candidates(b"\x40" + EID20[:5]) == []  # truncated


@pytest.mark.parametrize("payload", [b"", b"\x20" + EID20])
def test_extract_eid_candidates_rejects_non_fmdn_frames(payload):
    assert extract_eid_candidates(payload) == []


def test_fmdn_addresses_classify_as_fmdn_devices():
    assert classify_address(f"{FMDN_ADDRESS_PREFIX}abc123") == ADDR_TYPE_FMDN_DEVICE


# ---------------------------------------------------------------------------
# GoogleFindMyResolver (adapter around the other integration)
# ---------------------------------------------------------------------------


def _resolver_with(hass, resolver_obj) -> GoogleFindMyResolver:
    hass.data[DOMAIN_GOOGLEFINDMY] = {"eid_resolver": resolver_obj}
    return GoogleFindMyResolver(hass)


async def test_resolver_unavailable_when_integration_absent(hass):
    resolver = GoogleFindMyResolver(hass)
    assert resolver.available is False
    assert resolver.resolve([PAYLOAD_20]) == ([], ResolutionStatus.RESOLVER_UNAVAILABLE)


@pytest.mark.parametrize("bucket", ["not a dict", {"eid_resolver": None}, {"eid_resolver": object()}, {}])
async def test_resolver_unavailable_for_malformed_bucket(hass, bucket):
    hass.data[DOMAIN_GOOGLEFINDMY] = bucket
    assert GoogleFindMyResolver(hass).resolve([PAYLOAD_20]) == ([], ResolutionStatus.RESOLVER_UNAVAILABLE)


async def test_resolver_is_looked_up_on_every_call(hass):
    resolver = GoogleFindMyResolver(hass)
    assert resolver.available is False
    hass.data[DOMAIN_GOOGLEFINDMY] = {"eid_resolver": FakeEidResolver({})}
    assert resolver.available is True
    del hass.data[DOMAIN_GOOGLEFINDMY]  # integration unloaded
    assert resolver.available is False


async def test_resolver_full_payload_match_short_circuits(hass):
    fake = FakeEidResolver({PAYLOAD_20: [_match()]})
    tags, status = _resolver_with(hass, fake).resolve([PAYLOAD_20])
    assert status is ResolutionStatus.RESOLVED
    assert tags == [FmdnTag("dev1", "canon1", "entry1")]
    assert fake.calls == [PAYLOAD_20]  # never fell back to the bare EID


async def test_resolver_falls_back_to_bare_eid(hass):
    fake = FakeEidResolver({EID20: [_match()]})
    tags, status = _resolver_with(hass, fake).resolve([PAYLOAD_20])
    assert status is ResolutionStatus.RESOLVED
    assert tags[0].device_id == "dev1"
    assert fake.calls == [PAYLOAD_20, EID20]


async def test_resolver_no_match(hass):
    tags, status = _resolver_with(hass, FakeEidResolver({})).resolve([PAYLOAD_20])
    assert (tags, status) == ([], ResolutionStatus.NO_KNOWN_EID_MATCH)


async def test_resolver_shared_tag_gives_one_tag_per_account(hass):
    fake = FakeEidResolver({PAYLOAD_20: [_match("devA", "canon"), _match("devB", "canon"), _match("devA", "canon")]})
    tags, _ = _resolver_with(hass, fake).resolve([PAYLOAD_20])
    assert [t.device_id for t in tags] == ["devA", "devB"]
    assert {t.canonical_id for t in tags} == {"canon"}


async def test_resolver_strips_entry_prefix_from_canonical_id(hass):
    fake = FakeEidResolver({PAYLOAD_20: [_match(canonical="entry1:google-uuid")]})
    tags, _ = _resolver_with(hass, fake).resolve([PAYLOAD_20])
    assert tags[0].canonical_id == "google-uuid"


async def test_resolver_ignores_matches_without_device_id(hass):
    fake = FakeEidResolver({PAYLOAD_20: [_match(device_id="")]})
    assert _resolver_with(hass, fake).resolve([PAYLOAD_20]) == ([], ResolutionStatus.NO_KNOWN_EID_MATCH)


async def test_resolver_supports_older_single_match_api(hass):
    tags, status = _resolver_with(hass, SingleOnlyResolver(_match())).resolve([PAYLOAD_20])
    assert status is ResolutionStatus.RESOLVED
    assert tags[0].device_id == "dev1"
    assert (
        _resolver_with(hass, SingleOnlyResolver(None)).resolve([PAYLOAD_20])[1] is ResolutionStatus.NO_KNOWN_EID_MATCH
    )


async def test_resolver_exception_does_not_propagate(hass):
    class Boom:
        def resolve_eid(self, eid):
            raise RuntimeError("keys not synced")

    assert _resolver_with(hass, Boom()).resolve([PAYLOAD_20]) == ([], ResolutionStatus.RESOLVER_ERROR)


# ---------------------------------------------------------------------------
# coordinator mixin
# ---------------------------------------------------------------------------

TAG = FmdnTag("dev1", "canon1", "entry1")
MAC_A = "c1:00:00:00:00:0a"
MAC_B = "c2:00:00:00:00:0b"
SERVICE_DATA = {FMDN_UUID_LONG: PAYLOAD_20}


def _coordinator(entry):
    return entry.runtime_data.coordinator


def _source(coordinator, mac: str) -> BermudaDevice:
    return coordinator._get_or_create_device(mac)  # noqa: SLF001


async def test_resolved_advert_creates_stable_metadevice(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    coordinator.fmdn_resolver = FakeTagResolver([TAG])
    source = _source(coordinator, MAC_A)

    coordinator.fmdn_handle_advertisement(source, SERVICE_DATA)

    meta = coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}dev1"]
    assert METADEVICE_FMDN_DEVICE in meta.metadevice_type
    assert METADEVICE_TYPE_FMDN_SOURCE in source.metadevice_type
    assert meta.metadevice_sources == [MAC_A]
    assert meta.fmdn_device_id == "dev1"
    assert meta.fmdn_canonical_id == "canon1"
    assert meta.create_sensor is True  # registered with Google on purpose, so always tracked
    assert coordinator.fmdn_stats["resolved"] == 1


async def test_mac_rotation_keeps_one_metadevice_with_both_sources(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    coordinator.fmdn_resolver = FakeTagResolver([TAG])
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_A), SERVICE_DATA)
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_B), SERVICE_DATA)
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_B), SERVICE_DATA)  # repeat: no duplicate

    assert len([a for a in coordinator.metadevices if a.startswith(FMDN_ADDRESS_PREFIX)]) == 1
    assert coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}dev1"].metadevice_sources == [MAC_B, MAC_A]


async def test_shared_tag_gets_one_metadevice_per_account(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    coordinator.fmdn_resolver = FakeTagResolver([TAG, FmdnTag("dev2", "canon1", "entry2")])
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_A), SERVICE_DATA)
    assert {f"{FMDN_ADDRESS_PREFIX}dev1", f"{FMDN_ADDRESS_PREFIX}dev2"} <= set(coordinator.metadevices)


async def test_non_fmdn_advert_is_ignored(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    resolver = coordinator.fmdn_resolver = FakeTagResolver([TAG])
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_A), {"0000fd5a-0000-1000-8000-00805f9b34fb": b"x"})
    assert resolver.calls == 0
    assert not any(a.startswith(FMDN_ADDRESS_PREFIX) for a in coordinator.metadevices)


async def test_unresolved_advert_is_retried_when_resolver_arrives(hass, setup_bermuda_entry, monkeypatch):
    coordinator = _coordinator(setup_bermuda_entry)
    resolver = coordinator.fmdn_resolver = FakeTagResolver([], ResolutionStatus.RESOLVER_UNAVAILABLE)
    scanner = _make_scanner(coordinator, "11:11:11:11:11:11", "Left", "kitchen")
    source = _source(coordinator, MAC_A)
    ad = _mock_ad()
    ad.service_data = SERVICE_DATA
    source.process_advertisement(scanner, ad)  # the advert hook offers it to the (unavailable) resolver
    assert resolver.calls == 1
    assert coordinator.fmdn_diagnostics()["pending_sources"] == 1
    assert not any(a.startswith(FMDN_ADDRESS_PREFIX) for a in coordinator.metadevices)

    # Too soon: no retry.
    coordinator.fmdn_retry_unresolved()
    assert resolver.calls == 1

    # Resolver shows up, enough time passes: retried and linked.
    resolver.tags, resolver.status = [TAG], ResolutionStatus.RESOLVED
    coordinator._fmdn_pending[MAC_A] -= FMDN_RETRY_INTERVAL + 1  # noqa: SLF001
    coordinator.fmdn_retry_unresolved()
    assert f"{FMDN_ADDRESS_PREFIX}dev1" in coordinator.metadevices
    assert coordinator.fmdn_diagnostics()["pending_sources"] == 0


async def test_retry_drops_sources_that_vanished_or_lost_their_adverts(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    coordinator._fmdn_pending["gone"] = 0.0  # noqa: SLF001  # never existed as a device
    ghost = _source(coordinator, MAC_A)
    coordinator.devices[MAC_A] = ghost
    coordinator._fmdn_pending[MAC_A] = 0.0  # noqa: SLF001  # exists but holds no adverts
    coordinator.fmdn_retry_unresolved()
    assert coordinator._fmdn_pending == {}  # noqa: SLF001


async def test_diagnostics_report_resolver_state(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    diag = coordinator.fmdn_diagnostics()
    assert diag["resolver_available"] is False  # GoogleFindMy-HA not loaded in the test hass
    assert diag["tags"] == 0
    coordinator.fmdn_resolver = FakeTagResolver([TAG])
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_A), SERVICE_DATA)
    assert coordinator.fmdn_diagnostics()["tags"] == 1
    assert coordinator.fmdn_diagnostics()["resolver_available"] is True


async def test_device_process_service_data_only_offers_fmdn(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    resolver = coordinator.fmdn_resolver = FakeTagResolver([TAG])
    scanner = _make_scanner(coordinator, "11:11:11:11:11:11", "Left", "kitchen")
    source = _source(coordinator, MAC_A)

    other = _mock_ad()
    other.service_data = {"0000fd5a-0000-1000-8000-00805f9b34fb": b"x"}
    source.process_advertisement(scanner, other)
    assert resolver.calls == 0

    fmdn = _mock_ad(-61)  # a different rssi, otherwise a stamp-less scanner treats it as nothing new
    fmdn.service_data = SERVICE_DATA
    source.process_advertisement(scanner, fmdn)
    assert resolver.calls == 1  # the advert hook fed the resolver
    assert f"{FMDN_ADDRESS_PREFIX}dev1" in coordinator.metadevices

    source.process_service_data(SimpleNamespace(service_data=[]))  # no service data at all: no-op
    assert resolver.calls == 1


# ---------------------------------------------------------------------------
# discovery from the GoogleFindMy-HA registry entries
# ---------------------------------------------------------------------------


async def test_discovery_seeds_metadevices_and_grabs_names(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    gfm_entry = MockConfigEntry(domain=DOMAIN_GOOGLEFINDMY, entry_id="gfm1")
    gfm_entry.add_to_hass(hass)
    devreg = dr.async_get(hass).async_get_or_create(
        config_entry_id="gfm1", identifiers={(DOMAIN_GOOGLEFINDMY, "gfm1:canon-xyz")}, name="Keys tag"
    )
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    er.async_get(hass).async_get_or_create(
        "device_tracker", DOMAIN_GOOGLEFINDMY, "keys", config_entry=gfm_entry, device_id=devreg.id
    )
    # a tracker with no device and a device with no registry entry are skipped quietly
    er.async_get(hass).async_get_or_create("device_tracker", DOMAIN_GOOGLEFINDMY, "orphan", config_entry=gfm_entry)
    er.async_get(hass).async_get_or_create("sensor", DOMAIN_GOOGLEFINDMY, "battery", config_entry=gfm_entry)

    coordinator._do_fmdn_init = True  # noqa: SLF001
    coordinator.discover_fmdn_metadevices()

    meta = coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}{devreg.id}"]
    assert meta.name == "Keys tag"
    assert meta.fmdn_canonical_id == "canon-xyz"
    assert meta.create_sensor is True
    # one-shot until flagged again
    assert coordinator._do_fmdn_init is False  # noqa: SLF001
    coordinator.metadevices.clear()
    coordinator.discover_fmdn_metadevices()
    assert coordinator.metadevices == {}


async def test_entities_attach_to_the_google_device(hass, setup_bermuda_entry):
    from custom_components.bermuda.entity import BermudaEntity  # noqa: PLC0415

    coordinator = _coordinator(setup_bermuda_entry)
    gfm_entry = MockConfigEntry(domain=DOMAIN_GOOGLEFINDMY, entry_id="gfm1")
    gfm_entry.add_to_hass(hass)
    devreg = dr.async_get(hass).async_get_or_create(
        config_entry_id="gfm1", identifiers={(DOMAIN_GOOGLEFINDMY, "gfm1:canon")}, name="Keys tag"
    )
    coordinator.fmdn_resolver = FakeTagResolver([FmdnTag(devreg.id, "canon", "gfm1")])
    coordinator.fmdn_handle_advertisement(_source(coordinator, MAC_A), SERVICE_DATA)
    meta = coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}{devreg.id}"]

    entity = BermudaEntity(coordinator, setup_bermuda_entry, meta.address)
    assert entity.device_info["identifiers"] == {(DOMAIN_GOOGLEFINDMY, "gfm1:canon")}
    assert "name" not in entity.device_info  # must not rename the Google device

    # If the Google device is gone, fall back to a device of our own.
    meta.fmdn_device_id = "no-such-device"
    fallback = entity.device_info
    assert fallback["identifiers"] == {(DOMAIN, meta.unique_id)}


# ---------------------------------------------------------------------------
# micro-locations survive the MAC rotating (the point of all this)
# ---------------------------------------------------------------------------


async def test_micro_location_survives_mac_rotation(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    await coordinator.fingerprints.async_load()
    coordinator.fmdn_resolver = FakeTagResolver([TAG])
    s1 = _make_scanner(coordinator, "11:11:11:11:11:11", "Kitchen Left", "kitchen")
    s2 = _make_scanner(coordinator, "22:22:22:22:22:22", "Kitchen Right", "kitchen")

    # The tag is heard under MAC A.
    src_a = _source(coordinator, MAC_A)
    for scanner, distance in ((s1, 1.0), (s2, 5.0)):
        _set_distance(src_a, scanner, distance)
    coordinator.fmdn_handle_advertisement(src_a, SERVICE_DATA)
    coordinator.update_metadevices()
    meta = coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}dev1"]

    # The user calibrates by the MAC it has right now: the spot lands on the stable identity.
    assert coordinator._resolve_device(MAC_A) is meta  # noqa: SLF001
    coordinator.calibrate_location(coordinator._resolve_device(MAC_A), "Key hook")  # noqa: SLF001
    assert coordinator.fingerprints.list(meta.address)
    assert not coordinator.fingerprints.list(MAC_A)

    # ~17 minutes later the tag rotates to MAC B at the same spot; MAC A goes silent.
    for advert in src_a.adverts.values():
        advert.rssi_distance = None
    src_b = _source(coordinator, MAC_B)
    for scanner, distance in ((s1, 1.1), (s2, 4.9)):
        _set_distance(src_b, scanner, distance)
    coordinator.fmdn_handle_advertisement(src_b, SERVICE_DATA)
    coordinator.update_metadevices()
    for _ in range(MICROLOC_HYSTERESIS_CYCLES):
        coordinator._refresh_microlocations()  # noqa: SLF001

    assert meta.micro_location_name == "Key hook"
    assert src_b.micro_location_name is None  # sources are never tracked in their own right

    # where_is accepts either MAC or the Google device and answers for the same tag.
    for ref in (MAC_B, MAC_A, meta.fmdn_device_id):
        where = await hass.services.async_call(DOMAIN, "where_is", {"device": ref}, blocking=True, return_response=True)
        assert where["device_address"] == meta.address
        assert where["micro_location"] == "Key hook"


async def test_resolve_device_passes_ordinary_devices_through(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    plain = _source(coordinator, "ec:00:00:00:00:99")
    coordinator.devices[plain.address] = plain
    assert coordinator._resolve_device("ec:00:00:00:00:99") is plain  # noqa: SLF001
    assert coordinator._resolve_device("no such thing") is None  # noqa: SLF001
    # a source whose tag metadevice has gone is returned as itself
    orphan = _source(coordinator, MAC_A)
    orphan.metadevice_type.add(METADEVICE_TYPE_FMDN_SOURCE)
    coordinator.devices[MAC_A] = orphan
    assert coordinator.fmdn_redirect(orphan) is orphan


async def test_adverts_of_dropped_sources_do_not_pile_up_on_the_tag(hass, setup_bermuda_entry):
    coordinator = _coordinator(setup_bermuda_entry)
    coordinator.fmdn_resolver = FakeTagResolver([TAG])
    scanner = _make_scanner(coordinator, "11:11:11:11:11:11", "Left", "kitchen")
    for mac in (MAC_A, MAC_B):
        source = _source(coordinator, mac)
        _set_distance(source, scanner, 2.0)
        coordinator.fmdn_handle_advertisement(source, SERVICE_DATA)
    coordinator.update_metadevices()
    meta = coordinator.metadevices[f"{FMDN_ADDRESS_PREFIX}dev1"]
    assert {key[0] for key in meta.adverts} == {MAC_A, MAC_B}

    # Pruning forgets the old MAC (it drops it from the sources); its copied advert must go too.
    meta.metadevice_sources.remove(MAC_A)
    coordinator.update_metadevices()
    assert {key[0] for key in meta.adverts} == {MAC_B}
