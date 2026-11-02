"""
Tests for the micro-location RF fingerprinting engine and store.

The engine maths (Fingerprint, FingerprintMatcher) is pure Python and tested
directly. FingerprintStore is exercised against Home Assistant's Store via the
``hass`` fixture from pytest-homeassistant-custom-component.
"""

from __future__ import annotations

import pytest

from custom_components.bermuda.location_fingerprints import (
    Fingerprint,
    FingerprintMatcher,
    FingerprintStore,
)

# ---------------------------------------------------------------------------
# Fingerprint (de)serialisation
# ---------------------------------------------------------------------------


def test_fingerprint_roundtrip():
    """A fingerprint survives a to_dict/from_dict round-trip intact."""
    fingerprint = Fingerprint(
        name="Key hook",
        device_address="aa:bb",
        vector={"s1": 1.0, "s2": 3.0},
        vector_std={"s1": 0.1},
        rssi_vector={"s1": -55.0},
        area_id="kitchen",
        floor_id="ground",
        sample_count=7,
    )
    restored = Fingerprint.from_dict(fingerprint.to_dict())
    assert restored.id == fingerprint.id
    assert restored.name == "Key hook"
    assert restored.device_address == "aa:bb"
    assert restored.vector == {"s1": 1.0, "s2": 3.0}
    assert restored.vector_std == {"s1": 0.1}
    assert restored.rssi_vector == {"s1": -55.0}
    assert restored.area_id == "kitchen"
    assert restored.floor_id == "ground"
    assert restored.sample_count == 7


def test_fingerprint_from_dict_tolerates_missing_keys():
    """from_dict fills sensible defaults and coerces types."""
    fingerprint = Fingerprint.from_dict({"name": "x", "device_address": "aa", "vector": {"s1": "2"}})
    assert fingerprint.vector == {"s1": 2.0}
    assert fingerprint.sample_count == 1
    assert isinstance(fingerprint.id, str) and fingerprint.id


# ---------------------------------------------------------------------------
# FingerprintMatcher
# ---------------------------------------------------------------------------


def _hook() -> Fingerprint:
    return Fingerprint(
        name="Key hook", device_address="d", vector={"s1": 1.0, "s2": 5.0}, vector_std={"s1": 0.1, "s2": 0.1}
    )


def _drawer() -> Fingerprint:
    return Fingerprint(
        name="Drawer", device_address="d", vector={"s1": 5.0, "s2": 1.0}, vector_std={"s1": 0.1, "s2": 0.1}
    )


def test_score_identical_is_near_zero():
    matcher = FingerprintMatcher()
    assert matcher.score_one({"s1": 1.0, "s2": 5.0}, _hook()) < 0.01


def test_score_missing_and_extra_scanner_penalised():
    matcher = FingerprintMatcher()
    fingerprint = _hook()
    baseline = matcher.score_one({"s1": 1.0, "s2": 5.0}, fingerprint)
    missing = matcher.score_one({"s1": 1.0}, fingerprint)  # s2 calibrated but now silent
    extra = matcher.score_one({"s1": 1.0, "s2": 5.0, "s3": 2.0}, fingerprint)  # heard a proxy the spot lacks
    assert missing > baseline
    assert extra > baseline


def test_score_empty_vector_returns_reject_distance():
    matcher = FingerprintMatcher()
    empty = Fingerprint(name="x", device_address="d", vector={})
    assert matcher.score_one({}, empty) == matcher.reject_distance


def test_match_picks_nearest_and_accepts():
    matcher = FingerprintMatcher()
    result = matcher.match({"s1": 1.1, "s2": 4.8}, [_hook(), _drawer()])
    assert result is not None
    assert result.name == "Key hook"
    assert result.accepted is True
    assert 0 < result.confidence <= 1
    assert result.scores[0][1] == "Key hook"  # best-first ordering
    assert result.second_score is not None and result.second_score > result.score


def test_match_ambiguous_is_less_confident_than_clear():
    matcher = FingerprintMatcher()
    ambiguous = matcher.match({"s1": 3.0, "s2": 3.0}, [_hook(), _drawer()])
    clear = matcher.match({"s1": 1.0, "s2": 5.0}, [_hook(), _drawer()])
    assert ambiguous.confidence < clear.confidence


def test_match_far_is_rejected():
    matcher = FingerprintMatcher()
    result = matcher.match({"s1": 40.0, "s2": 40.0}, [_hook()])
    assert result.accepted is False


def test_match_single_candidate_has_no_second():
    matcher = FingerprintMatcher()
    result = matcher.match({"s1": 1.0, "s2": 5.0}, [_hook()])
    assert result.second_score is None
    assert result.accepted is True
    assert result.confidence > 0


def test_match_accepts_on_margin_when_outside_accept_band():
    """A best that's past accept_distance can still win on a clear margin."""
    matcher = FingerprintMatcher(accept_distance=0.5, reject_distance=10.0, min_margin=0.4)
    result = matcher.match({"s1": 1.8, "s2": 5.2}, [_hook(), _drawer()])
    assert result.name == "Key hook"
    assert result.score > matcher.accept_distance  # past the easy band...
    assert result.accepted is True  # ...but the margin carries it


def test_match_empty_inputs_return_none():
    matcher = FingerprintMatcher()
    assert matcher.match({}, [_hook()]) is None
    assert matcher.match({"s1": 1.0}, []) is None


# ---------------------------------------------------------------------------
# FingerprintStore (requires hass for the Store helper)
# ---------------------------------------------------------------------------


async def test_store_load_empty(hass):
    store = FingerprintStore(hass)
    await store.async_load()
    assert store.loaded is True
    assert store.list() == []


async def test_store_add_list_find_rename_remove(hass):
    store = FingerprintStore(hass)
    await store.async_load()
    fingerprint = Fingerprint(name="Key hook", device_address="aa:bb", vector={"s1": 1.0})
    store.add(fingerprint)

    assert store.list("aa:bb") == [fingerprint]
    assert store.list("somewhere-else") == []
    assert store.get(fingerprint.id) is fingerprint
    assert store.find_by_name("KEY hook") is fingerprint  # case-insensitive, house-wide
    assert store.find_by_name("  key hook ", "aa:bb") is fingerprint  # optionally limited to who marked it
    assert store.find_by_name("key hook", "somewhere-else") is None
    assert store.find_by_name("nope") is None

    assert store.rename(fingerprint.id, "Coat hook") is True
    assert fingerprint.name == "Coat hook"
    assert store.rename("does-not-exist", "x") is False

    assert store.remove(fingerprint.id) is True
    assert store.remove(fingerprint.id) is False
    assert store.list() == []


async def test_store_persists_across_reload(hass):
    store = FingerprintStore(hass)
    await store.async_load()
    fingerprint = Fingerprint(
        name="Hook", device_address="aa:bb", vector={"s1": 1.0, "s2": 2.0}, vector_std={"s1": 0.1}
    )
    store.add(fingerprint)
    # Persist immediately (the real store debounces) then load into a fresh store.
    await store._store.async_save(store._data_to_save())  # noqa: SLF001 - test seam

    reloaded = FingerprintStore(hass)
    await reloaded.async_load()
    loaded = reloaded.get(fingerprint.id)
    assert loaded is not None
    assert loaded.name == "Hook"
    assert loaded.vector == {"s1": 1.0, "s2": 2.0}
    assert reloaded.list("aa:bb") == [loaded]


async def test_store_load_skips_malformed_rows(hass):
    store = FingerprintStore(hass)
    await store._store.async_save(  # noqa: SLF001 - seed bad + good rows
        {"fingerprints": [{"bad": "row"}, {"name": "ok", "device_address": "d", "vector": {"s1": 1.0}}]}
    )
    await store.async_load()
    saved = store.list()
    assert len(saved) == 1
    assert saved[0].name == "ok"


# ---------------------------------------------------------------------------
# match_rssi: places are matched on raw signal strength, not distances
# ---------------------------------------------------------------------------

from custom_components.bermuda.calibration import attenuation_warning  # noqa: E402
from custom_components.bermuda.location_fingerprints import RSSI_ACCEPT_SCORE  # noqa: E402

DESK = {"s1": -80.0, "s2": -87.0, "s3": -93.0, "s4": -93.0, "s5": -100.0}


def _place(name: str, readings: dict[str, float], std: float | None = 2.0) -> Fingerprint:
    return Fingerprint(
        name=name,
        device_address="d",
        vector={},
        rssi_vector=dict(readings),
        rssi_std={s: std for s in readings} if std is not None else {},
    )


def test_a_place_is_recognised_by_its_pattern():
    desk = _place("Desk", DESK)
    result = FingerprintMatcher().match_rssi({s: v + 1.0 for s, v in DESK.items()}, [desk])
    assert result.accepted
    assert result.name == "Desk"
    assert result.score < RSSI_ACCEPT_SCORE


@pytest.mark.parametrize("offset", [-15.0, -6.0, 8.0])
def test_a_louder_or_quieter_device_matches_the_same_place(offset):
    sofa = _place("Sofa", {"s1": -60.0, "s2": -75.0, "s3": -88.0})
    desk = _place("Desk", {"s1": -85.0, "s2": -70.0, "s3": -62.0})
    result = FingerprintMatcher().match_rssi(
        {s: v + offset for s, v in {"s1": -60.0, "s2": -75.0, "s3": -88.0}.items()}, [sofa, desk]
    )
    assert result.accepted
    assert result.name == "Sofa"


def test_a_lone_place_is_not_matched_from_a_different_spot():
    """The reported bug: only 'Desktop' was marked and it matched from the sofa with full confidence."""
    desk = _place("Desktop", DESK)
    sofa_reading = {"s1": -93.0, "s2": -78.0, "s3": -87.0, "s4": -100.0, "s5": -93.0}  # a different pattern
    result = FingerprintMatcher().match_rssi(sofa_reading, [desk])
    assert not result.accepted
    assert result.score > RSSI_ACCEPT_SCORE


def test_two_similar_places_are_not_claimed():
    a = _place("Left of table", {"s1": -60.0, "s2": -75.0})
    b = _place("Right of table", {"s1": -60.5, "s2": -75.5})
    assert not FingerprintMatcher().match_rssi({"s1": -60.0, "s2": -75.0}, [a, b]).accepted


def test_a_place_saved_before_spreads_were_recorded_still_matches():
    legacy = _place("Old", DESK, std=None)
    assert FingerprintMatcher().match_rssi(dict(DESK), [legacy]).accepted


def test_one_shared_proxy_is_not_enough_for_a_place():
    desk = _place("Desk", DESK)
    result = FingerprintMatcher().match_rssi({"s1": -80.0}, [desk])
    assert not result.accepted


def test_match_rssi_has_no_answer_without_readings_or_places():
    matcher = FingerprintMatcher()
    assert matcher.match_rssi({}, [_place("Desk", DESK)]) is None
    assert matcher.match_rssi(dict(DESK), []) is None


def test_rssi_spread_roundtrips():
    restored = Fingerprint.from_dict(_place("Desk", DESK, std=3.0).to_dict())
    assert restored.rssi_std["s1"] == 3.0


def test_a_too_low_fade_rate_is_called_out():
    assert attenuation_warning(2.4) is None
    assert attenuation_warning(1.5) is None
    assert "1.05" in attenuation_warning(1.05)


def test_a_place_already_held_is_kept_through_a_fair_match():
    from custom_components.bermuda.location_fingerprints import RSSI_HOLD_SCORE

    desk = _place("Desk", {"s1": -60.0, "s2": -75.0, "s3": -85.0})
    drift = {"s1": -60.0, "s2": -71.0, "s3": -89.0}  # 4 dB off on two proxies: a fair, not good, fit
    matcher = FingerprintMatcher()
    fresh = matcher.match_rssi(drift, [desk])
    assert fresh.score > RSSI_ACCEPT_SCORE
    assert not fresh.accepted  # not good enough to get in...
    held = matcher.match_rssi(drift, [desk], holding=desk.id)
    assert held.score <= RSSI_HOLD_SCORE
    assert held.accepted  # ...but good enough to stay


def test_a_held_place_is_let_go_when_the_fit_is_poor_or_another_place_wins():
    desk = _place("Desk", {"s1": -60.0, "s2": -75.0, "s3": -85.0})
    sofa = _place("Sofa", {"s1": -85.0, "s2": -62.0, "s3": -60.0})
    matcher = FingerprintMatcher()
    far = {"s1": -95.0, "s2": -60.0, "s3": -70.0}
    assert not matcher.match_rssi(far, [desk], holding=desk.id).accepted  # poor fit: let go
    won = matcher.match_rssi({"s1": -86.0, "s2": -62.0, "s3": -61.0}, [desk, sofa], holding=desk.id)
    assert won.accepted
    assert won.name == "Sofa"  # a clearly better place beats the one being held
    assert matcher.match_rssi(dict({"s1": -60.0, "s2": -75.0, "s3": -85.0}), [desk], holding="gone").accepted
