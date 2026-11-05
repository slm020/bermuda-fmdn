"""
Tests for confidence-triggered room learning.

The maths (ConfidenceTracker, RoomProfile, match_rooms) is pure and tested
directly. Trained rooms are house-wide (any device teaches, every device is
matched, whatever its loudness). The coordinator behaviour (prompt when poor,
train, apply trained rooms, reset) is driven through a loaded entry with a
hand-built tracked device and a controllable clock.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import slugify

from tests.flow_helpers import goto_setting
from custom_components.bermuda.const import (
    CONF_LEARNING_ENABLED,
    CONF_LEARNING_SUSTAINED,
    CONF_LEARNING_THRESHOLD,
    DOMAIN,
    LEARN_INTERVAL,
    LEARN_MAX_SECONDS,
    LEARN_PROMPT_QUIET,
    LEARN_SAMPLES,
    REPAIR_TRAIN_PREFIX,
    SIGNAL_DEVICE_NEW,
)
from custom_components.bermuda.repairs import async_create_fix_flow
from custom_components.bermuda.room_learning import (
    MAX_SPOTS_PER_ROOM,
    STORAGE_KEY,
    STORAGE_VERSION,
    ConfidenceTracker,
    ProfileStore,
    RoomProfile,
    match_rooms,
)
from tests.test_services import _make_scanner, _make_tracked_device, _set_distance

# ---------------------------------------------------------------------------
# ConfidenceTracker
# ---------------------------------------------------------------------------


def _run(tracker: ConfidenceTracker, script, *, step: float = 1.0, start: float = 0.0):
    """
    Feed a tracker like the real coordinator: one cycle per ``step`` seconds.

    ``script`` is a list of (area, floor, seconds_in_that_state, margin); the area stays
    constant for that many seconds (long runs, as real hysteresis gives), cycling every second.
    """
    now = start
    for area, floor, seconds, margin in script:
        for _ in range(int(seconds / step)):
            tracker.add(now, area, floor, margin)
            now += step
    return tracker.result(now), now


def test_confidence_is_untroubled_until_half_a_window_has_been_observed():
    result, _ = _run(ConfidenceTracker(300), [("a", "f", 20, 0.5), ("b", "f", 20, 0.5), ("a", "f", 20, 0.5)])
    assert result.score == 1.0  # 60s < 150s, even though it flipped
    assert result.top_areas == ()


def test_stable_room_has_full_confidence():
    result, _ = _run(ConfidenceTracker(300), [("kitchen", "ground", 400, 2.0)])
    assert result.score == 1.0
    assert result.top_areas == ("kitchen",)


def test_flapping_with_realistic_dwell_is_poor_confidence():
    # With hysteresis a hard-to-place device holds each answer for 20-30 s: AAAA BBBB AAAA BBBB ...
    script = [("lounge" if i % 2 else "hall", "ground", 25, 0.5) for i in range(14)]
    result, _ = _run(ConfidenceTracker(300), script)
    assert result.flapping == pytest.approx(1.0)
    assert result.score < 0.5
    assert set(result.top_areas) == {"lounge", "hall"}


def test_one_trip_to_the_next_room_and_back_is_not_flapping():
    result, _ = _run(
        ConfidenceTracker(300),
        [("kitchen", "f", 120, 1.0), ("lounge", "f", 150, 1.0), ("kitchen", "f", 130, 1.0)],
    )
    assert result.flapping == 0.0  # the detour lasted 150 s: someone moved, nothing flapped
    assert result.score == 1.0


def test_walking_through_rooms_is_not_flapping():
    result, _ = _run(
        ConfidenceTracker(300),
        [("a", "f", 100, 1.0), ("b", "f", 100, 1.0), ("c", "f", 100, 1.0), ("d", "f", 100, 1.0)],
    )
    assert result.flapping == 0.0
    assert result.score == 1.0


def test_near_equal_readings_alone_never_trigger_a_prompt():
    result, _ = _run(ConfidenceTracker(300), [("hall", "f", 400, 0.05)])
    assert result.near_equal == 1.0
    assert result.score == pytest.approx(0.6)  # stable but ambiguous: above the 0.5 default threshold


def test_floor_bouncing_is_poor_confidence():
    script = [("a", "up" if i % 2 else "down", 20, None) for i in range(16)]
    result, _ = _run(ConfidenceTracker(300), script)
    assert result.floor_bounce == pytest.approx(1.0)
    assert result.score < 0.5


def test_old_history_ages_out_of_the_window():
    tracker = ConfidenceTracker(300)
    _result, now = _run(tracker, [("lounge" if i % 2 else "hall", "f", 25, 0.5) for i in range(14)])
    assert tracker.result(now).score < 0.5
    settled, _ = _run(tracker, [("hall", "f", 400, 2.0)], start=now)
    assert settled.score == 1.0


def test_clear_forgets_the_history():
    tracker = ConfidenceTracker(300)
    _result, now = _run(tracker, [("a" if i % 2 else "b", "f", 25, None) for i in range(14)])
    tracker.clear()
    assert tracker.result(now).score == 1.0


# ---------------------------------------------------------------------------
# RoomProfile + match_rooms
# ---------------------------------------------------------------------------


def _profile(area: str, readings: dict[str, float], jitter: float = 1.5, n: int = 12) -> RoomProfile:
    """A trained room: each scanner reads around its mean, +/- jitter."""
    profile = RoomProfile(area)
    for i in range(n):
        sign = 1 if i % 2 else -1
        profile.add_sample({s: rssi + sign * jitter for s, rssi in readings.items()})
    return profile


def test_profile_roundtrip_preserves_statistics():
    original = _profile("kitchen", {"s1": -55.0, "s2": -70.0})
    restored = RoomProfile.from_dict(original.to_dict())
    assert restored.area_id == "kitchen"
    assert restored.samples == original.samples
    assert restored.stats["s1"].mean == pytest.approx(original.stats["s1"].mean)
    assert restored.stats["s2"].variance == pytest.approx(original.stats["s2"].variance)


@pytest.mark.parametrize("raw", [{}, {"area_id": "x"}, {"area_id": "x", "scanners": {"s": [1, 2]}}])
def test_profile_from_dict_rejects_malformed_data(raw):
    with pytest.raises((KeyError, TypeError, ValueError)):
        RoomProfile.from_dict(raw)


def test_single_sample_variance_is_floored():
    profile = RoomProfile("x")
    profile.add_sample({"s1": -60.0})
    assert profile.stats["s1"].variance == 4.0


def _rooms(*profiles: RoomProfile) -> dict[str, list[RoomProfile]]:
    rooms: dict[str, list[RoomProfile]] = {}
    for profile in profiles:
        rooms.setdefault(profile.area_id, []).append(profile)
    return rooms


def test_match_picks_the_room_the_readings_look_like():
    rooms = _rooms(
        _profile("kitchen", {"s1": -50.0, "s2": -75.0}),
        _profile("hall", {"s1": -75.0, "s2": -50.0}),
    )
    match = match_rooms({"s1": -51.0, "s2": -74.0}, rooms)
    assert match.area_id == "kitchen"
    assert [a for a, _s in match.scores] == ["kitchen", "hall"]
    assert match.best_score < 2.0


def test_match_refuses_a_poor_fit():
    rooms = _rooms(_profile("kitchen", {"s1": -50.0, "s2": -75.0}))
    match = match_rooms({"s1": -90.0, "s2": -40.0}, rooms)
    assert match.area_id is None
    assert match.best_score > 6.0


def test_match_refuses_when_two_rooms_are_too_alike():
    rooms = _rooms(
        _profile("kitchen", {"s1": -50.0, "s2": -70.0}),
        _profile("utility", {"s1": -50.5, "s2": -70.5}),
    )
    assert match_rooms({"s1": -50.0, "s2": -70.0}, rooms).area_id is None


def test_match_penalises_a_silent_scanner_that_should_be_heard():
    rooms = _rooms(_profile("kitchen", {"s1": -50.0, "s2": -60.0, "s3": -65.0}))
    heard_all = match_rooms({"s1": -50.0, "s2": -60.0, "s3": -65.0}, rooms)
    one_missing = match_rooms({"s1": -50.0, "s2": -60.0}, rooms)
    assert one_missing.best_score > heard_all.best_score


def test_match_with_nothing_usable_gives_no_answer():
    thin = RoomProfile("x")
    thin.add_sample({"s1": -50.0, "s2": -60.0})  # a single sample is below MIN_SCANNER_SAMPLES
    assert match_rooms({"s1": -50.0, "s2": -60.0}, _rooms(thin)).area_id is None
    assert match_rooms({"s1": -50.0}, {}).best_score == math.inf


def test_one_shared_scanner_says_nothing():
    """With a single scanner in common the fitted loudness would absorb everything: never a match."""
    rooms = _rooms(_profile("kitchen", {"s1": -50.0, "s2": -75.0}))
    assert match_rooms({"s1": -50.0}, rooms).area_id is None
    assert match_rooms({"s1": -50.0, "other": -60.0}, rooms).best_score == math.inf


ROOMS_3 = {"kitchen": {"s1": -50.0, "s2": -68.0, "s3": -82.0}, "hall": {"s1": -80.0, "s2": -62.0, "s3": -50.0}}


@pytest.mark.parametrize("offset", [0.0, -6.0, -15.0, 8.0])
def test_a_louder_or_quieter_device_matches_the_same_room(offset):
    """The requirement: transmit power must not change the answer (all scanners shift together)."""
    rooms = _rooms(*(_profile(a, r) for a, r in ROOMS_3.items()))
    for area, readings in ROOMS_3.items():
        match = match_rooms({s: v + offset for s, v in readings.items()}, rooms)
        assert match.area_id == area
        assert match.offset_db == pytest.approx(offset, abs=0.5)
        assert match.best_score < 2.0


def test_a_quiet_device_that_loses_its_far_scanner_still_matches():
    """A -15 dB device is simply not heard by the farthest proxy: that silence is expected, not evidence."""
    rooms = _rooms(*(_profile(a, r) for a, r in ROOMS_3.items()))
    quiet = {s: v - 15.0 for s, v in ROOMS_3["kitchen"].items() if s != "s3"}  # s3 would be -97: below hearing
    match = match_rooms(quiet, rooms)
    assert match.area_id == "kitchen"
    assert match.best_score < 2.0


def test_a_silent_scanner_that_should_be_loud_still_counts_against():
    rooms = _rooms(*(_profile(a, r) for a, r in ROOMS_3.items()))
    fine = match_rooms(dict(ROOMS_3["kitchen"]), rooms)
    no_s1 = match_rooms({s: v for s, v in ROOMS_3["kitchen"].items() if s != "s1"}, rooms)  # s1 at -50 would be heard
    assert no_s1.best_score > fine.best_score + 1.0


def test_the_best_spot_in_a_room_is_used():
    sofa = _profile("living", {"s1": -50.0, "s2": -80.0, "s3": -70.0})
    desk = _profile("living", {"s1": -70.0, "s2": -55.0, "s3": -60.0})
    rooms = _rooms(sofa, desk, _profile("bed", {"s1": -85.0, "s2": -70.0, "s3": -50.0}))
    assert match_rooms({"s1": -69.0, "s2": -56.0, "s3": -61.0}, rooms).area_id == "living"
    assert match_rooms({"s1": -51.0, "s2": -79.0, "s3": -71.0}, rooms).area_id == "living"


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------


async def test_store_roundtrip_spots_and_forget(hass, hass_storage):
    store = ProfileStore(hass)
    await store.async_load()
    assert store.loaded
    store.add_spot(_profile("kitchen", {"s1": -50.0}))
    store.add_spot(_profile("kitchen", {"s1": -52.0}))
    store.add_spot(_profile("hall", {"s1": -70.0}))
    assert sorted(store.rooms()) == ["hall", "kitchen"]
    assert len(store.spots("kitchen")) == 2

    hass_storage[STORAGE_KEY] = {"version": STORAGE_VERSION, "key": STORAGE_KEY, "data": store._data_to_save()}  # noqa: SLF001
    fresh = ProfileStore(hass)
    await fresh.async_load()
    assert sorted(fresh.rooms()) == ["hall", "kitchen"]
    assert len(fresh.spots("kitchen")) == 2
    assert fresh.spots("kitchen")[0].stats["s1"].n == 12

    assert fresh.forget("kitchen") is True
    assert fresh.forget("kitchen") is False
    assert sorted(fresh.rooms()) == ["hall"]
    assert fresh.forget() is True
    assert fresh.forget() is False
    assert fresh.rooms() == {}


async def test_spots_are_named_retrained_by_name_and_forgotten_one_at_a_time(hass):
    store = ProfileStore(hass)
    for label, rssi in (("Sofa", -50.0), ("Door", -60.0), ("", -70.0), ("", -71.0)):
        profile = _profile("living", {"s1": rssi})
        profile.label = label
        store.add_spot(profile)
    assert store.labels() == {"living": ["Sofa", "Door", "Spot 3", "Spot 4"]}

    again = _profile("living", {"s1": -55.0})
    again.label = "sofa"  # same name, any case: retrains that spot in place
    store.add_spot(again)
    assert store.labels() == {"living": ["sofa", "Door", "Spot 3", "Spot 4"]}
    assert store.spots("living")[0].stats["s1"].mean == pytest.approx(-55.0)

    assert store.forget("living", "DOOR") is True
    assert store.forget("living", "door") is False
    assert store.labels() == {"living": ["sofa", "Spot 3", "Spot 4"]}
    for label in ("sofa", "Spot 3", "Spot 4"):
        store.forget("living", label)
    assert store.rooms() == {}  # a room with no spots left is gone


def test_labels_survive_the_roundtrip_and_old_data_gets_none():
    profile = _profile("kitchen", {"s1": -50.0})
    profile.label = "table"
    assert RoomProfile.from_dict(profile.to_dict()).label == "table"
    legacy = profile.to_dict()
    del legacy["label"]
    assert RoomProfile.from_dict(legacy).label == ""


async def test_store_replace_and_spot_cap(hass):
    store = ProfileStore(hass)
    for i in range(MAX_SPOTS_PER_ROOM + 3):
        store.add_spot(_profile("kitchen", {"s1": -50.0 - i}))
    spots = store.spots("kitchen")
    assert len(spots) == MAX_SPOTS_PER_ROOM
    assert spots[-1].stats["s1"].mean == pytest.approx(-50.0 - (MAX_SPOTS_PER_ROOM + 2))  # the newest survive
    store.add_spot(_profile("kitchen", {"s1": -60.0}), replace=True)
    assert len(store.spots("kitchen")) == 1
    assert store.spots("nowhere") == []


async def test_spots_saved_before_names_existed_get_names_on_load(hass, hass_storage):
    old = _profile("living", {"s1": -50.0}).to_dict()
    del old["label"]  # what the previous build saved
    named = _profile("living", {"s1": -60.0})
    named.label = "Spot 1"  # already took the first free name
    hass_storage[STORAGE_KEY] = {
        "version": STORAGE_VERSION,
        "key": STORAGE_KEY,
        "data": {"rooms": {"living": [old, named.to_dict(), {**old}]}},
    }
    store = ProfileStore(hass)
    await store.async_load()
    labels = store.labels()["living"]
    assert labels[1] == "Spot 1"
    assert len({label.casefold() for label in labels}) == 3  # unique, none empty
    assert all(labels)
    assert store.forget("living", labels[0]) is True  # and each can now be forgotten alone


async def test_the_oldest_spot_is_dropped_and_reported_when_a_room_is_full(hass):
    store = ProfileStore(hass)
    for i in range(MAX_SPOTS_PER_ROOM):
        profile = _profile("living", {"s1": -50.0 - i})
        profile.label = f"Wall {i}"
        assert store.add_spot(profile) is None
    extra = _profile("living", {"s1": -90.0})
    extra.label = "Corner"
    assert store.add_spot(extra) == "Wall 0"
    assert "Wall 0" not in store.labels()["living"]
    assert store.labels()["living"][-1] == "Corner"


async def test_store_skips_malformed_entries(hass, hass_storage):
    good = _profile("kitchen", {"s1": -50.0}).to_dict()
    hass_storage[STORAGE_KEY] = {
        "version": STORAGE_VERSION,
        "key": STORAGE_KEY,
        "data": {"rooms": {"kitchen": [good, {"area_id": "broken"}], "hall": "not a list"}},
    }
    store = ProfileStore(hass)
    await store.async_load()
    assert len(store.spots("kitchen")) == 1
    assert store.spots("hall") == []


# ---------------------------------------------------------------------------
# coordinator behaviour
# ---------------------------------------------------------------------------


class Clock:
    """Controllable monotonic clock for the learning mixin."""

    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("custom_components.bermuda.coordinator_learning.monotonic_time_coarse", clock)
    return clock


async def _setup(hass, entry, clock):
    """A loaded coordinator, two kitchen/hall proxies, a tracked 'Keys' device and two real areas."""
    coordinator = entry.runtime_data.coordinator
    await coordinator.room_profiles.async_load()
    areas = ar.async_get(hass)
    kitchen = areas.async_get_or_create("Kitchen")
    hall = areas.async_get_or_create("Hall")
    s1 = _make_scanner(coordinator, "11:11:11:11:11:11", "Kitchen proxy", kitchen.id)
    s2 = _make_scanner(coordinator, "22:22:22:22:22:22", "Hall proxy", hall.id)
    keys = _make_tracked_device(coordinator, "ec:00:00:00:00:01", "Keys")
    return coordinator, keys, s1, s2, kitchen, hall


def _hear(keys, s1, s2, rssi1: float, rssi2: float, d1: float = 2.0, d2: float = 3.0, stamp: float = 0.0):
    """Give Keys a current reading from both proxies."""
    for scanner, rssi, distance in ((s1, rssi1, d1), (s2, rssi2, d2)):
        advert = _set_distance(keys, scanner, distance, int(rssi))
        advert.rssi = rssi
        advert.hist_rssi = [int(rssi)] * 3  # real adverts keep their recent readings here
        advert.stamp = stamp


def _cycle(coordinator, keys, area_id: str | None, floor_id: str | None = "ground"):
    """One update cycle with the distance-picked area fixed to ``area_id``."""
    keys.area_id = area_id
    keys.floor_id = floor_id
    coordinator._refresh_learning()  # noqa: SLF001


def _dwell(coordinator, keys, clock, area_id: str | None, seconds: int, floor_id: str | None = "ground"):
    """Stay in ``area_id`` for ``seconds``, one update cycle per second."""
    for _ in range(seconds):
        clock.now += 1
        _cycle(coordinator, keys, area_id, floor_id)


def _flap(coordinator, keys, clock, area_a, area_b, *, seconds: int, spell: int = 25):
    """Flip between two rooms with realistic hysteresis: each answer held for ``spell`` seconds."""
    for i in range(seconds // spell):
        _dwell(coordinator, keys, clock, area_a if i % 2 else area_b, spell)


def _issue(hass, keys):
    return ir.async_get(hass).async_get_issue(DOMAIN, f"{REPAIR_TRAIN_PREFIX}{slugify(keys.address)}")


async def test_flapping_device_gets_a_training_prompt_after_the_sustained_time(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)  # both rooms read near-equal
    start = clock.now

    # Half a window of observation is needed before confidence can be judged, then the device must
    # stay poor for the sustained time (default 300 s) before anyone is bothered.
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=325)
    assert _issue(hass, keys) is None

    for _ in range(40):
        _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=25)
        if _issue(hass, keys) is not None:
            break
    issue = _issue(hass, keys)
    assert issue is not None
    assert 450 <= clock.now - start <= 800
    assert issue.is_fixable is True
    assert issue.translation_placeholders["device"] == "Keys"
    assert "Kitchen" in issue.translation_placeholders["areas"]
    assert coordinator.learning_diagnostics()["open_prompts"] == 1

    # Still flapping, so the prompt stays. Once readings separate and the answer settles it is withdrawn.
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=100)
    assert _issue(hass, keys) is not None
    _hear(keys, s1, s2, -50, -75, d1=1.0, d2=4.0)
    _dwell(coordinator, keys, clock, kitchen.id, 400)
    assert _issue(hass, keys) is None
    assert coordinator.learning_diagnostics()["open_prompts"] == 0


async def test_a_dismissed_prompt_stays_dismissed_if_confidence_drops_again(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=800)
    issue = _issue(hass, keys)
    assert issue is not None

    ir.async_ignore_issue(hass, DOMAIN, issue.issue_id, True)  # the user dismisses it
    _hear(keys, s1, s2, -50, -75, d1=1.0, d2=4.0)
    _dwell(coordinator, keys, clock, kitchen.id, 400)  # recovers: must not erase the dismissal
    assert _issue(hass, keys).dismissed_version is not None

    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=800)  # regresses
    assert _issue(hass, keys).dismissed_version is not None  # still dismissed: no nagging

    coordinator.start_training(keys, kitchen.id)  # acting on the device clears it for good
    assert _issue(hass, keys) is None


async def test_well_recognised_device_is_never_prompted(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -75, d1=1.0, d2=4.0)
    _dwell(coordinator, keys, clock, kitchen.id, 1500)
    for _ in range(3):  # ordinary use: wander to the next room for a few minutes and come back
        _dwell(coordinator, keys, clock, hall.id, 200)
        _dwell(coordinator, keys, clock, kitchen.id, 400)
    assert _issue(hass, keys) is None


async def test_prompts_can_be_turned_off(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    coordinator.options[CONF_LEARNING_ENABLED] = False
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=1000)
    assert _issue(hass, keys) is None


async def test_threshold_and_sustained_time_are_configurable(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    coordinator.options[CONF_LEARNING_SUSTAINED] = 30
    coordinator.options[CONF_LEARNING_THRESHOLD] = 0.9  # near-equal readings alone (0.6) now count as poor
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _dwell(coordinator, keys, clock, kitchen.id, 200)  # 150 s to build evidence, 30 s sustained
    assert _issue(hass, keys) is not None


async def test_learning_ignores_scanners_and_untracked_devices(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    keys.create_sensor = False
    for _ in range(3):
        _cycle(coordinator, keys, kitchen.id)
    assert coordinator._confidence == {}  # noqa: SLF001  (neither Keys nor the proxies are scored)


def _train(coordinator, keys, s1, s2, area, rssi1, rssi2, clock, base_stamp=1.0):
    """Run a full training run in ``area`` with the given readings."""
    coordinator.start_training(keys, area.id)
    for i in range(LEARN_SAMPLES + 5):
        clock.now += LEARN_INTERVAL
        wobble = 1 if i % 2 else -1
        _hear(keys, s1, s2, rssi1 + wobble, rssi2 - wobble, stamp=base_stamp + i)
        coordinator._refresh_learning()  # noqa: SLF001


async def test_learn_a_room_screen_counts_down_then_reports(hass, setup_bermuda_entry, clock, monkeypatch):
    from custom_components.bermuda import options_learning

    monkeypatch.setattr(options_learning, "CHUNK_SECONDS", 0.05)
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    flows = hass.config_entries.options
    result = await flows.async_init(setup_bermuda_entry.entry_id)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
    assert "learn" in result["menu_options"]
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "learn"})
    assert result["step_id"] == "learn"

    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": kitchen.id}
    )
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "learning"
    shown = result["description_placeholders"]
    assert shown["room"] == "Kitchen"
    assert shown["device"] == "Keys"  # the friendly name, never the long id
    assert shown["samples"] == "0"
    assert shown["wanted"] == str(LEARN_SAMPLES)
    assert shown["left"] == "5:00"
    assert coordinator.training_status(keys.address)["training"]

    # Let the readings come in (the real thing does this in the background), then follow the flow on.
    for i in range(LEARN_SAMPLES + 2):
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -50 + (i % 2), -75, stamp=1.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    steps = []
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        steps.append(result.get("step_id") or result.get("next_step_id"))
        await hass.async_block_till_done()
        result = await flows.async_configure(result["flow_id"])
    assert result["step_id"] == "learn_done"
    assert "Done." in result["description_placeholders"]["outcome"]
    assert "Keys" in result["description_placeholders"]["outcome"]
    assert coordinator.room_profiles.rooms()

    result = await flows.async_configure(result["flow_id"], user_input={})
    assert result["type"] is FlowResultType.MENU  # back at the main menu


async def test_learn_a_room_screen_reports_a_run_that_heard_too_little(hass, setup_bermuda_entry, clock, monkeypatch):
    from custom_components.bermuda import options_learning

    monkeypatch.setattr(options_learning, "CHUNK_SECONDS", 0.05)
    coordinator, keys, _s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    flows = hass.config_entries.options
    result = await flows.async_init(setup_bermuda_entry.entry_id)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "learn"})
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": kitchen.id}
    )
    clock.now += LEARN_MAX_SECONDS + 1
    coordinator._refresh_learning()  # noqa: SLF001
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        await hass.async_block_till_done()
        result = await flows.async_configure(result["flow_id"])
    assert result["step_id"] == "learn_done"
    assert "nothing was saved" in result["description_placeholders"]["outcome"]


async def test_learn_a_room_screen_shows_an_error_when_training_cannot_start(
    hass, setup_bermuda_entry, clock, monkeypatch
):
    coordinator, keys, _s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)

    def refuse(*_a, **_k):
        raise ServiceValidationError("nope")

    monkeypatch.setattr(coordinator, "start_training", refuse)
    flows = hass.config_entries.options
    result = await flows.async_init(setup_bermuda_entry.entry_id)
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
    result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "learn"})
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": kitchen.id}
    )
    assert result["step_id"] == "learn"
    assert result["errors"] == {"base": "learn_cannot_start"}


async def test_training_collects_fresh_samples_and_saves_the_room(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    status = coordinator.start_training(keys, kitchen.id)
    assert status["training"] is True
    assert status["samples"] == 0

    _hear(keys, s1, s2, -50, -75, stamp=1.0)
    clock.now += LEARN_INTERVAL
    coordinator._refresh_learning()  # noqa: SLF001
    assert coordinator.training_status(keys.address)["samples"] == 1

    # The same advert (same stamp) is never counted twice, and nothing is taken before the interval.
    clock.now += LEARN_INTERVAL
    coordinator._refresh_learning()  # noqa: SLF001
    assert coordinator.training_status(keys.address)["samples"] == 1
    _hear(keys, s1, s2, -51, -74, stamp=2.0)
    coordinator._refresh_learning()  # noqa: SLF001
    assert coordinator.training_status(keys.address)["samples"] == 2

    coordinator.reset_learning()
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    assert not coordinator.training_status(keys.address)["training"]
    profile = coordinator.room_profiles.spots(kitchen.id)[0]
    assert profile.samples == LEARN_SAMPLES
    assert profile.stats[s1.address].mean == pytest.approx(-50, abs=1)
    assert coordinator.training_status(keys.address)["trained_rooms"] == [kitchen.id]


async def test_training_that_times_out_keeps_or_discards_by_sample_count(hass, setup_bermuda_entry, clock, caplog):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)

    coordinator.start_training(keys, kitchen.id)
    for i in range(12):  # enough samples, but the device then goes quiet
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -50, -75, stamp=1.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    clock.now += LEARN_MAX_SECONDS
    coordinator._refresh_learning()  # noqa: SLF001
    assert kitchen.id in coordinator.room_profiles.rooms()

    coordinator.start_training(keys, hall.id)
    for i in range(3):  # too few
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -75, -50, stamp=100.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    clock.now += LEARN_MAX_SECONDS
    coordinator._refresh_learning()  # noqa: SLF001
    assert hall.id not in coordinator.room_profiles.rooms()
    assert "gathered only 3 samples" in caplog.text


async def test_start_training_validates_its_arguments(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    with pytest.raises(ServiceValidationError, match="No Home Assistant area"):
        coordinator.start_training(keys, "no-such-area")
    with pytest.raises(ServiceValidationError, match="isn't a tracked device"):
        coordinator.start_training(s1, kitchen.id)
    plain = coordinator._get_or_create_device("ec:00:00:00:00:77")  # noqa: SLF001
    with pytest.raises(ServiceValidationError, match="isn't a tracked device"):
        coordinator.start_training(plain, kitchen.id)


async def test_starting_training_withdraws_an_open_prompt(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=800)
    assert _issue(hass, keys) is not None
    coordinator.start_training(keys, kitchen.id)
    assert _issue(hass, keys) is None


async def _trained_keys(hass, entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, entry, clock)
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    _train(coordinator, keys, s1, s2, hall, -75, -50, clock, base_stamp=500.0)
    coordinator._confidence.clear()  # noqa: SLF001
    return coordinator, keys, s1, s2, kitchen, hall


async def test_trained_room_overrides_a_confident_but_wrong_distance_answer(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _trained_keys(hass, setup_bermuda_entry, clock)
    # Live readings look like the Hall, but plain distance (wrongly) says Kitchen, steadily.
    _hear(keys, s1, s2, -75, -50, d1=1.0, d2=1.1, stamp=900.0)
    _dwell(coordinator, keys, clock, kitchen.id, 400)
    assert keys.area_id == hall.id  # training wins even though distance never doubted itself
    assert keys.area_advert is None  # an override, not a winning advert


async def test_trained_device_is_not_prompted_and_unmatched_readings_change_nothing(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _trained_keys(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -90, -40, d1=2.0, d2=2.1, stamp=900.0)  # looks like neither trained room
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=900)
    assert _issue(hass, keys) is None
    _dwell(coordinator, keys, clock, kitchen.id, 1)
    assert keys.area_id == kitchen.id  # untouched by the matcher

    # Readings that look like the Hall while the answer is already Hall: nothing to override.
    _hear(keys, s1, s2, -75, -50, d1=2.0, d2=2.1, stamp=901.0)
    _dwell(coordinator, keys, clock, hall.id, 1)
    assert keys.area_id == hall.id


async def test_reset_forgets_one_room_or_everything(hass, setup_bermuda_entry, clock):
    coordinator, _keys, _s1, _s2, _kitchen, hall = await _trained_keys(hass, setup_bermuda_entry, clock)
    assert coordinator.reset_learning(hall.id) is True  # one room
    assert len(coordinator.room_profiles.rooms()) == 1
    assert coordinator.reset_learning() is True  # everything
    assert coordinator.room_profiles.rooms() == {}
    assert coordinator.reset_learning() is False


async def test_rooms_taught_by_one_device_are_used_for_another_and_a_quiet_one(hass, setup_bermuda_entry, clock):
    """The house-wide point: the S23 (here 'Keys') teaches, a quieter tracker is matched with no training of its own."""
    coordinator, keys, s1, s2, kitchen, hall = await _trained_keys(hass, setup_bermuda_entry, clock)
    tag = _make_tracked_device(coordinator, "ec:00:00:00:00:02", "Quiet tag")
    for rssi1, rssi2, expected in ((-62, -87, kitchen), (-87, -62, hall)):  # both 12 dB quieter than the teacher
        _hear(tag, s1, s2, rssi1, rssi2, d1=1.0, d2=1.1, stamp=900.0)
        wrong = hall if expected is kitchen else kitchen
        tag.area_id = wrong.id
        coordinator._refresh_learning()  # noqa: SLF001
        assert tag.area_id == expected.id


async def test_training_adds_spots_and_replace_starts_the_room_over(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    _train(coordinator, keys, s1, s2, kitchen, -60, -65, clock, base_stamp=300.0)
    assert len(coordinator.room_profiles.spots(kitchen.id)) == 2
    assert coordinator.learning_diagnostics()["trained_spots"] == 2

    coordinator.start_training(keys, kitchen.id, replace=True)
    for i in range(LEARN_SAMPLES + 2):
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -55 + (i % 2), -70, stamp=900.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    assert len(coordinator.room_profiles.spots(kitchen.id)) == 1


async def test_labelled_training_via_the_screen_and_forgetting_one_spot(hass, setup_bermuda_entry, clock, monkeypatch):
    from custom_components.bermuda import options_learning

    monkeypatch.setattr(options_learning, "CHUNK_SECONDS", 0.05)
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    _train(coordinator, keys, s1, s2, kitchen, -60, -65, clock)  # an earlier, unnamed spot
    flows = hass.config_entries.options

    async def open_learn():
        result = await flows.async_init(setup_bermuda_entry.entry_id)
        result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
        return await flows.async_configure(result["flow_id"], user_input={"next_step_id": "learn"})

    result = await open_learn()
    assert "Kitchen**: Spot 1" in result["description_placeholders"]["trained"]

    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": kitchen.id, "label": "  Table "}
    )
    assert result["description_placeholders"]["room"] == "Kitchen (Table)"
    for i in range(LEARN_SAMPLES + 2):
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -50 + (i % 2), -75, stamp=3000.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        await hass.async_block_till_done()
        result = await flows.async_configure(result["flow_id"])
    assert "Kitchen (Table)" in result["description_placeholders"]["outcome"]
    assert coordinator.room_profiles.labels() == {kitchen.id: ["Spot 1", "Table"]}

    result = await flows.async_configure(result["flow_id"], user_input={})
    result = await open_learn()
    result = await flows.async_configure(
        result["flow_id"],
        user_input={"device": keys.address.upper(), "room": kitchen.id, "label": "table", "forget": True},
    )
    assert "Forgot the **table** spot" in result["description_placeholders"]["outcome"]
    assert coordinator.room_profiles.labels() == {kitchen.id: ["Spot 1"]}

    result = await flows.async_configure(result["flow_id"], user_input={})
    result = await open_learn()
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "label": "table", "forget": True}
    )
    assert result["errors"] == {"base": "learn_cannot_start"}  # a spot name needs its room


async def test_services_label_a_spot_and_forget_it(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    started = await hass.services.async_call(
        DOMAIN,
        "learn_room",
        {"device": "Keys", "area": "Kitchen", "label": "Table"},
        blocking=True,
        return_response=True,
    )
    assert started["training"] is True
    for i in range(LEARN_SAMPLES + 2):
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -50, -75, stamp=1.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    assert coordinator.training_status(keys.address)["trained_spots"] == {kitchen.id: ["Table"]}
    assert coordinator.learning_diagnostics()["trained_spots"] == 1
    with pytest.raises(ServiceValidationError, match="Say which area"):
        await hass.services.async_call(DOMAIN, "reset_learning", {"label": "Table"}, blocking=True)
    gone = await hass.services.async_call(
        DOMAIN, "reset_learning", {"area": "Kitchen", "label": "table"}, blocking=True, return_response=True
    )
    assert gone == {"area_id": kitchen.id, "label": "table", "reset": True}


async def test_prompt_is_only_for_rooms_that_are_not_trained_yet(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)  # only the kitchen is trained
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1, stamp=900.0)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=900)
    assert _issue(hass, keys) is None  # just trained: it stays quiet for a while
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=LEARN_PROMPT_QUIET)
    assert _issue(hass, keys) is not None  # the hall is still untrained, so it is worth asking
    assert "Hall" in _issue(hass, keys).translation_placeholders["areas"]
    assert "Kitchen" not in _issue(hass, keys).translation_placeholders["areas"]

    _train(coordinator, keys, s1, s2, hall, -75, -50, clock, base_stamp=2000.0)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1, stamp=3000.0)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=100)
    assert _issue(hass, keys) is None  # both rooms are now trained house-wide: nothing left to offer


async def test_services_learn_and_reset(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    started = await hass.services.async_call(
        DOMAIN, "learn_room", {"device": "Keys", "area": "Kitchen"}, blocking=True, return_response=True
    )
    assert started["training"] is True
    assert started["area_id"] == kitchen.id

    with pytest.raises(ServiceValidationError, match="No Home Assistant area"):
        await hass.services.async_call(DOMAIN, "learn_room", {"device": "Keys", "area": "Nowhere"}, blocking=True)
    with pytest.raises(ServiceValidationError, match="doesn't know any device"):
        await hass.services.async_call(DOMAIN, "learn_room", {"device": "Ghost", "area": "Kitchen"}, blocking=True)

    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    reset = await hass.services.async_call(
        DOMAIN, "reset_learning", {"area": "Kitchen"}, blocking=True, return_response=True
    )
    assert reset == {"area_id": kitchen.id, "label": None, "reset": True}
    assert await hass.services.async_call(DOMAIN, "reset_learning", {}, blocking=True, return_response=True) == {
        "area_id": None,
        "label": None,
        "reset": False,
    }
    with pytest.raises(ServiceValidationError, match="No Home Assistant area"):
        await hass.services.async_call(DOMAIN, "reset_learning", {"area": "Nowhere"}, blocking=True)


# ---------------------------------------------------------------------------
# Repairs flow
# ---------------------------------------------------------------------------


async def test_repairs_flow_offers_teach_or_skip_then_asks_for_the_room(hass, setup_bermuda_entry, clock):
    coordinator, keys, _s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    flow = await async_create_fix_flow(hass, f"{REPAIR_TRAIN_PREFIX}x", {"device_key": keys.address})
    flow.hass, flow.issue_id = hass, f"{REPAIR_TRAIN_PREFIX}x"

    menu = await flow.async_step_init()
    assert menu["type"] is FlowResultType.MENU
    assert menu["menu_options"] == ["train", "skip"]
    assert menu["description_placeholders"] == {
        "device": "Keys",
        "kind": "Unidentified device",
        "areas": "several rooms",
    }

    form = await flow.async_step_train()
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "train"
    flow.flow_id, flow.handler = "f1", DOMAIN
    shown = await flow.async_step_train({"area": kitchen.id})
    assert shown["type"] is FlowResultType.SHOW_PROGRESS  # the screen stays open and counts down
    assert shown["description_placeholders"]["room"] == "Kitchen"
    assert coordinator.training_status(keys.address)["area_id"] == kitchen.id
    flow._task.cancel()  # noqa: SLF001


async def test_repairs_flow_counts_down_then_says_it_was_saved(hass, setup_bermuda_entry, clock, monkeypatch):
    from custom_components.bermuda import repairs

    monkeypatch.setattr(repairs, "CHUNK_SECONDS", 0.01)
    coordinator, keys, s1, s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    flow = await async_create_fix_flow(hass, f"{REPAIR_TRAIN_PREFIX}x", {"device_key": keys.address})
    flow.hass, flow.flow_id, flow.handler, flow.issue_id = hass, "f1", DOMAIN, f"{REPAIR_TRAIN_PREFIX}x"
    shown = await flow.async_step_train({"area": kitchen.id})
    assert shown["step_id"] == "train_run"
    assert shown["description_placeholders"]["samples"] == "0"

    await flow._task  # noqa: SLF001
    nxt = await flow.async_step_train_run()
    assert nxt["type"] is FlowResultType.SHOW_PROGRESS_DONE
    assert nxt["step_id"] == "train_listen"  # alternates so the frontend refreshes

    for i in range(LEARN_SAMPLES + 2):
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -50 + (i % 2), -75, stamp=10.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    shown = await flow.async_step_train_listen()
    await flow._task  # noqa: SLF001
    nxt = await flow.async_step_train_listen()
    assert nxt["step_id"] == "train_done"
    done = await flow.async_step_train_done()
    assert "**Kitchen** is trained" in done["description_placeholders"]["outcome"]
    closed = await flow.async_step_train_done({})
    assert closed["type"] is FlowResultType.CREATE_ENTRY
    assert kitchen.id in coordinator.room_profiles.rooms()


async def test_no_prompt_while_training_or_just_after(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    coordinator.start_training(keys, hall.id)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=900)  # flapping during the run
    assert _issue(hass, keys) is None  # asking now would restart the run


async def test_one_second_blips_are_not_flapping():
    tracker = ConfidenceTracker()
    now = 0.0
    tracker.add(now, "living", None, None)
    for _ in range(30):  # every 10 s the answer blips to another room for one second
        now += 9
        tracker.add(now, "bedroom", None, None)
        now += 1
        tracker.add(now, "living", None, None)
    result = tracker.result(now)
    assert result.flapping == 0.0
    assert result.top_areas == ("living",)

    for _ in range(10):  # real 30-second stays in the other room still count
        now += 30
        tracker.add(now, "bedroom", None, None)
        now += 30
        tracker.add(now, "living", None, None)
    assert tracker.result(now).flapping == 1.0


async def test_repairs_flow_skip_dismisses_the_prompt_for_good(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _hear(keys, s1, s2, -50, -55, d1=2.0, d2=2.1)
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=800)
    issue = _issue(hass, keys)
    assert issue is not None
    assert issue.dismissed_version is None

    flow = await async_create_fix_flow(hass, issue.issue_id, {"device_key": keys.address})
    flow.hass = hass
    flow.issue_id = issue.issue_id
    done = await flow.async_step_skip()
    assert done["type"] is FlowResultType.CREATE_ENTRY
    assert _issue(hass, keys).dismissed_version is not None  # dismissed, so it will not nag again
    assert not coordinator.training_status(keys.address)["training"]  # and nothing was trained

    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=800)  # still flapping: it stays dismissed
    assert _issue(hass, keys).dismissed_version is not None


async def test_repairs_flow_aborts_if_the_device_has_gone(hass, setup_bermuda_entry):
    flow = await async_create_fix_flow(hass, f"{REPAIR_TRAIN_PREFIX}x", {"device_key": "no:such:device"})
    flow.hass = hass
    assert (await flow.async_step_init())["reason"] == "device_gone"
    assert (await flow.async_step_train())["reason"] == "device_gone"


async def test_repairs_flow_rejects_unknown_issues(hass):
    with pytest.raises(ValueError, match="unknown repair issue"):
        await async_create_fix_flow(hass, "something_else", None)


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------


async def test_learn_screen_can_forget_a_room_or_everything(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    _train(coordinator, keys, s1, s2, hall, -75, -50, clock, base_stamp=500.0)
    flows = hass.config_entries.options

    async def open_learn():
        result = await flows.async_init(setup_bermuda_entry.entry_id)
        result = await flows.async_configure(result["flow_id"], user_input={"next_step_id": "calibrate"})
        return await flows.async_configure(result["flow_id"], user_input={"next_step_id": "learn"})

    # Forget without picking a room, and without ticking "everything", is refused (it must not wipe the house).
    result = await open_learn()
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper(), "forget": True})
    assert result["errors"] == {"base": "learn_cannot_start"}
    assert sorted(coordinator.room_profiles.rooms()) == sorted([kitchen.id, hall.id])

    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": kitchen.id, "forget": True}
    )
    assert "Forgot what was learned for **Kitchen**" in result["description_placeholders"]["outcome"]
    assert sorted(coordinator.room_profiles.rooms()) == [hall.id]

    result = await flows.async_configure(result["flow_id"], user_input={})
    result = await open_learn()
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "room": hall.id, "forget_all": True}
    )
    assert "Forgot every room" in result["description_placeholders"]["outcome"]
    assert not coordinator.room_profiles.rooms()

    result = await flows.async_configure(result["flow_id"], user_input={})
    result = await open_learn()
    result = await flows.async_configure(
        result["flow_id"], user_input={"device": keys.address.upper(), "forget_all": True}
    )
    assert "Nothing matched" in result["description_placeholders"]["outcome"]
    # Without Forget, a room is required.
    result = await flows.async_configure(result["flow_id"], user_input={})
    result = await open_learn()
    result = await flows.async_configure(result["flow_id"], user_input={"device": keys.address.upper()})
    assert result["errors"] == {"base": "learn_cannot_start"}


# ---------------------------------------------------------------------------
# options + inert paths
# ---------------------------------------------------------------------------


async def test_options_flow_saves_learning_settings(hass, setup_bermuda_entry):
    result = await hass.config_entries.options.async_init(setup_bermuda_entry.entry_id)
    result = await goto_setting(hass, result, "globalopts")
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "distance_model": {"ref_power": -55.0, "attenuation": 3.0, "max_area_radius": 10.0},
            "tracking": {"devtracker_nothome_timeout": 30, "update_interval": 1.0},
            "smoothing": {"smoothing_samples": 20, "max_velocity": 3.0},
            "learning": {CONF_LEARNING_ENABLED: False, CONF_LEARNING_THRESHOLD: 0.4, CONF_LEARNING_SUSTAINED: 120},
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_bermuda_entry.options[CONF_LEARNING_ENABLED] is False
    assert setup_bermuda_entry.options[CONF_LEARNING_THRESHOLD] == 0.4
    assert setup_bermuda_entry.options[CONF_LEARNING_SUSTAINED] == 120


async def test_refresh_is_inert_until_profiles_have_loaded(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _setup(hass, setup_bermuda_entry, clock)
    coordinator.room_profiles.loaded = False
    _cycle(coordinator, keys, kitchen.id)
    assert coordinator._confidence == {}  # noqa: SLF001
