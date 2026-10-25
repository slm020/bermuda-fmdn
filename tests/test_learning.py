"""
Tests for confidence-triggered room learning.

The maths (ConfidenceTracker, RoomProfile, match_rooms) is pure and tested
directly. The coordinator behaviour (prompt when poor, train, apply trained
rooms only when doubtful, reset) is driven through a loaded entry with a
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
from custom_components.bermuda.button import BermudaLearnRoomButton, BermudaResetLearningButton
from custom_components.bermuda.const import (
    CONF_LEARNING_ENABLED,
    CONF_LEARNING_SUSTAINED,
    CONF_LEARNING_THRESHOLD,
    DOMAIN,
    LEARN_INTERVAL,
    LEARN_MAX_SECONDS,
    LEARN_SAMPLES,
    REPAIR_TRAIN_PREFIX,
    SIGNAL_DEVICE_NEW,
)
from custom_components.bermuda.repairs import async_create_fix_flow
from custom_components.bermuda.room_learning import (
    STORAGE_KEY,
    STORAGE_VERSION,
    ConfidenceTracker,
    ProfileStore,
    RoomProfile,
    match_rooms,
)
from custom_components.bermuda.select import BermudaTrainingRoomSelect
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


def test_match_picks_the_room_the_readings_look_like():
    profiles = {
        "kitchen": _profile("kitchen", {"s1": -50.0, "s2": -75.0}),
        "hall": _profile("hall", {"s1": -75.0, "s2": -50.0}),
    }
    match = match_rooms({"s1": -51.0, "s2": -74.0}, profiles)
    assert match.area_id == "kitchen"
    assert [a for a, _s in match.scores] == ["kitchen", "hall"]
    assert match.best_score < 2.0


def test_match_refuses_a_poor_fit():
    profiles = {"kitchen": _profile("kitchen", {"s1": -50.0, "s2": -75.0})}
    match = match_rooms({"s1": -90.0, "s2": -40.0}, profiles)
    assert match.area_id is None
    assert match.best_score > 6.0


def test_match_refuses_when_two_rooms_are_too_alike():
    profiles = {
        "kitchen": _profile("kitchen", {"s1": -50.0, "s2": -70.0}),
        "utility": _profile("utility", {"s1": -50.5, "s2": -70.5}),
    }
    assert match_rooms({"s1": -50.0, "s2": -70.0}, profiles).area_id is None


def test_match_penalises_a_silent_scanner():
    profiles = {"kitchen": _profile("kitchen", {"s1": -50.0, "s2": -60.0})}
    heard_all = match_rooms({"s1": -50.0, "s2": -60.0}, profiles)
    one_missing = match_rooms({"s1": -50.0}, profiles)
    assert one_missing.best_score > heard_all.best_score


def test_match_with_nothing_usable_gives_no_answer():
    thin = RoomProfile("x")
    thin.add_sample({"s1": -50.0})  # a single sample is below MIN_SCANNER_SAMPLES
    assert match_rooms({"s1": -50.0}, {"x": thin}).area_id is None
    assert match_rooms({"s1": -50.0}, {}).best_score == math.inf


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------


async def test_store_roundtrip_and_reset(hass, hass_storage):
    store = ProfileStore(hass)
    await store.async_load()
    assert store.loaded
    store.put("dev1", _profile("kitchen", {"s1": -50.0}))
    store.put("dev1", _profile("hall", {"s1": -70.0}))
    assert sorted(store.get("dev1")) == ["hall", "kitchen"]
    assert store.device_keys() == ["dev1"]

    hass_storage[STORAGE_KEY] = {"version": STORAGE_VERSION, "key": STORAGE_KEY, "data": store._data_to_save()}  # noqa: SLF001
    fresh = ProfileStore(hass)
    await fresh.async_load()
    assert sorted(fresh.get("dev1")) == ["hall", "kitchen"]
    assert fresh.get("dev1")["kitchen"].stats["s1"].n == 12

    assert fresh.reset("dev1") is True
    assert fresh.reset("dev1") is False
    assert fresh.get("dev1") == {}


async def test_store_skips_malformed_entries(hass, hass_storage):
    good = _profile("kitchen", {"s1": -50.0}).to_dict()
    hass_storage[STORAGE_KEY] = {
        "version": STORAGE_VERSION,
        "key": STORAGE_KEY,
        "data": {"devices": {"dev1": [good, {"area_id": "broken"}], "dev2": "not a list"}},
    }
    store = ProfileStore(hass)
    await store.async_load()
    assert sorted(store.get("dev1")) == ["kitchen"]
    assert store.get("dev2") == {}


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

    coordinator.reset_learning(keys)
    _train(coordinator, keys, s1, s2, kitchen, -50, -75, clock)
    assert not coordinator.training_status(keys.address)["training"]
    profile = coordinator.room_profiles.get(keys.address)[kitchen.id]
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
    assert kitchen.id in coordinator.room_profiles.get(keys.address)

    coordinator.start_training(keys, hall.id)
    for i in range(3):  # too few
        clock.now += LEARN_INTERVAL
        _hear(keys, s1, s2, -75, -50, stamp=100.0 + i)
        coordinator._refresh_learning()  # noqa: SLF001
    clock.now += LEARN_MAX_SECONDS
    coordinator._refresh_learning()  # noqa: SLF001
    assert hall.id not in coordinator.room_profiles.get(keys.address)
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


async def test_trained_room_steps_in_only_when_confidence_is_poor(hass, setup_bermuda_entry, clock):
    coordinator, keys, s1, s2, kitchen, hall = await _trained_keys(hass, setup_bermuda_entry, clock)
    # Live readings look like the Hall, but plain distance (wrongly) says Kitchen.
    _hear(keys, s1, s2, -75, -50, d1=1.0, d2=1.1, stamp=900.0)

    # Distance is steady, so confidence is full: a trained room never overrides a confident answer.
    _dwell(coordinator, keys, clock, kitchen.id, 400)
    assert keys.area_id == kitchen.id

    # Now the distance answer starts flapping: confidence drops and the trained Hall wins.
    _flap(coordinator, keys, clock, kitchen.id, hall.id, seconds=200)
    _dwell(coordinator, keys, clock, kitchen.id, 1)
    assert keys.area_id == hall.id
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


async def test_reset_forgets_everything(hass, setup_bermuda_entry, clock):
    coordinator, keys, *_rest = await _trained_keys(hass, setup_bermuda_entry, clock)
    assert coordinator.reset_learning(keys) is True
    assert coordinator.room_profiles.get(keys.address) == {}
    assert coordinator.reset_learning(keys) is False


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
        DOMAIN, "reset_learning", {"device": "Keys"}, blocking=True, return_response=True
    )
    assert reset == {"device": "Keys", "reset": True}


# ---------------------------------------------------------------------------
# Repairs flow
# ---------------------------------------------------------------------------


async def test_repairs_flow_asks_for_the_room_and_starts_training(hass, setup_bermuda_entry, clock):
    coordinator, keys, _s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    flow = await async_create_fix_flow(hass, f"{REPAIR_TRAIN_PREFIX}x", {"device_key": keys.address})
    flow.hass = hass

    form = await flow.async_step_init()
    assert form["type"] is FlowResultType.FORM
    assert form["description_placeholders"] == {"device": "Keys"}

    done = await flow.async_step_init({"area": kitchen.id})
    assert done["type"] is FlowResultType.CREATE_ENTRY
    assert coordinator.training_status(keys.address)["area_id"] == kitchen.id


async def test_repairs_flow_aborts_if_the_device_has_gone(hass, setup_bermuda_entry):
    flow = await async_create_fix_flow(hass, f"{REPAIR_TRAIN_PREFIX}x", {"device_key": "no:such:device"})
    flow.hass = hass
    assert (await flow.async_step_init())["reason"] == "device_gone"


async def test_repairs_flow_rejects_unknown_issues(hass):
    with pytest.raises(ValueError, match="unknown repair issue"):
        await async_create_fix_flow(hass, "something_else", None)


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------


async def test_training_room_select_and_buttons(hass, setup_bermuda_entry, clock):
    coordinator, keys, _s1, _s2, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    entry = setup_bermuda_entry

    select = BermudaTrainingRoomSelect(coordinator, entry, keys.address)
    select.async_write_ha_state = MagicMock()
    assert "Kitchen" in select.options
    assert select.current_option is None
    await select.async_select_option("Kitchen")
    assert select.current_option == "Kitchen"
    await select.async_select_option("No such room")  # ignored
    assert select.current_option == "Kitchen"
    assert select.unique_id.endswith("_training_room")

    learn = BermudaLearnRoomButton(coordinator, entry, keys.address)
    reset = BermudaResetLearningButton(coordinator, entry, keys.address)
    assert learn.unique_id.endswith("_learn_room")
    assert reset.unique_id.endswith("_reset_learning")

    await learn.async_press()
    assert coordinator.training_status(keys.address)["area_id"] == kitchen.id
    await reset.async_press()
    assert coordinator.training_status(keys.address)["training"] is False

    coordinator.training_room.clear()
    with pytest.raises(ServiceValidationError, match="Pick a training room"):
        await learn.async_press()


async def test_select_forgets_a_deleted_area(hass, setup_bermuda_entry, clock):
    coordinator, keys, *_rest, kitchen, _hall = await _setup(hass, setup_bermuda_entry, clock)
    select = BermudaTrainingRoomSelect(coordinator, setup_bermuda_entry, keys.address)
    coordinator.training_room[keys.address] = kitchen.id
    ar.async_get(hass).async_delete(kitchen.id)
    assert select.current_option is None


async def test_new_tracked_devices_get_the_learning_entities(hass, setup_bermuda_entry, clock):
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    coordinator, keys, *_rest = await _setup(hass, setup_bermuda_entry, clock)
    async_dispatcher_send(hass, SIGNAL_DEVICE_NEW, keys.address)
    await hass.async_block_till_done()
    async_dispatcher_send(hass, SIGNAL_DEVICE_NEW, keys.address)  # a repeat creates nothing new
    await hass.async_block_till_done()
    unique_ids = {e.unique_id for e in er.async_get(hass).entities.values() if e.platform == DOMAIN}
    for suffix in ("training_room", "learn_room", "reset_learning"):
        assert f"{keys.unique_id}_{suffix}" in unique_ids


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
