"""Room filter: evidence from every proxy, built up over time."""

from __future__ import annotations

import random

from custom_components.bermuda.room_filter import RoomFilter, room_evidence

AREAS = {"lr": "living_room", "bed": "bedroom", "bath": "bathroom", "kit": "kitchen", "din": "dining_room"}
CANDIDATES = set(AREAS.values())


def _run(f, base, n, rng, spread=8.0, pattern=None):
    shown = []
    for _ in range(n):
        live = {s: v + rng.uniform(-spread, spread) for s, v in base.items()}
        shown.append(f.update(room_evidence(live, AREAS, pattern or {}, CANDIDATES)))
    return shown


def test_weak_noisy_tag_on_the_sofa_never_jumps():
    rng = random.Random(1)
    f = RoomFilter()
    sofa = {"lr": -80.0, "bed": -90.0, "bath": -91.0, "din": -92.0, "kit": -96.0}
    shown = _run(f, sofa, 600, rng)
    assert set(shown[10:]) == {"living_room"}


def test_walking_into_another_room_shows_within_seconds():
    rng = random.Random(2)
    f = RoomFilter()
    _run(f, {"lr": -78.0, "bed": -92.0, "bath": -92.0}, 60, rng)
    shown = _run(f, {"lr": -92.0, "bed": -75.0, "bath": -93.0}, 15, rng)
    assert shown[-1] == "bedroom"
    assert shown.index("bedroom") <= 6


def test_trained_pattern_beats_an_untrained_room_that_is_about_as_loud():
    f = RoomFilter()
    live = {"lr": -88.0, "bed": -87.0}
    for _ in range(30):
        shown = f.update(room_evidence(live, AREAS, {"living_room": 0.3, "bathroom": 3.0}, CANDIDATES))
    assert shown == "living_room"


def test_unheard_rooms_and_empty_evidence():
    f = RoomFilter()
    assert f.update({}) is None
    ev = room_evidence({"lr": -80.0, "zz": -60.0}, AREAS, {}, CANDIDATES)
    assert ev["kitchen"] < ev["living_room"]  # an unheard room scores the floor; unknown proxy ignored
    for _ in range(10):
        f.update(ev)
    assert f.shown == "living_room"
    f.reset()
    assert f.shown is None
    assert not f.belief
    assert room_evidence({}, AREAS, {}, {"bedroom"}) == {"bedroom": -8.0}
