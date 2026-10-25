"""
Confidence-triggered room learning for Bermuda (pure maths plus a thin Store wrapper).

Bermuda normally places a device by distance alone. This module adds an opt-in
second opinion for the devices where that goes wrong:

* :class:`ConfidenceTracker` scores how well a device's room is being
  recognised from its recent history (flapping back and forth between two
  rooms, near-equal readings from rooms either side of a wall, bouncing
  between floors).
* :class:`RoomProfile` is what the user *trains*: per-scanner RSSI statistics
  captured while the device sits in a known room.
* :func:`match_rooms` compares live readings with those profiles.

Nothing here learns on its own: profiles exist only for devices and rooms the
user trained, so devices that are recognised well are never touched. The ideas
(RSSI profiles per room, z-score matching, confidence from history) are adapted
from jleinenbach/bermuda (MIT), slimmed to the user-trained case.

The maths avoids Home Assistant imports; only :class:`ProfileStore` touches HA.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .const import (
    LEARN_FLAP_FULL,
    LEARN_FLAP_MAX_DWELL,
    LEARN_NEAR_EQUAL_MARGIN,
    LEARN_WINDOW_SECONDS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant

STORAGE_VERSION = 1
STORAGE_KEY = "bermuda_learning"
SAVE_DELAY = 10  # seconds to debounce disk writes

# Confidence penalty weights. Flapping and floor bounces are strong evidence of
# poor recognition; near-equal readings alone (a stable answer next to a wall)
# deliberately cannot push a device under the default threshold by themselves.
WEIGHT_FLAPPING = 1.0
WEIGHT_NEAR_EQUAL = 0.4
WEIGHT_FLOOR_BOUNCE = 1.0

RSSI_VARIANCE_FLOOR = 4.0  # dB^2: never trust a profile tighter than 2 dB
MIN_SCANNER_SAMPLES = 5  # a scanner needs this many samples in a profile to count
MISSING_SCANNER_Z2 = 9.0  # a profiled scanner that is silent now: as if 3 sigma off
UNKNOWN_SCANNER_Z2 = 1.0  # a scanner heard now that the profile never saw: mild
ACCEPT_SCORE = 6.0  # mean squared z-score; ~2.4 sigma rms. Above this no room matches.
ACCEPT_MARGIN = 1.0  # the best room must beat the runner-up by this to be believed


@dataclass(frozen=True)
class ConfidenceResult:
    """How well a device's room is being recognised (1.0 = no doubt)."""

    score: float
    flapping: float
    near_equal: float
    floor_bounce: float
    top_areas: tuple[str, ...]  # most-seen areas in the window, for the user-facing prompt


def _quick_returns(changes: list[tuple[float, str]], max_dwell: float) -> int:
    """
    Count A -> B -> A returns where the detour to B was brief.

    ``changes`` are (time, value) with consecutive values already different. A device that
    wanders off and stays away (A -> B for minutes) is moving, not flapping.
    """
    return sum(
        1
        for i in range(2, len(changes))
        if changes[i][1] == changes[i - 2][1] and changes[i][0] - changes[i - 1][0] <= max_dwell
    )


class ConfidenceTracker:
    """
    Score room recognition from the recent *time-ordered* history of a device.

    Area selection has hysteresis, so a device that is hard to place looks like long runs
    (AAAAAA BBBBBB AAAAAA), not one-cycle blips. The tracker therefore records area/floor
    *changes* with timestamps, and judges quick there-and-back changes over a time window,
    independent of how often the coordinator happens to cycle.
    """

    def __init__(self, window: float = LEARN_WINDOW_SECONDS) -> None:
        """Create an empty tracker scoring the last ``window`` seconds."""
        self._window = window
        self._started: float | None = None
        self._areas: deque[tuple[float, str]] = deque()  # (time, area) at each change
        self._floors: deque[tuple[float, str]] = deque()  # (time, floor) at each change
        self._margins: deque[tuple[float, bool]] = deque()  # (time, readings were near-equal)

    def add(self, now: float, area_id: str | None, floor_id: str | None, margin: float | None) -> None:
        """
        Record one cycle at monotonic time ``now``.

        ``margin`` is how much farther the nearest scanner in a *different* area is than the
        winning one, as a fraction (0.1 = 10% farther); ``None`` when no other area can hear
        the device.
        """
        if self._started is None:
            self._started = now
        if area_id is not None and (not self._areas or self._areas[-1][1] != area_id):
            self._areas.append((now, area_id))
        if floor_id is not None and (not self._floors or self._floors[-1][1] != floor_id):
            self._floors.append((now, floor_id))
        if margin is not None:
            self._margins.append((now, margin < LEARN_NEAR_EQUAL_MARGIN))
        cutoff = now - self._window
        for series in (self._areas, self._floors, self._margins):
            while series and series[0][0] < cutoff:
                series.popleft()

    def clear(self) -> None:
        """Forget the history (eg after training)."""
        self._started = None
        self._areas.clear()
        self._floors.clear()
        self._margins.clear()

    def result(self, now: float) -> ConfidenceResult:
        """Score the window. Until half a window has been observed there is no evidence of a problem."""
        if self._started is None or now - self._started < self._window / 2:
            return ConfidenceResult(1.0, 0.0, 0.0, 0.0, ())
        flapping = min(1.0, _quick_returns(list(self._areas), LEARN_FLAP_MAX_DWELL) / LEARN_FLAP_FULL)
        floor_bounce = min(1.0, _quick_returns(list(self._floors), LEARN_FLAP_MAX_DWELL) / (LEARN_FLAP_FULL / 2))
        near_equal = sum(1 for _t, near in self._margins if near) / len(self._margins) if self._margins else 0.0
        penalty = min(
            1.0, WEIGHT_FLAPPING * flapping + WEIGHT_NEAR_EQUAL * near_equal + WEIGHT_FLOOR_BOUNCE * floor_bounce
        )
        top = tuple(area for area, _count in Counter(a for _t, a in self._areas).most_common(2))
        return ConfidenceResult(round(1.0 - penalty, 3), flapping, near_equal, floor_bounce, top)


class _Stat:
    """Running mean/variance of one scanner's RSSI (Welford)."""

    __slots__ = ("m2", "mean", "n")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.n = n
        self.mean = mean
        self.m2 = m2

    def add(self, value: float) -> None:
        """Fold in one RSSI reading."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        """Sample variance, floored so a quiet scanner can't make matching brittle."""
        return max(self.m2 / (self.n - 1), RSSI_VARIANCE_FLOOR) if self.n > 1 else RSSI_VARIANCE_FLOOR


class RoomProfile:
    """What one device's RSSI looks like from each scanner while it sat in one room."""

    def __init__(self, area_id: str) -> None:
        """Create an empty profile for ``area_id``."""
        self.area_id = area_id
        self.stats: dict[str, _Stat] = {}
        self.samples = 0

    def add_sample(self, rssi_by_scanner: Mapping[str, float]) -> None:
        """Fold in one snapshot of per-scanner RSSI."""
        for scanner, rssi in rssi_by_scanner.items():
            self.stats.setdefault(scanner, _Stat()).add(rssi)
        self.samples += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the Store."""
        return {
            "area_id": self.area_id,
            "samples": self.samples,
            "scanners": {s: [st.n, st.mean, st.m2] for s, st in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RoomProfile:
        """Rebuild from :meth:`to_dict` output (raises KeyError/TypeError/ValueError if malformed)."""
        profile = cls(str(raw["area_id"]))
        profile.samples = int(raw.get("samples", 0))
        for scanner, (n, mean, m2) in raw["scanners"].items():
            profile.stats[str(scanner)] = _Stat(int(n), float(mean), float(m2))
        return profile


@dataclass(frozen=True)
class RoomMatch:
    """Result of comparing live readings with a device's trained rooms."""

    area_id: str | None  # the believed room, or None when no trained room fits clearly
    best_score: float
    scores: tuple[tuple[str, float], ...]  # (area_id, score), best first; lower is better


def match_rooms(live_rssi: Mapping[str, float], profiles: Mapping[str, RoomProfile]) -> RoomMatch:
    """
    Compare live per-scanner RSSI with each trained room.

    A room's score is the mean squared z-score over the scanners it knows well:
    near 1 means the reading looks like that room; a scanner that has gone silent
    counts as 3 sigma off. The best room is only believed if it is a decent fit
    and clearly better than the runner-up.
    """
    scored: list[tuple[str, float]] = []
    for area_id, profile in profiles.items():
        terms = [
            (live_rssi[s] - st.mean) ** 2 / st.variance if s in live_rssi else MISSING_SCANNER_Z2
            for s, st in profile.stats.items()
            if st.n >= MIN_SCANNER_SAMPLES
        ]
        if not terms:
            continue
        terms.extend(UNKNOWN_SCANNER_Z2 for s in live_rssi if s not in profile.stats)
        scored.append((area_id, sum(terms) / len(terms)))
    scored.sort(key=lambda item: item[1])
    if not scored:
        return RoomMatch(None, math.inf, ())
    best_area, best = scored[0]
    clear_winner = len(scored) == 1 or scored[1][1] - best >= ACCEPT_MARGIN
    return RoomMatch(best_area if best <= ACCEPT_SCORE and clear_winner else None, best, tuple(scored))


class ProfileStore:
    """Persists the user-trained room profiles, keyed by stable device address then area id."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Create the (empty, unloaded) store; HA's Store is imported lazily like FingerprintStore."""
        from homeassistant.helpers.storage import Store  # noqa: PLC0415 (lazy by design)

        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._items: dict[str, dict[str, RoomProfile]] = {}
        self.loaded = False

    async def async_load(self) -> None:
        """Load saved profiles into memory, skipping anything malformed."""
        data = await self._store.async_load()
        self._items = {}
        if data and isinstance(data.get("devices"), dict):
            for device_key, rooms in data["devices"].items():
                for raw in rooms if isinstance(rooms, list) else []:
                    try:
                        profile = RoomProfile.from_dict(raw)
                    except KeyError, TypeError, ValueError:
                        continue
                    self._items.setdefault(str(device_key), {})[profile.area_id] = profile
        self.loaded = True

    def _data_to_save(self) -> dict[str, Any]:
        return {"devices": {k: [p.to_dict() for p in rooms.values()] for k, rooms in self._items.items()}}

    def get(self, device_key: str) -> dict[str, RoomProfile]:
        """Return a device's trained rooms (empty if none). Treat as read-only."""
        return self._items.get(device_key, {})

    def put(self, device_key: str, profile: RoomProfile) -> None:
        """Save a (re)trained room and schedule a write."""
        self._items.setdefault(device_key, {})[profile.area_id] = profile
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    def reset(self, device_key: str) -> bool:
        """Forget everything trained for a device. Returns whether there was anything."""
        existed = self._items.pop(device_key, None) is not None
        if existed:
            self._store.async_delay_save(self._data_to_save, SAVE_DELAY)
        return existed

    def device_keys(self) -> list[str]:
        """Every device with trained rooms."""
        return list(self._items)
