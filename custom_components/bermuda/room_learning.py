"""
Confidence-triggered room learning for Bermuda (pure maths plus a thin Store wrapper).

Bermuda normally places a device by distance alone. This module adds an opt-in
second opinion for the devices where that goes wrong:

* :class:`ConfidenceTracker` scores how well a device's room is being
  recognised from its recent history (flapping back and forth between two
  rooms, near-equal readings from rooms either side of a wall, bouncing
  between floors).
* :class:`RoomProfile` is what the user *trains*: per-scanner RSSI statistics
  captured at one spot in a known room. Profiles belong to the HOUSE, not to a
  device: train a room once (with any phone) and every tracked device is matched
  against it. A room can hold several spots (sofa and bed in the same room).
* :func:`match_rooms` compares live readings with those profiles. A louder or
  quieter device shifts every scanner's RSSI by the same number of dB (a constant
  power ratio), so the match fits and removes that one offset before comparing; only the
  *differences between proxies* count.

Nothing here learns on its own: profiles exist only for rooms the user trained. The ideas
(RSSI profiles per room, z-score matching, confidence from history) are adapted
from jleinenbach/bermuda (MIT), slimmed to the user-trained case.

The maths avoids Home Assistant imports; only :class:`ProfileStore` touches HA.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

from .const import (
    LEARN_FLAP_FULL,
    LEARN_FLAP_MAX_DWELL,
    LEARN_MEDIAN_COUNT,
    LEARN_NEAR_EQUAL_MARGIN,
    LEARN_WINDOW_SECONDS,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from homeassistant.core import HomeAssistant

STORAGE_VERSION = 1
STORAGE_KEY = "bermuda_room_profiles"  # house-wide; the old per-device "bermuda_learning" file is no longer read
MAX_SPOTS_PER_ROOM = 12  # adding more drops the oldest (and says so)
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
MIN_SHARED_SCANNERS = 2  # with one shared scanner the fitted offset would absorb everything
HEARING_FLOOR_DBM = -95.0  # a scanner expected to hear the device below this is allowed to be silent
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
    """What the RSSI looks like from each scanner at one trained spot in a room (raw dBm from the training device)."""

    def __init__(self, area_id: str, label: str = "") -> None:
        """Create an empty profile for ``area_id``; ``label`` names the spot (sofa, door, north wall...)."""
        self.area_id = area_id
        self.label = label
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
            "label": self.label,
            "samples": self.samples,
            "scanners": {s: [st.n, st.mean, st.m2] for s, st in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RoomProfile:
        """Rebuild from :meth:`to_dict` output (raises KeyError/TypeError/ValueError if malformed)."""
        profile = cls(str(raw["area_id"]), str(raw.get("label", "")))
        profile.samples = int(raw.get("samples", 0))
        for scanner, (n, mean, m2) in raw["scanners"].items():
            profile.stats[str(scanner)] = _Stat(int(n), float(mean), float(m2))
        return profile


@dataclass(frozen=True)
class RoomMatch:
    """Result of comparing live readings with the house's trained rooms."""

    area_id: str | None  # the believed room, or None when no trained room fits clearly
    best_score: float
    scores: tuple[tuple[str, float], ...]  # (area_id, score), best first; lower is better
    offset_db: float | None = None  # how much louder (+) or quieter (-) the device is than the trained one


def fit_score(
    live_rssi: Mapping[str, float], known: Mapping[str, tuple[float, float]], seen: Collection[str]
) -> tuple[float, float] | None:
    """
    Score live readings against a reference pattern, ignoring the device's overall loudness.

    ``known`` maps each scanner the reference knows well to ``(mean dBm, variance)``; ``seen`` is
    every scanner the reference has any record of. Returns ``(score, offset_db)`` or None when too
    few scanners are shared to say anything. ``offset_db`` is the variance-weighted mean of
    (live - reference) over the shared scanners: the one number that a louder/quieter transmitter
    shifts every scanner by. After removing it, the leftover per-scanner differences are scored as
    squared z-scores (near 1 = a good fit).
    """
    shared = [s for s in known if s in live_rssi]
    if len(shared) < MIN_SHARED_SCANNERS:
        return None
    weights = {s: 1.0 / known[s][1] for s in shared}
    offset = sum(weights[s] * (live_rssi[s] - known[s][0]) for s in shared) / sum(weights.values())
    # One degree of freedom went into the offset, so scale the residuals back up to chi-square size.
    scale = len(shared) / (len(shared) - 1)
    terms = [scale * (live_rssi[s] - known[s][0] - offset) ** 2 / known[s][1] for s in shared]
    # A reference scanner that has gone silent only counts against it if this device
    # (at its fitted loudness) should still have been audible there.
    terms.extend(
        MISSING_SCANNER_Z2
        for s, (mean, _var) in known.items()
        if s not in live_rssi and mean + offset >= HEARING_FLOOR_DBM
    )
    terms.extend(UNKNOWN_SCANNER_Z2 for s in live_rssi if s not in seen)
    return sum(terms) / len(terms), offset


def _score_spot(live_rssi: Mapping[str, float], profile: RoomProfile) -> tuple[float, float] | None:
    """Score live readings against one trained spot (see :func:`fit_score`)."""
    known = {s: (st.mean, st.variance) for s, st in profile.stats.items() if st.n >= MIN_SCANNER_SAMPLES}
    return fit_score(live_rssi, known, profile.stats)


def device_live_rssi(device: Any) -> dict[str, float]:
    """
    Raw RSSI from each scanner that can hear ``device`` right now.

    Raw (no per-proxy offsets, no distance model) so anything learned from it stays valid across a
    calibration. Each reading is the median of the scanner's last few adverts, which tames the
    dropouts and spikes a single reading would pass straight into a match.
    """
    live: dict[str, float] = {}
    for advert in device.adverts.values():
        if advert.rssi_distance is None or advert.rssi is None:
            continue
        recent = [r for r in advert.hist_rssi[:LEARN_MEDIAN_COUNT] if r is not None] or [advert.rssi]
        value = float(median(recent))
        if value > live.get(advert.scanner_address, float("-inf")):
            live[advert.scanner_address] = value
    return live


def match_rooms(live_rssi: Mapping[str, float], rooms: Mapping[str, Sequence[RoomProfile]]) -> RoomMatch:
    """
    Compare live per-scanner RSSI with each trained room (best spot per room).

    A room's score is the mean squared z-score of the differences between scanners left
    after the device's overall loudness is removed (see :func:`_score_spot`): near 1 means
    the reading looks like that room. The best room is only believed if it is a decent fit
    and clearly better than the runner-up.
    """
    scored: list[tuple[str, float, float]] = []
    for area_id, spots in rooms.items():
        results = [r for spot in spots if (r := _score_spot(live_rssi, spot)) is not None]
        if results:
            score, offset = min(results)
            scored.append((area_id, score, offset))
    scored.sort(key=lambda item: item[1])
    if not scored:
        return RoomMatch(None, math.inf, ())
    best_area, best, offset = scored[0]
    clear_winner = len(scored) == 1 or scored[1][1] - best >= ACCEPT_MARGIN
    return RoomMatch(
        best_area if best <= ACCEPT_SCORE and clear_winner else None,
        best,
        tuple((a, sc) for a, sc, _o in scored),
        offset,
    )


class ProfileStore:
    """Persists the house's user-trained room profiles: area id -> its spots."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Create the (empty, unloaded) store; HA's Store is imported lazily like FingerprintStore."""
        from homeassistant.helpers.storage import Store  # noqa: PLC0415 (lazy by design)

        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._items: dict[str, list[RoomProfile]] = {}
        self.loaded = False

    async def async_load(self) -> None:
        """Load saved profiles into memory, skipping anything malformed."""
        data = await self._store.async_load()
        self._items = {}
        if data and isinstance(data.get("rooms"), dict):
            for area_id, spots in data["rooms"].items():
                for raw in spots if isinstance(spots, list) else []:
                    try:
                        profile = RoomProfile.from_dict(raw)
                    except KeyError, TypeError, ValueError:
                        continue
                    self._items.setdefault(str(area_id), []).append(profile)
        for spots in self._items.values():  # spots saved before spots had names get one
            for spot in spots:
                if not spot.label:
                    spot.label = self._free_label(spots)
        self.loaded = True

    @staticmethod
    def _free_label(spots: Sequence[RoomProfile]) -> str:
        """The next unused "Spot N" name among a room's spots."""
        taken = {p.label.casefold() for p in spots}
        n = len(spots) + 1
        while f"spot {n}" in taken:
            n += 1
        return f"Spot {n}"

    def _data_to_save(self) -> dict[str, Any]:
        return {"rooms": {a: [p.to_dict() for p in spots] for a, spots in self._items.items()}}

    def rooms(self) -> Mapping[str, Sequence[RoomProfile]]:
        """Every trained room with its spots. Treat as read-only."""
        return self._items

    def spots(self, area_id: str) -> Sequence[RoomProfile]:
        """One room's trained spots (empty if untrained)."""
        return self._items.get(area_id, [])

    def add_spot(self, profile: RoomProfile, *, replace: bool = False) -> str | None:
        """
        Save a trained spot and schedule a write.

        A spot with the same label (ignoring case) in the same room is retrained in place; an
        unlabelled one is named "Spot N". ``replace`` first forgets every other spot in the room.
        Returns the label of the oldest spot dropped to stay within ``MAX_SPOTS_PER_ROOM``, if any.
        """
        spots = [] if replace else list(self._items.get(profile.area_id, []))
        label = profile.label.strip()
        if not label:
            label = self._free_label(spots)
        profile.label = label
        for i, existing in enumerate(spots):
            if existing.label.casefold() == label.casefold():
                spots[i] = profile
                break
        else:
            spots.append(profile)
        dropped = spots[0].label if len(spots) > MAX_SPOTS_PER_ROOM else None
        self._items[profile.area_id] = spots[-MAX_SPOTS_PER_ROOM:]
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)
        return dropped

    def forget(self, area_id: str | None = None, label: str | None = None) -> bool:
        """
        Forget one spot (``area_id`` and ``label``), one room, or every room (no arguments).

        Returns whether there was anything to forget.
        """
        if area_id is None:
            existed = bool(self._items)
            self._items = {}
        elif label:
            spots = self._items.get(area_id, [])
            kept = [p for p in spots if p.label.casefold() != label.strip().casefold()]
            existed = len(kept) != len(spots)
            if kept:
                self._items[area_id] = kept
            else:
                self._items.pop(area_id, None)
        else:
            existed = self._items.pop(area_id, None) is not None
        if existed:
            self._store.async_delay_save(self._data_to_save, SAVE_DELAY)
        return existed

    def labels(self) -> dict[str, list[str]]:
        """Every trained room's spot labels, for display."""
        return {a: [p.label for p in spots] for a, spots in self._items.items()}
