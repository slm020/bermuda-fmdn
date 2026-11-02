"""
RF fingerprinting for sub-area "micro-location" detection in Bermuda.

A *micro-location* is a named spot (for example "Key hook") that is defined by
example: when the user tells Bermuda that a given device is *here right now*,
we snapshot the vector of smoothed distances from every scanner that can
currently hear it. Later, we compare a device's live vector against its saved
spots and report the best match. This gives a location that is finer-grained
than Home Assistant's Areas, which Bermuda can otherwise only resolve down to
the nearest scanner.

Design notes
------------
- Fingerprints are *tied to the device* that was used to calibrate them. Two
  different transmitters radiate differently, so a spot calibrated with your
  keys is only ever matched against your keys. This keeps matching accurate and
  matches the "where are my keys" mental model.
- The maths in this module is deliberately free of Home Assistant imports so it
  can be unit-tested in isolation. The only HA touch-point is
  :class:`FingerprintStore`, which persists data via HA's ``Store`` helper and
  imports it lazily so that the rest of the module remains importable without a
  Home Assistant installation present.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .room_learning import RSSI_VARIANCE_FLOOR, fit_score

if TYPE_CHECKING:
    from collections.abc import Iterable

    from homeassistant.core import HomeAssistant

# Persistence (HA Store) -----------------------------------------------------
STORAGE_VERSION = 1
STORAGE_KEY = "bermuda_locations"
SAVE_DELAY = 10  # seconds to debounce writes to disk

# Matcher tuning defaults ----------------------------------------------------
# All distances are in metres, matching Bermuda's smoothed ``rssi_distance``.
DEFAULT_ACCEPT_DISTANCE = 2.5  # at/below this weighted error, a match is solid
DEFAULT_REJECT_DISTANCE = 5.0  # at/above this weighted error, reject outright
DEFAULT_MIN_MARGIN = 0.4  # best must beat runner-up by this to be "confident"
DEFAULT_MARGIN_FULL = 2.0  # margin at which we award full margin-confidence
DEFAULT_MISS_PENALTY = 6.0  # error charged when a calibrated scanner is silent
DEFAULT_EXTRA_PENALTY = 3.0  # error charged per scanner heard that the spot lacks
DEFAULT_STD_EPS = 0.25  # floor on variance so a perfect calib can't dominate

# Signal-strength matching (what places are matched on) ------------------------
# Scores are mean squared z-scores after removing the device's overall loudness
# (see room_learning.fit_score): about 1 is a good fit, so these are far stricter than a room's.
RSSI_ACCEPT_SCORE = 3.0  # at/below this a place is a solid match
RSSI_REJECT_SCORE = 8.0  # at/above this it is no match at all
RSSI_HOLD_SCORE = 6.0  # a place the device is already reported at is kept until its score passes this
RSSI_MIN_MARGIN = 0.75  # the best place must beat the runner-up by this
RSSI_MARGIN_FULL = 3.0  # margin at which confidence from the margin is full
RSSI_LEGACY_STD = 4.0  # dB: assumed spread for places saved before their spread was recorded


@dataclass
class Fingerprint:
    """A single named spot, captured for one device."""

    name: str
    device_address: str
    vector: dict[str, float]  # scanner_address -> mean smoothed distance (m)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    area_id: str | None = None
    floor_id: str | None = None
    vector_std: dict[str, float] = field(default_factory=dict)  # per-scanner std-dev
    rssi_vector: dict[str, float] = field(default_factory=dict)  # scanner -> mean raw RSSI (dBm): what is matched
    rssi_std: dict[str, float] = field(default_factory=dict)  # scanner -> its spread at calibration (dB)
    sample_count: int = 1
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict for the Store."""
        return {
            "id": self.id,
            "name": self.name,
            "device_address": self.device_address,
            "area_id": self.area_id,
            "floor_id": self.floor_id,
            "vector": dict(self.vector),
            "vector_std": dict(self.vector_std),
            "rssi_vector": dict(self.rssi_vector),
            "rssi_std": dict(self.rssi_std),
            "sample_count": self.sample_count,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        """Re-create a Fingerprint from stored data, tolerating missing keys."""
        return cls(
            name=data["name"],
            device_address=data["device_address"],
            vector={str(k): float(v) for k, v in data.get("vector", {}).items()},
            id=data.get("id") or uuid.uuid4().hex,
            area_id=data.get("area_id"),
            floor_id=data.get("floor_id"),
            vector_std={str(k): float(v) for k, v in data.get("vector_std", {}).items()},
            rssi_vector={str(k): float(v) for k, v in data.get("rssi_vector", {}).items()},
            rssi_std={str(k): float(v) for k, v in data.get("rssi_std", {}).items()},
            sample_count=int(data.get("sample_count", 1)),
            created=float(data.get("created", time.time())),
            updated=float(data.get("updated", time.time())),
        )


@dataclass
class MatchResult:
    """Outcome of comparing a live vector against a device's saved spots."""

    id: str | None
    name: str | None
    score: float  # weighted error of the winner (lower is better)
    confidence: float  # 0..1, blends absolute closeness with the runner-up margin
    accepted: bool  # did the winner clear the accept threshold + margin?
    second_score: float | None  # runner-up error, for diagnostics / hysteresis
    scores: list[tuple[str, str, float]]  # [(id, name, score)] sorted best-first


class FingerprintMatcher:
    """
    Compares a live distance vector against a set of saved Fingerprints.

    The metric is a weighted RMS error over the *union* of scanners. Scanners
    that were part of the calibrated spot count most (weighted by how steady
    they were at calibration time); a calibrated scanner that is now silent, or
    a scanner now heard that the spot never had, both count *against* the match,
    because presence/absence of a proxy is itself location information.
    """

    def __init__(
        self,
        accept_distance: float = DEFAULT_ACCEPT_DISTANCE,
        reject_distance: float = DEFAULT_REJECT_DISTANCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        margin_full: float = DEFAULT_MARGIN_FULL,
        miss_penalty: float = DEFAULT_MISS_PENALTY,
        extra_penalty: float = DEFAULT_EXTRA_PENALTY,
        std_eps: float = DEFAULT_STD_EPS,
    ) -> None:
        """Store the scoring/acceptance thresholds used to match a live vector against spots."""
        self.accept_distance = accept_distance
        self.reject_distance = reject_distance
        self.min_margin = min_margin
        self.margin_full = margin_full
        self.miss_penalty = miss_penalty
        self.extra_penalty = extra_penalty
        self.std_eps = std_eps

    def score_one(self, live: dict[str, float], fingerprint: Fingerprint) -> float:
        """Return the weighted RMS error between a live vector and one spot."""
        weighted_sq = 0.0
        weight_total = 0.0

        for scanner, ref_distance in fingerprint.vector.items():
            std = fingerprint.vector_std.get(scanner, 0.0)
            weight = 1.0 / ((std * std) + self.std_eps)
            weight_total += weight
            if scanner in live:
                diff = live[scanner] - ref_distance
            else:
                # The spot expects this proxy to hear the device, but it can't.
                diff = self.miss_penalty
            weighted_sq += weight * diff * diff

        # Scanners we hear now that the calibrated spot didn't include.
        for scanner in live:
            if scanner not in fingerprint.vector:
                weight = 1.0 / self.std_eps  # treat as a confident "shouldn't be here"
                weight_total += weight
                weighted_sq += weight * self.extra_penalty * self.extra_penalty

        if weight_total <= 0:
            return self.reject_distance
        return math.sqrt(weighted_sq / weight_total)

    def match(self, live: dict[str, float], candidates: Iterable[Fingerprint]) -> MatchResult | None:
        """
        Score ``live`` against every candidate and pick the best.

        Returns ``None`` if there are no candidates or the live vector is empty
        (nothing to go on). Otherwise always returns a result; inspect
        ``accepted``/``confidence`` to decide whether to believe it.
        """
        scored: list[tuple[str, str, float]] = []
        best: tuple[Fingerprint, float] | None = None
        for fingerprint in candidates:
            score = self.score_one(live, fingerprint)
            scored.append((fingerprint.id, fingerprint.name, score))
            if best is None or score < best[1]:
                best = (fingerprint, score)

        if best is None or not live:
            return None

        scored.sort(key=lambda item: item[2])
        best_fp, best_score = best
        second_score = scored[1][2] if len(scored) > 1 else None

        confidence = self._confidence(best_score, second_score)
        within_accept = best_score <= self.accept_distance
        clear_margin = second_score is not None and (second_score - best_score) >= self.min_margin
        accepted = best_score <= self.reject_distance and confidence > 0 and (within_accept or clear_margin)

        return MatchResult(
            id=best_fp.id,
            name=best_fp.name,
            score=best_score,
            confidence=confidence,
            accepted=accepted,
            second_score=second_score,
            scores=scored,
        )

    @staticmethod
    def _rssi_reference(fingerprint: Fingerprint) -> dict[str, tuple[float, float]]:
        """A place's scanners as ``{scanner: (mean dBm, variance)}``."""
        return {
            scanner: (mean, max(fingerprint.rssi_std.get(scanner, RSSI_LEGACY_STD) ** 2, RSSI_VARIANCE_FLOOR))
            for scanner, mean in fingerprint.rssi_vector.items()
        }

    def match_rssi(
        self, live_rssi: dict[str, float], candidates: Iterable[Fingerprint], holding: str | None = None
    ) -> MatchResult | None:
        """
        Compare live raw RSSI with each place, ignoring how loud the device is.

        Same idea as room learning: one fitted offset absorbs the device's transmit power and only the
        differences between proxies are compared, so a place is recognised by its *pattern*, not by
        distances (which depend on the distance calibration). A place with no usable overlap scores as
        rejected. Returns ``None`` when there is nothing to go on.

        ``holding`` is the id of the place the device is already reported at. Getting in takes a good match
        (``RSSI_ACCEPT_SCORE``) but staying only needs a fair one (``RSSI_HOLD_SCORE``), unless a different
        place now clearly wins, so a device sitting on the edge of a place does not flicker in and out.
        """
        scored: list[tuple[str, str, float]] = []
        for fingerprint in candidates:
            reference = self._rssi_reference(fingerprint)
            fit = fit_score(live_rssi, reference, reference)
            scored.append((fingerprint.id, fingerprint.name, fit[0] if fit else RSSI_REJECT_SCORE))
        if not scored or not live_rssi:
            return None
        scored.sort(key=lambda item: item[2])
        best_id, best_name, best = scored[0]
        second = scored[1][2] if len(scored) > 1 else None
        closeness = _clamp((RSSI_REJECT_SCORE - best) / (RSSI_REJECT_SCORE - RSSI_ACCEPT_SCORE), 0.0, 1.0)
        margin_conf = 1.0 if second is None else _clamp((second - best) / RSSI_MARGIN_FULL, 0.0, 1.0)
        accepted = best <= RSSI_ACCEPT_SCORE and (second is None or second - best >= RSSI_MIN_MARGIN)
        if not accepted and holding is not None:
            kept = next((entry for entry in scored if entry[0] == holding), None)
            if kept is not None and kept[2] <= RSSI_HOLD_SCORE:
                # Keep the place it is already at: nothing else has clearly won.
                best_id, best_name, best = kept
                accepted = True
                second = next((entry[2] for entry in scored if entry[0] != holding), None)
        return MatchResult(
            id=best_id,
            name=best_name,
            score=best,
            confidence=round(closeness * margin_conf, 4),
            accepted=accepted,
            second_score=second,
            scores=scored,
        )

    def _confidence(self, best_score: float, second_score: float | None) -> float:
        """Blend absolute closeness with the runner-up margin into 0..1."""
        span = self.reject_distance - self.accept_distance
        if span <= 0:  # pragma: no cover - guards against silly config
            closeness = 1.0 if best_score <= self.accept_distance else 0.0
        else:
            closeness = (self.reject_distance - best_score) / span
        closeness = _clamp(closeness, 0.0, 1.0)

        if second_score is None:
            margin_conf = 1.0  # only one spot to choose from, no ambiguity
        else:
            margin_conf = _clamp((second_score - best_score) / self.margin_full, 0.0, 1.0)

        return round(closeness * margin_conf, 4)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


class FingerprintStore:
    """
    Persists Fingerprints via Home Assistant's Store helper.

    Places belong to the house: every tracked device is matched against all of them, and
    ``device_address`` only records which device was used to mark the place. Kept deliberately
    thin: it owns an in-memory dict of fingerprints plus a ``device_address -> [ids]`` index
    (for "marked by" lookups) and debounces writes to disk. ``Store`` is imported lazily so
    that the rest of this module stays importable without Home Assistant.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Set up the (empty, unloaded) in-memory fingerprint store and its HA Store backend."""
        from homeassistant.helpers.storage import Store  # noqa: PLC0415 (lazy by design)

        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._items: dict[str, Fingerprint] = {}
        self._by_device: dict[str, list[str]] = {}
        self.loaded = False

    async def async_load(self) -> None:
        """Load saved fingerprints from disk into memory (idempotent)."""
        data = await self._store.async_load()
        self._items = {}
        if data and isinstance(data.get("fingerprints"), list):
            for raw in data["fingerprints"]:
                try:
                    fingerprint = Fingerprint.from_dict(raw)
                except KeyError, TypeError, ValueError:
                    continue
                self._items[fingerprint.id] = fingerprint
        self._reindex()
        self.loaded = True

    def _reindex(self) -> None:
        """Rebuild the device_address -> [ids] lookup index."""
        index: dict[str, list[str]] = {}
        for fingerprint in self._items.values():
            index.setdefault(fingerprint.device_address, []).append(fingerprint.id)
        self._by_device = index

    def _data_to_save(self) -> dict[str, Any]:
        """Build the JSON-serialisable payload for the Store."""
        return {"fingerprints": [fp.to_dict() for fp in self._items.values()]}

    def _schedule_save(self) -> None:
        """Persist to disk after a short debounce."""
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    # --- read helpers -------------------------------------------------------

    def list(self, device_address: str | None = None) -> list[Fingerprint]:
        """Return all fingerprints, optionally filtered to one device."""
        if device_address is None:
            return list(self._items.values())
        return [self._items[fid] for fid in self._by_device.get(device_address, [])]

    def get(self, fingerprint_id: str) -> Fingerprint | None:
        """Return a fingerprint by id, or None."""
        return self._items.get(fingerprint_id)

    def find_by_name(self, name: str, device_address: str | None = None) -> Fingerprint | None:
        """Return the place called ``name`` (case-insensitive); places are house-wide, so any device may match."""
        lowered = name.strip().casefold()
        for fingerprint in self.list(device_address):
            if fingerprint.name.casefold() == lowered:
                return fingerprint
        return None

    # --- mutations ----------------------------------------------------------

    def add(self, fingerprint: Fingerprint) -> None:
        """Add or replace a fingerprint (keyed by id) and schedule a save."""
        self._items[fingerprint.id] = fingerprint
        self._reindex()
        self._schedule_save()

    def remove(self, fingerprint_id: str) -> bool:
        """Remove a fingerprint by id. Returns True if it existed."""
        if fingerprint_id in self._items:
            del self._items[fingerprint_id]
            self._reindex()
            self._schedule_save()
            return True
        return False

    def rename(self, fingerprint_id: str, new_name: str) -> bool:
        """Rename a fingerprint by id. Returns True if it existed."""
        fingerprint = self._items.get(fingerprint_id)
        if fingerprint is None:
            return False
        fingerprint.name = new_name
        fingerprint.updated = time.time()
        self._schedule_save()
        return True
