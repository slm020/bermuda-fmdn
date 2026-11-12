"""
Room estimate that uses every proxy at once and builds up evidence over time.

Picking "the nearest proxy" each second fails for quiet transmitters such as Find Hub tags on a
dog's collar: they reach every proxy at -80..-100 dBm, readings jump by ~10 dB from one advert to
the next, and per-proxy calibration offsets can shrink a real 15 dB lead to a 2 dB one. So instead,
each update cycle every candidate room gets a score from two kinds of evidence:

* **Loudness** (every room with a proxy): how far that room's loudest proxy is behind the loudest
  proxy overall, in raw dBm. Raw on purpose: the proxies are identical boards, and the offsets
  measured with a phone held by hand turned out to point at the wrong room from the sofa.
* **Pattern** (trained rooms): how well the whole reading from all proxies fits the room's trained
  spots (:func:`room_learning.match_rooms` scores). Rooms with no training get a fixed penalty, so
  a trained room that fits well beats an untrained one that is merely about as loud.

Those scores feed a simple forward (hidden Markov) filter: a belief per room that carries over from
cycle to cycle, with a small chance each cycle that the device has moved. One noisy second can only
nudge the belief; a room has to keep winning for a few seconds before it is shown, and the shown
room only changes once the new room is clearly believed. A dog walking into another room still
shows up within a few seconds because the evidence there is strong and consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# Loudness: a room whose loudest proxy is this many dB behind the loudest proxy overall scores -1
# (twice as far behind scores -4: squared, so near-ties barely matter and big gaps matter a lot).
LOUDNESS_DB = 5.0
LOUDNESS_FLOOR = -6.0  # cap for rooms far behind, or not heard at all
UNHEARD = LOUDNESS_FLOOR
# Pattern: the trained-room fit score (mean squared z, ~0-1 is a good fit) counts against the room.
# Capped low: past ~4 it just means "does not fit", and a bad fit must not cost a trained room more
# than an untrained room that is clearly quieter.
PATTERN_CAP = 4.0
EMPTY_ROOM_PENALTY = 1.0  # presence sensors say nobody is there: a tie-breaker only, as radars can miss a dog
UNTRAINED_PENALTY = 2.0  # what a room with no trained spots (or too few shared proxies) pays instead

EVIDENCE_WEIGHT = 0.3  # how much one cycle of evidence moves the belief (cycles are ~1 s)
MOVE_CHANCE = 0.05  # per cycle chance the device moved: keeps every room reachable
SHOW_BELIEF = 0.75  # a different room is shown only once believed this strongly
# A proxy often misses a few of a quiet tag's adverts in a row; that is not evidence the tag left.
# Its last reading is kept for this long before the proxy counts as not hearing the tag.
MEMORY_SECONDS = 20.0


def room_evidence(
    live_rssi: Mapping[str, float],
    scanner_areas: Mapping[str, str],
    pattern_scores: Mapping[str, float],
    candidate_areas: set[str],
    empty_areas: set[str] | None = None,
) -> dict[str, float]:
    """
    Log-likelihood style score per room (0 = perfect, more negative = less likely).

    ``live_rssi``: scanner address -> raw RSSI now. ``scanner_areas``: scanner address -> its room.
    ``pattern_scores``: trained room -> fit score (only rooms that could be scored).
    ``empty_areas``: rooms whose presence sensors say nobody is there.
    """
    loudest_by_area: dict[str, float] = {}
    for scanner, rssi in live_rssi.items():
        area = scanner_areas.get(scanner)
        if area is not None and rssi > loudest_by_area.get(area, -math.inf):
            loudest_by_area[area] = rssi
    top = max(loudest_by_area.values(), default=None)

    evidence: dict[str, float] = {}
    for area in candidate_areas:
        if top is not None and area in loudest_by_area:
            behind = (top - loudest_by_area[area]) / LOUDNESS_DB
            loud = max(-(behind**2), LOUDNESS_FLOOR)
        else:
            loud = UNHEARD
        pattern = -min(pattern_scores[area], PATTERN_CAP) if area in pattern_scores else -UNTRAINED_PENALTY
        empty = -EMPTY_ROOM_PENALTY if empty_areas and area in empty_areas else 0.0
        evidence[area] = loud + pattern + empty
    return evidence


@dataclass
class RoomFilter:
    """Per-device belief over rooms, and the room currently shown."""

    belief: dict[str, float] = field(default_factory=dict)
    shown: str | None = None
    recent: dict[str, tuple[float, float]] = field(default_factory=dict)  # scanner -> (when, rssi)

    def remember(self, live_rssi: Mapping[str, float], now: float) -> dict[str, float]:
        """Merge this cycle's readings with each proxy's last reading from the past MEMORY_SECONDS."""
        for scanner, rssi in live_rssi.items():
            self.recent[scanner] = (now, rssi)
        self.recent = {s: (t, r) for s, (t, r) in self.recent.items() if now - t <= MEMORY_SECONDS}
        return {s: r for s, (_t, r) in self.recent.items()}

    def update(self, evidence: Mapping[str, float]) -> str | None:
        """Fold one cycle of evidence into the belief and return the room to show (None = no opinion yet)."""
        if not evidence:
            return self.shown
        areas = set(evidence) | set(self.belief)
        n = len(areas)
        if not self.belief:
            prior = dict.fromkeys(areas, 1.0 / n)
        else:
            # Predict: stay put most of the time, otherwise could be anywhere.
            prior = {a: (1 - MOVE_CHANCE) * self.belief.get(a, 0.0) + MOVE_CHANCE / n for a in areas}
        floor = min(evidence.values()) if evidence else 0.0
        weighted = {a: prior[a] * math.exp(EVIDENCE_WEIGHT * evidence.get(a, floor)) for a in areas}
        total = sum(weighted.values()) or 1.0
        self.belief = {a: w / total for a, w in weighted.items()}

        best = max(self.belief, key=self.belief.__getitem__)
        if self.shown is None or self.shown not in self.belief:
            # First opinion: show the front-runner once it leads at all clearly.
            if self.belief[best] >= 0.5:
                self.shown = best
        elif best != self.shown and self.belief[best] >= SHOW_BELIEF:
            self.shown = best
        return self.shown

    def reset(self) -> None:
        """Forget the belief (eg after the device has not been heard for a while)."""
        self.belief.clear()
        self.shown = None
        self.recent.clear()
