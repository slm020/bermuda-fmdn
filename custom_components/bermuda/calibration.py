"""
Distance calibration: turning a few measured signal strengths into Bermuda's settings.

Bermuda estimates distance as ``10 ** ((ref_power - rssi) / (10 * attenuation))`` where
``ref_power`` is the signal strength (dBm) at 1 m and ``attenuation`` says how quickly the
signal fades in your home. Rather than ask for those numbers, the guided calibration measures
them: hold a phone 1 m from a proxy, then 3 m, and (optionally) 1 m from each other proxy.

* ``ref_power`` is the reading at 1 m.
* ``attenuation`` follows from the drop between 1 m and 3 m.
* each proxy's ``rssi_offset`` is whatever lines its 1 m reading up with ``ref_power``.

The maths here is pure; ``apply_scanner_offsets`` is the one function that touches Home Assistant.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from homeassistant.config_entries import ConfigSubentry

from .const import (
    CONF_RSSI_OFFSET,
    CONF_SCANNER,
    OPT_RSSI_OFFSET_MAX,
    OPT_RSSI_OFFSET_MIN,
    SUBENTRY_TYPE_CALIBRATION,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from homeassistant.core import HomeAssistant

    from . import BermudaConfigEntry

MEASURE_SECONDS: Final = 60  # the longest one measurement listens
ENOUGH_SAMPLES: Final = 12  # a measurement ends early once it has this many readings
CHUNK_SECONDS: Final = 5  # the progress screen refreshes this often
SAMPLE_INTERVAL: Final = 0.5  # how often it checks for a new reading
MIN_SAMPLES: Final = 3  # fewer readings than this and the proxy barely heard the phone
MAX_SPREAD_DB: Final = 8.0  # a jumpier signal than this (std dev, dB) gives a poor measurement
FAR_METRES: Final = 3.0  # the second distance used to work out attenuation (default choice)
FAR_CHOICES: Final = (2.0, 3.0)  # a proxy that cannot hear the phone at 3 m can be measured at 2 m instead
MIN_ATTENUATION: Final = 1.0  # outside 1..6 the two readings can't both be right
MAX_ATTENUATION: Final = 6.0


@dataclass(frozen=True)
class Measurement:
    """A summary of one listening session."""

    median: float  # dBm, robust against the odd outlier
    spread: float  # standard deviation, dB
    count: int


def summarise(samples: Sequence[float]) -> Measurement | None:
    """Median and spread of the readings, or None if there were too few to trust."""
    if len(samples) < MIN_SAMPLES:
        return None
    return Measurement(round(statistics.median(samples), 1), round(statistics.pstdev(samples), 1), len(samples))


def is_stable(measurement: Measurement) -> bool:
    """Whether the reading held steady enough to use."""
    return measurement.spread <= MAX_SPREAD_DB


def attenuation_from(rssi_1m: float, rssi_far: float, far_metres: float = FAR_METRES) -> float:
    """Path-loss exponent from a reading at 1 m and one at ``far_metres``."""
    return (rssi_1m - rssi_far) / (10 * math.log10(far_metres))


def is_plausible_attenuation(attenuation: float) -> bool:
    """Whether the exponent is in the range real homes give."""
    return MIN_ATTENUATION <= attenuation <= MAX_ATTENUATION


LOW_ATTENUATION_WARNING: Final = 1.5  # in a real home the fade rate is usually 2-4; below this something is off


def attenuation_warning(attenuation: float) -> str | None:
    """A plain-words warning when the measured fade rate looks too low to be right, else None."""
    if attenuation >= LOW_ATTENUATION_WARNING:
        return None
    return (
        f"A fade rate of {attenuation:.2f} is lower than a home normally gives (usually 2 to 4). The 3 m reading "
        "was barely weaker than the 1 m one, so distances would be squashed and far-away rooms would look close. "
        "Try again with the far reading at 2 m, or move away from walls and the proxy, or skip the far step."
    )


def offset_for(ref_power: float, raw_rssi_at_1m: float) -> float:
    """The offset (dB) that makes a proxy's 1 m reading equal ``ref_power``, within the allowed range."""
    return round(min(max(ref_power - raw_rssi_at_1m, OPT_RSSI_OFFSET_MIN), OPT_RSSI_OFFSET_MAX), 1)


def apply_scanner_offsets(hass: HomeAssistant, entry: BermudaConfigEntry, offsets: Mapping[str, float]) -> None:
    """
    Save per-proxy RSSI offsets as calibration subentries (the source of truth since config v2).

    Updates a proxy's existing subentry or adds one. Writing the legacy options dict instead
    would be silently overwritten at the next load.
    """
    existing = {
        se.data.get(CONF_SCANNER): se
        for se in entry.subentries.values()
        if se.subentry_type == SUBENTRY_TYPE_CALIBRATION
    }
    # Names first: saving an entry can reload the integration, after which the live devices are gone.
    titles = {scanner: _scanner_title(entry, scanner) for scanner in offsets}
    for scanner, offset in offsets.items():
        data = MappingProxyType({CONF_SCANNER: scanner, CONF_RSSI_OFFSET: offset})
        if (subentry := existing.get(scanner)) is not None:
            hass.config_entries.async_update_subentry(entry, subentry, data=data)
        else:
            hass.config_entries.async_add_subentry(
                entry,
                ConfigSubentry(
                    data=data,
                    subentry_type=SUBENTRY_TYPE_CALIBRATION,
                    title=titles[scanner],
                    unique_id=scanner,
                ),
            )


def _scanner_title(entry: BermudaConfigEntry, scanner: str) -> str:
    """A proxy's friendly name for a calibration entry's title, falling back to its address."""
    device = entry.runtime_data.coordinator.devices.get(scanner)
    return (device.name if device and device.name else None) or scanner


def retitle_calibration_subentries(hass: HomeAssistant, entry: BermudaConfigEntry) -> int:
    """
    Give calibration entries still titled by a bare address the proxy's friendly name.

    Only titles that are exactly the address are touched, so a name the user chose is kept and
    this settles after one pass. Returns how many were renamed.
    """
    renamed = 0
    for se in list(entry.subentries.values()):
        if se.subentry_type != SUBENTRY_TYPE_CALIBRATION:
            continue
        scanner = se.data.get(CONF_SCANNER)
        title = _scanner_title(entry, scanner) if scanner else None
        if scanner and se.title == scanner and title and title != scanner:
            hass.config_entries.async_update_subentry(entry, se, title=title)
            renamed += 1
    return renamed
