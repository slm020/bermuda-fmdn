"""
Interface for resolving a rotating Google Find Hub EID to a stable tag.

Bermuda does no Google authentication or key handling of its own. The
GoogleFindMy-HA integration owns the account login and key material and
publishes an EID resolver in ``hass.data``; :class:`GoogleFindMyResolver`
calls it duck-typed at runtime (no import of that GPL-3.0 integration).
Anything implementing :class:`FmdnResolver` can replace it later.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from .const import _LOGGER, DOMAIN_GOOGLEFINDMY
from .fmdn_extraction import extract_eid_candidates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant

DATA_EID_RESOLVER = "eid_resolver"


class ResolutionStatus(StrEnum):
    """Outcome of one resolution attempt (kept as statistics for diagnostics)."""

    RESOLVED = "resolved"
    NO_KNOWN_EID_MATCH = "no_known_eid_match"
    RESOLVER_UNAVAILABLE = "resolver_unavailable"
    RESOLVER_ERROR = "resolver_error"


class FmdnTag(NamedTuple):
    """A resolved tag: one per Google account the tag is registered in."""

    device_id: str  # Home Assistant device-registry id (unique per Google account)
    canonical_id: str  # Google's id for the physical tag (shared across accounts)
    config_entry_id: str


class FmdnResolver(Protocol):
    """Resolve the payloads of one advert to the tag(s) they belong to."""

    def resolve(self, payloads: Sequence[bytes]) -> tuple[list[FmdnTag], ResolutionStatus]:
        """Return the matching tags (empty if none) and why."""
        ...

    @property
    def available(self) -> bool:
        """Whether the resolver can currently be reached (for diagnostics)."""
        ...


def _to_tag(match: Any) -> FmdnTag | None:
    """Convert a resolver's match object to an FmdnTag (None if it lacks a device id)."""
    device_id = getattr(match, "device_id", None)
    if not device_id:
        return None
    return FmdnTag(
        str(device_id),
        str(getattr(match, "canonical_id", "") or "").split(":")[-1],
        str(getattr(match, "config_entry_id", "") or ""),
    )


class GoogleFindMyResolver:
    """Resolver backed by the GoogleFindMy-HA integration (>= 1.7.0-3)."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Store hass; the resolver itself is looked up on every call."""
        self._hass = hass

    @property
    def available(self) -> bool:
        """Whether GoogleFindMy-HA is loaded and has published its resolver."""
        return self._get_resolver() is not None

    def _get_resolver(self) -> Any | None:
        """
        Fetch the live resolver object, or None when the integration isn't loaded.

        Not cached: GoogleFindMy-HA publishes it on setup and removes it on unload.
        """
        bucket = self._hass.data.get(DOMAIN_GOOGLEFINDMY)
        if not isinstance(bucket, dict):
            return None
        resolver = bucket.get(DATA_EID_RESOLVER)
        if resolver is None or not callable(getattr(resolver, "resolve_eid", None)):
            return None
        return resolver

    def resolve(self, payloads: Sequence[bytes]) -> tuple[list[FmdnTag], ResolutionStatus]:
        """Try the full payloads first, then bare EIDs; the first attempt that matches wins."""
        resolver = self._get_resolver()
        if resolver is None:
            return [], ResolutionStatus.RESOLVER_UNAVAILABLE

        attempts: list[bytes] = list(payloads)
        for payload in payloads:
            attempts.extend(extract_eid_candidates(payload))

        for attempt in attempts:
            try:
                raw_matches = self._call(resolver, attempt)
            except Exception:  # noqa: BLE001  # a third-party resolver must never break the update loop
                _LOGGER.debug("FMDN resolver raised while resolving an advert", exc_info=True)
                return [], ResolutionStatus.RESOLVER_ERROR
            tags = {tag.device_id: tag for match in raw_matches if (tag := _to_tag(match)) is not None}
            if tags:
                return list(tags.values()), ResolutionStatus.RESOLVED
        return [], ResolutionStatus.NO_KNOWN_EID_MATCH

    @staticmethod
    def _call(resolver: Any, eid: bytes) -> list[Any]:
        """Ask for all matches (shared tags) where supported, else the single best one."""
        resolve_all = getattr(resolver, "resolve_eid_all", None)
        if callable(resolve_all):
            return list(resolve_all(eid) or [])
        single = resolver.resolve_eid(eid)
        return [single] if single is not None else []
