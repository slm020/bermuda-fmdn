"""
Pull Google Find Hub (FMDN) payloads out of a BLE advert's service data.

A tag advertises Eddystone service ``0xFEAA`` with frame type ``0x40`` (or
``0x41`` for unwanted-tracking mode), then a 20 or 32 byte ephemeral identifier
(EID), then an optional hashed-flags byte. The EID rotates every ~17 minutes
and the MAC rotates with it. Adapted from jleinenbach/bermuda (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .const import FMDN_EID_LENGTHS, FMDN_FRAME_TYPES, FMDN_SERVICE_UUID
from .trackers import short_uuid

if TYPE_CHECKING:
    from collections.abc import Mapping


def extract_raw_payloads(service_data: Mapping[str, bytes]) -> list[bytes]:
    """
    Return the unmodified FMDN service-data payloads in an advert.

    The resolver wants the full payload (frame type and flags byte included) so
    that it can detect the geometry itself.
    """
    return [
        bytes(payload)
        for uuid, payload in service_data.items()
        if short_uuid(str(uuid)) == FMDN_SERVICE_UUID and isinstance(payload, bytes | bytearray) and payload
    ]


def extract_eid_candidates(payload: bytes) -> list[bytes]:
    """
    Return bare EID candidates from one raw payload, for resolvers that want only the EID.

    Strips the frame-type byte and takes a 20 or 32 byte prefix (any trailing
    hashed-flags byte is dropped). Payloads that are not FMDN frames give nothing.
    """
    if len(payload) < 1 or payload[0] not in FMDN_FRAME_TYPES:
        return []
    body = payload[1:]
    return [body[:length] for length in FMDN_EID_LENGTHS if len(body) >= length]
