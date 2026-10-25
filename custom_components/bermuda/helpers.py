"""
Pure helper functions for Bermuda.

These functions take Bermuda objects/values and return values, with no hass
access and no I/O, so they can be unit-tested in isolation. Shared by the
options flow steps that offer devices for tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .const import (
    ADDR_TYPE_PRIVATE_BLE_DEVICE,
    BDADDR_TYPE_RANDOM_RESOLVABLE,
    CONF_BLOCKED_DEVICES,
    CONF_BLOCKED_KINDS,
    CONF_DEVICES,
)
from .device_kind import device_kind

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from .bermuda_device import BermudaDevice


def is_device_selectable(device: BermudaDevice, stale_cutoff: float) -> bool:
    """
    Whether a device should be offered in the tracking pickers.

    Scanners aren't tracked; Private BLE devices configure themselves; and a
    random (resolvable) MAC not seen since ``stale_cutoff`` (monotonic stamp)
    has almost certainly rotated away and would only clutter the list.
    """
    if device.is_scanner or device.address_type == ADDR_TYPE_PRIVATE_BLE_DEVICE:
        return False
    return not (device.address_type == BDADDR_TYPE_RANDOM_RESOLVABLE and device.last_seen < stale_cutoff)


def is_blocked(device: BermudaDevice, options: Mapping[str, Any]) -> bool:
    """
    Whether the user has hidden this device ("never show it again").

    Blocked by address, or by *kind*: every device of a hidden kind (handy for the
    neighbours' phones, whose addresses keep changing). Devices the user explicitly
    tracks are never hidden by kind.
    """
    address = device.address.upper()
    if address in {a.upper() for a in options.get(CONF_BLOCKED_DEVICES, []) if isinstance(a, str)}:
        return True
    kinds = options.get(CONF_BLOCKED_KINDS, [])
    if not kinds:
        return False
    if address in {a.upper() for a in options.get(CONF_DEVICES, []) if isinstance(a, str)}:
        return False
    return device_kind(device) in kinds
