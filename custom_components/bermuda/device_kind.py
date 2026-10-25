"""
Human-friendly descriptions of Bluetooth devices, for the pickers and the hide list.

Bluetooth adverts rarely say "iPhone 15" or "Galaxy S24"; most only carry a
company id, a few service ids, and sometimes a name. This module turns what *is*
there into the most honest label it can ("Apple device (iPhone, iPad, Mac or
Watch)", "Windows PC", "Heart-rate monitor") instead of a bare MAC address, and
never guesses a model it cannot know.

``describe_kind`` is pure (plain data in, string out) so it can be tested alone;
``device_kind`` and ``device_label`` adapt a BermudaDevice to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from bluetooth_data_tools import monotonic_time_coarse

from .const import (
    BDADDR_TYPE_RANDOM_RESERVED,
    BDADDR_TYPE_RANDOM_RESOLVABLE,
    BDADDR_TYPE_RANDOM_STATIC,
    BDADDR_TYPE_RANDOM_UNRESOLVABLE,
)
from .trackers import short_uuid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .bermuda_advert import BermudaAdvert
    from .bermuda_device import BermudaDevice

COMPANY_APPLE: Final = 0x004C
COMPANY_MICROSOFT: Final = 0x0006
COMPANY_SAMSUNG: Final = 0x0075
COMPANY_GOOGLE: Final = 0x00E0
COMPANY_AMAZON: Final = 0x0171

# Apple "Continuity" message type (first byte of Apple manufacturer data) -> what is advertising.
_APPLE_TYPES: Final[dict[int, str]] = {
    0x02: "iBeacon",
    0x05: "Apple device (AirDrop)",
    0x06: "HomeKit accessory",
    0x07: "AirPods / Beats earbuds",
    0x09: "Apple TV / HomePod / AirPlay speaker",
    0x0C: "Apple device (iPhone, iPad, Mac or Watch)",
    0x0F: "Apple device (iPhone, iPad, Mac or Watch)",
    0x10: "Apple device (iPhone, iPad, Mac or Watch)",
}

# GATT / vendor service uuid (16-bit, upper-case) -> kind, for devices with no company id worth using.
_SERVICE_KINDS: Final[dict[str, str]] = {
    "180D": "Heart-rate monitor",
    "1812": "Keyboard / mouse / controller (HID)",
    "181A": "Environment sensor",
    "1816": "Cycling sensor",
    "1814": "Running sensor",
    "181D": "Smart scale",
    "1826": "Fitness machine",
    "110B": "Bluetooth speaker / headphones",
    "110A": "Bluetooth speaker / headphones",
    "FE2C": "Android / Fast Pair device (earbuds, watch, phone...)",
    "FD6F": "Phone (exposure-notification beacon)",
    "FEF3": "Google Nest / Chromecast",
}

_NAME_HINTS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("iphone", "ipad", "macbook", "imac", "apple watch", "airpods"), "Apple device"),
    (("galaxy", "pixel", "oneplus", "xiaomi", "redmi", "huawei"), "Android phone / watch"),
    (("bravia", "[tv]", " tv", "roku", "fire tv"), "TV"),
    (("buds", "headphone", "earbud", "soundbar", "speaker", "jbl", "bose", "sony wh", "sony wf"), "Audio device"),
    (("watch", "band", "fitbit", "garmin"), "Watch / fitness band"),
    (("printer", "hp ", "canon", "epson", "brother"), "Printer"),
)

_RANDOM_TYPES: Final = frozenset(
    {
        BDADDR_TYPE_RANDOM_RESOLVABLE,
        BDADDR_TYPE_RANDOM_UNRESOLVABLE,
        BDADDR_TYPE_RANDOM_STATIC,
        BDADDR_TYPE_RANDOM_RESERVED,
    }
)


def describe_kind(  # noqa: PLR0911 - one early return per kind of evidence reads best
    *,
    manufacturer_data: Mapping[int, bytes],
    service_uuids: set[str],
    name: str | None,
    tracker_type: str | None,
    manufacturer: str | None,
) -> str:
    """
    Best honest description of a device from what its adverts carry.

    Order: a recognised item-tracker, then vendor-specific payloads (Apple Continuity,
    Microsoft, Samsung...), then well-known services, then hints in the name, then just
    the company name. ``service_uuids`` holds 16-bit upper-case short forms.
    """
    if tracker_type:
        return tracker_type

    apple = manufacturer_data.get(COMPANY_APPLE)
    if apple is not None:
        return _APPLE_TYPES.get(apple[0], "Apple device") if apple else "Apple device"
    if COMPANY_MICROSOFT in manufacturer_data:
        return "Windows PC / Microsoft device"
    if COMPANY_SAMSUNG in manufacturer_data:
        return "Samsung device (phone, TV, watch or earbuds)"
    if COMPANY_GOOGLE in manufacturer_data:
        return "Google / Android device"
    if COMPANY_AMAZON in manufacturer_data:
        return "Amazon device (Echo, Fire, Ring...)"

    for uuid in sorted(service_uuids):
        if uuid in _SERVICE_KINDS:
            return _SERVICE_KINDS[uuid]

    lowered = (name or "").lower()
    for needles, kind in _NAME_HINTS:
        if any(needle in lowered for needle in needles):
            return kind

    if manufacturer:
        return f"{manufacturer} device"
    return "Unidentified device"


def is_random_address(device: BermudaDevice) -> bool:
    """Whether the device uses a random (rotating or private) Bluetooth address."""
    return device.address_type in _RANDOM_TYPES


def _latest_advert(device: BermudaDevice) -> BermudaAdvert | None:
    """The advert with the newest timestamp, or None."""
    return max(device.adverts.values(), key=lambda advert: advert.stamp or 0.0, default=None)


def device_kind(device: BermudaDevice) -> str:
    """Friendly kind label for a BermudaDevice, from its most recent advert."""
    advert = _latest_advert(device)
    manufacturer_data: Mapping[int, bytes] = advert.manufacturer_data[0] if advert and advert.manufacturer_data else {}
    uuids: set[str] = {short_uuid(str(u)) for u in advert.service_uuids} if advert else set()
    if advert and advert.service_data:
        uuids |= {short_uuid(str(u)) for u in advert.service_data[0]}
    return describe_kind(
        manufacturer_data=manufacturer_data,
        service_uuids=uuids,
        name=device.name_bt_local_name or device.name_bt_serviceinfo,
        tracker_type=device.tracker_type,
        manufacturer=device.manufacturer,
    )


def strongest_rssi(device: BermudaDevice) -> float | None:
    """The best signal strength (dBm) any proxy currently reports for the device."""
    readings = [advert.rssi for advert in device.adverts.values() if advert.rssi is not None]
    return max(readings) if readings else None


def _signal_text(rssi: float | None) -> str:
    """Signal strength with a plain-English distance hint: the closest things are usually yours."""
    if rssi is None:
        return "no signal"
    if rssi >= -55:
        hint = "very close"
    elif rssi >= -70:
        hint = "near"
    elif rssi >= -85:
        hint = "far"
    else:
        hint = "very weak"
    return f"{rssi:.0f} dBm, {hint}"


def device_label(device: BermudaDevice, *, now: float | None = None) -> str:
    """
    One-line picker label: kind, real name (if it has one), signal, age, short address.

    Example: ``Apple device (iPhone, iPad, Mac or Watch) · "Steve's phone" · -48 dBm, very close · 3s ago · …EE:FF``.
    """
    now = monotonic_time_coarse() if now is None else now
    parts = [device_kind(device)]
    real_name = device.name_bt_local_name or device.name_bt_serviceinfo or device.name_devreg
    if real_name:
        parts.append(f'"{real_name}"')
    parts.append(_signal_text(strongest_rssi(device)))
    if device.last_seen:
        parts.append(f"{max(0, round(now - device.last_seen))}s ago")
    parts.append(f"…{device.address.upper()[-8:]}")
    return " · ".join(parts)
