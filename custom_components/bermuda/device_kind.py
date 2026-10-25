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
from homeassistant.helpers import device_registry as dr

from .const import (
    BDADDR_TYPE_RANDOM_RESOLVABLE,
    BDADDR_TYPE_RANDOM_STATIC,
    BDADDR_TYPE_RANDOM_UNRESOLVABLE,
    DOMAIN,
)
from .trackers import short_uuid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant

    from .bermuda_advert import BermudaAdvert
    from .bermuda_device import BermudaDevice

COMPANY_APPLE: Final = 0x004C
COMPANY_MICROSOFT: Final = 0x0006
COMPANY_SAMSUNG: Final = 0x0075
COMPANY_GOOGLE: Final = 0x00E0
COMPANY_AMAZON: Final = 0x0171

RSSI_RIGHT_BY: Final = -55  # dBm: stronger than this and the device is right next to a proxy
RSSI_CLOSE: Final = -75  # dBm: weaker than this and it is not meaningfully near any proxy
MAX_LABEL_DISTANCE: Final = 10.0  # metres: beyond this a device is "far from everywhere"

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


def strongest_advert(device: BermudaDevice) -> BermudaAdvert | None:
    """The advert with the best signal: the proxy that hears this device loudest."""
    heard = [advert for advert in device.adverts.values() if advert.rssi is not None]
    return max(heard, key=lambda advert: advert.rssi, default=None)  # type: ignore[arg-type,return-value]


def strongest_rssi(device: BermudaDevice) -> float | None:
    """The best signal strength (dBm) any proxy currently reports for the device."""
    advert = strongest_advert(device)
    return advert.rssi if advert is not None else None


def device_room(device: BermudaDevice) -> tuple[str, str] | None:
    """(area id, area name) of the proxy hearing the device loudest: roughly, where it is."""
    advert = strongest_advert(device)
    if advert is None or not advert.area_id:
        return None
    return advert.area_id, advert.area_name or advert.area_id


def signal_phrase(rssi: float | None, room: str | None) -> str:
    """
    Say where a device is, in words: "right by Study", "close to Study", or "far from everywhere".

    A faint signal is not "near" anywhere, so below the cut-off no room is named at all.
    """
    if rssi is None:
        return "no signal"
    if rssi >= RSSI_RIGHT_BY:
        return f"right by {room}" if room else "very close"
    if rssi >= RSSI_CLOSE:
        return f"close to {room}" if room else "close"
    return "far from everywhere"


def distance_text(metres: float) -> str:
    """A distance a person can use: 70 cm, 3.2 m, 8 m."""
    if metres < 0.95:
        return f"{max(10, round(metres * 100, -1)):.0f} cm"
    return f"{metres:.1f} m" if metres < 10 else f"{metres:.0f} m"


def nearest_estimate(device: BermudaDevice) -> tuple[float, str | None] | None:
    """
    Estimated distance (metres) to the nearest proxy, with that proxy's room.

    Uses Bermuda's own smoothed, calibrated distances, so it is only as good as the calibration
    (see the Calibrate screen). A distance is a better clue than "near" or "far": 70 cm is
    probably your phone; 3 m from the living room might be a neighbour a floor up.
    """
    estimates = [(a.rssi_distance, a.area_name) for a in device.adverts.values() if a.rssi_distance is not None]
    return min(estimates, key=lambda e: e[0]) if estimates else None


def where_text(device: BermudaDevice) -> str:
    """Where a device is, in words: "~70 cm from Study", or "far from everywhere"."""
    estimate = nearest_estimate(device)
    if estimate is not None:
        metres, room = estimate
        if metres > MAX_LABEL_DISTANCE:
            return "far from everywhere"
        return f"~{distance_text(metres)} from {room}" if room else f"~{distance_text(metres)} from a proxy"
    room_info = device_room(device)
    return signal_phrase(strongest_rssi(device), room_info[1] if room_info else None)


def identity_hint(device: BermudaDevice) -> str | None:
    """
    Raw clues for a device that could not be identified: its company id, or a service id.

    Searchable ("0x004C" or "0xFE2C"), so the user can look it up when the label says nothing.
    """
    if device_kind(device) != "Unidentified device":
        return None
    advert = _latest_advert(device)
    if advert is None:
        return None
    if advert.manufacturer_data and (ids := sorted(advert.manufacturer_data[0])):
        return f"company 0x{ids[0]:04X}"
    uuids = sorted({short_uuid(str(u)) for u in advert.service_uuids})
    if advert.service_data:
        uuids = sorted({*uuids, *(short_uuid(str(u)) for u in advert.service_data[0])})
    return f"service 0x{uuids[0]}" if uuids else None


def device_label(device: BermudaDevice) -> str:
    """
    Short picker label: the name you gave it (if any), what it is, its broadcast name, how far away, end of address.

    Example: ``Windows PC / Microsoft device · "DESKTOP-1" · ~70 cm from Study · …EE:FF``.
    """
    chosen_name = device.name_by_user or device.name_subentry  # a name the user gave always leads
    parts = [chosen_name, device_kind(device)] if chosen_name else [device_kind(device)]
    if hint := identity_hint(device):
        parts.append(hint)
    real_name = device.name_bt_local_name or device.name_bt_serviceinfo or device.name_devreg
    if real_name and real_name != chosen_name:
        parts.append(f'"{real_name}"')
    parts.append(where_text(device))
    parts.append(f"…{device.address.upper()[-8:]}")
    return " · ".join(parts)


def ha_device_name(hass: HomeAssistant, device: BermudaDevice) -> str:
    """The device's name as Home Assistant shows it (a rename wins), else Bermuda's own name for it."""
    entry = dr.async_get(hass).async_get_device_by_identifier(DOMAIN, str(device.unique_id))
    return (entry and (entry.name_by_user or entry.name)) or device.name_by_user or device.name_subentry or device.name


_ADDRESS_NOTES: Final = {
    BDADDR_TYPE_RANDOM_RESOLVABLE: "random (resolvable: a phone-style private address that changes)",
    BDADDR_TYPE_RANDOM_UNRESOLVABLE: "random (changes)",
    BDADDR_TYPE_RANDOM_STATIC: "random (fixed until power-cycled)",
}


def _hex(data: bytes, limit: int = 24) -> str:
    """Hex of the first bytes of a payload, with an ellipsis if cut."""
    text = data[:limit].hex(" ")
    return text + (" …" if len(data) > limit else "")


def device_facts(device: BermudaDevice, *, now: float | None = None) -> list[tuple[str, str]]:
    """
    Everything Bermuda knows about a device, as (label, value) rows, for the "Identify" screen.

    Raw ids are given in hex so they can be searched for; nothing is guessed beyond the kind.
    """
    now = monotonic_time_coarse() if now is None else now
    advert = _latest_advert(device)
    rows: list[tuple[str, str]] = [("Looks like", device_kind(device))]
    names = [n for n in (device.name_bt_local_name, device.name_bt_serviceinfo, device.name_devreg) if n]
    rows.append(("Name", ", ".join(dict.fromkeys(names)) if names else "none broadcast"))
    rows.append(("Address", f"{device.address.upper()} · {_ADDRESS_NOTES.get(device.address_type, 'public')}"))
    if device.manufacturer_id is not None or device.manufacturer:
        company = f"0x{device.manufacturer_id:04X}" if device.manufacturer_id is not None else ""
        rows.append(("Company", f"{device.manufacturer or 'unknown'} {company}".strip()))
    if advert is not None:
        for company_id, payload in (advert.manufacturer_data[0] if advert.manufacturer_data else {}).items():
            rows.append((f"Manufacturer data 0x{company_id:04X}", _hex(payload) or "empty"))
        if advert.service_uuids:
            rows.append(("Services", ", ".join(f"0x{short_uuid(str(u))}" for u in advert.service_uuids[:8])))
        for uuid, payload in (advert.service_data[0] if advert.service_data else {}).items():
            rows.append((f"Service data 0x{short_uuid(str(uuid))}", _hex(bytes(payload)) or "empty"))
    heard = sorted(
        (a for a in device.adverts.values() if a.rssi is not None), key=lambda a: a.rssi or -999, reverse=True
    )
    if heard:
        rows.append(
            (
                "Heard by",
                "; ".join(
                    f"{a.name} ({a.area_name or 'no room'}) {a.rssi:.0f} dBm"
                    + (f", ~{distance_text(a.rssi_distance)}" if a.rssi_distance is not None else "")
                    for a in heard[:5]
                ),
            )
        )
    if device.last_seen:
        rows.append(("Last heard", f"{max(0, round(now - device.last_seen))} s ago"))
    return rows
