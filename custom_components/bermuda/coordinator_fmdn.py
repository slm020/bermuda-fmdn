"""
Google Find Hub (FMDN) tag management mixin for the coordinator.

A tag's BLE address rotates with its ephemeral identifier, so each advert is
passed to the EID resolver (see ``fmdn.resolver``). A resolved tag becomes a
metadevice ``fmdn:<device-registry id>`` whose sources are the current (and
recent) rotating MACs, exactly like an iBeacon or Private BLE device. Anything
keyed on the metadevice (micro-location spots, entities) therefore survives
the MAC rotating.

Mixed into BermudaDataUpdateCoordinator.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.const import Platform

from .const import (
    _LOGGER,
    CONF_BLOCKED_DEVICES,
    DOMAIN_GOOGLEFINDMY,
    FMDN_ADDRESS_PREFIX,
    FMDN_RETRY_INTERVAL,
    METADEVICE_FMDN_DEVICE,
    METADEVICE_TYPE_FMDN_SOURCE,
)
from .fmdn_extraction import extract_raw_payloads
from .fmdn_resolver import FmdnResolver, FmdnTag, GoogleFindMyResolver

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.device_registry import DeviceRegistry
    from homeassistant.helpers.entity_registry import EntityRegistry

    from .bermuda_device import BermudaDevice


class BermudaFmdnMixin:
    """Resolve rotating Google Find Hub adverts to stable metadevices."""

    # Class-level defaults so a coordinator that skipped __init__ (as some tests build) stays inert.
    _do_fmdn_init: bool = False

    if TYPE_CHECKING:
        # Provided by BermudaDataUpdateCoordinator (see coordinator.py:__init__).
        hass: HomeAssistant
        er: EntityRegistry
        dr: DeviceRegistry
        metadevices: dict[str, BermudaDevice]

        def _get_or_create_device(self, address: str) -> BermudaDevice: ...
        def _get_device(self, address: str) -> BermudaDevice | None: ...

    def _fmdn_init(self, hass: HomeAssistant) -> None:
        """Set up the resolver interface and bookkeeping."""
        self.fmdn_resolver: FmdnResolver = GoogleFindMyResolver(hass)
        self.fmdn_stats: Counter[str] = Counter()
        # Sources whose advert did not resolve yet -> monotonic stamp of the last attempt.
        self._fmdn_pending: dict[str, float] = {}
        self._do_fmdn_init = True

    # --- per-advert resolution ----------------------------------------------

    def fmdn_handle_advertisement(self, source_device: BermudaDevice, service_data: Mapping[str, bytes]) -> None:
        """
        Resolve an advert's FMDN payload and link its source MAC to the tag's metadevice.

        Called when a device's service data changes. Adverts without an FMDN
        payload are ignored; unresolved ones are retried (see fmdn_retry_unresolved).
        """
        payloads = extract_raw_payloads(service_data)
        if not payloads:
            return
        tags, status = self.fmdn_resolver.resolve(payloads)
        self.fmdn_stats[status.value] += 1
        if not tags:
            self._fmdn_pending[source_device.address] = monotonic_time_coarse()
            return
        self._fmdn_pending.pop(source_device.address, None)
        source_device.metadevice_type.add(METADEVICE_TYPE_FMDN_SOURCE)
        for tag in tags:
            if self._fmdn_is_hidden(tag.device_id):
                continue
            metadevice = self._fmdn_get_metadevice(tag)
            if source_device.address not in metadevice.metadevice_sources:
                metadevice.metadevice_sources.insert(0, source_device.address)
                _LOGGER.debug("FMDN tag %s now advertising from %s", metadevice.name, source_device.address)

    def fmdn_retry_unresolved(self) -> None:
        """Re-offer adverts that did not resolve (resolver not loaded yet, keys not synced...)."""
        if not getattr(self, "_fmdn_pending", None):
            return
        now = monotonic_time_coarse()
        for address, stamp in list(self._fmdn_pending.items()):
            source = self._get_device(address)
            if source is None:
                del self._fmdn_pending[address]
                continue
            if now - stamp < FMDN_RETRY_INTERVAL:
                continue
            latest = next((adv.service_data[0] for adv in source.adverts.values() if adv.service_data), None)
            if latest is None:
                del self._fmdn_pending[address]
                continue
            self.fmdn_handle_advertisement(source, latest)

    def _fmdn_is_hidden(self, device_id: str) -> bool:
        """Whether the user hid this Google device, so it must not be re-created at restart."""
        address = f"{FMDN_ADDRESS_PREFIX}{device_id}".upper()
        return address in {a.upper() for a in self.options.get(CONF_BLOCKED_DEVICES, []) if isinstance(a, str)}

    # --- metadevices --------------------------------------------------------

    def _fmdn_get_metadevice(self, tag: FmdnTag) -> BermudaDevice:
        """Get (creating on first sight) the metadevice for a resolved tag."""
        metadevice = self._get_or_create_device(f"{FMDN_ADDRESS_PREFIX}{tag.device_id}")
        if metadevice.address not in self.metadevices:
            self.metadevices[metadevice.address] = metadevice
        metadevice.fmdn_device_id = tag.device_id
        metadevice.fmdn_canonical_id = tag.canonical_id
        # The user registered this tag with Google Find Hub on purpose, so always track it.
        metadevice.create_sensor = True
        if devreg := self.dr.async_get(tag.device_id):
            metadevice.name_by_user = devreg.name_by_user
            metadevice.name_devreg = devreg.name
            metadevice.make_name()
        return metadevice

    def discover_fmdn_metadevices(self) -> None:
        """
        Seed a metadevice for every device GoogleFindMy-HA knows, before any advert arrives.

        Means entities exist (and restore) at startup instead of waiting for a
        tag to be heard. Only runs when flagged (startup, registry changes).
        """
        if not self._do_fmdn_init:
            return
        self._do_fmdn_init = False
        for entry in self.hass.config_entries.async_entries(DOMAIN_GOOGLEFINDMY, include_disabled=False):
            for entity in self.er.entities.get_entries_for_config_entry_id(entry.entry_id):
                if entity.domain != Platform.DEVICE_TRACKER or entity.device_id is None:
                    continue
                devreg = self.dr.async_get(entity.device_id)
                if devreg is None:
                    continue
                canonical = next(
                    (ident.split(":")[-1] for domain, ident in devreg.identifiers if domain == DOMAIN_GOOGLEFINDMY),
                    "",
                )
                if self._fmdn_is_hidden(entity.device_id):
                    continue
                self._fmdn_get_metadevice(FmdnTag(entity.device_id, canonical, entry.entry_id))

    def fmdn_redirect(self, device: BermudaDevice) -> BermudaDevice:
        """
        Map a rotating source MAC to its tag's stable metadevice (other devices pass through).

        Lets users calibrate micro-locations or ask ``where_is`` by whichever MAC the
        tag has right now and still land on the persistent identity.
        """
        if METADEVICE_TYPE_FMDN_SOURCE not in device.metadevice_type:
            return device
        for metadevice in self.metadevices.values():
            if METADEVICE_FMDN_DEVICE in metadevice.metadevice_type and device.address in metadevice.metadevice_sources:
                return metadevice
        return device

    def fmdn_metadevice_for_registry_id(self, device_id: str) -> BermudaDevice | None:
        """Find the tag metadevice for a Home Assistant device-registry id (eg from a device picker)."""
        return self.metadevices.get(f"{FMDN_ADDRESS_PREFIX}{device_id}")

    def fmdn_diagnostics(self) -> dict[str, Any]:
        """Resolution statistics for the diagnostics download."""
        return {
            "resolver_available": self.fmdn_resolver.available,
            "resolutions": dict(self.fmdn_stats),
            "tags": len([d for d in self.metadevices.values() if METADEVICE_FMDN_DEVICE in d.metadevice_type]),
            "pending_sources": len(self._fmdn_pending),
        }
