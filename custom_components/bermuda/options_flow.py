"""
Options (and calibration) flow for Bermuda BLE Trilateration.

Split out of config_flow.py: holds the large options/calibration wizard
(BermudaOptionsFlowHandler) and its inline UI-text table. Imports are broad
and pruned by ruff.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.config_entries import OptionsFlow
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)
from homeassistant.helpers.translation import async_get_translations

from .const import (
    _LOGGER,
    CONF_AREA_ENTITIES,
    CONF_AREA_ENTITY_DISTANCE,
    CONF_AREA_ENTITY_DISTANCES,
    CONF_ATTENUATION,
    CONF_BLOCKED_DEVICES,
    CONF_BLOCKED_KINDS,
    CONF_DEVICES,
    CONF_DEVTRACK_TIMEOUT,
    CONF_IRK,
    CONF_LEARNING_ENABLED,
    CONF_LEARNING_SUSTAINED,
    CONF_LEARNING_THRESHOLD,
    CONF_MAX_RADIUS,
    CONF_MAX_VELOCITY,
    CONF_REF_POWER,
    CONF_SMOOTHING_SAMPLES,
    CONF_TRACK_CATEGORIES,
    CONF_UPDATE_INTERVAL,
    DEFAULT_AREA_ENTITY_DISTANCE,
    DEFAULT_ATTENUATION,
    DEFAULT_DEVTRACK_TIMEOUT,
    DEFAULT_LEARNING_ENABLED,
    DEFAULT_LEARNING_SUSTAINED,
    DEFAULT_LEARNING_THRESHOLD,
    DEFAULT_MAX_RADIUS,
    DEFAULT_MAX_VELOCITY,
    DEFAULT_REF_POWER,
    DEFAULT_SMOOTHING_SAMPLES,
    DEFAULT_UPDATE_INTERVAL,
    DISTANCE_INFINITE,
    DOMAIN,
    METADEVICE_FMDN_DEVICE,
    NAME,
    OPT_MIN_ATTENUATION,
    OPT_MIN_DEVTRACK_TIMEOUT,
    OPT_MIN_LEARNING_SUSTAINED,
    OPT_MIN_MAX_RADIUS,
    OPT_MIN_MAX_VELOCITY,
    OPT_MIN_SMOOTHING_SAMPLES,
    OPT_MIN_UPDATE_INTERVAL,
    OPT_REF_POWER_MAX,
    OPT_REF_POWER_MIN,
    PICKER_STALE_AGE,
    SCANNER_AGE_FRESH,
    SCANNER_AGE_WORRY,
    TRACK_CATEGORIES,
)
from .device_kind import device_facts, device_kind, device_label, strongest_rssi
from .helpers import is_blocked, is_device_selectable
from .options_text import _DESCRIPTION_TEXTS
from .private_enrol import async_enrol_private_device
from .util import mac_redact

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowResult

    from .bermuda_device import BermudaDevice
    from .coordinator import BermudaDataUpdateCoordinator


class BermudaOptionsFlowHandler(OptionsFlow):
    """Config flow options handler for bermuda."""

    def __init__(self) -> None:
        """Initialize Bermuda options flow."""
        self.coordinator: BermudaDataUpdateCoordinator
        self.devices: dict[str, BermudaDevice]
        self._translations_cache: dict[str, str] | None = None
        self._options: dict[str, Any] | None = None
        self._identify = ""  # address of the device being looked at on the Identify screen

    @property
    def options(self) -> dict[str, Any]:
        """Return a mutable working copy of the config entry options."""
        if self._options is None:
            self._options = deepcopy(dict(self.config_entry.options))
        return self._options

    async def _get_options_translation(self, key: str, **kwargs: str) -> str:
        """
        Get a translated string from options translations.

        Keys starting with "description_text." are resolved from the inline
        _DESCRIPTION_TEXTS dict (not part of HA's translation schema).
        All other keys are fetched via async_get_translations.
        """
        if key.startswith("description_text."):
            sub_key = key.removeprefix("description_text.")
            lang = self.hass.config.language
            texts = _DESCRIPTION_TEXTS.get(lang, _DESCRIPTION_TEXTS["en"])
            text = texts.get(sub_key, _DESCRIPTION_TEXTS["en"].get(sub_key, ""))
        else:
            if self._translations_cache is None:
                self._translations_cache = await async_get_translations(
                    self.hass, self.hass.config.language, "options", integrations=[DOMAIN]
                )
            full_key = f"component.{DOMAIN}.options.{key}"
            text = self._translations_cache.get(full_key, "")
        if kwargs:
            with contextlib.suppress(KeyError, IndexError, ValueError):
                text = text.format(**kwargs)
        return text

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:  # pylint: disable=unused-argument
        """Manage the options."""
        self.coordinator = self.config_entry.runtime_data.coordinator
        self.devices = self.coordinator.devices

        messages = {}
        active_devices = self.coordinator.count_active_devices()
        active_scanners = self.coordinator.count_active_scanners()

        messages["device_counter_active"] = f"{active_devices}"
        messages["device_counter_devices"] = f"{len(self.devices)}"
        messages["scanner_counter_active"] = f"{active_scanners}"
        messages["scanner_counter_scanners"] = f"{len(self.coordinator.scanner_list)}"

        if len(self.coordinator.scanner_list) == 0:
            messages["status"] = await self._get_options_translation("error.no_scanners")
        elif active_devices == 0:
            messages["status"] = await self._get_options_translation("error.no_devices")
        else:
            messages["status"] = await self._get_options_translation("error.some_active")

        # Build a markdown table of scanners so the user can see what's up.
        t_title = await self._get_options_translation("description_text.scanner_table_title")
        t_col_scanner = await self._get_options_translation("description_text.scanner_table_col_scanner")
        t_col_address = await self._get_options_translation("description_text.scanner_table_col_address")
        t_col_last_ad = await self._get_options_translation("description_text.scanner_table_col_last_ad")
        t_seconds_ago = await self._get_options_translation("description_text.seconds_ago")
        scanner_table = f"\n\n{t_title}\n\n|{t_col_scanner}|{t_col_address}|{t_col_last_ad}|\n|---|---|---:|\n"
        # Use emoji to indicate if age is "good"
        for scanner in self.coordinator.get_active_scanner_summary():
            age = int(scanner.get("last_stamp_age", DISTANCE_INFINITE))
            if age < SCANNER_AGE_FRESH:
                status = '<ha-icon icon="mdi:check-circle-outline"></ha-icon>'
            elif age < SCANNER_AGE_WORRY:
                status = '<ha-icon icon="mdi:alert-outline"></ha-icon>'
            else:
                status = '<ha-icon icon="mdi:skull-crossbones"></ha-icon>'
            # Remove centre octets from mac for condensed, privatised display
            shortmac = mac_redact(scanner.get("address", "ERR"))
            scanner_table += (
                f"| {scanner.get('name', 'NAME_ERR')}| [{shortmac}]"
                f"| {status} {(scanner.get('last_stamp_age', DISTANCE_INFINITE)):.2f} {t_seconds_ago}|\n"
            )
        messages["status"] += scanner_table

        # Menu labels come from translations (options.step.init.menu_options.*),
        # so every menu entry is localised like the rest of the flow.
        return self.async_show_menu(
            step_id="init",
            menu_options=["add", "hide", "identify", "tracked", "hidden", "google", "settings"],
            description_placeholders=messages,
        )

    # --- device lists -------------------------------------------------------

    def _candidates(self, *, include_tracked: bool) -> list[BermudaDevice]:
        """
        Devices worth offering in a picker: seen recently, not hidden, strongest signal first.

        Proxies, self-configuring Private BLE devices and long-gone random MACs are left out
        (see is_device_selectable), as is anything the user has hidden. Strongest first,
        because the things closest to a proxy are usually yours and the neighbours' are faint.
        """
        coordinator = self.config_entry.runtime_data.coordinator
        tracked = {a.upper() for a in self.options.get(CONF_DEVICES, []) if isinstance(a, str)}
        stale_cutoff = monotonic_time_coarse() - PICKER_STALE_AGE
        found = [
            device
            for device in coordinator.devices.values()
            if is_device_selectable(device, stale_cutoff)
            and not is_blocked(device, self.options)
            and (include_tracked or not (device.address.upper() in tracked or device.create_sensor))
        ]

        def _signal(device: BermudaDevice) -> float:
            rssi = strongest_rssi(device)
            return rssi if rssi is not None else -9999.0

        found.sort(key=_signal, reverse=True)
        return found

    def _hide(self, addresses: list[str], kinds: list[str]) -> None:
        """Add addresses and kinds to the hide lists (and drop them from tracking)."""
        if addresses:
            blocked = [a.upper() for a in self.options.get(CONF_BLOCKED_DEVICES, []) if isinstance(a, str)]
            self.options[CONF_BLOCKED_DEVICES] = list(dict.fromkeys(blocked + [a.upper() for a in addresses]))
            self.options[CONF_DEVICES] = [
                a for a in self.options.get(CONF_DEVICES, []) if not (isinstance(a, str) and a.upper() in addresses)
            ]
        if kinds:
            self.options[CONF_BLOCKED_KINDS] = list(dict.fromkeys([*self.options.get(CONF_BLOCKED_KINDS, []), *kinds]))

    @staticmethod
    def _list_selector(options: list[SelectOptionDict], *, multiple: bool = True) -> SelectSelector:
        """Checkbox list (or radio list if single-choice) rather than a dropdown."""
        return SelectSelector(
            SelectSelectorConfig(options=options, multiple=multiple, sort=False, mode=SelectSelectorMode.LIST)
        )

    # --- add ----------------------------------------------------------------

    async def async_step_add(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Add devices: tick the ones to track."""
        if user_input is not None:
            chosen = [a.upper() for a in user_input.get("devices", [])]
            current = [a.upper() for a in self.options.get(CONF_DEVICES, []) if isinstance(a, str)]
            self.options[CONF_DEVICES] = list(dict.fromkeys(current + chosen))
            return await self._update_options()
        devices = self._candidates(include_tracked=False)
        options = [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices]
        return self.async_show_form(
            step_id="add",
            data_schema=vol.Schema({vol.Optional("devices", default=[]): self._list_selector(options)}),
            description_placeholders={"count": str(len(options))},
        )

    # --- hide ---------------------------------------------------------------

    async def async_step_hide(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """
        Hide devices, 1/2: hide whole kinds at once (every Apple device, every Windows PC...).

        Phones and gadgets change their address constantly, so hiding by kind is what lasts.
        Devices you track by hand are never hidden by kind.
        """
        if user_input is not None:
            self._hide([], list(user_input.get("kinds", [])))
            if user_input.get("individually"):
                return await self.async_step_hide_pick()
            return await self._update_options()
        kinds = Counter(device_kind(d) for d in self._candidates(include_tracked=False))
        options = [SelectOptionDict(value=kind, label=f"{kind} ({count})") for kind, count in kinds.most_common()]
        return self.async_show_form(
            step_id="hide",
            data_schema=vol.Schema(
                {
                    vol.Optional("kinds", default=[]): self._list_selector(options),
                    vol.Optional("individually", default=False): BooleanSelector(),
                }
            ),
        )

    async def async_step_hide_pick(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Hide devices, step 2 of 2: tick the individual devices to hide."""
        if user_input is not None:
            self._hide([a.upper() for a in user_input.get("devices", [])], [])
            return await self._update_options()
        devices = self._candidates(include_tracked=False)
        options = [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices]
        return self.async_show_form(
            step_id="hide_pick",
            data_schema=vol.Schema({vol.Optional("devices", default=[]): self._list_selector(options)}),
            description_placeholders={"count": str(len(options))},
        )

    # --- identify -----------------------------------------------------------

    async def async_step_identify(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Identify a device, step 1 of 2: pick one."""
        if user_input is not None:
            self._identify = user_input["device"]
            return await self.async_step_identify_detail()
        devices = self._candidates(include_tracked=True)
        options = [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices]
        return self.async_show_form(
            step_id="identify",
            data_schema=vol.Schema({vol.Required("device"): self._list_selector(options, multiple=False)}),
        )

    async def async_step_identify_detail(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Identify a device, step 2 of 2: everything Bermuda knows about it."""
        if user_input is not None:
            return await self.async_step_init()
        coordinator = self.config_entry.runtime_data.coordinator
        device = coordinator.devices.get(self._identify.lower()) or coordinator.devices.get(self._identify)
        rows = device_facts(device) if device is not None else [("Device", "no longer seen")]
        table = "\n".join(f"| {label} | {value} |" for label, value in rows)
        return self.async_show_form(
            step_id="identify_detail",
            data_schema=vol.Schema({}),
            description_placeholders={"facts": f"| | |\n|---|---|\n{table}"},
        )

    # --- manage -------------------------------------------------------------

    async def async_step_tracked(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Tracked devices: untick to stop tracking."""
        tracked = [a.upper() for a in self.options.get(CONF_DEVICES, []) if isinstance(a, str)]
        if user_input is not None:
            self.options[CONF_DEVICES] = [a.upper() for a in user_input.get("devices", [])]
            return await self._update_options()
        known = {d.address.upper(): d for d in self.config_entry.runtime_data.coordinator.devices.values()}
        options = [
            SelectOptionDict(
                value=address, label=device_label(known[address]) if address in known else f"{address} (not seen now)"
            )
            for address in tracked
        ]
        return self.async_show_form(
            step_id="tracked",
            data_schema=vol.Schema({vol.Optional("devices", default=tracked): self._list_selector(options)}),
            description_placeholders={"count": str(len(options))},
        )

    async def async_step_hidden(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Hidden devices and kinds: untick to show them again."""
        hidden_devices = [a.upper() for a in self.options.get(CONF_BLOCKED_DEVICES, []) if isinstance(a, str)]
        hidden_kinds = list(self.options.get(CONF_BLOCKED_KINDS, []))
        if user_input is not None:
            self.options[CONF_BLOCKED_DEVICES] = [a.upper() for a in user_input.get("devices", [])]
            self.options[CONF_BLOCKED_KINDS] = list(user_input.get("kinds", []))
            return await self._update_options()
        known = {d.address.upper(): d for d in self.config_entry.runtime_data.coordinator.devices.values()}
        device_options = [
            SelectOptionDict(
                value=address, label=device_label(known[address]) if address in known else f"{address} (not seen now)"
            )
            for address in hidden_devices
        ]
        kind_options = [SelectOptionDict(value=kind, label=kind) for kind in hidden_kinds]
        return self.async_show_form(
            step_id="hidden",
            data_schema=vol.Schema(
                {
                    vol.Optional("kinds", default=hidden_kinds): self._list_selector(kind_options),
                    vol.Optional("devices", default=hidden_devices): self._list_selector(device_options),
                }
            ),
            description_placeholders={"count": str(len(hidden_devices) + len(hidden_kinds))},
        )

    # --- google tags --------------------------------------------------------

    async def async_step_google(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Google Find Hub tags: is the sign-in in place, and which tags are known."""
        if user_input is not None:
            return await self.async_step_init()
        coordinator = self.config_entry.runtime_data.coordinator
        info = coordinator.fmdn_diagnostics()
        tags = sorted(
            (d for d in coordinator.metadevices.values() if METADEVICE_FMDN_DEVICE in d.metadevice_type),
            key=lambda d: d.name,
        )
        if not info["resolver_available"]:
            status = "**Not connected yet.** Follow the steps below."
        elif not tags:
            status = "**Connected**, but it knows of no tags yet."
        else:
            now = monotonic_time_coarse()
            lines = [
                f"- {d.name}: heard {round(now - d.last_seen)} s ago" if d.last_seen else f"- {d.name}: not heard yet"
                for d in tags
            ]
            status = "**Connected.** Tags:\n" + "\n".join(lines)
        return self.async_show_form(
            step_id="google",
            data_schema=vol.Schema({}),
            description_placeholders={"status": status},
        )

    # --- settings -----------------------------------------------------------

    async def async_step_settings(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Everything that is not about choosing devices."""
        return self.async_show_menu(
            step_id="settings",
            menu_options=["globalopts", "categories", "area_entities", "enrol_private"],
        )

    async def async_step_categories(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Auto-track every device of a category."""
        if user_input is not None:
            self.options[CONF_TRACK_CATEGORIES] = user_input.get("track_categories", [])
            return await self._update_options()
        category_selector = SelectSelector(
            SelectSelectorConfig(
                options=[SelectOptionDict(value=cat, label=cat) for cat in TRACK_CATEGORIES],
                multiple=True,
                sort=False,
                mode=SelectSelectorMode.LIST,
                translation_key="track_category",
            )
        )
        return self.async_show_form(
            step_id="categories",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "track_categories", default=list(self.options.get(CONF_TRACK_CATEGORIES, []))
                    ): category_selector
                }
            ),
        )

    async def async_step_globalopts(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Global options, grouped into collapsible sections for readability."""
        if user_input is not None:
            # Each section nests its fields; flatten them back into the flat options.
            for value in user_input.values():
                self.options.update(value)
            return await self._update_options()

        def _opt(key: str, default: float) -> float:
            # self.options is a loosely-typed working copy of config_entry.options;
            # these keys are always numeric config values.
            return cast("float", self.options.get(key, default))

        def _float(min_: float | None = None, max_: float | None = None) -> vol.All:
            return vol.All(vol.Coerce(float), vol.Range(min=min_, max=max_))

        def _int(min_: int | None = None) -> vol.All:
            return vol.All(vol.Coerce(int), vol.Range(min=min_))

        data_schema = vol.Schema(
            {
                vol.Required("distance_model"): section(
                    vol.Schema(
                        {
                            vol.Required(CONF_REF_POWER, default=_opt(CONF_REF_POWER, DEFAULT_REF_POWER)): _float(
                                OPT_REF_POWER_MIN, OPT_REF_POWER_MAX
                            ),
                            vol.Required(CONF_ATTENUATION, default=_opt(CONF_ATTENUATION, DEFAULT_ATTENUATION)): _float(
                                OPT_MIN_ATTENUATION
                            ),
                            vol.Required(CONF_MAX_RADIUS, default=_opt(CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS)): _float(
                                OPT_MIN_MAX_RADIUS
                            ),
                        }
                    ),
                    {"collapsed": False},
                ),
                vol.Required("tracking"): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_DEVTRACK_TIMEOUT, default=_opt(CONF_DEVTRACK_TIMEOUT, DEFAULT_DEVTRACK_TIMEOUT)
                            ): _int(OPT_MIN_DEVTRACK_TIMEOUT),
                            vol.Required(
                                CONF_UPDATE_INTERVAL, default=_opt(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
                            ): _float(OPT_MIN_UPDATE_INTERVAL),
                        }
                    ),
                    {"collapsed": True},
                ),
                vol.Required("smoothing"): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_SMOOTHING_SAMPLES, default=_opt(CONF_SMOOTHING_SAMPLES, DEFAULT_SMOOTHING_SAMPLES)
                            ): _int(OPT_MIN_SMOOTHING_SAMPLES),
                            vol.Required(
                                CONF_MAX_VELOCITY, default=_opt(CONF_MAX_VELOCITY, DEFAULT_MAX_VELOCITY)
                            ): _float(OPT_MIN_MAX_VELOCITY),
                        }
                    ),
                    {"collapsed": True},
                ),
                vol.Optional("learning"): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_LEARNING_ENABLED,
                                default=bool(self.options.get(CONF_LEARNING_ENABLED, DEFAULT_LEARNING_ENABLED)),
                            ): bool,
                            vol.Required(
                                CONF_LEARNING_THRESHOLD,
                                default=_opt(CONF_LEARNING_THRESHOLD, DEFAULT_LEARNING_THRESHOLD),
                            ): _float(0.0, 1.0),
                            vol.Required(
                                CONF_LEARNING_SUSTAINED,
                                default=_opt(CONF_LEARNING_SUSTAINED, DEFAULT_LEARNING_SUSTAINED),
                            ): _int(OPT_MIN_LEARNING_SUSTAINED),
                        }
                    ),
                    {"collapsed": True},
                ),
            }
        )

        return self.async_show_form(step_id="globalopts", data_schema=data_schema)

    async def async_step_enrol_private(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """
        Enrol a privacy device (iPhone/Watch) by its IRK.

        Bermuda's proxies can't pair like ESPresense firmware, so we hand the IRK
        to HA's private_ble_device integration; Bermuda then tracks it on its own.
        """
        errors: dict[str, str] = {}
        if user_input is not None:
            error = await async_enrol_private_device(self.hass, user_input[CONF_IRK], user_input.get(CONF_NAME, ""))
            if not error:
                # Nudge the coordinator so the new private device shows up promptly.
                # Best-effort only: enrolment already succeeded, so a refresh
                # hiccup must not fail the flow.
                try:
                    await self.coordinator.async_request_refresh()
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("Post-enrolment refresh failed; device will appear on the next cycle")
                return await self._update_options()
            errors["base" if error == "bluetooth_not_available" else CONF_IRK] = error

        data_schema = vol.Schema(
            {
                vol.Required(CONF_IRK): TextSelector(),
                vol.Optional(CONF_NAME, default=""): TextSelector(),
            }
        )
        return self.async_show_form(step_id="enrol_private", data_schema=data_schema, errors=errors)

    async def async_step_area_entities(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Select presence entities and the global default virtual distance."""
        if user_input is not None:
            self.options[CONF_AREA_ENTITIES] = user_input.get(CONF_AREA_ENTITIES, [])
            self.options[CONF_AREA_ENTITY_DISTANCE] = user_input.get(
                CONF_AREA_ENTITY_DISTANCE, DEFAULT_AREA_ENTITY_DISTANCE
            )
            if self.options[CONF_AREA_ENTITIES]:
                return await self.async_step_area_entities_distance()
            return await self._update_options()

        data_schema = {
            vol.Optional(
                CONF_AREA_ENTITIES,
                default=self.options.get(CONF_AREA_ENTITIES, []),
            ): EntitySelector(EntitySelectorConfig(multiple=True)),
            vol.Optional(
                CONF_AREA_ENTITY_DISTANCE,
                default=self.options.get(CONF_AREA_ENTITY_DISTANCE, DEFAULT_AREA_ENTITY_DISTANCE),
            ): NumberSelector(
                NumberSelectorConfig(min=0.01, max=999, step=0.1, mode=NumberSelectorMode.BOX, unit_of_measurement="m")
            ),
        }
        return self.async_show_form(step_id="area_entities", data_schema=vol.Schema(data_schema))

    async def async_step_area_entities_distance(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Set the per-entity virtual distance, grouped by area for readability."""
        entities = self.options.get(CONF_AREA_ENTITIES, [])
        if user_input is not None:
            distances: dict[str, float] = {}
            for entity_id in entities:
                value = user_input.get(entity_id)
                if value is not None:
                    distances[entity_id] = float(value)
            self.options[CONF_AREA_ENTITY_DISTANCES] = distances
            return await self._update_options()

        existing = self.options.get(CONF_AREA_ENTITY_DISTANCES, {})
        default_distance = self.options.get(CONF_AREA_ENTITY_DISTANCE, DEFAULT_AREA_ENTITY_DISTANCE)
        manager = self.coordinator.area_entity_manager

        # Group the entities by their resolved area, for a readable form.
        area_groups: dict[str, list[str]] = {}
        for entity_id in entities:
            _area_id, area_name = manager.resolve_entity_area(entity_id)
            area_groups.setdefault(area_name or "(no area)", []).append(entity_id)

        data_schema = {}
        for area_name in sorted(area_groups):
            for entity_id in sorted(area_groups[area_name]):
                data_schema[vol.Optional(entity_id, default=existing.get(entity_id, default_distance))] = (
                    NumberSelector(
                        NumberSelectorConfig(
                            min=0.01, max=999, step=0.1, mode=NumberSelectorMode.BOX, unit_of_measurement="m"
                        )
                    )
                )

        summary = "\n".join(
            f"**{area_name}**: " + ", ".join(e.split(".")[-1] for e in sorted(area_groups[area_name]))
            for area_name in sorted(area_groups)
        )
        return self.async_show_form(
            step_id="area_entities_distance",
            data_schema=vol.Schema(data_schema),
            description_placeholders={"area_summary": summary},
        )

    async def _update_options(self) -> ConfigFlowResult:
        """Update config entry options."""
        return self.async_create_entry(title=NAME, data=self.options)
