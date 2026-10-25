"""
The guided calibration wizard, as a mixin for the options flow.

Walks the user through it slowly: pick the phone, then measure it at 1 m and 3 m from one
proxy (this sets how distance is worked out), then optionally at 1 m from each other proxy
(this evens out differences between proxies). Each measurement is taken automatically while
a progress spinner shows. Nothing is saved until the user presses Save, so a half-finished
or abandoned run changes nothing.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.config_entries import OptionsFlow
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .calibration import (
    FAR_CHOICES,
    FAR_METRES,
    MEASURE_SECONDS,
    SAMPLE_INTERVAL,
    apply_scanner_offsets,
    attenuation_from,
    is_plausible_attenuation,
    is_stable,
    offset_for,
    summarise,
)
from .const import (
    CONF_ATTENUATION,
    CONF_REF_POWER,
    CONF_RSSI_OFFSETS,
    DEFAULT_ATTENUATION,
    DEFAULT_REF_POWER,
)
from .device_kind import device_label

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowResult

    from .bermuda_device import BermudaDevice
    from .coordinator import BermudaDataUpdateCoordinator


class CalibrationFlowMixin(OptionsFlow):
    """Calibration steps; the options-flow handler subclasses this and provides the helpers used here."""

    @property
    def options(self) -> dict[str, Any]:
        """The working copy of the options (the handler provides it)."""
        raise NotImplementedError

    if TYPE_CHECKING:

        def _candidates(self, *, include_tracked: bool) -> list[BermudaDevice]:
            """Provided by the handler."""
            ...

        @staticmethod
        def _list_selector(options: list[SelectOptionDict], *, multiple: bool = True) -> SelectSelector:
            """Provided by the handler."""
            ...

        async def _update_options(self) -> ConfigFlowResult:
            """Provided by the handler."""
            ...

        async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
            """Provided by the handler (the main menu)."""
            ...

    _cal: dict[str, Any] | None = None
    _cal_task: asyncio.Task[list[float]] | None = None
    _cal_error: str | None = None

    # --- helpers ------------------------------------------------------------

    @property
    def _cal_coordinator(self) -> BermudaDataUpdateCoordinator:
        return cast("BermudaDataUpdateCoordinator", self.config_entry.runtime_data.coordinator)

    def _cal_proxies(self) -> list[BermudaDevice]:
        """The proxies, in a stable order."""
        return sorted(self._cal_coordinator.get_scanners, key=lambda s: (s.area_name or "", s.name))

    @staticmethod
    def _proxy_name(proxy: BermudaDevice) -> str:
        return f"{proxy.area_name} ({proxy.name})" if proxy.area_name else proxy.name

    def _cal_ref_power(self) -> float:
        assert self._cal is not None  # noqa: S101
        if self._cal.get("ref_power") is not None:
            return float(self._cal["ref_power"])
        return float(self.options.get(CONF_REF_POWER, DEFAULT_REF_POWER))

    def _cal_status(self) -> str:
        """A short progress summary for the hub screen."""
        assert self._cal is not None  # noqa: S101
        cal = self._cal
        if cal.get("ref_power") is None:
            distance = "**Distance:** not measured yet."
        elif cal.get("skipped_far"):
            distance = (
                f"**Distance:** 1 m reads {cal['ref_power']:.1f} dBm, fade rate {cal['attenuation']:.2f}"
                " (your current setting, kept because the far reading was skipped; not saved yet)."
            )
        else:
            distance = (
                f"**Distance:** 1 m reads {cal['ref_power']:.1f} dBm, fade rate {cal['attenuation']:.2f}"
                " (measured, not saved yet)."
            )
        done = [p for p in self._cal_proxies() if p.address in cal["raw_1m"]]
        proxies = ", ".join(self._proxy_name(p) for p in done) if done else "none yet"
        return f"{distance}\n\n**Proxies measured:** {proxies}."

    # --- steps --------------------------------------------------------------

    async def async_step_calibrate(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Calibrate, step 1: pick the phone to hold."""
        if user_input is not None:
            self._cal = {"device": user_input["device"].lower(), "raw_1m": {}, "ref_power": None}
            return await self.async_step_calibrate_hub()
        devices = self._candidates(include_tracked=True)
        if not devices:
            return self.async_abort(reason="no_devices")
        options = [SelectOptionDict(value=d.address.upper(), label=device_label(d)) for d in devices]
        return self.async_show_form(
            step_id="calibrate",
            data_schema=vol.Schema({vol.Required("device"): self._list_selector(options, multiple=False)}),
        )

    async def async_step_calibrate_hub(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Calibrate: what to measure next, or save."""
        return self.async_show_menu(
            step_id="calibrate_hub",
            menu_options=["calibrate_distance", "calibrate_proxy", "calibrate_save", "calibrate_discard"],
            description_placeholders={"status": self._cal_status()},
        )

    async def async_step_calibrate_discard(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Throw the measurements away."""
        self._cal = None
        return await self.async_step_init()

    def _proxy_picker(self, step_id: str, *, mark_done: bool) -> ConfigFlowResult:
        assert self._cal is not None  # noqa: S101
        options = [
            SelectOptionDict(
                value=p.address,
                label=("✓ " if mark_done and p.address in self._cal["raw_1m"] else "") + self._proxy_name(p),
            )
            for p in self._cal_proxies()
        ]
        return self.async_show_form(
            step_id=step_id,
            data_schema=vol.Schema({vol.Required("proxy"): self._list_selector(options, multiple=False)}),
        )

    async def async_step_calibrate_distance(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Distance: pick the proxy you can stand 1 m and 3 m from."""
        assert self._cal is not None  # noqa: S101
        if user_input is not None:
            self._cal.update(goal="1m", target=user_input["proxy"], reference=user_input["proxy"])
            return await self.async_step_calibrate_ready()
        return self._proxy_picker("calibrate_distance", mark_done=False)

    async def async_step_calibrate_proxy(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Proxy: pick which proxy to even out."""
        assert self._cal is not None  # noqa: S101
        if user_input is not None:
            self._cal.update(goal="proxy", target=user_input["proxy"])
            return await self.async_step_calibrate_ready()
        return self._proxy_picker("calibrate_proxy", mark_done=True)

    async def async_step_calibrate_ready(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Tell the user exactly where to hold the phone, then measure when they press Submit."""
        assert self._cal is not None  # noqa: S101
        if user_input is not None:
            if self._cal["goal"] == "far":
                if user_input.get("skip"):
                    return await self._cal_skip_far()
                self._cal["far_metres"] = float(user_input.get("distance", FAR_METRES))
            return await self.async_step_calibrate_measure()
        proxy = self._cal_coordinator.devices.get(self._cal["target"])
        name = self._proxy_name(proxy) if proxy else self._cal["target"]
        far_metres = float(self._cal.get("far_metres", FAR_METRES))
        where = {"1m": "**1 m**", "far": f"**{far_metres:.0f} m**", "proxy": "**1 m**"}[self._cal["goal"]]
        step = {
            "1m": "Step 1 of 2.",
            "far": (
                "Step 2 of 2: move back, in a straight line. If this proxy can't hear the phone that far, "
                "pick a nearer distance below, or tick Skip to keep your current fade rate."
            ),
            "proxy": "",
        }[self._cal["goal"]]
        schema: dict[Any, Any] = {}
        if self._cal["goal"] == "far":
            schema = {
                vol.Required("distance", default=str(int(far_metres))): SelectSelector(
                    SelectSelectorConfig(
                        options=[SelectOptionDict(value=str(int(m)), label=f"{int(m)} m") for m in FAR_CHOICES],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional("skip", default=False): BooleanSelector(),
            }
        errors = {"base": self._cal_error} if self._cal_error else None
        self._cal_error = None
        return self.async_show_form(
            step_id="calibrate_ready",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={
                "instruction": (
                    f"{step} Hold the phone {where} from **{name}**, screen on and still, with nothing "
                    f"between you. Press Submit; measuring takes about {MEASURE_SECONDS} seconds."
                ).strip()
            },
        )

    async def _measure(self) -> list[float]:
        """Listen for MEASURE_SECONDS and collect the phone's raw signal strength as the target proxy hears it."""
        assert self._cal is not None  # noqa: S101
        samples: list[float] = []
        last_stamp = 0.0
        end = time.monotonic() + MEASURE_SECONDS
        while time.monotonic() < end:
            device = self._cal_coordinator.devices.get(self._cal["device"])
            advert = device.get_scanner(self._cal["target"]) if device else None
            if advert is not None and advert.rssi is not None and advert.stamp > last_stamp:
                samples.append(advert.rssi)
                last_stamp = advert.stamp
            await asyncio.sleep(SAMPLE_INTERVAL)
        return samples

    async def async_step_calibrate_measure(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Measure, showing a progress spinner meanwhile."""
        if self._cal_task is None:
            self._cal_task = self.hass.async_create_task(self._measure())
        if not self._cal_task.done():
            return self.async_show_progress(
                step_id="calibrate_measure", progress_action="measuring", progress_task=self._cal_task
            )
        return self.async_show_progress_done(next_step_id="calibrate_measured")

    async def async_step_calibrate_measured(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Judge the measurement and move on, or ask for it to be repeated."""
        assert self._cal is not None  # noqa: S101
        assert self._cal_task is not None  # noqa: S101
        cal = self._cal
        samples, self._cal_task = self._cal_task.result(), None
        measurement = summarise(samples)
        if measurement is None:
            self._cal_error = "no_signal"
            return await self.async_step_calibrate_ready()
        if not is_stable(measurement):
            self._cal_error = "unstable"
            return await self.async_step_calibrate_ready()

        goal, target = cal["goal"], cal["target"]
        if goal == "1m":
            cal["raw_1m"][target] = measurement.median
            cal["adjusted_1m"] = measurement.median + self._cal_offset_before(target)
            cal["goal"] = "far"
            return await self.async_step_calibrate_ready()
        if goal == "far":
            adjusted_far = measurement.median + self._cal_offset_before(target)
            attenuation = attenuation_from(cal["adjusted_1m"], adjusted_far, float(cal.get("far_metres", FAR_METRES)))
            if not is_plausible_attenuation(attenuation):
                self._cal_error = "inconsistent"
                cal["goal"] = "1m"  # the pair is suspect: redo both
                return await self.async_step_calibrate_ready()
            cal["ref_power"] = round(cal["adjusted_1m"], 1)
            cal["attenuation"] = round(attenuation, 2)
        else:
            cal["raw_1m"][target] = measurement.median
        cal["goal"] = None
        return await self.async_step_calibrate_hub()

    async def _cal_skip_far(self) -> ConfigFlowResult:
        """Keep the user's current fade rate: only the 1 m reading is taken from this wizard."""
        assert self._cal is not None  # noqa: S101
        cal = self._cal
        cal["ref_power"] = round(cal["adjusted_1m"], 1)
        cal["attenuation"] = float(self.options.get(CONF_ATTENUATION, DEFAULT_ATTENUATION))
        cal["skipped_far"] = True
        cal["goal"] = None
        return await self.async_step_calibrate_hub()

    def _cal_offset_before(self, address: str) -> float:
        """A proxy's saved offset: readings are adjusted by it, so distance maths must include it."""
        return float(self._cal_coordinator.options.get(CONF_RSSI_OFFSETS, {}).get(address, 0.0))

    async def async_step_calibrate_save(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Write the results: distance settings, and each measured proxy's offset."""
        assert self._cal is not None  # noqa: S101
        cal = self._cal
        if cal.get("ref_power") is not None:
            self.options[CONF_REF_POWER] = cal["ref_power"]
            self.options[CONF_ATTENUATION] = cal["attenuation"]
        ref = self._cal_ref_power()
        if cal["raw_1m"]:
            apply_scanner_offsets(
                self.hass, self.config_entry, {a: offset_for(ref, raw) for a, raw in cal["raw_1m"].items()}
            )
        self._cal = None
        return await self._update_options()
