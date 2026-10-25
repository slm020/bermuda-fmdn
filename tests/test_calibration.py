"""
Tests for the guided calibration wizard and its maths.

The measurement itself (listening to a phone for ~40 s) is replaced by canned readings for the
flow tests, and exercised separately against a fake advert feed with a short listening time.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from homeassistant.data_entry_flow import FlowResultType

from custom_components.bermuda import options_calibration
from custom_components.bermuda.calibration import (
    Measurement,
    apply_scanner_offsets,
    attenuation_from,
    is_plausible_attenuation,
    is_stable,
    offset_for,
    summarise,
)
from custom_components.bermuda.const import CONF_ATTENUATION, CONF_REF_POWER, SUBENTRY_TYPE_CALIBRATION
from custom_components.bermuda.device_kind import distance_text, where_text
from custom_components.bermuda.options_calibration import CalibrationFlowMixin
from tests.test_device_picker import _hear, _scanner, _submit

# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------


def test_summarise_needs_enough_readings_and_uses_the_median():
    assert summarise([-60.0] * 3) is None
    result = summarise([-60, -61, -60, -59, -60, -90])  # one wild outlier doesn't drag the median
    assert result == Measurement(-60.0, result.spread, 6)
    assert result.count == 6


def test_stability_is_judged_on_spread():
    assert is_stable(Measurement(-60, 2.0, 10))
    assert not is_stable(Measurement(-60, 9.0, 10))


def test_attenuation_recovers_the_exponent_used_to_make_the_readings():
    ref = -60.0
    for exponent in (2.0, 2.7, 3.4):
        rssi_far = ref - 10 * exponent * math.log10(3)
        assert attenuation_from(ref, rssi_far) == pytest.approx(exponent)


@pytest.mark.parametrize(
    ("value", "ok"), [(0.4, False), (1.0, True), (3.0, True), (6.0, True), (7.5, False), (-2, False)]
)
def test_plausible_attenuation_range(value, ok):
    assert is_plausible_attenuation(value) is ok


def test_offset_lines_the_proxy_up_with_the_reference_and_stays_in_range():
    assert offset_for(-60.0, -62.5) == 2.5
    assert offset_for(-60.0, -57.0) == -3.0
    assert offset_for(-60.0, -120.0) == 30.0  # clamped
    assert offset_for(-60.0, -20.0) == -30.0


@pytest.mark.parametrize(
    ("metres", "text"),
    [(0.04, "10 cm"), (0.7, "70 cm"), (0.96, "1.0 m"), (3.24, "3.2 m"), (9.96, "10.0 m"), (14.2, "14 m")],
)
def test_distance_text_is_human_sized(metres, text):
    assert distance_text(metres) == text


# ---------------------------------------------------------------------------
# saving offsets
# ---------------------------------------------------------------------------


def _offsets(entry):
    return {
        se.data["scanner"]: se.data["rssi_offset"]
        for se in entry.subentries.values()
        if se.subentry_type == SUBENTRY_TYPE_CALIBRATION
    }


async def test_apply_scanner_offsets_adds_then_updates(hass, setup_bermuda_entry):
    entry = setup_bermuda_entry
    apply_scanner_offsets(hass, entry, {"aa:aa:aa:aa:aa:01": 2.0, "aa:aa:aa:aa:aa:02": -1.0})
    await hass.async_block_till_done()
    assert _offsets(entry) == {"aa:aa:aa:aa:aa:01": 2.0, "aa:aa:aa:aa:aa:02": -1.0}
    apply_scanner_offsets(hass, entry, {"aa:aa:aa:aa:aa:01": 3.5})
    await hass.async_block_till_done()
    assert _offsets(entry) == {"aa:aa:aa:aa:aa:01": 3.5, "aa:aa:aa:aa:aa:02": -1.0}


# ---------------------------------------------------------------------------
# where_text
# ---------------------------------------------------------------------------


async def test_labels_use_estimated_distance_when_known(hass, setup_bermuda_entry):
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    scanner = _scanner(coordinator, "11:11:11:11:11:11", "study", "Study")
    device = _hear(coordinator, "AA:BB:CC:DD:EE:50", rssi=-50, scanner=scanner)
    advert = next(iter(device.adverts.values()))
    for metres, expected in ((0.7, "~70 cm from Study"), (3.2, "~3.2 m from Study"), (14.0, "far from everywhere")):
        advert.rssi_distance = metres
        assert where_text(device) == expected
    advert.area_name = None
    advert.rssi_distance = 2.0
    assert where_text(device) == "~2.0 m from a proxy"
    advert.rssi_distance = None  # no estimate yet: fall back to describing the signal
    advert.area_name = "Study"
    assert where_text(device) == "right by Study" or where_text(device) == "close to Study"


# ---------------------------------------------------------------------------
# the wizard
# ---------------------------------------------------------------------------


@pytest.fixture
def readings(monkeypatch):
    """Canned measurements: tests append lists of RSSI readings in the order they will be taken."""
    queue: list[list[float]] = []

    async def fake_measure(self):
        return queue.pop(0)

    monkeypatch.setattr(CalibrationFlowMixin, "_measure", fake_measure)
    return queue


async def _finish_progress(hass, result):
    """Let the (instant) measurement finish and follow the flow on to the next screen."""
    while result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
        await hass.async_block_till_done()
        result = await hass.config_entries.options.async_configure(result["flow_id"])
    return result


async def _wizard(hass, entry):
    """Two proxies, a phone heard by both, and the wizard opened on the hub menu."""
    coordinator = entry.runtime_data.coordinator
    lounge = _scanner(coordinator, "11:11:11:11:11:11", "lounge", "Lounge")
    kitchen = _scanner(coordinator, "22:22:22:22:22:22", "kitchen", "Kitchen")
    for scanner in (lounge, kitchen):
        scanner._is_scanner = False  # noqa: SLF001
        coordinator.scanner_list_add(scanner)
        scanner._is_scanner = True  # noqa: SLF001
    _hear(coordinator, "AA:BB:CC:DD:EE:60", rssi=-60, scanner=lounge, name="My phone")
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate"}
    )
    assert result["step_id"] == "calibrate"
    result = await _submit(hass, result, device="AA:BB:CC:DD:EE:60")
    assert result["step_id"] == "calibrate_hub"
    return result, lounge, kitchen


async def _pick(hass, result, step: str, **fields):
    result = await hass.config_entries.options.async_configure(result["flow_id"], user_input={"next_step_id": step})
    return await _submit(hass, result, **fields) if fields else result


async def test_full_wizard_measures_distance_and_a_proxy_then_saves(hass, setup_bermuda_entry, readings):
    result, lounge, kitchen = await _wizard(hass, setup_bermuda_entry)
    assert "not measured yet" in result["description_placeholders"]["status"]

    # Distance: 1 m then 3 m from the lounge proxy.
    result = await _pick(hass, result, "calibrate_distance", proxy=lounge.address)
    assert result["step_id"] == "calibrate_ready"
    assert "Step 1 of 2" in result["description_placeholders"]["instruction"]
    assert "**1 m**" in result["description_placeholders"]["instruction"]
    assert "Lounge" in result["description_placeholders"]["instruction"]
    readings.append([-60.0] * 10)
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["step_id"] == "calibrate_ready"
    assert "Step 2 of 2" in result["description_placeholders"]["instruction"]
    assert "**3 m**" in result["description_placeholders"]["instruction"]
    readings.append([-74.3] * 10)  # 14.3 dB weaker at 3 m  =>  exponent 3.0
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["step_id"] == "calibrate_hub"
    status = result["description_placeholders"]["status"]
    assert "1 m reads -60.0 dBm, fade rate 3.00" in status
    assert "Lounge" in status  # the reference proxy counts as measured

    # Even out the kitchen proxy, which hears the phone 2 dB weaker at 1 m.
    result = await _pick(hass, result, "calibrate_proxy", proxy=kitchen.address)
    assert "**1 m**" in result["description_placeholders"]["instruction"]
    assert "Step" not in result["description_placeholders"]["instruction"]
    readings.append([-62.0] * 10)
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["step_id"] == "calibrate_hub"
    assert "Kitchen" in result["description_placeholders"]["status"]

    # Nothing is saved until the end.
    assert CONF_REF_POWER not in setup_bermuda_entry.options or setup_bermuda_entry.options[CONF_REF_POWER] != -60.0
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate_save"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert setup_bermuda_entry.options[CONF_REF_POWER] == -60.0
    assert setup_bermuda_entry.options[CONF_ATTENUATION] == 3.0
    offsets = _offsets(setup_bermuda_entry)
    assert offsets[kitchen.address] == 2.0  # -60 - (-62)
    assert offsets[lounge.address] == 0.0


async def test_each_failure_asks_to_try_again_with_a_clear_reason(hass, setup_bermuda_entry, readings):
    result, lounge, _kitchen = await _wizard(hass, setup_bermuda_entry)
    result = await _pick(hass, result, "calibrate_proxy", proxy=lounge.address)

    readings.append([-60.0] * 3)  # barely heard
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["step_id"] == "calibrate_ready"
    assert result["errors"] == {"base": "no_signal"}

    readings.append([-40.0, -80.0] * 6)  # all over the place
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["errors"] == {"base": "unstable"}

    readings.append([-60.0] * 10)  # fine this time
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["step_id"] == "calibrate_hub"


async def test_inconsistent_distance_readings_restart_the_pair(hass, setup_bermuda_entry, readings):
    result, lounge, _kitchen = await _wizard(hass, setup_bermuda_entry)
    result = await _pick(hass, result, "calibrate_distance", proxy=lounge.address)
    readings.append([-60.0] * 10)
    result = await _finish_progress(hass, await _submit(hass, result))
    readings.append([-58.0] * 10)  # "3 m" reads stronger than "1 m": the user got the distances wrong
    result = await _finish_progress(hass, await _submit(hass, result))
    assert result["errors"] == {"base": "inconsistent"}
    assert "Step 1 of 2" in result["description_placeholders"]["instruction"]  # back to the start of the pair


async def test_discarding_changes_nothing(hass, setup_bermuda_entry, readings):
    result, lounge, _kitchen = await _wizard(hass, setup_bermuda_entry)
    before = (dict(setup_bermuda_entry.options), _offsets(setup_bermuda_entry))
    result = await _pick(hass, result, "calibrate_proxy", proxy=lounge.address)
    readings.append([-70.0] * 10)
    result = await _finish_progress(hass, await _submit(hass, result))
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate_discard"}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    assert (dict(setup_bermuda_entry.options), _offsets(setup_bermuda_entry)) == before


async def test_saving_with_nothing_measured_just_closes(hass, setup_bermuda_entry, readings):
    result, _lounge, _kitchen = await _wizard(hass, setup_bermuda_entry)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate_save"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_calibrate_with_no_devices_nearby_explains_itself(hass, setup_bermuda_entry):
    result = await hass.config_entries.options.async_init(setup_bermuda_entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "calibrate"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_devices"


async def test_measure_collects_only_fresh_readings(hass, setup_bermuda_entry, monkeypatch):
    monkeypatch.setattr(options_calibration, "MEASURE_SECONDS", 0.3)
    monkeypatch.setattr(options_calibration, "SAMPLE_INTERVAL", 0.02)
    coordinator = setup_bermuda_entry.runtime_data.coordinator
    scanner = _scanner(coordinator, "11:11:11:11:11:11", "lounge", "Lounge")
    phone = _hear(coordinator, "AA:BB:CC:DD:EE:61", rssi=-60, scanner=scanner)
    advert = phone.get_scanner(scanner.address)

    from custom_components.bermuda.options_flow import BermudaOptionsFlowHandler  # noqa: PLC0415

    flow = BermudaOptionsFlowHandler()
    flow.hass = hass
    flow.handler = setup_bermuda_entry.entry_id
    flow._cal = {"device": phone.address, "target": scanner.address}  # noqa: SLF001

    base = advert.stamp

    async def feed():
        for i in range(1, 8):
            advert.rssi, advert.stamp = -60.0 - i, base + i
            if i == 4:
                await asyncio.sleep(0.05)  # the same stamp seen twice must not be counted twice
            await asyncio.sleep(0.03)

    feeder = asyncio.create_task(feed())
    samples = await flow._measure()  # noqa: SLF001
    await feeder
    assert samples == sorted(samples, reverse=True)  # in order, each reading once
    assert len(samples) == len(set(samples)) >= 3
