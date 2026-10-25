"""Small helpers for driving Bermuda's options-flow menus in tests."""

from __future__ import annotations

from typing import Any


async def goto_setting(hass: Any, result: dict[str, Any], step: str) -> dict[str, Any]:
    """From the main options menu, open a screen that lives under Settings."""
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input={"next_step_id": "settings"}
    )
    return await hass.config_entries.options.async_configure(result["flow_id"], user_input={"next_step_id": step})
