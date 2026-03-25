from __future__ import annotations

from typing import Literal, cast

XMappingMode = Literal["auto", "monotone", "time_guided"]

ALLOWED_X_MAPPING_MODES: tuple[XMappingMode, ...] = ("auto", "monotone", "time_guided")


def normalize_x_mapping_mode(value: object) -> str:
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not mode or mode == "auto":
        return "auto"
    if mode in {"monotone", "monotone_only", "monotoneonly"}:
        return "monotone"
    if mode in {"time_guided", "timeguided"}:
        return "time_guided"
    return mode


def parse_x_mapping_mode(value: object) -> XMappingMode:
    mode = normalize_x_mapping_mode(value)
    if mode not in ALLOWED_X_MAPPING_MODES:
        raise ValueError(
            f"Invalid x_mapping_mode '{value}'. Expected one of: {', '.join(ALLOWED_X_MAPPING_MODES)}."
        )
    return cast(XMappingMode, mode)

