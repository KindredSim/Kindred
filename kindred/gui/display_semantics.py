from __future__ import annotations

from dataclasses import dataclass
import re

__all__ = [
    "AxisDisplaySemantic",
    "canonical_concentration_axis",
    "canonical_dataset_x_axis",
    "canonical_time_axis",
    "concentration_series_header",
    "dataset_axis_semantics",
    "format_axis_header",
    "normalize_axis_semantic",
]

_PAREN_UNITS_RE = re.compile(r"^(?P<label>.*?)\s*\((?P<units>[^()]*)\)\s*$")


@dataclass(frozen=True, slots=True)
class AxisDisplaySemantic:
    label: str
    units: str | None = None

    @property
    def header(self) -> str:
        return format_axis_header(self.label, self.units)


def format_axis_header(label: str, units: object = None) -> str:
    label_s = str(label or "").strip()
    units_s = str(units or "").strip()
    if label_s and units_s:
        return f"{label_s} ({units_s})"
    return label_s


def normalize_axis_semantic(
    label: object,
    *,
    units: object = None,
    default_label: str,
    default_units: object = None,
) -> AxisDisplaySemantic:
    label_s = str(label or "").strip()
    units_s = str(units or "").strip()
    if not label_s:
        label_s = str(default_label or "").strip()
    match = _PAREN_UNITS_RE.match(label_s)
    if match is not None:
        label_s = str(match.group("label") or "").strip()
        if not units_s:
            units_s = str(match.group("units") or "").strip()
    if not units_s and default_units is not None:
        units_s = str(default_units or "").strip()
    return AxisDisplaySemantic(label=label_s, units=units_s or None)


def canonical_time_axis(label: object = "Time", *, units: object = "s") -> AxisDisplaySemantic:
    return normalize_axis_semantic(
        label,
        units=units,
        default_label="Time",
        default_units="s",
    )


def canonical_dataset_x_axis(label: object = "Time", *, units: object = None) -> AxisDisplaySemantic:
    label_s = str(label or "").strip()
    if not label_s or label_s.lower() == "time":
        return canonical_time_axis(label_s or "Time", units=units if units is not None else "s")
    return normalize_axis_semantic(
        label_s,
        units=units,
        default_label=label_s,
        default_units=None,
    )


def canonical_concentration_axis(
    label: object = "Concentration",
    *,
    units: object = "M",
) -> AxisDisplaySemantic:
    return normalize_axis_semantic(
        label,
        units=units,
        default_label="Concentration",
        default_units="M",
    )


def concentration_series_header(series_name: object, *, units: object = "M") -> str:
    name = str(series_name or "").strip()
    if not name:
        return canonical_concentration_axis(units=units).header
    return canonical_concentration_axis(f"[{name}]", units=units).header


def dataset_axis_semantics(
    *,
    x_label: object = "Time",
    x_units: object = None,
    y_label: object = "Concentration",
    y_units: object = "M",
) -> tuple[AxisDisplaySemantic, AxisDisplaySemantic]:
    return (
        canonical_dataset_x_axis(x_label, units=x_units),
        canonical_concentration_axis(y_label, units=y_units),
    )
