"""Typed Fit-local setup, transition, snapshot, and identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Mapping

FitResolvedXMode = Literal["monotone", "time_guided"]

__all__ = [
    "FitParametricSettings",
    "FitRunIdentity",
    "FitTargetSelection",
    "copy_target_selection",
]


def _name(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank.")
    return text


def _finite(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


@dataclass(frozen=True, slots=True)
class FitParametricSettings:
    dataset_id: str
    x_name: str
    requested_mode: Literal["auto", "monotone", "time_guided"]
    resolved_mode: FitResolvedXMode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _name(self.dataset_id, "dataset ID"))
        object.__setattr__(self, "x_name", _name(self.x_name, "X series"))
        if self.x_name == "t":
            raise ValueError("Parametric-X settings cannot target time.")
        if self.requested_mode not in {"auto", "monotone", "time_guided"}:
            raise ValueError("Unknown parametric-X mode.")
        if self.requested_mode == "auto":
            if self.resolved_mode not in {None, "monotone", "time_guided"}:
                raise ValueError("Unknown resolved parametric-X mode.")
        elif self.requested_mode != self.resolved_mode:
            raise ValueError(
                "An explicit parametric-X mode cannot resolve to another mode."
            )


@dataclass(frozen=True, slots=True)
class FitTargetSelection:
    dataset_id: str
    eligible_targets: tuple[str, ...]
    selected_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        dataset_id = _name(self.dataset_id, "dataset ID")
        eligible = tuple(_name(item, "eligible target") for item in self.eligible_targets)
        selected = tuple(_name(item, "selected target") for item in self.selected_targets)
        if (
            len(eligible) != len(set(eligible))
            or len(selected) != len(set(selected))
            or not set(selected).issubset(eligible)
        ):
            raise ValueError("Fit target selection is not a subset of eligible targets.")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "eligible_targets", eligible)
        object.__setattr__(self, "selected_targets", selected)

    def all(self) -> "FitTargetSelection":
        return replace(self, selected_targets=self.eligible_targets)

    def none(self) -> "FitTargetSelection":
        return replace(self, selected_targets=())

    def invert(self) -> "FitTargetSelection":
        selected = set(self.selected_targets)
        return replace(
            self,
            selected_targets=tuple(
                item for item in self.eligible_targets if item not in selected
            ),
        )


def copy_target_selection(
    source: FitTargetSelection,
    destinations: tuple[FitTargetSelection, ...],
) -> tuple[tuple[FitTargetSelection, ...], int, int]:
    """Replace each wholly compatible destination; never partially copy."""

    destination_ids = [item.dataset_id for item in destinations]
    if (
        source.dataset_id in destination_ids
        or len(destination_ids) != len(set(destination_ids))
    ):
        raise ValueError(
            "Copy Targets requires distinct destination datasets other than "
            "the source."
        )
    source_targets = set(source.selected_targets)
    if not source_targets:
        return destinations, 0, 0
    copied: list[FitTargetSelection] = []
    matched = 0
    unmatched = 0
    for destination in destinations:
        if source_targets.issubset(destination.eligible_targets):
            copied.append(
                replace(
                    destination,
                    selected_targets=tuple(
                        item
                        for item in destination.eligible_targets
                        if item in source_targets
                    ),
                )
            )
            matched += 1
        else:
            copied.append(destination)
            unmatched += 1
    return tuple(copied), matched, unmatched


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fit run identity contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(
        f"Fit run identity contains unsupported value {type(value).__name__}."
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                str(key): _freeze_json(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FitRunIdentity:
    """Deeply immutable compact scientific run identity.

    The identity retains only scalar metadata and digest descriptors.  The
    canonical bytes used to derive ``short_hash`` are intentionally transient;
    verbose export is a presentation concern and is assembled on demand.
    """

    payload: Mapping[str, object]
    short_hash: str

    @classmethod
    def build(cls, payload: Mapping[str, object]) -> "FitRunIdentity":
        normalized = _json_value(payload)
        if not isinstance(normalized, dict):
            raise TypeError("Fit run identity must be a JSON object.")
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return cls(
            payload=_freeze_json(normalized),
            short_hash=hashlib.sha256(encoded).hexdigest()[:12],
        )

    def readable_json(self) -> str:
        return json.dumps(
            _plain_json(self.payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
