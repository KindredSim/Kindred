"""Small typed contracts used by the Fit presentation surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace


__all__ = [
    "FitParametricSettings",
    "FitTargetSelection",
    "copy_target_selection",
]


@dataclass(frozen=True, slots=True)
class FitParametricSettings:
    dataset_id: str
    x_name: str


@dataclass(frozen=True, slots=True)
class FitTargetSelection:
    dataset_id: str
    eligible_targets: tuple[str, ...]
    selected_targets: tuple[str, ...]


def copy_target_selection(
    source: FitTargetSelection,
    destinations: tuple[FitTargetSelection, ...],
) -> tuple[tuple[FitTargetSelection, ...], int, int]:
    """Replace each wholly compatible destination; never partially copy."""

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
