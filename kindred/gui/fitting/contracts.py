"""Small typed contracts used by the Fit presentation surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace


__all__ = [
    "FitParametricSettings",
    "FitTargetSelection",
    "copy_target_selection",
]


def _name(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank.")
    return text


@dataclass(frozen=True, slots=True)
class FitParametricSettings:
    dataset_id: str
    x_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _name(self.dataset_id, "dataset ID"))
        object.__setattr__(self, "x_name", _name(self.x_name, "X series"))
        if self.x_name == "t":
            raise ValueError("Parametric-X settings cannot target time.")


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
