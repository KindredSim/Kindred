"""Small immutable snapshots used by dataset widgets and import staging."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kindred.core.project_document import (
    MechanismTabDocument,
    ProjectDocument,
    RawDatasetRecord,
)
from kindred.core.project_presentation_state import ProjectPresentationState


@dataclass(frozen=True)
class ActiveDatasetLinkSnapshot:
    dataset_id: str
    set_id: str | None
    set_name: str | None
    observation_mapping: tuple[tuple[str, str], ...]
    include_in_fit: bool
    overlay_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataManagerSnapshot:
    datasets: tuple[RawDatasetRecord, ...]
    active_tab_id: str | None
    active_links: tuple[ActiveDatasetLinkSnapshot, ...]
    available_sets: tuple[tuple[str, str], ...]

    def link_by_dataset_id(self) -> dict[str, ActiveDatasetLinkSnapshot]:
        return {link.dataset_id: link for link in self.active_links}


@dataclass(frozen=True, slots=True)
class ExperimentalLayerInput:
    """One immutable selected raw series with separate mapped-model identity."""

    layer_id: str
    dataset_id: str
    raw_observation_id: str
    dataset_name: str
    raw_column: str
    model_species: str | None
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.layer_id, "layer"),
            (self.dataset_id, "dataset"),
            (self.raw_observation_id, "raw observation"),
            (self.dataset_name, "dataset name"),
            (self.raw_column, "raw column"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"Experimental {label} identity must be a nonblank "
                    "edge-normalized string."
                )
        if self.model_species is not None and (
            not isinstance(self.model_species, str)
            or not self.model_species
            or self.model_species != self.model_species.strip()
        ):
            raise ValueError(
                "Experimental mapped-model identity must be None or a "
                "nonblank edge-normalized string."
            )
        if (
            type(self.t) is not tuple
            or type(self.y) is not tuple
            or type(self.source_rows) is not tuple
            or not self.t
            or len(self.t) != len(self.y)
            or len(self.t) != len(self.source_rows)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (*self.t, *self.y)
            )
            or any(
                type(row) is not int or row <= 0
                for row in self.source_rows
            )
            or len(set(self.source_rows)) != len(self.source_rows)
        ):
            raise ValueError(
                "Experimental layer values require immutable aligned finite "
                "float t/y tuples and unique positive physical source rows."
            )

    @property
    def presented_species(self) -> str:
        return self.model_species or self.raw_column


def build_data_manager_snapshot(
    *,
    document: ProjectDocument,
    active_tab: MechanismTabDocument | None,
    presentation: ProjectPresentationState | None,
) -> DataManagerSnapshot:
    """Compose the Data dock's complete immutable render input."""
    if active_tab is None:
        return DataManagerSnapshot(
            datasets=document.datasets,
            active_tab_id=None,
            active_links=(),
            available_sets=(),
        )
    canonical_links = dict(active_tab.dataset_links)
    overlays = dict(
        () if presentation is None else presentation.overlay_columns_by_dataset
    )
    sets_by_id = {
        item.set_id: item.name for item in active_tab.initial_condition_sets
    }
    links = tuple(
        ActiveDatasetLinkSnapshot(
            dataset_id=record.dataset_id,
            set_id=(
                canonical_links[record.dataset_id].set_id
                if record.dataset_id in canonical_links
                else None
            ),
            set_name=sets_by_id.get(
                canonical_links[record.dataset_id].set_id
                if record.dataset_id in canonical_links
                else ""
            ),
            observation_mapping=(
                canonical_links[record.dataset_id].observation_mapping
                if record.dataset_id in canonical_links
                else ()
            ),
            include_in_fit=(
                canonical_links[record.dataset_id].include_in_fit
                if record.dataset_id in canonical_links
                else False
            ),
            overlay_columns=tuple(overlays.get(record.dataset_id, ())),
        )
        for record in document.datasets
    )
    return DataManagerSnapshot(
        datasets=document.datasets,
        active_tab_id=active_tab.tab_id,
        active_links=links,
        available_sets=tuple(
            (item.set_id, item.name)
            for item in active_tab.initial_condition_sets
        ),
    )


def build_experimental_layer_inputs(
    *,
    document: ProjectDocument,
    active_tab: MechanismTabDocument,
    presentation: ProjectPresentationState,
) -> tuple[ExperimentalLayerInput, ...]:
    """Build the presenter's selected experimental inputs without plotting."""
    overlays = dict(presentation.overlay_columns_by_dataset)
    links = dict(active_tab.dataset_links)
    layers: list[ExperimentalLayerInput] = []
    for record in document.datasets:
        selected = set(overlays.get(record.dataset_id, ()))
        if not selected:
            continue
        link = links.get(record.dataset_id)
        mapping = dict(link.observation_mapping) if link is not None else {}
        for series in record.observations:
            raw_column = str(series.column_name)
            if raw_column not in selected:
                continue
            layers.append(
                ExperimentalLayerInput(
                    layer_id=(
                        f"{record.dataset_id}:"
                        f"{series.raw_observation_id}"
                    ),
                    dataset_id=record.dataset_id,
                    raw_observation_id=series.raw_observation_id,
                    dataset_name=record.display_name,
                    raw_column=raw_column,
                    model_species=mapping.get(raw_column),
                    t=series.t,
                    y=series.y,
                    source_rows=series.source_rows,
                )
            )
    return tuple(layers)
