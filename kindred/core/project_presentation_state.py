"""Immutable project-saved workspace presentation choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from kindred.core.project_document import (
    MechanismTabDocument,
    ProjectDocument,
    validate_raw_observation_id,
)

__all__ = [
    "AxisSeriesSelection",
    "ExperimentalAxisSelection",
    "ProjectPresentationState",
    "SimulationSeriesAxisSelection",
    "WorkspacePresentationState",
    "default_project_presentation_state",
    "default_workspace_presentation_state",
    "reconcile_workspace_presentation_state",
    "validate_project_presentation_state",
    "validate_workspace_presentation_state",
]


@dataclass(frozen=True, slots=True)
class SimulationSeriesAxisSelection:
    """One simulation-owned named series selected for an axis."""

    series_name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.series_name, str)
            or not self.series_name
            or self.series_name != self.series_name.strip()
        ):
            raise ValueError(
                "Simulation axis series must be a nonblank edge-normalized "
                "name."
            )


@dataclass(frozen=True, slots=True)
class ExperimentalAxisSelection:
    """One dataset-local raw observation selected for an axis."""

    dataset_id: str
    raw_observation_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dataset_id, str)
            or not self.dataset_id
            or self.dataset_id != self.dataset_id.strip()
        ):
            raise ValueError(
                "Experimental axis dataset ID must be a nonblank "
                "edge-normalized string."
            )
        validate_raw_observation_id(
            self.raw_observation_id,
            "experimental axis raw observation ID",
        )


AxisSeriesSelection: TypeAlias = (
    SimulationSeriesAxisSelection | ExperimentalAxisSelection
)


@dataclass(frozen=True, slots=True)
class ProjectPresentationState:
    current_set_id: str | None = None
    plot_set_ids: tuple[str, ...] = ()
    pinned_slider_set_ids: tuple[str, ...] = ()
    overlay_raw_observation_ids_by_dataset: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    selected_x_series: AxisSeriesSelection = SimulationSeriesAxisSelection("t")
    selected_y_series: tuple[AxisSeriesSelection, ...] = ()
    selected_y_uses_default_all: bool = True
    reference_lines_visible: bool = True
    intervention_annotations_visible: bool = True
    log_x: bool = False
    log_y: bool = False
    legend_visible: bool = True
    x_inverted: bool = False
    y_inverted: bool = False

    def reconciled(
        self,
        document: ProjectDocument,
        tab: MechanismTabDocument,
    ) -> "ProjectPresentationState":
        set_ids = {item.set_id for item in tab.initial_condition_sets}
        datasets = {item.dataset_id: item for item in document.datasets}
        overlays: list[tuple[str, tuple[str, ...]]] = []
        for (
            dataset_id,
            raw_observation_ids,
        ) in self.overlay_raw_observation_ids_by_dataset:
            dataset = datasets.get(dataset_id)
            if dataset is None:
                continue
            available = {
                series.raw_observation_id
                for series in dataset.observations
            }
            retained = tuple(
                raw_observation_id
                for raw_observation_id in raw_observation_ids
                if raw_observation_id in available
            )
            if retained:
                overlays.append((dataset_id, retained))
        simulation_series = _simulation_axis_series(tab)
        experimental_series = frozenset(
            ExperimentalAxisSelection(dataset_id, raw_observation_id)
            for dataset_id, raw_observation_ids in overlays
            for raw_observation_id in raw_observation_ids
        )
        selected_x = self.selected_x_series
        if not isinstance(
            selected_x,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError(
                "selected_x_series must be a typed simulation or "
                "experimental axis selection."
            )
        if not isinstance(self.selected_y_series, tuple) or any(
            not isinstance(
                selection,
                (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
            )
            for selection in self.selected_y_series
        ):
            raise TypeError(
                "selected_y_series must be a tuple of typed simulation or "
                "experimental axis selections."
            )
        if not _axis_selection_is_current(
            selected_x,
            simulation_series=simulation_series,
            experimental_series=experimental_series,
        ):
            selected_x = SimulationSeriesAxisSelection("t")
        selected_y = tuple(
            selection
            for selection in self.selected_y_series
            if _axis_selection_is_current(
                selection,
                simulation_series=simulation_series,
                experimental_series=experimental_series,
            )
        )
        current_set_id = (
            self.current_set_id
            if self.current_set_id in set_ids
            else tab.initial_condition_sets[0].set_id
        )
        return ProjectPresentationState(
            current_set_id=current_set_id,
            plot_set_ids=tuple(
                item
                for item in self.plot_set_ids
                if item in set_ids
            ),
            pinned_slider_set_ids=tuple(
                item
                for item in self.pinned_slider_set_ids
                if item in set_ids
            ),
            overlay_raw_observation_ids_by_dataset=tuple(overlays),
            selected_x_series=selected_x,
            selected_y_series=selected_y,
            selected_y_uses_default_all=self.selected_y_uses_default_all,
            reference_lines_visible=self.reference_lines_visible,
            intervention_annotations_visible=(
                self.intervention_annotations_visible
            ),
            log_x=self.log_x,
            log_y=self.log_y,
            legend_visible=self.legend_visible,
            x_inverted=self.x_inverted,
            y_inverted=self.y_inverted,
        )

    def is_default_for(self, tab: MechanismTabDocument) -> bool:
        if len(tab.initial_condition_sets) != 1:
            return False
        return self == default_project_presentation_state(tab)


@dataclass(frozen=True, slots=True)
class WorkspacePresentationState:
    active_tab_id: str
    presentation_by_tab_id: tuple[
        tuple[str, ProjectPresentationState], ...
    ]

    def presentation(self, tab_id: str) -> ProjectPresentationState:
        target = str(tab_id)
        for current_id, state in self.presentation_by_tab_id:
            if current_id == target:
                return state
        raise KeyError(f"Unknown presentation tab ID: {target!r}.")


def default_project_presentation_state(
    tab: MechanismTabDocument,
) -> ProjectPresentationState:
    default_set_id = tab.initial_condition_sets[0].set_id
    return ProjectPresentationState(
        current_set_id=default_set_id,
        plot_set_ids=(default_set_id,),
        selected_x_series=SimulationSeriesAxisSelection("t"),
        reference_lines_visible=True,
        intervention_annotations_visible=True,
    )


def default_workspace_presentation_state(
    document: ProjectDocument,
    *,
    active_tab_id: str | None = None,
) -> WorkspacePresentationState:
    active = (
        document.tabs[0].tab_id
        if active_tab_id is None
        else str(active_tab_id)
    )
    return validate_workspace_presentation_state(
        document,
        WorkspacePresentationState(
            active_tab_id=active,
            presentation_by_tab_id=tuple(
                (
                    tab.tab_id,
                    default_project_presentation_state(tab),
                )
                for tab in document.tabs
            ),
        ),
    )


def _simulation_axis_series(
    tab: MechanismTabDocument,
) -> frozenset[SimulationSeriesAxisSelection]:
    analysis = tab.dsl_analysis
    names = (
        ()
        if analysis is None
        else (*analysis.species_names, *analysis.observable_names)
    )
    return frozenset(
        SimulationSeriesAxisSelection(name)
        for name in dict.fromkeys(("t", *names))
    )


def _axis_selection_is_current(
    value: object,
    *,
    simulation_series: frozenset[SimulationSeriesAxisSelection],
    experimental_series: frozenset[ExperimentalAxisSelection],
) -> bool:
    if isinstance(value, SimulationSeriesAxisSelection):
        return value in simulation_series
    return bool(
        isinstance(value, ExperimentalAxisSelection)
        and value in experimental_series
    )


def _validate_axis_selection(
    value: object,
    label: str,
    *,
    simulation_series: frozenset[SimulationSeriesAxisSelection],
    experimental_series: frozenset[ExperimentalAxisSelection],
) -> None:
    if not isinstance(
        value,
        (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
    ):
        raise TypeError(
            f"{label} must be a typed simulation or experimental axis "
            "selection."
        )
    if _axis_selection_is_current(
        value,
        simulation_series=simulation_series,
        experimental_series=experimental_series,
    ):
        return
    if isinstance(value, ExperimentalAxisSelection):
        raise ValueError(
            f"{label} references raw observation "
            f"{value.raw_observation_id!r} in dataset {value.dataset_id!r}, "
            "but that observation is not selected as an overlay."
        )
    raise ValueError(
        f"{label} references simulation series {value.series_name!r}, "
        "which is not present in the current typed mechanism series."
    )


def validate_project_presentation_state(
    document: ProjectDocument,
    tab: MechanismTabDocument,
    state: ProjectPresentationState,
) -> ProjectPresentationState:
    if not isinstance(state, ProjectPresentationState):
        raise TypeError("state must be ProjectPresentationState.")
    if (
        state.current_set_id is not None
        and not isinstance(state.current_set_id, str)
    ):
        raise TypeError("current_set_id must be a str or None.")
    for field_name in (
        "selected_y_uses_default_all",
        "reference_lines_visible",
        "intervention_annotations_visible",
        "log_x",
        "log_y",
        "legend_visible",
        "x_inverted",
        "y_inverted",
    ):
        if type(getattr(state, field_name)) is not bool:
            raise TypeError(f"{field_name} must be a bool.")
    set_ids = {item.set_id for item in tab.initial_condition_sets}
    if (
        state.current_set_id is not None
        and state.current_set_id not in set_ids
    ):
        raise ValueError(
            f"Presentation state for tab {tab.tab_id!r} references "
            f"missing current set {state.current_set_id!r}."
        )
    if len(state.plot_set_ids) != len(set(state.plot_set_ids)):
        raise ValueError("Plot set IDs must be unique.")
    if len(state.pinned_slider_set_ids) != len(
        set(state.pinned_slider_set_ids)
    ):
        raise ValueError("Pinned Slider set IDs must be unique.")
    for set_id in (*state.plot_set_ids, *state.pinned_slider_set_ids):
        if set_id not in set_ids:
            raise ValueError(
                f"Presentation state for tab {tab.tab_id!r} references "
                f"missing set {set_id!r}."
            )

    dataset_by_id = {
        item.dataset_id: item
        for item in document.datasets
    }
    overlay_dataset_ids = [
        dataset_id
        for dataset_id, _
        in state.overlay_raw_observation_ids_by_dataset
    ]
    if len(overlay_dataset_ids) != len(set(overlay_dataset_ids)):
        raise ValueError("Overlay dataset IDs must be unique.")
    for (
        dataset_id,
        raw_observation_ids,
    ) in state.overlay_raw_observation_ids_by_dataset:
        dataset = dataset_by_id.get(dataset_id)
        if dataset is None:
            raise ValueError(
                f"Presentation state references missing dataset "
                f"{dataset_id!r}."
            )
        if len(raw_observation_ids) != len(set(raw_observation_ids)):
            raise ValueError(
                f"Overlay raw observation IDs for dataset {dataset_id!r} "
                "must be unique."
            )
        available = {
            series.raw_observation_id
            for series in dataset.observations
        }
        for raw_observation_id in raw_observation_ids:
            validate_raw_observation_id(
                raw_observation_id,
                "overlay raw observation ID",
            )
            if raw_observation_id not in available:
                raise ValueError(
                    "Overlay references missing raw observation ID "
                    f"{raw_observation_id!r} in dataset {dataset_id!r}."
                )

    simulation_series = _simulation_axis_series(tab)
    experimental_series = frozenset(
        ExperimentalAxisSelection(dataset_id, raw_observation_id)
        for dataset_id, raw_observation_ids
        in state.overlay_raw_observation_ids_by_dataset
        for raw_observation_id in raw_observation_ids
    )
    _validate_axis_selection(
        state.selected_x_series,
        "selected_x_series",
        simulation_series=simulation_series,
        experimental_series=experimental_series,
    )
    if not isinstance(state.selected_y_series, tuple):
        raise TypeError("selected_y_series must be a tuple.")
    if len(state.selected_y_series) != len(set(state.selected_y_series)):
        raise ValueError("Selected Y series must be unique.")
    for index, selection in enumerate(state.selected_y_series):
        _validate_axis_selection(
            selection,
            f"selected_y_series[{index}]",
            simulation_series=simulation_series,
            experimental_series=experimental_series,
        )
    if state.selected_y_uses_default_all and state.selected_y_series:
        raise ValueError(
            "Default-all Y selection must not also contain an explicit "
            "selected_y_series subset."
        )
    return state


def reconcile_workspace_presentation_state(
    document: ProjectDocument,
    state: WorkspacePresentationState,
) -> WorkspacePresentationState:
    """Resolve every saved axis against one complete current project value."""

    if not isinstance(state, WorkspacePresentationState):
        raise TypeError("state must be WorkspacePresentationState.")
    document.tab(state.active_tab_id)
    entries = state.presentation_by_tab_id
    ids = [tab_id for tab_id, _ in entries]
    document_ids = [tab.tab_id for tab in document.tabs]
    if ids != document_ids:
        raise ValueError(
            "Workspace presentation must contain every tab exactly once "
            "in document order."
        )
    if any(
        not isinstance(tab_state, ProjectPresentationState)
        for _tab_id, tab_state in entries
    ):
        raise TypeError(
            "Workspace presentation entries must contain "
            "ProjectPresentationState values."
        )
    return validate_workspace_presentation_state(
        document,
        WorkspacePresentationState(
            active_tab_id=state.active_tab_id,
            presentation_by_tab_id=tuple(
                (
                    tab_id,
                    tab_state.reconciled(document, document.tab(tab_id)),
                )
                for tab_id, tab_state in entries
            ),
        ),
    )


def validate_workspace_presentation_state(
    document: ProjectDocument,
    state: WorkspacePresentationState,
) -> WorkspacePresentationState:
    if not isinstance(state, WorkspacePresentationState):
        raise TypeError(
            "state must be WorkspacePresentationState."
        )
    document.tab(state.active_tab_id)
    entries = state.presentation_by_tab_id
    ids = [tab_id for tab_id, _ in entries]
    document_ids = [tab.tab_id for tab in document.tabs]
    if ids != document_ids:
        raise ValueError(
            "Workspace presentation must contain every tab exactly once "
            "in document order."
        )
    for tab_id, tab_state in entries:
        validate_project_presentation_state(
            document,
            document.tab(tab_id),
            tab_state,
        )
    return state
