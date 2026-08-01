"""Immutable project-saved workspace presentation choices."""

from __future__ import annotations

from dataclasses import dataclass

from kindred.core.project_document import MechanismTabDocument, ProjectDocument

__all__ = [
    "ProjectPresentationState",
    "WorkspacePresentationState",
    "default_project_presentation_state",
    "default_workspace_presentation_state",
    "validate_project_presentation_state",
    "validate_workspace_presentation_state",
]


@dataclass(frozen=True, slots=True)
class ProjectPresentationState:
    current_set_id: str | None = None
    plot_set_ids: tuple[str, ...] = ()
    pinned_slider_set_ids: tuple[str, ...] = ()
    overlay_columns_by_dataset: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    selected_x_series: str | None = None
    selected_y_series: tuple[str, ...] = ()
    selected_y_uses_default_all: bool = True
    parametric_mode: bool = False
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
        for dataset_id, columns in self.overlay_columns_by_dataset:
            dataset = datasets.get(dataset_id)
            if dataset is None:
                continue
            available = {
                series.column_name
                for series in dataset.observations
            }
            retained = tuple(
                column
                for column in columns
                if column in available
            )
            if retained:
                overlays.append((dataset_id, retained))
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
            overlay_columns_by_dataset=tuple(overlays),
            selected_x_series=self.selected_x_series,
            selected_y_series=self.selected_y_series,
            selected_y_uses_default_all=self.selected_y_uses_default_all,
            parametric_mode=self.parametric_mode,
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


def _validate_optional_series(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str or None.")
    if not value.strip():
        raise ValueError(f"{label} must not be blank.")


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
        "parametric_mode",
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
    _validate_optional_series(
        state.selected_x_series,
        "selected_x_series",
    )
    if len(state.selected_y_series) != len(set(state.selected_y_series)):
        raise ValueError("Selected Y series must be unique.")
    for series_name in state.selected_y_series:
        if not isinstance(series_name, str):
            raise TypeError("Selected Y series names must be strings.")
        if not series_name.strip():
            raise ValueError("Selected Y series names must not be blank.")
    if (
        state.selected_y_uses_default_all
        and state.selected_y_series
    ):
        raise ValueError(
            "Default-all Y selection must not also contain an explicit "
            "selected_y_series subset."
        )

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
        for dataset_id, _ in state.overlay_columns_by_dataset
    ]
    if len(overlay_dataset_ids) != len(set(overlay_dataset_ids)):
        raise ValueError("Overlay dataset IDs must be unique.")
    for dataset_id, columns in state.overlay_columns_by_dataset:
        dataset = dataset_by_id.get(dataset_id)
        if dataset is None:
            raise ValueError(
                f"Presentation state references missing dataset "
                f"{dataset_id!r}."
            )
        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Overlay columns for dataset {dataset_id!r} "
                "must be unique."
            )
        available = {
            series.column_name
            for series in dataset.observations
        }
        for column in columns:
            if column not in available:
                raise ValueError(
                    f"Overlay references missing column {column!r} "
                    f"in dataset {dataset_id!r}."
                )
    return state


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
