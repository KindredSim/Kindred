"""Active mechanism-tab lifecycle and noncanonical per-tab state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Iterable, Protocol, runtime_checkable

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
    validate_project_document,
)
from kindred.core.project_session import ProjectSession
from kindred.core.project_presentation_state import (
    ProjectPresentationState,
    WorkspacePresentationState,
    default_project_presentation_state,
    validate_project_presentation_state,
    validate_workspace_presentation_state,
)

if TYPE_CHECKING:
    from kindred.core.result_repository import ResultRepository

__all__ = [
    "DocumentDeletionResultFence",
    "PreparedWorkspaceInstall",
    "ScientificDocumentDelta",
    "TabRuntimeState",
    "WorkspaceTabRegistry",
]


@runtime_checkable
class DocumentDeletionResultFence(Protocol):
    """Direct result-owner seam for canonical document deletions."""

    def purge_sets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]: ...

    def purge_tab(
        self,
        tab_id: str,
        set_ids: Iterable[str],
    ) -> tuple[tuple[str, int], ...]: ...


@dataclass(slots=True)
class TabRuntimeState:
    """Noncanonical authoring and transient view state for exactly one tab."""

    reactions_draft: MechanismAuthoringSource | None = None
    notes_draft: str | None = None
    invalid_initial_text_by_set_id: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    selected_run_set_ids: tuple[str, ...] = ()
    slider_parameter_draft: dict[str, float] = field(default_factory=dict)
    slider_species_drafts_by_set_id: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    result_repository: "ResultRepository | None" = None

    @property
    def has_dirty_drafts(self) -> bool:
        return bool(
            self.reactions_draft is not None
            or self.notes_draft is not None
            or self.invalid_initial_text_by_set_id
            or self.slider_parameter_draft
            or self.slider_species_drafts_by_set_id
        )

@dataclass(frozen=True, slots=True)
class PreparedWorkspaceInstall:
    """Validated per-tab presentation and preconstructed transient runtimes."""

    active_tab_id: str
    presentation_by_tab_id: tuple[
        tuple[str, ProjectPresentationState], ...
    ]
    runtime_by_tab_id: tuple[tuple[str, TabRuntimeState], ...]


@dataclass(frozen=True, slots=True)
class ScientificDocumentDelta:
    """Affected existing targets from one accepted canonical document delta."""

    affected_targets_by_tab: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def affected_set_ids(self, tab_id: str) -> tuple[str, ...]:
        target = str(tab_id)
        for current_tab_id, set_ids in self.affected_targets_by_tab:
            if current_tab_id == target:
                return set_ids
        return ()


class WorkspaceTabRegistry:
    """Sole live active-tab ID plus presentation and transient tab state."""

    def __init__(
        self,
        session: ProjectSession,
        *,
        deletion_result_fence: DocumentDeletionResultFence,
        runtime_factory: Callable[[str], TabRuntimeState] | None = None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be a ProjectSession.")
        if not isinstance(deletion_result_fence, DocumentDeletionResultFence):
            raise TypeError(
                "deletion_result_fence must implement DocumentDeletionResultFence."
            )
        self._session = session
        self._deletion_result_fence = deletion_result_fence
        self._runtime_factory = runtime_factory or (lambda _tab_id: TabRuntimeState())
        self._installed = self.prepare_install(
            document=session.document,
            presentation=session.saved_baseline.presentation,
        )

    @property
    def active_tab_id(self) -> str:
        return self._installed.active_tab_id

    @property
    def active_document_identity(self) -> tuple[int, str]:
        """Session-only identity used to reset native editor history."""

        return (
            self._session.identity_generation,
            self._installed.active_tab_id,
        )

    @property
    def active_tab(self) -> MechanismTabDocument:
        return self._session.document.tab(self.active_tab_id)

    @property
    def active_runtime(self) -> TabRuntimeState:
        return self.tab_runtime(self.active_tab_id)

    @property
    def active_presentation(self) -> ProjectPresentationState:
        return self.presentation(self.active_tab_id)

    def presentation(self, tab_id: str) -> ProjectPresentationState:
        target = str(tab_id)
        self._session.document.tab(target)
        for current_id, state in self._installed.presentation_by_tab_id:
            if current_id == target:
                return state
        raise KeyError(f"Unknown presentation tab ID: {target!r}.")

    def tab_runtime(self, tab_id: str) -> TabRuntimeState:
        target = str(tab_id)
        self._session.document.tab(target)
        for current_id, runtime in self._installed.runtime_by_tab_id:
            if current_id == target:
                return runtime
        raise KeyError(f"Unknown runtime tab ID: {target!r}.")

    def presentation_items(self) -> tuple[tuple[str, ProjectPresentationState], ...]:
        return self._installed.presentation_by_tab_id

    def workspace_presentation(self) -> WorkspacePresentationState:
        return validate_workspace_presentation_state(
            self._session.document,
            WorkspacePresentationState(
                active_tab_id=self.active_tab_id,
                presentation_by_tab_id=self.presentation_items(),
            ),
        )

    def prepare_install(
        self,
        *,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
    ) -> PreparedWorkspaceInstall:
        """Validate and construct all replacement state without live mutation."""

        candidate = validate_project_document(document)
        validated_presentation = validate_workspace_presentation_state(
            candidate,
            presentation,
        )
        target_active = validated_presentation.active_tab_id
        supplied = dict(validated_presentation.presentation_by_tab_id)
        next_presentation: list[tuple[str, ProjectPresentationState]] = []
        next_runtime: list[tuple[str, TabRuntimeState]] = []
        for tab in candidate.tabs:
            state = supplied[tab.tab_id]
            next_presentation.append(
                (
                    tab.tab_id,
                    validate_project_presentation_state(
                        candidate,
                        tab,
                        state,
                    ),
                )
            )
            runtime = self._runtime_factory(tab.tab_id)
            if not isinstance(runtime, TabRuntimeState):
                raise TypeError("runtime_factory must return TabRuntimeState.")
            next_runtime.append((tab.tab_id, runtime))
        return PreparedWorkspaceInstall(
            active_tab_id=target_active,
            presentation_by_tab_id=tuple(next_presentation),
            runtime_by_tab_id=tuple(next_runtime),
        )

    def commit_install(self, prepared: PreparedWorkspaceInstall) -> None:
        """Commit one prepared registry replacement using assignments only."""

        if not isinstance(prepared, PreparedWorkspaceInstall):
            raise TypeError("prepared must be PreparedWorkspaceInstall.")
        self._installed = prepared

    def reconcile_document(
        self,
        *,
        previous_document: ProjectDocument,
    ) -> ScientificDocumentDelta:
        """Reconcile transient state and return the accepted scientific delta."""
        document = self._session.document
        scientific_delta = self._scientific_document_delta(
            previous_document,
            document,
        )
        tab_ids = {tab.tab_id for tab in document.tabs}
        next_presentation = {
            tab_id: state
            for tab_id, state in self._installed.presentation_by_tab_id
            if tab_id in tab_ids
        }
        next_runtime = {
            tab_id: state
            for tab_id, state in self._installed.runtime_by_tab_id
            if tab_id in tab_ids
        }
        previous_tabs = {
            tab.tab_id: tab for tab in previous_document.tabs
        }
        for tab in document.tabs:
            state = next_presentation.get(
                tab.tab_id,
                default_project_presentation_state(tab),
            )
            next_presentation[tab.tab_id] = state.reconciled(document, tab)
            runtime = next_runtime.get(tab.tab_id)
            if runtime is None:
                runtime = self._runtime_factory(tab.tab_id)
                if not isinstance(runtime, TabRuntimeState):
                    raise TypeError("runtime_factory must return TabRuntimeState.")
                next_runtime[tab.tab_id] = runtime
            set_ids = {item.set_id for item in tab.initial_condition_sets}
            runtime.invalid_initial_text_by_set_id = {
                set_id: values
                for set_id, values
                in runtime.invalid_initial_text_by_set_id.items()
                if set_id in set_ids
            }
            runtime.selected_run_set_ids = tuple(
                set_id
                for set_id in runtime.selected_run_set_ids
                if set_id in set_ids
            )
            previous_tab = previous_tabs.get(tab.tab_id)
            current_species = set(
                ()
                if tab.dsl_analysis is None
                else tab.dsl_analysis.species_names
            )
            previous_species = set(
                ()
                if previous_tab is None or previous_tab.dsl_analysis is None
                else previous_tab.dsl_analysis.species_names
            )
            runtime.slider_species_drafts_by_set_id = {
                set_id: {
                    species: value
                    for species, value in values.items()
                    if species in current_species
                    and species in previous_species
                }
                for set_id, values
                in runtime.slider_species_drafts_by_set_id.items()
                if set_id in set_ids
            }
            runtime.slider_species_drafts_by_set_id = {
                set_id: values
                for set_id, values
                in runtime.slider_species_drafts_by_set_id.items()
                if values
            }
            previous_parameters = (
                {}
                if previous_tab is None or previous_tab.dsl_analysis is None
                else {
                    item.name: item
                    for item in previous_tab.dsl_analysis.parameters
                }
            )
            current_parameters = (
                {}
                if tab.dsl_analysis is None
                else {
                    item.name: item for item in tab.dsl_analysis.parameters
                }
            )
            runtime.slider_parameter_draft = {
                name: value
                for name, value in runtime.slider_parameter_draft.items()
                if name in previous_parameters
                and current_parameters.get(name) == previous_parameters[name]
            }
        active_tab_id = self.active_tab_id
        if active_tab_id not in tab_ids:
            active_tab_id = document.tabs[0].tab_id
        self._installed = PreparedWorkspaceInstall(
            active_tab_id=active_tab_id,
            presentation_by_tab_id=tuple(
                (tab.tab_id, next_presentation[tab.tab_id])
                for tab in document.tabs
            ),
            runtime_by_tab_id=tuple(
                (tab.tab_id, next_runtime[tab.tab_id])
                for tab in document.tabs
            ),
        )
        return scientific_delta

    def replace_document(
        self,
        document: ProjectDocument,
    ) -> ScientificDocumentDelta | None:
        """Install one canonical document change and report affected targets."""
        previous = self._session.document
        changed = self._session.replace_document(
            document,
            fence_deletions=self._fence_document_deletions,
        )
        if changed:
            return self.reconcile_document(previous_document=previous)
        return None

    def _fence_document_deletions(
        self,
        previous: ProjectDocument,
        current: ProjectDocument,
    ) -> None:
        """Fence and purge deletion deltas before canonical document mutation."""

        current_tabs = {tab.tab_id: tab for tab in current.tabs}
        for old_tab in previous.tabs:
            current_tab = current_tabs.get(old_tab.tab_id)
            old_set_ids = tuple(
                initial_set.set_id
                for initial_set in old_tab.initial_condition_sets
            )
            if current_tab is None:
                self._deletion_result_fence.purge_tab(
                    old_tab.tab_id,
                    old_set_ids,
                )
                continue
            current_set_ids = {
                initial_set.set_id
                for initial_set in current_tab.initial_condition_sets
            }
            removed_set_ids = tuple(
                set_id for set_id in old_set_ids if set_id not in current_set_ids
            )
            if removed_set_ids:
                self._deletion_result_fence.purge_sets(
                    old_tab.tab_id,
                    removed_set_ids,
                )

    def select_tab(self, tab_id: str) -> None:
        target = str(tab_id)
        self._session.document.tab(target)
        self._installed = replace(self._installed, active_tab_id=target)

    def set_presentation(
        self,
        tab_id: str,
        state: ProjectPresentationState,
    ) -> None:
        tab = self._session.document.tab(tab_id)
        candidate = validate_project_presentation_state(
            self._session.document,
            tab,
            state,
        )
        self._installed = replace(
            self._installed,
            presentation_by_tab_id=tuple(
                (current_id, candidate if current_id == tab.tab_id else current)
                for current_id, current
                in self._installed.presentation_by_tab_id
            ),
        )

    def create_tab(
        self,
        *,
        run_defaults: SimulationExecutionSettings,
        scientific_settings: TabScientificSettings,
    ) -> MechanismTabDocument:
        candidate = self._session.document.add_tab(
            run_defaults=run_defaults,
            scientific_settings=scientific_settings,
        )
        created = candidate.tabs[-1]
        self.replace_document(candidate)
        self._installed = replace(
            self._installed,
            active_tab_id=created.tab_id,
        )
        return created

    def rename_tab(
        self,
        tab_id: str,
        name: str,
    ) -> ScientificDocumentDelta | None:
        candidate = self._session.document.rename_tab(tab_id, name)
        return self.replace_document(candidate)

    def close_tab(self, tab_id: str) -> str:
        """Close an already-confirmed eligible tab and return the new active ID."""
        document = self._session.document
        target = str(tab_id)
        index = next(
            (index for index, tab in enumerate(document.tabs) if tab.tab_id == target),
            None,
        )
        if index is None:
            raise KeyError(f"Unknown mechanism tab ID: {target!r}.")
        was_active = self.active_tab_id == target
        candidate = document.remove_tab(target)
        self.replace_document(candidate)
        if was_active:
            self._installed = replace(
                self._installed,
                active_tab_id=candidate.tabs[
                    min(index, len(candidate.tabs) - 1)
                ].tab_id,
            )
        return self.active_tab_id

    def tab_has_content_or_drafts(
        self,
        tab_id: str,
        *,
        fresh_run_defaults: SimulationExecutionSettings | None = None,
        fresh_scientific_settings: TabScientificSettings | None = None,
    ) -> bool:
        tab = self._session.document.tab(tab_id)
        return bool(
            tab.has_canonical_content(
                fresh_run_defaults=fresh_run_defaults,
                fresh_scientific_settings=fresh_scientific_settings,
            )
            or not self.presentation(tab_id).is_default_for(tab)
            or self.tab_runtime(tab_id).has_dirty_drafts
        )

    def has_dirty_drafts(self) -> bool:
        return any(
            runtime.has_dirty_drafts
            for _tab_id, runtime in self._installed.runtime_by_tab_id
        )

    def replace_active_tab(
        self,
        tab: MechanismTabDocument,
    ) -> ScientificDocumentDelta | None:
        if tab.tab_id != self.active_tab_id:
            raise ValueError("Replacement tab must be the active tab.")
        candidate = self._session.document.replace_tab(tab)
        return self.replace_document(candidate)

    def mutate_active_tab(
        self,
        mutation: Callable[[MechanismTabDocument], MechanismTabDocument],
    ) -> ScientificDocumentDelta | None:
        if not callable(mutation):
            raise TypeError("mutation must be callable.")
        candidate = mutation(self.active_tab)
        if not isinstance(candidate, MechanismTabDocument):
            raise TypeError("mutation must return a MechanismTabDocument.")
        return self.replace_active_tab(candidate)

    def replace_active_initial_condition_set(
        self,
        initial_set: InitialConditionSet,
    ) -> ScientificDocumentDelta | None:
        candidate = self.active_tab.replace_initial_condition_set(initial_set)
        return self.replace_active_tab(candidate)

    def set_active_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> ScientificDocumentDelta | None:
        return self.replace_active_tab(self.active_tab.with_run_defaults(settings))

    def set_active_run_override(
        self,
        set_id: str,
        settings: SimulationExecutionSettings | None,
    ) -> ScientificDocumentDelta | None:
        target = self.active_tab.initial_condition_set(set_id).set_id
        return self.replace_active_tab(
            self.active_tab.with_run_override(target, settings),
        )

    def set_active_scientific_settings(
        self,
        settings: TabScientificSettings,
    ) -> ScientificDocumentDelta | None:
        return self.replace_active_tab(
            replace(self.active_tab, scientific_settings=settings),
        )

    def remove_initial_condition_sets(
        self,
        set_ids: Iterable[str],
    ) -> ScientificDocumentDelta | None:
        targets = tuple(dict.fromkeys(str(set_id) for set_id in set_ids))
        if not targets:
            return None
        candidate = self.active_tab.without_initial_condition_sets(targets)
        return self.replace_active_tab(candidate)

    @staticmethod
    def _scientific_document_delta(
        previous: ProjectDocument,
        current: ProjectDocument,
    ) -> ScientificDocumentDelta:
        previous_tabs = {tab.tab_id: tab for tab in previous.tabs}
        affected_by_tab: list[tuple[str, tuple[str, ...]]] = []
        for tab in current.tabs:
            old_tab = previous_tabs.get(tab.tab_id)
            if old_tab is None:
                continue
            old_sets = {
                initial_set.set_id: initial_set
                for initial_set in old_tab.initial_condition_sets
            }
            existing_set_ids = tuple(
                initial_set.set_id
                for initial_set in tab.initial_condition_sets
                if initial_set.set_id in old_sets
            )
            if old_tab.mechanism_source != tab.mechanism_source:
                if existing_set_ids:
                    affected_by_tab.append(
                        (
                            tab.tab_id,
                            existing_set_ids,
                        )
                    )
                continue
            changed: list[str] = []
            for initial_set in tab.initial_condition_sets:
                old_set = old_sets.get(initial_set.set_id)
                if old_set is None:
                    continue
                if old_set.values_dict() != initial_set.values_dict():
                    changed.append(initial_set.set_id)
                    continue
                if (
                    old_tab.resolved_execution_settings(initial_set.set_id)
                    != tab.resolved_execution_settings(initial_set.set_id)
                ):
                    changed.append(initial_set.set_id)
            if old_tab.scientific_settings != tab.scientific_settings:
                changed = list(existing_set_ids)
            if changed:
                affected_by_tab.append(
                    (
                        tab.tab_id,
                        tuple(dict.fromkeys(changed)),
                    )
                )
        return ScientificDocumentDelta(
            affected_targets_by_tab=tuple(affected_by_tab),
        )
