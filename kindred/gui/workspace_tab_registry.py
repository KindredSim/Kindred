"""Active mechanism-tab lifecycle and noncanonical per-tab state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
    prepare_tab_scientific_settings,
)
from kindred.core.project_session import ProjectContentToken, ProjectSession
from kindred.core.project_presentation_state import (
    ProjectPresentationState,
    WorkspacePresentationState,
    default_project_presentation_state,
    reconcile_workspace_presentation_state,
    validate_project_presentation_state,
    validate_workspace_presentation_state,
)

if TYPE_CHECKING:
    from kindred.core.result_repository import ResultRepository

__all__ = [
    "ScientificDocumentDelta",
    "TabRuntimeState",
    "WorkspaceTabRegistry",
]


@dataclass(slots=True)
class TabRuntimeState:
    """Noncanonical authoring and transient view state for exactly one tab."""

    reactions_draft: MechanismAuthoringSource | None = None
    notes_draft: str | None = None
    # Raw authored concentration text which has not yet crossed the normal
    # cell-settlement boundary.  Entries intentionally include both valid and
    # invalid text; validity is derived at the owner that admits a value.
    initial_text_by_set_id: dict[str, dict[str, str]] = field(
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
            or self.initial_text_by_set_id
            or self.slider_parameter_draft
            or self.slider_species_drafts_by_set_id
        )


@dataclass(frozen=True, slots=True)
class ScientificDocumentDelta:
    """The surviving and retired targets in one canonical document delta.

    ``affected_targets_by_tab`` is deliberately limited to set identities that
    survive in the replacement document.  ``removed_targets_by_tab`` contains
    identities that must be retired from the old physical result owners.
    """

    affected_targets_by_tab: tuple[tuple[str, tuple[str, ...]], ...] = ()
    removed_targets_by_tab: tuple[tuple[str, tuple[str, ...]], ...] = ()

class WorkspaceTabRegistry:
    """Sole live active-tab ID plus presentation and transient tab state."""

    def __init__(
        self,
        session: ProjectSession,
        *,
        computation_fence: Callable[[ScientificDocumentDelta], object],
        retire_interactive_plot_targets: Callable[
            [ScientificDocumentDelta], None
        ],
        post_document_publication: Callable[[ScientificDocumentDelta], None],
        runtime_factory: Callable[[str], TabRuntimeState] | None = None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be a ProjectSession.")
        if not callable(computation_fence):
            raise TypeError("computation_fence must be callable.")
        if not callable(retire_interactive_plot_targets):
            raise TypeError(
                "retire_interactive_plot_targets must be callable."
            )
        if not callable(post_document_publication):
            raise TypeError("post_document_publication must be callable.")
        self._session = session
        self._computation_fence = computation_fence
        self._retire_interactive_plot_targets = retire_interactive_plot_targets
        self._post_document_publication = post_document_publication
        self._runtime_factory = runtime_factory or (lambda _tab_id: TabRuntimeState())
        self._active_tab_id = ""
        self._presentation_by_tab_id: tuple[
            tuple[str, ProjectPresentationState], ...
        ] = ()
        self._runtime_by_tab_id: tuple[tuple[str, TabRuntimeState], ...] = ()
        self._assign_workspace_state(
            session.document,
            session.saved_baseline.presentation,
        )

    def _assign_workspace_state(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        retain_runtime: bool = True,
        runtime_by_tab_id: tuple[tuple[str, TabRuntimeState], ...] | None = None,
    ) -> None:
        """Assign the registry's direct presentation/runtime owners once."""

        self._active_tab_id = presentation.active_tab_id
        self._presentation_by_tab_id = presentation.presentation_by_tab_id
        if runtime_by_tab_id is not None:
            self._runtime_by_tab_id = runtime_by_tab_id
            return
        existing_runtime = dict(self._runtime_by_tab_id) if retain_runtime else {}
        runtime_items: list[tuple[str, TabRuntimeState]] = []
        for tab in document.tabs:
            runtime = existing_runtime.get(tab.tab_id)
            if runtime is None:
                runtime = self._runtime_factory(tab.tab_id)
            runtime_items.append((tab.tab_id, runtime))
        self._runtime_by_tab_id = tuple(runtime_items)

    @property
    def active_tab_id(self) -> str:
        return self._active_tab_id

    @property
    def document(self) -> ProjectDocument:
        """Expose current canonical truth to staged physical consumers."""

        return self._session.document

    def tab(self, tab_id: str) -> MechanismTabDocument:
        """Return canonical truth for exactly one caller-supplied tab ID."""

        return self._session.document.tab(str(tab_id))

    def capture_content_token(self) -> ProjectContentToken:
        return self._session.capture_content_token()

    def presentation(self, tab_id: str) -> ProjectPresentationState:
        target = str(tab_id)
        self._session.document.tab(target)
        for current_id, state in self._presentation_by_tab_id:
            if current_id == target:
                return state
        raise KeyError(f"Unknown presentation tab ID: {target!r}.")

    def tab_runtime(self, tab_id: str) -> TabRuntimeState:
        target = str(tab_id)
        self._session.document.tab(target)
        for current_id, runtime in self._runtime_by_tab_id:
            if current_id == target:
                return runtime
        raise KeyError(f"Unknown runtime tab ID: {target!r}.")

    def presentation_items(self) -> tuple[tuple[str, ProjectPresentationState], ...]:
        return self._presentation_by_tab_id

    def workspace_presentation(self) -> WorkspacePresentationState:
        return WorkspacePresentationState(
            active_tab_id=self.active_tab_id,
            presentation_by_tab_id=self.presentation_items(),
        )

    def replace_project(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | None,
        baseline_document: ProjectDocument,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Commit one admitted complete project and its workspace state."""

        if not self._session.is_content_token_current(expected_token):
            return None
        runtime_by_tab_id = tuple(
            (tab.tab_id, self._runtime_factory(tab.tab_id))
            for tab in document.tabs
        )
        previous = self._session.document
        delta = self.scientific_document_delta(previous, document)
        if document != previous and (
            delta.affected_targets_by_tab or delta.removed_targets_by_tab
        ):
            self._prepare_document_delta(delta, previous, document)
        token = self._session.replace_project(
            document,
            presentation,
            path=path,
            baseline_document=baseline_document,
            expected_token=expected_token,
        )
        if token is None:
            return None
        self._assign_workspace_state(
            document,
            presentation,
            retain_runtime=False,
            runtime_by_tab_id=runtime_by_tab_id,
        )
        return token

    def replace_document(
        self,
        document: ProjectDocument,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Publish one admitted document mutation directly."""

        if not self._session.is_content_token_current(expected_token):
            return None
        previous = self._session.document
        changed = document != previous
        existing_presentation = dict(self._presentation_by_tab_id)
        active_tab_id = (
            self.active_tab_id
            if any(tab.tab_id == self.active_tab_id for tab in document.tabs)
            else document.tabs[0].tab_id
        )
        next_workspace = WorkspacePresentationState(
            active_tab_id=active_tab_id,
            presentation_by_tab_id=tuple(
                (
                    tab.tab_id,
                    (
                        existing_presentation[tab.tab_id].reconciled(
                            document,
                            tab,
                        )
                        if tab.tab_id in existing_presentation
                        else default_project_presentation_state(tab)
                    ),
                )
                for tab in document.tabs
            ),
        )
        next_workspace = reconcile_workspace_presentation_state(
            document,
            next_workspace,
        )
        next_workspace = validate_workspace_presentation_state(
            document,
            next_workspace,
        )
        existing_runtime = dict(self._runtime_by_tab_id)
        next_runtime = tuple(
            (
                tab.tab_id,
                existing_runtime[tab.tab_id]
                if tab.tab_id in existing_runtime
                else self._runtime_factory(tab.tab_id),
            )
            for tab in document.tabs
        )
        delta = self.scientific_document_delta(previous, document)
        if changed and (
            delta.affected_targets_by_tab or delta.removed_targets_by_tab
        ):
            self._prepare_document_delta(delta, previous, document)
        token = self._session.replace_document(
            document,
            expected_token=expected_token,
        )
        if token is None:
            return None
        self._assign_workspace_state(
            document,
            next_workspace,
            runtime_by_tab_id=next_runtime,
        )
        if changed:
            self._post_document_publication(delta)
        return token

    def _prepare_document_delta(
        self,
        delta: ScientificDocumentDelta,
        previous: ProjectDocument,
        replacement: ProjectDocument,
    ) -> None:
        """Retire computation and old physical results before publication."""

        self._computation_fence(delta)
        self._retire_interactive_plot_targets(delta)

        existing_runtime = dict(self._runtime_by_tab_id)
        previous_by_tab_id = {tab.tab_id: tab for tab in previous.tabs}
        replacement_by_tab_id = {tab.tab_id: tab for tab in replacement.tabs}
        affected_by_tab: dict[str, list[str]] = {}
        removed_by_tab: dict[str, set[str]] = {}
        for tab_id, set_ids in delta.removed_targets_by_tab:
            tab = str(tab_id)
            removed = removed_by_tab.setdefault(tab, set())
            removed.update(str(set_id) for set_id in set_ids)
        for tab_id, set_ids in delta.affected_targets_by_tab:
            tab = str(tab_id)
            targets = affected_by_tab.setdefault(tab, [])
            removed = removed_by_tab.get(tab, set())
            for set_id in set_ids:
                value = str(set_id)
                if value not in removed and value not in targets:
                    targets.append(value)
        for tab_id, affected_ids in affected_by_tab.items():
            if not affected_ids:
                continue
            runtime = existing_runtime.get(tab_id)
            repository = None if runtime is None else runtime.result_repository
            if repository is not None:
                previous_tab = previous_by_tab_id.get(tab_id)
                replacement_tab = replacement_by_tab_id.get(tab_id)
                if (
                    previous_tab is not None
                    and replacement_tab is not None
                    and previous_tab.prepared_dsl is not replacement_tab.prepared_dsl
                ):
                    repository.purge_sets(tuple(affected_ids))
                else:
                    repository.deauthorize_sets(tuple(affected_ids))
        for tab_id, removed_ids in removed_by_tab.items():
            if not removed_ids:
                continue
            runtime = existing_runtime.get(tab_id)
            if runtime is not None:
                removed = set(removed_ids)
                runtime.selected_run_set_ids = tuple(
                    set_id
                    for set_id in runtime.selected_run_set_ids
                    if set_id not in removed
                )
                for set_id in removed:
                    runtime.initial_text_by_set_id.pop(set_id, None)
                    runtime.slider_species_drafts_by_set_id.pop(set_id, None)
            repository = None if runtime is None else runtime.result_repository
            if repository is not None:
                repository.purge_sets(tuple(removed_ids))

    def select_tab(
        self,
        tab_id: str,
    ) -> None:
        target = str(tab_id)
        self._session.document.tab(target)
        if target == self.active_tab_id:
            return
        self._active_tab_id = target

    def set_presentation(
        self,
        tab_id: str,
        state: ProjectPresentationState,
    ) -> None:
        tab = self._session.document.tab(tab_id)
        candidate = validate_project_presentation_state(
            self._session.document,
            tab,
            state.reconciled(self._session.document, tab),
        )
        current = self.presentation(tab.tab_id)
        if candidate == current:
            return
        self._presentation_by_tab_id = tuple(
            (current_id, candidate if current_id == tab.tab_id else current)
            for current_id, current in self._presentation_by_tab_id
        )

    def create_tab(
        self,
        *,
        run_defaults: SimulationExecutionSettings,
        scientific_settings: TabScientificSettings,
        expected_token: ProjectContentToken,
    ) -> MechanismTabDocument | None:
        candidate = self._session.document.add_tab(
            run_defaults=run_defaults,
            scientific_settings=scientific_settings,
        )
        created = candidate.tabs[-1]
        if self.replace_document(
            candidate,
            expected_token=expected_token,
        ) is None:
            return None
        self._active_tab_id = created.tab_id
        return created

    def rename_tab(
        self,
        tab_id: str,
        name: str,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        candidate = self._session.document.rename_tab(tab_id, name)
        return self.replace_document(candidate, expected_token=expected_token)

    def close_tab(
        self,
        tab_id: str,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Close an eligible tab and return its successor content token.

        Return ``None`` when the expected project content is no longer current.
        """
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
        result = self.replace_document(
            candidate,
            expected_token=expected_token,
        )
        if result is None:
            return None
        if was_active:
            self._active_tab_id = candidate.tabs[
                min(index, len(candidate.tabs) - 1)
            ].tab_id
        return result

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
            for _tab_id, runtime in self._runtime_by_tab_id
        )

    def replace_tab(
        self,
        tab_id: str,
        tab: MechanismTabDocument,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        target = self.tab(tab_id).tab_id
        if tab.tab_id != target:
            raise ValueError("Replacement tab must match the captured tab ID.")
        candidate = self._session.document.replace_tab(tab)
        return self.replace_document(candidate, expected_token=expected_token)

    def mutate_tab(
        self,
        tab_id: str,
        mutation: Callable[[MechanismTabDocument], MechanismTabDocument],
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        if not callable(mutation):
            raise TypeError("mutation must be callable.")
        target = self.tab(tab_id)
        candidate = mutation(target)
        if not isinstance(candidate, MechanismTabDocument):
            raise TypeError("mutation must return a MechanismTabDocument.")
        return self.replace_tab(
            target.tab_id,
            candidate,
            expected_token=expected_token,
        )

    def replace_initial_condition_set(
        self,
        tab_id: str,
        initial_set: InitialConditionSet,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        tab = self.tab(tab_id)
        candidate = tab.replace_initial_condition_set(initial_set)
        return self.replace_tab(
            tab.tab_id,
            candidate,
            expected_token=expected_token,
        )

    def set_run_defaults(
        self,
        tab_id: str,
        settings: SimulationExecutionSettings,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        tab = self.tab(tab_id)
        return self.replace_tab(
            tab.tab_id,
            tab.with_run_defaults(settings),
            expected_token=expected_token,
        )

    def set_run_override(
        self,
        tab_id: str,
        set_id: str,
        settings: SimulationExecutionSettings | None,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        tab = self.tab(tab_id)
        target = tab.initial_condition_set(set_id).set_id
        return self.replace_tab(
            tab.tab_id,
            tab.with_run_override(target, settings),
            expected_token=expected_token,
        )

    def set_scientific_settings(
        self,
        tab_id: str,
        settings: TabScientificSettings,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        tab = self.tab(tab_id)
        candidate_tab = prepare_tab_scientific_settings(tab, settings)
        candidate_document = self._session.document.replace_tab(candidate_tab)
        return self.replace_document(
            candidate_document,
            expected_token=expected_token,
        )

    def remove_initial_condition_sets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        targets = tuple(dict.fromkeys(str(set_id) for set_id in set_ids))
        if not targets:
            if not self._session.is_content_token_current(expected_token):
                return None
            return self._session.capture_content_token()
        tab = self.tab(tab_id)
        candidate = tab.without_initial_condition_sets(targets)
        return self.replace_tab(
            tab.tab_id,
            candidate,
            expected_token=expected_token,
        )

    @staticmethod
    def scientific_document_delta(
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
        current_tab_ids = {tab.tab_id for tab in current.tabs}
        removed_by_tab: list[tuple[str, tuple[str, ...]]] = []
        for old_tab in previous.tabs:
            old_set_ids = tuple(
                initial_set.set_id
                for initial_set in old_tab.initial_condition_sets
            )
            if old_tab.tab_id not in current_tab_ids:
                if old_set_ids:
                    removed_by_tab.append((old_tab.tab_id, old_set_ids))
                continue
            current_tab = next(
                tab for tab in current.tabs if tab.tab_id == old_tab.tab_id
            )
            current_set_ids = {
                initial_set.set_id
                for initial_set in current_tab.initial_condition_sets
            }
            removed_set_ids = tuple(
                set_id for set_id in old_set_ids if set_id not in current_set_ids
            )
            if removed_set_ids:
                removed_by_tab.append((old_tab.tab_id, removed_set_ids))
        return ScientificDocumentDelta(
            affected_targets_by_tab=tuple(affected_by_tab),
            removed_targets_by_tab=tuple(removed_by_tab),
        )
