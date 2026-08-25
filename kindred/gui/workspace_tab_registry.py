"""Active mechanism-tab lifecycle and noncanonical per-tab state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Iterable

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
)
from kindred.core.project_session import ProjectContentToken, ProjectSession
from kindred.core.project_presentation_state import (
    ProjectPresentationState,
    WorkspacePresentationState,
    default_workspace_presentation_state,
    reconcile_workspace_presentation_state,
    validate_project_presentation_state,
)
from kindred.gui.parameter_enumeration import (
    GuiParameterCoordinate,
    enumerate_parameter_coordinates,
)

if TYPE_CHECKING:
    from kindred.core.result_repository import ResultRepository

__all__ = [
    "CanonicalResultFence",
    "CanonicalMutationResult",
    "ScientificDocumentDelta",
    "TabRuntimeState",
    "WorkspaceTabRegistry",
]


CanonicalResultFence = Callable[["ScientificDocumentDelta"], None]


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
class ScientificDocumentDelta:
    """Affected existing targets from one accepted canonical document delta."""

    affected_targets_by_tab: tuple[tuple[str, tuple[str, ...]], ...] = ()

@dataclass(frozen=True, slots=True)
class CanonicalMutationResult:
    """One accepted exact-revision document mutation and its result fence."""

    token: ProjectContentToken
    changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.token, ProjectContentToken):
            raise TypeError("token must be ProjectContentToken.")
        if type(self.changed) is not bool:
            raise TypeError("changed must be bool.")


class WorkspaceTabRegistry:
    """Sole live active-tab ID plus presentation and transient tab state."""

    def __init__(
        self,
        session: ProjectSession,
        *,
        deletion_result_fence: object,
        canonical_result_fence: CanonicalResultFence,
        interactive_preview_fence: Callable[[ScientificDocumentDelta], None],
        runtime_factory: Callable[[str], TabRuntimeState] | None = None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be a ProjectSession.")
        if not callable(canonical_result_fence):
            raise TypeError("canonical_result_fence must be callable.")
        if not callable(interactive_preview_fence):
            raise TypeError("interactive_preview_fence must be callable.")
        self._session = session
        self._deletion_result_fence = deletion_result_fence
        self._canonical_result_fence = canonical_result_fence
        self._interactive_preview_fence = interactive_preview_fence
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

    def _workspace_state_for(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState | None = None,
    ) -> WorkspacePresentationState:
        if presentation is not None:
            return presentation
        defaults = default_workspace_presentation_state(document)
        default_by_tab_id = dict(defaults.presentation_by_tab_id)
        existing_by_tab_id = dict(self._presentation_by_tab_id)
        active_tab_id = (
            self.active_tab_id
            if any(tab.tab_id == self.active_tab_id for tab in document.tabs)
            else defaults.active_tab_id
        )
        candidate = WorkspacePresentationState(
            active_tab_id=active_tab_id,
            presentation_by_tab_id=tuple(
                (
                    tab.tab_id,
                    existing_by_tab_id.get(
                        tab.tab_id,
                        default_by_tab_id[tab.tab_id],
                    ),
                )
                for tab in document.tabs
            ),
        )
        return reconcile_workspace_presentation_state(
            document,
            candidate,
        )

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

    def document_identity(self, tab_id: str) -> tuple[int, str]:
        """Return the native-editor history identity for exactly one tab."""

        target = self.tab(tab_id).tab_id
        return self._session.identity_generation, target

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

    @staticmethod
    def _slider_coordinates(
        tab: MechanismTabDocument | None,
    ) -> dict[str, GuiParameterCoordinate]:
        """Return typed writable Slider identity, including retained scalars."""

        if tab is None or tab.prepared_dsl is None:
            return {}
        coordinates = enumerate_parameter_coordinates(
            tab.prepared_dsl,
        )
        return {
            item.name: item
            for item in coordinates
            if item.writable and not item.derived
        }

    def _prepare_replacement_runtime(
        self,
        document: ProjectDocument,
    ) -> tuple[tuple[str, TabRuntimeState], ...]:
        """Construct incoming transient owners before semantic publication."""

        return tuple(
            (tab.tab_id, self._runtime_factory(tab.tab_id))
            for tab in document.tabs
        )

    def replace_project(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | None,
        baseline_document: ProjectDocument,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
        """Install one fully prepared project by direct assignment."""

        if not self._session.is_content_token_current(expected_token):
            return None
        previous = self._session.document
        prepared_runtime = self._prepare_replacement_runtime(document)
        delta = self.scientific_document_delta(previous, document)
        self._publish_document_delta(previous, document, delta)
        token = self._session.replace_project(
            document,
            presentation,
            path=path,
            expected_token=expected_token,
            baseline_document=baseline_document,
        )
        if token is None:
            return None
        self._assign_workspace_state(
            self._session.document,
            presentation,
            retain_runtime=False,
            runtime_by_tab_id=prepared_runtime,
        )
        return CanonicalMutationResult(token=token, changed=True)

    def replace_document(
        self,
        document: ProjectDocument,
        *,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
        """Publish one admitted document mutation directly."""

        if not self._session.is_content_token_current(expected_token):
            return None
        previous = self._session.document
        changed = document != previous
        next_workspace = self._workspace_state_for(document)
        delta = self.scientific_document_delta(previous, document)
        if changed:
            self._publish_document_delta(previous, document, delta)
        token = self._session.replace_document(
            document,
            expected_token=expected_token,
        )
        if token is None:
            return None
        if changed:
            self._assign_workspace_state(self._session.document, next_workspace)
        return CanonicalMutationResult(token=token, changed=changed)

    def _publish_document_delta(
        self,
        previous: ProjectDocument,
        current: ProjectDocument,
        delta: ScientificDocumentDelta,
    ) -> None:
        """Fence result owners before the session publishes the document."""

        self._canonical_result_fence(delta)
        self._interactive_preview_fence(delta)
        self._fence_document_deletions(previous, current)

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
                self._interactive_preview_fence(
                    ScientificDocumentDelta(
                        affected_targets_by_tab=(
                            (old_tab.tab_id, old_set_ids),
                        )
                    )
                )
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
                self._interactive_preview_fence(
                    ScientificDocumentDelta(
                        affected_targets_by_tab=(
                            (old_tab.tab_id, removed_set_ids),
                        )
                    )
                )
                self._deletion_result_fence.purge_sets(
                    old_tab.tab_id,
                    removed_set_ids,
                )

    def select_tab(
        self,
        tab_id: str,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        target = str(tab_id)
        self._session.document.tab(target)
        if not self._session.is_content_token_current(expected_token):
            return None
        if target == self.active_tab_id:
            return self._session.capture_content_token()
        previous = self.workspace_presentation()
        self._active_tab_id = target
        return self._session.record_presentation_change(
            previous,
            self.workspace_presentation(),
        )

    def set_presentation(
        self,
        tab_id: str,
        state: ProjectPresentationState,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        if not isinstance(state, ProjectPresentationState):
            raise TypeError("state must be ProjectPresentationState.")
        if not self._session.is_content_token_current(expected_token):
            return None
        tab = self._session.document.tab(tab_id)
        candidate = validate_project_presentation_state(
            self._session.document,
            tab,
            state.reconciled(self._session.document, tab),
        )
        current = self.presentation(tab.tab_id)
        if candidate == current:
            return self._session.capture_content_token()
        previous = self.workspace_presentation()
        self._presentation_by_tab_id = tuple(
                (current_id, candidate if current_id == tab.tab_id else current)
                for current_id, current in self._presentation_by_tab_id
            )
        return self._session.record_presentation_change(
            previous,
            self.workspace_presentation(),
        )

    def create_tab(
        self,
        *,
        run_defaults: SimulationExecutionSettings,
        scientific_settings: TabScientificSettings,
        expected_token: ProjectContentToken,
    ) -> tuple[MechanismTabDocument, CanonicalMutationResult] | None:
        candidate = self._session.document.add_tab(
            run_defaults=run_defaults,
            scientific_settings=scientific_settings,
        )
        created = candidate.tabs[-1]
        result = self.replace_document(
            candidate,
            expected_token=expected_token,
        )
        if result is None:
            return None
        self._active_tab_id = created.tab_id
        self.workspace_presentation()
        return created, result

    def rename_tab(
        self,
        tab_id: str,
        name: str,
        *,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
        candidate = self._session.document.rename_tab(tab_id, name)
        return self.replace_document(candidate, expected_token=expected_token)

    def close_tab(
        self,
        tab_id: str,
        *,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
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
            self.workspace_presentation()
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
    ) -> CanonicalMutationResult | None:
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
    ) -> CanonicalMutationResult | None:
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
    ) -> CanonicalMutationResult | None:
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
    ) -> CanonicalMutationResult | None:
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
    ) -> CanonicalMutationResult | None:
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
    ) -> CanonicalMutationResult | None:
        tab = self.tab(tab_id)
        return self.replace_tab(
            tab.tab_id,
            replace(tab, scientific_settings=settings),
            expected_token=expected_token,
        )

    def remove_initial_condition_sets(
        self,
        tab_id: str,
        set_ids: Iterable[str],
        *,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
        targets = tuple(dict.fromkeys(str(set_id) for set_id in set_ids))
        if not targets:
            if not self._session.is_content_token_current(expected_token):
                return None
            return CanonicalMutationResult(
                token=self._session.capture_content_token(),
                changed=False,
            )
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
        return ScientificDocumentDelta(
            affected_targets_by_tab=tuple(affected_by_tab),
        )
