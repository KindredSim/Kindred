"""Active mechanism-tab lifecycle and noncanonical per-tab state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Callable, Iterable, Protocol, runtime_checkable

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
    default_project_presentation_state,
    validate_project_presentation_state,
    validate_workspace_presentation_state,
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
        deletion_result_fence: DocumentDeletionResultFence,
        canonical_result_fence: CanonicalResultFence,
        runtime_factory: Callable[[str], TabRuntimeState] | None = None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be a ProjectSession.")
        if not isinstance(deletion_result_fence, DocumentDeletionResultFence):
            raise TypeError(
                "deletion_result_fence must implement DocumentDeletionResultFence."
            )
        if not callable(canonical_result_fence):
            raise TypeError("canonical_result_fence must be callable.")
        self._session = session
        self._deletion_result_fence = deletion_result_fence
        self._canonical_result_fence = canonical_result_fence
        self._runtime_factory = runtime_factory or (lambda _tab_id: TabRuntimeState())
        self._installed = self.prepare_install(
            document=session.document,
            presentation=session.saved_baseline.presentation,
        )

    @property
    def active_tab_id(self) -> str:
        return self._installed.active_tab_id

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
        """Construct presentation and transient replacement state."""

        if not isinstance(document, ProjectDocument):
            raise TypeError("document must be a ProjectDocument.")
        validated_presentation = validate_workspace_presentation_state(
            document,
            presentation,
        )
        target_active = validated_presentation.active_tab_id
        supplied = dict(validated_presentation.presentation_by_tab_id)
        next_presentation: list[tuple[str, ProjectPresentationState]] = []
        next_runtime: list[tuple[str, TabRuntimeState]] = []
        for tab in document.tabs:
            state = supplied[tab.tab_id]
            next_presentation.append(
                (
                    tab.tab_id,
                    validate_project_presentation_state(
                        document,
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

    @staticmethod
    def _clone_runtime(runtime: TabRuntimeState) -> TabRuntimeState:
        """Copy mutable transient containers while retaining owned resources."""

        if not isinstance(runtime, TabRuntimeState):
            raise TypeError("runtime must be TabRuntimeState.")
        return TabRuntimeState(
            reactions_draft=runtime.reactions_draft,
            notes_draft=runtime.notes_draft,
            invalid_initial_text_by_set_id={
                set_id: dict(values)
                for set_id, values in runtime.invalid_initial_text_by_set_id.items()
            },
            selected_run_set_ids=tuple(runtime.selected_run_set_ids),
            slider_parameter_draft=dict(runtime.slider_parameter_draft),
            slider_species_drafts_by_set_id={
                set_id: dict(values)
                for set_id, values in runtime.slider_species_drafts_by_set_id.items()
            },
            result_repository=runtime.result_repository,
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
            wegscheider_cyclicity_enabled=bool(
                tab.scientific_settings.wegscheider_policy
            ),
        )
        return {
            item.name: item
            for item in coordinates
            if item.writable and not item.derived
        }

    @staticmethod
    def _same_slider_coordinate(
        previous: GuiParameterCoordinate,
        current: GuiParameterCoordinate,
    ) -> bool:
        """Compare stable coordinate ownership, not incidental source/value."""

        return (
            previous.name,
            previous.family,
            previous.role,
            previous.writable,
            previous.derived,
            previous.unit,
        ) == (
            current.name,
            current.family,
            current.role,
            current.writable,
            current.derived,
            current.unit,
        )

    def _prepare_reconciled_install(
        self,
        *,
        previous_document: ProjectDocument,
        document: ProjectDocument,
    ) -> tuple[PreparedWorkspaceInstall, ScientificDocumentDelta]:
        """Build a complete reconciled registry value without live mutation."""

        scientific_delta = self.scientific_document_delta(
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
            tab_id: self._clone_runtime(state)
            for tab_id, state in self._installed.runtime_by_tab_id
            if tab_id in tab_ids
        }
        previous_tabs = {
            tab.tab_id: tab for tab in previous_document.tabs
        }
        invalidated_slider_targets: dict[str, set[str]] = {}
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
            if (
                runtime.reactions_draft is not None
                and runtime.reactions_draft.reactions_text
                == tab.mechanism_source.reactions_text
            ):
                # Runtime replacement establishes the Draft/canonical invariant
                # before callers perform any later persistence or view work.
                runtime.reactions_draft = None
            set_ids = {item.set_id for item in tab.initial_condition_sets}
            current_species = set(
                ()
                if tab.dsl_analysis is None
                else tab.dsl_analysis.species_names
            )
            runtime.invalid_initial_text_by_set_id = {
                set_id: {
                    species: raw
                    for species, raw in values.items()
                    if species in current_species
                }
                for set_id, values
                in runtime.invalid_initial_text_by_set_id.items()
                if set_id in set_ids
            }
            runtime.invalid_initial_text_by_set_id = {
                set_id: values
                for set_id, values
                in runtime.invalid_initial_text_by_set_id.items()
                if values
            }
            runtime.selected_run_set_ids = tuple(
                set_id
                for set_id in runtime.selected_run_set_ids
                if set_id in set_ids
            )
            previous_tab = previous_tabs.get(tab.tab_id)
            previous_species = set(
                ()
                if previous_tab is None or previous_tab.dsl_analysis is None
                else previous_tab.dsl_analysis.species_names
            )
            current_initials_by_set_id = {
                initial_set.set_id: initial_set.values_dict()
                for initial_set in tab.initial_condition_sets
            }
            previous_species_drafts = {
                set_id: dict(values)
                for set_id, values
                in runtime.slider_species_drafts_by_set_id.items()
            }
            runtime.slider_species_drafts_by_set_id = {
                set_id: {
                    species: value
                    for species, value in values.items()
                    if species in current_species
                    and species in previous_species
                    and float(value)
                    != float(
                        current_initials_by_set_id[set_id].get(species, 0.0)
                    )
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
            previous_parameters = self._slider_coordinates(previous_tab)
            current_parameters = self._slider_coordinates(tab)
            previous_parameter_drafts = dict(runtime.slider_parameter_draft)
            runtime.slider_parameter_draft = {
                name: value
                for name, value in runtime.slider_parameter_draft.items()
                if (
                    (previous_coordinate := previous_parameters.get(name))
                    is not None
                    and (current_coordinate := current_parameters.get(name))
                    is not None
                    and self._same_slider_coordinate(
                        previous_coordinate,
                        current_coordinate,
                    )
                    and math.isfinite(float(value))
                    and float(value) != float(current_coordinate.value)
                )
            }
            affected = invalidated_slider_targets.setdefault(tab.tab_id, set())
            if set(previous_parameter_drafts) != set(runtime.slider_parameter_draft):
                affected.update(set_ids)
            for set_id, values in previous_species_drafts.items():
                if set(values) != set(
                    runtime.slider_species_drafts_by_set_id.get(set_id, {})
                ) and set_id in set_ids:
                    affected.add(set_id)
        active_tab_id = self.active_tab_id
        if active_tab_id not in tab_ids:
            active_tab_id = document.tabs[0].tab_id
        prepared = PreparedWorkspaceInstall(
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
        if any(invalidated_slider_targets.values()):
            affected_by_tab = {
                tab_id: list(set_ids)
                for tab_id, set_ids in scientific_delta.affected_targets_by_tab
            }
            for tab in document.tabs:
                additions = invalidated_slider_targets.get(tab.tab_id, set())
                if not additions:
                    continue
                existing = affected_by_tab.setdefault(tab.tab_id, [])
                existing.extend(
                    initial_set.set_id
                    for initial_set in tab.initial_condition_sets
                    if initial_set.set_id in additions
                    and initial_set.set_id not in existing
                )
            scientific_delta = ScientificDocumentDelta(
                affected_targets_by_tab=tuple(
                    (tab.tab_id, tuple(affected_by_tab[tab.tab_id]))
                    for tab in document.tabs
                    if affected_by_tab.get(tab.tab_id)
                )
            )
        return prepared, scientific_delta

    def replace_document(
        self,
        document: ProjectDocument,
        *,
        expected_token: ProjectContentToken,
    ) -> CanonicalMutationResult | None:
        """Conditionally install one document and return its exact result fence."""

        previous = self._session.document
        prepared_document = self._session.prepare_document_mutation(document)
        candidate = prepared_document.document
        changed = candidate != previous
        if changed:
            prepared_workspace, delta = self._prepare_reconciled_install(
                previous_document=previous,
                document=candidate,
            )
        else:
            prepared_workspace = self._installed
            delta = ScientificDocumentDelta()
        if changed:
            if not self._session.is_content_token_current(expected_token):
                return None
            # Fence the already-computed non-deletion target delta before the
            # session publishes the candidate document.  The callback is the
            # one exact result-owner boundary; callers must not deauthorize
            # this delta after replace_document returns.
            self._canonical_result_fence(delta)
        token = self._session.commit_document_mutation(
            prepared_document,
            expected_token=expected_token,
            fence_deletions=self._fence_document_deletions,
        )
        if token is None:
            return None
        if changed:
            self._installed = prepared_workspace
        return CanonicalMutationResult(token=token, changed=changed)

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
        self._installed = replace(self._installed, active_tab_id=target)
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
        self._installed = replace(
            self._installed,
            presentation_by_tab_id=tuple(
                (current_id, candidate if current_id == tab.tab_id else current)
                for current_id, current
                in self._installed.presentation_by_tab_id
            ),
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
        previous = self.workspace_presentation()
        self._installed = replace(
            self._installed,
            active_tab_id=created.tab_id,
        )
        token = self._session.record_presentation_change(
            previous,
            self.workspace_presentation(),
        )
        return created, replace(result, token=token)

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
            previous = self.workspace_presentation()
            self._installed = replace(
                self._installed,
                active_tab_id=candidate.tabs[
                    min(index, len(candidate.tabs) - 1)
                ].tab_id,
            )
            token = self._session.record_presentation_change(
                previous,
                self.workspace_presentation(),
            )
            result = replace(result, token=token)
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
            for _tab_id, runtime in self._installed.runtime_by_tab_id
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
