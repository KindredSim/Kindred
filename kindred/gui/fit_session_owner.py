"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from PySide6 import QtCore, QtWidgets

from kindred.core.algebra.observable_introspection import (
    FitLocalObservable,
    validate_fit_local_observable,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    InitialConditionSet,
    MechanismTabDocument,
    SimulationExecutionSettings,
)
from kindred.core.project_session import ProjectSession
from kindred.core.simulator.dsl_parse import (
    DslDerivedObservableDefinition,
    DslDerivedScalarDefinition,
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_prepared_dsl_with_observables,
    derive_prepared_dsl_with_parameter_updates,
)
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitExecutionRequest,
    FitInitialVariable,
    FitOriginSnapshot,
    FitParameterVariable,
    FitSetupSnapshot,
    build_fit_execution_request,
    capture_fit_origin,
    revalidate_fit_request_inclusions,
)
from kindred.gui.fitting.contracts import FitParametricSettings, FitRunIdentity
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitLiveUpdate,
    GlobalFitWorker,
)
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionRequest,
)
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry

__all__ = [
    "CanonicalDetailedBalanceSourceCommit",
    "FitSessionOwner",
]

WindowFactory = Callable[..., FittingWindow]
ColorResolver = Callable[[str], object]
DeauthorizeTargets = Callable[[str, Iterable[str]], object]
CanonicalDetailedBalanceSourceCommit = Callable[
    [str, MechanismAuthoringSource],
    PreparedDslRequestSet | None,
]


@dataclass(frozen=True, slots=True)
class _PendingFitDetailedBalanceCorrection:
    generation: int
    correction: DetailedBalanceCorrectionRequest
    execution_request: FitExecutionRequest


def _finite_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _bound(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if math.isnan(number):
        raise ValueError(f"{field} cannot be NaN.")
    return number


def _int(value: object, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc


class FitSessionOwner(QtCore.QObject):
    """Retain the window, typed local request, worker, candidate, and setup."""

    window_changed = QtCore.Signal(bool)
    execution_changed = QtCore.Signal(bool)
    canonical_applied = QtCore.Signal(str, object)
    status_changed = QtCore.Signal(str)
    detailed_balance_correction_requested = QtCore.Signal(object)
    owned_qthreads_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        session: ProjectSession,
        registry: WorkspaceTabRegistry,
        gate: ComputationGate,
        deauthorize_targets: DeauthorizeTargets,
        commit_detailed_balance_source: CanonicalDetailedBalanceSourceCommit,
        color_resolver: ColorResolver | None = None,
        parent_widget: QtWidgets.QWidget | None = None,
        window_factory: WindowFactory = FittingWindow,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(session, ProjectSession):
            raise TypeError("session must be ProjectSession.")
        if not isinstance(registry, WorkspaceTabRegistry):
            raise TypeError("registry must be WorkspaceTabRegistry.")
        if not isinstance(gate, ComputationGate):
            raise TypeError("gate must be ComputationGate.")
        if not callable(deauthorize_targets):
            raise TypeError("deauthorize_targets must be callable.")
        if not callable(commit_detailed_balance_source):
            raise TypeError("commit_detailed_balance_source must be callable.")
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        if not callable(window_factory):
            raise TypeError("window_factory must be callable.")
        self._session = session
        self._registry = registry
        self._gate = gate
        self._deauthorize_targets = deauthorize_targets
        self._commit_detailed_balance_source = commit_detailed_balance_source
        self._color_resolver = color_resolver
        self._parent_widget = parent_widget
        self._window_factory = window_factory
        self._window: FittingWindow | None = None
        self._worker: GlobalFitWorker | None = None
        self._candidate: FitCandidate | None = None
        self._origin: FitOriginSnapshot | None = None
        self._local_observables: tuple[FitLocalObservable, ...] = ()
        self._local_request: FitExecutionRequest | None = None
        self._local_setup_snapshot: FitSetupSnapshot | None = None
        self._pending_detailed_balance: (
            _PendingFitDetailedBalanceCorrection | None
        ) = None
        self._worker_generation = 0
        self._publication_fenced_generation: int | None = None
        self._closing = False
        self._application_close_fenced = False

    @property
    def has_window(self) -> bool:
        return self._window is not None

    @property
    def project_defaults(self) -> FitDefaults:
        return self._session.document.fit_defaults

    def owned_qthreads(self) -> tuple[QtCore.QThread, ...]:
        """Return the exact native Fit threads still retained by this owner."""

        worker = self._worker
        if worker is None:
            return ()
        return (worker,)

    def open_or_focus(self, tab_id: str | None = None) -> FittingWindow | None:
        """Open one modal origin window, or focus the existing singleton."""
        self._require_thread()
        if self._application_close_fenced:
            return None
        window = self._window
        if window is not None:
            window.show()
            window.raise_()
            window.activateWindow()
            return window
        if not self._gate.is_idle:
            self._set_status(
                f"Fit Open rejected: {self._gate.active_kind or 'another computation'} is active."
            )
            return None
        requested_id = str(tab_id or self._registry.active_tab_id)
        try:
            origin = capture_fit_origin(self._session.document, requested_id)
            window = self._window_factory(
                origin,
                close_handler=self._request_window_close,
                color_resolver=self._color_resolver,
                parent=self._parent_widget,
            )
        except Exception as exc:
            self._set_status(f"Fit Open failed: {exc}")
            return None
        if not isinstance(window, FittingWindow):
            raise TypeError("window_factory must return FittingWindow.")
        self._origin = origin
        self._window = window
        self._local_observables = ()
        self._local_request = None
        self._local_setup_snapshot = None
        self._pending_detailed_balance = None
        self._candidate = None
        try:
            initial_request = self._parse_request(window.raw_request_values())
            window.install_initial_snapshot(
                FitSetupSnapshot.from_request(initial_request)
            )
            window.set_effective_source(
                initial_request.mechanism_source,
                initial_request.prepared_dsl,
            )
        except (TypeError, ValueError, KeyError) as exc:
            window.deleteLater()
            self._window = None
            self._origin = None
            self._set_status(f"Fit Open failed: {exc}")
            return None
        window.executeRequested.connect(self.execute)
        window.cancelRequested.connect(self.cancel)
        window.pauseRequested.connect(self.pause)
        window.resumeRequested.connect(self.resume)
        window.applyRequested.connect(self.apply_candidate)
        window.addObservableRequested.connect(self.add_local_observable)
        window.restoreSetupRequested.connect(self.restore_setup_snapshot)
        window.projectDefaultsRequested.connect(self.set_project_defaults)
        window.destroyed.connect(self._on_window_destroyed)
        self.window_changed.emit(True)
        window.show()
        window.raise_()
        window.activateWindow()
        self._set_status(f"Fit opened for {origin.tab_name}.")
        return window

    @QtCore.Slot()
    def execute(self) -> bool:
        """Build typed local setup, then directly admit one Fit worker."""
        self._require_thread()
        window = self._window
        origin = self._origin
        if window is None or origin is None:
            self._set_status("Fit rejected: there is no open Fit window.")
            return False
        if self._worker is not None:
            self._set_status("Fit rejected: this Fit is already executing.")
            return False
        if self._closing or self._application_close_fenced:
            self._set_status("Fit rejected while the window is closing.")
            return False
        if not self._origin_is_current():
            self._set_status("Fit rejected: origin science is no longer current.")
            return False
        if not self._gate.is_idle:
            self._set_status(
                f"Fit rejected: {self._gate.active_kind or 'another computation'} is active."
            )
            return False
        if window.has_pending_target_changes:
            self._set_status(
                "Fit rejected: Apply or Revert the pending target edits first."
            )
            return False
        try:
            request = self._build_request(window.raw_request_values())
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup is invalid: {exc}")
            return False
        if not self._gate.try_acquire(self, "fit"):
            self._set_status("Fit rejected: another computation is active.")
            return False
        setup_snapshot = FitSetupSnapshot.from_request(request)
        try:
            worker = GlobalFitWorker(
                request,
                setup_snapshot=setup_snapshot,
                parent=self,
            )
        except Exception as exc:
            self._gate.release(self)
            self._set_status(f"Fit preparation failed before dispatch: {exc}")
            return False
        self._local_observables = request.local_observables
        window.set_effective_source(
            request.mechanism_source,
            request.prepared_dsl,
        )
        self._local_request = request
        self._local_setup_snapshot = setup_snapshot
        self._pending_detailed_balance = None
        self._worker_generation += 1
        generation = self._worker_generation
        self._publication_fenced_generation = None
        worker.progress.connect(
            lambda value, text, g=generation: self._on_progress(g, value, text)
        )
        worker.auto_modes_resolved.connect(
            lambda modes, g=generation: self._on_auto_modes(g, modes)
        )
        worker.best_updated.connect(
            lambda update, g=generation: self._on_best(g, update)
        )
        worker.curves_updated.connect(
            lambda curves, g=generation: self._on_live_curves(g, curves)
        )
        worker.run_identity_ready.connect(
            lambda identity, g=generation: self._on_run_identity(
                g,
                identity,
            )
        )
        worker.succeeded.connect(
            lambda candidate, g=generation: self._on_succeeded(g, candidate)
        )
        worker.failed.connect(
            lambda message, g=generation: self._on_failed(g, message)
        )
        worker.detailed_balance_correction_required.connect(
            lambda correction, g=generation: (
                self._on_detailed_balance_correction_required(
                    g,
                    correction,
                )
            )
        )
        worker.finished.connect(
            lambda g=generation: self._on_worker_finished(g)
        )
        self._worker = worker
        self.owned_qthreads_changed.emit()
        self._candidate = None
        window.clear_candidate()
        window.set_executing(True)
        self.execution_changed.emit(True)
        self._set_status("Fit started.")
        worker.start()
        return True

    @QtCore.Slot()
    def cancel(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        self._pending_detailed_balance = None
        self._publication_fenced_generation = self._worker_generation
        worker.cancel()
        self._set_status("Fit cancellation requested.")
        return True

    @QtCore.Slot()
    def pause(self) -> bool:
        if self._application_close_fenced or self._worker is None:
            return False
        self._worker.pause()
        self._set_status("Fit paused.")
        return True

    @QtCore.Slot()
    def resume(self) -> bool:
        if self._application_close_fenced or self._worker is None:
            return False
        self._worker.resume()
        self._set_status("Fit resumed.")
        return True

    def complete_detailed_balance_correction(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> bool:
        """Commit one verified correction or abort the waiting Fit branch."""

        self._require_thread()
        if self._application_close_fenced or self._closing:
            return False
        if not isinstance(outcome, DetailedBalanceCorrectionOutcome):
            raise TypeError("outcome must be DetailedBalanceCorrectionOutcome.")
        pending = self._pending_detailed_balance
        worker = self._worker
        if pending is None or worker is None:
            return False
        request = pending.correction
        if (
            pending.generation != self._worker_generation
            or pending.generation == self._publication_fenced_generation
            or (
                outcome.request_id,
                outcome.origin,
                outcome.tab_id,
                outcome.source,
                outcome.issue,
            )
            != (
                request.request_id,
                request.origin,
                request.tab_id,
                request.source,
                request.issue,
            )
        ):
            self._set_status(
                "Detailed-balance correction rejected: request identity "
                "is no longer current."
            )
            return False
        if outcome.cancelled:
            self._pending_detailed_balance = None
            worker.submit_detailed_balance_correction(
                request.request_id,
                None,
                rejection_message=(
                    "Detailed-balance correction cancelled; Fit aborted."
                ),
            )
            self._set_status("Detailed-balance correction cancelled; Fit aborted.")
            return True

        corrected_text = outcome.corrected_source
        if corrected_text is None:
            raise ValueError("Accepted correction has no corrected source.")
        corrected_canonical = (
            pending.execution_request.origin.mechanism_source.with_reactions_text(
                corrected_text
            )
        )
        try:
            canonical_prepared = self._commit_detailed_balance_source(
                request.tab_id,
                corrected_canonical,
            )
            if not isinstance(canonical_prepared, PreparedDslRequestSet):
                raise RuntimeError(
                    "Canonical source admission rejected the correction."
                )
            self._verify_detailed_balance_resolved(canonical_prepared)
            live_tab = self._session.document.tab(request.tab_id)
            live_runtime = self._registry.tab_runtime(request.tab_id)
            if (
                live_tab.mechanism_source != corrected_canonical
                or live_tab.prepared_dsl is not canonical_prepared
                or live_runtime.reactions_draft is not None
            ):
                raise RuntimeError(
                    "Canonical source commit did not install the exact "
                    "verified correction and clear its matching draft."
                )
            rebased = capture_fit_origin(
                self._session.document,
                request.tab_id,
            )
            previous_origin = pending.execution_request.origin
            if (
                rebased.scientific_settings
                != previous_origin.scientific_settings
                or rebased.mapped_datasets != previous_origin.mapped_datasets
            ):
                raise RuntimeError(
                    "Fit origin changed beyond the committed mechanism source."
                )
            definitions = tuple(
                DslDerivedObservableDefinition(
                    name=item.name,
                    expression=item.expression,
                    scalar_parameters=tuple(
                        DslDerivedScalarDefinition(
                            name=scalar.name,
                            value=scalar.value,
                            minimum=scalar.minimum,
                            maximum=scalar.maximum,
                            scope=scalar.scope,
                        )
                        for scalar in item.scalar_parameters
                    ),
                )
                for item in pending.execution_request.local_observables
            )
            corrected_prepared = (
                rebased.prepared_dsl
                if not definitions
                else derive_prepared_dsl_with_observables(
                    rebased.prepared_dsl,
                    definitions,
                )
            )
            corrected_effective = MechanismAuthoringSource.from_full_dsl_text(
                corrected_prepared.analysis.normalized_text
            )
            self._verify_detailed_balance_resolved(corrected_prepared)
            writable_coordinates = {
                coordinate.name: coordinate
                for coordinate in enumerate_parameter_coordinates(
                    corrected_prepared,
                    wegscheider_cyclicity_enabled=bool(
                        rebased.scientific_settings.wegscheider_policy
                    ),
                )
            }
            missing_parameters = sorted(
                {
                    parameter.name
                    for parameter in pending.execution_request.parameters
                }
                - set(writable_coordinates)
            )
            if missing_parameters:
                raise RuntimeError(
                    "Detailed-balance correction removed writable Fit "
                    "coordinates: "
                    + ", ".join(missing_parameters)
                )
            prior_parameters = {
                parameter.name: parameter
                for parameter in pending.execution_request.parameters
            }
            corrected_parameter_items: list[FitParameterVariable] = []
            canonical_names: set[str] = set()
            for canonical_default in rebased.parameter_defaults:
                canonical_names.add(canonical_default.name)
                prior = prior_parameters.get(canonical_default.name)
                corrected_parameter_items.append(
                    canonical_default
                    if prior is None
                    else replace(
                        prior,
                        display_name=writable_coordinates[
                            prior.name
                        ].display_name,
                        family=writable_coordinates[
                            prior.name
                        ].family,
                        role=writable_coordinates[
                            prior.name
                        ].role,
                        unit=writable_coordinates[
                            prior.name
                        ].unit,
                    )
                )
            corrected_parameter_items.extend(
                replace(
                    parameter,
                    display_name=writable_coordinates[
                        parameter.name
                    ].display_name,
                    family=writable_coordinates[
                        parameter.name
                    ].family,
                    role=writable_coordinates[
                        parameter.name
                    ].role,
                    unit=writable_coordinates[
                        parameter.name
                    ].unit,
                )
                for parameter in pending.execution_request.parameters
                if parameter.name not in canonical_names
            )
            corrected_parameters = tuple(corrected_parameter_items)
            corrected_request = build_fit_execution_request(
                origin=rebased,
                mechanism_source=corrected_effective,
                prepared_dsl=corrected_prepared,
                local_observables=(
                    pending.execution_request.local_observables
                ),
                datasets=pending.execution_request.datasets,
                parameters=corrected_parameters,
                initials=pending.execution_request.initials,
                defaults=pending.execution_request.defaults,
            )
            revalidate_fit_request_inclusions(
                document=self._session.document,
                request=corrected_request,
            )
            corrected_snapshot = FitSetupSnapshot.from_request(
                corrected_request
            )
            worker.submit_detailed_balance_correction(
                request.request_id,
                corrected_request,
                corrected_snapshot,
            )
        except Exception as exc:
            self._pending_detailed_balance = None
            rejection_message = (
                f"Detailed-balance correction rejected; Fit aborted: {exc}"
            )
            worker.submit_detailed_balance_correction(
                request.request_id,
                None,
                rejection_message=rejection_message,
            )
            self._set_status(rejection_message)
            return False
        self._pending_detailed_balance = None
        self._origin = rebased
        self._local_request = corrected_request
        self._local_setup_snapshot = corrected_snapshot
        if self._window is not None:
            preserved = self._window.raw_request_values()
            local_scalar_names = {
                scalar.name
                for observable in corrected_request.local_observables
                for scalar in observable.scalar_parameters
            }
            self._window.rebase(
                rebased,
                preserved=preserved,
                supplemental_parameters=tuple(
                    parameter
                    for parameter in corrected_request.parameters
                    if parameter.name in local_scalar_names
                ),
            )
            self._window.commit_current_target_selection()
            self._window.install_initial_snapshot(corrected_snapshot)
            self._window.set_effective_source(
                corrected_effective,
                corrected_prepared,
            )
            self._window.set_executing(True)
        self._set_status(
            "Detailed-balance correction committed; continuing Fit once."
        )
        return True

    @QtCore.Slot(str, str, str, str, str)
    def add_local_observable(
        self,
        name: str,
        expression: str,
        dataset_id: str,
        raw_column: str,
        scalar_scope: str,
    ) -> bool:
        """Validate and retain a local observable without project mutation."""
        self._require_thread()
        if self._application_close_fenced:
            return False
        window = self._window
        origin = self._origin
        if window is None or origin is None or self._worker is not None:
            return False
        try:
            prior_definitions = tuple(
                DslDerivedObservableDefinition(
                    name=item.name,
                    expression=item.expression,
                    scalar_parameters=tuple(
                        DslDerivedScalarDefinition(
                            name=scalar.name,
                            value=scalar.value,
                            minimum=scalar.minimum,
                            maximum=scalar.maximum,
                            scope=scalar.scope,
                        )
                        for scalar in item.scalar_parameters
                    ),
                )
                for item in self._local_observables
            )
            prior_prepared = (
                origin.prepared_dsl
                if not prior_definitions
                else derive_prepared_dsl_with_observables(
                    origin.prepared_dsl,
                    prior_definitions,
                )
            )
            observable = validate_fit_local_observable(
                name=name,
                expression=expression,
                prepared_dsl=prior_prepared,
                additional_known_observables=tuple(
                    item.name for item in self._local_observables
                ),
                unknown_scalar_scope=(
                    "per_dataset"
                    if str(scalar_scope).strip().lower() == "per-dataset"
                    else "shared"
                ),
            )
            mapped = next(
                item
                for item in origin.mapped_datasets
                if item.dataset_id == str(dataset_id)
            )
            if not any(
                item.raw_column == str(raw_column)
                for item in mapped.observations
            ):
                raise ValueError("Selected observation column is not captured.")
            self._local_observables = (*self._local_observables, observable)
            definitions = (
                *prior_definitions,
                DslDerivedObservableDefinition(
                    name=observable.name,
                    expression=observable.expression,
                    scalar_parameters=tuple(
                        DslDerivedScalarDefinition(
                            name=scalar.name,
                            value=scalar.value,
                            minimum=scalar.minimum,
                            maximum=scalar.maximum,
                            scope=scalar.scope,
                        )
                        for scalar in observable.scalar_parameters
                    ),
                ),
            )
            effective_prepared = derive_prepared_dsl_with_observables(
                origin.prepared_dsl,
                definitions,
            )
            effective = MechanismAuthoringSource.from_full_dsl_text(
                effective_prepared.analysis.normalized_text
            )
            window.install_local_observable(
                dataset_id=mapped.dataset_id,
                raw_column=str(raw_column),
                observable_name=observable.name,
                scalar_parameters=observable.scalar_parameters,
            )
            window.set_effective_source(effective, effective_prepared)
        except (StopIteration, TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Add Observable rejected: {exc}")
            return False
        self._candidate = None
        self._local_request = None
        window.clear_candidate()
        self._set_status(
            f"Observable {observable.name!r} is Fit-local until Apply."
        )
        return True

    @QtCore.Slot(object)
    def restore_setup_snapshot(self, snapshot: object) -> bool:
        """Restore one owner-held typed setup without canonical mutation."""

        self._require_thread()
        window = self._window
        origin = self._origin
        if (
            self._application_close_fenced
            or self._worker is not None
            or window is None
            or origin is None
            or not isinstance(snapshot, FitSetupSnapshot)
        ):
            return False
        if (
            snapshot.origin_tab_id != origin.tab_id
            or snapshot.origin_capture_id != origin.capture_id
        ):
            self._set_status(
                "Fit setup reset rejected: the snapshot origin is stale."
            )
            return False
        try:
            window.clear_candidate()
            restored_snapshot = window.restore_snapshot(snapshot)
            definitions = tuple(
                DslDerivedObservableDefinition(
                    name=item.name,
                    expression=item.expression,
                    scalar_parameters=tuple(
                        DslDerivedScalarDefinition(
                            name=scalar.name,
                            value=scalar.value,
                            minimum=scalar.minimum,
                            maximum=scalar.maximum,
                            scope=scalar.scope,
                        )
                        for scalar in item.scalar_parameters
                    ),
                )
                for item in restored_snapshot.local_observables
            )
            effective_prepared = (
                origin.prepared_dsl
                if not definitions
                else derive_prepared_dsl_with_observables(
                    origin.prepared_dsl,
                    definitions,
                )
            )
            effective_source = MechanismAuthoringSource.from_full_dsl_text(
                effective_prepared.analysis.normalized_text
            )
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup reset failed: {exc}")
            return False
        self._local_observables = restored_snapshot.local_observables
        self._local_request = None
        self._local_setup_snapshot = None
        self._candidate = None
        window.set_effective_source(
            effective_source,
            effective_prepared,
        )
        self._set_status("Fit setup snapshot restored.")
        return True

    @QtCore.Slot(object)
    def apply_candidate(self, scopes: object) -> bool:
        """Apply any selected independent scopes through one document replacement."""
        self._require_thread()
        if self._application_close_fenced:
            return False
        candidate = self._candidate
        origin = self._origin
        window = self._window
        if candidate is None or origin is None or window is None:
            self._set_status("There is no completed Fit candidate to Apply.")
            return False
        selected = frozenset(str(item) for item in self._scope_items(scopes))
        available = frozenset(candidate.available_scopes)
        if not selected or not selected.issubset(available):
            self._set_status("Apply contains an unavailable or empty scope selection.")
            return False
        if not self._origin_is_current():
            self._set_status("Apply rejected: origin science is no longer current.")
            return False
        target_changes_pending = window.has_pending_target_changes
        preserved = window.raw_request_values()
        previous_document = self._session.document
        try:
            local_scalar_names = self._candidate_local_scalar_names(
                candidate
            )
            applied_canonical_parameter_names = tuple(
                sorted(
                    name
                    for name, _value in candidate.parameters
                    if (
                        "parameters" in selected
                        and name not in local_scalar_names
                    )
                )
            )
            unapplied_local_items = tuple(
                sorted(
                    item.name
                    for item in candidate.setup_snapshot.parameters
                    if (
                        item.fit
                        and item.name in local_scalar_names
                        and "parameters" in selected
                        and "observable_definitions" not in selected
                    )
                )
            )
            remaining_candidate = self._remaining_candidate_after_apply(
                candidate,
                selected,
            )
            preserved = self._preserved_state_with_terminal_values(
                preserved,
                candidate,
            )
            current_tab = previous_document.tab(origin.tab_id)
            next_tab = self._build_applied_tab(
                current_tab,
                candidate,
                selected,
                origin,
            )
            next_document = previous_document.replace_tab(next_tab)
        except Exception as exc:
            self._set_status(f"Apply rejected: {exc}")
            return False
        try:
            delta = self._registry.replace_document(next_document)
            changed = delta is not None
            if delta is not None:
                for affected_tab_id, set_ids in delta.affected_targets_by_tab:
                    self._deauthorize_targets(affected_tab_id, set_ids)
        except Exception as exc:
            self._set_status(
                f"Apply failed at the document boundary: {exc}"
            )
            return False
        self._local_request = None
        self._local_setup_snapshot = None
        self._local_observables = (
            ()
            if remaining_candidate is None
            else remaining_candidate.local_observables
        )
        self.canonical_applied.emit(origin.tab_id, selected)
        try:
            rebased = capture_fit_origin(self._session.document, origin.tab_id)
            self._origin = rebased
            remaining_local_scalar_names = {
                scalar.name
                for observable in self._local_observables
                for scalar in observable.scalar_parameters
            }
            window.rebase(
                rebased,
                preserved=preserved,
                canonicalized_scopes=(
                    selected
                    & frozenset(
                        {
                            "initial_concentrations",
                            "simulation_setup",
                        }
                    )
                ),
                supplemental_parameters=tuple(
                    parameter
                    for parameter in candidate.setup_snapshot.parameters
                    if parameter.name in remaining_local_scalar_names
                ),
                preserve_snapshots=True,
            )
            if not target_changes_pending:
                window.commit_current_target_selection()
            refreshed_request = self._parse_request(
                window.raw_request_values()
            )
            refreshed_snapshot = FitSetupSnapshot.from_request(
                refreshed_request
            )
            window.set_effective_source(
                refreshed_request.mechanism_source,
                refreshed_request.prepared_dsl,
            )
            self._local_observables = refreshed_request.local_observables
            if remaining_candidate is None:
                self._candidate = None
                window.clear_candidate()
            else:
                remaining_setup = replace(
                    candidate.setup_snapshot,
                    origin_tab_id=rebased.tab_id,
                    origin_capture_id=rebased.capture_id,
                    mechanism_source=refreshed_request.mechanism_source,
                    prepared_dsl=refreshed_request.prepared_dsl,
                    datasets=(
                        refreshed_snapshot.datasets
                        if "simulation_setup" in selected
                        else candidate.setup_snapshot.datasets
                    ),
                    initials=(
                        refreshed_snapshot.initials
                        if "initial_concentrations" in selected
                        else candidate.setup_snapshot.initials
                    ),
                    local_observables=refreshed_request.local_observables,
                )
                remaining_candidate = replace(
                    remaining_candidate,
                    origin_capture_id=rebased.capture_id,
                    mechanism_source=refreshed_request.mechanism_source,
                    prepared_dsl=refreshed_request.prepared_dsl,
                    local_observables=refreshed_request.local_observables,
                    setup_snapshot=remaining_setup,
                )
                self._candidate = remaining_candidate
                window.set_candidate(remaining_candidate)
        except Exception as exc:
            self._candidate = None
            self._set_status(
                "Fit values were applied, but local setup could not rebase: "
                f"{exc}"
            )
            return True
        if unapplied_local_items:
            names = ", ".join(unapplied_local_items)
            canonical_message = (
                "Canonical Parameter values applied: "
                + ", ".join(applied_canonical_parameter_names)
                + "."
                if applied_canonical_parameter_names
                else "Parameters had no canonical authority to publish."
            )
            self._set_status(
                canonical_message
                + " Fit-local terminal scalars were not applied: "
                + names
                + ". Apply Observable Definitions to publish them."
            )
        elif remaining_candidate is not None:
            self._set_status(
                "Selected Fit scopes applied; remaining terminal Fit "
                "values are still available."
            )
        else:
            self._set_status(
                "Fit candidate applied."
                if changed
                else "Selected Fit values already matched the project; "
                "nothing was invalidated."
            )
        return True

    def set_project_defaults(self, defaults: FitDefaults) -> bool:
        self._require_thread()
        if self._application_close_fenced:
            return False
        if not isinstance(defaults, FitDefaults):
            raise TypeError("defaults must be FitDefaults.")
        try:
            delta = self._registry.replace_document(
                replace(self._session.document, fit_defaults=defaults)
            )
            if delta is not None:
                for affected_tab_id, set_ids in (
                    delta.affected_targets_by_tab
                ):
                    self._deauthorize_targets(
                        affected_tab_id,
                        set_ids,
                    )
        except Exception as exc:
            self._set_status(
                f"Project Fit defaults update failed: {exc}"
            )
            return False
        if delta is not None:
            self._set_status("Current-project Fit defaults updated.")
        return delta is not None

    def close_for_project_replacement(self, deadline: float) -> bool:
        """Blocking, resumable cleanup solely for project replacement."""

        return self._close_and_destroy_for_project_replacement(float(deadline))

    def fence_application_close_admission(self) -> None:
        """Permanently reject every later Fit admission and continuation."""

        self._require_thread()
        if self._application_close_fenced:
            return
        self._application_close_fenced = True
        self._publication_fenced_generation = self._worker_generation
        self._candidate = None
        self._local_request = None
        self._local_setup_snapshot = None
        self._pending_detailed_balance = None
        if self._window is not None:
            self._window.clear_candidate()

    def cancel_for_application_close(self) -> None:
        """Fence publication and request cancellation without waiting."""

        self._require_thread()
        self.fence_application_close_admission()
        worker = self._worker
        if worker is not None:
            # GlobalFitWorker.cancel wakes both pause and correction
            # conditions in addition to requesting interruption.
            worker.cancel()

    def _request_window_close(self, _deadline: float) -> bool:
        self._closing = True
        if self._worker is not None:
            self._pending_detailed_balance = None
            self._publication_fenced_generation = self._worker_generation
            self._worker.cancel()
            self._set_status("Fit cancellation requested; the window will close when the worker stops.")
            return False
        self._discard_local()
        return True

    def _close_and_destroy_for_project_replacement(
        self,
        deadline: float,
    ) -> bool:
        self._require_thread()
        window = self._window
        if window is None:
            return self._settle_worker_for_project_replacement(deadline)
        self._closing = True
        if not self._settle_worker_for_project_replacement(deadline):
            self._closing = False
            return False
        self._discard_local()
        window.force_close_after_cleanup()
        QtCore.QCoreApplication.sendPostedEvents(
            window,
            QtCore.QEvent.Type.DeferredDelete,
        )
        if self._window is not None:
            window.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(
                window,
                QtCore.QEvent.Type.DeferredDelete,
            )
        return self._window is None

    def _settle_worker_for_project_replacement(self, deadline: float) -> bool:
        worker = self._worker
        if worker is None:
            return True
        worker.cancel()
        remaining_ms = max(
            0,
            int((float(deadline) - time.monotonic()) * 1000.0),
        )
        if not worker.wait(remaining_ms):
            return False
        self._worker_generation += 1
        self._finish_worker()
        return True

    def _discard_local(self) -> None:
        self._candidate = None
        self._local_request = None
        self._local_setup_snapshot = None
        self._local_observables = ()
        self._origin = None
        self._pending_detailed_balance = None
        if self._window is not None:
            self._window.clear_candidate()

    @QtCore.Slot()
    def _on_window_destroyed(self, *_args: object) -> None:
        if self._window is None:
            return
        self._window = None
        self._candidate = None
        self._local_request = None
        self._local_setup_snapshot = None
        self._local_observables = ()
        self._origin = None
        self._pending_detailed_balance = None
        self._closing = False
        worker = self._worker
        if worker is not None and worker.isFinished():
            self._worker_generation += 1
            self._finish_worker()
        self.window_changed.emit(False)
        self._set_status("Fit closed; all transient Fit state was discarded.")

    def _on_progress(self, generation: int, value: int, text: str) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
        ):
            return
        if self._window is not None:
            self._window.set_progress(value, text)
        self._set_status(text)

    def _on_best(self, generation: int, update: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
            or not isinstance(update, FitLiveUpdate)
        ):
            return
        if self._window is not None:
            self._window.set_live_best(update)

    def _on_auto_modes(self, generation: int, modes: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
            or not isinstance(modes, tuple)
        ):
            return
        normalized = tuple(
            (str(dataset_id), str(mode))
            for dataset_id, mode in modes
        )
        if any(mode not in {"monotone", "time_guided"} for _item, mode in normalized):
            self._worker.cancel()
            self._set_status("Fit emitted an invalid resolved Auto map.")
            return
        if self._window is not None:
            self._window.set_resolved_auto_modes(normalized)

    def _on_live_curves(self, generation: int, curves: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
            or not isinstance(curves, tuple)
        ):
            return
        if self._window is not None:
            self._window.set_live_curves(curves)

    def _on_run_identity(self, generation: int, identity: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
            or not isinstance(identity, FitRunIdentity)
        ):
            return
        if self._window is not None:
            self._window.set_run_identity(identity)

    def _on_succeeded(self, generation: int, candidate: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
        ):
            return
        if not isinstance(candidate, FitCandidate):
            self._on_failed(generation, "Fit returned an invalid candidate.")
            return
        if (
            self._origin is None
            or candidate.origin_capture_id != self._origin.capture_id
        ):
            self._on_failed(generation, "Fit candidate no longer matches this origin capture.")
            return
        self._candidate = candidate
        rendered = True
        if self._window is not None:
            rendered = self._window.set_candidate(candidate)
        self._set_status(
            "Fit completed; candidate is ready to Apply."
            if rendered
            else "Fit completed; plotting failed, but the candidate remains ready to Apply."
        )

    def _on_failed(self, generation: int, message: str) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
        ):
            return
        worker = self._worker
        if self._window is not None and worker is not None:
            if worker.resolved_auto_modes:
                self._window.set_resolved_auto_modes(
                    worker.resolved_auto_modes
                )
            if worker.run_identity is not None:
                self._window.set_run_identity(worker.run_identity)
        self._set_status(str(message))

    def _on_detailed_balance_correction_required(
        self,
        generation: int,
        correction: object,
    ) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or generation != self._worker_generation
            or generation == self._publication_fenced_generation
            or self._worker is None
            or self._worker.publication_fenced
        ):
            return
        if not isinstance(correction, DetailedBalanceCorrectionRequest):
            self._worker.cancel()
            self._set_status(
                "Fit emitted an invalid detailed-balance correction request; "
                "Fit aborted."
            )
            return
        execution_request = self._local_request
        setup_snapshot = self._local_setup_snapshot
        expected_request_id = (
            ""
            if execution_request is None
            else FitRunIdentity.build(
                {
                    "kind": "detailed_balance_correction",
                    "schema_version": 1,
                    "origin_capture_id": (
                        execution_request.origin.capture_id
                    ),
                    "tab_id": execution_request.origin.tab_id,
                    "source": (
                        execution_request.origin.mechanism_source.reactions_text
                    ),
                    "temperature_K": float(
                        execution_request.origin.scientific_settings.temperature
                    ),
                    "wegscheider_cyclicity_enabled": bool(
                        execution_request.origin.scientific_settings.wegscheider_policy
                    ),
                    "issue": correction.issue,
                }
            ).short_hash
        )
        if (
            execution_request is None
            or setup_snapshot is None
            or correction.origin.casefold() != "fit"
            or correction.tab_id != execution_request.origin.tab_id
            or correction.request_id != expected_request_id
            or correction.source
            != execution_request.origin.mechanism_source.reactions_text
            or correction.temperature_K
            != float(
                execution_request.origin.scientific_settings.temperature
            )
        ):
            self._worker.submit_detailed_balance_correction(
                correction.request_id,
                None,
                rejection_message=(
                    "Detailed-balance correction request was stale; Fit aborted."
                ),
            )
            self._set_status(
                "Detailed-balance correction request was stale; Fit aborted."
            )
            return
        self._pending_detailed_balance = _PendingFitDetailedBalanceCorrection(
            generation=generation,
            correction=correction,
            execution_request=execution_request,
        )
        if self._window is not None:
            self._window.set_detailed_balance_correction_pending()
        self._set_status(
            "Fit requires a detailed-balance source correction before it can "
            "continue."
        )
        self.detailed_balance_correction_requested.emit(correction)

    def _on_worker_finished(self, generation: int) -> None:
        if generation != self._worker_generation or self._worker is None:
            return
        self._finish_worker()
        if self._closing and self._window is not None:
            self._discard_local()
            self._window.force_close_after_cleanup()

    def _finish_worker(self) -> None:
        worker = self._worker
        self._worker = None
        self._pending_detailed_balance = None
        self._gate.release(self)
        if self._window is not None and not self._application_close_fenced:
            self._window.set_executing(False)
        self.execution_changed.emit(False)
        if worker is not None:
            worker.deleteLater()
            self.owned_qthreads_changed.emit()

    @staticmethod
    def _verify_detailed_balance_resolved(
        prepared_dsl: PreparedDslRequestSet,
    ) -> None:
        """Verify the retained accepted correction resolved every cycle."""

        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )
        from kindred.core.simulator.wegscheider_symbolic import (
            validate_wegscheider_cyclicity_resolved,
        )
        mechanism = prepared_dsl.mechanism_for_execution()
        apply_parameter_algebra_spec_to_mechanism(
            prepared_dsl.parameter_algebra_spec_for_execution(),
            mechanism=mechanism,
            require_mutable=False,
        )
        validate_wegscheider_cyclicity_resolved(mechanism)

    def _origin_is_current(self) -> bool:
        origin = self._origin
        if origin is None:
            return False
        try:
            current = capture_fit_origin(
                self._session.document,
                origin.tab_id,
            )
        except Exception:
            return False
        return bool(
            current.mechanism_source == origin.mechanism_source
            and current.scientific_settings == origin.scientific_settings
            and current.mapped_datasets == origin.mapped_datasets
        )

    def _build_request(
        self,
        raw: Mapping[str, object],
    ) -> FitExecutionRequest:
        request = self._parse_request(raw)
        validated = build_fit_execution_request(
            origin=request.origin,
            mechanism_source=request.mechanism_source,
            prepared_dsl=request.prepared_dsl,
            local_observables=request.local_observables,
            datasets=request.datasets,
            parameters=request.parameters,
            initials=request.initials,
            defaults=request.defaults,
        )
        revalidate_fit_request_inclusions(
            document=self._session.document,
            request=validated,
        )
        return validated

    def _parse_request(
        self,
        raw: Mapping[str, object],
    ) -> FitExecutionRequest:
        origin = self._origin
        if origin is None:
            raise RuntimeError("Fit has no origin.")
        datasets: list[FitDatasetLocalSettings] = []
        for item in self._mapping_items(raw.get("datasets"), "datasets"):
            execution_raw = item.get("execution")
            if not isinstance(execution_raw, Mapping):
                raise ValueError("Each dataset requires complete simulation setup.")
            execution = SimulationExecutionSettings(
                solver=str(execution_raw.get("solver") or ""),
                t_start=_finite_float(execution_raw.get("t_start"), "t start"),
                t_end=_finite_float(execution_raw.get("t_end"), "t end"),
                integration_points=_int(
                    execution_raw.get("integration_points"),
                    "integration points",
                ),
            )
            targets = [
                target
                for target in self._mapping_items(
                    item.get("targets"),
                    "targets",
                )
                if bool(target.get("use"))
            ]
            points = item.get("sample_points")
            datasets.append(
                FitDatasetLocalSettings(
                    dataset_id=str(item.get("dataset_id") or ""),
                    included=bool(item.get("included")),
                    sample_t_min=_finite_float(
                        item.get("sample_t_min"),
                        "sampling t min",
                    ),
                    sample_t_max=_finite_float(
                        item.get("sample_t_max"),
                        "sampling t max",
                    ),
                    sample_points=(
                        None
                        if points is None
                        else _int(points, "sampling Points")
                    ),
                    execution=execution,
                    weight=_finite_float(item.get("weight"), "dataset weight"),
                    target_species=tuple(
                        str(target.get("model_target") or "")
                        for target in targets
                    ),
                    target_weights=tuple(
                        (
                            str(target.get("model_target") or ""),
                            _finite_float(
                                target.get("weight"),
                                "target weight",
                            ),
                        )
                        for target in targets
                    ),
                    observation_targets=tuple(
                        (
                            str(target.get("raw_observation_id") or ""),
                            str(target.get("raw_column") or ""),
                            str(target.get("model_target") or ""),
                        )
                        for target in targets
                    ),
                    pre_boundary_segment_keys=tuple(
                        (
                            str(
                                target.get(
                                    "raw_observation_id"
                                )
                                or ""
                            ),
                            int(identity.get("source_row")),
                            str(identity.get("segment_key") or ""),
                        )
                        for target in targets
                        for identity in self._mapping_items(
                            target.get(
                                "pre_boundary_segment_keys"
                            ),
                            "pre-boundary segment keys",
                        )
                    ),
                    parametric=(
                        None
                        if str(item.get("x_name") or "t") == "t"
                        else FitParametricSettings(
                            dataset_id=str(item.get("dataset_id") or ""),
                            x_name=str(item.get("x_name") or ""),
                            requested_mode={
                                "auto": "auto",
                                "monotone": "monotone",
                                "time-guided": "time_guided",
                            }[
                                str(item.get("x_mapping_mode") or "Auto")
                                .strip()
                                .lower()
                            ],
                            resolved_mode=(
                                "time_guided"
                                if str(item.get("x_mapping_mode") or "")
                                .strip()
                                .lower()
                                == "time-guided"
                                else "monotone"
                            ),
                        )
                    ),
                )
            )
        captured_parameters = {
            item.name: item for item in origin.parameter_defaults
        }
        local_scalars = {
            scalar.name: scalar
            for observable in self._local_observables
            for scalar in observable.scalar_parameters
        }

        def parameter_variable(
            item: Mapping[str, object],
        ) -> FitParameterVariable:
            name = str(item.get("name") or "").strip()
            captured = captured_parameters.get(name)
            scalar = local_scalars.get(name)
            if captured is None and scalar is None:
                raise ValueError(
                    f"Parameter {name!r} is not a captured writable coordinate."
                )
            return FitParameterVariable(
                name=name,
                display_name=(
                    captured.display_name
                    if captured is not None
                    else name
                ),
                initial=_finite_float(
                    item.get("initial"),
                    "parameter initial",
                ),
                minimum=_bound(
                    item.get("minimum"),
                    "parameter minimum",
                ),
                maximum=_bound(
                    item.get("maximum"),
                    "parameter maximum",
                ),
                fit=bool(item.get("fit")),
                log10=bool(item.get("log10")),
                scope=(
                    "per_dataset"
                    if str(item.get("scope") or "").strip().lower()
                    == "per-dataset"
                    else "shared"
                ),
                dataset_values=tuple(
                    (
                        str(value_item.get("dataset_id") or ""),
                        _finite_float(
                            value_item.get("value"),
                            "per-dataset parameter value",
                        ),
                    )
                    for value_item in self._mapping_items(
                        item.get("dataset_values"),
                        "dataset parameter values",
                    )
                ),
                family=(
                    captured.family
                    if captured is not None
                    else "scalar"
                ),
                role=(
                    captured.role
                    if captured is not None
                    else "scalar"
                ),
                unit=(
                    captured.unit
                    if captured is not None
                    else "1"
                ),
            )
        parameters = tuple(
            parameter_variable(item)
            for item in self._mapping_items(raw.get("parameters"), "parameters")
        )
        parameter_by_name = {
            parameter.name: parameter for parameter in parameters
        }
        local_observables = tuple(
            replace(
                observable,
                scalar_parameters=tuple(
                    replace(
                        scalar,
                        value=parameter_by_name[scalar.name].initial,
                        minimum=parameter_by_name[scalar.name].minimum,
                        maximum=parameter_by_name[scalar.name].maximum,
                        scope=parameter_by_name[scalar.name].scope,
                    )
                    for scalar in observable.scalar_parameters
                ),
            )
            for observable in self._local_observables
        )
        initials = tuple(
            FitInitialVariable(
                set_id=str(item.get("set_id") or ""),
                species=str(item.get("species") or ""),
                initial=_finite_float(
                    item.get("initial"),
                    "initial concentration",
                ),
                minimum=_finite_float(
                    item.get("minimum"),
                    "initial minimum",
                ),
                maximum=_finite_float(
                    item.get("maximum"),
                    "initial maximum",
                ),
                fit=bool(item.get("fit")),
                log10=bool(item.get("log10")),
                scope=str(item.get("scope") or "Global").strip().lower(),
            )
            for item in self._mapping_items(raw.get("initials"), "initials")
        )
        included_dataset_ids = {
            item.dataset_id for item in datasets if item.included
        }
        included_set_ids = {
            mapped.set_id
            for mapped in origin.mapped_datasets
            if mapped.dataset_id in included_dataset_ids
        }
        for species in {item.species for item in initials}:
            global_rows = [
                item for item in initials
                if (
                    item.species == species
                    and item.scope == "global"
                    and item.set_id in included_set_ids
                )
            ]
            if len({item.initial for item in global_rows}) > 1:
                raise ValueError(
                    f"Initial {species!r} has Multiple values after Local→Global; "
                    "enter one explicit replacement value or switch back to Local."
                )
        defaults_raw = raw.get("defaults")
        if not isinstance(defaults_raw, Mapping):
            raise ValueError("Optimizer defaults are missing.")
        defaults = FitDefaults(
            method=str(defaults_raw.get("method") or ""),
            max_nfev=_int(defaults_raw.get("max_nfev"), "maximum evaluations"),
            ftol=_finite_float(
                defaults_raw.get("ftol"),
                "function tolerance",
            ),
            xtol=_finite_float(
                defaults_raw.get("xtol"),
                "step tolerance",
            ),
            use_seed=bool(defaults_raw.get("use_seed")),
            seed=_int(defaults_raw.get("seed"), "seed"),
        )
        definitions = tuple(
            DslDerivedObservableDefinition(
                name=item.name,
                expression=item.expression,
                scalar_parameters=tuple(
                    DslDerivedScalarDefinition(
                        name=scalar.name,
                        value=scalar.value,
                        minimum=scalar.minimum,
                        maximum=scalar.maximum,
                        scope=scalar.scope,
                    )
                    for scalar in item.scalar_parameters
                ),
            )
            for item in local_observables
        )
        effective_prepared = (
            origin.prepared_dsl
            if not definitions
            else derive_prepared_dsl_with_observables(
                origin.prepared_dsl,
                definitions,
            )
        )
        return FitExecutionRequest(
            origin=origin,
            mechanism_source=MechanismAuthoringSource.from_full_dsl_text(
                effective_prepared.analysis.normalized_text
            ),
            prepared_dsl=effective_prepared,
            local_observables=local_observables,
            datasets=tuple(datasets),
            parameters=parameters,
            initials=initials,
            defaults=defaults,
        )

    @staticmethod
    def _mapping_items(
        value: object,
        field: str,
    ) -> tuple[Mapping[str, object], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field} must be a sequence.")
        items = tuple(value)
        if any(not isinstance(item, Mapping) for item in items):
            raise ValueError(f"{field} must contain mappings.")
        return items

    @staticmethod
    def _scope_items(scopes: object) -> tuple[str, ...]:
        if isinstance(scopes, (set, frozenset, tuple, list)):
            return tuple(str(item) for item in scopes)
        raise TypeError("Apply scopes must be a set of independent selectors.")

    @staticmethod
    def _candidate_local_scalar_names(
        candidate: FitCandidate,
    ) -> frozenset[str]:
        names = [
            scalar.name
            for observable in candidate.local_observables
            for scalar in observable.scalar_parameters
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "Fit candidate repeats a Fit-local scalar identity."
            )
        return frozenset(names)

    @staticmethod
    def _preserved_state_with_terminal_values(
        preserved: Mapping[str, object],
        candidate: FitCandidate,
    ) -> dict[str, object]:
        """Overlay terminal fitted values without replacing unrelated UI state."""

        terminal_parameters = {
            item.name: item
            for item in candidate.setup_snapshot.parameters
            if item.fit
        }
        if len(terminal_parameters) != sum(
            1 for item in candidate.setup_snapshot.parameters if item.fit
        ):
            raise ValueError(
                "Fit terminal setup repeats a fitted parameter identity."
            )
        parameters: list[object] = []
        for raw_item in list(preserved.get("parameters") or ()):
            if not isinstance(raw_item, Mapping):
                parameters.append(raw_item)
                continue
            item = dict(raw_item)
            terminal = terminal_parameters.get(
                str(item.get("name") or "")
            )
            if terminal is not None:
                item["initial"] = terminal.initial
                item["dataset_values"] = [
                    {
                        "dataset_id": dataset_id,
                        "value": value,
                    }
                    for dataset_id, value in terminal.dataset_values
                ]
            parameters.append(item)

        terminal_initials: dict[tuple[str, str], float] = {}
        seen_sets: set[str] = set()
        for set_id, values in candidate.initial_values_by_set:
            if set_id in seen_sets:
                raise ValueError(
                    f"Fit candidate repeats initial set {set_id!r}."
                )
            seen_sets.add(set_id)
            for species, raw_value in values:
                key = (set_id, species)
                if key in terminal_initials:
                    raise ValueError(
                        f"Fit candidate repeats fitted initial {key!r}."
                    )
                value = float(raw_value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Fitted initial {key!r} must be finite and "
                        "nonnegative."
                    )
                terminal_initials[key] = value
        for initial in candidate.setup_snapshot.initials:
            if not initial.fit or initial.scope != "global":
                continue
            key = (initial.set_id, initial.species)
            previous = terminal_initials.get(key)
            if (
                previous is not None
                and previous != initial.initial
            ):
                raise ValueError(
                    f"Global fitted initial {key!r} disagrees with its "
                    "terminal setup row."
                )
            terminal_initials[key] = initial.initial
        initials: list[object] = []
        for raw_item in list(preserved.get("initials") or ()):
            if not isinstance(raw_item, Mapping):
                initials.append(raw_item)
                continue
            item = dict(raw_item)
            key = (
                str(item.get("set_id") or ""),
                str(item.get("species") or ""),
            )
            if key in terminal_initials:
                item["initial"] = terminal_initials[key]
            initials.append(item)

        merged = dict(preserved)
        merged["parameters"] = parameters
        merged["initials"] = initials
        return merged

    @classmethod
    def _remaining_candidate_after_apply(
        cls,
        candidate: FitCandidate,
        scopes: frozenset[str],
    ) -> FitCandidate | None:
        local_scalar_names = cls._candidate_local_scalar_names(candidate)
        retired_parameter_names: set[str] = set()
        if "parameters" in scopes:
            retired_parameter_names.update(
                name
                for name, _value in candidate.parameters
                if name not in local_scalar_names
            )
        if "observable_definitions" in scopes:
            retired_parameter_names.update(local_scalar_names)
        remaining = replace(
            candidate,
            parameters=tuple(
                (name, value)
                for name, value in candidate.parameters
                if name not in retired_parameter_names
            ),
            scalar_values_by_dataset=tuple(
                (
                    dataset_id,
                    tuple(
                        (name, value)
                        for name, value in values
                        if name not in retired_parameter_names
                    ),
                )
                for dataset_id, values in candidate.scalar_values_by_dataset
            ),
            initial_values_by_set=(
                ()
                if "initial_concentrations" in scopes
                else candidate.initial_values_by_set
            ),
            execution_by_set=(
                ()
                if "simulation_setup" in scopes
                else candidate.execution_by_set
            ),
            local_observables=(
                ()
                if "observable_definitions" in scopes
                else candidate.local_observables
            ),
        )
        return remaining if remaining.available_scopes else None

    @classmethod
    def _validate_terminal_observable_definitions(
        cls,
        candidate: FitCandidate,
    ) -> None:
        parameters = {
            item.name: item
            for item in candidate.setup_snapshot.parameters
        }
        if len(parameters) != len(candidate.setup_snapshot.parameters):
            raise ValueError(
                "Fit terminal setup repeats a parameter identity."
            )
        cls._candidate_local_scalar_names(candidate)
        for observable in candidate.local_observables:
            for scalar in observable.scalar_parameters:
                parameter = parameters.get(scalar.name)
                if parameter is None:
                    raise ValueError(
                        f"Fit-local scalar {scalar.name!r} has no terminal "
                        "parameter row."
                    )
                terminal_value = parameter.initial
                if parameter.fit and parameter.scope == "per_dataset":
                    values = {
                        float(value)
                        for _dataset_id, value in parameter.dataset_values
                    }
                    if len(values) != 1:
                        raise ValueError(
                            "Observable Definitions Apply cannot publish "
                            f"per-dataset fitted scalar {scalar.name!r}: "
                            "its terminal dataset values differ."
                        )
                    terminal_value = values.pop()
                if (
                    not math.isfinite(float(terminal_value))
                    or scalar.value != terminal_value
                    or scalar.minimum != parameter.minimum
                    or scalar.maximum != parameter.maximum
                    or scalar.scope != parameter.scope
                ):
                    raise ValueError(
                        f"Fit-local scalar {scalar.name!r} does not match "
                        "its terminal parameter state."
                    )

    @classmethod
    def _build_applied_tab(
        cls,
        tab: MechanismTabDocument,
        candidate: FitCandidate,
        scopes: frozenset[str],
        origin: FitOriginSnapshot,
    ) -> MechanismTabDocument:
        next_tab = tab
        prepared = next_tab.prepared_dsl
        if not isinstance(prepared, PreparedDslRequestSet):
            raise ValueError("Fit Apply requires the retained accepted mechanism.")
        if (
            "observable_definitions" in scopes
            and candidate.local_observables
        ):
            cls._validate_terminal_observable_definitions(candidate)
            definitions = tuple(
                DslDerivedObservableDefinition(
                    name=item.name,
                    expression=item.expression,
                    scalar_parameters=tuple(
                        DslDerivedScalarDefinition(
                            name=scalar.name,
                            value=scalar.value,
                            minimum=scalar.minimum,
                            maximum=scalar.maximum,
                            scope=scalar.scope,
                        )
                        for scalar in item.scalar_parameters
                    ),
                )
                for item in candidate.local_observables
            )
            prepared = derive_prepared_dsl_with_observables(
                prepared,
                definitions,
            )
            next_tab = replace(
                next_tab,
                mechanism_source=MechanismAuthoringSource.from_full_dsl_text(
                    prepared.analysis.normalized_text
                ),
                prepared_dsl=prepared,
            )
        if "parameters" in scopes:
            wegscheider_policy = bool(
                next_tab.scientific_settings.wegscheider_policy
            )
            local_scalar_names = cls._candidate_local_scalar_names(
                candidate
            )
            setup_parameters = candidate.setup_snapshot.parameters
            expected = {
                item.name: item
                for item in setup_parameters
            }
            if len(expected) != len(setup_parameters):
                raise ValueError(
                    "Fit terminal setup repeats a parameter identity."
                )
            coordinates = {
                item.name: item
                for item in enumerate_parameter_coordinates(
                    prepared,
                    wegscheider_cyclicity_enabled=wegscheider_policy,
                )
            }
            writable_values = {
                name: float(value)
                for name, value in candidate.parameters
                if name not in local_scalar_names
            }
            for name in writable_values:
                coordinate = coordinates.get(name)
                if coordinate is None:
                    raise ValueError(
                        f"Fitted parameter {name!r} is no longer writable."
                    )
                captured = expected.get(name)
                if (
                    captured is None
                    or coordinate.family != captured.family
                    or coordinate.role != captured.role
                    or not coordinate.writable
                    or coordinate.derived
                ):
                    raise ValueError(
                        f"Fitted parameter {name!r} changed typed family, "
                        "origin, or write ownership."
                    )
            updates = tuple(
                DslParameterCoordinateUpdate(
                    name=name,
                    value=value,
                )
                for name, value in sorted(writable_values.items())
            )
            if updates:
                prior_identity = prepared.analysis.identity
                prepared = derive_prepared_dsl_with_parameter_updates(
                    prepared,
                    updates,
                    wegscheider_cyclicity_enabled=wegscheider_policy,
                )
                derived_from = str(
                    prepared.analysis.structure.get(
                        "derived_from_analysis_identity"
                    )
                    or ""
                )
                applied_updates = tuple(
                    (
                        str(item.get("name") or ""),
                        float(item.get("value")),
                    )
                    for item in prepared.analysis.structure.get(
                        "parameter_updates",
                        (),
                    )
                    if isinstance(item, Mapping)
                )
                expected_updates = tuple(
                    (item.name, item.value) for item in updates
                )
                if (
                    derived_from != prior_identity
                    or prepared.analysis.identity == prior_identity
                    or applied_updates != expected_updates
                ):
                    raise RuntimeError(
                        "Typed Fit parameter Apply lost its exact origin or "
                        "derived analysis identity."
                    )
                derived_source = MechanismAuthoringSource.from_full_dsl_text(
                    prepared.analysis.normalized_text
                )
                if (
                    derived_source.full_dsl.strip()
                    != prepared.analysis.normalized_text
                ):
                    raise RuntimeError(
                        "Typed Fit parameter Apply lost its normalized "
                        "authoring source."
                    )
                next_tab = replace(
                    next_tab,
                    mechanism_source=derived_source,
                    prepared_dsl=prepared,
                )
        if "initial_concentrations" in scopes:
            included_dataset_ids = {
                item.dataset_id
                for item in candidate.setup_snapshot.datasets
                if item.included
            }
            included_set_ids = {
                mapped.set_id
                for mapped in origin.mapped_datasets
                if mapped.dataset_id in included_dataset_ids
            }
            expected_key_rows = [
                (item.set_id, item.species)
                for item in candidate.setup_snapshot.initials
                if item.fit and item.set_id in included_set_ids
            ]
            expected_keys = set(expected_key_rows)
            if len(expected_key_rows) != len(expected_keys):
                raise ValueError(
                    "Fit terminal setup repeats an expected fitted initial "
                    "identity."
                )
            actual_values: dict[tuple[str, str], float] = {}
            seen_set_ids: set[str] = set()
            for set_id, values in candidate.initial_values_by_set:
                if set_id in seen_set_ids:
                    raise ValueError(
                        f"Fit candidate repeats initial set {set_id!r}."
                    )
                seen_set_ids.add(set_id)
                seen_species: set[str] = set()
                for species, raw_value in values:
                    if species in seen_species:
                        raise ValueError(
                            f"Fit candidate repeats initial "
                            f"{(set_id, species)!r}."
                        )
                    seen_species.add(species)
                    value = float(raw_value)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"Fitted initial {(set_id, species)!r} must "
                            "be finite and nonnegative."
                        )
                    actual_values[(set_id, species)] = value
            actual_keys = set(actual_values)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                details = []
                if missing:
                    details.append(f"missing={missing!r}")
                if unexpected:
                    details.append(f"unexpected={unexpected!r}")
                raise ValueError(
                    "Fit candidate initial identities do not exactly match "
                    "the fitted terminal rows"
                    + (": " + "; ".join(details) if details else ".")
                )
            current_set_ids = {
                item.set_id for item in next_tab.initial_condition_sets
            }
            missing_sets = sorted(
                {set_id for set_id, _species in expected_keys}
                - current_set_ids
            )
            if missing_sets:
                raise ValueError(
                    "Fit candidate references missing initial-condition "
                    f"sets: {', '.join(missing_sets)}."
                )
            updated: list[InitialConditionSet] = []
            for initial_set in next_tab.initial_condition_sets:
                current_names = [
                    name for name, _value in initial_set.values
                ]
                if len(current_names) != len(set(current_names)):
                    raise ValueError(
                        f"Initial set {initial_set.name!r} repeats a species."
                    )
                if any(
                    not math.isfinite(float(value)) or float(value) < 0.0
                    for _name, value in initial_set.values
                ):
                    raise ValueError(
                        f"Initial set {initial_set.name!r} must be finite "
                        "and nonnegative before Fit values are merged."
                    )
                fitted_names = {
                    species
                    for set_id, species in expected_keys
                    if set_id == initial_set.set_id
                }
                if not fitted_names:
                    updated.append(initial_set)
                    continue
                missing_species = sorted(
                    fitted_names - set(current_names)
                )
                if missing_species:
                    raise ValueError(
                        f"Initial set {initial_set.name!r} no longer contains "
                        f"fitted species: {', '.join(missing_species)}."
                    )
                merged = tuple(
                    (
                        name,
                        actual_values.get(
                            (initial_set.set_id, name),
                            float(current_value),
                        ),
                    )
                    for name, current_value in initial_set.values
                )
                if any(
                    not math.isfinite(float(value)) or float(value) < 0.0
                    for _name, value in merged
                ):
                    raise ValueError(
                        f"Merged initial set {initial_set.name!r} must be "
                        "finite and nonnegative."
                    )
                updated.append(
                    initial_set.with_values(merged)
                )
            next_tab = next_tab.with_initial_condition_sets(updated)
        if "simulation_setup" in scopes:
            for set_id, execution in candidate.execution_by_set:
                if (
                    set_id not in dict(next_tab.run_overrides_by_set_id)
                    and execution == next_tab.resolved_execution_settings(set_id)
                ):
                    continue
                next_tab = next_tab.with_run_override(set_id, execution)
        return next_tab

    def _set_status(self, text: str) -> None:
        if self._application_close_fenced:
            return
        message = str(text)
        origin = self._origin
        if origin is not None:
            for mapped in sorted(
                origin.mapped_datasets,
                key=lambda item: max(
                    len(item.dataset_id),
                    len(item.set_id),
                ),
                reverse=True,
            ):
                message = message.replace(
                    mapped.dataset_id,
                    mapped.display_name,
                )
                message = message.replace(
                    mapped.set_id,
                    mapped.set_name,
                )
        if self._window is not None:
            try:
                self._window.set_status(message)
            except RuntimeError:
                pass
        self.status_changed.emit(message)

    def _require_thread(self) -> None:
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError(
                "FitSessionOwner may be used only from its GUI thread."
            )
