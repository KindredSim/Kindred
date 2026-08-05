"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    FitDatasetSetupEdit,
    FitDatasetWeightModeSetupEdit,
    FitDefaultsSetupEdit,
    FitExecutionRequest,
    FitInitialSetupEdit,
    FitOriginSnapshot,
    FitParameterSetupEdit,
    FitParameterVariable,
    FitSetupSnapshot,
    build_fit_execution_request,
    capture_fit_origin,
    create_initial_fit_setup,
    fit_origin_science_matches,
    revalidate_fit_request_inclusions,
)
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitLiveUpdate,
    FitWorkerTerminal,
    GlobalFitWorker,
)
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionRequest,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    projected_species_names,
)
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry

__all__ = [
    "CanonicalDetailedBalanceSourceCommit",
    "FitSessionOwner",
]

WindowFactory = Callable[..., FittingWindow]
ColorResolver = Callable[[str, Sequence[str]], object]
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
        self._candidate_available_scopes: frozenset[str] = frozenset()
        self._candidate_unavailable_reasons: dict[str, str] = {}
        self._origin: FitOriginSnapshot | None = None
        self._local_observables: tuple[FitLocalObservable, ...] = ()
        self._local_request: FitExecutionRequest | None = None
        self._local_setup_snapshot: FitSetupSnapshot | None = None
        self._initial_setup_snapshot: FitSetupSnapshot | None = None
        self._last_fit_setup_snapshot: FitSetupSnapshot | None = None
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
            requested_id = str(tab_id or "").strip()
            if (
                requested_id
                and self._origin is not None
                and requested_id != self._origin.tab_id
            ):
                try:
                    requested_name = self._session.document.tab(
                        requested_id
                    ).name
                except Exception:
                    requested_name = requested_id
                window.show_origin_conflict(requested_name)
            else:
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
            modeled_species = tuple(
                projected_species_names(
                    self._session.document.tab(requested_id)
                )
            )
            fit_color_resolver = (
                None
                if self._color_resolver is None
                else lambda species: self._color_resolver(
                    str(species),
                    modeled_species,
                )
            )
            window = self._window_factory(
                origin,
                close_handler=self._request_window_close,
                color_resolver=fit_color_resolver,
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
        initial_setup = create_initial_fit_setup(origin)
        self._local_setup_snapshot = initial_setup
        self._initial_setup_snapshot = initial_setup
        self._last_fit_setup_snapshot = None
        self._pending_detailed_balance = None
        self._candidate = None
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
        try:
            window.rebase(origin, setup=initial_setup)
            window.install_initial_snapshot(initial_setup)
        except (TypeError, ValueError, KeyError) as exc:
            window.deleteLater()
            self._window = None
            self._origin = None
            self._local_setup_snapshot = None
            self._initial_setup_snapshot = None
            self._last_fit_setup_snapshot = None
            self._set_status(f"Fit Open failed: {exc}")
            return None
        window.executeRequested.connect(self.execute)
        window.cancelRequested.connect(self.cancel)
        window.pauseRequested.connect(self.pause)
        window.resumeRequested.connect(self.resume)
        window.applyRequested.connect(self.apply_candidate)
        window.addObservableRequested.connect(self.add_local_observable)
        window.setupEditsRequested.connect(self._on_setup_edits)
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
        if window.has_invalid_setup_edits:
            self._set_status(
                "Fit rejected: correct the invalid setup edit first."
            )
            return False
        try:
            request = self._build_request()
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
        self._worker_generation += 1
        generation = self._worker_generation
        self._publication_fenced_generation = None
        connection_specs = (
            (
                worker.progress,
                lambda value, text, g=generation: self._on_progress(
                    g,
                    value,
                    text,
                ),
            ),
            (
                worker.auto_modes_resolved,
                lambda modes, g=generation: self._on_auto_modes(g, modes),
            ),
            (
                worker.best_updated,
                lambda update, g=generation: self._on_best(g, update),
            ),
            (
                worker.curves_updated,
                lambda curves, g=generation: self._on_live_curves(g, curves),
            ),
            (
                worker.detailed_balance_correction_required,
                lambda correction, g=generation: (
                    self._on_detailed_balance_correction_required(
                        g,
                        correction,
                    )
                ),
            ),
            (
                worker.finished,
                lambda g=generation, w=worker: self._on_worker_finished(g, w),
            ),
        )
        connected_signals: list[tuple[object, object]] = []
        start_attempted = False
        try:
            for signal, callback in connection_specs:
                signal.connect(callback)
                connected_signals.append((signal, callback))
            self._worker = worker
            start_attempted = True
            worker.start()
        except Exception as exc:
            running: bool | None = False if not start_attempted else None
            finished: bool | None = False if not start_attempted else None
            if start_attempted:
                try:
                    running = bool(worker.isRunning())
                except Exception:
                    running = None
                try:
                    finished = bool(worker.isFinished())
                except Exception:
                    finished = None
            verified_unstarted = running is False and finished is False
            if verified_unstarted:
                if self._worker is worker:
                    self._worker = None
                for signal, callback in reversed(connected_signals):
                    try:
                        signal.disconnect(callback)
                    except (RuntimeError, TypeError):
                        pass
                self._publication_fenced_generation = generation
                self._gate.release(self)
                worker.deleteLater()
                window.set_executing(False)
                self._set_status(
                    f"Fit dispatch failed before QThread start: {exc}"
                )
                return False

            if running is True or finished is True:
                self._install_started_request_state(
                    request=request,
                    setup_snapshot=setup_snapshot,
                    window=window,
                )
                custody_state = "running" if running is True else "finished"
            else:
                custody_state = "unknown"
            self.owned_qthreads_changed.emit()
            window.set_executing(True)
            self.execution_changed.emit(True)
            self._set_status(
                "Fit QThread start raised; native state is "
                f"{custody_state}. Retaining worker and computation gate for "
                f"native terminal cleanup: {exc}"
            )
            return True
        self._install_started_request_state(
            request=request,
            setup_snapshot=setup_snapshot,
            window=window,
        )
        self.owned_qthreads_changed.emit()
        window.set_executing(True)
        self.execution_changed.emit(True)
        self._set_status("Fit started.")
        return True

    def _install_started_request_state(
        self,
        *,
        request: FitExecutionRequest,
        setup_snapshot: FitSetupSnapshot,
        window: FittingWindow,
    ) -> None:
        """Replace prior local/candidate state only after native start is known."""

        self._local_observables = request.local_observables
        self._local_request = request
        self._local_setup_snapshot = setup_snapshot
        self._pending_detailed_balance = None
        self._candidate = None
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
        window.clear_candidate()

    @QtCore.Slot(object)
    def _on_setup_edits(self, intents: object) -> None:
        """Fold one or more typed UI intents into the sole local setup owner."""

        window = self._window
        origin = self._origin
        setup = self._local_setup_snapshot
        if window is None or origin is None or setup is None:
            return
        if not isinstance(intents, tuple) or not intents:
            window.set_status("Fit setup edits must be a nonempty tuple.")
            return
        try:
            if self._worker is not None:
                raise ValueError("Fit setup is locked while execution is active.")
            for intent in intents:
                intent_tab_id = str(getattr(intent, "origin_tab_id", ""))
                intent_capture_id = str(
                    getattr(intent, "origin_capture_id", "")
                )
                if (
                    intent_tab_id != origin.tab_id
                    or intent_capture_id != origin.capture_id
                ):
                    raise ValueError(
                        "Fit setup edit no longer matches its origin."
                    )
                if isinstance(intent, FitDatasetSetupEdit):
                    replacement = intent.dataset
                    existing_ids = [item.dataset_id for item in setup.datasets]
                    if replacement.dataset_id not in existing_ids:
                        raise ValueError(
                            "Fit dataset edit changed stable identity."
                        )
                    captured = next(
                        item
                        for item in origin.mapped_datasets
                        if item.dataset_id == replacement.dataset_id
                    )
                    allowed_raw_ids = {
                        item.raw_observation_id for item in captured.observations
                    }
                    edited_raw_ids = {
                        raw_id
                        for raw_id, _model in replacement.observation_targets
                    }
                    if not edited_raw_ids.issubset(allowed_raw_ids):
                        raise ValueError(
                            "Fit dataset edit references an uncaptured raw observation ID."
                        )
                    if (
                        intent.raw_observation_id is not None
                        and intent.raw_observation_id not in allowed_raw_ids
                    ):
                        raise ValueError(
                            "Fit observation edit changed raw observation identity."
                        )
                    setup = replace(
                        setup,
                        datasets=tuple(
                            replacement
                            if item.dataset_id == replacement.dataset_id
                            else item
                            for item in setup.datasets
                        ),
                    )
                elif isinstance(intent, FitParameterSetupEdit):
                    replacement = intent.parameter
                    existing_names = [item.name for item in setup.parameters]
                    if replacement.name not in existing_names:
                        raise ValueError(
                            "Fit parameter edit changed stable identity."
                        )
                    parameters = tuple(
                        replacement if item.name == replacement.name else item
                        for item in setup.parameters
                    )
                    by_name = {item.name: item for item in parameters}
                    local_observables = tuple(
                        replace(
                            observable,
                            scalar_parameters=tuple(
                                replace(
                                    scalar,
                                    value=by_name[scalar.name].initial,
                                    minimum=by_name[scalar.name].minimum,
                                    maximum=by_name[scalar.name].maximum,
                                    scope=by_name[scalar.name].scope,
                                )
                                for scalar in observable.scalar_parameters
                            ),
                        )
                        for observable in setup.local_observables
                    )
                    setup = self._setup_with_local_observables(
                        replace(setup, parameters=parameters),
                        local_observables,
                    )
                elif isinstance(intent, FitInitialSetupEdit):
                    replacement = intent.initial
                    key = (replacement.set_id, replacement.species)
                    existing_keys = {
                        (item.set_id, item.species) for item in setup.initials
                    }
                    if key not in existing_keys:
                        raise ValueError(
                            "Fit initial edit changed stable identity."
                        )
                    setup = replace(
                        setup,
                        initials=tuple(
                            replacement
                            if (item.set_id, item.species) == key
                            else item
                            for item in setup.initials
                        ),
                    )
                elif isinstance(intent, FitDefaultsSetupEdit):
                    setup = replace(setup, defaults=intent.defaults)
                elif isinstance(intent, FitDatasetWeightModeSetupEdit):
                    setup = replace(
                        setup,
                        dataset_weight_mode=intent.dataset_weight_mode,
                    )
                else:
                    raise TypeError(
                        "Fit window emitted an unknown setup edit intent."
                    )
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            window.reject_setup_edits(
                intents,
                f"Fit setup edit rejected: {exc}",
            )
            return
        self._local_setup_snapshot = setup
        self._local_observables = setup.local_observables
        self._local_request = None
        window.accept_setup_edits(setup, intents)

    def _setup_with_local_observables(
        self,
        setup: FitSetupSnapshot,
        observables: tuple[FitLocalObservable, ...],
    ) -> FitSetupSnapshot:
        origin = self._origin
        if origin is None:
            raise RuntimeError("Fit has no origin.")
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
            for item in observables
        )
        effective_prepared = (
            origin.prepared_dsl
            if not definitions
            else derive_prepared_dsl_with_observables(
                origin.prepared_dsl,
                definitions,
            )
        )
        return replace(
            setup,
            mechanism_source=MechanismAuthoringSource.from_parts(
                reactions_text=effective_prepared.analysis.normalized_text
            ),
            prepared_dsl=effective_prepared,
            local_observables=observables,
        )

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
            if (
                self._pending_detailed_balance is not pending
                or self._worker is not worker
                or not self._gate.is_owned_by(self)
                or worker.publication_fenced
            ):
                raise RuntimeError(
                    "Detailed-balance correction no longer owns the exact "
                    "waiting Fit admission."
                )
            canonical_prepared = self._commit_detailed_balance_source(
                request.tab_id,
                corrected_canonical,
            )
            if (
                self._application_close_fenced
                or self._closing
                or self._pending_detailed_balance is not pending
                or self._worker is not worker
                or not self._gate.is_owned_by(self)
                or worker.publication_fenced
            ):
                raise RuntimeError(
                    "Detailed-balance correction was retired during its "
                    "canonical commit."
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
                or rebased.fit_defaults != previous_origin.fit_defaults
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
            corrected_effective = MechanismAuthoringSource.from_parts(
                reactions_text=corrected_prepared.analysis.normalized_text
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
            changed_parameter_ownership = sorted(
                parameter.name
                for parameter in pending.execution_request.parameters
                if (
                    (coordinate := writable_coordinates[parameter.name]).family
                    != parameter.family
                    or coordinate.role != parameter.role
                    or not coordinate.writable
                    or coordinate.derived
                )
            )
            if changed_parameter_ownership:
                raise RuntimeError(
                    "Detailed-balance correction changed Fit coordinate "
                    "ownership: " + ", ".join(changed_parameter_ownership)
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
                coordinate = writable_coordinates[canonical_default.name]
                if prior is None:
                    canonical_default = replace(
                        canonical_default,
                        dataset_values=tuple(
                            (
                                dataset.dataset_id,
                                canonical_default.initial,
                            )
                            for dataset in pending.execution_request.datasets
                        ),
                    )
                if prior is not None and not prior.fit:
                    corrected_dataset_values = tuple(
                        (dataset_id, canonical_default.initial)
                        for dataset_id, _value in prior.dataset_values
                    )
                    corrected_parameter_items.append(
                        replace(
                            prior,
                            initial=canonical_default.initial,
                            dataset_values=corrected_dataset_values,
                            display_name=coordinate.display_name,
                            family=coordinate.family,
                            role=coordinate.role,
                            unit=coordinate.unit,
                        )
                    )
                    continue
                corrected_parameter_items.append(
                    canonical_default
                    if prior is None
                    else replace(
                        prior,
                        display_name=coordinate.display_name,
                        family=coordinate.family,
                        role=coordinate.role,
                        unit=coordinate.unit,
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
                dataset_weight_mode=(
                    pending.execution_request.dataset_weight_mode
                ),
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
        self._initial_setup_snapshot = corrected_snapshot
        self._last_fit_setup_snapshot = None
        if self._window is not None:
            self._window.rebase(
                rebased,
                setup=corrected_snapshot,
            )
            self._window.commit_current_target_selection()
            self._window.install_initial_snapshot(corrected_snapshot)
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
        raw_observation_id: str,
        scalar_scope: str,
    ) -> bool:
        """Validate and retain a local observable without project mutation."""
        self._require_thread()
        if self._application_close_fenced:
            return False
        window = self._window
        origin = self._origin
        setup = self._local_setup_snapshot
        if (
            window is None
            or origin is None
            or setup is None
            or self._worker is not None
        ):
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
            if scalar_scope not in {"shared", "per_dataset"}:
                raise ValueError(
                    "Unknown scalar scope must be shared or per_dataset."
                )
            observable = validate_fit_local_observable(
                name=name,
                expression=expression,
                prepared_dsl=prior_prepared,
                additional_known_observables=tuple(
                    item.name for item in self._local_observables
                ),
                unknown_scalar_scope=scalar_scope,
            )
            mapped = next(
                item
                for item in origin.mapped_datasets
                if item.dataset_id == str(dataset_id)
            )
            if not any(
                item.raw_observation_id == str(raw_observation_id)
                for item in mapped.observations
            ):
                raise ValueError(
                    "Selected raw observation ID is not captured."
                )
            local = next(
                item
                for item in setup.datasets
                if item.dataset_id == mapped.dataset_id
            )
            targets_by_raw_id = dict(local.observation_targets)
            previous_target = targets_by_raw_id.get(str(raw_observation_id))
            targets_by_raw_id[str(raw_observation_id)] = observable.name
            weight_by_target = dict(local.target_weights)
            previous_weight = weight_by_target.get(previous_target or "", 1.0)
            selected_raw_ids = set(targets_by_raw_id)
            ordered_targets = tuple(
                (
                    observation.raw_observation_id,
                    targets_by_raw_id[observation.raw_observation_id],
                )
                for observation in mapped.observations
                if observation.raw_observation_id in selected_raw_ids
            )
            updated_local = replace(
                local,
                target_species=tuple(
                    model_target for _raw_id, model_target in ordered_targets
                ),
                target_weights=tuple(
                    (
                        model_target,
                        previous_weight
                        if raw_id == str(raw_observation_id)
                        else weight_by_target.get(model_target, 1.0),
                    )
                    for raw_id, model_target in ordered_targets
                ),
                observation_targets=ordered_targets,
            )
            retained_dataset_ids = tuple(
                item.dataset_id for item in setup.datasets
            )
            supplemental_parameters = tuple(
                FitParameterVariable(
                    name=scalar.name,
                    display_name=scalar.name,
                    initial=scalar.value,
                    minimum=scalar.minimum,
                    maximum=scalar.maximum,
                    fit=True,
                    log10=False,
                    scope=scalar.scope,
                    dataset_values=tuple(
                        (retained_id, scalar.value)
                        for retained_id in retained_dataset_ids
                    ),
                    family="scalar",
                    role="scalar",
                    unit="1",
                )
                for scalar in observable.scalar_parameters
            )
            existing_parameter_names = {
                item.name for item in setup.parameters
            }
            if any(
                item.name in existing_parameter_names
                for item in supplemental_parameters
            ):
                raise ValueError(
                    "Fit-local observable scalar identity already exists."
                )
            local_observables = (*setup.local_observables, observable)
            setup = self._setup_with_local_observables(
                replace(
                    setup,
                    datasets=tuple(
                        updated_local
                        if item.dataset_id == updated_local.dataset_id
                        else item
                        for item in setup.datasets
                    ),
                    parameters=(*setup.parameters, *supplemental_parameters),
                ),
                tuple(local_observables),
            )
            self._local_observables = tuple(local_observables)
            self._local_setup_snapshot = setup
            window.project_setup_snapshot(setup)
            window.commit_current_target_selection()
        except (StopIteration, TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Add Observable rejected: {exc}")
            return False
        self._candidate = None
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
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
        if not any(
            retained is not None and snapshot == retained
            for retained in (
                self._initial_setup_snapshot,
                self._last_fit_setup_snapshot,
            )
        ):
            self._set_status(
                "Fit setup reset rejected: the snapshot is not an owner-retained "
                "Initial or Last Fit snapshot."
            )
            return False
        try:
            window.clear_candidate()
            restored_snapshot = self._setup_with_local_observables(
                snapshot,
                snapshot.local_observables,
            )
            window.restore_snapshot(restored_snapshot)
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup reset failed: {exc}")
            return False
        self._local_observables = restored_snapshot.local_observables
        self._local_request = None
        self._local_setup_snapshot = restored_snapshot
        self._candidate = None
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
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
        available = self._candidate_available_scopes
        if not selected or not selected.issubset(available):
            self._set_status("Apply contains an unavailable or empty scope selection.")
            return False
        if not self._origin_is_current():
            self._set_status("Apply rejected: origin science is no longer current.")
            return False
        origin = self._origin
        if origin is None:
            return False
        if window.has_pending_target_changes:
            self._set_status(
                "Apply rejected: Apply or Revert pending target edits first."
            )
            return False
        preserved = self._local_setup_snapshot
        if preserved is None:
            self._set_status("Apply rejected: Fit has no typed local setup.")
            return False
        expected_token = self._session.capture_content_token()
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
            preserved = self._preserved_setup_with_terminal_values(
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
            result = self._registry.replace_document(
                next_document,
                expected_token=expected_token,
            )
            if result is None:
                self._set_status(
                    "Apply rejected: project content changed before publication."
                )
                return False
            for affected_tab_id, set_ids in (
                result.delta.affected_targets_by_tab
            ):
                self._deauthorize_targets(affected_tab_id, set_ids)
            changed = result.changed
        except Exception as exc:
            self._set_status(
                f"Apply failed at the document boundary: {exc}"
            )
            return False
        self._local_request = None
        self._local_observables = (
            ()
            if remaining_candidate is None
            else remaining_candidate.local_observables
        )
        self.canonical_applied.emit(origin.tab_id, selected)
        try:
            rebased = capture_fit_origin(self._session.document, origin.tab_id)
            self._origin = rebased
            self._rebase_retained_setup_snapshots(rebased)
            refreshed_snapshot = self._rebase_setup_after_apply(
                preserved,
                rebased,
                selected,
                self._local_observables,
            )
            self._local_setup_snapshot = refreshed_snapshot
            window.rebase(
                rebased,
                setup=refreshed_snapshot,
                preserve_snapshots=True,
            )
            window.commit_current_target_selection()
            self._local_observables = refreshed_snapshot.local_observables
            if remaining_candidate is None:
                self._candidate = None
                self._candidate_available_scopes = frozenset()
                self._candidate_unavailable_reasons = {}
                window.clear_candidate()
            else:
                remaining_candidate = replace(
                    remaining_candidate,
                    origin_capture_id=rebased.capture_id,
                    mechanism_source=refreshed_snapshot.mechanism_source,
                    prepared_dsl=refreshed_snapshot.prepared_dsl,
                    local_observables=refreshed_snapshot.local_observables,
                    setup_snapshot=refreshed_snapshot,
                )
                self._candidate = remaining_candidate
                (
                    self._candidate_available_scopes,
                    self._candidate_unavailable_reasons,
                ) = self._candidate_apply_feasibility(remaining_candidate)
                window.set_candidate(
                    remaining_candidate,
                    available_scopes=self._candidate_available_scopes,
                    unavailable_reasons=self._candidate_unavailable_reasons,
                )
        except Exception as exc:
            self._candidate = None
            self._candidate_available_scopes = frozenset()
            self._candidate_unavailable_reasons = {}
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
        expected_token = self._session.capture_content_token()
        try:
            result = self._registry.replace_document(
                replace(self._session.document, fit_defaults=defaults),
                expected_token=expected_token,
            )
            if result is None:
                self._set_status(
                    "Project Fit defaults update rejected: project content "
                    "changed before publication."
                )
                return False
            for affected_tab_id, set_ids in (
                result.delta.affected_targets_by_tab
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
        if result.changed:
            self._set_status("Current-project Fit defaults updated.")
        return result.changed

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

    def cancel_for_application_close(self) -> None:
        """Dispose Fit state and close through the existing owner lifecycle."""

        self._require_thread()
        self.fence_application_close_admission()
        self._closing = True
        window = self._window
        worker = self._worker
        self._discard_local()
        if worker is None:
            if window is not None:
                window.force_close_after_cleanup()
            return
        # GlobalFitWorker.cancel wakes both pause and correction conditions in
        # addition to requesting interruption.  The existing finished route
        # performs the physical close after native settlement.
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
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
        self._local_request = None
        self._local_setup_snapshot = None
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
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
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
        self._local_request = None
        self._local_setup_snapshot = None
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
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
        if (
            execution_request is None
            or setup_snapshot is None
            or correction.origin.casefold() != "fit"
            or correction.tab_id != execution_request.origin.tab_id
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

    def _on_worker_finished(
        self,
        generation: int,
        worker: GlobalFitWorker,
    ) -> None:
        if (
            generation != self._worker_generation
            or self._worker is not worker
        ):
            return
        terminal = worker.terminal_result
        publication_allowed = bool(
            not self._application_close_fenced
            and not self._closing
            and generation != self._publication_fenced_generation
            and not worker.publication_fenced
            and isinstance(terminal, FitWorkerTerminal)
            and terminal.candidate is not None
            and self._origin_is_current()
        )
        candidate = (
            terminal.candidate
            if isinstance(terminal, FitWorkerTerminal)
            else None
        )
        if publication_allowed:
            origin = self._origin
            publication_allowed = bool(
                origin is not None
                and candidate is not None
                and candidate.origin_tab_id == origin.tab_id
                and candidate.origin_capture_id == origin.capture_id
            )
        failure_message = (
            "Fit worker exited without one terminal result."
            if not isinstance(terminal, FitWorkerTerminal)
            else terminal.message
        )
        self._finish_worker(worker)
        if self._closing and self._window is not None:
            if not self._application_close_fenced:
                self._discard_local()
            self._window.force_close_after_cleanup()
            return
        if not publication_allowed or candidate is None:
            if (
                not self._application_close_fenced
                and not self._closing
                and failure_message
            ):
                self._set_status(failure_message)
            return
        self._candidate = candidate
        self._last_fit_setup_snapshot = candidate.setup_snapshot
        (
            self._candidate_available_scopes,
            self._candidate_unavailable_reasons,
        ) = self._candidate_apply_feasibility(candidate)
        rendered = (
            self._window.set_candidate(
                candidate,
                available_scopes=self._candidate_available_scopes,
                unavailable_reasons=self._candidate_unavailable_reasons,
            )
            if self._window
            else False
        )
        self._set_status(
            "Fit completed; candidate is ready to Apply."
            if rendered
            else "Fit completed; plotting failed, but the candidate remains ready to Apply."
        )

    def _finish_worker(
        self,
        expected: GlobalFitWorker | None = None,
    ) -> None:
        worker = self._worker
        if expected is not None and worker is not expected:
            return
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
        if not fit_origin_science_matches(origin, current):
            return False
        if current != origin:
            setup = self._local_setup_snapshot
            self._origin = current
            self._rebase_retained_setup_snapshots(current)
            if setup is not None:
                setup = replace(
                    setup,
                    origin_tab_id=current.tab_id,
                    origin_capture_id=current.capture_id,
                )
                self._local_setup_snapshot = setup
                if self._window is not None:
                    self._window.rebase(
                        current,
                        setup=setup,
                        preserve_snapshots=True,
                    )
        return True

    def _rebase_retained_setup_snapshots(
        self,
        origin: FitOriginSnapshot,
    ) -> None:
        """Keep session-lifetime reset snapshots keyed to a preserving rebase."""

        def rebased(
            snapshot: FitSetupSnapshot | None,
        ) -> FitSetupSnapshot | None:
            if snapshot is None:
                return None
            return replace(
                snapshot,
                origin_tab_id=origin.tab_id,
                origin_capture_id=origin.capture_id,
            )

        self._initial_setup_snapshot = rebased(self._initial_setup_snapshot)
        self._last_fit_setup_snapshot = rebased(self._last_fit_setup_snapshot)

    def _build_request(self) -> FitExecutionRequest:
        origin = self._origin
        setup = self._local_setup_snapshot
        if origin is None or setup is None:
            raise RuntimeError("Fit has no typed local setup.")
        validated = build_fit_execution_request(
            origin=origin,
            mechanism_source=setup.mechanism_source,
            prepared_dsl=setup.prepared_dsl,
            local_observables=setup.local_observables,
            datasets=setup.datasets,
            parameters=setup.parameters,
            initials=setup.initials,
            dataset_weight_mode=setup.dataset_weight_mode,
            defaults=setup.defaults,
        )
        revalidate_fit_request_inclusions(
            document=self._session.document,
            request=validated,
        )
        return validated

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

    def _preserved_setup_with_terminal_values(
        self,
        preserved: FitSetupSnapshot,
        candidate: FitCandidate,
    ) -> FitSetupSnapshot:
        """Overlay terminal values into the owner-held typed setup."""

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
        terminal_initials = {
            (item.set_id, item.species): item
            for item in candidate.setup_snapshot.initials
            if item.fit
        }
        parameters = tuple(
            (
                replace(
                    item,
                    initial=terminal_parameters[item.name].initial,
                    dataset_values=terminal_parameters[
                        item.name
                    ].dataset_values,
                )
                if item.name in terminal_parameters
                else item
            )
            for item in preserved.parameters
        )
        initials = tuple(
            (
                replace(
                    item,
                    initial=terminal_initials[
                        (item.set_id, item.species)
                    ].initial,
                )
                if (item.set_id, item.species) in terminal_initials
                else item
            )
            for item in preserved.initials
        )
        return self._setup_with_local_observables(
            replace(
                preserved,
                parameters=parameters,
                initials=initials,
            ),
            candidate.setup_snapshot.local_observables,
        )

    def _rebase_setup_after_apply(
        self,
        preserved: FitSetupSnapshot,
        origin: FitOriginSnapshot,
        canonicalized_scopes: frozenset[str],
        local_observables: tuple[FitLocalObservable, ...],
    ) -> FitSetupSnapshot:
        """Re-key retained typed setup directly onto the new canonical origin."""

        baseline = create_initial_fit_setup(origin)
        prior_datasets = {item.dataset_id: item for item in preserved.datasets}
        datasets = tuple(
            replace(
                prior,
                execution=(
                    fresh.execution
                    if "simulation_setup" in canonicalized_scopes
                    else prior.execution
                ),
            )
            if (prior := prior_datasets.get(fresh.dataset_id)) is not None
            else fresh
            for fresh in baseline.datasets
        )
        retained_dataset_ids = {item.dataset_id for item in datasets}
        prior_parameters = {
            item.name: item for item in preserved.parameters
        }
        canonical_parameters: list[FitParameterVariable] = []
        for fresh in baseline.parameters:
            prior = prior_parameters.get(fresh.name)
            if prior is None:
                canonical_parameters.append(fresh)
                continue
            canonical_parameters.append(
                replace(
                    prior,
                    display_name=fresh.display_name,
                    initial=(
                        fresh.initial
                        if "parameters" in canonicalized_scopes
                        else prior.initial
                    ),
                    dataset_values=tuple(
                        (dataset_id, value)
                        for dataset_id, value in prior.dataset_values
                        if dataset_id in retained_dataset_ids
                    ),
                    family=fresh.family,
                    role=fresh.role,
                    unit=fresh.unit,
                )
            )
        local_scalar_names = {
            scalar.name
            for observable in local_observables
            for scalar in observable.scalar_parameters
        }
        supplemental = tuple(
            replace(
                item,
                dataset_values=tuple(
                    (dataset_id, value)
                    for dataset_id, value in item.dataset_values
                    if dataset_id in retained_dataset_ids
                ),
            )
            for item in preserved.parameters
            if item.name in local_scalar_names
        )
        prior_initials = {
            (item.set_id, item.species): item for item in preserved.initials
        }
        initials = tuple(
            replace(
                prior,
                initial=(
                    fresh.initial
                    if "initial_concentrations" in canonicalized_scopes
                    else prior.initial
                ),
            )
            if (
                prior := prior_initials.get((fresh.set_id, fresh.species))
            )
            is not None
            else fresh
            for fresh in baseline.initials
        )
        rebased = replace(
            baseline,
            datasets=datasets,
            parameters=(*canonical_parameters, *supplemental),
            initials=initials,
            defaults=preserved.defaults,
            dataset_weight_mode=preserved.dataset_weight_mode,
        )
        return self._setup_with_local_observables(
            rebased,
            local_observables,
        )

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
        return (
            remaining
            if (
                remaining.parameters
                or remaining.initial_values_by_set
                or remaining.execution_by_set
                or remaining.local_observables
            )
            else None
        )

    @classmethod
    def _candidate_apply_feasibility(
        cls,
        candidate: FitCandidate,
    ) -> tuple[frozenset[str], dict[str, str]]:
        """Decide terminal Apply feasibility once in the sole Fit owner."""

        local_scalar_names = cls._candidate_local_scalar_names(candidate)
        available: set[str] = set()
        unavailable: dict[str, str] = {}
        if any(
            name not in local_scalar_names
            for name, _value in candidate.parameters
        ):
            available.add("parameters")
        else:
            unavailable["parameters"] = (
                "Parameters Apply is unavailable because the candidate "
                "contains no fitted project parameters."
            )
        if candidate.initial_values_by_set:
            available.add("initial_concentrations")
        else:
            unavailable["initial_concentrations"] = (
                "Initial Concentrations Apply is unavailable because the "
                "candidate contains no fitted initial-concentration results."
            )
        if candidate.execution_by_set:
            available.add("simulation_setup")
        else:
            unavailable["simulation_setup"] = (
                "Simulation Setup Apply is unavailable because the candidate "
                "contains no execution-setup results."
            )
        if candidate.local_observables:
            try:
                cls._validate_terminal_observable_definitions(candidate)
            except ValueError as exc:
                unavailable["observable_definitions"] = (
                    "Observable Definitions Apply is unavailable because "
                    f"{exc}"
                )
            else:
                available.add("observable_definitions")
        else:
            unavailable["observable_definitions"] = (
                "Observable Definitions Apply is unavailable because the "
                "candidate contains no Fit-local observable definitions."
            )
        return frozenset(available), unavailable

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
                mechanism_source=MechanismAuthoringSource.from_parts(
                    reactions_text=prepared.analysis.normalized_text
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
                derived_source = MechanismAuthoringSource.from_parts(
                    reactions_text=prepared.analysis.normalized_text
                )
                if (
                    derived_source.reactions_text.strip()
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
