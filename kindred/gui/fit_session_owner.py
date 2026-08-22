"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from PySide6 import QtCore, QtWidgets

from kindred.core.algebra.observable_introspection import (
    FitLocalObservable,
    apply_admitted_algebra,
    validate_fit_local_observable,
)
from kindred.core.analysis.global_fitting import SharedRefitPromotion
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    InitialConditionSet,
    MechanismTabDocument,
    canonical_species_names,
)
from kindred.core.project_session import ProjectSession
from kindred.core.species import validate_concentration
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_source_with_parameter_updates,
)
from kindred.core.simulator.wegscheider_symbolic import (
    UnresolvedWegscheiderCyclicityError,
)
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.fitting.launch import (
    FitApplySelection,
    FitDatasetLocalSettings,
    FitExecutionRequest,
    FitInitialApplyCellProposal,
    FitInitialApplyProposal,
    FitLocalScalarProvenance,
    FitOriginSnapshot,
    FitObservationSnapshot,
    FitParameterConfiguration,
    FitParameterVariable,
    FitSetupSnapshot,
    _replace_fit_parameter,
    build_fit_execution_request,
    capture_fit_origin,
    create_initial_fit_setup,
    build_fit_local_admission,
    fit_origin_science_matches,
    revalidate_fit_request_inclusions,
    selected_observations,
)
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.reporting import (
    build_fit_action_projection,
    fit_action_projection_hash,
    fit_action_projection_json,
    fit_action_projection_report_text,
    fit_action_projection_summary,
)
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitLiveUpdate,
    FitWorkerTerminal,
    GlobalFitWorker,
    project_fit_candidate_curves,
)
from kindred.gui.parameter_enumeration import (
    admit_gui_parameter_value,
    enumerate_parameter_coordinates,
)
from kindred.gui.workflow_contracts import (
    DetailedBalanceCorrectionCommitResult,
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionProposal,
    DetailedBalanceCorrectionRequest,
)
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry

__all__ = [
    "FitReplacementDispositionSummary",
    "FitSessionOwner",
]

WindowFactory = Callable[..., FittingWindow]
ColorResolver = Callable[[str, Sequence[str]], object]
CanonicalDetailedBalanceSourceProposal = Callable[
    [str, MechanismAuthoringSource],
    DetailedBalanceCorrectionProposal | None,
]
CanonicalDetailedBalanceSourceCommit = Callable[
    [DetailedBalanceCorrectionProposal],
    DetailedBalanceCorrectionCommitResult,
]


@dataclass(frozen=True, slots=True)
class _PendingFitDetailedBalanceCorrection:
    correction: DetailedBalanceCorrectionRequest
    setup: FitSetupSnapshot


@dataclass(frozen=True, slots=True)
class _SharedRefitRetryState:
    """One atomic owner transaction for a Shared-refit retry.

    The source candidate, promotion selection, and accepted retry setup are
    one custody unit.  Keeping them together prevents a failed/cancelled
    retry from leaving a promotion-bearing setup without its source (or a
    source whose accepted setup has already been replaced).
    """

    source_candidate: FitCandidate
    promotions: tuple[SharedRefitPromotion, ...]
    accepted_setup: FitSetupSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.source_candidate, FitCandidate):
            raise TypeError("Shared-refit retry source must be FitCandidate.")
        promotions = tuple(self.promotions)
        if not promotions or any(
            not isinstance(item, SharedRefitPromotion) for item in promotions
        ):
            raise ValueError("Shared-refit retry requires typed promotions.")
        if not isinstance(self.accepted_setup, FitSetupSnapshot):
            raise TypeError("Shared-refit retry setup must be FitSetupSnapshot.")
        if tuple(self.accepted_setup.shared_refit_promotions) != promotions:
            raise ValueError(
                "Shared-refit retry promotions must match its accepted setup."
            )
        object.__setattr__(self, "promotions", promotions)


@dataclass(frozen=True, slots=True)
class FitReplacementDispositionSummary:
    """Immutable Fit state disclosed to project replacement disposition.

    The summary is derived from :class:`FitSessionOwner` custody only.  It is
    deliberately a projection rather than a second owner: candidate/setup,
    terminal, worker, and Shared-refit source objects remain private to the
    owner.  ``active_worker`` is reported separately from
    ``has_retained_work`` so replacement code can request cancellation first
    and then make a retained-work decision after native settlement.

    ``apply_ready_scopes`` contains only scopes still owned by this Fit
    candidate.  ``apply_ready_initial_keys`` is the remaining exact
    ``(set_id, species)`` identity roster, not a copy of fitted values.
    """

    active_worker: bool = False
    retained_completed_fit: bool = False
    apply_ready_scopes: frozenset[str] = frozenset()
    apply_ready_initial_keys: tuple[tuple[str, str], ...] = ()
    retained_diagnostic_only: bool = False
    retained_nonconverged: bool = False
    retained_terminal: bool = False
    setup_validation_blocked: bool = False
    shared_refit_retry: bool = False
    untouched_initial_setup: bool = False
    terminal_lifecycle_cause: Literal[
        "completed", "user_stop", "cancelled", "failed", "scientific_unavailable"
    ] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "active_worker",
            "retained_completed_fit",
            "retained_diagnostic_only",
            "retained_nonconverged",
            "retained_terminal",
            "setup_validation_blocked",
            "shared_refit_retry",
            "untouched_initial_setup",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool.")
        scopes = frozenset(
            str(scope or "").strip() for scope in self.apply_ready_scopes
        )
        if any(not scope for scope in scopes):
            raise ValueError("Fit Apply-ready scope identities must be nonblank.")
        object.__setattr__(self, "apply_ready_scopes", scopes)
        keys = tuple(
            (str(set_id or "").strip(), str(species or "").strip())
            for set_id, species in self.apply_ready_initial_keys
        )
        if any(not set_id or not species for set_id, species in keys):
            raise ValueError("Fit Apply-ready initial identities must be nonblank.")
        if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise ValueError(
                "Fit Apply-ready initial identities must be unique and sorted."
            )
        object.__setattr__(self, "apply_ready_initial_keys", keys)
        if self.terminal_lifecycle_cause not in {
            None,
            "completed",
            "user_stop",
            "cancelled",
            "failed",
            "scientific_unavailable",
        }:
            raise ValueError("Fit terminal lifecycle cause is invalid.")
        if self.untouched_initial_setup and self.has_retained_work:
            raise ValueError(
                "Untouched initial Fit setup cannot also disclose retained work."
            )

    @property
    def apply_ready(self) -> bool:
        """Whether any independently selectable Fit replacement remains."""

        return bool(self.apply_ready_scopes or self.apply_ready_initial_keys)

    @property
    def retained_candidate(self) -> bool:
        """Whether a terminal candidate, including diagnostic-only, is retained."""

        return bool(self.retained_completed_fit or self.retained_diagnostic_only)

    @property
    def has_retained_work(self) -> bool:
        """Whether replacement must disclose non-active Fit-local custody."""

        return bool(
            self.retained_candidate
            or self.retained_terminal
            or self.setup_validation_blocked
            or self.shared_refit_retry
        )

    @property
    def requires_disposition(self) -> bool:
        """Whether project replacement must present this retained Fit state."""

        return self.has_retained_work


class FitSessionOwner(QtCore.QObject):
    """Retain the window, accepted setup, worker, and candidate custody."""

    window_changed = QtCore.Signal(bool)
    execution_changed = QtCore.Signal(bool)
    canonical_applied = QtCore.Signal(str, object)
    status_changed = QtCore.Signal(str)
    detailed_balance_correction_requested = QtCore.Signal(object)
    application_close_settlement_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        session: ProjectSession,
        registry: WorkspaceTabRegistry,
        gate: ComputationGate,
        propose_detailed_balance_source: CanonicalDetailedBalanceSourceProposal,
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
        if not callable(propose_detailed_balance_source):
            raise TypeError("propose_detailed_balance_source must be callable.")
        if not callable(commit_detailed_balance_source):
            raise TypeError("commit_detailed_balance_source must be callable.")
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        if not callable(window_factory):
            raise TypeError("window_factory must be callable.")
        self._session = session
        self._registry = registry
        self._gate = gate
        self._propose_detailed_balance_source = propose_detailed_balance_source
        self._commit_detailed_balance_source = commit_detailed_balance_source
        self._color_resolver = color_resolver
        self._parent_widget = parent_widget
        self._window_factory = window_factory
        self._window: FittingWindow | None = None
        self._worker: GlobalFitWorker | None = None
        self._candidate: FitCandidate | None = None
        self._setup_state: (
            FitSetupSnapshot | _SharedRefitRetryState | None
        ) = None
        self._candidate_applied_scopes: frozenset[str] = frozenset()
        self._candidate_applied_initial_keys: frozenset[
            tuple[str, str]
        ] = frozenset()
        self._initial_setup_snapshot: FitSetupSnapshot | None = None
        self._last_fit_setup_snapshot: FitSetupSnapshot | None = None
        self._pending_detailed_balance: (
            _PendingFitDetailedBalanceCorrection | None
        ) = None
        self._closing = False
        self._application_close_fenced = False
        self._application_close_settlement_pending = False

    @property
    def project_defaults(self) -> FitDefaults:
        return self._session.document.fit_defaults

    @property
    def replacement_disposition(self) -> FitReplacementDispositionSummary:
        """Return immutable Fit state for project replacement disposition.

        This projection intentionally reads only owner custody and the
        existing window-level setup-validation diagnostic.  It never consults
        Results-tab widgets and never transfers candidate/result ownership to
        the project controller.
        """

        worker_active = self._worker is not None
        retry_state = self._retry_state()
        candidate = self._candidate
        setup_validation_blocked = False
        window = self._window
        if window is not None:
            try:
                setup_validation_blocked = bool(window.has_invalid_setup_edits)
            except (AttributeError, RuntimeError):
                # A Qt window can be in deferred destruction while this
                # read-only projection is queried.  Owner custody remains the
                # authoritative part of the summary in that interval.
                setup_validation_blocked = False

        retained_completed_fit = False
        retained_diagnostic_only = False
        retained_nonconverged = False
        apply_ready_scopes: frozenset[str] = frozenset()
        apply_ready_initial_keys: tuple[tuple[str, str], ...] = ()
        if candidate is not None:
            result = candidate.result
            retained_completed_fit = self._candidate_result_is_apply_eligible(
                candidate
            )
            retained_diagnostic_only = not retained_completed_fit
            retained_nonconverged = not result.optimizer_converged
            if not worker_active:
                apply_ready_scopes, _ = self._candidate_apply_feasibility(candidate)
                if "initial_concentrations" in apply_ready_scopes:
                    try:
                        proposal = self.current_initial_apply_proposal()
                    except (AttributeError, KeyError, TypeError, ValueError):
                        proposal = None
                    if proposal is not None:
                        apply_ready_initial_keys = tuple(
                            sorted(item.key for item in proposal.cells)
                        )

        untouched_initial_setup = False
        initial_setup = self._initial_setup_snapshot
        accepted_setup = self._accepted_setup()
        if (
            self._window is not None
            and not worker_active
            and candidate is None
            and retry_state is None
            and self._worker is None
            and not setup_validation_blocked
            and initial_setup is not None
            and accepted_setup == initial_setup
        ):
            untouched_initial_setup = True

        return FitReplacementDispositionSummary(
            active_worker=worker_active,
            retained_completed_fit=retained_completed_fit,
            apply_ready_scopes=apply_ready_scopes,
            apply_ready_initial_keys=apply_ready_initial_keys,
            retained_diagnostic_only=retained_diagnostic_only,
            retained_nonconverged=retained_nonconverged,
            retained_terminal=False,
            setup_validation_blocked=setup_validation_blocked,
            shared_refit_retry=retry_state is not None,
            untouched_initial_setup=untouched_initial_setup,
            terminal_lifecycle_cause=None,
        )

    @property
    def application_close_settled(self) -> bool:
        """Return whether Close-facing Fit cleanup is fully owner-settled."""

        return bool(
            self._worker is None
            and (
                not (self._application_close_fenced or self._closing)
                or self._window is None
            )
        )

    def _notify_application_close_settled(self) -> None:
        """Publish only the transition reached after complete owner cleanup."""

        if (
            self._application_close_settlement_pending
            and self.application_close_settled
        ):
            self._application_close_settlement_pending = False
            self.application_close_settlement_changed.emit()

    def open_or_focus(self, tab_id: str | None = None) -> FittingWindow | None:
        """Open one modal origin window, or focus the existing singleton."""
        self._require_thread()
        if self._application_close_fenced or self._closing:
            return None
        window = self._window
        if window is not None:
            requested_id = str(tab_id or "").strip()
            if (
                requested_id
                and self._current_origin() is not None
                and requested_id != self._current_origin().tab_id
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
                canonical_species_names(
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
        self._window = window
        initial_setup = create_initial_fit_setup(origin)
        self._install_setup(initial_setup)
        self._initial_setup_snapshot = initial_setup
        self._last_fit_setup_snapshot = None
        self._pending_detailed_balance = None
        self._candidate = None
        self._candidate_applied_scopes = frozenset()
        self._candidate_applied_initial_keys = frozenset()
        try:
            window.rebase(origin, setup=initial_setup)
            self._project_initial_apply_proposal(window)
            window.set_initial_reset_available(initial_setup)
        except (TypeError, ValueError, KeyError) as exc:
            window.deleteLater()
            self._window = None
            self._setup_state = None
            self._initial_setup_snapshot = None
            self._last_fit_setup_snapshot = None
            self._set_status(f"Fit Open failed: {exc}")
            return None
        window.executeRequested.connect(self.execute)
        window.cancelRequested.connect(self.cancel)
        window.pauseRequested.connect(self.pause)
        window.resumeRequested.connect(self.resume)
        window.applyRequested.connect(self.apply_candidate)
        window.copyHashRequested.connect(self.copy_hash)
        window.copyRunJsonRequested.connect(self.copy_run_json)
        window.resultsSummaryRequested.connect(self.results_summary)
        window.sharedRefitRequested.connect(self.accept_shared_refit)
        window.addObservableRequested.connect(self.add_local_observable)
        window.setupSnapshotRequested.connect(self._on_setup_snapshot)
        window.resetRequested.connect(self.reset_setup)
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
        origin = self._current_origin()
        if window is None or origin is None:
            self._set_status("Fit rejected: there is no open Fit window.")
            return False
        setup = self._accepted_setup()
        if setup is None:
            self._set_status("Fit rejected: there is no accepted Fit setup.")
            return False
        if self._worker is not None:
            self._set_status("Fit rejected: this Fit is already executing.")
            return False
        if self._closing or self._application_close_fenced:
            self._set_status("Fit rejected while the window is closing.")
            return False
        if not self._origin_is_current():
            if self._retry_state() is not None:
                self._invalidate_candidate_custody()
                self._retire_shared_refit_retry()
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
        if window.has_pending_data_changes:
            self._set_status(
                "Fit rejected: Apply or Revert the pending Fit data edits first."
            )
            return False
        if window.has_invalid_setup_edits:
            self._set_status(
                "Fit rejected: correct the invalid setup edit first."
            )
            return False
        try:
            request = self._build_request()
        except UnresolvedWegscheiderCyclicityError as exc:
            report = exc.report
            if report is None:
                self._set_status("Fit detailed-balance admission returned no report.")
                return False
            correction = DetailedBalanceCorrectionRequest(
                origin="Fit",
                tab_id=origin.tab_id,
                source=origin.mechanism_source.reactions_text,
                issue=str(exc),
                temperature_K=float(origin.scientific_settings.temperature),
                report=report,
                source_energy_unit=str(origin.prepared_dsl.analysis.energy_unit),
                accepted_analysis=origin.prepared_dsl.analysis,
            )
            self._pending_detailed_balance = _PendingFitDetailedBalanceCorrection(
                correction=correction,
                setup=setup,
            )
            if self._window is not None:
                self._window.set_detailed_balance_correction_pending()
            self.detailed_balance_correction_requested.emit(correction)
            self._set_status(
                "Fit requires a detailed-balance source correction before dispatch."
            )
            return False
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup is invalid: {exc}")
            return False
        retry_state = self._retry_state()
        retry_source = (
            retry_state.source_candidate if retry_state is not None else None
        )
        if not self._gate.try_acquire(self, "fit"):
            self._set_status("Fit rejected: another computation is active.")
            return False
        try:
            worker = GlobalFitWorker(
                request,
                parent=self,
            )
        except Exception as exc:
            self._gate.release(self)
            if retry_source is not None:
                self._restore_shared_refit_source_presentation(
                    retry_source,
                    setup=FitSetupSnapshot.from_request(request),
                )
            self._set_status(f"Fit preparation failed before dispatch: {exc}")
            return False
        start_attempted = False
        try:
            # The worker object is the sole callback fence.  Queued callbacks
            # from a settled/replaced QThread are harmless because every
            # callback checks exact QObject custody before touching state.
            worker.progress.connect(
                lambda value, text, w=worker: self._on_progress(w, value, text)
            )
            worker.auto_modes_resolved.connect(
                lambda modes, w=worker: self._on_auto_modes(w, modes)
            )
            worker.best_updated.connect(
                lambda update, w=worker: self._on_best(w, update)
            )
            worker.curves_updated.connect(
                lambda curves, w=worker: self._on_live_curves(w, curves)
            )
            worker.finished.connect(
                lambda w=worker: self._on_worker_finished(w)
            )
            self._worker = worker
            self._application_close_settlement_pending = True
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
                self._gate.release(self)
                worker.deleteLater()
                self._notify_application_close_settled()
                window.set_executing(False)
                if retry_source is not None:
                    self._restore_shared_refit_source_presentation(
                        retry_source,
                        setup=FitSetupSnapshot.from_request(request),
                    )
                self._set_status(
                    f"Fit dispatch failed before QThread start: {exc}"
                )
                return False

            if running is True or finished is True:
                window.set_executing(True)
                self._install_started_request_state(
                    request=request,
                    window=window,
                )
                custody_state = "running" if running is True else "finished"
            else:
                # Native state is ambiguous: retain the exact request/setup
                # before any queued worker callback can publish against the
                # old local state.  Worker and gate custody remains retained.
                window.set_executing(True)
                self._install_started_request_state(
                    request=request,
                    window=window,
                )
                custody_state = "unknown"
            self.execution_changed.emit(True)
            self._set_status(
                "Fit QThread start raised; native state is "
                f"{custody_state}. Retaining worker and computation gate for "
                f"native terminal cleanup: {exc}"
            )
            return True
        window.set_executing(True)
        self._install_started_request_state(
            request=request,
            window=window,
        )
        self.execution_changed.emit(True)
        self._set_status("Fit started.")
        return True

    def _install_started_request_state(
        self,
        *,
        request: FitExecutionRequest,
        window: FittingWindow,
    ) -> None:
        """Replace prior local/candidate state only after native start is known."""

        started_setup = FitSetupSnapshot.from_request(request)
        retry_state = self._retry_state()
        if request.shared_refit_promotions:
            if retry_state is None:
                raise RuntimeError(
                    "Shared-refit launch lost its accepted retry custody."
                )
            self._install_setup(
                started_setup,
                retry_source=retry_state.source_candidate,
            )
            # Keep the source candidate's partial-Apply markers call-local
            # while the replacement worker runs.  They are restored with the
            # source on every unsuccessful Shared-refit settlement.
            self._candidate = None
        else:
            self._install_setup(started_setup)
            self._invalidate_candidate_custody()
        # Shared-refit source custody is call-local execution input.  It must
        # survive dispatch failure, cancellation, and warning terminals; only
        # a successfully accepted replacement retires it.
        self._pending_detailed_balance = None
        self._clear_candidate_presentation(window)

    def _restore_shared_refit_source_presentation(
        self,
        source: FitCandidate,
        *,
        setup: FitSetupSnapshot,
        failure: object | None = None,
    ) -> None:
        """Restore the retained Per-dataset source after an unsuccessful dispatch."""

        self._install_setup(setup, retry_source=source)
        self._candidate = source
        (
            projection,
            copy_hash,
            report_text,
            report_failure,
            curves,
            curve_failure,
        ) = self._build_presentations(source)
        available, unavailable = self._candidate_apply_feasibility(
            source,
            applied_scopes=self._candidate_applied_scopes,
            applied_initial_keys=self._candidate_applied_initial_keys,
        )
        proposal = self.current_initial_apply_proposal()
        window = self._window
        if window is None:
            return
        try:
            window.project_setup_snapshot(setup)
            window.set_terminal_presentation(
                projection,
                copy_hash=copy_hash,
                report_text=report_text,
                curves=curves,
                report_failure=report_failure,
                curve_failure=curve_failure,
                apply_labels=unavailable,
                enabled_scopes=available,
                initial_proposal=proposal,
                older_retained_state=True,
            )
            window.set_last_fit_reset_available(self._last_fit_setup_snapshot)
            if failure is not None:
                window.set_terminal_failure(failure)
        except Exception as exc:
            self._set_status(
                "Shared refit failed before retry presentation was restored: "
                f"{exc}"
            )

    @QtCore.Slot(object)
    def _on_setup_snapshot(self, snapshot: object) -> bool:
        """Admit one complete leaf-authored setup snapshot."""

        if self._closing or self._application_close_fenced:
            return False
        window = self._window
        origin = self._current_origin()
        previous = self._accepted_setup()
        if window is None or origin is None or previous is None:
            return False
        if not isinstance(snapshot, FitSetupSnapshot):
            window.set_status("Fit setup snapshot is not typed.")
            return False
        if snapshot.origin is not origin:
            window.set_status("Fit setup snapshot no longer matches its origin.")
            return False
        if self._worker is not None:
            window.set_status("Fit setup is locked while execution is active.")
            return False
        retry_state = self._retry_state()
        source = retry_state.source_candidate if retry_state is not None else None
        if snapshot.shared_refit_promotions:
            if snapshot.shared_refit_promotions != previous.shared_refit_promotions:
                source = source or self._candidate
            if source is None or not self._candidate_origin_is_current(source):
                window.set_status(
                    "Shared refit promotion requires one current Per-dataset candidate."
                )
                return False
        else:
            source = None
        if snapshot == previous:
            return False
        self._install_setup(snapshot, retry_source=source)
        had_candidate = self._candidate is not None
        if source is None:
            self._invalidate_candidate_custody()
            if had_candidate:
                self._clear_candidate_presentation(window)
        window.project_setup_snapshot(snapshot)
        if snapshot.shared_refit_promotions:
            self._set_status("Shared refit setup accepted; run Fit to execute it.")
        return True

    @QtCore.Slot(tuple, str)
    def accept_shared_refit(
        self,
        parameter_names: tuple[str, ...],
        seed_statistic: str,
    ) -> bool:
        """Accept one Results selection using the owner's current candidate."""

        self._require_thread()
        setup = self._accepted_setup()
        # After an unsuccessful Shared terminal, the retained Per-dataset
        # candidate is the only valid source for the next promotion intent.
        retry_state = self._retry_state()
        candidate = retry_state.source_candidate if retry_state is not None else self._candidate
        origin = self._current_origin()
        if setup is None or candidate is None or origin is None:
            self._set_status("Shared refit rejected: no terminal Fit source is available.")
            return False
        if not self._candidate_result_is_apply_eligible(candidate):
            self._set_status(
                "Shared refit rejected: source Fit result is not converged and "
                "non-diagnostic."
            )
            return False
        if not self._candidate_origin_is_current(candidate):
            self._retire_shared_refit_retry()
            self._set_status("Shared refit rejected: source science is no longer current.")
            return False
        parameter_by_name = {
            item.name: item for item in setup.parameters
        }
        local_scalar_names = self._candidate_local_scalar_names(candidate)
        promotions: list[SharedRefitPromotion] = []
        try:
            for name in parameter_names:
                parameter = parameter_by_name.get(name)
                if (
                    parameter is None
                    or parameter.scope != "per_dataset"
                    or not parameter.selected_config.fit
                    or name in local_scalar_names
                ):
                    raise ValueError(
                        f"{name!r} is not an eligible current Per-dataset coordinate."
                    )
                config = parameter.selected_config
                promotions.append(
                    SharedRefitPromotion(
                        parameter_name=name,
                        minimum=config.minimum,
                        maximum=config.maximum,
                        log10=config.log10,
                        seed_statistic=seed_statistic,
                    )
                )
        except (TypeError, ValueError) as exc:
            self._set_status(f"Shared refit rejected: {exc}")
            return False
        admitted_promotions = tuple(promotions)
        snapshot = replace(
            setup,
            shared_refit_promotions=admitted_promotions,
            profile_likelihood=None,
            residual_bootstrap=None,
        )
        if not self._on_setup_snapshot(snapshot):
            return False
        retry_state = self._retry_state()
        accepted = (
            retry_state is not None
            and retry_state.source_candidate is candidate
            and retry_state.promotions == admitted_promotions
        )
        if not accepted:
            self._set_status("Shared refit rejected: owner setup admission failed.")
            return False
        # Acceptance and execution are one owner transaction.  The existing
        # execute path retains the source candidate until the new Shared Fit
        # reaches its terminal outcome, including dispatch failures.
        return self.execute()

    def _setup_with_local_observables(
        self,
        setup: FitSetupSnapshot,
        observables: tuple[FitLocalObservable, ...],
        *,
        origin: FitOriginSnapshot | None = None,
    ) -> FitSetupSnapshot:
        base_origin = origin if origin is not None else self._current_origin()
        if base_origin is None:
            raise RuntimeError("Fit has no origin.")
        build_fit_local_admission(
            base_origin.prepared_dsl,
            observables,
            setup.parameters,
        )
        return replace(
            setup,
            local_observables=observables,
            origin=base_origin,
        )

    @QtCore.Slot()
    def cancel(self) -> bool:
        if self._closing or self._application_close_fenced:
            return False
        worker = self._worker
        if worker is None:
            return False
        self._pending_detailed_balance = None
        worker.request_cancel(publish_best=True)
        self._set_status(
            "Fit stop requested; the best fully evaluated valid candidate will "
            "be finalized if one exists."
        )
        return True

    @QtCore.Slot()
    def pause(self) -> bool:
        if (
            self._closing
            or self._application_close_fenced
            or self._worker is None
        ):
            return False
        self._worker.pause()
        self._set_status("Fit paused.")
        return True

    @QtCore.Slot()
    def resume(self) -> bool:
        if (
            self._closing
            or self._application_close_fenced
            or self._worker is None
        ):
            return False
        self._worker.resume()
        self._set_status("Fit resumed.")
        return True

    def complete_detailed_balance_correction(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> bool:
        """Commit one verified correction, then admit one fresh request.

        Correction is requested during setup admission, before the gate or a
        QThread is acquired.  Consequently this path has no worker restart or
        wake-up protocol: the corrected origin is installed and the ordinary
        ``execute`` route performs the sole request construction.
        """

        self._require_thread()
        if self._application_close_fenced or self._closing:
            return False
        if not isinstance(outcome, DetailedBalanceCorrectionOutcome):
            raise TypeError("outcome must be DetailedBalanceCorrectionOutcome.")
        pending = self._pending_detailed_balance
        if pending is None or self._worker is not None:
            return False
        request = pending.correction
        if (
            (
                outcome.origin,
                outcome.tab_id,
                outcome.source,
                outcome.issue,
            )
            != (
                request.origin,
                request.tab_id,
                request.source,
                request.issue,
            )
        ):
            self._set_status(
                "Detailed-balance correction rejected: request currentness "
                "is no longer current."
            )
            return False
        if outcome.cancelled:
            self._pending_detailed_balance = None
            self._set_status("Detailed-balance correction cancelled; Fit aborted.")
            return True

        corrected_text = outcome.corrected_source
        if corrected_text is None:
            raise ValueError("Accepted correction has no corrected source.")
        current_origin = self._current_origin()
        corrected_canonical = current_origin.mechanism_source.with_reactions_text(
            corrected_text
        ) if current_origin is not None else None
        if corrected_canonical is None:
            return False
        try:
            if (
                self._pending_detailed_balance is not pending
            ):
                return False
            proposal = self._propose_detailed_balance_source(
                request.tab_id,
                corrected_canonical,
            )
            if not isinstance(proposal, DetailedBalanceCorrectionProposal):
                raise RuntimeError(
                    "Canonical source proposal rejected the correction."
                )
            proposed_tab = proposal.proposed_document.tab(request.tab_id)
            canonical_prepared = proposed_tab.prepared_dsl
            if not isinstance(canonical_prepared, PreparedDslRequestSet):
                raise RuntimeError(
                    "Proposed canonical source has no retained prepared DSL."
                )
            self._verify_detailed_balance_resolved(canonical_prepared)
            rebased = capture_fit_origin(
                proposal.proposed_document,
                request.tab_id,
            )
            previous_origin = self._current_origin()
            if previous_origin is None:
                raise RuntimeError("Fit origin was released before correction.")
            if (
                rebased.scientific_settings
                != previous_origin.scientific_settings
                or rebased.mapped_datasets != previous_origin.mapped_datasets
                or rebased.fit_defaults != previous_origin.fit_defaults
            ):
                raise RuntimeError(
                    "Fit origin changed beyond the committed mechanism source."
                )
            corrected_prepared = rebased.prepared_dsl
            self._verify_detailed_balance_resolved(corrected_prepared)
            corrected_snapshot = replace(
                pending.setup,
                origin=rebased,
            )
            commit_result = self._commit_detailed_balance_source(proposal)
            if not isinstance(
                commit_result,
                DetailedBalanceCorrectionCommitResult,
            ):
                raise RuntimeError("Canonical source commit returned no result.")
            if not commit_result.committed:
                raise RuntimeError(
                    "Canonical source commit rejected the correction"
                    + (
                        f": {commit_result.error}"
                        if commit_result.error
                        else "."
                    )
            )
        except Exception as exc:
            self._pending_detailed_balance = None
            rejection_message = (
                "Detailed-balance correction rejected before canonical commit; "
                "source remains an uncommitted draft; Fit aborted: "
                f"{exc}"
            )
            self._set_status(rejection_message)
            return False
        try:
            # Install the corrected owner/window state before the fresh
            # request admission.
            if commit_result.error:
                raise RuntimeError(
                    "Canonical source commit completed with a postcommit "
                    f"error: {commit_result.error}"
                )
            live_tab = self._session.document.tab(request.tab_id)
            live_runtime = self._registry.tab_runtime(request.tab_id)
            if (
                live_tab.mechanism_source != proposal.source
                or live_tab.prepared_dsl is not proposal.prepared_dsl
                or live_runtime.reactions_draft is not None
            ):
                raise RuntimeError(
                    "Canonical source commit did not install the exact "
                    "verified correction and clear its matching draft."
                )
            retry_state = self._retry_state()
            if corrected_snapshot.shared_refit_promotions:
                self._install_setup(
                    corrected_snapshot,
                    retry_source=(
                        None if retry_state is None else retry_state.source_candidate
                    ),
                )
            else:
                self._install_setup(corrected_snapshot)
            self._initial_setup_snapshot = replace(
                corrected_snapshot,
                shared_refit_promotions=(),
            )
            self._last_fit_setup_snapshot = None
            self._rebase_owner_snapshots(rebased)
            if self._window is not None:
                self._window.rebase(
                    rebased,
                    setup=corrected_snapshot,
                )
                self._project_initial_apply_proposal(self._window)
                self._window.set_initial_reset_available(corrected_snapshot)
        except Exception as exc:
            self._pending_detailed_balance = None
            self._set_status(
                "Detailed-balance correction committed; Fit owner rebase failed: "
                f"{exc}"
            )
            return False
        self._pending_detailed_balance = None
        self._set_status(
            "Detailed-balance correction committed; admitting a fresh Fit request."
        )
        return self.execute()

    @QtCore.Slot(str, str, object, str)
    def add_local_observable(
        self,
        name: str,
        expression: str,
        observations: object,
        scalar_scope: str,
    ) -> bool:
        """Atomically admit one observable across the selected dataset roster."""
        self._require_thread()
        if self._closing or self._application_close_fenced:
            return False
        window = self._window
        origin = self._current_origin()
        setup = self._accepted_setup()
        if (
            window is None
            or origin is None
            or setup is None
            or self._worker is not None
        ):
            return False
        try:
            if scalar_scope not in {"shared", "per_dataset"}:
                raise ValueError(
                    "Unknown scalar scope must be shared or per_dataset."
                )
            selected_ids = tuple(window.selected_fit_dataset_ids())
            selected = tuple(observations) if isinstance(observations, (tuple, list)) else ()
            if not selected_ids or len(selected) != len(selected_ids):
                raise ValueError(
                    "Selected Fit observations must cover the complete selected dataset roster."
                )
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError("Selected Fit dataset roster repeats an identity.")
            included_ids = tuple(item.dataset_id for item in setup.datasets if item.included)
            if tuple(selected_ids) != included_ids:
                raise ValueError(
                    "Selected Fit observations do not match the active accepted target roster."
                )
            if any(not isinstance(item, FitObservationSnapshot) for item in selected):
                raise TypeError("Selected Fit observations must be typed snapshots.")
            for dataset_id in selected_ids:
                conflict = window.add_observable_conflict(dataset_id)
                if conflict is not None:
                    raise ValueError(conflict)
            observable = validate_fit_local_observable(
                name=name,
                expression=expression,
                prepared_dsl=origin.prepared_dsl,
                additional_known_observables=tuple(
                    item.name for item in setup.local_observables
                ),
                additional_known_scalars=tuple(
                    scalar_name
                    for item in setup.local_observables
                    for scalar_name in item.scalar_names
                ),
            )
            updates: dict[str, FitDatasetLocalSettings] = {}
            for dataset_id, selected_observation in zip(selected_ids, selected, strict=True):
                mapped = next(
                    item for item in origin.mapped_datasets
                    if item.dataset_id == dataset_id
                )
                captured = next(
                    (
                        item for item in mapped.observations
                        if item.raw_observation_id == selected_observation.raw_observation_id
                    ),
                    None,
                )
                if captured is None or captured != selected_observation:
                    raise ValueError(
                        f"Dataset {dataset_id!r} selected raw observation is stale or changed."
                    )
                local = next(item for item in setup.datasets if item.dataset_id == dataset_id)
                targets_by_raw_id = dict(local.observation_targets)
                raw_observation_id = selected_observation.raw_observation_id
                previous_target = targets_by_raw_id.get(raw_observation_id)
                if previous_target is None or previous_target not in local.target_species:
                    raise ValueError(
                        f"Dataset {dataset_id!r} selected raw observation has no active accepted target."
                    )
                if selected_observation.model_target != previous_target:
                    raise ValueError(
                        f"Dataset {dataset_id!r} selected raw observation target is stale."
                    )
                weight_by_target = dict(local.target_weights)
                if previous_target not in weight_by_target or weight_by_target[previous_target] <= 0.0:
                    raise ValueError(
                        f"Dataset {dataset_id!r} selected target has no valid weight."
                    )
                targets_by_raw_id[raw_observation_id] = observable.name
                active_targets = tuple(
                    observable.name if target == previous_target else target
                    for target in local.target_species
                )
                if len(active_targets) != len(set(active_targets)):
                    raise ValueError(
                        f"Observable {observable.name!r} duplicates an active Fit target."
                    )
                ordered_targets = tuple(
                    (raw_id, targets_by_raw_id[raw_id])
                    for raw_id, _target in local.observation_targets
                )
                weight_roster = (
                    (*active_targets, local.parametric.x_name)
                    if local.parametric is not None
                    else active_targets
                )
                updates[dataset_id] = replace(
                    local,
                    target_species=active_targets,
                    target_weights=tuple(
                        (
                            target,
                            weight_by_target[previous_target]
                            if target == observable.name
                            else weight_by_target[target],
                        )
                        for target in weight_roster
                    ),
                    observation_targets=ordered_targets,
                )
            if observable.name in {
                target
                for item in setup.datasets
                for target in item.target_species
            }:
                raise ValueError(
                    f"Observable {observable.name!r} duplicates an active Fit target."
                )
            retained_dataset_ids = tuple(
                item.dataset_id for item in setup.datasets
            )
            supplemental_parameters = tuple(
                FitParameterVariable(
                    name=scalar_name,
                    display_name=scalar_name,
                    scope=scalar_scope,
                    shared_config=FitParameterConfiguration(
                        value=1.0,
                        minimum=1.0e-12,
                        maximum=1.0e12,
                        fit=True,
                        log10=False,
                        dataset_values=tuple(
                            (retained_id, 1.0)
                            for retained_id in retained_dataset_ids
                        ),
                    ),
                    per_dataset_config=FitParameterConfiguration(
                        value=1.0,
                        minimum=1.0e-12,
                        maximum=1.0e12,
                        fit=True,
                        log10=False,
                        dataset_values=tuple(
                            (retained_id, 1.0)
                            for retained_id in retained_dataset_ids
                        ),
                    ),
                    family="scalar",
                    role="scalar",
                    scientific_category="finite_scalar",
                    allow_exact_zero=True,
                    unit="1",
                    fit_local_provenance=FitLocalScalarProvenance(
                        observable_name=observable.name,
                        admission_scope=scalar_scope,
                        admitted_value=1.0,
                    ),
                )
                for scalar_name in observable.scalar_names
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
            setup = replace(
                setup,
                shared_refit_promotions=(),
            )
            setup = self._setup_with_local_observables(
                replace(
                    setup,
                    datasets=tuple(updates.get(item.dataset_id, item) for item in setup.datasets),
                    parameters=(*setup.parameters, *supplemental_parameters),
                ),
                tuple(local_observables),
            )
            window.project_setup_snapshot(setup)
            self._install_setup(setup)
        except (StopIteration, TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Add Observable rejected: {exc}")
            return False
        self._invalidate_candidate_custody()
        self._clear_candidate_presentation(window)
        self._set_status(
            f"Observable {observable.name!r} is Fit-local until Apply."
        )
        return True

    @QtCore.Slot(object)
    def reset_setup(self, reset_kind: object) -> bool:
        """Restore one owner-retained reset snapshot selected by kind."""

        self._require_thread()
        window = self._window
        origin = self._current_origin()
        if (
            self._closing
            or self._application_close_fenced
            or self._worker is not None
            or window is None
            or origin is None
            or str(reset_kind or "").strip().lower() not in {"initial", "last_fit"}
        ):
            return False
        kind = str(reset_kind).strip().lower()
        snapshot = (
            self._initial_setup_snapshot
            if kind == "initial"
            else self._last_fit_setup_snapshot
        )
        if snapshot is None:
            self._set_status(
                "Fit setup reset rejected: the selected owner snapshot is unavailable."
            )
            return False
        if snapshot.origin is not origin:
            self._set_status("Fit setup reset rejected: the owner snapshot is stale.")
            return False
        try:
            current = self._accepted_setup()
            if current is None:
                return False
            retained = self._reset_snapshot_against_current_origin(
                snapshot,
                origin,
            )
            restored_snapshot = self._parameters_reset_snapshot(
                current=current,
                retained=retained,
                origin=origin,
            )
            restored_snapshot = replace(
                restored_snapshot,
                shared_refit_promotions=(),
            )
            self._params_reset_projection()
            self._clear_candidate_presentation(window)
            window.project_setup_snapshot(restored_snapshot)
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup reset failed: {exc}")
            return False
        self._install_setup(restored_snapshot)
        self._invalidate_candidate_custody()
        self._set_status(f"Fit setup {kind.replace('_', ' ')} snapshot restored.")
        return True

    def _params_reset_projection(self) -> None:
        """Clear only presentation drafts owned by the Parameters surface."""

        window = self._window
        if window is not None:
            window.clear_parameters_pending_edits()

    def _clear_candidate_presentation(self, window: FittingWindow) -> None:
        """Clear the candidate presentation and its exact IC proposal together."""

        if not isinstance(window, FittingWindow):
            raise TypeError("Fit candidate presentation requires FittingWindow.")
        window.clear_candidate()

    def _current_candidate_projection(
        self,
    ) -> tuple[FitCandidate, dict[str, object]] | None:
        candidate = self._candidate
        if candidate is None:
            self._set_status("No completed Fit run is available.")
            return None
        try:
            projection = build_fit_action_projection(candidate)
        except Exception as exc:
            self._set_status(f"Fit action projection failed: {exc}")
            return None
        return candidate, projection

    @QtCore.Slot()
    def copy_hash(self) -> bool:
        """Build one fresh action projection and copy its deterministic hash."""

        projected = self._current_candidate_projection()
        if projected is None:
            return False
        _candidate, projection = projected
        try:
            value = fit_action_projection_hash(projection)
            QtWidgets.QApplication.clipboard().setText(value)
            if self._window is not None:
                self._window.set_copy_hash(value)
            self._set_status("Copied the Fit run hash.")
            return True
        except Exception as exc:
            self._set_status(f"Fit hash copy failed: {exc}")
            return False

    @QtCore.Slot()
    def copy_run_json(self) -> bool:
        """Build one fresh action projection and copy its JSON payload."""

        projected = self._current_candidate_projection()
        if projected is None:
            return False
        _candidate, projection = projected
        try:
            value = fit_action_projection_json(projection)
            QtWidgets.QApplication.clipboard().setText(value)
            self._set_status("Copied the complete Fit run JSON.")
            return True
        except Exception as exc:
            self._set_status(f"Fit Run JSON copy failed: {exc}")
            return False

    @QtCore.Slot()
    def results_summary(self) -> bool:
        """Render Summary from one fresh projection and its matching hash."""

        projected = self._current_candidate_projection()
        if projected is None or self._window is None:
            return False
        _candidate, projection = projected
        try:
            summary = fit_action_projection_summary(projection)
            copy_hash = str(summary["copy_hash"])
            self._window.render_results_summary(
                summary,
                identity=copy_hash,
                copy_hash_requested=self.copy_hash,
                copy_json_requested=self.copy_run_json,
            )
            return True
        except Exception as exc:
            self._set_status(f"Fit Results Summary failed: {exc}")
            return False

    def _invalidate_candidate_custody(self) -> None:
        """Release the current candidate and its Apply custody."""

        self._candidate = None
        self._candidate_applied_scopes = frozenset()
        self._candidate_applied_initial_keys = frozenset()

    def _project_initial_apply_proposal(
        self,
        window: FittingWindow,
    ) -> FitInitialApplyProposal | None:
        """Project the owner's current origin-bound proposal to the window."""

        if not isinstance(window, FittingWindow):
            raise TypeError("Fit IC proposal presentation requires FittingWindow.")
        proposal = self.current_initial_apply_proposal()
        window.set_initial_apply_proposal(proposal)
        return proposal

    def _reset_snapshot_against_current_origin(
        self,
        snapshot: FitSetupSnapshot,
        origin: FitOriginSnapshot,
    ) -> FitSetupSnapshot:
        """Rebase retained values onto canonical science without old promotions."""

        canonical_observable_names = set(
            origin.prepared_dsl.analysis.observable_names
        )
        promoted = tuple(
            item
            for item in snapshot.local_observables
            if item.name in canonical_observable_names
        )
        retained = self._without_promoted_local_observables(
            snapshot,
            promoted,
        )
        return self._rebase_setup_after_apply(
            retained,
            origin,
            frozenset(),
            retained.local_observables,
        )

    def _parameters_reset_snapshot(
        self,
        *,
        current: FitSetupSnapshot,
        retained: FitSetupSnapshot,
        origin: FitOriginSnapshot,
    ) -> FitSetupSnapshot:
        """Restore setup values while retaining only executable target routes."""

        current_by_dataset = {
            item.dataset_id: item for item in current.datasets
        }
        if set(current_by_dataset) != {
            item.dataset_id for item in retained.datasets
        }:
            raise ValueError(
                "Fit reset changed the stable dataset identities."
            )
        datasets = []
        for retained_dataset in retained.datasets:
            current_dataset = current_by_dataset[retained_dataset.dataset_id]
            retain_current_route = self._dataset_target_route_is_executable(
                origin=origin,
                setup=retained,
                dataset=current_dataset,
            )
            route_owner = (
                current_dataset
                if retain_current_route
                else retained_dataset
            )
            restored_dataset = replace(
                retained_dataset,
                # These fields are authored by Data and survive Parameters Reset.
                t_start=current_dataset.t_start,
                included=current_dataset.included,
                data_range_min=current_dataset.data_range_min,
                data_range_max=current_dataset.data_range_max,
                maximum_observations_per_target=(
                    current_dataset.maximum_observations_per_target
                ),
                # Target/weight/mapping state survives only as one executable route.
                weight=route_owner.weight,
                target_species=route_owner.target_species,
                target_weights=route_owner.target_weights,
                observation_targets=route_owner.observation_targets,
                parametric=route_owner.parametric,
            )
            if not self._dataset_target_route_is_executable(
                origin=origin,
                setup=retained,
                dataset=restored_dataset,
            ):
                raise ValueError(
                    "Fit reset cannot restore an executable target/mapping "
                    f"for dataset {retained_dataset.dataset_id!r}."
                )
            datasets.append(restored_dataset)
        return replace(
            retained,
            datasets=tuple(datasets),
            solver=current.solver,
            dataset_weight_mode=current.dataset_weight_mode,
        )

    @staticmethod
    def _dataset_target_route_is_executable(
        *,
        origin: FitOriginSnapshot,
        setup: FitSetupSnapshot,
        dataset: FitDatasetLocalSettings,
    ) -> bool:
        """Check one target/mapping route against canonical plus local science."""

        try:
            if not isinstance(dataset, FitDatasetLocalSettings):
                return False
            mapped = next(
                item
                for item in origin.mapped_datasets
                if item.dataset_id == dataset.dataset_id
            )
            captured_by_raw_id = {
                item.raw_observation_id: item
                for item in mapped.observations
            }
            allowed_observables = set(
                origin.prepared_dsl.analysis.observable_names
            )
            allowed_observables.update(
                item.name for item in setup.local_observables
            )
            for raw_id, target in dataset.observation_targets:
                captured = captured_by_raw_id.get(raw_id)
                if captured is None or (
                    target != captured.model_target
                    and target not in allowed_observables
                ):
                    return False
            # Exclusion suspends execution; it does not make an empty or stale
            # target route valid.  Reset must preserve a route that can execute
            # immediately if the dataset is re-included later.
            selected = selected_observations(mapped, dataset)
            return bool(selected)
        except (AttributeError, StopIteration, TypeError, ValueError, KeyError):
            return False

    @QtCore.Slot(tuple, tuple)
    def apply_candidate(
        self,
        scopes: tuple[str, ...],
        initial_keys: tuple[tuple[str, str], ...],
    ) -> bool:
        """Apply any selected independent scopes through one document replacement."""
        self._require_thread()
        if self._closing or self._application_close_fenced:
            return False
        candidate = self._candidate
        origin = self._current_origin()
        window = self._window
        if candidate is None or origin is None or window is None:
            self._set_status("There is no completed Fit candidate to Apply.")
            return False
        selection = FitApplySelection(
            origin=origin,
            scopes=tuple(scopes),
            initial_keys=tuple(initial_keys),
        )
        selected = frozenset(selection.scopes)
        selected_initial_keys = frozenset(selection.initial_keys)
        if selection.origin is not origin:
            self._set_status("Apply rejected: the selection belongs to a stale Fit origin.")
            return False
        if not self._candidate_origin_is_current(candidate):
            self._set_status("Apply rejected: origin science is no longer current.")
            return False
        origin = self._current_origin()
        if origin is None:
            return False
        available, unavailable_reasons = self._candidate_apply_feasibility(candidate)
        if not selected.issubset(available):
            unavailable_selected = sorted(selected.difference(available))
            reason = "; ".join(
                unavailable_reasons.get(scope, "")
                for scope in unavailable_selected
            )
            self._set_status(
                "Apply rejected: "
                + (reason or "the selected Apply scope is unavailable.")
            )
            return False
        if window.has_pending_target_changes:
            self._set_status(
                "Apply rejected: Apply or Revert pending target edits first."
            )
            return False
        preserved = self._accepted_setup()
        if preserved is None:
            self._set_status("Apply rejected: Fit has no typed local setup.")
            return False
        expected_token = self._session.capture_content_token()
        previous_document = self._session.document
        try:
            initial_proposal = self._initial_apply_proposal(candidate)
            proposal_keys = frozenset(initial_proposal.selected_keys)
            if "initial_concentrations" in selected:
                unavailable_initial_keys = selected_initial_keys - proposal_keys
                unavailable_initial_keys |= (
                    selected_initial_keys & self._candidate_applied_initial_keys
                )
                if unavailable_initial_keys:
                    raise ValueError(
                        "Initial Conditions Apply contains unavailable cells: "
                        + ", ".join(repr(item) for item in sorted(unavailable_initial_keys))
                    )
            local_scalar_names = self._candidate_local_scalar_names(
                candidate
            )
            terminal_parameter_by_name = {
                item.name: item
                for item in candidate.setup_snapshot.parameters
            }
            publishable_values = self._candidate_publishable_parameter_values(candidate)
            applied_canonical_parameter_names = tuple(
                sorted(
                    name
                    for name in publishable_values
                    if (
                        "parameters" in selected
                        and name not in local_scalar_names
                        and terminal_parameter_by_name.get(name) is not None
                    )
                )
            )
            unapplied_local_items = tuple(
                sorted(
                    item.name
                    for item in candidate.setup_snapshot.parameters
                    if (
                        item.selected_config.fit
                        and item.name in local_scalar_names
                        and "parameters" in selected
                        and "observable_definitions" not in selected
                    )
                )
            )
            per_dataset_not_applied = tuple(
                sorted(
                    item.name
                    for item in candidate.setup_snapshot.parameters
                    if item.selected_config.fit and item.scope == "per_dataset"
                )
            )
            applied_initial_keys = (
                self._candidate_applied_initial_keys | selected_initial_keys
            )
            applied_scopes = self._candidate_applied_scopes | (
                selected - {"initial_concentrations"}
            )
            if proposal_keys and proposal_keys <= applied_initial_keys:
                applied_scopes = applied_scopes | {"initial_concentrations"}
            preserved = self._preserved_setup_with_terminal_values(
                preserved,
                candidate,
            )
            # Any partial Apply publishes a new canonical origin.  Retire
            # promotion intent in the prepared rebase so retry custody cannot
            # survive the accepted document replacement.
            preserved = replace(
                preserved,
                shared_refit_promotions=(),
            )
            if "observable_definitions" in selected:
                preserved = self._without_promoted_local_observables(
                    preserved,
                    candidate.local_observables,
                )
            current_tab = previous_document.tab(origin.tab_id)
            next_tab = self._build_applied_tab(
                current_tab,
                candidate,
                selected,
                origin,
                selected_initial_keys=selected_initial_keys,
            )
            next_document = previous_document.replace_tab(next_tab)
            # Prepare the complete semantic rebase against the exact proposed
            # document.  Nothing canonical is published unless this succeeds.
            rebased = capture_fit_origin(next_document, origin.tab_id)
            refreshed_snapshot = self._rebase_setup_after_apply(
                preserved,
                rebased,
                selected,
                preserved.local_observables,
                selected_initial_keys=selected_initial_keys,
            )
            refreshed_initial_snapshot = self._reset_owner_snapshot_after_apply(
                self._initial_setup_snapshot,
                origin=rebased,
                canonicalized_scopes=selected,
                promoted=candidate.local_observables,
                selected_initial_keys=selected_initial_keys,
            )
            refreshed_last_fit_snapshot = self._reset_owner_snapshot_after_apply(
                self._last_fit_setup_snapshot,
                origin=rebased,
                canonicalized_scopes=selected,
                promoted=candidate.local_observables,
                selected_initial_keys=selected_initial_keys,
            )
            available_scopes, unavailable_reasons = (
                self._candidate_apply_feasibility(
                    candidate,
                    applied_scopes=applied_scopes,
                    applied_initial_keys=applied_initial_keys,
                )
            )
        except Exception as exc:
            self._set_status(f"Apply rejected: {exc}")
            return False

        try:
            result = self._registry.replace_document(
                next_document,
                expected_token=expected_token,
            )
        except Exception as exc:
            self._set_status(
                "Apply rejected before document replacement; the Fit candidate "
                f"remains available: {exc}"
            )
            return False
        if result is None:
            self._set_status(
                "Apply rejected before document replacement: project content "
                "changed before the Fit candidate could be published."
            )
            return False
        changed = result.changed

        # The accepted document and its complete semantic Fit rebase are now
        # installed together before any fallible result/view projection.
        self._initial_setup_snapshot = refreshed_initial_snapshot
        self._last_fit_setup_snapshot = refreshed_last_fit_snapshot
        self._install_setup(refreshed_snapshot)
        # Keep the complete terminal candidate as the sole Summary/Copy
        # carrier.  The rebased context above was call-local Apply state only.
        self._candidate = candidate
        self._candidate_applied_scopes = applied_scopes
        self._candidate_applied_initial_keys = applied_initial_keys
        curve_failure: str | None = None
        (
            projection,
            copy_hash,
            report_text,
            report_failure,
            curves,
            curve_failure,
        ) = self._build_presentations(candidate)
        initial_proposal = self.current_initial_apply_proposal()
        self.canonical_applied.emit(origin.tab_id, selected)

        try:
            # Clear old Apply/Copy authority before rebuilding the physical view.
            self._clear_candidate_presentation(window)
            window.rebase(
                rebased,
                setup=refreshed_snapshot,
            )
            rendered = window.set_terminal_presentation(
                projection,
                copy_hash=copy_hash,
                report_text=report_text,
                report_failure=report_failure,
                curve_failure=curve_failure,
                curves=curves,
                apply_labels=unavailable_reasons,
                enabled_scopes=available_scopes,
                initial_proposal=initial_proposal,
            )
        except Exception as exc:
            projection = None
            curves = ()
            report_failure = None
            curve_failure = f"{type(exc).__name__}: {exc}"
            try:
                window.set_terminal_presentation(
                    projection,
                    copy_hash=None,
                    report_text=None,
                    report_failure=report_failure,
                    curve_failure=curve_failure,
                    curves=curves,
                    apply_labels=unavailable_reasons,
                    enabled_scopes=available_scopes,
                    initial_proposal=initial_proposal,
                )
            except Exception as recovery_exc:
                exc = RuntimeError(f"{exc}; Apply recovery failed: {recovery_exc}")
            self._set_status(
                "Selected Fit values were applied and remaining scopes were "
                "preserved, but their view projection failed: "
                f"{exc}"
            )
            return True
        if not rendered:
            self._set_status(
                "Selected Fit values were applied and remaining scopes were "
                "preserved, but fitted values remain available while their "
                "plot is unavailable."
            )
            return True
        parameter_disposition: list[str] = []
        if "parameters" in selected:
            parameter_disposition.append(
                "Shared canonical Parameter values applied: "
                + ", ".join(applied_canonical_parameter_names)
                + "."
            )
            if per_dataset_not_applied:
                parameter_disposition.append(
                    "Per-dataset values remain Fit-local and were not "
                    "applied to Run: "
                    + ", ".join(per_dataset_not_applied)
                    + "."
                )
        if unapplied_local_items:
            parameter_disposition.append(
                "Fit-local observable scalars were not applied through "
                "Parameters: "
                + ", ".join(unapplied_local_items)
                + ". Apply Observable Definitions to publish them."
            )
        if parameter_disposition:
            if available_scopes:
                parameter_disposition.append(
                    "Remaining terminal Fit scopes are still available."
                )
            self._set_status(" ".join(parameter_disposition))
        elif available_scopes:
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
        if self._closing or self._application_close_fenced:
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
                    "changed before document replacement."
                )
                return False
        except Exception as exc:
            self._set_status(
                f"Project Fit defaults update failed: {exc}"
            )
            return False
        if result.changed:
            self._set_status("Current-project Fit defaults updated.")
        return result.changed

    def request_project_replacement_cancellation(self) -> None:
        """Broadcast a nonblocking Fit cancellation request.

        Project replacement calls this on every active owner before waiting on
        any one owner.  Worker cancellation remains owned by this session;
        this method never waits and never publishes a terminal candidate.
        """

        self._require_thread()
        if self._closing:
            return
        self._closing = True
        self._pending_detailed_balance = None
        worker = self._worker
        if worker is not None:
            worker.request_cancel()

    def settle_project_replacement(self, deadline: float) -> bool:
        """Settle a previously requested Fit cancellation by a common deadline."""

        self._require_thread()
        worker = self._worker
        if worker is not None:
            remaining_ms = max(
                0,
                int((float(deadline) - time.monotonic()) * 1000.0),
            )
            if not worker.wait(remaining_ms):
                # Replacement remains failure-atomic: leave the current Fit
                # worker/candidate/window installed and reopen admission for
                # a later owner-controlled retry.
                self._closing = False
                return False
            self._finish_worker()
        self._discard_local()
        if self._window is not None:
            return self._destroy_window_after_cleanup()
        self._closing = False
        self._notify_application_close_settled()
        return True

    def close_idle_for_application_close(self) -> bool:
        """Synchronously discard and destroy an idle Fit dialog for Close."""

        self._require_thread()
        if self._worker is not None:
            return False
        self._closing = True
        self._discard_local()
        if self._window is None:
            self._closing = False
            self._notify_application_close_settled()
            return True
        return self._destroy_window_after_cleanup()

    def fence_application_close_admission(self) -> None:
        """Reject Fit admission and continuation for the active Close attempt."""

        self._require_thread()
        if self._application_close_fenced:
            return
        self._application_close_fenced = True
        if not self.application_close_settled:
            self._application_close_settlement_pending = True

    def cancel_for_application_close(self) -> None:
        """Dispose Fit state and close through the existing owner lifecycle."""

        self._require_thread()
        was_closing = self._closing
        if not self._application_close_fenced:
            self.fence_application_close_admission()
        self._closing = True
        window = self._window
        worker = self._worker
        self._discard_local()
        if worker is None:
            if window is not None:
                self._destroy_window_after_cleanup()
            return
        if window is not None:
            window.enter_stopping()
        if not was_closing:
            # Close is explicit so a stopped-best terminal can never win
            # application teardown.
            worker.request_cancel(close=True)

    def _request_window_close(self) -> bool:
        if self._closing:
            return False
        self._closing = True
        if self._worker is not None:
            self._pending_detailed_balance = None
            if self._window is not None:
                self._window.enter_stopping()
            self._worker.request_cancel(close=True)
            return False
        self._discard_local()
        return True

    def _destroy_window_after_cleanup(self) -> bool:
        """Use the owner-held deferred-delete path and verify destruction."""

        window = self._window
        if window is None:
            self._notify_application_close_settled()
            return True
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
        destroyed = self._window is None
        if destroyed:
            self._notify_application_close_settled()
        return destroyed

    def _discard_local(self) -> None:
        self._invalidate_candidate_custody()
        self._setup_state = None
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
        self._pending_detailed_balance = None
        if self._window is not None:
            self._clear_candidate_presentation(self._window)

    @QtCore.Slot()
    def _on_window_destroyed(self, *_args: object) -> None:
        if self._window is None:
            return
        self._window = None
        self._setup_state = None
        self._invalidate_candidate_custody()
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
        self._pending_detailed_balance = None
        worker = self._worker
        self._closing = worker is not None
        self.window_changed.emit(False)
        self._notify_application_close_settled()
        self._set_status("Fit closed; all transient Fit state was discarded.")

    def _on_progress(self, worker: GlobalFitWorker, value: int, text: str) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        if self._window is not None:
            self._window.set_progress(value, text)
        self._set_status(text)

    def _on_best(self, worker: GlobalFitWorker, update: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
            or not isinstance(update, FitLiveUpdate)
        ):
            return
        if self._window is not None:
            self._window.set_live_best(update)

    def _on_auto_modes(self, worker: GlobalFitWorker, modes: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
            or not isinstance(modes, tuple)
        ):
            return
        normalized = tuple(
            (str(dataset_id), str(mode))
            for dataset_id, mode in modes
        )
        if any(mode not in {"monotone", "time_guided"} for _item, mode in normalized):
            worker.request_cancel()
            self._set_status("Fit emitted an invalid resolved Auto map.")
            return
        if self._window is not None:
            self._window.set_resolved_auto_modes(normalized)

    def _on_live_curves(self, worker: GlobalFitWorker, curves: object) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
            or not isinstance(curves, tuple)
        ):
            return
        if self._window is not None:
            self._window.set_live_curves(curves)

    def _on_worker_finished(
        self,
        worker: GlobalFitWorker,
    ) -> None:
        if self._worker is not worker:
            return
        worker.wait(0)
        terminal = worker.terminal_result
        if not isinstance(terminal, FitWorkerTerminal):
            self._finish_worker(worker)
            self._set_status("Fit worker exited without one terminal result.")
            return
        retry_state = self._retry_state()
        shared_refit_source_candidate = (
            retry_state.source_candidate if retry_state is not None else None
        )
        request = worker.request
        candidate = terminal.candidate
        shared_refit_run = bool(
            retry_state is not None
            or (
                request is not None
                and request.dispatch.shared_refit_execution is not None
            )
            or (
                candidate is not None
                and candidate.request.dispatch.shared_refit_execution is not None
            )
        )
        launch_setup_snapshot = self._accepted_setup()
        terminal_candidate_accepted = bool(
            not self._application_close_fenced
            and not self._closing
            and candidate is not None
            and terminal.failure is None
            and terminal.lifecycle_cause in {"completed", "user_stop"}
            and self._origin_is_current()
        )
        if terminal_candidate_accepted:
            origin = self._current_origin()
            terminal_candidate_accepted = bool(
                origin is not None
                and candidate is not None
                and candidate.origin is origin
            )
        failure_message = terminal.message
        if self._closing:
            if not self._application_close_fenced:
                self._discard_local()
            self._finish_worker(worker)
            self._destroy_window_after_cleanup()
            return
        self._finish_worker(worker)
        if not terminal_candidate_accepted or candidate is None:
            if (
                shared_refit_run
                and shared_refit_source_candidate is not None
                and launch_setup_snapshot is not None
            ):
                self._restore_shared_refit_source_presentation(
                    shared_refit_source_candidate,
                    setup=launch_setup_snapshot,
                    failure=(
                        terminal.failure
                        if isinstance(terminal, FitWorkerTerminal)
                        else None
                    ),
                )
            if (
                terminal.lifecycle_cause == "scientific_unavailable"
                and self._window is not None
                and not self._application_close_fenced
                and not self._closing
            ):
                cause = (
                    terminal.unavailable.factual_cause
                    if terminal.unavailable is not None
                    else None
                )
                if cause is not None:
                    # The unavailable attempt has no candidate or result of
                    # its own.  If an older Last Fit was retained, mark that
                    # separate presentation explicitly; Shared-refit source
                    # restoration above remains the source's own custody.
                    self._window.set_terminal_unavailable(
                        cause,
                        older_retained_last_fit=(
                            self._last_fit_setup_snapshot is not None
                        ),
                    )
            if (
                isinstance(terminal, FitWorkerTerminal)
                and terminal.failure is not None
                and not shared_refit_run
                and self._window is not None
                and not self._application_close_fenced
                and not self._closing
            ):
                try:
                    self._window.set_terminal_failure(terminal.failure)
                except Exception as exc:
                    self._set_status(
                        "Fit failed and typed failure presentation failed: "
                        f"{exc}"
                    )
                    return
            if (
                not self._application_close_fenced
                and not self._closing
                and failure_message
            ):
                self._set_status(failure_message)
            return
        shared_refit_replacement_succeeded = bool(
            shared_refit_run
            and self._candidate_result_is_apply_eligible(candidate)
        )
        if (
            shared_refit_run
            and not shared_refit_replacement_succeeded
            and shared_refit_source_candidate is not None
            and launch_setup_snapshot is not None
        ):
            self._restore_shared_refit_source_presentation(
                shared_refit_source_candidate,
                setup=launch_setup_snapshot,
            )
            if terminal.lifecycle_cause == "user_stop":
                self._set_status(terminal.message)
            else:
                self._set_status(
                    "Fit completed with warning; finite values are diagnostic seed "
                    "material only and cannot be applied."
                )
            return
        self._candidate = candidate
        terminal_setup = candidate.setup_snapshot
        (
            projection,
            copy_hash,
            report_text,
            report_failure,
            curves,
            curve_failure,
        ) = self._build_presentations(candidate)
        if shared_refit_replacement_succeeded:
            self._install_setup(
                replace(
                    terminal_setup,
                    shared_refit_promotions=(),
                )
            )
        elif (
            shared_refit_source_candidate is not None
            and launch_setup_snapshot is not None
        ):
            self._install_setup(
                launch_setup_snapshot,
                retry_source=shared_refit_source_candidate,
            )
        else:
            self._install_setup(terminal_setup)
        if self._initial_setup_snapshot is not None:
            self._initial_setup_snapshot = replace(
                self._initial_setup_snapshot,
                shared_refit_promotions=(),
            )
        self._candidate_applied_scopes = frozenset()
        self._candidate_applied_initial_keys = frozenset()
        self._last_fit_setup_snapshot = replace(
            terminal_setup,
            shared_refit_promotions=(),
        )
        available_scopes, unavailable_reasons = self._candidate_apply_feasibility(
            candidate
        )
        initial_proposal = self.current_initial_apply_proposal()
        rendered = False
        if self._window is not None:
            try:
                accepted_setup = self._accepted_setup()
                if accepted_setup is None:
                    raise RuntimeError("Fit terminal lost its accepted setup.")
                self._window.project_setup_snapshot(accepted_setup)
                rendered = self._window.set_terminal_presentation(
                    projection,
                    copy_hash=copy_hash,
                    report_text=report_text,
                    report_failure=report_failure,
                    curve_failure=curve_failure,
                    curves=curves,
                    apply_labels=unavailable_reasons,
                    enabled_scopes=available_scopes,
                    initial_proposal=initial_proposal,
                )
            except Exception as exc:
                # A curve/view failure is presentation-only; candidate and
                # Apply custody remain untouched.
                curves = ()
                curve_failure = f"{type(exc).__name__}: {exc}"
                try:
                    self._window.set_terminal_presentation(
                        projection,
                        copy_hash=copy_hash,
                        report_text=report_text,
                        report_failure=report_failure,
                        curve_failure=curve_failure,
                        curves=curves,
                        apply_labels=unavailable_reasons,
                        enabled_scopes=available_scopes,
                        initial_proposal=initial_proposal,
                    )
                except Exception as recovery_exc:
                    exc = RuntimeError(
                        f"{exc}; Fit presentation recovery failed: {recovery_exc}"
                    )
        if self._window is not None:
            self._window.set_last_fit_reset_available(self._last_fit_setup_snapshot)
        if terminal.lifecycle_cause == "user_stop":
            self._set_status(terminal.message)
            return
        if not self._candidate_result_is_apply_eligible(candidate):
            self._set_status(
                "Fit completed with warning; finite values are diagnostic seed "
                "material only and cannot be applied."
            )
            return
        if rendered:
            self._set_status("Fit completed; candidate is ready to Apply.")
        elif curve_failure is not None:
            self._set_status(
                "Fit completed; plotting failed, but the candidate remains "
                f"ready to Apply: {curve_failure}"
            )
        else:
            self._set_status(
                "Fit completed; plotting failed, but the candidate remains ready to Apply."
            )

    def _build_presentations(
        self,
        candidate: FitCandidate,
    ) -> tuple[
        dict[str, object] | None,
        str | None,
        str | None,
        str | None,
        tuple[object, ...],
        str | None,
    ]:
        """Build report and curves independently from retained candidate truth."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("Fit presentation candidate must be FitCandidate.")
        projection: dict[str, object] | None
        copy_hash: str | None = None
        projection_failure: str | None = None
        try:
            projection = build_fit_action_projection(candidate)
        except Exception as exc:
            projection = None
            projection_failure = f"{type(exc).__name__}: {exc}"
        if projection is not None:
            try:
                copy_hash = fit_action_projection_hash(projection)
            except Exception as exc:
                projection_failure = f"{type(exc).__name__}: {exc}"
            try:
                report_text = fit_action_projection_report_text(projection)
            except Exception as exc:
                report_text = None
                report_failure = f"{type(exc).__name__}: {exc}"
            else:
                report_failure = None
        else:
            report_text = None
            report_failure = projection_failure
        try:
            curves = tuple(project_fit_candidate_curves(candidate))
            if not curves:
                raise ValueError("Fit result produced no curve datasets.")
        except Exception as exc:
            curves = ()
            curve_failure = f"{type(exc).__name__}: {exc}"
        else:
            curve_failure = None
        return (
            projection,
            copy_hash,
            report_text,
            report_failure,
            curves,
            curve_failure,
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
        if (
            self._window is not None
            and not self._application_close_fenced
            and not self._closing
        ):
            self._window.set_executing(False)
        self.execution_changed.emit(False)
        if worker is not None:
            worker.deleteLater()
        self._notify_application_close_settled()

    @staticmethod
    def _verify_detailed_balance_resolved(
        prepared_dsl: PreparedDslRequestSet,
    ) -> None:
        """Verify the retained accepted correction resolved every cycle."""

        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
        from kindred.core.simulator.wegscheider_symbolic import (
            require_wegscheider_admission_satisfied,
        )
        require_wegscheider_admission_satisfied(
            prepared_dsl.wegscheider_admission_report()
        )

    def _origin_is_current(self) -> bool:
        origin = self._current_origin()
        if origin is None:
            return False
        setup = self._accepted_setup()
        active_dataset_ids = frozenset(
            item.dataset_id
            for item in (
                self._worker.request.datasets
                if self._worker is not None
                else setup.datasets
                if setup is not None
                else ()
            )
            if item.included
        )
        try:
            current = capture_fit_origin(
                self._session.document,
                origin.tab_id,
                dataset_ids=active_dataset_ids,
            )
        except Exception:
            return False
        return fit_origin_science_matches(
            origin,
            current,
            included_dataset_ids=active_dataset_ids,
        )

    def _candidate_origin_is_current(self, candidate: FitCandidate) -> bool:
        """Check Apply against the roster admitted for the terminal candidate."""

        if not isinstance(candidate, FitCandidate):
            return False
        comparison_origin = (
            self._current_origin()
            if self._candidate_applied_scopes or self._candidate_applied_initial_keys
            else candidate.origin
        )
        if comparison_origin is None:
            return False
        active_dataset_ids = frozenset(
            item.dataset_id
            for item in candidate.setup_snapshot.datasets
            if item.included
        )
        try:
            current = capture_fit_origin(
                self._session.document,
                comparison_origin.tab_id,
                dataset_ids=active_dataset_ids,
            )
        except Exception:
            return False
        return fit_origin_science_matches(
            comparison_origin,
            current,
            included_dataset_ids=active_dataset_ids,
        )

    def _rebase_owner_snapshots(self, origin: FitOriginSnapshot) -> None:
        """Re-key owner-retained reset snapshots after a currentness rebase."""

        def rebased(snapshot: FitSetupSnapshot | None) -> FitSetupSnapshot | None:
            if snapshot is None:
                return None
            return replace(snapshot, origin=origin)

        self._initial_setup_snapshot = rebased(self._initial_setup_snapshot)
        self._last_fit_setup_snapshot = rebased(self._last_fit_setup_snapshot)

    def _build_request(self) -> FitExecutionRequest:
        setup = self._accepted_setup()
        origin = None if setup is None else setup.origin
        if origin is None:
            origin = self._current_origin()
        if origin is None or setup is None:
            raise RuntimeError("Fit has no typed local setup.")
        retry_state = self._retry_state()
        validated = build_fit_execution_request(
            origin=origin,
            mechanism_source=origin.mechanism_source,
            prepared_dsl=origin.prepared_dsl,
            solver=setup.solver,
            local_observables=setup.local_observables,
            datasets=setup.datasets,
            parameters=setup.parameters,
            initials=setup.initials,
            global_initials=setup.global_initials,
            initial_scopes=setup.initial_scopes,
            dataset_weight_mode=setup.dataset_weight_mode,
            defaults=setup.defaults,
            profile_likelihood=setup.profile_likelihood,
            residual_bootstrap=setup.residual_bootstrap,
            shared_refit_promotions=setup.shared_refit_promotions,
            shared_refit_source_result=(
                None if retry_state is None else retry_state.source_candidate.result
            ),
        )
        revalidate_fit_request_inclusions(
            document=self._session.document,
            request=validated,
        )
        return validated

    @staticmethod
    def _candidate_local_scalar_names(
        candidate: FitCandidate,
    ) -> frozenset[str]:
        names = [
            scalar_name
            for observable in candidate.local_observables
            for scalar_name in observable.scalar_names
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "Fit candidate repeats a Fit-local scalar identity."
            )
        return frozenset(names)

    @staticmethod
    def _candidate_publishable_parameter_values(
        candidate: FitCandidate,
    ) -> dict[str, float]:
        """Return fitted values owned by current canonical coordinates."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("Project parameter values require FitCandidate.")
        parameters_by_name = {
            item.name: item for item in candidate.setup_snapshot.parameters
        }
        values = {
            str(name): float(value)
            for name, value in candidate.result.shared_params.items()
            if (
                parameters_by_name.get(name) is not None
                and parameters_by_name[name].scope == "shared"
                and parameters_by_name[name].selected_config.fit
            )
        }
        centred_by_name = {
            item.parameter_name: item
            for item in candidate.result.centred_fixed_effects_settlements
        }
        for name, settlement in centred_by_name.items():
            parameter = parameters_by_name.get(name)
            if (
                parameter is not None
                and parameter.scope == "centred_fixed_effects"
                and parameter.selected_config.fit
            ):
                values[name] = float(settlement.overall_decoded_value)
        return values

    def _accepted_setup(self) -> FitSetupSnapshot | None:
        """Return the one accepted setup owner for the current route."""

        state = self._setup_state
        if isinstance(state, _SharedRefitRetryState):
            return state.accepted_setup
        if state is not None and not isinstance(state, FitSetupSnapshot):
            raise TypeError("Fit setup owner contains an invalid state.")
        return state

    def _retry_state(self) -> _SharedRefitRetryState | None:
        """Return retry-specific custody when the accepted setup is a retry."""

        state = self._setup_state
        return state if isinstance(state, _SharedRefitRetryState) else None

    def _current_origin(self) -> FitOriginSnapshot | None:
        """Return the concrete origin retained by the accepted setup."""

        setup = self._accepted_setup()
        return None if setup is None else setup.origin

    def _install_setup(
        self,
        setup: FitSetupSnapshot,
        *,
        retry_source: FitCandidate | None = None,
    ) -> None:
        """Install one ordinary setup or one complete Shared-refit transaction."""

        if not isinstance(setup, FitSetupSnapshot):
            raise TypeError("Accepted Fit setup must be FitSetupSnapshot.")
        if setup.shared_refit_promotions:
            if retry_source is None:
                raise ValueError(
                    "Promotion-bearing setup requires Shared-refit source custody."
                )
            self._setup_state = _SharedRefitRetryState(
                source_candidate=retry_source,
                promotions=setup.shared_refit_promotions,
                accepted_setup=setup,
            )
            return
        if retry_source is not None:
            raise ValueError(
                "Shared-refit source custody requires promotion-bearing setup."
            )
        self._setup_state = setup

    def _retire_shared_refit_retry(self) -> None:
        """Retire retry custody into one promotion-free ordinary setup."""

        retry_state = self._setup_state
        if not isinstance(retry_state, _SharedRefitRetryState):
            return
        setup = replace(
            retry_state.accepted_setup,
            shared_refit_promotions=(),
        )
        self._install_setup(setup)

    @staticmethod
    def _without_promoted_local_observables(
        setup: FitSetupSnapshot,
        promoted: Iterable[FitLocalObservable],
    ) -> FitSetupSnapshot:
        """Remove definitions and scalar rows that now belong to canonical DSL."""

        promoted_items = tuple(promoted)
        promoted_names = {item.name for item in promoted_items}
        promoted_scalar_names = {
            scalar_name
            for item in promoted_items
            for scalar_name in item.scalar_names
        }
        remaining_observables = tuple(
            item
            for item in setup.local_observables
            if item.name not in promoted_names
        )
        remaining_scalar_names = {
            scalar_name
            for item in remaining_observables
            for scalar_name in item.scalar_names
        }
        retired_scalar_names = promoted_scalar_names - remaining_scalar_names
        return replace(
            setup,
            local_observables=remaining_observables,
            parameters=tuple(
                item
                for item in setup.parameters
                if item.name not in retired_scalar_names
            ),
        )

    def _preserved_setup_with_terminal_values(
        self,
        preserved: FitSetupSnapshot,
        candidate: FitCandidate,
    ) -> FitSetupSnapshot:
        """Overlay terminal values using terminal scope/schema as authority.

        The pre-run setup is still authoritative for non-result controls, but
        it is not allowed to supply the shape of a fitted scalar.  In
        particular, a Shared-refit terminal row must not be reconstituted as
        a stale Per-dataset row merely because the setup object held before
        the refit used that scope.
        """

        terminal_parameters = {
            item.name: item
            for item in candidate.setup_snapshot.parameters
            if item.selected_config.fit
        }
        if len(terminal_parameters) != sum(
            1
            for item in candidate.setup_snapshot.parameters
            if item.selected_config.fit
        ):
            raise ValueError(
                "Fit terminal setup repeats a fitted parameter identity."
            )
        terminal_initials = {
            (item.set_id, item.species): item
            for item in candidate.setup_snapshot.initials
            if (
                item.fit
                and dict(candidate.setup_snapshot.initial_scopes).get(
                    item.species,
                    "local",
                )
                == "local"
            )
        }
        terminal_globals = {
            item.species: item
            for item in candidate.setup_snapshot.global_initials
            if item.fit
        }
        terminal_parameter_rows = tuple(candidate.setup_snapshot.parameters)
        if len({item.name for item in terminal_parameter_rows}) != len(terminal_parameter_rows):
            raise ValueError("Fit terminal setup repeats a parameter identity.")
        terminal_names = {item.name for item in terminal_parameter_rows}
        preserved_names = {item.name for item in preserved.parameters}
        if terminal_names != preserved_names:
            raise ValueError(
                "Fit terminal parameter schema no longer matches the retained "
                "Fit setup; Apply cannot construct a hybrid scope."
            )
        # Terminal rows carry the exact terminal scope, active configuration,
        # and fitted values.  Preserve their inactive scope configuration too:
        # it is part of the terminal schema and is needed by a later explicit
        # scope transition.
        parameters = terminal_parameter_rows
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
        global_initials = tuple(
            (
                replace(
                    item,
                    initial=terminal_globals[item.species].initial,
                )
                if item.species in terminal_globals
                else item
            )
            for item in preserved.global_initials
        )
        return self._setup_with_local_observables(
            replace(
                preserved,
                parameters=parameters,
                initials=initials,
                global_initials=global_initials,
                initial_scopes=preserved.initial_scopes,
            ),
            preserved.local_observables,
        )

    def _rebase_setup_after_apply(
        self,
        preserved: FitSetupSnapshot,
        origin: FitOriginSnapshot,
        canonicalized_scopes: frozenset[str],
        local_observables: tuple[FitLocalObservable, ...],
        *,
        selected_initial_keys: frozenset[tuple[str, str]] = frozenset(),
    ) -> FitSetupSnapshot:
        """Re-key retained setup without replacing inactive Local IC rows."""

        baseline = create_initial_fit_setup(origin)
        prior_datasets = {item.dataset_id: item for item in preserved.datasets}
        datasets = tuple(
            prior
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
                    _replace_fit_parameter(
                        prior,
                        initial=(
                            fresh.selected_config.value
                            if "parameters" in canonicalized_scopes
                            else prior.selected_config.value
                        ),
                        dataset_values=tuple(
                            (dataset_id, value)
                            for dataset_id, value
                            in prior.per_dataset_config.dataset_values
                            if dataset_id in retained_dataset_ids
                        )
                        if prior.scope != "shared"
                        else (),
                    ),
                    display_name=fresh.display_name,
                    family=fresh.family,
                    role=fresh.role,
                    unit=fresh.unit,
                )
            )
        local_scalar_names = {
            scalar_name
            for observable in local_observables
            for scalar_name in observable.scalar_names
        }
        supplemental = tuple(
            _replace_fit_parameter(
                item,
                dataset_values=tuple(
                    (dataset_id, value)
                    for dataset_id, value
                    in item.per_dataset_config.dataset_values
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
                    if (
                        "initial_concentrations" in canonicalized_scopes
                        and (fresh.set_id, fresh.species) in selected_initial_keys
                        and dict(preserved.initial_scopes).get(
                            fresh.species,
                            "local",
                        )
                        == "local"
                    )
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
        prior_globals = {item.species: item for item in preserved.global_initials}
        prior_scopes = dict(preserved.initial_scopes)
        initial_scopes = tuple(
            (
                species,
                (
                    "global"
                    if prior_scopes.get(species) == "global"
                    and species in prior_globals
                    else "local"
                ),
            )
            for species, _scope in baseline.initial_scopes
        )
        active_global_species = {
            species
            for species, scope in initial_scopes
            if scope == "global"
        }
        global_initials = tuple(
            prior_globals[species]
            for species in sorted(active_global_species)
        )
        # Validate the retained typed local plan against the *new* canonical
        # source before constructing the replacement.  The constructor below
        # receives the complete local-observable/parameter pair in one call;
        # there is deliberately no intermediate canonical-only snapshot for
        # FitSetupSnapshot.__post_init__ to misclassify.
        build_fit_local_admission(
            origin.prepared_dsl,
            local_observables,
            (*canonical_parameters, *supplemental),
        )
        rebased = replace(
            baseline,
            origin=origin,
            datasets=datasets,
            solver=preserved.solver,
            parameters=(*canonical_parameters, *supplemental),
            initials=initials,
            global_initials=global_initials,
            initial_scopes=initial_scopes,
            defaults=preserved.defaults,
            dataset_weight_mode=preserved.dataset_weight_mode,
            profile_likelihood=preserved.profile_likelihood,
            residual_bootstrap=preserved.residual_bootstrap,
            shared_refit_promotions=preserved.shared_refit_promotions,
            local_observables=local_observables,
        )
        return rebased

    def _reset_owner_snapshot_after_apply(
        self,
        snapshot: FitSetupSnapshot | None,
        *,
        origin: FitOriginSnapshot,
        canonicalized_scopes: frozenset[str],
        promoted: tuple[FitLocalObservable, ...],
        selected_initial_keys: frozenset[tuple[str, str]] = frozenset(),
    ) -> FitSetupSnapshot | None:
        """Rebase one Reset owner while retiring newly canonical locals."""

        if snapshot is None:
            return None
        retained = replace(snapshot, origin=origin)
        if "observable_definitions" in canonicalized_scopes:
            retained = self._without_promoted_local_observables(
                retained,
                promoted,
            )
        return self._rebase_setup_after_apply(
            retained,
            origin,
            canonicalized_scopes,
            retained.local_observables,
            selected_initial_keys=selected_initial_keys,
        )

    def current_initial_apply_proposal(self) -> FitInitialApplyProposal | None:
        """Return the complete remaining, independently selectable IC proposal."""

        candidate = self._candidate
        if candidate is None:
            return None
        proposal = self._initial_apply_proposal(
            candidate,
            excluded_keys=self._candidate_applied_initial_keys,
        )
        origin = self._current_origin()
        if origin is None:
            return None
        return replace(proposal, origin=origin)

    @classmethod
    def _candidate_initial_application(
        cls,
        candidate: FitCandidate,
    ) -> tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], object],
        dict[str, tuple[str, ...]],
    ]:
        """Validate terminal IC identity and return one canonical cell map."""

        origin = candidate.origin
        nominal_setup = candidate.nominal_setup_snapshot
        included_dataset_ids = {
            item.dataset_id
            for item in nominal_setup.datasets
            if item.included
        }
        included_mapped = tuple(
            item
            for item in origin.mapped_datasets
            if item.dataset_id in included_dataset_ids
        )
        included_set_ids = {item.set_id for item in included_mapped}
        datasets_by_set: dict[str, tuple[str, ...]] = {
            mapped.set_id: tuple(
                sorted(
                    item.dataset_id
                    for item in included_mapped
                    if item.set_id == mapped.set_id
                )
            )
            for mapped in included_mapped
        }
        selected_scopes = dict(nominal_setup.initial_scopes)
        expected: dict[tuple[str, str], object] = {}
        for item in nominal_setup.initials:
            key = (item.set_id, item.species)
            if (
                item.fit
                and item.set_id in included_set_ids
                and selected_scopes.get(item.species) == "local"
            ):
                if key in expected:
                    raise ValueError(
                        "Fit terminal setup repeats a fitted initial identity."
                    )
                expected[key] = item
        for item in nominal_setup.global_initials:
            if not item.fit or selected_scopes.get(item.species) != "global":
                continue
            for mapped in included_mapped:
                if item.species not in dict(mapped.initial_values):
                    continue
                key = (mapped.set_id, item.species)
                if key in expected:
                    raise ValueError(
                        "Fit terminal setup repeats a fitted initial identity."
                    )
                expected[key] = item

        actual: dict[tuple[str, str], float] = {}
        seen_set_ids: set[str] = set()
        for set_id, values in candidate.initial_values_by_set:
            if set_id in seen_set_ids:
                raise ValueError(f"Fit candidate repeats initial set {set_id!r}.")
            seen_set_ids.add(set_id)
            seen_species: set[str] = set()
            for species, raw_value in values:
                if species in seen_species:
                    raise ValueError(
                        f"Fit candidate repeats initial {(set_id, species)!r}."
                    )
                seen_species.add(species)
                value = validate_concentration(
                    raw_value,
                    field=f"fitted initial {(set_id, species)!r}",
                )
                actual[(set_id, species)] = value
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            details = []
            if missing:
                details.append(f"missing={missing!r}")
            if unexpected:
                details.append(f"unexpected={unexpected!r}")
            raise ValueError(
                "Fit candidate initial identities do not exactly match the "
                "fitted terminal cells: " + "; ".join(details)
            )
        return actual, expected, datasets_by_set

    @classmethod
    def _initial_apply_proposal(
        cls,
        candidate: FitCandidate,
        *,
        excluded_keys: frozenset[tuple[str, str]] = frozenset(),
    ) -> FitInitialApplyProposal:
        actual, expected, datasets_by_set = cls._candidate_initial_application(
            candidate
        )
        captured_by_key = {
            (mapped.set_id, species): float(value)
            for mapped in candidate.origin.mapped_datasets
            for species, value in mapped.initial_values
        }
        cells: list[FitInitialApplyCellProposal] = []
        for key in sorted(expected):
            if key in excluded_keys:
                continue
            set_id, species = key
            configuration = expected[key]
            cells.append(
                FitInitialApplyCellProposal(
                    set_id=set_id,
                    species=species,
                    dataset_ids=datasets_by_set.get(set_id, ()),
                    captured_value=captured_by_key[key],
                    nominal_value=float(getattr(configuration, "initial")),
                    fitted_value=actual[key],
                    minimum=float(getattr(configuration, "minimum")),
                    maximum=float(getattr(configuration, "maximum")),
                    log10=bool(getattr(configuration, "log10")),
                )
            )
        return FitInitialApplyProposal(
            origin=candidate.origin,
            cells=tuple(cells),
        )

    def _candidate_apply_feasibility(
        self,
        candidate: FitCandidate,
        *,
        applied_scopes: frozenset[str] = frozenset(),
        applied_initial_keys: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[frozenset[str], dict[str, str]]:
        """Decide terminal Apply feasibility once in the sole Fit owner."""

        local_scalar_names = self._candidate_local_scalar_names(candidate)
        publishable_values = self._candidate_publishable_parameter_values(candidate)
        terminal_parameters = {
            item.name: item
            for item in candidate.setup_snapshot.parameters
        }
        available: set[str] = set()
        unavailable: dict[str, str] = {}
        if not self._candidate_result_is_apply_eligible(candidate):
            reason = (
                "Terminal optimizer did not converge; values are finite diagnostic "
                "seed material only and cannot be applied."
            )
            unavailable.update(
                {
                    "parameters": reason,
                    "initial_concentrations": reason,
                    "observable_definitions": reason,
                }
            )
            return frozenset(), unavailable
        if any(
            name not in local_scalar_names
            and terminal_parameters.get(name) is not None
            and name in publishable_values
            for name in publishable_values
        ):
            available.add("parameters")
        else:
            has_per_dataset_values = any(
                item.selected_config.fit
                and item.scope == "per_dataset"
                for item in candidate.setup_snapshot.parameters
            )
            unavailable["parameters"] = (
                "Parameters Apply is unavailable because the candidate "
                "contains no fitted shared project parameters."
                + (
                    " Per-dataset values remain Fit-local and are not "
                    "applied to Run."
                    if has_per_dataset_values
                    else ""
                )
            )
        try:
            initial_proposal = self._initial_apply_proposal(
                candidate,
                excluded_keys=applied_initial_keys,
            )
        except (KeyError, TypeError, ValueError) as exc:
            unavailable["initial_concentrations"] = (
                "Initial Conditions Apply is unavailable because the terminal "
                f"cell proposal is invalid: {exc}"
            )
        else:
            if initial_proposal.cells:
                available.add("initial_concentrations")
            else:
                unavailable["initial_concentrations"] = (
                    "Initial Conditions Apply is unavailable because the "
                    "candidate contains no remaining fitted initial cells. "
                    "Fit-local simulation start is never published to Run settings."
                )
        if candidate.local_observables:
            try:
                self._validate_terminal_observable_definitions(candidate)
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
        applied = frozenset(str(scope) for scope in applied_scopes)
        available.difference_update(applied)
        for scope in applied:
            unavailable[scope] = "This Fit scope has already been applied."
        return frozenset(available), unavailable

    @staticmethod
    def _candidate_result_is_apply_eligible(
        candidate: FitCandidate,
    ) -> bool:
        """Share Apply's canonical result gate with Shared-refit admission."""

        result = candidate.result
        return bool(
            result.optimizer_converged
            and not result.diagnostic_only
            and result.optimizer_diagnostic is None
            and result.status == "ok"
        )

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
            for scalar_name in observable.scalar_names:
                parameter = parameters.get(scalar_name)
                if parameter is None:
                    raise ValueError(
                        f"Fit-local scalar {scalar_name!r} has no terminal "
                        "parameter row."
                    )
                cls._terminal_observable_scalar_value(candidate, parameter)

    @staticmethod
    def _terminal_observable_scalar_value(
        candidate: FitCandidate,
        parameter: FitParameterVariable,
    ) -> float:
        """Resolve one promotable scalar over the included terminal roster."""

        if parameter.scope != "shared":
            raise ValueError(
                "Observable Definitions Apply cannot publish Per-dataset "
                f"scalar {parameter.name!r}; run an explicit Shared refit."
            )
        try:
            value = candidate.result.shared_params[parameter.name]
        except KeyError as exc:
            raise ValueError(
                f"Fit-local scalar {parameter.name!r} has no fitted value."
            ) from exc
        return admit_gui_parameter_value(
            value,
            scientific_category=parameter.scientific_category,
            allow_exact_zero=parameter.allow_exact_zero,
            field=f"Fit-local scalar {parameter.name!r} terminal value",
        )

    @classmethod
    def _build_applied_tab(
        cls,
        tab: MechanismTabDocument,
        candidate: FitCandidate,
        scopes: frozenset[str],
        origin: FitOriginSnapshot,
        *,
        selected_initial_keys: frozenset[tuple[str, str]] = frozenset(),
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
            local_scalar_names = cls._candidate_local_scalar_names(candidate)
            promoted_parameters = tuple(
                _replace_fit_parameter(
                    parameter,
                    initial=cls._terminal_observable_scalar_value(
                        candidate,
                        parameter,
                    ),
                )
                if parameter.name in local_scalar_names
                else parameter
                for parameter in candidate.setup_snapshot.parameters
            )
            admission = candidate.request.fit_local_admission
            if admission is None or tuple(admission.observables) != tuple(
                candidate.local_observables
            ):
                raise ValueError(
                    "Observable Definitions Apply lost its admitted Algebra program."
                )
            applied_source, prepared = apply_admitted_algebra(
                source=next_tab.mechanism_source.reactions_text,
                base=prepared,
                admission=admission,
                scalar_values={
                    name: cls._terminal_observable_scalar_value(
                        candidate,
                        next(
                            parameter
                            for parameter in promoted_parameters
                            if parameter.name == name
                        ),
                    )
                    for name in admission.scalar_names
                },
            )
            next_tab = replace(
                next_tab,
                mechanism_source=MechanismAuthoringSource.from_parts(
                    reactions_text=applied_source
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
                name: value
                for name, value in cls._candidate_publishable_parameter_values(candidate).items()
                if (
                    name not in local_scalar_names
                    and expected.get(name) is not None
                )
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
                    or coordinate.scientific_category
                    != captured.scientific_category
                    or coordinate.allow_exact_zero
                    != captured.allow_exact_zero
                    or not coordinate.writable
                    or coordinate.derived
                ):
                    raise ValueError(
                        f"Fitted parameter {name!r} changed typed family, "
                        "origin, or write ownership."
                    )
                writable_values[name] = admit_gui_parameter_value(
                    writable_values[name],
                    scientific_category=coordinate.scientific_category,
                    allow_exact_zero=coordinate.allow_exact_zero,
                    field=f"fitted parameter {name!r}",
                )
            updates = tuple(
                DslParameterCoordinateUpdate(
                    name=name,
                    value=value,
                )
                for name, value in sorted(writable_values.items())
            )
            if updates:
                derived_source, prepared = derive_source_with_parameter_updates(
                    next_tab.mechanism_source.reactions_text,
                    prepared,
                    updates,
                    wegscheider_cyclicity_enabled=wegscheider_policy,
                )
                derived_source = MechanismAuthoringSource.from_parts(
                    reactions_text=derived_source,
                )
                next_tab = replace(
                    next_tab,
                    mechanism_source=derived_source,
                    prepared_dsl=prepared,
        )
        if "initial_concentrations" in scopes:
            # The terminal candidate remains immutable across partial Apply.
            # The current origin owns the document mutation; the candidate
            # owns the fitted cell values and immutable terminal identities.
            if candidate.origin.tab_id != origin.tab_id:
                raise ValueError("Fit candidate IC proposal changed tab origin.")
            actual_values, expected, _datasets_by_set = (
                cls._candidate_initial_application(candidate)
            )
            expected_keys = set(expected)
            if not selected_initial_keys or not selected_initial_keys <= expected_keys:
                raise ValueError(
                    "Initial Conditions Apply must select available fitted cells."
                )
            current_set_ids = {
                item.set_id for item in next_tab.initial_condition_sets
            }
            missing_sets = sorted(
                {set_id for set_id, _species in selected_initial_keys}
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
                for species, value in initial_set.values:
                    validate_concentration(
                        value,
                        field=(
                            f"initial concentration {species!r} in "
                            f"{initial_set.name!r}"
                        ),
                    )
                fitted_names = {
                    species
                    for set_id, species in selected_initial_keys
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
                        (
                            actual_values[(initial_set.set_id, name)]
                            if (initial_set.set_id, name) in selected_initial_keys
                            else float(current_value)
                        ),
                    )
                    for name, current_value in initial_set.values
                )
                for species, value in merged:
                    validate_concentration(
                        value,
                        field=(
                            f"merged initial concentration {species!r} in "
                            f"{initial_set.name!r}"
                        ),
                    )
                updated.append(
                    initial_set.with_values(merged)
                )
            next_tab = next_tab.with_initial_condition_sets(updated)
        return next_tab

    def _set_status(self, text: str) -> None:
        if self._application_close_fenced:
            return
        message = str(text)
        origin = self._current_origin()
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
