"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    FitDefaults,
    MechanismSourceCollision,
    reconcile_tab_mechanism_source,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_session import ProjectSession
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    derive_source_with_parameter_updates,
)
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.parameter_enumeration import admit_gui_parameter_value
from kindred.gui.fitting.launch import (
    FitExecutionRequest,
    FitOriginSnapshot,
    FitParameterVariable,
    FitSetupSnapshot,
    build_fit_execution_request,
    capture_fit_origin,
    create_initial_fit_setup,
    prune_fit_analysis_coordinates,
    convert_fit_parameter_scope,
)
from kindred.gui.fitting.reporting import fit_completion_presentation
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitLiveUpdate,
    GlobalFitWorker,
    project_fit_candidate_curves,
)
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry

__all__ = [
    "FitSessionOwner",
]

WindowFactory = Callable[..., FittingWindow]
ColorResolver = Callable[[str, Sequence[str]], object]


class FitSessionOwner(QtCore.QObject):
    """Retain the window, accepted setup, worker, and candidate custody."""

    window_changed = QtCore.Signal(bool)
    execution_changed = QtCore.Signal(bool)
    status_changed = QtCore.Signal(str)
    project_document_changed = QtCore.Signal(object)
    application_close_settlement_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        session: ProjectSession,
        registry: WorkspaceTabRegistry,
        gate: ComputationGate,
        observation_style_getter: Callable[[], tuple[str, float, float]],
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
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        if not callable(window_factory):
            raise TypeError("window_factory must be callable.")
        self._session = session
        self._registry = registry
        self._gate = gate
        self._color_resolver = color_resolver
        self._observation_style_getter = observation_style_getter
        self._parent_widget = parent_widget
        self._window_factory = window_factory
        self._window: FittingWindow | None = None
        self._worker: GlobalFitWorker | None = None
        self._candidate: FitCandidate | None = None
        self._setup_state: FitSetupSnapshot | None = None
        self._initial_setup_snapshot: FitSetupSnapshot | None = None
        self._last_fit_setup_snapshot: FitSetupSnapshot | None = None
        self._closing = False
        self._application_close_fenced = False
        self._application_close_settlement_pending = False

    def refresh_plot_preferences(self) -> None:
        if self._window is not None:
            self._window.set_observation_style(*self._observation_style_getter())

    @property
    def project_defaults(self) -> FitDefaults:
        return self._session.document.fit_defaults

    @property
    def active(self) -> bool:
        """Return whether Fit owns an open window or worker."""

        return bool(self._window is not None or self._worker is not None)

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

    @QtCore.Slot(object)
    def open_fit_configuration(
        self,
        tab_id: str | None = None,
    ) -> FittingWindow | None:
        """Open an idle configuration from one admitted document origin."""

        self._require_thread()
        if (
            self._application_close_fenced
            or self._closing
            or self._window is not None
            or self._worker is not None
        ):
            return None
        if not self._gate.is_idle:
            self._set_status(
                "Fit Open rejected: "
                f"{self._gate.active_kind or 'another computation'} is active."
            )
            return None
        requested_id = str(tab_id or self._registry.active_tab_id)
        try:
            origin = capture_fit_origin(self._session.document, requested_id)
            setup = create_initial_fit_setup(origin)
            analysis = origin.prepared_dsl.analysis
            modeled_species = tuple(
                dict.fromkeys(
                    (*analysis.species_names, *analysis.observable_names)
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
                observation_style=self._observation_style_getter(),
                parent=self._parent_widget,
            )
        except (TypeError, ValueError, KeyError, LookupError) as exc:
            self._set_status(f"Fit Open failed: {exc}")
            return None
        if not isinstance(window, FittingWindow):
            raise TypeError("window_factory must return FittingWindow.")

        self._window = window
        self._install_setup(setup)
        self._initial_setup_snapshot = setup
        self._last_fit_setup_snapshot = None
        self._candidate = None
        try:
            window.project_setup_snapshot(setup)
            window.set_initial_reset_available(setup)
            window.set_shared_seed_available(False)
            window.set_apply_available(False, "Run a fit to obtain values to apply.")
        except (TypeError, ValueError, KeyError) as exc:
            window.deleteLater()
            self._window = None
            self._setup_state = None
            self._initial_setup_snapshot = None
            self._last_fit_setup_snapshot = None
            self._set_status(f"Fit Open failed: {exc}")
            return None

        window.executeRequested.connect(self.execute)
        window.stopRequested.connect(self.stop)
        window.pauseRequested.connect(self.pause)
        window.resumeRequested.connect(self.resume)
        window.resultsSummaryRequested.connect(self.results_summary)
        window.sharedSeedRequested.connect(self.seed_shared_fit)
        window.applyRequested.connect(self.apply_fit_to_project)
        window.setupSnapshotRequested.connect(self._on_setup_snapshot)
        window.resetRequested.connect(self.reset_setup)
        window.destroyed.connect(self._on_window_destroyed)
        self.window_changed.emit(True)
        window.show()
        window.raise_()
        window.activateWindow()
        self._set_status(f"Fit opened for {origin.tab_name}.")
        return window

    @QtCore.Slot()
    def execute(self) -> bool:
        """Admit one Fit worker from the current accepted setup."""
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
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup is invalid: {exc}")
            return False
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
            worker.candidate_ready.connect(
                lambda candidate, w=worker: self._on_candidate(w, candidate)
            )
            worker.factual_failure.connect(
                lambda failure, w=worker: self._on_failure(w, failure)
            )
            worker.scientific_unavailable.connect(
                lambda unavailable, w=worker: self._on_unavailable(w, unavailable)
            )
            worker.cancelled.connect(
                lambda w=worker: self._on_cancelled(w)
            )
            worker.finished.connect(
                lambda w=worker, r=request: self._finish_worker(w, r)
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
                self._set_status(
                    f"Fit dispatch failed before QThread start: {exc}"
                )
                return False

            if running is True or finished is True:
                window.set_executing(True)
                self._install_started_request_state(
                    setup=setup,
                    window=window,
                )
                custody_state = "running" if running is True else "finished"
            else:
                # Native state is ambiguous: retain the exact request/setup
                # before any queued worker callback can publish against the
                # old local state.  Worker and gate custody remains retained.
                window.set_executing(True)
                self._install_started_request_state(
                    setup=setup,
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
            setup=setup,
            window=window,
        )
        self.execution_changed.emit(True)
        self._set_status("Fit started.", phase="Running", detail="Waiting for the first complete candidate")
        return True

    def _install_started_request_state(
        self,
        *,
        setup: FitSetupSnapshot,
        window: FittingWindow,
    ) -> None:
        """Install the admitted setup only after native start is known."""

        candidate = self._candidate
        if candidate is not None and candidate.setup_snapshot != setup:
            self._invalidate_candidate_custody()
        self._install_setup(setup)
        # Execution owns the physical presentation until terminal settlement;
        # custody remains in ``self._candidate`` for a failed or cancelled
        # replacement attempt.
        self._clear_candidate_presentation(window)

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
        if snapshot == previous:
            return False
        had_candidate = self._candidate is not None
        self._install_setup(snapshot)
        self._invalidate_candidate_custody()
        if had_candidate:
            self._clear_candidate_presentation(window)
        window.project_setup_snapshot(snapshot)
        return True

    @QtCore.Slot()
    def stop(self) -> bool:
        if self._closing or self._application_close_fenced:
            return False
        worker = self._worker
        if worker is None:
            return False
        worker.request_stop()
        self._set_status("Fit stop requested; retaining this run's best complete candidate.",
                         phase="Stopping", detail="Best complete candidate will be retained")
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
        self._set_status(
            "Fit pause requested; the current evaluation will finish before execution pauses.",
            phase="Pause requested", detail="Current evaluation is finishing"
        )
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
        self._set_status("Fit resumed.", phase="Running", detail="")
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
            window.clear_pending_setup_edits()
            self._clear_candidate_presentation(window)
            window.project_setup_snapshot(snapshot)
        except (TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Fit setup reset failed: {exc}")
            return False
        self._install_setup(snapshot)
        self._invalidate_candidate_custody()
        self._set_status(f"Fit setup {kind.replace('_', ' ')} snapshot restored.")
        return True

    def _clear_candidate_presentation(self, window: FittingWindow) -> None:
        """Clear the candidate presentation."""

        if not isinstance(window, FittingWindow):
            raise TypeError("Fit candidate presentation requires FittingWindow.")
        window.clear_candidate()
        window.set_apply_available(False, "Run a fit to obtain current values to apply.")

    @QtCore.Slot()
    def results_summary(self) -> bool:
        """Render the human-readable Summary from retained Fit facts."""

        if self._window is None:
            return False
        # Results already projects the current execution or terminal. Retained
        # candidate custody may still belong to a previous attempt during retry.
        self._window.render_results_summary()
        return True

    @staticmethod
    def _candidate_curves(
        candidate: FitCandidate,
    ) -> tuple[tuple[object, ...], str | None]:
        """Project terminal curves independently from report/UI admission."""

        try:
            curves = tuple(project_fit_candidate_curves(candidate))
        except Exception as exc:  # noqa: BLE001
            return (), f"{type(exc).__name__}: {exc}"
        if not curves:
            return (), "Fit result produced no curve datasets."
        return curves, None

    def _invalidate_candidate_custody(self) -> None:
        """Release the current terminal candidate."""

        self._candidate = None

    def set_project_defaults(self, defaults: FitDefaults) -> bool:
        self._require_thread()
        if self._closing or self._application_close_fenced:
            return False
        if not isinstance(defaults, FitDefaults):
            raise TypeError("defaults must be FitDefaults.")
        expected_token = self._session.capture_content_token()
        try:
            successor_token = self._registry.replace_document(
                replace(self._session.document, fit_defaults=defaults),
                expected_token=expected_token,
            )
            if successor_token is None:
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
        if successor_token == expected_token:
            return False
        self.project_document_changed.emit(None)
        self._set_status("Current-project Fit defaults updated.")
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
            worker.request_cancel()

    def _request_window_close(self) -> bool:
        if self._closing:
            return False
        self._closing = True
        if self._worker is not None:
            if self._window is not None:
                self._window.enter_stopping()
            self._worker.request_cancel()
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
        if self._window is not None and value < 100:
            self._window.set_progress(value)
        self._set_status(text, present=value < 100)

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

    def _on_auto_modes(
        self,
        worker: GlobalFitWorker,
        modes: tuple[tuple[str, str], ...],
    ) -> None:
        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        if self._window is not None:
            self._window.set_resolved_auto_modes(modes)

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

    def _on_candidate(self, worker: GlobalFitWorker, candidate: object) -> None:
        """Retain and present the worker's direct typed candidate terminal."""

        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        self._candidate = candidate
        # Keep terminal actions disabled until the worker's finished callback
        # releases execution custody.  The candidate itself remains in owner
        # custody until finished releases the worker.
        self._set_status(candidate.result.message, present=False)

    def _present_candidate(
        self,
        candidate: FitCandidate,
    ) -> None:
        """Project one retained candidate into the terminal Fit presentation."""

        terminal_setup = candidate.setup_snapshot
        self._last_fit_setup_snapshot = terminal_setup
        self._install_setup(terminal_setup)
        window = self._window
        if window is None:
            return
        try:
            window.project_setup_snapshot(terminal_setup)
            window.set_last_fit_reset_available(terminal_setup)
            curves, curve_failure = self._candidate_curves(candidate)
            _completion, complete = fit_completion_presentation(candidate)
            if complete:
                window.set_progress(100)
            window.set_terminal_presentation(
                candidate,
                curves=curves,
                curve_failure=curve_failure,
            )
            apply_reason = self._candidate_apply_reason(candidate)
            window.set_apply_available(not apply_reason, apply_reason)
        except Exception as exc:
            self._set_status(f"Fit result presentation failed: {exc}")

    @QtCore.Slot(str)
    def seed_shared_fit(self, statistic: str) -> bool:
        """Convert local dataset-based values into an idle Shared setup."""

        self._require_thread()
        setup = self._accepted_setup()
        origin = self._current_origin()
        window = self._window
        if (
            self._closing
            or self._application_close_fenced
            or setup is None
            or origin is None
            or window is None
            or self._worker is not None
        ):
            return False
        if not self._setup_can_seed_shared(setup):
            self._set_status(
                "Shared Fit setup rejected: no local dataset-based values are available."
            )
            return False
        statistic = str(statistic or "").strip().lower()
        if statistic not in {"arithmetic", "geometric"}:
            self._set_status("Shared Fit setup rejected: unknown seed statistic.")
            return False
        parameters: list[FitParameterVariable] = []
        changed_parameter_names: list[str] = []
        try:
            for parameter in setup.parameters:
                if parameter.scope == "shared":
                    parameters.append(parameter)
                    continue
                parameters.append(
                    convert_fit_parameter_scope(parameter, "shared", setup.datasets, statistic=statistic)
                )
                changed_parameter_names.append(parameter.name)
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
            self._set_status(f"Shared Fit setup rejected: {exc}")
            return False
        shared_setup = replace(
            setup,
            parameters=tuple(parameters),
            profile_likelihood=prune_fit_analysis_coordinates(
                setup.profile_likelihood,
                changed_parameter_names,
            ),
            residual_bootstrap=prune_fit_analysis_coordinates(
                setup.residual_bootstrap,
                changed_parameter_names,
            ),
        )
        self._install_setup(shared_setup)
        self._invalidate_candidate_custody()
        self._clear_candidate_presentation(window)
        window.project_setup_snapshot(shared_setup)
        window.set_shared_seed_available(False)
        self._set_status(
            "Shared Fit setup prepared from "
            f"{statistic} Per-dataset estimates; press Run Fit to execute."
        )
        return True

    @classmethod
    def _candidate_publishable_parameter_values(
        cls,
        candidate: FitCandidate,
    ) -> dict[str, float]:
        """Return complete canonical Shared/centred parameter values."""

        parameters = {item.name: item for item in candidate.setup_snapshot.parameters}
        if len(parameters) != len(candidate.setup_snapshot.parameters):
            raise ValueError("Fit terminal setup repeats a parameter identity.")
        values: dict[str, float] = {}
        for parameter in candidate.setup_snapshot.parameters:
            if (
                not parameter.selected_config.fit
            ):
                continue
            if parameter.scope == "shared":
                raw_value = candidate.result.shared_params.get(parameter.name)
                if raw_value is None:
                    raise ValueError(
                        f"Fit result omitted shared parameter {parameter.name!r}."
                    )
            elif parameter.scope == "centred_fixed_effects":
                settlements = tuple(
                    item
                    for item in candidate.result.centred_fixed_effects_settlements
                    if item.parameter_name == parameter.name
                )
                if len(settlements) != 1:
                    raise ValueError(
                        f"Fit result omitted centred parameter {parameter.name!r}."
                    )
                raw_value = settlements[0].overall_decoded_value
            else:
                continue
            values[parameter.name] = admit_gui_parameter_value(
                raw_value,
                scientific_category=parameter.scientific_category,
                allow_exact_zero=parameter.allow_exact_zero,
                field=f"fitted parameter {parameter.name!r}",
            )
        return values

    @staticmethod
    def _setup_can_seed_shared(setup: FitSetupSnapshot) -> bool:
        return bool(
            any(
                parameter.scope != "shared"
                for parameter in setup.parameters
            )
        )

    @classmethod
    def _candidate_apply_reason(cls, candidate: FitCandidate) -> str:
        if any(parameter.scope == "per_dataset" and parameter.selected_config.fit
               for parameter in candidate.setup_snapshot.parameters):
            return "Configure an ordinary Shared Fit before applying per-dataset parameters."
        try:
            parameter_values = cls._candidate_publishable_parameter_values(candidate)
            initial_values = candidate.initial_values_by_set
        except (TypeError, ValueError, KeyError) as exc:
            return f"Fit values cannot be applied: {exc}"
        return "" if parameter_values or initial_values else "This fit has no fitted values to apply."

    @QtCore.Slot()
    def apply_fit_to_project(self) -> bool:
        """Apply one complete, current terminal Fit candidate atomically."""

        self._require_thread()
        candidate = self._candidate
        origin = self._current_origin()
        setup = self._accepted_setup()
        window = self._window
        if (
            self._closing
            or self._application_close_fenced
            or candidate is None
            or origin is None
            or setup is None
            or window is None
            or self._worker is not None
        ):
            return False
        if not self._origin_science_is_current(candidate.origin):
            self._set_status("Fit Apply rejected: origin science is no longer current.")
            return False
        if any(
            parameter.scope == "per_dataset"
            and parameter.selected_config.fit
            for parameter in candidate.setup_snapshot.parameters
        ):
            self._set_status(
                "Fit Apply rejected: configure an ordinary Shared Fit before applying."
            )
            return False
        try:
            parameter_values = self._candidate_publishable_parameter_values(candidate)
            centred_provenance = self._candidate_centred_fixed_effects_provenance(candidate)
            initial_values = {
                (set_id, species): value
                for set_id, values in candidate.initial_values_by_set
                for species, value in values
            }
            source_text = origin.mechanism_source.reactions_text
            updates = tuple(
                DslParameterCoordinateUpdate(name=name, value=value)
                for name, value in sorted(parameter_values.items())
            )
            if updates:
                source_text = derive_source_with_parameter_updates(
                    source_text,
                    origin.prepared_dsl,
                    updates,
                )
            source_text = self._append_centred_fixed_effects_provenance(source_text, centred_provenance)
            reconciliation = reconcile_tab_mechanism_source(
                self._registry.document,
                origin.tab_id,
                MechanismAuthoringSource.from_parts(reactions_text=source_text),
            )
            if isinstance(reconciliation, MechanismSourceCollision):
                reconciliation = reconciliation.replacement
            proposed_document = reconciliation.proposed_document
            proposed_tab = proposed_document.tab(origin.tab_id)
            if initial_values:
                found_keys: set[tuple[str, str]] = set()
                proposed_sets = []
                for initial_set in proposed_tab.initial_condition_sets:
                    values = initial_set.values_dict()
                    for key, value in initial_values.items():
                        if key[0] != initial_set.set_id:
                            continue
                        # Fit captures the complete canonical species roster;
                        # assigning here materializes an authored omission
                        # whose accepted initial concentration was exact zero.
                        values[key[1]] = value
                        found_keys.add(key)
                    proposed_sets.append(initial_set.with_values(values))
                if found_keys != set(initial_values):
                    raise ValueError("Fit Apply initial-condition identity is stale.")
                proposed_document = proposed_document.replace_tab(
                    proposed_tab.with_initial_condition_sets(tuple(proposed_sets))
                )
            successor_origin = capture_fit_origin(proposed_document, origin.tab_id)
            rebased_setup = replace(setup, origin=successor_origin)
            rebased_initial_setup = (
                None
                if self._initial_setup_snapshot is None
                else replace(
                    self._initial_setup_snapshot,
                    origin=successor_origin,
                )
            )
            rebased_last_fit_setup = replace(
                candidate.setup_snapshot,
                origin=successor_origin,
            )
        except (TypeError, ValueError, KeyError, LookupError) as exc:
            self._set_status(f"Fit Apply rejected: {exc}")
            return False
        token = self._session.capture_content_token()
        if self._registry.replace_document(proposed_document, expected_token=token) is None:
            self._set_status("Fit Apply rejected: the captured project is stale.")
            return False
        self._install_setup(rebased_setup)
        self._initial_setup_snapshot = rebased_initial_setup
        self._last_fit_setup_snapshot = rebased_last_fit_setup
        window.project_setup_snapshot(rebased_setup)
        window.set_last_fit_reset_available(rebased_last_fit_setup)
        self.project_document_changed.emit(origin.tab_id)
        window.set_apply_available(False, "These Fit values have already been applied to the project.")
        self._set_status("Centred Fit applied: overall values published; dataset estimates not applied (recorded as mechanism-source provenance)." if centred_provenance else "Fit candidate applied to the project.")
        return True

    @staticmethod
    def _candidate_centred_fixed_effects_provenance(candidate: FitCandidate) -> tuple[str, ...]:
        """Build non-executable provenance for fitted centred parameters."""

        settlements = tuple(sorted(candidate.result.centred_fixed_effects_settlements, key=lambda item: item.parameter_name))
        if not settlements:
            return ()
        display_names = {mapped.dataset_id: mapped.display_name for mapped in candidate.origin.mapped_datasets}
        return tuple(
            f"# Centred Fit applied: overall {settlement.parameter_name} = "
            f"{settlement.overall_decoded_value:.17g}; "
            "dataset estimates not applied: "
            + ", ".join(
                (
                    " ".join(str(display_names[dataset_id]).splitlines())
                    + f" = {settlement.dataset_values[dataset_id]:.17g}"
                )
                for dataset_id in settlement.dataset_ids
            )
            for settlement in settlements
        )

    @staticmethod
    def _append_centred_fixed_effects_provenance(
        source: str,
        comments: Sequence[str],
    ) -> str:
        if not comments:
            return source
        newline = "\r\n" if "\r\n" in source else "\n"
        separator = "" if source.endswith(("\n", "\r")) else newline
        return source + separator + newline.join(comments)

    def _on_failure(self, worker: GlobalFitWorker, failure: object) -> None:
        """Present the worker's direct factual failure terminal once."""

        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        if self._window is not None:
            self._clear_candidate_presentation(self._window)
            self._window.set_terminal_failure(failure)
        self._set_status(f"Fit failed: {failure.message}", present=False)

    def _on_unavailable(
        self,
        worker: GlobalFitWorker,
        unavailable: object,
    ) -> None:
        """Present the worker's direct scientific-unavailability terminal."""

        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        if self._window is not None:
            self._window.set_terminal_unavailable(unavailable)
        cause = unavailable.factual_cause
        self._set_status(
            "Fit result scientifically unavailable: "
            + (cause.message if cause is not None else unavailable.stage), present=False
        )

    def _on_cancelled(self, worker: GlobalFitWorker) -> None:
        """Publish the direct cancellation terminal once."""

        if (
            self._application_close_fenced
            or self._closing
            or self._worker is not worker
        ):
            return
        if self._window is not None:
            self._clear_candidate_presentation(self._window)
        self._invalidate_candidate_custody()
        self._set_status("Fit stopped before any complete candidate was evaluated.",
                         phase="Stopped", detail="No complete candidate was evaluated")

    def _finish_worker(self, expected: GlobalFitWorker, request: FitExecutionRequest) -> None:
        """Release custody and present only this request's terminal candidate."""
        worker = self._worker
        if worker is not expected:
            return
        self._worker = None
        self._gate.release(self)
        closing = self._closing
        if self._window is not None and not self._application_close_fenced and not closing:
            self._window.set_executing(False)
            if self._candidate is not None and self._candidate.request is request:
                self._present_candidate(self._candidate)
            setup = self._accepted_setup()
            self._window.set_shared_seed_available(setup is not None and self._setup_can_seed_shared(setup))
        self.execution_changed.emit(False)
        if worker is not None:
            worker.deleteLater()
        if closing:
            self._discard_local()
            self._destroy_window_after_cleanup()
        else:
            self._notify_application_close_settled()

    def _origin_is_current(self) -> bool:
        origin = self._current_origin()
        if origin is None:
            return False
        return self._origin_science_is_current(origin)

    def _origin_science_is_current(
        self,
        captured: FitOriginSnapshot,
    ) -> bool:
        """Compare retained Fit science while excluding defaults/presentation."""

        try:
            current = capture_fit_origin(
                self._session.document,
                captured.tab_id,
            )
        except (TypeError, ValueError, KeyError, LookupError):
            return False

        def dataset_science(item: object) -> tuple[object, ...]:
            return (
                item.dataset_id,
                item.set_id,
                item.included_at_launch,
                item.initial_values,
                item.resolved_t_start,
                tuple(
                    (
                        observation.raw_observation_id,
                        observation.model_target,
                        observation.t,
                        observation.y,
                        observation.source_rows,
                    )
                    for observation in item.observations
                ),
            )

        return bool(
            captured.tab_id == current.tab_id
            and captured.mechanism_source == current.mechanism_source
            and captured.prepared_dsl is current.prepared_dsl
            and captured.scientific_settings == current.scientific_settings
            and tuple(
                dataset_science(item)
                for item in sorted(
                    captured.mapped_datasets,
                    key=lambda value: value.dataset_id,
                )
            )
            == tuple(
                dataset_science(item)
                for item in sorted(
                    current.mapped_datasets,
                    key=lambda value: value.dataset_id,
                )
            )
            and captured.parameter_defaults == current.parameter_defaults
        )

    def _build_request(self) -> FitExecutionRequest:
        setup = self._accepted_setup()
        if setup is None:
            raise RuntimeError("Fit has no typed local setup.")
        return build_fit_execution_request(setup=setup)

    def _accepted_setup(self) -> FitSetupSnapshot | None:
        """Return the one accepted setup owner for the current route."""

        state = self._setup_state
        if state is not None and not isinstance(state, FitSetupSnapshot):
            raise TypeError("Fit setup owner contains an invalid state.")
        return state

    def _current_origin(self) -> FitOriginSnapshot | None:
        """Return the concrete origin retained by the accepted setup."""

        setup = self._accepted_setup()
        return None if setup is None else setup.origin

    def _install_setup(
        self,
        setup: FitSetupSnapshot,
    ) -> None:
        """Install one complete accepted Fit setup."""

        if not isinstance(setup, FitSetupSnapshot):
            raise TypeError("Accepted Fit setup must be FitSetupSnapshot.")
        self._setup_state = setup
        if self._window is not None:
            self._window.set_shared_seed_available(self._setup_can_seed_shared(setup))

    def _set_status(self, text: str, *, phase: str | None = None,
                    detail: str = "", present: bool = True) -> None:
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
                if phase is not None:
                    self._window.set_execution_status(phase, detail)
                elif present:
                    self._window.set_status(message)
            except RuntimeError:
                pass
        self.status_changed.emit(message)

    def _require_thread(self) -> None:
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError(
                "FitSessionOwner may be used only from its GUI thread."
            )
