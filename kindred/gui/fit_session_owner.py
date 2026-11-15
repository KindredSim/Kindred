"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from PySide6 import QtCore, QtWidgets

from kindred.core.exceptions import located
from kindred.core.project_document import (
    MechanismSourceReconciliation,
    ProjectDocument,
    reconcile_tab_mechanism_source,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_session import ProjectSession
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    derive_source_with_parameter_updates,
)
from kindred.gui.computation_gate import ComputationGate
from kindred.core.validation import six_digit_text, six_digit_value
from kindred.gui.fitting.launch import (
    FitExecutionRequest,
    FitOriginSnapshot,
    FitSetupSnapshot,
    build_fit_execution_request,
    capture_fit_origin,
    create_initial_fit_setup,
    fit_parameter_overall,
)
from kindred.gui.fitting.reporting import fit_better_fit_text, fit_identity_text
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.execution import FitCandidate
from kindred.gui.fitting.worker import GlobalFitWorker
from kindred.gui.workspace_tab_registry import WorkspaceTabRegistry

ColorResolver = Callable[[str, Sequence[str]], object]

# The setup fields each tab owns; Reset, Undo and Redo replace them by part.
_DATA_FIELDS = ("datasets", "initials", "shared_initial_groups", "dataset_weight_mode")
_PARAMETER_FIELDS = ("parameters", "solver", "defaults", "profile")


def _part_differs(fields: tuple[str, ...], first: FitSetupSnapshot, second: FitSetupSnapshot) -> bool:
    return any(getattr(first, field) != getattr(second, field) for field in fields)


class FitSessionOwner(QtCore.QObject):
    """Retain the window, accepted setup, worker, and candidate custody."""

    window_changed = QtCore.Signal(bool)
    execution_changed = QtCore.Signal(bool)
    status_changed = QtCore.Signal(str)
    project_document_changed = QtCore.Signal(str)
    application_close_settlement_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        session: ProjectSession,
        registry: WorkspaceTabRegistry,
        gate: ComputationGate,
        observation_style_getter: Callable[[], tuple[float, float]],
        color_resolver: ColorResolver,
        parallel_workers: Callable[[], int],
        parent_widget: QtWidgets.QWidget,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._registry = registry
        self._gate = gate
        self._color_resolver = color_resolver
        self._observation_style_getter = observation_style_getter
        self._parallel_workers = parallel_workers
        self._parent_widget = parent_widget
        self._window: FittingWindow | None = None
        self._worker: GlobalFitWorker | None = None
        self._candidate: FitCandidate | None = None
        self._candidate_copy_name: str | None = None
        self._candidate_applied_to_project = False
        self._candidate_projection = None
        # A failed retry's status, shown after the earlier result returns.
        self._failure_status: str | None = None
        self._setup_state: FitSetupSnapshot | None = None
        self._initial_setup_snapshot: FitSetupSnapshot | None = None
        self._last_fit_setup_snapshot: FitSetupSnapshot | None = None
        # Accepted setups before and after the current one, for Undo and Redo.
        self._undo_setups: list[FitSetupSnapshot] = []
        self._redo_setups: list[FitSetupSnapshot] = []
        self._closing = False
        self._application_close_fenced = False

    def refresh_plot_preferences(self) -> None:
        if self._window is not None:
            self._window.set_observation_style(*self._observation_style_getter())

    @property
    def active(self) -> bool:
        """Return whether Fit owns an open window or worker."""

        return self._window is not None or self._worker is not None

    @property
    def application_close_settled(self) -> bool:
        """Return whether Close-facing Fit cleanup is fully owner-settled."""

        return self._worker is None and (not self._closing or self._window is None)

    @QtCore.Slot(object)
    def open_fit_configuration(
        self,
        tab_id: str,
    ) -> FittingWindow | None:
        """Open an idle configuration from one admitted document origin."""

        if (
            self._closing
            or self._window is not None
            or self._worker is not None
        ):
            return None
        if not self._gate.is_idle:
            self._set_status(
                "Fit Open rejected: "
                f"{self._gate.active_kind} is active."
            )
            return None
        origin = capture_fit_origin(self._session.document, tab_id)
        setup = create_initial_fit_setup(origin)
        analysis = origin.prepared_dsl.analysis
        modeled_species = tuple(
            dict.fromkeys(
                (*analysis.species_names, *analysis.observable_names)
            )
        )
        def fit_color_resolver(species):
            return self._color_resolver(species, modeled_species)

        window = FittingWindow(
            setup,
            close_handler=self._request_window_close,
            color_resolver=fit_color_resolver,
            observation_style=self._observation_style_getter(),
            parent=self._parent_widget,
        )
        self._window = window
        self._install_setup(setup)
        # The previous window's destruction cleared its last Fit and result.
        self._initial_setup_snapshot = setup
        window.project_setup_snapshot(setup)

        window.executeRequested.connect(self.execute)
        window.stopRequested.connect(self.stop)
        window.pauseRequested.connect(self.pause)
        window.resumeRequested.connect(self.resume)
        window.applyRequested.connect(self.apply_fit_to_project)
        window.applyToCopyRequested.connect(self.apply_fit_to_tab_copy)
        window.setupSnapshotRequested.connect(self._on_setup_snapshot)
        window.resetRequested.connect(self.reset_setup)
        window.undoRequested.connect(self.undo_setup)
        window.redoRequested.connect(self.redo_setup)
        window.destroyed.connect(self._on_window_destroyed)
        self.window_changed.emit(True)
        window.show()
        window.raise_()
        window.activateWindow()
        declarations = origin.prepared_dsl.authored_stop_declarations()
        if declarations and origin.scientific_settings.use_authored_stops:
            message = QtWidgets.QMessageBox(window)
            message.setWindowTitle("Stopping Interventions During Fitting")
            message.setIcon(QtWidgets.QMessageBox.Information)
            message.setTextFormat(QtCore.Qt.PlainText)
            message.setText(
                "Datasets fitted against time run over the interval their experimental data "
                "require, so these stopping interventions are ignored for them:\n\n"
                + "\n".join(declarations)
                + "\n\nDatasets fitted against a measured X use them, as Run does. "
                "All other interventions remain active."
            )
            message.setStandardButtons(QtWidgets.QMessageBox.Ok)
            message.exec()
        self._set_status(f"Fit opened for {origin.tab_name}.")
        return window

    @QtCore.Slot()
    def execute(self, initial_starts: Mapping[str, float] = MappingProxyType({})) -> None:
        """Admit one Fit worker from the current accepted setup.

        ``initial_starts`` starts named varied starting concentrations
        elsewhere than their Starting IC, for this run only.
        """
        window = self._window
        if self._closing:
            self._set_status("Fit rejected while the window is closing.")
            return
        if not self._gate.is_idle:
            self._set_status(
                f"Fit rejected: {self._gate.active_kind} is active."
            )
            return
        request = build_fit_execution_request(
            setup=self._accepted_setup(),
            workers=self._parallel_workers(),
            initial_starts=initial_starts,
        )
        # The gate was idle just above, with nothing in between to take it.
        self._gate.try_acquire(self, "fit")
        # A retained result of another setup ends here; one of this setup
        # stays for a failed or cancelled attempt.
        if self._candidate is not None and self._candidate.setup_snapshot != request.setup:
            self._invalidate_candidate_custody()
        worker = GlobalFitWorker(request, parent=self)
        # One worker exists at a time, and its finished signal is queued
        # after every other signal from its run, so callbacks need no
        # worker-identity check.
        worker.execution_started.connect(self._on_execution_started)
        worker.progress.connect(self._on_progress)
        worker.best_updated.connect(self._on_best)
        worker.curves_updated.connect(self._on_live_curves)
        worker.candidate_ready.connect(
            lambda candidate, w=worker: self._on_candidate(w, candidate)
        )
        worker.factual_failure.connect(self._on_failure)
        worker.scientific_unavailable.connect(self._on_unavailable)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(lambda r=request: self._finish_worker(r))
        self._worker = worker
        window.set_executing(True)
        self.execution_changed.emit(True)
        window.prepare_fit_plots(request, waiting=True)
        self._set_status("Preparing Fit plots.", phase="Preparing", detail="Preparing all dataset views")
        QtCore.QTimer.singleShot(0, lambda: self._launch_worker(worker, request))

    def _launch_worker(self, worker: GlobalFitWorker, request: FitExecutionRequest) -> None:
        if self._closing:
            self._finish_worker(request)
            return
        worker.start()

    def _on_execution_started(self) -> None:
        if self._closing:
            return
        self._set_status("Fit started.", phase="Running", detail="Waiting for the first complete candidate")

    @QtCore.Slot(object)
    def _on_setup_snapshot(self, snapshot: object) -> None:
        """Admit one complete leaf-authored setup snapshot."""

        # Setup edits come only from the open, idle window: every setup
        # control is disabled while a Fit runs.
        if self._closing:
            return
        window = self._window
        previous = self._accepted_setup()
        if snapshot == previous:
            return
        self._record_setup_change(previous)
        self._install_setup(snapshot)
        # An accepted edit is the user's latest action; an action that reports
        # its outcome posts it after this returns.
        window.set_status("")
        self._refresh_result_currency()
        window.project_setup_snapshot(snapshot)

    @QtCore.Slot()
    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.request_stop()
        self._set_status("Fit stop requested. The best complete candidate of this run is kept.",
                         phase="Stopping", detail="Best complete candidate will be retained")

    @QtCore.Slot()
    def pause(self) -> None:
        # Reached only through the window's Pause, which checks it is running.
        self._worker.pause()
        self._set_status(
            "Fit pause requested. The current evaluation will finish before execution pauses.",
            phase="Pause requested", detail="Pauses after the current evaluation"
        )

    @QtCore.Slot()
    def resume(self) -> None:
        self._worker.resume()
        self._set_status("Fit resumed.", phase="Running", detail="")

    @QtCore.Slot(object)
    def reset_setup(self, reset_kind: str, part: str) -> None:
        """Restore Data and Targets, Parameters or both from a retained setup.

        Data and Targets owns the datasets, initial concentrations, shared
        IC groups and weight mode; Parameters owns the parameters, Fit
        settings and Profile.
        """

        # Reset is offered only by the open, idle window.
        # "Last Fit" is offered only once a Fit set that snapshot, and Apply
        # rebases both snapshots with the setup.
        window = self._window
        snapshot = (
            self._initial_setup_snapshot
            if reset_kind == "initial"
            else self._last_fit_setup_snapshot
        )
        current = self._accepted_setup()
        if part != "both":
            fields = _DATA_FIELDS if part == "data" else _PARAMETER_FIELDS
            snapshot = replace(current, **{field: getattr(snapshot, field) for field in fields})
        if snapshot != current:
            self._record_setup_change(current)
        # A reset discards held invalid text only in the part it replaces.
        window.clear_pending_setup_edits(data=part != "parameters", parameters=part != "data")
        self._restore_setup(snapshot)
        target = "the setup when this window opened" if reset_kind == "initial" else "the last Fit"
        scope = {"data": "Data and Targets", "parameters": "Parameters", "both": "Fit setup"}[part]
        self._set_status(f"{scope} reset to {target}.")

    @QtCore.Slot()
    def undo_setup(self) -> None:
        """Restore the accepted setup before the last change."""

        self._step_setup_history(self._undo_setups, self._redo_setups, "undone")

    @QtCore.Slot()
    def redo_setup(self) -> None:
        """Restore the accepted setup the last Undo left."""

        self._step_setup_history(self._redo_setups, self._undo_setups, "redone")

    def _step_setup_history(
        self,
        source: list[FitSetupSnapshot],
        destination: list[FitSetupSnapshot],
        verb: str,
    ) -> None:
        # Ctrl+Z and Ctrl+Y fire during a run and while stopping.
        if self._closing or self._worker is not None or not source:
            return
        current = self._accepted_setup()
        snapshot = source.pop()
        # Held invalid text keeps its whole row or dataset, so like a reset,
        # a step discards it in the part it changes; the other part keeps it.
        self._window.clear_pending_setup_edits(
            data=_part_differs(_DATA_FIELDS, current, snapshot),
            parameters=_part_differs(_PARAMETER_FIELDS, current, snapshot),
        )
        destination.append(current)
        self._restore_setup(snapshot)
        self._set_status(f"Fit setup change {verb}.")

    def _record_setup_change(self, previous: FitSetupSnapshot) -> None:
        """Keep the setup a change replaces for Undo; a new change ends Redo."""

        self._undo_setups.append(previous)
        self._redo_setups.clear()

    def _restore_setup(self, snapshot: FitSetupSnapshot) -> None:
        """Install and project an earlier setup; callers discard held text first."""

        self._install_setup(snapshot)
        self._refresh_result_currency()
        self._window.project_setup_snapshot(snapshot)

    def _candidate_is_current(self) -> bool:
        """Whether the accepted setup is the one the retained result left.

        That is the setup the result ran with, carrying its fitted parameter
        values as starting values.  Apply to Project only rebases the setup
        onto the applied origin.
        """

        candidate = self._candidate
        setup = self._accepted_setup()
        return (
            candidate is not None
            and replace(setup, origin=candidate.setup_snapshot.origin) == candidate.setup_snapshot
        )

    def _refresh_result_currency(self) -> None:
        """Mark the shown result stale, or current again after Undo or Reset."""

        window = self._window
        if self._candidate is None:
            # A failure or stop shown without a result is stale after any change.
            window.mark_results_stale()
            return
        if self._candidate_is_current():
            window.mark_results_current()
            self._refresh_apply_availability()
        else:
            window.mark_results_stale()

    def _invalidate_candidate_custody(self) -> None:
        """Release the current terminal candidate."""

        self._candidate = None
        self._candidate_projection = None
        self._candidate_copy_name = None
        self._candidate_applied_to_project = False

    def cancel_for_application_close(self) -> None:
        """Dispose Fit state and close through the existing owner lifecycle."""

        was_closing = self._closing
        # The fence silences Fit status lines for the rest of Close.
        self._application_close_fenced = True
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
        # Only the window's closeEvent calls this, once it has entered
        # stopping and while not already closing.
        self._closing = True
        if self._worker is not None:
            self._worker.request_cancel()
            return False
        self._discard_local()
        return True

    def _destroy_window_after_cleanup(self) -> None:
        """Close the window through its owner-authorized path and delete it now."""

        window = self._window
        window.force_close_after_cleanup()
        QtCore.QCoreApplication.sendPostedEvents(
            window,
            QtCore.QEvent.Type.DeferredDelete,
        )

    def _discard_local(self) -> None:
        self._invalidate_candidate_custody()
        self._setup_state = None
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
        self._undo_setups.clear()
        self._redo_setups.clear()
        if self._window is not None:
            self._window.clear_candidate()

    @QtCore.Slot()
    def _on_window_destroyed(self, *_args: object) -> None:
        self._window = None
        self._setup_state = None
        self._invalidate_candidate_custody()
        self._initial_setup_snapshot = None
        self._last_fit_setup_snapshot = None
        self._undo_setups.clear()
        self._redo_setups.clear()
        self._closing = False
        self.window_changed.emit(False)
        self.application_close_settlement_changed.emit()
        self._set_status("Fit closed. All temporary fit state was discarded.")

    def _on_progress(self, done: int, total: int, text: str) -> None:
        if self._closing:
            return
        self._window.set_progress(done, total)
        self._set_status(text)

    def _on_best(self, update: object) -> None:
        if self._closing:
            return
        self._window.set_live_best(update)

    def _on_live_curves(self, curves: object) -> None:
        if self._closing:
            return
        self._window.set_live_curves(curves)

    def _on_candidate(self, worker: GlobalFitWorker, candidate: object) -> None:
        """Retain and present the worker's direct typed candidate terminal."""

        if self._closing:
            return
        self._invalidate_candidate_custody()
        self._candidate = candidate
        self._candidate_projection = worker.terminal_projection
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
        self._undo_setups.clear()
        self._redo_setups.clear()
        window = self._window
        if window is None:
            return
        window.project_setup_snapshot(terminal_setup)
        window.set_last_fit_reset_available(terminal_setup)
        curves = self._candidate_projection
        # A run that ended without Stop shows a full bar; its text states how
        # Profile settled.
        profile = candidate.result.profile_likelihood_result
        if candidate.result.ending != "stopped" and (profile is None or profile.status != "stopped"):
            window.set_progress(1, 1)
        window.set_terminal_presentation(
            candidate,
            curves=curves,
        )
        self._refresh_apply_availability()

    @classmethod
    def _candidate_publishable_parameter_values(
        cls,
        candidate: FitCandidate,
    ) -> dict[str, float]:
        """Return the fitted Shared values and each Per-dataset parameter's mean."""

        setup = candidate.setup_snapshot
        values: dict[str, float] = {}
        for parameter in setup.parameters:
            if not parameter.selected_config.fit:
                continue
            if parameter.scope == "shared":
                raw_value = candidate.result.shared_params[parameter.name]
            else:
                overall = fit_parameter_overall(parameter, setup.datasets, candidate.origin)
                if overall is None:
                    continue
                raw_value = overall[1]
            values[parameter.name] = raw_value
        return values

    @classmethod
    def _candidate_apply_reason(cls, candidate: FitCandidate) -> str:
        parameter_values = cls._candidate_publishable_parameter_values(candidate)
        initial_values = candidate.initial_values_by_set
        return "" if parameter_values or initial_values else "This fit has no fitted values to apply."

    @QtCore.Slot()
    def apply_fit_to_project(self) -> None:
        """Apply one complete, current terminal Fit candidate atomically."""

        self._apply_fit(to_copy=False)

    @QtCore.Slot()
    def apply_fit_to_tab_copy(self) -> None:
        """Publish one fitted copy while retaining the original Fit origin."""

        self._apply_fit(to_copy=True)

    def _refresh_apply_availability(self) -> None:
        window = self._window
        if self._candidate_applied_to_project:
            reason = "These Fit values have already been applied to the project. Run a new fit to apply again."
        elif self._candidate is None:
            reason = "Run a fit to obtain current values to apply."
        elif not self._candidate_is_current():
            reason = "Run a fit to obtain current values to apply."
        else:
            reason = self._candidate_apply_reason(self._candidate)
        copy_reason = reason
        if self._candidate_copy_name is not None:
            copy_reason = f"Fit already applied to '{self._candidate_copy_name}'. Run a new fit to create another fitted copy."
        window.set_apply_available(
            not reason, reason, copy_available=not copy_reason, copy_reason=copy_reason,
        )

    def _apply_fit(self, *, to_copy: bool) -> None:
        # Both Apply actions are enabled only while idle with a current,
        # unapplied candidate; every change to those disables them at once.
        candidate = self._candidate
        origin = self._current_origin()
        setup = self._accepted_setup()
        window = self._window
        reason = self._candidate_apply_reason(candidate)
        if reason:
            self._set_status(f"Fit Apply rejected: {reason}")
            return
        if to_copy:
            document, copied, set_ids = self._registry.prepare_tab_copy(
                origin.tab_id, name_suffix="Fit",
            )
            proposed_document = self._prepare_fit_application(
                candidate, document, copied.tab_id, set_ids=set_ids,
            )
            published = self._registry.publish_tab_copy(
                origin.tab_id, copied, set_ids, proposed_document, activate=False,
            )
            self._candidate_copy_name = published.name
            self._refresh_apply_availability()
            self.project_document_changed.emit(published.tab_id)
            origin_name = self._registry.tab(origin.tab_id).name
            message = (
                f"Fit applied to '{published.name}'. "
                f"This fitting session remains attached to '{origin_name}'."
            )
            self._set_status(message)
            return
        proposed_document = self._prepare_fit_application(
            candidate, self._registry.document, origin.tab_id,
        )
        successor_origin = capture_fit_origin(proposed_document, origin.tab_id)
        rebased_setup = replace(setup, origin=successor_origin)
        rebased_initial_setup = replace(
            self._initial_setup_snapshot,
            origin=successor_origin,
        )
        rebased_last_fit_setup = replace(
            candidate.setup_snapshot,
            origin=successor_origin,
        )
        self._registry.replace_document(proposed_document)
        # Applied starting concentrations supersede their sets' Initial
        # Conditions table drafts, as a Slider Apply's do.
        runtime = self._registry.tab_runtime(origin.tab_id)
        for set_id, _values in candidate.initial_values_by_set:
            runtime.initial_text_by_set_id.pop(set_id, None)
        self._candidate_applied_to_project = True
        self._install_setup(rebased_setup)
        self._initial_setup_snapshot = rebased_initial_setup
        self._last_fit_setup_snapshot = rebased_last_fit_setup
        for history in (self._undo_setups, self._redo_setups):
            history[:] = [replace(item, origin=successor_origin) for item in history]
        window.project_setup_snapshot(rebased_setup)
        window.set_last_fit_reset_available(rebased_last_fit_setup)
        self.project_document_changed.emit(origin.tab_id)
        self._refresh_apply_availability()
        self._set_status("Fit applied to the project.")

    def _prepare_fit_application(
        self,
        candidate: FitCandidate,
        document: ProjectDocument,
        tab_id: str,
        *,
        set_ids: Mapping[str, str] | None = None,
    ) -> ProjectDocument:
        """Admit the same complete Fit values for either destination, detached.

        Admitted values are rounded once to six significant digits; the Fit
        session keeps the exact values.
        """

        exact_values = self._candidate_publishable_parameter_values(candidate)
        initial_values = {
            (set_ids[set_id] if set_ids is not None else set_id, species): six_digit_value(value)
            for set_id, values in candidate.initial_values_by_set
            for species, value in values
        }
        origin = candidate.origin
        source_text = origin.mechanism_source.reactions_text
        # A value whose exact or rounded value equals the admitted one keeps the
        # authored text.
        admitted = {item.name: item.shared_config.value for item in origin.parameter_defaults}
        updates = tuple(
            DslParameterCoordinateUpdate(name=name, value=six_digit_value(value))
            for name, value in sorted(exact_values.items())
            if admitted[name] not in (value, six_digit_value(value))
        )
        if updates:
            source_text = derive_source_with_parameter_updates(source_text, origin.prepared_dsl, updates)
        comments = self._per_dataset_apply_comments(candidate)
        if comments:
            newline = "\r\n" if "\r\n" in source_text else "\n"
            separator = "" if source_text.endswith(("\n", "\r")) else newline
            source_text += separator + newline.join(comments) + ("" if separator else newline)
        source = MechanismAuthoringSource(reactions_text=source_text)
        if source == document.tab(tab_id).mechanism_source:
            # IC-only and unchanged-value Apply retain the accepted owner so
            # unaffected Run requests remain reusable by ordinary Preview.
            reconciliation = MechanismSourceReconciliation(proposed_document=document)
        else:
            reconciliation = reconcile_tab_mechanism_source(document, tab_id, source)
        proposed_document = reconciliation.proposed_document
        proposed_tab = proposed_document.tab(tab_id)
        if initial_values:
            proposed_sets = []
            for initial_set in proposed_tab.initial_condition_sets:
                values = initial_set.values_dict()
                for key, value in initial_values.items():
                    if key[0] != initial_set.set_id:
                        continue
                    # Materialize an authored omission whose admitted value was zero.
                    values[key[1]] = value
                proposed_sets.append(initial_set.with_values(values))
            proposed_document = proposed_document.replace_tab(
                proposed_tab.with_initial_condition_sets(tuple(proposed_sets))
            )
        return proposed_document

    @staticmethod
    def _per_dataset_apply_comments(candidate: FitCandidate) -> tuple[str, ...]:
        """One comment line per fitted Per-dataset parameter: its mean and values."""

        setup = candidate.setup_snapshot
        lines: list[str] = []
        for parameter in setup.parameters:
            if parameter.scope != "per_dataset":
                continue
            overall = fit_parameter_overall(parameter, setup.datasets, candidate.origin)
            if overall is None:
                continue
            entries, mean = overall
            lines.append(
                f"# Fit applied: {parameter.name} = {six_digit_text(mean)} (mean). "
                "Per-dataset values not applied: "
                + ", ".join(
                    f"{' '.join(label.splitlines())} = {six_digit_text(value)}"
                    for label, value in entries
                )
            )
        return tuple(lines)

    def _on_failure(self, failure: object) -> None:
        """Present the worker's direct factual failure terminal once."""

        if self._closing:
            return
        self._window.clear_candidate()
        self._window.set_terminal_failure(failure)
        self._failure_status = f"Fit failed: {located(failure.message, *failure.location())}"
        self._set_status(self._failure_status, present=False)

    def _on_unavailable(self, unavailable: object) -> None:
        """Present the worker's direct scientific-unavailability terminal."""

        if self._closing:
            return
        self._window.set_terminal_unavailable(unavailable)
        cause = unavailable.factual_cause
        self._set_status(
            "No valid candidate found: "
            + located(cause.message, *cause.location()),
            present=False,
        )

    def _on_cancelled(self) -> None:
        """Publish the direct cancellation terminal once."""

        if self._closing:
            return
        self._window.clear_candidate()
        self._invalidate_candidate_custody()
        self._set_status("Fit stopped before any complete candidate was evaluated.",
                         phase="Stopped", detail="No complete candidate was evaluated")

    def _finish_worker(self, request: FitExecutionRequest) -> None:
        """Release custody and present only this request's terminal candidate."""
        # Its own finished signal, posted last; one worker exists at a time.
        worker = self._worker
        self._worker = None
        self._gate.release(self)
        failure_status, self._failure_status = self._failure_status, None
        closing = self._closing
        if not closing:
            self._window.set_executing(False)
            if self._candidate is not None:
                if self._candidate.request is request:
                    self._present_candidate(self._candidate)
                elif not worker.process_started and not self._candidate_applied_to_project:
                    # An applied result is not re-shown: its setup predates
                    # Apply, and the failure stays visible instead.  A result
                    # that returns keeps the failure in the status line.
                    self._window.prepare_fit_plots(self._candidate.request, waiting=False)
                    self._present_candidate(self._candidate)
                    if failure_status is not None:
                        self._window.set_status(failure_status)
                else:
                    # The attempt now shown supersedes an older result that
                    # is not shown again, so it can no longer be applied.
                    self._invalidate_candidate_custody()
        self.execution_changed.emit(False)
        worker.deleteLater()
        if closing:
            self._discard_local()
            self._destroy_window_after_cleanup()
        else:
            self.application_close_settlement_changed.emit()
            candidate = self._candidate
            if candidate is not None and candidate.request is request and candidate.better_setup is not None:
                self._offer_better_fit(candidate)

    def _offer_better_fit(self, candidate: FitCandidate) -> None:
        """Offer to run again from the refit Profile found clearly better."""

        message = QtWidgets.QMessageBox(self._window)
        message.setWindowTitle("Better Fit Found")
        message.setIcon(QtWidgets.QMessageBox.Information)
        message.setTextFormat(QtCore.Qt.RichText)
        message.setText(fit_better_fit_text(candidate))
        start = message.addButton("Run Fit From There", QtWidgets.QMessageBox.AcceptRole)
        message.addButton("Not Now", QtWidgets.QMessageBox.RejectRole)
        message.exec()
        if message.clickedButton() is start:
            # The refit's parameter values become the setup, as a fit's own do.
            # Its varied starting concentrations start this one run only, so
            # every Starting IC and its range stay as the user set them.
            self._record_setup_change(self._accepted_setup())
            self._restore_setup(candidate.better_setup)
            self.execute(dict(candidate.better_initial_starts))

    def _accepted_setup(self) -> FitSetupSnapshot | None:
        """Return the one accepted setup owner for the current route."""

        return self._setup_state

    def _current_origin(self) -> FitOriginSnapshot | None:
        """Return the concrete origin retained by the accepted setup."""

        setup = self._accepted_setup()
        return None if setup is None else setup.origin

    def _install_setup(
        self,
        setup: FitSetupSnapshot,
    ) -> None:
        """Install one complete accepted Fit setup."""

        self._setup_state = setup

    def _set_status(self, text: str, *, phase: str | None = None,
                    detail: str = "", present: bool = True) -> None:
        if self._application_close_fenced:
            return
        origin = self._current_origin()
        message = text if origin is None else fit_identity_text(text, origin)
        if self._window is not None:
            if phase is not None:
                self._window.set_execution_status(phase, detail)
            elif present:
                self._window.set_status(message)
        self.status_changed.emit(message)
