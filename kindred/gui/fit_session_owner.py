"""Sole owner of one origin-bound modal Fit session."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace

from PySide6 import QtCore, QtWidgets

from kindred.core.algebra.observable_introspection import (
    FitLocalObservable,
    validate_fit_local_observable,
)
from kindred.core.dataset_mapping import with_confirmed_observation_mapping
from kindred.core.project_document import (
    FitDefaults,
    MechanismSourceCollision,
    canonical_observation_target_names,
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
    FitDatasetLocalSettings,
    FitExecutionRequest,
    FitLocalScalarProvenance,
    FitOriginSnapshot,
    FitObservationSnapshot,
    FitParameterConfiguration,
    FitParameterVariable,
    FitSetupSnapshot,
    build_fit_execution_request,
    capture_fit_origin,
    build_fit_local_admission,
    create_initial_fit_setup,
    revalidate_fit_request_inclusions,
    prune_fit_analysis_coordinates,
    _replace_fit_parameter,
)
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.reporting import (
    fit_candidate_report_text,
)
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
            modeled_species = tuple(origin.prepared_dsl.analysis.species_names)
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
            window.set_apply_available(False)
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
        window.resultsSummaryRequested.connect(self.results_summary)
        window.sharedSeedRequested.connect(self.seed_shared_fit)
        window.applyRequested.connect(self.apply_fit_to_project)
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
                setup,
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
        self._set_status("Fit started.")
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
        worker.request_cancel()
        self._set_status(
            "Fit stop requested; the current terminal candidate will be "
            "restored after settlement."
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
        self._set_status(
            "Fit pause requested; the current evaluation will finish before execution pauses."
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
        self._set_status("Fit resumed.")
        return True

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

    @QtCore.Slot()
    def results_summary(self) -> bool:
        """Render the human-readable Summary from retained Fit facts."""

        candidate = self._candidate
        if candidate is None or self._window is None:
            self._set_status("No completed Fit run is available.")
            return False
        try:
            self._window.render_results_summary(fit_candidate_report_text(candidate))
            return True
        except Exception as exc:
            self._set_status(f"Fit Results Summary failed: {exc}")
            return False

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
            # Close is explicit so a stopped-best terminal can never win
            # application teardown.
            worker.request_cancel(close=True)

    def _request_window_close(self) -> bool:
        if self._closing:
            return False
        self._closing = True
        if self._worker is not None:
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
        # custody so a failed/cancelled replacement can restore it.
        self._set_status("Fit complete.")

    def _present_candidate(
        self,
        candidate: FitCandidate,
        *,
        announce: bool = True,
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
            window.set_terminal_presentation(
                candidate,
                curves=curves,
                curve_failure=curve_failure,
            )
            window.set_apply_available(self._candidate_can_apply(candidate))
            window.set_shared_seed_available(self._candidate_can_seed_shared(candidate))
        except Exception as exc:
            self._set_status(f"Fit result presentation failed: {exc}")
        else:
            if announce:
                self._set_status("Fit complete.")

    def _restore_candidate_presentation(self) -> None:
        """Restore retained terminal Fit state after a transient attempt."""

        candidate = self._candidate
        if candidate is None or self._window is None:
            return
        self._present_candidate(candidate, announce=False)

    @QtCore.Slot(str)
    def seed_shared_fit(self, statistic: str) -> bool:
        """Convert successful Per-dataset estimates into an idle Shared setup."""

        self._require_thread()
        candidate = self._candidate
        setup = self._accepted_setup()
        origin = self._current_origin()
        window = self._window
        if (
            self._closing
            or self._application_close_fenced
            or candidate is None
            or setup is None
            or origin is None
            or window is None
            or self._worker is not None
        ):
            return False
        if not self._candidate_can_seed_shared(candidate):
            self._set_status(
                "Shared Fit setup rejected: no successful Per-dataset estimates are available."
            )
            return False
        statistic = str(statistic or "").strip().lower()
        if statistic not in {"arithmetic", "geometric"}:
            self._set_status("Shared Fit setup rejected: unknown seed statistic.")
            return False
        included_ids = tuple(
            item.dataset_id for item in setup.datasets if item.included
        )
        result_values = candidate.result.dataset_params
        parameters: list[FitParameterVariable] = []
        changed_parameter_names: list[str] = []
        try:
            for parameter in setup.parameters:
                if (
                    parameter.scope != "per_dataset"
                    or not parameter.selected_config.fit
                    or parameter.fit_local_provenance is not None
                ):
                    parameters.append(parameter)
                    continue
                values = []
                for dataset_id in included_ids:
                    value = result_values.get(dataset_id, {}).get(parameter.name)
                    if value is None or not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Fit result omitted {parameter.name!r} for {dataset_id!r}."
                        )
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Fit result contains a non-finite value for {parameter.name!r}."
                        )
                    values.append(value)
                if statistic == "geometric":
                    if any(value <= 0.0 for value in values):
                        raise ValueError(
                            f"Geometric Shared seed requires positive {parameter.name!r} values."
                        )
                    seed = math.exp(math.fsum(math.log(value) for value in values) / len(values))
                else:
                    seed = math.fsum(values) / len(values)
                parameters.append(
                    _replace_fit_parameter(
                        parameter,
                        scope="shared",
                        initial=seed,
                        dataset_values=(),
                    )
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

    @staticmethod
    def _candidate_result_is_apply_eligible(candidate: FitCandidate) -> bool:
        result = candidate.result
        return bool(
            result.optimizer_converged
            and not result.diagnostic_only
            and result.optimizer_diagnostic is None
            and result.status == "ok"
        )

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
                or parameter.fit_local_provenance is not None
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
    def _candidate_local_scalar_values(candidate: FitCandidate) -> dict[str, float]:
        """Return only Shared Fit-local observable scalar values."""

        parameters = {item.name: item for item in candidate.setup_snapshot.parameters}
        names = tuple(
            scalar_name
            for observable in candidate.local_observables
            for scalar_name in observable.scalar_names
        )
        if len(names) != len(set(names)):
            raise ValueError("Fit candidate repeats a Fit-local scalar identity.")
        values: dict[str, float] = {}
        for name in names:
            parameter = parameters.get(name)
            if parameter is None or parameter.fit_local_provenance is None:
                raise ValueError(f"Observable scalar {name!r} has no admitted row.")
            if parameter.scope != "shared" or parameter.selected_config.fit is not True:
                raise ValueError(
                    f"Observable scalar {name!r} must use a fitted Shared scope."
                )
            if parameter.fit_local_provenance.admission_scope != "shared":
                raise ValueError(
                    f"Observable scalar {name!r} was admitted Per-dataset and cannot be applied."
                )
            raw_value = candidate.result.shared_params.get(name)
            if raw_value is None:
                raise ValueError(f"Observable scalar {name!r} has no fitted value.")
            values[name] = admit_gui_parameter_value(
                raw_value,
                scientific_category=parameter.scientific_category,
                allow_exact_zero=parameter.allow_exact_zero,
                field=f"observable scalar {name!r}",
            )
        return values

    @staticmethod
    def _candidate_can_seed_shared(candidate: FitCandidate) -> bool:
        return bool(
            FitSessionOwner._candidate_result_is_apply_eligible(candidate)
            and any(
                parameter.scope == "per_dataset"
                and parameter.selected_config.fit
                and parameter.fit_local_provenance is None
                for parameter in candidate.setup_snapshot.parameters
            )
        )

    @classmethod
    def _candidate_can_apply(cls, candidate: FitCandidate) -> bool:
        if not cls._candidate_result_is_apply_eligible(candidate):
            return False
        if any(
            parameter.scope == "per_dataset"
            and parameter.selected_config.fit
            and parameter.fit_local_provenance is None
            for parameter in candidate.setup_snapshot.parameters
        ):
            return False
        try:
            parameter_values = cls._candidate_publishable_parameter_values(candidate)
            scalar_values = cls._candidate_local_scalar_values(candidate)
            initial_values = candidate.initial_values_by_set
        except (TypeError, ValueError, KeyError):
            return False
        return bool(parameter_values or scalar_values or initial_values)

    @QtCore.Slot()
    def apply_fit_to_project(self) -> bool:
        """Apply one complete, current terminal Fit candidate atomically."""

        self._require_thread()
        candidate = self._candidate
        origin = self._current_origin()
        window = self._window
        if (
            self._closing
            or self._application_close_fenced
            or candidate is None
            or origin is None
            or window is None
            or self._worker is not None
        ):
            return False
        if not self._candidate_result_is_apply_eligible(candidate):
            self._set_status("Fit Apply rejected: the terminal result is not converged.")
            return False
        if not self._origin_science_is_current(candidate.origin):
            self._set_status("Fit Apply rejected: origin science is no longer current.")
            return False
        if any(
            parameter.scope == "per_dataset"
            and parameter.selected_config.fit
            and parameter.fit_local_provenance is None
            for parameter in candidate.setup_snapshot.parameters
        ):
            self._set_status(
                "Fit Apply rejected: configure an ordinary Shared Fit before applying."
            )
            return False
        try:
            parameter_values = self._candidate_publishable_parameter_values(candidate)
            self._candidate_local_scalar_values(candidate)
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
            if candidate.local_observables:
                source_text = self._append_observable_source(candidate, source_text)
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
            for local in candidate.setup_snapshot.datasets:
                current_link = proposed_tab.dataset_link(local.dataset_id)
                current_mapping = (
                    ()
                    if current_link is None
                    else current_link.observation_mapping
                )
                if current_mapping == local.observation_targets:
                    continue
                proposed_tab = with_confirmed_observation_mapping(
                    document=proposed_document,
                    tab=proposed_tab,
                    dataset=proposed_document.dataset(local.dataset_id),
                    model_targets=canonical_observation_target_names(proposed_tab),
                    confirmed_mapping=local.observation_targets,
                )
                proposed_document = proposed_document.replace_tab(proposed_tab)
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
            capture_fit_origin(proposed_document, origin.tab_id)
        except (TypeError, ValueError, KeyError, LookupError) as exc:
            self._set_status(f"Fit Apply rejected: {exc}")
            return False
        token = self._session.capture_content_token()
        if self._registry.replace_document(proposed_document, expected_token=token) is None:
            self._set_status("Fit Apply rejected: the captured project is stale.")
            return False
        self.project_document_changed.emit(origin.tab_id)
        window.set_apply_available(False)
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

    @classmethod
    def _append_observable_source(cls, candidate: FitCandidate, source: str) -> str:
        values = cls._candidate_local_scalar_values(candidate)
        observables = tuple(candidate.local_observables)
        if not observables:
            raise ValueError("Fit candidate contains no observable definitions.")
        newline = "\r\n" if "\r\n" in source else "\n"
        separator = "" if source.endswith(("\n", "\r")) else newline
        declarations = tuple(
            f"param {name} = {values[name]:.17g}" for name in sorted(values)
        )
        algebra = tuple(
            f"let {item.name} = {item.expression}" for item in observables
        )
        return source + separator + newline.join((*declarations, *algebra))

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
        self._set_status(f"Fit failed: {failure.message}")

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
            + (cause.message if cause is not None else unavailable.stage)
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
            self._window.set_status("Fit cancelled.")
        self._set_status("Fit cancelled.")

    def _on_worker_finished(self, worker: GlobalFitWorker) -> None:
        """Release owner custody only after the direct terminal was emitted."""

        self._finish_worker(expected=worker)

    def _finish_worker(
        self,
        expected: GlobalFitWorker | None = None,
    ) -> None:
        worker = self._worker
        if expected is not None and worker is not expected:
            return
        self._worker = None
        self._gate.release(self)
        closing = self._closing
        if self._window is not None and not self._application_close_fenced and not closing:
            self._window.set_executing(False)
            self._restore_candidate_presentation()
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

    def _build_request_from_setup(
        self,
        setup: FitSetupSnapshot,
    ) -> FitExecutionRequest:
        request = build_fit_execution_request(
            origin=setup.origin,
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
        )
        revalidate_fit_request_inclusions(
            document=self._session.document,
            request=request,
        )
        return request

    def _build_request(self) -> FitExecutionRequest:
        setup = self._accepted_setup()
        if setup is None:
            raise RuntimeError("Fit has no typed local setup.")
        return self._build_request_from_setup(setup)

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
