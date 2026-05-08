from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore

from kindred.core.analysis.dataset_parameter_overrides import split_fit_dataset_parameter_overrides
from kindred.gui.fitting.runtime_readiness import FittingRuntimeAcceptedLaunch
from kindred.gui.fitting.worker import GlobalFitWorker

if TYPE_CHECKING:
    from kindred.gui.fitting.window import FittingWindow


class FittingAcceptedLaunchWorkerOwner:
    """Owns worker construction and initial worker publication for accepted fitting launches."""

    def __init__(self, window: "FittingWindow") -> None:
        self._window = window

    def start_runtime_launch(self, accepted_launch: FittingRuntimeAcceptedLaunch) -> None:
        window = self._window
        ready_identity = accepted_launch.identity
        window.fit_run_state_owner.set_active_dataset_ids(
            [spec.dataset_id for spec in ready_identity.datasets]
        )
        _dataset_params_map, variable_params_map = split_fit_dataset_parameter_overrides(ready_identity.dataset_overrides)
        window._active_variable_specs = variable_params_map
        run_stamp = accepted_launch.stamp
        run_stamp_hash = accepted_launch.stamp_hash
        run_stamp_short = accepted_launch.stamp_short
        window._run_results_tab.set_run_stamp(
            run_stamp,
            run_stamp_hash,
            run_stamp_short,
        )
        window._results_summary_button.setEnabled(True)
        self.start_worker(accepted_launch)

    def start_worker(
        self,
        accepted_launch: FittingRuntimeAcceptedLaunch,
    ) -> None:
        window = self._window
        datasets = list(accepted_launch.datasets)
        config = accepted_launch.config
        dataset_overrides = list(accepted_launch.dataset_overrides)
        weights = accepted_launch.weights
        fit_evaluator = accepted_launch.fit_evaluator
        runtime_session = accepted_launch.session
        stamp = accepted_launch.stamp
        stamp_hash = accepted_launch.stamp_hash
        stamp_short = accepted_launch.stamp_short
        window.fit_run_state_owner.set_active_run_stamp_hash(str(stamp_hash or ""))
        fit_runtime_ledger = getattr(runtime_session, "ledger", None)
        worker = GlobalFitWorker(
            datasets,
            dict(config["parameters"]),
            dataset_overrides=list(dataset_overrides),
            bounds=config.get("bounds"),
            weights=weights,
            method=config.get("method", "trf"),
            max_nfev=config.get("max_nfev", 1000),
            ftol=config.get("ftol", 1e-10),
            xtol=config.get("xtol", 1e-10),
            seed=config.get("seed"),
            log10_params=config.get("log10_params"),
            fit_evaluator=fit_evaluator,
            fit_runtime_session=runtime_session,
            fit_runtime_max_lanes=int(accepted_launch.lane_count),
            fit_runtime_ledger=fit_runtime_ledger,
            fit_func=window._fit_func,
            solver=str(accepted_launch.identity.requested_solver),
            rtol=float(accepted_launch.identity.requested_rtol),
            atol=float(accepted_launch.identity.requested_atol),
            best_update_interval_s=0.25,
            plot_update_interval_s=2.0,
            run_stamp=dict(stamp),
            run_stamp_hash=str(stamp_hash),
            run_stamp_short=str(stamp_short),
            parent=window,
        )
        window._worker = worker
        worker.progress.connect(window._dispatch_fit_worker_progress)
        if hasattr(worker, "bestUpdated"):
            try:
                worker.bestUpdated.connect(
                    window._dispatch_fit_worker_best_update,
                    QtCore.Qt.ConnectionType.QueuedConnection,
                )
            except Exception:
                worker.bestUpdated.connect(window._dispatch_fit_worker_best_update)
        worker.finished.connect(window._dispatch_fit_worker_finished)
        worker.error.connect(window._dispatch_fit_worker_error)
        worker.start()
        if window._worker_is_running(worker):
            window._set_running_state(True)
        window._paused = False
        window._pause_button.setEnabled(True)
        window._resume_button.setEnabled(False)
