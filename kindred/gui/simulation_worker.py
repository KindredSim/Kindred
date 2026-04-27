"""
Simulation worker thread with progress tracking and cancellation support.

This module provides:
- SimulationWorker: QThread for background simulation execution
- Progress tracking with real-time updates
- Cancellation support with polling in solver
- Robust cleanup and error handling
"""

import logging
import traceback
import numpy as np
from typing import Any, Dict, Mapping, Optional
from PySide6 import QtCore
from PySide6.QtCore import Signal

from kindred.core.exceptions import KindredError, SimulationCancelled
from kindred.core.simulation_failure import (
    build_simulation_failure,
    simulation_failure_from_exception,
)
from kindred.core.simulation_containment import contained_payloads_equal
from kindred.core.simulation_result_finalization import evaluate_simulation_result_algebra
from kindred.core.simulation_result_payload import (
    build_secondary_simulation_success_payload,
    build_simulation_success_payload,
)

logger = logging.getLogger(__name__)


def _formatted_exception_stack_trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _failure_payload_with_stack_trace(payload: dict[str, Any], *, exc: BaseException) -> dict[str, Any]:
    context = payload.get("context")
    if isinstance(context, dict) and str(context.get("stack_trace") or "").strip():
        return payload

    stack_trace = _formatted_exception_stack_trace(exc)
    if not stack_trace:
        return payload

    enriched_payload = dict(payload)
    enriched_context = dict(context) if isinstance(context, dict) else {}
    enriched_context["stack_trace"] = stack_trace
    enriched_payload["context"] = enriched_context
    return enriched_payload


def _pre_active_timeout_failure(exc: BaseException, *, phase: str) -> dict[str, Any]:
    timeout_s = float(getattr(exc, "timeout_s", 0.0) or 0.0)
    return build_simulation_failure(
        "timeout",
        str(exc),
        code="E306",
        details={
            "timeout_phase": str(phase),
            f"{phase}_timeout_s": timeout_s,
        },
        exc_type=exc.__class__.__name__,
    )


class SimulationWorker(QtCore.QThread):
    """
    Background worker for running ODE simulations.

    Runs simulations in a separate thread to avoid blocking the GUI.
    Provides progress updates and supports cancellation.

    Signals:
        progress(int, str): Emitted during simulation with (percent, message)
        result_ready(dict): Emitted when simulation completes successfully
        error(object): Emitted when an error occurs or simulation is cancelled

    Usage:
        worker = SimulationWorker(mechanism, initials, t_span, solver_config, parent)
        worker.progress.connect(update_progress_dialog)
        worker.result_ready.connect(on_simulation_complete)
        worker.error.connect(on_simulation_error)
        worker.start()

        # To cancel:
        worker.cancel()
    """

    progress = Signal(int, str)  # (percent_complete, status_message)
    result_ready = Signal(dict)  # Result: {t, Y, species_names, stats}
    error = Signal(object)       # Structured failure payload

    def __init__(
        self,
        mechanism_text: str,
        initials: Dict[str, float],
        t_span: tuple,
        solver_config: Dict[str, Any],
        parent=None,
        prepared: Optional[Dict[str, Any]] = None,
        include_mechanism_in_result_payload: bool = True,
    ):
        """
        Initialize simulation worker.

        Args:
            mechanism_text: Mechanism DSL text
            initials: Initial conditions {species: concentration}
            t_span: Time span (t_start, t_end)
            solver_config: Solver configuration:
                - solver: 'Radau' or 'BDF'
                - rtol: Relative tolerance (default: 1e-6)
                - atol: Absolute tolerance (default: 1e-12)
                - grid: Grid configuration (default: {'N': 100})
                - temperature_K: float (default: 298.15)
            parent: Parent QObject
        """
        super().__init__(parent)

        self._mechanism_text = mechanism_text
        self._initials = initials
        self._t_span = t_span
        self._solver_config = solver_config
        self._prepared = prepared
        self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)

        self._cancelled = False
        self._is_running = False

    def cancel(self):
        """Request cancellation of the simulation."""
        logger.info("Simulation cancellation requested")
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def _emit_cancelled(self) -> None:
        self.error.emit(build_simulation_failure("cancelled", "Simulation cancelled by user", code="E305"))

    def _build_cancel_event(self):
        def cancel_event(t, y):  # noqa: ANN001
            return -1.0 if self._cancelled else 1.0

        cancel_event.terminal = True
        cancel_event._kindred_cancel_event = True  # type: ignore[attr-defined]
        cancel_event._kindred_cancelled = lambda: bool(self._cancelled)  # type: ignore[attr-defined]
        return cancel_event

    def _build_progress_callback(self):
        def progress_callback(t: float, t_start: float, t_end: float):
            if self._cancelled:
                return
            percent = 30 + int(60 * (t - t_start) / (t_end - t_start))
            percent = max(30, min(90, percent))
            self.progress.emit(percent, f"Solving ODEs... t={t:.3f}/{t_end:.3f}")

        return progress_callback

    def _execution_request_from_attached_plan(self) -> Optional[dict]:
        simulation_plan = getattr(self, "_simulation_plan", None)
        if simulation_plan is None:
            return None

        from kindred.core.simulation_plan import SimulationPlan

        if isinstance(simulation_plan, SimulationPlan):
            return simulation_plan.to_execution_request().to_payload()
        if isinstance(simulation_plan, Mapping):
            return SimulationPlan.from_payload(simulation_plan).to_execution_request().to_payload()
        raise TypeError("Simulation worker _simulation_plan must be a SimulationPlan or mapping.")

    def _prepare_worker_run(
        self,
        *,
        prepared_payload: Optional[dict],
        execution_request: Optional[dict],
        cancel_event,
        progress_callback,
    ):
        from kindred.core.simulation_preparation import (
            SimulationPreparationError,
            prepare_simulation_worker_run,
        )

        try:
            return prepare_simulation_worker_run(
                mechanism_text=str(self._mechanism_text),
                initials=dict(self._initials or {}),
                t_span=(float(self._t_span[0]), float(self._t_span[1])),
                solver_config=dict(self._solver_config or {}),
                prepared_payload=prepared_payload if isinstance(prepared_payload, dict) else None,
                execution_request=execution_request if isinstance(execution_request, dict) else None,
                events=[cancel_event],
                progress_callback=progress_callback,
            )
        except SimulationPreparationError as exc:
            logger.error("Simulation preparation failed (%s): %s", exc.stage, exc, exc_info=True)
            self.error.emit(
                _failure_payload_with_stack_trace(
                    simulation_failure_from_exception(
                        exc,
                        kind="preparation_error",
                        details={"stage": str(exc.stage or "unknown")},
                    ),
                    exc=exc,
                )
            )
            return None

    def _solve_prepared_request(self, *, solve_ode, request) -> object:  # noqa: ANN001
        if self._cancelled:
            raise SimulationCancelled()
        return solve_ode(request)

    def _evaluate_algebra_outputs(
        self,
        *,
        mechanism,
        result,
        species_names: list[str],
        initials_for_algebra: Dict[str, float],
    ) -> tuple[np.ndarray, list[str], dict, list[dict], list[dict]]:
        finalized = evaluate_simulation_result_algebra(
            mechanism=mechanism,
            result=result,
            species_names=species_names,
            initials_for_algebra=initials_for_algebra,
            simulation_plan=getattr(self, "_simulation_plan", None),
        )
        if len(finalized.species_names) > len(species_names) or finalized.algebra_scalars or finalized.algebra_errors:
            self.progress.emit(96, "Evaluating algebraic species...")
            logger.info("Evaluating algebraic species...")
        return (
            finalized.y,
            finalized.species_names,
            finalized.algebra_scalars,
            finalized.algebra_errors,
            finalized.warnings,
        )

    def _build_result_payload(
        self,
        *,
        mechanism,
        result,
        base_species_count: int,
        extended_Y: np.ndarray,
        extended_species_names: list[str],
        algebra_scalars: dict,
        algebra_errors: list[dict],
        warnings: list[dict],
        solver: str,
    ) -> Dict[str, Any]:
        builder = (
            build_simulation_success_payload
            if self._include_mechanism_in_result_payload
            else build_secondary_simulation_success_payload
        )
        kwargs: Dict[str, Any] = {
            "result": result,
            "y": extended_Y,
            "species_names": extended_species_names,
            "base_species_count": int(base_species_count),
            "algebra_scalars": algebra_scalars,
            "algebra_errors": algebra_errors,
            "warnings": warnings,
            "solver": solver,
            "mechanism_text": self._mechanism_text,
            "solver_config": self._solver_config,
        }
        if self._include_mechanism_in_result_payload:
            kwargs["mechanism"] = mechanism
        return builder(**kwargs)

    def run(self):
        """
        Execute simulation in background thread.

        This method is called automatically when start() is invoked.
        Do not call directly.
        """
        self._is_running = True
        stage = "worker_setup"

        try:
            logger.info("Starting simulation in worker thread")

            # Check cancellation before starting
            if self._cancelled:
                self._emit_cancelled()
                return

            self.progress.emit(0, "Initializing simulation...")

            from kindred.core.simulator.solvers import solve_ode

            execution_request = self._execution_request_from_attached_plan()
            prepared_payload = getattr(self, "_prepared", None)
            if isinstance(execution_request, dict) and execution_request.get("prepared_payload") is not None:
                self.progress.emit(10, "Using structured execution request...")
            elif prepared_payload is not None:
                self.progress.emit(10, "Using precompiled mechanism...")
            else:
                self.progress.emit(10, "Parsing mechanism...")

            cancel_event = self._build_cancel_event()
            progress_callback = self._build_progress_callback()

            # Prepare simulation request
            stage = "prepare_simulation"
            self.progress.emit(30, "Preparing solver...")
            prepared = self._prepare_worker_run(
                prepared_payload=prepared_payload if isinstance(prepared_payload, dict) else None,
                execution_request=execution_request if isinstance(execution_request, dict) else None,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )
            if prepared is None:
                return

            mechanism = prepared.mechanism
            species_names = list(prepared.species_names)
            base_species_count = len(species_names)
            initials_for_algebra = prepared.initials_for_algebra
            prepared_warning_payloads = [
                build_simulation_failure(
                    "preparation_warning",
                    str(message),
                    details={"stage": "prepare_run_context"},
                )
                for message in (getattr(prepared, "warnings", None) or [])
            ]
            request = prepared.request
            solver = str(getattr(request, "solver", "") or "")
            if prepared.solver_warning:
                logger.warning(
                    "Solver normalization: %s (requested=%r)",
                    prepared.solver_warning,
                    prepared.solver_input,
                )
            if prepared.temperature_schedule is not None:
                logger.info("Using temperature schedule from DSL: %s", prepared.temperature_schedule)

            # Run simulation
            stage = "solve_request"
            self.progress.emit(30, f"Running {solver} solver...")
            try:
                result = self._solve_prepared_request(solve_ode=solve_ode, request=request)
            except SimulationCancelled:
                self._emit_cancelled()
                return
            except KindredError as exc:
                logger.error("Simulation failed (%s): %s", exc.code, exc, exc_info=True)
                self.error.emit(
                    _failure_payload_with_stack_trace(
                        simulation_failure_from_exception(exc),
                        exc=exc,
                    )
                )
                return
            except Exception as exc:
                logger.error("Simulation failed: %s", exc, exc_info=True)
                self.error.emit(
                    _failure_payload_with_stack_trace(
                        simulation_failure_from_exception(exc),
                        exc=exc,
                    )
                )
                return

            if self._cancelled:
                self._emit_cancelled()
                return

            # Prepare results
            stage = "finalize_results"
            self.progress.emit(95, "Finalizing results...")
            stage = "algebra_evaluation"
            extended_Y, extended_species_names, algebra_scalars, algebra_errors, warnings = self._evaluate_algebra_outputs(
                mechanism=mechanism,
                result=result,
                species_names=species_names,
                initials_for_algebra=initials_for_algebra,
            )
            warnings = list(prepared_warning_payloads) + list(warnings)
            stage = "emit_result"
            result_dict = self._build_result_payload(
                mechanism=mechanism,
                result=result,
                base_species_count=base_species_count,
                extended_Y=extended_Y,
                extended_species_names=extended_species_names,
                algebra_scalars=algebra_scalars,
                algebra_errors=algebra_errors,
                warnings=warnings,
                solver=solver,
            )

            self.progress.emit(100, "Complete!")
            logger.info("Simulation completed successfully")
            self.result_ready.emit(result_dict)

        except Exception as e:
            logger.error("Unexpected error in simulation worker during %s: %s", stage, e, exc_info=True)
            self.error.emit(
                _failure_payload_with_stack_trace(
                    simulation_failure_from_exception(
                        e,
                        kind="worker_internal_error",
                        details={"stage": stage},
                    ),
                    exc=e,
                )
            )

        finally:
            self._is_running = False
            logger.info("Simulation worker thread finished")

    def cleanup(self):
        """
        Clean up resources before thread termination.

        Contract: initiates shutdown only. This method sets cancellation flags and
        may trigger worker-side early returns, but it does not block and does not
        guarantee the thread has stopped when it returns.
        """
        logger.info("Cleaning up simulation worker")
        self._cancelled = True
        self._is_running = False

        # Wait for thread to finish (with timeout)
        if self.isRunning():
            logger.warning("Simulation thread is still running; cleanup() does not block waiting for it to finish")


class ContainedSimulationWorker(QtCore.QThread):
    """Qt adapter for a controller-owned warm contained simulation owner."""

    progress = Signal(int, str)
    result_ready = Signal(dict)
    error = Signal(object)

    def __init__(
        self,
        *,
        owner,
        simulation_plan_payload: Mapping[str, Any],
        include_mechanism_in_result_payload: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._owner = owner
        self._simulation_plan_payload = dict(simulation_plan_payload)
        self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
        self._cancelled = False
        self._is_running = False
        self._owner_closed = False

    def cancel(self) -> None:
        logger.info("Contained simulation cancellation requested")
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return bool(self._cancelled)

    def _close_owner_once(self, *, kill: bool) -> None:
        if self._owner_closed:
            return
        self._owner_closed = True
        owner = self._owner
        if owner is None or not hasattr(owner, "close"):
            return
        try:
            owner.close(kill=bool(kill))
        except Exception as exc:
            logger.debug("Contained simulation owner cleanup failed: %s", exc, exc_info=True)

    def run(self) -> None:
        self._is_running = True
        try:
            if self._cancelled:
                raise SimulationCancelled()
            owner = self._owner
            if owner is None:
                raise RuntimeError("Contained simulation owner is unavailable.")
            owner_running = False
            try:
                owner_running = bool(getattr(owner, "is_running", False))
            except Exception:
                owner_running = False
            self.progress.emit(
                0,
                "Running simulation..." if owner_running else "Initializing simulation...",
            )
            request_payload = {
                "include_mechanism_in_result_payload": bool(
                    self._include_mechanism_in_result_payload
                ),
            }
            owner_plan_payload = getattr(owner, "simulation_plan_payload", None)
            if not isinstance(owner_plan_payload, Mapping) or not contained_payloads_equal(
                dict(owner_plan_payload),
                self._simulation_plan_payload,
            ):
                request_payload["simulation_plan_payload"] = dict(self._simulation_plan_payload)
            result = owner.solve(
                request_payload,
                cancellation_check=self.is_cancelled,
            )
            if self._cancelled:
                raise SimulationCancelled()
            self.progress.emit(100, "Complete!")
            self.result_ready.emit(dict(result))
        except SimulationCancelled as exc:
            self._close_owner_once(kill=True)
            self.error.emit(simulation_failure_from_exception(exc))
        except BaseException as exc:  # noqa: BLE001 - Qt boundary must serialize all failures
            self._close_owner_once(kill=True)
            from kindred.core.simulation_containment import (
                SimulationContainmentAcceptTimeout,
                SimulationContainmentChildFailure,
                SimulationContainmentProtocolError,
                SimulationContainmentStartupTimeout,
                SimulationContainmentTimeout,
            )

            if isinstance(exc, SimulationContainmentTimeout):
                self.error.emit(dict(exc.failure))
            elif isinstance(exc, SimulationContainmentStartupTimeout):
                self.error.emit(_pre_active_timeout_failure(exc, phase="startup"))
            elif isinstance(exc, SimulationContainmentAcceptTimeout):
                self.error.emit(_pre_active_timeout_failure(exc, phase="accept"))
            elif isinstance(exc, SimulationContainmentChildFailure):
                failure = dict(exc.failure)
                details = failure.get("details")
                failure["details"] = dict(details) if isinstance(details, Mapping) else {}
                failure["details"]["source"] = "simulation_containment"
                self.error.emit(failure)
            elif isinstance(exc, SimulationContainmentProtocolError):
                self.error.emit(simulation_failure_from_exception(exc, kind="simulation_containment_protocol"))
            else:
                self.error.emit(simulation_failure_from_exception(exc))
        finally:
            self._is_running = False

    def cleanup(self) -> None:
        self._cancelled = True
        self._close_owner_once(kill=True)
