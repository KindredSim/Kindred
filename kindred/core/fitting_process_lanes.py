"""Process-backed fitting evaluator lanes for the supported serial evaluator path."""

from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled
from kindred.core.fitting_evaluation import (
    PreparedFittingExecutionContext,
    SerialFittingEvaluator,
    evaluate_fitting_series,
)
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    SimulationExecutionRequest,
)

_SLOT_EVALUATOR: Optional[SerialFittingEvaluator] = None
_SLOT_ID: Optional[int] = None
_SLOT_PREPARE_COUNT = 0
_SLOT_EVAL_COUNT = 0


def fitting_process_lane_payload_from_evaluator(fit_evaluator) -> Optional[Dict[str, Any]]:
    if type(fit_evaluator) is not SerialFittingEvaluator:
        return None
    export = getattr(fit_evaluator, "_kindred_process_lane_payload", None)
    if not callable(export):
        return None
    try:
        payload = copy.deepcopy(export())
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None
    return payload


def _build_serial_evaluator_from_process_payload(payload: Mapping[str, Any]) -> SerialFittingEvaluator:
    context = PreparedFittingExecutionContext(
        execution_request=SimulationExecutionRequest.from_mapping(
            dict(payload.get("execution_request") or {})
        ),
        requested_param_names=list(payload.get("requested_param_names") or []),
        prepared_metadata=PreparedSimulationMetadata.from_mapping(
            dict(payload.get("prepared_metadata") or {})
        ),
        temperature_K=float(payload.get("temperature_K") or 0.0),
        initial_prefix=str(payload.get("initial_prefix") or "init:"),
    )
    return SerialFittingEvaluator(
        context,
        fixed_params=dict(payload.get("fixed_params") or {}),
        fixed_param_origins=dict(payload.get("fixed_param_origins") or {}),
    )


def initialize_fitting_process_lane(slot: int, evaluator_payload: Mapping[str, Any]) -> None:
    global _SLOT_EVALUATOR, _SLOT_EVAL_COUNT, _SLOT_ID, _SLOT_PREPARE_COUNT

    _SLOT_ID = int(slot)
    _SLOT_EVALUATOR = _build_serial_evaluator_from_process_payload(evaluator_payload)
    _SLOT_PREPARE_COUNT = 0
    _SLOT_EVAL_COUNT = 0


def _error_context_payload(context: object) -> Optional[Dict[str, Any]]:
    if context is None:
        return None
    return {
        "line": getattr(context, "line", None),
        "col": getattr(context, "col", None),
        "line_text": getattr(context, "line_text", None),
        "file_path": getattr(context, "file_path", None),
        "stack_trace": getattr(context, "stack_trace", None),
    }


def _error_payload(exc: BaseException, *, dataset_id: str, failed_params: Mapping[str, float]) -> Dict[str, Any]:
    if isinstance(exc, FitSimulationError):
        return {
            "kind": "fit_simulation",
            "message": str(getattr(exc, "message", None) or str(exc)),
            "failed_params": dict(getattr(exc, "failed_params", None) or {}),
            "details": dict(getattr(exc, "details", {}) or {}),
            "context": _error_context_payload(getattr(exc, "context", None)),
            "error_provenance": {
                "dataset": str(dataset_id),
                "provenance": getattr(exc, "provenance", None),
            },
            "final_error_message": str(getattr(exc, "message", None) or str(exc)),
        }
    return {
        "kind": "generic",
        "message": str(exc),
        "failed_params": dict(failed_params or {}),
        "details": {},
        "context": None,
        "error_provenance": {"dataset": str(dataset_id)},
        "final_error_message": str(exc),
    }


def run_fitting_dataset_evaluation_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    global _SLOT_EVAL_COUNT, _SLOT_PREPARE_COUNT

    if _SLOT_EVALUATOR is None or _SLOT_ID is None:
        raise RuntimeError("Fitting process lane was not initialized.")

    slot = int(task.get("slot"))
    if slot != int(_SLOT_ID):
        raise RuntimeError(f"Fitting process lane received slot {slot}, expected {_SLOT_ID}.")

    dataset_id = str(task.get("dataset_id") or "")
    failed_params = dict(task.get("failed_param_snapshot") or {})
    was_prepared = getattr(_SLOT_EVALUATOR, "_prepared_run", None) is not None
    base_payload: Dict[str, Any] = {
        "index": int(task.get("index")),
        "dataset_id": dataset_id,
        "slot": int(_SLOT_ID),
        "worker_pid": int(os.getpid()),
    }
    try:
        sim_result = evaluate_fitting_series(
            _SLOT_EVALUATOR,
            dict(task.get("full_params") or {}),
            origins=dict(task.get("parameter_origins") or {}),
            failed_params=failed_params,
        )
        is_prepared = getattr(_SLOT_EVALUATOR, "_prepared_run", None) is not None
        if not was_prepared and is_prepared:
            _SLOT_PREPARE_COUNT += 1
        _SLOT_EVAL_COUNT += 1
        species = {
            str(name): np.asarray(values, dtype=float).reshape(-1).copy()
            for name, values in dict(sim_result.species or {}).items()
        }
        base_payload.update(
            {
                "ok": True,
                "sim_time": np.asarray(sim_result.t, dtype=float).reshape(-1).copy(),
                "sim_species": species,
                "cold_start": bool(not was_prepared and is_prepared),
                "prepare_count": int(_SLOT_PREPARE_COUNT),
                "eval_count": int(_SLOT_EVAL_COUNT),
            }
        )
        return base_payload
    except (FittingCancelled, SimulationCancelled):
        raise
    except Exception as exc:
        if not was_prepared and getattr(_SLOT_EVALUATOR, "_prepared_run", None) is not None:
            _SLOT_PREPARE_COUNT += 1
        _SLOT_EVAL_COUNT += 1
        base_payload.update(
            {
                "ok": False,
                "sim_time": None,
                "sim_species": {},
                "cold_start": False,
                "prepare_count": int(_SLOT_PREPARE_COUNT),
                "eval_count": int(_SLOT_EVAL_COUNT),
                "error": _error_payload(exc, dataset_id=dataset_id, failed_params=failed_params),
            }
        )
        return base_payload


@dataclass
class _ProcessLaneSlot:
    slot: int
    executor: Any


class ProcessBackedFittingEvaluatorLanePool:
    def __init__(self, evaluator_payload: Mapping[str, Any], *, max_lanes: int) -> None:
        self._evaluator_payload = copy.deepcopy(dict(evaluator_payload or {}))
        self._max_lanes = max(1, int(max_lanes))
        self._slots: Dict[int, _ProcessLaneSlot] = {}
        self._worker_pids: set[int] = set()
        self._slot_stats: Dict[int, Dict[str, int]] = {}
        self._closed = False

    @property
    def max_lanes(self) -> int:
        return self._max_lanes

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(self, slot: int, task: Mapping[str, Any]):
        slot_index = int(slot)
        if slot_index < 0 or slot_index >= self._max_lanes:
            raise RuntimeError(
                f"Fitting process lane slot {slot_index} is outside the retained lane cap {self._max_lanes}."
            )
        if self._closed:
            raise RuntimeError("Fitting process lane pool is closed.")
        lane_slot = self._slots.get(slot_index)
        if lane_slot is None:
            lane_slot = self._create_slot(slot_index)
            self._slots[slot_index] = lane_slot
        payload = dict(task)
        payload["slot"] = slot_index
        return lane_slot.executor.submit(run_fitting_dataset_evaluation_task, payload)

    def record_result(self, payload: Mapping[str, Any]) -> None:
        slot = int(payload.get("slot"))
        pid = int(payload.get("worker_pid"))
        self._worker_pids.add(pid)
        current = self._slot_stats.setdefault(
            slot,
            {"pid": pid, "cold_starts": 0, "eval_count": 0},
        )
        current["pid"] = pid
        current["cold_starts"] = max(int(current.get("cold_starts", 0)), int(payload.get("prepare_count") or 0))
        current["eval_count"] = max(int(current.get("eval_count", 0)), int(payload.get("eval_count") or 0))

    def worker_pids(self) -> tuple[int, ...]:
        return tuple(sorted(self._worker_pids))

    def slot_stats(self) -> Dict[int, Dict[str, int]]:
        return {int(slot): dict(stats) for slot, stats in self._slot_stats.items()}

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = True, terminate: bool = False) -> None:
        self._closed = True
        for lane_slot in list(self._slots.values()):
            executor = lane_slot.executor
            if terminate:
                if self._terminate_executor_workers(executor, cancel_futures=cancel_futures):
                    continue
            executor.shutdown(wait=bool(wait), cancel_futures=bool(cancel_futures))
        self._slots.clear()

    def _create_slot(self, slot: int) -> _ProcessLaneSlot:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_fitting_process_lane,
            initargs=(int(slot), copy.deepcopy(self._evaluator_payload)),
        )
        return _ProcessLaneSlot(slot=int(slot), executor=executor)

    @staticmethod
    def _terminate_executor_workers(executor: Any, *, cancel_futures: bool) -> bool:
        terminate_workers = getattr(executor, "terminate_workers", None)
        if callable(terminate_workers):
            terminate_workers()
            return True

        processes_map = getattr(executor, "_processes", None)
        if not processes_map:
            return False
        processes = list(processes_map.values())
        for process in processes:
            is_alive = getattr(process, "is_alive", None)
            terminate = getattr(process, "terminate", None)
            if callable(terminate) and (not callable(is_alive) or bool(is_alive())):
                terminate()
        executor.shutdown(wait=False, cancel_futures=bool(cancel_futures))
        for process in processes:
            join = getattr(process, "join", None)
            if callable(join):
                join(timeout=0.2)
        for process in processes:
            is_alive = getattr(process, "is_alive", None)
            kill = getattr(process, "kill", None)
            if callable(kill) and callable(is_alive) and bool(is_alive()):
                kill()
                join = getattr(process, "join", None)
                if callable(join):
                    join(timeout=0.2)
        return True
