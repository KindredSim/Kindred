from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np

from kindred.core.simulation_identity import coerce_simulation_identity
from kindred.core.simulation_failure import (
    build_simulation_failure,
    serialize_algebra_error,
    simulation_failure_from_exception,
)
from kindred.core.simulation_result_payload import (
    build_secondary_simulation_success_payload,
    build_simulation_success_payload,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "apply_worker_blas_limits",
    "batch_mechanism_signature",
    "compute_effective_batch_workers",
    "initialize_batch_worker",
    "run_batch_simulation_task",
]

BLAS_THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_WORKER_CACHE_MAXSIZE = 8
_WORKER_PREPARED_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def compute_effective_batch_workers(*, num_sets: int, max_parallel_workers: int) -> int:
    """
    Return effective worker count using:
    min(num_sets, max(1, cpu_count-1), max_parallel_workers).
    """
    n_sets = max(0, int(num_sets))
    cap = max(1, int(max_parallel_workers))
    cpu = os.cpu_count()
    cpu_cap = max(1, int(cpu) - 1) if isinstance(cpu, int) and cpu > 0 else 1
    if n_sets <= 0:
        return 1
    return int(min(n_sets, cpu_cap, cap))


def apply_worker_blas_limits(*, enabled: bool, environ: MutableMapping[str, str] | None = None) -> None:
    """Set BLAS/OpenMP thread environment vars to 1 when enabled."""
    if not bool(enabled):
        return
    env = os.environ if environ is None else environ
    for name in BLAS_THREAD_ENV_VARS:
        env[str(name)] = "1"


def initialize_batch_worker(limit_blas_threads: bool) -> None:
    """Process-pool initializer for batch simulation workers."""
    apply_worker_blas_limits(enabled=bool(limit_blas_threads))


def batch_mechanism_signature(
    *,
    mechanism_text: str = "",
    temperature_K: float = 298.15,
    use_sparse_jacobian: bool = False,
    wegscheider_cyclicity_enabled: bool = False,
    simulation_identity: Mapping[str, Any] | object | None = None,
) -> str:
    """Stable signature for per-process prepared-runtime reuse."""
    identity = coerce_simulation_identity(simulation_identity)
    if identity is not None:
        return identity.prepared_runtime_key()
    payload = {
        "dsl": str(mechanism_text or ""),
        "temperature_K": float(temperature_K),
        "use_sparse_jacobian": bool(use_sparse_jacobian),
        "wegscheider_cyclicity_enabled": bool(wegscheider_cyclicity_enabled),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8", "ignore")).hexdigest()


def _ensure_worker_cache_bound() -> None:
    while len(_WORKER_PREPARED_CACHE) > int(_WORKER_CACHE_MAXSIZE):
        _key, _entry = _WORKER_PREPARED_CACHE.popitem(last=False)
        try:
            if isinstance(_entry, dict):
                _entry.clear()
        except Exception as exc:
            logger.debug("Failed to clear evicted worker cache entry: %s", exc, exc_info=True)
        del _key
        del _entry


def _prepared_entry(
    *,
    signature: str,
    mechanism_text: str,
    temperature_K: float,
    wegscheider_cyclicity_enabled: bool,
) -> Dict[str, Any]:
    entry = _WORKER_PREPARED_CACHE.get(signature)
    if isinstance(entry, dict):
        _WORKER_PREPARED_CACHE.move_to_end(signature)
        return entry

    from kindred.core.simulation_preparation import prepare_bound_mechanism

    bound = prepare_bound_mechanism(
        mechanism_text=str(mechanism_text or ""),
        param_names=[],
        temperature_K=float(temperature_K),
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    entry = {
        "bound": bound,
        "prepared_payload": bound.as_worker_payload(),
    }
    _WORKER_PREPARED_CACHE[signature] = entry
    _ensure_worker_cache_bound()
    return entry


def _prepared_payload_from_bound(bound: Any) -> Dict[str, Any]:
    if hasattr(bound, "as_worker_payload"):
        payload = bound.as_worker_payload()
        if isinstance(payload, dict):
            return dict(payload)
    mechanism = getattr(bound, "mechanism", None)
    species_names = [str(name) for name in list(getattr(bound, "species_names", []) or [])]
    return {
        "version": 1,
        "mechanism": mechanism,
        "rhs": getattr(bound, "rhs"),
        "y0": np.array(getattr(bound, "y0", []), copy=True, dtype=float).reshape(-1),
        "species_names": species_names,
        "mechanism_text": "",
        "temperature_schedule": None,
        "jacobian_func": None,
    }


def _run_batch_simulation_task_impl(task: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Execute one batch-set simulation in a worker process.

    This function is process-safe (no Qt objects) and supports per-process
    prepared-runtime reuse keyed by `mechanism_signature`.
    """
    from kindred.core.algebra.simulation_series import evaluate_algebra_series_for_simulation
    from kindred.core.simulation_preparation import (
        SimulationPreparationError,
        prepare_simulation_worker_run,
    )
    from kindred.core.simulator.solvers import solve_ode

    execution_request = task.get("execution_request")
    mechanism_text = str(task.get("mechanism_text") or "")
    solver_config = dict(task.get("solver_config") or {})
    initials = dict(task.get("initials") or {})
    signature = str(task.get("mechanism_signature") or "").strip()
    simulation_identity = task.get("simulation_identity")
    include_mechanism_in_result_payload = bool(task.get("include_mechanism_in_result_payload", False))
    structured_prepared_request = isinstance(execution_request, Mapping) and execution_request.get("prepared_payload") is not None
    if isinstance(execution_request, Mapping):
        if structured_prepared_request:
            mechanism_text = str(execution_request.get("mechanism_text") or "")
        else:
            mechanism_text = str(execution_request.get("mechanism_text") or mechanism_text or "")
        solver_config = dict(execution_request.get("solver_config") or solver_config)
        initials = dict(execution_request.get("initials") or initials)
        simulation_identity = execution_request.get("simulation_identity") or simulation_identity
    temperature_K = float(solver_config.get("temperature_K") or 298.15)
    wegscheider_enabled = bool(solver_config.get("wegscheider_cyclicity_enabled", False))
    use_sparse_jacobian = bool(solver_config.get("use_sparse_jacobian"))
    # Structured prepared execution is authoritative and may safely reuse a
    # structural prepared-runtime key. Text-driven batch preview tasks must key
    # the worker cache by the actual mechanism text, otherwise later slider
    # positions can reuse the first prepared runtime and ignore new parameter
    # values.
    if not structured_prepared_request and mechanism_text:
        signature = batch_mechanism_signature(
            mechanism_text=mechanism_text,
            temperature_K=temperature_K,
            use_sparse_jacobian=use_sparse_jacobian,
            wegscheider_cyclicity_enabled=bool(wegscheider_enabled),
        )
    elif not signature:
        signature = batch_mechanism_signature(
            simulation_identity=simulation_identity,
            mechanism_text=mechanism_text,
            temperature_K=temperature_K,
            use_sparse_jacobian=use_sparse_jacobian,
            wegscheider_cyclicity_enabled=bool(wegscheider_enabled),
        )

    t_span_raw = task.get("t_span") or (0.0, float(task.get("t_end") or 0.0))
    try:
        t_start, t_end = float(t_span_raw[0]), float(t_span_raw[1])
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("Invalid t_span %r in batch task; falling back to (0, t_end): %s", t_span_raw, exc)
        t_start, t_end = 0.0, float(task.get("t_end") or 0.0)
    t_span = (float(t_start), float(t_end))

    if isinstance(execution_request, Mapping):
        prepared = prepare_simulation_worker_run(execution_request=execution_request)
        if structured_prepared_request:
            mechanism_text = str(execution_request.get("mechanism_text") or "")
        else:
            mechanism_text = str(execution_request.get("mechanism_text") or mechanism_text or "")
    else:
        try:
            entry = _prepared_entry(
                signature=signature,
                mechanism_text=mechanism_text,
                temperature_K=temperature_K,
                wegscheider_cyclicity_enabled=bool(wegscheider_enabled),
            )
        except Exception as exc:
            try:
                prepare_simulation_worker_run(
                    mechanism_text=mechanism_text,
                    initials=initials,
                    t_span=t_span,
                    solver_config=solver_config,
                    prepared_payload={"version": 0},
                )
            except SimulationPreparationError as prep_exc:
                if prep_exc.stage == "solver_config":
                    raise prep_exc from exc
            raise

        bound = entry["bound"]
        prepared_payload = dict(entry.get("prepared_payload") or _prepared_payload_from_bound(bound))
        prepared_payload["mechanism"] = bound.mechanism.clone()
        prepared_payload["y0"] = np.array(prepared_payload.get("y0", bound.y0), copy=True, dtype=float)

        prepared = prepare_simulation_worker_run(
            mechanism_text=mechanism_text,
            initials=initials,
            t_span=t_span,
            solver_config=solver_config,
            prepared_payload=prepared_payload,
        )

    result = solve_ode(prepared.request)

    algebra_scalars: Dict[str, float] = {}
    species_names = list(prepared.species_names)
    base_species_count = len(species_names)
    species_series = {sp: result.Y[i, :] for i, sp in enumerate(species_names)}
    initials_map = dict(prepared.initials_for_algebra or {})
    try:
        algebra_series, algebra_scalars = evaluate_algebra_series_for_simulation(
            prepared.mechanism,
            t=result.t,
            species_series=species_series,
            initials=initials_map,
        )
        if algebra_series:
            algebra_names = list(algebra_series.keys())
            algebra_matrix = np.vstack([algebra_series[name] for name in algebra_names])
            extended_y = np.vstack([result.Y, algebra_matrix])
            extended_species_names = species_names + algebra_names
        else:
            extended_y = result.Y
            extended_species_names = species_names
        algebra_errors: list[dict[str, Any]] = []
    except Exception as exc:
        logger.warning("Algebra evaluation failed in batch worker: %s", exc)
        extended_y = result.Y
        extended_species_names = species_names
        algebra_errors = [serialize_algebra_error(exc, name="__algebra__")]

    builder = (
        build_simulation_success_payload
        if include_mechanism_in_result_payload
        else build_secondary_simulation_success_payload
    )
    payload_kwargs: Dict[str, Any] = {
        "result": result,
        "y": extended_y,
        "species_names": extended_species_names,
        "base_species_count": int(base_species_count),
        "algebra_scalars": algebra_scalars,
        "algebra_errors": algebra_errors,
        "warnings": [],
        "solver": str(prepared.request.solver),
        "mechanism_text": mechanism_text,
        "solver_config": {
            **dict(solver_config),
            "solver": str(prepared.request.solver),
            "solver_warning": (
                str(prepared.solver_warning)
                if getattr(prepared, "solver_warning", None)
                else None
            ),
            "rtol": float(prepared.request.rtol),
            "atol": float(prepared.request.atol),
            "grid": dict(getattr(prepared.request, "grid", {}) or {}),
            "temperature_K": float(temperature_K),
            "wegscheider_cyclicity_enabled": bool(wegscheider_enabled),
            "use_sparse_jacobian": bool(use_sparse_jacobian),
        },
        "extra_fields": {
            "run_id": int(task.get("run_id") or 0),
            "set_id": str(task.get("set_id") or ""),
            "set_name": str(task.get("set_name") or ""),
        },
    }
    if include_mechanism_in_result_payload:
        payload_kwargs["mechanism"] = prepared.mechanism
    return builder(**payload_kwargs)


def run_batch_simulation_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return _run_batch_simulation_task_impl(task)
    except Exception as exc:
        from kindred.core.simulation_preparation import SimulationPreparationError

        if isinstance(exc, SimulationPreparationError):
            return {
                "success": False,
                "run_id": int(task.get("run_id") or 0),
                "set_id": str(task.get("set_id") or ""),
                "set_name": str(task.get("set_name") or ""),
                "error": build_simulation_failure(
                    "preparation_error",
                    str(exc),
                    details={"stage": str(exc.stage or "unknown")},
                    exc_type=exc.__class__.__name__,
                ),
            }
        return {
            "success": False,
            "run_id": int(task.get("run_id") or 0),
            "set_id": str(task.get("set_id") or ""),
            "set_name": str(task.get("set_name") or ""),
            "error": simulation_failure_from_exception(exc),
        }
