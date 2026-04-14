# SPDX-License-Identifier: MIT
"""
ODE solver wrappers for Kindred.

SciPy is a hard dependency; all integration routes through `scipy.integrate.solve_ivp` with:
- Deterministic `t_eval` output grid
- Optional TemperatureSchedule wrapper that injects T into RHS(T=...) calls
- Optional progress callback wrapper
- Optional positivity clamp (post-processing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    cast,
)

import numpy as np
from kindred.core.scipy_integrate import load_scipy_integrate

from kindred.core.temperature import TemperatureScheduleDictProtocol, TemperatureScheduleProtocol
from kindred.core.exceptions import (
    InitialConditionError,
    SimulationCancelled,
    TimeGridError,
    create_solver_error,
)
from kindred.core.time_grid import build_time_grid
from .jacobian import JacobianConfig, compute_jacobian

logger = logging.getLogger(__name__)

DEFAULT_SOLVER_NAME = "BDF"


def _solve_ivp(*, fun: Callable[[float, np.ndarray], np.ndarray], t_span: Tuple[float, float], y0: np.ndarray, **kwargs: Any):
    solve_ivp = load_scipy_integrate()
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


class ODERhsNoTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray: ...


class ODERhsWithTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


ODERhs = ODERhsNoTemp | ODERhsWithTemp

Rhs2 = Callable[[float, np.ndarray], np.ndarray]


__all__ = [
    "SimulationRequest",
    "SimulationOutput",
    "DEFAULT_SOLVER_NAME",
    "build_time_grid",
    "normalize_solver_name",
    "solve_ode",
]


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
    t_span: Tuple[float, float]
    y0: np.ndarray
    solver: str = DEFAULT_SOLVER_NAME
    rtol: float = 1e-6
    atol: float = 1e-12
    max_step: Optional[float] = None
    first_step: Optional[float] = None
    t_eval: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, float | int]] = None
    rosenbrock_jacobian: JacobianConfig = JacobianConfig()
    jacobian_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None
    event_terminal: Optional[Iterable[bool]] = None
    positivity: Optional[str] = None
    pos_indices: Optional[Iterable[int]] = None

    progress_callback: Optional[Callable[[float, float, float], None]] = None
    temperature_schedule: TemperatureScheduleProtocol | None = None


@dataclass(frozen=True)
class SimulationOutput:
    t: np.ndarray
    Y: np.ndarray
    provenance: Dict[str, object]
    fallback_occurred: bool = False
    fallback_message: Optional[str] = None


def _scipy_method_for(name: object) -> Tuple[str, Optional[str]]:
    """Map solver name to SciPy method name with correct capitalization."""
    n = str(name or "").strip().upper()
    scipy_methods = {
        "RADAU": "Radau",
        "BDF": "BDF",
    }
    if n in scipy_methods:
        return scipy_methods[n], None
    return DEFAULT_SOLVER_NAME, f"Unknown solver name; using {DEFAULT_SOLVER_NAME}"


def normalize_solver_name(name: object) -> Tuple[str, Optional[str]]:
    """
    Normalize a user-specified solver name to a SciPy `solve_ivp` method name.

    Returns (method, warning). The warning is non-empty when an unknown
    name is mapped to a supported solver.
    """
    return _scipy_method_for(name)


def _unpack_banded_jacobian(J: np.ndarray, *, ml: int, mu: int) -> np.ndarray:
    n = J.shape[1]
    Jd = np.zeros((n, n), float)
    for j in range(n):
        i_min = max(0, j - mu)
        i_max = min(n - 1, j + ml)
        band_rows = slice(mu + i_min - j, mu + i_max - j + 1)
        Jd[i_min : i_max + 1, j] = J[band_rows, j]
    return Jd


def _make_scipy_jac(rhs: Callable[[float, np.ndarray], np.ndarray], cfg: "JacobianConfig"):
    def jac(t: float, y: np.ndarray) -> np.ndarray:
        J, kind = compute_jacobian(rhs, t, y, cfg=cfg)
        J_arr = np.asarray(J, dtype=float)
        if kind.startswith("banded("):
            ml, mu = cfg.validate_for(J_arr.shape[1])
            return _unpack_banded_jacobian(J_arr, ml=ml, mu=mu)
        return J_arr

    return jac


class _TemperatureInjectedRhs:
    def __init__(self, rhs: ODERhsWithTemp, schedule: TemperatureScheduleProtocol) -> None:
        self._rhs = rhs
        self._schedule = schedule

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        T = float(self._schedule(t))
        return self._rhs(t, y, T=T)


class _ProgressRhs:
    def __init__(
        self,
        rhs: Rhs2,
        *,
        callback: Callable[[float, float, float], None],
        t0: float,
        t1: float,
        every: int = 10,
    ) -> None:
        self._rhs = rhs
        self._callback = callback
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._every = int(every)
        self._span = float(self._t1 - self._t0)
        self._last_bucket = -1

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        if self._every > 0 and np.isfinite(self._span):
            if self._span <= 0.0:
                bucket = 0
            else:
                frac = (float(t) - self._t0) / self._span
                if frac < 0.0:
                    frac = 0.0
                elif frac > 1.0:
                    frac = 1.0
                bucket = int(frac * float(self._every))
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                self._callback(float(t), self._t0, self._t1)
        return self._rhs(t, y)


def _prepare_time_grid(req: SimulationRequest, *, t0: float, t1: float) -> np.ndarray:
    if req.t_eval is None:
        return build_time_grid(t0, t1, req.grid or {"N": 2})
    t_eval = np.asarray(req.t_eval, dtype=float).reshape(-1)
    if t_eval.size == 0:
        raise TimeGridError("t_eval must contain at least one time point")
    if np.any(~np.isfinite(t_eval)):
        raise TimeGridError("t_eval must contain only finite values")
    if np.any(np.diff(t_eval) <= 0):
        raise TimeGridError("t_eval must be strictly increasing")
    if t_eval[0] < t0 or t_eval[-1] > t1:
        raise TimeGridError("t_eval must lie within t_span")
    return t_eval


def _prepare_rhs(req: SimulationRequest, *, t0: float, t1: float) -> tuple[Rhs2, Rhs2]:
    if req.temperature_schedule is None:
        rhs_for_jac: Rhs2 = cast(ODERhsNoTemp, req.rhs)
    else:
        rhs_for_jac = _TemperatureInjectedRhs(cast(ODERhsWithTemp, req.rhs), req.temperature_schedule)

    rhs: Rhs2 = rhs_for_jac
    if req.progress_callback is not None:
        rhs = _ProgressRhs(rhs_for_jac, callback=req.progress_callback, t0=t0, t1=t1)
    return rhs, rhs_for_jac


def _build_provenance(req: SimulationRequest, *, t_eval: np.ndarray) -> Dict[str, object]:
    prov: Dict[str, object] = {
        "solver_requested": req.solver,
        "rtol": float(req.rtol),
        "atol": float(req.atol),
        "max_step": (None if req.max_step is None else float(req.max_step)),
        "first_step": (None if req.first_step is None else float(req.first_step)),
        "grid_type": ("explicit" if req.t_eval is not None else ("dt" if "dt" in (req.grid or {}) else "N")),
        "grid_params": ({"t_eval": list(t_eval)} if req.t_eval is not None else dict(req.grid or {})),
        "rosenbrock_jacobian": {
            "mode": getattr(req.rosenbrock_jacobian, "mode", None),
            "ml": getattr(req.rosenbrock_jacobian, "ml", None),
            "mu": getattr(req.rosenbrock_jacobian, "mu", None),
        },
        "custom_jacobian": bool(req.jacobian_func),
        "positivity": (req.positivity or None),
        "pos_indices": (list(req.pos_indices) if req.pos_indices is not None else None),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }

    if req.temperature_schedule is not None:
        if isinstance(req.temperature_schedule, TemperatureScheduleDictProtocol):
            try:
                prov["temperature_schedule"] = req.temperature_schedule.to_dict()
            except Exception:
                prov["temperature_schedule"] = str(req.temperature_schedule)
        else:
            prov["temperature_schedule"] = str(req.temperature_schedule)

    return prov


def _implicit_scipy_alternatives(primary: str) -> List[str]:
    order = {
        "BDF": ["Radau"],
        "Radau": ["BDF"],
    }
    return [m for m in order.get(primary, []) if m != primary]


def _apply_positivity_to_trajectory(
    Y: np.ndarray,
    *,
    mode: Optional[str],
    indices: Optional[Iterable[int]],
) -> np.ndarray:
    if not mode:
        return Y
    if indices is None:
        return np.maximum(Y, 0.0)
    Y2 = np.asarray(Y, float).copy()
    for j in indices:
        jj = int(j)
        if 0 <= jj < Y2.shape[0]:
            Y2[jj, :] = np.maximum(Y2[jj, :], 0.0)
    return Y2


def _execute_scipy(
    req: SimulationRequest,
    *,
    rhs: Rhs2,
    rhs_for_jac: Rhs2,
    t0: float,
    t1: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    prov: Dict[str, object],
    method: str,
    note: Optional[str],
) -> SimulationOutput:
    if note:
        prov["emulation_note"] = note

    base_kwargs: Dict[str, Any] = dict(t_eval=t_eval, rtol=float(req.rtol), atol=float(req.atol))
    if req.max_step is not None:
        base_kwargs["max_step"] = float(req.max_step)
    if req.first_step is not None:
        base_kwargs["first_step"] = float(req.first_step)
    events_list: Optional[List[Callable[[float, np.ndarray], float]]] = None
    if req.events is not None:
        events_list = list(req.events)
        base_kwargs["events"] = events_list

    jac_callable = req.jacobian_func or _make_scipy_jac(rhs_for_jac, req.rosenbrock_jacobian)
    banded_jacobian_active = (
        getattr(req.rosenbrock_jacobian, "mode", None) == "banded"
        and req.rosenbrock_jacobian.ml is not None
        and req.rosenbrock_jacobian.mu is not None
    )

    def _kwargs_for_method(method_name: str) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["method"] = method_name
        if method_name in ("Radau", "BDF"):
            kwargs["jac"] = jac_callable
            if banded_jacobian_active:
                # SciPy implicit solvers expect an n x n Jacobian when a jac
                # callable is supplied; jac_sparsity is ignored in that case.
                kwargs.pop("jac_sparsity", None)
        else:
            kwargs.pop("jac", None)
            kwargs.pop("jac_sparsity", None)
        return kwargs

    attempted_methods: List[str] = []
    fallback_occurred = False
    fallback_message: Optional[str] = None
    primary_method = method

    def _looks_like_brentq_sign_error(exc: BaseException) -> bool:
        return "different signs" in str(exc).lower()

    def _cancel_requested_via_event() -> bool:
        for ev in events_list or []:
            if not bool(getattr(ev, "_kindred_cancel_event", False)):
                continue
            cancelled_cb = getattr(ev, "_kindred_cancelled", None)
            if callable(cancelled_cb) and bool(cancelled_cb()):
                return True
        return False

    def _solve_or_cancelled(method_name: str):
        try:
            return _solve_ivp(fun=rhs, t_span=(t0, t1), y0=y0, **_kwargs_for_method(method_name))  # type: ignore[misc]
        except ValueError as exc:
            # SciPy event root-finder can raise on cancellation races (brentq sign error).
            # If we have an explicit cancellation event and it reports cancellation requested,
            # translate to a typed cancellation signal.
            if events_list and _looks_like_brentq_sign_error(exc) and _cancel_requested_via_event():
                raise SimulationCancelled() from exc
            raise

    sol = _solve_or_cancelled(method)
    attempted_methods.append(method)

    if events_list and getattr(sol, "status", 0) == 1 and getattr(sol, "t_events", None):
        for idx, ev in enumerate(events_list):
            if not bool(getattr(ev, "_kindred_cancel_event", False)):
                continue
            t_events = sol.t_events[idx] if idx < len(sol.t_events) else None
            if t_events is not None and len(t_events) > 0:
                raise SimulationCancelled()

    if not sol.success:
        prov["solver_failure"] = str(sol.message)
        last_message = str(sol.message)
        for alt_method in _implicit_scipy_alternatives(method):
            alt_sol = _solve_or_cancelled(alt_method)
            attempted_methods.append(alt_method)
            if alt_sol.success:
                sol = alt_sol
                prov["solver_alternative_used"] = alt_method
                fallback_occurred = True
                fallback_message = f"{primary_method} failed; succeeded with {alt_method}"
                method = alt_method
                break
            last_message = str(alt_sol.message)
        if not sol.success:
            t_fail = float(sol.t[-1]) if getattr(sol, "t", None) is not None and len(sol.t) > 0 else float(t0)
            attempt_list = ", ".join(attempted_methods)
            raise create_solver_error(
                req.solver,
                t_fail,
                f"{last_message}; attempted methods: {attempt_list}",
            )

    prov["solver_used"] = method
    t_out = np.asarray(sol.t, float)
    Y_out = _apply_positivity_to_trajectory(
        np.asarray(sol.y, float),
        mode=req.positivity,
        indices=(list(req.pos_indices) if req.pos_indices is not None else None),
    )
    if hasattr(sol, "t_events") and sol.t_events:
        prov["events"] = [list(te) for te in sol.t_events]
    return SimulationOutput(
        t=t_out,
        Y=Y_out,
        provenance=prov,
        fallback_occurred=fallback_occurred,
        fallback_message=fallback_message,
    )

def _request_from_mapping(payload: Mapping[str, Any], *, allow_unknown_keys: bool) -> SimulationRequest:
    field_names = {f.name for f in fields(SimulationRequest)}
    payload_dict = dict(payload)
    missing = [name for name in ("rhs", "t_span", "y0") if name not in payload_dict]
    if missing:
        raise TypeError(f"Missing required SimulationRequest field(s): {', '.join(missing)}")
    unknown = sorted(set(payload_dict) - field_names)
    if unknown and not allow_unknown_keys:
        raise TypeError(f"Unknown SimulationRequest field(s): {', '.join(unknown)}")
    filtered = {k: v for k, v in payload_dict.items() if k in field_names}
    return SimulationRequest(**filtered)  # type: ignore[arg-type]


def solve_ode(req: SimulationRequest | Mapping[str, Any], *, allow_unknown_keys: bool = False) -> SimulationOutput:
    if not isinstance(req, SimulationRequest):
        if isinstance(req, Mapping):
            req = _request_from_mapping(req, allow_unknown_keys=bool(allow_unknown_keys))
        else:
            raise TypeError(f"solve_ode expects SimulationRequest or Mapping, got {type(req)!r}")
    y0 = np.asarray(req.y0, float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise InitialConditionError("y0 must be a 1D finite array")
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    rhs, rhs_for_jac = _prepare_rhs(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method, note = _scipy_method_for(req.solver)
    return _execute_scipy(
        req,
        rhs=rhs,
        rhs_for_jac=rhs_for_jac,
        t0=t0,
        t1=t1,
        y0=y0,
        t_eval=t_eval,
        prov=prov,
        method=method,
        note=note,
    )
