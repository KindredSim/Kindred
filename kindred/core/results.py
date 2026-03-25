"""
Time-series container with CTC integration and provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple, Any
import numpy as np

from kindred.core.time_grid import is_uniform_time_grid

__all__ = ["integrate_ctc", "SimulationResult"]

# ---------------- CTC helpers ----------------


def _trapezoid_segment(h: float, y0: float, y1: float) -> float:
    return 0.5 * h * (y0 + y1)


def _composite_trapezoid(t: np.ndarray, y: np.ndarray) -> float:
    dt = np.diff(t)
    return float(np.sum(dt * 0.5 * (y[:-1] + y[1:])))


def _simpson13_uniform(t: np.ndarray, y: np.ndarray) -> float:
    """Composite Simpson 1/3 on uniform grid with even number of intervals."""
    n = t.size - 1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 requires an even number of intervals")
    h = (t[-1] - t[0]) / n
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return float(s * h / 3.0)


def _normalize_tail_strategy(val: str | None) -> str:
    s = str(val or "").strip().lower()
    if not s:
        return "38"
    if s in {"38", "3/8", "three-eighths", "three_eighths"} or s.startswith("38"):
        return "38"
    return "trapezoid"


def integrate_ctc(
    t: np.ndarray,
    y: np.ndarray,
    *,
    eps_uniform: float | None = None,
    uniformity_eps: float | None = None,
    tail_strategy: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> Tuple[float, str, bool, float, str]:
    """
    Integrate y(t) on the native grid and return
        (value, method_label, is_uniform, used_uniformity_eps, used_tail_strategy).

    Policy knobs:
      - eps_uniform or uniformity_eps: relative tolerance ε_uniform for uniform detection.
      - tail_strategy: "trapezoid" or "38" for odd-interval uniform grids.
      - policy: optional dict with keys {"uniformity_eps", "tail_strategy"}.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("t and y must be 1D arrays of equal length")
    n = t.size

    # Merge policy
    if policy:
        uniformity_eps = policy.get("uniformity_eps", uniformity_eps)
        tail_strategy = policy.get("tail_strategy", tail_strategy)
    if eps_uniform is not None:
        uniformity_eps = eps_uniform
    if uniformity_eps is None:
        uniformity_eps = 1e-6  # Default tolerance for the uniform-grid check.
    used_eps = float(uniformity_eps)

    used_tail = _normalize_tail_strategy(tail_strategy)

    if n < 2:
        # Degenerate grid: zero integral; report as trapezoidal path for label conformity.
        return 0.0, "Trapezoidal", True, used_eps, used_tail

    # Basic monotonicity and uniformity check (relative metric)
    monotonic = bool(np.all(np.diff(t) > 0))
    uniform = monotonic and is_uniform_time_grid(t, used_eps)

    if not uniform:
        return _composite_trapezoid(t, y), "Trapezoidal", False, used_eps, used_tail

    # Uniform path
    intervals = n - 1
    if intervals % 2 == 0:
        # Even number of intervals -> composite Simpson 1/3
        val = _simpson13_uniform(t, y)
        return val, "Simpson13", True, used_eps, used_tail

    # Odd number of intervals on a uniform grid
    h = float(np.median(np.diff(t)))  # stable, deterministic spacing

    if used_tail == "38":
        if n >= 4:
            # Simpson 1/3 on the first (n-4) intervals (first n-3 points)
            head_pts = n - 3
            head_val = 0.0
            if head_pts >= 3:
                # (head_pts - 1) = n - 4 intervals, guaranteed even
                head_val = _simpson13_uniform(t[:head_pts], y[:head_pts])
            # Simpson 3/8 on the last three intervals (last four points)
            tail_val = 3.0 * h * (y[-4] + 3.0 * y[-3] + 3.0 * y[-2] + y[-1]) / 8.0
            return float(head_val + tail_val), "Simpson13+38", True, used_eps, used_tail
        # Tiny uniform grids: fallback to trapezoid tail
        tail = _trapezoid_segment(t[-1] - t[-2], y[-2], y[-1])
        # For consistency, pair with Simpson 1/3 on the head if possible
        head_val = 0.0
        if n - 1 >= 2:
            # first n-1 points => (n-2) intervals which is even since (n-1) is odd
            head_val = _simpson13_uniform(t[:-1], y[:-1])
        return float(head_val + tail), "Simpson13+TrapezoidTail", True, used_eps, used_tail

    # tail strategy: trapezoid
    head_val = 0.0
    if n - 1 >= 2:
        # Simpson 1/3 on first n-2 intervals (first n-1 points)
        head_val = _simpson13_uniform(t[:-1], y[:-1])
    tail = _trapezoid_segment(t[-1] - t[-2], y[-2], y[-1])
    return float(head_val + tail), "Simpson13+TrapezoidTail", True, used_eps, used_tail


# ---------------- Results container ----------------

@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    series: Dict[str, np.ndarray]
    scalars: Dict[str, float] = field(default_factory=dict)
    ctc: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        t: np.ndarray,
        series: Mapping[str, np.ndarray],
        scalars: Mapping[str, float] | None = None,
        provenance: Mapping[str, Any] | None = None,
        ctc_policy: Mapping[str, Any] | None = None,
    ) -> "SimulationResult":
        t = np.asarray(t, dtype=float).copy()
        ser = {str(k): np.asarray(v, dtype=float).copy() for k, v in series.items()}
        scal_map = dict(scalars or {})
        prov = dict(provenance or {})

        # Compute CTC per series
        ctc_map: Dict[str, float] = {}
        # Keep one metadata block for uniform detection/method
        method_used: str | None = None
        uniform_used: bool | None = None
        eps_used: float | None = None
        tail_used: str | None = None

        for k, arr in ser.items():
            v, method, uniform, used_eps, used_tail = integrate_ctc(
                t, arr, policy=ctc_policy or {}
            )
            ctc_map[k] = float(v)
            # Record method/flags from the last series (consistent, deterministic)
            method_used = method
            uniform_used = uniform
            eps_used = float(used_eps)
            tail_used = used_tail

        # Stash CTC metadata in provenance for downstream inspection.
        prov.setdefault("ctc", {})
        prov["ctc"].update({
            "integration_method": method_used or "",
            "uniform_grid_detected": bool(uniform_used) if uniform_used is not None else False,
            "uniformity_eps": float(eps_used) if eps_used is not None else 1e-6,
            "tail_strategy": str(tail_used or "38"),
        })

        return cls(t=t, series=ser, scalars=scal_map, ctc=ctc_map, provenance=prov)
