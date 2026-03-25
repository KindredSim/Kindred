"""
Parametric alignment helpers for global fitting.

When a dataset chooses X != t, the fit residuals are computed as:
    y_model(x_obs) - y_obs
where y_model(x_obs) is obtained by interpolating along the simulated parametric
curve (x_model(t), y_model(t)) over the dataset's sampled time window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kindred.core.exceptions import FitSimulationError

__all__ = [
    "align_y_on_x_obs",
    "align_y_on_x_obs_time_guided",
    "align_y_on_x_obs_time_guided_penalized",
    "ParametricXAlignment",
    "is_non_monotone_in_sampled_window_error",
]


_NON_MONOTONE_SUBSTRING = "is not strictly monotone in sampled window"


def is_non_monotone_in_sampled_window_error(exc: BaseException) -> bool:
    if not isinstance(exc, FitSimulationError):
        return False
    return _NON_MONOTONE_SUBSTRING in str(exc)


def _as_1d_float(name: str, values: object) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive
        raise FitSimulationError(f"Invalid {name} array for parametric alignment: {exc}") from exc
    return arr


def align_y_on_x_obs(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: Optional[np.ndarray],
    x_model: np.ndarray,
    y_model: np.ndarray,
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> np.ndarray:
    """
    Align y_model onto the observed x grid using parametric interpolation.

    Rules (Phase 1b):
    - Select a simulation segment using the sampled observed time window [min(t_obs), max(t_obs)].
    - Require x_model(t) to be strictly monotone on that segment (increasing or decreasing).
    - Reject if any x_obs lies outside the model x range on that segment (no clamping).
    """
    ds = str(dataset_label or "dataset").strip() or "dataset"
    x_name = str(x_name or "").strip() or "X"
    y_name = str(y_name or "").strip() or "Y"

    t_obs_arr = _as_1d_float("t_obs", t_obs)
    x_obs_arr = _as_1d_float("x_obs", x_obs)
    if t_obs_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': empty t array for parametric alignment.")
    if x_obs_arr.size != t_obs_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': x_obs length {x_obs_arr.size} does not match t length {t_obs_arr.size}.",
        )
    if not np.all(np.isfinite(x_obs_arr)):
        raise FitSimulationError(f"Dataset '{ds}': x_obs contains non-finite values for X='{x_name}'.")

    if t_sim is None:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation did not provide a time axis required for X='{x_name}' parametric alignment.",
        )
    t_sim_arr = _as_1d_float("t_sim", t_sim)
    x_model_arr = _as_1d_float("x_model", x_model)
    y_model_arr = _as_1d_float("y_model", y_model)
    if t_sim_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': simulation time axis is empty.")
    if x_model_arr.size != t_sim_arr.size or y_model_arr.size != t_sim_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation series length mismatch for X='{x_name}', Y='{y_name}'.",
        )

    try:
        t0 = float(np.min(t_obs_arr))
        t1 = float(np.max(t_obs_arr))
    except Exception as exc:  # pragma: no cover - defensive
        raise FitSimulationError(f"Dataset '{ds}': failed to determine sampled time window: {exc}") from exc
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise FitSimulationError(f"Dataset '{ds}': non-finite sampled time window for parametric alignment.")
    if t0 > t1:
        t0, t1 = t1, t0

    t_scale = max(1.0, abs(t0), abs(t1))
    t_pad = 1e-12 * t_scale
    mask = (t_sim_arr >= (t0 - t_pad)) & (t_sim_arr <= (t1 + t_pad))
    if not np.any(mask):
        raise FitSimulationError(
            f"Dataset '{ds}': no simulation points fall within sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )

    x_seg = x_model_arr[mask]
    y_seg = y_model_arr[mask]
    if x_seg.size < 2 or y_seg.size < 2:
        raise FitSimulationError(
            f"Dataset '{ds}': insufficient simulation points in sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )
    if not (np.all(np.isfinite(x_seg)) and np.all(np.isfinite(y_seg))):
        raise FitSimulationError(
            f"Dataset '{ds}': simulation produced non-finite values for X='{x_name}' or Y='{y_name}' in the sampled window.",
        )

    # Strict monotonicity check on the window segment only.
    x_scale = max(1.0, float(np.max(np.abs(x_seg))))
    x_tol = 1e-12 * x_scale
    diffs = np.diff(x_seg)
    inc = bool(np.all(diffs > x_tol))
    dec = bool(np.all(diffs < -x_tol))
    if not (inc or dec):
        raise FitSimulationError(
            f"Dataset '{ds}': X(t) for '{x_name}' is not strictly monotone in sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}]. Narrow or shift t_min/t_max.",
        )
    if dec:
        x_seg = x_seg[::-1]
        y_seg = y_seg[::-1]

    x_min = float(x_seg[0])
    x_max = float(x_seg[-1])
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    too_low = x_obs_arr < (x_min - x_tol)
    too_high = x_obs_arr > (x_max + x_tol)
    if bool(np.any(too_low) or np.any(too_high)):
        raise FitSimulationError(
            f"Dataset '{ds}': observed X values for '{x_name}' fall outside model range "
            f"[{x_min:.6g}, {x_max:.6g}] within sampled window [t_min={t0:.6g}, t_max={t1:.6g}]. "
            "Adjust t_min/t_max (no clamping is applied).",
        )

    aligned = np.interp(x_obs_arr, x_seg, y_seg)
    return np.asarray(aligned, dtype=float).reshape(-1)


def align_y_on_x_obs_time_guided(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: Optional[np.ndarray],
    x_model: np.ndarray,
    y_model: np.ndarray,
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> np.ndarray:
    """
    Align y_model onto the observed x grid using time-guided branch selection.

    Rules (Phase 1c):
    - Consider only the dataset's sampled time window [min(t_obs), max(t_obs)].
    - Allow non-monotone x_model(t) within that window.
    - For each observed point i, find all times t* in the sampled window such that x_model(t*) == x_obs[i]
      (via piecewise-linear crossings) and choose the solution with minimal |t* - t_obs[i]|.
    - Reject if any x_obs has no solution within the sampled window.
    """
    ds = str(dataset_label or "dataset").strip() or "dataset"
    x_name = str(x_name or "").strip() or "X"
    y_name = str(y_name or "").strip() or "Y"

    t_obs_arr = _as_1d_float("t_obs", t_obs)
    x_obs_arr = _as_1d_float("x_obs", x_obs)
    if t_obs_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': empty t array for parametric alignment.")
    if x_obs_arr.size != t_obs_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': x_obs length {x_obs_arr.size} does not match t length {t_obs_arr.size}.",
        )
    if not np.all(np.isfinite(x_obs_arr)):
        raise FitSimulationError(f"Dataset '{ds}': x_obs contains non-finite values for X='{x_name}'.")
    if not np.all(np.isfinite(t_obs_arr)):
        raise FitSimulationError(f"Dataset '{ds}': t contains non-finite values for X='{x_name}'.")

    if t_sim is None:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation did not provide a time axis required for X='{x_name}' parametric alignment.",
        )
    t_sim_arr = _as_1d_float("t_sim", t_sim)
    x_model_arr = _as_1d_float("x_model", x_model)
    y_model_arr = _as_1d_float("y_model", y_model)
    if t_sim_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': simulation time axis is empty.")
    if x_model_arr.size != t_sim_arr.size or y_model_arr.size != t_sim_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation series length mismatch for X='{x_name}', Y='{y_name}'.",
        )

    try:
        t0 = float(np.min(t_obs_arr))
        t1 = float(np.max(t_obs_arr))
    except Exception as exc:  # pragma: no cover - defensive
        raise FitSimulationError(f"Dataset '{ds}': failed to determine sampled time window: {exc}") from exc
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise FitSimulationError(f"Dataset '{ds}': non-finite sampled time window for parametric alignment.")
    if t0 > t1:
        t0, t1 = t1, t0

    t_scale = max(1.0, abs(t0), abs(t1))
    t_pad = 1e-12 * t_scale
    mask = (t_sim_arr >= (t0 - t_pad)) & (t_sim_arr <= (t1 + t_pad))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise FitSimulationError(
            f"Dataset '{ds}': no simulation points fall within sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )

    # Include one neighbor point on each side (if available) so boundary crossings can be detected
    # while still restricting solution selection to [t0, t1].
    start = max(0, int(idx[0]) - 1)
    stop = min(int(t_sim_arr.size), int(idx[-1]) + 2)
    t_seg = t_sim_arr[start:stop]
    x_seg = x_model_arr[start:stop]
    y_seg = y_model_arr[start:stop]
    if t_seg.size < 2:
        raise FitSimulationError(
            f"Dataset '{ds}': insufficient simulation points in sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )
    if not (np.all(np.isfinite(x_seg)) and np.all(np.isfinite(y_seg)) and np.all(np.isfinite(t_seg))):
        raise FitSimulationError(
            f"Dataset '{ds}': simulation produced non-finite values for X='{x_name}' or Y='{y_name}' in the sampled window.",
        )

    x_scale = max(1.0, float(np.max(np.abs(x_seg))), float(np.max(np.abs(x_obs_arr))))
    x_tol = 1e-12 * x_scale

    aligned = np.empty_like(x_obs_arr, dtype=float)

    # For each observed point, search all piecewise-linear crossings within the sampled window.
    for i, (t_obs_i, x_obs_i) in enumerate(zip(t_obs_arr, x_obs_arr)):
        best_t: Optional[float] = None
        best_dist = float("inf")

        for j in range(int(t_seg.size) - 1):
            x0 = float(x_seg[j])
            x1 = float(x_seg[j + 1])
            if x_obs_i < (min(x0, x1) - x_tol) or x_obs_i > (max(x0, x1) + x_tol):
                continue

            dx = x1 - x0
            if abs(dx) <= x_tol:
                if abs(x_obs_i - x0) <= x_tol:
                    lo = float(min(t_seg[j], t_seg[j + 1]))
                    hi = float(max(t_seg[j], t_seg[j + 1]))
                    t_star = float(min(hi, max(lo, float(t_obs_i))))
                else:
                    continue
            else:
                frac = (float(x_obs_i) - x0) / dx
                frac = float(min(1.0, max(0.0, frac)))
                t_star = float(t_seg[j] + frac * (t_seg[j + 1] - t_seg[j]))

            if t_star < (t0 - t_pad) or t_star > (t1 + t_pad):
                continue

            dist = abs(t_star - float(t_obs_i))
            if dist < best_dist - 1e-15:
                best_dist = dist
                best_t = t_star
            elif abs(dist - best_dist) <= 1e-15:
                if best_t is None or t_star < best_t:
                    best_t = t_star

        if best_t is None:
            raise FitSimulationError(
                f"Dataset '{ds}': No solution for X='{x_name}' at observed value {float(x_obs_i):.6g} "
                f"within sampled window [t_min={t0:.6g}, t_max={t1:.6g}]; adjust t_min/t_max or choose different X.",
            )

        aligned[i] = float(np.interp(float(best_t), t_seg, y_seg))

    return np.asarray(aligned, dtype=float).reshape(-1)


@dataclass(frozen=True)
class ParametricXAlignment:
    """
    Parametric-X alignment result for X != t.

    Attributes
    ----------
    y_aligned
        y_model evaluated at the chosen t* per point.
    t_star
        Chosen time t* per observed point (restricted to the sampled time window).
    x_star
        x_model(t*) per observed point.
    dx
        x_model(t*) - x_obs per observed point. Zero (within tolerance) indicates an exact crossing.
    exact
        Boolean mask indicating points treated as exact crossings (|dx| <= tol).
    """

    y_aligned: np.ndarray
    t_star: np.ndarray
    x_star: np.ndarray
    dx: np.ndarray
    exact: np.ndarray


def align_y_on_x_obs_time_guided_penalized(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: Optional[np.ndarray],
    x_model: np.ndarray,
    y_model: np.ndarray,
    dataset_label: str,
    x_name: str,
    y_name: str,
    beta: Optional[float] = None,
) -> ParametricXAlignment:
    """
    Time-guided parametric alignment with a penalized fallback (never hard-fails for no crossing/out-of-range).

    This chooses t* in the sampled time window to minimize, per observed point i:
        (x_model(t*) - x_obs[i])^2 + (beta * (t* - t_obs[i]))^2
    where x_model(t) is treated as piecewise-linear over the simulation grid.

    Notes
    -----
    - Crossings (when present) naturally yield dx ~= 0 and are preferred; the time term breaks ties by proximity to t_obs.
    - When no crossing exists within the window (including x_obs outside range), this returns the best-available t* and a nonzero dx.
    - Exceptions are reserved for invalid inputs (shape mismatches, empty arrays, non-finite values, or no simulation points in window).
    """
    ds = str(dataset_label or "dataset").strip() or "dataset"
    x_name = str(x_name or "").strip() or "X"
    y_name = str(y_name or "").strip() or "Y"

    t_obs_arr = _as_1d_float("t_obs", t_obs)
    x_obs_arr = _as_1d_float("x_obs", x_obs)
    if t_obs_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': empty t array for parametric alignment.")
    if x_obs_arr.size != t_obs_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': x_obs length {x_obs_arr.size} does not match t length {t_obs_arr.size}.",
        )
    if not (np.all(np.isfinite(x_obs_arr)) and np.all(np.isfinite(t_obs_arr))):
        raise FitSimulationError(f"Dataset '{ds}': non-finite t/x values for X='{x_name}'.")

    if t_sim is None:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation did not provide a time axis required for X='{x_name}' parametric alignment.",
        )
    t_sim_arr = _as_1d_float("t_sim", t_sim)
    x_model_arr = _as_1d_float("x_model", x_model)
    y_model_arr = _as_1d_float("y_model", y_model)
    if t_sim_arr.size == 0:
        raise FitSimulationError(f"Dataset '{ds}': simulation time axis is empty.")
    if x_model_arr.size != t_sim_arr.size or y_model_arr.size != t_sim_arr.size:
        raise FitSimulationError(
            f"Dataset '{ds}': simulation series length mismatch for X='{x_name}', Y='{y_name}'.",
        )

    try:
        t0 = float(np.min(t_obs_arr))
        t1 = float(np.max(t_obs_arr))
    except Exception as exc:  # pragma: no cover - defensive
        raise FitSimulationError(f"Dataset '{ds}': failed to determine sampled time window: {exc}") from exc
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise FitSimulationError(f"Dataset '{ds}': non-finite sampled time window for parametric alignment.")
    if t0 > t1:
        t0, t1 = t1, t0

    # Work only within the sampled window, with one neighbor point on each side to preserve segment coverage.
    t_scale = max(1.0, abs(t0), abs(t1))
    t_pad = 1e-12 * t_scale
    mask = (t_sim_arr >= (t0 - t_pad)) & (t_sim_arr <= (t1 + t_pad))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise FitSimulationError(
            f"Dataset '{ds}': no simulation points fall within sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )
    start = max(0, int(idx[0]) - 1)
    stop = min(int(t_sim_arr.size), int(idx[-1]) + 2)

    t_seg = t_sim_arr[start:stop]
    x_seg = x_model_arr[start:stop]
    y_seg = y_model_arr[start:stop]
    if t_seg.size < 2:
        raise FitSimulationError(
            f"Dataset '{ds}': insufficient simulation points in sampled window "
            f"[t_min={t0:.6g}, t_max={t1:.6g}] for X='{x_name}'. Adjust t_min/t_max.",
        )
    if not (np.all(np.isfinite(t_seg)) and np.all(np.isfinite(x_seg)) and np.all(np.isfinite(y_seg))):
        raise FitSimulationError(
            f"Dataset '{ds}': simulation produced non-finite values for X='{x_name}' or Y='{y_name}' in the sampled window.",
        )

    # Ensure increasing time for segment computations and interpolation.
    if not np.all(np.diff(t_seg) > 0):
        order = np.argsort(t_seg, kind="mergesort")
        t_seg = t_seg[order]
        x_seg = x_seg[order]
        y_seg = y_seg[order]

    dt = np.diff(t_seg)
    valid = dt > 0
    if not np.any(valid):
        raise FitSimulationError(
            f"Dataset '{ds}': simulation time axis is not strictly increasing in the sampled window for X='{x_name}'.",
        )

    t0s = t_seg[:-1][valid]
    dts = dt[valid]
    x0s = x_seg[:-1][valid]
    dxs = (x_seg[1:][valid] - x0s)
    m = dxs / dts  # x units per time

    # Choose beta in x/time units (so beta*(t-t_obs) has x units).
    beta_val: float
    if beta is None:
        slopes = np.abs(m[np.isfinite(m)])
        beta_val = float(np.median(slopes)) if slopes.size else 0.0
        if not np.isfinite(beta_val):
            beta_val = 0.0
    else:
        beta_val = float(beta)
        if not np.isfinite(beta_val) or beta_val < 0.0:
            raise FitSimulationError(f"Dataset '{ds}': invalid beta for penalized alignment: {beta!r}.")

    beta2 = beta_val * beta_val
    denom = (m * m) + beta2

    t_star = np.empty_like(t_obs_arr, dtype=float)
    x_star = np.empty_like(x_obs_arr, dtype=float)

    for i, (t_obs_i, x_obs_i) in enumerate(zip(t_obs_arr, x_obs_arr)):
        # Phase 1: prefer exact crossings (time-guided branch selection).
        x_scale_cross = max(1.0, float(np.max(np.abs(x_seg))), float(np.max(np.abs(x_obs_arr))))
        x_tol_cross = 1e-12 * x_scale_cross

        best_t: Optional[float] = None
        best_dist = float("inf")
        for j in range(int(t_seg.size) - 1):
            x0 = float(x_seg[j])
            x1 = float(x_seg[j + 1])
            if float(x_obs_i) < (min(x0, x1) - x_tol_cross) or float(x_obs_i) > (max(x0, x1) + x_tol_cross):
                continue

            dx0 = x1 - x0
            if abs(dx0) <= x_tol_cross:
                if abs(float(x_obs_i) - x0) <= x_tol_cross:
                    lo = float(min(t_seg[j], t_seg[j + 1]))
                    hi = float(max(t_seg[j], t_seg[j + 1]))
                    t_c = float(min(hi, max(lo, float(t_obs_i))))
                else:
                    continue
            else:
                frac = (float(x_obs_i) - x0) / dx0
                frac = float(min(1.0, max(0.0, frac)))
                t_c = float(t_seg[j] + frac * (t_seg[j + 1] - t_seg[j]))

            if t_c < (t0 - t_pad) or t_c > (t1 + t_pad):
                continue
            dist = abs(t_c - float(t_obs_i))
            if dist < best_dist - 1e-15:
                best_dist = dist
                best_t = t_c
            elif abs(dist - best_dist) <= 1e-15:
                if best_t is None or t_c < best_t:
                    best_t = t_c

        if best_t is not None:
            t_star[i] = min(t1, max(t0, float(best_t)))
            x_star[i] = float(np.interp(t_star[i], t_seg, x_seg))
            continue

        # Phase 2: penalized fallback (no crossing/out-of-range): choose best available t*.
        numer = (m * (x0s - float(x_obs_i))) + (beta2 * (t0s - float(t_obs_i)))
        with np.errstate(divide="ignore", invalid="ignore"):
            d_star = -numer / denom
        flat = ~np.isfinite(d_star) | (denom <= 0.0)
        if np.any(flat):
            d_star = np.asarray(d_star, dtype=float)
            d_star[flat] = float(t_obs_i) - t0s[flat]
        d_star = np.clip(d_star, 0.0, dts)

        frac = d_star / dts
        t_cand = t0s + d_star
        x_cand = x0s + frac * dxs

        inside = (t_cand >= (t0 - t_pad)) & (t_cand <= (t1 + t_pad))
        dx_err = x_cand - float(x_obs_i)
        dt_err = t_cand - float(t_obs_i)
        costs = (dx_err * dx_err) + (beta2 * (dt_err * dt_err))
        costs = np.where(inside, costs, float("inf"))

        j_best = int(np.argmin(costs))
        t_best = float(t_cand[j_best])

        best_cost = float(costs[j_best])
        ties = np.flatnonzero(np.isfinite(costs) & (np.abs(costs - best_cost) <= 1e-15))
        if ties.size > 1:
            dt_abs = np.abs(t_cand[ties] - float(t_obs_i))
            k = int(ties[int(np.argmin(dt_abs))])
            t_best = float(t_cand[k])
            best_dt = float(np.min(dt_abs))
            ties2 = ties[np.abs(dt_abs - best_dt) <= 1e-15]
            if ties2.size > 1:
                t_best = float(np.min(t_cand[ties2]))

        t_star[i] = min(t1, max(t0, t_best))
        x_star[i] = float(np.interp(t_star[i], t_seg, x_seg))

    y_aligned = np.asarray(np.interp(t_star, t_seg, y_seg), dtype=float).reshape(-1)
    dx = np.asarray(x_star - x_obs_arr, dtype=float).reshape(-1)

    x_scale = max(1.0, float(np.max(np.abs(x_seg))), float(np.max(np.abs(x_obs_arr))))
    x_tol = 1e-10 * x_scale
    exact = np.asarray(np.abs(dx) <= x_tol, dtype=bool).reshape(-1)

    return ParametricXAlignment(
        y_aligned=y_aligned,
        t_star=t_star.reshape(-1),
        x_star=x_star.reshape(-1),
        dx=dx,
        exact=exact,
    )
