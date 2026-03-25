"""
Jacobian utilities for simulator solvers.

Current contract
----------------
- Solver contract exposes SciPy-compatible Jacobian options for `solve_ivp`
  integrations using `Radau`, `BDF`, and related banded Jacobian handling:
    {"mode": "auto" | "analytic" | "banded", "ml": int | None, "mu": int | None}
- This module provides:
    * Finite-difference "auto" Jacobian (dense)
    * "analytic" pass-through (dense or banded conversion)
    * "banded" storage conversion using (ml, mu) in LAPACK/ SciPy convention

Conventions
-----------
- Dense Jacobian shape: (n, n) with J[i, j] = ∂f_i/∂y_j
- Banded storage (ab): shape (ml + mu + 1, n) with:
      ab[mu + i - j, j] = J[i, j]      for max(0, j-mu) ≤ i ≤ min(n-1, j+ml)
  This matches SciPy `solve_banded` and classic LAPACK `gbtrf/gbtrs` layout.
- No sparsity detection; "banded" mode is a representation choice. If the
  provided analytic Jacobian is dense, it is converted to banded by truncation.

Numerics
--------
- Finite differences use forward differences with perturbation:
      h_j = sqrt(eps) * max(|y_j|, 1) + atol + rtol * |y_j|
  and guard against h_j == 0 by falling back to sqrt(eps).
- All calculations are in float64; inputs are coerced to contiguous arrays.

No I/O. Deterministic. Boring on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


__all__ = [
    "JacobianConfig",
    "compute_jacobian",
    "finite_difference_jacobian",
    "to_banded",
]


# ------------------------------ config model ---------------------------------


@dataclass(frozen=True)
class JacobianConfig:
    """
    Configuration for Jacobian construction.

    Fields
    ------
    mode : "auto" | "analytic" | "banded"
        - "auto": finite-difference dense Jacobian
        - "analytic": use provided jac_func to get dense Jacobian
        - "banded": produce banded storage using (ml, mu). Source is either
                    analytic (preferred) or auto FD then truncated.
    ml, mu : int | None
        Lower/upper bandwidth when mode == "banded". Both must be provided.
    rtol, atol : float
        Tolerances used to scale finite-difference steps in "auto".
    """
    mode: str = "auto"
    ml: Optional[int] = None
    mu: Optional[int] = None
    rtol: float = 1e-6
    atol: float = 1e-12

    def validate_for(self, n: int) -> Tuple[int, int]:
        mode = self.mode
        if mode not in ("auto", "analytic", "banded"):
            raise ValueError("JacobianConfig.mode must be 'auto', 'analytic', or 'banded'")
        if mode == "banded":
            if self.ml is None or self.mu is None:
                raise ValueError("banded mode requires both ml and mu to be set")
            ml = int(self.ml)
            mu = int(self.mu)
            if ml < 0 or mu < 0:
                raise ValueError("ml and mu must be non-negative")
            if ml >= n or mu >= n:
                # Allow but clamp to n-1 for sanity to avoid empty bands
                ml = min(ml, n - 1)
                mu = min(mu, n - 1)
            return ml, mu
        return 0, 0


# ------------------------------ core builders --------------------------------


def compute_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    *,
    cfg: JacobianConfig,
    jac_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Build a Jacobian per configuration.

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> dy/dt, shape (n,).
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    cfg : JacobianConfig
        Mode and options.
    jac_func : callable | None
        Analytic Jacobian provider J(t, y) -> array. If mode="analytic" and this
        is None, a ValueError is raised. If mode="banded" and provided, we
        convert its dense output to banded. If not provided in "banded", we
        use auto FD then convert.

    Returns
    -------
    (J, kind)
        If kind == "dense", J has shape (n, n).
        If kind == "banded(ml,mu)", J has shape (ml+mu+1, n) in banded layout.

    Notes
    -----
    - This function does not attempt to detect sparsity. If you ask for
      "banded" we simply pack the band from a dense J (analytic or FD).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    n = y.size
    ml, mu = cfg.validate_for(n)

    if cfg.mode == "analytic":
        if jac_func is None:
            raise ValueError("analytic mode requires jac_func")
        J = np.asarray(jac_func(t, y), dtype=float)
        if J.shape != (n, n):
            raise ValueError(f"analytic jacobian must have shape {(n, n)}, got {J.shape}")
        return J, "dense"

    if cfg.mode == "banded":
        # Prefer analytic if provided, else auto FD
        if jac_func is not None:
            Jd = np.asarray(jac_func(t, y), dtype=float)
            if Jd.shape != (n, n):
                raise ValueError(f"analytic jacobian must have shape {(n, n)}, got {Jd.shape}")
        else:
            Jd = finite_difference_jacobian(f, t, y, rtol=cfg.rtol, atol=cfg.atol)
        Jb = to_banded(Jd, ml=ml, mu=mu)
        return Jb, f"banded({ml},{mu})"

    # auto → dense FD
    J = finite_difference_jacobian(f, t, y, rtol=cfg.rtol, atol=cfg.atol)
    return J, "dense"


def finite_difference_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Forward-difference dense Jacobian.

        J[:, j] = ( f(t, y + h_j e_j) - f(t, y) ) / h_j

    Step size per component j:
        h_j = sqrt(eps) * max(|y_j|, 1) + atol + rtol * |y_j|

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> dy/dt, shape (n,).
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    rtol, atol : float
        Tolerances used to scale perturbations.

    Returns
    -------
    J : ndarray, shape (n, n)
        Dense Jacobian.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    n = y.size

    f0 = np.asarray(f(t, y), dtype=float)
    if f0.shape != (n,):
        raise ValueError(f"f(t,y) must return shape {(n,)}, got {f0.shape}")

    J = np.empty((n, n), dtype=float)
    eps = np.finfo(float).eps
    sqrt_eps = np.sqrt(eps)

    for j in range(n):
        yj = y[j]
        h = sqrt_eps * max(abs(yj), 1.0) + float(atol) + float(rtol) * abs(yj)
        if h == 0.0 or not np.isfinite(h):
            h = sqrt_eps
        y_pert = y.copy()
        y_pert[j] = yj + h
        fj = np.asarray(f(t, y_pert), dtype=float)
        if fj.shape != (n,):
            raise ValueError(f"f(t,y) must return shape {(n,)}, got {fj.shape}")
        J[:, j] = (fj - f0) / h

    return J


# ------------------------------ representation -------------------------------


def to_banded(J: np.ndarray, *, ml: int, mu: int) -> np.ndarray:
    """
    Convert dense J (n x n) to banded storage with (ml, mu).

    Parameters
    ----------
    J : ndarray, shape (n, n)
        Dense Jacobian with J[i, j] = ∂f_i/∂y_j.
    ml, mu : int
        Lower and upper bandwidth.

    Returns
    -------
    ab : ndarray, shape (ml + mu + 1, n)
        Banded matrix where ab[mu + i - j, j] = J[i, j] for in-band entries.
        Out-of-band entries are discarded.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square (n, n) array")
    n = J.shape[0]
    ml = int(ml)
    mu = int(mu)
    if ml < 0 or mu < 0:
        raise ValueError("ml and mu must be non-negative")
    ab = np.zeros((ml + mu + 1, n), dtype=float)
    # Fill in-band entries
    for j in range(n):
        i_min = max(0, j - mu)
        i_max = min(n - 1, j + ml)
        row = mu + np.arange(i_min - j, i_max - j + 1)
        ab[row, j] = J[i_min : i_max + 1, j]
    return ab
