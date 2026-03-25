from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

import kindred.core.analysis.global_fitting as global_fitting
import kindred.core.fitting_optimization as fitting_optimization
from kindred.core.optimization_de import compute_de_popsize_maxiter
from kindred.core.fitting_optimization import fit_parameters


@dataclass
class _DummyDEResult:
    x: np.ndarray
    success: bool = True
    message: str = "ok"
    nfev: int = 0


def _make_de_stub(calls: List[Tuple[int, int, int, Optional[bool]]]):
    def _stub(func, bounds, **kwargs):
        popsize = int(kwargs["popsize"])
        maxiter = int(kwargs["maxiter"])
        polish = kwargs.get("polish")
        calls.append((popsize, maxiter, len(bounds), polish))
        return _DummyDEResult(x=np.zeros(len(bounds), dtype=float))

    return _stub


def test_compute_de_popsize_maxiter_regression() -> None:
    cases = [
        (1, 1, 5, 1),
        (25, 1, 5, 4),
        (80, 1, 6, 12),
        (200, 2, 7, 13),
        (1000, 1, 15, 65),
        (2000, 10, 10, 19),
    ]
    for budget, dim, expected_popsize, expected_maxiter in cases:
        popsize, maxiter = compute_de_popsize_maxiter(budget=budget, dim=dim)
        assert (popsize, maxiter) == (expected_popsize, expected_maxiter)


def test_de_budget_call_sites_match_helper(monkeypatch) -> None:
    cases = [
        (1, 1),
        (80, 1),
        (200, 2),
        (1000, 1),
        (2000, 10),
    ]

    fitting_calls: List[Tuple[int, int, int, Optional[bool]]] = []
    global_calls: List[Tuple[int, int, int, Optional[bool]]] = []
    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (lambda *_a, **_k: None, _make_de_stub(global_calls)),
    )

    def objective(_params: np.ndarray) -> np.ndarray:
        return np.ones(1, dtype=float)

    t = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)

    def simulation_func(_params):
        return {"t": t, "species": {"A": y}}

    datasets = [{"id": "ds1", "t": t, "y": y, "species": "A"}]

    for budget, dim in cases:
        expected_popsize, expected_maxiter = compute_de_popsize_maxiter(budget=budget, dim=dim)

        initial_params = {f"p{i}": 0.0 for i in range(dim)}
        bounds = {name: (0.0, 1.0) for name in initial_params}
        fit_parameters(
            objective,
            initial_params,
            bounds=bounds,
            method="de",
            max_nfev=budget,
            seed=123,
            scipy_loader=lambda: (lambda *_a, **_k: None, _make_de_stub(fitting_calls)),
        )

        shared_params = {f"p{i}": 0.0 for i in range(dim)}
        global_fitting.fit_global(
            simulation_func,
            datasets,
            shared_params,
            bounds=bounds,
            method="de",
            max_nfev=budget,
            seed=123,
        )

        assert fitting_calls[-1] == (expected_popsize, expected_maxiter, dim, False)
        assert global_calls[-1] == (expected_popsize, expected_maxiter, dim, False)
