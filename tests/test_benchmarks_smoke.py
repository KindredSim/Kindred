from __future__ import annotations

from pathlib import Path

import numpy as np

from kindred.core.datasets.csv_import import load_csv_dataset
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "regression_suite"


def test_nonstiff_case_simulates_without_regression():
    text = (BENCH_DIR / "nonstiff_first_order.dsl").read_text()
    mech = parse_dsl_to_mechanism(text)
    species = mech.species_names()
    rhs = build_ode_rhs_from_mechanism(mech)
    y0 = np.array([mech.species[name].initial_conc for name in species], dtype=float)

    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 4.0),
        y0=y0,
        grid={"N": 12},
        solver="LSODA",
        rtol=1e-7,
        atol=1e-10,
    )
    out = solve_ode(req)

    assert out.t.size == out.Y.shape[1]
    assert out.Y.shape[0] == len(species)
    assert np.all(out.Y >= 0)


def test_global_datasets_share_grid_and_columns():
    csv_paths = sorted(BENCH_DIR.glob("global_consecutive_dataset_*.csv"))
    assert csv_paths, "Global consecutive benchmark datasets are missing"

    baseline_t = None
    for path in csv_paths:
        _, payload = load_csv_dataset(str(path))
        t = payload["t"]
        species = payload["species"]

        assert set(species) == {"A", "B", "C"}
        assert np.all(np.diff(t) > 0)
        assert all(arr.shape == t.shape for arr in species.values())

        if baseline_t is None:
            baseline_t = t
        else:
            assert np.allclose(baseline_t, t)
