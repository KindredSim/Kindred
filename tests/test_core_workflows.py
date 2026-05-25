"""
Targeted regression tests for low-level simulation and fitting workflows.

These tests focus on adversarial scenarios that were not covered by the
integration suite—minimal solver grids and mismatched fitting targets.
"""

import numpy as np
import pytest

from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
from kindred.core.fitting_optimization import FitResult
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode

pytestmark = pytest.mark.integration



def test_simulation_pipeline_handles_minimal_grid():
    """A simple mechanism should integrate correctly even with the smallest grid."""
    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.5",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    y0 = np.array([mechanism.species["A"].initial_conc, mechanism.species["B"].initial_conc])
    rhs = build_ode_rhs_from_mechanism(mechanism)

    request = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 2.0),
        y0=y0,
        solver="BDF",
        rtol=1e-7,
        atol=1e-12,
        grid={"N": 4},
    )
    result = solve_ode(request)
    assert result.t.size == 4
    assert np.all(np.diff(result.t) > 0)
    # Concentration of A should monotonically decrease for a first-order decay.
    assert np.all(np.diff(result.Y[0]) <= 1e-12)


def test_reversible_equilibrium_converges_to_expected_ratio():
    """A reversible equilibrium should settle to the kf/kr ratio."""
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    species_names = list(mechanism.species.keys())
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names])
    rhs = build_ode_rhs_from_mechanism(mechanism)

    request = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 10.0),
        y0=y0,
        solver="BDF",
        rtol=1e-8,
        atol=1e-12,
        grid={"N": 50},
    )
    result = solve_ode(request)
    final_A = result.Y[species_names.index("A"), -1]
    final_B = result.Y[species_names.index("B"), -1]
    total = final_A + final_B

    # Mass balance must hold and the equilibrium ratio should match kf/kr.
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert pytest.approx(final_B / final_A, rel=1e-2) == 2.0


def test_fit_global_final_replay_executes_parameterized_intervention_schedule(monkeypatch):
    import kindred.core.analysis.global_fitting as global_fitting

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["dose"],
        t_end=2.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    def _fit_parameters(_objective, _initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters={"dose": 3.0},
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(3, dtype=float),
            nfev=1,
            message="forced optimum",
        )

    monkeypatch.setattr(global_fitting, "fit_parameters", _fit_parameters)

    result = global_fitting.fit_global(
        evaluator,
        [
            {
                "id": "ds1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species": "A",
                "y": np.asarray([1.0, 4.0, 4.0], dtype=float),
            }
        ],
        shared_params={"dose": 1.0},
        bounds={"dose": (0.0, 10.0)},
        method="trf",
        max_nfev=1,
        max_runtime_lanes=1,
    )

    assert result.completion.status == "ok"
    assert result.shared_params["dose"] == pytest.approx(3.0)
    replayed_a = np.asarray(result.model_series["ds1"]["A"], dtype=float)
    assert float(replayed_a[-1]) == pytest.approx(4.0, abs=1e-6)
