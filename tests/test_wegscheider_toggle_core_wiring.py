import pytest
import numpy as np

from kindred.core.fitting_objective import build_fitting_objective
from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism


def _as_float(x):
    return float(x()) if callable(x) else float(x)


def _triangle_dsl() -> str:
    return "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )


def test_prepare_bound_mechanism_binds_wegscheider_dependents_when_enabled():
    dsl = _triangle_dsl()
    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kf1", "kr1", "kf2", "kr2", "kf3"],  # omit kr3 (expected dependent)
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=True,
    )

    assert "kr3" in bound.bindings

    mech = bound.mechanism
    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=True)
    kf3 = _as_float(mech.equilibria[2].kf)
    kr3 = _as_float(mech.equilibria[2].kr)
    assert kf3 / kr3 == pytest.approx(1.0 / 6.0, rel=0, abs=1e-15)

    constrained = (getattr(mech, "metadata", {}) or {}).get("constrained_params") or {}
    assert "kr3" in constrained


def test_build_fitting_objective_accepts_toggle_and_enforces_cyclicity():
    dsl = _triangle_dsl()

    t_exp = np.array([0.0, 1.0], dtype=float)
    y_exp = np.array([0.0, 0.0], dtype=float)

    objective = build_fitting_objective(
        mechanism_text=dsl,
        param_names=["kf1", "kr1", "kf2", "kr2", "kf3"],  # omit kr3 (expected dependent)
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="A",
        temperature_K=298.15,
        initials={},
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        wegscheider_cyclicity_enabled=True,
    )

    residuals = objective(np.array([2.0, 1.0, 3.0, 1.0, 1.0], dtype=float))
    assert np.asarray(residuals).shape == y_exp.shape

    fn = getattr(objective, "_fn", None)
    assert fn is not None and getattr(fn, "__closure__", None) is not None
    closure = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
    mech = closure.get("mechanism")
    assert mech is not None
    kf3 = _as_float(mech.equilibria[2].kf)
    kr3 = _as_float(mech.equilibria[2].kr)
    assert kf3 / kr3 == pytest.approx(1.0 / 6.0, rel=0, abs=1e-15)
