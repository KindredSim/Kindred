import numpy as np
import pytest

from kindred.core.analysis.global_fitting import fit_global
from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism


def _as_float(x):
    return float(x()) if callable(x) else float(x)


def test_global_fit_log10_params_respect_wegscheider_constraints():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )

    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kf1", "kr1", "kf2", "kr2", "kf3", "kr3"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
    )
    mech = bound.mechanism
    mech.metadata["wegscheider_cyclicity_enabled"] = True

    t = np.array([0.0, 1.0], dtype=float)
    datasets = [
        {"id": "d0", "t": t, "y": np.array([0.0, 0.0], dtype=float), "species": "A"},
    ]

    seen = {"called": 0}

    def sim(params):
        seen["called"] += 1
        for name, value in params.items():
            if name in bound.bindings:
                bound.bindings[name].set(float(value))
        _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=True)
        kf3 = _as_float(mech.equilibria[2].kf)
        kr3 = _as_float(mech.equilibria[2].kr)
        assert kf3 / kr3 == pytest.approx(1.0 / 6.0, rel=0, abs=1e-15)
        return {"t": t, "species": {"A": np.array([0.0, 0.0], dtype=float)}}

    result = fit_global(
        sim,
        datasets,
        shared_params={"kf1": 2.0, "kr1": 1.0, "kf2": 3.0, "kr2": 1.0, "kf3": 1.0, "kr3": 1.0},
        bounds={
            "kf1": (1e-6, 1e6),
            "kr1": (1e-6, 1e6),
            "kf2": (1e-6, 1e6),
            "kr2": (1e-6, 1e6),
            "kf3": (1e-6, 1e6),
            "kr3": (1e-6, 1e6),
        },
        log10_params={"kf1": True},
        method="trf",
        max_nfev=2,
    )
    assert seen["called"] >= 1
    assert result is not None
