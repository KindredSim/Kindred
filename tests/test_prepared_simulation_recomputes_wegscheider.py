import math

import pytest

from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

pytestmark = pytest.mark.unit



def _as_float(x):
    return float(x()) if callable(x) else float(x)


def test_prepared_mechanism_recomputes_wegscheider_on_updates():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = 1 / (Keq1 * Keq2)",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["Keq1", "Keq2", "kf3"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=True,
    )
    mech = bound.mechanism

    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=True)
    kf3 = _as_float(mech.equilibria[2].kf)
    kr3 = _as_float(mech.equilibria[2].kr)
    assert kf3 / kr3 == pytest.approx(1.0 / 6.0, rel=0, abs=1e-15)

    # Change a free parameter and re-apply; dependent should update deterministically.
    bound.bindings["Keq1"].set(4.0)
    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=True)
    kf3b = _as_float(mech.equilibria[2].kf)
    kr3b = _as_float(mech.equilibria[2].kr)
    assert math.isfinite(kf3b) and math.isfinite(kr3b)
    assert kf3b / kr3b == pytest.approx(1.0 / 12.0, rel=0, abs=1e-15)
    assert "Keq3" not in bound.bindings
