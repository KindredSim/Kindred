import math

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism


def _as_float(x):
    return float(x()) if callable(x) else float(x)


def _ln_k_ratio(mech, step_index: int) -> float:
    eq = mech.equilibria[step_index - 1]
    kf = _as_float(eq.kf)
    kr = _as_float(eq.kr)
    assert kf > 0.0 and kr > 0.0
    return math.log(kf) - math.log(kr)


def test_wegscheider_triangle_enforces_cycle_and_marks_derived():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    mech.metadata["wegscheider_cyclicity_enabled"] = True

    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)

    # Cycle condition: ln(K1) + ln(K2) + ln(K3) == 0 (forward orientations match the DSL sides).
    cycle_sum = _ln_k_ratio(mech, 1) + _ln_k_ratio(mech, 2) + _ln_k_ratio(mech, 3)
    assert cycle_sum == pytest.approx(0.0, abs=1e-12)

    constrained = (mech.metadata or {}).get("constrained_params") or {}
    assert "kr3" in constrained

