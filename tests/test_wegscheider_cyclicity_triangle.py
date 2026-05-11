import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

pytestmark = pytest.mark.unit



def _as_float(x):
    return float(x()) if callable(x) else float(x)


def _triangle_dsl(*, resolved: bool) -> str:
    lines = [
        "equilibrium: A <-> B; kf=2.0; K=2.0",
        "equilibrium: B <-> C; kf=3.0; K=3.0",
        "equilibrium: C <-> A; kf=1.0; K=1.0",
    ]
    if resolved:
        lines.append("param Keq3 = 1 / (Keq1 * Keq2)")
    lines.append("init: A=1.0, B=0.0, C=0.0")
    return "\n".join(lines)


def test_wegscheider_triangle_requires_symbolic_keq_dependency():
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
        analyze_wegscheider_cyclicity,
    )

    dsl = _triangle_dsl(resolved=False)
    mech = parse_dsl_to_mechanism(dsl, initials={})
    mech.metadata["wegscheider_cyclicity_enabled"] = True

    report = analyze_wegscheider_cyclicity(mech)
    assert report.cycles[0].step_indices == (1, 2, 3)
    assert report.cycles[0].coefficients == (1, 1, 1)
    assert report.is_resolved is False

    with pytest.raises(UnresolvedWegscheiderCyclicityError):
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)


def test_wegscheider_triangle_symbolic_dependency_marks_keq_not_rate_derived():
    dsl = _triangle_dsl(resolved=True)
    mech = parse_dsl_to_mechanism(dsl, initials={})
    mech.metadata["wegscheider_cyclicity_enabled"] = True

    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)

    assert _as_float(mech.equilibria[2].metadata["Keq_input"]) == pytest.approx(1.0 / 6.0)
    assert _as_float(mech.equilibria[2].kr) == pytest.approx(6.0)

    constrained = (mech.metadata or {}).get("constrained_params") or {}
    assert constrained["Keq3"]["constraint_reason"] == "algebra"
    assert "kr3" not in constrained
