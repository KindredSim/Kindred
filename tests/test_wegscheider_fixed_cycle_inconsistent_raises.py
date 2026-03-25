import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism


def test_wegscheider_inconsistent_fixed_cycle_raises_when_enabled():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=1.0; K=2.0",
            "equilibrium: B <-> C; kf=1.0; K=2.0",
            "equilibrium: C <-> A; kf=1.0; K=2.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    mech.metadata["wegscheider_cyclicity_enabled"] = True

    with pytest.raises(DSLError, match="Wegscheider cyclicity constraints are unsatisfiable"):
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
