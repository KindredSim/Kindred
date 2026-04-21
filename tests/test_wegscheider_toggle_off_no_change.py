from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
import pytest

pytestmark = pytest.mark.unit



def test_wegscheider_toggle_off_leaves_rates_unchanged():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    mech.metadata["wegscheider_cyclicity_enabled"] = False

    before = [(float(eq.kf), float(eq.kr)) for eq in mech.equilibria]
    _ = apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    after = [(float(eq.kf), float(eq.kr)) for eq in mech.equilibria]

    assert after == before

