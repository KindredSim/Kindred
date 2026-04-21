from kindred.core.simulator.dsl import parse_dsl_to_mechanism
import pytest

pytestmark = pytest.mark.unit



def test_parse_dsl_to_mechanism_allows_state_network_with_no_initials_mapping():
    dsl_text = "\n".join(
        [
            "reaction: X -> Y; k=1.0",
            "init: X=1.0, Y=0.0",
            "",
            "# State Network",
            "state: name=A; kind=GS; energy=0.0; degeneracy=1",
            "state: name=B; kind=GS; energy=5.0; degeneracy=1",
            "state: name=TS1; kind=TS; energy=20.0; degeneracy=1",
            "edge: A,TS1",
            "edge: B,TS1",
            "",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl_text)
    assert "X" in mech.species
    assert "Y" in mech.species

