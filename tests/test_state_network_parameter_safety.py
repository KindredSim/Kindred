import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError

pytestmark = pytest.mark.unit



def _dsl_with_state_network() -> str:
    return "\n".join(
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


@pytest.mark.parametrize("binding_name", ["k1", "kf2"])
def test_state_network_generated_steps_must_not_introduce_canonical_named_bindings(monkeypatch, binding_name):
    from kindred.core.mechanism import Mechanism
    from kindred.core.ode_builder import RateBinding
    import kindred.core.simulator.state_network_converter as snc

    def _fake_convert(*_args, **_kwargs):
        mech = Mechanism()
        mech.add_species("A", 0.0)
        mech.add_species("B", 0.0)
        mech.add_reaction(
            reactants={"A": 1.0},
            products={"B": 1.0},
            rate=RateBinding(name=binding_name, value=1.0),
        )
        return mech

    monkeypatch.setattr(snc, "convert_state_network_to_mechanism", _fake_convert)

    with pytest.raises(DSLError) as exc:
        parse_dsl_to_mechanism(_dsl_with_state_network(), initials={})
    msg = str(exc.value)
    assert "State-network generated steps currently do not participate in step indexing" in msg
    assert f"introduced a canonical-looking parameter name {binding_name!r}" in msg
    assert "non-numeric binding/callable" not in msg
    assert "parameters.\n\nState-network reaction" in msg
    assert "\\n" not in msg


def test_state_network_generated_steps_must_not_introduce_any_non_numeric_bindings(monkeypatch):
    from kindred.core.mechanism import Mechanism
    from kindred.core.ode_builder import RateBinding
    import kindred.core.simulator.state_network_converter as snc

    def _fake_convert(*_args, **_kwargs):
        mech = Mechanism()
        mech.add_species("A", 0.0)
        mech.add_species("B", 0.0)
        mech.add_reaction(
            reactants={"A": 1.0},
            products={"B": 1.0},
            rate=RateBinding(name="state_rate", value=1.0),
        )
        return mech

    monkeypatch.setattr(snc, "convert_state_network_to_mechanism", _fake_convert)

    with pytest.raises(DSLError) as exc:
        parse_dsl_to_mechanism(_dsl_with_state_network(), initials={})
    assert "non-numeric binding" in str(exc.value).lower() or "binding" in str(exc.value).lower()
