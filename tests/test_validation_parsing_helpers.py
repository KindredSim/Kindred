from dataclasses import replace

import pytest

pytestmark = pytest.mark.unit



def test_try_parse_callable_finite_float_accepts_scalar_and_callable():
    from kindred.core.validation import try_parse_callable_finite_float

    parsed, ok = try_parse_callable_finite_float("1.25")
    assert ok is True
    assert parsed == pytest.approx(1.25)

    parsed2, ok2 = try_parse_callable_finite_float(lambda: "2.5")
    assert ok2 is True
    assert parsed2 == pytest.approx(2.5)

    parsed3, ok3 = try_parse_callable_finite_float(lambda: "nan")
    assert ok3 is False
    assert parsed3 == pytest.approx(0.0)


def test_bind_parameters_defaults_when_callable_raises_or_is_nonfinite():
    from kindred.core.simulation_preparation import _bind_parameters_to_mechanism
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.units import UnitsModel

    dsl = "\n".join(
        [
            "reaction: A -> B ; k=0.1",
            "init: A=1, B=0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={}, units=UnitsModel(temperature_K=300.0))
    rxn0 = mech.reactions[0]

    class _Boom:
        def __call__(self) -> float:
            raise RuntimeError("boom")

    mech.reactions[0] = replace(rxn0, rate=_Boom())
    bindings = _bind_parameters_to_mechanism(mech, ["k1"])
    assert bindings["k1"]() == pytest.approx(1.0)

    mech2 = parse_dsl_to_mechanism(dsl, initials={}, units=UnitsModel(temperature_K=300.0))
    rxn0b = mech2.reactions[0]
    mech2.reactions[0] = replace(rxn0b, rate=float("nan"))
    bindings2 = _bind_parameters_to_mechanism(mech2, ["k1"])
    assert bindings2["k1"]() == pytest.approx(1.0)

