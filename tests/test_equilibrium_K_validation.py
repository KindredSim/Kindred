import pytest

from kindred.core.mechanism import Equilibrium, Reaction
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.step_indexing import canonical_parameter_names


def test_equilibrium_kf_kr_K_inconsistent_raises_with_context():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=4.0; kr=2.0; K=3.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    with pytest.raises(DSLError) as exc:
        parse_dsl_to_mechanism(dsl, initials={})
    msg = str(exc.value)
    assert "inconsistent" in msg.lower()
    assert "Line 1:" in msg
    assert "equilibrium: A <-> B" in msg


def test_equilibrium_kf_kr_K_consistent_is_accepted_without_overriding():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=4.0; kr=2.0; K=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert float(mech.equilibria[0].kf) == pytest.approx(4.0)
    assert float(mech.equilibria[0].kr) == pytest.approx(2.0)
    assert mech.equilibria[0].metadata.get("Keq_input") == pytest.approx(2.0)
    assert canonical_parameter_names(mech) == {"kf1", "kr1", "Keq1"}


def test_equilibrium_K_only_requires_anchor_rate():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; K=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    with pytest.raises(DSLError) as exc:
        parse_dsl_to_mechanism(dsl, initials={})
    msg = str(exc.value).lower()
    assert "kf" in msg or "kr" in msg
    assert "Line 1:" in str(exc.value)


def test_equilibrium_kf_and_K_derives_kr_deterministically():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=10.0; K=5.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert float(mech.equilibria[0].kf) == pytest.approx(10.0)
    assert float(mech.equilibria[0].kr) == pytest.approx(2.0)
    assert canonical_parameter_names(mech) == {"kf1", "kr1", "Keq1"}


def test_equilibrium_kr_and_K_derives_kf_deterministically():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kr=2.0; K=5.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert float(mech.equilibria[0].kr) == pytest.approx(2.0)
    assert float(mech.equilibria[0].kf) == pytest.approx(10.0)
    assert canonical_parameter_names(mech) == {"kf1", "kr1", "Keq1"}


def test_equilibrium_kf_kr_without_K_does_not_expose_K_param():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=4.0; kr=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert canonical_parameter_names(mech) == {"kf1", "kr1"}


def test_equilibrium_defensively_copies_mutable_inputs() -> None:
    stoich_forward = {"A": 1.0}
    stoich_back = {"B": 1.0}
    metadata = {"source": "user"}

    eq = Equilibrium(
        stoich_forward=stoich_forward,
        stoich_back=stoich_back,
        Keq=2.0,
        kf=4.0,
        kr=2.0,
        metadata=metadata,
    )

    stoich_forward["A"] = 9.0
    stoich_back["B"] = 7.0
    metadata["source"] = "mutated"

    assert eq.stoich_forward == {"A": 1.0}
    assert eq.stoich_back == {"B": 1.0}
    assert eq.metadata == {"source": "user"}

    eq.stoich_forward["A"] = 3.0
    eq.stoich_back["B"] = 4.0
    eq.metadata["source"] = "local"

    assert stoich_forward == {"A": 9.0}
    assert stoich_back == {"B": 7.0}
    assert metadata == {"source": "mutated"}


def test_reaction_defensively_copies_mutable_inputs() -> None:
    stoich = {"A": -1.0, "B": 1.0}
    overrides = {"model": "Arrhenius"}

    reaction = Reaction(stoich=stoich, rate=1.0, overrides=overrides)

    stoich["A"] = -2.0
    overrides["model"] = "Eyring"

    assert reaction.stoich == {"A": -1.0, "B": 1.0}
    assert reaction.overrides == {"model": "Arrhenius"}

    reaction.stoich["A"] = -3.0
    reaction.overrides["model"] = "Custom"

    assert stoich == {"A": -2.0, "B": 1.0}
    assert overrides == {"model": "Eyring"}
