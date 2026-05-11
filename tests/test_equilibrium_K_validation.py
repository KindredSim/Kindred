import pytest

from kindred.core.mechanism import Equilibrium, Mechanism, Reaction
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.step_indexing import canonical_parameter_names

pytestmark = pytest.mark.unit



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


def test_equilibrium_kf_kr_without_K_still_has_canonical_keq_param():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=4.0; kr=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert canonical_parameter_names(mech) == {"kf1", "kr1", "Keq1"}


def test_equilibrium_defensively_copies_and_freezes_mutable_inputs() -> None:
    stoich_forward = {"A": 1.0}
    stoich_back = {"B": 1.0}
    metadata = {
        "source": "user",
        "explicit_rates": [1.0, 2.0],
        "forward_model": {"A": 1.0},
    }

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
    metadata["explicit_rates"].append(3.0)
    metadata["forward_model"]["A"] = 9.0

    assert eq.stoich_forward == {"A": 1.0}
    assert eq.stoich_back == {"B": 1.0}
    assert eq.metadata["source"] == "user"
    assert eq.metadata["explicit_rates"] == (1.0, 2.0)
    assert eq.metadata["forward_model"] == {"A": 1.0}

    with pytest.raises(TypeError):
        eq.stoich_forward["A"] = 3.0
    with pytest.raises(TypeError):
        eq.stoich_back["B"] = 4.0
    with pytest.raises(TypeError):
        eq.metadata["source"] = "local"
    with pytest.raises(TypeError):
        eq.metadata["explicit_rates"][0] = 9.0
    with pytest.raises(TypeError):
        eq.metadata["forward_model"]["A"] = 9.0

    assert stoich_forward == {"A": 9.0}
    assert stoich_back == {"B": 7.0}
    assert metadata == {
        "source": "mutated",
        "explicit_rates": [1.0, 2.0, 3.0],
        "forward_model": {"A": 9.0},
    }


def test_equilibrium_rejects_negative_physical_side_coefficients() -> None:
    with pytest.raises(ValueError, match="stoich_forward coefficient"):
        Equilibrium(stoich_forward={"A": -1.0}, stoich_back={"B": 1.0}, Keq=2.0)


def test_equilibrium_rejects_empty_physical_sides() -> None:
    with pytest.raises(ValueError, match="stoich_forward cannot be empty"):
        Equilibrium(stoich_forward={}, stoich_back={"B": 1.0}, Keq=2.0)
    with pytest.raises(ValueError, match="stoich_back cannot be empty"):
        Equilibrium(stoich_forward={"A": 1.0}, stoich_back={}, Keq=2.0)

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    with pytest.raises(ValueError, match="stoich_forward cannot be empty"):
        mechanism.add_equilibrium({}, {"B": 1.0}, Keq=2.0)
    with pytest.raises(ValueError, match="stoich_back cannot be empty"):
        mechanism.add_equilibrium({"A": 1.0}, {}, Keq=2.0)


def test_reaction_defensively_copies_and_freezes_mutable_inputs() -> None:
    reactants = {"A": 1.0}
    products = {"B": 1.0}
    overrides = {"model": "Arrhenius"}

    reaction = Reaction(reactants=reactants, products=products, rate=1.0, overrides=overrides)

    reactants["A"] = 2.0
    products["B"] = 2.0
    overrides["model"] = "Eyring"

    assert reaction.reactants == {"A": 1.0}
    assert reaction.products == {"B": 1.0}
    assert reaction.net_stoich == {"A": -1.0, "B": 1.0}
    assert reaction.overrides == {"model": "Arrhenius"}

    with pytest.raises(TypeError):
        reaction.reactants["A"] = 3.0
    with pytest.raises(TypeError):
        reaction.products["B"] = 4.0
    with pytest.raises(TypeError):
        reaction.net_stoich["A"] = -3.0
    with pytest.raises(TypeError):
        reaction.rate_orders["A"] = 3.0
    with pytest.raises(TypeError):
        reaction.overrides["model"] = "Custom"

    assert reactants == {"A": 2.0}
    assert products == {"B": 2.0}
    assert overrides == {"model": "Eyring"}


def test_mechanism_add_reaction_preserves_explicit_empty_rate_orders() -> None:
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)

    reaction = mechanism.add_reaction(
        reactants={"A": 1.0},
        products={"B": 1.0},
        rate=1.0,
        rate_orders={},
    )

    assert reaction.rate_orders == {}
    assert reaction.order == 0
