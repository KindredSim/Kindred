import numpy as np
import pytest

from kindred.core.simulator.state_model import Edge, StateNetwork, StateType, TSDegreeError


pytestmark = pytest.mark.unit


def test_state_network_add_state_converts_units_and_validates_fields():
    net = StateNetwork()
    st = net.add_state(
        "A",
        kind=StateType.GS,
        energy=(1.0, "kJ/mol"),
        degeneracy=2.0,
        standard_state=" C0 ",
        members=["X", "Y"],
        std_conc_product_M=3.0,
    )

    assert st.name == "A"
    assert st.kind == "GS"
    assert st.energy_jmol == pytest.approx(1000.0)
    assert st.degeneracy == pytest.approx(2.0)
    assert st.standard_state == "C0"
    assert st.members == ("X", "Y")
    assert st.std_conc_product_M == pytest.approx(3.0)

    with pytest.raises(ValueError, match="unsupported energy unit"):
        net.add_state("B", kind=StateType.GS, energy=(1.0, "eV"))

    with pytest.raises(ValueError, match="degeneracy must be positive and finite"):
        net.add_state("C", kind=StateType.GS, energy=0.0, degeneracy=0.0)

    with pytest.raises(ValueError, match="standard_state must be 'C0'"):
        net.add_state("D", kind=StateType.GS, energy=0.0, standard_state="bad")

    with pytest.raises(ValueError, match="kind must be 'GS' or 'TS'"):
        net.add_state("E", kind="weird", energy=0.0)


def test_state_network_rename_state_updates_edges_and_adjacency():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=0.0)
    net.add_state("TS1", kind=StateType.TS, energy=50.0)
    net.add_state("B", kind=StateType.GS, energy=-1.0)
    net.add_edge("A", "TS1")
    net.add_edge("TS1", "B")

    assert net.degree("TS1") == 2

    net.rename_state("TS1", "TS2")

    assert net.get("TS2").name == "TS2"
    assert net.degree("TS2") == 2
    assert {(e.a, e.b) for e in net.edges()} == {("A", "TS2"), ("B", "TS2")}

    net.rename_state("TS2", "TS2")
    assert net.get("TS2").name == "TS2"

    with pytest.raises(KeyError, match="unknown state"):
        net.rename_state("MISSING", "X")

    with pytest.raises(ValueError, match="already exists"):
        net.rename_state("A", "B")


def test_state_network_remove_state_enforces_ts_degree_and_cleans_edges_for_gs():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=0.0)
    net.add_state("TS", kind=StateType.TS, energy=1.0)
    net.add_state("B", kind=StateType.GS, energy=0.0)
    net.add_edge("A", "TS")
    net.add_edge("TS", "B")

    with pytest.raises(TSDegreeError, match="cannot remove TS"):
        net.remove_state("TS")

    net.remove_state("A")
    assert [s.name for s in net.states()] == ["TS", "B"]
    assert {(e.a, e.b) for e in net.edges()} == {("B", "TS")}

    with pytest.raises(KeyError, match="unknown state"):
        net.remove_state("MISSING")


def test_state_network_add_edge_errors_idempotency_and_ts_degree_limit():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=0.0)
    net.add_state("B", kind=StateType.GS, energy=0.0)
    net.add_state("C", kind=StateType.GS, energy=0.0)
    net.add_state("TS", kind=StateType.TS, energy=0.0)

    with pytest.raises(ValueError, match="self-loops are not allowed"):
        net.add_edge("A", "A")

    with pytest.raises(KeyError, match="unknown state\\(s\\)"):
        net.add_edge("A", "MISSING")

    net.add_edge("A", "TS")
    net.add_edge("TS", "B")
    assert net.degree("TS") == 2

    e = net.add_edge("TS", "B")
    assert (e.a, e.b) == ("B", "TS")
    assert net.degree("TS") == 2

    with pytest.raises(TSDegreeError, match="would exceed degree 2"):
        net.add_edge("TS", "C")


def test_state_network_validate_detects_inconsistent_edges_and_ts_degree():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=0.0)
    net.add_state("TS", kind=StateType.TS, energy=0.0)
    net.add_state("B", kind=StateType.GS, energy=0.0)
    net.add_edge("A", "TS")

    with pytest.raises(TSDegreeError, match="must have degree 2"):
        net.validate()

    net.add_edge("TS", "B")
    net.validate()

    net._adj["A"].discard("TS")
    with pytest.raises(ValueError, match="adjacency inconsistent"):
        net.validate()

    net._adj["A"].add("TS")
    net._edges.add(("A", "MISSING"))
    with pytest.raises(ValueError, match="edge references unknown state"):
        net.validate()


def test_state_network_setters_and_serialization_are_deterministic():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=0.0, members=["X"], std_conc_product_M=1.0)
    net.add_state("TS", kind=StateType.TS, energy=1.0)
    net.add_state("B", kind=StateType.GS, energy=0.0)
    net.add_edge("A", "TS")
    net.add_edge("TS", "B")
    net.validate()

    net.set_energy("A", 2.0, "kJ/mol")
    net.set_degeneracy("A", 3.0)
    net.set_standard_state("A", "p0")
    assert net.get("A").energy_jmol == pytest.approx(2000.0)
    assert net.get("A").degeneracy == pytest.approx(3.0)
    assert net.get("A").standard_state == "p0"

    with pytest.raises(KeyError, match="unknown state"):
        net.set_energy("MISSING", 1.0, "J/mol")

    with pytest.raises(ValueError, match="standard_state must be"):
        net.set_standard_state("A", "bad")

    blob = net.to_serializable()
    assert list(blob["states"].keys()) == ["A", "TS", "B"]
    assert blob["states"]["A"]["members"] == ["X"]
    assert blob["states"]["A"]["std_conc_product_M"] == pytest.approx(1.0)
    assert blob["edges"] == [["A", "TS"], ["B", "TS"]]


def test_state_model_covers_edge_and_error_branches():
    e = Edge("A", "B")
    assert e.endpoints() == ("A", "B")

    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=(1.0, "kcal/mol"))
    assert np.isfinite(net.get("A").energy_jmol)

    with pytest.raises(ValueError, match="energy must be finite"):
        net.add_state("B", kind=StateType.GS, energy=(np.inf, "J/mol"))

    with pytest.raises(ValueError, match="standard_state must be a string"):
        net.add_state("C", kind=StateType.GS, energy=0.0, standard_state=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="degeneracy must be numeric"):
        net.add_state("D", kind=StateType.GS, energy=0.0, degeneracy="nope")  # type: ignore[arg-type]

    trimmed = net.add_state("  TRIM  ", kind=StateType.GS, energy=0.0)
    assert trimmed.name == "TRIM"
    assert net.get("TRIM") is trimmed

    with pytest.raises(TypeError, match="species name must be a string"):
        net.degree(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="species name cannot be empty or whitespace"):
        net.get("   ")

    with pytest.raises(KeyError, match="unknown state"):
        net.degree("MISSING")

    with pytest.raises(KeyError, match="unknown state"):
        net.get("MISSING")

    with pytest.raises(ValueError, match="already exists"):
        net.add_state("A", kind=StateType.GS, energy=0.0)

    net.add_state("TS", kind=StateType.TS, energy=0.0)
    net.add_state("X", kind=StateType.GS, energy=0.0)
    net.add_state("Y", kind=StateType.GS, energy=0.0)
    net.add_state("Z", kind=StateType.GS, energy=0.0)
    net.add_edge("X", "TS")
    net.add_edge("Y", "TS")
    with pytest.raises(TSDegreeError, match="would exceed degree 2"):
        net.add_edge("Z", "TS")

    net.remove_edge("X", "Z")

    with pytest.raises(KeyError, match="unknown state"):
        net.set_degeneracy("MISSING", 2.0)

    with pytest.raises(KeyError, match="unknown state"):
        net.set_standard_state("MISSING", "C0")
