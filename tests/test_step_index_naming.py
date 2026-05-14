import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
from kindred.core.simulator.step_indexing import canonical_parameter_names, get_step_index_map
from kindred.core.algebra.symbol_table import build_algebra_symbol_table
from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui

pytestmark = pytest.mark.unit



def test_canonical_step_names_mixed_mechanism_no_K_param():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "equilibrium: B <-> C; kf=2.0; kr=0.5",
            "reaction: C -> D; k=0.2",
            "init: A=1.0, B=0.0, C=0.0, D=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert canonical_parameter_names(mech) == {"k1", "kf2", "kr2", "Keq2", "k3"}
    # Guard against regressions to per-type ordinal naming (reactions-only and equilibria-only counters).
    assert canonical_parameter_names(mech) != {"k1", "k2", "kf1", "kr1"}

    step_map = get_step_index_map(mech)
    assert [e.get("step_index") for e in step_map] == [1, 2, 3]
    assert [e.get("kind") for e in step_map] == ["reaction", "equilibrium", "reaction"]


def test_parameter_algebra_targets_canonical_step_names_in_mixed_mechanism():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "equilibrium: B <-> C; kf=2.0; kr=0.5",
            "reaction: C -> D; k=0.2",
            "init: A=1.0, B=0.0, C=0.0, D=0.0",
            "",
            "# Algebra",
            "param a = 4",
            "param k3 = a*kr2",
            "",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.reactions[1].rate) == pytest.approx(4.0 * float(mech.equilibria[0].kr))


def test_post_solve_symbol_table_exposes_canonical_names_only():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "equilibrium: B <-> C; kf=2.0; kr=0.5",
            "reaction: C -> D; k=0.2",
            "init: A=1.0, B=0.0, C=0.0, D=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    symtab = build_algebra_symbol_table(mech)
    assert symtab.get("k1") == pytest.approx(float(mech.reactions[0].rate))
    assert symtab.get("kf2") == pytest.approx(float(mech.equilibria[0].kf))
    assert symtab.get("kr2") == pytest.approx(float(mech.equilibria[0].kr))
    assert symtab.get("Keq2") == pytest.approx(float(mech.equilibria[0].kf) / float(mech.equilibria[0].kr))
    assert "K2" not in symtab.user_names()


def test_fully_explicit_equilibrium_does_not_mark_a_derived_rate():
    mech = parse_dsl_to_mechanism(
        "equilibrium: A <-> B ; kf=10 ; kr=5 ; Keq=2\ninitial: A=1.0\ninitial: B=0.0",
        initials={},
    )

    step_map = get_step_index_map(mech)

    assert step_map[0]["derive_rate"] is None


def test_symbol_table_exposes_keq_without_indexed_k_alias_for_explicit_equilibrium_constants():
    mech = parse_dsl_to_mechanism(
        "equilibrium: A <-> B ; kf=10 ; K=2\ninitial: A=1.0\ninitial: B=0.0",
        initials={},
    )

    symtab = build_algebra_symbol_table(mech)

    with pytest.raises(KeyError):
        symtab.get("K1")
    assert symtab.get("Keq1") == pytest.approx(2.0)


def test_symbol_table_requires_authoritative_parameter_namespace():
    class _MechanismWithoutStepMap:
        reactions = []
        equilibria = []
        metadata = {}

    with pytest.raises(ValueError, match="step_index_map"):
        build_algebra_symbol_table(_MechanismWithoutStepMap())


def test_gui_parameter_enumeration_returns_canonical_names_and_derived_flags():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "equilibrium: B <-> C; kr=2.0; K=5.0",
            "reaction: C -> D; k=0.2",
            "init: A=1.0, B=0.0, C=0.0, D=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    # Even with no `param ...` statements, K-implied constraints should apply.
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)

    variables, metadata = enumerate_step_parameters_for_gui(mech)
    assert list(variables.keys()) == ["k1", "kf2", "kr2", "Keq2", "k3"]

    # For "kr=...; K=..." (kf not explicit), policy derives kf from kr*K and disables kf slider.
    assert float(mech.equilibria[0].kf) == pytest.approx(float(mech.equilibria[0].kr) * float(mech.equilibria[0].metadata["Keq_input"]))
    assert metadata["Keq2"].get("editable") is not False
    assert metadata["kf2"].get("derived") is True
    assert metadata["kf2"].get("editable") is False
    assert metadata["kr2"].get("derived") is not True


def test_gui_parameter_enumeration_includes_implicit_equilibrium_constant_identity():
    mech = parse_dsl_to_mechanism(
        "equilibrium: A <-> B ; kf=10 ; kr=2\ninitial: A=1.0\ninitial: B=0.0",
        initials={},
    )

    variables, metadata = enumerate_step_parameters_for_gui(mech)

    assert list(variables.keys()) == ["kf1", "kr1", "Keq1"]
    assert variables["Keq1"] == pytest.approx(5.0)
    assert metadata["Keq1"]["value_valid"] is True
    assert metadata["Keq1"].get("editable") is False
    assert metadata["Keq1"].get("derived") is True
    assert metadata["kf1"].get("derived") is not True
    assert metadata["kr1"].get("derived") is not True


def test_gui_parameter_enumeration_implicit_keq_tracks_algebra_mutated_rates():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kf=2 ; kr=1",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param kf1 = 4",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)

    variables, metadata = enumerate_step_parameters_for_gui(mech)

    assert float(mech.equilibria[0].kf) == pytest.approx(4.0)
    assert float(mech.equilibria[0].Keq) == pytest.approx(2.0)
    assert variables["Keq1"] == pytest.approx(4.0)
    assert metadata["Keq1"].get("editable") is False
    assert metadata["Keq1"].get("derived") is True


def test_state_network_generated_steps_do_not_consume_step_indices():
    dsl = "\n".join(
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
    mech = parse_dsl_to_mechanism(dsl, initials={})
    assert len(mech.reactions) >= 1

    step_map = get_step_index_map(mech)
    assert len(step_map) == 1
    assert canonical_parameter_names(mech) == {"k1"}
