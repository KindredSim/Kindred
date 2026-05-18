from kindred.core.simulator.dsl_text_update import analyze_step_parameter_update
from kindred.core.simulator.dsl import extract_parameters_from_dsl, parse_dsl_to_mechanism
from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui


def test_updating_derived_kr_on_keq_authority_equilibrium_is_blocked():
    source = "equilibrium: A <-> B; kf=6.0; K=3.0"

    outcome = analyze_step_parameter_update(source, "kr1", 4.0)

    assert not outcome.writable
    assert outcome.warning_reason == "target_unwritable"
    assert outcome.updated_text == source


def test_updating_derived_kr_on_dg_authority_equilibrium_is_blocked():
    source = "equilibrium: A <-> B; kf=6.0; dG_eq=-1.0"

    outcome = analyze_step_parameter_update(source, "kr1", 4.0)

    assert not outcome.writable
    assert outcome.warning_reason == "target_unwritable"
    assert outcome.updated_text == source


def test_gui_enumeration_marks_dg_authority_reverse_rate_as_derived():
    mechanism = parse_dsl_to_mechanism("equilibrium: A <-> B; kf=6.0; dG_eq=-1.0", initials={})

    _variables, metadata = enumerate_step_parameters_for_gui(mechanism)

    assert metadata["kr1"]["editable"] is False
    assert metadata["kr1"]["derived"] is True


def test_updating_kf_on_dg_authority_equilibrium_preserves_public_authority():
    source = "equilibrium: A <-> B; kf=6.0; dG_eq=-1.0"

    outcome = analyze_step_parameter_update(source, "kf1", 7.0)

    assert outcome.writable
    assert "kf=7" in outcome.updated_text
    assert "dG_eq=-1.0" in outcome.updated_text
    assert "kr=" not in outcome.updated_text
    parse_dsl_to_mechanism(outcome.updated_text, initials={})


def test_dg_authority_keq_display_uses_metadata_value_with_std_ratio():
    source = (
        "# === Generated from Computational Mode ===\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq; cm_std_ratio=0.5\n"
        "# === End Generated from Computational Mode ==="
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})

    variables, metadata = enumerate_step_parameters_for_gui(mechanism)
    scan = {item.name: item for item in extract_parameters_from_dsl(source)}

    assert variables["Keq1"] == mechanism.equilibria[0].Keq
    assert scan["Keq1"].value == mechanism.equilibria[0].Keq
    assert metadata["Keq1"]["editable"] is False
