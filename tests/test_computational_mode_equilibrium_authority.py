from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest
from PySide6.QtCore import Qt

from kindred.core.document_parameter_store import DocumentParameterStore
from kindred.core.exceptions import FitSimulationError
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    authority_readout_updates_from_step_entry,
    effective_equilibrium_keq,
    normalize_existing_equilibrium_rate_authority,
)
from kindred.core.simulator.computational_mode import CompReaction, CompSpec, CompSpecies, compile_comp_spec
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
from kindred.core.simulator.dsl import (
    extract_parameter_names_from_dsl,
    extract_parameters_from_dsl,
    parse_dsl_to_mechanism,
)
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
)
from kindred.core.simulator.state_model import StateNetwork, StateType
from kindred.core.simulator.state_network_converter import StateNetworkConverter
from kindred.core.symbolic.jacobian import classify_symbolic_jacobian_support
from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab


pytestmark = pytest.mark.unit


def test_computational_mode_fast_equilibrium_generates_kf_dg_authority_without_kr():
    spec = CompSpec(
        temperature_K=298.15,
        energy_unit="kJ/mol",
        std_default_M=1.0,
        species={
            "A": CompSpecies(name="A", kind="GS", G_value=0.0, std_M=1.0, cref_M=1.0),
            "B": CompSpecies(name="B", kind="GS", G_value=1.0, std_M=1.0, cref_M=1.0),
        },
        reactions=[
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=10.0),
        ],
    )

    compiled = compile_comp_spec(spec, output_energy_unit="kJ/mol")

    assert "equilibrium: A <-> B" in compiled.generated_reaction_dsl
    assert "; kf=10" in compiled.generated_reaction_dsl
    assert "; dG_eq=" in compiled.generated_reaction_dsl
    assert "; cm_std_ratio=" in compiled.generated_reaction_dsl
    assert "; kr=" not in compiled.generated_reaction_dsl


def test_computational_mode_fast_equilibrium_parses_to_effective_reverse_rate():
    spec = CompSpec(
        temperature_K=298.15,
        energy_unit="kJ/mol",
        std_default_M=1.0,
        species={
            "A": CompSpecies(name="A", kind="GS", G_value=0.0, std_M=1.0, cref_M=1.0),
            "B": CompSpecies(name="B", kind="GS", G_value=1.0, std_M=1.0, cref_M=1.0),
        },
        reactions=[
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=10.0),
        ],
    )
    compiled = compile_comp_spec(spec, output_energy_unit="kJ/mol")
    mechanism = parse_dsl_to_mechanism(
        f"{GENERATED_BLOCK_START}\n{compiled.generated_reaction_dsl}\n{GENERATED_BLOCK_END}\ninit: A=1.0, B=0.0",
        initials={},
    )
    rhs = build_ode_rhs_from_mechanism(mechanism)

    eq = mechanism.equilibria[0]
    expected_keq = math.exp(-1000.0 / (8.31446261815324 * 298.15))
    expected_kr = 10.0 / expected_keq

    assert float(eq.kf) == pytest.approx(10.0)
    assert float(eq.kr) == pytest.approx(expected_kr)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [expected_kr, -expected_kr])


def test_state_network_direct_fast_equilibrium_uses_thermodynamic_authority():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_edge("A", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(network, initials={"A": 0.0, "B": 1.0})
    rhs = build_ode_rhs_from_mechanism(mechanism)

    eq = mechanism.equilibria[0]
    authority = normalize_existing_equilibrium_rate_authority(eq)
    expected_keq = math.exp(-1000.0 / (8.31446261815324 * 298.15))
    expected_std_ratio = 0.5
    expected_kr = float(eq.kf) / (expected_keq * expected_std_ratio)

    assert authority.kind == EquilibriumRateAuthorityKind.KEQ
    assert eq.Keq == pytest.approx(expected_keq)
    assert eq.kr == pytest.approx(expected_kr)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [expected_kr, -expected_kr])


def test_state_network_transition_state_readback_uses_explicit_rate_authority_for_keq():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_state("TS", kind=StateType.TS, energy=(50.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_edge("A", "TS")
    network.add_edge("TS", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(
        network,
        initials={"A": 0.0, "B": 1.0},
    )

    eq = mechanism.equilibria[0]
    authority = normalize_existing_equilibrium_rate_authority(eq)
    expected_keq = float(eq.kf) / float(eq.kr)
    algebra_values = read_mechanism_parameter_values(mechanism, names={"Keq1"})

    assert authority.kind == EquilibriumRateAuthorityKind.KR
    assert eq.metadata["Keq"] != pytest.approx(expected_keq)
    assert effective_equilibrium_keq(eq) == pytest.approx(expected_keq)
    assert algebra_values["Keq1"] == pytest.approx(expected_keq)


def test_state_network_transition_state_readout_refresh_derives_keq_from_current_rates():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_state("TS", kind=StateType.TS, energy=(50.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_edge("A", "TS")
    network.add_edge("TS", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(
        network,
        initials={"A": 0.0, "B": 1.0},
    )
    entry = mechanism.metadata["step_index_map"][0]

    updates = {
        update.name: update.value
        for update in authority_readout_updates_from_step_entry(
            entry,
            {"kf1": 4.0, "kr1": 1.0, "Keq1": 2.0},
        )
    }

    assert updates["Keq1"] == pytest.approx(4.0)


def test_generated_computational_mode_payload_rejects_legacy_kr_without_std_ratio():
    legacy = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=legacy\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="generated Computational Mode kr"):
        parse_dsl_to_mechanism(legacy, initials={})


def test_generated_computational_mode_payload_rejects_legacy_kr_with_std_ratio():
    legacy = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=legacy; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="generated Computational Mode kr"):
        parse_dsl_to_mechanism(legacy, initials={})


def test_generated_computational_mode_payload_rejects_missing_std_ratio():
    incomplete = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=missing_std\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="cm_std_ratio"):
        parse_dsl_to_mechanism(incomplete, initials={})


def test_generated_computational_mode_payload_rejects_missing_cm_id():
    incomplete = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_std_ratio=0.5\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="cm_id"):
        parse_dsl_to_mechanism(incomplete, initials={})


def test_generated_computational_mode_payload_rejects_missing_dg_eq():
    incomplete = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; cm_id=missing_dg; cm_std_ratio=0.5\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="dG_eq"):
        parse_dsl_to_mechanism(incomplete, initials={})


def test_generated_computational_mode_payload_rejects_keq_authority():
    incompatible = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; Keq=2.0; cm_id=keq_authority; cm_std_ratio=0.5\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="Keq"):
        parse_dsl_to_mechanism(incompatible, initials={})


def test_public_dsl_cannot_self_declare_computational_mode_legacy_authority():
    public_spoof = (
        "energy=kJ/mol\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=spoof\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="generated equilibrium fields"):
        parse_dsl_to_mechanism(public_spoof, initials={})


def test_generated_fast_runtime_parameters_are_owned_by_generated_metadata_only():
    mixed_source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "equilibrium: C <-> D; kf=4.0; dG_eq=2.0\n"
        "init: A=1.0, B=0.0, C=1.0, D=0.0"
    )
    store = DocumentParameterStore()
    store.set_schema(mixed_source)

    names = set(store.parameter_names())

    assert "dG_eq_fast__feq__A__B" in names
    assert "kf1" not in names
    assert "kr1" not in names
    assert "Keq1" not in names
    assert "dG_eq_fast__feq__C__D" not in names

    generated_bound = prepare_bound_mechanism(
        mixed_source,
        ["dG_eq_fast__feq__A__B"],
        initials={},
    )
    assert "dG_eq_fast__feq__A__B" in generated_bound.bindings
    assert "kf1" not in generated_bound.bindings
    assert "kr1" not in generated_bound.bindings
    assert "Keq1" not in generated_bound.bindings

    for indexed_name in ("kf1", "kr1", "Keq1"):
        with pytest.raises(FitSimulationError, match="not a valid indexed|Unknown request parameter"):
            prepare_bound_mechanism(
                mixed_source,
                [indexed_name],
                initials={},
            )

    with pytest.raises(FitSimulationError, match="Unknown request parameter"):
        prepare_bound_mechanism(
            mixed_source,
            ["dG_eq_fast__feq__C__D"],
            initials={},
        )


def test_fitting_tab_does_not_publish_noneditable_equilibrium_readouts(qt_app, qtbot):
    source = "energy=kJ/mol\nequilibrium: A <-> B; kf=6.0; dG_eq=-1.0\ninit: A=1.0, B=0.0"
    definitions = [asdict(param) for param in extract_parameters_from_dsl(source)]
    state_owner = type(
        "_StateOwner",
        (),
        {"_global_dataset_variable_params": {}},
    )()
    parameter_state = ParametersIcsTab._build_parameter_state(state_owner, definitions)

    rows_by_name = {row["param_name"]: row for row in parameter_state}
    assert rows_by_name["kf1"]["fit"] is True
    assert rows_by_name["kr1"]["fit"] is False
    assert rows_by_name["Keq1"]["fit"] is False

    tab = ParametersIcsTab(
        parameter_state=parameter_state,
        initial_parameter_snapshot=parameter_state,
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={definition["name"]: definition for definition in definitions},
        mechanism_species=["A", "B"],
        dataset_entries=[{"id": "dataset"}],
        prepared_param_names=[definition["name"] for definition in definitions],
        selected_dataset_ids_getter=lambda: ["dataset"],
        dataset_entries_getter=lambda: [{"id": "dataset"}],
        worker_running_getter=lambda: False,
        dataset_fit_settings_store_getter=lambda: None,
        mechanism_parameter_scan_owner_getter=lambda: None,
        mechanism_text_getter=lambda: source,
        integration_defaults=("BDF", 1e-6, 1e-12),
        config_defaults={},
    )
    qtbot.addWidget(tab)

    row_index = {tab._param_table.item(row, 2).text(): row for row in range(tab._param_table.rowCount())}
    for name in ("kr1", "Keq1"):
        fit_item = tab._param_table.item(row_index[name], 0)
        assert fit_item.checkState() == Qt.Unchecked
        assert not (fit_item.flags() & Qt.ItemIsUserCheckable)

    bundle = tab.collect_parameter_config_bundle(show_errors=False)
    assert bundle is not None
    config, _dataset_params, _dataset_variable_params = bundle

    assert set(config["parameters"]) == {"kf1"}
    assert "kr1" not in config["fixed_params"]
    assert "Keq1" not in config["fixed_params"]


def test_generated_fast_endpoint_name_cannot_be_redeclared_as_scalar_param():
    source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "param dG_eq_fast__feq__A__B = 7.0\n"
        "init: A=1.0, B=0.0"
    )

    with pytest.raises(DSLError, match="reserved generated fast-equilibrium parameter"):
        extract_parameter_names_from_dsl(source)

    with pytest.raises(FitSimulationError, match="reserved generated fast-equilibrium parameter"):
        prepare_bound_mechanism(
            source,
            ["dG_eq_fast__feq__A__B"],
            initials={},
        )


def test_generated_fast_execution_request_override_is_worker_runtime_parameter():
    source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.0, "B": 1.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=source,
        parameter_overrides={"dG_eq_fast__feq__A__B": 2.5},
    )

    prepared = prepare_simulation_worker_run(execution_request=request)

    rate = np.asarray(prepared.request.rhs(0.0, prepared.request.y0), dtype=float)
    expected_keq = math.exp(-2500.0 / (8.31446261815324 * 298.15))
    expected_kr = 10.0 / expected_keq
    np.testing.assert_allclose(rate, [expected_kr, -expected_kr])

    bound = prepare_bound_mechanism(
        source,
        ["dG_eq_fast__feq__A__B"],
        initials={"A": 0.0, "B": 1.0},
    )
    prepared_request = SimulationExecutionRequest(
        prepared_payload=bound.as_serializable_execution_payload(),
        initials={"A": 0.0, "B": 1.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=source,
        parameter_overrides={"dG_eq_fast__feq__A__B": 3.0},
    )

    prepared_from_payload = prepare_simulation_worker_run(execution_request=prepared_request)

    prepared_rate = np.asarray(
        prepared_from_payload.request.rhs(0.0, prepared_from_payload.request.y0),
        dtype=float,
    )
    prepared_expected_keq = math.exp(-3000.0 / (8.31446261815324 * 298.15))
    prepared_expected_kr = 10.0 / prepared_expected_keq
    np.testing.assert_allclose(prepared_rate, [prepared_expected_kr, -prepared_expected_kr])

    cached_bound = prepare_bound_mechanism(
        source,
        [],
        initials={"A": 0.0, "B": 1.0},
    )
    version1_payload = dict(cached_bound.as_worker_payload())
    version1_payload["mechanism"] = cached_bound.mechanism.clone()
    version1_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.0, "B": 1.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=source,
        parameter_overrides={"dG_eq_fast__feq__A__B": 3.5},
    )

    prepared_from_version1_payload = prepare_simulation_worker_run(
        execution_request=version1_request,
        prepared_payload=version1_payload,
    )

    version1_rate = np.asarray(
        prepared_from_version1_payload.request.rhs(
            0.0,
            prepared_from_version1_payload.request.y0,
        ),
        dtype=float,
    )
    version1_expected_keq = math.exp(-3500.0 / (8.31446261815324 * 298.15))
    version1_expected_kr = 10.0 / version1_expected_keq
    np.testing.assert_allclose(version1_rate, [version1_expected_kr, -version1_expected_kr])


def test_runtime_keq_binding_is_authoritative_for_equilibrium_readouts():
    source = (
        "energy=kJ/mol\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0\n"
        "init: A=0.0, B=1.0"
    )
    desired_keq = math.exp(-3000.0 / (8.31446261815324 * 298.15))
    requested_kr = 10.0 / desired_keq
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.0, "B": 1.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=source,
        parameter_overrides={"kr1": requested_kr},
    )

    prepared = prepare_simulation_worker_run(execution_request=request)
    values = read_mechanism_parameter_values(prepared.mechanism, names={"Keq1", "kr1"})

    assert effective_equilibrium_keq(prepared.mechanism.equilibria[0]) == pytest.approx(desired_keq)
    assert values["Keq1"] == pytest.approx(desired_keq)
    assert values["kr1"] == pytest.approx(requested_kr)


def test_generated_fast_parameter_discovery_uses_generated_metadata_name():
    mixed_source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "equilibrium: C <-> D; kf=4.0; dG_eq=2.0\n"
        "init: A=1.0, B=0.0, C=1.0, D=0.0"
    )

    definitions = {param.name: param for param in extract_parameters_from_dsl(mixed_source)}
    names = extract_parameter_names_from_dsl(mixed_source)

    assert "dG_eq_fast__feq__A__B" in definitions
    assert definitions["dG_eq_fast__feq__A__B"].value == pytest.approx(1.0)
    assert definitions["dG_eq_fast__feq__A__B"].editable is True
    assert "dG_eq_fast__feq__A__B" in names
    assert "kf1" not in definitions
    assert "kr1" not in definitions
    assert "Keq1" not in definitions
    assert {"kf2", "kr2", "Keq2"}.issubset(definitions)


def test_generated_fast_parameter_name_preserves_exact_cm_id_identity():
    source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A + B <-> C; kf=10.0; dG_eq=1.0; cm_id=feq__A+B__C; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=1.0, B=1.0, C=0.0"
    )

    names = extract_parameter_names_from_dsl(source)

    assert "dG_eq_fast__feq__A+B__C" in names
    assert "dG_eq_fast__feq__A_B__C" not in names
    bound = prepare_bound_mechanism(
        source,
        ["dG_eq_fast__feq__A+B__C"],
        initials={},
    )
    assert "dG_eq_fast__feq__A+B__C" in bound.bindings


def test_generated_computational_mode_payload_rejects_duplicate_cm_id():
    duplicate = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        "equilibrium: A <-> B; kf=20.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=1.0, B=0.0"
    )

    with pytest.raises(DSLError, match="duplicate generated Computational Mode cm_id"):
        parse_dsl_to_mechanism(duplicate, initials={})


def test_computational_mode_generation_rejects_duplicate_fast_equilibrium_identity():
    spec = CompSpec(
        temperature_K=298.15,
        energy_unit="kJ/mol",
        std_default_M=1.0,
        species={
            "A": CompSpecies(name="A", kind="GS", G_value=0.0, std_M=1.0, cref_M=1.0),
            "B": CompSpecies(name="B", kind="GS", G_value=1.0, std_M=1.0, cref_M=1.0),
        },
        reactions=[
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=10.0),
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=20.0),
        ],
    )

    with pytest.raises(ValueError, match="duplicate generated fast-equilibrium cm_id"):
        compile_comp_spec(spec, output_energy_unit="kJ/mol")


def test_generated_fast_symbolic_jacobian_status_does_not_fabricate_indexed_parameters():
    source = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=1.0, B=0.0"
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})

    support = classify_symbolic_jacobian_support(mechanism)

    assert not support.supported
    assert support.code == "temperature-dependent-equilibrium"
    assert "kf1" not in support.reason
    assert "kr1" not in support.reason
    assert "Keq1" not in support.reason
