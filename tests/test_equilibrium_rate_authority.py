from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kindred.core.cache import generate_mechanism_hash
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateInputContext,
    step_entry_role_editable,
)
from kindred.core.mechanism import Mechanism
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataKeys
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.rate_binding import RateBinding
from kindred.core.simulation_preparation import (
    SimulationPreparationError,
    apply_parameter_overrides_to_prepared_mechanism,
    partition_simulation_parameter_values,
    prepare_bound_mechanism,
)
from kindred.core.algebra.symbol_table import build_algebra_symbol_table
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.dsl_parameter_scan import extract_parameters_from_dsl
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism, read_mechanism_parameter_values
from kindred.gui.parameter_enumeration import enumerate_step_parameters_for_gui


pytestmark = pytest.mark.unit


def _metadata_authority_mechanism(dg_eq_j_per_mol: float) -> Mechanism:
    mechanism = parse_dsl_to_mechanism(
        "energy=J/mol\n"
        f"equilibrium: A <-> B; kf=2.0; dG_eq={float(dg_eq_j_per_mol):.17g}\n"
        "init: A=0.0, B=1.0",
        initials={},
    )
    return mechanism


def test_equilibrium_authority_metadata_participates_in_hash_and_serialization_identity():
    mechanism_a = _metadata_authority_mechanism(0.0)
    mechanism_b = _metadata_authority_mechanism(1000.0)

    rhs_a = build_ode_rhs_from_mechanism(mechanism_a)
    rhs_b = build_ode_rhs_from_mechanism(mechanism_b)

    assert generate_mechanism_hash(mechanism_a) != generate_mechanism_hash(mechanism_b)
    assert mechanism_a.to_serializable()["equilibria"] != mechanism_b.to_serializable()["equilibria"]
    assert not np.allclose(
        rhs_a(0.0, np.asarray([0.0, 1.0], dtype=float)),
        rhs_b(0.0, np.asarray([0.0, 1.0], dtype=float)),
    )


def test_public_equilibrium_rejects_explicit_reverse_rate_plus_dg_metadata_authority():
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)

    with pytest.raises(ValueError, match="kf.*exactly one"):
        mechanism.add_equilibrium(
            stoich_forward={"A": 1.0},
            stoich_back={"B": 1.0},
            kf=2.0,
            kr=0.5,
            metadata={
                EquilibriumMetadataKeys.USER_PROVIDED_KF: True,
                EquilibriumMetadataKeys.USER_PROVIDED_KR: True,
                EquilibriumMetadataKeys.DG_EQ_J_PER_MOL: 0.0,
            },
        )


def test_public_equilibrium_accepts_kf_plus_dg_metadata_authority_for_runtime():
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=2.0,
        metadata={EquilibriumMetadataKeys.DG_EQ_J_PER_MOL: 0.0},
    )
    rhs = build_ode_rhs_from_mechanism(mechanism)

    np.testing.assert_allclose(rhs(0.0, np.asarray([1.0, 0.0], dtype=float)), [-2.0, 2.0])


def test_public_dg_equilibrium_exposes_effective_reverse_rate_consistently():
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        "equilibrium: A <-> B; kf=2.0; dG_eq=0.0\n"
        "init: A=0.0, B=1.0\n"
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})
    eq = mechanism.equilibria[0]

    assert float(eq.Keq) == pytest.approx(1.0)
    assert float(eq.kr) == pytest.approx(2.0)
    assert mechanism.to_serializable()["equilibria"][0]["kr"] == pytest.approx(2.0)

    gui_variables, _ = enumerate_step_parameters_for_gui(mechanism)
    scan_parameters = {param.name: param for param in extract_parameters_from_dsl(source)}
    algebra_values = read_mechanism_parameter_values(mechanism, names={"kr1", "Keq1"})

    assert gui_variables["kr1"] == pytest.approx(2.0)
    assert scan_parameters["kr1"].value == pytest.approx(2.0)
    assert algebra_values["kr1"] == pytest.approx(2.0)
    assert algebra_values["Keq1"] == pytest.approx(1.0)


def test_direct_public_dg_equilibrium_exposes_effective_reverse_rate_consistently():
    mechanism = Mechanism()
    mechanism.metadata[MechanismMetadataKeys.TEMPERATURE_K] = 298.15
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 1.0)

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=2.0,
        metadata={EquilibriumMetadataKeys.DG_EQ_J_PER_MOL: 0.0},
    )
    eq = mechanism.equilibria[0]

    assert float(eq.Keq) == pytest.approx(1.0)
    assert float(eq.kr) == pytest.approx(2.0)
    assert mechanism.to_serializable()["equilibria"][0]["kr"] == pytest.approx(2.0)

    gui_variables, gui_metadata = enumerate_step_parameters_for_gui(mechanism)
    algebra_values = read_mechanism_parameter_values(mechanism, names={"kr1", "Keq1"})
    symtab = build_algebra_symbol_table(mechanism)

    assert gui_variables["kr1"] == pytest.approx(2.0)
    assert gui_variables["Keq1"] == pytest.approx(1.0)
    assert gui_metadata["kr1"]["editable"] is False
    assert algebra_values["kr1"] == pytest.approx(2.0)
    assert algebra_values["Keq1"] == pytest.approx(1.0)
    assert symtab.get("kr1") == pytest.approx(2.0)
    assert symtab.get("Keq1") == pytest.approx(1.0)


def test_dg_authority_rejects_parameter_algebra_keq_override():
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        "equilibrium: A <-> B; kf=2.0; dG_eq=0.0\n"
        "param Keq1 = 4.0\n"
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})

    with pytest.raises(ValueError, match="Keq1.*not editable"):
        apply_parameter_algebra_to_mechanism(source, mechanism=mechanism, require_mutable=False)


@pytest.mark.parametrize(
    ("source", "param_name"),
    [
        (
            "T=298.15\n"
            "energy=J/mol\n"
            "equilibrium: A <-> B; kf=6.0; dG_eq=0.0\n"
            "init: A=0.0, B=1.0\n",
            "kr1",
        ),
        (
            "equilibrium: A <-> B; kf=6.0; Keq=3.0\n"
            "init: A=0.0, B=1.0\n",
            "kr1",
        ),
        (
            "equilibrium: A <-> B; kf=6.0; kr=2.0\n"
            "init: A=0.0, B=1.0\n",
            "Keq1",
        ),
    ],
)
def test_parameter_algebra_rejects_noneditable_equilibrium_roles_before_mutation(source, param_name):
    mechanism = parse_dsl_to_mechanism(f"{source}param {param_name} = 99.0\n", initials={})
    before_raw = mechanism.equilibria[0]
    before_values = read_mechanism_parameter_values(mechanism, names={"kf1", "kr1", "Keq1"})

    with pytest.raises(ValueError, match=rf"{param_name}.*not editable"):
        apply_parameter_algebra_to_mechanism(
            f"{source}param {param_name} = 99.0\n",
            mechanism=mechanism,
            require_mutable=False,
        )

    after_raw = mechanism.equilibria[0]
    assert after_raw.kf == before_raw.kf
    assert after_raw.kr == before_raw.kr
    assert after_raw.Keq == before_raw.Keq
    assert dict(after_raw.metadata) == dict(before_raw.metadata)
    assert read_mechanism_parameter_values(mechanism, names={"kf1", "kr1", "Keq1"}) == before_values


def test_parameter_algebra_accepts_editable_equilibrium_roles():
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        "equilibrium: A <-> B; kf=6.0; dG_eq=0.0\n"
        "init: A=0.0, B=1.0\n"
        "param kf1 = 12.0\n"
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})

    applied = apply_parameter_algebra_to_mechanism(source, mechanism=mechanism, require_mutable=False)

    assert applied == {"kf1": pytest.approx(12.0)}
    assert read_mechanism_parameter_values(mechanism, names={"kf1", "kr1", "Keq1"}) == {
        "kf1": pytest.approx(12.0),
        "kr1": pytest.approx(12.0),
        "Keq1": pytest.approx(1.0),
    }


def test_prepared_runtime_rejects_noneditable_derived_equilibrium_override():
    source = "equilibrium: A <-> B; kf=2.0; Keq=4.0\ninit: A=0.0, B=1.0"
    bound = prepare_bound_mechanism(
        source,
        ["kf1"],
        wegscheider_cyclicity_enabled=False,
    )
    before_raw = bound.mechanism.equilibria[0].kr

    with pytest.raises(SimulationPreparationError, match="kr1.*not editable"):
        partition = partition_simulation_parameter_values(
            mechanism=bound.mechanism,
            parameter_overrides={"kr1": 99.0},
            unresolved_intervention_schedule=None,
        )
        apply_parameter_overrides_to_prepared_mechanism(
            bound.mechanism,
            parameter_partition=partition,
        )

    assert bound.mechanism.equilibria[0].kr == before_raw


def test_programmatic_dynamic_thermodynamic_authority_stays_lazy_for_rhs():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 1.0)
    kf_binding = RateBinding("kf1", 2.0)
    keq_binding = RateBinding("Keq1", 4.0)

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=kf_binding,
        Keq=keq_binding,
    )
    eq = mechanism.equilibria[0]
    assert eq.kr is None

    rhs = build_ode_rhs_from_mechanism(mechanism)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [0.5, -0.5])

    keq_binding.set(2.0)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [1.0, -1.0])

    kf_binding.set(6.0)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [3.0, -3.0])


def test_programmatic_callable_thermodynamic_authority_stays_lazy_for_rhs():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 1.0)
    values = {"kf": 2.0, "Keq": 4.0}

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=lambda: values["kf"],
        Keq=lambda: values["Keq"],
    )
    assert mechanism.equilibria[0].kr is None

    rhs = build_ode_rhs_from_mechanism(mechanism)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [0.5, -0.5])

    values["Keq"] = 2.0
    values["kf"] = 6.0
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [3.0, -3.0])


def test_serialization_uses_effective_authority_values_not_stale_raw_derived_slots():
    mechanism = _metadata_authority_mechanism(0.0)
    mechanism.equilibria[0] = replace(mechanism.equilibria[0], kr=999.0)

    serialized = mechanism.to_serializable()["equilibria"][0]

    assert serialized["kr"] == pytest.approx(2.0)
    assert serialized["Keq"] == pytest.approx(1.0)


def test_cache_identity_ignores_equilibrium_display_metadata():
    mechanism = _metadata_authority_mechanism(0.0)
    original_hash = generate_mechanism_hash(mechanism)

    mechanism.metadata["step_index_map"][0]["equilibrium_authority"]["editable"]["kr"] = True
    mechanism.metadata["step_index_map"][0]["equilibrium_authority"]["derived"]["kr"] = False

    assert generate_mechanism_hash(mechanism) == original_hash


def test_algebra_symbol_table_uses_dg_authority_after_kf_algebra_update():
    source = (
        "T=298.15\n"
        "energy=J/mol\n"
        "equilibrium: A <-> B; kf=6.0; dG_eq=0.0\n"
        "param kf1 = 12.0\n"
    )
    mechanism = parse_dsl_to_mechanism(source, initials={})
    apply_parameter_algebra_to_mechanism(source, mechanism=mechanism, require_mutable=False)

    symtab = build_algebra_symbol_table(mechanism)

    assert symtab.get("kf1") == pytest.approx(12.0)
    assert symtab.get("Keq1") == pytest.approx(1.0)
    assert symtab.get("kr1") == pytest.approx(12.0)


def test_public_api_cannot_use_internal_authority_context_to_accept_all_three():
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)

    with pytest.raises(TypeError):
        mechanism.add_equilibrium(
            stoich_forward={"A": 1.0},
            stoich_back={"B": 1.0},
            kf=1.0,
            kr=2.0,
            Keq=3.0,
            metadata={
                EquilibriumMetadataKeys.USER_PROVIDED_KF: True,
                EquilibriumMetadataKeys.USER_PROVIDED_KR: False,
            },
            authority_context=EquilibriumRateInputContext.NORMALIZED_PUBLIC,
        )


def test_direct_api_records_normalized_authority_snapshot_for_step_index():
    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=2.0,
        Keq=4.0,
    )

    entry = mechanism.metadata["step_index_map"][0]
    assert step_entry_role_editable(entry, "Keq") is True
    assert step_entry_role_editable(entry, "kr") is False
