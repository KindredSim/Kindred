from __future__ import annotations

import numpy as np
import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAssignment,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism


pytestmark = pytest.mark.unit


def _spec(text: str, *, scalar_input_names=None):
    mechanism = parse_dsl_to_mechanism(text, initials={})
    return parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
        scalar_input_names=set(scalar_input_names or ()),
    )


def _assignment(expr_src: str, *, name: str = "Keq3") -> ParameterAssignment:
    return ParameterAssignment(
        name=name,
        expr_src=expr_src,
        line_number=1,
        line_content=f"param {name} = {expr_src}",
    )


def test_symbolic_state_context_keeps_species_display_separate_from_internal_state_symbols():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure
    from kindred.core.symbolic.namespaces import make_state_symbol_context

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=2.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        initials={},
    )

    context = make_state_symbol_context(mechanism.species_names())
    structure = build_symbolic_jacobian_structure(mechanism)
    artifact = structure.bind({"k1": 2.0})

    assert context.kind == "state-vector"
    assert context.display_symbols == ("[A]", "[B]")
    assert context.symbol_names == ("y_0", "y_1")
    assert structure.state_symbol_context == context.to_payload()
    assert artifact.state_symbol_context == context.to_payload()
    np.testing.assert_allclose(artifact.jacobian_func(0.0, np.asarray([1.0, 0.0])), [[-2.0, 0.0], [2.0, 0.0]])


def test_parameter_expression_records_parameter_namespace_and_rejects_state_symbols():
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.namespaces import make_parameter_namespace_context
    from kindred.core.symbolic.parameter_expression import translate_parameter_expression

    spec = _spec(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=4",
                "param Keq3 = 1 / (Keq1 * Keq2)",
                "init: A=1, B=0, C=0",
            ]
        )
    )

    namespace = make_parameter_namespace_context(spec)
    translated = translate_parameter_expression(_assignment("1 / (Keq1 * Keq2)"), namespace=namespace)

    assert translated.symbol_context["kind"] == "parameter-expression"
    assert translated.symbol_context["allows_state_symbols"] is False
    assert translated.symbol_context["canonical_identifiers"] == ["Keq1", "Keq2"]
    assert translated.symbol_context["mechanism_parameters"] == ["Keq1", "Keq2", "Keq3", "kf1", "kf2", "kf3", "kr1", "kr2", "kr3"]
    with pytest.raises(UnsupportedSymbolicExpressionError, match="State concentration symbols"):
        translate_parameter_expression(_assignment("[A]"), namespace=namespace)


def test_symbolic_parameter_namespace_uses_direct_casefold_policy_only():
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.namespaces import make_parameter_namespace_context

    spec = _spec(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "equilibrium: B <-> C ; kf=1 ; kr=0.5",
                "init: A=1, B=0, C=0",
            ]
        )
    )
    namespace = make_parameter_namespace_context(spec)

    assert namespace.resolve_identifier("k1") == "k1"
    assert namespace.resolve_identifier("K1") == "k1"
    assert namespace.resolve_identifier("KF2") == "kf2"
    assert namespace.resolve_identifier("KR2") == "kr2"
    assert namespace.resolve_identifier("KEQ2") == "Keq2"
    with pytest.raises(UnsupportedSymbolicExpressionError, match="Protected indexed mechanism parameter 'K2'"):
        namespace.resolve_identifier("K2")


def test_symbolic_parameter_namespace_resolves_K1_through_mechanism_before_scalar_input():
    from kindred.core.symbolic.namespaces import make_parameter_namespace_context

    spec = _spec(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1, B=0",
            ]
        ),
        scalar_input_names={"K1"},
    )
    namespace = make_parameter_namespace_context(spec)

    assert namespace.resolve_identifier("K1") == "k1"


def test_symbolic_jacobian_canonicalizes_explicit_indexed_k_binding_name_for_irreversible_step():
    from dataclasses import replace

    from kindred.core.rate_binding import RateBinding
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1, B=0",
            ]
        ),
        initials={},
    )
    mechanism.reactions[0] = replace(mechanism.reactions[0], rate=RateBinding(name="K1", value=2.0))

    structure = build_symbolic_jacobian_structure(mechanism)

    assert structure.parameter_symbols == ("k1",)


def test_symbolic_jacobian_rejects_explicit_indexed_k_binding_name_for_reversible_step():
    from dataclasses import replace

    from kindred.core.rate_binding import RateBinding
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "equilibrium: A <-> B; kf=1.0; kr=0.5",
                "init: A=1, B=0",
            ]
        ),
        initials={},
    )
    mechanism.equilibria[0] = replace(mechanism.equilibria[0], kf=RateBinding(name="K1", value=1.0))

    with pytest.raises(UnsupportedSymbolicExpressionError, match="K1.*not a valid indexed parameter identifier"):
        build_symbolic_jacobian_structure(mechanism)


def test_wegscheider_proof_records_parameter_only_proof_namespace():
    from kindred.core.symbolic.namespaces import make_product_identity_proof_context
    from kindred.core.symbolic.proof import prove_product_identity

    spec = _spec(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=4",
                "param Keq3 = 1 / (Keq1 * Keq2)",
                "init: A=1, B=0, C=0",
            ]
        )
    )

    target_factors = {"Keq1": 1, "Keq2": 1, "Keq3": 1}
    proof_context = make_product_identity_proof_context(
        target_factors=target_factors,
        spec=spec,
    )

    result = prove_product_identity(
        target_factors=target_factors,
        candidate=_assignment("1 / (Keq1 * Keq2)"),
        proof_context=proof_context,
    )

    assert result.proven is True
    assert result.symbol_context["kind"] == "wegscheider-parameter-proof"
    assert result.symbol_context["allows_state_symbols"] is False
    assert result.symbol_context["proof_symbols"] == ["Keq1", "Keq2", "Keq3"]
    assert result.symbol_context["assignment_names"] == ["Keq3"]


def test_symbolic_proof_context_is_immutable_after_spec_construction():
    from dataclasses import replace

    from kindred.core.symbolic.namespaces import make_product_identity_proof_context
    from kindred.core.symbolic.proof import prove_product_identity

    spec = _spec(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=4",
                "param inv = 1 / (Keq1 * Keq2)",
                "param Keq3 = inv",
                "init: A=1, B=0, C=0",
            ]
        )
    )
    target_factors = {"Keq1": 1, "Keq2": 1, "Keq3": 1}
    proof_context = make_product_identity_proof_context(target_factors=target_factors, spec=spec)
    mutated_spec = replace(spec, param_statements=[])

    result = prove_product_identity(
        target_factors=target_factors,
        candidate=_assignment("inv", name="Keq3"),
        proof_context=proof_context,
        spec=mutated_spec,
    )

    assert result.proven is True
    assert result.symbol_context["assignment_names"] == ["Keq3", "inv"]


def test_symbolic_namespace_contexts_do_not_expose_mutable_mapping_state():
    from kindred.core.symbolic.namespaces import (
        make_parameter_namespace_context,
        make_product_identity_proof_context,
    )

    spec = _spec(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "param derived = k1 * 2",
                "init: A=1.0, B=0.0",
            ]
        )
    )

    namespace = make_parameter_namespace_context(spec)
    proof_context = make_product_identity_proof_context(target_factors={"k1": 1}, spec=spec)

    with pytest.raises(TypeError):
        namespace.canonical_by_lower["bad"] = "bad"
    with pytest.raises(TypeError):
        proof_context.assignments["bad"] = _assignment("k1", name="bad")


def test_unsupported_symbolic_jacobian_status_reaches_prepared_request_and_solver_provenance():
    from kindred.core.simulation_preparation import prepare_simulation_worker_run
    from kindred.core.simulator.solvers import solve_ode

    prepared = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "equilibrium: A <-> B; kf=2.0; dG_eq=0",
                "init: A=1.0, B=0.0",
            ]
        ),
        initials={},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "use_sparse_jacobian": True, "grid": {"N": 5}},
    )

    assert prepared.request.jacobian_func is None
    assert prepared.request.jac_sparsity is None
    assert prepared.request.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "temperature-dependent-equilibrium",
        "reason": "Temperature-dependent equilibrium models are outside symbolic Jacobian support.",
    }

    result = solve_ode(prepared.request)

    assert result.provenance["symbolic_jacobian"] is False
    assert "symbolic_jacobian_identity" not in result.provenance
    assert result.provenance["symbolic_jacobian_status"] == prepared.request.symbolic_jacobian_status


def test_fitting_records_unsupported_symbolic_status_without_blocking_numeric_execution():
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "equilibrium: A <-> B; kf=2.0; dG_eq=0",
                "init: A=1.0, B=0.0",
            ]
        ),
        param_names=["kf1"],
        t_end=0.2,
        num_points=4,
        solver="BDF",
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
    )
    evaluator = SerialFittingEvaluator(context)

    result = evaluator({"kf1": 2.0})

    assert np.asarray(result.t, dtype=float).shape == (4,)
    assert evaluator.prepared_metadata.symbolic_jacobian_identity is None
    assert evaluator.prepared_metadata.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "temperature-dependent-equilibrium",
        "reason": "Temperature-dependent equilibrium models are outside symbolic Jacobian support.",
    }
    payload = evaluator.to_process_payload()
    assert payload["prepared_metadata"]["symbolic_jacobian_status"] == evaluator.prepared_metadata.symbolic_jacobian_status
    restored = SerialFittingEvaluator.from_process_payload(payload)
    assert restored.prepared_metadata.symbolic_jacobian_status == evaluator.prepared_metadata.symbolic_jacobian_status


def test_wegscheider_unresolved_policy_remains_hard_gate_when_validation_enabled():
    from kindred.core.simulation_preparation import SimulationPreparationError, prepare_simulation_worker_run

    with pytest.raises(SimulationPreparationError) as excinfo:
        prepare_simulation_worker_run(
            mechanism_text="\n".join(
                [
                    "equilibrium: A <-> B ; kf=1 ; K=2",
                    "equilibrium: B <-> C ; kf=1 ; K=3",
                    "equilibrium: C <-> A ; kf=1 ; K=0.16666666666666666",
                    "init: A=1, B=0, C=0",
                ]
            ),
            initials={},
            t_span=(0.0, 0.1),
            solver_config={"solver": "BDF", "wegscheider_cyclicity_enabled": True},
        )

    assert excinfo.value.stage == "wegscheider_cyclicity"


def test_gui_provenance_copies_symbolic_status_without_adding_ui_controls():
    from kindred.gui.simulation_provenance_owner import SimulationProvenanceOwner

    owner = SimulationProvenanceOwner(
        dataset_snapshot_getter=lambda: {},
        fit_metadata_getter=lambda: None,
    )
    status = {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "temperature-dependent-equilibrium",
        "reason": "Temperature-dependent equilibrium models are outside symbolic Jacobian support.",
    }

    provenance = owner.publish_simulation_completion_provenance(
        mechanism_text="equilibrium: A <-> B; kf=2.0; dG_eq=0",
        solver_method="BDF",
        solver_label="BDF",
        solver_warning=None,
        solver_config={"rtol": 1e-6, "atol": 1e-12},
        temperature_K=298.15,
        temperature_source="mechanism",
        energy_unit=None,
        energy_mode=False,
        simulation_time=0.1,
        num_points_requested=2,
        species_names=["A", "B"],
        t=np.asarray([0.0, 0.1], dtype=float),
        series={"A": np.asarray([1.0, 0.9]), "B": np.asarray([0.0, 0.1])},
        solver_provenance={
            "symbolic_jacobian": False,
            "symbolic_jacobian_status": status,
        },
    )

    assert provenance["symbolic_jacobian"] is False
    assert provenance["symbolic_jacobian_status"] == status
    assert provenance["solver_provenance"]["symbolic_jacobian_status"] == status
