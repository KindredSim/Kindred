import pytest
import numpy as np

from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    SimulationPreparationError,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
)
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_to_mechanism,
    read_mechanism_parameter_values,
)

pytestmark = pytest.mark.unit


PBM_STOICHIOMETRIC_CYCLE = "\n".join(
    [
        "equilibrium: PBMproduct <-> Methidequinone + Amine ; kf=1 ; K=2",
        "equilibrium: Methidequinone <-> Methidequinone_CIS ; kf=1 ; K=3",
        "equilibrium: Methidequinone_CIS + Amine <-> PBMproduct ; kf=1 ; K=7",
        "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
    ]
)


PBM_SYMBOLICALLY_RESOLVED = "\n".join(
    [
        "equilibrium: PBMproduct <-> Methidequinone + Amine ; kf=1 ; K=2",
        "equilibrium: Methidequinone <-> Methidequinone_CIS ; kf=1 ; K=3",
        "equilibrium: Methidequinone_CIS + Amine <-> PBMproduct ; kf=1 ; K=7",
        "param Keq3 = 1 / (Keq1 * Keq2)",
        "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
    ]
)


PBM_NUMERICALLY_CONSISTENT_BUT_UNRESOLVED = "\n".join(
    [
        "equilibrium: PBMproduct <-> Methidequinone + Amine ; kf=1 ; K=2",
        "equilibrium: Methidequinone <-> Methidequinone_CIS ; kf=1 ; K=3",
        "equilibrium: Methidequinone_CIS + Amine <-> PBMproduct ; kf=1 ; K=0.16666666666666666",
        "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
    ]
)


TRIANGLE_SYMBOLICALLY_RESOLVED = "\n".join(
    [
        "equilibrium: A <-> B ; kf=2 ; K=2",
        "equilibrium: B <-> C ; kf=3 ; K=3",
        "equilibrium: C <-> A ; kf=1 ; K=1",
        "param Keq3 = 1 / (Keq1 * Keq2)",
        "init: A=1, B=0, C=0",
    ]
)


NON_UNIT_CYCLE = "\n".join(
    [
        "equilibrium: A <-> B ; kf=1 ; K=2",
        "equilibrium: 2 B <-> C ; kf=1 ; K=3",
        "equilibrium: C <-> 2 A ; kf=1 ; K=5",
        "init: A=1, B=0, C=0",
    ]
)


FRACTIONAL_NON_CYCLE = "\n".join(
    [
        "equilibrium: C <-> B ; kf=1 ; K=2",
        "equilibrium: B <-> 0.5 A + 1.5 C ; kf=1 ; K=3",
        "init: A=0, B=0, C=1",
    ]
)


def _enabled_mechanism(text: str):
    mechanism = parse_dsl_to_mechanism(text, initials={})
    mechanism.metadata["wegscheider_cyclicity_enabled"] = True
    return mechanism


def _assert_unresolved(text: str) -> None:
    from kindred.core.simulator.wegscheider_symbolic import (
        UnresolvedWegscheiderCyclicityError,
        validate_wegscheider_cyclicity_resolved,
    )

    with pytest.raises(UnresolvedWegscheiderCyclicityError) as excinfo:
        validate_wegscheider_cyclicity_resolved(_enabled_mechanism(text))
    assert excinfo.value.stage == "wegscheider_cyclicity"
    assert excinfo.value.cycles


def test_stoichiometric_cycle_detects_pbm_cycle_that_complex_graph_misses():
    from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_cyclicity

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(PBM_STOICHIOMETRIC_CYCLE))

    assert [cycle.step_indices for cycle in report.cycles] == [(1, 2, 3)]
    assert [cycle.coefficients for cycle in report.cycles] == [(1, 1, 1)]
    assert report.cycles[0].parameter_names == ("Keq1", "Keq2", "Keq3")
    assert not report.is_resolved


def test_numeric_coincidence_does_not_resolve_stoichiometric_cycle():
    _assert_unresolved(PBM_NUMERICALLY_CONSISTENT_BUT_UNRESOLVED)


def test_probe_coincidence_expression_does_not_resolve_stoichiometric_cycle():
    crafted_probe_match = "\n".join(
        [
            "equilibrium: PBMproduct <-> Methidequinone + Amine ; kf=1 ; K=2",
            "equilibrium: Methidequinone <-> Methidequinone_CIS ; kf=1 ; K=3",
            "equilibrium: Methidequinone_CIS + Amine <-> PBMproduct ; kf=1 ; K=7",
            "param Keq3 = 1 / (Keq1 * Keq2) + (Keq1 - 2) * (Keq1 - 1.5) * (Keq1 - 8) * 0.001",
            "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
        ]
    )

    _assert_unresolved(crafted_probe_match)


def test_numeric_inconsistency_is_not_hidden_by_rate_mutation():
    _assert_unresolved(PBM_STOICHIOMETRIC_CYCLE)


def test_simple_triangle_uses_stoichiometric_nullspace_contract():
    from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_cyclicity

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(TRIANGLE_SYMBOLICALLY_RESOLVED))

    assert [cycle.step_indices for cycle in report.cycles] == [(1, 2, 3)]
    assert [cycle.coefficients for cycle in report.cycles] == [(1, 1, 1)]
    assert report.is_resolved
    assert report.cycles[0].resolved_by == "Keq3"


def test_symbolic_wegscheider_report_records_proof_identity():
    from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_cyclicity

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(TRIANGLE_SYMBOLICALLY_RESOLVED))
    cycle = report.cycles[0]

    assert cycle.resolved_by == "Keq3"
    assert cycle.resolved_proof_fingerprint
    assert report.symbolic_identity["kind"] == "wegscheider_cyclicity"
    assert report.symbolic_identity["fingerprint"]
    assert report.symbolic_identity["cycles"][0]["proof_fingerprint"] == cycle.resolved_proof_fingerprint


def test_fractional_stoichiometry_is_not_truncated_into_false_cycle():
    from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_cyclicity

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(FRACTIONAL_NON_CYCLE))

    assert report.cycles == ()


def test_transitive_symbolic_keq_dependency_resolves_cycle():
    from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_cyclicity

    text = TRIANGLE_SYMBOLICALLY_RESOLVED.replace(
        "param Keq3 = 1 / (Keq1 * Keq2)",
        "param inv = 1e0 / (Keq1 * Keq2)\nparam Keq3 = inv",
    )
    report = analyze_wegscheider_cyclicity(_enabled_mechanism(text))

    assert report.is_resolved
    assert report.cycles[0].resolved_by == "Keq3"


def test_symbolic_keq_dependency_resolves_cycle_and_recomputes_derived_rate():
    mechanism = _enabled_mechanism(PBM_SYMBOLICALLY_RESOLVED)

    derived = apply_parameter_algebra_to_mechanism(
        PBM_SYMBOLICALLY_RESOLVED,
        mechanism=mechanism,
        require_mutable=False,
    )
    values = read_mechanism_parameter_values(mechanism, names={"Keq1", "Keq2", "Keq3", "kr3"})

    assert derived["Keq3"] == pytest.approx(1.0 / 6.0)
    assert values["Keq3"] == pytest.approx(1.0 / 6.0)
    assert values["kr3"] == pytest.approx(6.0)
    assert mechanism.metadata["constrained_params"]["Keq3"]["constraint_reason"] == "algebra"
    assert "kr3" not in mechanism.metadata["constrained_params"]


def test_prepared_bound_mechanism_recomputes_dependent_keq_without_reparse():
    bound = prepare_bound_mechanism(
        TRIANGLE_SYMBOLICALLY_RESOLVED,
        ["Keq1", "Keq2", "kf3"],
        wegscheider_cyclicity_enabled=True,
    )

    assert "Keq3" not in bound.bindings
    assert "Keq1" in bound.bindings
    assert "Keq2" in bound.bindings

    bound.bindings["Keq1"].set(4.0)
    apply_parameter_algebra_to_mechanism(
        TRIANGLE_SYMBOLICALLY_RESOLVED,
        mechanism=bound.mechanism,
        require_mutable=True,
    )
    values = read_mechanism_parameter_values(bound.mechanism, names={"Keq3", "kr3"})

    assert values["Keq3"] == pytest.approx(1.0 / 12.0)
    assert values["kr3"] == pytest.approx(12.0)


def test_source_resolution_writes_durable_param_line_and_reparse_resolves():
    from kindred.core.simulator.wegscheider_symbolic import (
        analyze_wegscheider_cyclicity,
        apply_wegscheider_resolution_to_source,
        build_wegscheider_resolution_updates,
    )

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(PBM_STOICHIOMETRIC_CYCLE))
    updates = build_wegscheider_resolution_updates(report, {"cycle_1": "Keq3"})
    updated = apply_wegscheider_resolution_to_source(PBM_STOICHIOMETRIC_CYCLE, updates)

    assert "param Keq3 = 1 / (Keq1 * Keq2)" in updated
    reparsed_report = analyze_wegscheider_cyclicity(_enabled_mechanism(updated))
    assert reparsed_report.is_resolved
    assert reparsed_report.cycles[0].resolved_by == "Keq3"


def test_source_resolution_replaces_existing_bad_param_line_instead_of_duplicating():
    from kindred.core.simulator.parameter_algebra import parse_parameter_algebra_spec_from_dsl_text
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
    from kindred.core.simulator.wegscheider_symbolic import (
        analyze_wegscheider_cyclicity,
        apply_wegscheider_resolution_to_source,
        build_wegscheider_resolution_updates,
    )

    bad_source = PBM_STOICHIOMETRIC_CYCLE.replace(
        "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
        "param Keq3 = Keq1\ninit: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
    )
    report = analyze_wegscheider_cyclicity(_enabled_mechanism(PBM_STOICHIOMETRIC_CYCLE))
    updates = build_wegscheider_resolution_updates(report, {"cycle_1": "Keq3"})

    updated = apply_wegscheider_resolution_to_source(bad_source, updates)

    assert updated.count("param Keq3 =") == 1
    assert "param Keq3 = 1 / (Keq1 * Keq2)" in updated
    reparsed_mechanism = _enabled_mechanism(updated)
    parse_parameter_algebra_spec_from_dsl_text(
        updated,
        mechanism_namespace=build_namespace_from_mechanism(reparsed_mechanism),
    )
    reparsed_report = analyze_wegscheider_cyclicity(reparsed_mechanism)
    assert reparsed_report.is_resolved


def test_source_resolution_handles_reciprocal_orientation_and_non_unit_coefficients():
    from kindred.core.simulator.wegscheider_symbolic import (
        analyze_wegscheider_cyclicity,
        build_wegscheider_resolution_updates,
    )

    report = analyze_wegscheider_cyclicity(_enabled_mechanism(NON_UNIT_CYCLE))
    updates = build_wegscheider_resolution_updates(report, {"cycle_1": "Keq3"})

    assert report.cycles[0].coefficients == (2, 1, 1)
    assert updates[0].parameter_name == "Keq3"
    assert updates[0].expr_src == "1 / (Keq1**2 * Keq2)"


def test_source_resolution_rejects_duplicate_dependent_parameter_choices():
    from kindred.core.simulator.wegscheider_symbolic import (
        WegscheiderCycle,
        WegscheiderCyclicityReport,
        build_wegscheider_resolution_updates,
    )

    report = WegscheiderCyclicityReport(
        cycles=(
            WegscheiderCycle(
                cycle_id="cycle_1",
                step_indices=(1, 2, 3),
                equilibrium_indices=(0, 1, 2),
                coefficients=(1, 1, 1),
                parameter_names=("Keq1", "Keq2", "Keq3"),
            ),
            WegscheiderCycle(
                cycle_id="cycle_2",
                step_indices=(3, 4, 5),
                equilibrium_indices=(2, 3, 4),
                coefficients=(1, 1, 1),
                parameter_names=("Keq3", "Keq4", "Keq5"),
            ),
        )
    )

    with pytest.raises(ValueError, match="selected for both"):
        build_wegscheider_resolution_updates(report, {"cycle_1": "Keq3", "cycle_2": "Keq3"})


def test_worker_preparation_reports_unresolved_cyclicity_as_own_stage():
    with pytest.raises(SimulationPreparationError) as excinfo:
        prepare_simulation_worker_run(
            mechanism_text=PBM_STOICHIOMETRIC_CYCLE,
            initials={},
            t_span=(0.0, 1.0),
            solver_config={"grid": {"N": 3}, "wegscheider_cyclicity_enabled": True},
        )

    assert excinfo.value.stage == "wegscheider_cyclicity"
    assert "Keq3" in str(excinfo.value)


def test_prepared_solver_provenance_includes_symbolic_wegscheider_identity():
    from kindred.core.simulator.solvers import solve_ode

    prepared = prepare_simulation_worker_run(
        mechanism_text=TRIANGLE_SYMBOLICALLY_RESOLVED,
        initials={},
        t_span=(0.0, 0.2),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "wegscheider_cyclicity_enabled": True,
        },
    )

    result = solve_ode(prepared.request)

    identity = result.provenance["symbolic_wegscheider_identity"]
    assert identity["kind"] == "wegscheider_cyclicity"
    assert identity["cycles"][0]["resolved_by"] == "Keq3"


def test_structured_prepared_worker_without_spec_rejects_unresolved_cyclicity():
    mechanism = _enabled_mechanism(PBM_STOICHIOMETRIC_CYCLE)

    with pytest.raises(SimulationPreparationError) as excinfo:
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload={
                    "version": 2,
                    "mechanism": mechanism,
                    "species_names": list(mechanism.species_names()),
                    "y0": np.asarray([mechanism.species[sp].initial_conc for sp in mechanism.species_names()]),
                    "mechanism_text": "",
                    "temperature_schedule": None,
                    "jacobian_func": None,
                },
                initials={},
                t_span=(0.0, 1.0),
                solver_config={"grid": {"N": 3}, "wegscheider_cyclicity_enabled": True},
                mechanism_text="",
                simulation_identity={"schema_id": "schema", "param_fingerprint": "fingerprint"},
            )
        )

    assert excinfo.value.stage == "wegscheider_cyclicity"
    assert "Keq3" in str(excinfo.value)


def test_prepared_bound_mechanism_rejects_unresolved_cyclicity_as_own_stage():
    from kindred.core.exceptions import FitSimulationError

    with pytest.raises(FitSimulationError, match="wegscheider_cyclicity"):
        prepare_bound_mechanism(
            PBM_STOICHIOMETRIC_CYCLE,
            ["Keq1", "Keq2", "Keq3"],
            wegscheider_cyclicity_enabled=True,
        )
