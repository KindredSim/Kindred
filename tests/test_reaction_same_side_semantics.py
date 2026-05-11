from __future__ import annotations

import json

import numpy as np
import pytest

from kindred.core import batch_parallel
from kindred.core.cache import generate_mechanism_hash
from kindred.core.fitting_evaluation import (
    SerialFittingEvaluator,
    prepare_fitting_execution_context,
)
from kindred.core.mechanism import Mechanism, Reaction
from kindred.core.mechanism_metadata import EquilibriumMetadataView
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulation_preparation import (
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
    prepared_simulation_run_for_execution_request,
)
from kindred.core.simulation_identity import SimulationIdentity
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


pytestmark = pytest.mark.unit


CATALYST_TEXT = "\n".join(
    [
        "reaction: A + E -> B + E; k=2.0",
        "initial: A=3.0",
        "initial: E=5.0",
        "initial: B=0.0",
    ]
)


def _reversible_catalyst_text(prefix: str) -> str:
    return "\n".join(
        [
            f"{prefix}: A + E <-> B + E; kf=2.0; kr=0.5",
            "initial: A=3.0",
            "initial: E=5.0",
            "initial: B=4.0",
        ]
    )


def _species_index(mechanism) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(mechanism.species_names())}


def _rhs_by_species(mechanism, initials: dict[str, float]) -> dict[str, float]:
    names = mechanism.species_names()
    idx = {name: pos for pos, name in enumerate(names)}
    y = np.zeros(len(names), dtype=float)
    for name, value in initials.items():
        y[idx[name]] = float(value)
    rhs = build_ode_rhs_from_mechanism(mechanism)
    values = np.asarray(rhs(0.0, y), dtype=float)
    return {name: float(values[pos]) for name, pos in idx.items()}


def _batch_task_with_plan(task: dict[str, object]) -> dict[str, object]:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    copied = dict(task)
    t_span_raw = copied.get("t_span") or (0.0, float(copied.get("t_end") or 0.0))
    t_span = (float(t_span_raw[0]), float(t_span_raw[1]))
    execution_request = {
        "prepared_payload": copied.get("prepared_payload"),
        "initials": dict(copied.get("initials") or {}),
        "t_span": t_span,
        "solver_config": dict(copied.get("solver_config") or {}),
        "mechanism_text": str(copied.get("mechanism_text") or ""),
        "simulation_identity": (
            dict(copied.get("simulation_identity") or {})
            if isinstance(copied.get("simulation_identity"), dict)
            else None
        ),
    }
    copied["simulation_plan"] = SimulationPlan.from_execution_request(
        execution_request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
        metadata={
            "set_id": str(copied.get("set_id") or ""),
            "set_name": str(copied.get("set_name") or ""),
        },
    ).to_payload()
    return copied


def _expected_catalyzed_product(*, initial_a: float, catalyst_e: float, rate: float, t_end: float) -> float:
    return float(initial_a) * (1.0 - float(np.exp(-float(rate) * float(catalyst_e) * float(t_end))))


def test_dsl_reaction_preserves_same_side_species_as_rate_order_participant() -> None:
    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: Ycross + AOH -> W + PO + AOH; k=594.987218120352",
                "initial: Ycross=1.0",
                "initial: AOH=2.0",
                "initial: W=0.0",
                "initial: PO=0.0",
            ]
        ),
        initials={},
    )

    reaction = mechanism.reactions[0]

    assert reaction.reactants == {"Ycross": 1.0, "AOH": 1.0}
    assert reaction.products == {"W": 1.0, "PO": 1.0, "AOH": 1.0}
    assert reaction.rate_orders == {"Ycross": 1.0, "AOH": 1.0}
    assert reaction.net_stoich == {"Ycross": -1.0, "W": 1.0, "PO": 1.0}
    assert reaction.order == 2


def test_same_side_catalyst_scales_rate_without_changing_catalyst_derivative() -> None:
    mechanism = parse_dsl_to_mechanism(CATALYST_TEXT, initials={})

    zero_catalyst = _rhs_by_species(mechanism, {"A": 3.0, "E": 0.0, "B": 0.0})
    active_catalyst = _rhs_by_species(mechanism, {"A": 3.0, "E": 5.0, "B": 0.0})

    assert zero_catalyst["A"] == pytest.approx(0.0)
    assert zero_catalyst["B"] == pytest.approx(0.0)
    assert zero_catalyst["E"] == pytest.approx(0.0)
    assert active_catalyst["A"] == pytest.approx(-30.0)
    assert active_catalyst["B"] == pytest.approx(30.0)
    assert active_catalyst["E"] == pytest.approx(0.0)


def test_reduced_net_reaction_keeps_full_reactant_order() -> None:
    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: B + B -> B + C; k=3.0",
                "initial: B=4.0",
                "initial: C=0.0",
            ]
        ),
        initials={},
    )

    values = _rhs_by_species(mechanism, {"B": 4.0, "C": 0.0})

    assert mechanism.reactions[0].rate_orders == {"B": 2.0}
    assert mechanism.reactions[0].net_stoich == {"B": -1.0, "C": 1.0}
    assert values["B"] == pytest.approx(-48.0)
    assert values["C"] == pytest.approx(48.0)


def test_serialization_and_hash_distinguish_same_side_catalyst_from_uncatalyzed_net() -> None:
    catalyzed = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: Z -> E; k=1.0",
                "reaction: A + E -> B + E; k=2.0",
                "initial: A=3.0",
                "initial: B=0.0",
                "initial: E=5.0",
                "initial: Z=1.0",
            ]
        ),
        initials={},
    )
    uncatalyzed = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: Z -> E; k=1.0",
                "reaction: A -> B; k=2.0",
                "initial: A=3.0",
                "initial: B=0.0",
                "initial: E=5.0",
                "initial: Z=1.0",
            ]
        ),
        initials={},
    )

    catalyzed_serial = catalyzed.to_serializable()
    uncatalyzed_serial = uncatalyzed.to_serializable()
    json.dumps(catalyzed_serial)
    json.dumps(uncatalyzed_serial)

    assert catalyzed_serial["reactions"][1]["reactants"] == {"A": 1.0, "E": 1.0}
    assert catalyzed_serial["reactions"][1]["products"] == {"B": 1.0, "E": 1.0}
    assert isinstance(catalyzed_serial["reactions"][1]["overrides"], dict)
    assert catalyzed_serial["reactions"] != uncatalyzed_serial["reactions"]
    assert generate_mechanism_hash(catalyzed) != generate_mechanism_hash(uncatalyzed)

    catalyzed_identity = SimulationIdentity.build(
        schema_id=generate_mechanism_hash(catalyzed),
        param_fingerprint="",
        canonical_initials_fingerprint="",
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        t_end=1.0,
    )
    uncatalyzed_identity = SimulationIdentity.build(
        schema_id=generate_mechanism_hash(uncatalyzed),
        param_fingerprint="",
        canonical_initials_fingerprint="",
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        t_end=1.0,
    )
    assert catalyzed_identity.cache_key() != uncatalyzed_identity.cache_key()
    assert catalyzed_identity.prepared_runtime_key() != uncatalyzed_identity.prepared_runtime_key()


def test_prepared_runtime_reuses_same_side_catalyst_semantics() -> None:
    prepared = prepare_simulation_worker_run(
        mechanism_text=CATALYST_TEXT,
        initials={"A": 3.0, "E": 5.0, "B": 0.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
    )

    values = np.asarray(prepared.request.rhs(0.0, prepared.request.y0), dtype=float)
    idx = {name: pos for pos, name in enumerate(prepared.species_names)}

    assert prepared.mechanism.reactions[0].rate_orders == {"A": 1.0, "E": 1.0}
    assert values[idx["A"]] == pytest.approx(-30.0)
    assert values[idx["B"]] == pytest.approx(30.0)
    assert values[idx["E"]] == pytest.approx(0.0)

    bound = prepare_bound_mechanism(
        mechanism_text=CATALYST_TEXT,
        param_names=[],
        initials={"A": 3.0, "E": 5.0, "B": 0.0},
        use_advanced_dsl=True,
    )
    prepared_payload = bound.as_serializable_execution_payload()
    reused = prepare_simulation_worker_run(
        mechanism_text=CATALYST_TEXT,
        prepared_payload=prepared_payload,
        initials={"A": 3.0, "E": 5.0, "B": 0.0},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
    )
    request_reused = prepared_simulation_run_for_execution_request(
        reused,
        {
            "mechanism_text": CATALYST_TEXT,
            "prepared_payload": prepared_payload,
            "initials": {"A": 3.0, "E": 5.0, "B": 0.0},
            "t_span": (0.0, 0.1),
            "solver_config": {"solver": "BDF", "grid": {"N": 3}},
        },
    )
    reused_values = np.asarray(request_reused.request.rhs(0.0, request_reused.request.y0), dtype=float)

    assert reused_values[idx["A"]] == pytest.approx(-30.0)
    assert reused_values[idx["B"]] == pytest.approx(30.0)
    assert reused_values[idx["E"]] == pytest.approx(0.0)


def test_fitting_serial_and_process_payload_paths_preserve_same_side_catalyst() -> None:
    context = prepare_fitting_execution_context(
        mechanism_text=CATALYST_TEXT,
        param_names=[],
        t_end=0.05,
        num_points=6,
        solver="BDF",
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)
    restored = SerialFittingEvaluator.from_process_payload(evaluator.to_process_payload())

    low = evaluator({"init:A": 3.0, "init:E": 1.0, "init:B": 0.0})
    high = restored({"init:A": 3.0, "init:E": 10.0, "init:B": 0.0})

    assert float(np.asarray(low.species["B"], dtype=float)[-1]) == pytest.approx(
        _expected_catalyzed_product(initial_a=3.0, catalyst_e=1.0, rate=2.0, t_end=0.05),
        rel=1e-3,
        abs=1e-6,
    )
    assert float(np.asarray(high.species["B"], dtype=float)[-1]) == pytest.approx(
        _expected_catalyzed_product(initial_a=3.0, catalyst_e=10.0, rate=2.0, t_end=0.05),
        rel=1e-3,
        abs=1e-6,
    )


def test_batch_parallel_task_preserves_same_side_catalyst() -> None:
    base_task = {
        "mechanism_text": CATALYST_TEXT,
        "solver_config": {"solver": "BDF", "grid": {"N": 6}},
        "t_end": 0.05,
        "simulation_identity": {"runtime": "same-side-catalyst"},
    }
    low_task = _batch_task_with_plan(
        {
            **base_task,
            "set_id": "low",
            "set_name": "low",
            "initials": {"A": 3.0, "E": 1.0, "B": 0.0},
        }
    )
    high_task = _batch_task_with_plan(
        {
            **base_task,
            "set_id": "high",
            "set_name": "high",
            "initials": {"A": 3.0, "E": 10.0, "B": 0.0},
        }
    )

    low = batch_parallel.run_batch_simulation_task(low_task)
    high = batch_parallel.run_batch_simulation_task(high_task)
    assert low["success"] is True
    assert high["success"] is True
    low_idx = list(low["species_names"]).index("B")
    high_idx = list(high["species_names"]).index("B")

    assert float(np.asarray(low["Y"], dtype=float)[low_idx, -1]) == pytest.approx(
        _expected_catalyzed_product(initial_a=3.0, catalyst_e=1.0, rate=2.0, t_end=0.05),
        rel=1e-3,
        abs=1e-6,
    )
    assert float(np.asarray(high["Y"], dtype=float)[high_idx, -1]) == pytest.approx(
        _expected_catalyzed_product(initial_a=3.0, catalyst_e=10.0, rate=2.0, t_end=0.05),
        rel=1e-3,
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("prefix", "expected_fast"),
    [
        ("reaction", False),
        ("equilibrium", True),
    ],
)
def test_reversible_same_side_catalyst_ode_math_preserves_both_directions(
    prefix: str,
    expected_fast: bool,
) -> None:
    mechanism = parse_dsl_to_mechanism(_reversible_catalyst_text(prefix), initials={})

    values = _rhs_by_species(mechanism, {"A": 3.0, "E": 5.0, "B": 4.0})

    assert mechanism.equilibria[0].fast is expected_fast
    assert mechanism.equilibria[0].stoich_forward == {"A": 1.0, "E": 1.0}
    assert mechanism.equilibria[0].stoich_back == {"B": 1.0, "E": 1.0}
    assert values["A"] == pytest.approx(-20.0)
    assert values["B"] == pytest.approx(20.0)
    assert values["E"] == pytest.approx(0.0)


@pytest.mark.parametrize("prefix", ["reaction", "equilibrium"])
def test_reversible_serialization_and_hash_distinguish_same_side_catalyst(prefix: str) -> None:
    catalyzed = parse_dsl_to_mechanism(_reversible_catalyst_text(prefix), initials={})
    uncatalyzed = parse_dsl_to_mechanism(
        "\n".join(
            [
                f"{prefix}: A <-> B; kf=2.0; kr=0.5",
                "initial: A=3.0",
                "initial: E=5.0",
                "initial: B=4.0",
            ]
        ),
        initials={},
    )

    catalyzed_serial = catalyzed.to_serializable()
    uncatalyzed_serial = uncatalyzed.to_serializable()

    assert catalyzed_serial["equilibria"][0]["stoich_forward"] == {"A": 1.0, "E": 1.0}
    assert catalyzed_serial["equilibria"][0]["stoich_back"] == {"B": 1.0, "E": 1.0}
    assert catalyzed_serial["equilibria"] != uncatalyzed_serial["equilibria"]
    assert generate_mechanism_hash(catalyzed) != generate_mechanism_hash(uncatalyzed)

    catalyzed_identity = SimulationIdentity.build(
        schema_id=generate_mechanism_hash(catalyzed),
        param_fingerprint="",
        canonical_initials_fingerprint="",
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        t_end=1.0,
    )
    uncatalyzed_identity = SimulationIdentity.build(
        schema_id=generate_mechanism_hash(uncatalyzed),
        param_fingerprint="",
        canonical_initials_fingerprint="",
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        t_end=1.0,
    )
    assert catalyzed_identity.cache_key() != uncatalyzed_identity.cache_key()
    assert catalyzed_identity.prepared_runtime_key() != uncatalyzed_identity.prepared_runtime_key()


def test_programmatic_reactions_use_explicit_physical_sides_and_clone_dicts() -> None:
    mechanism = Mechanism()
    for name in ("A", "E", "B"):
        mechanism.add_species(name, 0.0)

    reactants = {"A": 1.0, "E": 1.0}
    products = {"B": 1.0, "E": 1.0}
    reaction = mechanism.add_reaction(reactants=reactants, products=products, rate=2.0)
    reactants["A"] = 9.0
    products["B"] = 9.0

    assert reaction.reactants == {"A": 1.0, "E": 1.0}
    assert reaction.products == {"B": 1.0, "E": 1.0}
    assert reaction.rate_orders == {"A": 1.0, "E": 1.0}
    assert reaction.net_stoich == {"A": -1.0, "B": 1.0}
    assert not hasattr(reaction, "stoich")

    with pytest.raises(ValueError, match="referenced by a reaction"):
        mechanism.remove_species("E")


def test_direct_reaction_constructor_rejects_ambiguous_net_stoich_input() -> None:
    with pytest.raises(TypeError):
        Reaction(stoich={"A": -1.0, "B": 1.0}, rate=2.0)


def test_direct_reaction_constructor_is_keyword_only_for_semantic_fields() -> None:
    with pytest.raises(TypeError):
        Reaction({"A": 1.0}, {"B": 1.0}, 2.0)


def test_explicit_empty_rate_orders_remain_zero_order() -> None:
    reaction = Reaction(reactants={"A": 1.0}, products={"B": 1.0}, rate=2.0, rate_orders={})

    assert reaction.rate_orders == {}
    assert reaction.order == 0


def test_reversible_eyring_forward_model_survives_immutable_metadata() -> None:
    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A <-> B; dG_act=50; dG_eq=5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )
    equilibrium = mechanism.equilibria[0]

    metadata_view = EquilibriumMetadataView.from_metadata(equilibrium.metadata)
    assert metadata_view.forward_model is not None
    assert metadata_view.forward_model["type"] == "Eyring"

    mechanism.metadata["temperature_K"] = 298.15
    rhs_298 = build_ode_rhs_from_mechanism(mechanism)
    dy_298 = rhs_298(0.0, np.array([1.0, 0.0], dtype=float))

    mechanism.metadata["temperature_K"] = 350.0
    rhs_350 = build_ode_rhs_from_mechanism(mechanism)
    dy_350 = rhs_350(0.0, np.array([1.0, 0.0], dtype=float))

    assert abs(float(dy_350[0])) > abs(float(dy_298[0])) * 10.0
