from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode

pytestmark = pytest.mark.integration


def _request_from_dsl(
    text: str,
    *,
    n: int = 7,
    t_span: tuple[float, float] = (0.0, 6.0),
    t_eval: np.ndarray | None = None,
) -> tuple[SimulationRequest, list[str]]:
    mechanism = parse_dsl_to_mechanism(text, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names], dtype=float)
    return (
        SimulationRequest(
            rhs=rhs,
            t_span=t_span,
            y0=y0,
            solver="BDF",
            grid=None if t_eval is not None else {"N": n},
            t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
            species_names=tuple(species_names),
            intervention_schedule=mechanism.metadata.get("intervention_schedule"),
        ),
        species_names,
    )


def test_instant_events_emit_single_post_event_boundary_sample() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
                "intervention: op=add; species=A; time=3.0; amount=1.5",
                "intervention: op=clear; species=B; time=3.0",
            ]
        )
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]
    b = result.Y[species_names.index("B")]

    assert list(result.t).count(3.0) == 1
    assert float(a[0]) == pytest.approx(2.0)
    assert float(a[np.where(result.t == 3.0)[0][0]]) == pytest.approx(3.5)
    assert float(b[np.where(result.t == 3.0)[0][0]]) == pytest.approx(0.0)


def test_final_instant_event_replaces_final_boundary_sample() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=6.0; value=4.0",
            ]
        )
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]

    assert float(result.t[-1]) == pytest.approx(6.0)
    assert float(a[-1]) == pytest.approx(4.0)
    assert list(result.t).count(6.0) == 1


def test_final_ulp_adjacent_sample_uses_post_event_state() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=6.0; value=4.0",
            ]
        ),
        t_eval=np.array([0.0, np.nextafter(6.0, -np.inf)], dtype=float),
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]

    assert float(result.t[-1]) == pytest.approx(6.0)
    assert float(a[-1]) == pytest.approx(4.0)
    assert list(result.t).count(6.0) == 1


def test_terminal_event_at_intervention_boundary_keeps_requested_boundary_sample() -> None:
    request, _species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=0.5; amount=1.0",
            ]
        ),
        t_span=(0.0, 1.0),
        t_eval=np.array([0.0, 0.5, 1.0], dtype=float),
    )

    def stop_at_half(t, _y):
        return float(t) - 0.5

    stop_at_half.terminal = True
    stop_at_half.direction = 1

    result = solve_ode(replace(request, events=(stop_at_half,)))

    assert list(result.t) == pytest.approx([0.0, 0.5])


def test_decimal_boundary_sample_uses_post_event_state() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=0.1; amount=1.0",
            ]
        ),
        n=4,
        t_span=(0.0, 0.3),
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]

    assert result.t[1] == pytest.approx(0.1)
    assert float(a[1]) == pytest.approx(2.0)
    assert float(a[2]) == pytest.approx(2.0)


def test_ulp_adjacent_boundary_samples_collapse_to_single_post_event_sample() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=0.1; amount=1.0",
            ]
        ),
        t_span=(0.0, 0.2),
        t_eval=np.array(
            [
                0.0,
                np.nextafter(0.1, -np.inf),
                0.1,
                np.nextafter(0.1, np.inf),
                0.2,
            ],
            dtype=float,
        ),
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]

    assert list(result.t).count(0.1) == 1
    assert float(a[np.where(result.t == 0.1)[0][0]]) == pytest.approx(2.0)


def test_repeated_pulses_and_source_sink_intervals_execute_through_shared_solver() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=0.0",
                "initial: B=2.0",
                "intervention: op=pulse; species=A; start=1.0; every=2.0; count=3; amount=1.0",
                "intervention: op=source; species=A; start=1.0; end=5.0; rate=0.5",
                "intervention: op=sink; species=B; start=2.0; end=4.0; rate=0.25",
            ]
        )
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]
    b = result.Y[species_names.index("B")]

    assert float(a[-1]) == pytest.approx(5.0, abs=1e-6)
    assert float(b[-1]) == pytest.approx(1.5, abs=1e-6)
    assert result.provenance["has_intervention_schedule"] is True


def test_parameterized_pulse_amount_executes_through_execution_request_overrides() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=1.0; amount_param=dose",
            ]
        ),
        parameter_overrides={"dose": 2.5},
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    result = solve_ode(prepared.request)
    a = result.Y[prepared.species_names.index("A")]

    assert float(a[-1]) == pytest.approx(3.5, abs=1e-6)


def test_parameterized_event_time_resegments_execution_request_schedule() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time_param=pulse_time; amount=2.0",
            ]
        ),
        parameter_overrides={"pulse_time": 2.0},
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    result = solve_ode(prepared.request)
    a = result.Y[prepared.species_names.index("A")]

    assert float(result.t[-1]) == pytest.approx(2.0)
    assert float(a[1]) == pytest.approx(1.0, abs=1e-6)
    assert float(a[-1]) == pytest.approx(3.0, abs=1e-6)


def test_parameterized_source_interval_rate_executes_through_execution_request_overrides() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=0.0",
                "initial: B=0.0",
                "intervention: op=source; species=A; start=0.0; end=2.0; rate_param=feed",
            ]
        ),
        parameter_overrides={"feed": 1.5},
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    result = solve_ode(prepared.request)
    a = result.Y[prepared.species_names.index("A")]

    assert float(a[-1]) == pytest.approx(3.0, abs=1e-6)


def test_parameterized_repeated_pulse_amount_executes_through_execution_request_overrides() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 3.0),
        solver_config={"solver": "BDF", "grid": {"N": 4}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=pulse; species=A; start=1.0; every=1.0; count=2; amount_param=dose",
            ]
        ),
        parameter_overrides={"dose": 1.25},
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    result = solve_ode(prepared.request)
    a = result.Y[prepared.species_names.index("A")]

    assert float(a[-1]) == pytest.approx(3.5, abs=1e-6)


def test_parameterized_repeated_pulse_timing_executes_through_execution_request_overrides() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 4.0),
        solver_config={"solver": "BDF", "grid": {"N": 5}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=pulse; species=A; start_param=pulse_start; every_param=pulse_gap; count=2; amount=1.0",
            ]
        ),
        parameter_overrides={"pulse_start": 1.0, "pulse_gap": 2.0},
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    result = solve_ode(prepared.request)
    a = result.Y[prepared.species_names.index("A")]

    assert float(a[np.where(result.t == 1.0)[0][0]]) == pytest.approx(2.0, abs=1e-6)
    assert float(a[np.where(result.t == 3.0)[0][0]]) == pytest.approx(3.0, abs=1e-6)
    assert float(a[-1]) == pytest.approx(3.0, abs=1e-6)


def test_reservoir_interval_holds_species_constant_without_hiding_other_dynamics() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=reservoir; species=A; start=1.0; end=5.0; value=2.0",
            ]
        ),
        n=13,
    )

    result = solve_ode(request)
    a = result.Y[species_names.index("A")]
    b = result.Y[species_names.index("B")]
    active = (result.t >= 1.0) & (result.t < 5.0)

    np.testing.assert_allclose(a[active], 2.0, atol=1e-7)
    assert float(b[-1]) > 0.0


def test_state_triggered_intervention_executes_through_shared_solver_root_event() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> C; k=1.0",
                "reaction: B -> D; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
                "initial: D=0.0",
                "intervention: op=trigger; trigger_species=A; threshold=0.5; direction=falling; action=add; species=B; amount=1.25; max_count=1; min_interval=0.0",
            ]
        ),
        n=9,
        t_span=(0.0, 2.0),
    )

    result = solve_ode(request)
    b = result.Y[species_names.index("B")]

    assert float(b[-1]) == pytest.approx(1.25, abs=1e-6)
    assert result.provenance["intervention_trigger_events"] == [
        {
            "time": pytest.approx(0.693147, abs=1e-5),
            "trigger_species": "A",
            "species": "B",
            "action": "add",
        }
    ]


def test_state_triggered_intervention_supports_rising_crossing_direction() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "reaction: C -> D; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
                "initial: D=0.0",
                "intervention: op=trigger; trigger_species=B; threshold=0.5; direction=rising; action=set; species=C; value=2.0; max_count=1; min_interval=0.0",
            ]
        ),
        n=9,
        t_span=(0.0, 2.0),
    )

    result = solve_ode(request)
    c = result.Y[species_names.index("C")]

    assert float(c[-1]) == pytest.approx(2.0, abs=1e-6)
    assert result.provenance["intervention_trigger_events"][0]["action"] == "set"


def test_state_triggered_intervention_repeat_respects_max_count() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "reaction: C -> D; k=0",
                "initial: A=0.0",
                "initial: B=0.0",
                "initial: C=0.0",
                "initial: D=0.0",
                "intervention: op=source; species=A; start=0.0; end=5.0; rate=1.0",
                "intervention: op=trigger; trigger_species=A; threshold=1.0; direction=rising; action=add; species=C; amount=1.0; max_count=2; min_interval=0.25",
                "intervention: op=clear; species=A; time=1.5",
            ]
        ),
        n=11,
        t_span=(0.0, 5.0),
    )

    result = solve_ode(request)
    c = result.Y[species_names.index("C")]

    assert float(c[-1]) == pytest.approx(2.0, abs=1e-6)
    assert len(result.provenance["intervention_trigger_events"]) == 2


def test_state_triggered_intervention_applies_simultaneous_triggers_deterministically() -> None:
    request, species_names = _request_from_dsl(
        "\n".join(
            [
                "reaction: A -> D; k=1.0",
                "reaction: B -> E; k=0",
                "reaction: C -> F; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
                "initial: D=0.0",
                "initial: E=0.0",
                "initial: F=0.0",
                "intervention: op=trigger; trigger_species=A; threshold=0.5; direction=either; action=add; species=B; amount=1.0; max_count=1; min_interval=0.0",
                "intervention: op=trigger; trigger_species=A; threshold=0.5; direction=either; action=add; species=C; amount=2.0; max_count=1; min_interval=0.0",
            ]
        ),
        n=9,
        t_span=(0.0, 2.0),
    )

    result = solve_ode(request)
    b = result.Y[species_names.index("B")]
    c = result.Y[species_names.index("C")]

    assert float(b[-1]) == pytest.approx(1.0, abs=1e-6)
    assert float(c[-1]) == pytest.approx(2.0, abs=1e-6)
    assert [
        (event["trigger_species"], event["species"], event["action"])
        for event in result.provenance["intervention_trigger_events"]
    ] == [("A", "B", "add"), ("A", "C", "add")]
