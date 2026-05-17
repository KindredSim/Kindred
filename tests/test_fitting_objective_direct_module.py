from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_objective import (
    build_fitting_objective,
    build_prepared_fitting_objective,
)
from kindred.core.simulation_preparation import (
    BoundMechanism,
    PreparedFittingObjectiveContext,
    SimulationPreparationError,
    materialize_request_intervention_schedule_for_parameter_values,
)
from kindred.core.simulator.solvers import SimulationOutput


MECHANISM = "\n".join(
    [
        "reaction: A -> B; k=0.5",
        "initial: A=1.0",
        "initial: B=0.0",
    ]
)


@pytest.mark.unit
def test_direct_module_build_fitting_objective_rejects_nonmonotone_time_grid() -> None:
    objective = build_fitting_objective(
        mechanism_text=MECHANISM,
        param_names=["k1"],
        t_exp=np.array([0.0, 0.5, 0.5]),
        y_exp=np.zeros(3, dtype=float),
        target_species="B",
        solver="BDF",
    )

    with pytest.raises(FitSimulationError, match="strictly increasing"):
        objective(np.array([0.25], dtype=float))

    assert getattr(objective, "failure_reason", "") == (
        "Experimental time points must be strictly increasing for solver evaluation."
    )


@pytest.mark.unit
def test_direct_prepare_fitting_objective_context_preserves_intervention_schedule() -> None:
    from kindred.core.simulation_preparation import prepare_fitting_objective_context

    prepared = prepare_fitting_objective_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=3.0",
            ]
        ),
        param_names=["k1"],
        t_exp=np.array([0.0, 0.5, 1.0], dtype=float),
        target_species="A",
        solver="BDF",
    )

    assert prepared.request.intervention_schedule is not None
    assert prepared.request.species_names == ("A", "B")


@pytest.mark.unit
def test_direct_fitting_objective_resolves_protected_schedule_name_through_canonical_mechanism_parameter() -> None:
    seen_amounts: list[float] = []

    def _solve_policy_factory(_prepared):
        def _solve_request(request: object, _param_values: np.ndarray) -> SimulationOutput:
            schedule = getattr(request, "intervention_schedule", None)
            payload = schedule.to_payload() if schedule is not None else {}
            seen_amounts.append(float(payload["instant_events"][0]["amount"]))
            return SimulationOutput(
                t=np.array([0.0, 1.0], dtype=float),
                Y=np.array([[2.0, 2.0], [0.0, 0.0]], dtype=float),
                provenance={"solver": "direct-schedule-test"},
            )

        return _solve_request

    objective = build_fitting_objective(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.0",
                "initial: A=0.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=0.0; amount_param=K1",
            ]
        ),
        param_names=["K1"],
        t_exp=np.array([0.0, 1.0], dtype=float),
        y_exp=np.array([2.0, 2.0], dtype=float),
        target_species="A",
        solver="BDF",
        solve_policy_factory=_solve_policy_factory,
        parameter_algebra_policy_factory=lambda _prepared: (lambda _params: None),
    )

    residuals = objective(np.array([2.0], dtype=float))

    assert np.allclose(residuals, 0.0)
    assert seen_amounts == [pytest.approx(2.0)]


@pytest.mark.unit
def test_build_fitting_objective_composes_prepared_context_factories() -> None:
    prepared = PreparedFittingObjectiveContext(
        bound=BoundMechanism(
            mechanism=SimpleNamespace(),
            rhs=lambda *_args, **_kwargs: np.array([], dtype=float),
            bindings={},
            species_names=["A"],
            y0=np.array([1.0], dtype=float),
            param_names=["k1"],
            mechanism_text=MECHANISM,
        ),
        requested_param_names=["k1"],
        request=SimpleNamespace(t_span=(0.0, 1.0)),
        target_species="A",
        target_is_species=True,
        target_species_index=0,
        compiled_algebra=None,
        initials_for_algebra={"A": 1.0},
        temperature_K=298.15,
    )
    events: list[tuple[str, object]] = []

    def _prepare_context(**kwargs: object) -> PreparedFittingObjectiveContext:
        events.append(("prepare", tuple(kwargs["param_names"])))  # type: ignore[index]
        return prepared

    def _solve_policy_factory(received: PreparedFittingObjectiveContext):
        events.append(("solve_factory", received is prepared))

        def _solve_request(request: object, param_values: np.ndarray) -> SimulationOutput:
            events.append(("solve", (request is prepared.request, tuple(param_values))))
            return SimulationOutput(
                t=np.array([0.0, 1.0], dtype=float),
                Y=np.array([[1.0, 0.5]], dtype=float),
                provenance={"solver": "prepared-pipeline-test"},
            )

        return _solve_request

    def _parameter_algebra_policy_factory(received: PreparedFittingObjectiveContext):
        events.append(("algebra_factory", received is prepared))

        def _apply(params: dict[str, float]) -> None:
            events.append(("algebra", dict(params)))

        return _apply

    objective = build_fitting_objective(
        mechanism_text=MECHANISM,
        param_names=["k1"],
        t_exp=np.array([0.0, 1.0], dtype=float),
        y_exp=np.array([1.0, 0.5], dtype=float),
        target_species="A",
        solver="BDF",
        prepare_context_func=_prepare_context,
        solve_policy_factory=_solve_policy_factory,
        parameter_algebra_policy_factory=_parameter_algebra_policy_factory,
    )

    residuals = objective(np.array([0.25], dtype=float))

    assert np.allclose(residuals, 0.0)
    assert events == [
        ("prepare", ("k1",)),
        ("solve_factory", True),
        ("algebra_factory", True),
        ("algebra", {"k1": 0.25}),
        ("solve", (True, (0.25,))),
    ]


@pytest.mark.unit
def test_direct_module_parameterized_schedule_preserves_structured_execution_request() -> None:
    from kindred.core.intervention_schedule import parse_intervention_schedule_from_dsl
    from kindred.core.simulation_preparation import SimulationExecutionRequest
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.0",
            "initial: A=0.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=0.0; amount_param=dose",
        ]
    )
    mechanism = parse_dsl_to_mechanism(mechanism_text, initials={})
    unresolved = parse_intervention_schedule_from_dsl(mechanism_text)
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF"},
        mechanism_text=mechanism_text,
    )
    prepared = PreparedFittingObjectiveContext(
        bound=BoundMechanism(
            mechanism=mechanism,
            rhs=lambda *_args, **_kwargs: np.array([], dtype=float),
            bindings={},
            species_names=["A", "B"],
            y0=np.array([0.0, 0.0], dtype=float),
            param_names=["dose"],
            mechanism_text=mechanism_text,
            unresolved_intervention_schedule=unresolved,
        ),
        requested_param_names=["dose"],
        request=request,
        target_species="A",
        target_is_species=True,
        target_species_index=0,
        compiled_algebra=None,
        initials_for_algebra={"A": 0.0, "B": 0.0},
        temperature_K=298.15,
        unresolved_intervention_schedule=unresolved,
    )

    updated = materialize_request_intervention_schedule_for_parameter_values(
        mechanism=prepared.bound.mechanism,
        request=prepared.request,
        unresolved_intervention_schedule=prepared.unresolved_intervention_schedule,
        parameter_values={"dose": 2.0},
        species_names=prepared.bound.species_names,
        runtime_parameter_names=prepared.bound.bindings.keys(),
    )

    assert isinstance(updated, SimulationExecutionRequest)
    assert updated.has_intervention_schedule_authority is True
    assert updated.intervention_schedule is not None
    payload = updated.intervention_schedule.to_payload()
    assert payload["instant_events"][0]["amount"] == pytest.approx(2.0)


@pytest.mark.unit
def test_direct_module_parameterized_schedule_rejects_unowned_request_type() -> None:
    from kindred.core.intervention_schedule import parse_intervention_schedule_from_dsl
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.0",
            "initial: A=0.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=0.0; amount_param=dose",
        ]
    )
    mechanism = parse_dsl_to_mechanism(mechanism_text, initials={})
    unresolved = parse_intervention_schedule_from_dsl(mechanism_text)
    prepared = PreparedFittingObjectiveContext(
        bound=BoundMechanism(
            mechanism=mechanism,
            rhs=lambda *_args, **_kwargs: np.array([], dtype=float),
            bindings={},
            species_names=["A", "B"],
            y0=np.array([0.0, 0.0], dtype=float),
            param_names=["dose"],
            mechanism_text=mechanism_text,
            unresolved_intervention_schedule=unresolved,
        ),
        requested_param_names=["dose"],
        request=SimpleNamespace(t_span=(0.0, 1.0)),
        target_species="A",
        target_is_species=True,
        target_species_index=0,
        compiled_algebra=None,
        initials_for_algebra={"A": 0.0, "B": 0.0},
        temperature_K=298.15,
        unresolved_intervention_schedule=unresolved,
    )

    with pytest.raises(
        SimulationPreparationError,
        match="does not support intervention schedule replacement",
    ):
        materialize_request_intervention_schedule_for_parameter_values(
            mechanism=prepared.bound.mechanism,
            request=prepared.request,
            unresolved_intervention_schedule=prepared.unresolved_intervention_schedule,
            parameter_values={"dose": 2.0},
            species_names=prepared.bound.species_names,
            runtime_parameter_names=prepared.bound.bindings.keys(),
        )


@pytest.mark.unit
def test_direct_module_prepared_objective_penalizes_nonfinite_model_series() -> None:
    prepared = PreparedFittingObjectiveContext(
        bound=BoundMechanism(
            mechanism=SimpleNamespace(),
            rhs=lambda *_args, **_kwargs: np.array([], dtype=float),
            bindings={},
            species_names=["A"],
            y0=np.array([1.0], dtype=float),
            param_names=["k1"],
            mechanism_text=MECHANISM,
        ),
        requested_param_names=["k1"],
        request=SimpleNamespace(t_span=(0.0, 1.0)),
        target_species="A",
        target_is_species=True,
        target_species_index=0,
        compiled_algebra=None,
        initials_for_algebra={"A": 1.0},
        temperature_K=298.15,
    )

    def _solve_request(_request: object, _param_values: np.ndarray) -> SimulationOutput:
        return SimulationOutput(
            t=np.array([0.0, 0.5, 1.0], dtype=float),
            Y=np.array([[0.0, np.nan, 0.0]], dtype=float),
            provenance={"solver": "direct-test"},
        )

    objective = build_prepared_fitting_objective(
        prepared,
        y_exp=np.zeros(3, dtype=float),
        solve_request=_solve_request,
        parameter_algebra_policy=lambda _params: None,
    )

    residuals = objective(np.array([0.5], dtype=float))

    assert np.allclose(residuals, 1e6)
    assert isinstance(getattr(objective, "last_error", None), FitSimulationError)
    assert getattr(objective, "last_error_provenance", None) == {"solver": "direct-test"}
