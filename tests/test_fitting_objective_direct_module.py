from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_objective import (
    build_fitting_objective,
    build_prepared_fitting_objective,
)
from kindred.core.simulation_preparation import BoundMechanism, PreparedFittingObjectiveContext
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
