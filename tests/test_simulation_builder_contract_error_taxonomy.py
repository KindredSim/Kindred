from __future__ import annotations

import pytest

from kindred.core.api.simulation import (
    SimulationBuilderContractError,
    coerce_simulation_builder,
)
from kindred.core.exceptions import IntegrationFailureError


@pytest.mark.unit
def test_simulation_builder_contract_error_uses_integration_taxonomy() -> None:
    def legacy_builder(_mechanism_text: str, _param_names: list[str]):
        return None

    builder = coerce_simulation_builder(legacy_builder)

    with pytest.raises(SimulationBuilderContractError) as excinfo:
        builder("A -> B", ["k1"], solver="BDF", rtol=1e-6, atol=1e-9)

    err = excinfo.value
    assert isinstance(err, IntegrationFailureError)
    assert err.code == "E602"
    assert err.suggestion == "Update the simulation builder to accept solver, rtol, and atol keyword arguments."
    assert err.details == {
        "contract": "(mechanism_text, param_names, *, solver, rtol, atol)",
        "missing_kwargs": ["solver", "rtol", "atol"],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy_builder",
    [
        lambda _mechanism_text, _param_names, *, solver, atol: None,
        lambda _mechanism_text, _param_names, *, solver, rtol: None,
    ],
)
def test_coerce_simulation_builder_maps_all_required_kwarg_mismatches(legacy_builder) -> None:
    builder = coerce_simulation_builder(legacy_builder)

    with pytest.raises(SimulationBuilderContractError):
        builder("A -> B", ["k1"], solver="BDF", rtol=1e-6, atol=1e-9)
