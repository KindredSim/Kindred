from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import pytest

from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    EquilibriumMetadataView,
    MechanismMetadataKeys,
    MechanismMetadataView,
)


@pytest.mark.unit
def test_equilibrium_metadata_view_coerces_serialized_bool_strings_and_nonfinite_numbers() -> None:
    view = EquilibriumMetadataView.from_metadata(
        {
            EquilibriumMetadataKeys.FAST_EQUILIBRIUM: "false",
            EquilibriumMetadataKeys.USER_PROVIDED_KF: "1",
            EquilibriumMetadataKeys.USER_PROVIDED_KR: "0",
            EquilibriumMetadataKeys.DG_EQ_J_PER_MOL: "nan",
            EquilibriumMetadataKeys.STANDARD_CONC_M: "inf",
            EquilibriumMetadataKeys.EXPLICIT_RATES: ["1.5", "nan", "bad", 2.5],
        }
    )

    assert view.fast_equilibrium is False
    assert view.user_provided_kf is True
    assert view.user_provided_kr is False
    assert view.dG_eq_J_per_mol is None
    assert view.standard_conc_M is None
    assert view.explicit_rates == (1.5, 2.5)


@pytest.mark.unit
def test_mechanism_metadata_view_defaults_invalid_energy_unit_and_numeric_inputs() -> None:
    view = MechanismMetadataView.from_metadata(
        {
            MechanismMetadataKeys.TEMPERATURE_K: object(),
            MechanismMetadataKeys.STANDARD_CONC_M: "bad",
            MechanismMetadataKeys.KAPPA_GLOBAL: None,
            MechanismMetadataKeys.ENERGY_UNIT: "hartree",
        }
    )

    assert math.isclose(view.temperature_K, 298.15)
    assert math.isclose(view.standard_conc_M, 1.0)
    assert math.isclose(view.kappa_global, 1.0)
    assert view.energy_unit == "kJ/mol"


@pytest.mark.unit
def test_bound_mechanism_worker_payload_uses_metadata_view_for_temperature_schedule(monkeypatch) -> None:
    from kindred.core.simulation_preparation import BoundMechanism

    sentinel_schedule = object()
    monkeypatch.setattr(
        "kindred.core.simulation_preparation._metadata_view_for_mechanism",
        lambda *_args, **_kwargs: SimpleNamespace(temperature_schedule=sentinel_schedule),
    )

    mechanism = SimpleNamespace(metadata={"temperature_schedule": None})
    bound = BoundMechanism(
        mechanism=mechanism,
        rhs=lambda _t, y: y,
        bindings={},
        species_names=["A"],
        y0=np.asarray([1.0], dtype=float),
        param_names=[],
        mechanism_text="reaction: A -> A; k=1.0",
    )

    payload = bound.as_worker_payload()

    assert payload["temperature_schedule"] is sentinel_schedule
