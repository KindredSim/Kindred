from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]


def test_prepared_simulation_returns_typed_series_payload() -> None:
    from kindred.core.simulation_preparation import build_prepared_simulation_func
    from kindred.core.simulation_series_payload import SimulationSeriesPayload

    prepared = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
    )

    result = prepared({"k1": 0.5})

    assert isinstance(result, SimulationSeriesPayload)
    assert np.asarray(result["t"], dtype=float).shape == (5,)
    assert "A" in result["species"]
    assert result.to_legacy_dict()["species"].keys() == result["species"].keys()


def test_prepared_simulation_executes_intervention_schedule_direct_path() -> None:
    from kindred.core.intervention_schedule import normalized_intervention_schedule_fingerprint_from_dsl_text
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=3.0",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
    )
    prepared_meta = prepared._kindred_prepared_simulation_meta  # type: ignore[attr-defined]

    result = prepared({"k1": 0.0})

    assert prepared_meta.intervention_schedule_fingerprint == normalized_intervention_schedule_fingerprint_from_dsl_text(
        mechanism_text
    )
    assert float(np.asarray(result["species"]["A"], dtype=float)[0]) == pytest.approx(3.0)


def test_prepared_simulation_metadata_canonicalizes_direct_indexed_request_names_before_execution() -> None:
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    prepared = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["K1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
    )

    prepared_meta = prepared._kindred_prepared_simulation_meta  # type: ignore[attr-defined]

    assert prepared_meta.param_names == ["k1"]


def test_global_fit_accepts_typed_simulation_series_payload() -> None:
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_series_payload import SimulationSeriesPayload

    t = np.array([0.0, 0.5, 1.0], dtype=float)
    y = np.array([1.0, 0.75, 0.5], dtype=float)

    def _simulation(_params):
        return SimulationSeriesPayload(
            t=t.copy(),
            species={"A": y.copy()},
            algebra_scalars={},
        )

    result = fit_global(
        _simulation,
        datasets=[{"id": "ds1", "t": t.copy(), "y": y.copy(), "species": "A"}],
        shared_params={"k1": 0.5},
        method="trf",
        max_nfev=2,
    )

    assert result.dataset_info
    assert result.dataset_info[0].dataset_id == "ds1"
