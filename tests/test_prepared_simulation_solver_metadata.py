from __future__ import annotations
import pytest

pytestmark = pytest.mark.unit



def test_build_prepared_simulation_meta_uses_typed_metadata_object() -> None:
    from kindred.core.simulation_preparation import (
        PreparedSimulationMetadata,
        build_prepared_simulation_func,
    )

    simulation_func = build_prepared_simulation_func(
        mechanism_text="rxn: A -> B; k=1.0",
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        temperature_K=298.15,
        solver="unknown_solver_name",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    meta = getattr(simulation_func, "_kindred_prepared_simulation_meta", None)

    assert isinstance(meta, PreparedSimulationMetadata)
    assert meta.solver_requested == "unknown_solver_name"
    assert meta.solver_normalized == "BDF"
    assert meta.param_names == ["k1"]
    serialized = meta.to_serializable_dict()
    assert serialized["solver_requested"] == "unknown_solver_name"
    assert serialized["solver_normalized"] == "BDF"
    assert "solver" not in serialized


def test_build_prepared_simulation_meta_does_not_publish_raw_schedule_fingerprint_after_parse_failure() -> None:
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    simulation_func = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "rxn: A -> B; k=1.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    meta = getattr(simulation_func, "_kindred_prepared_simulation_meta", None)

    assert meta.intervention_schedule_fingerprint == ""
