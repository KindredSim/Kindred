from __future__ import annotations

from dataclasses import replace

import pytest

from kindred.core.simulation_preparation import PreparedSimulationMetadata
from kindred.gui.fitting.run_stamp import build_global_fit_run_stamp, hash_global_fit_run_stamp

pytestmark = pytest.mark.unit


def _prepared_metadata() -> PreparedSimulationMetadata:
    return PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256="0" * 64,
        mechanism_text_len=12,
        param_names=["k"],
        t_end=1.0,
        num_points=3,
        temperature_K=298.15,
        solver_requested="BDF",
        solver_normalized="BDF",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
        intervention_schedule_declarative_fingerprint="declarative-fp",
        intervention_schedule_executable_fingerprint="executable-fp",
    )


def _stamp(prepared: PreparedSimulationMetadata) -> dict:
    return build_global_fit_run_stamp(
        dataset_rows=[],
        included_ids=[],
        applied_fit_targets={},
        weights_used=None,
        weight_mode="none",
        fit_config={
            "parameters": {"k": 1.0},
            "fixed_params": {},
            "bounds": {"k": (0.1, 10.0)},
            "log10_params": {},
            "method": "trf",
            "max_nfev": 10,
            "ftol": 1e-8,
            "xtol": 1e-8,
        },
        mechanism_text="reaction: A -> B; k=1",
        reactions_text="reaction: A -> B; k=1",
        prepared_simulation=prepared,
    )


def test_global_fit_run_stamp_uses_declarative_and_executable_schedule_identity() -> None:
    prepared = _prepared_metadata()
    stamp = _stamp(prepared)

    block = stamp["prepared_simulation"]

    assert block["intervention_schedule_declarative_fingerprint"] == "declarative-fp"
    assert block["intervention_schedule_executable_fingerprint"] == "executable-fp"
    assert "intervention_schedule_fingerprint" not in block


def test_global_fit_run_stamp_hash_changes_with_executable_schedule_identity() -> None:
    first = _stamp(_prepared_metadata())
    second = _stamp(
        replace(
            _prepared_metadata(),
            intervention_schedule_executable_fingerprint="different-executable-fp",
        )
    )

    assert hash_global_fit_run_stamp(first) != hash_global_fit_run_stamp(second)
