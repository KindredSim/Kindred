from __future__ import annotations

import pytest

from kindred.core.simulation_identity import contained_simulation_owner_identity


pytestmark = pytest.mark.unit


def _preview_mechanism_digest(mechanism_text: str) -> str:
    payload = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text=mechanism_text,
        solver_config={
            "solver": "BDF",
            "rtol": 1e-6,
            "atol": 1e-12,
            "grid": {"N": 10},
            "temperature_K": 298.15,
        },
        t_end=1.0,
        set_id="default",
        parameter_names=["Keq1"],
    )
    return str(payload["mechanism_digest"])


@pytest.mark.parametrize("key", ["Keq", "keq", "KEQ"])
def test_preview_identity_tracks_exact_keq_key_value_case_insensitively(key):
    assert _preview_mechanism_digest(f"equilibrium: A <-> B; kf=1; {key}=2") != (
        _preview_mechanism_digest(f"equilibrium: A <-> B; kf=1; {key}=3")
    )


@pytest.mark.parametrize("key", ["K", "K_eq", "k_eq"])
def test_preview_identity_tracks_legacy_keq_alias_values(key):
    first_digest = _preview_mechanism_digest(f"equilibrium: A <-> B; kf=1; {key}=2")
    second_digest = _preview_mechanism_digest(f"equilibrium: A <-> B; kf=1; {key}=3")
    canonical_digest = _preview_mechanism_digest("equilibrium: A <-> B; kf=1; Keq=2")

    assert first_digest != second_digest
    assert first_digest != canonical_digest
