from __future__ import annotations

import pytest

from kindred.core.simulation_identity import contained_simulation_owner_identity


pytestmark = pytest.mark.unit


def _preview_structural_digest(mechanism_text: str) -> str:
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
    return str(payload["structural_mechanism_digest"])


@pytest.mark.parametrize("key", ["Keq", "keq", "KEQ"])
def test_preview_identity_masks_exact_keq_key_case_insensitively(key):
    assert _preview_structural_digest(f"equilibrium: A <-> B; kf=1; {key}=2") == (
        _preview_structural_digest(f"equilibrium: A <-> B; kf=1; {key}=3")
    )


@pytest.mark.parametrize("key", ["K", "K_eq", "k_eq"])
def test_preview_identity_does_not_mask_legacy_keq_aliases(key):
    first_digest = _preview_structural_digest(f"equilibrium: A <-> B; kf=1; {key}=2")
    second_digest = _preview_structural_digest(f"equilibrium: A <-> B; kf=1; {key}=3")
    canonical_digest = _preview_structural_digest("equilibrium: A <-> B; kf=1; Keq=2")

    assert first_digest != second_digest
    assert first_digest != canonical_digest


def test_simulation_identity_distinguishes_declarative_protocol_fingerprint():
    from kindred.core.intervention_schedule import InterventionSchedule, compile_intervention_schedule
    from kindred.core.simulation_identity import SimulationIdentity

    base_schedule = InterventionSchedule.from_payload(
        {
            "protocols": [
                {
                    "kind": "repeat",
                    "name": "light",
                    "start": 0.0,
                    "every": 1.0,
                    "duration": 0.5,
                    "count": 1,
                    "during": [{"kind": "source", "species": "A", "rate": 1.0}],
                }
            ]
        }
    )
    labelled_schedule = InterventionSchedule.from_payload(
        {
            "metadata": {"label": "display label"},
            "protocols": [
                {
                    "kind": "repeat",
                    "name": "light",
                    "start": 0.0,
                    "every": 1.0,
                    "duration": 0.5,
                    "count": 1,
                    "during": [{"kind": "source", "species": "A", "rate": 1.0}],
                }
            ],
        }
    )

    base_compiled = compile_intervention_schedule(base_schedule)
    labelled_compiled = compile_intervention_schedule(labelled_schedule)
    assert base_compiled.executable_fingerprint == labelled_compiled.executable_fingerprint

    base_identity = SimulationIdentity.build(
        schema_id="schema",
        param_fingerprint="params",
        canonical_initials_fingerprint="initials",
        solver_config={"solver": "BDF", "temperature_K": 298.15},
        t_end=1.0,
        intervention_schedule_declarative_fingerprint=base_compiled.declarative_fingerprint,
        intervention_schedule_executable_fingerprint=base_compiled.executable_fingerprint,
    )
    labelled_identity = SimulationIdentity.build(
        schema_id="schema",
        param_fingerprint="params",
        canonical_initials_fingerprint="initials",
        solver_config={"solver": "BDF", "temperature_K": 298.15},
        t_end=1.0,
        intervention_schedule_declarative_fingerprint=labelled_compiled.declarative_fingerprint,
        intervention_schedule_executable_fingerprint=labelled_compiled.executable_fingerprint,
    )

    assert base_identity.cache_key() != labelled_identity.cache_key()
    assert base_identity.prepared_runtime_key() == labelled_identity.prepared_runtime_key()
    assert "intervention_schedule_fingerprint" not in base_identity.to_payload()


def test_simulation_identity_rejects_stale_single_intervention_schedule_fingerprint() -> None:
    from kindred.core.simulation_identity import SimulationIdentity

    payload = {
        "schema_id": "schema",
        "param_fingerprint": "params",
        "canonical_initials_fingerprint": "initials",
        "solver": {"solver": "BDF", "temperature_K": 298.15},
        "t_end": 1.0,
        "intervention_schedule_fingerprint": "stale",
    }

    with pytest.raises(ValueError, match="stale intervention_schedule_fingerprint"):
        SimulationIdentity.from_payload(payload)
