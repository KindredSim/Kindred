from __future__ import annotations

import numpy as np
import pytest

from kindred.gui.simulation_provenance_owner import SimulationProvenanceOwner


pytestmark = pytest.mark.unit


def test_publish_simulation_completion_provenance_owns_ctc_and_metadata() -> None:
    owner = SimulationProvenanceOwner(
        dataset_snapshot_getter=lambda: {"datasets": ["d1"]},
        fit_metadata_getter=lambda: {"run": "fit-1"},
    )

    provenance = owner.publish_simulation_completion_provenance(
        mechanism_text="reaction: A -> B; k=1",
        solver_method="BDF",
        solver_label="BDF",
        solver_warning=None,
        solver_config={"rtol": 1e-6, "atol": 1e-12},
        temperature_K=298.15,
        temperature_source="ui",
        energy_unit=None,
        energy_mode=False,
        simulation_time=1.0,
        num_points_requested=2,
        species_names=["A"],
        t=np.asarray([0.0, 1.0], dtype=float),
        series={"A": np.asarray([1.0, 0.0], dtype=float)},
        algebra_scalars={"S": 1.0},
        dataset_overlays=[{"name": "overlay"}],
        solver_provenance={
            "symbolic_jacobian": True,
            "symbolic_jacobian_identity": {"kind": "jacobian", "fingerprint": "abc"},
            "symbolic_wegscheider_identity": {
                "kind": "wegscheider_cyclicity",
                "fingerprint": "proof",
            },
        },
        warnings=[{"kind": "preparation_warning", "message": "symbolic fallback"}],
    )

    assert provenance["datasets"] == {"datasets": ["d1"]}
    assert provenance["fit"] == {"run": "fit-1"}
    assert provenance["algebra_scalars"] == {"S": 1.0}
    assert provenance["symbolic_jacobian"] is True
    assert provenance["symbolic_jacobian_identity"]["fingerprint"] == "abc"
    assert provenance["symbolic_wegscheider_identity"]["fingerprint"] == "proof"
    assert provenance["solver_provenance"]["symbolic_jacobian"] is True
    assert provenance["warnings"] == [{"kind": "preparation_warning", "message": "symbolic fallback"}]
    assert owner.last_simulation_provenance == provenance
    assert set(owner.last_simulation_ctc) == {"A"}
    assert provenance["ctc"]["tail_strategy"] == "38"
