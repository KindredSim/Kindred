import numpy as np
import pytest

pytestmark = pytest.mark.gui



def test_simulate_mechanism_includes_algebra_observables(main_window):
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "total = [A] + [B]",
        ]
    )

    result = main_window._simulate_mechanism(dsl, t_end=0.2, num_points=8)
    assert "species" in result
    species = result["species"]
    assert "total" in species
    assert species["total"].shape == np.asarray(result["t"]).shape

