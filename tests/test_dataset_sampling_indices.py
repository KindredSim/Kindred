from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit



def test_compute_sampled_indices_evenly_spaced_includes_endpoints():
    from kindred.core.analysis.dataset_sampling import compute_sampled_indices

    t = np.arange(0, 200, dtype=float)
    idx = compute_sampled_indices(t=t, t_min=50.0, t_max=149.0, n_points=10)

    assert idx.shape == (10,)
    assert idx.dtype.kind in {"i", "u"}
    assert np.all(np.diff(idx) > 0)

    sampled = t[idx]
    assert sampled[0] == 50.0
    assert sampled[-1] == 149.0
    assert np.all((sampled >= 50.0) & (sampled <= 149.0))
    assert list(sampled) == [50.0, 61.0, 72.0, 83.0, 94.0, 105.0, 116.0, 127.0, 138.0, 149.0]


def test_compute_sampled_indices_n_equals_2_returns_endpoints_only():
    from kindred.core.analysis.dataset_sampling import compute_sampled_indices

    t = np.arange(0, 10, dtype=float)
    idx = compute_sampled_indices(t=t, t_min=2.0, t_max=7.0, n_points=2)

    assert idx.shape == (2,)
    sampled = t[idx]
    assert list(sampled) == [2.0, 7.0]

