from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.unit]


def test_penalized_time_guided_alignment_selects_branch_by_time() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs_time_guided_penalized

    t_sim = np.linspace(0.0, 1.0, 1001, dtype=float)
    x_model = t_sim * (1.0 - t_sim)  # non-monotone, max=0.25 at t=0.5
    y_model = t_sim.copy()

    x_obs = np.asarray([0.2, 0.2], dtype=float)
    t_obs = np.asarray([0.25, 0.75], dtype=float)

    out = align_y_on_x_obs_time_guided_penalized(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="Int",
        y_name="Y",
    )

    # x=t(1-t)=0.2 has two solutions: ~0.2764 and ~0.7236. Choose nearest to t_obs.
    assert out.t_star.shape == t_obs.shape
    assert out.y_aligned.shape == t_obs.shape
    assert np.isfinite(out.t_star).all()
    assert np.isfinite(out.y_aligned).all()
    assert np.isfinite(out.dx).all()

    assert out.y_aligned[0] == pytest.approx(0.2764, abs=5e-3)
    assert out.y_aligned[1] == pytest.approx(0.7236, abs=5e-3)
    assert bool(np.all(out.exact))


def test_penalized_alignment_out_of_range_returns_finite_dx_no_exception() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs_time_guided_penalized

    t_sim = np.linspace(0.0, 1.0, 501, dtype=float)
    x_model = t_sim * (1.0 - t_sim)  # max=0.25
    y_model = t_sim.copy()

    t_obs = np.asarray([0.2, 0.8], dtype=float)
    x_obs = np.asarray([0.3, 0.3], dtype=float)  # out of range (no crossing)

    out1 = align_y_on_x_obs_time_guided_penalized(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="Int",
        y_name="Y",
    )
    out2 = align_y_on_x_obs_time_guided_penalized(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="Int",
        y_name="Y",
    )

    assert np.isfinite(out1.y_aligned).all()
    assert np.isfinite(out1.dx).all()
    # Since x_obs > max(x_model), dx = x_model(t*) - x_obs should be <= 0.
    assert float(np.max(out1.dx)) <= 0.0
    assert bool(np.all(~out1.exact))

    # Determinism: same inputs => same outputs.
    assert np.allclose(out1.t_star, out2.t_star)
    assert np.allclose(out1.y_aligned, out2.y_aligned)
    assert np.allclose(out1.dx, out2.dx)

