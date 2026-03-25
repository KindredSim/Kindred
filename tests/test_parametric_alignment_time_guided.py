from __future__ import annotations

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError


pytestmark = [pytest.mark.unit]


def _roots_for_x_target(x_target: float) -> tuple[float, float]:
    disc = 1.0 - 4.0 * float(x_target)
    assert disc >= 0.0
    s = float(np.sqrt(disc))
    return (0.5 * (1.0 - s), 0.5 * (1.0 + s))


def test_time_guided_alignment_selects_branch_by_t_obs() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs_time_guided

    t_sim = np.linspace(0.0, 1.0, 2001, dtype=float)
    x_model = t_sim * (1.0 - t_sim)
    y_model = t_sim.copy()

    x_target = 0.2
    t1, t2 = _roots_for_x_target(x_target)
    t_obs = np.asarray([t1, t2], dtype=float)
    x_obs = np.asarray([x_target, x_target], dtype=float)

    aligned = align_y_on_x_obs_time_guided(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="X",
        y_name="Y",
    )

    assert aligned.shape == (2,)
    assert float(aligned[0]) == pytest.approx(t1, abs=1e-3)
    assert float(aligned[1]) == pytest.approx(t2, abs=1e-3)


def test_time_guided_alignment_respects_sampled_time_window() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs_time_guided

    t_sim = np.linspace(0.0, 1.0, 2001, dtype=float)
    x_model = t_sim * (1.0 - t_sim)
    y_model = t_sim.copy()

    x_target = 0.2
    _t1, t2 = _roots_for_x_target(x_target)

    # Window [0.65, 0.75] includes only the late root (~0.7236).
    t_obs = np.asarray([0.65, 0.75], dtype=float)
    x_obs = np.asarray([x_target, x_target], dtype=float)

    aligned = align_y_on_x_obs_time_guided(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="X",
        y_name="Y",
    )

    assert aligned.shape == (2,)
    assert float(aligned[0]) == pytest.approx(t2, abs=1e-3)
    assert float(aligned[1]) == pytest.approx(t2, abs=1e-3)


def test_time_guided_alignment_rejects_no_solution_in_window() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs_time_guided

    t_sim = np.linspace(0.0, 1.0, 2001, dtype=float)
    x_model = t_sim * (1.0 - t_sim)
    y_model = t_sim.copy()

    t_obs = np.asarray([0.0, 1.0], dtype=float)
    x_obs = np.asarray([0.3, 0.3], dtype=float)  # max(x_model) = 0.25

    with pytest.raises(FitSimulationError) as exc_info:
        _ = align_y_on_x_obs_time_guided(
            t_obs=t_obs,
            x_obs=x_obs,
            t_sim=t_sim,
            x_model=x_model,
            y_model=y_model,
            dataset_label="ds",
            x_name="X",
            y_name="Y",
        )

    msg = str(exc_info.value)
    assert "no solution" in msg.lower()
    assert "adjust t_min/t_max" in msg
