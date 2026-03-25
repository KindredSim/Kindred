from __future__ import annotations

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError


pytestmark = [pytest.mark.unit]


def test_align_y_on_x_obs_monotone_increasing() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs

    t_sim = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    x_model = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=float)
    y_model = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)

    # Window is [1, 2] -> segment uses points at t=1,2 (x=[1,4], y=[3,5]).
    t_obs = np.asarray([1.0, 2.0], dtype=float)
    x_obs = np.asarray([2.0, 4.0], dtype=float)

    aligned = align_y_on_x_obs(
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
    assert aligned[1] == pytest.approx(5.0)
    # Linear interp between (1,3) and (4,5) at x=2 -> 3 + (1/3)*2 = 11/3.
    assert aligned[0] == pytest.approx(11.0 / 3.0)


def test_align_y_on_x_obs_monotone_decreasing() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs

    t_sim = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    x_model = np.asarray([3.0, 2.0, 1.0, 0.0], dtype=float)
    y_model = np.asarray([13.0, 12.0, 11.0, 10.0], dtype=float)  # y = 10 + x

    t_obs = np.asarray([0.0, 3.0], dtype=float)
    x_obs = np.asarray([1.5, 2.5], dtype=float)

    aligned = align_y_on_x_obs(
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
    assert aligned[0] == pytest.approx(11.5)
    assert aligned[1] == pytest.approx(12.5)


def test_align_y_on_x_obs_rejects_non_monotone_in_window() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs

    t_sim = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    x_model = np.asarray([0.0, 1.0, 0.5, 2.0], dtype=float)  # turns within window
    y_model = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)

    t_obs = np.asarray([0.0, 3.0], dtype=float)
    x_obs = np.asarray([1.0, 1.0], dtype=float)

    with pytest.raises(FitSimulationError) as exc_info:
        _ = align_y_on_x_obs(
            t_obs=t_obs,
            x_obs=x_obs,
            t_sim=t_sim,
            x_model=x_model,
            y_model=y_model,
            dataset_label="ds1",
            x_name="X",
            y_name="Y",
        )
    msg = str(exc_info.value)
    assert "monotone" in msg.lower()
    assert "t_min" in msg and "t_max" in msg


def test_align_y_on_x_obs_rejects_x_obs_out_of_range() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs

    t_sim = np.asarray([0.0, 1.0, 2.0], dtype=float)
    x_model = np.asarray([0.0, 1.0, 2.0], dtype=float)
    y_model = np.asarray([0.0, 10.0, 20.0], dtype=float)

    t_obs = np.asarray([0.0, 2.0], dtype=float)
    x_obs = np.asarray([3.0, 3.0], dtype=float)

    with pytest.raises(FitSimulationError) as exc_info:
        _ = align_y_on_x_obs(
            t_obs=t_obs,
            x_obs=x_obs,
            t_sim=t_sim,
            x_model=x_model,
            y_model=y_model,
            dataset_label="ds1",
            x_name="X",
            y_name="Y",
        )
    assert "outside" in str(exc_info.value).lower()


def test_align_y_on_x_obs_monotone_only_in_sampled_window() -> None:
    from kindred.core.analysis.parametric_alignment import align_y_on_x_obs

    t_sim = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    x_model = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)  # non-monotone globally
    y_model = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)

    # Window [0,2] -> x_model segment [0,1,2] is strictly increasing.
    t_obs = np.asarray([0.0, 1.0, 2.0], dtype=float)
    x_obs = np.asarray([0.0, 1.5, 2.0], dtype=float)

    aligned = align_y_on_x_obs(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        dataset_label="ds",
        x_name="X",
        y_name="Y",
    )
    assert aligned.shape == (3,)
    assert aligned[0] == pytest.approx(0.0)
    assert aligned[-1] == pytest.approx(20.0)
