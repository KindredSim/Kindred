import math

import numpy as np
import pytest

from kindred.core.results import integrate_ctc
from kindred.core.time_grid import build_time_grid, is_uniform_time_grid

pytestmark = pytest.mark.unit



def _make_jitter_grid(delta: float) -> np.ndarray:
    dt = np.array([1.0, 1.0 + delta, 1.0, 1.0 - delta, 1.0], dtype=float)
    return np.concatenate([[0.0], np.cumsum(dt)])


def test_build_time_grid_dt_non_integer_division_hits_t1_exactly() -> None:
    t0, t1 = 0.0, 1.0
    dt = 0.3
    t = build_time_grid(t0, t1, {"dt": dt})

    assert t[0] == t0
    assert t[-1] == t1
    assert np.all(np.diff(t) > 0)

    n_int = max(1, int(math.ceil((t1 - t0) / dt)))
    assert t.size == n_int + 1
    h_adj = (t1 - t0) / n_int
    assert np.allclose(np.diff(t), h_adj)


def test_build_time_grid_N_mode_matches_linspace() -> None:
    t = build_time_grid(2.0, 5.0, {"N": 4})
    assert np.allclose(t, np.linspace(2.0, 5.0, 4, dtype=float))
    assert np.all(np.diff(t) > 0)


def test_is_uniform_time_grid_median_relative_threshold() -> None:
    eps = 1e-6
    assert is_uniform_time_grid(_make_jitter_grid(5e-7), eps)
    assert not is_uniform_time_grid(_make_jitter_grid(2e-6), eps)


def test_integrate_ctc_simpson13_on_uniform_even_intervals() -> None:
    t = np.linspace(0.0, 1.0, 5)
    y = t**2
    val, method, uniform, _used_eps, _used_tail = integrate_ctc(t, y, uniformity_eps=1e-6)

    assert uniform is True
    assert method == "Simpson13"
    assert val == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_integrate_ctc_trapezoid_on_nonuniform() -> None:
    t = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)
    y = np.ones_like(t)
    val, method, uniform, _used_eps, _used_tail = integrate_ctc(t, y, uniformity_eps=1e-6)

    assert uniform is False
    assert method == "Trapezoidal"
    assert val == pytest.approx(1.0, abs=1e-12)


def test_integrate_ctc_uniformity_threshold_controls_method() -> None:
    eps = 1e-6

    t_uniformish = _make_jitter_grid(5e-7)
    y1 = np.ones_like(t_uniformish)
    val1, method1, uniform1, _used_eps, _used_tail = integrate_ctc(t_uniformish, y1, uniformity_eps=eps)
    assert uniform1 is True
    assert method1 == "Simpson13+38"
    assert val1 == pytest.approx(float(t_uniformish[-1] - t_uniformish[0]), abs=2e-6)

    t_nonuniform = _make_jitter_grid(2e-6)
    y2 = np.ones_like(t_nonuniform)
    _val2, method2, uniform2, _used_eps2, _used_tail2 = integrate_ctc(t_nonuniform, y2, uniformity_eps=eps)
    assert uniform2 is False
    assert method2 == "Trapezoidal"
