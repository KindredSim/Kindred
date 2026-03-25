"""
Test parallel fitting with proper picklable objective functions.

This module demonstrates and tests the correct way to use parallel_fit
with objective functions that can be pickled for multiprocessing.

Local (non-picklable) functions will be forced into sequential execution.
Module-level objective functions are still required for true parallelism.
"""

from __future__ import annotations

import pytest
import numpy as np
from kindred.core.parallel_fitting import parallel_fit

# Check for multiprocessing support
def _has_semlock():
    try:
        import multiprocessing
        return hasattr(multiprocessing, 'SemLock')
    except (ImportError, OSError):
        return False

HAS_SEMLOCK = _has_semlock()
SEMLOCK_REASON = "requires multiprocessing.SemLock support for parallel workers"


# Module-level objective functions (picklable)
def _simple_quadratic(params):
    """Simple quadratic objective for testing (x - 2)^2."""
    x = params.get('x', params.get(0, 0.0))
    return np.array([x - 2.0])


def _rosenbrock_2d(params):
    """Rosenbrock function for testing."""
    x = params.get('x', params.get(0, 0.0))
    y = params.get('y', params.get(1, 0.0))
    return np.array([
        10 * (y - x**2),
        1 - x
    ])


def _noisy_exponential(params):
    """Exponential decay with noise."""
    k = params.get('k', 1.0)
    t = np.linspace(0, 5, 50)
    y_true = np.exp(-k * t)
    # Add fixed noise for reproducibility
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, len(t))
    y_obs = y_true + noise
    y_pred = np.exp(-k * t)
    return y_pred - y_obs


class TestParallelFittingPickling:
    """Test parallel fitting with picklable objective functions."""

    @pytest.mark.skipif(not HAS_SEMLOCK, reason=SEMLOCK_REASON)
    def test_parallel_fit_with_module_level_function(self):
        """Test that module-level functions work with parallel_fit."""
        result = parallel_fit(
            _simple_quadratic,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=3,
            n_workers=1,  # Use 1 worker for deterministic testing
            seed=42,
        )

        # Should find minimum near x=2
        assert result.success
        assert abs(result.best_params['x'] - 2.0) < 0.1
        assert len(result.all_results) == 3

    @pytest.mark.skipif(not HAS_SEMLOCK, reason=SEMLOCK_REASON)
    def test_parallel_fit_with_multiple_parameters(self):
        """Test parallel fit with multiple parameters."""
        result = parallel_fit(
            _rosenbrock_2d,
            nominal_params={'x': 0.0, 'y': 0.0},
            bounds={'x': (-2.0, 2.0), 'y': (-2.0, 2.0)},
            n_starts=5,
            n_workers=1,
            max_nfev=500,
            seed=42,
        )

        # Rosenbrock minimum is at (1, 1)
        assert result.success
        assert abs(result.best_params['x'] - 1.0) < 0.2
        assert abs(result.best_params['y'] - 1.0) < 0.2

    @pytest.mark.skipif(not HAS_SEMLOCK, reason=SEMLOCK_REASON)
    def test_parallel_fit_different_sampling_methods(self):
        """Test all sampling methods with picklable function."""
        for method in ['sobol', 'random', 'lhs', 'grid']:
            result = parallel_fit(
                _simple_quadratic,
                nominal_params={'x': 0.0},
                bounds={'x': (-5.0, 5.0)},
                n_starts=4,
                sampling_method=method,
                n_workers=1,
                max_nfev=100,
                seed=42,
            )
            assert result.success, f"Method {method} failed"
            assert abs(result.best_params['x'] - 2.0) < 0.2, f"Method {method} found wrong minimum"

    def test_parallel_fit_with_local_function_forces_sequential(self):
        """
        Local functions are unpicklable; parallel_fit should force sequential mode.
        """
        offset = 1.0

        def local_objective(params):
            """This local function cannot be pickled."""
            return np.array([params['x'] - offset])

        result = parallel_fit(
            local_objective,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=2,
            n_workers=2,  # Request parallel, should fall back deterministically
            max_nfev=50,
            seed=0,
        )

        assert result.success
        assert result.sequential_fallback is True
        assert result.parallel_effective is False
        assert result.parallel_disabled_reason
        assert "pickl" in result.parallel_disabled_reason.lower()

    def test_parallel_fit_fallback_to_sequential(self):
        """
        Test that parallel_fit can fall back to sequential mode.

        When n_workers=1 or multiprocessing is unavailable, parallel_fit
        should use a sequential fallback that CAN handle local functions.
        """
        def local_objective(params):
            """Local function works in sequential mode."""
            return np.array([params['x'] - 3.0])

        # With n_workers=1, should use sequential mode (no pickling required)
        result = parallel_fit(
            local_objective,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=3,
            n_workers=1,  # Sequential mode
            max_nfev=100,
            seed=42,
        )

        assert result.success
        assert abs(result.best_params['x'] - 3.0) < 0.2


class TestParallelFittingRobustness:
    """Test parallel fitting edge cases and error handling."""

    def test_parallel_fit_with_no_bounds(self):
        """Test that parallel_fit works without explicit bounds."""
        result = parallel_fit(
            _simple_quadratic,
            nominal_params={'x': 0.0},
            bounds=None,  # No bounds
            n_starts=3,
            n_workers=1,
            max_nfev=100,
            seed=42,
        )
        # Should still work, using default bounds
        assert result.success

    def test_parallel_fit_with_single_start(self):
        """Test parallel fit with only one starting point."""
        result = parallel_fit(
            _simple_quadratic,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=1,  # Single start
            n_workers=1,
            max_nfev=100,
        )
        assert result.success
        assert len(result.all_results) == 1

    @pytest.mark.skipif(not HAS_SEMLOCK, reason="multiprocessing.SemLock unavailable")
    def test_parallel_fit_convergence_statistics(self):
        """Test that parallel fit provides convergence statistics."""
        result = parallel_fit(
            _simple_quadratic,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=5,
            n_workers=1,
            seed=42,
        )

        # Check result structure
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_cost')
        assert hasattr(result, 'all_results')
        assert hasattr(result, 'success')

        # Should have results from all starts
        assert len(result.all_results) == 5

        # All individual results should have timing info
        for res in result.all_results:
            assert hasattr(res, 'duration')


class TestObjectiveFunctionRequirements:
    """Document and test requirements for objective functions."""

    def test_objective_must_accept_dict(self):
        """Objective functions must accept parameter dict."""
        params = {'x': 1.5, 'y': 2.3}
        result = _rosenbrock_2d(params)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_objective_must_return_array(self):
        """Objective functions must return numpy array."""
        params = {'x': 1.0}
        result = _simple_quadratic(params)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_objective_should_handle_missing_keys(self):
        """Objective functions should handle missing parameter keys gracefully."""
        # Using dict.get() with defaults
        params = {}  # Empty dict
        result = _simple_quadratic(params)
        # Should not crash, should use default value 0.0
        assert isinstance(result, np.ndarray)
