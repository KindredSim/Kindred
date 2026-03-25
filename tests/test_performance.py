"""
Tests for performance optimization features (Sprint 2).

Tests cover:
- Parallel fitting functionality
- Sparse Jacobian detection and construction
- Initial guess generation
- Benchmark mechanisms
"""

import multiprocessing as mp
from pathlib import Path

import pytest
import numpy as np

from kindred.core.parallel_fitting import (
    generate_initial_guesses,
    parallel_fit,
    WorkerPool,
)
from kindred.core.exceptions import OptimizationError
from kindred.core.sparse_jacobian import (
    detect_sparsity_pattern,
    estimate_sparsity_ratio,
    SparsityInfo,
)
from kindred.core.mechanism import Mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _has_semlock() -> bool:
    """Return True if multiprocessing can allocate SemLock primitives."""
    try:
        sem = mp.Semaphore(1)
        closer = getattr(sem, "close", None)
        if callable(closer):
            closer()
        return True
    except (OSError, AttributeError, RuntimeError, PermissionError):
        return False


HAS_SEMLOCK = _has_semlock()
SEMLOCK_REASON = "requires multiprocessing.SemLock support for parallel workers"
BENCHMARK_REASON = "requires benchmark mechanism fixtures under benchmarks/mechanisms/*.txt"


# ----------------------- Module-level functions for pickling ----------------------
# All functions used in WorkerPool.map/starmap and parallel_fit must be defined
# at module level to support multiprocessing pickling.

def _square(x):
    """Square function for test_worker_pool_map."""
    return x ** 2


def _add(x, y):
    """Add function for test_worker_pool_starmap."""
    return x + y


def _simple_quadratic_objective(params):
    """
    Simple quadratic objective for test_parallel_fit_simple.
    Minimizes (x - 3)^2 + (y - 4)^2
    """
    x = params[0]
    y = params[1]
    return np.array([x - 3.0, y - 4.0])


def _single_param_objective(params):
    """Simple objective for test_parallel_fit_all_methods."""
    return np.array([params[0] - 1.0])


def _dummy_objective_for_failure_test(vec):
    """Dummy objective that should work (but will be mocked to fail)."""
    return np.ones(1)


class TestInitialGuessGeneration:
    """Test initial guess generation for multi-start optimization."""

    def test_sobol_sampling(self):
        """Test Sobol sequence sampling."""
        nominal = {'k1': 1.0, 'k2': 0.5}
        bounds = {'k1': (0.1, 10.0), 'k2': (0.01, 5.0)}

        guesses = generate_initial_guesses(
            nominal, bounds, n_starts=10, method='sobol', seed=42
        )

        assert len(guesses) == 10
        assert all(isinstance(g, dict) for g in guesses)

        # Check all guesses are within bounds
        for guess in guesses:
            assert 0.1 <= guess['k1'] <= 10.0
            assert 0.01 <= guess['k2'] <= 5.0

    def test_random_sampling(self):
        """Test random uniform sampling."""
        nominal = {'k': 1.0}
        bounds = {'k': (0.1, 10.0)}

        guesses = generate_initial_guesses(
            nominal, bounds, n_starts=5, method='random', seed=42
        )

        assert len(guesses) == 5
        # First guess should be nominal
        assert guesses[0] == nominal

    def test_grid_sampling(self):
        """Test grid-based sampling."""
        nominal = {'k1': 1.0, 'k2': 1.0}
        bounds = {'k1': (0.0, 2.0), 'k2': (0.0, 2.0)}

        guesses = generate_initial_guesses(
            nominal, bounds, n_starts=9, method='grid'
        )

        # Grid sampling should produce points on regular grid
        assert len(guesses) >= 4  # At least 2x2 grid

    def test_lhs_sampling(self):
        """Test Latin Hypercube Sampling."""
        nominal = {'k1': 1.0, 'k2': 0.5}
        bounds = {'k1': (0.1, 10.0), 'k2': (0.01, 5.0)}

        guesses = generate_initial_guesses(
            nominal, bounds, n_starts=8, method='lhs', seed=42
        )

        assert len(guesses) == 8

        # Check space-filling property (rough test)
        k1_vals = [g['k1'] for g in guesses]
        k2_vals = [g['k2'] for g in guesses]

        # Should have good spread (coefficient of variation > 0.2)
        assert np.std(k1_vals) / np.mean(k1_vals) > 0.2
        assert np.std(k2_vals) / np.mean(k2_vals) > 0.2

    def test_generate_initial_guesses_requires_positive_starts(self):
        """Edge cases: zero starts or missing parameters must raise ValueError."""
        nominal = {'k': 1.0}
        bounds = {'k': (0.1, 1.0)}

        with pytest.raises(ValueError):
            generate_initial_guesses(nominal, bounds, n_starts=0)

        with pytest.raises(ValueError):
            generate_initial_guesses({}, bounds, n_starts=2)


@pytest.mark.skipif(not HAS_SEMLOCK, reason=SEMLOCK_REASON)
class TestWorkerPool:
    """Test worker pool management."""

    def test_worker_pool_context_manager(self):
        """Test worker pool with context manager."""
        with WorkerPool(n_workers=2) as pool:
            assert pool.pool is not None

        # Pool should be closed after context
        assert pool.pool is None

    def test_worker_pool_map(self):
        """Test parallel map operation."""
        with WorkerPool(n_workers=2) as pool:
            results = pool.map(_square, [1, 2, 3, 4, 5])

        assert results == [1, 4, 9, 16, 25]

    def test_worker_pool_starmap(self):
        """Test parallel starmap operation."""
        with WorkerPool(n_workers=2) as pool:
            results = pool.starmap(_add, [(1, 2), (3, 4), (5, 6)])

        assert results == [3, 7, 11]


class TestParallelFitFailures:
    """Edge cases for parallel fitting degradation paths."""

    def test_parallel_fit_raises_when_all_workers_fail(self, monkeypatch):
        """If every worker fails, OptimizationError should bubble up with context."""

        def _explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("kindred.core.parallel_fitting.fit_parameters", _explode)

        with pytest.raises(OptimizationError) as exc_info:
            parallel_fit(
                _dummy_objective_for_failure_test,
                nominal_params={'k': 0.0},
                bounds={'k': (-1.0, 1.0)},
                n_starts=3,
                n_workers=1,
            )

        assert "parallel fitting failed" in str(exc_info.value).lower()


class TestSparsityDetection:
    """Test sparse Jacobian sparsity pattern detection."""

    def test_small_mechanism_sparsity(self):
        """Test sparsity detection for small mechanism."""
        mech = Mechanism()
        mech.add_species('A', 1.0)
        mech.add_species('B', 0.0)
        mech.add_species('C', 0.0)

        # Linear chain: A -> B -> C
        mech.add_reaction({'A': -1.0, 'B': 1.0}, rate=1.0)
        mech.add_reaction({'B': -1.0, 'C': 1.0}, rate=0.5)

        info = detect_sparsity_pattern(mech)

        assert isinstance(info, SparsityInfo)
        assert len(info.species_names) == 3
        assert info.n_nonzero > 0
        assert 0.0 < info.sparsity_ratio <= 1.0

    def test_dense_mechanism_sparsity(self):
        """Test sparsity for fully connected mechanism."""
        mech = Mechanism()
        mech.add_species('A', 1.0)
        mech.add_species('B', 0.0)

        # All species interact
        mech.add_reaction({'A': -1.0, 'B': 1.0}, rate=1.0)
        mech.add_reaction({'B': -1.0, 'A': 1.0}, rate=0.5)

        info = detect_sparsity_pattern(mech)

        # Should be dense (all elements nonzero)
        assert info.sparsity_ratio == 1.0

    def test_sparsity_estimation(self):
        """Test quick sparsity estimation."""
        mech = Mechanism()
        for i in range(10):
            mech.add_species(f'S{i}', 0.0 if i > 0 else 1.0)

        # Add linear chain reactions
        for i in range(9):
            mech.add_reaction({f'S{i}': -1.0, f'S{i+1}': 1.0}, rate=1.0)

        ratio = estimate_sparsity_ratio(mech)

        # Should be sparse (much less than 1.0)
        assert 0.0 < ratio < 0.5

    def test_coupling_graph(self):
        """Test species coupling graph construction."""
        mech = Mechanism()
        mech.add_species('A', 1.0)
        mech.add_species('B', 0.0)
        mech.add_species('C', 0.0)

        mech.add_reaction({'A': -1.0, 'B': 1.0}, rate=1.0)

        info = detect_sparsity_pattern(mech)

        # A and B should be coupled
        assert 'B' in info.coupling_graph['A']
        assert 'A' in info.coupling_graph['B']

        # C should only be coupled with itself (diagonal)
        assert info.coupling_graph['C'] == {'C'}


class TestBenchmarkMechanisms:
    """Test benchmark mechanisms load and simulate correctly."""

    def test_small_mechanism_loads(self):
        """Test that small benchmark mechanism loads."""
        mech_file = Path(__file__).parent.parent / "benchmarks" / "mechanisms" / "small.txt"

        if not mech_file.exists():
            pytest.skip(BENCHMARK_REASON)

        with open(mech_file) as f:
            dsl_text = f.read()

        mech = parse_dsl_to_mechanism(dsl_text, initials={})

        assert len(mech.species_names()) == 5
        assert len(mech.reactions) == 4

    def test_medium_mechanism_loads(self):
        """Test that medium benchmark mechanism loads."""
        mech_file = Path(__file__).parent.parent / "benchmarks" / "mechanisms" / "medium.txt"

        if not mech_file.exists():
            pytest.skip(BENCHMARK_REASON)

        with open(mech_file) as f:
            dsl_text = f.read()

        mech = parse_dsl_to_mechanism(dsl_text, initials={})

        assert len(mech.species_names()) == 25
        # Should have reactions and equilibria
        assert len(mech.reactions) + len(mech.equilibria) > 30

    def test_large_mechanism_loads(self):
        """Test that large benchmark mechanism loads."""
        mech_file = Path(__file__).parent.parent / "benchmarks" / "mechanisms" / "large.txt"

        if not mech_file.exists():
            pytest.skip(BENCHMARK_REASON)

        with open(mech_file) as f:
            dsl_text = f.read()

        mech = parse_dsl_to_mechanism(dsl_text, initials={})

        assert len(mech.species_names()) >= 100
        # Should be a large network
        assert len(mech.reactions) > 100


class TestParallelFitting:
    """Test parallel multi-start fitting."""

    def test_parallel_fit_simple(self):
        """Test parallel fitting with simple quadratic objective."""
        # Objective: minimize (x - 3)^2 + (y - 4)^2
        # Uses module-level function for multiprocessing pickling
        nominal = {'x': 0.0, 'y': 0.0}
        bounds = {'x': (-10.0, 10.0), 'y': (-10.0, 10.0)}

        result = parallel_fit(
            _simple_quadratic_objective,
            nominal,
            bounds,
            n_starts=4,
            n_workers=2,
            max_nfev=100,
            seed=42,
        )

        assert result.n_starts == 4
        assert result.n_success > 0
        assert result.best_result is not None

        # Best result should be close to (3, 4)
        x_opt = result.best_result.parameters['x']
        y_opt = result.best_result.parameters['y']

        assert 2.5 < x_opt < 3.5
        assert 3.5 < y_opt < 4.5

    def test_parallel_fit_all_methods(self):
        """Test all sampling methods work."""
        # Uses module-level function for multiprocessing pickling
        nominal = {'k': 0.5}
        bounds = {'k': (0.1, 2.0)}

        for method in ['sobol', 'random', 'lhs', 'grid']:
            result = parallel_fit(
                _single_param_objective,
                nominal,
                bounds,
                n_starts=3,
                sampling_method=method,
                n_workers=1,
                max_nfev=50,
                seed=42,
            )

            assert result.n_starts == 3
            assert len(result.all_results) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sparse_jacobian_empty_mechanism(self):
        """Test sparsity detection for empty mechanism."""
        mech = Mechanism()

        info = detect_sparsity_pattern(mech)

        assert info.n_nonzero == 0
        assert info.sparsity_ratio == 0.0

    def test_single_species_mechanism(self):
        """Test mechanisms with single species."""
        mech = Mechanism()
        mech.add_species('A', 1.0)

        info = detect_sparsity_pattern(mech)

        # Single species, only diagonal element
        assert info.n_nonzero == 1
        assert info.sparsity_ratio == 1.0

    def test_initial_guesses_single_param(self):
        """Test initial guess generation with single parameter."""
        nominal = {'k': 1.0}
        bounds = {'k': (0.1, 10.0)}

        guesses = generate_initial_guesses(nominal, bounds, n_starts=5, seed=42)

        assert len(guesses) == 5
        assert all('k' in g for g in guesses)
