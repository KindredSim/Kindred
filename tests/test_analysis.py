"""
Tests for data analysis module.

Tests cover:
- Global fitting (multi-dataset parameter estimation)
- Caching functionality
"""

import pytest
import numpy as np
from kindred.core.analysis.global_fitting import fit_global, GlobalFitResult, DatasetFitInfo
from kindred.core.cache import (
    generate_mechanism_hash,
    cache_simulation,
    get_cache_stats,
    clear_cache,
)
from kindred.core.mechanism import Mechanism


class TestGlobalFitting:
    """Test global fitting across multiple datasets."""

    def test_simple_global_fit(self):
        """Test fitting single parameter to two datasets."""
        # Simulation function: exponential decay with rate k
        def simulate(params):
            k = params['k']
            t = np.linspace(0, 10, 50)
            c = np.exp(-k * t)
            return {'t': t, 'A': c}

        # Create two synthetic datasets with k=0.5
        true_k = 0.5
        t1 = np.linspace(0, 10, 50)
        y1 = np.exp(-true_k * t1) + np.random.normal(0, 0.01, 50)

        t2 = np.linspace(0, 15, 60)
        y2 = np.exp(-true_k * t2) + np.random.normal(0, 0.01, 60)

        datasets = [
            {'id': 'exp1', 't': t1, 'y': y1, 'species': 'A'},
            {'id': 'exp2', 't': t2, 'y': y2, 'species': 'A'},
        ]

        shared_params = {'k': 0.3}  # Initial guess
        bounds = {'k': (0.01, 2.0)}

        result = fit_global(simulate, datasets, shared_params, bounds=bounds, max_nfev=500)

        assert isinstance(result, GlobalFitResult)
        assert result.completion.status == 'ok'
        assert 0.45 < result.shared_params['k'] < 0.55  # Close to true value 0.5
        assert result.global_r_squared > 0.9
        assert len(result.dataset_info) == 2

        # Check per-dataset statistics
        for info in result.dataset_info:
            assert isinstance(info, DatasetFitInfo)
            assert info.r_squared > 0.8
            assert info.chi_squared < 0.01

    def test_global_fit_with_weights(self):
        """Test weighted global fitting."""
        def simulate(params):
            k = params['k']
            t = np.linspace(0, 10, 50)
            c = np.exp(-k * t)
            return {'t': t, 'A': c}

        true_k = 0.5
        t1 = np.linspace(0, 10, 50)
        y1 = np.exp(-true_k * t1) + np.random.normal(0, 0.01, 50)

        t2 = np.linspace(0, 10, 50)
        y2 = np.exp(-true_k * t2) + np.random.normal(0, 0.05, 50)  # Noisier

        datasets = [
            {'id': 'exp1', 't': t1, 'y': y1, 'species': 'A'},
            {'id': 'exp2', 't': t2, 'y': y2, 'species': 'A'},
        ]

        # Weight first dataset more (less noisy)
        weights = {'exp1': 2.0, 'exp2': 1.0}

        result = fit_global(
            simulate, datasets, {'k': 0.3},
            bounds={'k': (0.01, 2.0)},
            weights=weights,
            max_nfev=500
        )

        assert result.completion.status == 'ok'
        assert 0.4 < result.shared_params['k'] < 0.6

    def test_global_fit_differential_evolution(self):
        """Global fitting should support differential evolution."""
        def simulate(params):
            k = params['k']
            t = np.linspace(0, 8, 40)
            return {'t': t, 'A': np.exp(-k * t)}

        true_k = 0.45
        t = np.linspace(0, 8, 40)
        y = np.exp(-true_k * t)
        datasets = [{'id': 'exp', 't': t, 'y': y, 'species': 'A'}]

        result = fit_global(
            simulate,
            datasets,
            {'k': 0.2},
            bounds={'k': (0.05, 1.0)},
            method="differential_evolution",
            max_nfev=120,
            seed=7,
        )

        assert result.completion.status == 'ok'
        assert 0.4 < result.shared_params['k'] < 0.5
        assert result.objective_residuals is not None

    def test_global_fit_weights_bias_result(self):
        """Heavier weights bias the solution toward preferred dataset."""
        def simulate(params):
            k = params['k']
            t = np.linspace(0, 5, 40)
            return {'t': t, 'A': np.exp(-k * t)}

        datasets = [
            {'id': 'slow', 't': np.linspace(0, 5, 40), 'y': np.exp(-0.3 * np.linspace(0, 5, 40)), 'species': 'A'},
            {'id': 'fast', 't': np.linspace(0, 5, 40), 'y': np.exp(-0.9 * np.linspace(0, 5, 40)), 'species': 'A'},
        ]

        unweighted = fit_global(simulate, datasets, {'k': 0.5})
        weighted = fit_global(
            simulate,
            datasets,
            {'k': 0.5},
            weights={'slow': 5.0, 'fast': 1.0},
        )

        assert abs(weighted.shared_params['k'] - 0.3) < abs(unweighted.shared_params['k'] - 0.3)

    def test_global_fit_target_weights_bias_result_within_dataset(self):
        """Per-target weights should bias a multi-target dataset toward the preferred target."""

        def simulate(params):
            k = params['k']
            t = np.linspace(0, 5, 40)
            return {
                't': t,
                'species': {
                    'A': np.exp(-k * t),
                    'B': np.exp(-2.0 * k * t),
                },
            }

        t = np.linspace(0, 5, 40)
        datasets_unweighted = [
            {
                'id': 'mixed',
                't': t,
                'y': np.vstack([
                    np.exp(-0.3 * t),
                    np.exp(-2.0 * 0.9 * t),
                ]),
                'species': ['A', 'B'],
            }
        ]
        datasets_weighted = [
            dict(datasets_unweighted[0], target_weights={'A': 5.0, 'B': 1.0})
        ]

        unweighted = fit_global(simulate, datasets_unweighted, {'k': 0.5})
        weighted = fit_global(simulate, datasets_weighted, {'k': 0.5})

        assert abs(weighted.shared_params['k'] - 0.3) < abs(unweighted.shared_params['k'] - 0.3)

    def test_global_fit_with_dataset_variable_initials(self):
        """Fit per-dataset initial concentrations with bounds."""
        def simulate(params):
            k = params['k']
            init_a = params.get('init:A', 1.0)
            t = np.linspace(0, 5, 40)
            return {'t': t, 'A': init_a * np.exp(-k * t)}

        t = np.linspace(0, 5, 40)
        datasets = [
            {'id': 'ds1', 't': t, 'y': 1.0 * np.exp(-0.4 * t), 'species': 'A'},
            {'id': 'ds2', 't': t, 'y': 2.0 * np.exp(-0.4 * t), 'species': 'A'},
        ]

        dataset_variable_params = {
            'ds1': {'init:A': {'initial': 0.5, 'min': 0.1, 'max': 3.0}},
            'ds2': {'init:A': {'initial': 0.5, 'min': 0.1, 'max': 5.0}},
        }

        result = fit_global(
            simulate,
            datasets,
            {'k': 0.2},
            dataset_variable_params=dataset_variable_params,
        )

        assert result.completion.status == 'ok'
        assert pytest.approx(result.shared_params['k'], rel=1e-2) == 0.4
        assert pytest.approx(result.dataset_params['ds1']['init:A'], rel=1e-2) == 1.0
        assert pytest.approx(result.dataset_params['ds2']['init:A'], rel=1e-2) == 2.0

    def test_global_fit_multi_species_dataset(self):
        """Ensure multi-species payloads contribute all residuals."""
        sim_time = np.linspace(0, 5, 200)

        def simulate(params):
            k = params['k']
            a = np.exp(-k * sim_time)
            b = 0.5 * np.exp(-2 * k * sim_time)
            return {'t': sim_time, 'species': {'A': a, 'B': b}}

        true_k = 0.4
        t_dataset = np.linspace(0, 5, 40)
        y_matrix = np.vstack([
            np.exp(-true_k * t_dataset),
            0.5 * np.exp(-2 * true_k * t_dataset),
        ])

        datasets = [{
            'id': 'multi',
            't': t_dataset,
            'y': y_matrix,
            'species': ['A', 'B'],
        }]

        result = fit_global(
            simulate,
            datasets,
            {'k': 0.2},
            max_nfev=200,
        )

        assert result.completion.status == 'ok'
        assert pytest.approx(result.shared_params['k'], rel=1e-2) == true_k
        assert result.dataset_info[0].n_points == t_dataset.size * 2
        assert result.dataset_info[0].residuals.shape[0] == t_dataset.size * 2

    def test_global_fit_allows_different_species_per_dataset(self):
        """Datasets can request different species combinations during fitting."""
        t_axis = np.linspace(0, 4, 60)

        def simulate(params):
            k = params['k']
            a = np.exp(-k * t_axis)
            b = 0.5 * np.exp(-0.5 * k * t_axis)
            return {'t': t_axis, 'species': {'A': a, 'B': b}}

        true_k = 0.35
        y_a = np.exp(-true_k * t_axis)
        y_b = 0.5 * np.exp(-0.5 * true_k * t_axis)
        datasets = [
            {'id': 'fit_A', 't': t_axis, 'y': y_a, 'species': 'A'},
            {'id': 'fit_B', 't': t_axis, 'y': y_b, 'species': 'B'},
            {'id': 'fit_both', 't': t_axis, 'y': np.vstack([y_a, y_b]), 'species': ['A', 'B']},
        ]

        result = fit_global(
            simulate,
            datasets,
            {'k': 0.2},
            max_nfev=200,
        )

        assert result.completion.status == 'ok'
        assert pytest.approx(result.shared_params['k'], rel=1e-2) == true_k
        info_map = {info.dataset_id: info for info in result.dataset_info}
        assert info_map['fit_A'].n_points == t_axis.size
        assert info_map['fit_B'].n_points == t_axis.size
        assert info_map['fit_both'].n_points == t_axis.size * 2

    def test_global_fit_exposes_series_and_objective(self):
        """Result should include model/residual series and objective residuals."""
        time_axis = np.linspace(0, 4, 30)

        def simulate(params):
            k = params['k']
            return {'t': time_axis, 'A': np.exp(-k * time_axis)}

        datasets = [
            {'id': 'exp1', 't': time_axis, 'y': np.exp(-0.4 * time_axis), 'species': 'A'},
            {'id': 'exp2', 't': time_axis, 'y': np.exp(-0.45 * time_axis), 'species': 'A'},
        ]

        result = fit_global(
            simulate,
            datasets,
            {'k': 0.2},
            max_nfev=200,
        )

        assert result.objective_residuals is not None
        total_points = sum(info.n_points for info in result.dataset_info)
        assert len(result.objective_residuals) == total_points
        for ds in datasets:
            ds_id = ds['id']
            assert ds_id in result.model_series
            model = result.model_series[ds_id]['A']
            assert model.shape == ds['y'].shape
            residuals = result.residual_series[ds_id]['A']
            np.testing.assert_allclose(residuals, model - ds['y'])

    def test_global_fit_no_datasets(self):
        """Test that empty dataset list raises error."""
        def simulate(params):
            return {'A': np.array([1.0])}

        with pytest.raises(ValueError, match="At least one dataset"):
            fit_global(simulate, [], {'k': 1.0})

    def test_fit_global_does_not_mutate_input_datasets(self):
        """fit_global must not mutate caller-provided dataset dicts (e.g., by injecting ids)."""
        def simulate(_params):
            return {"A": np.array([1.0])}

        datasets = [{"t": np.array([0.0]), "y": np.array([1.0]), "species": "A"}]
        original_keys = [set(ds.keys()) for ds in datasets]

        with pytest.raises(ValueError, match="Unknown optimization method"):
            fit_global(simulate, datasets, {"k": 1.0}, method="not_a_method")

        assert [set(ds.keys()) for ds in datasets] == original_keys

    @pytest.mark.parametrize("method", ["trf", "dogbox"])
    def test_x0_on_lower_bound_no_infeasible_error(self, method):
        """x0 sitting exactly on the lower bound must not crash dogbox/trf."""
        def simulate(params):
            k = params['k']
            t = np.linspace(0, 5, 30)
            return {'t': t, 'A': np.exp(-k * t)}

        true_k = 0.5
        t = np.linspace(0, 5, 30)
        y = np.exp(-true_k * t)
        datasets = [{'id': 'exp', 't': t, 'y': y, 'species': 'A'}]

        # Initial guess exactly on the lower bound
        result = fit_global(
            simulate,
            datasets,
            {'k': 0.1},
            bounds={'k': (0.1, 2.0)},
            method=method,
            max_nfev=200,
        )
        assert result.completion.status == 'ok'


class TestCaching:
    """Test simulation caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_mechanism_hash_deterministic(self):
        """Test that mechanism hash is deterministic."""
        mech1 = Mechanism()
        mech1.add_species('A', 1.0)
        mech1.add_species('B', 0.0)

        mech2 = Mechanism()
        mech2.add_species('A', 1.0)
        mech2.add_species('B', 0.0)

        hash1 = generate_mechanism_hash(mech1)
        hash2 = generate_mechanism_hash(mech2)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_mechanism_hash_different(self):
        """Test that different mechanisms produce different hashes."""
        mech1 = Mechanism()
        mech1.add_species('A', 1.0)

        mech2 = Mechanism()
        mech2.add_species('A', 2.0)  # Different initial concentration

        hash1 = generate_mechanism_hash(mech1)
        hash2 = generate_mechanism_hash(mech2)

        assert hash1 != hash2

    def test_cache_decorator_basic(self):
        """Test basic cache decorator functionality."""
        call_count = [0]

        @cache_simulation(maxsize=128)
        def expensive_func(mechanism, value):
            call_count[0] += 1
            return value ** 2

        mech = Mechanism()
        mech.add_species('A', 1.0)

        # First call: cache miss
        result1 = expensive_func(mech, 5)
        assert result1 == 25
        assert call_count[0] == 1

        # Second call with same args: cache hit
        result2 = expensive_func(mech, 5)
        assert result2 == 25
        assert call_count[0] == 1  # Not called again

        # Call with different args: cache miss
        result3 = expensive_func(mech, 10)
        assert result3 == 100
        assert call_count[0] == 2

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        clear_cache()

        @cache_simulation(maxsize=128)
        def dummy_func(mechanism, x):
            return x * 2

        mech = Mechanism()
        mech.add_species('A', 1.0)

        # Generate some hits and misses
        dummy_func(mech, 1)  # Miss
        dummy_func(mech, 1)  # Hit
        dummy_func(mech, 2)  # Miss
        dummy_func(mech, 1)  # Hit

        stats = get_cache_stats()
        assert stats.hits >= 2
        assert stats.misses >= 2
        assert 0 < stats.hit_rate < 1

    def test_clear_cache(self):
        """Test cache clearing."""
        @cache_simulation(maxsize=128)
        def dummy_func(mechanism, x):
            return x * 2

        mech = Mechanism()
        mech.add_species('A', 1.0)

        dummy_func(mech, 1)
        _ = get_cache_stats()

        clear_cache()
        stats_after = get_cache_stats()

        # Stats should be reset
        assert stats_after.hits == 0
        assert stats_after.misses == 0

    def test_equivalent_mechanisms_share_cache_entries(self):
        """Mechanisms with identical structure should hit the same cache slot."""
        call_count = [0]

        @cache_simulation(maxsize=16)
        def expensive(mechanism, scale):
            call_count[0] += 1
            return mechanism.species_names(), scale * 2

        mech1 = Mechanism()
        mech1.add_species('A', 1.0)

        mech2 = Mechanism()
        mech2.add_species('A', 1.0)

        first = expensive(mech1, 5)
        assert first[1] == 10
        assert call_count[0] == 1

        second = expensive(mech2, 5)
        assert second == first
        assert call_count[0] == 1  # cache hit despite different object instances

    def test_clear_cache_invalidates_wrapped_functions(self):
        """Ensure clear_cache flushes entries held by decorated callables."""
        call_count = [0]

        @cache_simulation(maxsize=4)
        def cached_func(mechanism, value):
            call_count[0] += 1
            return value ** 2

        mech = Mechanism()
        mech.add_species('A', 1.0)

        cached_func(mech, 3)
        cached_func(mech, 3)
        assert call_count[0] == 1

        clear_cache()

        cached_func(mech, 3)
        assert call_count[0] == 2  # cache was flushed


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_global_fit_simulation_failure(self):
        """Test global fit when simulation raises exception."""
        def broken_simulate(params):
            raise RuntimeError("Simulation failed")

        datasets = [
            {'id': 'exp1', 't': np.array([0, 1]), 'y': np.array([1, 0.9]), 'species': 'A'}
        ]

        result = fit_global(broken_simulate, datasets, {'k': 1.0}, max_nfev=10)

        # Should handle gracefully
        assert result.completion.status == 'fail' or result.global_chi_squared == np.inf
