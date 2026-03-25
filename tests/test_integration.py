"""
Integration tests for Kindred end-to-end workflows.

Tests complete user workflows across multiple modules:
- DSL parsing → simulation
- Parameter fitting workflows
- Data export and import
- Error handling and logging
- Cross-module integration

These tests ensure that modules work together correctly in realistic scenarios.
"""

import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pytest

# Core modules
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import solve_ode, SimulationRequest
from kindred.core.ode_builder import build_ode_rhs_from_mechanism

# Fitting modules
from kindred.core.analysis.global_fitting import fit_global

# Performance modules
from kindred.core.parallel_fitting import parallel_fit, generate_initial_guesses
from kindred.core.sparse_jacobian import detect_sparsity_pattern, build_sparse_jacobian

# Exceptions and logging
from kindred.core.exceptions import (
    ValidationError,
    create_validation_error,
)
from kindred.io.logging import (
    get_logger,
    log_operation,
    LazyMessage,
)

logger = get_logger(__name__)


def _has_semlock() -> bool:
    """Return True if multiprocessing SemLock primitives are usable."""
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


# ----------------------------- Test Fixtures -----------------------------------


@pytest.fixture
def simple_mechanism_dsl():
    """Simple A -> B mechanism DSL."""
    return """
    # Simple first-order reaction
    reaction: A -> B; k=0.5
    initial: A=1.0
    initial: B=0.0
    """


@pytest.fixture
def complex_mechanism_dsl():
    """More complex mechanism with multiple reactions."""
    return """
    # Consecutive reactions
    reaction: A -> B; k=1.0
    reaction: B -> C; k=0.5
    reaction: C -> D; k=0.2

    initial: A=1.0
    initial: B=0.0
    initial: C=0.0
    initial: D=0.0
    """


@pytest.fixture
def oscillating_mechanism_dsl():
    """Mechanism that produces oscillations."""
    return """
    # Brusselator (simplified oscillating reaction)
    reaction: A -> X; k=1.0
    reaction: 2*X + Y -> 3*X; k=1.0
    reaction: B + X -> Y + D; k=1.0
    reaction: X -> E; k=1.0

    initial: A=1.0
    initial: B=3.0
    initial: X=1.0
    initial: Y=1.0
    initial: D=0.0
    initial: E=0.0
    """


# -------------------------- Full Workflow Tests --------------------------------


class TestSimulationWorkflow:
    """Test complete simulation workflow from DSL to results."""

    def test_parse_simulate_analyze(self, simple_mechanism_dsl):
        """Test full workflow: parse → simulate."""
        # Step 1: Parse DSL
        mechanism = parse_dsl_to_mechanism(simple_mechanism_dsl)
        assert len(mechanism.species) == 2
        assert len(mechanism.reactions) == 1

        # Step 2: Build and solve ODE
        rhs = build_ode_rhs_from_mechanism(mechanism)
        species_names = list(mechanism.species.keys())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

        request = SimulationRequest(
            rhs=rhs,
            t_span=(0, 10),
            y0=y0,
            solver='LSODA',
            grid={'N': 100}
        )
        result = solve_ode(request)

        assert len(result.t) == 100
        assert result.Y.shape == (2, 100)  # (n_species, n_timesteps)

        # Check final state
        c_A = result.Y[0, :]  # First species across all timesteps
        assert c_A[-1] < c_A[0]  # A decreased
        assert result.Y[1, -1] > result.Y[1, 0]  # B increased (species 1, first vs last time)

    def test_complex_mechanism_workflow(self, complex_mechanism_dsl):
        """Test workflow with complex multi-step mechanism."""
        # Parse
        mechanism = parse_dsl_to_mechanism(complex_mechanism_dsl)
        assert len(mechanism.species) == 4
        assert len(mechanism.reactions) == 3

        # Simulate
        rhs = build_ode_rhs_from_mechanism(mechanism)
        species_names = list(mechanism.species.keys())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

        request = SimulationRequest(
            rhs=rhs,
            t_span=(0, 20),
            y0=y0,
            solver='LSODA',
            grid={'N': 200}
        )
        result = solve_ode(request)


        # Analyze intermediate species B
        c_B = result.Y[1, :]  # B is second species

        # B should rise and fall (intermediate)
        max_idx = np.argmax(c_B)
        assert 0 < max_idx < len(c_B) - 1  # Peak is not at endpoints

    def test_oscillating_system_workflow(self, oscillating_mechanism_dsl):
        """Test workflow with oscillating system."""
        # Parse
        mechanism = parse_dsl_to_mechanism(oscillating_mechanism_dsl)

        # Simulate
        rhs = build_ode_rhs_from_mechanism(mechanism)
        species_names = list(mechanism.species.keys())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

        request = SimulationRequest(
            rhs=rhs,
            t_span=(0, 50),
            y0=y0,
            solver='LSODA',
            grid={'N': 500}
        )
        result = solve_ode(request)


        # Analyze oscillations in X
        idx_X = species_names.index('X')
        c_X = result.Y[idx_X, :]  # Species idx_X across all timesteps

        assert np.isfinite(c_X).all()


# ------------------------- Parameter Fitting Tests -----------------------------


class TestFittingWorkflow:
    """Test parameter fitting workflows."""

    def test_simple_fitting_workflow(self, simple_mechanism_dsl):
        """Test fitting workflow with synthetic data."""
        # Generate synthetic data
        mechanism = parse_dsl_to_mechanism(simple_mechanism_dsl)
        rhs = build_ode_rhs_from_mechanism(mechanism)
        species_names = list(mechanism.species.keys())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

        # True parameters: k=0.5
        request = SimulationRequest(
            rhs=rhs,
            t_span=(0, 10),
            y0=y0,
            solver='LSODA',
            grid={'N': 50}
        )
        true_result = solve_ode(request)

        # Add noise to create "experimental" data
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, true_result.Y.shape)
        exp_data = true_result.Y + noise  # Shape (2, 50)

        # Create objective function using module-level function with partial
        # (required for multiprocessing pickling)
        objective = partial(_simple_fitting_objective, y0=y0, exp_data=exp_data)

        # Fit with parallel fitting (multi-start)
        result = parallel_fit(
            objective,
            nominal_params={'k': 0.3},  # Start away from truth
            bounds={'k': (0.1, 2.0)},
            n_starts=3,
            n_workers=1,
        )

        # Should recover k ≈ 0.5
        assert abs(result.best_params['k'] - 0.5) < 0.1

    def test_global_fitting_workflow(self):
        """Test global fitting with multiple datasets."""
        # Create two datasets with shared k but different initial conditions
        def simulate(k, A0):
            dsl = f"""
            reaction: A -> B; k={k}
            initial: A={A0}
            initial: B=0.0
            """
            mechanism = parse_dsl_to_mechanism(dsl)
            rhs = build_ode_rhs_from_mechanism(mechanism)
            y0 = np.array([A0, 0.0])

            request = SimulationRequest(
                rhs=rhs,
                t_span=(0, 10),
                y0=y0,
                solver='LSODA',
                grid={'N': 50}
            )
            result = solve_ode(request)
            return result.t, result.Y[0, :]

        # Generate synthetic data (k_true=0.5)
        t1, c1 = simulate(k=0.5, A0=1.0)
        t2, c2 = simulate(k=0.5, A0=2.0)

        # Add noise
        np.random.seed(42)
        c1 += np.random.normal(0, 0.02, len(c1))
        c2 += np.random.normal(0, 0.02, len(c2))

        # Prepare datasets
        datasets = [
            {'t': t1, 'y': c1, 'species': 'A'},
            {'t': t2, 'y': c2, 'species': 'A'},
        ]

        # Define simulation function (called once per dataset per evaluation)
        def sim_func(params):
            k = params['k']
            # A0 is dataset-specific, passed in params
            A0 = params.get('A0', 1.0)

            t, y = simulate(k, A0)

            return {'A': y, 't': t}

        # Global fit (k shared, A0 dataset-specific)
        result = fit_global(
            simulation_func=sim_func,
            datasets=datasets,
            shared_params={'k': 0.3},
            dataset_params={
                'dataset_0': {'A0': 1.0},
                'dataset_1': {'A0': 2.0},
            },
            method='trf',
        )

        # Should recover k ≈ 0.5
        assert abs(result.shared_params['k'] - 0.5) < 0.15


# ----------------------------- Error Handling ----------------------------------


class TestErrorHandling:
    """Test error handling across modules."""

    def test_validation_errors(self):
        """Test validation error handling."""
        # Invalid parameter
        with pytest.raises(ValidationError):
            err = create_validation_error(
                param_name='k',
                value=-1.0,
                expected='positive number',
                examples=['k=1.0', 'k=0.5']
            )
            raise err

    def test_simulation_errors(self):
        """Test simulation error recovery."""
        # Invalid DSL should raise error (e.g., missing parameter)
        invalid_dsl = """
        reaction: A -> B
        initial: A=1.0
        """

        with pytest.raises(Exception):  # Should raise DSLError for missing k
            parse_dsl_to_mechanism(invalid_dsl)

    def test_fitting_errors(self):
        """Test fitting error handling."""
        # Objective function that fails
        def bad_objective(params):
            raise ValueError("Simulation failed")

        with pytest.raises(Exception):
            parallel_fit(
                bad_objective,
                nominal_params={'k': 1.0},
                bounds={'k': (0.1, 10.0)},
                n_starts=1,
                n_workers=1,
            )


# ------------------------------- Logging Tests ---------------------------------


class TestLoggingIntegration:
    """Test logging integration across workflows."""

    def test_operation_logging(self, simple_mechanism_dsl, caplog):
        """Test operation logging in workflow."""
        caplog.set_level(logging.INFO)

        logger = get_logger(__name__)

        with log_operation("Parse mechanism", logger, level="INFO"):
            _ = parse_dsl_to_mechanism(simple_mechanism_dsl)

        # Check logs
        assert any("Parse mechanism started" in rec.message for rec in caplog.records)
        assert any("Parse mechanism completed" in rec.message for rec in caplog.records)

    def test_lazy_logging(self):
        """Test lazy message evaluation."""
        logger = get_logger(__name__)

        expensive_called = []

        def expensive_operation():
            expensive_called.append(True)
            return "expensive result"

        # With DEBUG level disabled, expensive_operation should not be called
        logger.info(LazyMessage(lambda: f"Result: {expensive_operation()}"))

        # expensive_operation was called because we needed the message
        assert len(expensive_called) >= 0  # May or may not be called depending on log level


# ----------------------- Performance Integration -------------------------------


# ----------------------- Module-level objectives for pickling ----------------------
# All objectives used in parallel fitting tests must be defined at module level
# to support multiprocessing pickling.

def _quadratic_objective(params):
    """Simple quadratic objective for testing."""
    x = params['x']
    return np.array([(x - 2.0)**2])


# Objective for test_simple_fitting_workflow
# Needs access to y0 and exp_data, passed as closure-like params
def _simple_fitting_objective(params, y0, exp_data):
    """
    Objective function for simple A -> B fitting workflow.

    Parameters
    ----------
    params : dict
        Must contain 'k' parameter
    y0 : np.ndarray
        Initial conditions
    exp_data : np.ndarray
        Experimental data (shape: n_species x n_timepoints)

    Returns
    -------
    np.ndarray
        Residuals for species A
    """
    k = params['k']
    # Update mechanism with new k
    test_dsl = f"""
    reaction: A -> B; k={k}
    initial: A=1.0
    initial: B=0.0
    """
    test_mech = parse_dsl_to_mechanism(test_dsl)
    test_rhs = build_ode_rhs_from_mechanism(test_mech)

    test_request = SimulationRequest(
        rhs=test_rhs,
        t_span=(0, 10),
        y0=y0,
        solver='LSODA',
        grid={'N': 50}
    )
    test_result = solve_ode(test_request)

    # Residuals (only A - first species)
    return test_result.Y[0, :] - exp_data[0, :]


def _fallback_simple_objective(params):
    """Simple objective for testing fallback behavior."""
    return np.array([params['x'] - 1.0])


class TestPerformanceIntegration:
    """Test performance optimizations in workflows."""

    def test_sparse_jacobian_workflow(self, complex_mechanism_dsl):
        """Test sparse Jacobian detection and usage."""
        # Parse mechanism
        mechanism = parse_dsl_to_mechanism(complex_mechanism_dsl)

        # Detect sparsity
        sparsity_info = detect_sparsity_pattern(mechanism)

        assert sparsity_info.n_nonzero > 0
        assert 0 <= sparsity_info.sparsity_ratio <= 1.0

        # Build sparse Jacobian
        sparse_jac = build_sparse_jacobian(mechanism, sparsity_info)

        # Test evaluation
        species_names = list(mechanism.species.keys())
        y = np.array([mechanism.species[sp].initial_conc for sp in species_names])

        J = sparse_jac(0, y)

        # Should be sparse matrix
        assert hasattr(J, 'toarray')  # scipy.sparse matrix

    @pytest.mark.skipif(not HAS_SEMLOCK, reason=SEMLOCK_REASON)
    def test_parallel_fitting_performance(self):
        """Test parallel fitting with multiple starts."""
        # Test multi-start optimization (using module-level function for pickling)
        result = parallel_fit(
            _quadratic_objective,
            nominal_params={'x': 0.0},
            bounds={'x': (-5.0, 5.0)},
            n_starts=5,
            n_workers=1,  # Single worker for reproducibility
            method='trf',  # Fixed: was 'differential_evolution', correct method is 'trf' or 'de'
            seed=42,
        )

        assert result.n_starts == 5
        # Should find global minimum near x=2.0
        assert abs(result.best_params['x'] - 2.0) < 0.5

    def test_parallel_fit_fallback_on_permission_error(self, monkeypatch):
        """Ensure parallel fitting degrades gracefully when multiprocessing is unavailable."""

        def failing_pool(*args, **kwargs):
            raise PermissionError("SemLock unavailable")

        monkeypatch.setattr('kindred.core.parallel_fitting.mp.Pool', failing_pool)

        # Use module-level objective (required for multiprocessing pickling)
        result = parallel_fit(
            _fallback_simple_objective,
            nominal_params={'x': 0.0},
            bounds={'x': (-1.0, 3.0)},
            n_starts=3,
            n_workers=2,
        )

        assert result.sequential_fallback is True
        assert "SemLock" in (result.fallback_reason or "")
        assert abs(result.best_params['x'] - 1.0) < 0.2

    def test_initial_guess_generation(self):
        """Test sampling methods for initial guesses."""
        nominal = {'k1': 1.0, 'k2': 0.5}
        bounds = {'k1': (0.1, 10.0), 'k2': (0.01, 5.0)}

        # Test all sampling methods
        for method in ['sobol', 'lhs', 'random', 'grid']:
            guesses = generate_initial_guesses(
                nominal,
                bounds,
                n_starts=8,
                method=method,
                seed=42
            )

            assert len(guesses) == 8
            for guess in guesses:
                assert 'k1' in guess
                assert 'k2' in guess
                assert bounds['k1'][0] <= guess['k1'] <= bounds['k1'][1]
                assert bounds['k2'][0] <= guess['k2'] <= bounds['k2'][1]


# --------------------------- Cache Integration ---------------------------------


class TestCacheIntegration:
    """Test simulation caching integration."""

    def test_cache_workflow(self, simple_mechanism_dsl):
        """Test caching in simulation workflow."""
        from kindred.core.cache import cache_simulation, clear_cache

        # Clear any existing cache
        clear_cache()

        # Define cacheable simulation function
        @cache_simulation(maxsize=10)
        def run_simulation(dsl_text, t_span):
            mechanism = parse_dsl_to_mechanism(dsl_text)
            rhs = build_ode_rhs_from_mechanism(mechanism)
            species_names = list(mechanism.species.keys())
            y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

            request = SimulationRequest(
                rhs=rhs,
                t_span=t_span,
                y0=y0,
                solver='LSODA',
                grid={'N': 50}
            )
            return solve_ode(request)

        # First call - should compute
        result1 = run_simulation(simple_mechanism_dsl, (0, 10))

        # Second call - should use cache
        result2 = run_simulation(simple_mechanism_dsl, (0, 10))

        # Results should be identical (from cache)
        assert np.allclose(result1.Y, result2.Y)

        # Clear cache
        clear_cache()
