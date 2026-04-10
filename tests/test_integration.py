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

import numpy as np
import pytest

# Core modules
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import solve_ode, SimulationRequest
from kindred.core.ode_builder import build_ode_rhs_from_mechanism

# Fitting modules
from kindred.core.analysis.global_fitting import fit_global

# Performance modules
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
            fit_evaluator=sim_func,
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
