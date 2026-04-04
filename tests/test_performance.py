"""
Tests for performance optimization features (Sprint 2).

Tests cover:
- Sparse Jacobian detection and construction
- Benchmark mechanisms
"""

from pathlib import Path

import pytest

from kindred.core.sparse_jacobian import (
    detect_sparsity_pattern,
    estimate_sparsity_ratio,
    SparsityInfo,
)
from kindred.core.mechanism import Mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


BENCHMARK_REASON = "requires benchmark mechanism fixtures under benchmarks/mechanisms/*.txt"


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
