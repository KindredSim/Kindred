"""
Tests for performance optimization features (Sprint 2).

Tests cover:
- Benchmark mechanisms
"""

from pathlib import Path

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism

pytestmark = pytest.mark.unit



BENCHMARK_REASON = "requires benchmark mechanism fixtures under benchmarks/mechanisms/*.txt"

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
