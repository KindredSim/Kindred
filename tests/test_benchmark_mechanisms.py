from __future__ import annotations

from pathlib import Path

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism


pytestmark = pytest.mark.unit


BENCHMARK_MECHANISM_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "mechanisms"


@pytest.mark.parametrize("mechanism_path", sorted(BENCHMARK_MECHANISM_DIR.glob("*.txt")))
def test_benchmark_mechanism_fixtures_parse(mechanism_path):
    text = mechanism_path.read_text(encoding="utf-8")

    parse_dsl_to_mechanism(text, initials={})
