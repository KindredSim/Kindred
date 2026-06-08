#!/usr/bin/env python
"""
Benchmark runner for Kindred performance testing.

Runs performance benchmarks on mechanisms of varying sizes to:
- Measure simulation time and memory usage
- Test scalability with mechanism size
- Detect performance regressions

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --verbose
    python benchmarks/run_benchmarks.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.solvers import solve_ode, SimulationRequest

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    mechanism_name: str
    n_species: int
    n_reactions: int
    integration_time: float  # seconds
    peak_memory_mb: float  # MB
    n_steps: int
    solver: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    timestamp: str
    results: List[BenchmarkResult]
    summary: Dict[str, float]


def load_mechanism_from_file(filepath: Path):
    """Load mechanism from DSL file."""
    try:
        with open(filepath, 'r') as f:
            dsl_text = f.read()

        mechanism = parse_dsl_to_mechanism(dsl_text, initials={})
        return mechanism

    except Exception as exc:
        logger.error(f"Failed to load mechanism from {filepath}: {exc}")
        raise


def benchmark_simulation(
    mechanism,
    mechanism_name: str,
    t_span=(0, 100),
    solver='BDF',
    rtol=1e-6,
    atol=1e-12,
) -> BenchmarkResult:
    """
    Benchmark a single mechanism simulation.

    Parameters
    ----------
    mechanism : Mechanism
        Mechanism to benchmark
    mechanism_name : str
        Name for reporting
    t_span : tuple
        Time span (start, end)
    solver : str
        ODE solver method
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    n_species = len(mechanism.species_names())
    n_reactions = len(mechanism.reactions) + len(mechanism.equilibria)

    logger.info(
        f"Benchmarking {mechanism_name}: "
        f"{n_species} species, {n_reactions} reactions"
    )

    # Build ODE system
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

    # Run simulation with timing
    start_time = time.time()
    success = False
    error_message = None
    n_steps = 0
    peak_memory = 0.0

    try:
        # Create simulation request
        request = SimulationRequest(
            rhs=rhs,
            t_span=t_span,
            y0=y0,
            solver=solver,
            rtol=rtol,
            atol=atol,
            grid={'N': 200}  # Fixed number of output points
        )

        # Run simulation
        result = solve_ode(request)

        success = True  # solve_ode doesn't return success flag
        n_steps = len(result.t)

    except Exception as exc:
        logger.error(f"Simulation failed for {mechanism_name}: {exc}")
        error_message = str(exc)
        success = False

    elapsed_time = time.time() - start_time

    # Estimate memory usage (rough approximation)
    # Memory ≈ n_species * n_steps * 8 bytes (float64)
    if n_steps > 0:
        peak_memory = (n_species * n_steps * 8) / (1024 * 1024)  # MB

    logger.info(
        f"  Result: success={success}, time={elapsed_time:.3f}s, "
        f"steps={n_steps}, memory≈{peak_memory:.2f}MB"
    )

    return BenchmarkResult(
        mechanism_name=mechanism_name,
        n_species=n_species,
        n_reactions=n_reactions,
        integration_time=elapsed_time,
        peak_memory_mb=peak_memory,
        n_steps=n_steps,
        solver=solver,
        success=success,
        error_message=error_message,
    )


def run_benchmark_suite(
    mechanisms_dir: Path,
    output_file: Optional[Path] = None,
) -> BenchmarkSuite:
    """
    Run complete benchmark suite.

    Parameters
    ----------
    mechanisms_dir : Path
        Directory containing mechanism files
    output_file : Path, optional
        Output file for JSON results

    Returns
    -------
    BenchmarkSuite
        Complete benchmark results
    """
    logger.info("=" * 70)
    logger.info("Kindred Performance Benchmark Suite")
    logger.info("=" * 70)

    # Find all mechanism files
    mechanism_files = sorted(mechanisms_dir.glob("*.txt"))

    if not mechanism_files:
        logger.warning(f"No mechanism files found in {mechanisms_dir}")
        return BenchmarkSuite(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results=[],
            summary={},
        )

    logger.info(f"Found {len(mechanism_files)} mechanisms to benchmark")
    logger.info("")

    results = []

    # Run benchmarks
    for mech_file in mechanism_files:
        try:
            mechanism = load_mechanism_from_file(mech_file)
            result = benchmark_simulation(mechanism, mech_file.stem)
            results.append(result)

        except Exception as exc:
            logger.error(f"Benchmark failed for {mech_file.name}: {exc}")
            # Add failed result
            results.append(
                BenchmarkResult(
                    mechanism_name=mech_file.stem,
                    n_species=0,
                    n_reactions=0,
                    integration_time=0.0,
                    peak_memory_mb=0.0,
                    n_steps=0,
                    solver="BDF",
                    success=False,
                    error_message=str(exc),
                )
            )

        logger.info("")

    # Calculate summary statistics
    successful = [r for r in results if r.success]

    if successful:
        summary = {
            "total_benchmarks": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_time_s": sum(r.integration_time for r in successful),
            "avg_time_s": np.mean([r.integration_time for r in successful]),
            "min_time_s": min(r.integration_time for r in successful),
            "max_time_s": max(r.integration_time for r in successful),
            "total_memory_mb": sum(r.peak_memory_mb for r in successful),
        }
    else:
        summary = {
            "total_benchmarks": len(results),
            "successful": 0,
            "failed": len(results),
        }

    suite = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        summary=summary,
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
    logger.info(f"Successful: {summary.get('successful', 0)}")
    logger.info(f"Failed: {summary.get('failed', 0)}")

    if successful:
        logger.info(f"Total time: {summary['total_time_s']:.3f}s")
        logger.info(f"Average time: {summary['avg_time_s']:.3f}s")
        logger.info(f"Time range: {summary['min_time_s']:.3f}s - {summary['max_time_s']:.3f}s")
        logger.info(f"Total memory: {summary['total_memory_mb']:.2f}MB")

    logger.info("=" * 70)

    # Write results to file if specified
    if output_file is not None:
        try:
            output_data = {
                "timestamp": suite.timestamp,
                "summary": suite.summary,
                "results": [asdict(r) for r in suite.results],
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Results written to {output_file}")

        except Exception as exc:
            logger.error(f"Failed to write results to {output_file}: {exc}")

    return suite


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Kindred performance benchmarks"
    )
    parser.add_argument(
        "--mechanisms-dir",
        type=Path,
        default=Path(__file__).parent / "mechanisms",
        help="Directory containing mechanism files (default: ./mechanisms)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for JSON results (default: no output file)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    # Run benchmarks
    suite = run_benchmark_suite(
        mechanisms_dir=args.mechanisms_dir,
        output_file=args.output,
    )

    # Exit with appropriate code
    if suite.summary.get("failed", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
