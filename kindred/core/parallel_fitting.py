"""
Parallel parameter fitting using multi-start optimization.

This module provides parallel fitting capabilities to avoid local minima and
improve parameter estimation reliability by running multiple optimization
attempts with different initial guesses simultaneously.

Features:
- Multi-start optimization across different initial parameter values
- Parallel execution using multiprocessing
- Progress aggregation across workers
- Automatic selection of best result
- Graceful error handling and worker shutdown
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from multiprocessing.reduction import ForkingPickler
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import time

import numpy as np

try:
    from scipy.stats import qmc  # Quasi-Monte Carlo sampling
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from kindred.core.exceptions import OptimizationError
from kindred.core.fitting_optimization import fit_parameters, FitResult

logger = logging.getLogger(__name__)

__all__ = [
    "ParallelFitResult",
    "WorkerPool",
    "parallel_fit",
    "generate_initial_guesses",
]

def _initialize_fitting_worker() -> None:
    """
    Pool initializer for fitting workers.

    Keeps imports lazy so module import does not pull extra dependencies into
    normal startup paths. In worker processes, cap BLAS/OpenMP threads to avoid
    oversubscription (mirrors batch worker behavior).
    """
    try:
        from kindred.core.batch_parallel import apply_worker_blas_limits

        apply_worker_blas_limits(enabled=True)
    except Exception as exc:
        logger.debug("Fitting worker BLAS init skipped: %s", exc)


def _env_flag_true(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return str(val).strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass
class ParallelFitResult:
    """
    Results from parallel multi-start optimization.

    Attributes
    ----------
    best_result : FitResult
        Best fit result among all starts
    all_results : list[FitResult]
        All individual fit outcomes (sorted by chi-squared)
    n_starts : int
        Number of optimization starts
    n_success : int
        Number of successful optimizations
    total_time : float
        Total wall-clock time (seconds)
    worker_times : list[float]
        Individual worker execution times
    sequential_fallback : bool
        Whether the pool fell back to sequential execution
    fallback_reason : str, optional
        Reason for falling back to sequential execution
    parallel_effective : bool
        True when parallel execution was used; False when forced to sequential
    parallel_disabled_reason : str, optional
        Reason parallel execution was disabled
    """
    best_result: FitResult
    all_results: List[FitResult]
    n_starts: int
    n_success: int
    total_time: float
    worker_times: List[float] = field(default_factory=list)
    sequential_fallback: bool = False
    fallback_reason: Optional[str] = None
    parallel_effective: bool = True
    parallel_disabled_reason: Optional[str] = None

    @property
    def best_params(self) -> Dict[str, float]:
        """Convenience accessor for the best-fit parameter dictionary."""
        return self.best_result.parameters

    @property
    def success(self) -> bool:
        """Return True when at least one worker succeeded."""
        return self.n_success > 0


class WorkerPool:
    """
    Process pool manager for parallel fitting.

    Manages worker processes, task distribution, and result aggregation.
    Provides graceful shutdown and error handling.
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize worker pool.

        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes. Defaults to CPU count - 1.
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)

        self.n_workers = n_workers
        self.pool: Optional[mp.Pool] = None
        self._sequential = False
        self._fallback_reason: Optional[str] = None

        logger.info("Initialized WorkerPool with %s workers", self.n_workers)

    def __enter__(self):
        """Context manager entry: create process pool."""
        self._fallback_reason = None
        if _env_flag_true("KINDRED_DISABLE_MULTIPROCESSING"):
            self.pool = None
            self._sequential = True
            self._fallback_reason = "disabled via KINDRED_DISABLE_MULTIPROCESSING"
            return self
        if self.n_workers == 1:
            # Purely sequential path to avoid pickling requirements
            self.pool = None
            self._sequential = True
            self._fallback_reason = "n_workers=1: using sequential execution"
            return self
        try:
            # Windows uses spawn by default; frozen executables are supported
            # when multiprocessing.freeze_support() is used in kindred.__main__.
            # Keep using mp.Pool so tests can monkeypatch kindred.core.parallel_fitting.mp.Pool.
            self.pool = mp.Pool(
                processes=self.n_workers,
                initializer=_initialize_fitting_worker,
            )
        except (OSError, PermissionError, ValueError, RuntimeError, TypeError) as exc:
            self.pool = None
            self._sequential = True
            self._fallback_reason = str(exc)
            logger.warning(
                "Multiprocessing pool unavailable (%s). Falling back to sequential execution.",
                exc,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: clean up process pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        self._sequential = False

    def map(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """
        Map function across tasks in parallel.

        Parameters
        ----------
        func : callable
            Function to apply to each task
        tasks : list
            List of task arguments

        Returns
        -------
        list
            Results from all tasks
        """
        if self.pool is None or self._sequential:
            return [func(task) for task in tasks]

        try:
            return self.pool.map(func, tasks)
        except Exception as exc:
            logger.warning("Multiprocessing map failed (%s); falling back to sequential.", exc)
            self._fallback_reason = self._fallback_reason or str(exc)
            self._sequential = True
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
            return [func(task) for task in tasks]

    def starmap(self, func: Callable, tasks: List[Tuple]) -> List[Any]:
        """
        Map function across tasks with multiple arguments.

        Parameters
        ----------
        func : callable
            Function to apply to each task
        tasks : list of tuples
            List of argument tuples

        Returns
        -------
        list
            Results from all tasks
        """
        if self.pool is None or self._sequential:
            return [func(*task) for task in tasks]

        try:
            return self.pool.starmap(func, tasks)
        except Exception as exc:
            logger.warning("Multiprocessing starmap failed (%s); falling back to sequential.", exc)
            self._fallback_reason = self._fallback_reason or str(exc)
            self._sequential = True
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
            return [func(*task) for task in tasks]

    @property
    def sequential(self) -> bool:
        """Return True if the pool fell back to sequential execution."""
        return self._sequential

    @property
    def fallback_reason(self) -> Optional[str]:
        """Return the reason for sequential fallback, if any."""
        return self._fallback_reason


def _is_picklable(obj: Any) -> Tuple[bool, Optional[str]]:
    """Return (True, None) if obj can be pickled, else (False, reason)."""
    try:
        ForkingPickler.dumps(obj)
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    return True, None


def generate_initial_guesses(
    nominal: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    n_starts: int,
    method: str = "sobol",
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Generate diverse initial parameter guesses for multi-start optimization.

    Parameters
    ----------
    nominal : dict
        Nominal parameter values {name: value}
    bounds : dict, optional
        Parameter bounds {name: (min, max)}. If None, bounds are inferred from
        nominal values using a 0.1x-10x heuristic.
    n_starts : int
        Number of initial guesses to generate
    method : str
        Sampling method:
        - 'sobol': Sobol quasi-random sequence (recommended)
        - 'lhs': Latin Hypercube Sampling
        - 'random': Uniform random sampling
        - 'grid': Grid-based sampling
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[dict]
        List of initial parameter dictionaries

    Examples
    --------
    >>> nominal = {'k1': 1.0, 'k2': 0.5}
    >>> bounds = {'k1': (0.1, 10.0), 'k2': (0.01, 5.0)}
    >>> guesses = generate_initial_guesses(nominal, bounds, n_starts=10)
    >>> len(guesses)
    10
    """
    if not nominal:
        raise ValueError("At least one nominal parameter is required.")

    try:
        n_starts = int(n_starts)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"n_starts must be a positive integer, got {n_starts!r}") from exc

    if n_starts <= 0:
        raise ValueError(f"n_starts must be positive, got {n_starts}")

    if not HAS_SCIPY:
        logger.warning("scipy not available, using simple random sampling")
        method = 'random'

    param_names = list(nominal.keys())
    n_params = len(param_names)

    def _default_bounds(value: float) -> Tuple[float, float]:
        if value == 0:
            return (-1.0, 1.0)
        lower, upper = 0.1 * value, 10.0 * value
        if lower > upper:
            lower, upper = upper, lower
        if lower == upper:
            lower, upper = lower - 1.0, upper + 1.0
        return (lower, upper)

    bounds = bounds or {}

    # Extract bounds arrays
    lower = np.array([
        (bounds.get(name) or _default_bounds(nominal[name]))[0]
        for name in param_names
    ])
    upper = np.array([
        (bounds.get(name) or _default_bounds(nominal[name]))[1]
        for name in param_names
    ])

    # Generate samples
    if method == 'sobol' and HAS_SCIPY:
        # Sobol sequence (quasi-random, excellent space-filling)
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
        samples = sampler.random(n=n_starts)
        # Scale to bounds
        samples = qmc.scale(samples, lower, upper)

    elif method == 'lhs' and HAS_SCIPY:
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        samples = sampler.random(n=n_starts)
        samples = qmc.scale(samples, lower, upper)

    elif method == 'grid':
        # Grid-based sampling
        n_per_dim = int(np.ceil(n_starts ** (1.0 / n_params)))
        grids = [np.linspace(lower[i], upper[i], n_per_dim) for i in range(n_params)]
        mesh = np.meshgrid(*grids, indexing='ij')
        samples = np.stack([m.flatten() for m in mesh], axis=1)[:n_starts]

    else:  # 'random' or fallback
        # Simple uniform random sampling
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.uniform(lower, upper, size=(n_starts, n_params))

    # Convert to list of dicts
    guesses = []
    for i in range(len(samples)):
        guess = {name: samples[i, j] for j, name in enumerate(param_names)}
        guesses.append(guess)

    # Always include nominal as first guess
    if nominal not in guesses:
        guesses[0] = nominal

    logger.debug("Generated %s initial guesses using %s method", len(guesses), method)

    return guesses


def _fit_worker(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    method: str,
    max_nfev: int,
    seed: Optional[int],
    ftol: float = 1e-10,
    xtol: float = 1e-10,
) -> Tuple[FitResult, float]:
    """
    Worker function for parallel fitting.

    Parameters
    ----------
    objective_func : callable
        Objective returning residuals for a given parameter vector.
    initial_params : dict
        Starting parameter guesses.
    bounds : dict, optional
        Parameter bounds.
    method : str
        Optimizer identifier.
    max_nfev : int
        Maximum function evaluations.
    seed : int, optional
        Random seed for stochastic optimizers.
    ftol : float
        Function tolerance for least_squares.
    xtol : float
        Parameter tolerance for least_squares.

    Returns
    -------
    tuple
        (FitResult, execution_time)
    """
    start_time = time.time()
    param_names = list(initial_params.keys())

    bridge_to_dict = not getattr(objective_func, "_kindred_vector_objective", False)
    if bridge_to_dict:
        sample = np.array([initial_params[name] for name in param_names], dtype=float)
        try:
            objective_func(sample)
        except Exception:
            bridge_to_dict = True
        else:
            bridge_to_dict = False

    if bridge_to_dict:
        def objective_callable(vector: np.ndarray) -> np.ndarray:
            params_dict = {name: vector[i] for i, name in enumerate(param_names)}
            return np.asarray(objective_func(params_dict))
    else:
        objective_callable = objective_func

    try:
        result = fit_parameters(
            objective_func=objective_callable,
            initial_params=initial_params,
            bounds=bounds,
            method=method,
            progress_callback=None,  # No callbacks in workers
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
        )
    except Exception as exc:
        logger.warning("Fit failed with initial params %s: %s", initial_params, exc)
        # Return failed result
        result = FitResult(
            success=False,
            parameters=initial_params,
            uncertainties=None,
            chi_squared=np.inf,
            r_squared=0.0,
            residuals=np.array([]),
            nfev=0,
            message=f"Worker exception: {exc}",
        )

    elapsed = time.time() - start_time

    return result, elapsed


def parallel_fit(
    objective_func: Callable[[np.ndarray], np.ndarray],
    nominal_params: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    n_starts: int = 10,
    method: str = "trf",
    sampling_method: str = "sobol",
    max_nfev: int = 1000,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, FitResult], None]] = None,
    seed: Optional[int] = None,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
) -> ParallelFitResult:
    """
    Perform parallel multi-start parameter fitting.

    Runs multiple optimization attempts with different initial guesses in parallel,
    selecting the best result. This helps avoid local minima and improves
    parameter estimation reliability.

    Parameters
    ----------
    objective_func : callable
        Objective function returning residuals array
    nominal_params : dict
        Nominal parameter values {name: value}
    bounds : dict, optional
        Parameter bounds {name: (min, max)}. If None, bounds are inferred from
        nominal values.
    n_starts : int
        Number of optimization starts (default: 10)
    method : str
        Optimization method ('trf', 'dogbox', 'lm', 'de')
    sampling_method : str
        Initial guess sampling method ('sobol', 'lhs', 'random', 'grid')
    max_nfev : int
        Maximum function evaluations per start
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    progress_callback : callable, optional
        Called with (completed, total, best_result) after each completion
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    ParallelFitResult
        Comprehensive results including best fit and all attempts

    Examples
    --------
    >>> def objective(params):
    ...     # ... compute residuals ...
    ...     return residuals
    >>>
    >>> result = parallel_fit(
    ...     objective,
    ...     nominal_params={'k': 1.0},
    ...     bounds={'k': (0.1, 10.0)},
    ...     n_starts=20,
    ...     n_workers=4
    ... )
    >>> print(f"Best chi²: {result.best_result.chi_squared:.4e}")
    >>> print(f"Success rate: {result.n_success}/{result.n_starts}")

    Notes
    -----
    - Uses multiprocessing for true parallelism (bypasses GIL)
    - Sobol sampling provides excellent parameter space coverage
    - All workers run independently (no shared state)
    - Progress callbacks run in main process (thread-safe)
    - Failed fits are included in results with chi² = inf
    """
    logger.info("Starting parallel fit: %s starts, method=%s", n_starts, method)

    start_time = time.time()

    # Generate initial guesses
    initial_guesses = generate_initial_guesses(
        nominal_params, bounds, n_starts, method=sampling_method, seed=seed
    )

    # Prepare worker arguments
    worker_args = [
        (objective_func, guess, bounds, method, max_nfev, seed, ftol, xtol)
        for guess in initial_guesses
    ]

    # Preflight pickling before launching multiprocessing
    requested_workers = n_workers if n_workers is not None else max(1, mp.cpu_count() - 1)
    pickling_error: Optional[str] = None
    if requested_workers > 1:
        picklable, pickling_error = _is_picklable(objective_func)
        if not picklable:
            logger.warning(
                "Objective is not picklable for multiprocessing (%s); using sequential execution.",
                pickling_error,
            )

    # Run parallel fitting
    pool_fallback_reason: Optional[str] = None
    pool_sequential = False
    parallel_effective = True
    parallel_disabled_reason: Optional[str] = None

    if requested_workers > 1 and pickling_error:
        pool_sequential = True
        pool_fallback_reason = f"objective not picklable: {pickling_error}"
        parallel_effective = False
        parallel_disabled_reason = pool_fallback_reason
        results_and_times = [_fit_worker(*args) for args in worker_args]
    else:
        with WorkerPool(n_workers=n_workers) as pool:
            results_and_times = pool.starmap(_fit_worker, worker_args)
            pool_fallback_reason = pool.fallback_reason
            pool_sequential = pool.sequential
        parallel_effective = not pool_sequential
        if pool_sequential:
            parallel_disabled_reason = pool_fallback_reason or "parallel execution disabled; using sequential path"

    # Unpack results
    all_results = [r for r, _ in results_and_times]
    worker_times = [t for _, t in results_and_times]

    # Sort by chi-squared (best first)
    all_results.sort(key=lambda r: r.chi_squared)

    # Select best result
    best_result = all_results[0]

    # Count successes
    n_success = sum(1 for r in all_results if r.success)

    # Total time
    total_time = time.time() - start_time

    logger.info(
        f"Parallel fit complete: {n_success}/{n_starts} successful, "
        f"best χ²={best_result.chi_squared:.4e}, time={total_time:.2f}s"
    )

    # Progress callback (final)
    if progress_callback is not None:
        progress_callback(n_starts, n_starts, best_result)

    if n_success == 0:
        reason = ", ".join(
            sorted({r.message for r in all_results if r.message})
        )
        summary = reason if reason else "no workers produced a valid result"
        logger.error("Parallel fit failed: %s", summary)
        raise OptimizationError(f"Parallel fitting failed: {summary}")

    return ParallelFitResult(
        best_result=best_result,
        all_results=all_results,
        n_starts=n_starts,
        n_success=n_success,
        total_time=total_time,
        worker_times=worker_times,
        sequential_fallback=pool_sequential,
        fallback_reason=pool_fallback_reason,
        parallel_effective=parallel_effective,
        parallel_disabled_reason=parallel_disabled_reason,
    )
