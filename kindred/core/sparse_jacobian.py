"""
Sparse Jacobian support for large chemical mechanisms.

This module provides sparse matrix representations for Jacobian matrices in
chemical kinetics ODE systems. For mechanisms with many species (>100), sparse
Jacobians significantly reduce memory usage and improve computational efficiency.

Features:
- Automatic sparsity pattern detection from reaction network
- scipy.sparse.csc_array for efficient storage
- Analytical Jacobian computation
- Integration with existing ODE solvers
"""

from __future__ import annotations

import math
import logging
from typing import Callable, Dict, List, Optional, Set

import numpy as np

try:
    from scipy.sparse import csc_array
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False

from kindred.core.mechanism import Mechanism

logger = logging.getLogger(__name__)

__all__ = [
    "detect_sparsity_pattern",
    "build_sparse_jacobian",
    "estimate_sparsity_ratio",
    "SparsityInfo",
]


def _pow_overflow_to_inf(base: float, exponent: float) -> float:
    try:
        return math.pow(base, exponent)
    except OverflowError:
        return math.inf


class SparsityInfo:
    """
    Information about Jacobian sparsity pattern.

    Attributes
    ----------
    pattern : np.ndarray
        Boolean array (n_species × n_species) indicating nonzero elements
    n_nonzero : int
        Number of nonzero elements
    sparsity_ratio : float
        Fraction of nonzero elements (0.0 = fully sparse, 1.0 = dense)
    species_names : list[str]
        Species names in order
    coupling_graph : dict
        Species coupling graph {species: set of coupled species}
    """

    def __init__(
        self,
        pattern: np.ndarray,
        species_names: List[str],
        coupling_graph: Dict[str, Set[str]],
    ):
        self.pattern = pattern
        self.species_names = species_names
        self.coupling_graph = coupling_graph

        self.n_nonzero = np.sum(pattern)
        total_elements = pattern.size
        self.sparsity_ratio = self.n_nonzero / total_elements if total_elements > 0 else 0.0

    def __repr__(self) -> str:
        n = len(self.species_names)
        return (
            f"SparsityInfo(species={n}, "
            f"nonzero={self.n_nonzero}/{n*n}, "
            f"sparsity={self.sparsity_ratio:.2%})"
        )


def detect_sparsity_pattern(mechanism: Mechanism) -> SparsityInfo:
    """
    Detect sparsity pattern of Jacobian from reaction network.

    The Jacobian J[i,j] = ∂(dC_i/dt)/∂C_j is nonzero only when species i and j
    participate in the same reaction. This function analyzes the mechanism to
    determine which matrix elements are structurally nonzero.

    Parameters
    ----------
    mechanism : Mechanism
        Chemical mechanism

    Returns
    -------
    SparsityInfo
        Sparsity pattern information

    Examples
    --------
    >>> from kindred.core.mechanism import Mechanism
    >>> mech = Mechanism()
    >>> mech.add_species('A', 1.0)
    >>> mech.add_species('B', 0.0)
    >>> # ... add reactions ...
    >>> info = detect_sparsity_pattern(mech)
    >>> print(info)
    SparsityInfo(species=2, nonzero=4/4, sparsity=100.00%)

    Notes
    -----
    The Jacobian sparsity pattern is determined by:
    1. Species that appear together in reactions are coupled
    2. J[i,j] ≠ 0 if species i and j are coupled
    3. Diagonal elements J[i,i] are always nonzero (self-dependence)

    For large mechanisms (>100 species), typical sparsity is 1-10%, leading to
    10-100× memory savings and faster matrix operations.
    """
    species_names = mechanism.species_names()
    n_species = len(species_names)
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Build species coupling graph
    coupling_graph: Dict[str, Set[str]] = {name: set() for name in species_names}

    # Add self-coupling (diagonal always nonzero)
    for name in species_names:
        coupling_graph[name].add(name)

    # Process reactions
    for rxn in mechanism.reactions:
        affected_species = list(rxn.net_stoich.keys())
        dependency_species = list(rxn.rate_orders.keys())

        for sp1 in affected_species:
            coupling_graph[sp1].update(dependency_species)

    # Process equilibria
    for eq in mechanism.equilibria:
        # Get all species (forward + backward)
        eq_species = set(eq.stoich_forward.keys()) | set(eq.stoich_back.keys())

        # Couple all species in this equilibrium
        for sp1 in eq_species:
            coupling_graph[sp1].update(eq_species)

    # Build boolean sparsity pattern matrix
    pattern = np.zeros((n_species, n_species), dtype=bool)
    if n_species:
        row_blocks: List[np.ndarray] = []
        col_blocks: List[np.ndarray] = []
        for sp_i, coupled in coupling_graph.items():
            js = np.fromiter((species_index[sp_j] for sp_j in coupled), dtype=np.int64, count=len(coupled))
            row_blocks.append(np.full(js.size, species_index[sp_i], dtype=np.int64))
            col_blocks.append(js)
        rows = np.concatenate(row_blocks) if row_blocks else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(col_blocks) if col_blocks else np.zeros(0, dtype=np.int64)
        pattern[rows, cols] = True

    info = SparsityInfo(pattern, species_names, coupling_graph)

    logger.info(
        f"Detected sparsity pattern: {n_species} species, "
        f"{info.n_nonzero} nonzero elements ({info.sparsity_ratio:.1%})"
    )

    return info


def build_sparse_jacobian(
    mechanism: Mechanism,
    sparsity_info: Optional[SparsityInfo] = None,
) -> Callable[[float, np.ndarray], csc_array]:
    """
    Build sparse Jacobian function for ODE system.

    Constructs analytical Jacobian J[i,j] = ∂(dC_i/dt)/∂C_j in sparse format.
    Uses sparsity pattern to avoid computing structurally zero elements.

    Parameters
    ----------
    mechanism : Mechanism
        Chemical mechanism
    sparsity_info : SparsityInfo, optional
        Pre-computed sparsity information. If None, will detect automatically.

    Returns
    -------
    callable
        Function jac(t, y) -> scipy.sparse.csc_array (Jacobian matrix)

    Raises
    ------
    ImportError
        If scipy.sparse is not available

    Examples
    --------
    >>> jac_func = build_sparse_jacobian(mechanism)
    >>> J = jac_func(0.0, y0)
    >>> print(f"Jacobian shape: {J.shape}, nnz: {J.nnz}")

    Notes
    -----
    The Jacobian is constructed analytically using the chain rule:

    For a reaction: r_i = k * Π[C_j]^n_j

    ∂r_i/∂C_k = k * n_k * C_k^(n_k-1) * Π_{j≠k} C_j^n_j

    And by chain rule:

    ∂(dC_i/dt)/∂C_k = Σ_reactions ν_i * ∂r/∂C_k

    where ν_i is the stoichiometric coefficient of species i.

    Sparse format is efficient for large mechanisms because:
    - Memory: O(nnz) instead of O(n²)
    - Matrix-vector multiply: O(nnz) instead of O(n²)
    - Typical sparsity: 1-10% for large mechanisms
    """
    if not HAS_SCIPY_SPARSE:
        raise ImportError(
            "scipy.sparse is required for sparse Jacobian. "
            "Install with: pip install scipy"
        )

    # Detect sparsity if not provided
    if sparsity_info is None:
        sparsity_info = detect_sparsity_pattern(mechanism)

    species_names = mechanism.species_names()
    n_species = len(species_names)
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Pre-allocate sparse Jacobian structure once (solver hot path updates only .data).
    pattern = sparsity_info.pattern
    if pattern.shape != (n_species, n_species):
        raise ValueError(
            "sparsity_info.pattern shape does not match mechanism species ordering "
            f"(pattern={pattern.shape}, expected={(n_species, n_species)})"
        )

    indptr = np.zeros(n_species + 1, dtype=np.int64)
    indices_by_col: List[np.ndarray] = []
    for j in range(n_species):
        rows = np.flatnonzero(pattern[:, j]).astype(np.int64, copy=False)
        indices_by_col.append(rows)
        indptr[j + 1] = indptr[j] + rows.size

    nnz = int(indptr[-1])
    indices = np.concatenate(indices_by_col) if nnz else np.zeros(0, dtype=np.int64)
    data = np.zeros(nnz, dtype=float)
    J = csc_array((data, indices, indptr), shape=(n_species, n_species), dtype=float)
    jac_data = J.data

    def _csc_offsets_for_rows(col: int, rows: np.ndarray) -> np.ndarray:
        start = int(indptr[col])
        end = int(indptr[col + 1])
        col_rows = indices[start:end]
        pos = np.searchsorted(col_rows, rows)
        ok = (pos < col_rows.size) & (col_rows[pos] == rows)
        if not np.all(ok):
            missing = rows[~ok]
            raise ValueError(
                "Jacobian sparsity pattern missing required entries for "
                f"column={col}, rows={missing.tolist()}"
            )
        return start + pos

    def _evaluate_scalar(value):
        if value is None:
            return None
        return float(value()) if callable(value) else float(value)

    def _require_positive_finite_runtime_Keq(Keq_value: float) -> float:
        Keq_float = float(Keq_value)
        if not (Keq_float > 0.0) or not math.isfinite(Keq_float):
            raise ValueError("Equilibrium Keq must be positive and finite for runtime anchoring")
        return Keq_float

    SMALL = 1e-30

    def _monomial_derivatives_inplace(
        rate_const: float,
        term_idx: np.ndarray,
        term_order: np.ndarray,
        term_nonzero_mask: np.ndarray,
        y: np.ndarray,
        y_terms: np.ndarray,
        valid_terms: np.ndarray,
        term_vals: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Compute ∂/∂y_i [ k * Π y_j^{order_j} ] for i in term_idx.

        Output is written in-place to `out` (same shape as term_idx).
        """
        if term_idx.size == 0:
            for out_pos in range(int(out.size)):
                out[out_pos] = 0.0
            return

        monomial = float(rate_const)
        for term_pos in range(int(term_idx.size)):
            y_value = float(y[int(term_idx[term_pos])])
            order = float(term_order[term_pos])
            y_terms[term_pos] = y_value
            valid_terms[term_pos] = bool(term_nonzero_mask[term_pos])
            if y_value > SMALL and bool(term_nonzero_mask[term_pos]):
                term_value = _pow_overflow_to_inf(y_value, order)
            elif bool(term_nonzero_mask[term_pos]):
                term_value = 0.0
            else:
                term_value = 1.0
            term_vals[term_pos] = term_value
            monomial *= term_value

        for term_pos in range(int(out.size)):
            if not bool(valid_terms[term_pos]):
                out[term_pos] = 0.0
                continue
            if monomial != 0.0 and float(y_terms[term_pos]) > SMALL:
                out[term_pos] = monomial * float(term_order[term_pos]) / float(y_terms[term_pos])
                continue
            deriv = float(rate_const) * float(term_order[term_pos])
            for factor_pos in range(int(term_idx.size)):
                y_value = float(y_terms[factor_pos])
                exponent = float(term_order[factor_pos])
                if factor_pos == term_pos:
                    exponent -= 1.0
                if exponent == 0.0:
                    factor = 1.0
                elif y_value > SMALL:
                    factor = _pow_overflow_to_inf(y_value, exponent)
                elif exponent > 0.0:
                    factor = 0.0
                else:
                    factor = math.inf
                deriv *= factor
            out[term_pos] = float(deriv)

    # Build list of steps (irreversible reactions + equilibria) with precomputed offsets into J.data.
    reaction_steps = []
    equilibrium_steps = []

    for rxn in mechanism.reactions:
        vec = np.asarray(rxn.net_stoich_vector(species_names), dtype=float)
        stoich_rows = np.flatnonzero(vec).astype(np.int64, copy=False)
        stoich_vals = vec[stoich_rows].astype(float, copy=False)

        rate_obj = rxn.rate
        k = float(rate_obj()) if callable(rate_obj) else float(rate_obj)

        term_pairs = [
            (species_index[sp_name], float(order))
            for sp_name, order in rxn.rate_orders.items()
        ]
        if term_pairs:
            term_idx = np.fromiter((p[0] for p in term_pairs), dtype=np.int64, count=len(term_pairs))
            term_order = np.fromiter((p[1] for p in term_pairs), dtype=float, count=len(term_pairs))
        else:
            term_idx = np.zeros(0, dtype=np.int64)
            term_order = np.zeros(0, dtype=float)

        term_nonzero_mask = term_order != 0.0
        offsets_by_term_list = [_csc_offsets_for_rows(int(col), stoich_rows) for col in term_idx.tolist()]
        if term_idx.size:
            offsets_by_term = np.vstack(offsets_by_term_list).astype(np.int64, copy=False)
        else:
            offsets_by_term = np.zeros((0, stoich_rows.size), dtype=np.int64)

        reaction_steps.append(
            (
                k,
                term_idx,
                term_order,
                term_nonzero_mask,
                np.empty(term_idx.size, dtype=float),   # y_terms
                np.empty(term_idx.size, dtype=bool),    # valid_terms
                np.empty(term_idx.size, dtype=float),   # term_vals
                np.empty(term_idx.size, dtype=float),   # out_derivs
                stoich_vals,
                offsets_by_term,
            )
        )

    for eq in mechanism.equilibria:
        fwd_vec = np.asarray(eq.forward_vector(species_names), dtype=float)
        back_vec = np.asarray(eq.back_vector(species_names), dtype=float)
        vec = back_vec - fwd_vec
        stoich_rows = np.flatnonzero(vec).astype(np.int64, copy=False)
        stoich_vals = vec[stoich_rows].astype(float, copy=False)

        kf = _evaluate_scalar(eq.kf)
        kr = _evaluate_scalar(eq.kr)
        Keq = _evaluate_scalar(eq.Keq)

        if kf is None and kr is None:
            if Keq is None:
                raise ValueError("Equilibrium missing kinetic parameters (need Keq or rates)")
            Keq = _require_positive_finite_runtime_Keq(Keq)
            kf = 1.0
            kr = kf / Keq
        elif kf is None:
            if Keq is None:
                raise ValueError("Equilibrium missing kf and equilibrium information to derive it")
            Keq = _require_positive_finite_runtime_Keq(Keq)
            kf = kr * Keq
        elif kr is None:
            if Keq is None:
                raise ValueError("Equilibrium missing kr and equilibrium information to derive it")
            Keq = _require_positive_finite_runtime_Keq(Keq)
            kr = kf / Keq

        fwd_pairs = [(species_index[n], v) for n, v in eq.stoich_forward.items()]
        rev_pairs = [(species_index[n], v) for n, v in eq.stoich_back.items()]

        fwd_idx = np.fromiter((p[0] for p in fwd_pairs), dtype=np.int64, count=len(fwd_pairs)) if fwd_pairs else np.zeros(0, dtype=np.int64)
        fwd_order = np.fromiter((p[1] for p in fwd_pairs), dtype=float, count=len(fwd_pairs)) if fwd_pairs else np.zeros(0, dtype=float)
        rev_idx = np.fromiter((p[0] for p in rev_pairs), dtype=np.int64, count=len(rev_pairs)) if rev_pairs else np.zeros(0, dtype=np.int64)
        rev_order = np.fromiter((p[1] for p in rev_pairs), dtype=float, count=len(rev_pairs)) if rev_pairs else np.zeros(0, dtype=float)

        cols = np.unique(np.concatenate([fwd_idx, rev_idx])).astype(np.int64, copy=False)
        offsets_by_col_list = [_csc_offsets_for_rows(int(col), stoich_rows) for col in cols.tolist()]
        if cols.size:
            offsets_by_col = np.vstack(offsets_by_col_list).astype(np.int64, copy=False)
        else:
            offsets_by_col = np.zeros((0, stoich_rows.size), dtype=np.int64)
        net_derivs = np.zeros(cols.size, dtype=float)
        fwd_pos = np.searchsorted(cols, fwd_idx) if fwd_idx.size else np.zeros(0, dtype=np.int64)
        rev_pos = np.searchsorted(cols, rev_idx) if rev_idx.size else np.zeros(0, dtype=np.int64)

        fwd_nonzero_mask = fwd_order != 0.0
        rev_nonzero_mask = rev_order != 0.0

        equilibrium_steps.append(
            (
                float(kf),
                float(kr),
                fwd_idx,
                fwd_order,
                fwd_nonzero_mask,
                np.empty(fwd_idx.size, dtype=float),  # y_terms
                np.empty(fwd_idx.size, dtype=bool),   # valid_terms
                np.empty(fwd_idx.size, dtype=float),  # term_vals
                np.empty(fwd_idx.size, dtype=float),  # out_derivs
                rev_idx,
                rev_order,
                rev_nonzero_mask,
                np.empty(rev_idx.size, dtype=float),  # y_terms
                np.empty(rev_idx.size, dtype=bool),   # valid_terms
                np.empty(rev_idx.size, dtype=float),  # term_vals
                np.empty(rev_idx.size, dtype=float),  # out_derivs
                net_derivs,
                fwd_pos,
                rev_pos,
                stoich_vals,
                offsets_by_col,
            )
        )

    def jacobian(t: float, y: np.ndarray) -> csc_array:
        """
        Compute sparse Jacobian matrix at current state.

        Notes
        -----
        Internal sparse storage is reused between calls, but callers receive a
        matrix snapshot so solver internals cannot observe later mutations.
        """
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y_arr = y
        else:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
        for data_pos in range(int(jac_data.size)):
            jac_data[data_pos] = 0.0

        for (
            k,
            term_idx,
            term_order,
            term_nonzero_mask,
            y_terms,
            valid_terms,
            term_vals,
            derivs,
            stoich_vals,
            offsets_by_term,
        ) in reaction_steps:
            _monomial_derivatives_inplace(
                k,
                term_idx,
                term_order,
                term_nonzero_mask,
                y_arr,
                y_terms,
                valid_terms,
                term_vals,
                derivs,
            )
            if derivs.size:
                for term_pos in range(int(derivs.size)):
                    deriv = float(derivs[term_pos])
                    if deriv == 0.0:
                        continue
                    offsets = offsets_by_term[term_pos]
                    for row_pos in range(int(stoich_vals.size)):
                        jac_data[int(offsets[row_pos])] += deriv * float(stoich_vals[row_pos])

        for (
            kf,
            kr,
            fwd_idx,
            fwd_order,
            fwd_nonzero_mask,
            fwd_y_terms,
            fwd_valid_terms,
            fwd_term_vals,
            fwd_derivs,
            rev_idx,
            rev_order,
            rev_nonzero_mask,
            rev_y_terms,
            rev_valid_terms,
            rev_term_vals,
            rev_derivs,
            net_derivs,
            fwd_pos,
            rev_pos,
            stoich_vals,
            offsets_by_col,
        ) in equilibrium_steps:
            for deriv_pos in range(int(net_derivs.size)):
                net_derivs[deriv_pos] = 0.0
            _monomial_derivatives_inplace(
                kf,
                fwd_idx,
                fwd_order,
                fwd_nonzero_mask,
                y_arr,
                fwd_y_terms,
                fwd_valid_terms,
                fwd_term_vals,
                fwd_derivs,
            )
            _monomial_derivatives_inplace(
                kr,
                rev_idx,
                rev_order,
                rev_nonzero_mask,
                y_arr,
                rev_y_terms,
                rev_valid_terms,
                rev_term_vals,
                rev_derivs,
            )
            if fwd_derivs.size:
                for deriv_pos in range(int(fwd_derivs.size)):
                    net_derivs[int(fwd_pos[deriv_pos])] += float(fwd_derivs[deriv_pos])
            if rev_derivs.size:
                for deriv_pos in range(int(rev_derivs.size)):
                    net_derivs[int(rev_pos[deriv_pos])] -= float(rev_derivs[deriv_pos])

            if net_derivs.size:
                for col_pos in range(int(net_derivs.size)):
                    deriv = float(net_derivs[col_pos])
                    if deriv == 0.0:
                        continue
                    offsets = offsets_by_col[col_pos]
                    for row_pos in range(int(stoich_vals.size)):
                        jac_data[int(offsets[row_pos])] += deriv * float(stoich_vals[row_pos])

        return J.copy()

    return jacobian


def estimate_sparsity_ratio(mechanism: Mechanism) -> float:
    """
    Estimate Jacobian sparsity ratio without full pattern detection.

    Quick approximation based on average reaction size.

    Parameters
    ----------
    mechanism : Mechanism
        Chemical mechanism

    Returns
    -------
    float
        Estimated sparsity ratio (0.0 = fully sparse, 1.0 = dense)
    """
    n_species = len(mechanism.species_names())

    if n_species == 0:
        return 0.0

    # Count total species participations
    total_participations = 0
    n_reactions = len(mechanism.reactions) + len(mechanism.equilibria)

    for rxn in mechanism.reactions:
        total_participations += len(set(rxn.net_stoich) | set(rxn.rate_orders))

    for eq in mechanism.equilibria:
        total_participations += len(eq.stoich_forward) + len(eq.stoich_back)

    if n_reactions == 0:
        return 1.0  # No reactions = diagonal only

    avg_rxn_size = total_participations / n_reactions

    # Estimate: each reaction couples ~avg_rxn_size² elements
    # Plus diagonal
    estimated_nonzero = n_species + n_reactions * avg_rxn_size**2

    sparsity_ratio = min(1.0, estimated_nonzero / (n_species**2))

    return sparsity_ratio
