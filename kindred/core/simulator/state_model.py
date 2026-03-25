"""
States and Transition-State network model.

Core model contract
-------------------
- States: name, type GS|TS, species membership, energy respecting unit setting,
  standard state, degeneracy.
- TS degree fixed to 2 and enforced. Attempts to change are blocked with a
  structured error and no model mutation.

Scope
-----
This module defines:
- State and Edge records
- A StateNetwork container with deterministic behavior
- TS-degree enforcement on all mutating operations
- Canonical energy handling in J/mol with helpers for kcal/mol and kJ/mol

Non-goals
---------
- No kinetics mapping (handled in kinetics.py)
- No fast-equilibrium policy (fast_eq.py)
- No DSL parsing (dsl.py)

Constraints
-----------
- No filesystem/registry/network access
- Pure in-memory model; deterministic ordering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

import math

from ..units import kcalmol_to_jmol, kjmol_to_jmol
from ..validation import validate_name

logger = logging.getLogger(__name__)


__all__ = [
    "StateType",
    "State",
    "Edge",
    "TSDegreeError",
    "StateNetwork",
]


# ------------------------------ data types -----------------------------------

class StateType:
    """Enumerated state kinds."""
    GS = "GS"   # ground state (well)
    TS = "TS"   # transition state (saddle)


@dataclass(frozen=True)
class State:
    """
    Chemical state node.

    Fields
    ------
    name : str
        Unique identifier.
    kind : "GS" | "TS"
        State type. If "TS", network enforces degree = 2.
    energy_jmol : float
        Canonical energy of the state in J/mol.
    degeneracy : float
        Statistical degeneracy (>= 1 recommended, but any positive finite float allowed).
    standard_state : str
        Either "C0" (1 M) or "p0" (1 bar), advisory for Eyring prefactors downstream.
    members : Optional[Tuple[str, ...]]
        Optional tuple of species names comprising the state (purely descriptive here).
    std_conc_product_M : float | None
        Optional standard concentration product for this state, in M^m where m is the
        molecularity implied by `members` (or 1 if `members` is None).

        - For GS states representing a stoichiometric set of species, this is typically Π std_i^{ν_i}.
        - For TS states, this is typically std_TS.

        If None, downstream converters may fall back to the global C0 convention.
    """
    name: str
    kind: str
    energy_jmol: float
    degeneracy: float = 1.0
    standard_state: str = "C0"   # "C0" or "p0"
    members: Optional[Tuple[str, ...]] = None
    std_conc_product_M: float | None = None

    def with_energy(self, value: float, unit: str) -> "State":
        """Return a copy with energy set from value in given unit ('kcal/mol' or 'kJ/mol' or 'J/mol')."""
        ej = _to_jmol(value, unit)
        return replace(self, energy_jmol=ej)

    def with_degeneracy(self, g: float) -> "State":
        """Return a copy with updated degeneracy (strictly positive finite)."""
        g = _coerce_pos_float(g, "degeneracy")
        return replace(self, degeneracy=g)

    def with_standard_state(self, ss: str) -> "State":
        """Return a copy with updated standard state string ('C0' or 'p0')."""
        ssn = _normalize_standard_state(ss)
        return replace(self, standard_state=ssn)


@dataclass(frozen=True)
class Edge:
    """
    Undirected connectivity between two states.

    This represents a reaction channel adjacency in the state graph. Kinetic
    modeling chooses directions and rates elsewhere.
    """
    a: str
    b: str

    def endpoints(self) -> Tuple[str, str]:
        return (self.a, self.b)


# ------------------------------ errors ---------------------------------------

class TSDegreeError(ValueError):
    """
    Raised when an operation would violate the TS degree = 2 constraint.

    The GUI should surface this as a structured error in the TS Advanced tab and
    avoid mutating the model when this is raised.
    """


# ------------------------------ helpers --------------------------------------

def _to_jmol(value: float, unit: str) -> float:
    if unit == "J/mol":
        ej = float(value)
    elif unit == "kJ/mol":
        ej = kjmol_to_jmol(float(value))
    elif unit == "kcal/mol":
        ej = kcalmol_to_jmol(float(value))
    else:
        raise ValueError(f"unsupported energy unit {unit!r} (expected 'J/mol', 'kJ/mol', or 'kcal/mol')")
    if not math.isfinite(ej):
        raise ValueError("energy must be finite")
    return ej


def _normalize_standard_state(ss: str) -> str:
    if not isinstance(ss, str):
        raise ValueError("standard_state must be a string")
    s = ss.strip()
    if s not in ("C0", "p0"):
        raise ValueError("standard_state must be 'C0' (1 M) or 'p0' (1 bar)")
    return s


def _coerce_pos_float(x: float, label: str) -> float:
    try:
        xf = float(x)
    except Exception as e:
        logger.debug(f"Failed to coerce {label} to float: {x}", exc_info=True)
        raise ValueError(f"{label} must be numeric") from e
    if not math.isfinite(xf) or xf <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return xf


# ------------------------------ network --------------------------------------

class StateNetwork:
    """
    Deterministic state graph with TS-degree enforcement.

    Invariants
    ----------
    - State names are unique.
    - Edges are undirected and unique up to endpoints set {a,b}.
    - Every TS node has degree exactly 2.
    - Attempts to exceed TS degree or to finalize with degree != 2 raise TSDegreeError.

    Typical usage
    -------------
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=(0.0,"kJ/mol"))
    net.add_state("TS1", kind=StateType.TS, energy=(50.0,"kJ/mol"))
    net.add_state("B", kind=StateType.GS, energy=(-10.0,"kJ/mol"))
    net.add_edge("A","TS1")
    net.add_edge("TS1","B")
    net.validate()   # OK; TS1 degree == 2
    """

    def __init__(self) -> None:
        self._states: Dict[str, State] = {}
        # adjacency: name -> set of neighbor names
        self._adj: Dict[str, Set[str]] = {}
        # edge set as sorted tuple keys to ensure uniqueness
        self._edges: Set[Tuple[str, str]] = set()

    # ------------------ queries ------------------

    def states(self) -> List[State]:
        """List states in deterministic insertion order."""
        return [self._states[k] for k in self._states.keys()]

    def edges(self) -> List[Edge]:
        """List edges in deterministic order (lexicographic by endpoint)."""
        return [Edge(a, b) for (a, b) in sorted(self._edges)]

    def degree(self, name: str) -> int:
        n = validate_name(name)
        try:
            return len(self._adj[n])
        except KeyError:
            raise KeyError(f"unknown state {n!r}") from None

    def get(self, name: str) -> State:
        n = validate_name(name)
        try:
            return self._states[n]
        except KeyError:
            raise KeyError(f"unknown state {n!r}") from None

    def is_ts(self, name: str) -> bool:
        st = self.get(name)
        return st.kind == StateType.TS

    # ------------------ mutation ------------------

    def add_state(
        self,
        name: str,
        *,
        kind: str,
        energy: Tuple[float, str] | float,
        degeneracy: float = 1.0,
        standard_state: str = "C0",
        members: Optional[Iterable[str]] = None,
        std_conc_product_M: float | None = None,
    ) -> State:
        """
        Add a new state.

        Parameters
        ----------
        name : str
            Unique state name.
        kind : "GS" | "TS"
        energy : (value, unit) | value
            Energy specified either as a tuple with unit ("kJ/mol" | "kcal/mol" | "J/mol"),
            or a bare float treated as J/mol.
        degeneracy : float
            Positive finite degeneracy.
        standard_state : str
            "C0" (1 M) or "p0" (1 bar).
        members : Iterable[str] | None
            Optional species names.

        Returns
        -------
        State
        """
        n = validate_name(name)
        if n in self._states:
            raise ValueError(f"state {n!r} already exists")

        if isinstance(energy, tuple):
            value, unit = energy
            ej = _to_jmol(float(value), str(unit))
        else:
            ej = _to_jmol(float(energy), "J/mol")

        g = _coerce_pos_float(degeneracy, "degeneracy")
        ss = _normalize_standard_state(standard_state)
        mem = tuple(members) if members is not None else None
        std_prod = None
        if std_conc_product_M is not None:
            std_prod = _coerce_pos_float(std_conc_product_M, "std_conc_product_M")

        k = str(kind).strip().upper()
        if k not in (StateType.GS, StateType.TS):
            raise ValueError("kind must be 'GS' or 'TS'")

        st = State(
            name=n,
            kind=k,
            energy_jmol=ej,
            degeneracy=g,
            standard_state=ss,
            members=mem,
            std_conc_product_M=std_prod,
        )
        self._states[n] = st
        self._adj[n] = set()
        # No degree check needed yet (TS degree is about edges)
        return st

    def rename_state(self, old: str, new: str) -> None:
        """Rename a state while preserving insertion order and edges."""
        o = validate_name(old)
        nn = validate_name(new)
        if o not in self._states:
            raise KeyError(f"unknown state {o!r}")
        if nn in self._states and nn != o:
            raise ValueError(f"state {nn!r} already exists")
        if nn == o:
            return

        st = self._states.pop(o)
        st = replace(st, name=nn)
        # Reinsert to preserve relative order by rebuilding dict
        self._states = {**self._states}
        # Insert at end; deterministic enough for our purposes
        self._states[nn] = st

        # Update adjacency
        neigh = self._adj.pop(o)
        self._adj[nn] = set()
        for v in neigh:
            # update neighbor adjacency set
            self._adj[v].remove(o)
            self._adj[v].add(nn)
            # update edge key
            a, b = (nn, v) if nn < v else (v, nn)
            old_key: Tuple[str, str] = (o, v) if o < v else (v, o)
            self._edges.discard(old_key)
            self._edges.add((a, b))
            self._adj[nn].add(v)

        # TS degree is unchanged by rename; no check necessary

    def remove_state(self, name: str) -> None:
        """Remove a state. If it's a TS, only allowed when it has no edges (degree 0)."""
        n = validate_name(name)
        if n not in self._states:
            raise KeyError(f"unknown state {n!r}")
        if self.is_ts(n) and self.degree(n) != 0:
            # Do not allow silent structural changes that could break the invariant
            raise TSDegreeError(f"cannot remove TS {n!r} while degree != 0")
        # Remove incident edges for GS nodes
        for v in list(self._adj[n]):
            self.remove_edge(n, v)
        del self._adj[n]
        del self._states[n]

    def add_edge(self, a: str, b: str) -> Edge:
        """
        Add an undirected edge between states a and b.

        Enforces TS degree ≤ 2 at the moment of insertion. Exact equality (=2)
        is validated by `validate()` to allow temporary degree 1 while building.
        """
        na = validate_name(a)
        nb = validate_name(b)
        if na == nb:
            raise ValueError("self-loops are not allowed")
        if na not in self._states or nb not in self._states:
            missing = [x for x in (na, nb) if x not in self._states]
            raise KeyError(f"unknown state(s): {', '.join(repr(m) for m in missing)}")
        a, b = (na, nb) if na < nb else (nb, na)
        key: Tuple[str, str] = (a, b)
        if key in self._edges:
            return Edge(a, b)  # idempotent

        # Provisional degree check
        deg_a = len(self._adj[na])
        deg_b = len(self._adj[nb])
        if self.is_ts(na) and deg_a >= 2:
            raise TSDegreeError(f"TS {na!r} would exceed degree 2")
        if self.is_ts(nb) and deg_b >= 2:
            raise TSDegreeError(f"TS {nb!r} would exceed degree 2")

        # Insert
        self._edges.add(key)
        self._adj[na].add(nb)
        self._adj[nb].add(na)
        return Edge(a, b)

    def remove_edge(self, a: str, b: str) -> None:
        """
        Remove an undirected edge. This can temporarily leave a TS with degree
        not equal to 2; `validate()` must be called by higher-level code before
        using the network to derive kinetics.
        """
        na = validate_name(a)
        nb = validate_name(b)
        a, b = (na, nb) if na < nb else (nb, na)
        key: Tuple[str, str] = (a, b)
        if key not in self._edges:
            return
        self._edges.remove(key)
        self._adj[na].discard(nb)
        self._adj[nb].discard(na)

    # ------------------ validation ------------------

    def validate(self) -> None:
        """
        Validate global invariants.

        - All TS nodes must have degree exactly 2.
        - All nodes must exist and edges be consistent.
        """
        # Graph consistency
        for (a, b) in self._edges:
            if a not in self._states or b not in self._states:
                raise ValueError("edge references unknown state")
            if b not in self._adj[a] or a not in self._adj[b]:
                raise ValueError("adjacency inconsistent with edge set")

        # TS degree = 2
        for name, st in self._states.items():
            if st.kind == StateType.TS:
                deg = len(self._adj[name])
                if deg != 2:
                    raise TSDegreeError(f"TS {name!r} must have degree 2, found {deg}")

    # ------------------ convenience ------------------

    def set_energy(self, name: str, value: float, unit: str) -> None:
        """Set state's energy with unit conversion to canonical J/mol."""
        n = validate_name(name)
        if n not in self._states:
            raise KeyError(f"unknown state {n!r}")
        st = self._states[n].with_energy(value, unit)
        self._states[n] = st

    def set_degeneracy(self, name: str, g: float) -> None:
        """Set state's degeneracy."""
        n = validate_name(name)
        if n not in self._states:
            raise KeyError(f"unknown state {n!r}")
        st = self._states[n].with_degeneracy(g)
        self._states[n] = st

    def set_standard_state(self, name: str, ss: str) -> None:
        """Set state's standard state ('C0' or 'p0')."""
        n = validate_name(name)
        if n not in self._states:
            raise KeyError(f"unknown state {n!r}")
        st = self._states[n].with_standard_state(ss)
        self._states[n] = st

    # ------------------ serialization ------------------

    def to_serializable(self) -> Dict[str, object]:
        """
        Deterministic serialization of states and edges.

        Returns
        -------
        dict with keys:
            states: {name: {kind, energy_jmol, degeneracy, standard_state, members}}
            edges: [[a, b], ...] with endpoints sorted
        """
        states_block = {
            name: {
                "kind": st.kind,
                "energy_jmol": float(st.energy_jmol),
                "degeneracy": float(st.degeneracy),
                "standard_state": st.standard_state,
                "members": list(st.members) if st.members is not None else None,
                "std_conc_product_M": (float(st.std_conc_product_M) if st.std_conc_product_M is not None else None),
            }
            for name, st in self._states.items()
        }
        edges_block = [[a, b] for (a, b) in sorted(self._edges)]
        return {"states": states_block, "edges": edges_block}
