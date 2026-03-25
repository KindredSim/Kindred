"""
Wegscheider cyclicity (thermodynamic cycle constraints) for reversible networks.

This module is intentionally Qt-free and solver-free. It provides deterministic
graph extraction and selection utilities that higher-level plumbing (parameter
algebra / fitting / GUI) can use to enforce cyclicity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.validation import try_parse_int

__all__ = [
    "WegscheiderEdge",
    "enumerate_reversible_edges",
    "derived_parameter_names_for_cyclicity",
    "select_spanning_forest_edges",
]


def _format_coeff(x: float) -> str:
    try:
        xf = float(x)
    except Exception:
        return "nan"
    if abs(xf - 1.0) <= 1e-12:
        return ""
    s = f"{xf:.12g}"
    s_lower = s.lower()
    if "e" in s_lower:
        e_idx = s_lower.index("e")
        mantissa = s[:e_idx]
        exponent = s[e_idx:]
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}{exponent}"
    if "." in s:
        return s.rstrip("0").rstrip(".")
    return s


def complex_key(side: Mapping[str, float]) -> str:
    """
    Deterministic key for a stoichiometry side (reactants or products).

    This treats the side as a "complex" node for cyclicity constraints.
    """
    parts: List[str] = []
    for name in sorted({str(k) for k in (side or {}).keys()}):
        try:
            coeff = float(side[name])
        except Exception:
            coeff = float("nan")
        if not math.isfinite(coeff) or coeff <= 0.0:
            # Keep a stable representation; validation occurs upstream.
            parts.append(f"{name}?")
            continue
        prefix = _format_coeff(coeff)
        parts.append(f"{prefix}{name}" if prefix else f"{name}")
    if not parts:
        return "∅"
    return "+".join(parts)


@dataclass(frozen=True)
class WegscheiderEdge:
    step_index: int
    equilibrium_index: int
    u: str
    v: str
    kf_name: str
    kr_name: str
    has_explicit_K: bool
    derive_rate: Optional[str]


def enumerate_reversible_edges(mechanism: object) -> List[WegscheiderEdge]:
    """
    Enumerate reversible step edges from mechanism.metadata["step_index_map"].

    Only entries with kind == "equilibrium" are included. Each edge corresponds
    to a kfN/krN pair.
    """
    step_map = get_step_index_map(mechanism)
    if not step_map:
        return []
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    out: List[WegscheiderEdge] = []
    for entry in step_map:
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        n, ok_n = try_parse_int(entry.get("step_index"))
        eq_idx, ok_eq = try_parse_int(entry.get("equilibrium_index"))
        if not (ok_n and ok_eq):
            continue
        if not (0 <= eq_idx < len(eqs)):
            continue
        eq = eqs[eq_idx]
        u = complex_key(getattr(eq, "stoich_forward", {}) or {})
        v = complex_key(getattr(eq, "stoich_back", {}) or {})
        out.append(
            WegscheiderEdge(
                step_index=n,
                equilibrium_index=eq_idx,
                u=u,
                v=v,
                kf_name=f"kf{n}",
                kr_name=f"kr{n}",
                has_explicit_K=bool(entry.get("has_K_param")),
                derive_rate=(str(entry.get("derive_rate")) if entry.get("derive_rate") is not None else None),
            )
        )
    return out


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent: Dict[str, str] = {str(x): str(x) for x in items}
        self._rank: Dict[str, int] = {str(x): 0 for x in items}

    def find(self, x: str) -> str:
        x0 = str(x)
        p = self._parent.get(x0, x0)
        if p != x0:
            self._parent[x0] = self.find(p)
        return self._parent.get(x0, x0)

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        pa = self._rank.get(ra, 0)
        pb = self._rank.get(rb, 0)
        if pa < pb:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if pa == pb:
            self._rank[ra] = pa + 1
        return True


def select_spanning_forest_edges(
    nodes: Sequence[str],
    edges: Sequence[WegscheiderEdge],
    *,
    prefer: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Return a deterministic spanning-forest edge index set.

    - `prefer` is an optional set of edge indices that are considered first.
    - The forest is built over the undirected graph induced by edges (u-v).
    """
    prefer = set(prefer or set())
    uf = _UnionFind(nodes)
    chosen: Set[int] = set()

    def _key(i: int) -> Tuple[int, int, int, str]:
        e = edges[i]
        return (
            0 if i in prefer else 1,
            int(e.step_index),
            int(e.equilibrium_index),
            str(e.kf_name),
        )

    for i in sorted(range(len(edges)), key=_key):
        e = edges[i]
        if uf.union(e.u, e.v):
            chosen.add(i)
    return chosen


def derived_parameter_names_for_cyclicity(
    mechanism: object,
    *,
    constrained_param_names: Set[str] | None = None,
) -> Set[str]:
    """
    Return the set of mechanism parameter names that will be derived when enforcing cyclicity.

    This mirrors the selection policy in the parameter-algebra cyclicity stage:
    - explicit-K edges and fully-locked edges are treated as fixed anchors,
    - a deterministic spanning forest is chosen,
    - eligible non-tree edges are derived by adjusting either krN (preferred) or kfN.

    The result is topology-dependent and does not depend on parameter values.
    """
    edges = enumerate_reversible_edges(mechanism)
    if len(edges) < 2:
        return set()
    locked = {str(x) for x in (constrained_param_names or set())}
    nodes: List[str] = sorted({e.u for e in edges} | {e.v for e in edges})
    if len(nodes) < 2:
        return set()

    forced_anchor: Set[int] = set()
    eligible: Set[int] = set()
    for i, e in enumerate(edges):
        if bool(e.has_explicit_K):
            forced_anchor.add(i)
            continue
        if str(e.kf_name) in locked and str(e.kr_name) in locked:
            forced_anchor.add(i)
            continue
        eligible.add(i)

    forest = select_spanning_forest_edges(nodes, edges, prefer=forced_anchor)
    anchor = set(forced_anchor) | set(forest)
    dependent = sorted(i for i in eligible if i not in anchor)
    if not dependent:
        return set()

    out: Set[str] = set()
    for i in dependent:
        e = edges[i]
        if str(e.kr_name) not in locked:
            out.add(str(e.kr_name))
        elif str(e.kf_name) not in locked:
            out.add(str(e.kf_name))
    return out
