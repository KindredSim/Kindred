"""
Parameter algebra for rate/equilibrium constants.

This adds an explicit, unambiguous syntax in the Algebra section:

    param k1 = 4*k2

`let` (and bare `name = expr`) remain for observables only.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

from kindred.core.rate_binding import RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_eval import (
    evaluate_parameter_algebra,
    evaluate_parameter_algebra_in_context as _evaluate_param_block_in_context,
)
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
    ParameterAssignment,
    extract_parameter_assignments_from_algebra_lines,
    extract_parameter_assignments_from_dsl_text,
    extract_observable_names_from_algebra_lines,
    mechanism_parameter_name_pattern,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.step_indexing import (
    canonical_parameter_names,
    get_step_index_map,
    iter_canonical_parameters,
    lookup_step_param_target,
)
from kindred.core.simulator.wegscheider import enumerate_reversible_edges, select_spanning_forest_edges

logger = logging.getLogger(__name__)

_MECH_PARAM_RE = mechanism_parameter_name_pattern()
_WEGSCHEIDER_META_KEY = "wegscheider_cyclicity_enabled"
_PARAMETER_ALGEBRA_SPEC_META_KEY = "parameter_algebra_spec"
_WEGSCHEIDER_TOL = 1e-10

_PUBLIC_REEXPORTS = (
    ParameterAssignment,
    ParameterAlgebraSpec,
    extract_parameter_assignments_from_algebra_lines,
    extract_parameter_assignments_from_dsl_text,
    extract_observable_names_from_algebra_lines,
    parse_parameter_algebra_spec_from_dsl_text,
    evaluate_parameter_algebra,
)


def mechanism_parameter_names(mechanism: object) -> Set[str]:
    # Canonical global step-index naming (kN / kfN / krN / KN).
    out = canonical_parameter_names(mechanism)
    if out:
        return out
    # Fallback for legacy/hand-constructed Mechanism instances without a step_index_map.
    n_rxn = len(getattr(mechanism, "reactions", []) or [])
    n_eq = len(getattr(mechanism, "equilibria", []) or [])
    legacy: Set[str] = set()
    for i in range(1, n_rxn + 1):
        legacy.add(f"k{i}")
    for i in range(1, n_eq + 1):
        legacy.add(f"kf{i}")
        legacy.add(f"kr{i}")
        legacy.add(f"K{i}")
    return legacy


def read_mechanism_parameter_values(mechanism: object, *, names: Optional[Set[str]] = None) -> Dict[str, float]:
    wanted = names or mechanism_parameter_names(mechanism)
    out: Dict[str, float] = {}

    def _as_float(x: object) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x()) if callable(x) else float(x)
        except Exception:
            return None

    # Prefer canonical step map if present.
    step_map = get_step_index_map(mechanism)
    if step_map:
        rxns = getattr(mechanism, "reactions", []) or []
        eqs = getattr(mechanism, "equilibria", []) or []
        for name, entry, role in iter_canonical_parameters(mechanism):
            if name not in wanted:
                continue
            kind = str(entry.get("kind") or "")
            if kind == "reaction" and role == "k":
                idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
                if 0 <= idx < len(rxns):
                    v = _as_float(getattr(rxns[idx], "rate", None))
                    if v is not None:
                        out[name] = v
            elif kind == "equilibrium":
                idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
                if not (0 <= idx < len(eqs)):
                    continue
                eq = eqs[idx]
                if role in {"kf", "kr"}:
                    v = _as_float(getattr(eq, role, None))
                    if v is not None:
                        out[name] = v
                elif role == "K":
                    meta = getattr(eq, "metadata", {}) or {}
                    v = _as_float(meta.get("K_input"))
                    if v is not None:
                        out[name] = v
        return out

    # Legacy fallback: per-type ordinals.
    for i, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
        key = f"k{i}"
        if key in wanted:
            v = _as_float(getattr(rxn, "rate", None))
            if v is not None:
                out[key] = v
    for i, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
        for base, attr in (("kf", "kf"), ("kr", "kr"), ("K", "K")):
            key = f"{base}{i}"
            if key in wanted:
                v = _as_float(getattr(eq, attr, None))
                if v is not None:
                    out[key] = v
    return out


def _read_scalar_param_values(mechanism: object, *, require_mutable: bool) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", {}) or {}
    if require_mutable:
        bindings = meta.get("scalar_param_bindings") or {}
        out: Dict[str, float] = {}
        if isinstance(bindings, dict):
            for nm, b in bindings.items():
                try:
                    out[str(nm)] = float(b()) if callable(b) else float(b)
                except Exception as exc:
                    logger.debug("Failed to read scalar param binding %r: %s", nm, exc, exc_info=True)
                    continue
        return out
    vals = meta.get("scalar_params") or {}
    out2: Dict[str, float] = {}
    if isinstance(vals, dict):
        for nm, v in vals.items():
            try:
                out2[str(nm)] = float(v)
            except (TypeError, ValueError, OverflowError) as exc:
                logger.debug("Skipping non-numeric scalar param %r=%r: %s", nm, v, exc, exc_info=True)
                continue
    return out2


def _ensure_scalar_param_storage(mechanism: object, *, require_mutable: bool) -> None:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return
    if require_mutable:
        meta.setdefault("scalar_param_bindings", {})
        meta.setdefault("scalar_param_info", {})
    else:
        meta.setdefault("scalar_params", {})
        meta.setdefault("scalar_param_info", {})


def _set_mechanism_param(
    mechanism: object,
    name: str,
    value: float,
    *,
    require_mutable: bool,
) -> None:
    target = lookup_step_param_target(mechanism, name)
    if target is None:
        raise DSLError(f"Unsupported parameter name {name!r} for parameter algebra")
    kind, idx, role, _entry = target
    if kind == "reaction" and role == "k":
        rxns = getattr(mechanism, "reactions", []) or []
        if not (0 <= idx < len(rxns)):
            raise DSLError(f"Unknown parameter {name!r} (reaction index out of range)")
        rxn = rxns[idx]
        current = getattr(rxn, "rate", None)
        if require_mutable:
            if hasattr(current, "set"):
                current.set(float(value))  # type: ignore[call-arg]
                return
            raise DSLError(f"Parameter {name!r} is not mutable in this run (missing binding)")
        rxns[idx] = replace(rxn, rate=float(value))
        return

    if kind == "equilibrium":
        eqs = getattr(mechanism, "equilibria", []) or []
        if not (0 <= idx < len(eqs)):
            raise DSLError(f"Unknown parameter {name!r} (equilibrium index out of range)")
        eq = eqs[idx]
        if role in {"kf", "kr"}:
            current = getattr(eq, role, None)
            if require_mutable:
                if hasattr(current, "set"):
                    current.set(float(value))  # type: ignore[call-arg]
                    return
                raise DSLError(f"Parameter {name!r} is not mutable in this run (missing binding)")
            eqs[idx] = replace(eq, **{role: float(value)})
            return
        if role == "K":
            meta = dict(getattr(eq, "metadata", {}) or {})
            current = meta.get("K_input")
            if require_mutable:
                if hasattr(current, "set"):
                    current.set(float(value))  # type: ignore[call-arg]
                    return
                # Allow creating a binding for K_input if not already mutable.
                b = RateBinding(name=str(name), value=float(value))
                meta["K_input"] = b
                eqs[idx] = replace(eq, metadata=meta)
                return
            meta["K_input"] = float(value)
            eqs[idx] = replace(eq, metadata=meta)
            return

    raise DSLError(f"Unsupported parameter target for {name!r}")


def _apply_equilibrium_K_constraints_to_values(mechanism: object, base_values: Dict[str, float]) -> None:
    """
    Populate derived equilibrium rate values implied by explicit K parameters.

    This ensures parameter-algebra expressions that reference the derived rate
    see a consistent value.
    """
    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        if not bool(entry.get("has_K_param")):
            continue
        try:
            n = int(entry.get("step_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping equilibrium K-constraint entry with invalid step_index=%r: %s", entry.get("step_index"), exc)
            continue
        derive_rate = str(entry.get("derive_rate") or "kr")
        kf_key = f"kf{n}"
        kr_key = f"kr{n}"
        K_key = f"K{n}"
        if K_key not in base_values:
            continue
        K = float(base_values[K_key])
        if not math.isfinite(K) or abs(K) < 1e-30:
            continue
        if derive_rate == "kf":
            if kr_key in base_values:
                base_values[kf_key] = float(base_values[kr_key]) * K
        else:
            if kf_key in base_values:
                base_values[kr_key] = float(base_values[kf_key]) / K


def _apply_equilibrium_K_constraints_to_mechanism(mechanism: object, *, require_mutable: bool) -> Dict[str, float]:
    """
    Apply equilibrium constraints implied by explicit K parameters to the mechanism in-place.

    Returns a map of derived rate updates (e.g., {'kr2': 1.23}).
    """
    updates: Dict[str, float] = {}
    values = read_mechanism_parameter_values(mechanism)
    _apply_equilibrium_K_constraints_to_values(mechanism, values)
    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        if not bool(entry.get("has_K_param")):
            continue
        try:
            n = int(entry.get("step_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping equilibrium K-constraint (mechanism) entry with invalid step_index=%r: %s", entry.get("step_index"), exc)
            continue
        derive_rate = str(entry.get("derive_rate") or "kr")
        if derive_rate == "kf":
            nm = f"kf{n}"
        else:
            nm = f"kr{n}"
        if nm not in values:
            continue
        v = float(values[nm])
        _set_mechanism_param(mechanism, nm, v, require_mutable=require_mutable)
        updates[nm] = v
    return updates


def _apply_wegscheider_cyclicity_constraints_to_mechanism(
    mechanism: object,
    *,
    require_mutable: bool,
    constrained_params: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    """
    Enforce Wegscheider cyclicity over reversible (equilibrium) steps.

    Policy (hard constraints):
    - Operates on ln(kf/kr) edge potentials over the complex graph.
    - Builds a deterministic spanning forest; non-tree, non-fixed edges are derived.
    - Explicit-K equilibria (step_index_map has_K_param) are treated as fixed ratios.
    - Derived targets are recorded into constrained_params for GUI + fit-scan exclusion.
    """
    meta = getattr(mechanism, "metadata", {}) or {}
    enabled = bool(meta.get(_WEGSCHEIDER_META_KEY, False)) if isinstance(meta, dict) else False
    if not enabled:
        return {}

    edges = enumerate_reversible_edges(mechanism)
    if len(edges) < 2:
        return {}

    locked: Set[str] = {str(k) for k in (constrained_params or {}).keys()}
    nodes: List[str] = sorted({e.u for e in edges} | {e.v for e in edges})
    if len(nodes) < 2:
        return {}

    forced_anchor: Set[int] = set()
    eligible: Set[int] = set()
    for i, e in enumerate(edges):
        if bool(e.has_explicit_K):
            forced_anchor.add(i)
            continue
        kf_locked = str(e.kf_name) in locked
        kr_locked = str(e.kr_name) in locked
        if kf_locked and kr_locked:
            forced_anchor.add(i)
            continue
        eligible.add(i)

    forest = select_spanning_forest_edges(nodes, edges, prefer=forced_anchor)
    anchor = set(forced_anchor) | set(forest)
    dependent = sorted(i for i in eligible if i not in anchor)

    eqs = list(getattr(mechanism, "equilibria", []) or [])

    def _as_float(x: object) -> float:
        return float(x()) if callable(x) else float(x)

    def _ln_ratio_for_edge(i: int) -> float:
        e = edges[i]
        if not (0 <= int(e.equilibrium_index) < len(eqs)):
            raise DSLError(f"Wegscheider cyclicity: invalid equilibrium index for step {int(e.step_index)}")
        eq = eqs[int(e.equilibrium_index)]
        try:
            kf = _as_float(getattr(eq, "kf"))
            kr = _as_float(getattr(eq, "kr"))
        except Exception as exc:
            raise DSLError(f"Wegscheider cyclicity: missing kf/kr values for step {int(e.step_index)}") from exc
        if not (math.isfinite(kf) and math.isfinite(kr) and kf > 0.0 and kr > 0.0):
            raise DSLError(
                f"Wegscheider cyclicity requires positive finite kf/kr for step {int(e.step_index)} "
                f"({e.kf_name}={kf!r}, {e.kr_name}={kr!r})."
            )
        return float(math.log(kf) - math.log(kr))

    # Build adjacency over anchor edges using ln(kf/kr) values.
    adj: Dict[str, List[Tuple[str, float, int]]] = {n: [] for n in nodes}
    for i in sorted(anchor):
        e = edges[i]
        lnK = _ln_ratio_for_edge(i)
        adj[e.u].append((e.v, lnK, i))
        adj[e.v].append((e.u, -lnK, i))

    phi: Dict[str, float] = {}
    for root in nodes:
        if root in phi:
            continue
        phi[root] = 0.0
        stack = [root]
        while stack:
            cur = stack.pop()
            cur_phi = float(phi[cur])
            for nxt, dphi, edge_i in adj.get(cur, []):
                expected = cur_phi + float(dphi)
                if nxt not in phi:
                    phi[nxt] = float(expected)
                    stack.append(nxt)
                    continue
                if abs(float(phi[nxt]) - float(expected)) > float(_WEGSCHEIDER_TOL):
                    e = edges[int(edge_i)]
                    raise DSLError(
                        "Wegscheider cyclicity constraints are unsatisfiable for the current fixed ratios. "
                        f"Conflict detected while traversing step {int(e.step_index)}."
                    )

    if not dependent:
        return {}

    updates: Dict[str, float] = {}
    for i in dependent:
        e = edges[i]
        desired_lnK = float(phi[e.v] - phi[e.u])
        derive_kr = (str(e.kr_name) not in locked)
        derive_kf = (str(e.kf_name) not in locked)
        if derive_kr:
            eq = eqs[int(e.equilibrium_index)]
            kf_val = _as_float(getattr(eq, "kf"))
            if not (math.isfinite(kf_val) and kf_val > 0.0):
                raise DSLError(f"Wegscheider cyclicity requires positive kf for {e.kf_name}.")
            lnkr = float(math.log(float(kf_val)) - desired_lnK)
            kr_new = float(math.exp(lnkr))
            if not (math.isfinite(kr_new) and kr_new > 0.0):
                raise DSLError(f"Wegscheider cyclicity produced invalid derived value for {e.kr_name}.")
            _set_mechanism_param(mechanism, str(e.kr_name), float(kr_new), require_mutable=require_mutable)
            updates[str(e.kr_name)] = float(kr_new)
            constrained_params.setdefault(
                str(e.kr_name),
                {"line": 0, "expr": "Wegscheider cyclicity", "constraint_reason": "wegscheider"},
            )
            continue
        if derive_kf:
            eq = eqs[int(e.equilibrium_index)]
            kr_val = _as_float(getattr(eq, "kr"))
            if not (math.isfinite(kr_val) and kr_val > 0.0):
                raise DSLError(f"Wegscheider cyclicity requires positive kr for {e.kr_name}.")
            lnkf = float(math.log(float(kr_val)) + desired_lnK)
            kf_new = float(math.exp(lnkf))
            if not (math.isfinite(kf_new) and kf_new > 0.0):
                raise DSLError(f"Wegscheider cyclicity produced invalid derived value for {e.kf_name}.")
            _set_mechanism_param(mechanism, str(e.kf_name), float(kf_new), require_mutable=require_mutable)
            updates[str(e.kf_name)] = float(kf_new)
            constrained_params.setdefault(
                str(e.kf_name),
                {"line": 0, "expr": "Wegscheider cyclicity", "constraint_reason": "wegscheider"},
            )
            continue
        raise DSLError(
            "Wegscheider cyclicity requires at least one adjustable parameter per dependent edge, but "
            f"both {e.kf_name} and {e.kr_name} are constrained."
        )

    return updates


def parameter_algebra_spec_from_mechanism(mechanism: object) -> ParameterAlgebraSpec | None:
    meta = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(meta, dict):
        return None
    spec = meta.get(_PARAMETER_ALGEBRA_SPEC_META_KEY)
    return spec if isinstance(spec, ParameterAlgebraSpec) else None


def apply_parameter_algebra_spec_to_mechanism(
    spec: ParameterAlgebraSpec,
    *,
    mechanism: object,
    require_mutable: bool,
) -> Dict[str, float]:
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[_PARAMETER_ALGEBRA_SPEC_META_KEY] = spec
    _ensure_scalar_param_storage(mechanism, require_mutable=require_mutable)
    base_values: Dict[str, float] = {}
    base_values.update(read_mechanism_parameter_values(mechanism, names=spec.mechanism_param_names))
    base_values.update(_read_scalar_param_values(mechanism, require_mutable=require_mutable))
    _apply_equilibrium_K_constraints_to_values(mechanism, base_values)

    derived: Dict[str, float] = {}
    if spec.param_statements:
        # Evaluate derived definitions, seeding scalar defaults in base_values if needed.
        derived = evaluate_parameter_algebra(spec, base_values=base_values)

    meta = getattr(mechanism, "metadata", {}) or {}
    if require_mutable:
        scalar_bindings = meta.get("scalar_param_bindings")
        if not isinstance(scalar_bindings, dict):
            scalar_bindings = {}
            meta["scalar_param_bindings"] = scalar_bindings
        scalar_info = meta.get("scalar_param_info")
        if not isinstance(scalar_info, dict):
            scalar_info = {}
            meta["scalar_param_info"] = scalar_info
    else:
        scalar_vals = meta.get("scalar_params")
        if not isinstance(scalar_vals, dict):
            scalar_vals = {}
            meta["scalar_params"] = scalar_vals
        scalar_info = meta.get("scalar_param_info")
        if not isinstance(scalar_info, dict):
            scalar_info = {}
            meta["scalar_param_info"] = scalar_info

    constrained: Dict[str, Dict[str, object]] = {}
    for stmt in spec.param_statements:
        is_mech = _MECH_PARAM_RE.match(stmt.name) is not None
        # Determine derived-ness for UI metadata: mechanism targets are always constrained.
        try:
            tmp_spec = ParameterAlgebraSpec(
                param_statements=[stmt],
                observable_names=set(spec.observable_names),
                mechanism_param_names=set(spec.mechanism_param_names),
                scalar_input_names=set(spec.scalar_input_names),
            )
            tmp_vals = _evaluate_param_block_in_context(tmp_spec, base_values=dict(base_values), enforce_defaults=False)
            default_val = float(tmp_vals.get(stmt.name, float("nan")))
        except Exception:
            default_val = float("nan")
        if not is_mech:
            # Base declaration iff expression has no parameter dependencies
            # (handled in evaluate_parameter_algebra), which seeds missing
            # base_values. Reflect base_values into storage.
            cur_val = base_values.get(stmt.name)
            if cur_val is None and math.isfinite(default_val):
                cur_val = default_val
            if cur_val is None:
                cur_val = float("nan")
            if require_mutable:
                b = scalar_bindings.get(stmt.name)
                if b is None:
                    b = RateBinding(name=str(stmt.name), value=float(cur_val))
                    scalar_bindings[stmt.name] = b
                # Do not overwrite base scalar params here; only derived scalar
                # params are present in `derived`.
                if stmt.name in derived:
                    b.set(float(derived[stmt.name]))
                scalar_info[stmt.name] = {
                    "line": stmt.line_number,
                    "expr": stmt.expr_src,
                    "derived": bool(stmt.name in derived),
                    "editable": not bool(stmt.name in derived),
                    "unit": "1",
                }
            else:
                if stmt.name not in scalar_vals and math.isfinite(cur_val):
                    scalar_vals[stmt.name] = float(cur_val)
                if stmt.name in derived:
                    scalar_vals[stmt.name] = float(derived[stmt.name])
                scalar_info[stmt.name] = {
                    "line": stmt.line_number,
                    "expr": stmt.expr_src,
                    "derived": bool(stmt.name in derived),
                    "editable": not bool(stmt.name in derived),
                    "unit": "1",
                }
        else:
            constrained[stmt.name] = {
                "line": stmt.line_number,
                "expr": stmt.expr_src,
                "constraint_reason": "algebra",
            }

    meta["constrained_params"] = constrained

    # Apply derived values to mechanism-bound parameters and derived scalar params.
    for nm, val in derived.items():
        if _MECH_PARAM_RE.match(nm):
            _set_mechanism_param(mechanism, nm, val, require_mutable=require_mutable)

    # Apply K-implied equilibrium constraints after any parameter algebra updates.
    eq_updates = _apply_equilibrium_K_constraints_to_mechanism(mechanism, require_mutable=require_mutable)
    if eq_updates:
        derived = dict(derived)
        derived.update(eq_updates)

    # Apply Wegscheider cyclicity constraints last (after explicit-K implied rates).
    cy_updates = _apply_wegscheider_cyclicity_constraints_to_mechanism(
        mechanism,
        require_mutable=require_mutable,
        constrained_params=constrained,
    )
    if cy_updates:
        derived = dict(derived)
        derived.update(dict(cy_updates))
    return derived


def apply_parameter_algebra_to_mechanism(
    dsl_text: str,
    *,
    mechanism: object,
    require_mutable: bool,
) -> Dict[str, float]:
    """
    Apply `param` algebra from the DSL text to the given mechanism in-place.

    - In non-prepared mode (`require_mutable=False`), the mechanism is updated by
      replacing Reaction/Equilibrium objects with new numeric values.
    - In prepared/bound mode (`require_mutable=True`), updates must target mutable
      bindings (RateBinding-like objects with `.set()`).
    """
    mech_names = mechanism_parameter_names(mechanism)
    spec = parse_parameter_algebra_spec_from_dsl_text(dsl_text, mechanism_param_names=mech_names)
    return apply_parameter_algebra_spec_to_mechanism(
        spec,
        mechanism=mechanism,
        require_mutable=require_mutable,
    )


def solver_parameter_units_from_mechanism(mechanism: object) -> Dict[str, str]:
    """
    Best-effort unit map for solver parameters.

    - k{i}: based on Reaction.order (mass-action), units M^(1-order)/s
    - kf{i}, kr{i}: based on equilibrium forward/back molecularity
    - K{i}: dimensionless ("1")
    - scalar params: dimensionless ("1") (if present in metadata)
    """
    from kindred.core.simulator.parameter_units import rate_constant_unit

    units: Dict[str, str] = {}
    step_map = get_step_index_map(mechanism)
    rxns = getattr(mechanism, "reactions", []) or []
    eqs = getattr(mechanism, "equilibria", []) or []
    if step_map:
        for name, entry, role in iter_canonical_parameters(mechanism):
            kind = str(entry.get("kind") or "")
            if kind == "reaction" and role == "k":
                try:
                    idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    logger.debug("Skipping reaction index with invalid reaction_index=%r: %s", entry.get("reaction_index"), exc)
                    continue
                if 0 <= idx < len(rxns):
                    try:
                        order = int(getattr(rxns[idx], "order", 1))
                    except Exception:
                        order = 1
                    units[name] = rate_constant_unit(order)
            elif kind == "equilibrium":
                try:
                    idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "Skipping equilibrium index with invalid equilibrium_index=%r: %s",
                        entry.get("equilibrium_index"),
                        exc,
                    )
                    continue
                if not (0 <= idx < len(eqs)):
                    continue
                eq = eqs[idx]
                if role == "K":
                    units[name] = "1"
                    continue
                try:
                    fwd_order = int(round(sum(getattr(eq, "stoich_forward", {}).values())))
                except Exception:
                    fwd_order = 1
                try:
                    back_order = int(round(sum(getattr(eq, "stoich_back", {}).values())))
                except Exception:
                    back_order = 1
                if role == "kf":
                    units[name] = rate_constant_unit(fwd_order)
                elif role == "kr":
                    units[name] = rate_constant_unit(back_order)
    else:
        # Legacy fallback: per-type ordinals.
        for i, rxn in enumerate(rxns, start=1):
            try:
                order = int(getattr(rxn, "order", 1))
            except Exception:
                order = 1
            units[f"k{i}"] = rate_constant_unit(order)
        for i, eq in enumerate(eqs, start=1):
            try:
                fwd_order = int(round(sum(getattr(eq, "stoich_forward", {}).values())))
            except Exception:
                fwd_order = 1
            try:
                back_order = int(round(sum(getattr(eq, "stoich_back", {}).values())))
            except Exception:
                back_order = 1
            units[f"kf{i}"] = rate_constant_unit(fwd_order)
            units[f"kr{i}"] = rate_constant_unit(back_order)
            units[f"K{i}"] = "1"

    meta = getattr(mechanism, "metadata", {}) or {}
    scalar_info = meta.get("scalar_param_info") or {}
    if isinstance(scalar_info, dict):
        for nm, info in scalar_info.items():
            if isinstance(info, dict):
                units[str(nm)] = str(info.get("unit") or "1")
            else:
                units[str(nm)] = "1"
    return units
