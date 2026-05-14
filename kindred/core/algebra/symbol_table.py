"""Symbol-table helpers for algebra evaluation."""

from __future__ import annotations

import logging

from typing import Dict, Mapping

from kindred.core.algebra.symbols import SymbolTable
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
    is_protected_indexed_identifier,
)

logger = logging.getLogger(__name__)

__all__ = ["build_algebra_symbol_table"]


def build_algebra_symbol_table(mechanism) -> SymbolTable:
    """Build a symbol table of rate/equilibrium constants and scalar params for algebra evaluation."""
    symtab = SymbolTable()

    skipped = 0
    max_debug_logs = 5

    def _log_skip(msg: str, exc: Exception) -> None:
        nonlocal skipped
        skipped += 1
        if skipped <= max_debug_logs:
            logger.debug("%s: %s", msg, exc, exc_info=True)

    namespace = build_namespace_from_mechanism(mechanism)
    metadata = getattr(mechanism, "metadata", {}) or {}
    step_map = metadata.get("step_index_map") if isinstance(metadata, dict) else None
    if not isinstance(step_map, list):
        raise ValueError("Mechanism step_index_map is missing; cannot build an authoritative algebra symbol table.")
    rxns = list(getattr(mechanism, "reactions", []) or [])
    eqs = list(getattr(mechanism, "equilibria", []) or [])

    for item in namespace.ordered_items:
        info = item.info
        name = str(item.canonical_name)
        source_index = item.source_index
        if source_index is None or not (0 <= int(source_index) < len(step_map)):
            raise ValueError(f"Cannot publish mechanism parameter {name!r}: missing authoritative step map entry.")
        entry = step_map[int(source_index)]
        if not isinstance(entry, dict):
            raise ValueError(f"Cannot publish mechanism parameter {name!r}: step map entry is invalid.")
        try:
            if info.step_kind == "reaction" and info.role == "k":
                rxn_idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
                if not (0 <= rxn_idx < len(rxns)):
                    raise ValueError(f"reaction index {rxn_idx} is out of range")
                value_obj = rxns[rxn_idx].rate
            elif info.step_kind == "equilibrium":
                eq_idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
                if not (0 <= eq_idx < len(eqs)):
                    raise ValueError(f"equilibrium index {eq_idx} is out of range")
                eq = eqs[eq_idx]
                if info.role in {"kf", "kr"}:
                    value_obj = getattr(eq, str(info.role), None)
                elif info.role == "Keq":
                    eq_meta = getattr(eq, "metadata", {}) or {}
                    value_obj = None
                    if isinstance(eq_meta, Mapping):
                        value_obj = eq_meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
                    if value_obj is None:
                        kf_obj = getattr(eq, "kf", None)
                        kr_obj = getattr(eq, "kr", None)
                        kf_raw = kf_obj() if callable(kf_obj) else kf_obj
                        kr_raw = kr_obj() if callable(kr_obj) else kr_obj
                        kr_value = float(kr_raw)
                        if kr_value == 0.0:
                            raise ValueError("reverse rate is zero")
                        value_obj = float(kf_raw) / kr_value
                else:
                    raise ValueError(f"unsupported equilibrium role {info.role!r}")
            else:
                raise ValueError(f"unsupported step kind {info.step_kind!r}")
            if value_obj is None:
                raise ValueError("value is missing")
            raw = value_obj() if callable(value_obj) else value_obj
            symtab.define_user(name, float(raw))
        except Exception as exc:
            raise ValueError(f"Failed to publish mechanism parameter {name!r}: {exc}") from exc

    # Add scalar params declared via Algebra param statements (param a = ...).
    meta = getattr(mechanism, "metadata", {}) or {}
    scalar_values: Dict[str, float] = {}
    if isinstance(meta, dict):
        scalar_params = meta.get("scalar_params") or {}
        if isinstance(scalar_params, dict):
            for nm, v in scalar_params.items():
                try:
                    scalar_values[str(nm)] = float(v)
                except Exception as exc:
                    _log_skip(f"Failed to coerce scalar param {nm!r}", exc)
                    continue
        scalar_bindings = meta.get("scalar_param_bindings") or {}
        if isinstance(scalar_bindings, dict):
            for nm, b in scalar_bindings.items():
                try:
                    scalar_values[str(nm)] = float(b()) if callable(b) else float(b)
                except Exception as exc:
                    _log_skip(f"Failed to evaluate scalar binding {nm!r}", exc)
                    continue
    for name, value in scalar_values.items():
        if not name.strip():
            continue
        resolution = namespace.resolve(name)
        if resolution.canonical_name is not None:
            raise ValueError(
                f"Algebra scalar symbol {name!r} is a protected indexed identifier that resolves to "
                f"mechanism parameter {resolution.canonical_name!r} and cannot be published as a scalar."
            )
        invalid_message = namespace.invalid_protected_indexed_identifier_message(name)
        if invalid_message is not None or is_protected_indexed_identifier(name):
            raise ValueError(
                f"Algebra scalar symbol {name!r} is a protected indexed identifier and cannot be published as a scalar."
            )
        try:
            symtab.define_user(name, float(value))
        except Exception as exc:
            logger.warning("Failed to define algebra scalar symbol %r: %s", name, exc, exc_info=True)
            continue
    if skipped > max_debug_logs:
        logger.debug("Suppressed %s additional symbol-table conversion error(s).", skipped - max_debug_logs)

    return symtab
