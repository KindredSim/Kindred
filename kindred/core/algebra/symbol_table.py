"""Symbol-table helpers for algebra evaluation."""

from __future__ import annotations

import logging

from typing import Dict

from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.step_indexing import get_step_index_map

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

    step_map = get_step_index_map(mechanism)
    if step_map:
        rxns = list(getattr(mechanism, "reactions", []) or [])
        eqs = list(getattr(mechanism, "equilibria", []) or [])
        for entry in step_map:
            kind = str(entry.get("kind") or "")
            try:
                n = int(entry.get("step_index"))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                _log_skip("Invalid step_index in step_index_map entry", exc)
                continue
            if kind == "reaction":
                try:
                    rxn_idx = int(entry.get("reaction_index", -1))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    _log_skip(f"Invalid reaction_index for k{n}", exc)
                    continue
                if not (0 <= rxn_idx < len(rxns)):
                    continue
                rate_obj = rxns[rxn_idx].rate
                try:
                    raw = rate_obj() if callable(rate_obj) else rate_obj
                    k_val = float(raw)
                    symtab.define_user(f"k{n}", k_val)
                except Exception as exc:
                    _log_skip(f"Failed to evaluate reaction rate for k{n}", exc)
                    continue
            elif kind == "equilibrium":
                try:
                    eq_idx = int(entry.get("equilibrium_index", -1))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    _log_skip(f"Invalid equilibrium_index for step {n}", exc)
                    continue
                if not (0 <= eq_idx < len(eqs)):
                    continue
                eq = eqs[eq_idx]
                if eq.kf is not None:
                    try:
                        raw = eq.kf() if callable(eq.kf) else eq.kf
                        kf_val = float(raw)
                        symtab.define_user(f"kf{n}", kf_val)
                    except Exception as exc:
                        _log_skip(f"Failed to evaluate equilibrium forward rate for kf{n}", exc)
                if eq.kr is not None:
                    try:
                        raw = eq.kr() if callable(eq.kr) else eq.kr
                        kr_val = float(raw)
                        symtab.define_user(f"kr{n}", kr_val)
                    except Exception as exc:
                        _log_skip(f"Failed to evaluate equilibrium reverse rate for kr{n}", exc)
                if bool(entry.get("has_K_param")):
                    meta = getattr(eq, "metadata", {}) or {}
                    K_obj = meta.get("K_input")
                    if K_obj is not None:
                        try:
                            raw = K_obj() if callable(K_obj) else K_obj
                            K_val = float(raw)
                            symtab.define_user(f"K{n}", K_val)
                        except Exception as exc:
                            _log_skip(f"Failed to evaluate equilibrium constant for K{n}", exc)
    else:
        # Legacy fallback: per-type ordinals.
        for i, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
            rate_obj = rxn.rate
            try:
                raw = rate_obj() if callable(rate_obj) else rate_obj
                k_val = float(raw)
                symtab.define_user(f"k{i}", k_val)
            except Exception as exc:
                _log_skip(f"Failed to evaluate reaction rate for k{i}", exc)
                continue
        for i, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
            if getattr(eq, "K", None) is not None:
                try:
                    raw = eq.K() if callable(eq.K) else eq.K
                    K_val = float(raw)
                    symtab.define_user(f"K{i}", K_val)
                except Exception as exc:
                    _log_skip(f"Failed to evaluate equilibrium constant for K{i}", exc)
            if getattr(eq, "kf", None) is not None:
                try:
                    raw = eq.kf() if callable(eq.kf) else eq.kf
                    kf_val = float(raw)
                    symtab.define_user(f"kf{i}", kf_val)
                except Exception as exc:
                    _log_skip(f"Failed to evaluate equilibrium forward rate for kf{i}", exc)
            if getattr(eq, "kr", None) is not None:
                try:
                    raw = eq.kr() if callable(eq.kr) else eq.kr
                    kr_val = float(raw)
                    symtab.define_user(f"kr{i}", kr_val)
                except Exception as exc:
                    _log_skip(f"Failed to evaluate equilibrium reverse rate for kr{i}", exc)

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
        try:
            symtab.define_user(name, float(value))
        except Exception as exc:
            logger.warning("Failed to define algebra scalar symbol %r: %s", name, exc, exc_info=True)
            continue
    if skipped > max_debug_logs:
        logger.debug("Suppressed %s additional symbol-table conversion error(s).", skipped - max_debug_logs)

    return symtab
