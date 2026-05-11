from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import numpy as np

from kindred.core.mechanism import Equilibrium, Mechanism, Reaction

from .artifacts import SYMBOLIC_JACOBIAN_IDENTITY_ATTR, SymbolicArtifactIdentity
from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError


@dataclass(frozen=True, slots=True)
class SymbolicJacobianArtifact:
    species_names: tuple[str, ...]
    rhs_expressions: tuple[str, ...]
    jacobian_expressions: tuple[tuple[str, ...], ...]
    identity: SymbolicArtifactIdentity
    jacobian_func: Callable[[float, np.ndarray], np.ndarray]


def _canonical_json(payload: object) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return data.encode("utf-8")


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _source_identity_payload(mechanism: Mechanism, species_names: tuple[str, ...]) -> dict[str, Any]:
    if not hasattr(mechanism, "to_serializable"):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires a serializable Kindred mechanism."
        )
    serializable = mechanism.to_serializable()
    mechanism_payload = dict(serializable or {})
    species_payload = mechanism_payload.get("species")
    if isinstance(species_payload, Mapping):
        mechanism_payload["species"] = {
            str(name): {}
            for name in species_payload.keys()
        }
    return {
        "species_names": species_names,
        "mechanism": mechanism_payload,
    }


def _finite_scalar(value: object, *, label: str) -> float:
    if callable(value):
        raise UnsupportedSymbolicExpressionError(f"Dynamic callable {label} is not supported for symbolic Jacobian.")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedSymbolicExpressionError(f"{label} must be a finite scalar.") from exc
    if not math.isfinite(out):
        raise UnsupportedSymbolicExpressionError(f"{label} must be finite.")
    return out


def _power_product(sympy: Any, symbols: Mapping[str, Any], powers: Mapping[str, object]) -> Any:
    expr = sympy.Integer(1)
    for species_name, raw_power in sorted((powers or {}).items()):
        power = _finite_scalar(raw_power, label=f"stoichiometric power for {species_name}")
        if species_name not in symbols:
            raise UnsupportedSymbolicExpressionError(f"Unknown species {species_name!r} in symbolic Jacobian.")
        expr *= symbols[str(species_name)] ** sympy.Rational(str(power))
    return expr


def _reaction_rate_expr(sympy: Any, rxn: Reaction, symbols: Mapping[str, Any]) -> Any:
    rate = _finite_scalar(getattr(rxn, "rate", None), label="reaction rate")
    return sympy.Float(rate) * _power_product(sympy, symbols, getattr(rxn, "rate_orders", {}) or {})


def _equilibrium_rates(eq: Equilibrium) -> tuple[float, float]:
    kf = getattr(eq, "kf", None)
    kr = getattr(eq, "kr", None)
    keq = getattr(eq, "Keq", None)
    if kf is None and kr is not None and keq is not None:
        kr_val = _finite_scalar(kr, label="equilibrium kr")
        keq_val = _finite_scalar(keq, label="equilibrium Keq")
        if keq_val <= 0.0:
            raise UnsupportedSymbolicExpressionError("equilibrium Keq must be positive.")
        return float(kr_val * keq_val), float(kr_val)
    if kr is None and kf is not None and keq is not None:
        kf_val = _finite_scalar(kf, label="equilibrium kf")
        keq_val = _finite_scalar(keq, label="equilibrium Keq")
        if keq_val <= 0.0:
            raise UnsupportedSymbolicExpressionError("equilibrium Keq must be positive.")
        return float(kf_val), float(kf_val / keq_val)
    if kf is None or kr is None:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires explicit equilibrium kf/kr or one rate plus Keq.")
    return (
        _finite_scalar(kf, label="equilibrium kf"),
        _finite_scalar(kr, label="equilibrium kr"),
    )


def _equilibrium_rate_expr(sympy: Any, eq: Equilibrium, symbols: Mapping[str, Any]) -> Any:
    meta = dict(getattr(eq, "metadata", {}) or {})
    if meta.get("forward_model") or meta.get("reverse_model") or meta.get("dG_eq_J_per_mol"):
        raise UnsupportedSymbolicExpressionError("Temperature-dependent equilibrium models are not supported yet.")
    kf, kr = _equilibrium_rates(eq)
    forward = sympy.Float(kf) * _power_product(sympy, symbols, getattr(eq, "stoich_forward", {}) or {})
    reverse = sympy.Float(kr) * _power_product(sympy, symbols, getattr(eq, "stoich_back", {}) or {})
    return forward - reverse


def build_symbolic_jacobian_artifact(mechanism: Mechanism) -> SymbolicJacobianArtifact:
    sympy = require_sympy()
    species_names_func = getattr(mechanism, "species_names", None)
    if not callable(species_names_func):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires a Kindred mechanism with species_names()."
        )
    species_names = tuple(str(name) for name in species_names_func())
    if not species_names:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires at least one species.")
    state_symbols = tuple(sympy.Symbol(f"y_{idx}") for idx, _name in enumerate(species_names))
    symbol_by_species = dict(zip(species_names, state_symbols))
    rhs = [sympy.Integer(0) for _name in species_names]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    for rxn in getattr(mechanism, "reactions", []) or []:
        rate_expr = _reaction_rate_expr(sympy, rxn, symbol_by_species)
        for species_name, coeff in getattr(rxn, "net_stoich", {}).items():
            rhs[species_index[str(species_name)]] += sympy.Float(float(coeff)) * rate_expr

    for eq in getattr(mechanism, "equilibria", []) or []:
        rate_expr = _equilibrium_rate_expr(sympy, eq, symbol_by_species)
        for species_name in species_names:
            coeff = float(getattr(eq, "stoich_back", {}).get(species_name, 0.0)) - float(
                getattr(eq, "stoich_forward", {}).get(species_name, 0.0)
            )
            if coeff:
                rhs[species_index[species_name]] += sympy.Float(coeff) * rate_expr

    rhs_matrix = sympy.Matrix(rhs)
    jacobian_matrix = rhs_matrix.jacobian(sympy.Matrix(state_symbols))
    rhs_strings = tuple(str(sympy.simplify(expr)) for expr in rhs_matrix)
    jacobian_strings = tuple(tuple(str(sympy.simplify(jacobian_matrix[i, j])) for j in range(len(species_names))) for i in range(len(species_names)))
    source_fingerprint = _fingerprint(_source_identity_payload(mechanism, species_names))
    artifact_fingerprint = _fingerprint(
        {
            "rhs": rhs_strings,
            "jacobian": jacobian_strings,
        }
    )
    metadata = get_symbolic_backend_metadata()
    identity = SymbolicArtifactIdentity.jacobian(
        metadata,
        source_fingerprint=source_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
    )
    compiled = sympy.lambdify(state_symbols, jacobian_matrix, modules="numpy")

    def jacobian_func(_t: float, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).reshape(-1)
        if values.size != len(species_names):
            raise ValueError(f"symbolic jacobian expected {len(species_names)} state values, got {values.size}")
        return np.asarray(compiled(*values), dtype=float)

    setattr(jacobian_func, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, identity.to_payload())
    return SymbolicJacobianArtifact(
        species_names=species_names,
        rhs_expressions=rhs_strings,
        jacobian_expressions=jacobian_strings,
        identity=identity,
        jacobian_func=jacobian_func,
    )
