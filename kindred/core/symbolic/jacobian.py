from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import numpy as np

from kindred.core.mechanism import Equilibrium, Mechanism, Reaction
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.rate_binding import RateBinding
from kindred.core.simulator.parameter_namespace import canonical_name_for_mechanism_step_parameter

from .artifacts import SYMBOLIC_JACOBIAN_IDENTITY_ATTR, SymbolicArtifactIdentity
from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .namespaces import make_evaluation_snapshot_context, make_state_symbol_context, symbolic_status_payload


@dataclass(frozen=True, slots=True)
class SymbolicJacobianArtifact:
    species_names: tuple[str, ...]
    rhs_expressions: tuple[str, ...]
    jacobian_expressions: tuple[tuple[str, ...], ...]
    identity: SymbolicArtifactIdentity
    jacobian_func: Callable[[float, np.ndarray], np.ndarray]
    parameter_symbols: tuple[str, ...] = ()
    evaluation_snapshot: tuple[tuple[str, float], ...] = ()
    state_symbol_context: Mapping[str, Any] | None = None
    evaluation_snapshot_context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructure:
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[str, ...]
    jacobian_expressions: tuple[tuple[str, ...], ...]
    structure_fingerprint: str
    artifact_fingerprint: str
    _compiled: Callable[..., Any]
    state_symbol_context: Mapping[str, Any]

    def bind(self, parameter_values: Mapping[str, object] | None = None) -> SymbolicJacobianArtifact:
        snapshot = _coerce_snapshot_values(
            self.parameter_symbols,
            parameter_values,
        )
        snapshot_context = make_evaluation_snapshot_context(snapshot).to_payload()
        snapshot_fingerprint = str(snapshot_context["fingerprint"])
        metadata = get_symbolic_backend_metadata()
        identity = SymbolicArtifactIdentity.jacobian(
            metadata,
            source_fingerprint=self.structure_fingerprint,
            artifact_fingerprint=self.artifact_fingerprint,
            structure_fingerprint=self.structure_fingerprint,
            evaluation_snapshot_fingerprint=snapshot_fingerprint,
            parameter_symbols=self.parameter_symbols,
        )
        parameter_tuple = tuple(value for _name, value in snapshot)

        def jacobian_func(_t: float, y: np.ndarray) -> np.ndarray:
            values = np.asarray(y, dtype=float).reshape(-1)
            if values.size != len(self.species_names):
                raise ValueError(f"symbolic jacobian expected {len(self.species_names)} state values, got {values.size}")
            return np.asarray(self._compiled(*values, *parameter_tuple), dtype=float)

        setattr(jacobian_func, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, identity.to_payload())
        return SymbolicJacobianArtifact(
            species_names=self.species_names,
            rhs_expressions=self.rhs_expressions,
            jacobian_expressions=self.jacobian_expressions,
            identity=identity,
            jacobian_func=jacobian_func,
            parameter_symbols=self.parameter_symbols,
            evaluation_snapshot=snapshot,
            state_symbol_context=dict(self.state_symbol_context),
            evaluation_snapshot_context=snapshot_context,
        )


@dataclass(frozen=True, slots=True)
class SymbolicJacobianSupport:
    supported: bool
    code: str
    reason: str
    payload: Mapping[str, Any]

    def raise_if_unsupported(self) -> None:
        if not self.supported:
            raise UnsupportedSymbolicExpressionError(self.reason)

    def to_status_payload(self) -> dict[str, str]:
        state = "supported" if self.supported else "unsupported"
        reason = self.reason if self.reason else "Symbolic Jacobian supported."
        return symbolic_status_payload(kind="jacobian", state=state, code=self.code, reason=reason)


class _ParameterRegistry:
    def __init__(self, sympy: Any) -> None:
        self._sympy = sympy
        self._symbols: dict[str, Any] = {}
        self._values: dict[str, float] = {}

    def parameter(self, value: object, *, label: str, default_name: str) -> Any:
        name = str(getattr(value, "name", None) or default_name or label).strip()
        if not name:
            name = str(default_name or label)
        scalar = _finite_scalar(value, label=label)
        existing = self._values.get(name)
        if existing is not None and not math.isclose(existing, scalar, rel_tol=0.0, abs_tol=0.0):
            raise UnsupportedSymbolicExpressionError(
                f"Conflicting values for symbolic parameter {name!r}."
            )
        self._values[name] = scalar
        symbol = self._symbols.get(name)
        if symbol is None:
            symbol = self._sympy.Symbol(name)
            self._symbols[name] = symbol
        return symbol

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._symbols.keys()))

    @property
    def parameter_values(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, float(self._values[name])) for name in self.parameter_names)


def _canonical_step_parameter_name(
    mechanism: Mechanism,
    *,
    kind: str,
    item_index: int,
    role: str,
    fallback_name: str,
    value: object = None,
) -> str:
    explicit_name = str(getattr(value, "name", None) or "").strip()
    if explicit_name:
        return explicit_name
    return canonical_name_for_mechanism_step_parameter(
        mechanism,
        kind=kind,
        item_index=int(item_index),
        role=role,
        fallback_name=str(fallback_name),
    )


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
    if isinstance(value, RateBinding):
        value = value()
    if callable(value):
        raise UnsupportedSymbolicExpressionError(f"Dynamic callable {label} is not supported for symbolic Jacobian.")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedSymbolicExpressionError(f"{label} must be a finite scalar.") from exc
    if not math.isfinite(out):
        raise UnsupportedSymbolicExpressionError(f"{label} must be finite.")
    return out


def _coerce_snapshot_values(
    parameter_symbols: tuple[str, ...],
    parameter_values: Mapping[str, object] | None,
) -> tuple[tuple[str, float], ...]:
    supplied = dict(parameter_values or {})
    out: list[tuple[str, float]] = []
    for name in parameter_symbols:
        raw = supplied.get(name)
        if raw is None:
            raise UnsupportedSymbolicExpressionError(f"Missing symbolic parameter value for {name!r}.")
        out.append((name, _finite_scalar(raw, label=f"symbolic parameter {name}")))
    return tuple(out)


def _power_product(sympy: Any, symbols: Mapping[str, Any], powers: Mapping[str, object]) -> Any:
    expr = sympy.Integer(1)
    for species_name, raw_power in sorted((powers or {}).items()):
        power = _finite_scalar(raw_power, label=f"stoichiometric power for {species_name}")
        if species_name not in symbols:
            raise UnsupportedSymbolicExpressionError(f"Unknown species {species_name!r} in symbolic Jacobian.")
        expr *= symbols[str(species_name)] ** sympy.Rational(str(power))
    return expr


def _reaction_rate_expr(
    sympy: Any,
    mechanism: Mechanism,
    rxn: Reaction,
    symbols: Mapping[str, Any],
    registry: _ParameterRegistry,
    *,
    reaction_index: int,
) -> Any:
    rate_value = getattr(rxn, "rate", None)
    rate = registry.parameter(
        rate_value,
        label="reaction rate",
        default_name=_canonical_step_parameter_name(
            mechanism,
            kind="reaction",
            item_index=int(reaction_index),
            role="k",
            fallback_name=f"k{int(reaction_index) + 1}",
            value=rate_value,
        ),
    )
    return rate * _power_product(sympy, symbols, getattr(rxn, "rate_orders", {}) or {})


def _equilibrium_rates(
    mechanism: Mechanism,
    eq: Equilibrium,
    registry: _ParameterRegistry,
    *,
    equilibrium_index: int,
) -> tuple[Any, Any]:
    kf = getattr(eq, "kf", None)
    kr = getattr(eq, "kr", None)
    keq = getattr(eq, "Keq", None)
    kf_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kf",
        fallback_name=f"kf{int(equilibrium_index) + 1}",
        value=kf,
    )
    kr_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kr",
        fallback_name=f"kr{int(equilibrium_index) + 1}",
        value=kr,
    )
    keq_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="Keq",
        fallback_name=f"Keq{int(equilibrium_index) + 1}",
        value=keq,
    )
    if kf is None and kr is not None and keq is not None:
        kr_symbol = registry.parameter(kr, label="equilibrium kr", default_name=kr_name)
        keq_val = _finite_scalar(keq, label="equilibrium Keq")
        if keq_val <= 0.0:
            raise UnsupportedSymbolicExpressionError("equilibrium Keq must be positive.")
        keq_symbol = registry.parameter(keq, label="equilibrium Keq", default_name=keq_name)
        return kr_symbol * keq_symbol, kr_symbol
    if kr is None and kf is not None and keq is not None:
        kf_symbol = registry.parameter(kf, label="equilibrium kf", default_name=kf_name)
        keq_val = _finite_scalar(keq, label="equilibrium Keq")
        if keq_val <= 0.0:
            raise UnsupportedSymbolicExpressionError("equilibrium Keq must be positive.")
        keq_symbol = registry.parameter(keq, label="equilibrium Keq", default_name=keq_name)
        return kf_symbol, kf_symbol / keq_symbol
    if kf is None or kr is None:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires explicit equilibrium kf/kr or one rate plus Keq.")
    return (
        registry.parameter(kf, label="equilibrium kf", default_name=kf_name),
        registry.parameter(kr, label="equilibrium kr", default_name=kr_name),
    )


def _equilibrium_rate_expr(
    sympy: Any,
    mechanism: Mechanism,
    eq: Equilibrium,
    symbols: Mapping[str, Any],
    registry: _ParameterRegistry,
    *,
    equilibrium_index: int,
) -> Any:
    kf, kr = _equilibrium_rates(mechanism, eq, registry, equilibrium_index=equilibrium_index)
    forward = kf * _power_product(sympy, symbols, getattr(eq, "stoich_forward", {}) or {})
    reverse = kr * _power_product(sympy, symbols, getattr(eq, "stoich_back", {}) or {})
    return forward - reverse


def _unsupported_support(code: str, reason: str, payload: Mapping[str, Any]) -> SymbolicJacobianSupport:
    return SymbolicJacobianSupport(
        supported=False,
        code=str(code),
        reason=str(reason),
        payload=dict(payload),
    )


def _symbolic_snapshot_scalar_value(value: object) -> float | None:
    try:
        if isinstance(value, RateBinding):
            scalar = float(value())
        else:
            scalar = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(scalar):
        return None
    return float(scalar)


def _symbolic_snapshot_value_kind(value: object) -> str:
    if callable(value) and not isinstance(value, RateBinding):
        return "callable"
    if _symbolic_snapshot_scalar_value(value) is None:
        return "unsupported"
    return "snapshot_scalar"


def classify_symbolic_jacobian_support(mechanism: Mechanism) -> SymbolicJacobianSupport:
    species_names_func = getattr(mechanism, "species_names", None)
    if not callable(species_names_func):
        return _unsupported_support(
            "missing-species",
            "Symbolic Jacobian requires a Kindred mechanism with species_names().",
            {"reactions": [], "equilibria": []},
        )
    species_names = tuple(str(name) for name in species_names_func())
    if not species_names:
        return _unsupported_support(
            "missing-species",
            "Symbolic Jacobian requires at least one species.",
            {"reactions": [], "equilibria": []},
        )

    reactions_payload = []
    unsupported_code = ""
    unsupported_reason = ""
    for idx, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
        rate_kind = _symbolic_snapshot_value_kind(getattr(rxn, "rate", None))
        reactions_payload.append(
            {
                "index": idx,
                "rate_value_kind": rate_kind,
            }
        )
        if not unsupported_code and rate_kind != "snapshot_scalar":
            unsupported_code = "dynamic-rate"
            unsupported_reason = f"Symbolic Jacobian does not support dynamic or non-finite reaction rate k{idx}."
    equilibria_payload = []
    for idx, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
        meta = dict(getattr(eq, "metadata", {}) or {})
        kf = getattr(eq, "kf", None)
        kr = getattr(eq, "kr", None)
        keq = getattr(eq, "Keq", None)
        keq_input = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        kf_kind = _symbolic_snapshot_value_kind(kf) if kf is not None else "missing"
        kr_kind = _symbolic_snapshot_value_kind(kr) if kr is not None else "missing"
        keq_kind = _symbolic_snapshot_value_kind(keq) if keq is not None else "missing"
        keq_input_kind = _symbolic_snapshot_value_kind(keq_input) if keq_input is not None else "missing"
        has_forward_model = bool(meta.get(EquilibriumMetadataKeys.FORWARD_MODEL))
        has_reverse_model = bool(meta.get("reverse_model"))
        has_dg_eq = bool(meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) is not None)
        equilibria_payload.append(
            {
                "index": idx,
                "kf_value_kind": kf_kind,
                "kr_value_kind": kr_kind,
                "Keq_value_kind": keq_kind,
                "Keq_input_value_kind": keq_input_kind,
                "has_forward_model": has_forward_model,
                "has_reverse_model": has_reverse_model,
                "has_dG_eq_J_per_mol": has_dg_eq,
            }
        )
        if unsupported_code:
            continue
        if has_forward_model or has_reverse_model or has_dg_eq:
            unsupported_code = "temperature-dependent-equilibrium"
            unsupported_reason = "Temperature-dependent equilibrium models are outside symbolic Jacobian support."
            continue
        if keq_input is not None and (
            isinstance(keq_input, RateBinding)
            or callable(keq_input)
            or keq_input_kind != "snapshot_scalar"
        ):
            unsupported_code = "unsupported-keq-input"
            unsupported_reason = f"Symbolic Jacobian does not support dynamic or non-finite Keq input for equilibrium {idx}."
            continue
        supported_pair = False
        if kf is not None and kr is not None:
            supported_pair = kf_kind == "snapshot_scalar" and kr_kind == "snapshot_scalar"
        elif kf is not None and keq is not None:
            supported_pair = kf_kind == "snapshot_scalar" and keq_kind == "snapshot_scalar"
        elif kr is not None and keq is not None:
            supported_pair = kr_kind == "snapshot_scalar" and keq_kind == "snapshot_scalar"
        if not supported_pair:
            unsupported_code = "unsupported-equilibrium-parameters"
            unsupported_reason = f"Symbolic Jacobian is missing finite equilibrium parameters for equilibrium {idx}."
            continue
        if keq is not None and keq_kind == "snapshot_scalar":
            keq_scalar = _symbolic_snapshot_scalar_value(keq)
            if keq_scalar is None or keq_scalar <= 0.0:
                unsupported_code = "unsupported-equilibrium-keq"
                unsupported_reason = f"Symbolic Jacobian requires positive Keq for equilibrium {idx}."
    payload = {
        "reactions": reactions_payload,
        "equilibria": equilibria_payload,
    }
    if unsupported_code:
        return _unsupported_support(unsupported_code, unsupported_reason, payload)
    return SymbolicJacobianSupport(
        supported=True,
        code="supported",
        reason="",
        payload=payload,
    )


def _structure_identity_payload(
    mechanism: Mechanism,
    species_names: tuple[str, ...],
    parameter_symbols: tuple[str, ...],
    support_payload: Mapping[str, Any],
) -> dict[str, Any]:
    reactions_payload = []
    for idx, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
        rate = getattr(rxn, "rate", None)
        reactions_payload.append(
            {
                "index": idx,
                "reactants": dict(getattr(rxn, "reactants", {}) or {}),
                "products": dict(getattr(rxn, "products", {}) or {}),
                "rate_orders": dict(getattr(rxn, "rate_orders", {}) or {}),
                "net_stoich": dict(getattr(rxn, "net_stoich", {}) or {}),
                "rate_parameter": _canonical_step_parameter_name(
                    mechanism,
                    kind="reaction",
                    item_index=idx - 1,
                    role="k",
                    fallback_name=f"k{idx}",
                    value=rate,
                ),
            }
        )
    equilibria_payload = []
    for idx, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
        equilibria_payload.append(
            {
                "index": idx,
                "stoich_forward": dict(getattr(eq, "stoich_forward", {}) or {}),
                "stoich_back": dict(getattr(eq, "stoich_back", {}) or {}),
                "kf_parameter": _canonical_step_parameter_name(
                    mechanism,
                    kind="equilibrium",
                    item_index=idx - 1,
                    role="kf",
                    fallback_name=f"kf{idx}",
                    value=getattr(eq, "kf", None),
                ) if getattr(eq, "kf", None) is not None else None,
                "kr_parameter": _canonical_step_parameter_name(
                    mechanism,
                    kind="equilibrium",
                    item_index=idx - 1,
                    role="kr",
                    fallback_name=f"kr{idx}",
                    value=getattr(eq, "kr", None),
                ) if getattr(eq, "kr", None) is not None else None,
                "keq_parameter": _canonical_step_parameter_name(
                    mechanism,
                    kind="equilibrium",
                    item_index=idx - 1,
                    role="Keq",
                    fallback_name=f"Keq{idx}",
                    value=getattr(eq, "Keq", None),
                ) if getattr(eq, "Keq", None) is not None else None,
            }
        )
    return {
        "species_names": species_names,
        "parameter_symbols": list(parameter_symbols),
        "reactions": reactions_payload,
        "equilibria": equilibria_payload,
        "symbolic_support": dict(support_payload),
    }


def _structure_parameter_symbols(mechanism: Mechanism) -> tuple[str, ...]:
    parameter_names: set[str] = set()
    for reaction_index, rxn in enumerate(getattr(mechanism, "reactions", []) or []):
        rate = getattr(rxn, "rate", None)
        parameter_names.add(
            _canonical_step_parameter_name(
                mechanism,
                kind="reaction",
                item_index=int(reaction_index),
                role="k",
                fallback_name=f"k{int(reaction_index) + 1}",
                value=rate,
            )
        )
    for equilibrium_index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        kf = getattr(eq, "kf", None)
        kr = getattr(eq, "kr", None)
        keq = getattr(eq, "Keq", None)
        kf_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="kf",
            fallback_name=f"kf{int(equilibrium_index) + 1}",
            value=kf,
        )
        kr_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="kr",
            fallback_name=f"kr{int(equilibrium_index) + 1}",
            value=kr,
        )
        keq_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="Keq",
            fallback_name=f"Keq{int(equilibrium_index) + 1}",
            value=keq,
        )
        if kf is None and kr is not None and keq is not None:
            parameter_names.update((kr_name, keq_name))
            continue
        if kr is None and kf is not None and keq is not None:
            parameter_names.update((kf_name, keq_name))
            continue
        if kf is not None and kr is not None:
            parameter_names.update((kf_name, kr_name))
            continue
    return tuple(sorted(parameter_names))


def _parameter_values_for_mechanism(
    mechanism: Mechanism,
    parameter_symbols: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for reaction_index, rxn in enumerate(getattr(mechanism, "reactions", []) or []):
        rate = getattr(rxn, "rate", None)
        name = _canonical_step_parameter_name(
            mechanism,
            kind="reaction",
            item_index=int(reaction_index),
            role="k",
            fallback_name=f"k{int(reaction_index) + 1}",
            value=rate,
        )
        values[name] = _finite_scalar(rate, label=f"symbolic parameter {name}")
    for equilibrium_index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        kf = getattr(eq, "kf", None)
        kr = getattr(eq, "kr", None)
        keq = getattr(eq, "Keq", None)
        kf_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="kf",
            fallback_name=f"kf{int(equilibrium_index) + 1}",
            value=kf,
        )
        kr_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="kr",
            fallback_name=f"kr{int(equilibrium_index) + 1}",
            value=kr,
        )
        keq_name = _canonical_step_parameter_name(
            mechanism,
            kind="equilibrium",
            item_index=int(equilibrium_index),
            role="Keq",
            fallback_name=f"Keq{int(equilibrium_index) + 1}",
            value=keq,
        )
        if kf is None and kr is not None and keq is not None:
            values[kr_name] = _finite_scalar(kr, label=f"symbolic parameter {kr_name}")
            values[keq_name] = _finite_scalar(keq, label=f"symbolic parameter {keq_name}")
            continue
        if kr is None and kf is not None and keq is not None:
            values[kf_name] = _finite_scalar(kf, label=f"symbolic parameter {kf_name}")
            values[keq_name] = _finite_scalar(keq, label=f"symbolic parameter {keq_name}")
            continue
        if kf is not None and kr is not None:
            values[kf_name] = _finite_scalar(kf, label=f"symbolic parameter {kf_name}")
            values[kr_name] = _finite_scalar(kr, label=f"symbolic parameter {kr_name}")
            continue

    out: dict[str, float] = {}
    for name in parameter_symbols:
        if name not in values:
            raise UnsupportedSymbolicExpressionError(f"Missing symbolic parameter value for {name!r}.")
        out[str(name)] = float(values[name])
    return out


def symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism: Mechanism) -> str:
    support = classify_symbolic_jacobian_support(mechanism)
    support.raise_if_unsupported()
    species_names_func = getattr(mechanism, "species_names", None)
    if not callable(species_names_func):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires a Kindred mechanism with species_names()."
        )
    species_names = tuple(str(name) for name in species_names_func())
    if not species_names:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires at least one species.")
    parameter_symbols = _structure_parameter_symbols(mechanism)
    return _fingerprint(_structure_identity_payload(mechanism, species_names, parameter_symbols, support.payload))


def build_symbolic_jacobian_structure(mechanism: Mechanism) -> SymbolicJacobianStructure:
    support = classify_symbolic_jacobian_support(mechanism)
    support.raise_if_unsupported()
    sympy = require_sympy()
    species_names_func = getattr(mechanism, "species_names", None)
    if not callable(species_names_func):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires a Kindred mechanism with species_names()."
        )
    species_names = tuple(str(name) for name in species_names_func())
    if not species_names:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires at least one species.")
    state_context = make_state_symbol_context(species_names)
    state_symbols = tuple(sympy.Symbol(name) for name in state_context.symbol_names)
    symbol_by_species = dict(zip(species_names, state_symbols))
    registry = _ParameterRegistry(sympy)
    rhs = [sympy.Integer(0) for _name in species_names]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    for reaction_index, rxn in enumerate(getattr(mechanism, "reactions", []) or []):
        rate_expr = _reaction_rate_expr(
            sympy,
            mechanism,
            rxn,
            symbol_by_species,
            registry,
            reaction_index=reaction_index,
        )
        for species_name, coeff in getattr(rxn, "net_stoich", {}).items():
            rhs[species_index[str(species_name)]] += sympy.Float(float(coeff)) * rate_expr

    for equilibrium_index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        rate_expr = _equilibrium_rate_expr(
            sympy,
            mechanism,
            eq,
            symbol_by_species,
            registry,
            equilibrium_index=equilibrium_index,
        )
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
    parameter_symbols = registry.parameter_names
    structure_fingerprint = _fingerprint(_structure_identity_payload(mechanism, species_names, parameter_symbols, support.payload))
    artifact_fingerprint = _fingerprint(
        {
            "rhs": rhs_strings,
            "jacobian": jacobian_strings,
            "parameter_symbols": list(parameter_symbols),
        }
    )
    parameter_sympy_symbols = tuple(sympy.Symbol(name) for name in parameter_symbols)
    compiled = sympy.lambdify((*state_symbols, *parameter_sympy_symbols), jacobian_matrix, modules="numpy")
    return SymbolicJacobianStructure(
        species_names=species_names,
        parameter_symbols=parameter_symbols,
        rhs_expressions=rhs_strings,
        jacobian_expressions=jacobian_strings,
        structure_fingerprint=structure_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        _compiled=compiled,
        state_symbol_context=state_context.to_payload(),
    )


def build_symbolic_jacobian_artifact(mechanism: Mechanism) -> SymbolicJacobianArtifact:
    structure = build_symbolic_jacobian_structure(mechanism)
    return structure.bind(_parameter_values_for_mechanism(mechanism, structure.parameter_symbols))
