from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np

from kindred.core.mechanism import Equilibrium, FrozenDict, Mechanism
from kindred.core.ode_builder import (
    EffectiveRateCoefficientEvaluator,
    build_effective_rate_coefficient_evaluator,
)
from kindred.core.temperature import TemperatureScheduleProtocol
from kindred.core.kinetics import (
    forward_rate_model_for_reaction,
    forward_rate_symbolic_relation,
    forward_rate_family,
    forward_rate_model_for_equilibrium,
)
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    admit_explicit_directional_rate_pair,
    equilibrium_rate_symbolic_relations,
    effective_reverse_rate_from_keq,
    normalize_existing_equilibrium_rate_authority,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
    canonical_name_for_mechanism_step_parameter,
)

from .artifacts import SymbolicArtifactIdentity
from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .identity import symbolic_fingerprint
from .namespaces import symbolic_status_payload


@dataclass(frozen=True, slots=True)
class SymbolicJacobianArtifact:
    identity: SymbolicArtifactIdentity
    jacobian_func: Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructure:
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[str, ...]
    jacobian_expressions: tuple[tuple[str, ...], ...]
    structure_fingerprint: str
    artifact_fingerprint: str
    _compiled: Callable[..., Any]
    jac_sparsity: tuple[tuple[bool, ...], ...]

    def bind(
        self,
        coefficient_evaluator: EffectiveRateCoefficientEvaluator,
        *,
        temperature_schedule: TemperatureScheduleProtocol | None = None,
        schedule_identity: Mapping[str, object] | None = None,
    ) -> SymbolicJacobianArtifact:
        if not isinstance(
            coefficient_evaluator,
            EffectiveRateCoefficientEvaluator,
        ):
            raise TypeError(
                "coefficient_evaluator must be "
                "EffectiveRateCoefficientEvaluator."
            )
        if temperature_schedule is not None and not callable(
            temperature_schedule
        ):
            raise TypeError("temperature_schedule must be callable.")
        nominal_temperature = float(
            coefficient_evaluator.nominal_temperature_K
        )
        coefficient_parameter_symbols = (
            _coefficient_parameter_symbols_from_step_kinds(
                coefficient_evaluator.step_kinds
            )
        )
        execution_payload = FrozenDict(
            {
                "kind": "runtime-effective-rate-evaluation",
                "evaluation_mode": (
                    "per-call-typed-schedule"
                    if temperature_schedule is not None
                    else "nominal-temperature-fallback"
                ),
                "nominal_temperature_K": nominal_temperature,
                "typed_schedule": (
                    FrozenDict(schedule_identity)
                    if schedule_identity is not None
                    else None
                ),
                "rate_authority_snapshot": (
                    coefficient_evaluator.authority_snapshot
                ),
                "parameter_symbols": self.parameter_symbols,
            }
        )
        snapshot_fingerprint = symbolic_fingerprint(execution_payload)
        metadata = get_symbolic_backend_metadata()
        identity = SymbolicArtifactIdentity.jacobian(
            metadata,
            source_fingerprint=self.structure_fingerprint,
            artifact_fingerprint=self.artifact_fingerprint,
            structure_fingerprint=self.structure_fingerprint,
            evaluation_snapshot_fingerprint=snapshot_fingerprint,
            parameter_symbols=self.parameter_symbols,
        )

        def jacobian_func(t: float, y: np.ndarray) -> np.ndarray:
            values = np.asarray(y, dtype=float).reshape(-1)
            if values.size != len(self.species_names):
                raise ValueError(f"symbolic jacobian expected {len(self.species_names)} state values, got {values.size}")
            temperature = (
                nominal_temperature
                if temperature_schedule is None
                else float(temperature_schedule(float(t)))
            )
            coefficients = coefficient_evaluator(temperature)
            coefficient_values = _coefficient_values_by_parameter_name(
                coefficient_evaluator,
                coefficients.forward,
                coefficients.reverse,
            )
            parameter_tuple = tuple(
                coefficient_values[name]
                for name in coefficient_parameter_symbols
            )
            return np.asarray(
                self._compiled(*values, *parameter_tuple),
                dtype=float,
            )

        return SymbolicJacobianArtifact(
            identity=identity,
            jacobian_func=jacobian_func,
        )


@dataclass(frozen=True, slots=True)
class _SymbolicMechanismExpressionModel:
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[Any, ...]
    jacobian_expressions: tuple[tuple[Any, ...], ...]
    state_symbols: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SymbolicMechanismPublicProjection:
    """Canonical public view of the single private symbolic mechanism graph."""

    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[Any, ...]
    jacobian_expressions: tuple[tuple[Any, ...], ...]
    state_symbols: tuple[Any, ...]
    chemistry_relations: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SymbolicJacobianSupport:
    supported: bool
    code: str
    reason: str

    def raise_if_unsupported(self) -> None:
        if not self.supported:
            raise UnsupportedSymbolicExpressionError(self.reason)

    def to_status_payload(self) -> dict[str, str]:
        state = "supported" if self.supported else "unsupported"
        reason = self.reason if self.reason else "Symbolic Jacobian supported."
        return symbolic_status_payload(kind="jacobian", state=state, code=self.code, reason=reason)


def _coefficient_parameter_name(
    step_index: int,
    direction: str,
) -> str:
    role = "kf" if direction == "forward" else "kr"
    return f"__kindred_effective_{role}_{int(step_index) + 1}"


def _symbolic_coefficient_parameter_symbols(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    names: list[str] = []
    step_index = 0
    for _reaction in getattr(mechanism, "reactions", ()) or ():
        names.append(
            _coefficient_parameter_name(step_index, "forward")
        )
        step_index += 1
    for _equilibrium in getattr(mechanism, "equilibria", ()) or ():
        names.extend(
            (
                _coefficient_parameter_name(step_index, "forward"),
                _coefficient_parameter_name(step_index, "reverse"),
            )
        )
        step_index += 1
    return tuple(sorted(names))


def _coefficient_parameter_symbols_from_step_kinds(
    step_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for step_index, kind in enumerate(step_kinds):
        names.append(_coefficient_parameter_name(step_index, "forward"))
        if str(kind) == "equilibrium":
            names.append(_coefficient_parameter_name(step_index, "reverse"))
    return tuple(sorted(names))


def _coefficient_values_by_parameter_name(
    evaluator: EffectiveRateCoefficientEvaluator,
    forward: tuple[float, ...],
    reverse: tuple[float, ...],
) -> dict[str, float]:
    if not (
        len(evaluator.step_kinds)
        == len(forward)
        == len(reverse)
    ):
        raise UnsupportedSymbolicExpressionError(
            "Effective rate coefficient shape conflicts with symbolic topology."
        )
    values: dict[str, float] = {}
    for step_index, kind in enumerate(evaluator.step_kinds):
        values[
            _coefficient_parameter_name(step_index, "forward")
        ] = float(forward[step_index])
        if kind == "equilibrium":
            values[
                _coefficient_parameter_name(step_index, "reverse")
            ] = float(reverse[step_index])
    return values


@dataclass(frozen=True, slots=True)
class _EquilibriumRateSource:
    kf: object | None
    kr: object | None
    keq: object | None
    kf_name: str | None
    kr_name: str | None
    keq_name: str | None


def _canonical_step_parameter_name(
    mechanism: Mechanism,
    *,
    kind: str,
    item_index: int,
    role: str,
    expected_name: str,
    value: object = None,
) -> str:
    canonical_name = canonical_name_for_mechanism_step_parameter(
        mechanism,
        kind=kind,
        item_index=int(item_index),
        role=role,
        expected_name=str(expected_name),
    )
    explicit_name = str(getattr(value, "name", None) or "").strip()
    if explicit_name:
        namespace = build_namespace_from_mechanism(mechanism)
        resolution = namespace.resolve(explicit_name)
        if resolution.canonical_name is not None:
            resolved = str(resolution.canonical_name)
            if resolved != canonical_name:
                raise UnsupportedSymbolicExpressionError(
                    f"Symbolic Jacobian binding name {explicit_name!r} resolves to {resolved!r}, "
                    f"but this step parameter is {canonical_name!r}."
                )
            return resolved
        invalid_message = namespace.invalid_protected_indexed_identifier_message(explicit_name)
        if invalid_message is not None:
            raise UnsupportedSymbolicExpressionError(invalid_message)
    return canonical_name


def _runtime_scalar_binding_value(value: object, *, label: str) -> object:
    if (
        callable(value)
        and callable(getattr(value, "set", None))
        and str(getattr(value, "name", "") or "").strip()
    ):
        try:
            return value()
        except Exception as exc:
            raise UnsupportedSymbolicExpressionError(
                f"{label} binding could not provide a finite scalar."
            ) from exc
    return value


def _finite_scalar(value: object, *, label: str) -> float:
    value = _runtime_scalar_binding_value(value, label=label)
    if callable(value):
        raise UnsupportedSymbolicExpressionError(f"Dynamic callable {label} is not supported for symbolic Jacobian.")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedSymbolicExpressionError(f"{label} must be a finite scalar.") from exc
    if not math.isfinite(out):
        raise UnsupportedSymbolicExpressionError(f"{label} must be finite.")
    return out


def _stoichiometric_coefficient_expr(sympy: Any, coeff: float, *, exact: bool) -> Any:
    if bool(exact):
        return sympy.Rational(str(float(coeff)))
    return sympy.Float(float(coeff))


def _power_product(sympy: Any, symbols: Mapping[str, Any], powers: Mapping[str, object]) -> Any:
    expr = sympy.Integer(1)
    for species_name, raw_power in sorted((powers or {}).items()):
        power = _finite_scalar(raw_power, label=f"stoichiometric power for {species_name}")
        if species_name not in symbols:
            raise UnsupportedSymbolicExpressionError(f"Unknown species {species_name!r} in symbolic Jacobian.")
        expr *= symbols[str(species_name)] ** sympy.Rational(str(power))
    return expr


def _explicit_keq_source(eq: Equilibrium) -> object | None:
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        keq_input = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        if keq_input is not None:
            return keq_input
    return getattr(eq, "Keq", None)


def _equilibrium_rate_source(
    mechanism: Mechanism,
    eq: Equilibrium,
    *,
    equilibrium_index: int,
) -> _EquilibriumRateSource:
    kf = getattr(eq, "kf", None)
    kr = getattr(eq, "kr", None)
    keq = _explicit_keq_source(eq)
    kf_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kf",
        expected_name=f"kf{int(equilibrium_index) + 1}",
        value=kf,
    )
    kr_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kr",
        expected_name=f"kr{int(equilibrium_index) + 1}",
        value=kr,
    )
    keq_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="Keq",
        expected_name=f"Keq{int(equilibrium_index) + 1}",
        value=keq,
    )

    try:
        authority = normalize_existing_equilibrium_rate_authority(eq)
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(
            f"Invalid equilibrium rate authority: {exc}"
        ) from exc
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        keq = None
    elif authority.kind == EquilibriumRateAuthorityKind.KEQ:
        keq = authority.Keq
    return _EquilibriumRateSource(
        kf=kf,
        kr=kr,
        keq=keq,
        kf_name=kf_name,
        kr_name=kr_name,
        keq_name=keq_name,
    )


def _unsupported_support(code: str, reason: str) -> SymbolicJacobianSupport:
    return SymbolicJacobianSupport(
        supported=False,
        code=str(code),
        reason=str(reason),
    )


def _symbolic_snapshot_scalar_value(value: object) -> float | None:
    try:
        scalar_value = _runtime_scalar_binding_value(value, label="symbolic snapshot scalar")
        if callable(scalar_value):
            return None
        scalar = float(scalar_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError, UnsupportedSymbolicExpressionError):
        return None
    if not math.isfinite(scalar):
        return None
    return float(scalar)


def _validate_equilibrium_parameter_values(
    source: _EquilibriumRateSource,
) -> None:
    kf_label = source.kf_name or "equilibrium kf"
    kr_label = source.kr_name or "equilibrium kr"
    keq_label = source.keq_name or "equilibrium Keq"
    kf_value = _finite_scalar(source.kf, label=f"symbolic parameter {kf_label}")
    if kf_value < 0.0:
        raise UnsupportedSymbolicExpressionError(
            "equilibrium kf must be finite and nonnegative."
        )
    kr_value = _symbolic_snapshot_scalar_value(source.kr) if source.kr is not None else None
    keq_value = _symbolic_snapshot_scalar_value(source.keq) if source.keq is not None else None

    if source.keq is not None and keq_value is None:
        raise UnsupportedSymbolicExpressionError(
            f"equilibrium Keq input {keq_label} must be finite for symbolic Jacobian."
        )
    if keq_value is not None:
        if keq_value <= 0.0:
            raise UnsupportedSymbolicExpressionError("equilibrium Keq must be positive.")
        _ = float(
            effective_reverse_rate_from_keq(
                kf_value,
                keq_value,
            )
        )
        return

    if kr_value is None:
        if source.kr is not None:
            raise UnsupportedSymbolicExpressionError(
                f"equilibrium kr input {kr_label} must be finite for symbolic Jacobian."
            )
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires finite equilibrium kf plus finite kr or Keq."
        )
    try:
        admit_explicit_directional_rate_pair(kf_value, kr_value)
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(
            "explicit equilibrium kf and kr must each be an admitted "
            "strictly positive rate."
        ) from exc


def _authored_equilibrium_forward_rate_family(
    mechanism: Mechanism,
    eq: Equilibrium,
) -> str:
    """Validate and return one authored typed forward-rate family."""

    mechanism_metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    try:
        model = forward_rate_model_for_equilibrium(
            eq,
            temperature_K=float(mechanism_metadata.temperature_K),
            default_standard_conc_M=float(
                mechanism_metadata.standard_conc_M
            ),
        )
        family = forward_rate_family(model)
    except (TypeError, ValueError) as exc:
        raise UnsupportedSymbolicExpressionError(
            f"Unsupported authored equilibrium forward-rate model: {exc}"
        ) from exc
    return family


def classify_symbolic_jacobian_support(mechanism: Mechanism) -> SymbolicJacobianSupport:
    species_names_func = getattr(mechanism, "species_names", None)
    if not callable(species_names_func):
        return _unsupported_support(
            "missing-species",
            "Symbolic Jacobian requires a Kindred mechanism with species_names().",
        )
    species_names = tuple(str(name) for name in species_names_func())
    if not species_names:
        return _unsupported_support(
            "missing-species",
            "Symbolic Jacobian requires at least one species.",
        )

    unsupported_code = ""
    unsupported_reason = ""
    for idx, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
        try:
            rate_value = _runtime_scalar_binding_value(
                getattr(rxn, "rate", None),
                label=f"reaction rate k{idx}",
            )
            if callable(rate_value):
                raise TypeError("dynamic reaction rate")
            rate_value = float(rate_value)
        except (TypeError, ValueError, OverflowError, UnsupportedSymbolicExpressionError):
            rate_value = None
        if not unsupported_code and (
            rate_value is None or not math.isfinite(rate_value)
        ):
            unsupported_code = "dynamic-rate"
            unsupported_reason = f"Symbolic Jacobian does not support dynamic or non-finite reaction rate k{idx}."
    for idx, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
        try:
            source = _equilibrium_rate_source(
                mechanism,
                eq,
                equilibrium_index=idx - 1,
            )
        except UnsupportedSymbolicExpressionError as exc:
            if not unsupported_code:
                unsupported_code = "invalid-equilibrium-rate-authority"
                unsupported_reason = str(exc)
            continue
        try:
            _authored_equilibrium_forward_rate_family(mechanism, eq)
        except UnsupportedSymbolicExpressionError as exc:
            if not unsupported_code:
                unsupported_code = "unsupported-equilibrium-rate-model"
                unsupported_reason = str(exc)
        if unsupported_code:
            continue
        try:
            _validate_equilibrium_parameter_values(source)
        except UnsupportedSymbolicExpressionError as exc:
            unsupported_reason = str(exc)
            unsupported_code = (
                "unsupported-equilibrium-keq"
                if "Keq" in unsupported_reason
                else "unsupported-equilibrium-parameters"
            )
            continue
    if unsupported_code:
        return _unsupported_support(unsupported_code, unsupported_reason)
    try:
        evaluator = build_effective_rate_coefficient_evaluator(mechanism)
        evaluator(evaluator.nominal_temperature_K)
    except (TypeError, ValueError, OverflowError) as exc:
        return _unsupported_support(
            "unsupported-effective-rate-authority",
            f"Symbolic Jacobian effective-rate authority is unsupported: {exc}",
        )
    return SymbolicJacobianSupport(
        supported=True,
        code="supported",
        reason="",
    )


def _structure_identity_payload(
    mechanism: Mechanism,
    species_names: tuple[str, ...],
    parameter_symbols: tuple[str, ...],
) -> dict[str, Any]:
    reactions_payload = []
    for idx, rxn in enumerate(getattr(mechanism, "reactions", []) or [], start=1):
        reactions_payload.append(
            {
                "index": idx,
                "reactants": dict(getattr(rxn, "reactants", {}) or {}),
                "products": dict(getattr(rxn, "products", {}) or {}),
                "rate_orders": dict(getattr(rxn, "rate_orders", {}) or {}),
                "net_stoich": dict(getattr(rxn, "net_stoich", {}) or {}),
                "forward_coefficient": _coefficient_parameter_name(
                    idx - 1,
                    "forward",
                ),
            }
        )
    equilibria_payload = []
    reaction_count = len(reactions_payload)
    for idx, eq in enumerate(getattr(mechanism, "equilibria", []) or [], start=1):
        step_index = reaction_count + idx - 1
        equilibria_payload.append(
            {
                "index": idx,
                "stoich_forward": dict(getattr(eq, "stoich_forward", {}) or {}),
                "stoich_back": dict(getattr(eq, "stoich_back", {}) or {}),
                "forward_coefficient": _coefficient_parameter_name(
                    step_index,
                    "forward",
                ),
                "reverse_coefficient": _coefficient_parameter_name(
                    step_index,
                    "reverse",
                ),
            }
        )
    return {
        "species_names": species_names,
        "parameter_symbols": list(parameter_symbols),
        "reactions": reactions_payload,
        "equilibria": equilibria_payload,
    }


def _coefficient_public_parameter_names(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    public_by_internal: dict[str, str] = {}
    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    try:
        for reaction_index, reaction in enumerate(reactions):
            internal_name = _coefficient_parameter_name(
                reaction_index,
                "forward",
            )
            public_by_internal[internal_name] = _canonical_step_parameter_name(
                mechanism,
                kind="reaction",
                item_index=reaction_index,
                role="k",
                expected_name=f"k{reaction_index + 1}",
                value=getattr(reaction, "rate", None),
            )
        reaction_count = len(reactions)
        for equilibrium_index, equilibrium in enumerate(equilibria):
            step_index = reaction_count + equilibrium_index
            source = _equilibrium_rate_source(
                mechanism,
                equilibrium,
                equilibrium_index=equilibrium_index,
            )
            if source.kf_name is None or source.kr_name is None:
                raise UnsupportedSymbolicExpressionError(
                    "Equilibrium public coefficient names are incomplete."
                )
            public_by_internal[
                _coefficient_parameter_name(step_index, "forward")
            ] = source.kf_name
            public_by_internal[
                _coefficient_parameter_name(step_index, "reverse")
            ] = source.kr_name
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(str(exc)) from exc
    internal_names = _symbolic_coefficient_parameter_symbols(mechanism)
    try:
        return tuple(public_by_internal[name] for name in internal_names)
    except KeyError as exc:
        raise UnsupportedSymbolicExpressionError(
            "Canonical symbolic coefficient projection is incomplete."
        ) from exc


def _canonical_public_parameter_symbols(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    try:
        namespace = build_namespace_from_mechanism(mechanism)
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(str(exc)) from exc
    return tuple(
        str(item.canonical_name)
        for item in namespace.ordered_items
    )


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
    parameter_symbols = _symbolic_coefficient_parameter_symbols(mechanism)
    identity_payload = _structure_identity_payload(
        mechanism,
        species_names,
        parameter_symbols,
    )
    identity_payload["public_parameter_symbols"] = list(
        _canonical_public_parameter_symbols(mechanism)
    )
    identity_payload["coefficient_public_projection"] = list(
        zip(
            parameter_symbols,
            _coefficient_public_parameter_names(mechanism),
            strict=True,
        )
    )
    return symbolic_fingerprint(identity_payload)


def _build_symbolic_mechanism_expression_model(
    mechanism: Mechanism,
    *,
    exact_stoichiometric_coefficients: bool = False,
) -> _SymbolicMechanismExpressionModel:
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
    state_symbols = tuple(
        sympy.Dummy(f"y_{idx}") for idx, _name in enumerate(species_names)
    )
    symbol_by_species = dict(zip(species_names, state_symbols))
    rhs = [sympy.Integer(0) for _name in species_names]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    for reaction_index, rxn in enumerate(getattr(mechanism, "reactions", []) or []):
        coefficient = sympy.Symbol(
            _coefficient_parameter_name(
                reaction_index,
                "forward",
            )
        )
        rate_expr = coefficient * _power_product(
            sympy,
            symbol_by_species,
            getattr(rxn, "rate_orders", {}) or {},
        )
        for species_name, coeff in getattr(rxn, "net_stoich", {}).items():
            rhs[species_index[str(species_name)]] += _stoichiometric_coefficient_expr(
                sympy,
                float(coeff),
                exact=bool(exact_stoichiometric_coefficients),
            ) * rate_expr

    reaction_count = len(tuple(getattr(mechanism, "reactions", []) or ()))
    for equilibrium_index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        step_index = reaction_count + equilibrium_index
        forward_coefficient = sympy.Symbol(
            _coefficient_parameter_name(step_index, "forward")
        )
        reverse_coefficient = sympy.Symbol(
            _coefficient_parameter_name(step_index, "reverse")
        )
        forward_expr = forward_coefficient * _power_product(
            sympy,
            symbol_by_species,
            getattr(eq, "stoich_forward", {}) or {},
        )
        reverse_expr = reverse_coefficient * _power_product(
            sympy,
            symbol_by_species,
            getattr(eq, "stoich_back", {}) or {},
        )
        rate_expr = forward_expr - reverse_expr
        for species_name in species_names:
            coeff = float(getattr(eq, "stoich_back", {}).get(species_name, 0.0)) - float(
                getattr(eq, "stoich_forward", {}).get(species_name, 0.0)
            )
            if coeff:
                rhs[species_index[species_name]] += _stoichiometric_coefficient_expr(
                    sympy,
                    coeff,
                    exact=bool(exact_stoichiometric_coefficients),
                ) * rate_expr

    rhs_matrix = sympy.Matrix(rhs)
    jacobian_matrix = rhs_matrix.jacobian(sympy.Matrix(state_symbols))
    parameter_symbols = _symbolic_coefficient_parameter_symbols(mechanism)
    return _SymbolicMechanismExpressionModel(
        species_names=species_names,
        parameter_symbols=parameter_symbols,
        rhs_expressions=tuple(sympy.simplify(expr) for expr in rhs_matrix),
        jacobian_expressions=tuple(
            tuple(sympy.simplify(jacobian_matrix[i, j]) for j in range(len(species_names)))
            for i in range(len(species_names))
        ),
        state_symbols=state_symbols,
    )


def _project_symbolic_mechanism_expression_model(
    mechanism: Mechanism,
    *,
    model: _SymbolicMechanismExpressionModel,
) -> SymbolicMechanismPublicProjection:
    """Project the single private graph onto canonical public chemistry atoms."""

    sympy = require_sympy()
    public_state_symbols = tuple(
        sympy.Symbol(str(name))
        for name in model.species_names
    )
    if len(public_state_symbols) != len(model.state_symbols):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic state projection is inconsistent."
        )

    internal_parameter_names = tuple(model.parameter_symbols)
    coefficient_public_names = _coefficient_public_parameter_names(mechanism)
    if len(coefficient_public_names) != len(internal_parameter_names):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic coefficient projection is inconsistent."
        )
    public_parameter_symbols = _canonical_public_parameter_symbols(mechanism)
    public_symbols_by_name = {
        name: sympy.Symbol(name)
        for name in public_parameter_symbols
    }
    replacements = {
        internal: public
        for internal, public in zip(
            model.state_symbols,
            public_state_symbols,
            strict=True,
        )
    }
    replacements.update(
        {
            sympy.Symbol(internal_name): public_symbols_by_name[public_name]
            for internal_name, public_name in zip(
                internal_parameter_names,
                coefficient_public_names,
                strict=True,
            )
        }
    )

    rhs_expressions = tuple(
        sympy.simplify(expression.xreplace(replacements))
        for expression in model.rhs_expressions
    )
    jacobian_expressions = tuple(
        tuple(
            sympy.simplify(expression.xreplace(replacements))
            for expression in row
        )
        for row in model.jacobian_expressions
    )

    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )

    def exact_literal(value: float) -> Any:
        return sympy.Rational(str(float(value)))

    temperature = sympy.Symbol("T")
    chemistry_relations: dict[str, Any] = {}
    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    for reaction_index, reaction in enumerate(reactions):
        public_name = _canonical_step_parameter_name(
            mechanism,
            kind="reaction",
            item_index=reaction_index,
            role="k",
            expected_name=f"k{reaction_index + 1}",
            value=getattr(reaction, "rate", None),
        )
        try:
            rate_model = forward_rate_model_for_reaction(
                reaction,
                default_standard_conc_M=float(
                    metadata.standard_conc_M
                ),
            )
            chemistry_relations[public_name] = (
                forward_rate_symbolic_relation(
                    rate_model,
                    temperature=temperature,
                    literal=exact_literal,
                    exponential=sympy.exp,
                )
            )
        except (TypeError, ValueError) as exc:
            raise UnsupportedSymbolicExpressionError(
                f"Unsupported symbolic reaction rate authority: {exc}"
            ) from exc

    for equilibrium_index, equilibrium in enumerate(
        tuple(getattr(mechanism, "equilibria", ()) or ())
    ):
        source = _equilibrium_rate_source(
            mechanism,
            equilibrium,
            equilibrium_index=equilibrium_index,
        )
        if (
            source.kf_name is None
            or source.kr_name is None
            or source.keq_name is None
        ):
            raise UnsupportedSymbolicExpressionError(
                "Equilibrium canonical public projection is incomplete."
            )
        try:
            forward_model = forward_rate_model_for_equilibrium(
                equilibrium,
                temperature_K=None,
                default_standard_conc_M=float(
                    metadata.standard_conc_M
                ),
            )
            forward_relation = forward_rate_symbolic_relation(
                forward_model,
                temperature=temperature,
                literal=exact_literal,
                exponential=sympy.exp,
            )
            equilibrium_metadata = getattr(
                equilibrium,
                "metadata",
                {},
            ) or {}
            standard_conc = (
                equilibrium_metadata.get(
                    EquilibriumMetadataKeys.STANDARD_CONC_M,
                    metadata.standard_conc_M,
                )
                if isinstance(equilibrium_metadata, Mapping)
                else metadata.standard_conc_M
            )
            if standard_conc is None:
                standard_conc = metadata.standard_conc_M
            relations = equilibrium_rate_symbolic_relations(
                equilibrium,
                forward_relation=forward_relation,
                temperature=temperature,
                standard_conc_M=float(standard_conc),
                literal=exact_literal,
                exponential=sympy.exp,
            )
        except (TypeError, ValueError) as exc:
            raise UnsupportedSymbolicExpressionError(
                f"Unsupported symbolic equilibrium rate authority: {exc}"
            ) from exc
        chemistry_relations[source.kf_name] = relations.kf
        chemistry_relations[source.kr_name] = relations.kr
        chemistry_relations[source.keq_name] = relations.Keq

    if set(chemistry_relations) != set(public_parameter_symbols):
        raise UnsupportedSymbolicExpressionError(
            "Canonical chemistry relation projection is incomplete."
        )
    return SymbolicMechanismPublicProjection(
        species_names=tuple(model.species_names),
        parameter_symbols=public_parameter_symbols,
        rhs_expressions=rhs_expressions,
        jacobian_expressions=jacobian_expressions,
        state_symbols=public_state_symbols,
        chemistry_relations=dict(chemistry_relations),
    )


def build_symbolic_mechanism_public_projection(
    mechanism: Mechanism,
    *,
    exact_stoichiometric_coefficients: bool = False,
) -> SymbolicMechanismPublicProjection:
    """Build the single graph and return only its canonical public projection."""

    model = _build_symbolic_mechanism_expression_model(
        mechanism,
        exact_stoichiometric_coefficients=bool(
            exact_stoichiometric_coefficients
        ),
    )
    return _project_symbolic_mechanism_expression_model(
        mechanism,
        model=model,
    )


def build_symbolic_jacobian_structure(mechanism: Mechanism) -> SymbolicJacobianStructure:
    sympy = require_sympy()
    model = _build_symbolic_mechanism_expression_model(
        mechanism,
        exact_stoichiometric_coefficients=False,
    )
    projection = _project_symbolic_mechanism_expression_model(
        mechanism,
        model=model,
    )
    species_names = model.species_names
    state_symbols = model.state_symbols
    jacobian_matrix = sympy.Matrix(model.jacobian_expressions)
    rhs_strings = tuple(
        str(expression)
        for expression in projection.rhs_expressions
    )
    jacobian_strings = tuple(
        tuple(
            str(expression)
            for expression in row
        )
        for row in projection.jacobian_expressions
    )
    jac_sparsity = tuple(
        tuple(bool(expression != 0) for expression in row)
        for row in jacobian_matrix.tolist()
    )
    coefficient_parameter_symbols = model.parameter_symbols
    parameter_symbols = projection.parameter_symbols
    structure_fingerprint = symbolic_jacobian_structure_fingerprint_for_mechanism(
        mechanism
    )
    artifact_fingerprint = symbolic_fingerprint(
        {
            "rhs": rhs_strings,
            "jacobian": jacobian_strings,
            "parameter_symbols": list(parameter_symbols),
        }
    )
    parameter_sympy_symbols = tuple(
        sympy.Symbol(name)
        for name in coefficient_parameter_symbols
    )
    compiled = sympy.lambdify((*state_symbols, *parameter_sympy_symbols), jacobian_matrix, modules="numpy")
    return SymbolicJacobianStructure(
        species_names=species_names,
        parameter_symbols=parameter_symbols,
        rhs_expressions=rhs_strings,
        jacobian_expressions=jacobian_strings,
        structure_fingerprint=structure_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        _compiled=compiled,
        jac_sparsity=jac_sparsity,
    )
