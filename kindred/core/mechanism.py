"""
Mechanism model: species registry, reactions, and equilibria.

Current contract
----------------
- Mechanism:
  - species: OrderedDict[str, Species] (preserves declaration order)
  - reactions: list[Reaction]
  - equilibria: list[Equilibrium]
  - metadata: dict includes declaration_order: list[str]
- Species:
  - name: str
  - initial_conc: finite non-negative float  (standard state basis)
- Reaction:
  - reactants/products: immutable Mapping[str, float] positive physical side maps
  - rate_orders: immutable Mapping[str, float] positive kinetic exponents;
                 None defaults to reactants, {} is explicit zero-order
  - net_stoich: immutable Mapping[str, float] products - reactants, zeros omitted
- order: float  exact finite molecularity from rate_orders
  - rate: StructuredRate | Callable    (opaque holder here; evaluated upstream)
  - overrides: {"model": "Constant" | "Eyring" | "Arrhenius" | None,
                "kappa": float | None, "A": float | None,
                "Ea_J_per_mol": float | None, "dG_act_J_per_mol": float | None,
                "standard_conc_M": float | None}
- Equilibrium:
  - stoich_forward: immutable Mapping[str, float] positive physical forward side
  - stoich_back: immutable Mapping[str, float] positive physical reverse side
  - Keq: float | Expr | None
  - kf: float | Expr | None
  - kr: float | Expr | None
  - fast: bool  (true if originated from `equilibrium:`)

Design notes
------------
- This file does not evaluate rates or assemble ODEs; it only holds a validated,
  deterministic description of the mechanism. Solver and simulator layers map
  these structures to kinetics later.
- Molecularity/order for Reaction is the exact finite floating-point sum of
  its positive directional rate-order coefficients. Reaction-side
  coefficients admit every positive finite real value.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Tuple, Union, Protocol, Optional

from .species import Species
from .validation import validate_name
from .equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    EquilibriumRateInputContext,
    effective_equilibrium_forward_rate,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    effective_reverse_rate_from_keq,
    json_safe_equilibrium_metadata_payload,
    json_safe_equilibrium_authority_payload,
    normalize_equilibrium_rate_authority,
    normalize_existing_equilibrium_rate_authority,
    plain_finite_float_or_none,
)
from .kinetics import K_from_deltaG_eq
from .mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataKeys
from .temperature import coerce_temperature_kelvin

class StructuredRate(Protocol):
    pass
class Expr(Protocol):
    pass
__all__ = [
    "Reaction",
    "Equilibrium",
    "Mechanism",
    "derive_molecularity",
]


class _SpeciesMap(OrderedDict):
    def __init__(self, values: Mapping[str, Species] | Iterable[tuple[str, Species]] | None = None) -> None:
        super().__init__()
        if values is not None:
            self.update(values)

    def __setitem__(self, key: object, value: object) -> None:
        name = validate_name(key)
        if not isinstance(value, Species):
            raise TypeError("mechanism species entries must be Species objects.")
        if value.name != name:
            raise ValueError("mechanism species map key must match Species.name.")
        super().__setitem__(name, value)

    def update(self, values: Mapping[str, Species] | Iterable[tuple[str, Species]] = (), **kwargs: Species) -> None:  # type: ignore[override]
        items = values.items() if isinstance(values, MappingABC) else values
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value


# --------------------------------- helpers ----------------------------------


class FrozenDict(MappingABC):
    """Small immutable, pickleable mapping used for mechanism semantic fields."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._data = {key: _freeze_value(value) for key, value in dict(values or {}).items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MappingABC):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[Dict[str, Any]]]:
        return (type(self), (self._data,))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, MappingABC):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _clone_authority_slot(value: Any) -> Any:
    if callable(getattr(value, "set", None)):
        return copy.deepcopy(value)
    return value


def _ensure_species_exist(stoich: Mapping[str, float], declared: Mapping[str, Species]) -> None:
    for n in stoich:
        if n not in declared:
            raise KeyError(f"unknown species in stoichiometry: {n!r}")


def _algebra_references_species(mechanism: object, species_name: str) -> bool:
    """Inspect the retained typed Algebra AST without reparsing source text."""

    from .algebra.parser import (
        BinaryNode,
        CallNode,
        SpeciesRefNode,
        UnaryNode,
    )

    def references(expression: object) -> bool:
        if isinstance(expression, SpeciesRefNode):
            return expression.name == species_name
        if isinstance(expression, UnaryNode):
            return references(expression.rhs)
        if isinstance(expression, BinaryNode):
            return references(expression.lhs) or references(expression.rhs)
        if isinstance(expression, CallNode):
            return any(references(argument) for argument in expression.args)
        return False

    metadata = getattr(mechanism, "metadata", {}) or {}
    spec = (
        metadata.get("parameter_algebra_spec")
        if isinstance(metadata, MappingABC)
        else None
    )
    compiled = getattr(mechanism, "_compiled_algebra_series", None)
    blocks = (
        getattr(spec, "parameter_block", None),
        getattr(spec, "observable_block", None),
        getattr(compiled, "block", None),
    )
    seen: set[int] = set()
    for block in blocks:
        if block is None or id(block) in seen:
            continue
        seen.add(id(block))
        for statement in tuple(getattr(block, "lines", ()) or ()):
            if references(getattr(statement, "expr", None)):
                return True
    return False


def derive_molecularity(rate_orders: Mapping[str, float]) -> float:
    """
    Derive reaction molecularity (order) as the sum of kinetic exponents.

    Coefficients like 0.5A + B -> ... yield order 1.5. Rate orders are
    already validated positive finite real coefficients at this boundary.
    """
    total = math.fsum(float(v) for v in rate_orders.values())
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("reaction directional order must be finite and non-negative")
    return float(total)


def _validate_positive_side(side: Mapping[str, float] | None, *, label: str) -> Dict[str, float]:
    """Validate a physical reaction side with positive coefficients."""
    if side is None:
        return {}
    if not isinstance(side, Mapping):
        raise TypeError(f"{label} must be a mapping of name -> positive coefficient")
    out: Dict[str, float] = {}
    for k, v in side.items():
        n = validate_name(k)
        if type(v) is bool:
            raise TypeError(f"{label} coefficient for {n!r} must be a positive finite real")
        try:
            fv = float(v)
        except (ValueError, TypeError, OverflowError) as e:
            raise TypeError(f"{label} coefficient for {n!r} must be a positive finite real, got {v!r}") from e
        if not math.isfinite(fv) or fv <= 0.0:
            raise ValueError(f"{label} coefficient for {n!r} must be a positive finite real")
        out[n] = fv
    return out


def _derive_net_stoich(reactants: Mapping[str, float], products: Mapping[str, float]) -> Dict[str, float]:
    net: Dict[str, float] = {}
    for sp, coef in reactants.items():
        net[sp] = net.get(sp, 0.0) - float(coef)
    for sp, coef in products.items():
        net[sp] = net.get(sp, 0.0) + float(coef)
    return {sp: coef for sp, coef in net.items() if coef != 0.0}


def _normalize_overrides(overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Validate final structured rate-override keys and normalize numeric values.
    """
    allowed = {
        "model",
        "kappa",
        "A",
        "Ea_J_per_mol",
        "dG_act_J_per_mol",
        "standard_conc_M",
    }
    out: Dict[str, Any] = {}
    if not overrides:
        return out
    unknown = sorted(str(key) for key in overrides if key not in allowed)
    if unknown:
        raise ValueError(
            "Unsupported reaction rate override field(s): "
            + ", ".join(unknown)
        )
    for k in allowed:
        if k in overrides:
            out[k] = overrides[k]
    # normalize model
    model = out.get("model", None)
    if model is not None:
        if model not in ("Constant", "Eyring", "Arrhenius"):
            raise ValueError(
                "overrides.model must be 'Constant', 'Eyring', "
                "'Arrhenius', or None"
            )
    # numeric sanity where applicable
    for num_key in (
        "kappa",
        "A",
        "Ea_J_per_mol",
        "dG_act_J_per_mol",
        "standard_conc_M",
    ):
        if num_key in out and out[num_key] is not None:
            if type(out[num_key]) is bool:
                raise TypeError(
                    f"overrides.{num_key} must be numeric if provided"
                )
            try:
                numeric = float(out[num_key])
            except (ValueError, TypeError, OverflowError) as e:
                raise TypeError(f"overrides.{num_key} must be numeric if provided") from e
            if not math.isfinite(numeric):
                raise ValueError(
                    f"overrides.{num_key} must be finite if provided"
                )
            if num_key in {"kappa", "A", "standard_conc_M"} and numeric <= 0.0:
                raise ValueError(
                    f"overrides.{num_key} must be positive and finite"
                )
            out[num_key] = numeric
    return out


# --------------------------------- models -----------------------------------


@dataclass(frozen=True, kw_only=True)
class Reaction:
    """
    Reaction step with physical sides and explicit kinetic order.

    Fields
    ------
    reactants : Mapping[str, float]
        Positive reactant-side coefficients.
    products : Mapping[str, float]
        Positive product-side coefficients.
    rate_orders : Mapping[str, float]
        Positive kinetic exponents; defaults to the reactant side.
    net_stoich : Mapping[str, float]
        Derived products - reactants mapping, zeros omitted.
    rate : Any
        Either a StructuredRate object or a Callable resolved upstream.
        Stored opaquely here.
    overrides : Mapping[str, Any]
        Optional per-step overrides for model parameters.
    order : float
        Exact finite molecularity: the sum of the directional rate orders.
    """
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate: Union[StructuredRate, Callable[[], float]]
    rate_orders: Mapping[str, float] | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    net_stoich: Mapping[str, float] = field(init=False)
    order: float = field(init=False)

    def __post_init__(self) -> None:
        # dataclass with frozen=True; use object.__setattr__
        reactants = _validate_positive_side(self.reactants, label="reactants")
        products = _validate_positive_side(self.products, label="products")
        if not reactants and not products:
            raise ValueError("reaction requires reactants or products")
        rate_orders = (
            dict(reactants)
            if self.rate_orders is None
            else _validate_positive_side(
                self.rate_orders,
                label="rate_orders",
            )
        )
        net_stoich = _derive_net_stoich(reactants, products)
        if not net_stoich:
            raise ValueError("reaction net stoichiometry cannot be empty")
        object.__setattr__(self, "reactants", FrozenDict(reactants))
        object.__setattr__(self, "products", FrozenDict(products))
        object.__setattr__(self, "rate_orders", FrozenDict(rate_orders))
        object.__setattr__(self, "net_stoich", FrozenDict(net_stoich))
        object.__setattr__(self, "overrides", FrozenDict(self.overrides))
        object.__setattr__(self, "order", derive_molecularity(rate_orders))

    def net_stoich_vector(self, species_order: Iterable[str]) -> List[float]:
        """Vectorize net stoichiometry against a given species order."""
        s = self.net_stoich
        return [float(s.get(n, 0.0)) for n in species_order]


@dataclass(frozen=True)
class Equilibrium:
    """
    Reversible equilibrium description.

    A forward rate kf is required. Exactly one reverse-side authority is selected:
    explicit kr, or thermodynamic Keq/dG_eq metadata. Backend consumers may carry
    derived values for display, but provenance must not become a second authority.
    Equilibria stay reversible single steps; the ODE builder consumes them as
    one column with forward/reverse power-law terms rather than duplicating
    into two reactions.

    fast=True marks an entry that originated from `equilibrium:` in the DSL.
    """
    stoich_forward: Mapping[str, float]
    stoich_back: Mapping[str, float]
    Keq: Union[float, Expr, None] = None
    kf: Union[float, Expr, None] = None
    kr: Union[float, Expr, None] = None
    fast: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stoich_forward = _validate_positive_side(self.stoich_forward, label="stoich_forward")
        stoich_back = _validate_positive_side(self.stoich_back, label="stoich_back")
        if not stoich_forward:
            raise ValueError("stoich_forward cannot be empty")
        if not stoich_back:
            raise ValueError("stoich_back cannot be empty")
        object.__setattr__(
            self,
            "stoich_forward",
            FrozenDict(stoich_forward),
        )
        object.__setattr__(
            self,
            "stoich_back",
            FrozenDict(stoich_back),
        )
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))

    def forward_vector(self, species_order: Iterable[str]) -> List[float]:
        return [float(self.stoich_forward.get(n, 0.0)) for n in species_order]

    def back_vector(self, species_order: Iterable[str]) -> List[float]:
        return [float(self.stoich_back.get(n, 0.0)) for n in species_order]


# --------------------------------- mechanism --------------------------------


class Mechanism:
    """
    Mechanism container with deterministic ordering and metadata.

    Attributes
    ----------
    species : OrderedDict[str, Species]
    reactions : list[Reaction]
    equilibria : list[Equilibrium]
    metadata : dict   (includes 'declaration_order')
    """

    def __init__(self) -> None:
        self._species: "OrderedDict[str, Species]" = _SpeciesMap()
        self.reactions: List[Reaction] = []
        self.equilibria: List[Equilibrium] = []
        self.metadata: Dict[str, Any] = {"declaration_order": []}

    @property
    def species(self) -> "OrderedDict[str, Species]":
        return self._species

    @species.setter
    def species(self, value: Mapping[str, Species] | Iterable[tuple[str, Species]] | None) -> None:
        self._species = _SpeciesMap(value)
        if hasattr(self, "metadata"):
            self._sync_metadata_order()

    def clone(self) -> "Mechanism":
        """
        Fast, structured clone for process-isolated mutation.

        Copies only Mechanism-owned mutable containers (and the dict payloads
        they directly own), while keeping potentially-heavy immutable objects
        (rate expressions, compiled callables, etc.) as shared references.
        """
        cloned = copy.copy(self)
        cloned.species = _SpeciesMap(self.species)
        cloned.reactions = [
            replace(
                r,
                reactants=dict(r.reactants),
                products=dict(r.products),
                rate=_clone_authority_slot(r.rate),
                rate_orders=dict(r.rate_orders),
                overrides=dict(r.overrides),
            )
            for r in self.reactions
        ]
        cloned.equilibria = [
            replace(
                e,
                stoich_forward=dict(e.stoich_forward),
                stoich_back=dict(e.stoich_back),
                kf=_clone_authority_slot(e.kf),
                kr=_clone_authority_slot(e.kr),
                Keq=_clone_authority_slot(e.Keq),
                metadata=copy.deepcopy(dict(e.metadata)),
            )
            for e in self.equilibria
        ]
        cloned.metadata = copy.deepcopy(self.metadata)
        cloned._sync_metadata_order()
        return cloned

    # ---------- species management ----------

    def add_species(self, name: str, initial_conc: float = 0.0) -> Species:
        n = validate_name(name)
        if n in self.species:
            raise ValueError(f"species {n!r} already exists")
        sp = Species(n, initial_conc)
        self.species[n] = sp
        self._sync_metadata_order()
        return sp

    def set_initial(self, name: str, initial_conc: float) -> None:
        n = validate_name(name)
        if n not in self.species:
            raise KeyError(f"unknown species {n!r}")
        self.species[n] = replace(self.species[n], initial_conc=initial_conc)

    def rename_species(self, old: str, new: str) -> None:
        o = validate_name(old)
        nn = validate_name(new)
        if o not in self.species:
            raise KeyError(f"unknown species {o!r}")
        if nn in self.species and nn != o:
            raise ValueError(f"species {nn!r} already exists")
        if nn == o:
            return
        if any(
            o in reaction.reactants
            or o in reaction.products
            or o in reaction.rate_orders
            for reaction in self.reactions
        ):
            raise ValueError(
                f"cannot rename species {o!r}: referenced by a reaction"
            )
        if any(
            o in equilibrium.stoich_forward
            or o in equilibrium.stoich_back
            for equilibrium in self.equilibria
        ):
            raise ValueError(
                f"cannot rename species {o!r}: referenced by an equilibrium"
            )
        if _algebra_references_species(self, o):
            raise ValueError(
                f"cannot rename species {o!r}: referenced by Algebra"
            )
        new_map: "OrderedDict[str, Species]" = _SpeciesMap()
        for k, sp in self.species.items():
            if k == o:
                new_map[nn] = replace(sp, name=nn)
            else:
                new_map[k] = sp
        self.species = new_map
        self._sync_metadata_order()

    def remove_species(self, name: str) -> None:
        n = validate_name(name)
        if n not in self.species:
            raise KeyError(f"unknown species {n!r}")
        # Removing a species invalidates any steps that reference it
        # Do the conservative thing and block removal if referenced.
        if any(
            n in r.reactants
            or n in r.products
            or n in r.rate_orders
            or n in r.net_stoich
            for r in self.reactions
        ):
            raise ValueError(f"cannot remove species {n!r}: referenced by a reaction")
        if any(n in e.stoich_forward or n in e.stoich_back for e in self.equilibria):
            raise ValueError(f"cannot remove species {n!r}: referenced by an equilibrium")
        del self.species[n]
        self._sync_metadata_order()

    def species_names(self) -> List[str]:
        return list(self.species.keys())

    def _sync_metadata_order(self) -> None:
        self.metadata["declaration_order"] = self.species_names()

    def _append_step_index_map_entry(self, entry: Mapping[str, Any]) -> None:
        raw_mapping = self.metadata.get("step_index_map")
        step_index_map = raw_mapping if isinstance(raw_mapping, list) else []
        step_index = len(step_index_map) + 1
        mapped_entry = {"step_index": int(step_index), **dict(entry)}
        step_index_map.append(mapped_entry)
        self.metadata["step_index_map"] = step_index_map

    # ---------- step additions ----------

    def add_reaction(
        self,
        *,
        reactants: Mapping[str, float],
        products: Mapping[str, float],
        rate: Any,
        rate_orders: Optional[Mapping[str, float]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        record_step_index: bool = True,
    ) -> Reaction:
        """
        Add a reaction step.

        Validates physical sides and ensures all species are declared.
        """
        r = _validate_positive_side(reactants, label="reactants")
        p = _validate_positive_side(products, label="products")
        ro = (
            dict(r)
            if rate_orders is None
            else _validate_positive_side(rate_orders, label="rate_orders")
        )
        net = _derive_net_stoich(r, p)
        if not net:
            raise ValueError("reaction net stoichiometry cannot be empty")
        _ensure_species_exist({**r, **p, **ro}, self.species)
        ov = _normalize_overrides(overrides)
        rxn = Reaction(reactants=r, products=p, rate=rate, rate_orders=ro, overrides=ov)
        self.reactions.append(rxn)
        if bool(record_step_index):
            self._append_step_index_map_entry(
                {
                    "kind": "reaction",
                    "reaction_index": len(self.reactions) - 1,
                }
            )
        return rxn

    def add_equilibrium(
        self,
        stoich_forward: Mapping[str, float],
        stoich_back: Mapping[str, float],
        *,
        Keq: Any | None = None,
        kf: Any | None = None,
        kr: Any | None = None,
        fast: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        record_step_index: bool = True,
    ) -> Equilibrium:
        """
        Add an equilibrium pair.

        kf is required, plus exactly one reverse-side authority: kr or thermodynamic
        Keq/dG_eq metadata. Equilibrium authority and numeric sanity are validated
        by the equilibrium-rate authority layer.
        """
        return self._add_equilibrium_with_authority_context(
            stoich_forward,
            stoich_back,
            Keq=Keq,
            kf=kf,
            kr=kr,
            fast=fast,
            metadata=metadata,
            record_step_index=record_step_index,
            authority_context=EquilibriumRateInputContext.PUBLIC,
        )

    def _add_equilibrium_with_authority_context(
        self,
        stoich_forward: Mapping[str, float],
        stoich_back: Mapping[str, float],
        *,
        Keq: Any | None = None,
        kf: Any | None = None,
        kr: Any | None = None,
        fast: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        record_step_index: bool = True,
        authority_context: EquilibriumRateInputContext | str | None = None,
    ) -> Equilibrium:
        sf = _validate_positive_side(stoich_forward, label="stoich_forward")
        sb = _validate_positive_side(stoich_back, label="stoich_back")
        if not sf:
            raise ValueError("stoich_forward cannot be empty")
        if not sb:
            raise ValueError("stoich_back cannot be empty")
        _ensure_species_exist(sf, self.species)
        _ensure_species_exist(sb, self.species)

        meta = dict(metadata) if metadata else {}
        standard_conc = meta.get(
            EquilibriumMetadataKeys.STANDARD_CONC_M,
            self.metadata.get(MechanismMetadataKeys.STANDARD_CONC_M, 1.0),
        )
        if type(standard_conc) is bool:
            raise TypeError(
                "equilibrium standard concentration must be a positive finite real"
            )
        try:
            standard_conc_value = float(standard_conc)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "equilibrium standard concentration must be a positive finite real"
            ) from exc
        if not math.isfinite(standard_conc_value) or standard_conc_value <= 0.0:
            raise ValueError(
                "equilibrium standard concentration must be a positive finite real"
            )
        # Resolve authored reverse-side authority before deriving any standard-
        # state factor.  Explicit kf+kr owns both concentration-form rates and
        # is therefore C0-inert; thermodynamic authority alone consumes the
        # directional C0 exponent.
        meta.pop("std_ratio", None)
        meta.setdefault(EquilibriumMetadataKeys.USER_PROVIDED_KF, bool(kf is not None))
        meta.setdefault(EquilibriumMetadataKeys.USER_PROVIDED_KR, bool(kr is not None))
        authority = normalize_equilibrium_rate_authority(
            kf=kf,
            kr=kr,
            Keq=Keq,
            metadata=meta,
            context=authority_context,
        )
        if authority.kind == EquilibriumRateAuthorityKind.KEQ:
            try:
                directional_delta = math.fsum(sb.values()) - math.fsum(sf.values())
                standard_ratio = math.pow(standard_conc_value, directional_delta)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "equilibrium directional standard-state ratio must be finite"
                ) from exc
            if not math.isfinite(standard_ratio) or standard_ratio <= 0.0:
                raise ValueError(
                    "equilibrium directional standard-state ratio must be positive and finite"
                )
            meta["std_ratio"] = standard_ratio
            authority = normalize_equilibrium_rate_authority(
                kf=kf,
                kr=kr,
                Keq=Keq,
                metadata=meta,
                context=authority_context,
            )
        if authority.kind == EquilibriumRateAuthorityKind.KEQ:
            effective_keq = Keq
            if effective_keq is None and meta.get(EquilibriumMetadataKeys.KEQ_INPUT) is not None:
                effective_keq = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
            if effective_keq is None and meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) is not None:
                temperature = plain_finite_float_or_none(self.metadata.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15))
                dg_value = plain_finite_float_or_none(meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL))
                if temperature is not None and dg_value is not None:
                    effective_keq = float(K_from_deltaG_eq(float(dg_value), float(temperature)))
            scalar_keq = plain_finite_float_or_none(effective_keq)
            if Keq is None and scalar_keq is not None:
                Keq = scalar_keq
            scalar_kf = plain_finite_float_or_none(kf)
            scalar_std_ratio = plain_finite_float_or_none(authority.reverse_std_ratio)
            if (
                kr is None
                and scalar_kf is not None
                and scalar_keq is not None
                and scalar_std_ratio is not None
                and float(scalar_keq) > 0.0
            ):
                kr = effective_reverse_rate_from_keq(scalar_kf, scalar_keq, scalar_std_ratio)

        eq = Equilibrium(
            stoich_forward=sf,
            stoich_back=sb,
            Keq=Keq,
            kf=kf,
            kr=kr,
            fast=bool(fast),
            metadata=meta,
        )
        self.equilibria.append(eq)
        if bool(record_step_index):
            entry: Dict[str, object] = {
                "kind": "equilibrium",
                "equilibrium_index": len(self.equilibria) - 1,
            }
            entry.update(authority.step_map_fields())
            self._append_step_index_map_entry(entry)
        return eq

    # ---------- vectorization helpers ----------

    def reaction_stoich_matrix(self) -> List[List[float]]:
        """
        Matrix with shape (n_species, n_reactions) in declaration order.

        Each column corresponds to a reaction's stoichiometric vector.
        """
        names = self.species_names()
        cols: List[List[float]] = [r.net_stoich_vector(names) for r in self.reactions]
        # transpose to species-major rows
        if not cols:
            return [[] for _ in names]
        rows = list(map(list, zip(*cols)))
        return rows

    def equilibrium_pair_vectors(self) -> List[Tuple[List[float], List[float]]]:
        """
        List of (forward_vector, back_vector), each aligned to declaration order.
        """
        names = self.species_names()
        return [(e.forward_vector(names), e.back_vector(names)) for e in self.equilibria]

    # ---------- serialization for provenance ----------

    def to_serializable(self) -> Dict[str, Any]:
        """
        Deterministic serialization of the mechanism structure.

        Equilibrium entries publish authored owner/basis coordinates only. Derived
        reverse-side readouts are not emitted as peer authored fields.
        """
        declaration_order = self.metadata["declaration_order"]
        order_map = {name: idx for idx, name in enumerate(declaration_order)}

        species_block = OrderedDict((n, {"initial_conc": sp.initial_conc}) for n, sp in self.species.items())
        reactions_block = [
            {
                "reactants": OrderedDict(sorted(r.reactants.items(), key=lambda kv: order_map[kv[0]])),
                "products": OrderedDict(sorted(r.products.items(), key=lambda kv: order_map[kv[0]])),
                "rate_orders": OrderedDict(sorted(r.rate_orders.items(), key=lambda kv: order_map[kv[0]])),
                "net_stoich": OrderedDict(sorted(r.net_stoich.items(), key=lambda kv: order_map[kv[0]])),
                "order": r.order,
                "overrides": OrderedDict(
                    (key, json_safe_equilibrium_authority_payload(value))
                    for key, value in sorted(r.overrides.items())
                ),
                "rate": json_safe_equilibrium_authority_payload(r.rate),
            }
            for r in self.reactions
        ]
        temperature_K = coerce_temperature_kelvin(
            self.metadata.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15),
            field=MechanismMetadataKeys.TEMPERATURE_K,
        )

        equilibria_block = []
        for e in self.equilibria:
            metadata = json_safe_equilibrium_metadata_payload(e)
            entry = {
                "stoich_forward": OrderedDict(sorted(e.stoich_forward.items(), key=lambda kv: order_map[kv[0]])),
                "stoich_back":   OrderedDict(sorted(e.stoich_back.items(),   key=lambda kv: order_map[kv[0]])),
                "fast": e.fast,
                "metadata": OrderedDict(sorted(metadata.items())),
            }
            effective_kf = effective_equilibrium_forward_rate(e, temperature_K=temperature_K)
            effective_keq = effective_equilibrium_keq(e, temperature_K=temperature_K)
            effective_kr = effective_equilibrium_reverse_rate(e, temperature_K=temperature_K)
            authority = normalize_existing_equilibrium_rate_authority(e)
            kf_value = effective_kf if effective_kf is not None else e.kf
            entry["kf"] = json_safe_equilibrium_authority_payload(kf_value)
            if authority.kind == EquilibriumRateAuthorityKind.KR:
                kr_value = effective_kr if effective_kr is not None else authority.kr
                entry["kr"] = json_safe_equilibrium_authority_payload(kr_value)
            elif authority.has_explicit_keq_param:
                keq_value = effective_keq if effective_keq is not None else authority.Keq
                entry["Keq"] = json_safe_equilibrium_authority_payload(keq_value)
            equilibria_block.append(entry)
        return {
            "species": species_block,
            "reactions": reactions_block,
            "equilibria": equilibria_block,
            "metadata": {"declaration_order": declaration_order[:]},
        }
