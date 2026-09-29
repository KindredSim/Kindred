"""Admitted species, immutable kinetic models, and reversible authority.

Reaction and equilibrium forward sides retain one typed physical model.
Thermodynamic reverse coordinates remain authored metadata; numerical rates
are derived only by scientific consumers at their selected temperature.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .kinetics import ForwardRateModel
from .species import Species
from .stoichiometry import directional_order_fraction
from .validation import validate_name
from .equilibrium_rate_authority import (
    EquilibriumRateAuthority,
    normalize_equilibrium_rate_authority,
)
from .mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataKeys

__all__ = [
    "Reaction",
    "Equilibrium",
    "Mechanism",
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


def _ensure_species_exist(stoich: Mapping[str, float], declared: Mapping[str, Species]) -> None:
    for n in stoich:
        if n not in declared:
            raise KeyError(f"unknown species in stoichiometry: {n!r}")


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


# --------------------------------- models -----------------------------------


@dataclass(frozen=True, kw_only=True)
class Reaction:
    """Immutable physical sides, kinetic orders, and one forward model."""
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    forward_model: ForwardRateModel
    rate_orders: Mapping[str, float] | None = None
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
        object.__setattr__(
            self,
            "order",
            float(
                directional_order_fraction(
                    rate_orders,
                )
            ),
        )

    def net_stoich_vector(self, species_order: Iterable[str]) -> List[float]:
        """Vectorize net stoichiometry against a given species order."""
        s = self.net_stoich
        return [float(s.get(n, 0.0)) for n in species_order]


@dataclass(frozen=True)
class Equilibrium:
    """One reversible step with a typed forward model and authored reverse authority."""
    stoich_forward: Mapping[str, float]
    stoich_back: Mapping[str, float]
    forward_model: ForwardRateModel
    kr: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    authority: EquilibriumRateAuthority = field(init=False, repr=False, compare=False)

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
        object.__setattr__(
            self,
            "authority",
            normalize_equilibrium_rate_authority(
                forward_model=self.forward_model, kr=self.kr, metadata=self.metadata,
            ),
        )

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
    metadata : dict
    """

    def __init__(self) -> None:
        self._species: "OrderedDict[str, Species]" = _SpeciesMap()
        self.reactions: List[Reaction] = []
        self.equilibria: List[Equilibrium] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def species(self) -> "OrderedDict[str, Species]":
        return self._species

    @species.setter
    def species(self, value: Mapping[str, Species] | Iterable[tuple[str, Species]] | None) -> None:
        self._species = _SpeciesMap(value)

    # ---------- species management ----------

    def add_species(self, name: str) -> Species:
        n = validate_name(name)
        if n in self.species:
            raise ValueError(f"species {n!r} already exists")
        sp = Species(n)
        self.species[n] = sp
        return sp

    def species_names(self) -> List[str]:
        return list(self.species.keys())

    # ---------- step additions ----------

    def add_reaction(
        self,
        *,
        reactants: Mapping[str, float],
        products: Mapping[str, float],
        forward_model: ForwardRateModel,
        rate_orders: Optional[Mapping[str, float]] = None,
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
        rxn = Reaction(reactants=r, products=p, forward_model=forward_model, rate_orders=ro)
        self.reactions.append(rxn)
        return rxn

    def add_equilibrium(
        self,
        stoich_forward: Mapping[str, float],
        stoich_back: Mapping[str, float],
        *,
        forward_model: ForwardRateModel,
        kr: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Equilibrium:
        """Add one admitted forward model and exactly one reverse authority."""
        sf = _validate_positive_side(stoich_forward, label="stoich_forward")
        sb = _validate_positive_side(stoich_back, label="stoich_back")
        _ensure_species_exist({**sf, **sb}, self.species)
        meta = dict(metadata or {})
        meta.setdefault(EquilibriumMetadataKeys.STANDARD_CONC_M,
                        self.metadata.get(MechanismMetadataKeys.STANDARD_CONC_M, 1.0))
        eq = Equilibrium(stoich_forward=sf, stoich_back=sb,
                         forward_model=forward_model, kr=kr, metadata=meta)
        self.equilibria.append(eq)
        return eq
