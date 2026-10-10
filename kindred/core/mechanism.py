"""Admitted species, immutable kinetic models, and reversible authority.

Reaction and equilibrium forward sides retain one typed physical model.
Thermodynamic reverse coordinates remain authored metadata; numerical rates
are derived only by scientific consumers at their selected temperature.
"""

from __future__ import annotations

from fractions import Fraction
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .kinetics import ForwardRateModel
from .species import Species
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


def _derive_net_stoich(reactants: Mapping[str, float], products: Mapping[str, float]) -> Dict[str, float]:
    net: Dict[str, Fraction] = {}
    for sp, coef in reactants.items():
        net[sp] = net.get(sp, Fraction(0)) - Fraction(str(coef))
    for sp, coef in products.items():
        net[sp] = net.get(sp, Fraction(0)) + Fraction(str(coef))
    return {sp: float(coef) for sp, coef in net.items() if coef != 0}


# --------------------------------- models -----------------------------------


# Species names and positive finite coefficients arrive admitted by DSL parsing.
@dataclass(frozen=True, kw_only=True)
class Reaction:
    """Immutable physical sides, kinetic orders, and one forward model."""
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    forward_model: ForwardRateModel
    rate_orders: Mapping[str, float] = field(init=False)
    net_stoich: Mapping[str, float] = field(init=False)

    def __post_init__(self) -> None:
        # dataclass with frozen=True; use object.__setattr__
        # DSL parsing refuses a step whose sides are equal.
        net_stoich = _derive_net_stoich(self.reactants, self.products)
        object.__setattr__(self, "reactants", FrozenDict(self.reactants))
        object.__setattr__(self, "products", FrozenDict(self.products))
        # Kinetic orders are the reactant coefficients (mass action).
        object.__setattr__(self, "rate_orders", self.reactants)
        object.__setattr__(self, "net_stoich", FrozenDict(net_stoich))

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
    net_stoich: Mapping[str, float] = field(init=False, repr=False, compare=False)
    authority: EquilibriumRateAuthority = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.stoich_forward:
            raise ValueError("stoich_forward cannot be empty")
        if not self.stoich_back:
            raise ValueError("stoich_back cannot be empty")
        object.__setattr__(
            self,
            "stoich_forward",
            FrozenDict(self.stoich_forward),
        )
        object.__setattr__(
            self,
            "stoich_back",
            FrozenDict(self.stoich_back),
        )
        object.__setattr__(
            self,
            "net_stoich",
            FrozenDict(_derive_net_stoich(self.stoich_forward, self.stoich_back)),
        )
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))
        object.__setattr__(
            self,
            "authority",
            normalize_equilibrium_rate_authority(
                forward_model=self.forward_model, kr=self.kr, metadata=self.metadata,
            ),
        )

    def net_stoich_vector(self, species_order: Iterable[str]) -> List[float]:
        """Vectorize net back-minus-forward stoichiometry against a species order."""
        s = self.net_stoich
        return [float(s.get(n, 0.0)) for n in species_order]


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
        self._species: "OrderedDict[str, Species]" = OrderedDict()
        self.reactions: List[Reaction] = []
        self.equilibria: List[Equilibrium] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def species(self) -> "OrderedDict[str, Species]":
        return self._species

    @species.setter
    def species(self, value: Mapping[str, Species] | Iterable[tuple[str, Species]] | None) -> None:
        self._species = OrderedDict(value)

    # ---------- species management ----------

    def add_species(self, name: str) -> Species:
        """Add one parser-admitted species name, declared once by its builder."""
        sp = Species(name)
        self.species[name] = sp
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
    ) -> Reaction:
        """Add a reaction step; its builder declared every species it names."""
        rxn = Reaction(reactants=reactants, products=products, forward_model=forward_model)
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
        meta = dict(metadata or {})
        meta.setdefault(EquilibriumMetadataKeys.STANDARD_CONC_M,
                        self.metadata.get(MechanismMetadataKeys.STANDARD_CONC_M, 1.0))
        eq = Equilibrium(stoich_forward=stoich_forward, stoich_back=stoich_back,
                         forward_model=forward_model, kr=kr, metadata=meta)
        self.equilibria.append(eq)
        return eq
