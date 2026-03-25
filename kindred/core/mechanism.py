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
  - initial_conc: float  (standard state basis)
- Reaction:
  - stoich: dict[str, float]  negative for reactants, positive for products
  - order: int  derived molecularity (sum of reactant stoich magnitudes)
  - rate: StructuredRate | Callable    (opaque holder here; evaluated upstream)
  - overrides: {"model": "Eyring" | "Arrhenius" | None,
                "kappa": float | None, "A": float | None, "Ea": float | None,
                "Ea_J_per_mol": float | None, "dG_act_J_per_mol": float | None,
                "standard_conc_M": float | None}
- Equilibrium:
  - stoich_forward: dict[str, float]
  - stoich_back: dict[str, float]
  - K: float | Expr | None
  - kf: float | Expr | None
  - kr: float | Expr | None
  - fast: bool  (true if originated from `equilibrium:`)

Design notes
------------
- This file does not evaluate rates or assemble ODEs; it only holds a validated,
  deterministic description of the mechanism. Solver and simulator layers map
  these structures to kinetics later.
- Molecularity/order for Reaction is derived from the sum of magnitudes of
  negative stoichiometric coefficients and rounded to nearest integer with a
  tight tolerance.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union, Protocol, Optional

from .species import Species, coerce_float
from .validation import validate_name

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


# --------------------------------- helpers ----------------------------------


def _validate_stoich(stoich: Mapping[str, float]) -> Dict[str, float]:
    """
    Validate a stoichiometry mapping and coerce values to floats.

    Returns a new plain dict with stripped names and float values.
    """
    if not isinstance(stoich, Mapping):
        raise TypeError("stoichiometry must be a mapping of name -> coefficient")
    out: Dict[str, float] = {}
    for k, v in stoich.items():
        n = validate_name(k)
        try:
            fv = float(v)
        except (ValueError, TypeError) as e:
            raise TypeError(f"stoichiometry for {n!r} must be numeric, got {v!r}") from e
        if fv == 0.0:
            # omit true zeros to keep the representation clean and deterministic
            continue
        out[n] = fv
    if not out:
        raise ValueError("stoichiometry cannot be empty (all-zero is not allowed)")
    return out


def _ensure_species_exist(stoich: Mapping[str, float], declared: Mapping[str, Species]) -> None:
    for n in stoich:
        if n not in declared:
            raise KeyError(f"unknown species in stoichiometry: {n!r}")


def _close_to_int(x: float, tol: float = 1e-9) -> int:
    r = round(x)
    if abs(x - r) <= tol:
        return int(r)
    raise ValueError(f"value {x} not within {tol} of an integer")


def derive_molecularity(stoich: Mapping[str, float]) -> int:
    """
    Derive reaction molecularity (order) as the sum of magnitudes
    of negative stoichiometric coefficients (reactants).

    Coefficients like 2A + B -> ... yield order 3.
    """
    total = sum(-v for v in stoich.values() if v < 0.0)
    if total < 0:
        # shouldn't happen, defensive
        total = 0.0
    return _close_to_int(total, tol=1e-9)


def _normalize_overrides(overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Keep only allowed override keys and normalize types conservatively.
    Unknown keys are dropped deterministically.
    """
    allowed = {
        "model",
        "kappa",
        "A",
        "Ea",
        "Ea_J_per_mol",
        "dG_act_J_per_mol",
        "standard_conc_M",
    }
    out: Dict[str, Any] = {}
    if not overrides:
        return out
    for k in allowed:
        if k in overrides:
            out[k] = overrides[k]
    # normalize model
    model = out.get("model", None)
    if model is not None:
        if model not in ("Eyring", "Arrhenius"):
            raise ValueError("overrides.model must be 'Eyring', 'Arrhenius', or None")
    # numeric sanity where applicable
    for num_key in ("kappa", "A", "Ea", "Ea_J_per_mol", "dG_act_J_per_mol", "standard_conc_M"):
        if num_key in out and out[num_key] is not None:
            try:
                out[num_key] = float(out[num_key])  # Ea and A accept float; kappa likewise
            except (ValueError, TypeError) as e:
                raise TypeError(f"overrides.{num_key} must be numeric if provided") from e
    return out


# --------------------------------- models -----------------------------------


@dataclass(frozen=True)
class Reaction:
    """
    Reaction step with general stoichiometry.

    Fields
    ------
    stoich : dict[str, float]
        Negative for reactants, positive for products, zeros omitted.
    rate : Any
        Either a StructuredRate object or a Callable resolved upstream.
        Stored opaquely here.
    overrides : dict
        Optional per-step overrides for model parameters.
    order : int
        Derived molecularity based on reactant stoichiometry.
    """
    stoich: Dict[str, float]
    rate: Union[StructuredRate, Callable[[], float]]
    overrides: Dict[str, Any] = field(default_factory=dict)
    order: int = field(init=False)

    def __post_init__(self) -> None:
        # dataclass with frozen=True; use object.__setattr__
        object.__setattr__(self, "order", derive_molecularity(self.stoich))

    def stoich_vector(self, species_order: Iterable[str]) -> List[float]:
        """Vectorize stoichiometry against a given species order."""
        s = self.stoich
        return [float(s.get(n, 0.0)) for n in species_order]


@dataclass(frozen=True)
class Equilibrium:
    """
    Reversible equilibrium description.

    Either K or kf/kr may be provided (or both). If only K is provided,
    kf/kr may be derived later based on a fast-equilibrium policy.
    Equilibria stay reversible single steps; the ODE builder consumes them as
    one column with forward/reverse power-law terms rather than duplicating
    into two reactions.

    fast=True marks an entry that originated from `equilibrium:` in the DSL.
    """
    stoich_forward: Dict[str, float]
    stoich_back: Dict[str, float]
    K: Union[float, Expr, None] = None
    kf: Union[float, Expr, None] = None
    kr: Union[float, Expr, None] = None
    fast: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        self.species: "OrderedDict[str, Species]" = OrderedDict()
        self.reactions: List[Reaction] = []
        self.equilibria: List[Equilibrium] = []
        self.metadata: Dict[str, Any] = {"declaration_order": []}

    def clone(self) -> "Mechanism":
        """
        Fast, structured clone for process-isolated mutation.

        Copies only Mechanism-owned mutable containers (and the dict payloads
        they directly own), while keeping potentially-heavy immutable objects
        (rate expressions, compiled callables, etc.) as shared references.
        """
        cloned = copy.copy(self)
        cloned.species = OrderedDict(self.species)
        cloned.reactions = [
            replace(r, stoich=dict(r.stoich), overrides=dict(r.overrides)) for r in self.reactions
        ]
        cloned.equilibria = [
            replace(
                e,
                stoich_forward=dict(e.stoich_forward),
                stoich_back=dict(e.stoich_back),
                metadata=dict(e.metadata),
            )
            for e in self.equilibria
        ]
        cloned.metadata = copy.deepcopy(self.metadata)
        return cloned

    # ---------- species management ----------

    def add_species(self, name: str, initial_conc: float = 0.0) -> Species:
        n = validate_name(name)
        if n in self.species:
            raise ValueError(f"species {n!r} already exists")
        sp = Species(n, coerce_float(initial_conc))
        self.species[n] = sp
        self._sync_metadata_order()
        return sp

    def set_initial(self, name: str, initial_conc: float) -> None:
        n = validate_name(name)
        if n not in self.species:
            raise KeyError(f"unknown species {n!r}")
        self.species[n] = replace(self.species[n], initial_conc=coerce_float(initial_conc))

    def rename_species(self, old: str, new: str) -> None:
        o = validate_name(old)
        nn = validate_name(new)
        if o not in self.species:
            raise KeyError(f"unknown species {o!r}")
        if nn in self.species and nn != o:
            raise ValueError(f"species {nn!r} already exists")
        if nn == o:
            return
        new_map: "OrderedDict[str, Species]" = OrderedDict()
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
        if any(n in r.stoich for r in self.reactions):
            raise ValueError(f"cannot remove species {n!r}: referenced by a reaction")
        if any(n in e.stoich_forward or n in e.stoich_back for e in self.equilibria):
            raise ValueError(f"cannot remove species {n!r}: referenced by an equilibrium")
        del self.species[n]
        self._sync_metadata_order()

    def species_names(self) -> List[str]:
        return list(self.species.keys())

    def _sync_metadata_order(self) -> None:
        self.metadata["declaration_order"] = self.species_names()

    # ---------- step additions ----------

    def add_reaction(
        self,
        stoich: Mapping[str, float],
        rate: Any,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Reaction:
        """
        Add a reaction step.

        Validates stoichiometry and ensures all species are declared.
        """
        s = _validate_stoich(stoich)
        _ensure_species_exist(s, self.species)
        ov = _normalize_overrides(overrides)
        rxn = Reaction(stoich=s, rate=rate, overrides=ov)
        self.reactions.append(rxn)
        return rxn

    def add_equilibrium(
        self,
        stoich_forward: Mapping[str, float],
        stoich_back: Mapping[str, float],
        *,
        K: Any | None = None,
        kf: Any | None = None,
        kr: Any | None = None,
        fast: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Equilibrium:
        """
        Add an equilibrium pair.

        At least one of {K, (kf and kr)} must be provided. Validation is structural;
        numeric sanity (positivity, units) is the responsibility of the simulator layer.
        """
        sf = _validate_stoich(stoich_forward)
        sb = _validate_stoich(stoich_back)
        _ensure_species_exist(sf, self.species)
        _ensure_species_exist(sb, self.species)

        if K is None and (kf is None or kr is None):
            raise ValueError("equilibrium requires K or both kf and kr")

        meta = dict(metadata) if metadata else {}

        eq = Equilibrium(
            stoich_forward=sf,
            stoich_back=sb,
            K=K,
            kf=kf,
            kr=kr,
            fast=bool(fast),
            metadata=meta,
        )
        self.equilibria.append(eq)
        return eq

    # ---------- vectorization helpers ----------

    def reaction_stoich_matrix(self) -> List[List[float]]:
        """
        Matrix with shape (n_species, n_reactions) in declaration order.

        Each column corresponds to a reaction's stoichiometric vector.
        """
        names = self.species_names()
        cols: List[List[float]] = [r.stoich_vector(names) for r in self.reactions]
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

        Note: `rate`, `K`, `kf`, `kr` are stored as-is and may be non-JSON types.
        Callers are responsible for serializing expressions if needed.
        """
        declaration_order = self.metadata["declaration_order"]
        order_map = {name: idx for idx, name in enumerate(declaration_order)}

        species_block = OrderedDict((n, {"initial_conc": sp.initial_conc}) for n, sp in self.species.items())
        reactions_block = [
            {
                "stoich": OrderedDict(sorted(r.stoich.items(), key=lambda kv: order_map[kv[0]])),
                "order": r.order,
                "overrides": r.overrides,
                "rate": r.rate,
            }
            for r in self.reactions
        ]
        equilibria_block = [
            {
                "stoich_forward": OrderedDict(sorted(e.stoich_forward.items(), key=lambda kv: order_map[kv[0]])),
                "stoich_back":   OrderedDict(sorted(e.stoich_back.items(),   key=lambda kv: order_map[kv[0]])),
                "K": e.K, "kf": e.kf, "kr": e.kr, "fast": e.fast,
            }
            for e in self.equilibria
        ]
        return {
            "species": species_block,
            "reactions": reactions_block,
            "equilibria": equilibria_block,
            "metadata": {"declaration_order": declaration_order[:]},
        }
