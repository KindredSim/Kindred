"""
Species model and ordered registry.

Current contract
----------------
- Species has fields: `name: str`, `initial_conc: float` (standard state basis).
- Mechanism keeps `species: OrderedDict[str, Species]` and preserves declaration order.
- No networking, no registry, no filesystem access.

Design notes
------------
- The registry enforces uniqueness and stable declaration order.
- Renames preserve order.
- Initial concentrations are real-valued floats (coerced); negative values are allowed
  by the core model (chemistry may constrain later), but NaN/inf are rejected.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .validation import validate_name

__all__ = [
    "Species",
    "SpeciesRegistry",
    "coerce_float",
]


@dataclass(frozen=True)
class Species:
    """Immutable species record."""
    name: str
    initial_conc: float


# --- helpers -----------------------------------------------------------------

def coerce_float(value) -> float:
    """
    Coerce a number-like to float with strict finite check.

    Raises
    ------
    TypeError, ValueError
        If the value is not convertible to a finite float.
    """
    try:
         f = float(value)
    except (ValueError, TypeError) as e:
         raise TypeError(f"initial concentration must be a real number, got {value!r}") from e
    if not (f == f):  # NaN check without importing math
        raise ValueError("initial concentration cannot be NaN")
    if f in (float("inf"), float("-inf")):
        raise ValueError("initial concentration must be finite")
    return f


# --- registry ----------------------------------------------------------------

class SpeciesRegistry:
    """
    Ordered registry for Species with deterministic behavior.

    Preserves insertion order, supports rename while keeping order,
    and exposes convenience accessors for mechanism assembly.
    """

    def __init__(self, items: Optional[Iterable[Tuple[str, float]]] = None) -> None:
        self._data: "OrderedDict[str, Species]" = OrderedDict()
        if items:
            for name, conc in items:
                self.add(name, conc)

    # --- basic protocol ------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return validate_name(name) in self._data

    def __len__(self) -> int:  # pragma: no cover (trivial)
        return len(self._data)

    def __iter__(self) -> Iterator[Species]:  # yields Species in order
        return iter(self._data.values())

    def names(self) -> List[str]:
        """Declaration-ordered list of species names."""
        return list(self._data.keys())

    def items(self) -> Iterable[Tuple[str, Species]]:
        """Declaration-ordered (name, Species) pairs."""
        return self._data.items()

    # --- CRUD ----------------------------------------------------------------

    def add(self, name: str, initial_conc: float = 0.0) -> Species:
        """
        Add a species. Raises if the name already exists.

        Returns the created Species.
        """
        n = validate_name(name)
        if n in self._data:
            raise ValueError(f"species {n!r} already exists")
        sp = Species(n, coerce_float(initial_conc))
        self._data[n] = sp
        return sp

    def remove(self, name: str) -> None:
        """Remove a species by name. Raises KeyError if missing."""
        n = validate_name(name)
        try:
            del self._data[n]
        except KeyError:
            raise KeyError(f"unknown species {n!r}") from None

    def rename(self, old: str, new: str) -> None:
        """
        Rename a species while preserving declaration order.

        Raises if old is missing or new already exists.
        """
        o = validate_name(old)
        nn = validate_name(new)
        if o not in self._data:
            raise KeyError(f"unknown species {o!r}")
        if nn in self._data and nn != o:
            raise ValueError(f"species {nn!r} already exists")
        if nn == o:
            return  # no-op

        # Preserve order: rebuild around the renamed key
        new_map: "OrderedDict[str, Species]" = OrderedDict()
        for k, sp in self._data.items():
            if k == o:
                new_map[nn] = replace(sp, name=nn)
            else:
                new_map[k] = sp
        self._data = new_map

    def set_initial(self, name: str, initial_conc: float) -> None:
        """Update initial concentration for an existing species."""
        n = validate_name(name)
        if n not in self._data:
            raise KeyError(f"unknown species {n!r}")
        sp = self._data[n]
        self._data[n] = replace(sp, initial_conc=coerce_float(initial_conc))

    # --- accessors -----------------------------------------------------------

    def get(self, name: str) -> Species:
        """Return Species by name, or raise KeyError if missing."""
        n = validate_name(name)
        try:
            return self._data[n]
        except KeyError:
            raise KeyError(f"unknown species {n!r}") from None

    def initials_dict(self) -> Dict[str, float]:
        """Return an ordered dict-like view of initial concentrations by name."""
        return OrderedDict((k, sp.initial_conc) for k, sp in self._data.items())

    def declaration_order(self) -> List[str]:
        """Alias of names(); explicit naming for provenance metadata."""
        return self.names()

    # --- serialization helpers ----------------------------------------------

    def to_serializable(self) -> Dict[str, Dict[str, float]]:
        """
        Serialize to a deterministic mapping:
        {name: {"initial_conc": value}, ...} in declaration order.
        """
        return OrderedDict((k, {"initial_conc": sp.initial_conc}) for k, sp in self._data.items())

    @classmethod
    def from_serializable(cls, obj: Dict[str, Dict[str, float]]) -> "SpeciesRegistry":
        """
        Construct from `to_serializable` shape. Unknown fields are ignored.
        """
        items: List[Tuple[str, float]] = []
        for name, payload in obj.items():
            if not isinstance(payload, dict):
                continue
            conc = payload.get("initial_conc", 0.0)
            items.append((name, conc))
        return cls(items)
