from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Sequence

from kindred.core.validation import and_join


_PROTECTED_INDEXED_IDENTIFIER_RE = re.compile(r"^(?:k|kf|kr|keq)([1-9]\d*)$")


@dataclass(frozen=True)
class MechanismParameterInfo:
    canonical_name: str
    step_index: int
    step_kind: str
    role: str


@dataclass(frozen=True)
class MechanismSourceCoordinateInfo:
    """One authored physical coordinate with reaction-local identity."""

    canonical_name: str
    step_index: int
    step_kind: str
    role: str
    source_index: int


@dataclass(frozen=True)
class MechanismParameterResolution:
    raw_name: str
    canonical_name: str | None = None


@dataclass(frozen=True)
class MechanismParameterNamespaceItem:
    canonical_name: str
    info: MechanismParameterInfo
    source_index: int


@dataclass(frozen=True)
class MechanismParameterNamespace:
    canonical_names: frozenset[str]
    ordered_items: tuple[MechanismParameterNamespaceItem, ...]
    source_coordinates: tuple[MechanismSourceCoordinateInfo, ...] = ()

    @property
    def source_coordinate_names(self) -> frozenset[str]:
        return frozenset(item.canonical_name for item in self.source_coordinates)

    def source_coordinate_info(
        self,
        name: str,
    ) -> MechanismSourceCoordinateInfo | None:
        requested = str(name)
        return next(
            (
                item
                for item in self.source_coordinates
                if item.canonical_name == requested
            ),
            None,
        )

    def flat_names(self) -> set[str]:
        return set(self.canonical_names)

    def resolve(self, raw_name: str) -> MechanismParameterResolution:
        name = str(raw_name)
        if name in self.canonical_names:
            return MechanismParameterResolution(raw_name=name, canonical_name=name)

        return MechanismParameterResolution(raw_name=name)

    def invalid_protected_indexed_identifier_message(self, raw_name: str) -> str | None:
        """The sentence for a reserved k, kf, kr or Keq name that names no parameter here."""
        name = str(raw_name)
        step_index = protected_indexed_identifier_step_index(name)
        if step_index is None or self.resolve(name).canonical_name is not None:
            return None
        names = tuple(dict.fromkeys(
            item.canonical_name
            for item in (*(entry.info for entry in self.ordered_items), *self.source_coordinates)
            if item.step_index == step_index
        ))
        if not names:
            return f"{name} is reserved for reaction parameters, and this mechanism has no step {step_index}."
        return (f"{name} is reserved for reaction parameters; "
                f"the parameters of step {step_index} are {and_join(names)}.")


@dataclass(frozen=True)
class _NamespaceStepDescriptor:
    step_index: int
    step_kind: str
    source_index: int
    source_coordinates: tuple[tuple[str, str], ...] = ()


def protected_indexed_identifier_step_index(raw_name: str) -> int | None:
    match = _protected_indexed_identifier_match(raw_name)
    if match is None:
        return None
    return int(match.group(1))


def _protected_indexed_identifier_match(raw_name: str) -> re.Match[str] | None:
    """Recognize the reserved indexed reaction-parameter names, any case.

    Resolution itself remains exact-case.  Folding here reserves every case
    spelling, so a param or let line cannot declare K1 beside k1.
    """

    return _PROTECTED_INDEXED_IDENTIFIER_RE.fullmatch(
        str(raw_name or "").casefold()
    )


def _iter_canonical_items(
    descriptors: Sequence[_NamespaceStepDescriptor],
) -> Iterator[MechanismParameterNamespaceItem]:
    for descriptor in descriptors:
        step_index = int(descriptor.step_index)
        step_kind = str(descriptor.step_kind)
        if step_kind == "reaction":
            name = f"k{step_index}"
            yield MechanismParameterNamespaceItem(
                canonical_name=name,
                info=MechanismParameterInfo(
                    canonical_name=name,
                    step_index=step_index,
                    step_kind=step_kind,
                    role="k",
                ),
                source_index=descriptor.source_index,
            )
            continue

        for role in ("kf", "kr"):
            name = f"{role}{step_index}"
            yield MechanismParameterNamespaceItem(
                canonical_name=name,
                info=MechanismParameterInfo(
                    canonical_name=name,
                    step_index=step_index,
                    step_kind=step_kind,
                    role=role,
                ),
                source_index=descriptor.source_index,
            )
        name = f"Keq{step_index}"
        yield MechanismParameterNamespaceItem(
            canonical_name=name,
            info=MechanismParameterInfo(
                canonical_name=name,
                step_index=step_index,
                step_kind=step_kind,
                role="Keq",
            ),
            source_index=descriptor.source_index,
        )


def _build_namespace(descriptors: Sequence[_NamespaceStepDescriptor]) -> MechanismParameterNamespace:
    # Unique step indices give unique indexed names and reaction-local source names.
    ordered_items = tuple(_iter_canonical_items(descriptors))
    source_coordinates = tuple(
        MechanismSourceCoordinateInfo(
            canonical_name=canonical_name,
            step_index=descriptor.step_index,
            step_kind=descriptor.step_kind,
            role=role,
            source_index=descriptor.source_index,
        )
        for descriptor in descriptors
        for role, canonical_name in descriptor.source_coordinates
    )
    return MechanismParameterNamespace(
        canonical_names=frozenset(
            (*(item.canonical_name for item in ordered_items),
             *(item.canonical_name for item in source_coordinates))
        ),
        ordered_items=ordered_items,
        source_coordinates=source_coordinates,
    )


def _mechanism_step_descriptors(mechanism: object) -> list[_NamespaceStepDescriptor]:
    # The DSL builder writes one entry per authored step, numbered from 1, with
    # its kind and canonical reaction-local source identities.
    return [
        _NamespaceStepDescriptor(
            step_index=entry["step_index"],
            step_kind=entry["kind"],
            source_index=source_index,
            source_coordinates=tuple(entry["source_coordinates"].items()),
        )
        for source_index, entry in enumerate(mechanism.metadata["step_index_map"])
    ]


def build_namespace_from_mechanism(mechanism: object) -> MechanismParameterNamespace:
    return _build_namespace(_mechanism_step_descriptors(mechanism))


def canonical_name_for_mechanism_step_parameter(
    mechanism: object,
    *,
    kind: str,
    item_index: int,
    role: str,
) -> str:
    """Return the indexed name of one mechanism step's rate role."""
    index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
    step_index = next(
        entry["step_index"]
        for entry in mechanism.metadata["step_index_map"]
        if entry["kind"] == kind and entry[index_key] == item_index
    )
    return f"{role}{step_index}"
