from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Mapping, Sequence

from kindred.core.validation import try_parse_int
from .dsl_build_input import (
    canonical_source_coordinate_name,
    source_coordinate_roles,
)

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
class InvalidProtectedIndexedIdentifier:
    raw_name: str
    step_index: int
    suggested_names: tuple[str, ...]


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_names", frozenset(self.canonical_names))
        object.__setattr__(self, "ordered_items", tuple(self.ordered_items))
        object.__setattr__(self, "source_coordinates", tuple(self.source_coordinates))

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

    def invalid_protected_indexed_identifier(self, raw_name: str) -> InvalidProtectedIndexedIdentifier | None:
        if self.resolve(str(raw_name)).canonical_name is not None:
            return None
        match = _protected_indexed_identifier_match(raw_name)
        if match is None:
            return None
        return self._invalid_indexed_identifier_match(raw_name, match.group(1))

    def invalid_protected_indexed_identifier_message(self, raw_name: str) -> str | None:
        invalid = self.invalid_protected_indexed_identifier(raw_name)
        if invalid is None:
            return None
        suggestions = ", ".join(invalid.suggested_names)
        suggestion_clause = (
            f" Existing canonical indexed parameter name(s) for this step: {suggestions}."
            if suggestions
            else " Exact protected indexed names must match an existing canonical mechanism parameter."
        )
        return (
            f"{invalid.raw_name!r} is not a valid indexed parameter identifier."
            f"{suggestion_clause} Choose a longer ordinary name such as 'K1_test' for an independent parameter."
        )

    def _invalid_indexed_identifier_match(
        self,
        raw_name: str,
        step_index_text: str,
    ) -> InvalidProtectedIndexedIdentifier | None:
        step_index, ok = try_parse_int(step_index_text)
        if not ok:
            return None
        suggestions = tuple(
            name
            for lookup in (f"k{step_index}", f"kf{step_index}", f"kr{step_index}", f"keq{step_index}")
            for name in [
                next(
                    (
                        canonical
                        for canonical in self.canonical_names
                        if canonical.casefold() == lookup.casefold()
                    ),
                    None,
                )
            ]
            if name is not None
        )
        return InvalidProtectedIndexedIdentifier(
            raw_name=str(raw_name),
            step_index=int(step_index),
            suggested_names=suggestions,
        )


@dataclass(frozen=True)
class _NamespaceStepDescriptor:
    step_index: int
    step_kind: str
    source_index: int
    source_coordinates: tuple[tuple[str, str], ...] = ()


def is_protected_indexed_identifier(raw_name: str) -> bool:
    return _protected_indexed_identifier_match(raw_name) is not None


def protected_indexed_identifier_step_index(raw_name: str) -> int | None:
    match = _protected_indexed_identifier_match(raw_name)
    if match is None:
        return None
    step_index, ok = try_parse_int(match.group(1))
    if not ok:
        return None
    return int(step_index)


def _protected_indexed_identifier_match(raw_name: str) -> re.Match[str] | None:
    """Recognize protected indexed spellings for diagnostics only.

    Resolution itself remains exact-case.  Folding here only lets diagnostics
    identify a misspelled protected name and offer its exact canonical form.
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

        if step_kind != "equilibrium":
            raise ValueError(f"Unsupported step kind in parameter namespace: {step_kind!r}")

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
    ordered_items = tuple(_iter_canonical_items(descriptors))
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    for item in ordered_items:
        if item.canonical_name in seen_names:
            raise ValueError(f"Duplicate mechanism parameter name {item.canonical_name!r} in authoritative namespace.")
        seen_names.add(item.canonical_name)
        ordered_names.append(item.canonical_name)
    source_names = {
        str(canonical_name)
        for descriptor in descriptors
        for _role, canonical_name in descriptor.source_coordinates
    }
    canonical_names = frozenset((*ordered_names, *source_names))
    source_coordinates = tuple(
        MechanismSourceCoordinateInfo(
            canonical_name=str(canonical_name),
            step_index=int(descriptor.step_index),
            step_kind=str(descriptor.step_kind),
            role=str(role),
            source_index=descriptor.source_index,
        )
        for descriptor in descriptors
        for role, canonical_name in descriptor.source_coordinates
    )
    source_names = [item.canonical_name for item in source_coordinates]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Duplicate reaction-local source coordinate identity.")
    return MechanismParameterNamespace(
        canonical_names=canonical_names,
        ordered_items=ordered_items,
        source_coordinates=source_coordinates,
    )


def _mechanism_step_descriptors(mechanism: object) -> list[_NamespaceStepDescriptor]:
    raw_mapping = mechanism.metadata["step_index_map"]
    descriptors: list[_NamespaceStepDescriptor] = []
    seen_step_indices: set[int] = set()
    for source_index, raw_entry in enumerate(raw_mapping):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Mechanism step_index_map entry {source_index} is not a dict.")
        step_index, ok = try_parse_int(raw_entry.get("step_index"))
        if not ok or step_index <= 0:
            raise ValueError(f"Mechanism step_index_map entry {source_index} has an invalid step_index.")
        if step_index in seen_step_indices:
            raise ValueError(
                f"Mechanism step_index_map entry {source_index} repeats step_index {step_index}; authoritative namespaces require unique step indices."
            )
        seen_step_indices.add(step_index)
        step_kind = str(raw_entry.get("kind") or "")
        if step_kind not in {"reaction", "equilibrium"}:
            raise ValueError(f"Mechanism step_index_map entry {source_index} has an invalid kind {step_kind!r}.")
        raw_source_coordinates = raw_entry.get("source_coordinates") or {}
        if not isinstance(raw_source_coordinates, Mapping):
            raise ValueError(
                f"Mechanism step_index_map entry {source_index} has invalid source coordinates."
            )
        source_coordinates: list[tuple[str, str]] = []
        for raw_role, raw_name in raw_source_coordinates.items():
            role = str(raw_role)
            name = str(raw_name)
            if role not in source_coordinate_roles():
                raise ValueError(
                    f"Mechanism step_index_map entry {source_index} has an invalid source coordinate role {role!r}."
                )
            expected_name = canonical_source_coordinate_name(step_index, role)
            if name != expected_name:
                raise ValueError(
                    f"Mechanism step_index_map entry {source_index} has a non-local source identity {name!r}; expected {expected_name!r}."
                )
            source_coordinates.append((role, name))
        descriptors.append(
            _NamespaceStepDescriptor(
                step_index=step_index,
                step_kind=step_kind,
                source_index=source_index,
                source_coordinates=tuple(source_coordinates),
            )
        )
    return descriptors


def build_namespace_from_mechanism(mechanism: object) -> MechanismParameterNamespace:
    return _build_namespace(_mechanism_step_descriptors(mechanism))


def canonical_name_for_mechanism_step_parameter(
    mechanism: object,
    *,
    kind: str,
    item_index: int,
    role: str,
    expected_name: str,
) -> str:
    raw_mapping = mechanism.metadata["step_index_map"]
    index_key = "reaction_index" if str(kind) == "reaction" else "equilibrium_index"
    step_index: int | None = None
    for raw_entry in raw_mapping:
        if not isinstance(raw_entry, dict):
            continue
        if str(raw_entry.get("kind") or "") != str(kind):
            continue
        parsed_item_index, item_ok = try_parse_int(raw_entry.get(index_key))
        parsed_step_index, step_ok = try_parse_int(raw_entry.get("step_index"))
        if not item_ok or not step_ok:
            continue
        if int(parsed_item_index) != int(item_index):
            continue
        step_index = int(parsed_step_index)
        break
    if step_index is None or step_index <= 0:
        raise ValueError(
            f"Cannot resolve authoritative mechanism parameter name {expected_name!r}: "
            f"no step_index_map entry matched {kind} item {item_index}."
        )

    role_s = str(role)
    candidate = f"{role_s}{step_index}"
    namespace = build_namespace_from_mechanism(mechanism)
    resolved = namespace.resolve(candidate)
    if resolved.canonical_name is not None:
        return str(resolved.canonical_name)
    raise ValueError(
        f"Cannot resolve authoritative mechanism parameter name {expected_name!r}: "
        f"{candidate!r} is not a canonical parameter in the mechanism namespace."
    )
