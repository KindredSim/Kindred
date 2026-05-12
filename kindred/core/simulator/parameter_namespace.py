from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator, Mapping, Sequence

from kindred.core.validation import try_parse_int

_CANONICAL_NAME_RE = re.compile(r"^(k|kf|kr|Keq)([1-9]\d*)$")
_K_ALIAS_RE = re.compile(r"^k(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class MechanismParameterInfo:
    canonical_name: str
    step_index: int | None
    step_kind: str | None
    role: str | None
    has_explicit_keq: bool | None


@dataclass(frozen=True)
class MechanismParameterResolution:
    raw_name: str
    canonical_name: str | None = None
    equilibrium_conflict_name: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.canonical_name is not None


@dataclass(frozen=True)
class MechanismParameterNamespaceItem:
    canonical_name: str
    info: MechanismParameterInfo
    source_index: int | None = None


@dataclass(frozen=True)
class MechanismParameterNamespace:
    canonical_names: frozenset[str]
    canonical_by_lower: Mapping[str, str]
    info_by_name: Mapping[str, MechanismParameterInfo]
    ordered_items: tuple[MechanismParameterNamespaceItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_names", frozenset(self.canonical_names))
        object.__setattr__(self, "canonical_by_lower", dict(self.canonical_by_lower))
        object.__setattr__(self, "info_by_name", dict(self.info_by_name))
        object.__setattr__(self, "ordered_items", tuple(self.ordered_items))

    def flat_names(self) -> set[str]:
        return set(self.canonical_names)

    def resolve(self, raw_name: str) -> MechanismParameterResolution:
        direct_match = self.canonical_by_lower.get(str(raw_name).lower())
        if direct_match is not None:
            return MechanismParameterResolution(raw_name=str(raw_name), canonical_name=direct_match)

        alias_match = _K_ALIAS_RE.match(str(raw_name))
        if alias_match is None:
            return MechanismParameterResolution(raw_name=str(raw_name))

        index = alias_match.group(1)
        equilibrium_name = self.canonical_by_lower.get(f"keq{index}")
        if equilibrium_name is not None:
            return MechanismParameterResolution(
                raw_name=str(raw_name),
                equilibrium_conflict_name=equilibrium_name,
            )

        reversible_forward_name = self.canonical_by_lower.get(f"kf{index}")
        if reversible_forward_name is not None:
            return MechanismParameterResolution(raw_name=str(raw_name), canonical_name=reversible_forward_name)

        irreversible_name = self.canonical_by_lower.get(f"k{index}")
        if irreversible_name is not None:
            return MechanismParameterResolution(raw_name=str(raw_name), canonical_name=irreversible_name)

        return MechanismParameterResolution(raw_name=str(raw_name))


@dataclass(frozen=True)
class _NamespaceStepDescriptor:
    step_index: int
    step_kind: str
    has_explicit_keq: bool
    source_index: int | None = None


@dataclass(frozen=True)
class _StepNamespacePolicy:
    step_kind: str
    has_explicit_keq: bool


def _canonical_lookup(names: Iterable[str]) -> dict[str, str]:
    canonical_by_lower: dict[str, str] = {}
    for name in names:
        lowered = str(name).lower()
        existing = canonical_by_lower.get(lowered)
        if existing is not None and existing != name:
            raise ValueError(
                f"Conflicting mechanism parameter names for case-insensitive lookup: {existing!r} and {name!r}"
            )
        canonical_by_lower[lowered] = str(name)
    return canonical_by_lower


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
                    has_explicit_keq=bool(descriptor.has_explicit_keq),
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
                    has_explicit_keq=bool(descriptor.has_explicit_keq),
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
                has_explicit_keq=bool(descriptor.has_explicit_keq),
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
    canonical_names = frozenset(ordered_names)
    canonical_by_lower = _canonical_lookup(ordered_names)
    info_by_name = {item.canonical_name: item.info for item in ordered_items}
    return MechanismParameterNamespace(
        canonical_names=canonical_names,
        canonical_by_lower=canonical_by_lower,
        info_by_name=info_by_name,
        ordered_items=ordered_items,
    )


def _namespace_policy_from_step(step: object) -> _StepNamespacePolicy:
    missing_attrs = [
        attr for attr in ("is_equilibrium", "reversible", "kr", "Keq_input") if not hasattr(step, attr)
    ]
    if missing_attrs:
        raise ValueError(
            "Step is missing required namespace metadata: " + ", ".join(sorted(missing_attrs))
        )
    is_equilibrium_step = bool(
        getattr(step, "is_equilibrium")
        or (getattr(step, "reversible") and getattr(step, "kr") is not None)
    )
    return _StepNamespacePolicy(
        step_kind="equilibrium" if is_equilibrium_step else "reaction",
        has_explicit_keq=bool(getattr(step, "Keq_input", None) is not None),
    )


def _mechanism_step_descriptors(mechanism: object) -> list[_NamespaceStepDescriptor]:
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Mechanism metadata is missing or invalid; expected a step_index_map-backed mechanism.")
    raw_mapping = metadata.get("step_index_map")
    if not isinstance(raw_mapping, list):
        raise ValueError("Mechanism step_index_map is missing; cannot build an authoritative parameter namespace.")
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
        descriptors.append(
            _NamespaceStepDescriptor(
                step_index=step_index,
                step_kind=step_kind,
                has_explicit_keq=bool(raw_entry.get("has_Keq_param")),
                source_index=source_index,
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
    fallback_name: str,
) -> str:
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, dict):
        return str(fallback_name)
    raw_mapping = metadata.get("step_index_map")
    if not isinstance(raw_mapping, list):
        return str(fallback_name)
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
        return str(fallback_name)

    role_s = str(role)
    candidate = f"{role_s}{step_index}"
    namespace = build_namespace_from_mechanism(mechanism)
    resolved = namespace.resolve(candidate)
    if resolved.canonical_name is not None:
        return str(resolved.canonical_name)
    return str(fallback_name)


def build_namespace_from_ir_steps(steps: Sequence[object]) -> MechanismParameterNamespace:
    descriptors: list[_NamespaceStepDescriptor] = []
    for step_index, step in enumerate(steps, start=1):
        policy = _namespace_policy_from_step(step)
        descriptors.append(
            _NamespaceStepDescriptor(
                step_index=step_index,
                step_kind=policy.step_kind,
                has_explicit_keq=policy.has_explicit_keq,
            )
        )
    return _build_namespace(descriptors)


def build_flat_compat_namespace(canonical_names: Iterable[str]) -> MechanismParameterNamespace:
    """
    Build a compatibility-only namespace from already-canonical parameter names.

    No step metadata is available on this path, and no authoritative reconstruction
    is performed. This constructor is only for callers that already have canonical
    names and cannot access a mechanism-backed or IR-backed namespace source.
    """
    ordered_names: list[str] = []
    seen: set[str] = set()
    for raw_name in canonical_names:
        name = str(raw_name)
        match = _CANONICAL_NAME_RE.match(name)
        if match is None:
            raise ValueError(
                f"Compatibility namespace requires already-canonical names; got {name!r}."
            )
        if name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)
    canonical_by_lower = _canonical_lookup(ordered_names)
    ordered_items = tuple(
        MechanismParameterNamespaceItem(
            canonical_name=name,
            info=MechanismParameterInfo(
                canonical_name=name,
                step_index=None,
                step_kind=None,
                role=None,
                has_explicit_keq=None,
            ),
            source_index=None,
        )
        for name in ordered_names
    )
    return MechanismParameterNamespace(
        canonical_names=frozenset(ordered_names),
        canonical_by_lower=canonical_by_lower,
        info_by_name={item.canonical_name: item.info for item in ordered_items},
        ordered_items=ordered_items,
    )
