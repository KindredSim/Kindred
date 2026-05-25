"""Canonical mechanism structure snapshot ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from kindred.core.mechanism_source import MechanismAuthoringSource


def _normalize_text_for_identity(text: str) -> str:
    return "\n".join(" ".join(str(line).split()) for line in str(text or "").splitlines()).strip()


def _normalize_identity_parts(parts: Sequence[object] | None) -> tuple[str, ...]:
    return tuple(str(part) for part in (parts or ()))


@dataclass(frozen=True)
class MechanismStructureSnapshot:
    identity: tuple[str, str, tuple[str, ...]]
    full_dsl: str
    mechanism: object


class MechanismStructureSnapshotOwner:
    """Owns reuse identity for authoritative parsed mechanism structure."""

    def __init__(self) -> None:
        self._snapshot: MechanismStructureSnapshot | None = None

    def clear(self) -> None:
        self._snapshot = None

    def snapshot_for(
        self,
        *,
        source: MechanismAuthoringSource,
        units_identity: Sequence[object] | None = None,
        builder: Callable[[str], object],
    ) -> MechanismStructureSnapshot:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        reactions_s = str(source.reactions_text or "")
        state_network_s = str(source.state_network_dsl or "")
        full_dsl = source.full_dsl
        identity = (
            _normalize_text_for_identity(reactions_s),
            _normalize_text_for_identity(state_network_s),
            _normalize_identity_parts(units_identity),
        )
        current = self._snapshot
        if current is not None and current.identity == identity:
            return current
        mechanism = builder(full_dsl)
        snapshot = MechanismStructureSnapshot(
            identity=identity,
            full_dsl=full_dsl,
            mechanism=mechanism,
        )
        self._snapshot = snapshot
        return snapshot
