from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from kindred.core.mechanism_structure_snapshot import (
    MechanismStructureSnapshot,
    MechanismStructureSnapshotOwner,
)
from kindred.core.mechanism_source import MechanismAuthoringSource

class MainWindowMechanismHelpers:
    """Owns cached mechanism snapshots for MainWindow's mechanism-helper seam."""

    def __init__(self) -> None:
        self._last_mechanism: object | None = None
        self._last_mechanism_context: dict[str, Any] = {}
        self._structure_owner = MechanismStructureSnapshotOwner()

    def last_mechanism(self) -> object | None:
        return self._last_mechanism

    def last_mechanism_context(self) -> dict[str, Any]:
        context = dict(self._last_mechanism_context or {})
        solver_config = context.get("solver_config")
        if isinstance(solver_config, dict):
            context["solver_config"] = dict(solver_config)
        return context

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: dict[str, Any]) -> None:
        self._last_mechanism = mechanism
        self._last_mechanism_context = {
            "dsl_text": str(mechanism_text),
            "solver_config": dict(solver_config or {}),
            "timestamp": datetime.now().isoformat(),
        }

    def authoritative_structure_snapshot(
        self,
        *,
        source: MechanismAuthoringSource,
        units_identity: tuple[object, ...] = (),
        builder: Callable[[str], object],
    ) -> MechanismStructureSnapshot:
        return self._structure_owner.snapshot_for(
            source=source,
            units_identity=units_identity,
            builder=builder,
        )

    def clear_last_mechanism(self) -> None:
        self._last_mechanism = None
        self._last_mechanism_context = {}
        self._structure_owner.clear()
