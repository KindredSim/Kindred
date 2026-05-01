import inspect

import pytest

from kindred.core.mechanism_structure_snapshot import MechanismStructureSnapshotOwner


pytestmark = [pytest.mark.unit]


def test_structure_snapshot_reuses_same_structure_across_runtime_initial_changes():
    signature = inspect.signature(MechanismStructureSnapshotOwner.snapshot_for)
    assert "runtime_initials_identity" not in signature.parameters

    owner = MechanismStructureSnapshotOwner()
    builds: list[str] = []

    def build(source: str) -> object:
        builds.append(str(source))
        return object()

    first = owner.snapshot_for(
        reactions_text="reaction: A -> B; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )
    second = owner.snapshot_for(
        reactions_text="reaction: A -> B; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )

    assert first is second
    assert len(builds) == 1


def test_structure_snapshot_rebuilds_when_structural_source_changes():
    owner = MechanismStructureSnapshotOwner()
    builds: list[str] = []

    def build(source: str) -> object:
        builds.append(str(source))
        return object()

    first = owner.snapshot_for(
        reactions_text="reaction: A -> B; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )
    second = owner.snapshot_for(
        reactions_text="reaction: A -> C; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )

    assert first is not second
    assert len(builds) == 2


def test_structure_snapshot_owner_reuses_only_current_identity():
    owner = MechanismStructureSnapshotOwner()
    builds: list[str] = []

    def build(source: str) -> object:
        builds.append(str(source))
        return object()

    first = owner.snapshot_for(
        reactions_text="reaction: A -> B; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )
    second = owner.snapshot_for(
        reactions_text="reaction: A -> C; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )
    third = owner.snapshot_for(
        reactions_text="reaction: A -> B; k=1.0",
        state_network_text="",
        units_identity=("temperature_K", "298.15"),
        builder=build,
    )

    assert first is not second
    assert third is not first
    assert len(builds) == 3
