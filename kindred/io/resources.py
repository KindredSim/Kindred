"""
Safe resource loader using importlib.resources.

Provides Python 3.10+ compatible resource loading for kindred/data assets.
Works with both installed wheels and editable installs.

Usage:
    from kindred.io.resources import get_resource_text, get_resource_path

    # Load a bundled preset mechanism
    preset_text = get_resource_text("presets/M1.txt")

    # Get path to icon
    icon_path = get_resource_path("assets/kindred.ico")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kindred.core.mechanism_source import MechanismAuthoringSource

__all__ = [
    "get_resource_text",
    "get_resource_path",
    "get_preset_mechanism_source",
    "PRESET_MECHANISM_TITLES",
    "get_intervention_example_source",
    "get_all_intervention_example_specs",
    "InterventionExampleSpec",
]


@dataclass(frozen=True, slots=True)
class InterventionExampleSpec:
    """One named packaged intervention example."""

    id: str
    title: str


_INTERVENTION_EXAMPLE_TITLES = {
    "I1": "Set a concentration",
    "I2": "Increase and decrease a concentration",
    "I3": "Continuous addition and removal",
    "I4": "Maintain constant reagent",
    "I5": "Trap an intermediate",
    "I6": "Replenish a reagent",
    "I7": "Stop at a concentration",
    "I8": "Stop at a conversion",
    "I9": "Repeated additions",
    "I10": "Intermittent feeding",
    "I11": "Light–dark photoswitching",
    "I12": "Mix, react, and quench",
    "I13": "Add stock solution and solvent",
    "I14": "Pump a stock solution",
    "I15": "Withdraw an aliquot",
    "I16": "Scale concentrations",
    "I17": "Repeated dosing and sampling",
}
PRESET_MECHANISM_TITLES = {
    "M1": "A -> B",
    "M2": "A <-> B",
    "M3": "A + B -> C",
    "M4": "2*A <-> A2",
    "M5": "A <-> B, B -> C",
    "M6": "A -> B, A -> C",
    "M7": "E + S <-> ES, ES -> E + P",
    "M8": "A + B <-> C, C -> D, C -> E (Arrhenius)",
    "M9": "A + B <-> C, C -> D, C -> E (Eyring)",
    "M10": "A + B <-> C, C -> D, D -> E (Temperature Change)",
    "M11": "PB cascade",
    "M12": "Intermolecular Competition",
    "M13": "Coupled Reversible Cycles",
}


def get_resource_text(relative_path: str, encoding: str = "utf-8") -> str:
    """
    Load text content from kindred/data resource.

    Parameters
    ----------
    relative_path : str
        Path relative to kindred/data (e.g., "presets/M1.txt")
    encoding : str
        Text encoding (default: utf-8)

    Returns
    -------
    str
        Resource text content

    Raises
    ------
    FileNotFoundError
        If resource doesn't exist
    """
    from importlib.resources import files

    return files("kindred").joinpath("data").joinpath(relative_path).read_text(encoding=encoding)


def get_resource_path(relative_path: str) -> Path:
    """Return the filesystem path of one packaged kindred/data resource.

    Kindred ships as a filesystem package (wheel or editable install), and
    every caller passes a packaged constant path.
    """
    from importlib.resources import files

    return Path(os.fspath(files("kindred").joinpath("data", relative_path)))


# Convenience functions for common resources


def get_preset_mechanism_source(preset_id: str) -> MechanismAuthoringSource:
    """
    Load a preset mechanism by ID as a complete authoring source.

    Parameters
    ----------
    preset_id : str
        Preset ID (e.g., "M1", "M2", ...)

    Returns
    -------
    MechanismAuthoringSource
        Complete mechanism source.
    """
    return MechanismAuthoringSource.from_parts(
        reactions_text=get_resource_text(f"presets/{preset_id}.txt")
    )


def get_intervention_example_source(example_id: str) -> MechanismAuthoringSource:
    """
    Load a bundled intervention example by ID as a complete authoring source.

    Parameters
    ----------
    example_id : str
        Intervention example ID (e.g., "I1", "I2", ...)

    Returns
    -------
    MechanismAuthoringSource
        Complete mechanism source.
    """
    return MechanismAuthoringSource.from_parts(
        reactions_text=get_resource_text(f"interventions/{example_id}.txt")
    )


def get_all_intervention_example_specs() -> tuple[InterventionExampleSpec, ...]:
    """Return the ordered catalog of packaged intervention examples."""

    return tuple(
        InterventionExampleSpec(id=example_id, title=title)
        for example_id, title in _INTERVENTION_EXAMPLE_TITLES.items()
    )
