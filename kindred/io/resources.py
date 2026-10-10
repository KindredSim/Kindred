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

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from kindred.core.mechanism_source import MechanismAuthoringSource

logger = logging.getLogger(__name__)

__all__ = [
    "get_resource_text",
    "get_resource_path",
    "get_preset_mechanism_source",
    "PRESET_MECHANISM_TITLES",
    "get_intervention_example_source",
    "get_all_intervention_example_specs",
    "InterventionExampleCatalogError",
    "InterventionExampleSpec",
]


class InterventionExampleCatalogError(RuntimeError):
    """The packaged intervention-example catalog is missing or unreadable."""


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

    try:
        data_pkg = files("kindred").joinpath("data")
        resource_file = data_pkg.joinpath(relative_path)

        text = resource_file.read_text(encoding=encoding)
        logger.debug(f"Loaded resource: kindred/data/{relative_path}")
        return text

    except (FileNotFoundError, AttributeError) as e:
        logger.error(f"Resource not found: kindred/data/{relative_path}")
        raise FileNotFoundError(
            f"Resource not found: kindred/data/{relative_path}"
        ) from e


def get_resource_path(relative_path: str) -> Path:
    """
    Get filesystem path to kindred/data resource.

    Returns the on-disk path under the installed or editable package.

    Parameters
    ----------
    relative_path : str
        Path relative to kindred/data (e.g., "assets/kindred.ico")

    Returns
    -------
    Path
        Path to resource file

    Raises
    ------
    FileNotFoundError
        If resource doesn't exist
    """
    from importlib.resources import files

    try:
        raw = (relative_path or "").strip()
        if not raw:
            raise FileNotFoundError(f"Resource not found: kindred/data/{relative_path}")
        if re.match(r"^[A-Za-z]:", raw):
            raise FileNotFoundError(f"Resource not found: kindred/data/{relative_path}")

        data_pkg = files("kindred").joinpath("data")
        normalized = raw
        if os.altsep:
            normalized = normalized.replace(os.altsep, os.sep)
        normalized = normalized.replace("\\", os.sep).lstrip(os.sep)
        parts = [p for p in normalized.split(os.sep) if p]
        if not parts or ".." in parts:
            raise FileNotFoundError(f"Resource not found: kindred/data/{relative_path}")

        resource_file = data_pkg.joinpath(*parts)
        if not (resource_file.is_file() or resource_file.is_dir()):
            raise FileNotFoundError(f"Resource not found: kindred/data/{relative_path}")

        # Kindred ships as a filesystem package (wheel or editable install);
        # a resource without a filesystem path is treated as missing.
        try:
            fs_path = Path(os.fspath(resource_file)).resolve()
        except TypeError as exc:
            raise FileNotFoundError(
                f"Resource has no filesystem path: kindred/data/{relative_path}"
            ) from exc
        if not fs_path.exists():
            raise FileNotFoundError(f"Resource not found: kindred/data/{relative_path}")
        return fs_path

    except (FileNotFoundError, AttributeError) as e:
        logger.error(f"Resource not found: kindred/data/{relative_path}")
        raise FileNotFoundError(
            f"Resource not found: kindred/data/{relative_path}"
        ) from e


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
    normalized = str(preset_id).strip()
    if normalized not in PRESET_MECHANISM_TITLES:
        raise ValueError(f"Unknown packaged preset mechanism {normalized!r}.")
    return MechanismAuthoringSource.from_parts(
        reactions_text=get_resource_text(f"presets/{normalized}.txt")
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
    normalized = str(example_id).strip()
    available = {spec.id for spec in get_all_intervention_example_specs()}
    if normalized not in available:
        raise ValueError(
            f"Unknown packaged intervention example {normalized!r}."
        )
    return MechanismAuthoringSource.from_parts(
        reactions_text=get_resource_text(f"interventions/{normalized}.txt")
    )


def get_all_intervention_example_specs() -> tuple[InterventionExampleSpec, ...]:
    """
    Return the ordered catalog after verifying every listed resource is readable.

    Returns
    -------
    tuple of InterventionExampleSpec
        Names and IDs for every cataloged intervention resource.

    Raises
    ------
    InterventionExampleCatalogError
        If any cataloged resource is missing or unreadable.
    """
    examples: list[InterventionExampleSpec] = []
    for example_id, title in _INTERVENTION_EXAMPLE_TITLES.items():
        try:
            get_resource_text(f"interventions/{example_id}.txt")
        except Exception as exc:
            raise InterventionExampleCatalogError(
                f"Packaged intervention example {example_id!r} is missing or unreadable."
            ) from exc
        examples.append(
            InterventionExampleSpec(
                id=example_id,
                title=title,
            )
        )

    return tuple(examples)
