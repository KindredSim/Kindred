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

import hashlib
import logging
import os
import re
import tempfile
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

    For resources that already exist on disk (typical wheel installs), returns
    the on-disk path under the installed package.

    For resources loaded from non-filesystem importers (e.g., zipimport),
    materializes the resource into a per-user cache directory and returns the
    cached on-disk path. The returned path remains valid after this function
    returns.

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

        try:
            fs_path = Path(os.fspath(resource_file)).resolve()
            if fs_path.exists():
                return fs_path
        except TypeError:
            pass

        cache_root = os.environ.get("KINDRED_RESOURCE_CACHE_DIR", "").strip()
        if cache_root:
            cache_base = Path(cache_root).expanduser().resolve()
        else:
            cache_base = Path(tempfile.gettempdir()).resolve() / "kindred_resource_cache"

        def _sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def _materialize_file(*, src, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = src.read_bytes()
            data_hash = hashlib.sha256(data).hexdigest()

            if dest.exists() and dest.is_file():
                try:
                    if dest.stat().st_size == len(data) and _sha256_file(dest) == data_hash:
                        return
                except OSError:
                    pass

            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    dir=str(dest.parent),
                    prefix=dest.name + ".",
                    suffix=".tmp",
                ) as handle:
                    handle.write(data)
                    tmp_path = Path(handle.name)
                os.replace(str(tmp_path), str(dest))
            finally:
                if tmp_path is not None and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        def _materialize_dir(*, src, dest_dir: Path) -> None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                out = dest_dir / child.name
                if child.is_dir():
                    _materialize_dir(src=child, dest_dir=out)
                elif child.is_file():
                    _materialize_file(src=child, dest=out)

        target = (cache_base / "kindred" / "data" / Path(*parts)).resolve()
        if resource_file.is_dir():
            _materialize_dir(src=resource_file, dest_dir=target)
            return target

        _materialize_file(src=resource_file, dest=target)
        return target

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
