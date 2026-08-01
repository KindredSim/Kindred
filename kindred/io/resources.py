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
    "list_resources",
    "get_preset_mechanism_source",
    "get_intervention_example_source",
    "get_all_intervention_example_specs",
    "InterventionExampleCatalogError",
    "InterventionExampleSpec",
]


class InterventionExampleCatalogError(RuntimeError):
    """The packaged intervention-example catalog is missing or unreadable."""


@dataclass(frozen=True, slots=True)
class InterventionExampleSpec:
    """One dynamically discovered packaged intervention example."""

    id: str
    title: str
    path: str
    type: str = "intervention"


_MANDATORY_INTERVENTION_EXAMPLE_IDS = frozenset(
    {"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"}
)
_FIXED_PRESET_IDS = frozenset(f"M{index}" for index in range(1, 10))


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


def list_resources(subdirectory: str = "") -> list[str]:
    """
    List available resources in kindred/data subdirectory.

    Parameters
    ----------
    subdirectory : str
        Subdirectory within kindred/data (e.g., "presets")
        Empty string lists root data directory

    Returns
    -------
    list of str
        Resource names (filenames only, not full paths)
    """
    from importlib.resources import files

    try:
        data_pkg = files("kindred").joinpath("data")
        target = data_pkg.joinpath(subdirectory) if subdirectory else data_pkg

        resources = []
        for item in target.iterdir():
            if item.is_file():
                resources.append(item.name)

        return sorted(resources)

    except Exception as e:
        logger.warning(f"Could not list resources in kindred/data/{subdirectory}: {e}")
        return []


# Convenience functions for common resources


def _natural_key(name: str) -> list[object]:
    """Split into numeric and text chunks for deterministic ordering."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _title_from_resource_text(text: str, *, fallback: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        prefix = "# title:"
        if stripped.lower().startswith(prefix):
            title = stripped[len(prefix):].strip()
            return title or fallback
        return fallback
    return fallback


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
    if normalized not in _FIXED_PRESET_IDS:
        raise ValueError("Preset mechanism must be one of M1 through M9.")
    return MechanismAuthoringSource.from_full_dsl_text(
        get_resource_text(f"presets/{normalized}.txt")
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
    return MechanismAuthoringSource.from_full_dsl_text(
        get_resource_text(f"interventions/{normalized}.txt")
    )


def get_all_intervention_example_specs() -> tuple[InterventionExampleSpec, ...]:
    """
    Discover and validate all bundled intervention examples.

    Discovery is dynamic and naturally ordered, so later non-contiguous IDs
    are included. The mandatory I1-I8 package contract fails closed when a
    required file is absent or any discovered resource cannot be read.

    Returns
    -------
    tuple of InterventionExampleSpec
        Typed metadata for every discovered intervention resource.

    Raises
    ------
    InterventionExampleCatalogError
        If discovery fails, I1-I8 are incomplete, or a discovered resource is
        unreadable.
    """
    from importlib.resources import files

    try:
        base = files("kindred").joinpath("data", "interventions")
        names = sorted(
            (
                item.name
                for item in base.iterdir()
                if item.is_file() and item.suffix == ".txt"
            ),
            key=_natural_key,
        )
    except Exception as exc:
        raise InterventionExampleCatalogError(
            "Unable to discover packaged intervention examples."
        ) from exc

    discovered_ids = {
        name[:-4] if name.endswith(".txt") else name
        for name in names
    }
    missing = sorted(
        _MANDATORY_INTERVENTION_EXAMPLE_IDS - discovered_ids,
        key=_natural_key,
    )
    if missing:
        raise InterventionExampleCatalogError(
            "Mandatory packaged intervention examples are missing: "
            + ", ".join(missing)
        )

    examples: list[InterventionExampleSpec] = []
    for name in names:
        base = name[:-4] if name.endswith(".txt") else name
        path = f"interventions/{name}"
        try:
            text = get_resource_text(path)
        except Exception as exc:
            raise InterventionExampleCatalogError(
                f"Packaged intervention example {base!r} is unreadable."
            ) from exc
        examples.append(
            InterventionExampleSpec(
                id=base,
                path=path,
                title=_title_from_resource_text(text, fallback=base),
            )
        )

    return tuple(examples)
