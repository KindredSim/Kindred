"""
Profile management for Kindred simulation settings.

Profiles are JSON files in kindred/data/presets/*.json that define:
- Solver settings (method, rtol, atol)
- Grid settings (N points)
- UI preferences (theme)
- Export defaults (format)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from kindred.core.simulator.solvers import normalize_solver_name

logger = logging.getLogger(__name__)

__all__ = ["Profile", "ProfileManager"]


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return float(default)
    return float(parsed)


@dataclass
class Profile:
    """
    Simulation and UI profile.

    Attributes
    ----------
    name : str
        Profile display name
    description : str
        Human-readable description
    solver_method : str
        ODE solver method (BDF, Radau)
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    grid_n : int
        Default number of grid points
    theme : str
        UI theme name
    export_format : str
        Default export format
    version : str
        Profile format version
    file_path : Path, optional
        Path to JSON file (if loaded from file)
    """
    name: str
    description: str = ""
    solver_method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-12
    grid_n: int = 1000
    theme: str = "default"
    export_format: str = "csv"
    version: str = "1.0"
    file_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], file_path: Optional[Path] = None) -> Profile:
        """
        Create Profile from dictionary.

        Parameters
        ----------
        data : dict
            Profile data from JSON
        file_path : Path, optional
            Source file path

        Returns
        -------
        Profile
            Profile instance
        """
        def _section(name: str) -> Dict[str, Any]:
            value = data.get(name, {})
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise TypeError(f"Profile section '{name}' must be a mapping.")
            return value

        solver = _section("solver")
        grid = _section("grid")
        ui = _section("ui")
        export = _section("export")
        solver_method, _warning = normalize_solver_name(solver.get("method", "BDF"))

        return cls(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            solver_method=solver_method,
            rtol=_coerce_positive_float(solver.get("rtol", 1e-6), default=1e-6),
            atol=_coerce_positive_float(solver.get("atol", 1e-12), default=1e-12),
            grid_n=grid.get("N", 1000),
            theme=ui.get("theme", "default"),
            export_format=export.get("default_format", "csv"),
            version=data.get("version", "1.0"),
            file_path=file_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Profile to dictionary.

        Returns
        -------
        dict
            Profile data suitable for JSON serialization
        """
        return {
            "name": self.name,
            "description": self.description,
            "solver": {
                "method": self.solver_method,
                "rtol": self.rtol,
                "atol": self.atol,
            },
            "grid": {
                "N": self.grid_n,
            },
            "ui": {
                "theme": self.theme,
            },
            "export": {
                "default_format": self.export_format,
            },
            "version": self.version,
        }

    def save(self, file_path: Path) -> None:
        """
        Save profile to JSON file.

        Parameters
        ----------
        file_path : Path
            Output file path
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        self.file_path = file_path
        logger.info(f"Saved profile '{self.name}' to {file_path}")


class ProfileManager:
    """
    Manager for loading and applying simulation profiles.

    Profiles are loaded from kindred/data/presets/*.json.
    """

    def __init__(self, presets_dir: Optional[Path] = None, auto_load: bool = True):
        """
        Initialize ProfileManager.

        Parameters
        ----------
        presets_dir : Path, optional
            Directory containing profile JSON files.
            If None, uses kindred/data/presets.
        auto_load : bool, optional
            If True, automatically load profiles during initialization.
            Default is True.
        """
        if presets_dir is None:
            # Default to kindred/data/presets
            from kindred.io.resources import get_resource_path
            presets_dir = get_resource_path("presets")

        self.presets_dir = Path(presets_dir)
        self._profiles: Dict[str, Profile] = {}
        self._active_profile: Optional[str] = None

        logger.debug(f"ProfileManager initialized with presets_dir: {self.presets_dir}")

        if auto_load:
            self.load_profiles()

    def load_profiles(self) -> None:
        """
        Load all profiles from presets directory.

        Only loads files with .json extension.
        Logs warnings for invalid profiles.
        """
        if not self.presets_dir.exists():
            logger.warning(f"Presets directory not found: {self.presets_dir}")
            return

        json_files = list(self.presets_dir.glob("*.json"))
        logger.info(f"Loading profiles from {self.presets_dir} ({len(json_files)} JSON files)")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                profile = Profile.from_dict(data, file_path=json_file)
                self._profiles[profile.name] = profile
                logger.debug(f"Loaded profile: {profile.name} from {json_file.name}")

            except Exception as e:
                logger.warning(f"Failed to load profile {json_file.name}: {e}")

        logger.info(f"Loaded {len(self._profiles)} profiles: {list(self._profiles.keys())}")

    def get_profile(self, name: str) -> Optional[Profile]:
        """
        Get profile by name.

        Parameters
        ----------
        name : str
            Profile name

        Returns
        -------
        Profile or None
            Profile instance, or None if not found
        """
        return self._profiles.get(name)

    def list_profiles(self) -> List[str]:
        """
        Get list of available profile names.

        Returns
        -------
        list of str
            Profile names sorted alphabetically
        """
        return sorted(self._profiles.keys())

    def get_active_profile(self) -> Optional[Profile]:
        """
        Get currently active profile.

        Returns
        -------
        Profile or None
            Active profile, or None if no profile is active
        """
        if self._active_profile is None:
            return None
        return self._profiles.get(self._active_profile)

    def set_active_profile(self, name: str) -> bool:
        """
        Set active profile by name.

        Parameters
        ----------
        name : str
            Profile name

        Returns
        -------
        bool
            True if profile was set successfully, False otherwise
        """
        if name not in self._profiles:
            logger.warning(f"Profile '{name}' not found")
            return False

        self._active_profile = name
        logger.info(f"Active profile set to '{name}'")
        return True

    def clear_active_profile(self) -> None:
        """Clear active profile (return to defaults)."""
        self._active_profile = None
        logger.info("Active profile cleared")

    def create_profile(
        self,
        name: str,
        description: str = "",
        solver_method: str = "BDF",
        rtol: float = 1e-6,
        atol: float = 1e-12,
        grid_n: int = 1000,
        theme: str = "default",
        export_format: str = "csv",
    ) -> Profile:
        """
        Create a new profile.

        Parameters
        ----------
        name : str
            Profile name
        description : str
            Profile description
        solver_method : str
            ODE solver method
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
        grid_n : int
            Grid points
        theme : str
            UI theme
        export_format : str
            Export format

        Returns
        -------
        Profile
            New profile instance
        """
        normalized_solver, _warning = normalize_solver_name(solver_method)
        profile = Profile(
            name=name,
            description=description,
            solver_method=normalized_solver,
            rtol=_coerce_positive_float(rtol, default=1e-6),
            atol=_coerce_positive_float(atol, default=1e-12),
            grid_n=grid_n,
            theme=theme,
            export_format=export_format,
        )

        self._profiles[name] = profile
        logger.info(f"Created profile '{name}'")
        return profile

    def delete_profile(self, name: str) -> bool:
        """
        Delete profile by name.

        Parameters
        ----------
        name : str
            Profile name

        Returns
        -------
        bool
            True if profile was deleted, False if not found
        """
        if name not in self._profiles:
            return False

        profile = self._profiles[name]

        # Delete file if it exists
        if profile.file_path and profile.file_path.exists():
            try:
                profile.file_path.unlink()
                logger.info(f"Deleted profile file: {profile.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete profile file: {e}")

        # Remove from registry
        del self._profiles[name]

        # Clear active if this was active
        if self._active_profile == name:
            self._active_profile = None

        logger.info(f"Deleted profile '{name}'")
        return True
