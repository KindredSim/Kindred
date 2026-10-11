"""Application configuration, defaults, and recent-project persistence."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.project_document import (
    FitDefaults,
    SIMULATION_POINT_RANGE,
    ProjectDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
    default_fit_defaults,
    default_project_document,
    default_simulation_execution_settings,
    default_tab_scientific_settings,
    validate_fit_defaults,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.gui.theme_manager import readable_plot_color

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigController",
]

_RECENT_PROJECTS_KEY = "recent_files"
_RECENT_PROJECT_LIMIT = 10
_THEME_KEY = "ui/dark_mode"

_RUN_SETTINGS_KEYS = {
    "solver": "simulation/solver",
    "t_start": "simulation/t_start",
    "t_end": "simulation/time",
    "integration_points": "simulation/points",
}
_SCIENTIFIC_SETTINGS_KEYS = {
    "temperature": "simulation/temperature",
    "standard_conc_M": "simulation/standard_conc_M",
    "energy_unit": "simulation/energy_unit",
    "rtol": "simulation/rtol",
    "atol": "simulation/atol",
    "analytic_jacobian_policy": "simulation/use_analytic_jacobian",
    "use_authored_stops": "simulation/use_authored_stops",
}
_FIT_SETTINGS_KEYS = {
    "method": "fitting/method",
    "max_nfev": "fitting/max_nfev",
    "ftol": "fitting/ftol",
    "xtol": "fitting/xtol",
    "use_seed": "fitting/use_seed",
    "seed": "fitting/seed",
}
_RUN_USER_MAXIMUM_KEY = "simulation/run_user_maximum"
_RESULT_CACHE_CAPACITY_KEY = "results/cache_capacity"
_PREVIEW_POINTS_KEY = "sliders/preview_points"
_PREVIEW_DELAY_KEY = "sliders/preview_delay_ms"
_DEFAULT_RESULT_CACHE_CAPACITY = 32
_QT_POSITIVE_INT_MAX = (1 << 31) - 1
_DEFAULT_PREVIEW_POINTS = 50
_DEFAULT_PREVIEW_DELAY_MS = 150

_SPECIES_COLORS_KEY = "plot/species_colors"
_EXPERIMENTAL_POINT_SHAPE_KEY = "plot/experimental_point_shape"
_EXPERIMENTAL_POINT_SIZE_KEY = "plot/experimental_point_size"
_EXPERIMENTAL_POINT_OPACITY_KEY = "plot/experimental_point_opacity"
_WINDOW_LAYOUT_KEYS = {
    "is_maximized": "window/is_maximized",
    "geometry": "window/geometry",
    "dock_state": "window/dock_state",
}
_DEFAULT_EXPERIMENTAL_POINT_SIZE = 6
_DEFAULT_EXPERIMENTAL_POINT_OPACITY = 180.0 / 255.0
_SPECIES_SUFFIXES = ("_conc", "_concentration")
_BASE_SPECIES_RGB: tuple[tuple[int, int, int], ...] = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (0, 134, 149),
)


class ConfigController(QtCore.QObject):
    """The sole QSettings-backed owner of admitted application settings."""

    EXPERIMENTAL_POINT_SHAPES = (
        ("Circle", "o"), ("Square", "s"), ("Triangle", "t"),
        ("Diamond", "d"), ("Cross", "x"), ("Plus", "+"),
    )

    applicationPlotPreferencesChanged = QtCore.Signal()
    recentProjectsChanged = QtCore.Signal()
    resultCacheCapacityChanged = QtCore.Signal(int)
    previewSettingsChanged = QtCore.Signal()

    def __init__(self, settings: QtCore.QSettings):
        super().__init__()
        self._settings_store = settings
        self._species_palette = self._build_species_palette()

    def _settings(self) -> QtCore.QSettings:
        return self._settings_store

    @staticmethod
    def _parsed_int(value: object) -> int | None:
        if type(value) is int:
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text and text.lstrip("+-").isdigit():
                return int(text)
        return None

    def run_user_maximum_preference(self) -> int | None:
        """Return a clamped explicit Run maximum, or ``None`` for Run fallback."""

        settings = self._settings()
        if not settings.contains(_RUN_USER_MAXIMUM_KEY):
            return None
        parsed = self._parsed_int(settings.value(_RUN_USER_MAXIMUM_KEY))
        if parsed is None:
            return None
        return min(MAX_PARALLEL_WORKERS_CEILING, max(1, parsed))

    def set_run_user_maximum(self, value: int | None) -> int | None:
        if value is None:
            settings = self._settings()
            snapshot = self._snapshot_settings(settings, (_RUN_USER_MAXIMUM_KEY,))
            settings.remove(_RUN_USER_MAXIMUM_KEY)
            self._sync_or_report(
                settings,
                "Run user maximum",
                snapshot=snapshot,
                success_message="Run parallel maximum reset to Automatic.",
            )
            return None
        # The dialog's spin box admits 1 to the ceiling.
        accepted = value
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_RUN_USER_MAXIMUM_KEY,))
        settings.setValue(_RUN_USER_MAXIMUM_KEY, accepted)
        self._sync_or_report(
            settings,
            "Run user maximum",
            snapshot=snapshot,
            success_message=f"Run parallel maximum set to {accepted}.",
        )
        return accepted

    def result_cache_capacity(self) -> int:
        """Return the validated per-tab explicit-result cache capacity."""

        parsed = self._parsed_int(
            self._settings().value(
                _RESULT_CACHE_CAPACITY_KEY,
                _DEFAULT_RESULT_CACHE_CAPACITY,
            )
        )
        if parsed is None or not 1 <= parsed <= _QT_POSITIVE_INT_MAX:
            logger.warning(
                "Ignoring invalid persisted result cache capacity."
            )
            return _DEFAULT_RESULT_CACHE_CAPACITY
        return parsed

    def set_result_cache_capacity(self, value: int) -> int:
        """Persist and synchronously publish one positive Qt integer."""

        previous = self.result_cache_capacity()
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_RESULT_CACHE_CAPACITY_KEY,))
        settings.setValue(_RESULT_CACHE_CAPACITY_KEY, value)
        self._sync_or_report(
            settings,
            "result cache capacity",
            snapshot=snapshot,
            success_message=(
                f"Result cache capacity set to {value} per mechanism tab."
            ),
        )
        if previous != value:
            self.resultCacheCapacityChanged.emit(value)
        return value

    def preview_points(self) -> int:
        return self._preview_setting(
            _PREVIEW_POINTS_KEY, _DEFAULT_PREVIEW_POINTS, SIMULATION_POINT_RANGE,
        )

    def set_preview_points(self, value: int) -> int:
        return self._set_preview_setting(
            _PREVIEW_POINTS_KEY, value, self.preview_points(), label="Preview points",
        )

    def preview_delay_ms(self) -> int:
        return self._preview_setting(
            _PREVIEW_DELAY_KEY, _DEFAULT_PREVIEW_DELAY_MS, (0, 1000),
        )

    def set_preview_delay_ms(self, value: int) -> int:
        return self._set_preview_setting(
            _PREVIEW_DELAY_KEY, value, self.preview_delay_ms(), label="Preview delay (ms)",
        )

    def _preview_setting(
        self, key: str, default: int, bounds: tuple[int, int],
    ) -> int:
        parsed = self._parsed_int(self._settings().value(key, default))
        return default if parsed is None else min(bounds[1], max(bounds[0], parsed))

    def _set_preview_setting(
        self, key: str, value: int, previous: int, *, label: str,
    ) -> int:
        # The Simulation Settings editors admit the range.
        accepted = value
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (key,))
        settings.setValue(key, accepted)
        self._sync_or_report(
            settings, label, snapshot=snapshot,
            success_message=f"{label} set to {accepted}.",
        )
        if previous != accepted:
            self.previewSettingsChanged.emit()
        return accepted

    @staticmethod
    def _setting(
        settings: QtCore.QSettings,
        key: str,
        default: object,
        expected_type: type,
    ) -> object:
        try:
            if expected_type is float:
                # PySide6 coerces typed float reads of stored text via float32.
                return float(settings.value(key, default))
            return settings.value(key, default, type=expected_type)
        except (TypeError, ValueError, OverflowError):
            logger.warning("Ignoring invalid persisted setting %s.", key)
            return default

    def application_run_defaults(self) -> SimulationExecutionSettings:
        """Return the complete Run default for a newly created tab."""
        settings = self._settings()
        factory = default_simulation_execution_settings()
        try:
            candidate = SimulationExecutionSettings(
                solver=normalize_solver_name(
                    self._setting(
                        settings,
                        _RUN_SETTINGS_KEYS["solver"],
                        factory.solver,
                        str,
                    )
                ),
                t_start=float(
                    self._setting(
                        settings,
                        _RUN_SETTINGS_KEYS["t_start"],
                        factory.t_start,
                        float,
                    )
                ),
                t_end=float(
                    self._setting(
                        settings,
                        _RUN_SETTINGS_KEYS["t_end"],
                        factory.t_end,
                        float,
                    )
                ),
                integration_points=int(
                    self._setting(
                        settings,
                        _RUN_SETTINGS_KEYS["integration_points"],
                        factory.integration_points,
                        int,
                    )
                ),
            )
            return validate_simulation_execution_settings(
                candidate,
                "application Run defaults",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Ignoring invalid persisted application Run defaults: %s", exc
            )
            return factory

    def application_scientific_defaults(self) -> TabScientificSettings:
        """Return the separate scientific default for a newly created tab."""
        settings = self._settings()
        factory = default_tab_scientific_settings()
        try:
            candidate = TabScientificSettings(
                standard_conc_M=float(self._setting(
                    settings, _SCIENTIFIC_SETTINGS_KEYS["standard_conc_M"],
                    factory.standard_conc_M, float,
                )),
                energy_unit=str(self._setting(
                    settings, _SCIENTIFIC_SETTINGS_KEYS["energy_unit"],
                    factory.energy_unit, str,
                )),
                temperature=float(
                    self._setting(
                        settings,
                        _SCIENTIFIC_SETTINGS_KEYS["temperature"],
                        factory.temperature,
                        float,
                    )
                ),
                rtol=float(
                    self._setting(
                        settings,
                        _SCIENTIFIC_SETTINGS_KEYS["rtol"],
                        factory.rtol,
                        float,
                    )
                ),
                atol=float(
                    self._setting(
                        settings,
                        _SCIENTIFIC_SETTINGS_KEYS["atol"],
                        factory.atol,
                        float,
                    )
                ),
                analytic_jacobian_policy=bool(
                    self._setting(
                        settings,
                        _SCIENTIFIC_SETTINGS_KEYS["analytic_jacobian_policy"],
                        factory.analytic_jacobian_policy,
                        bool,
                    )
                ),
                use_authored_stops=bool(self._setting(
                    settings, _SCIENTIFIC_SETTINGS_KEYS["use_authored_stops"],
                    factory.use_authored_stops, bool,
                )),
            )
            return validate_tab_scientific_settings(
                candidate,
                "application tab-scientific defaults",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Ignoring invalid persisted tab-scientific defaults: %s", exc
            )
            return factory

    def application_fit_defaults(self) -> FitDefaults:
        """Return only optimizer defaults; obsolete Fit simulation keys are ignored."""
        settings = self._settings()
        factory = default_fit_defaults()
        try:
            candidate = FitDefaults(
                method=str(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["method"],
                        factory.method,
                        str,
                    )
                )
                .strip()
                .lower(),
                max_nfev=int(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["max_nfev"],
                        factory.max_nfev,
                        int,
                    )
                ),
                ftol=float(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["ftol"],
                        factory.ftol,
                        float,
                    )
                ),
                xtol=float(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["xtol"],
                        factory.xtol,
                        float,
                    )
                ),
                use_seed=bool(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["use_seed"],
                        factory.use_seed,
                        bool,
                    )
                ),
                seed=int(
                    self._setting(
                        settings,
                        _FIT_SETTINGS_KEYS["seed"],
                        factory.seed,
                        int,
                    )
                ),
            )
            return validate_fit_defaults(candidate)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Ignoring invalid persisted application Fit defaults: %s", exc
            )
            return factory

    def default_project_document(self) -> ProjectDocument:
        """Seed a new project from current application defaults."""
        return default_project_document(
            run_defaults=self.application_run_defaults(),
            scientific_settings=self.application_scientific_defaults(),
            fit_defaults=self.application_fit_defaults(),
        )

    def persist_run_defaults(
        self, candidate: SimulationExecutionSettings,
    ) -> SimulationExecutionSettings:
        # The Simulation Settings dialog admitted the candidate.
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, tuple(_RUN_SETTINGS_KEYS.values()))
        for field_name, key in _RUN_SETTINGS_KEYS.items():
            settings.setValue(key, getattr(candidate, field_name))
        self._sync_or_report(settings, "Run defaults", snapshot=snapshot)
        return candidate

    def persist_scientific_defaults(
        self,
        candidate: TabScientificSettings,
    ) -> TabScientificSettings:
        # The Simulation Settings dialog admitted the candidate.
        settings = self._settings()
        snapshot = self._snapshot_settings(
            settings,
            tuple(_SCIENTIFIC_SETTINGS_KEYS.values()),
        )
        for field_name, key in _SCIENTIFIC_SETTINGS_KEYS.items():
            settings.setValue(key, getattr(candidate, field_name))
        self._sync_or_report(
            settings,
            "default scientific settings",
            snapshot=snapshot,
        )
        return candidate

    def persist_fit_defaults(self, candidate: FitDefaults) -> FitDefaults:
        # The Fit defaults dialog admitted the candidate.
        settings = self._settings()
        snapshot = self._snapshot_settings(
            settings,
            tuple(_FIT_SETTINGS_KEYS.values()),
        )
        for field_name, key in _FIT_SETTINGS_KEYS.items():
            settings.setValue(key, getattr(candidate, field_name))
        self._sync_or_report(
            settings,
            "Fit defaults",
            snapshot=snapshot,
            success_message=(
                "Application Fit defaults updated for newly created projects."
            ),
        )
        return candidate

    def persist_theme(self, dark_mode: bool) -> None:
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_THEME_KEY,))
        settings.setValue(_THEME_KEY, bool(dark_mode))
        self._sync_or_report(settings, "theme", snapshot=snapshot)

    def theme_preference(self) -> bool:
        """Return the persisted application theme preference."""

        return bool(
            self._setting(
                self._settings(),
                _THEME_KEY,
                True,
                bool,
            )
        )

    def window_layout_preference(
        self,
    ) -> tuple[bool, object | None, object | None]:
        """Return persisted physical window state and optional Qt payloads."""

        settings = self._settings()
        return (
            bool(
                self._setting(
                    settings,
                    _WINDOW_LAYOUT_KEYS["is_maximized"],
                    False,
                    bool,
                )
            ),
            settings.value(_WINDOW_LAYOUT_KEYS["geometry"]),
            settings.value(_WINDOW_LAYOUT_KEYS["dock_state"]),
        )

    def persist_window_layout(
        self,
        is_maximized: bool,
        geometry: object,
        dock_state: object,
    ) -> None:
        """Persist the complete MainWindow layout as one settings write."""

        settings = self._settings()
        keys = tuple(_WINDOW_LAYOUT_KEYS.values())
        snapshot = self._snapshot_settings(settings, keys)
        settings.setValue(_WINDOW_LAYOUT_KEYS["is_maximized"], is_maximized)
        settings.setValue(_WINDOW_LAYOUT_KEYS["geometry"], geometry)
        settings.setValue(_WINDOW_LAYOUT_KEYS["dock_state"], dock_state)
        self._sync_or_report(
            settings,
            "window layout",
            snapshot=snapshot,
        )

    @staticmethod
    def _snapshot_settings(
        settings: QtCore.QSettings,
        keys: Sequence[str],
    ) -> tuple[tuple[str, bool, object], ...]:
        snapshot: list[tuple[str, bool, object]] = []
        for key in keys:
            existed = settings.contains(key)
            snapshot.append((key, existed, settings.value(key) if existed else None))
        return tuple(snapshot)

    @staticmethod
    def _restore_settings(
        settings: QtCore.QSettings,
        snapshot: Sequence[tuple[str, bool, object]],
    ) -> None:
        for key, existed, raw_value in snapshot:
            try:
                if existed:
                    settings.setValue(key, raw_value)
                else:
                    settings.remove(key)
            except Exception:
                logger.exception("Could not restore failed setting %s.", key)

    def _sync_or_report(
        self,
        settings: QtCore.QSettings,
        label: str,
        *,
        snapshot: Sequence[tuple[str, bool, object]],
        success_message: str | None = None,
    ) -> None:
        message = f"{label[:1].upper()}{label[1:]} could not be saved."
        try:
            settings.sync()
            if settings.status() != QtCore.QSettings.Status.NoError:
                raise OSError(f"QSettings status {settings.status()}")
        except Exception as exc:
            logger.warning("%s (%s)", message, exc)
            self._restore_settings(settings, snapshot)
            try:
                # QSettings may write new values while retaining a prior
                # FormatError status. Persist the restored accepted values.
                settings.sync()
            except Exception:
                logger.exception("Could not synchronize restored setting values for %s.", label)
            raise OSError(message) from exc
        if success_message is not None:
            logger.info(success_message)

    def recent_projects(self) -> tuple[str, ...]:
        raw = self._settings().value(_RECENT_PROJECTS_KEY, [])
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, (list, tuple)):
            candidates = list(raw)
        else:
            candidates = []
        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            path = str(candidate).strip()
            identity = os.path.normcase(os.path.abspath(path))
            if not path or identity in seen:
                continue
            seen.add(identity)
            result.append(path)
            if len(result) == _RECENT_PROJECT_LIMIT:
                break
        return tuple(result)

    def record_recent_project(self, path: str) -> tuple[str, ...]:
        identity = os.path.normcase(path)
        current = [
            item
            for item in self.recent_projects()
            if os.path.normcase(os.path.abspath(item)) != identity
        ]
        result = tuple([path, *current][:_RECENT_PROJECT_LIMIT])
        self._persist_recent_projects(result)
        return result

    def _persist_recent_projects(self, projects: Sequence[str]) -> None:
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_RECENT_PROJECTS_KEY,))
        settings.setValue(_RECENT_PROJECTS_KEY, list(projects))
        self._sync_or_report(
            settings,
            "recent projects",
            snapshot=snapshot,
        )
        self.recentProjectsChanged.emit()

    def clear_recent_files(self) -> None:
        self._persist_recent_projects(())

    def species_color(
        self,
        species: str,
        known_species: Sequence[str] = (),
    ) -> QtGui.QColor:
        """Return one application preference color as an independent QColor."""
        canonical = self._resolve_species_name(species, known_species)
        override = self._species_color_overrides().get(canonical)
        if override is not None:
            return QtGui.QColor(override)
        slot = self._default_species_slot(canonical, known_species)
        # qdarktheme supplies application Text while styling Base per widget.
        dark_mode = (
            QtWidgets.QApplication.palette().color(QtGui.QPalette.Text).lightness() > 128
        )
        return readable_plot_color(self._species_palette[slot], dark_mode)

    def set_species_color(
        self,
        species: str,
        color: QtGui.QColor,
        known_species: Sequence[str] = (),
    ) -> None:
        canonical = self._resolve_species_name(species, known_species)
        overrides = dict(self._species_color_overrides())
        overrides[canonical] = QtGui.QColor(color).name(QtGui.QColor.NameFormat.HexRgb)
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_SPECIES_COLORS_KEY,))
        settings.setValue(_SPECIES_COLORS_KEY, overrides)
        self._sync_or_report(
            settings,
            "species colors",
            snapshot=snapshot,
        )
        self.applicationPlotPreferencesChanged.emit()

    def experimental_point_shape(self) -> str:
        value = self._settings().value(_EXPERIMENTAL_POINT_SHAPE_KEY, "o")
        return (
            value
            if value in tuple(code for _label, code in self.EXPERIMENTAL_POINT_SHAPES)
            else "o"
        )

    def set_experimental_point_shape(self, shape: str) -> str:
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_EXPERIMENTAL_POINT_SHAPE_KEY,))
        settings.setValue(_EXPERIMENTAL_POINT_SHAPE_KEY, shape)
        self._sync_or_report(settings, "experimental point shape", snapshot=snapshot)
        self.applicationPlotPreferencesChanged.emit()
        return shape

    def experimental_point_size(self) -> int:
        value = self._setting(
            self._settings(),
            _EXPERIMENTAL_POINT_SIZE_KEY,
            _DEFAULT_EXPERIMENTAL_POINT_SIZE,
            int,
        )
        try:
            return self._validate_experimental_point_size(value)
        except (TypeError, ValueError):
            return _DEFAULT_EXPERIMENTAL_POINT_SIZE

    def set_experimental_point_size(self, size: int) -> int:
        accepted = size
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_EXPERIMENTAL_POINT_SIZE_KEY,))
        settings.setValue(_EXPERIMENTAL_POINT_SIZE_KEY, accepted)
        self._sync_or_report(
            settings,
            "experimental point size",
            snapshot=snapshot,
        )
        self.applicationPlotPreferencesChanged.emit()
        return accepted

    def experimental_point_opacity(self) -> float:
        value = self._setting(
            self._settings(),
            _EXPERIMENTAL_POINT_OPACITY_KEY,
            _DEFAULT_EXPERIMENTAL_POINT_OPACITY,
            float,
        )
        try:
            return self._validate_experimental_point_opacity(value)
        except (TypeError, ValueError):
            return _DEFAULT_EXPERIMENTAL_POINT_OPACITY

    def set_experimental_point_opacity(self, opacity: float) -> float:
        accepted = opacity
        settings = self._settings()
        snapshot = self._snapshot_settings(settings, (_EXPERIMENTAL_POINT_OPACITY_KEY,))
        settings.setValue(_EXPERIMENTAL_POINT_OPACITY_KEY, accepted)
        self._sync_or_report(
            settings,
            "experimental point opacity",
            snapshot=snapshot,
        )
        self.applicationPlotPreferencesChanged.emit()
        return accepted

    @staticmethod
    def _validate_experimental_point_size(value: object) -> int:
        if type(value) is bool or not isinstance(value, int):
            raise TypeError("Experimental point size must be an integer.")
        if not 1 <= value <= 20:
            raise ValueError("Experimental point size must be between 1 and 20.")
        return int(value)

    @staticmethod
    def _validate_experimental_point_opacity(value: object) -> float:
        if type(value) is bool:
            raise TypeError("Experimental point opacity must be numeric.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Experimental point opacity must be numeric.") from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError("Experimental point opacity must be between zero and one.")
        return parsed

    def _species_color_overrides(self) -> Mapping[str, str]:
        raw = self._settings().value(_SPECIES_COLORS_KEY, {})
        if not isinstance(raw, Mapping):
            return MappingProxyType({})
        accepted: dict[str, str] = {}
        for name, value in raw.items():
            key = str(name).strip()
            color = QtGui.QColor(str(value))
            if key and color.isValid():
                accepted[key] = color.name(QtGui.QColor.NameFormat.HexRgb)
        return MappingProxyType(accepted)

    @classmethod
    def _resolve_species_name(
        cls,
        species: str,
        known_species: Sequence[str],
    ) -> str:
        raw = species
        roster = tuple(known_species)
        if not roster:
            return raw
        exact = {name: name for name in roster}
        if raw in exact:
            return exact[raw]
        folded = {name.casefold(): name for name in roster}
        if raw.casefold() in folded:
            return folded[raw.casefold()]
        lowered = raw.casefold()
        for suffix in _SPECIES_SUFFIXES:
            if lowered.endswith(suffix) and len(raw) > len(suffix):
                base = raw[: -len(suffix)].casefold()
                if base in folded:
                    return folded[base]
        return raw

    def _default_species_slot(
        self,
        canonical: str,
        known_species: Sequence[str],
    ) -> int:
        roster = sorted(
            known_species,
            key=str.casefold,
        )
        if canonical in roster:
            return roster.index(canonical) % len(self._species_palette)
        return sum(ord(character) for character in canonical) % len(
            self._species_palette
        )

    @classmethod
    def _build_species_palette(cls) -> tuple[QtGui.QColor, ...]:
        base = [QtGui.QColor(*rgb) for rgb in _BASE_SPECIES_RGB]
        darker = [
            cls._mix_color(color, factor=0.22, toward_white=False) for color in base
        ]
        lighter = [
            cls._mix_color(color, factor=0.18, toward_white=True) for color in base
        ]
        return tuple(base + darker + lighter)

    @staticmethod
    def _mix_color(
        color: QtGui.QColor,
        *,
        factor: float,
        toward_white: bool,
    ) -> QtGui.QColor:
        target = 255 if toward_white else 0

        def channel(value: int) -> int:
            mixed = (
                value + (target - value) * factor
                if toward_white
                else value * (1.0 - factor)
            )
            return max(0, min(int(round(mixed)), 255))

        return QtGui.QColor(
            channel(color.red()),
            channel(color.green()),
            channel(color.blue()),
        )
