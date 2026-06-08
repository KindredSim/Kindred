from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import os
from typing import Callable, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

logger = logging.getLogger(__name__)

__all__ = ["ConfigController"]


@dataclass(frozen=True, slots=True)
class ConfigControllerPort:
    """Explicit injected wiring surface for ConfigController."""

    parent: QtCore.QObject
    settings: Callable[[], object]
    is_maximized: Callable[[], bool]
    restore_maximized_state: Callable[[], None]
    restore_geometry: Callable[[object], None]
    save_geometry: Callable[[], object]
    restore_window_state: Callable[[object], None]
    save_window_state: Callable[[], object]
    schedule_restored_floating_dock_recovery: Callable[[], None]
    splitter: Callable[[], object | None]
    set_temperature: Callable[[float], None]
    temperature: Callable[[], float]
    set_simulation_time_text: Callable[[str], None]
    simulation_time_text: Callable[[], str]
    set_num_points: Callable[[int], None]
    num_points: Callable[[], int]
    set_slider_preview_points: Callable[[int], None]
    slider_preview_points: Callable[[], int]
    set_slider_preview_solver: Callable[[str], None]
    slider_preview_solver: Callable[[], str]
    apply_solver_runtime_state: Callable[..., None]
    has_explicit_startup_solver_override: Callable[[], bool]
    has_explicit_startup_rtol_override: Callable[[], bool]
    has_explicit_startup_atol_override: Callable[[], bool]
    initial_solver_name: Callable[[], Optional[str]]
    explicit_startup_solver_name: Callable[[], Optional[str]]
    initial_rtol: Callable[[], Optional[float]]
    explicit_startup_rtol: Callable[[], Optional[float]]
    initial_atol: Callable[[], Optional[float]]
    explicit_startup_atol: Callable[[], Optional[float]]
    set_use_sparse_jacobian: Callable[[bool], None]
    use_sparse_jacobian: Callable[[], bool]
    set_wegscheider_cyclicity_enabled: Callable[[bool], None]
    wegscheider_cyclicity_enabled: Callable[[], bool]
    max_parallel_batch_workers: Callable[[], int]
    set_max_parallel_batch_workers: Callable[[int], None]
    batch_runtime_lane_budget: Callable[[], int]
    set_batch_runtime_lane_budget: Callable[[int], None]
    limit_blas_threads_per_worker: Callable[[], bool]
    set_limit_blas_threads_per_worker: Callable[[bool], None]
    result_cache_cap: Callable[[], int]
    preview_cache_cap: Callable[[], int]
    set_cache_caps: Callable[..., None]
    set_ribbon_collapsed: Callable[[bool], None]
    ribbon_collapsed: Callable[[], bool]
    set_dark_mode: Callable[[bool], None]
    dark_mode: Callable[[], bool]
    dark_mode_action: Callable[[], object | None]
    apply_theme: Callable[[bool], None]
    profile_manager: Callable[[], object]
    apply_profile: Callable[[object], None]
    set_profile_indicator_text: Callable[[str], None]
    set_status_text: Callable[[str], None]
    update_profiles_menu: Callable[[], None]
    profiles_menu_available: Callable[[], bool]
    load_custom_shortcuts: Callable[[dict], None]
    recent_menu: Callable[[], object | None]
    set_recent_menu: Callable[[object | None], None]
    menu_bar: Callable[[], object]
    load_recent_project: Callable[[str], None]


class ConfigController(QtCore.QObject):
    """
    Global configuration / settings controller.

    Consolidates QSettings persistence and settings-driven UI application behind
    an explicit injected UI port instead of a raw MainWindow back-reference.
    """

    def __init__(self, ui: ConfigControllerPort):
        super().__init__(ui.parent)
        self._ui = ui
        # User preference snapshot — not overwritten by document loads.
        self._user_preferences: dict[str, object] = {}

    def update_user_preference(self, key: str, value: object) -> None:
        """Record a user-chosen value for a dual-persisted settings key."""
        from kindred.gui.project_schema import QSETTINGS_KEY_MAP
        if key not in QSETTINGS_KEY_MAP:
            raise ValueError(f"Unknown user preference key: {key!r}")
        if key in {"max_parallel_batch_workers", "batch_runtime_lane_budget"}:
            value = min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, int(value)),
            )
        self._user_preferences[key] = value

    def get_user_preference(self, key: str) -> object:
        """Return the stored user preference, falling back to factory default."""
        from kindred.gui.project_schema import PROJECT_DEFAULTS
        return self._user_preferences.get(key, PROJECT_DEFAULTS.get(key))

    @staticmethod
    def _read_int_setting(settings: QtCore.QSettings, key: str, default: int) -> int:
        raw_value = settings.value(key, default)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            logger.warning("Invalid integer setting %s=%r; using %s", key, raw_value, int(default))
            return int(default)

    @staticmethod
    def _discard_invalid_gui_state(settings: QtCore.QSettings, key: str) -> None:
        with suppress(RuntimeError, TypeError):
            settings.remove(key)
            settings.sync()

    def _read_gui_state_setting(self, settings: QtCore.QSettings, key: str) -> object | None:
        try:
            return settings.value(key)
        except (EOFError, TypeError, ValueError) as exc:
            logger.warning("Ignoring unreadable persisted GUI state %s: %s", key, exc)
            self._discard_invalid_gui_state(settings, key)
            return None

    def _restore_gui_state_setting(
        self,
        settings: QtCore.QSettings,
        key: str,
        restore_state: Callable[[object], object],
        *,
        state_name: str,
    ) -> bool:
        state = self._read_gui_state_setting(settings, key)
        if not state:
            return False
        try:
            restored = restore_state(state)
        except (EOFError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid persisted %s %s: %s", state_name, key, exc)
            self._discard_invalid_gui_state(settings, key)
            return False
        if restored is False:
            logger.warning("Ignoring invalid persisted %s %s: restore rejected stored state", state_name, key)
            self._discard_invalid_gui_state(settings, key)
            return False
        return True

    def load_settings(self) -> None:
        settings = self._settings()
        from kindred.core.runtime_defaults import PREVIEW_CACHE_CAP_DEFAULT, RESULT_CACHE_CAP_DEFAULT
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME
        from kindred.gui.project_schema import PROJECT_DEFAULTS

        restore_maximized = settings.value("window/is_maximized", False, type=bool)
        self._restore_gui_state_setting(
            settings,
            "window/geometry",
            self._ui.restore_geometry,
            state_name="window geometry",
        )

        splitter = self._ui.splitter()
        if splitter:
            self._restore_gui_state_setting(
                settings,
                "window/splitter_state",
                splitter.restoreState,
                state_name="workspace splitter state",
            )

        temperature = settings.value("simulation/temperature", 298.15, type=float)
        self._ui.set_temperature(temperature)

        sim_time_raw = settings.value("simulation/time", "10.0")
        if isinstance(sim_time_raw, (int, float)):
            sim_time_text = f"{float(sim_time_raw):g}"
        else:
            sim_time_text = str(sim_time_raw)
        self._ui.set_simulation_time_text(sim_time_text)

        num_points = settings.value("simulation/points", 100, type=int)
        self._ui.set_num_points(num_points)
        slider_preview_points = self._read_int_setting(settings, "simulation/slider_preview_points", 100)
        self._ui.set_slider_preview_points(max(50, min(20000, slider_preview_points)))
        slider_preview_solver = str(settings.value("simulation/slider_preview_solver", "BDF") or "BDF").strip() or "BDF"
        self._ui.set_slider_preview_solver(slider_preview_solver)
        explicit_solver_value = self._ui.explicit_startup_solver_name()
        explicit_rtol_value = self._ui.explicit_startup_rtol()
        explicit_atol_value = self._ui.explicit_startup_atol()
        if self._ui.has_explicit_startup_solver_override():
            solver_value = explicit_solver_value or str(DEFAULT_SOLVER_NAME)
        else:
            solver_value = settings.value(
                "simulation/solver",
                self._ui.initial_solver_name() or str(DEFAULT_SOLVER_NAME),
            )
        if self._ui.has_explicit_startup_rtol_override():
            rtol_value = explicit_rtol_value
        else:
            rtol_value = settings.value("simulation/rtol", self._ui.initial_rtol() or 1e-6)
        if self._ui.has_explicit_startup_atol_override():
            atol_value = explicit_atol_value
        else:
            atol_value = settings.value("simulation/atol", self._ui.initial_atol() or 1e-12)
        self._ui.apply_solver_runtime_state(
            solver=str(solver_value or DEFAULT_SOLVER_NAME),
            rtol=rtol_value,
            atol=atol_value,
        )

        self._ui.set_use_sparse_jacobian(
            settings.value(
                "simulation/use_sparse_jacobian",
                bool(PROJECT_DEFAULTS["use_sparse_jacobian"]),
                type=bool,
            )
        )
        self._ui.set_wegscheider_cyclicity_enabled(
            settings.value(
                "simulation/wegscheider_cyclicity_enabled",
                bool(PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"]),
                type=bool,
            )
        )
        self._ui.set_max_parallel_batch_workers(
            min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(
                    1,
                    self._read_int_setting(
                        settings,
                        "simulation/max_parallel_batch_workers",
                        int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
                    ),
                ),
            )
        )
        self._ui.set_batch_runtime_lane_budget(
            min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(
                    1,
                    self._read_int_setting(
                        settings,
                        "simulation/batch_runtime_lane_budget",
                        int(PROJECT_DEFAULTS["batch_runtime_lane_budget"]),
                    ),
                ),
            )
        )

        self._ui.set_limit_blas_threads_per_worker(
            settings.value(
                "simulation/limit_blas_threads_per_worker",
                bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
                type=bool,
            )
        )

        default_result_cap = int(RESULT_CACHE_CAP_DEFAULT)
        default_preview_cap = int(PREVIEW_CACHE_CAP_DEFAULT)
        result_cap = self._read_int_setting(settings, "simulation/result_cache_cap", default_result_cap)
        preview_cap = self._read_int_setting(settings, "simulation/preview_cache_cap", default_preview_cap)
        self._ui.set_cache_caps(result_cap=result_cap, preview_cap=preview_cap, persist=False)

        if self._restore_gui_state_setting(
            settings,
            "window/dock_state",
            self._ui.restore_window_state,
            state_name="window dock state",
        ):
            self._ui.schedule_restored_floating_dock_recovery()

        if restore_maximized:
            self._ui.restore_maximized_state()

        ribbon_collapsed = settings.value("ui/ribbon_collapsed", False, type=bool)
        self._ui.set_ribbon_collapsed(ribbon_collapsed)

        profiles_available = self._ui.profiles_menu_available()
        if not profiles_available:
            # Profiles menu is hidden — clear any stranded key so the hidden
            # feature does not silently override user settings each launch.
            stored = settings.value("profiles/active", "", type=str)
            if stored:
                logger.info(
                    "Clearing stranded profiles/active='%s' — "
                    "profiles menu is not available",
                    stored,
                )
                settings.remove("profiles/active")
                settings.sync()

        profile_to_apply = None
        active_profile = None
        stored_profile = ""
        if profiles_available:
            profile_manager = self._ui.profile_manager()
            active_profile = profile_manager.get_active_profile()
            if active_profile is None:
                stored_profile = settings.value("profiles/active", "", type=str)
                if stored_profile:
                    profile = profile_manager.get_profile(stored_profile)
                    if profile:
                        profile_manager.set_active_profile(stored_profile)
                        profile_to_apply = profile

        dark_mode = settings.value("ui/dark_mode", True, type=bool)
        if active_profile is None and profile_to_apply is None:
            self._ui.set_dark_mode(dark_mode)
            dark_mode_action = self._ui.dark_mode_action()
            if dark_mode_action is not None:
                dark_mode_action.setChecked(dark_mode)
            self._ui.apply_theme(dark_mode)

        if profile_to_apply is not None:
            self._ui.apply_profile(profile_to_apply)
            if (
                self._ui.has_explicit_startup_solver_override()
                or self._ui.has_explicit_startup_rtol_override()
                or self._ui.has_explicit_startup_atol_override()
            ):
                self._ui.apply_solver_runtime_state(
                    solver=(
                        explicit_solver_value
                        if self._ui.has_explicit_startup_solver_override()
                        else (self._ui.initial_solver_name() or str(DEFAULT_SOLVER_NAME))
                    ),
                    rtol=(
                        explicit_rtol_value
                        if self._ui.has_explicit_startup_rtol_override()
                        else (self._ui.initial_rtol() or 1e-6)
                    ),
                    atol=(
                        explicit_atol_value
                        if self._ui.has_explicit_startup_atol_override()
                        else (self._ui.initial_atol() or 1e-12)
                    ),
                )
            self._ui.set_profile_indicator_text(f"Profile: {stored_profile}")
            self._ui.set_status_text(f"Applied profile: {stored_profile}")
            self._ui.update_profiles_menu()

        saved_shortcuts = settings.value("keyboard/shortcuts", {})
        if saved_shortcuts:
            self._ui.load_custom_shortcuts(saved_shortcuts)

        # Snapshot resolved startup values as the user preference baseline.
        # Reads from QSettings, then patches in any CLI/profile overrides that
        # were applied to live state above but not written to QSettings.
        from kindred.gui.project_schema import get_user_preference_payload
        self._user_preferences = get_user_preference_payload(settings)
        if self._ui.has_explicit_startup_solver_override():
            self._user_preferences["solver"] = str(solver_value or DEFAULT_SOLVER_NAME)
        if self._ui.has_explicit_startup_rtol_override():
            self._user_preferences["rtol"] = rtol_value
        if self._ui.has_explicit_startup_atol_override():
            self._user_preferences["atol"] = atol_value

        logger.debug("User settings loaded from QSettings (dark_mode=%s)", self._ui.dark_mode())

    def save_settings(self) -> None:
        from kindred.gui.project_schema import PROJECT_DEFAULTS, QSETTINGS_KEY_MAP

        settings = self._settings()

        settings.setValue("window/is_maximized", self._ui.is_maximized())
        settings.setValue("window/geometry", self._ui.save_geometry())
        settings.setValue("window/dock_state", self._ui.save_window_state())

        splitter = self._ui.splitter()
        if splitter:
            settings.setValue("window/splitter_state", splitter.saveState())

        # Dual-persisted keys — write user preferences, not live document state.
        for key, qs_key in QSETTINGS_KEY_MAP.items():
            value = self._user_preferences.get(key, PROJECT_DEFAULTS.get(key))
            if value is not None:
                settings.setValue(qs_key, value)

        # Application-level keys — always read from live UI state.
        settings.setValue("simulation/slider_preview_points", self._ui.slider_preview_points())
        settings.setValue("simulation/slider_preview_solver", self._ui.slider_preview_solver())
        settings.setValue(
            "simulation/result_cache_cap",
            self._ui.result_cache_cap(),
        )
        settings.setValue(
            "simulation/preview_cache_cap",
            self._ui.preview_cache_cap(),
        )

        settings.setValue("ui/ribbon_collapsed", self._ui.ribbon_collapsed())

        active_profile = self._ui.profile_manager().get_active_profile()
        if active_profile:
            settings.setValue("profiles/active", active_profile.name)
        else:
            settings.remove("profiles/active")

        settings.sync()
        logger.debug("User settings saved to QSettings")

    def persist_keyboard_shortcuts(self, shortcuts_dict: dict) -> None:
        self._settings().setValue("keyboard/shortcuts", shortcuts_dict)

    def toggle_theme(self) -> None:
        action = self._ui.dark_mode_action()
        if action is None:
            return
        dark_mode = bool(action.isChecked())
        self._ui.set_dark_mode(dark_mode)
        self._ui.apply_theme(dark_mode)
        self._settings().setValue("ui/dark_mode", dark_mode)

    def _settings(self) -> QtCore.QSettings:
        settings = self._ui.settings()
        if not isinstance(settings, QtCore.QSettings):
            raise TypeError(f"ConfigControllerPort.settings() must return QSettings, got {type(settings).__name__}")
        return settings

    def get_recent_menu(self, *, force_recreate: bool = False) -> Optional[QtWidgets.QMenu]:
        """
        Return a live Recent Projects submenu, recreating it when Qt deleted it.

        Qt can invalidate Python wrappers for menu objects during lifecycle churn
        (platform integration, deferred deletes, or menu rebuilds). This method
        always returns a currently valid menu or ``None`` when File menu lookup
        fails.
        """
        recent_menu = self._ui.recent_menu()
        if not force_recreate and recent_menu is not None and isValid(recent_menu):
            return recent_menu

        for _ in range(2):
            menubar = self._ui.menu_bar()
            file_menu = None
            for candidate in menubar.findChildren(QtWidgets.QMenu):
                if candidate is None or not isValid(candidate):
                    continue
                if candidate.title().replace("&", "") == "File":
                    file_menu = candidate
                    break
            if file_menu is None:
                for action in menubar.actions():
                    candidate = action.menu()
                    if candidate is None or not isValid(candidate):
                        continue
                    if candidate.title().replace("&", "") == "File":
                        file_menu = candidate
                        break

            if file_menu is None:
                logger.warning("Recent Projects menu unavailable: File menu is missing or invalid")
                return None

            try:
                recent_actions = [action for action in list(file_menu.actions()) if action.text() == "Recent Projects"]
                if not force_recreate:
                    valid_recent_actions: List[Tuple[QtGui.QAction, QtWidgets.QMenu]] = []
                    for action in recent_actions:
                        action_menu = action.menu()
                        if action_menu is not None and isValid(action_menu):
                            valid_recent_actions.append((action, action_menu))

                    if valid_recent_actions:
                        keep_action, keep_menu = valid_recent_actions[0]
                        for action in recent_actions:
                            if action is keep_action:
                                continue
                            file_menu.removeAction(action)
                        self._ui.set_recent_menu(keep_menu)
                        return keep_menu

                for stale_action in recent_actions:
                    file_menu.removeAction(stale_action)

                recent_menu = QtWidgets.QMenu("Recent Projects", file_menu)
                insert_before = next((action for action in file_menu.actions() if action.isSeparator()), None)
                if insert_before is not None:
                    file_menu.insertMenu(insert_before, recent_menu)
                else:
                    file_menu.addMenu(recent_menu)
                self._ui.set_recent_menu(recent_menu)
                if isValid(recent_menu):
                    return recent_menu
            except RuntimeError:
                continue

        logger.warning("Recent Projects menu unavailable: failed to acquire a live File menu")
        return None

    def update_recent_files_menu(self) -> None:
        def _populate(menu: QtWidgets.QMenu) -> None:
            menu.clear()

            recent_files = self._settings().value("recent_files", [])
            if not isinstance(recent_files, list):
                recent_files = []

            if not recent_files:
                no_files_action = menu.addAction("No recent projects")
                no_files_action.setEnabled(False)
                no_files_action.setToolTip(
                    "No projects have been opened recently. Load a project to populate this list."
                )
                return

            for filepath in recent_files[:10]:
                action = menu.addAction(
                    os.path.basename(filepath),
                    lambda p=filepath: self._ui.load_recent_project(p),
                )
                action.setToolTip(filepath)

            menu.addSeparator()
            clear_action = menu.addAction("Clear Recent Projects")
            clear_action.triggered.connect(self.clear_recent_files)

        last_error: Optional[BaseException] = None
        for attempt in range(2):
            recent_menu = self.get_recent_menu(force_recreate=bool(attempt))
            if recent_menu is None or not isValid(recent_menu):
                continue

            try:
                _populate(recent_menu)
                return
            except RuntimeError as exc:
                last_error = exc

        if last_error is not None:
            logger.warning("Recent Projects menu update skipped: %s", last_error)
        else:
            logger.warning("Recent Projects menu update skipped: unable to acquire a valid menu")

    def add_to_recent_files(self, filepath: str) -> None:
        recent_files = self._settings().value("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []

        if filepath in recent_files:
            recent_files.remove(filepath)

        recent_files.insert(0, filepath)
        recent_files = recent_files[:10]
        self._settings().setValue("recent_files", recent_files)
        self.update_recent_files_menu()

    def clear_recent_files(self) -> None:
        self._settings().setValue("recent_files", [])
        self.update_recent_files_menu()
