# kindred/gui/mixins/profile_mixin.py
"""ProfileMixin - Profile menu handling and application logic."""

from __future__ import annotations

import logging

import shiboken6
from PySide6 import QtWidgets

from kindred.config.profiles import Profile
from kindred.gui.mixins.ports import ProfileMixinPorts

logger = logging.getLogger(__name__)


class ProfileMixin:
    """
    Mixin providing profile menu management for MainWindow.

    MainWindow provides an explicit `ProfileMixinPorts` object at `self._profile_ports`.
    """

    def _require_profile_ports(self) -> ProfileMixinPorts:
        ports = getattr(self, "_profile_ports", None)
        if isinstance(ports, ProfileMixinPorts):
            return ports

        profile_manager = getattr(self, "_profile_manager", None)
        profile_indicator = getattr(self, "_profile_indicator", None)
        status_label = getattr(self, "_status_label", None)
        num_points_spinbox = getattr(self, "_num_points_spinbox", None)
        dark_mode_action = getattr(self, "_dark_mode_action", None)
        if profile_manager is None or profile_indicator is None or status_label is None:
            raise RuntimeError("ProfileMixin ports are not initialized.")

        return ProfileMixinPorts(
            profile_manager=profile_manager,
            profiles_menu_getter=lambda: getattr(self, "_profiles_menu", None),
            profile_indicator_setter=lambda text: profile_indicator.setText(str(text)),
            status_setter=lambda text: status_label.setText(str(text)),
            settings_set_value=self.settings_set_value,
            settings_remove=self.settings_remove,
            num_points_spinbox=num_points_spinbox,
            dark_mode_action=dark_mode_action,
            toggle_theme=self._toggle_theme,
            update_solver_summary_label=self._update_solver_summary_label,
        )

    def _update_profiles_menu(self) -> None:
        """Update Profiles menu with available profiles."""
        ports = self._require_profile_ports()
        profiles_menu = ports.profiles_menu_getter()
        if profiles_menu is None:
            return
        if not shiboken6.isValid(profiles_menu):
            return

        profiles_menu.clear()

        # Get list of available profiles
        profile_names = ports.profile_manager.list_profiles()

        if not profile_names:
            no_profiles_action = profiles_menu.addAction("No profiles available")
            no_profiles_action.setEnabled(False)
            no_profiles_action.setToolTip("No solver/grid profiles found. Create profiles in kindred/data/presets/ to use this feature.")
            return

        # Add profile actions
        for profile_name in profile_names:
            action = profiles_menu.addAction(profile_name)
            action.setCheckable(True)

            # Check if this is the active profile
            active = ports.profile_manager.get_active_profile()
            if active and active.name == profile_name:
                action.setChecked(True)

            # Connect to slot
            action.triggered.connect(lambda checked, name=profile_name: self._activate_profile(name))

        # Add separator and Clear action
        profiles_menu.addSeparator()
        clear_action = profiles_menu.addAction("Clear Active Profile")
        clear_action.triggered.connect(self._clear_profile)

    def _activate_profile(self, profile_name: str) -> None:
        """
        Activate a profile and apply its settings.

        Parameters
        ----------
        profile_name : str
            Name of profile to activate
        """
        logger.info(f"Activating profile: {profile_name}")

        # Get the profile
        ports = self._require_profile_ports()
        profile = ports.profile_manager.get_profile(profile_name)
        if profile is None:
            QtWidgets.QMessageBox.warning(
                self, "Profile Error", f"Profile '{profile_name}' not found"
            )
            return

        # Set as active
        ports.profile_manager.set_active_profile(profile_name)

        # Apply profile settings
        self._apply_profile(profile)

        # Update menu
        self._update_profiles_menu()

        # Update status bar
        ports.profile_indicator_setter(f"Profile: {profile_name}")
        ports.status_setter(f"Applied profile: {profile_name}")

        logger.info(f"Profile '{profile_name}' activated and applied")

    def _clear_profile(self) -> None:
        """Clear active profile and return to default settings."""
        logger.info("Clearing active profile")

        ports = self._require_profile_ports()
        ports.profile_manager.clear_active_profile()

        # Reset runtime defaults and widgets
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

        ports.num_points_spinbox.setValue(100)
        self.apply_solver_runtime_state(
            solver=str(DEFAULT_SOLVER_NAME),
            rtol=1e-6,
            atol=1e-12,
        )

        if ports.dark_mode_action.isChecked():
            ports.dark_mode_action.blockSignals(True)
            ports.dark_mode_action.setChecked(False)
            ports.dark_mode_action.blockSignals(False)
            ports.toggle_theme()

        ports.settings_remove("profiles/active")

        # Update menu
        self._update_profiles_menu()

        # Update status bar
        ports.profile_indicator_setter("Profile: None")
        ports.status_setter("Profile cleared - using default settings")

        logger.info("Active profile cleared")

    def _apply_profile(self, profile: Profile) -> None:
        """
        Apply profile settings to GUI and simulation.

        Parameters
        ----------
        profile : Profile
            Profile to apply
        """
        logger.info(f"Applying profile settings: {profile.name}")

        # Update solver defaults used by simulations
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

        # Update grid density widget
        ports = self._require_profile_ports()
        grid_value = int(profile.grid_n or ports.num_points_spinbox.value())
        grid_value = max(ports.num_points_spinbox.minimum(), min(grid_value, ports.num_points_spinbox.maximum()))
        ports.num_points_spinbox.setValue(grid_value)
        self.apply_solver_runtime_state(
            solver=profile.solver_method or str(DEFAULT_SOLVER_NAME),
            rtol=profile.rtol or 1e-6,
            atol=profile.atol or 1e-12,
        )

        # Update theme if profile requests it
        desired_dark = (profile.theme or "default").lower() == "dark"
        if desired_dark != ports.dark_mode_action.isChecked():
            ports.dark_mode_action.blockSignals(True)
            ports.dark_mode_action.setChecked(desired_dark)
            ports.dark_mode_action.blockSignals(False)
            ports.toggle_theme()

        # Persist active profile selection
        ports.settings_set_value("profiles/active", profile.name)

        logger.info(
            "Profile '%s' applied (solver=%s, rtol=%s, atol=%s, grid=%d, theme=%s)",
            profile.name,
            self._initial_solver,
            self._initial_rtol,
            self._initial_atol,
            grid_value,
            profile.theme,
        )
