from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FittingMixinPorts:
    mechanism_editor: Any
    dataset_registry: Any
    dataset_fit_settings_store: Any
    dataset_view_publisher: Any
    mechanism_parameter_scan_owner: Any
    status_setter: Callable[[str], None]
    temperature_getter: Callable[[], float]
    num_points_getter: Callable[[], int]


@dataclass(frozen=True)
class ProfileMixinPorts:
    profile_manager: Any
    profiles_menu_getter: Callable[[], Any]
    profile_indicator_setter: Callable[[str], None]
    status_setter: Callable[[str], None]
    settings_set_value: Callable[[str, object], None]
    settings_remove: Callable[[str], None]
    num_points_spinbox: Any
    dark_mode_action: Any
    toggle_theme: Callable[[], None]
    update_solver_summary_label: Callable[[], None]
