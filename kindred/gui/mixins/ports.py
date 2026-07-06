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
    wegscheider_cyclicity_enabled_getter: Callable[[], bool]
