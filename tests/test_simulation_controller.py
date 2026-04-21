from __future__ import annotations

from contextlib import suppress
from concurrent.futures import Future
from dataclasses import dataclass
from queue import SimpleQueue
import warnings
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.dsl_text_update import format_authoritative_parameter_value
from kindred.core.simulation_identity import SimulationIdentity, SimulationScopeIdentity
from kindred.gui.controllers.simulation_controller import (
    SimulationController,
    _default_batch_executor_factory,
)
from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers
from kindred.gui.simulation_worker import SimulationWorker
from kindred.gui.ports import SimulationUiPorts
from tests.worker_stubs import make_stubborn_worker


@dataclass
class _FakeButton:
    enabled: bool = True

    def isEnabled(self) -> bool:
        return bool(self.enabled)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


@dataclass
class _FakeLabel:
    text: str = ""

    def setText(self, text: str) -> None:
        self.text = str(text)

    def repaint(self) -> None:
        return


@dataclass
class _FakeProgress:
    value: int = 0

    def setValue(self, value: int) -> None:
        self.value = int(value)

    def repaint(self) -> None:
        return


@dataclass
class _FakeSpinBox:
    _value: float

    def value(self) -> float:
        return float(self._value)

    def setValue(self, value: float) -> None:
        self._value = float(value)


class _FakeSignal:
    def __init__(self, *, disconnect_raises_typeerror: bool = False) -> None:
        self._handlers: list[Callable[..., Any]] = []
        self._disconnect_raises_typeerror = bool(disconnect_raises_typeerror)

    def connect(self, handler: Callable[..., Any]) -> None:
        self._handlers.append(handler)

    def disconnect(self) -> None:
        if self._disconnect_raises_typeerror:
            raise TypeError("not connected")
        self._handlers.clear()


class _FakeWorker(QtCore.QObject):
    def __init__(
        self,
        *,
        running: bool = False,
        wait_returns: bool = True,
        signal_disconnect_typeerror: bool = False,
    ) -> None:
        super().__init__()
        self._running = bool(running)
        self._wait_returns = bool(wait_returns)
        self._cancelled = False
        self._terminated = False

        self.progress = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)
        self.finished = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)
        self.error = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)

    def isRunning(self) -> bool:  # Qt-ish API
        return bool(self._running)

    def cancel(self) -> None:
        self._cancelled = True
        self._running = False

    def wait(self, _ms: Optional[int] = None) -> bool:
        return bool(self._wait_returns)

    def terminate(self) -> None:
        self._terminated = True
        self._running = False

    def start(self) -> None:
        self._running = True


class _QtSignalWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, str)
    result_ready = QtCore.Signal(dict)
    error = QtCore.Signal(object)

    def __init__(self, *, running: bool = True) -> None:
        super().__init__()
        self._running = bool(running)
        self.cancel_calls: list[None] = []
        self.wait_calls: list[int] = []
        self.progress.connect(lambda *_args: None)
        self.result_ready.connect(lambda *_args: None)

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_calls.append(None)

    def wait(self, ms: Optional[int] = None) -> bool:
        self.wait_calls.append(int(ms or 0))
        return False


def _successful_result_payload() -> dict[str, Any]:
    return {
        "t": np.linspace(0.0, 1.0, 3),
        "Y": np.asarray([[1.0, 0.5, 0.1], [0.0, 0.5, 0.9]], dtype=float),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": "reaction: A -> B ; k=0.1",
        "solver_config": {"solver": "Radau", "temperature_K": 298.15},
        "algebra_scalars": {},
        "algebra_errors": [],
        "fallback_occurred": False,
        "fallback_message": None,
    }


class _FakeMainWindow(QtCore.QObject):
    def settings_set_value(self, key: str, value: object) -> None:
        self._settings.setValue(str(key), value)

    def settings_sync(self) -> None:
        self._settings.sync()

    def run_button_is_enabled(self) -> bool:
        return bool(self._run_btn.isEnabled())

    def set_run_button_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(bool(enabled))

    def set_stop_button_enabled(self, enabled: bool) -> None:
        self._stop_btn.setEnabled(bool(enabled))

    def set_status_text(self, text: str) -> None:
        self._status_label.setText(str(text))

    def set_sim_progress_value(self, value: int) -> None:
        self._sim_progress.setValue(int(value))

    def repaint_simulation_widgets(self) -> None:
        self._sim_progress.repaint()
        self._status_label.repaint()

    def set_algebra_status_text(self, text: str) -> None:
        self._algebra_status_label.setText(str(text))

    def message_box_warning(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(None, str(title), str(message))

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None:
        full_message = str(message)
        if details:
            full_message = f"{full_message}\n\nDetails:\n{details}"
        QtWidgets.QMessageBox.critical(None, str(title), full_message)

    def mechanism_reactions_text_raw(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is not None and hasattr(editor, "_reactions_text"):
            return str(editor._reactions_text.toPlainText())
        return str(self._get_mechanism_text() or "")

    def mechanism_state_network_dsl_raw(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        state_editor = getattr(editor, "_state_network_editor", None)
        if state_editor is not None and hasattr(state_editor, "get_state_network_dsl"):
            return str(state_editor.get_state_network_dsl() or "")
        return ""

    def mechanism_slider_points_value(self) -> Optional[int]:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "slider_points_value"):
            return None
        try:
            return int(editor.slider_points_value())
        except Exception:
            return None

    def mechanism_slider_solver_value(self) -> Optional[str]:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "slider_solver_value"):
            return None
        try:
            value = editor.slider_solver_value()
        except Exception:
            return None
        return str(value) if value is not None else None

    def auto_lock_for_run(self) -> bool:
        self._auto_lock_for_run_calls = int(getattr(self, "_auto_lock_for_run_calls", 0)) + 1
        return bool(getattr(self, "_auto_lock_for_run_result", True))

    def is_mechanism_ready_for_run(self) -> bool:
        self._is_mechanism_ready_for_run_calls = int(getattr(self, "_is_mechanism_ready_for_run_calls", 0)) + 1
        return bool(getattr(self, "_is_mechanism_ready_for_run_result", True))

    def temperature_spinbox_value(self) -> float:
        return float(self._temperature_spinbox.value())

    def num_points_spinbox_value(self) -> int:
        return int(self._num_points_spinbox.value())

    def sim_time_spinbox_text(self) -> str:
        return str(self._sim_time_spinbox.text())

    def parse_sim_time_seconds(self) -> float:
        return float(self._parse_sim_time_seconds())

    def use_sparse_jacobian(self) -> bool:
        return bool(self._use_sparse_jacobian)

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_cyclicity_enabled)

    def main_plot(self) -> object:
        return self._plot_tabs._main_plot

    def set_results_table(self, table: object) -> None:
        self._results_table = table

    def sync_main_plot_copy_labels(self, primary_set_id: str, selected_set_ids) -> None:
        pass

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._temperature_mode_indicator.setText(str(text))

    def set_mechanism_reactions_text_with_optional_undo(
        self,
        new_text: str,
        description: str,
        *,
        record_undo: bool,
    ) -> None:
        self._set_text_with_optional_undo(
            self._mechanism_editor._reactions_text,
            str(new_text),
            str(description),
            bool(record_undo),
        )

    def stop_slider_release_commit_timer(self) -> None:
        timer = self._slider_release_commit_timer
        if timer is not None and timer.isActive():
            timer.stop()

    def has_pending_slider_values(self) -> bool:
        return bool(self._pending_slider_values)

    def finalize_slider_release_commit(self) -> None:
        self._finalize_slider_release_commit()

    def stop_variable_update_timer(self) -> None:
        timer = self._variable_update_timer
        if timer is not None:
            timer.stop()

    def stop_species_slider_update_timer(self) -> None:
        timer = getattr(self, "_species_slider_update_timer", None)
        if timer is not None:
            timer.stop()

    def set_slider_triggered_simulation(self, value: bool) -> None:
        self._slider_triggered_simulation = bool(value)

    def slider_triggered_simulation(self) -> bool:
        return bool(self._slider_triggered_simulation)

    def last_slider_change_name(self) -> str:
        return str(self._last_slider_change_name or "")

    def slider_drag_active(self) -> bool:
        return bool(self._slider_drag_active)

    def suppress_slider_refresh(self) -> bool:
        return bool(self._suppress_slider_refresh)

    def slider_gesture_target_set_ids_snapshot(self) -> list[str]:
        return [str(set_id) for set_id in getattr(self, "_slider_gesture_target_set_ids_snapshot", [])]

    def preview_initials_for_row(self, row: int, baseline: dict[str, float]) -> dict[str, float]:
        _ = row
        return {str(key): float(value) for key, value in dict(baseline or {}).items()}

    def preview_batch_cache_token(self, rows: list[int]) -> str:
        _ = rows
        return ""

    def _remember_last_mechanism(self, mechanism: object, dsl_text: str, solver_config: dict[str, Any]) -> None:
        self._mechanism_helpers.remember_last_mechanism(mechanism, dsl_text, solver_config)

    def _clear_last_mechanism(self) -> None:
        self._mechanism_helpers.clear_last_mechanism()

    def last_mechanism(self) -> Optional[object]:
        return self._mechanism_helpers.last_mechanism()

    def last_mechanism_context(self) -> dict[str, Any]:
        return self._mechanism_helpers.last_mechanism_context()

    def batch_rows_for_scope(self, scope: str) -> list[int]:
        return [int(row) for row in (self._batch_rows_for_scope(str(scope)) or [])]

    def batch_set_ids_for_scope(self, scope: str) -> list[str]:
        return [str(set_id) for set_id in (self._batch_set_ids_for_scope(str(scope)) or [])]

    def shown_batch_set_ids(self) -> list[str]:
        return [str(set_id) for set_id in (self._shown_batch_set_ids() or [])]

    def batch_current_row(self) -> Optional[int]:
        row = self._batch_current_row()
        return int(row) if row is not None else None

    def batch_set_id_for_row(self, row: int) -> Optional[str]:
        value = self._batch_set_id_for_row(int(row))
        return str(value) if value is not None else None

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]:
        value = self._batch_set_name_for_id(str(set_id))
        return str(value) if value is not None else None

    def batch_set_id_for_name(self, name: str) -> Optional[str]:
        value = self._batch_set_id_for_name(str(name))
        return str(value) if value is not None else None

    def batch_preferred_primary_set_id(self, rows: list[int]) -> Optional[str]:
        value = self._batch_preferred_primary_set_id(list(rows))
        return str(value) if value is not None else None

    def set_active_batch_selection(self, set_id: str, set_name: str, selected_ids: list[str]) -> None:
        _ = (set_id, set_name, selected_ids)

    def clear_display_selection_state(self) -> None:
        return None

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: dict[str, Any] | None = None,
        t_end: float = 0.0,
    ) -> str:
        if scope_identity is not None and hasattr(scope_identity, "cache_key"):
            return str(scope_identity.cache_key())
        return str(
            self._batch_cache_key(
                mechanism_text=str(mechanism_text),
                solver_config=dict(solver_config or {}),
                t_end=float(t_end),
            )
        )

    def batch_store_row_count(self) -> int:
        return int(self._batch_store.row_count())

    def batch_store_set_names(self) -> list[str]:
        return [str(name) for name in (self._batch_store.set_names() or [])]

    def batch_store_visible_species(self) -> list[str]:
        return [str(species) for species in (self._batch_store.visible_species() or [])]

    def batch_model_validate_rows(self, rows: list[int]) -> set[tuple[int, str]]:
        invalid = self._batch_model.validate_rows(list(rows))
        if not invalid:
            return set()
        return {(int(row), str(species)) for row, species in invalid}

    def batch_initials_for_row(self, row: int) -> dict[str, float]:
        initials = self._batch_initials_for_row(int(row))
        if not initials:
            return {}
        return {str(key): float(value) for key, value in dict(initials).items()}

    def display_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: list[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[object] = None,
        valid_set_ids: Optional[list[str] | tuple[str, ...]] = None,
        allow_fallback: bool = True,
    ) -> bool:
        return bool(
            self._display_cached_batch_selection(
                cache_key=str(cache_key),
                selected_sets=list(selected_sets),
                prefer_set=str(prefer_set) if prefer_set is not None else None,
                cache_store=cache_store,
                valid_set_ids=(
                    tuple(str(set_id) for set_id in valid_set_ids)
                    if valid_set_ids is not None
                    else None
                ),
                allow_fallback=bool(allow_fallback),
            )
        )

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(self, species_names: list[str], *, preserve_active_cache: bool = False) -> None:
        self._sync_batch_species_columns(list(species_names), preserve_active_cache=bool(preserve_active_cache))

    def has_slider_overrides(self) -> bool:
        return bool(self._slider_overrides)

    def simulation_schema_id(self) -> str:
        return str(getattr(self, "_simulation_schema_id", "schema-default"))

    def simulation_param_fingerprint(self, set_id: Optional[str] = None) -> str:
        mapping = getattr(self, "_simulation_param_fingerprints", None)
        if isinstance(mapping, dict):
            set_id_s = str(set_id or "")
            if set_id_s in mapping:
                return str(mapping[set_id_s])
            if "" in mapping:
                return str(mapping[""])
        return str(getattr(self, "_simulation_param_fingerprint", "params-default"))

    def slider_overrides(self, set_id: Optional[str] = None) -> dict[str, float]:
        _ = set_id
        overrides: dict[str, float] = {}
        for key, value in (self._slider_overrides or {}).items():
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(parsed):
                continue
            overrides[str(key)] = float(parsed)
        return overrides

    def apply_overrides_to_text(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        _ = set_id
        return str(self._apply_overrides_to_text(str(base_text)))

    def apply_overrides_to_state_network_dsl(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        _ = set_id
        return str(self._apply_overrides_to_state_network_dsl(str(base_text)))

    def apply_parameter_overrides_to_dsl(self, mechanism_text: str, parameters: dict[str, float]) -> str:
        return str(self._apply_parameter_overrides_to_dsl(str(mechanism_text), dict(parameters)))

    def get_mechanism_text(self) -> str:
        return str(self._get_mechanism_text() or "")

    def initial_solver_name(self) -> Optional[str]:
        solver = getattr(self, "_initial_solver", None)
        return str(solver) if solver is not None else None

    def initial_rtol(self) -> Optional[float]:
        value = getattr(self, "_initial_rtol", None)
        return float(value) if value is not None else None

    def initial_atol(self) -> Optional[float]:
        value = getattr(self, "_initial_atol", None)
        return float(value) if value is not None else None

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]:
        value = self._dsl_global_temperature_K(str(dsl_text))
        return float(value) if value is not None else None

    def prepare_slider_runtime(self, *, set_id: Optional[str] = None) -> Optional[object]:
        return self._prepare_slider_runtime(set_id=set_id)

    def apply_slider_overrides_to_bindings(self, runtime: object, *, set_id: Optional[str] = None) -> bool:
        return bool(self._apply_slider_overrides_to_bindings(runtime, set_id=set_id))

    def set_slider_runtime_dirty(self, value: bool) -> None:
        self._slider_runtime_dirty = bool(value)

    def snapshot_datasets(self) -> dict[str, Any]:
        return dict(self._snapshot_datasets() or {})

    def last_fit_metadata(self) -> Optional[dict[str, Any]]:
        value = self._last_fit_metadata
        return dict(value) if isinstance(value, dict) else None

    def set_last_simulation_provenance(self, provenance: dict[str, Any]) -> None:
        self._last_simulation_provenance = dict(provenance)

    def set_last_simulation_ctc(self, ctc: dict[str, float]) -> None:
        self._last_simulation_ctc = {str(key): float(value) for key, value in (ctc or {}).items()}

    def integrate_ctc(
        self,
        t: object,
        y: object,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> tuple[float, str, bool, float, str]:
        _ = (t, y, uniformity_eps, tail_strategy)
        return (1.0, "trapz", True, float(uniformity_eps), str(tail_strategy))

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: dict[str, Any]) -> None:
        self._remember_last_mechanism(mechanism, str(mechanism_text), dict(solver_config))

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        return bool(self._is_energy_mode_mechanism(mechanism))

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool:
        return bool(self._dsl_has_computational_mode_generated_block(str(mechanism_text)))

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        self._sync_energy_mode_temperature_from_mechanism(mechanism)

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self._populate_energy_mode_variables_from_mechanism(
            mechanism,
            refresh_sliders=bool(refresh_sliders),
            preserve_visibility=bool(preserve_visibility),
        )

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None:
        self._extract_and_populate_variables(preserve_visibility=bool(preserve_visibility))

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        self._sync_mechanism_controls_to_focused_batch_set(use_workspace=bool(use_workspace))

    def apply_pending_init_migration(self, *, seed_sets: dict[str, dict[str, float]], rewrite: str) -> bool:
        if not seed_sets or not rewrite:
            return False
        for set_name, seed in dict(seed_sets).items():
            row_idx = self._batch_store.ensure_set(str(set_name))
            for species, value in dict(seed).items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not np.isfinite(parsed):
                    continue
                self._batch_store.set_value(row_idx, str(species), f"{float(parsed):.6g}")
        self.set_mechanism_reactions_text_with_optional_undo(
            str(rewrite),
            "Migrate initial concentrations to batch table",
            record_undo=True,
        )
        return True

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        self._invalidate_pending_init_preserved_results_after_failed_run()

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None:
        self._arm_pending_init_result_invalidation_guard(rewrite=rewrite)


@pytest.fixture
def mw(qt_app) -> _FakeMainWindow:
    _ = qt_app
    window = _FakeMainWindow()
    window._settings = MagicMock()

    window._run_btn = _FakeButton(True)
    window._stop_btn = _FakeButton(False)
    window._status_label = _FakeLabel()
    window._algebra_status_label = _FakeLabel()
    window._sim_progress = _FakeProgress()
    window._temperature_spinbox = _FakeSpinBox(298.15)
    window._num_points_spinbox = _FakeSpinBox(100.0)
    window._initial_solver = "BDF"
    window._initial_rtol = 1e-6
    window._initial_atol = 1e-12
    window._last_fit_metadata = None
    window._last_simulation_provenance = {}
    window._last_simulation_ctc = {}

    window._slider_triggered_simulation = False
    window._pending_slider_values = {}
    window._slider_overrides = {}
    window._slider_drag_active = False
    window._slider_gesture_target_set_ids_snapshot = []
    window._last_slider_change_name = ""
    window._slider_runtime_dirty = False
    window._use_sparse_jacobian = False
    window._wegscheider_cyclicity_enabled = False

    window._batch_store = MagicMock()
    window._batch_store.row_count.return_value = 0
    window._batch_store.set_names.return_value = []
    window._batch_store.visible_species.return_value = []

    window._batch_set_ids_for_scope = MagicMock(return_value=[])
    window._shown_batch_set_ids = MagicMock(return_value=[])
    window._batch_rows_for_scope = MagicMock(return_value=[])
    window._batch_current_row = MagicMock(return_value=None)
    window._batch_set_id_for_row = MagicMock(return_value=None)
    window._batch_set_id_for_name = MagicMock(return_value=None)
    window._batch_set_name_for_id = MagicMock(return_value=None)
    window._batch_preferred_primary_set_id = MagicMock(return_value=None)
    window._batch_cache_key = MagicMock(return_value="cache-key")
    window._simulation_schema_id = "schema-default"
    window._simulation_param_fingerprints = {"": "params-default"}
    window._auto_lock_for_run_result = True
    window._auto_lock_for_run_calls = 0
    window._is_mechanism_ready_for_run_result = True
    window._is_mechanism_ready_for_run_calls = 0

    window._display_cached_batch_selection = MagicMock(return_value=False)
    window._flush_slider_plot_updates = MagicMock(return_value=False)
    window.set_data = MagicMock()
    window._plot_tabs = MagicMock()
    window._plot_tabs._main_plot = MagicMock()
    window._plot_tabs._main_plot.stats_table.return_value = MagicMock()
    window._results_table = MagicMock()
    window._results_table.viewport.return_value = MagicMock()

    window._parse_sim_time_seconds = MagicMock(return_value=10.0)
    window._dsl_global_temperature_K = MagicMock(return_value=None)
    window._sync_batch_species_columns = MagicMock()
    window._sim_time_spinbox = MagicMock()
    window._sim_time_spinbox.text.return_value = "10.0"
    window._snapshot_datasets = MagicMock(return_value={})

    window._prepare_slider_runtime = MagicMock(return_value=None)
    window._apply_slider_overrides_to_bindings = MagicMock(return_value=False)
    window._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    window._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text: str(text))
    window._apply_parameter_overrides_to_dsl = MagicMock(side_effect=lambda mechanism_text, parameters: str(mechanism_text))
    window.reset_mechanism_workspaces = MagicMock(return_value=False)
    window.discard_concentration_overlays_for_set_ids = MagicMock(return_value=False)
    window.discard_concentration_overlays_for_rows = MagicMock(return_value=False)
    window._dirty_state_generations = {}
    window.has_dirty_state_for_set = MagicMock(
        side_effect=lambda set_id: int(window._dirty_state_generations.get(str(set_id or ""), 1) or 0) > 0
    )
    window.dirty_state_generation = MagicMock(
        side_effect=lambda set_id: int(window._dirty_state_generations.get(str(set_id or ""), 1) or 0)
    )
    window._run_simulation_internal = MagicMock()

    window._get_mechanism_text = MagicMock(return_value="")
    window._is_energy_mode_mechanism = MagicMock(return_value=False)
    window._dsl_has_computational_mode_generated_block = MagicMock(return_value=False)
    window._sync_energy_mode_temperature_from_mechanism = MagicMock()
    window._set_temperature_override_state = MagicMock()
    window._update_temperature_mode_indicator = MagicMock()
    window._temperature_mode_indicator = _FakeLabel()

    window._remember_last_mechanism = MagicMock()
    window._populate_energy_mode_variables_from_mechanism = MagicMock()
    window._extract_and_populate_variables = MagicMock()
    window._sync_mechanism_controls_to_focused_batch_set = MagicMock()
    window._update_batch_row_controls_state = MagicMock()
    window._batch_model = MagicMock()
    window._batch_model.columnCount.return_value = 1
    window._batch_model.index.side_effect = lambda *_args, **_kwargs: object()
    window._batch_model.dataChanged = MagicMock()
    window._batch_model.dataChanged.emit = MagicMock()
    window._batch_model.validate_rows = MagicMock(return_value=set())

    window._set_text_with_optional_undo = MagicMock()
    window._invalidate_pending_init_preserved_results_after_failed_run = MagicMock()
    window._arm_pending_init_result_invalidation_guard = MagicMock()
    window._suppress_slider_runtime_invalidation = False
    window._suppress_slider_refresh = False

    window._variable_update_timer = MagicMock()
    window._variable_update_timer.isActive.return_value = False
    window._species_slider_update_timer = MagicMock()
    window._species_slider_update_timer.isActive.return_value = False
    window._slider_release_commit_timer = MagicMock()
    window._slider_release_commit_timer.isActive.return_value = False
    window._finalize_slider_release_commit = MagicMock()

    window._batch_initials_for_row = MagicMock(return_value={})
    window._variable_runtime = window
    window._mechanism_helpers = MainWindowMechanismHelpers(window)

    return window


@pytest.fixture
def controller(mw: _FakeMainWindow) -> SimulationController:
    ui = SimulationUiPorts(
        dialogs=mw,
        settings=mw,
        run_ui=mw,
        slider=mw,
        batch=mw,
        mechanism=mw,
        solver=mw,
        runtime=mw,
        results=mw,
        provenance=mw,
        mechanism_helpers=mw._mechanism_helpers,
    )
    c = SimulationController(ui, parent=mw)
    try:
        yield c
    finally:
        timer = getattr(c, "_slider_plot_coalesce_timer", None)
        with suppress(RuntimeError, TypeError):
            if timer is not None and timer.isActive():
                timer.stop()
        timer = getattr(c, "_batch_future_poll_timer", None)
        with suppress(RuntimeError, TypeError):
            if timer is not None and timer.isActive():
                timer.stop()


@pytest.mark.unit
def test_default_batch_executor_factory_is_injectable(monkeypatch):
    created = {}

    def _fake_get_context(name: str):
        created["context_name"] = name
        return f"ctx:{name}"

    class _FakeExecutor:
        def __init__(self, **kwargs):
            created["kwargs"] = dict(kwargs)

    monkeypatch.setattr("multiprocessing.get_context", _fake_get_context)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _FakeExecutor)

    _default_batch_executor_factory(3, True)
    assert created["context_name"] == "spawn"
    assert created["kwargs"]["max_workers"] == 3
    assert created["kwargs"]["mp_context"] == "ctx:spawn"


@pytest.mark.unit
def test_set_simulation_cache_caps_clamps_and_persists(mw: _FakeMainWindow, controller: SimulationController):
    result = controller.set_simulation_cache_caps(result_cap=-5, preview_cap="7", persist=True)
    assert result.ok is True
    assert result.operation == "set_caps"
    assert controller.batch_cache.result_cache.max_entries() == 0
    assert controller.batch_cache.preview_cache.max_entries() == 7
    mw._settings.setValue.assert_any_call("simulation/result_cache_cap", 0)
    mw._settings.setValue.assert_any_call("simulation/preview_cache_cap", 7)


@pytest.mark.unit
def test_simulation_cache_stats_surfaces_failures(controller: SimulationController):
    controller.batch_cache.result_cache = MagicMock()
    controller.batch_cache.result_cache.used_entries.side_effect = RuntimeError("boom")
    controller.batch_cache.preview_cache = MagicMock()
    controller.batch_cache.preview_cache.used_entries.side_effect = RuntimeError("boom")
    result = controller.simulation_cache_stats()
    assert result.ok is False
    assert result.operation == "stats"
    assert result.stats is None
    assert "Failed to read simulation cache status" in result.message


@pytest.mark.unit
def test_purge_simulation_result_cache_surfaces_failures(controller: SimulationController):
    controller.batch_cache.purge_result_cache = MagicMock(side_effect=RuntimeError("purge boom"))

    result = controller.purge_simulation_result_cache()

    assert result.ok is False
    assert result.operation == "purge_result_cache"
    assert "Failed to clear simulation result cache" in result.message


@pytest.mark.unit
def test_cleanup_worker_safely_does_not_force_terminate(controller: SimulationController):
    worker = _FakeWorker(running=True, wait_returns=False, signal_disconnect_typeerror=True)
    controller._cleanup_worker_safely(worker, "test worker")
    assert worker._cancelled is True
    assert worker._terminated is False


@pytest.mark.unit
def test_cleanup_worker_safely_defers_qthread_deletion_until_finished(controller: SimulationController, monkeypatch):
    send_called = {"n": 0}

    def _boom_send_posted_events(*_args, **_kwargs) -> None:
        send_called["n"] += 1
        raise AssertionError("sendPostedEvents must not run while worker thread is still running")

    monkeypatch.setattr(QtCore.QCoreApplication, "sendPostedEvents", _boom_send_posted_events)

    worker = make_stubborn_worker(_FakeWorker)
    controller._cleanup_worker_safely(worker, "test worker")
    assert worker._cancelled is True
    assert worker._delete_later_called is False
    assert worker.deleteLater in worker.finished._handlers
    assert send_called["n"] == 0


@pytest.mark.unit
def test_release_current_simulation_worker_skips_unregistered_qt_signal_disconnect_warning(
    controller: SimulationController,
):
    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._release_current_simulation_worker()

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]
    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]


@pytest.mark.unit
def test_cleanup_worker_safely_disconnects_registered_qt_signal_handlers_without_warning(
    controller: SimulationController,
    monkeypatch,
):
    worker = _QtSignalWorker(running=False)
    progress = MagicMock()
    complete = MagicMock()
    error = MagicMock()
    controller.on_simulation_progress = progress
    controller.on_simulation_complete = complete
    controller.on_simulation_error = error
    monkeypatch.setattr(controller, "_delete_worker_if_stopped", MagicMock())

    controller._connect_simulation_worker_application_signals(
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    worker.progress.emit(10, "running")
    worker.result_ready.emit({"payload": True})
    worker.error.emit({"kind": "failure"})

    assert progress.call_count == 1
    assert complete.call_count == 1
    assert error.call_count == 1

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._cleanup_worker_safely(worker, "simulation worker")

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]
    assert getattr(worker, "_kindred_controller_worker_signal_handlers", ()) == ()
    controller._delete_worker_if_stopped.assert_called_once_with(worker, "simulation worker")

    worker.progress.emit(20, "after")
    worker.result_ready.emit({"payload": False})
    worker.error.emit({"kind": "ignored"})

    assert progress.call_count == 1
    assert complete.call_count == 1
    assert error.call_count == 1


@pytest.mark.unit
def test_disconnect_simulation_worker_application_signals_preserves_failed_runtime_disconnects(
    controller: SimulationController,
):
    class _Signal:
        def __init__(self, *, raise_runtimeerror: bool = False) -> None:
            self.raise_runtimeerror = bool(raise_runtimeerror)
            self.handlers: list[Callable[..., Any]] = []

        def connect(self, handler: Callable[..., Any]) -> None:
            self.handlers.append(handler)

        def disconnect(self, handler: Callable[..., Any]) -> None:
            if self.raise_runtimeerror:
                raise RuntimeError("disconnect failed")
            self.handlers.remove(handler)

        def emit(self, *args: Any) -> None:
            for handler in tuple(self.handlers):
                handler(*args)

    class _Worker:
        def __init__(self) -> None:
            self.progress = _Signal(raise_runtimeerror=True)
            self.result_ready = _Signal()
            self.error = _Signal()

    worker = _Worker()
    progress = MagicMock()
    complete = MagicMock()
    error = MagicMock()
    controller.on_simulation_progress = progress
    controller.on_simulation_complete = complete
    controller.on_simulation_error = error
    controller._record_nonfatal_exception = MagicMock()

    controller._connect_simulation_worker_application_signals(
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._disconnect_simulation_worker_application_signals(worker)

    remaining = getattr(worker, "_kindred_controller_worker_signal_handlers", ())
    assert len(remaining) == 1
    assert remaining[0][0] == "progress"
    controller._record_nonfatal_exception.assert_called_once()

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]

    worker.progress.emit(20, "done")
    worker.result_ready.emit({"payload": False})
    worker.error.emit({"kind": "ignored"})

    assert progress.call_count == 1
    assert complete.call_count == 0
    assert error.call_count == 0


@pytest.mark.unit
def test_connect_simulation_worker_application_signals_preserves_tracked_disconnect_failures_on_reconnect(
    controller: SimulationController,
):
    class _Signal:
        def __init__(self, *, raise_runtimeerror: bool = False) -> None:
            self.raise_runtimeerror = bool(raise_runtimeerror)
            self.handlers: list[Callable[..., Any]] = []

        def connect(self, handler: Callable[..., Any]) -> None:
            self.handlers.append(handler)

        def disconnect(self, handler: Callable[..., Any]) -> None:
            if self.raise_runtimeerror:
                raise RuntimeError("disconnect failed")
            self.handlers.remove(handler)

    class _Worker:
        def __init__(self) -> None:
            self.progress = _Signal(raise_runtimeerror=True)
            self.result_ready = _Signal()
            self.error = _Signal()

    worker = _Worker()
    controller._record_nonfatal_exception = MagicMock()

    controller._connect_simulation_worker_application_signals(
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    controller._connect_simulation_worker_application_signals(
        worker,
        run_id=8,
        fast_mode=False,
        request_id=12,
        set_name="set2",
        set_id="id2",
        cache_key="ck2",
    )

    connections = getattr(worker, "_kindred_controller_worker_signal_handlers", ())
    names = [signal_name for signal_name, _handler in connections]
    assert names.count("progress") == 2
    assert names.count("result_ready") == 1
    assert names.count("error") == 1
    controller._record_nonfatal_exception.assert_called_once()


@pytest.mark.unit
def test_prepare_simulation_shutdown_for_close_keeps_window_recoverable_when_worker_errors_after_deferred_close(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(42)

    worker.error.connect(
        lambda msg: controller.on_simulation_error(
            msg,
            run_id=3,
            fast_mode=False,
            request_id=5,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="ck",
        )
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._status_label.text == "Simulation cancelled by user"
    assert mw._sim_progress.value == 0


@pytest.mark.unit
def test_prepare_simulation_shutdown_for_close_ignores_deleted_retained_worker(controller: SimulationController):
    worker = QtCore.QThread(parent=controller)
    controller._retained_simulation_workers.append(worker)
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)

    close_ready = controller.prepare_simulation_shutdown_for_close()

    assert close_ready is True
    assert controller._shutdown_requested_for_close is False
    assert controller._retained_simulation_workers == []


@pytest.mark.unit
def test_release_current_simulation_worker_ignores_deleted_worker(controller: SimulationController):
    worker = QtCore.QThread(parent=controller)
    controller._simulation_worker = worker
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)

    controller.release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == []


@pytest.mark.unit
def test_run_simulation_from_slider_ignores_deleted_current_worker(
    mw: _FakeMainWindow, controller: SimulationController
):
    calls: list[dict[str, Any]] = []

    def _record_run(**kwargs) -> None:
        calls.append(dict(kwargs))

    controller.run_simulation_internal = _record_run
    worker = QtCore.QThread(parent=controller)
    controller._simulation_worker = worker
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)
    controller._latest_sim_request_id = 1
    controller._pending_slider_sim_request_id = 1

    controller._run_simulation_from_slider()

    assert controller._simulation_worker is None
    assert calls and calls[0]["fast_mode"] is True


@pytest.mark.unit
def test_deferred_close_successful_completion_does_not_schedule_next_serial_batch_run(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._simulation_running = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "total": 2,
    }

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker.result_ready.connect(
        lambda payload: controller.on_simulation_complete(
            payload,
            run_id=7,
            fast_mode=False,
            request_id=11,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="ck",
        )
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.result_ready.emit(_successful_result_payload())

    assert scheduled == []
    assert controller._batch_run_context.get("active") is False
    assert controller._simulation_running is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers


@pytest.mark.unit
def test_deferred_close_successful_completion_does_not_schedule_pending_slider_rerun_and_still_recovers_ui(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 9
    controller._latest_sim_request_id = 15
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._pending_slider_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(42)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker.result_ready.connect(
        lambda payload: controller.on_simulation_complete(
            payload,
            run_id=9,
            fast_mode=False,
            request_id=15,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="ck",
        )
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.result_ready.emit(_successful_result_payload())

    assert scheduled == []
    assert controller._pending_slider_simulation is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 100
    assert mw._status_label.text == "Simulation complete: 2 species, 3 points"


@pytest.mark.unit
def test_deferred_close_error_recovery_restores_later_serial_batch_continuation(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True

    worker.error.connect(
        lambda msg: controller.on_simulation_error(
            msg,
            run_id=3,
            fast_mode=False,
            request_id=5,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="old-ck",
        )
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert controller._shutdown_requested_for_close is True

    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._shutdown_requested_for_close is True
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.finished.emit()

    assert controller._shutdown_requested_for_close is False
    assert worker not in controller._retained_simulation_workers

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._active_run_id = 13
    controller._latest_sim_request_id = 12
    controller._simulation_running = True
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "total": 2,
    }

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=13,
        fast_mode=False,
        request_id=12,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="new-ck",
    )

    assert scheduled == [controller._start_next_batch_simulation]
    assert controller._batch_run_context["active"] is True
    assert controller._batch_run_context["pos"] == 1


@pytest.mark.unit
def test_deferred_close_error_recovery_restores_later_pending_slider_rerun(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 7
    controller._latest_sim_request_id = 9
    controller._simulation_running = True

    worker.error.connect(
        lambda msg: controller.on_simulation_error(
            msg,
            run_id=7,
            fast_mode=False,
            request_id=9,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="old-ck",
        )
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert controller._shutdown_requested_for_close is True

    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._shutdown_requested_for_close is True
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.finished.emit()

    assert controller._shutdown_requested_for_close is False
    assert worker not in controller._retained_simulation_workers

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._active_run_id = 15
    controller._latest_sim_request_id = 16
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._pending_slider_simulation = True
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "total": 1,
    }

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=15,
        fast_mode=False,
        request_id=16,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="new-ck",
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is False


@pytest.mark.unit
def test_simulation_worker_does_not_shadow_qthread_finished_signal():
    assert "finished" not in SimulationWorker.__dict__


@pytest.mark.unit
def test_shutdown_batch_executor_falls_back_on_typeerror_and_terminates_processes(
    controller: SimulationController,
):
    proc = MagicMock()
    proc.is_alive.return_value = True

    executor = MagicMock()
    executor.shutdown.side_effect = [TypeError("no cancel_futures"), None]
    executor._processes = {"p": proc}

    controller._batch_parallel.executor = executor
    controller._shutdown_batch_executor(force_terminate=True)
    assert proc.terminate.call_count == 1


@pytest.mark.unit
def test_slider_request_during_parallel_full_run_defers_without_force_terminate(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._run_simulation_internal = MagicMock()
    mw._run_btn = _FakeButton(False)

    worker = _FakeWorker(running=True, wait_returns=False, signal_disconnect_typeerror=True)
    worker._fast_mode = False  # type: ignore[attr-defined]

    def _boom_thread_terminate() -> None:
        raise AssertionError("QThread.terminate must not be called from slider deferral path")

    worker.terminate = _boom_thread_terminate  # type: ignore[assignment]
    controller._simulation_worker = worker

    proc = MagicMock()
    proc.is_alive.return_value = True
    proc.terminate.side_effect = AssertionError("Process terminate must not be called from slider deferral path")

    executor = MagicMock()
    executor.shutdown = MagicMock()
    executor._processes = {"p": proc}
    controller._batch_parallel.executor = executor

    controller._batch_run_context = {"active": True, "parallel": True}
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is True
    controller._run_simulation_internal.assert_not_called()


@pytest.mark.unit
def test_slider_request_while_fast_worker_running_is_latest_only_and_does_not_cancel(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._run_simulation_internal = MagicMock()
    mw._run_btn = _FakeButton(True)

    worker = _FakeWorker(running=True, wait_returns=True)
    worker._fast_mode = True  # type: ignore[attr-defined]
    worker.cancel = MagicMock(side_effect=AssertionError("Fast worker must not be cancelled for latest-only scheduling"))
    controller._simulation_worker = worker

    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is True
    controller._run_simulation_internal.assert_not_called()


@pytest.mark.unit
def test_stale_fast_completion_schedules_pending_slider_run(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True

    controller._active_run_id = 5
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True}

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=5,
        fast_mode=True,
        request_id=int(rid_old),
    )
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_superseded_multiset_preview_completion_still_displays_current_result_before_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = False
    controller._active_run_id = 7
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 7,
        "request_id": int(rid_old),
        "fast_mode": True,
        "cache_key": "preview-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "completed_set_ids": ["id1"],
        "preview_scope_set_ids": ("id1", "id2"),
        "preview_batch_cache_token_by_set_id": {"id1": "", "id2": ""},
    }
    mw._batch_set_ids_for_scope.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.side_effect = lambda row: ("id1", "id2")[int(row)]
    mw._display_cached_batch_selection.return_value = True

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=True,
        request_id=int(rid_old),
        batch_set="set2",
        batch_set_id="id2",
        cache_key="preview-cache",
    )

    assert mw._display_cached_batch_selection.call_count == 1
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_superseded_multiset_preview_partial_completion_keeps_parallel_batch_active_until_full_batch_finishes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = False
    controller._active_run_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 8,
        "request_id": int(rid_old),
        "fast_mode": True,
        "cache_key": "preview-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "completed_set_ids": [],
        "preview_scope_set_ids": ("id1", "id2"),
        "preview_batch_cache_token_by_set_id": {"id1": "", "id2": ""},
    }
    mw._batch_set_ids_for_scope.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.side_effect = lambda row: ("id1", "id2")[int(row)]
    mw._display_cached_batch_selection.return_value = True

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=8,
        fast_mode=True,
        request_id=int(rid_old),
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
    )

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is True
    assert ctx.get("completed_set_ids") == ["id1"]
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert scheduled == []


@pytest.mark.unit
def test_stale_fast_completion_without_pending_still_cleans_up_active_run(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 2
    controller._pending_slider_sim_request_id = 2
    controller._pending_slider_simulation = False

    controller._active_run_id = 11
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True}
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(0)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=1,
    )

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_on_simulation_complete_uses_base_species_count_for_algebra_status_without_mechanism(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "keep_executor_alive": False,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": True,
        "cache_key": "preview-cache",
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "total": 1,
        "preview_scope_set_ids": ("id1",),
        "preview_batch_cache_token_by_set_id": {"id1": ""},
    }
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    result = _successful_result_payload()
    result["species_names"] = ["A", "Alg"]
    result["algebra_errors"] = [{"kind": "algebra_error", "message": "bad algebra"}]
    result["base_species_count"] = 1

    controller._on_simulation_complete(
        result,
        run_id=7,
        fast_mode=True,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
    )

    assert mw._algebra_status_label.text == "Algebra: 1 ok, 1 error"


@pytest.mark.unit
def test_on_simulation_complete_prefers_payload_base_species_count_over_mechanism_for_algebra_status(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "keep_executor_alive": False,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": True,
        "cache_key": "preview-cache",
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "total": 1,
        "preview_scope_set_ids": ("id1",),
        "preview_batch_cache_token_by_set_id": {"id1": ""},
    }
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    mechanism = MagicMock()
    mechanism.species_names.return_value = ["A", "Alg"]

    result = _successful_result_payload()
    result["species_names"] = ["A", "Alg"]
    result["mechanism"] = mechanism
    result["algebra_errors"] = [{"kind": "algebra_error", "message": "bad algebra"}]
    result["base_species_count"] = 1

    controller._on_simulation_complete(
        result,
        run_id=7,
        fast_mode=True,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
    )

    assert mw._algebra_status_label.text == "Algebra: 1 ok, 1 error"


@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_ui_active_when_full_run_still_in_flight(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._batch_run_context = {"active": True, "parallel": True, "fast_mode": False}

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    controller.invalidate_slider_preview_work()

    assert controller._pending_slider_simulation is False
    assert controller._batch_run_context.get("active") is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57


@pytest.mark.unit
def test_invalidate_slider_preview_work_supersedes_active_fast_parallel_batch(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "fast_mode": True,
        "request_id": int(rid),
    }

    def _fake_supersede() -> None:
        ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
        ctx["active"] = False
        controller._batch_run_context = ctx

    controller._supersede_parallel_batch_run_soft = MagicMock(side_effect=_fake_supersede)

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)

    controller.invalidate_slider_preview_work()

    controller._supersede_parallel_batch_run_soft.assert_called_once_with()
    assert controller._batch_run_context.get("active") is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0


@pytest.mark.unit
def test_invalidate_slider_preview_work_suppresses_stale_completion_ui_after_discard(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True}
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    controller._pending_slider_plot_cache_key = "preview-ck"
    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(73)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller.invalidate_slider_preview_work()

    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
    )

    assert scheduled == []
    assert controller._pending_slider_plot_cache_key is None
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_active_after_stale_completion(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "fast_mode": False,
        "request_id": 99,
    }
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 99  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller.invalidate_slider_preview_work()
    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
    )

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is True
    assert ctx.get("fast_mode") is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57


@pytest.mark.unit
def test_nonowning_stale_fast_completion_does_not_reset_explicit_run_status_progress(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 3
    controller._pending_slider_sim_request_id = None
    controller._pending_slider_simulation = False
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "fast_mode": False,
        "request_id": 3,
    }
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 3  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=1,
    )

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is True
    assert ctx.get("fast_mode") is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57


@pytest.mark.unit
def test_invalidate_slider_preview_work_suppresses_stale_error_ui_after_discard(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True}
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller.invalidate_slider_preview_work()

    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    controller._on_simulation_error("boom", run_id=11, fast_mode=True, request_id=int(rid))

    assert scheduled == []
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    message_box.assert_not_called()


@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_active_after_stale_error(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "fast_mode": False,
        "request_id": 101,
    }
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 101  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller.invalidate_slider_preview_work()
    controller._on_simulation_error("boom", run_id=11, fast_mode=True, request_id=int(rid))

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is True
    assert ctx.get("fast_mode") is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 41
    message_box.assert_not_called()


@pytest.mark.unit
def test_invalidate_active_explicit_simulation_for_authoritative_change_cancels_run_and_ignores_old_completion(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._run_sequence_id = 11
    controller._active_run_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "fast_mode": False,
        "request_id": 101,
    }
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 101  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)
    mw.set_data.reset_mock()

    controller.invalidate_active_explicit_simulation_for_authoritative_change()

    assert controller._active_run_id == 12
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert explicit_worker._cancelled is True
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=11,
        fast_mode=False,
        request_id=101,
    )

    mw.set_data.assert_not_called()


@pytest.mark.unit
def test_nonowning_stale_fast_error_does_not_reset_explicit_run_status_progress(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 3
    controller._pending_slider_sim_request_id = None
    controller._pending_slider_simulation = False
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "fast_mode": False,
        "request_id": 3,
    }
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 3  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller._on_simulation_error("boom", run_id=11, fast_mode=True, request_id=1)

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is True
    assert ctx.get("fast_mode") is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 41
    message_box.assert_not_called()


@pytest.mark.unit
def test_stale_fast_error_without_pending_still_cleans_up_active_run(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._latest_sim_request_id = 2
    controller._pending_slider_sim_request_id = 2
    controller._pending_slider_simulation = False

    controller._active_run_id = 7
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True}
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(0)

    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_error("boom", run_id=7, fast_mode=True, request_id=1)

    ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
    assert ctx.get("active") is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert scheduled == [controller._run_simulation_from_slider]
    message_box.assert_not_called()


@pytest.mark.unit
def test_slider_run_deferral_does_not_set_updating_status_when_no_new_run_starts(
    mw: _FakeMainWindow, controller: SimulationController
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    controller._batch_run_context = {"active": True, "fast_mode": True}
    controller._simulation_worker = None
    controller._simulation_running = False

    mw._status_label.setText("Ready")

    controller._run_simulation_from_slider()

    assert controller._pending_slider_simulation is True
    assert controller._simulation_running is False
    assert mw._status_label.text == "Ready"


@pytest.mark.unit
def test_slider_run_supersedes_active_fast_parallel_preview_for_newer_request(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._latest_sim_request_id = int(rid_new)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "fast_mode": True,
        "request_id": int(rid_old),
    }

    def _fake_supersede() -> None:
        ctx = dict(getattr(controller, "_batch_run_context", {}) or {})
        ctx["active"] = False
        controller._batch_run_context = ctx

    controller._supersede_parallel_batch_run_soft = MagicMock(side_effect=_fake_supersede)

    controller._run_simulation_from_slider()

    controller._supersede_parallel_batch_run_soft.assert_called_once_with()
    controller.run_simulation_internal.assert_called_once()
    assert controller.run_simulation_internal.call_args.kwargs["request_id"] == int(rid_new)
    assert controller.run_simulation_internal.call_args.kwargs["reuse_parallel_executor"] is True


@pytest.mark.unit
def test_slider_run_blocks_launch_while_retained_worker_is_still_running(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    worker = make_stubborn_worker(_FakeWorker)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    controller._simulation_running = False
    mw._status_label.setText("Ready")
    mw._run_btn.setEnabled(True)
    mw._stop_btn.setEnabled(False)

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert controller.has_running_owned_simulation_workers() is True
    assert controller._retained_simulation_workers == [worker]
    assert mw._status_label.text == "Cancelling previous simulation..."
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_retained_worker_finish_replays_latest_pending_slider_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._simulation_running = False

    mw._status_label.setText("Ready")
    mw._run_btn.setEnabled(True)
    mw._stop_btn.setEnabled(False)

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == int(rid)
    assert scheduled == []

    worker._running = False
    worker.finished.emit()

    assert controller._retained_simulation_workers == []
    assert controller.has_running_owned_simulation_workers() is False
    assert scheduled == [controller._run_simulation_from_slider]

    scheduled[0]()

    assert controller.run_simulation_internal.call_count == 1
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True


@pytest.mark.unit
def test_retained_worker_finish_preserves_reserved_future_slider_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    controller._latest_sim_request_id = 5
    controller._pending_slider_sim_request_id = 6
    controller._pending_slider_simulation = True
    controller._simulation_running = False

    worker._running = False
    worker.finished.emit()

    assert controller._retained_simulation_workers == []
    assert controller.has_running_owned_simulation_workers() is False
    assert controller._pending_slider_sim_request_id == 6
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_retained_worker_finish_cancels_species_timer_before_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    species_timer = _ActiveTimer()
    mw._species_slider_update_timer = species_timer

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._simulation_running = False

    def _fire_species_timeout_if_still_active() -> None:
        if species_timer.isActive():
            controller.run_simulation_from_slider()

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._pending_slider_simulation is True
    assert scheduled == []

    worker._running = False
    worker.finished.emit()

    assert species_timer.stop_calls == 1
    assert species_timer.isActive() is False
    assert scheduled == [controller._run_simulation_from_slider]

    scheduled[0]()
    assert controller.run_simulation_internal.call_count == 1

    _fire_species_timeout_if_still_active()
    assert controller.run_simulation_internal.call_count == 1


@pytest.mark.unit
def test_supersede_parallel_batch_run_soft_cancels_futures_and_stops_timer(controller: SimulationController):
    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_future_poll_timer = timer

    fut_cancelled = MagicMock()
    fut_cancelled.cancel.return_value = True
    fut_running = MagicMock()
    fut_running.cancel.return_value = False

    controller._batch_parallel.future_map = {"a": fut_cancelled, "b": fut_running}
    controller._batch_parallel.future_meta = {"a": {"set_name": "A"}, "b": {"set_name": "B"}}
    controller._batch_run_context = {"active": True, "parallel": True}

    controller._supersede_parallel_batch_run_soft()
    assert controller._batch_parallel.future_map == {}
    assert controller._batch_parallel.future_meta == {}
    assert controller._batch_parallel.superseded_future_map == {"b": fut_running}
    assert controller._batch_parallel.superseded_future_meta == {
        "b": {"set_name": "B", "set_id": "b", "superseded": "1"}
    }
    timer.stop.assert_called()


@pytest.mark.unit
def test_superseded_parallel_batch_future_error_is_drained_deterministically(controller: SimulationController):
    class _RunningFuture:
        def __init__(self) -> None:
            self._done = False
            self._exc: Exception | None = None

        def cancel(self) -> bool:
            return False

        def done(self) -> bool:
            return bool(self._done)

        def set_exception(self, exc: Exception) -> None:
            self._exc = exc
            self._done = True

        def result(self):
            if self._exc is not None:
                raise self._exc
            return {"ok": True}

    fut = _RunningFuture()

    controller._batch_parallel.future_map = {"sid": fut}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}
    controller._batch_run_context = {"active": False, "parallel": False}
    controller._record_nonfatal_exception = MagicMock()

    controller._supersede_parallel_batch_run_soft()
    fut.set_exception(RuntimeError("boom"))

    controller._poll_parallel_batch_futures()

    assert controller._batch_parallel.superseded_future_map == {}
    assert controller._batch_parallel.superseded_future_meta == {}
    controller._record_nonfatal_exception.assert_called_once()


@pytest.mark.unit
def test_superseded_future_error_does_not_abort_active_run(controller: SimulationController):
    submitted: list[tuple[str, dict[str, object]]] = []

    class _PendingFuture:
        def __init__(self) -> None:
            self._done = False
            self._result = None

        def add_done_callback(self, _callback) -> None:
            return

        def done(self) -> bool:
            return bool(self._done)

        def set_result(self, result) -> None:
            self._done = True
            self._result = result

        def result(self):
            return self._result

    class _SupersededFuture:
        def done(self) -> bool:
            return True

        def result(self):
            raise RuntimeError("superseded boom")

    class _FakeExecutor:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.shutdown_calls: list[dict[str, object]] = []

        def submit(self, _fn, *args, **_kwargs):
            if args:
                submitted.append((self.label, dict(args[0])))
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    created: list[_FakeExecutor] = []

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _FakeExecutor:
        executor = _FakeExecutor(label=f"executor-{len(created) + 1}-w{int(max_workers)}")
        created.append(executor)
        return executor

    current = _PendingFuture()
    first = _FakeExecutor("initial")
    controller.parallel_batch.executor_factory = _factory
    controller.parallel_batch.executor = first
    controller.parallel_batch._current_max_workers = 2
    controller._pool_eagerly_created = True
    controller._batch_parallel.future_map = {"current": current}
    controller._batch_parallel.future_meta = {"current": {"set_name": "current-set"}}
    controller._batch_parallel.superseded_future_map = {"stale": _SupersededFuture()}
    controller._batch_parallel.superseded_future_meta = {
        "stale": {"set_id": "stale", "set_name": "stale-set", "superseded": "1"}
    }
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "run_id": 11,
        "request_id": 22,
        "fast_mode": False,
        "cache_key": "current-cache",
    }
    controller._on_simulation_error = MagicMock()
    controller._on_simulation_complete = MagicMock()
    controller._record_nonfatal_exception = MagicMock()

    controller._poll_parallel_batch_futures()

    controller._on_simulation_error.assert_not_called()
    controller._record_nonfatal_exception.assert_called_once()
    assert controller._batch_run_context["active"] is True
    assert first.shutdown_calls == []
    assert controller.parallel_batch.executor is first
    assert controller.parallel_batch.future_map == {"current": current}
    assert controller.parallel_batch.superseded_future_map == {}
    assert controller._pool_eagerly_created is True

    current.set_result({"payload": "stale"})
    controller._batch_parallel.completed_queue.put(("current", 1.0))
    controller._poll_parallel_batch_futures()
    controller._on_simulation_complete.assert_called_once()

    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["fresh"],
        "queue_names": ["fresh-set"],
        "run_id": 12,
        "request_id": 23,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {},
        "pending_init_applied": True,
    }
    controller.ui.batch.batch_initials_for_row = MagicMock(return_value={"A": 1.0})

    controller._start_parallel_batch_simulations()

    assert created == []
    assert controller.parallel_batch.executor is first
    assert [label for label, task in submitted if task.get("set_id") == "fresh"] == [first.label]


@pytest.mark.unit
def test_parallel_keep_executor_alive_completion_keeps_polling_until_superseded_future_drains(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _SupersededFuture:
        def __init__(self) -> None:
            self._done = False

        def done(self) -> bool:
            return bool(self._done)

        def finish(self) -> None:
            self._done = True

        def result(self):
            return {"ok": True}

    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_future_poll_timer = timer

    fut = _SupersededFuture()
    controller._batch_parallel.superseded_future_map = {"sid": fut}
    controller._batch_parallel.superseded_future_meta = {"sid": {"set_id": "sid", "set_name": "set1", "superseded": "1"}}
    controller._record_nonfatal_exception = MagicMock()
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "ck",
        "queue_ids": ["sid"],
        "queue_names": ["set1"],
        "total": 1,
    }

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="sid",
        cache_key="ck",
    )

    assert controller._batch_run_context.get("active") is False
    assert controller._batch_parallel.future_map == {}
    assert controller._batch_parallel.superseded_future_map == {"sid": fut}
    timer.stop.assert_not_called()

    fut.finish()
    controller._poll_parallel_batch_futures()

    assert controller._batch_parallel.superseded_future_map == {}
    assert controller._batch_parallel.superseded_future_meta == {}
    controller._record_nonfatal_exception.assert_not_called()
    timer.stop.assert_called_once()


@pytest.mark.unit
def test_superseded_parallel_batch_future_error_payload_keeps_healthy_pool_alive(controller: SimulationController):
    class _SupersededFuture:
        def done(self) -> bool:
            return True

        def result(self):
            return {
                "success": False,
                "error": {"kind": "simulation_error", "message": "solver blew up", "code": "E301"},
            }

    class _FakeExecutor:
        def __init__(self) -> None:
            self.shutdown_calls: list[dict[str, object]] = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    executor = _FakeExecutor()
    controller.parallel_batch.executor = executor
    controller.parallel_batch._current_max_workers = 2
    controller._pool_eagerly_created = True
    controller._batch_run_context = {"active": False, "parallel": False}
    controller._batch_parallel.superseded_future_map = {"sid": _SupersededFuture()}
    controller._batch_parallel.superseded_future_meta = {
        "sid": {"set_id": "sid", "set_name": "set1", "superseded": "1"}
    }
    controller._record_nonfatal_exception = MagicMock()

    controller._poll_parallel_batch_futures()

    assert controller.parallel_batch.executor is executor
    assert executor.shutdown_calls == []
    assert controller.parallel_batch.superseded_future_map == {}
    assert controller.parallel_batch.superseded_future_meta == {}
    assert controller._pool_eagerly_created is True
    controller._record_nonfatal_exception.assert_called_once()


@pytest.mark.unit
def test_primary_explicit_completion_preserves_fresh_cache_during_post_run_species_sync(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _Mechanism:
        def species_names(self) -> list[str]:
            return ["A", "C"]

    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "keep_executor_alive": False,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "fresh-current-cache",
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "total": 1,
    }
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    mw._sync_batch_species_columns = MagicMock()

    result = _successful_result_payload()
    result["mechanism"] = _Mechanism()
    result["mechanism_text"] = "reaction: A -> C ; k=0.1"
    result["species_names"] = ["A", "C"]

    controller._on_simulation_complete(
        result,
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="fresh-current-cache",
    )

    mw._sync_batch_species_columns.assert_called_once_with(["A", "C"], preserve_active_cache=True)
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)


@pytest.mark.unit
def test_on_simulation_complete_later_completion_does_not_widen_narrowed_valid_subset(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "fresh-current-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "explicit_cache_preview_token": "narrow-preview-token",
        "explicit_cache_preview_scope_set_ids": ("id1",),
        "explicit_cache_valid_set_ids": ("id1",),
    }
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_preview_token = "narrow-preview-token"
    controller.batch_cache.active_cache_preview_scope_set_ids = ("id1",)
    controller.batch_cache.active_cache_valid_set_ids = ("id1",)

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
    )

    assert controller.batch_cache.active_cache_preview_token == "narrow-preview-token"
    assert controller.batch_cache.active_cache_preview_scope_set_ids == ("id1",)
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)


@pytest.mark.unit
def test_on_simulation_complete_redraw_falls_back_to_current_result_when_constrained_subset_draw_returns_false(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "keep_executor_alive": False,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "fresh-current-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "explicit_cache_valid_set_ids": ("id2",),
    }
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id2",)
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False
    mw.set_data.reset_mock()

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
    )

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["valid_set_ids"] == ("id2",)
    assert kwargs["allow_fallback"] is False
    mw.set_data.assert_called_once()
    assert mw.set_data.call_args.kwargs["label"] == "set2"


@pytest.mark.unit
def test_on_simulation_complete_coalesced_flush_uses_valid_subset_without_fallback(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "fresh-current-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "explicit_cache_valid_set_ids": ("id2",),
    }
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id2",)
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
    )

    assert mw._display_cached_batch_selection.call_count == 0

    controller._flush_slider_plot_updates()

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["valid_set_ids"] == ("id2",)
    assert kwargs["allow_fallback"] is False


@pytest.mark.unit
def test_on_simulation_complete_coalesced_flush_keeps_valid_subset_after_dirty_reset(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "run_id": 7,
        "request_id": 11,
        "fast_mode": False,
        "cache_key": "fresh-current-cache",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "total": 2,
        "completed_set_ids": ["id1"],
        "explicit_cache_valid_set_ids": ("id1", "id2"),
        "pending_workspace_reset_set_ids": ["id2"],
        "pending_dirty_reset_generation_by_set_id": {"id2": 1},
    }
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id1", "id2")
    mw._dirty_state_generations = {"id2": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = False
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._shown_batch_set_ids.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = True

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
    )

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["selected_sets"] == ["id1", "id2"]
    assert kwargs["valid_set_ids"] == ("id1", "id2")
    assert kwargs["allow_fallback"] is False


@pytest.mark.unit
def test_queue_slider_plot_update_gates_by_request_and_run_ids(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 2
    controller._active_run_id = 10

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=1,
        run_id=10,
        slider_triggered=True,
    )
    assert controller._pending_slider_plot_set_ids == set()

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=2,
        run_id=11,
        slider_triggered=True,
    )
    assert controller._pending_slider_plot_set_ids == set()

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=2,
        run_id=10,
        slider_triggered=False,
    )
    assert controller._pending_slider_plot_set_ids == {"s1"}
    assert controller._pending_slider_plot_cache_kind == "result"

    assert controller._slider_plot_coalesce_timer.isActive()
    assert int(controller._slider_plot_coalesce_timer.interval()) >= 1


@pytest.mark.unit
def test_flush_slider_plot_updates_merges_pending_sets_and_calls_display(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = {"b"}
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2

    mw._shown_batch_set_ids.return_value = ["a", "b"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "a"
    mw._display_cached_batch_selection.return_value = True

    ok = controller._flush_slider_plot_updates()
    assert ok is True
    args, kwargs = mw._display_cached_batch_selection.call_args
    assert kwargs["cache_key"] == "cache-key"
    assert kwargs["prefer_set"] == "a"
    assert kwargs["selected_sets"] == ["a", "b"]


@pytest.mark.unit
def test_flush_slider_plot_updates_force_uses_cache_keys_when_no_selection(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = set()
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2

    mw._shown_batch_set_ids.return_value = []
    mw._display_cached_batch_selection.return_value = True

    controller.batch_cache.preview_cache.put("cache-key::x", {"t": [], "series": {}})
    controller.batch_cache.preview_cache.put("cache-key::y", {"t": [], "series": {}})

    ok = controller._flush_slider_plot_updates(force=True)
    assert ok is True
    _args, kwargs = mw._display_cached_batch_selection.call_args
    assert sorted(kwargs["selected_sets"]) == ["x", "y"]


@pytest.mark.unit
def test_flush_slider_plot_updates_uses_shown_sets_not_highlighted_selection(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = {"dirty"}
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2

    mw._batch_set_ids_for_scope.return_value = ["selected-only"]
    mw._shown_batch_set_ids.return_value = ["shown-a", "dirty"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "shown-a"
    mw._display_cached_batch_selection.return_value = True

    ok = controller._flush_slider_plot_updates()
    assert ok is True
    _args, kwargs = mw._display_cached_batch_selection.call_args
    assert kwargs["selected_sets"] == ["shown-a", "dirty"]


@pytest.mark.unit
def test_consume_parallel_batch_future_success_calls_on_complete_and_clears_maps(mw: _FakeMainWindow, controller: SimulationController):
    fut: Future = Future()
    fut.set_result({"payload": 123})

    controller._batch_parallel.future_map = {"sid": fut}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}

    controller._on_simulation_complete = MagicMock()
    ok = controller._consume_parallel_batch_future(
        set_id="sid",
        fut=fut,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        source="scan",
        completed_ts=1.0,
    )
    assert ok is True
    assert "sid" not in controller._batch_parallel.future_map
    assert "sid" not in controller._batch_parallel.future_meta
    controller._on_simulation_complete.assert_called_once()


@pytest.mark.unit
def test_consume_parallel_batch_future_on_complete_exception_reports_error_and_shutdown(
    mw: _FakeMainWindow, controller: SimulationController
):
    fut: Future = Future()
    fut.set_result({"payload": 123})

    controller._batch_parallel.future_map = {"sid": fut}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}

    controller._on_simulation_complete = MagicMock(side_effect=RuntimeError("ui boom"))
    controller._on_simulation_error = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    ok = controller._consume_parallel_batch_future(
        set_id="sid",
        fut=fut,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        source="scan",
        completed_ts=1.0,
    )
    assert ok is False
    controller._on_simulation_error.assert_called_once()
    controller._shutdown_batch_executor.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_consume_parallel_batch_future_error_calls_on_error_and_shutdown(mw: _FakeMainWindow, controller: SimulationController):
    fut: Future = Future()
    fut.set_exception(RuntimeError("boom"))
    controller._batch_parallel.future_map = {"sid": fut}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}

    controller._on_simulation_error = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    ok = controller._consume_parallel_batch_future(
        set_id="sid",
        fut=fut,
        run_id=1,
        request_id=2,
        fast_mode=True,
        cache_key="ck",
        source="callback",
        completed_ts=1.0,
    )
    assert ok is False
    controller._on_simulation_error.assert_called_once()
    controller._shutdown_batch_executor.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_consume_parallel_batch_future_exception_tears_down_pool_and_next_parallel_run_recreates_executor(
    controller: SimulationController,
):
    submitted: list[tuple[str, dict[str, object]]] = []

    class _PendingFuture:
        def __init__(self) -> None:
            self._done = False

        def add_done_callback(self, _callback) -> None:
            return

        def done(self) -> bool:
            return bool(self._done)

    class _BoomFuture:
        def result(self):
            raise RuntimeError("boom")

    class _FakeExecutor:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.shutdown_calls: list[dict[str, object]] = []

        def submit(self, _fn, *args, **_kwargs):
            if args:
                submitted.append((self.label, dict(args[0])))
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    created: list[_FakeExecutor] = []

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _FakeExecutor:
        executor = _FakeExecutor(label=f"executor-{len(created) + 1}-w{int(max_workers)}")
        created.append(executor)
        return executor

    first = _FakeExecutor("initial")
    controller.parallel_batch.executor_factory = _factory
    controller.parallel_batch.executor = first
    controller.parallel_batch._current_max_workers = 2
    controller._pool_eagerly_created = True
    controller._batch_parallel.future_map = {"sid": _BoomFuture()}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "run_id": 1,
        "request_id": 2,
        "fast_mode": False,
        "cache_key": "ck",
    }
    controller.on_simulation_error = MagicMock()

    ok = controller._consume_parallel_batch_future(
        set_id="sid",
        fut=controller._batch_parallel.future_map["sid"],
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        source="scan",
    )

    assert ok is False
    controller.on_simulation_error.assert_called_once()
    assert controller._batch_run_context["active"] is False
    assert first.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert controller.parallel_batch.executor is None
    assert controller.parallel_batch.future_map == {}
    assert controller._pool_eagerly_created is False

    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["fresh"],
        "queue_names": ["fresh-set"],
        "run_id": 3,
        "request_id": 11,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {},
        "pending_init_applied": True,
    }
    controller.ui.batch.batch_initials_for_row = MagicMock(return_value={"A": 1.0})

    controller._start_parallel_batch_simulations()

    assert len(created) == 1
    assert controller.parallel_batch.executor is created[0]
    assert created[0] is not first
    assert [label for label, task in submitted if task.get("set_id") == "fresh"] == [created[0].label]


@pytest.mark.unit
def test_poll_parallel_batch_futures_consumes_callback_then_scan(controller: SimulationController):
    fut1: Future = Future()
    fut1.set_result({"ok": 1})
    fut2: Future = Future()
    fut2.set_result({"ok": 2})

    controller._batch_parallel.completed_queue = SimpleQueue()
    controller._batch_parallel.completed_queue.put(("a", 1.234))
    controller._batch_parallel.future_map = {"a": fut1, "b": fut2}
    controller._batch_parallel.future_meta = {"a": {"set_name": "A"}, "b": {"set_name": "B"}}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "run_id": 9,
        "request_id": 8,
        "fast_mode": False,
        "cache_key": "ck",
    }

    controller._consume_parallel_batch_future = MagicMock(return_value=True)
    controller._poll_parallel_batch_futures()

    assert controller._consume_parallel_batch_future.call_count == 2
    sources = [kwargs["source"] for _args, kwargs in controller._consume_parallel_batch_future.call_args_list]
    assert sources == ["callback", "scan"]


@pytest.mark.unit
def test_poll_parallel_batch_futures_catches_unhandled_exceptions_and_shuts_down(mw: _FakeMainWindow, controller: SimulationController):
    fut1: Future = Future()
    fut1.set_result({"ok": 1})

    controller._batch_parallel.future_map = {"a": fut1}
    controller._batch_parallel.future_meta = {"a": {"set_name": "A"}}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "run_id": 9,
        "request_id": 8,
        "fast_mode": False,
        "cache_key": "ck",
    }

    controller._on_simulation_error = MagicMock()
    controller._shutdown_batch_executor = MagicMock()
    controller._consume_parallel_batch_future = MagicMock(side_effect=RuntimeError("boom"))

    controller._poll_parallel_batch_futures()
    controller._shutdown_batch_executor.assert_called_once_with(force_terminate=True)
    controller._on_simulation_error.assert_called_once()


@pytest.mark.unit
def test_flush_pending_slider_updates_for_run_stops_timers_and_finalizes(mw: _FakeMainWindow, controller: SimulationController):
    release_timer = MagicMock()
    release_timer.isActive.return_value = True
    debounce_timer = MagicMock()
    debounce_timer.isActive.return_value = True
    mw._slider_release_commit_timer = release_timer
    mw._variable_update_timer = debounce_timer
    mw._pending_slider_values = {"a": 1}
    mw._finalize_slider_release_commit = MagicMock()
    mw._slider_triggered_simulation = True

    controller._pending_slider_simulation = True
    controller._pending_slider_plot_cache_key = "ck"

    controller._flush_pending_slider_updates_for_run()
    release_timer.stop.assert_called_once()
    debounce_timer.stop.assert_called_once()
    mw._finalize_slider_release_commit.assert_called_once()
    assert controller._pending_slider_simulation is True
    assert mw._slider_triggered_simulation is False


@pytest.mark.unit
def test_flush_pending_slider_updates_for_run_stops_species_timer_and_preserves_replay_until_success(
    mw: _FakeMainWindow, controller: SimulationController
):
    species_timer = MagicMock()
    species_timer.isActive.return_value = True
    mw._species_slider_update_timer = species_timer

    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller.run_state.pending_slider_target_set_ids = ("id1",)

    controller._flush_pending_slider_updates_for_run(reset_set_ids=("id1",))

    species_timer.stop.assert_called_once()
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert tuple(controller.run_state.pending_slider_target_set_ids) == ("id1",)


@pytest.mark.unit
def test_run_simulation_from_slider_discards_stale_request(mw: _FakeMainWindow, controller: SimulationController):
    controller._pending_slider_sim_request_id = 1
    controller._latest_sim_request_id = 2

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None


@pytest.mark.unit
def test_run_simulation_from_slider_promotes_reserved_future_request_to_latest(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    controller._latest_sim_request_id = 5
    controller._pending_slider_sim_request_id = 6
    mw._run_btn.setEnabled(True)

    controller._run_simulation_from_slider()

    assert controller._latest_sim_request_id == 6
    controller.run_simulation_internal.assert_called_once()
    assert controller.run_simulation_internal.call_args.kwargs["request_id"] == 6


@pytest.mark.unit
def test_run_simulation_from_slider_uses_snapshotted_target_rows(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._latest_sim_request_id = int(rid)
    mw._batch_rows_for_scope.return_value = [2]
    mw._batch_store.row_count.return_value = 3
    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id3"]
    mw._slider_gesture_target_set_ids_snapshot = ["id1", "id2"]
    mw._last_slider_change_name = "k1"

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["request_id"] == int(rid)
    assert kwargs["batch_rows"] == [0, 1]


@pytest.mark.unit
def test_run_simulation_from_slider_ignores_stale_mechanism_snapshot_for_species_preview(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._latest_sim_request_id = int(rid)
    mw._batch_rows_for_scope.return_value = [2]
    mw._batch_store.row_count.return_value = 3
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2", 2: "id3"}[int(row)]
    mw._slider_gesture_target_set_ids_snapshot = ["id1", "id2"]
    mw._last_slider_change_name = "init:A"

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["batch_rows"] == [2]


@pytest.mark.unit
def test_run_simulation_from_slider_preflight_abort_clears_slider_triggered_flag(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return ""

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation_from_slider()

    assert mw._slider_triggered_simulation is False


@pytest.mark.unit
def test_run_simulation_from_slider_defers_when_full_run_in_progress(mw: _FakeMainWindow, controller: SimulationController):
    mw._run_btn = _FakeButton(False)
    controller._simulation_worker = None
    controller._pending_slider_sim_request_id = None
    controller._latest_sim_request_id = 0
    controller._batch_run_context = {"active": False, "parallel": False}

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is True


@pytest.mark.unit
def test_cancel_active_run_for_restart_resets_ui_and_shuts_down(mw: _FakeMainWindow, controller: SimulationController):
    controller._batch_run_context = {"active": True}
    controller._shutdown_batch_executor = MagicMock()
    worker = _FakeWorker(running=True, wait_returns=True)
    controller._simulation_worker = worker

    controller._cancel_active_run_for_restart()
    assert controller._batch_run_context["active"] is False
    controller._shutdown_batch_executor.assert_called_once_with(force_terminate=True)
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_run_simulation_blocks_restart_while_retained_worker_is_still_running(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_rows_for_scope.return_value = [0]
    controller.run_simulation_internal = MagicMock()
    controller._batch_run_context = {"active": True}
    controller._shutdown_batch_executor = MagicMock()
    worker = make_stubborn_worker(_FakeWorker)
    controller._simulation_worker = worker
    controller._simulation_running = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)

    controller.run_simulation()
    controller.run_simulation()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert controller.has_running_owned_simulation_workers() is True
    assert controller._retained_simulation_workers == [worker]
    assert mw._status_label.text == "Cancelling previous simulation..."
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_run_simulation_reuses_parallel_executor_for_explicit_multi_set_runs(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_rows_for_scope.return_value = [0, 1]
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    controller.run_simulation_internal.assert_called_once()
    _args, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is False
    assert kwargs["batch_rows"] == [0, 1]
    assert kwargs["reuse_parallel_executor"] is True


@pytest.mark.unit
def test_run_auto_locks_editor(mw: _FakeMainWindow, controller: SimulationController):
    mw._batch_rows_for_scope.return_value = [0]
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._auto_lock_for_run_calls == 1
    controller.run_simulation_internal.assert_called_once()


@pytest.mark.unit
def test_run_aborts_if_mechanism_invalid_while_unlocked(mw: _FakeMainWindow, controller: SimulationController):
    mw._batch_rows_for_scope.return_value = [0]
    mw._auto_lock_for_run_result = False
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._auto_lock_for_run_calls == 1
    controller.run_simulation_internal.assert_not_called()
    assert mw._status_label.text == "Cannot run: mechanism has errors. Fix and try again."


@pytest.mark.unit
def test_start_parallel_batch_simulations_falls_back_to_serial_when_executor_factory_fails(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._batch_parallel.executor = None
    controller.parallel_batch.executor_factory = MagicMock(side_effect=RuntimeError("no executor"))
    controller._start_next_batch_simulation = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "run_id": 1,
        "effective_workers": 2,
    }

    controller._start_parallel_batch_simulations()
    assert controller._batch_run_context["parallel"] is False
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_run_simulation_internal_builds_context_and_calls_start_next(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return "state: A, kind=GS, energy=0"

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\ninitial: A=1")
            self._state_network_editor = _StateNetworkEditor()

    batch_names = ["set1"]
    mw._batch_store.row_count.side_effect = lambda: len(batch_names)
    mw._batch_store.set_names.side_effect = lambda: list(batch_names)
    mw._batch_store.ensure_set.side_effect = (
        lambda name: batch_names.index(str(name))
        if str(name) in batch_names
        else (batch_names.append(str(name)) or (len(batch_names) - 1))
    )
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": (
            {"randomname3": {"A": 1.0}},
            text.replace("initial: A=1", "# Initial concentrations moved to Batch Initial Conditions table (randomname3). Edit there."),
        ),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text.replace("initial:", "# stripped initial:"),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._apply_parameter_override_fallback_to_dsl = MagicMock(
        side_effect=lambda text, *, set_id=None: str(text).replace("PRIMARY", str(set_id))
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()
    mw.discard_concentration_overlays_for_rows.return_value = True

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    ctx = controller._batch_run_context
    assert ctx["active"] is True
    assert ctx["parallel"] is False
    assert isinstance(ctx["cache_key"], str)
    assert ctx["cache_key"] != "ck"
    assert "# State Network" in ctx["full_dsl"]
    assert ctx["queue_names"] == ["randomname3"]
    assert ctx["pending_init_seed"] == {"randomname3": {"A": 1.0}}
    assert isinstance(ctx["pending_init_rewrite"], str) and ctx["pending_init_rewrite"]
    assert ctx["pending_init_applied"] is True
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_run_simulation_internal_merges_empty_default_named_block_with_legacy_initials(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 1

    class _Text:
        def __init__(self, text):
            self._text = text

        def toPlainText(self):
            return self._text

        def setPlainText(self, value):
            self._text = str(value)

    class _StateNetworkEditor:
        def get_dsl_text(self) -> str:
            return ""

        def getPlainText(self) -> str:
            return ""

        def setPlainText(self, value: str) -> None:
            self._text = str(value)

        def toPlainText(self) -> str:
            return getattr(self, "_text", "")

        def get_state_network_dsl(self) -> str:
            return "state: A, kind=GS, energy=0"

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text(
                "reaction: A -> B; k=1\n\nset1 = {\n}\n\n# Initial concentrations\n[A] = 1.0\n"
            )
            self._state_network_editor = _StateNetworkEditor()

    batch_names = ["set1"]
    mw._batch_store.row_count.side_effect = lambda: len(batch_names)
    mw._batch_store.set_names.side_effect = lambda: list(batch_names)
    mw._batch_store.ensure_set.side_effect = (
        lambda name: batch_names.index(str(name))
        if str(name) in batch_names
        else (batch_names.append(str(name)) or (len(batch_names) - 1))
    )
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._apply_parameter_override_fallback_to_dsl = MagicMock(
        side_effect=lambda text, *, set_id=None: str(text).replace("PRIMARY", str(set_id))
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    ctx = controller._batch_run_context

    assert ctx["pending_init_seed"] == {"set1": {"A": 1.0}}
    assert isinstance(ctx["pending_init_rewrite"], str) and ctx["pending_init_rewrite"]
    assert ctx["pending_init_applied"] is True
    assert ctx["pending_init_rewrite"].count(
        "Initial concentrations moved to Batch Initial Conditions table (set1). Edit there."
    ) == 2
    mw._batch_store.set_value.assert_any_call(0, "A", "1")
    mw._set_text_with_optional_undo.assert_called()
    rewritten = str(mw._set_text_with_optional_undo.call_args.args[1])
    assert "set1 = {" not in rewritten
    assert "[A] = 1.0" not in rewritten
    assert rewritten.count(
        "Initial concentrations moved to Batch Initial Conditions table (set1). Edit there."
    ) == 2


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_isolates_prepared_payloads_per_set(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime()
        created_runtimes.append(runtime)
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 4.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 2.5}, {"A": 5.5}])
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._apply_parameter_override_fallback_to_dsl = MagicMock(
        side_effect=lambda text, *, set_id=None: str(text).replace("PRIMARY", str(set_id))
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_executor=False)

    prepared_by_set_id = controller._batch_run_context["prepared_by_set_id"]
    execution_request_by_set_id = controller._batch_run_context["execution_request_by_set_id"]
    assert controller._batch_run_context["prepared"] is None
    assert controller._batch_run_context["execution_request"] is None
    assert prepared_by_set_id["id1"]["mechanism"]["bound_set_id"] == "id1"
    assert prepared_by_set_id["id2"]["mechanism"]["bound_set_id"] == "id2"
    assert execution_request_by_set_id["id1"]["prepared_payload"]["mechanism"]["bound_set_id"] == "id1"
    assert execution_request_by_set_id["id2"]["prepared_payload"]["mechanism"]["bound_set_id"] == "id2"
    assert execution_request_by_set_id["id1"]["initials"] == {"A": 2.5}
    assert execution_request_by_set_id["id2"]["initials"] == {"A": 5.5}
    assert created_runtimes[0] is not created_runtimes[1]


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_refreshes_runtime_after_multi_set_preview(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.mechanism = {"runtime": self.label}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime(label=f"runtime-{len(created_runtimes) + 1}")
        created_runtimes.append(runtime)
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.side_effect = lambda rows: "id2" if list(rows) == [1] else "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._apply_parameter_override_fallback_to_dsl = MagicMock(
        side_effect=lambda text, *, set_id=None: str(text).replace("PRIMARY", str(set_id))
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_executor=False)

    assert len(created_runtimes) == 2

    controller._batch_run_context = {"active": False}
    controller._run_simulation_internal(fast_mode=True, request_id=8, batch_rows=[1], reuse_parallel_executor=False)

    # After a multi-set loop the runtime is marked dirty so a subsequent
    # single-set run creates a fresh runtime rather than reusing the last
    # set's bindings.
    assert len(created_runtimes) == 3


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_marks_runtime_dirty_after_multi_set_loop(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """After a multi-set fast-mode loop, the runtime must be marked dirty.

    If the runtime remains "clean" after the last iteration, a subsequent
    single-set slider tick could reuse bindings from the wrong set.
    """
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.mechanism = {"runtime": self.label}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime(label=f"runtime-{set_id}")
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.side_effect = lambda rows: "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=10, batch_rows=[0, 1], reuse_parallel_executor=False)

    # After the multi-set loop, the runtime must be marked dirty so the next
    # single-set interaction forces a fresh prepare instead of reusing the
    # last set's bindings.
    assert mw._slider_runtime_dirty is True, (
        "Runtime was left 'clean' after multi-set loop; a subsequent single-set "
        "interaction would reuse the last set's bindings"
    )


@pytest.mark.unit
def test_fast_preview_completion_uses_dispatch_time_overlay_token_snapshot(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    row_to_set_id = {0: "id1", 1: "id2"}

    def _preview_token(rows: list[int]) -> str:
        tokens: list[str] = []
        for row in rows or []:
            set_id = row_to_set_id.get(int(row))
            if set_id:
                tokens.append(f"token:{set_id}")
        return "|".join(tokens)

    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: row_to_set_id[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(side_effect=_preview_token)

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()
    controller._latest_sim_request_id = 7
    controller._queue_slider_plot_update = MagicMock()
    mw._mechanism_editor._reactions_text = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_executor=False)

    assert controller._batch_run_context["preview_batch_cache_token_by_set_id"] == {
        "id1": "token:id1",
        "id2": "token:id2",
    }

    row_to_set_id.clear()
    row_to_set_id.update({0: "id9", 1: "id1", 2: "id2"})

    result = _successful_result_payload()
    cache_key = str(controller._batch_run_context["cache_key"])
    controller._on_simulation_complete(
        result,
        run_id=None,
        fast_mode=True,
        request_id=7,
        batch_set="set1",
        batch_set_id="id1",
        cache_key=cache_key,
    )

    payload = controller.batch_cache.preview_cache.get(f"{cache_key}::id1")
    assert isinstance(payload, dict)
    assert payload.get("preview_batch_cache_token") == "token:id1"


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_parallel_signatures_follow_preview_mechanism_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    def _signature(**kwargs):
        mechanism_text = str(kwargs.get("mechanism_text") or "")
        if mechanism_text:
            return f"text:{mechanism_text}"
        identity = dict(kwargs.get("simulation_identity") or {})
        return f"id:{identity.get('param_fingerprint')}"

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._parse_sim_time_seconds.return_value = 10.0
    mw._simulation_param_fingerprints = {"id1": "params-a", "id2": "params-b"}
    mw._slider_overrides = {"k1": 2.0}
    mw.apply_overrides_to_text = MagicMock(
        side_effect=lambda text, *, set_id=None: f"{text}\n# preview {set_id} k={mw._slider_overrides['k1']}"
    )
    mw.apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text, *, set_id=None: str(text))

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        _signature,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 2,
    )
    controller._start_parallel_batch_simulations = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=7,
        batch_rows=[0, 1],
        reuse_parallel_executor=False,
    )
    first_ctx = dict(controller._batch_run_context)

    mw._slider_overrides = {"k1": 5.0}
    controller._batch_run_context = {"active": False}
    controller._run_simulation_internal(
        fast_mode=True,
        request_id=8,
        batch_rows=[0, 1],
        reuse_parallel_executor=False,
    )
    second_ctx = dict(controller._batch_run_context)

    first_text = dict(first_ctx["mechanism_text_by_set_id"])
    second_text = dict(second_ctx["mechanism_text_by_set_id"])
    first_sig = dict(first_ctx["mechanism_signature_by_set_id"])
    second_sig = dict(second_ctx["mechanism_signature_by_set_id"])

    assert first_sig == {
        set_id: f"text:{text}"
        for set_id, text in first_text.items()
    }
    assert second_sig == {
        set_id: f"text:{text}"
        for set_id, text in second_text.items()
    }
    assert first_sig["id1"] != second_sig["id1"]
    assert first_sig["id2"] != second_sig["id2"]


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_keeps_scalar_override_in_worker_dsl_when_bindings_cannot_apply(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\n# Algebra\nparam a = 5\n")
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"a": 2.0}
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text: str(text))
    mw._apply_parameter_overrides_to_dsl = MagicMock(
        side_effect=lambda text, parameters: str(text).replace(
            "param a = 5",
            f"param a = {format_authoritative_parameter_value(parameters['a'])}",
        )
    )
    mw._prepare_slider_runtime = MagicMock(return_value=object())
    mw._apply_slider_overrides_to_bindings = MagicMock(return_value=False)
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=8, batch_rows=[0], reuse_parallel_executor=False)

    assert "param a = 2" in controller._batch_run_context["mechanism_text_by_set_id"]["id1"]
    mw._apply_parameter_overrides_to_dsl.assert_called()


@pytest.mark.unit
def test_run_simulation_internal_explicit_run_uses_overlay_cache_token(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "baseline-cache"
    mw._simulation_schema_id = "schema-explicit"
    mw._simulation_param_fingerprints = {"id1": "params-id1"}
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="set:id1|A=2.5")

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    solver_cfg = controller._batch_run_context["solver_config"]
    expected = SimulationScopeIdentity.build(
        queue_ids=["id1"],
        identity_by_set_id={
            "id1": SimulationIdentity.build(
                schema_id="schema-explicit",
                param_fingerprint="",
                solver_config=dict(solver_cfg),
                t_end=10.0,
                preview_batch_cache_token="",
                execution_flags=(),
            )
        },
    ).cache_key()
    assert controller._batch_run_context["cache_key"] == expected
    assert controller.batch_cache.active_cache_key == expected
    assert controller.batch_cache.active_cache_preview_token is None
    assert controller.batch_cache.active_cache_preview_scope_set_ids is None
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)
    assert mw.preview_batch_cache_token.call_args_list == []
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_run_simulation_internal_explicit_cache_key_ignores_non_primary_set_fingerprint_changes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="")
    mw._simulation_schema_id = "schema-explicit"
    mw._simulation_param_fingerprints = {"id1": "params-id1", "id2": "params-id2a"}

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0, 1], reuse_parallel_executor=False)
    first_key = str(controller._batch_run_context["cache_key"])

    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id1", "id2"]
    mw._simulation_param_fingerprints = {"id1": "params-id1", "id2": "params-id2b"}
    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0, 1], reuse_parallel_executor=False)
    second_key = str(controller._batch_run_context["cache_key"])

    assert first_key == second_key


@pytest.mark.unit
def test_run_simulation_internal_baseline_explicit_run_leaves_overlay_cache_token_empty(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "baseline-cache"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="")

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert isinstance(controller._batch_run_context["cache_key"], str)
    assert controller._batch_run_context["cache_key"] != "baseline-cache"
    assert controller.batch_cache.active_cache_key == controller._batch_run_context["cache_key"]
    assert controller.batch_cache.active_cache_preview_token is None
    assert controller.batch_cache.active_cache_preview_scope_set_ids is None
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)
    assert mw.preview_batch_cache_token.call_args_list == []
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_start_next_batch_simulation_explicit_run_ignores_staged_concentration_overlay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["t_span"] = tuple(t_span)
            created["solver_config"] = dict(solver_config)
            created["prepared"] = prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id2"],
        "queue_names": ["set2"],
        "full_dsl": "reaction: A -> B; k=1",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,
        "request_id": 7,
        "cache_key": "explicit-cache",
        "pending_init_seed": {},
            "pending_init_applied": True,
        }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["initials"] == {"A": 1.0}
    assert created["started"] is True
    mw.preview_initials_for_row.assert_not_called()


@pytest.mark.unit
def test_start_parallel_batch_simulations_explicit_run_ignores_staged_concentration_overlay(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    class _FakeParallelFuture:
        def add_done_callback(self, _callback) -> None:
            return

        def done(self) -> bool:
            return False

    class _FakeExecutor:
        def submit(self, _fn, *args, **_kwargs):
            if args:
                submitted.append(dict(args[0]))
            return _FakeParallelFuture()

        def shutdown(self, *args, **kwargs) -> None:
            _ = args, kwargs
            return

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller.parallel_batch.executor_factory = MagicMock(return_value=_FakeExecutor())
    controller._batch_parallel.executor = None
    controller._batch_parallel.future_map = {}
    controller._batch_parallel.future_meta = {}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["id2"],
        "queue_names": ["set2"],
        "run_id": 3,
        "request_id": 11,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    controller._start_parallel_batch_simulations()

    assert submitted and submitted[0]["initials"] == {"A": 1.0}
    mw.preview_initials_for_row.assert_not_called()


@pytest.mark.unit
def test_start_parallel_batch_simulations_marks_only_primary_explicit_result_for_mechanism_payload(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    class _FakeParallelFuture:
        def add_done_callback(self, _callback) -> None:
            return

        def done(self) -> bool:
            return False

    class _FakeExecutor:
        def submit(self, _fn, *args, **_kwargs):
            if args:
                submitted.append(dict(args[0]))
            return _FakeParallelFuture()

        def shutdown(self, *args, **kwargs) -> None:
            _ = args, kwargs
            return

    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 2.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 1.0}, {"A": 2.0}])
    controller.parallel_batch.executor_factory = MagicMock(return_value=_FakeExecutor())
    controller._batch_parallel.executor = None
    controller._batch_parallel.future_map = {}
    controller._batch_parallel.future_meta = {}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0, 1],
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "run_id": 3,
        "request_id": 11,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {"id1": "sig-1", "id2": "sig-2"},
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    controller._start_parallel_batch_simulations()

    assert len(submitted) == 2
    by_set_id = {str(task["set_id"]): task for task in submitted}
    assert by_set_id["id1"]["include_mechanism_in_result_payload"] is True
    assert by_set_id["id2"]["include_mechanism_in_result_payload"] is False


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_uses_set_specific_prepared_payload_and_mechanism_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 1,
        "rows": [0, 1],
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_text_by_set_id": {
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        },
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {
            "id1": "sig-2",
            "id2": "sig-3",
        },
        "prepared": None,
        "prepared_by_set_id": {
            "id1": {"prepared_for": "id1"},
            "id2": {"prepared_for": "id2"},
        },
        "execution_request_by_set_id": {
            "id2": {
                "prepared_payload": {"version": 2, "prepared_for": "id2"},
                "initials": {"A": 1.5},
                "t_span": (0.0, 10.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "reaction: A -> B; k=3",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        },
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": True,
        "request_id": 7,
        "cache_key": "slider-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["prepared"] == {"prepared_for": "id2"}
    assert created["started"] is True
    worker = controller._simulation_worker
    assert getattr(worker, "_execution_request", None) is not None
    assert worker._execution_request["prepared_payload"] == {"version": 2, "prepared_for": "id2"}  # type: ignore[index]
    assert worker._execution_request["initials"] == {"A": 1.5}  # type: ignore[index]
    assert worker._execution_request["mechanism_text"] == "reaction: A -> B; k=3"  # type: ignore[index]


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_does_not_borrow_batch_global_prepared_payload(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 1,
        "rows": [0, 1],
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_text_by_set_id": {
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        },
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {
            "id1": "sig-2",
            "id2": "sig-3",
        },
        "prepared": {"prepared_for": "id1"},
        "prepared_by_set_id": {
            "id1": {"prepared_for": "id1"},
        },
        "execution_request": None,
        "execution_request_by_set_id": {},
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": True,
        "request_id": 11,
        "cache_key": "slider-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["initials"] == {"A": 4.0}
    assert created["started"] is True
    assert getattr(controller._simulation_worker, "_execution_request", None) is None


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_does_not_borrow_batch_global_execution_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["solver_config"] = dict(solver_config)
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 1,
        "rows": [0, 1],
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_text_by_set_id": {
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        },
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {
            "id1": "sig-2",
            "id2": "sig-3",
        },
        "prepared": None,
        "prepared_by_set_id": {},
        "execution_request": {
            "prepared_payload": {"version": 2, "prepared_for": "id1"},
            "initials": {"A": 9.0},
            "t_span": (0.0, 10.0),
            "solver_config": {"solver": "BDF"},
            "mechanism_text": "reaction: A -> B; k=2",
            "simulation_identity": {"schema_id": "schema-a", "param_fingerprint": "fingerprint-a"},
        },
        "execution_request_by_set_id": {
            "id1": {
                "prepared_payload": {"version": 2, "prepared_for": "id1"},
                "initials": {"A": 9.0},
                "t_span": (0.0, 10.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "reaction: A -> B; k=2",
                "simulation_identity": {"schema_id": "schema-a", "param_fingerprint": "fingerprint-a"},
            },
        },
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": True,
        "request_id": 12,
        "cache_key": "slider-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["initials"] == {"A": 4.0}
    assert created["solver_config"] == {"solver": "BDF"}
    assert created["started"] is True
    assert getattr(controller._simulation_worker, "_execution_request", None) is None


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_reapplies_parameter_override_fallback_when_prepared_missing(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    mw._slider_overrides = {"a": 2.0}
    mw._apply_parameter_overrides_to_dsl = MagicMock(
        side_effect=lambda mechanism_text, parameters: str(mechanism_text).replace(
            "param a = 5",
            f"param a = {format_authoritative_parameter_value(parameters['a'])}",
        )
    )
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "full_dsl": "reaction: A -> B; k=1\n# Algebra\nparam a = 5\n",
        "mechanism_text_by_set_id": {
            "id1": "reaction: A -> B; k=1\n# Algebra\nparam a = 5\n",
        },
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {"id1": "sig-1"},
        "prepared": None,
        "prepared_by_set_id": {},
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": True,
        "request_id": 9,
        "cache_key": "slider-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert "param a = 2" in str(created["mechanism_text"])
    mw._apply_parameter_overrides_to_dsl.assert_called_once()
    assert created["started"] is True


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_fallback_cache_key_ignores_rewritten_worker_dsl_witness(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            _ = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            return

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    mw._slider_overrides = {"a": 2.0}
    mw._simulation_schema_id = "schema-preview"
    mw._simulation_param_fingerprints = {"id1": "params-id1"}
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "full_dsl": "reaction: A -> B; k=1\n# Algebra\nparam a = 5\n",
        "mechanism_text_by_set_id": {
            "id1": "reaction: A -> B; k=1\n# Algebra\nparam a = 5\n",
        },
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {"id1": "sig-1"},
        "prepared": None,
        "prepared_by_set_id": {},
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": True,
        "request_id": 9,
        "cache_key": "slider-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
        "simulation_identity_by_set_id": {
            "id1": {
                "version": 1,
                "schema_id": "schema-preview",
                "param_fingerprint": "params-id1",
                "solver": {
                    "solver": "BDF",
                    "rtol": 1e-6,
                    "atol": 1e-12,
                    "grid_n": 100,
                    "temperature_K": 298.15,
                    "use_sparse_jacobian": False,
                    "wegscheider_cyclicity_enabled": False,
                },
                "t_end": 10.0,
                "preview_batch_cache_token": "",
                "execution_flags": ("fast_mode",),
            }
        },
    }

    rewritten_texts = [
        "reaction: A -> B; k=1\n# Algebra\nparam a = 2\n# witness one\n",
        "reaction: A -> B; k=1\n# Algebra\nparam a = 2\n# witness two\n",
    ]
    mw._apply_parameter_overrides_to_dsl = MagicMock(side_effect=list(rewritten_texts))

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()
    first_key = str(controller._batch_run_context["cache_key"])

    controller._batch_run_context["pos"] = 0
    controller._batch_run_context["cache_key"] = "slider-cache"
    controller._start_next_batch_simulation()
    second_key = str(controller._batch_run_context["cache_key"])

    assert first_key == second_key


@pytest.mark.unit
def test_start_next_batch_simulation_explicit_run_uses_canonical_pending_init_seed(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["initials"] = dict(initials)
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 0.25}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["randomname3"],
        "full_dsl": "reaction: A -> B; k=1",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,
        "request_id": 7,
        "cache_key": "explicit-cache",
        "pending_init_seed": {"randomname3": {"A": 1.0}},
        "pending_init_applied": False,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["initials"] == {"A": 1.0}
    mw.preview_initials_for_row.assert_not_called()


@pytest.mark.unit
def test_start_parallel_batch_simulations_explicit_run_uses_canonical_pending_init_seed(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    class _FakeParallelFuture:
        def add_done_callback(self, _callback) -> None:
            return

        def done(self) -> bool:
            return False

    class _FakeExecutor:
        def submit(self, _fn, *args, **_kwargs):
            if args:
                submitted.append(dict(args[0]))
            return _FakeParallelFuture()

        def shutdown(self, *args, **kwargs) -> None:
            _ = args, kwargs
            return

    mw._batch_initials_for_row.return_value = {"A": 0.25}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller.parallel_batch.executor_factory = MagicMock(return_value=_FakeExecutor())
    controller._batch_parallel.executor = None
    controller._batch_parallel.future_map = {}
    controller._batch_parallel.future_meta = {}
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["randomname3"],
        "run_id": 3,
        "request_id": 11,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {"randomname3": {"A": 1.0}},
        "pending_init_applied": False,
    }

    controller._start_parallel_batch_simulations()

    assert submitted and submitted[0]["initials"] == {"A": 1.0}
    mw.preview_initials_for_row.assert_not_called()


@pytest.mark.unit
def test_parallel_batch_pool_settings_changed_shuts_down_idle_pool_immediately(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _FakeExecutor:
        def __init__(self) -> None:
            self._max_workers = 2
            self.shutdown_calls: list[dict[str, object]] = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    fake = _FakeExecutor()
    controller.parallel_batch.executor = fake
    controller.batch_run_context = {}
    controller.parallel_batch.future_map = {}
    controller.parallel_batch.superseded_future_map = {}

    controller.parallel_batch_pool_settings_changed()

    assert fake.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert controller.parallel_batch.executor is None
    assert controller._pool_eagerly_created is False


@pytest.mark.unit
def test_parallel_batch_pool_settings_changed_defers_shutdown_until_parallel_completion(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self._max_workers = int(max_workers)
            self.shutdown_calls: list[dict[str, object]] = []

        def submit(self, _fn, *_args, **_kwargs):
            return object()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    current = _FakeExecutor(2)
    created: list[tuple[int, bool, _FakeExecutor]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        executor = _FakeExecutor(max_workers)
        created.append((int(max_workers), bool(limit_blas_threads), executor))
        return executor

    controller.parallel_batch.executor = current
    controller.parallel_batch.executor_factory = _factory
    controller.parallel_batch.max_parallel_workers = 6
    controller.batch_run_context = {
        "active": True,
        "parallel": True,
        "keep_executor_alive": True,
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "completed_set_ids": [],
        "total": 1,
        "fast_mode": False,
        "primary_set_id": "other-id",
    }
    controller._active_run_id = 3
    controller.run_state.latest_sim_request_id = 11
    mw._batch_current_row.return_value = None
    monkeypatch.setattr(controller, "_resolve_completion_mechanism", MagicMock(return_value=None))
    monkeypatch.setattr(controller, "_update_primary_result_materialization_contract", MagicMock(return_value=False))

    controller.parallel_batch_pool_settings_changed()

    assert controller.parallel_batch.executor is current
    assert controller.parallel_batch.is_pool_stale is True
    assert current.shutdown_calls == []

    controller.on_simulation_complete(
        {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.asarray([[1.0, 1.0]], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "",
            "solver_config": {},
            "fallback_occurred": False,
            "fallback_message": None,
        },
        run_id=3,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="cache-key",
    )

    assert current.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert controller.parallel_batch.executor is None

    recreated = controller.parallel_batch.ensure_executor(max_workers=6)

    assert created == [(6, True, recreated)]
    assert controller.parallel_batch.executor is recreated


@pytest.mark.unit
def test_ensure_parallel_batch_pool_eagerly_created_only_once(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    created: list[tuple[int, bool]] = []

    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self._max_workers = int(max_workers)

        def submit(self, _fn, *_args, **_kwargs):
            return object()

        def shutdown(self, *args, **kwargs):
            _ = args, kwargs
            return None

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers)

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.executor_factory = _factory
    controller._pool_eagerly_created = False
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_pool_eagerly_created()
    first = controller.parallel_batch.executor
    controller.ensure_parallel_batch_pool_eagerly_created()
    controller.parallel_batch_pool_settings_changed()
    controller.ensure_parallel_batch_pool_eagerly_created()

    assert created == [(3, True), (3, True)]
    assert controller.parallel_batch.executor is not None
    assert controller.parallel_batch.executor is not first


@pytest.mark.unit
def test_ensure_parallel_batch_pool_eagerly_created_retries_after_failure(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    attempts: list[tuple[int, bool]] = []
    recorded: list[tuple[str, str]] = []

    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self._max_workers = int(max_workers)

        def submit(self, _fn, *_args, **_kwargs):
            return object()

        def shutdown(self, *args, **kwargs):
            _ = args, kwargs
            return None

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        attempts.append((int(max_workers), bool(limit_blas_threads)))
        if len(attempts) == 1:
            raise RuntimeError("factory boom")
        return _FakeExecutor(max_workers)

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.executor_factory = _factory
    controller.parallel_batch.record_nonfatal_exception = _record
    controller._record_nonfatal_exception = _record
    controller._pool_eagerly_created = False
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_pool_eagerly_created()

    assert attempts == [(3, True)]
    assert controller.parallel_batch.executor is None
    assert controller._pool_eagerly_created is False
    assert recorded == [("Failed to create and prewarm batch executor", "factory boom")]

    controller.ensure_parallel_batch_pool_eagerly_created()

    assert attempts == [(3, True), (3, True)]
    assert controller.parallel_batch.executor is not None
    assert controller._pool_eagerly_created is True


@pytest.mark.unit
def test_ensure_parallel_batch_pool_eagerly_created_prewarm_failure_records_once(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    recorded: list[tuple[str, str]] = []

    class _SubmitFailExecutor:
        def __init__(self, max_workers: int) -> None:
            self._max_workers = int(max_workers)
            self.shutdown_calls: list[dict[str, object]] = []

        def submit(self, _fn, *_args, **_kwargs):
            raise RuntimeError("submit boom")

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    created: list[_SubmitFailExecutor] = []

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _SubmitFailExecutor:
        executor = _SubmitFailExecutor(max_workers)
        created.append(executor)
        return executor

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.executor_factory = _factory
    controller.parallel_batch.record_nonfatal_exception = _record
    controller._record_nonfatal_exception = _record
    controller._pool_eagerly_created = False
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_pool_eagerly_created()

    assert len(created) == 1
    assert created[0].shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert controller.parallel_batch.executor is None
    assert controller._pool_eagerly_created is False
    assert recorded == [("Failed to create and prewarm batch executor", "submit boom")]


@pytest.mark.unit
def test_poll_parallel_batch_futures_shuts_down_stale_pool_after_superseded_futures_drain(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _SupersededFuture:
        def done(self) -> bool:
            return True

        def result(self):
            return {"ok": True}

    class _FakeExecutor:
        def __init__(self) -> None:
            self.shutdown_calls: list[dict[str, object]] = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append(
                {
                    "wait": bool(wait),
                    "cancel_futures": bool(cancel_futures),
                }
            )

    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_future_poll_timer = timer
    executor = _FakeExecutor()
    controller.parallel_batch.executor = executor
    controller.parallel_batch.mark_pool_stale()
    controller._batch_run_context = {"active": False, "parallel": False}
    controller._batch_parallel.future_map = {}
    controller._batch_parallel.superseded_future_map = {"sid": _SupersededFuture()}
    controller._batch_parallel.superseded_future_meta = {"sid": {"set_id": "sid", "set_name": "set1", "superseded": "1"}}

    controller._poll_parallel_batch_futures()

    assert controller.parallel_batch.superseded_future_map == {}
    assert controller.parallel_batch.superseded_future_meta == {}
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert controller.parallel_batch.executor is None
    assert timer.stop.called


@pytest.mark.unit
def test_start_next_batch_simulation_invalid_initials_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_initials_for_row.side_effect = ValueError("bad initials")
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "full_dsl": "reaction: A -> B; k=1",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,
        "request_id": 7,
        "cache_key": "explicit-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    controller._start_next_batch_simulation()

    assert warned == [("Invalid Initial Conditions", "Set 'set1' has invalid initial conditions:\n\nbad initials")]
    mw._batch_model.validate_rows.assert_called_once_with([0])
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    assert controller._batch_run_context["pending_init_applied"] is False


@pytest.mark.unit
def test_start_parallel_batch_simulations_invalid_initials_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    class _FakeExecutor:
        def submit(self, *_args, **_kwargs):
            raise AssertionError("submit should not be reached when initials are invalid")

        def shutdown(self, *args, **kwargs) -> None:
            _ = args, kwargs
            return

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_initials_for_row.side_effect = ValueError("bad initials")
    controller._batch_parallel.executor = _FakeExecutor()
    controller._shutdown_batch_executor = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": True,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "run_id": 3,
        "request_id": 11,
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_signature": "sig",
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "effective_workers": 2,
        "fast_mode": False,
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    controller._start_parallel_batch_simulations()

    assert warned == [("Invalid Initial Conditions", "Set 'set1' has invalid initial conditions:\n\nbad initials")]
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._shutdown_batch_executor.assert_called_once_with(force_terminate=True)
    assert controller._batch_run_context["pending_init_applied"] is False


@pytest.mark.unit
def test_run_simulation_internal_aborts_and_unlocks_on_invalid_batch_rows(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\ninitial: A=1")
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._parse_sim_time_seconds.return_value = 10.0

    mw._batch_model.validate_rows.return_value = {(0, "A")}

    class _MechTmp:
        def species_names(self):
            return ["A", "B"]

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", lambda *_a, **_k: _MechTmp())
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({"set1": {"A": 1.0}}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()
    controller._simulation_running = True
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pending_init_applied": True,
    }
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "Invalid Initial Conditions"
    assert controller._simulation_running is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    assert controller._batch_run_context["pending_init_applied"] is False
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_preview_mode_caps_points(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

        def slider_points_value(self) -> int:
            return 500

        def slider_solver_value(self) -> str:
            return "Radau"

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0
    mw._slider_drag_active = True
    mw._last_slider_change_name = "Keq12"
    mw._batch_store.visible_species.return_value = ["A"]

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    solver_cfg = controller._batch_run_context["solver_config"]
    assert solver_cfg["solver"] == "Radau"
    assert int(solver_cfg["grid"]["N"]) <= 120
    assert int(solver_cfg["grid"]["N"]) >= 50


@pytest.mark.unit
def test_run_simulation_internal_invalid_t_end_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "Invalid t_end"
    mw.reset_mechanism_workspaces.assert_not_called()
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_invalid_t_end_does_not_schedule_pending_slider_replay_after_preflight_abort(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []
    scheduled: list[object] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._latest_sim_request_id = 4
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 3
    controller.run_state.pending_slider_target_set_ids = ("id1",)
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "Invalid t_end"
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id > 4
    assert tuple(getattr(controller.run_state, "pending_slider_target_set_ids", ())) == ("id1",)
    assert scheduled == []
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_invalid_t_end_reinvalidates_preserved_pending_init_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1\ninitial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "Invalid t_end"
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_no_mechanism_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "initial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "No Mechanism"
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._start_next_batch_simulation.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_invalid_initials_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    class _MechTmp:
        def species_names(self):
            return ["A", "B"]

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", lambda *_a, **_k: _MechTmp())

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._batch_initials_for_row.side_effect = ValueError("bad initials")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert warned and warned[0][0] == "Invalid Initial Conditions"
    mw.reset_mechanism_workspaces.assert_not_called()
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_explicit_run_worker_error_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert controller._batch_run_context.get("pending_workspace_reset_set_ids") == ["id1"]
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.discard_concentration_overlays_for_rows.assert_not_called()

    controller._on_simulation_error(
        {"kind": "simulation_error", "message": "ode build failed"},
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.discard_concentration_overlays_for_rows.assert_not_called()


@pytest.mark.unit
def test_explicit_run_worker_error_reinvalidates_preserved_pending_init_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1\ninitial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    controller._on_simulation_error(
        {"kind": "simulation_error", "message": "ode build failed"},
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()


@pytest.mark.unit
def test_explicit_run_success_clears_targeted_dirty_workspaces_after_completion(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert controller._batch_run_context.get("pending_workspace_reset_set_ids") == ["id1"]
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.discard_concentration_overlays_for_rows.assert_not_called()

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_rows.assert_not_called()
    assert mw._sync_batch_species_columns.call_count == 2
    assert mw._sync_batch_species_columns.call_args_list[-1] == call(["A", "B"], preserve_active_cache=True)


@pytest.mark.unit
def test_explicit_run_success_clears_targeted_concentration_overlays_by_set_id_after_row_reorder(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}.get(int(row))
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    assert controller._batch_run_context.get("pending_workspace_reset_set_ids") == ["id1"]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id2", 1: "id1"}.get(int(row))

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_rows.assert_not_called()


@pytest.mark.unit
def test_explicit_run_success_resyncs_focused_mechanism_controls_after_targeted_workspace_reset(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw._sync_mechanism_controls_to_focused_batch_set.assert_called_once_with(use_workspace=True)

@pytest.mark.unit
def test_explicit_run_success_clears_targeted_concentration_overlays_by_set_id_not_row(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)

    mw._batch_set_id_for_row.return_value = "id2"
    mw.discard_concentration_overlays_for_set_ids.return_value = True

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_rows.assert_not_called()


@pytest.mark.unit
def test_explicit_run_success_cancels_pending_species_preview_after_targeted_overlay_reset(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    assert mw.discard_concentration_overlays_for_set_ids.call_count == 1
    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None
    assert scheduled == []


@pytest.mark.unit
def test_explicit_run_success_preserves_pending_species_preview_replay_when_no_targeted_dirty_reset_occurred(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.reset_mechanism_workspaces.return_value = False
    mw.discard_concentration_overlays_for_set_ids.return_value = False
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id == 7
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_explicit_run_success_preserves_pending_slider_replay_for_non_targeted_dirty_set(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1, "id2": 3}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller.run_state.pending_slider_target_set_ids = ("id2",)

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None
    assert tuple(getattr(controller.run_state, "pending_slider_target_set_ids", ())) == ("id2",)
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_explicit_run_preflight_abort_does_not_schedule_pending_slider_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return ""

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_current_row.return_value = 0
    mw._last_slider_change_name = "k1"

    controller._latest_sim_request_id = 2
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 1
    controller.run_state.pending_slider_target_set_ids = ("id1", "id2")

    controller._run_simulation()

    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id > 2
    assert tuple(getattr(controller.run_state, "pending_slider_target_set_ids", ())) == ("id1", "id2")
    assert scheduled == []


@pytest.mark.unit
def test_explicit_run_success_requeues_surviving_pending_slider_replay_with_fresh_request_id(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1, "id2": 2}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._last_slider_change_name = "k1"

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0], reuse_parallel_executor=False)
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 1
    controller.run_state.pending_slider_target_set_ids = ("id1", "id2")

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    controller.run_simulation_internal = MagicMock()
    assert scheduled == [controller._run_simulation_from_slider]
    controller._latest_sim_request_id = 2
    scheduled[0]()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["batch_rows"] == [1]


@pytest.mark.unit
def test_explicit_run_success_preserves_targeted_dirty_state_edited_after_run_start(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_executor=False)
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    mw._dirty_state_generations["id1"] = 2

    controller._on_simulation_complete(
        _successful_result_payload(),
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id == 7
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_on_simulation_complete_updates_cache_and_marks_pending_init_applied(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    mw._slider_triggered_simulation = True
    controller._queue_slider_plot_update = MagicMock()
    mw._mechanism_editor = MagicMock()
    mw._mechanism_editor._reactions_text = MagicMock()

    mw._batch_store.ensure_set = MagicMock(return_value=0)
    mw._batch_store.set_value = MagicMock()

    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "queue_names": ["set1"],
        "queue_ids": ["id1"],
        "cache_key": "ck",
        "pending_init_seed": {"A": 1.0},
        "pending_init_rewrite": "reaction: A -> B; k=1",
        "pending_init_applied": False,
        "primary_set_id": "id1",
    }

    result = {
        "t": np.linspace(0.0, 1.0, 3),
        "Y": np.asarray([[1.0, 0.5, 0.1], [0.0, 0.5, 0.9]], dtype=float),
        "species_names": ["A", "B"],
        "mechanism": object(),
        "mechanism_text": "reaction: A -> B; k=1",
        "solver_config": {"solver": "Radau", "temperature_K": 298.15},
        "algebra_scalars": {},
        "algebra_errors": [],
    }

    controller._on_simulation_complete(
        result,
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    payload = controller.batch_cache.result_cache.get("ck::id1")
    assert isinstance(payload, dict)
    assert np.allclose(np.asarray(payload.get("t")), result["t"])
    assert controller._batch_run_context["pending_init_applied"] is True
    mw._arm_pending_init_result_invalidation_guard.assert_called_once_with(rewrite="reaction: A -> B; k=1")
    controller._queue_slider_plot_update.assert_called_once()


@pytest.mark.unit
def test_on_simulation_complete_uses_truthful_scipy_fallback_warning_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warnings: list[tuple[str, str]] = []

    def _fake_warning(_parent, title: str, message: str):
        warnings.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2

    result = _successful_result_payload()
    result["fallback_occurred"] = True
    result["fallback_message"] = "BDF failed; succeeded with Radau"

    controller._on_simulation_complete(
        result,
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    assert warnings == [("Solver fallback", warnings[0][1])]
    message = warnings[0][1]
    assert "BDF failed; succeeded with Radau" in message
    assert "alternative stiff SciPy solver" in message
    assert "RK4" not in message
    assert "fixed-step" not in message


@pytest.mark.unit
def test_on_simulation_error_cancelled_schedules_pending_slider(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    scheduled = {}

    def _fake_single_shot(_ms: int, fn: Callable[[], Any]) -> None:
        scheduled["fn"] = fn

    monkeypatch.setattr(QtCore.QTimer, "singleShot", _fake_single_shot)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_simulation = True

    controller._on_simulation_error(
        {"kind": "cancelled", "message": "Simulation cancelled by user"},
        run_id=2,
        fast_mode=True,
        request_id=1,
    )
    assert "fn" in scheduled
    mw._variable_update_timer.stop.assert_called_once_with()
    mw._species_slider_update_timer.stop.assert_called_once_with()


@pytest.mark.unit
def test_on_simulation_error_non_cancelled_explicit_requeues_preserved_pending_slider_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": False, "request_id": 5}
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._last_slider_change_name = "k1"
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 4
    controller.run_state.pending_slider_target_set_ids = ("id2",)

    controller._on_simulation_error(
        "boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id > 5
    assert tuple(getattr(controller.run_state, "pending_slider_target_set_ids", ())) == ("id2",)
    critical.assert_called_once()
    controller.run_simulation_internal = MagicMock()
    scheduled[0]()
    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["batch_rows"] == [1]


@pytest.mark.unit
def test_on_simulation_error_surfaces_stack_trace_as_dialog_details_and_log(
    caplog, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock()
    mw.message_box_critical = critical
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._batch_run_context = {"active": True, "parallel": False, "fast_mode": True, "request_id": 5}
    controller._simulation_running = True
    controller._slider_simulation_active = True

    stack_trace = "Traceback line 1\nTraceback line 2"

    with caplog.at_level("WARNING", logger="kindred.gui.controllers.simulation_controller"):
        controller._on_simulation_error(
            {
                "kind": "simulation_error",
                "message": "solver blew up",
                "context": {"stack_trace": stack_trace},
            },
            run_id=3,
            fast_mode=True,
            request_id=5,
        )

    critical.assert_called_once_with(
        "Simulation Error",
        "Simulation failed:\n\nsolver blew up",
        details=stack_trace,
    )
    assert "Traceback" not in critical.call_args.args[1]
    messages = [record.getMessage() for record in caplog.records]
    assert "Simulation error surfaced to UI: solver blew up" in messages
    assert stack_trace in messages
    assert mw._status_label.text == "Simulation failed"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_consume_parallel_batch_future_error_payload_calls_on_error(controller: SimulationController):
    fut: Future = Future()
    fut.set_result(
        {
            "success": False,
            "error": {"kind": "simulation_error", "message": "solver blew up", "code": "E301"},
        }
    )

    controller._batch_parallel.future_map = {"sid": fut}
    controller._batch_parallel.future_meta = {"sid": {"set_name": "set1"}}
    controller.on_simulation_error = MagicMock()

    ok = controller._consume_parallel_batch_future(
        set_id="sid",
        fut=fut,
        run_id=1,
        request_id=1,
        fast_mode=False,
        cache_key="ck",
        source="scan",
    )

    assert ok is False
    controller.on_simulation_error.assert_called_once()


@pytest.mark.unit
def test_has_running_workers_is_pure_query(controller: SimulationController):
    worker = _FakeWorker(running=False)
    controller._simulation_worker = worker
    controller._retained_simulation_workers = [worker]
    controller._delete_worker_if_stopped = MagicMock()

    assert controller._has_running_owned_simulation_workers() is False
    controller._delete_worker_if_stopped.assert_not_called()
    assert controller._simulation_worker is worker
    assert controller._retained_simulation_workers == [worker]


# ---------------------------------------------------------------------------
# Structured execution request: non-fast-mode regression tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_does_not_build_prepared_payloads(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """Explicit runs must stay canonical-only and avoid preview prepared payloads."""

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        runtime = _FakeRuntime()
        created_runtimes.append(runtime)
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    # Explicit run: fast_mode=False
    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0], reuse_parallel_executor=False)

    prepared_by_set_id = controller._batch_run_context.get("prepared_by_set_id", {})
    assert prepared_by_set_id == {}
    assert controller._batch_run_context.get("prepared") is None
    assert created_runtimes == []
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_builds_execution_requests(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        return _FakeRuntime()

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0], reuse_parallel_executor=False)

    execution_request_by_set_id = controller._batch_run_context.get("execution_request_by_set_id", {})
    assert "id1" in execution_request_by_set_id
    request = execution_request_by_set_id["id1"]
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 1.0}
    assert tuple(request["t_span"]) == (0.0, 10.0)
    assert request["mechanism_text"] == "reaction: A -> B; k=1"
    assert request["simulation_identity"]["schema_id"] != ""
    assert request["simulation_identity"]["param_fingerprint"] == ""
    mw.preview_initials_for_row.assert_not_called()
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_multiset_execution_requests_do_not_inherit_primary_dsl(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=PRIMARY")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=PRIMARY",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

        def as_serializable_execution_payload(self) -> dict[str, object]:
            return {
                "version": 2,
                "mechanism": self.mechanism,
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=PRIMARY",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        return _FakeRuntime()

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 4.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 2.5}, {"A": 5.5}])
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()
    controller._apply_parameter_override_fallback_to_dsl = MagicMock(
        side_effect=lambda text, *, set_id=None: str(text).replace("PRIMARY", str(set_id))
    )

    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0, 1], reuse_parallel_executor=False)

    execution_request_by_set_id = controller._batch_run_context.get("execution_request_by_set_id", {})
    assert execution_request_by_set_id["id1"]["prepared_payload"] is None
    assert execution_request_by_set_id["id2"]["prepared_payload"] is None
    assert execution_request_by_set_id["id1"]["mechanism_text"] == "reaction: A -> B; k=PRIMARY"
    assert execution_request_by_set_id["id2"]["mechanism_text"] == "reaction: A -> B; k=PRIMARY"
    assert execution_request_by_set_id["id1"]["initials"] == {"A": 1.0}
    assert execution_request_by_set_id["id2"]["initials"] == {"A": 4.0}
    mw.preview_initials_for_row.assert_not_called()
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()


@pytest.mark.unit
def test_start_next_batch_simulation_non_fast_mode_ignores_prepared_payload(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """Serial explicit runs must parse canonical DSL instead of using preview prepared payloads."""

    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_text_by_set_id": {"id1": "reaction: A -> B; k=1"},
        "mechanism_signature": "sig",
        "mechanism_signature_by_set_id": {"id1": "sig-1"},
        "prepared": None,
        "prepared_by_set_id": {
            "id1": {"prepared_for": "id1", "version": 1},
        },
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,  # explicit run
        "request_id": 42,
        "cache_key": "explicit-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["started"] is True
    mw.preview_initials_for_row.assert_not_called()


@pytest.mark.unit
def test_start_next_batch_simulation_non_fast_mode_sets_structured_execution_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "full_dsl": "reaction: A -> B; k=1",
        "execution_request_by_set_id": {
            "id1": {
                "prepared_payload": {"version": 1, "prepared_for": "id1"},
                "initials": {"A": 3.0},
                "t_span": (0.0, 10.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "reaction: A -> B; k=1",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        },
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,
        "request_id": 42,
        "cache_key": "explicit-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["mechanism_text"] == "reaction: A -> B; k=1"
    worker = controller._simulation_worker
    assert getattr(worker, "_execution_request", None) is not None
    assert worker._execution_request["initials"] == {"A": 3.0}  # type: ignore[index]
    assert worker._execution_request["prepared_payload"] is None  # type: ignore[index]
    assert worker._execution_request["mechanism_text"] == "reaction: A -> B; k=1"  # type: ignore[index]


@pytest.mark.unit
def test_start_next_batch_simulation_non_primary_explicit_worker_uses_secondary_result_payload_mode(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload,
        ):
            _ = mechanism_text, initials, t_span, solver_config, parent, prepared
            created["include_mechanism_in_result_payload"] = bool(include_mechanism_in_result_payload)
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    controller._batch_run_context = {
        "active": True,
        "parallel": False,
        "pos": 1,
        "rows": [0, 1],
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "primary_set_id": "id1",
        "full_dsl": "reaction: A -> B; k=1",
        "execution_request_by_set_id": {
            "id2": {
                "prepared_payload": {"version": 1, "prepared_for": "id2"},
                "initials": {"A": 3.0},
                "t_span": (0.0, 10.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "reaction: A -> B; k=2",
                "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
            },
        },
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "fast_mode": False,
        "request_id": 42,
        "cache_key": "explicit-cache",
        "pending_init_seed": {},
        "pending_init_applied": True,
    }

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _RecordingWorker)

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["include_mechanism_in_result_payload"] is False


@pytest.mark.unit
def test_run_simulation_internal_energy_mode_builds_structured_execution_requests(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return "\n".join(
                [
                    "energy=kJ/mol",
                    "T=298.15",
                    "state: A, kind=GS, energy=0, members=A",
                    "state: B, kind=GS, energy=5, members=B",
                    "state: TS1, kind=TS, energy=25",
                    "edge: A,TS1",
                    "edge: TS1,B",
                ]
            )

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("")
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {
        "dGact_fwd__TS1__A__B": 32.0,
        "dG_eq__TS1__A__B": 7.0,
    }
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0
    runtime = MainWindowVariableRuntime(mw)
    runtime.set_variable_metadata(
        {
            "dGact_fwd__TS1__A__B": {
                "type": "energy",
                "role": "dG_act_fwd",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
            "dG_eq__TS1__A__B": {
                "type": "energy",
                "role": "dG_eq",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
        }
    )
    mw._prepare_slider_runtime = MagicMock(
        side_effect=lambda *, set_id=None: runtime.prepare_slider_runtime(set_id=set_id)
    )
    mw._apply_slider_overrides_to_bindings = MagicMock(
        side_effect=lambda prepared, *, set_id=None: runtime.apply_slider_overrides_to_bindings(
            prepared,
            set_id=set_id,
        )
    )

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_executor = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=99, batch_rows=[0], reuse_parallel_executor=False)

    prepared_by_set_id = controller._batch_run_context.get("prepared_by_set_id", {})
    execution_request_by_set_id = controller._batch_run_context.get("execution_request_by_set_id", {})

    assert prepared_by_set_id == {}
    assert "id1" in execution_request_by_set_id
    assert execution_request_by_set_id["id1"]["prepared_payload"] is None
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()
