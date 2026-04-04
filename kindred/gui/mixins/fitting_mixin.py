# kindred/gui/mixins/fitting_mixin.py
"""
FittingMixin - Parameter fitting operations for MainWindow.

Provides methods for:
- Fitting configuration dialogs
- Fitting diagnostics display
- Parameter override/update in mechanism DSL
- Modeless fitting window management
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.dsl_text_update import (
    StepParameterUpdateOutcome,
    analyze_parameter_updates_to_dsl_text,
    authoritative_parameter_values_match,
)
from kindred.gui.diagnostics import record_best_effort_failure as record_gui_best_effort_failure
from kindred.gui.mixins.ports import FittingMixinPorts
from kindred.gui.ui_helpers import safe_float_parse, setup_scientific_validator

logger = logging.getLogger(__name__)

_FITTING_KEY_TO_SHORT: dict[str, str] = {
    "fitting_method": "method",
    "fitting_max_nfev": "max_nfev",
    "fitting_ftol": "ftol",
    "fitting_xtol": "xtol",
    "fitting_use_parallel": "use_parallel",
    "fitting_use_seed": "use_seed",
    "fitting_seed": "seed",
    "fitting_solver": "solver",
    "fitting_rtol": "rtol",
    "fitting_atol": "atol",
}

_FIT_PARAMETER_APPLY_STATUS_REWRITTEN = "rewritten"
_FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT = "already_current"
_FIT_PARAMETER_APPLY_STATUS_FAILED = "failed"
_FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE = "unappliable"
_FIT_PARAMETER_APPLY_OUTCOME_REWRITE_INTENDED = "rewrite_intended"


@dataclass(frozen=True)
class _FitParameterApplyDelta:
    parameters: Dict[str, float]
    authoritative_values: Dict[str, float]
    changed_names: tuple[str, ...]
    current_text: str
    updated_text: str
    missing: tuple[str, ...]
    update_errors: tuple[dict[str, str], ...]
    step_outcomes: tuple[StepParameterUpdateOutcome, ...]
    warning_messages: tuple[str, ...]
    has_real_change: bool
    needs_dsl_rewrite: bool


@dataclass(frozen=True)
class _FitParameterApplyOutcome:
    status: str
    warning_messages: tuple[str, ...]
    rewrite_intended: bool
    rewrite_performed: bool
    rewrite_failed: bool
    applied_any: bool


@dataclass(frozen=True)
class _FitDatasetInitialConditionApplyItem:
    dataset_id: str
    row: int
    set_id: str
    set_name: str
    canonical_updates: Dict[str, float]
    settings_sync_updates: Dict[str, float]
    mapping_sync_needed: bool


@dataclass(frozen=True)
class _FitProjectApplyPlan:
    scope: str
    parameter_delta: Optional[_FitParameterApplyDelta]
    parameter_outcome: Optional[_FitParameterApplyOutcome]
    initial_condition_items: tuple[_FitDatasetInitialConditionApplyItem, ...]
    canonical_ic_affected_rows: tuple[int, ...]
    canonical_ic_affected_set_ids: tuple[str, ...]
    has_dataset_settings_sync: bool
    can_execute_parameter_apply: bool
    needs_slider_guard: bool
    needs_display_refresh: bool
    is_true_noop: bool


if TYPE_CHECKING:
    from kindred.gui.fitting.launch import GlobalFitLaunchContext


def launch_global_fit_session(context):
    from kindred.gui.fitting.launch import launch_global_fit_session as _impl

    return _impl(context)


class FittingMixin:
    """
    Mixin providing parameter fitting functionality for MainWindow.

    This mixin encapsulates fitting-related methods including parameter
    fitting workflows, fitting diagnostics, result handling, and mechanism
    DSL parameter updates.

    MainWindow provides an explicit `FittingMixinPorts` object at `self._fitting_ports`.
    """

    def _require_fitting_ports(self) -> FittingMixinPorts:
        ports = getattr(self, "_fitting_ports", None)
        if isinstance(ports, FittingMixinPorts):
            return ports

        mechanism_editor = getattr(self, "_mechanism_editor", None)
        dataset_manager = getattr(self, "_dataset_manager", None)
        data_manager = getattr(getattr(self, "_right_panel", None), "_data_manager", None)
        status_label = getattr(self, "_status_label", None)
        temperature_spinbox = getattr(self, "_temperature_spinbox", None)
        num_points_spinbox = getattr(self, "_num_points_spinbox", None)

        if status_label is None:
            raise RuntimeError("FittingMixin ports are not initialized.")

        return FittingMixinPorts(
            mechanism_editor=mechanism_editor,
            dataset_manager=dataset_manager,
            data_manager_getter=lambda: getattr(getattr(self, "_right_panel", None), "_data_manager", data_manager),
            status_setter=lambda text: status_label.setText(str(text)),
            temperature_getter=lambda: float(temperature_spinbox.value()) if temperature_spinbox is not None else 298.15,
            num_points_getter=lambda: int(num_points_spinbox.value()) if num_points_spinbox is not None else 100,
        )

    def _set_fitting_status(self, text: str) -> None:
        self._require_fitting_ports().status_setter(str(text))

    def _record_fitting_best_effort_failure(
        self,
        key: str,
        *,
        message: str,
        exc: Optional[Exception] = None,
        max_logs: int = 3,
    ) -> None:
        record_gui_best_effort_failure(
            self,
            str(key),
            message=message,
            exc=exc,
            log=logger,
            max_logs=int(max_logs),
            failures_attr="_fitting_best_effort_failures",
            counts_attr="_fitting_best_effort_failure_counts",
        )

    def _register_fit_window(self, window: QtWidgets.QWidget) -> None:
        """Track modeless fitting windows to keep them alive."""
        if not hasattr(self, "_active_fit_windows"):
            self._active_fit_windows: List[QtWidgets.QWidget] = []

        self._active_fit_windows.append(window)

        def _cleanup(*_args):
            windows = getattr(self, "_active_fit_windows", None)
            if isinstance(windows, list) and window in windows:
                windows.remove(window)

        window.destroyed.connect(_cleanup)
        window.show()
        window.raise_()
        window.activateWindow()

    def _write_fit_results_to_mechanism(
        self,
        parameters: dict,
        *,
        parameter_delta: Optional[_FitParameterApplyDelta] = None,
        refresh_display: bool = True,
    ) -> str:
        """
        Update mechanism DSL text with fitted parameter values.

        Parameters
        ----------
        parameters : dict
            Dictionary of {param_name: fitted_value}
        """
        normalized_parameters = {
            str(name): float(value)
            for name, value in dict(parameters or {}).items()
        }
        delta = parameter_delta or self._build_fit_parameter_apply_delta(normalized_parameters)
        updated_text = str(delta.updated_text or "")
        mechanism_editor = self._require_fitting_ports().mechanism_editor
        before_text = str(mechanism_editor.reactions_text())
        preview_outcome = self._build_fit_parameter_apply_outcome(delta, before_text=before_text)
        missing = [str(name) for name in delta.missing]
        update_errors = [dict(error) for error in delta.update_errors]
        for name in missing:
            logger.warning("Parameter %r not updated (missing/invalid)", name)
        for error in update_errors:
            logger.warning(
                "Parameter %r failed while applying fitted value (%s): %s",
                error.get("name"),
                error.get("exc_type", "Exception"),
                error.get("message", ""),
            )
        if preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE:
            logger.info("Fitted parameter project apply found unappliable requested targets; no rewrite will be performed")
            return _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE
        if preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT and not delta.has_real_change:
            logger.info("Fitted parameter project apply made no authoritative numeric change")
            return _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT
        if preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT and before_text == updated_text:
            logger.info("Fitted parameter project apply found the mechanism text already current")
            return _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT
        if preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT:
            logger.info("Fitted parameter project apply found no authoritative DSL rewrite to perform")
            return _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT

        set_with_undo = getattr(self, "set_mechanism_reactions_text_with_optional_undo", None)
        if callable(set_with_undo):
            set_with_undo(
                updated_text,
                "Global Fit: update Reactions",
                record_undo=True,
            )
        else:
            mechanism_editor.set_reactions_text(updated_text)

        if str(mechanism_editor.reactions_text()) == before_text:
            logger.warning("Fitted parameter project apply did not rewrite mechanism text")
            return _FIT_PARAMETER_APPLY_STATUS_FAILED

        if callable(getattr(self, "_on_programmatic_mechanism_load", None)):
            self._on_programmatic_mechanism_load()
        if callable(getattr(self, "_extract_and_populate_variables", None)):
            self._extract_and_populate_variables(preserve_visibility=True)
        if callable(getattr(self, "_sync_batch_species_columns", None)):
            mechanism_initials = self._extract_mechanism_initials(updated_text)
            self._sync_batch_species_columns(list(mechanism_initials.keys()))
        if refresh_display and callable(getattr(self, "_refresh_batch_display_from_focus_and_shown", None)):
            self._refresh_batch_display_from_focus_and_shown()
        logger.info("Wrote %d fitted parameter(s) to mechanism editor", len(normalized_parameters))
        return _FIT_PARAMETER_APPLY_STATUS_REWRITTEN

    def _prepare_fit_parameter_project_text(self, parameters: dict) -> Dict[str, object]:
        ports = self._require_fitting_ports()
        current_text = ports.mechanism_editor.reactions_text()
        analysis = analyze_parameter_updates_to_dsl_text(
            current_text,
            parameters,
            authoritative_values=self._current_fit_authoritative_parameter_values(),
            step_constraint_context=self._current_fit_step_constraint_context(),
        )
        return {
            "current_text": current_text,
            "updated_text": analysis.updated_text,
            "missing": list(analysis.missing),
            "update_errors": [dict(error) for error in analysis.update_errors],
            "step_outcomes": tuple(analysis.step_outcomes),
        }

    def _current_fit_authoritative_parameter_values(self) -> Dict[str, float]:
        preview_session = getattr(self, "_preview_session", None)
        param_store = getattr(preview_session, "param_store", None)
        return {
            str(name): float(value)
            for name, value in dict(getattr(param_store, "shared_params", {}) or {}).items()
        }

    def _current_fit_step_constraint_context(self) -> Dict[str, object]:
        ports = self._require_fitting_ports()
        solver_settings_getter = getattr(self, "_get_solver_settings", None)
        solver_settings = solver_settings_getter() if callable(solver_settings_getter) else {}
        return {
            "temperature_K": float(ports.temperature_getter()),
            "wegscheider_cyclicity_enabled": bool(
                dict(solver_settings or {}).get("wegscheider_cyclicity_enabled", False)
            ),
        }

    @staticmethod
    def _fit_parameter_warning_messages(
        missing: tuple[str, ...],
        update_errors: tuple[dict[str, str], ...],
        step_outcomes: tuple[StepParameterUpdateOutcome, ...],
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        warned_names = {str(outcome.parameter_name) for outcome in step_outcomes}
        for outcome in step_outcomes:
            if not outcome.found_target:
                warnings.append(
                    f"Step parameter '{outcome.parameter_name}' no longer matches any writable step in the current mechanism text."
                )
                continue
            if not outcome.writable:
                warnings.append(
                    f"Step parameter '{outcome.parameter_name}' is no longer writable in the current mechanism text."
                )
        for name in missing:
            name_str = str(name)
            if name_str in warned_names:
                continue
            warnings.append(f"Parameter '{name_str}' no longer exists in the current mechanism text.")
        for error in update_errors:
            name = str(error.get("name") or "unknown")
            exc_type = str(error.get("exc_type") or "Exception")
            message = str(error.get("message") or "").strip()
            if message:
                warnings.append(
                    f"Parameter '{name}' could not be applied to the current mechanism text ({exc_type}): {message}"
                )
            else:
                warnings.append(
                    f"Parameter '{name}' could not be applied to the current mechanism text ({exc_type})."
                )
        deduped: list[str] = []
        seen: set[str] = set()
        for warning in warnings:
            text = str(warning).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return tuple(deduped)

    @staticmethod
    def _build_fit_parameter_apply_outcome(
        delta: _FitParameterApplyDelta,
        *,
        before_text: str | None = None,
        write_status: str | None = None,
    ) -> _FitParameterApplyOutcome:
        warning_messages = tuple(str(message) for message in delta.warning_messages if str(message))
        current_text = str(delta.current_text if before_text is None else before_text)
        updated_text = str(delta.updated_text or "")
        rewrite_intended = bool(
            delta.has_real_change
            and delta.needs_dsl_rewrite
            and current_text != updated_text
        )
        if write_status == _FIT_PARAMETER_APPLY_STATUS_REWRITTEN:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_STATUS_REWRITTEN,
                warning_messages=warning_messages,
                rewrite_intended=rewrite_intended,
                rewrite_performed=True,
                rewrite_failed=False,
                applied_any=True,
            )
        if write_status == _FIT_PARAMETER_APPLY_STATUS_FAILED:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_STATUS_FAILED,
                warning_messages=warning_messages,
                rewrite_intended=rewrite_intended,
                rewrite_performed=False,
                rewrite_failed=True,
                applied_any=False,
            )
        if write_status == _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE,
                warning_messages=warning_messages,
                rewrite_intended=False,
                rewrite_performed=False,
                rewrite_failed=False,
                applied_any=False,
            )
        if warning_messages and not rewrite_intended:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE,
                warning_messages=warning_messages,
                rewrite_intended=False,
                rewrite_performed=False,
                rewrite_failed=False,
                applied_any=False,
            )
        if rewrite_intended:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_OUTCOME_REWRITE_INTENDED,
                warning_messages=warning_messages,
                rewrite_intended=True,
                rewrite_performed=False,
                rewrite_failed=False,
                applied_any=False,
            )
        return _FitParameterApplyOutcome(
            status=_FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT,
            warning_messages=warning_messages,
            rewrite_intended=False,
            rewrite_performed=False,
            rewrite_failed=False,
            applied_any=True,
        )

    @staticmethod
    def _fit_initial_condition_values_match(current_value: object, target_value: float) -> bool:
        try:
            current_float = float(current_value)
        except (TypeError, ValueError):
            return False
        return math.isclose(
            current_float,
            float(target_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    def _build_fit_parameter_apply_delta(self, parameters: Dict[str, float]) -> _FitParameterApplyDelta:
        normalized_parameters = {
            str(name): float(value)
            for name, value in dict(parameters or {}).items()
        }
        authoritative_values = self._current_fit_authoritative_parameter_values()
        update = self._prepare_fit_parameter_project_text(normalized_parameters)
        current_text = str(update["current_text"] or "")
        updated_text = str(update["updated_text"] or "")
        step_outcomes = tuple(update.get("step_outcomes") or ())
        step_outcomes_by_name = {
            str(outcome.parameter_name): outcome
            for outcome in step_outcomes
        }
        changed_names = tuple(
            str(name)
            for name, value in normalized_parameters.items()
            if (
                step_outcomes_by_name[str(name)].semantic_value_change
                if str(name) in step_outcomes_by_name
                else not authoritative_parameter_values_match(
                    authoritative_values.get(str(name)),
                    float(value),
                )
            )
        )
        missing = tuple(str(name) for name in update["missing"])
        update_errors = tuple(dict(error) for error in update["update_errors"])
        warning_messages = self._fit_parameter_warning_messages(missing, update_errors, step_outcomes)
        return _FitParameterApplyDelta(
            parameters=dict(normalized_parameters),
            authoritative_values=dict(authoritative_values),
            changed_names=changed_names,
            current_text=current_text,
            updated_text=updated_text,
            missing=missing,
            update_errors=update_errors,
            step_outcomes=step_outcomes,
            warning_messages=warning_messages,
            has_real_change=bool(changed_names),
            needs_dsl_rewrite=(updated_text != current_text),
        )

    def _init_fitting_defaults(self) -> None:
        """Initialize document-override fitting defaults as empty (no document loaded yet)."""
        self._fitting_defaults: Dict[str, object] = {}

    def _load_fitting_defaults(self) -> Dict[str, object]:
        """Return user-level fitting defaults (short keys) for the Fitting Defaults dialog.

        Reads from config_controller (tier 2 — global user preferences), NOT from
        self._fitting_defaults which may contain document overrides (tier 3).
        """
        _pref = self.config_controller.get_user_preference
        return {
            short: _pref(full)
            for full, short in _FITTING_KEY_TO_SHORT.items()
        }

    def _get_fitting_session_defaults(self) -> Dict[str, object]:
        """Return project-effective fitting defaults (short keys) for the fitting window.

        Document overrides (tier 3) take precedence.  Non-overridden keys
        read live from the user preference (tier 2), which itself falls
        back to factory defaults.
        """
        _pref = self.config_controller.get_user_preference
        return {
            short: (self._fitting_defaults[full]
                    if full in self._fitting_defaults
                    else _pref(full))
            for full, short in _FITTING_KEY_TO_SHORT.items()
        }

    def _configure_fitting(self):
        """Configure default fitting settings for new fitting windows."""
        defaults = self._load_fitting_defaults()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fitting Defaults")
        dialog.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(dialog)

        info = QtWidgets.QLabel(
            "Set default settings for new fitting windows."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        algo_group = QtWidgets.QGroupBox("Algorithm")
        algo_layout = QtWidgets.QFormLayout(algo_group)

        method_combo = QtWidgets.QComboBox()
        method_combo.addItems(["lm", "trf", "dogbox", "differential_evolution"])
        method_default = defaults.get("method", "trf")
        if method_default in {"lm", "trf", "dogbox", "differential_evolution"}:
            method_combo.setCurrentText(method_default)
        algo_layout.addRow("Method:", method_combo)

        max_nfev_spin = QtWidgets.QSpinBox()
        max_nfev_spin.setRange(10, 10000)
        max_nfev_spin.setValue(int(defaults.get("max_nfev", 1000)))
        algo_layout.addRow("Max evaluations:", max_nfev_spin)

        ftol_edit = QtWidgets.QLineEdit(str(defaults.get("ftol", "1e-10")))
        _ftol_val = setup_scientific_validator(ftol_edit)
        algo_layout.addRow("ftol:", ftol_edit)

        xtol_edit = QtWidgets.QLineEdit(str(defaults.get("xtol", "1e-10")))
        _xtol_val = setup_scientific_validator(xtol_edit)
        algo_layout.addRow("xtol:", xtol_edit)

        use_parallel_check = QtWidgets.QCheckBox("Parallel multi-start (DE only)")
        use_parallel_check.setChecked(bool(defaults.get("use_parallel", False)))
        algo_layout.addRow(use_parallel_check)

        use_seed_check = QtWidgets.QCheckBox("Use fixed random seed")
        use_seed_check.setChecked(bool(defaults.get("use_seed", True)))
        seed_spin = QtWidgets.QSpinBox()
        seed_spin.setRange(0, 999_999)
        seed_spin.setValue(int(defaults.get("seed", 42)))
        seed_spin.setEnabled(use_seed_check.isChecked())
        use_seed_check.toggled.connect(seed_spin.setEnabled)
        algo_layout.addRow(use_seed_check, seed_spin)

        layout.addWidget(algo_group)

        integration_group = QtWidgets.QGroupBox("Integration")
        integration_layout = QtWidgets.QFormLayout(integration_group)

        solver_combo = QtWidgets.QComboBox()
        solver_combo.addItems(["LSODA", "Radau", "BDF"])
        solver_combo.setCurrentText(str(defaults.get("solver", "LSODA")))
        integration_layout.addRow("Solver:", solver_combo)

        rtol_edit = QtWidgets.QLineEdit(str(defaults.get("rtol", "1e-6")))
        _rtol_val = setup_scientific_validator(rtol_edit)
        integration_layout.addRow("rtol:", rtol_edit)

        atol_edit = QtWidgets.QLineEdit(str(defaults.get("atol", "1e-12")))
        _atol_val = setup_scientific_validator(atol_edit)
        integration_layout.addRow("atol:", atol_edit)

        layout.addWidget(integration_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            updates = {
                "fitting_method": method_combo.currentText().strip().lower(),
                "fitting_max_nfev": int(max_nfev_spin.value()),
                "fitting_ftol": max(safe_float_parse(ftol_edit.text(), 1e-10), 1e-15),
                "fitting_xtol": max(safe_float_parse(xtol_edit.text(), 1e-10), 1e-15),
                "fitting_use_parallel": bool(use_parallel_check.isChecked()),
                "fitting_use_seed": bool(use_seed_check.isChecked()),
                "fitting_seed": int(seed_spin.value()),
                "fitting_solver": solver_combo.currentText(),
                "fitting_rtol": max(safe_float_parse(rtol_edit.text(), 1e-6), 1e-15),
                "fitting_atol": max(safe_float_parse(atol_edit.text(), 1e-12), 1e-18),
            }
            for key, value in updates.items():
                self.config_controller.update_user_preference(key, value)
            logger.info("Fitting defaults updated")
            self._set_fitting_status("Fitting defaults updated")

    def _show_fitting_diagnostics(self):
        """Show fitting diagnostics dialog with statistical analysis."""
        if not hasattr(self, '_last_fit_result') or self._last_fit_result is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Fit Results",
                "No fitting results available.\n\n"
                "Please run a global fit first using Fitting → Global Fit..."
            )
            return

        fit_result = self._last_fit_result

        def _array(value):
            if value is None:
                return np.array([])
            if isinstance(value, np.ndarray):
                return value
            return np.asarray(value)

        residuals = _array(fit_result.get('residuals'))
        observed = _array(fit_result.get('observed'))
        predicted = _array(fit_result.get('predicted'))

        if predicted.size == 0 and observed.size == residuals.size and observed.size > 0:
            predicted = observed + residuals

        if observed.size == 0:
            dataset_info = fit_result.get("dataset", {})
            dataset_name = dataset_info.get("name")
            if dataset_name:
                data_manager = self._require_fitting_ports().data_manager_getter()
                dataset_payload = data_manager.get_dataset(dataset_name) if data_manager is not None else None
                species_map = (dataset_payload or {}).get("species", {})
                target = dataset_info.get("target_species")
                if target and target in species_map:
                    observed = _array(species_map[target])
                elif species_map:
                    observed = _array(next(iter(species_map.values())))
                if observed.size and residuals.size == observed.size and predicted.size == 0:
                    predicted = observed + residuals

        result_dict = {
            'parameters': fit_result.get('parameters', {}),
            'residuals': residuals,
            'predicted': predicted,
            'observed': observed,
            'jacobian': None,
            'covariance': fit_result.get('covariance'),
            'success': bool(fit_result.get('success')),
            'message': fit_result.get('message', 'N/A'),
            'nfev': int(fit_result.get('nfev', 0)),
            'cost': float(fit_result.get('chi_squared', 0.0)),
        }

        from kindred.gui.widgets.fitting_diagnostics_dialog import FittingDiagnosticsDialog
        dialog = FittingDiagnosticsDialog(result_dict, self)
        dialog.show()  # Non-modal to allow comparison with plots

        logger.info("Fitting diagnostics dialog opened")

    def _extract_mechanism_initials(self, mechanism_text: str) -> Dict[str, float]:
        """Return mechanism species initial concentrations without running a simulation."""
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism
        from kindred.core.units import UnitsModel

        temperature = self._require_fitting_ports().temperature_getter()
        units = UnitsModel(temperature_K=temperature, energy_unit="kJ/mol")

        mechanism = parse_dsl_to_mechanism(mechanism_text, initials={}, units=units)
        return {name: species.initial_conc for name, species in mechanism.species.items()}

    def _build_global_fit_launch_context(self) -> GlobalFitLaunchContext:
        from kindred.gui.fitting.launch import GlobalFitLaunchContext

        ports = self._require_fitting_ports()
        reactions_widget = getattr(ports.mechanism_editor, "_reactions_text", None)

        def _get_reactions_text() -> str:
            if reactions_widget is None:
                return ""
            try:
                return str(reactions_widget.toPlainText() or "")
            except Exception:
                return ""

        def _set_reactions_text(new_text: str) -> None:
            if reactions_widget is None:
                raise RuntimeError("Reactions editor unavailable.")
            if callable(getattr(self, "_set_text_with_optional_undo", None)):
                self._set_text_with_optional_undo(
                    reactions_widget,
                    str(new_text or ""),
                    "Global Fit: update Reactions",
                    True,
                )
                return
            reactions_widget.setPlainText(str(new_text or ""))

        return GlobalFitLaunchContext(
            parent=self,
            dataset_manager=ports.dataset_manager,
            data_manager_getter=ports.data_manager_getter,
            mechanism_text_getter=lambda: str(self._get_mechanism_text() or ""),
            reactions_text_getter=_get_reactions_text,
            reactions_text_setter=_set_reactions_text,
            extract_mechanism_initials=self._extract_mechanism_initials,
            record_best_effort_failure=self._record_fitting_best_effort_failure,
            set_status=self._set_fitting_status,
            sync_batch_species_columns=self._sync_batch_species_columns,
            batch_initials_for_row=self._batch_initials_for_row,
            get_solver_settings=self._get_solver_settings,
            temperature_getter=ports.temperature_getter,
            num_points_getter=ports.num_points_getter,
            register_fit_window=self._register_fit_window,
            write_fit_results_to_mechanism=self._write_fit_results_to_mechanism,
            apply_fit_results_to_project=self._apply_fit_results_to_project,
            apply_dataset_initial_updates=self._apply_dataset_initial_updates,
            load_fitting_defaults=self._get_fitting_session_defaults,
            batch_store=getattr(self, "_batch_store", None),
            batch_model=getattr(self, "_batch_model", None),
            batch_table=getattr(self, "_batch_table", None),
        )

    def _run_global_fit(self):
        """Delegate global-fit launch ownership to the fitting package."""
        return launch_global_fit_session(self._build_global_fit_launch_context())

    @staticmethod
    def _normalize_fit_project_apply_scope(scope: str) -> str:
        normalized = str(scope or "").strip().lower()
        if normalized in {"parameters", "initial_conditions", "both"}:
            return normalized
        raise ValueError(f"Unsupported fit project-apply scope: {scope!r}")

    @staticmethod
    def _extract_fit_initial_condition_updates(
        dataset_params: Dict[str, Dict[str, float]] | None,
    ) -> Dict[str, Dict[str, float]]:
        updates_by_dataset: Dict[str, Dict[str, float]] = {}
        for dataset_id, param_map in (dataset_params or {}).items():
            if not isinstance(param_map, dict):
                continue
            species_updates: Dict[str, float] = {}
            for key, value in param_map.items():
                key_str = str(key)
                if not key_str.startswith("init:"):
                    continue
                species = key_str[len("init:") :]
                if not species:
                    continue
                species_updates[species] = float(value)
            if species_updates:
                updates_by_dataset[str(dataset_id)] = species_updates
        return updates_by_dataset

    def _build_fit_initial_condition_apply_items(
        self,
        dataset_params: Dict[str, Dict[str, float]] | None,
    ) -> tuple[tuple[_FitDatasetInitialConditionApplyItem, ...], bool]:
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        updates_by_dataset = self._extract_fit_initial_condition_updates(dataset_params)
        has_requested_updates = bool(updates_by_dataset)
        if not updates_by_dataset:
            return (), False

        batch_store = getattr(self, "_batch_store", None)
        if batch_store is None:
            raise RuntimeError("Initial conditions store is unavailable.")

        dataset_manager = self._require_fitting_ports().dataset_manager
        if dataset_manager is None:
            raise RuntimeError("Dataset manager is unavailable.")

        plan: list[_FitDatasetInitialConditionApplyItem] = []
        for dataset_id, species_updates in updates_by_dataset.items():
            try:
                settings = dataset_manager.get_fit_settings(dataset_id)
            except DatasetManagerError as exc:
                raise RuntimeError(f"Dataset '{dataset_id}' is no longer available.") from exc

            mapped_set_id = str(getattr(settings, "batch_set_id", "") or "").strip()
            mapped_set_name = str(getattr(settings, "batch_set", "") or "").strip()
            row = batch_store.row_for_set_id(mapped_set_id) if mapped_set_id else None
            if row is None and mapped_set_name:
                row = batch_store.row_for_set(mapped_set_name)
            if row is None:
                raise RuntimeError(
                    f"Dataset '{dataset_id}' is no longer mapped to an Initial Conditions set."
                )
            row_i = int(row)
            resolved_set_id = str(batch_store.set_id_for_row(row_i))
            resolved_set_name = str(batch_store.set_name_for_row(row_i))
            canonical_updates: Dict[str, float] = {}
            settings_sync_updates: Dict[str, float] = {}
            for species, value in species_updates.items():
                target_value = float(value)
                if not self._fit_initial_condition_values_match(
                    batch_store.get_value(row_i, str(species)),
                    target_value,
                ):
                    canonical_updates[str(species)] = target_value
                if not self._fit_initial_condition_values_match(
                    getattr(settings, "initial_conditions", {}).get(str(species)),
                    target_value,
                ):
                    settings_sync_updates[str(species)] = target_value
            mapping_sync_needed = (
                str(getattr(settings, "batch_set_id", "") or "").strip() != resolved_set_id
                or str(getattr(settings, "batch_set", "") or "").strip() != resolved_set_name
            )
            if not (canonical_updates or settings_sync_updates or mapping_sync_needed):
                continue
            plan.append(
                _FitDatasetInitialConditionApplyItem(
                    dataset_id=str(dataset_id),
                    row=row_i,
                    set_id=resolved_set_id,
                    set_name=resolved_set_name,
                    canonical_updates=dict(canonical_updates),
                    settings_sync_updates=dict(settings_sync_updates),
                    mapping_sync_needed=bool(mapping_sync_needed),
                )
            )
        return tuple(plan), has_requested_updates

    def _build_fit_project_apply_plan(
        self,
        scope: str,
        parameters: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
    ) -> _FitProjectApplyPlan:
        normalized_scope = self._normalize_fit_project_apply_scope(scope)
        if normalized_scope in {"parameters", "both"} and not parameters:
            raise RuntimeError("No fitted parameter values are available to apply.")

        parameter_delta: Optional[_FitParameterApplyDelta] = None
        parameter_outcome: Optional[_FitParameterApplyOutcome] = None
        if normalized_scope in {"parameters", "both"}:
            parameter_delta = self._build_fit_parameter_apply_delta(dict(parameters))
            parameter_outcome = self._build_fit_parameter_apply_outcome(parameter_delta)

        initial_condition_items: tuple[_FitDatasetInitialConditionApplyItem, ...] = ()
        if normalized_scope in {"initial_conditions", "both"}:
            initial_condition_items, has_initial_updates = self._build_fit_initial_condition_apply_items(dataset_params)
            if not has_initial_updates:
                raise RuntimeError("No fitted initial-condition values are available to apply.")

        canonical_rows: list[int] = []
        canonical_set_ids: list[str] = []
        has_dataset_settings_sync = False
        has_canonical_ic_change = False
        for item in initial_condition_items:
            if item.settings_sync_updates or item.mapping_sync_needed:
                has_dataset_settings_sync = True
            if item.canonical_updates:
                has_canonical_ic_change = True
                if int(item.row) not in canonical_rows:
                    canonical_rows.append(int(item.row))
                set_id = str(item.set_id or "").strip()
                if set_id and set_id not in canonical_set_ids:
                    canonical_set_ids.append(set_id)

        has_parameter_warnings = bool(parameter_outcome is not None and parameter_outcome.warning_messages)
        can_execute_parameter_apply = bool(parameter_outcome is not None and parameter_outcome.rewrite_intended)
        return _FitProjectApplyPlan(
            scope=normalized_scope,
            parameter_delta=parameter_delta,
            parameter_outcome=parameter_outcome,
            initial_condition_items=tuple(initial_condition_items),
            canonical_ic_affected_rows=tuple(canonical_rows),
            canonical_ic_affected_set_ids=tuple(canonical_set_ids),
            has_dataset_settings_sync=bool(has_dataset_settings_sync),
            can_execute_parameter_apply=bool(can_execute_parameter_apply),
            needs_slider_guard=bool(can_execute_parameter_apply),
            needs_display_refresh=bool(can_execute_parameter_apply or has_canonical_ic_change),
            is_true_noop=not bool(
                can_execute_parameter_apply or has_canonical_ic_change or has_dataset_settings_sync or has_parameter_warnings
            ),
        )

    def _apply_fit_initial_condition_project_updates(
        self,
        plan: tuple[_FitDatasetInitialConditionApplyItem, ...],
    ) -> None:
        batch_store = getattr(self, "_batch_store", None)
        batch_model = getattr(self, "_batch_model", None)
        dataset_manager = self._require_fitting_ports().dataset_manager
        if batch_store is None or dataset_manager is None:
            raise RuntimeError("Batch initial conditions project apply is unavailable.")

        affected_rows: list[int] = []
        for item in plan:
            row = int(item.row)
            for species, value in item.canonical_updates.items():
                batch_store.set_value(row, str(species), f"{float(value):.6g}")
            if item.canonical_updates:
                affected_rows.append(row)
            if item.settings_sync_updates or item.mapping_sync_needed:
                settings = dataset_manager.get_fit_settings(item.dataset_id)
                for species, value in item.settings_sync_updates.items():
                    settings.initial_conditions[str(species)] = float(value)
                settings.batch_set_id = item.set_id
                settings.batch_set = item.set_name
                dataset_manager.update_fit_settings(item.dataset_id, settings)

        if batch_model is not None and affected_rows:
            for row in sorted(set(affected_rows)):
                top_left = batch_model.index(int(row), 0)
                bottom_right = batch_model.index(int(row), max(0, int(batch_model.columnCount()) - 1))
                batch_model.dataChanged.emit(
                    top_left,
                    bottom_right,
                    [QtCore.Qt.DisplayRole, QtCore.Qt.BackgroundRole],
                )
            if hasattr(batch_model, "validate_rows"):
                batch_model.validate_rows(sorted(set(affected_rows)))

    def _apply_fit_results_to_project(
        self,
        scope: str,
        parameters: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
    ) -> bool | str:
        plan = self._build_fit_project_apply_plan(scope, parameters, dataset_params)
        if plan.is_true_noop:
            if plan.parameter_outcome is not None and plan.parameter_outcome.warning_messages:
                return "\n".join(str(message) for message in plan.parameter_outcome.warning_messages)
            return True

        if plan.needs_slider_guard:
            guard = getattr(self, "_guard_slider_transaction_invalidation", None)
            if callable(guard):
                if not guard(action_text="Applying fitted parameters to the project"):
                    return False

        applied_any = False
        parameter_outcome = plan.parameter_outcome
        parameter_warnings = list(parameter_outcome.warning_messages) if parameter_outcome is not None else []
        parameter_apply_failed = False
        if plan.parameter_delta is not None and plan.can_execute_parameter_apply:
            parameter_apply_status = self._write_fit_results_to_mechanism(
                dict(parameters),
                parameter_delta=plan.parameter_delta,
                refresh_display=False,
            )
            parameter_outcome = self._build_fit_parameter_apply_outcome(
                plan.parameter_delta,
                write_status=parameter_apply_status,
            )
        if parameter_outcome is not None:
            parameter_warnings = list(parameter_outcome.warning_messages)
            applied_any = bool(parameter_outcome.applied_any)
            parameter_apply_failed = bool(parameter_outcome.rewrite_failed)
        if plan.initial_condition_items:
            self._apply_fit_initial_condition_project_updates(plan.initial_condition_items)
            applied_any = True

        batch_cache = getattr(getattr(self, "_sim_controller", None), "batch_cache", None)
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        if batch_cache is not None and active_cache_key and plan.canonical_ic_affected_set_ids:
            merged_invalidated: list[str] = [
                str(set_id)
                for set_id in (getattr(batch_cache, "active_cache_invalidated_set_ids", None) or ())
                if str(set_id)
            ]
            for set_id in plan.canonical_ic_affected_set_ids:
                if set_id not in merged_invalidated:
                    merged_invalidated.append(str(set_id))
            batch_cache.active_cache_invalidated_set_ids = tuple(merged_invalidated) or None

        if plan.needs_display_refresh and callable(getattr(self, "_refresh_batch_display_from_focus_and_shown", None)):
            self._refresh_batch_display_from_focus_and_shown()
        if parameter_apply_failed and applied_any:
            parameter_warnings.append(
                "Fitted parameter values were not written to the current mechanism text."
            )
        if parameter_warnings:
            warning_text = "\n".join(str(message) for message in parameter_warnings)
            if applied_any:
                return (
                    "Applied remaining valid project updates, but skipped these parameter updates:\n"
                    f"{warning_text}"
                )
            return warning_text
        if parameter_apply_failed:
            return False
        return True

    def _apply_dataset_initial_updates(self, dataset_id: str, updates: Dict[str, float]) -> None:
        """Persist fitted initial concentrations back to dataset settings."""
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        try:
            settings = self._require_fitting_ports().dataset_manager.get_fit_settings(dataset_id)
        except DatasetManagerError:
            return
        changed = False
        for species, value in updates.items():
            if settings.initial_conditions.get(species) != value:
                settings.initial_conditions[species] = float(value)
                changed = True
        if changed:
            self._require_fitting_ports().dataset_manager.update_fit_settings(dataset_id, settings)
