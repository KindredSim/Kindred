# kindred/gui/mixins/fitting_mixin.py
"""
FittingMixin - Parameter fitting operations for MainWindow.

Provides methods for:
- Fitting configuration dialogs
- Parameter override/update in mechanism DSL
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
from typing import Callable, Dict, Mapping, Optional, TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.simulator.dsl_text_update import (
    StepParameterUpdateOutcome,
    analyze_parameter_updates_to_dsl_text,
    authoritative_parameter_values_match,
)
from kindred.gui.diagnostics import record_best_effort_failure as record_gui_best_effort_failure
from kindred.gui.fitting.constants import (
    FITTING_MAX_NFEV_RANGE,
    FITTING_METHODS,
    FITTING_SEED_RANGE,
    FITTING_SOLVERS,
)
from kindred.gui.ports import AuthoritativeRewriteStatus
from kindred.gui.mixins.ports import FittingMixinPorts
from kindred.gui.ui_helpers import safe_float_parse, setup_scientific_validator

logger = logging.getLogger(__name__)

_FITTING_KEY_TO_SHORT: dict[str, str] = {
    "fitting_method": "method",
    "fitting_max_nfev": "max_nfev",
    "fitting_ftol": "ftol",
    "fitting_xtol": "xtol",
    "fitting_use_seed": "use_seed",
    "fitting_seed": "seed",
    "fitting_solver": "solver",
    "fitting_rtol": "rtol",
    "fitting_atol": "atol",
}

_FIT_PARAMETER_APPLY_STATUS_REWRITTEN = "rewritten"
_FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT = "already_current"
_FIT_PARAMETER_APPLY_STATUS_FAILED = "failed"
_FIT_PARAMETER_APPLY_STATUS_CANCELED = "canceled"
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
class _FitParameterApplyWriteResult:
    status: str
    delta: _FitParameterApplyDelta


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
    fit_parameter_dirty_discard_set_ids: tuple[str, ...]
    fit_ic_dirty_discard_set_ids: tuple[str, ...]
    has_dataset_settings_sync: bool
    can_execute_parameter_apply: bool
    needs_display_refresh: bool
    is_true_noop: bool


if TYPE_CHECKING:
    from kindred.gui.fitting.launch import GlobalFitLaunchContext


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
        raise RuntimeError("FittingMixin ports are not initialized.")

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

    def _write_fit_results_to_mechanism(
        self,
        parameters: dict,
        *,
        parameter_delta: Optional[_FitParameterApplyDelta] = None,
        refresh_display: bool = True,
        before_publish: Callable[[], None] | None = None,
    ) -> _FitParameterApplyWriteResult:
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
        get_canonical_reactions = getattr(self, "mechanism_reactions_text_raw", None)
        if not callable(get_canonical_reactions):
            raise RuntimeError("Canonical mechanism Reactions getter is unavailable.")
        before_text = str(get_canonical_reactions() or "")
        preview_outcome = self._build_fit_parameter_apply_outcome(delta, before_text=before_text)
        force_fresh_boundary = bool(
            callable(getattr(self, "_reactions_draft_dirty", None))
            and self._reactions_draft_dirty()
        )
        if not force_fresh_boundary and preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE:
            logger.info("Fitted parameter project apply found unappliable requested targets; no rewrite will be performed")
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE,
                delta,
            )
        if not force_fresh_boundary and preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT and not delta.has_real_change:
            logger.info("Fitted parameter project apply made no authoritative numeric change")
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT,
                delta,
            )
        if not force_fresh_boundary and preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT and before_text == str(delta.updated_text or ""):
            logger.info("Fitted parameter project apply found the mechanism text already current")
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT,
                delta,
            )
        if not force_fresh_boundary and preview_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT:
            logger.info("Fitted parameter project apply found no authoritative DSL rewrite to perform")
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT,
                delta,
            )

        rewrite = getattr(self, "rewrite_authoritative_reactions_text", None)
        if not callable(rewrite):
            raise RuntimeError("Authoritative mechanism Reactions rewriter is unavailable.")

        applied_delta = delta

        def _rewrite(current_text: str) -> str:
            nonlocal applied_delta
            applied_delta = self._build_fit_parameter_apply_delta(
                normalized_parameters,
                source_text=str(current_text or ""),
            )
            return str(applied_delta.updated_text or "")

        outcome = rewrite(
            action_label="Applying fitted parameters to the project",
            description="Global Fit: update Reactions",
            rewrite_text=_rewrite,
            record_undo=True,
            transition_source="global_fit_project_apply",
            before_publish=before_publish,
            discard_dirty_preview=False,
        )
        if outcome.status is AuthoritativeRewriteStatus.CANCELLED:
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_CANCELED,
                applied_delta,
            )
        if outcome.status is AuthoritativeRewriteStatus.FAILED:
            logger.warning(
                "Fitted parameter project apply rewrite failed: %s",
                str(getattr(outcome, "error", "") or "authoritative rewrite failed"),
            )
            return _FitParameterApplyWriteResult(
                _FIT_PARAMETER_APPLY_STATUS_FAILED,
                applied_delta,
            )

        missing = [str(name) for name in applied_delta.missing]
        update_errors = [dict(error) for error in applied_delta.update_errors]
        for name in missing:
            logger.warning("Parameter %r not updated (missing/invalid)", name)
        for error in update_errors:
            logger.warning(
                "Parameter %r failed while applying fitted value (%s): %s",
                error.get("name"),
                error.get("exc_type", "Exception"),
                error.get("message", ""),
            )
        applied_outcome = self._build_fit_parameter_apply_outcome(
            applied_delta,
            before_text=str(applied_delta.current_text or ""),
        )
        final_status = _FIT_PARAMETER_APPLY_STATUS_REWRITTEN
        if applied_outcome.status == _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE:
            logger.info("Fitted parameter project apply found unappliable requested targets after draft resolution")
            final_status = _FIT_PARAMETER_APPLY_STATUS_UNAPPLIABLE
        elif applied_outcome.status == _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT:
            logger.info("Fitted parameter project apply found no authoritative DSL rewrite after draft resolution")
            final_status = _FIT_PARAMETER_APPLY_STATUS_ALREADY_CURRENT
        else:
            after_text = str(get_canonical_reactions() or "")
            if after_text == str(applied_delta.current_text or ""):
                logger.warning("Fitted parameter project apply did not rewrite mechanism text")
                return _FitParameterApplyWriteResult(
                    _FIT_PARAMETER_APPLY_STATUS_FAILED,
                    applied_delta,
                )

        if bool(getattr(outcome, "source_updated", False)):
            self._refresh_after_fit_authoritative_source_change(
                applied_text=str(applied_delta.updated_text or ""),
                refresh_display=bool(refresh_display),
            )

        if final_status == _FIT_PARAMETER_APPLY_STATUS_REWRITTEN:
            logger.info(
                "Wrote %d fitted parameter(s) to mechanism editor",
                len(normalized_parameters),
            )
        return _FitParameterApplyWriteResult(
            final_status,
            applied_delta,
        )

    def _refresh_after_fit_authoritative_source_change(
        self,
        *,
        applied_text: str,
        refresh_display: bool,
    ) -> None:
        sync_authoritative_source = getattr(
            self,
            "_sync_mechanism_session_owner_after_authoritative_widget_write",
            None,
        )
        if callable(sync_authoritative_source):
            sync_authoritative_source(dispatch_consumers=False)
        try:
            self._variable_runtime.extract_and_populate_variables(
                preserve_visibility=True,
            )
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.parameter_slider_refresh",
                message="Failed to refresh parameter sliders after fitted parameter apply",
                exc=exc,
            )
        try:
            self._variable_runtime.update_parameter_table_from_sliders()
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.parameter_table_refresh",
                message="Failed to refresh parameter table after fitted parameter apply",
                exc=exc,
            )
        if callable(getattr(self, "_sync_batch_species_columns", None)):
            mechanism_initials = self._extract_mechanism_initials(str(applied_text or ""))
            self._sync_batch_species_columns(list(mechanism_initials.keys()))
        if refresh_display and callable(getattr(self, "_refresh_batch_display_from_request_scope", None)):
            self._refresh_batch_display_from_request_scope()

    def _prepare_fit_parameter_project_text(
        self,
        parameters: dict,
        *,
        source_text: str | None = None,
    ) -> Dict[str, object]:
        get_canonical_reactions = getattr(self, "mechanism_reactions_text_raw", None)
        if source_text is None and not callable(get_canonical_reactions):
            raise RuntimeError("Canonical mechanism Reactions getter is unavailable.")
        current_text = (
            str(source_text or "")
            if source_text is not None
            else str(get_canonical_reactions() or "")
        )
        normalized_parameters = {str(name): value for name, value in dict(parameters or {}).items()}
        generated_fast_metadata = self._fit_generated_fast_parameter_metadata(current_text)
        generated_fast_parameters = {
            str(name): value
            for name, value in normalized_parameters.items()
            if str(name) in generated_fast_metadata
        }
        source_authoritative_values = {
            str(name): float(meta["value"])
            for name, meta in generated_fast_metadata.items()
            if isinstance(meta, Mapping) and meta.get("value") is not None
        }
        ordinary_parameters = {
            str(name): value
            for name, value in normalized_parameters.items()
            if str(name) not in generated_fast_parameters
        }
        analysis = analyze_parameter_updates_to_dsl_text(
            current_text,
            ordinary_parameters,
            step_constraint_context=self._current_fit_step_constraint_context(),
        )
        updated_text = str(analysis.updated_text)
        missing = list(analysis.missing)
        update_errors = [dict(error) for error in analysis.update_errors]
        if generated_fast_parameters:
            generated_writer = getattr(self, "_apply_energy_overrides_to_computational_mode_fast_equilibria", None)
            if callable(generated_writer):
                before_generated = updated_text
                updated_text = str(
                    generated_writer(
                        updated_text,
                        overrides=generated_fast_parameters,
                        override_metadata=generated_fast_metadata,
                    )
                )
                for name, value in generated_fast_parameters.items():
                    current_value = generated_fast_metadata.get(name, {}).get("value")
                    if (
                        updated_text == before_generated
                        and current_value is not None
                        and not authoritative_parameter_values_match(current_value, value)
                    ):
                        missing.append(str(name))
            else:
                missing.extend(str(name) for name in generated_fast_parameters)
        return {
            "current_text": current_text,
            "updated_text": updated_text,
            "missing": missing,
            "update_errors": update_errors,
            "step_outcomes": tuple(analysis.step_outcomes),
            "source_authoritative_values": source_authoritative_values,
        }

    @staticmethod
    def _fit_generated_fast_parameter_metadata(source_text: str) -> Dict[str, Dict[str, object]]:
        try:
            from kindred.core.equilibrium_rate_authority import generated_fast_equilibrium_runtime_parameter_name
            from kindred.core.mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataKeys
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            mechanism = parse_dsl_to_mechanism(str(source_text or ""), initials={})
        except Exception:
            return {}
        metadata_by_name: Dict[str, Dict[str, object]] = {}
        mechanism_meta = getattr(mechanism, "metadata", {}) or {}
        energy_unit = str(
            mechanism_meta.get(MechanismMetadataKeys.ENERGY_UNIT)
            if isinstance(mechanism_meta, dict)
            else "kJ/mol"
        ) or "kJ/mol"
        for eq in list(getattr(mechanism, "equilibria", []) or []):
            metadata = dict(getattr(eq, "metadata", {}) or {})
            name = generated_fast_equilibrium_runtime_parameter_name(eq)
            if name is None:
                continue
            try:
                dG_eq_j = float(metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL))
            except Exception:
                continue
            try:
                kf_fixed = float(getattr(eq, "kf"))
            except Exception:
                kf_fixed = metadata.get("kf")
            try:
                std_ratio = float(metadata.get("std_ratio"))
            except Exception:
                std_ratio = metadata.get("std_ratio")
            source_energy_unit = str(
                metadata.get(EquilibriumMetadataKeys.SOURCE_ENERGY_UNIT)
                or energy_unit
                or "kJ/mol"
            )
            source_unit_conv = UnitsModel(energy_unit=source_energy_unit)
            source_value = float(source_unit_conv.from_jmol(dG_eq_j))
            metadata_by_name[str(name)] = {
                "type": "energy",
                "role": "dG_eq_fast",
                "cm_id": str(metadata.get("cm_id") or ""),
                "unit": source_energy_unit,
                "source_unit": source_energy_unit,
                "kf_fixed": kf_fixed,
                "std_ratio": std_ratio,
                "value": source_value,
            }
        return metadata_by_name

    def _current_fit_step_constraint_context(self) -> Dict[str, object]:
        ports = self._require_fitting_ports()
        wegscheider_enabled = ports.wegscheider_cyclicity_enabled_getter()
        if not isinstance(wegscheider_enabled, bool):
            raise RuntimeError("Fitting constraint context requires an explicit Wegscheider setting.")
        return {
            "temperature_K": float(ports.temperature_getter()),
            "wegscheider_cyclicity_enabled": wegscheider_enabled,
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
        if write_status == _FIT_PARAMETER_APPLY_STATUS_CANCELED:
            return _FitParameterApplyOutcome(
                status=_FIT_PARAMETER_APPLY_STATUS_CANCELED,
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

    def _build_fit_parameter_apply_delta(
        self,
        parameters: Dict[str, float],
        *,
        source_text: str | None = None,
    ) -> _FitParameterApplyDelta:
        normalized_parameters = {
            str(name): float(value)
            for name, value in dict(parameters or {}).items()
        }
        update = self._prepare_fit_parameter_project_text(
            normalized_parameters,
            source_text=source_text,
        )
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
                else self._source_parameter_value_changed(
                    str(name),
                    float(value),
                    update=update,
                )
            )
        )
        missing = tuple(str(name) for name in update["missing"])
        update_errors = tuple(dict(error) for error in update["update_errors"])
        warning_messages = self._fit_parameter_warning_messages(missing, update_errors, step_outcomes)
        return _FitParameterApplyDelta(
            parameters=dict(normalized_parameters),
            authoritative_values=dict(update.get("source_authoritative_values") or {}),
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

    @staticmethod
    def _source_parameter_value_changed(
        name: str,
        value: float,
        *,
        update: Mapping[str, object],
    ) -> bool:
        source_values = dict(update.get("source_authoritative_values") or {})
        if str(name) in source_values:
            return not authoritative_parameter_values_match(
                source_values.get(str(name)),
                float(value),
            )
        if str(name) in set(str(item) for item in (update.get("missing") or ())):
            return False
        return str(update.get("updated_text") or "") != str(update.get("current_text") or "")

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
        method_combo.addItems(list(FITTING_METHODS))
        method_default = defaults.get("method", "trf")
        if method_default in FITTING_METHODS:
            method_combo.setCurrentText(method_default)
        algo_layout.addRow("Method:", method_combo)

        max_nfev_spin = QtWidgets.QSpinBox()
        max_nfev_spin.setRange(*FITTING_MAX_NFEV_RANGE)
        max_nfev_spin.setValue(int(defaults.get("max_nfev", 1000)))
        algo_layout.addRow("Max evaluations:", max_nfev_spin)

        ftol_edit = QtWidgets.QLineEdit(str(defaults.get("ftol", "1e-10")))
        _ftol_val = setup_scientific_validator(ftol_edit)
        algo_layout.addRow("ftol:", ftol_edit)

        xtol_edit = QtWidgets.QLineEdit(str(defaults.get("xtol", "1e-10")))
        _xtol_val = setup_scientific_validator(xtol_edit)
        algo_layout.addRow("xtol:", xtol_edit)

        use_seed_check = QtWidgets.QCheckBox("Use fixed random seed")
        use_seed_check.setChecked(bool(defaults.get("use_seed", True)))
        seed_spin = QtWidgets.QSpinBox()
        seed_spin.setRange(*FITTING_SEED_RANGE)
        seed_spin.setValue(int(defaults.get("seed", 42)))
        seed_spin.setEnabled(use_seed_check.isChecked())
        use_seed_check.toggled.connect(seed_spin.setEnabled)
        algo_layout.addRow(use_seed_check, seed_spin)

        layout.addWidget(algo_group)

        integration_group = QtWidgets.QGroupBox("Integration")
        integration_layout = QtWidgets.QFormLayout(integration_group)

        solver_combo = QtWidgets.QComboBox()
        solver_combo.addItems(list(FITTING_SOLVERS))
        solver_name = normalize_solver_name(defaults.get("solver", "BDF"))
        solver_combo.setCurrentText(solver_name)
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

        def _get_reactions_text() -> str:
            get_canonical_reactions = getattr(self, "mechanism_reactions_text_raw", None)
            if not callable(get_canonical_reactions):
                return ""
            try:
                return str(get_canonical_reactions() or "")
            except Exception:
                return ""

        rewrite_reactions_text = getattr(self, "rewrite_authoritative_reactions_text", None)
        if not callable(rewrite_reactions_text):
            raise RuntimeError("Authoritative mechanism Reactions rewriter is unavailable.")

        return GlobalFitLaunchContext(
            parent=self,
            dataset_registry=ports.dataset_registry,
            dataset_fit_settings_store=ports.dataset_fit_settings_store,
            dataset_view_publisher=ports.dataset_view_publisher,
            mechanism_parameter_scan_owner=ports.mechanism_parameter_scan_owner,
            mechanism_text_getter=lambda: str(self._get_mechanism_text() or ""),
            reactions_text_getter=_get_reactions_text,
            reactions_text_rewriter=rewrite_reactions_text,
            extract_mechanism_initials=self._extract_mechanism_initials,
            record_best_effort_failure=self._record_fitting_best_effort_failure,
            set_status=self._set_fitting_status,
            sync_batch_species_columns=self._sync_batch_species_columns,
            batch_initials_for_row=self._batch_initials_for_row,
            fitting_settings_getter=self._get_global_fit_launch_settings,
            num_points_getter=ports.num_points_getter,
            register_fit_window=self._fitting_runtime_input_publisher.register_window,
            apply_fit_results_to_project=self._apply_fit_results_to_project,
            load_fitting_defaults=self._get_fitting_session_defaults,
            batch_store=getattr(self, "_batch_store", None),
            batch_model=getattr(self, "_batch_model", None),
            batch_table=getattr(self, "_batch_table", None),
        )

    def _run_global_fit(self):
        """Delegate global-fit launch ownership to the fitting package."""
        from kindred.gui.fitting.launch import launch_global_fit_session

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
        from kindred.gui.controllers.dataset_errors import DatasetOwnerError

        updates_by_dataset = self._extract_fit_initial_condition_updates(dataset_params)
        has_requested_updates = bool(updates_by_dataset)
        if not updates_by_dataset:
            return (), False

        batch_store = getattr(self, "_batch_store", None)
        if batch_store is None:
            raise RuntimeError("Initial conditions store is unavailable.")

        dataset_fit_settings_store = self._require_fitting_ports().dataset_fit_settings_store

        plan: list[_FitDatasetInitialConditionApplyItem] = []
        for dataset_id, species_updates in updates_by_dataset.items():
            try:
                settings = dataset_fit_settings_store.get_fit_settings(dataset_id)
            except DatasetOwnerError as exc:
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

        fit_parameter_dirty_discard_set_ids: tuple[str, ...] = ()
        if parameter_delta is not None:
            fit_parameter_dirty_discard_set_ids = self._fit_parameter_dirty_preview_set_ids()

        initial_condition_items: tuple[_FitDatasetInitialConditionApplyItem, ...] = ()
        if normalized_scope in {"initial_conditions", "both"}:
            initial_condition_items, has_initial_updates = self._build_fit_initial_condition_apply_items(dataset_params)
            if not has_initial_updates:
                raise RuntimeError("No fitted initial-condition values are available to apply.")

        fit_ic_dirty_discard_set_ids: list[str] = []
        has_dataset_settings_sync = False
        has_canonical_ic_change = False
        for item in initial_condition_items:
            set_id = str(item.set_id or "").strip()
            item_changes_project = bool(
                item.canonical_updates
                or item.settings_sync_updates
                or item.mapping_sync_needed
            )
            item_has_dirty_preview = self._fit_initial_condition_set_has_dirty_preview(set_id)
            if (
                set_id
                and (item_changes_project or item_has_dirty_preview)
                and set_id not in fit_ic_dirty_discard_set_ids
            ):
                fit_ic_dirty_discard_set_ids.append(set_id)
            if item.settings_sync_updates or item.mapping_sync_needed:
                has_dataset_settings_sync = True
            if item.canonical_updates:
                has_canonical_ic_change = True

        has_parameter_warnings = bool(parameter_outcome is not None and parameter_outcome.warning_messages)
        can_execute_parameter_apply = bool(parameter_outcome is not None and parameter_outcome.rewrite_intended)
        has_fit_parameter_dirty_discard = bool(fit_parameter_dirty_discard_set_ids)
        has_fit_ic_dirty_discard = bool(fit_ic_dirty_discard_set_ids)
        return _FitProjectApplyPlan(
            scope=normalized_scope,
            parameter_delta=parameter_delta,
            parameter_outcome=parameter_outcome,
            initial_condition_items=tuple(initial_condition_items),
            fit_parameter_dirty_discard_set_ids=tuple(fit_parameter_dirty_discard_set_ids),
            fit_ic_dirty_discard_set_ids=tuple(fit_ic_dirty_discard_set_ids),
            has_dataset_settings_sync=bool(has_dataset_settings_sync),
            can_execute_parameter_apply=bool(can_execute_parameter_apply),
            needs_display_refresh=bool(
                can_execute_parameter_apply
                or has_fit_parameter_dirty_discard
                or has_canonical_ic_change
                or has_fit_ic_dirty_discard
            ),
            is_true_noop=not bool(
                can_execute_parameter_apply
                or has_fit_parameter_dirty_discard
                or has_canonical_ic_change
                or has_dataset_settings_sync
                or has_fit_ic_dirty_discard
                or has_parameter_warnings
            ),
        )

    def _fit_parameter_dirty_preview_set_ids(self) -> tuple[str, ...]:
        preview_session = getattr(self, "_preview_session", None)
        dirty_workspace_ids = getattr(preview_session, "local_mechanism_workspace_set_ids", None)
        if not callable(dirty_workspace_ids):
            return ()
        try:
            set_ids = tuple(
                dict.fromkeys(
                    str(set_id).strip()
                    for set_id in dirty_workspace_ids()
                    if str(set_id).strip()
                )
            )
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.dirty_preview_scope",
                message="Failed to inspect dirty slider preview scope before fitted parameter apply",
                exc=exc,
            )
            raise RuntimeError("Dirty slider preview discard scope is unavailable.") from exc
        return set_ids

    def _fit_initial_condition_set_has_dirty_preview(self, set_id: str) -> bool:
        set_id_s = str(set_id or "").strip()
        if not set_id_s:
            return False
        preview_session = getattr(self, "_preview_session", None)
        has_dirty_state = getattr(preview_session, "has_dirty_state_for_set", None)
        if not callable(has_dirty_state):
            return False
        try:
            return bool(has_dirty_state(set_id_s))
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.dirty_preview_probe",
                message="Failed to inspect dirty initial-condition preview state before fit apply",
                exc=exc,
            )
            return False

    def _normalize_fit_dirty_discard_set_ids(
        self,
        target_set_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(set_id).strip()
                for set_id in target_set_ids
                if str(set_id).strip()
            )
        )

    def _discard_fit_parameter_apply_dirty_preview_state(
        self,
        target_set_ids: tuple[str, ...],
        *,
        refresh_parameter_ui: bool = True,
    ) -> bool:
        normalized_set_ids = self._normalize_fit_dirty_discard_set_ids(target_set_ids)
        if not normalized_set_ids:
            return True
        preview_session = getattr(self, "_preview_session", None)
        clear_workspace = getattr(preview_session, "clear_local_mechanism_workspace", None)
        if not callable(clear_workspace):
            raise RuntimeError("Dirty slider preview discard is unavailable.")
        changed = False
        for set_id in normalized_set_ids:
            changed = bool(clear_workspace(set_id)) or changed
        if changed:
            sim_controller = getattr(self, "_sim_controller", None)
            invalidate_preview = getattr(sim_controller, "invalidate_slider_preview_work", None)
            if callable(invalidate_preview):
                invalidate_preview(target_set_ids=normalized_set_ids)
        if changed and bool(refresh_parameter_ui):
            try:
                self._variable_runtime.extract_and_populate_variables(
                    preserve_visibility=True,
                )
            except Exception as exc:
                self._record_fitting_best_effort_failure(
                    "fitting.apply_project.parameter_slider_refresh",
                    message="Failed to refresh parameter sliders after fitted parameter apply",
                    exc=exc,
                )
            try:
                self._variable_runtime.update_parameter_table_from_sliders()
            except Exception as exc:
                self._record_fitting_best_effort_failure(
                    "fitting.apply_project.parameter_table_refresh",
                    message="Failed to refresh parameter table after fitted parameter apply",
                    exc=exc,
                )
        try:
            has_parameter_workspace = getattr(preview_session, "has_local_mechanism_workspace", None)
            if not callable(has_parameter_workspace):
                raise RuntimeError("Dirty slider preview verification is unavailable.")
            return not any(bool(has_parameter_workspace(set_id)) for set_id in normalized_set_ids)
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.dirty_preview_discard",
                message="Failed to verify dirty slider preview discard after fitted parameter apply",
                exc=exc,
            )
            return False

    def _discard_fit_initial_condition_apply_dirty_preview_state(
        self,
        target_set_ids: tuple[str, ...],
    ) -> bool:
        normalized_set_ids = self._normalize_fit_dirty_discard_set_ids(target_set_ids)
        if not normalized_set_ids:
            return True
        discard_pending = getattr(self, "_discard_pending_slider_edits_for_invalidation", None)
        if not callable(discard_pending):
            raise RuntimeError("Dirty initial-condition preview discard is unavailable.")
        discard_pending(target_set_ids=normalized_set_ids)
        preview_session = getattr(self, "_preview_session", None)
        has_dirty_state = getattr(preview_session, "has_dirty_state_for_set", None)
        if not callable(has_dirty_state):
            raise RuntimeError("Dirty initial-condition preview verification is unavailable.")
        try:
            return not any(bool(has_dirty_state(set_id)) for set_id in normalized_set_ids)
        except Exception as exc:
            self._record_fitting_best_effort_failure(
                "fitting.apply_project.dirty_initial_condition_preview_discard",
                message="Failed to verify dirty initial-condition preview discard after fit apply",
                exc=exc,
            )
            return False

    def _apply_fit_initial_condition_project_updates(
        self,
        plan: tuple[_FitDatasetInitialConditionApplyItem, ...],
    ) -> None:
        batch_store = getattr(self, "_batch_store", None)
        batch_model = getattr(self, "_batch_model", None)
        dataset_fit_settings_store = self._require_fitting_ports().dataset_fit_settings_store
        if batch_store is None:
            raise RuntimeError("Batch initial conditions project apply is unavailable.")

        affected_rows: list[int] = []
        for item in plan:
            row = int(item.row)
            for species, value in item.canonical_updates.items():
                batch_store.set_value(row, str(species), f"{float(value):.6g}")
            if item.canonical_updates:
                affected_rows.append(row)
            if item.settings_sync_updates or item.mapping_sync_needed:
                settings = dataset_fit_settings_store.get_fit_settings(item.dataset_id)
                for species, value in item.settings_sync_updates.items():
                    settings.initial_conditions[str(species)] = float(value)
                settings.batch_set_id = item.set_id
                settings.batch_set = item.set_name
                dataset_fit_settings_store.update_fit_settings(item.dataset_id, settings)

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
        affected_set_ids = tuple(
            dict.fromkeys(
                str(item.set_id).strip()
                for item in plan
                if item.canonical_updates and str(item.set_id).strip()
            )
        )
        if affected_set_ids:
            self._canonical_initials_transition_owner.publish_canonical_initials_transition(
                affected_set_ids=affected_set_ids,
                transition_source="fitting_project_apply_initial_conditions",
                discard_dirty_preview=False,
            )

    def _apply_fit_results_to_project(
        self,
        scope: str,
        parameters: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
    ) -> bool | str:
        plan = self._build_fit_project_apply_plan(scope, parameters, dataset_params)
        needs_fresh_parameter_boundary = bool(
            plan.parameter_delta is not None
            and parameters
            and callable(getattr(self, "_reactions_draft_dirty", None))
            and self._reactions_draft_dirty()
        )

        if plan.is_true_noop and not needs_fresh_parameter_boundary:
            if plan.parameter_outcome is not None and plan.parameter_outcome.warning_messages:
                return "\n".join(str(message) for message in plan.parameter_outcome.warning_messages)
            return True

        publisher = getattr(self, "_fitting_runtime_input_publisher", None)
        transaction = getattr(publisher, "transaction", None)
        publish_transaction = nullcontext()
        if bool(plan.initial_condition_items) and plan.parameter_delta is not None and callable(
            transaction
        ):
            publish_transaction = transaction(reason="fitting project apply")

        with publish_transaction:
            applied_any = False
            parameter_outcome = plan.parameter_outcome
            parameter_warnings = (
                list(parameter_outcome.warning_messages)
                if parameter_outcome is not None
                else []
            )
            parameter_apply_failed = False
            parameter_dirty_preview_discarded = False
            parameter_dirty_preview_discard_failed = False

            def _discard_parameter_dirty_preview_after_commit() -> None:
                nonlocal parameter_dirty_preview_discarded
                nonlocal parameter_dirty_preview_discard_failed
                if not plan.fit_parameter_dirty_discard_set_ids:
                    return
                parameter_dirty_preview_discarded = bool(
                    self._discard_fit_parameter_apply_dirty_preview_state(
                        plan.fit_parameter_dirty_discard_set_ids,
                        refresh_parameter_ui=False,
                    )
                )
                parameter_dirty_preview_discard_failed = not parameter_dirty_preview_discarded

            if plan.parameter_delta is not None and (
                plan.can_execute_parameter_apply or needs_fresh_parameter_boundary
            ):
                parameter_write_result = self._write_fit_results_to_mechanism(
                    dict(parameters),
                    parameter_delta=plan.parameter_delta,
                    refresh_display=False,
                    before_publish=_discard_parameter_dirty_preview_after_commit,
                )
                if parameter_write_result.status == _FIT_PARAMETER_APPLY_STATUS_CANCELED:
                    return False
                parameter_outcome = self._build_fit_parameter_apply_outcome(
                    parameter_write_result.delta,
                    write_status=parameter_write_result.status,
                )
            if parameter_outcome is not None:
                parameter_warnings = list(parameter_outcome.warning_messages)
                applied_any = bool(parameter_outcome.applied_any)
                parameter_apply_failed = bool(parameter_outcome.rewrite_failed)
            if bool(parameter_outcome is not None and parameter_outcome.applied_any):
                if parameter_dirty_preview_discard_failed:
                    return False
                if not parameter_dirty_preview_discarded and not self._discard_fit_parameter_apply_dirty_preview_state(
                    plan.fit_parameter_dirty_discard_set_ids
                ):
                    return False
                applied_any = True
            if plan.initial_condition_items:
                self._apply_fit_initial_condition_project_updates(
                    plan.initial_condition_items,
                )
                if not self._discard_fit_initial_condition_apply_dirty_preview_state(
                    plan.fit_ic_dirty_discard_set_ids
                ):
                    return False
                applied_any = True

            if (
                plan.needs_display_refresh
                and applied_any
                and callable(getattr(self, "_refresh_batch_display_from_request_scope", None))
            ):
                self._refresh_batch_display_from_request_scope()
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
