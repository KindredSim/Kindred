"""Fit parameter presentation with current owner-facing hooks.

The widget construction in this module is intentionally close to the executable
Fit presentation.  The widget is only a view: accepted setup, request
construction, Fit results, defaults persistence, and worker ownership remain
outside it.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.analysis.fit_uncertainty import (
    BOOTSTRAP_SEED_RANGE,
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
)
from kindred.core.analysis.global_fit_execution import FitCoordinate

from kindred.core.fitting_method_policy import SUPPORTED_FIT_METHODS
from kindred.gui.fitting.constants import FIT_METHOD_LABELS
from kindred.core.project_document import (
    FIT_MAX_NFEV_RANGE,
    FIT_SEED_RANGE,
    FitDefaults,
    SUPPORTED_SOLVERS,
    default_fit_defaults,
    validate_fit_defaults,
)
from kindred.gui.fitting.launch import (
    FitInitialGlobalVariable,
    FitMappedDatasetSnapshot,
    FitOriginSnapshot,
    FitObservationSnapshot,
    FitParameterConfiguration,
    FitParameterVariable,
    FitSetupSnapshot,
    fit_uncertainty_coordinates,
    _replace_fit_parameter,
)
from kindred.gui.fitting.constants import (
    CENTRED_FIXED_EFFECTS_SCOPE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
)
from kindred.gui.parameter_enumeration import admit_gui_parameter_value
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = ["ParametersIcsTab"]


@dataclass(frozen=True, slots=True)
class _InvalidNumericDraft:
    raw_fields: tuple[tuple[str, str], ...]
    message: str


def _format_number(value: object) -> str:
    try:
        return repr(float(value))
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _format_scientific(value: object) -> str:
    # Keep the accepted binary64 value intact; this is only a projection.
    text = repr(float(value))
    if "e" not in text:
        return text
    base, exponent = text.split("e", 1)
    sign = ""
    digits = exponent
    if digits and digits[0] in "+-":
        sign = digits[0]
        digits = digits[1:]
    return f"{base}e{sign}{digits.lstrip('0') or '0'}"


def _resolve_coordinate_selection(
    values: Sequence[object],
    available: Sequence[FitCoordinate],
) -> set[FitCoordinate]:
    """Resolve stored coordinate identities without guessing display aliases."""

    roster = tuple(available)
    resolved: set[FitCoordinate] = set()
    for value in values:
        if isinstance(value, FitCoordinate):
            if value not in roster:
                raise ValueError(f"Unavailable Fit coordinate: {value.display_name!r}.")
            resolved.add(value)
            continue
        display = str(value or "").strip()
        matches = tuple(
            coordinate for coordinate in roster if coordinate.display_name == display
        )
        if len(matches) != 1:
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous Fit coordinate display name: {display!r}."
                )
            raise ValueError(f"Unavailable Fit coordinate: {display!r}.")
        resolved.add(matches[0])
    return resolved


class _FitLocalObservableDialog(QtWidgets.QDialog):
    """Author a Fit-local observable against one column per selected dataset."""

    def __init__(
        self,
        *,
        dataset_ids: Sequence[str],
        available_observations: Sequence[FitMappedDatasetSnapshot] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Define New Fit-local Observable")
        self.setModal(True)
        self.resize(520, 340)

        self._dataset_ids = tuple(
            dict.fromkeys(str(x).strip() for x in (dataset_ids or []) if str(x).strip())
        )
        self._available_observations = {
            mapped.dataset_id: mapped
            for mapped in self._normalize_available_observations(
                available_observations
            )
        }
        self._selection: tuple[
            str,
            str,
            tuple[FitObservationSnapshot, ...],
            str,
        ] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self._observable_tab = QtWidgets.QWidget(self)
        layout.addWidget(self._observable_tab, stretch=1)
        self._init_observable_tab()
        self._add_button_box(layout)

    @staticmethod
    def _normalize_available_observations(
        available_observations: Sequence[FitMappedDatasetSnapshot],
    ) -> tuple[FitMappedDatasetSnapshot, ...]:
        normalized: list[FitMappedDatasetSnapshot] = []
        for mapped in available_observations:
            if not isinstance(mapped, FitMappedDatasetSnapshot):
                raise TypeError(
                    "Observable choices must be typed mapped dataset snapshots."
                )
            if any(item.dataset_id == mapped.dataset_id for item in normalized):
                raise ValueError(
                    "Observable choices repeat a mapped dataset snapshot."
                )
            if any(
                not isinstance(observation, FitObservationSnapshot)
                for observation in mapped.observations
            ):
                raise TypeError("Mapped observations must be typed snapshots.")
            normalized.append(mapped)
        return tuple(normalized)

    def _init_observable_tab(self) -> None:
        observable_layout = QtWidgets.QVBoxLayout(self._observable_tab)
        header = QtWidgets.QLabel(
            "Define an algebraic observable for this Fit session.",
            self._observable_tab,
        )
        header.setWordWrap(True)
        observable_layout.addWidget(header)

        target_form = QtWidgets.QFormLayout()
        self._observable_column_combos: dict[str, QtWidgets.QComboBox] = {}
        for dataset_id in self._dataset_ids:
            entry = self._available_observations.get(dataset_id)
            combo = QtWidgets.QComboBox(self._observable_tab)
            combo.setObjectName(f"fit_local_observable_column_{dataset_id}")
            if entry is None:
                combo.addItem("No captured observations available")
                combo.setEnabled(False)
                label = f"{dataset_id} observed column:"
            else:
                for observation in entry.observations:
                    # Store the existing immutable snapshot itself.  The
                    # accepted selection must not reconstruct an observation
                    # from raw IDs.
                    combo.addItem(observation.raw_column, observation)
                label = f"{entry.display_name} observed column:"
            self._observable_column_combos[dataset_id] = combo
            target_form.addRow(label, combo)
        observable_layout.addLayout(target_form)

        scope_form = QtWidgets.QFormLayout()
        scalar_scope_label = QtWidgets.QLabel(
            "Per-dataset (choose Shared later in Parameters and enter a fresh value)",
            self._observable_tab,
        )
        scalar_scope_label.setWordWrap(True)
        scope_form.addRow("Auto-added scalars:", scalar_scope_label)
        ds_count = len(self._dataset_ids)
        self._observable_dataset_label = QtWidgets.QLabel(
            f"Applies to {ds_count} selected dataset{'' if ds_count == 1 else 's'}."
        )
        scope_form.addRow("", self._observable_dataset_label)
        observable_layout.addLayout(scope_form)

        self._new_observable_container = QtWidgets.QGroupBox(
            "Fit-local observable", self._observable_tab
        )
        new_layout = QtWidgets.QFormLayout(self._new_observable_container)
        self._new_observable_name_edit = QtWidgets.QLineEdit(
            self._new_observable_container
        )
        self._new_observable_name_edit.setPlaceholderText("e.g. signal")
        new_layout.addRow("Observable name:", self._new_observable_name_edit)
        self._new_observable_expr_edit = QtWidgets.QLineEdit(
            self._new_observable_container
        )
        self._new_observable_expr_edit.setPlaceholderText("e.g. scale * ([A] + [B])")
        new_layout.addRow("Expression:", self._new_observable_expr_edit)
        observable_layout.addWidget(self._new_observable_container)

        observable_help = QtWidgets.QLabel(
            "Species must be referenced using brackets: [A], [A]_0 (bare A is invalid).",
            self._observable_tab,
        )
        observable_help.setWordWrap(True)
        observable_layout.addWidget(observable_help)
        observable_layout.addStretch(1)

    def _add_button_box(self, layout: QtWidgets.QVBoxLayout) -> None:
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selection(
        self,
    ) -> tuple[str, str, tuple[FitObservationSnapshot, ...], str] | None:
        return self._selection

    def _selected_observations(self) -> tuple[FitObservationSnapshot, ...] | None:
        selected: list[FitObservationSnapshot] = []
        for dataset_id in self._dataset_ids:
            combo = self._observable_column_combos.get(dataset_id)
            if combo is None:
                return None
            observation = combo.currentData()
            if not isinstance(observation, FitObservationSnapshot):
                return None
            selected.append(observation)
        if len(selected) != len(self._dataset_ids):
            return None
        return tuple(selected)

    def accept(self) -> None:  # noqa: D401 - Qt override
        mode = "per_dataset"
        observations = self._selected_observations()
        if observations is None:
            return
        name = str(self._new_observable_name_edit.text()).strip()
        expression = str(self._new_observable_expr_edit.text()).strip()
        if not name or not expression:
            return
        self._selection = (name, expression, observations, mode)
        super().accept()


class _InitialConcentrationsDialog(QtWidgets.QDialog):
    """Choose one existing IC configuration and its consumption scope."""

    def __init__(
        self,
        *,
        species: Sequence[str],
        current_scopes: Mapping[str, str],
        current_globals: Mapping[str, FitInitialGlobalVariable] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Initial Concentrations")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)
        self._species = QtWidgets.QComboBox(self)
        self._species.addItems([str(item) for item in species if str(item)])
        self._scope = QtWidgets.QComboBox(self)
        self._scope.addItem("Local", "local")
        self._scope.addItem("Global", "global")
        if self._species.count():
            current = str(current_scopes.get(self._species.currentText(), "local"))
            index = self._scope.findData(current)
            self._scope.setCurrentIndex(max(0, index))
        self._species.currentTextChanged.connect(
            lambda value: self._scope.setCurrentIndex(
                max(0, self._scope.findData(current_scopes.get(value, "local")))
            )
        )
        layout.addRow("Species:", self._species)
        layout.addRow("Optimizer mode:", self._scope)
        self._global_initial = QtWidgets.QLineEdit(self)
        self._global_minimum = QtWidgets.QLineEdit(self)
        self._global_maximum = QtWidgets.QLineEdit(self)
        self._global_fit = QtWidgets.QCheckBox("Fit Global value", self)
        self._global_fit.setObjectName("global_fit_ic_fit")
        self._global_log10 = QtWidgets.QCheckBox("Fit in Log10", self)
        self._global_log10.setObjectName("global_fit_ic_log10")
        for edit in (
            self._global_initial,
            self._global_minimum,
            self._global_maximum,
        ):
            setup_scientific_validator(edit)
        layout.addRow("Global value:", self._global_initial)
        layout.addRow("Global minimum:", self._global_minimum)
        layout.addRow("Global maximum:", self._global_maximum)
        layout.addRow("Global Fit:", self._global_fit)
        layout.addRow("Global Log10:", self._global_log10)
        self._current_scopes = {
            str(name): str(scope).strip().lower()
            for name, scope in current_scopes.items()
        }
        self._current_globals = dict(current_globals or {})
        self._project_global_fields()
        self._species.currentTextChanged.connect(self._project_global_fields)
        self._scope.currentIndexChanged.connect(
            lambda _index: self._set_global_fields_enabled(
                self._scope.currentData() == "global"
            )
        )
        note = QtWidgets.QLabel(
            "Global activation requires fresh value and bounds. Cancel leaves "
            "the current Local setup unchanged; Local removes Global state.",
            self,
        )
        note.setWordWrap(True)
        layout.addRow("", note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow("", buttons)

    def selection(self) -> tuple[str, str] | None:
        species = str(self._species.currentText() or "").strip()
        scope = str(self._scope.currentData() or "").strip().lower()
        if not species or scope not in {"local", "global"}:
            return None
        return species, scope

    def _project_global_fields(self, *_args: object) -> None:
        species = self._species.currentText()
        current = (
            self._current_globals.get(species)
            if self._current_scopes.get(species) == "global"
            else None
        )
        blocker = QtCore.QSignalBlocker(self)
        if current is None:
            # A Local species has no dormant Global scientific values. Global
            # activation therefore starts with required, empty fields.
            self._global_initial.clear()
            self._global_minimum.clear()
            self._global_maximum.clear()
            self._global_fit.setChecked(False)
            self._global_log10.setChecked(False)
        else:
            self._global_initial.setText(_format_scientific(current.initial))
            self._global_minimum.setText(_format_scientific(current.minimum))
            self._global_maximum.setText(_format_scientific(current.maximum))
            self._global_fit.setChecked(bool(current.fit))
            self._global_log10.setChecked(bool(current.log10))
        del blocker
        self._set_global_fields_enabled(
            self._scope.currentData() == "global"
        )

    def _set_global_fields_enabled(self, enabled: bool) -> None:
        for edit in (
            self._global_initial,
            self._global_minimum,
            self._global_maximum,
            self._global_fit,
            self._global_log10,
        ):
            edit.setEnabled(bool(enabled))

    def global_configuration(self) -> FitInitialGlobalVariable | None:
        if self._scope.currentData() != "global":
            return None
        try:
            return FitInitialGlobalVariable(
                species=self._species.currentText(),
                initial=float(self._global_initial.text()),
                minimum=float(self._global_minimum.text()),
                maximum=float(self._global_maximum.text()),
                fit=self._global_fit.isChecked(),
                log10=self._global_log10.isChecked(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


class _ConvergenceDialog(QtWidgets.QDialog):
    """Transactional Fit convergence settings editor."""

    def __init__(self, defaults: FitDefaults, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Fit settings")
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self._ftol = QtWidgets.QLineEdit(_format_scientific(defaults.ftol), self)
        self._xtol = QtWidgets.QLineEdit(_format_scientific(defaults.xtol), self)
        setup_scientific_validator(self._ftol)
        setup_scientific_validator(self._xtol)
        form.addRow("Objective-improvement tolerance:", self._ftol)
        xtol_label = QtWidgets.QLabel("Parameter-step tolerance:", self)
        form.addRow(xtol_label, self._xtol)
        if defaults.method == DIFFERENTIAL_EVOLUTION_METHOD:
            xtol_label.hide()
            self._xtol.hide()
        note = QtWidgets.QLabel(
            "These values are applied together only after the Fit owner accepts the edit.",
            self,
        )
        note.setWordWrap(True)
        form.addRow("", note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[float, float] | None:
        try:
            values = (float(self._ftol.text()), float(self._xtol.text()))
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            return None
        return values

    def accept(self) -> None:  # noqa: D401 - Qt override
        if self.values() is None:
            return
        super().accept()


class _ProfileLikelihoodDialog(QtWidgets.QDialog):
    """Transactional Profile likelihood configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str] | None = None,
        configuration: ProfileLikelihoodConfig | None,
        default_selection: Sequence[FitCoordinate] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profile likelihood")
        self.setModal(True)
        self.resize(520, 420)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        display_names = dict(display_names or {})
        self._typed_identity_mode = any(
            coordinate.scope in {"centred_overall", "centred_deviation"}
            for coordinate in available
        )
        selected = (
            (
                set(default_selection)
                if configuration is not None
                and not configuration.parameter_names
                and default_selection is not None
                else _resolve_coordinate_selection(
                    configuration.parameter_names,
                    available,
                )
            )
            if configuration
            else set(
                default_selection if default_selection is not None else available
            )
        )
        for coordinate in available:
            if not isinstance(coordinate, FitCoordinate):
                raise TypeError("Uncertainty rosters require typed FitCoordinate values.")
            label = display_names.get(coordinate, coordinate.display_name)
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
            item.setToolTip(coordinate.display_name)
            item.setSelected(coordinate in selected)
        self._confidence = QtWidgets.QLineEdit(
            _format_scientific(configuration.confidence_level if configuration else 0.95),
            self,
        )
        self._points = QtWidgets.QSpinBox(self)
        self._points.setRange(2, 10_000)
        self._points.setValue(
            configuration.max_bracketing_steps_per_side if configuration else 8
        )
        self._budget = QtWidgets.QSpinBox(self)
        self._budget.setRange(1, 100_000)
        self._budget.setValue(configuration.max_nfev_per_point if configuration else 300)
        self._method = QtWidgets.QComboBox(self)
        ParametersIcsTab._add_method_items(self._method)
        method = configuration.optimizer_method if configuration else "trf"
        self._method.setCurrentIndex(max(0, self._method.findData(method)))
        setup_scientific_validator(self._confidence)
        form.addRow("Coordinates:", self._coordinates)
        form.addRow("Confidence level:", self._confidence)
        form.addRow("Maximum outward search steps per side", self._points)
        form.addRow("Maximum evaluations per refit:", self._budget)
        form.addRow("Optimization method:", self._method)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def configuration(self) -> ProfileLikelihoodConfig:
        names = tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self._coordinates.selectedItems()
        )
        if not self._typed_identity_mode:
            self._reject_ambiguous_display_names(names)
        return ProfileLikelihoodConfig(
            parameter_names=names,
            confidence_level=float(self._confidence.text()),
            max_bracketing_steps_per_side=self._points.value(),
            max_nfev_per_point=self._budget.value(),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    @staticmethod
    def _reject_ambiguous_display_names(
        coordinates: Sequence[FitCoordinate],
    ) -> None:
        labels = [coordinate.display_name for coordinate in coordinates]
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Selected uncertainty coordinates have ambiguous display names; "
                "select only one coordinate with each displayed identity."
            )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError):
            return
        super().accept()


class _ResidualBootstrapDialog(QtWidgets.QDialog):
    """Transactional residual Bootstrap configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str] | None = None,
        configuration: ResidualBootstrapConfig | None,
        default_selection: Sequence[FitCoordinate] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Residual bootstrap")
        self.setModal(True)
        self.resize(520, 440)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        display_names = dict(display_names or {})
        self._typed_identity_mode = any(
            coordinate.scope in {"centred_overall", "centred_deviation"}
            for coordinate in available
        )
        selected = (
            _resolve_coordinate_selection(
                configuration.parameter_names,
                available,
            )
            if configuration
            else set(
                default_selection if default_selection is not None else available
            )
        )
        for coordinate in available:
            if not isinstance(coordinate, FitCoordinate):
                raise TypeError("Uncertainty rosters require typed FitCoordinate values.")
            label = display_names.get(coordinate, coordinate.display_name)
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
            item.setToolTip(coordinate.display_name)
            item.setSelected(coordinate in selected)
        self._replicates = QtWidgets.QSpinBox(self)
        self._replicates.setRange(1, 1_000_000)
        self._replicates.setValue(configuration.replicates if configuration else 200)
        self._minimum = QtWidgets.QSpinBox(self)
        self._minimum.setRange(1, 1_000_000)
        self._minimum.setValue(
            configuration.minimum_successful_replicates if configuration else 50
        )
        self._fraction = QtWidgets.QLineEdit(
            _format_scientific(configuration.minimum_success_fraction if configuration else 0.8),
            self,
        )
        self._confidence = QtWidgets.QLineEdit(
            _format_scientific(configuration.confidence_level if configuration else 0.95),
            self,
        )
        self._seed = QtWidgets.QSpinBox(self)
        self._seed.setRange(*BOOTSTRAP_SEED_RANGE)
        bootstrap_seed = (
            configuration.seed
            if configuration is not None
            else int(
                np.random.default_rng().integers(
                    BOOTSTRAP_SEED_RANGE[0],
                    BOOTSTRAP_SEED_RANGE[1],
                    endpoint=True,
                )
            )
        )
        self._seed.setValue(bootstrap_seed)
        self._method = QtWidgets.QComboBox(self)
        ParametersIcsTab._add_method_items(self._method)
        method = configuration.optimizer_method if configuration else "trf"
        self._method.setCurrentIndex(max(0, self._method.findData(method)))
        setup_scientific_validator(self._fraction)
        setup_scientific_validator(self._confidence)
        form.addRow("Coordinates:", self._coordinates)
        form.addRow("Number of refits:", self._replicates)
        form.addRow("Minimum successful refits:", self._minimum)
        form.addRow("Required success rate:", self._fraction)
        form.addRow("Confidence level:", self._confidence)
        form.addRow("Random seed:", self._seed)
        form.addRow("Optimization method:", self._method)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def configuration(self) -> ResidualBootstrapConfig:
        names = tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self._coordinates.selectedItems()
        )
        if not self._typed_identity_mode:
            _ProfileLikelihoodDialog._reject_ambiguous_display_names(names)
        return ResidualBootstrapConfig(
            parameter_names=names,
            seed=self._seed.value(),
            replicates=self._replicates.value(),
            confidence_level=float(self._confidence.text()),
            minimum_successful_replicates=self._minimum.value(),
            minimum_success_fraction=float(self._fraction.text()),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError):
            return
        super().accept()


class ParametersIcsTab(QtWidgets.QWidget):
    """Presentation-only parameter, IC-scope, and optimizer panel."""

    setupSnapshotRequested = QtCore.Signal(object)
    addObservableRequested = QtCore.Signal(str, str, object, str)
    resetRequested = QtCore.Signal(str)
    viewStepsRequested = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    validityChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        origin: FitOriginSnapshot | None = None,
        parameter_choices: Mapping[str, object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if origin is not None and not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot or None.")
        self._origin = origin
        self._projected_setup: FitSetupSnapshot | None = None
        self._last_fit_parameters: dict[str, FitParameterVariable] = {}
        self._can_reset_initial = False
        self._can_reset_last_fit = False
        self._selected_dataset_ids: tuple[str, ...] = ()
        self._parameter_choices: dict[str, object] = {}
        self._invalid_numeric_drafts: dict[
            tuple[str, ...], _InvalidNumericDraft
        ] = {}
        self._projecting = False
        self._running = False
        self._set_parameter_choices(parameter_choices or {})
        self._build_ui()


    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_parameters_panel(), stretch=1)
    def _set_parameter_choices(self, choices: Mapping[str, object]) -> None:
        self._parameter_choices = {str(key): value for key, value in choices.items()}
        dataset_ids = self._parameter_choices.get("dataset_ids", ())
        self._selected_dataset_ids = tuple(
            dict.fromkeys(
                str(item).strip() for item in (dataset_ids or ()) if str(item).strip()
            )
        )

    def set_selected_dataset_ids(self, dataset_ids: Sequence[str]) -> None:
        self._selected_dataset_ids = tuple(
            dict.fromkeys(
                str(item).strip() for item in (dataset_ids or ()) if str(item).strip()
            )
        )
        # Selection controls the Fit-local observable target only. Parameter
        # rows always retain every captured dataset value, including excluded
        # datasets.


    # ------------------------------------------------------------------
    # Parameter presentation and complete snapshot replacements
    # ------------------------------------------------------------------


    def _add_fit_local_observable(self) -> None:
        if self._running:
            self.statusMessage.emit("Stop the current fit before adding an observable.")
            return
        dataset_ids = self._selected_dataset_ids
        if not dataset_ids:
            self.statusMessage.emit(
                "Select at least one dataset to include before adding an observable."
            )
            return
        dialog = _FitLocalObservableDialog(
            dataset_ids=dataset_ids,
            available_observations=self._available_observation_choices(dataset_ids),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selection = dialog.selection()
        if not selection:
            return
        # Window/owner handoff: name, expression, exact selected snapshot
        # roster (in selected-dataset order), and scalar scope.  No edit object
        # or raw-observation reconstruction crosses this boundary.
        name, expression, observations, scalar_scope = selection
        self.addObservableRequested.emit(
            name,
            expression,
            observations,
            scalar_scope,
        )

    def _add_initial_concentrations(self) -> None:
        if self._running:
            self.statusMessage.emit(
                "Stop the current fit before changing initial-condition scope."
            )
            return
        setup = self._projected_setup
        if setup is None:
            self.statusMessage.emit("Fit setup is not available.")
            return
        species = tuple(dict.fromkeys(item.species for item in setup.initials))
        if not species:
            self.statusMessage.emit("No retained initial concentrations are available.")
            return
        dialog = _InitialConcentrationsDialog(
            species=species,
            current_scopes=dict(setup.initial_scopes),
            current_globals={
                item.species: item for item in setup.global_initials
            },
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selection = dialog.selection()
        if selection is None:
            return
        species_name, scope = selection
        configured: FitInitialGlobalVariable | None = None
        if scope == "global":
            configured = dialog.global_configuration()
            if configured is None:
                self.statusMessage.emit(
                    "Global initial value and bounds are required and must be valid."
                )
                return
        else:
            local_rows = tuple(
                item
                for item in setup.initials
                if item.species == species_name
            )
            if not local_rows:
                self.statusMessage.emit(
                    "The selected species has no retained Local configuration."
                )
                return
        scope_map = dict(setup.initial_scopes)
        scope_map[species_name] = scope
        global_initials = tuple(
            item for item in setup.global_initials if item.species != species_name
        )
        if configured is not None:
            global_initials = (*global_initials, configured)
            global_initials = tuple(sorted(global_initials, key=lambda item: item.species))
        self.setupSnapshotRequested.emit(
            replace(
                setup,
                initial_scopes=tuple(scope_map.items()),
                global_initials=global_initials,
            )
        )

    def _reset_to_initial(self) -> None:
        if not self._can_reset_initial:
            self.statusMessage.emit("Initial Fit setup is not available.")
            return
        self.resetRequested.emit("initial")

    def _reset_to_last_fit(self) -> None:
        if not self._can_reset_last_fit:
            self.statusMessage.emit("No completed Fit is available to restore.")
            return
        self.resetRequested.emit("last_fit")


    def _available_observation_choices(
        self,
        dataset_ids: Sequence[str],
    ) -> tuple[FitMappedDatasetSnapshot, ...]:
        selected = {str(dataset_id) for dataset_id in dataset_ids}
        if self._origin is None:
            return ()
        return tuple(
            mapped
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in selected
        )

    _SCOPE_LABELS = {
        "shared": "One shared value",
        "per_dataset": "Independent value for each dataset",
        "centred_fixed_effects": CENTRED_FIXED_EFFECTS_SCOPE_LABEL,
    }

    def _build_parameters_panel(self) -> QtWidgets.QWidget:  # noqa: C901
        widget = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(widget)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)
        container = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(container)

        params_group = QtWidgets.QGroupBox("Parameters", container)
        params_group.setObjectName("fit_parameters_group")
        params_layout = QtWidgets.QVBoxLayout(params_group)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, params_group)
        split.setObjectName("fit_parameter_split")
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 58)
        split.setStretchFactor(1, 42)

        left = QtWidgets.QWidget(split)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.addWidget(QtWidgets.QLabel("Parameter list", left))
        self._param_table = QtWidgets.QTableWidget(0, 3, left)
        self._param_table.setObjectName("fit_parameter_roster")
        self._param_table.setHorizontalHeaderLabels(("Vary", "Parameter", "Last result"))
        self._param_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._param_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._param_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._param_table.horizontalHeader().setStretchLastSection(True)
        self._param_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._param_table.itemChanged.connect(self._on_roster_item_changed)
        self._param_table.itemSelectionChanged.connect(self._on_parameter_selection_changed)
        left_layout.addWidget(self._param_table, stretch=1)
        split.addWidget(left)

        right = QtWidgets.QWidget(split)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        self._selected_editor = QtWidgets.QGroupBox("Selected parameter", right)
        self._selected_editor.setObjectName("fit_selected_parameter_editor")
        editor_layout = QtWidgets.QVBoxLayout(self._selected_editor)
        form = QtWidgets.QFormLayout()
        self._vary_check = QtWidgets.QCheckBox("Vary this parameter", self._selected_editor)
        self._starting_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._minimum_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._maximum_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._scale_combo = QtWidgets.QComboBox(self._selected_editor)
        self._scale_combo.addItem("Linear", False)
        self._scale_combo.addItem("Logarithmic", True)
        self._scope_combo = QtWidgets.QComboBox(self._selected_editor)
        for scope, label in self._SCOPE_LABELS.items():
            self._scope_combo.addItem(label, scope)
        self._scope_combo.setObjectName("fit_parameter_scope")
        self._starting_edit.setObjectName("fit_parameter_starting_value")
        self._minimum_edit.setObjectName("fit_parameter_minimum")
        self._maximum_edit.setObjectName("fit_parameter_maximum")
        self._vary_check.setObjectName("fit_parameter_vary")
        self._scale_combo.setObjectName("fit_parameter_scale")
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            setup_scientific_validator(edit)
        form.addRow(self._vary_check)
        form.addRow("Starting value:", self._starting_edit)
        form.addRow("Finite minimum:", self._minimum_edit)
        form.addRow("Finite maximum:", self._maximum_edit)
        form.addRow("Scale:", self._scale_combo)
        form.addRow("Across datasets:", self._scope_combo)
        editor_layout.addLayout(form)

        self._revert_parameter_draft_button = QtWidgets.QPushButton(
            "Revert invalid values", self._selected_editor
        )
        self._revert_parameter_draft_button.setObjectName(
            "fit_parameter_revert_invalid_values"
        )
        self._revert_parameter_draft_button.setEnabled(False)
        self._revert_parameter_draft_button.clicked.connect(
            self._revert_selected_parameter_draft
        )
        editor_layout.addWidget(self._revert_parameter_draft_button)

        self._dataset_values_group = QtWidgets.QGroupBox("Values by dataset", self._selected_editor)
        self._dataset_values_group.setObjectName("fit_parameter_values_by_dataset")
        dataset_layout = QtWidgets.QVBoxLayout(self._dataset_values_group)
        self._dataset_values_table = QtWidgets.QTableWidget(0, 3, self._dataset_values_group)
        self._dataset_values_table.setObjectName("fit_parameter_dataset_values")
        self._dataset_values_table.setHorizontalHeaderLabels(
            ("Dataset", "Starting value", "Last result")
        )
        self._dataset_values_table.horizontalHeader().setStretchLastSection(True)
        self._dataset_values_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._dataset_values_table.itemChanged.connect(self._on_dataset_value_changed)
        dataset_layout.addWidget(self._dataset_values_table)
        editor_layout.addWidget(self._dataset_values_group)

        right_layout.addWidget(self._selected_editor)

        self._last_fit_group = QtWidgets.QGroupBox("Last Fit", right)
        last_group = self._last_fit_group
        last_layout = QtWidgets.QFormLayout(last_group)
        self._last_result_label = QtWidgets.QLabel("—", last_group)
        self._last_result_label.setWordWrap(True)
        last_layout.addRow("Last result:", self._last_result_label)
        right_layout.addWidget(last_group)
        right_layout.addStretch(1)
        split.addWidget(right)
        params_layout.addWidget(split, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self._add_button = QtWidgets.QToolButton(params_group)
        self._add_button.setText("Add parameter ▼")
        self._add_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QtWidgets.QMenu(self._add_button)
        self._add_initial_action = add_menu.addAction("Initial Concentrations")
        self._add_initial_action.triggered.connect(self._add_initial_concentrations)
        self._add_button.setMenu(add_menu)
        self._add_observable_button = QtWidgets.QPushButton(
            "Define Fit-local observable…", params_group
        )
        self._add_observable_button.clicked.connect(self._add_fit_local_observable)
        self._view_steps_button = QtWidgets.QPushButton("View mechanism steps", params_group)
        self._view_steps_button.setObjectName("global_fit_parameters_view_steps")
        self._view_steps_button.clicked.connect(self.viewStepsRequested)
        self._reset_initial_button = QtWidgets.QPushButton(
            "Reset selected to initial", params_group
        )
        self._reset_initial_button.clicked.connect(self._reset_to_initial)
        self._reset_last_button = QtWidgets.QPushButton(
            "Reset selected to last fit", params_group
        )
        self._reset_last_button.clicked.connect(self._reset_to_last_fit)
        for control in (
            self._add_button,
            self._add_observable_button,
            self._view_steps_button,
            self._reset_initial_button,
            self._reset_last_button,
        ):
            actions.addWidget(control)
        actions.addStretch(1)
        params_layout.addLayout(actions)
        layout.addWidget(params_group, stretch=1)

        self._analysis_cards_host = QtWidgets.QWidget(container)
        self._analysis_cards_layout = QtWidgets.QGridLayout(self._analysis_cards_host)
        self._analysis_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._analysis_cards_layout.setHorizontalSpacing(8)
        self._analysis_cards_layout.setVerticalSpacing(8)

        settings_group = QtWidgets.QGroupBox("Fit settings", self._analysis_cards_host)
        settings_group.setObjectName("fit_settings_group")
        settings_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        settings_form = QtWidgets.QFormLayout(settings_group)
        self._solver_combo = QtWidgets.QComboBox(settings_group)
        self._solver_combo.setObjectName("global_fit_solver")
        for solver in SUPPORTED_SOLVERS:
            self._solver_combo.addItem(str(solver), str(solver))
        self._method_combo = QtWidgets.QComboBox(settings_group)
        self._method_combo.setObjectName("global_fit_method")
        for method in SUPPORTED_FIT_METHODS:
            self._method_combo.addItem(FIT_METHOD_LABELS.get(method, method), method)
        defaults = default_fit_defaults()
        self._max_eval_spin = QtWidgets.QSpinBox(settings_group)
        self._max_eval_spin.setRange(*FIT_MAX_NFEV_RANGE)
        self._max_eval_spin.setValue(defaults.max_nfev)
        self._seed_check = QtWidgets.QCheckBox(
            "Use reproducible random seed", settings_group
        )
        self._seed_spin = QtWidgets.QSpinBox(settings_group)
        self._seed_spin.setRange(*FIT_SEED_RANGE)
        self._seed_spin.setValue(defaults.seed)
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        seed_widget = QtWidgets.QWidget(settings_group)
        seed_layout = QtWidgets.QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(self._seed_check)
        seed_layout.addWidget(self._seed_spin)
        settings_form.addRow("Simulation solver:", self._solver_combo)
        settings_form.addRow("Optimization method:", self._method_combo)
        settings_form.addRow("Maximum evaluations:", self._max_eval_spin)
        settings_form.addRow(seed_widget)
        self._settings_summary = QtWidgets.QLabel(settings_group)
        self._settings_summary.setObjectName("fit_settings_summary")
        self._settings_summary.setWordWrap(True)
        self._advanced_button = QtWidgets.QPushButton("Advanced…", settings_group)
        self._advanced_button.clicked.connect(self._open_convergence_dialog)
        settings_form.addRow("Accepted:", self._settings_summary)
        settings_form.addRow("Convergence:", self._advanced_button)

        profile_card = QtWidgets.QGroupBox("Profile likelihood", self._analysis_cards_host)
        profile_card.setObjectName("fit_profile_card")
        profile_layout = QtWidgets.QVBoxLayout(profile_card)
        self._profile_enabled = QtWidgets.QCheckBox("Enabled", profile_card)
        self._profile_summary = QtWidgets.QLabel("Not configured", profile_card)
        self._profile_summary.setWordWrap(True)
        self._profile_configure = QtWidgets.QPushButton("Configure…", profile_card)
        profile_layout.addWidget(self._profile_enabled)
        profile_layout.addWidget(self._profile_summary)
        profile_layout.addStretch(1)
        profile_layout.addWidget(self._profile_configure)

        bootstrap_card = QtWidgets.QGroupBox("Residual bootstrap", self._analysis_cards_host)
        bootstrap_card.setObjectName("fit_bootstrap_card")
        bootstrap_layout = QtWidgets.QVBoxLayout(bootstrap_card)
        self._bootstrap_enabled = QtWidgets.QCheckBox("Enabled", bootstrap_card)
        self._bootstrap_summary = QtWidgets.QLabel("Not configured", bootstrap_card)
        self._bootstrap_summary.setWordWrap(True)
        self._bootstrap_configure = QtWidgets.QPushButton("Configure…", bootstrap_card)
        bootstrap_layout.addWidget(self._bootstrap_enabled)
        bootstrap_layout.addWidget(self._bootstrap_summary)
        bootstrap_layout.addStretch(1)
        bootstrap_layout.addWidget(self._bootstrap_configure)

        self._analysis_cards = (settings_group, profile_card, bootstrap_card)
        for column, card in enumerate(self._analysis_cards):
            self._analysis_cards_layout.addWidget(card, 0, column)
            self._analysis_cards_layout.setColumnStretch(column, 1)
        layout.addWidget(self._analysis_cards_host)

        # Signals are attached after construction so projection can use one
        # coherent blocker set and cannot author a half-formed snapshot.
        self._solver_combo.currentIndexChanged.connect(self._emit_solver_edit)
        self._method_combo.currentIndexChanged.connect(
            self._emit_defaults_edit
        )
        self._max_eval_spin.valueChanged.connect(
            self._emit_defaults_edit
        )
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        self._seed_check.toggled.connect(
            self._emit_defaults_edit
        )
        self._seed_spin.valueChanged.connect(
            self._emit_defaults_edit
        )
        self._profile_enabled.toggled.connect(self._on_profile_toggled)
        self._bootstrap_enabled.toggled.connect(self._on_bootstrap_toggled)
        self._profile_configure.clicked.connect(self._open_profile_dialog)
        self._bootstrap_configure.clicked.connect(self._open_bootstrap_dialog)
        self._vary_check.toggled.connect(self._emit_selected_parameter_vary_edit)
        self._scale_combo.currentIndexChanged.connect(self._emit_selected_parameter_edit)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            edit.textChanged.connect(self._on_parameter_numeric_text_changed)
            edit.editingFinished.connect(self._emit_selected_parameter_edit)
        self._dataset_values_group.setVisible(False)
        self._selected_editor.setEnabled(False)
        self._project_defaults(defaults)
        return widget

    @staticmethod
    def _add_method_items(combo: QtWidgets.QComboBox) -> None:
        for method in SUPPORTED_FIT_METHODS:
            combo.addItem(FIT_METHOD_LABELS.get(method, method), method)

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if self._origin is not None and (
            snapshot.origin is not self._origin
        ):
            raise ValueError("Fit snapshot does not match this origin.")

    def _parameter_dataset_ids(
        self,
        snapshot: FitSetupSnapshot,
        parameters: Sequence[FitParameterVariable],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [item.dataset_id for item in snapshot.datasets]
                + [
                    dataset_id
                    for parameter in parameters
                    for dataset_id, _value in parameter.per_dataset_config.dataset_values
                ]
            )
        )

    def _dataset_display_name(self, dataset_id: str) -> str:
        if self._origin is not None:
            for mapped in self._origin.mapped_datasets:
                if mapped.dataset_id == dataset_id:
                    return str(mapped.display_name)
        if self._projected_setup is not None:
            for dataset in self._projected_setup.datasets:
                if dataset.dataset_id == dataset_id:
                    return str(dataset_id)
        return str(dataset_id)

    def _parameter_by_name(self, name: str) -> FitParameterVariable | None:
        if self._projected_setup is None:
            return None
        return next((item for item in self._projected_setup.parameters if item.name == name), None)

    def _display_parameter_name(self, name: str) -> str:
        parameter = self._parameter_by_name(name)
        return parameter.display_name if parameter is not None else name

    def _coordinate_display_name(self, coordinate: FitCoordinate) -> str:
        if not isinstance(coordinate, FitCoordinate):
            raise TypeError("Uncertainty roster entries require FitCoordinate values.")
        if coordinate.scope == "shared":
            return self._display_parameter_name(coordinate.name)
        if coordinate.scope == "centred_overall":
            return f"{self._display_parameter_name(coordinate.name)} — overall"
        if coordinate.scope == "dataset":
            return (
                f"{self._display_parameter_name(coordinate.name)} — "
                f"{self._dataset_display_name(str(coordinate.dataset_id))}"
            )
        return coordinate.display_name

    def _available_bootstrap_coordinates(self) -> tuple[FitCoordinate, ...]:
        return self._available_analysis_coordinates()

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self._projected_setup = snapshot
        blocker = QtCore.QSignalBlocker(self._param_table)
        self._projecting = True
        try:
            self._param_table.setRowCount(len(snapshot.parameters))
            for row, parameter in enumerate(snapshot.parameters):
                self._render_roster_row(row, parameter)
            self._solver_combo.setCurrentText(snapshot.solver)
            self._project_defaults(snapshot.defaults)
            self._project_advanced_analysis(snapshot)
        finally:
            self._projecting = False
            del blocker
        if snapshot.parameters:
            current = self._param_table.currentRow()
            self._param_table.selectRow(
                min(max(current, 0), len(snapshot.parameters) - 1)
            )
        else:
            self._clear_selected_editor()
        self._project_last_fit_columns()

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        if snapshot is not None:
            self._check_snapshot(snapshot)
            self._last_fit_parameters = {item.name: item for item in snapshot.parameters}
        else:
            self._last_fit_parameters.clear()
        self.set_reset_availability(
            initial=self._can_reset_initial,
            last_fit=snapshot is not None,
        )
        self._project_last_fit_columns()
        self._render_selected_parameter()

    def set_last_fit_presentation(self) -> None:
        """Label retained Last Fit values without changing their ownership."""

        self._last_fit_group.setTitle("Last Fit")
        header = "Last result"
        self._param_table.setHorizontalHeaderItem(
            2,
            QtWidgets.QTableWidgetItem(header),
        )
        self._dataset_values_table.setHorizontalHeaderItem(
            2,
            QtWidgets.QTableWidgetItem(header),
        )

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self.set_reset_availability(initial=True, last_fit=self._can_reset_last_fit)

    def set_reset_availability(self, *, initial: bool, last_fit: bool) -> None:
        self._can_reset_initial = bool(initial)
        self._can_reset_last_fit = bool(last_fit)
        self._reset_initial_button.setEnabled(self._can_reset_initial)
        self._reset_last_button.setEnabled(self._can_reset_last_fit)

    def _render_roster_row(self, row: int, parameter: FitParameterVariable) -> None:
        config = parameter.selected_config
        vary = QtWidgets.QTableWidgetItem()
        vary.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        vary.setCheckState(
            QtCore.Qt.CheckState.Checked if config.fit else QtCore.Qt.CheckState.Unchecked
        )
        vary.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        self._param_table.setItem(row, 0, vary)
        name_item = QtWidgets.QTableWidgetItem(parameter.display_name)
        name_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        name_item.setToolTip(f"{parameter.display_name}\nParameter ID: {parameter.name}")
        self._param_table.setItem(row, 1, name_item)
        last = QtWidgets.QTableWidgetItem(self._last_fit_summary(parameter.name))
        last.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        self._param_table.setItem(row, 2, last)

    def _last_fit_summary(self, name: str) -> str:
        last = self._last_fit_parameters.get(name)
        if last is None:
            return "—"
        if last.scope == "shared":
            return f"{_format_number(last.selected_config.value)} (Shared)"
        return (
            CENTRED_FIXED_EFFECTS_SCOPE_LABEL
            if last.scope == "centred_fixed_effects"
            else "Per dataset"
        )

    def _project_last_fit_columns(self) -> None:
        blocker = QtCore.QSignalBlocker(self._param_table)
        try:
            for row in range(self._param_table.rowCount()):
                item = self._param_table.item(row, 1)
                last = self._param_table.item(row, 2)
                if item is not None and last is not None:
                    last.setText(self._last_fit_summary(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")))
        finally:
            del blocker
        self._render_selected_parameter()

    def _on_roster_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 0:
            return
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if name:
            self._emit_vary_edit(
                name,
                item.checkState() == QtCore.Qt.CheckState.Checked,
            )

    def _on_parameter_selection_changed(self) -> None:
        if not self._projecting:
            self._render_selected_parameter()

    def _selected_parameter(self) -> FitParameterVariable | None:
        row = self._param_table.currentRow()
        item = self._param_table.item(row, 1) if row >= 0 else None
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        return self._parameter_by_name(name) if name else None

    def _clear_selected_editor(self) -> None:
        self._selected_editor.setEnabled(False)
        self._dataset_values_group.setVisible(False)
        self._last_result_label.setText("—")
        self._revert_parameter_draft_button.setEnabled(False)

    def _render_selected_parameter(self) -> None:  # noqa: C901
        parameter = self._selected_parameter()
        if parameter is None:
            self._clear_selected_editor()
            return
        self._selected_editor.setEnabled(True)
        config = parameter.selected_config
        validator_minimum = (
            -sys.float_info.max
            if parameter.scientific_category == "finite_scalar"
            else 0.0
        )
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            setup_scientific_validator(edit, minimum=validator_minimum)
        draft = self._invalid_numeric_drafts.get(("parameter", parameter.name))
        raw_fields = dict(draft.raw_fields) if draft is not None else {}
        blockers = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._vary_check,
                self._starting_edit,
                self._minimum_edit,
                self._maximum_edit,
                self._scale_combo,
                self._scope_combo,
                self._dataset_values_table,
            )
        ]
        try:
            self._project_scope_choices(parameter)
            self._vary_check.setChecked(config.fit)
            self._minimum_edit.setText(_format_number(config.minimum))
            self._maximum_edit.setText(_format_number(config.maximum))
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(config.log10))
            self._scope_combo.setCurrentIndex(self._scope_combo.findData(parameter.scope))
            if parameter.scope == "shared":
                self._starting_edit.setReadOnly(False)
                self._starting_edit.setText(_format_number(config.value))
            else:
                self._starting_edit.setReadOnly(True)
                self._starting_edit.setText("Values by dataset")
            controls = {
                "starting": self._starting_edit,
                "minimum": self._minimum_edit,
                "maximum": self._maximum_edit,
            }
            for field, edit in controls.items():
                edit.setStyleSheet("")
                edit.setToolTip("")
                if field not in raw_fields:
                    continue
                if field == "starting" and parameter.scope != "shared":
                    continue
                edit.setText(raw_fields[field])
                edit.setStyleSheet(
                    "background: rgb(255, 200, 200); color: rgb(80, 0, 0);"
                )
                edit.setToolTip(draft.message)
            self._render_dataset_values(parameter)
            last = self._last_fit_parameters.get(parameter.name)
            if last is None:
                self._last_result_label.setText("—")
            elif last.scope == "shared":
                self._last_result_label.setText(f"{_format_number(last.selected_config.value)} (Shared)")
            else:
                self._last_result_label.setText(
                    CENTRED_FIXED_EFFECTS_SCOPE_LABEL
                    if last.scope == "centred_fixed_effects"
                    else "Per dataset"
                )
        finally:
            del blockers
        self._revert_parameter_draft_button.setEnabled(
            any(
                len(key) > 1 and key[1] == parameter.name
                for key in self._invalid_numeric_drafts
            )
        )

    def _render_dataset_values(self, parameter: FitParameterVariable) -> None:
        visible = parameter.scope in {"per_dataset", "centred_fixed_effects"}
        self._dataset_values_group.setVisible(visible)
        if not visible:
            self._dataset_values_table.setRowCount(0)
            return
        snapshot = self._projected_setup
        dataset_ids = (
            self._parameter_dataset_ids(snapshot, (parameter,))
            if snapshot is not None
            else tuple(dataset_id for dataset_id, _value in parameter.per_dataset_config.dataset_values)
        )
        values = dict(parameter.per_dataset_config.dataset_values)
        last = self._last_fit_parameters.get(parameter.name)
        last_values = dict(last.selected_config.dataset_values) if last is not None and last.scope != "shared" else {}
        blocker = QtCore.QSignalBlocker(self._dataset_values_table)
        self._dataset_values_table.setRowCount(len(dataset_ids))
        for row, dataset_id in enumerate(dataset_ids):
            dataset_item = QtWidgets.QTableWidgetItem(self._dataset_display_name(dataset_id))
            dataset_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            dataset_item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset_id)
            dataset_item.setToolTip(f"{self._dataset_display_name(dataset_id)}\nDataset ID: {dataset_id}")
            self._dataset_values_table.setItem(row, 0, dataset_item)
            value_item = QtWidgets.QTableWidgetItem(
                _format_number(values[dataset_id]) if dataset_id in values else ""
            )
            value_item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset_id)
            value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            key = ("parameter_dataset", parameter.name, dataset_id, "initial")
            draft = self._invalid_numeric_drafts.get(key)
            if draft is not None:
                value_item.setText(dict(draft.raw_fields)["initial"])
                value_item.setBackground(QtGui.QColor(255, 200, 200))
                value_item.setForeground(QtGui.QColor(80, 0, 0))
                value_item.setToolTip(draft.message)
            self._dataset_values_table.setItem(row, 1, value_item)
            last_item = QtWidgets.QTableWidgetItem(
                _format_number(last_values[dataset_id]) if dataset_id in last_values else "—"
            )
            last_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            last_item.setToolTip(f"Dataset ID: {dataset_id}")
            self._dataset_values_table.setItem(row, 2, last_item)
        del blocker

    def _project_scope_choices(self, parameter: FitParameterVariable) -> None:
        """Offer only scopes supported by the accepted parameter authority."""

        allowed = (
            ("shared", "per_dataset")
            if parameter.fit_local_provenance is not None
            else tuple(self._SCOPE_LABELS)
        )
        current = tuple(
            str(self._scope_combo.itemData(index) or "")
            for index in range(self._scope_combo.count())
        )
        if current == allowed:
            return
        self._scope_combo.clear()
        for scope in allowed:
            self._scope_combo.addItem(self._SCOPE_LABELS[scope], scope)

    def _on_dataset_value_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 1:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        dataset_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        key = ("parameter_dataset", parameter.name, dataset_id, "initial")
        try:
            parsed = self._parse_number(
                item.text(), f"{parameter.name} value for dataset {dataset_id}"
            )
            admit_gui_parameter_value(
                parsed,
                scientific_category=parameter.scientific_category,
                allow_exact_zero=parameter.allow_exact_zero,
                field=f"{parameter.name} value for dataset {dataset_id}",
            )
            if key in self._invalid_numeric_drafts:
                self._store_invalid_numeric_draft(
                    key,
                    {"initial": item.text()},
                    "Apply the corrected parameter values or revert this draft.",
                )
                item.setBackground(QtGui.QColor(255, 200, 200))
                item.setForeground(QtGui.QColor(80, 0, 0))
                item.setToolTip(
                    "Apply the corrected parameter values or revert this draft."
                )
            self._emit_selected_parameter_edit()
        except (TypeError, ValueError, OverflowError) as exc:
            self._store_invalid_numeric_draft(
                key,
                {"initial": item.text()},
                str(exc),
            )
            item.setBackground(QtGui.QColor(255, 200, 200))
            item.setForeground(QtGui.QColor(80, 0, 0))
            item.setToolTip(str(exc))
            self.statusMessage.emit(str(exc))

    def _on_parameter_numeric_text_changed(self, _raw_text: str) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        try:
            self._parameter_projection_for_change(parameter)
        except (TypeError, ValueError, OverflowError) as exc:
            self._store_parameter_editor_draft(parameter, str(exc))
            return
        if ("parameter", parameter.name) in self._invalid_numeric_drafts:
            self._store_parameter_editor_draft(
                parameter,
                "Apply the corrected parameter values or revert this draft.",
            )

    def _store_parameter_editor_draft(
        self,
        parameter: FitParameterVariable,
        message: str,
    ) -> None:
        fields = {
            "minimum": self._minimum_edit,
            "maximum": self._maximum_edit,
        }
        if parameter.scope == "shared":
            fields = {"starting": self._starting_edit, **fields}
        self._store_invalid_numeric_draft(
            ("parameter", parameter.name),
            {field: edit.text() for field, edit in fields.items()},
            message,
        )
        for edit in fields.values():
            edit.setStyleSheet(
                "background: rgb(255, 200, 200); color: rgb(80, 0, 0);"
            )
            edit.setToolTip(str(message))
        self._revert_parameter_draft_button.setEnabled(True)

    def _revert_selected_parameter_draft(self) -> None:
        parameter = self._selected_parameter()
        if parameter is None:
            return
        if not self._retire_numeric_drafts({parameter.name}):
            return
        self._render_selected_parameter()
        self.statusMessage.emit(
            f"Invalid values for {parameter.display_name} reverted."
        )

    def _parameter_projection_for_change(self, parameter: FitParameterVariable) -> FitParameterVariable:
        if self._projected_setup is None:
            raise ValueError("Fit setup is unavailable.")
        scope = str(self._scope_combo.currentData() or parameter.scope)
        log10 = bool(self._scale_combo.currentData())
        minimum = self._parse_number(self._minimum_edit.text(), f"{parameter.name} minimum")
        maximum = self._parse_number(self._maximum_edit.text(), f"{parameter.name} maximum")
        config = parameter.shared_config if scope == "shared" else parameter.per_dataset_config
        if scope == "shared":
            value = self._parse_number(self._starting_edit.text(), f"{parameter.name} starting value")
            dataset_values = config.dataset_values
        else:
            value = config.value
            dataset_values_list: list[tuple[str, float]] = []
            for row in range(self._dataset_values_table.rowCount()):
                dataset_item = self._dataset_values_table.item(row, 0)
                value_item = self._dataset_values_table.item(row, 1)
                dataset_id = str(dataset_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if dataset_item else ""
                raw = value_item.text() if value_item is not None else ""
                if not dataset_id or not raw.strip():
                    raise ValueError(f"{parameter.name} requires a starting value for every dataset.")
                dataset_value = self._parse_number(raw, f"{parameter.name} value for dataset {dataset_id}")
                dataset_values_list.append((dataset_id, dataset_value))
            dataset_values = tuple(dataset_values_list)
        active = FitParameterConfiguration(
            value=value,
            minimum=minimum,
            maximum=maximum,
            fit=self._vary_check.isChecked(),
            log10=log10,
            dataset_values=dataset_values,
        )
        return FitParameterVariable(
            name=parameter.name,
            display_name=parameter.display_name,
            scope=scope,
            shared_config=active if scope == "shared" else parameter.shared_config,
            per_dataset_config=active if scope != "shared" else parameter.per_dataset_config,
            family=parameter.family,
            role=parameter.role,
            scientific_category=parameter.scientific_category,
            allow_exact_zero=parameter.allow_exact_zero,
            unit=parameter.unit,
            fit_local_provenance=parameter.fit_local_provenance,
        )

    def _uncertainty_effects_for_parameter(
        self,
        parameter: FitParameterVariable,
    ) -> dict[str, tuple[str, ...]]:
        """Return accepted typed uncertainty coordinates affected by Vary off."""

        setup = self._projected_setup
        if setup is None:
            return {}
        available = set(fit_uncertainty_coordinates(setup.parameters, setup.datasets))
        effects: dict[str, tuple[str, ...]] = {}
        for label, configuration in (
            ("Profile", setup.profile_likelihood),
            ("Bootstrap", setup.residual_bootstrap),
        ):
            if configuration is None:
                continue
            coordinates = tuple(
                coordinate
                for coordinate in configuration.parameter_names
                if isinstance(coordinate, FitCoordinate)
                and coordinate in available
                and coordinate.name == parameter.name
            )
            if coordinates:
                effects[label] = tuple(
                    self._coordinate_display_name(coordinate)
                    for coordinate in coordinates
                )
        return effects

    def _confirm_vary_analysis_update(
        self,
        parameter: FitParameterVariable,
    ) -> bool:
        effects = self._uncertainty_effects_for_parameter(parameter)
        if not effects:
            return True
        details = [
            f"Stopping Vary for {parameter.display_name} will update these analyses:"
        ]
        details.extend(
            f"{label}: {', '.join(coordinates)}"
            for label, coordinates in effects.items()
        )
        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Update uncertainty analyses?")
        message.setText("\n".join(details))
        stop_button = message.addButton(
            "Stop varying and update analyses",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = message.addButton(
            "Cancel",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is stop_button

    def _reproject_vary_truth(self, parameter_name: str) -> None:
        """Restore the visible Vary controls from the accepted setup only."""

        setup = self._projected_setup
        parameter = self._parameter_by_name(parameter_name)
        if setup is None or parameter is None:
            return
        blockers = (
            QtCore.QSignalBlocker(self._vary_check),
            QtCore.QSignalBlocker(self._param_table),
        )
        self._vary_check.setChecked(parameter.selected_config.fit)
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 0)
            if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == parameter_name:
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if parameter.selected_config.fit
                    else QtCore.Qt.CheckState.Unchecked
                )
                break
        del blockers

    def _emit_vary_edit(self, name: str, desired_fit: bool) -> bool:
        if self._projecting or self._running or self._projected_setup is None:
            return False
        parameter = self._parameter_by_name(str(name or ""))
        if parameter is None:
            return False
        desired = bool(desired_fit)
        if desired == parameter.selected_config.fit:
            self._reproject_vary_truth(parameter.name)
            return False
        if not desired and not self._confirm_vary_analysis_update(parameter):
            self._reproject_vary_truth(parameter.name)
            return False
        setup = self._projected_setup
        self._reproject_vary_truth(parameter.name)
        replacement = _replace_fit_parameter(parameter, fit=desired)
        profile = setup.profile_likelihood
        bootstrap = setup.residual_bootstrap
        if not desired:
            available = set(fit_uncertainty_coordinates(setup.parameters, setup.datasets))

            def without_parameter_coordinates(configuration: object):
                if configuration is None:
                    return None
                remaining = tuple(
                    coordinate
                    for coordinate in configuration.parameter_names
                    if not (
                        isinstance(coordinate, FitCoordinate)
                        and coordinate in available
                        and coordinate.name == parameter.name
                    )
                )
                return None if not remaining else replace(configuration, parameter_names=remaining)

            profile = without_parameter_coordinates(profile)
            bootstrap = without_parameter_coordinates(bootstrap)
        self.setupSnapshotRequested.emit(
            replace(
                setup,
                parameters=tuple(
                    replacement if item.name == replacement.name else item
                    for item in setup.parameters
                ),
                profile_likelihood=profile,
                residual_bootstrap=bootstrap,
            )
        )
        return True

    def _emit_selected_parameter_vary_edit(self, desired_fit: bool) -> bool:
        parameter = self._selected_parameter()
        if parameter is None:
            return False
        return self._emit_vary_edit(parameter.name, desired_fit)

    def _emit_selected_parameter_edit(self, *_args: object, name: str | None = None) -> bool:
        if self._projecting or self._running or self._projected_setup is None:
            return False
        parameter = self._parameter_by_name(name or (self._selected_parameter().name if self._selected_parameter() else ""))
        if parameter is None:
            return False
        try:
            edited = self._parameter_projection_for_change(parameter)
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit parameter edit is invalid: {exc}")
            self._store_parameter_editor_draft(parameter, str(exc))
            return False
        updated = replace(
            self._projected_setup,
            parameters=tuple(
                edited if item.name == edited.name else item
                for item in self._projected_setup.parameters
            ),
        )
        self._retire_numeric_drafts({parameter.name})
        self.setupSnapshotRequested.emit(updated)
        return True

    def _on_scope_changed(self, _index: int) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        desired = str(self._scope_combo.currentData() or "")
        if (
            parameter.fit_local_provenance is not None
            and desired == "centred_fixed_effects"
        ):
            self._restore_scope(parameter.scope)
            self.statusMessage.emit(
                "Fit-local observable scalars cannot use centred dataset deviations."
            )
            return
        if desired == parameter.scope:
            return
        if desired not in {"shared", "per_dataset", "centred_fixed_effects"}:
            self._restore_scope(parameter.scope)
            return
        replacement = replace(parameter, scope=desired)
        if desired == "centred_fixed_effects" and parameter.scope == "shared":
            config = parameter.shared_config
            if config is None:
                self._restore_scope(parameter.scope)
                return
            dataset_ids = tuple(item.dataset_id for item in self._projected_setup.datasets)
            replacement = replace(
                parameter,
                scope=desired,
                per_dataset_config=replace(
                    config,
                    dataset_values=tuple((dataset_id, config.value) for dataset_id in dataset_ids),
                ),
            )
        self.setupSnapshotRequested.emit(
            replace(
                self._projected_setup,
                parameters=tuple(
                    replacement if item.name == replacement.name else item
                    for item in self._projected_setup.parameters
                ),
            )
        )

    def _restore_scope(self, scope: str) -> None:
        blocker = QtCore.QSignalBlocker(self._scope_combo)
        self._scope_combo.setCurrentIndex(self._scope_combo.findData(scope))
        del blocker

    @property
    def has_invalid_edits(self) -> bool:
        return bool(self._invalid_numeric_drafts)

    def _store_invalid_numeric_draft(
        self,
        key: tuple[str, ...],
        raw_fields: Mapping[str, str],
        message: str,
    ) -> None:
        draft = _InvalidNumericDraft(
            tuple((str(field), str(raw)) for field, raw in raw_fields.items()),
            str(message),
        )
        if self._invalid_numeric_drafts.get(key) != draft:
            self._invalid_numeric_drafts[key] = draft
            self.validityChanged.emit()

    def _retire_numeric_drafts(
        self,
        parameter_names: set[str] | None = None,
    ) -> bool:
        keys = tuple(
            key
            for key in self._invalid_numeric_drafts
            if parameter_names is None
            or (len(key) > 1 and key[1] in parameter_names)
        )
        for key in keys:
            self._invalid_numeric_drafts.pop(key)
        if keys:
            self.validityChanged.emit()
        return bool(keys)

    @staticmethod
    def _parse_number(raw_text: str, field: str) -> float:
        try:
            value = float(raw_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite.")
        return value

    def _project_defaults(self, defaults: FitDefaults) -> None:
        blocker = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._method_combo,
                self._max_eval_spin,
                self._seed_check,
                self._seed_spin,
            )
        ]
        self._method_combo.setCurrentIndex(self._method_combo.findData(defaults.method))
        self._max_eval_spin.setValue(defaults.max_nfev)
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        del blocker
        self._settings_summary.setText(
            f"{self._solver_combo.currentText()} · "
            f"{FIT_METHOD_LABELS.get(defaults.method, defaults.method)} · "
            f"max {defaults.max_nfev:,} · "
            f"{'seed ' + str(defaults.seed) if defaults.use_seed else 'no seed'}"
        )

    def _defaults_projection(
        self,
        *,
        ftol: float | None = None,
        xtol: float | None = None,
    ) -> FitDefaults:
        if self._projected_setup is None:
            raise ValueError("Fit defaults are not available.")
        defaults = self._projected_setup.defaults
        return validate_fit_defaults(
            replace(
                defaults,
                method=str(self._method_combo.currentData() or "trf"),
                max_nfev=self._max_eval_spin.value(),
                ftol=defaults.ftol if ftol is None else ftol,
                xtol=defaults.xtol if xtol is None else xtol,
                use_seed=self._seed_check.isChecked(),
                seed=self._seed_spin.value(),
            )
        )

    def _emit_defaults_edit(self, *_args: object) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        try:
            snapshot = replace(
                self._projected_setup,
                defaults=self._defaults_projection(),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit defaults are invalid: {exc}")
            return
        self.setupSnapshotRequested.emit(snapshot)

    def _emit_solver_edit(self, *_args: object) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        try:
            snapshot = replace(
                self._projected_setup,
                solver=str(self._solver_combo.currentData() or "BDF"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit solver edit is invalid: {exc}")
            return
        self.setupSnapshotRequested.emit(snapshot)

    def _project_advanced_analysis(self, snapshot: FitSetupSnapshot) -> None:
        profile = snapshot.profile_likelihood
        bootstrap = snapshot.residual_bootstrap
        blockers = [
            QtCore.QSignalBlocker(control)
            for control in (self._profile_enabled, self._bootstrap_enabled)
        ]
        try:
            self._profile_enabled.setChecked(profile is not None)
            self._bootstrap_enabled.setChecked(bootstrap is not None)
            self._profile_summary.setText(self._analysis_summary(profile, "Profile"))
            self._bootstrap_summary.setText(
                self._analysis_summary(bootstrap, "Bootstrap")
            )
        finally:
            del blockers

    @staticmethod
    def _analysis_summary(configuration: object | None, label: str) -> str:
        if configuration is None:
            return "Not configured"
        names = tuple(getattr(configuration, "parameter_names", ()))
        method = str(getattr(configuration, "optimizer_method", "trf"))
        if label == "Profile":
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''} · "
                f"{getattr(configuration, 'confidence_level', 0.95):.0%} · "
                f"{getattr(configuration, 'max_bracketing_steps_per_side', 0)} "
                "maximum outward search steps/side"
            )
        else:
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''} · "
                f"{getattr(configuration, 'replicates', 0)} refits · "
                f"{getattr(configuration, 'minimum_success_fraction', 0.0):.0%} minimum"
            )
        return f"Enabled · {detail} · {FIT_METHOD_LABELS.get(method, method)}"

    def _analysis_snapshot(
        self,
        *,
        profile: ProfileLikelihoodConfig | None,
        bootstrap: ResidualBootstrapConfig | None,
    ) -> FitSetupSnapshot:
        setup = self._projected_setup
        if setup is None:
            raise ValueError("Fit setup is unavailable.")
        return replace(
            setup,
            profile_likelihood=profile,
            residual_bootstrap=bootstrap,
        )

    def _open_convergence_dialog(self) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        dialog = _ConvergenceDialog(
            self._projected_setup.defaults,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if values is None:
            return
        try:
            setup = self._projected_setup
            if setup is None:
                return
            snapshot = replace(
                setup,
                defaults=self._defaults_projection(ftol=values[0], xtol=values[1]),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Advanced Fit settings are invalid: {exc}")
            return
        self.setupSnapshotRequested.emit(snapshot)

    def _available_analysis_coordinates(self) -> tuple[FitCoordinate, ...]:
        setup = self._projected_setup
        if setup is None:
            return ()
        return fit_uncertainty_coordinates(setup.parameters, setup.datasets)

    def _analysis_coordinate_display_names(self) -> dict[FitCoordinate, str]:
        return {
            raw: self._coordinate_display_name(raw)
            for raw in self._available_analysis_coordinates()
        }

    def _open_profile_dialog(self) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        dialog = _ProfileLikelihoodDialog(
            available=self._available_analysis_coordinates(),
            display_names=self._analysis_coordinate_display_names(),
            configuration=setup.profile_likelihood,
            default_selection=None,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            if setup.profile_likelihood is None:
                self._profile_enabled.setChecked(False)
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=dialog.configuration(),
                    bootstrap=setup.residual_bootstrap,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Profile likelihood settings are invalid: {exc}")

    def _open_bootstrap_dialog(self) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        dialog = _ResidualBootstrapDialog(
            available=self._available_bootstrap_coordinates(),
            display_names=self._analysis_coordinate_display_names(),
            configuration=setup.residual_bootstrap,
            default_selection=None,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            if setup.residual_bootstrap is None:
                self._bootstrap_enabled.setChecked(False)
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=setup.profile_likelihood,
                    bootstrap=dialog.configuration(),
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Residual bootstrap settings are invalid: {exc}")

    def _on_profile_toggled(self, enabled: bool) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        # The checkbox is a projection of the accepted owner snapshot.  Keep
        # it there while a dialog is pending; the accepted snapshot reprojects
        # the new value atomically.
        blocker = QtCore.QSignalBlocker(self._profile_enabled)
        self._profile_enabled.setChecked(setup.profile_likelihood is not None)
        del blocker
        if enabled:
            self._open_profile_dialog()
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=None,
                    bootstrap=setup.residual_bootstrap,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Profile likelihood settings are invalid: {exc}")

    def _on_bootstrap_toggled(self, enabled: bool) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        blocker = QtCore.QSignalBlocker(self._bootstrap_enabled)
        self._bootstrap_enabled.setChecked(setup.residual_bootstrap is not None)
        del blocker
        if enabled:
            self._open_bootstrap_dialog()
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=setup.profile_likelihood,
                    bootstrap=None,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Residual bootstrap settings are invalid: {exc}")

    def _relayout_analysis_cards(self) -> None:
        if not hasattr(self, "_analysis_cards"):
            return
        narrow = self.width() < 900
        columns = 1 if narrow else 3
        for index, card in enumerate(self._analysis_cards):
            row, column = divmod(index, columns)
            self._analysis_cards_layout.addWidget(card, row, column)
        for column in range(3):
            self._analysis_cards_layout.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout_analysis_cards()

    def set_running_state(self, running: bool) -> None:
        self._running = bool(running)
        self._add_observable_button.setEnabled(not self._running)
        self._add_button.setEnabled(not self._running)

    def clear_pending_setup_edits(self) -> None:
        self._retire_numeric_drafts()
        if self._projected_setup is not None:
            self.project_setup_snapshot(self._projected_setup)
