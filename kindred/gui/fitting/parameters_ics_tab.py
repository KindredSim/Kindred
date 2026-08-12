"""Fit parameter presentation with current owner-facing hooks.

The widget construction in this module is intentionally close to the executable
Fit presentation.  The widget is only a view: accepted setup, request
construction, Fit results, defaults persistence, and worker ownership remain
outside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.analysis.fit_uncertainty import (
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
)
from kindred.core.analysis.hierarchical_fit import HierarchicalFitConfig

from kindred.core.project_document import (
    FIT_MAX_NFEV_RANGE,
    FIT_METHOD_LABELS,
    FIT_SEED_RANGE,
    FitDefaults,
    SUPPORTED_FIT_METHODS,
    SUPPORTED_SOLVERS,
    default_fit_defaults,
    validate_fit_defaults,
)
from kindred.gui.fitting.constants import DIFFERENTIAL_EVOLUTION_METHOD
from kindred.gui.fitting.launch import (
    FitAdvancedAnalysisSetupEdit,
    FitDefaultsSetupEdit,
    FitInitialScopeSetupEdit,
    FitInitialGlobalVariable,
    FitMappedDatasetSnapshot,
    FitOriginSnapshot,
    FitObservationSnapshot,
    FitParameterScopeSetupEdit,
    FitParameterConfiguration,
    FitParameterSetupEdit,
    FitParameterVariable,
    FitSetupSnapshot,
    FitSolverSetupEdit,
    fit_uncertainty_coordinate_names,
)
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = ["ParametersIcsTab"]


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


class _FitLocalObservableDialog(QtWidgets.QDialog):
    """Algebraic-observable page used for one Fit-local intent."""

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

        self._dataset_ids = [str(x) for x in (dataset_ids or []) if str(x)]
        self._available_observations = {
            mapped.dataset_id: mapped
            for mapped in self._normalize_available_observations(
                available_observations
            )
        }
        self._selection: dict[str, str] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self._observable_tab = QtWidgets.QWidget(self)
        layout.addWidget(self._observable_tab, stretch=1)
        self._init_observable_tab()
        self._connect_observable_signals()
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
        self._observable_dataset_combo = QtWidgets.QComboBox(self._observable_tab)
        self._observable_column_combo = QtWidgets.QComboBox(self._observable_tab)
        for dataset_id, entry in self._available_observations.items():
            self._observable_dataset_combo.addItem(
                entry.display_name,
                dataset_id,
            )
        target_form.addRow("Dataset:", self._observable_dataset_combo)
        target_form.addRow(
            "Observed column:",
            self._observable_column_combo,
        )
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

    def _connect_observable_signals(self) -> None:
        self._observable_dataset_combo.currentIndexChanged.connect(
            self._refresh_observable_columns
        )
        self._refresh_observable_columns()

    def _add_button_box(self, layout: QtWidgets.QVBoxLayout) -> None:
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selection(self) -> dict[str, str] | None:
        return dict(self._selection or {}) if self._selection else None

    def _refresh_observable_columns(self) -> None:
        self._observable_column_combo.clear()
        dataset_id = str(self._observable_dataset_combo.currentData() or "")
        entry = self._available_observations.get(dataset_id)
        if entry is None:
            return
        for observation in entry.observations:
            self._observable_column_combo.addItem(
                observation.raw_column,
                observation.raw_observation_id,
            )

    def _observable_target_selection(self) -> dict[str, str] | None:
        dataset_id = str(self._observable_dataset_combo.currentData() or "").strip()
        raw_observation_id = str(
            self._observable_column_combo.currentData() or ""
        ).strip()
        raw_column = str(self._observable_column_combo.currentText() or "").strip()
        if not dataset_id or not raw_observation_id or not raw_column:
            return None
        return {
            "dataset_id": dataset_id,
            "raw_observation_id": raw_observation_id,
            "raw_column": raw_column,
        }

    def accept(self) -> None:  # noqa: D401 - Qt override
        mode = "per_dataset"
        target = self._observable_target_selection()
        if target is None:
            return
        name = str(self._new_observable_name_edit.text()).strip()
        expression = str(self._new_observable_expr_edit.text()).strip()
        if not name or not expression:
            return
        self._selection = {
            "type": "observable_new",
            "name": name,
            "expr": expression,
            "scalar_scope": mode,
            **target,
        }
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


class _HierarchicalConfigDialog(QtWidgets.QDialog):
    """Author one complete F-owned Hierarchical configuration."""

    def __init__(
        self,
        parameter: FitParameterVariable,
        configuration: HierarchicalFitConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Hierarchical Fit — {parameter.display_name}")
        self._parameter = parameter
        if configuration is not None and configuration.parameter_name != parameter.name:
            raise ValueError("Hierarchical configuration changed parameter identity.")
        parameter_config = parameter.per_dataset_config
        self._log10 = (
            parameter_config.log10 if configuration is None else configuration.log10
        )
        form = QtWidgets.QFormLayout(self)
        self._population = QtWidgets.QLineEdit(
            _format_scientific(
                parameter_config.value
                if configuration is None
                else configuration.initial_population
            ),
            self,
        )
        self._minimum = QtWidgets.QLineEdit(
            _format_scientific(
                parameter_config.minimum
                if configuration is None
                else configuration.minimum
            ),
            self,
        )
        self._maximum = QtWidgets.QLineEdit(
            _format_scientific(
                parameter_config.maximum
                if configuration is None
                else configuration.maximum
            ),
            self,
        )
        self._dispersion = QtWidgets.QLineEdit(
            _format_scientific(
                0.1
                if configuration is None
                else configuration.initial_dispersion
            ),
            self,
        )
        self._dispersion_minimum = QtWidgets.QLineEdit(
            _format_scientific(
                1e-8
                if configuration is None
                else configuration.minimum_dispersion
            ),
            self,
        )
        self._dispersion_maximum = QtWidgets.QLineEdit(
            _format_scientific(
                10.0
                if configuration is None
                else configuration.maximum_dispersion
            ),
            self,
        )
        self._max_nfev = QtWidgets.QSpinBox(self)
        self._max_nfev.setRange(1, 2_000_000_000)
        self._max_nfev.setValue(
            500 if configuration is None else configuration.max_nfev
        )
        self._inner_max_nfev = QtWidgets.QSpinBox(self)
        self._inner_max_nfev.setRange(1, 2_000_000_000)
        self._inner_max_nfev.setValue(
            200 if configuration is None else configuration.inner_max_nfev
        )
        self._method = QtWidgets.QComboBox(self)
        for method in SUPPORTED_FIT_METHODS:
            self._method.addItem(FIT_METHOD_LABELS.get(method, method), method)
        method_id = (
            "trf" if configuration is None else configuration.optimizer_method
        )
        self._method.setCurrentIndex(max(0, self._method.findData(method_id)))
        for edit in (
            self._population,
            self._minimum,
            self._maximum,
            self._dispersion,
            self._dispersion_minimum,
            self._dispersion_maximum,
        ):
            setup_scientific_validator(edit)
        for label, widget in (
            ("Population starting value:", self._population),
            ("Population allowed minimum:", self._minimum),
            ("Population allowed maximum:", self._maximum),
            ("Expected between-dataset variation:", self._dispersion),
            ("Variation allowed minimum:", self._dispersion_minimum),
            ("Variation allowed maximum:", self._dispersion_maximum),
        ):
            form.addRow(label, widget)
        advanced = QtWidgets.QGroupBox("Advanced", self)
        advanced_form = QtWidgets.QFormLayout(advanced)
        advanced_form.addRow("Optimizer method:", self._method)
        advanced_form.addRow("Outer evaluations:", self._max_nfev)
        advanced_form.addRow("Inner evaluations:", self._inner_max_nfev)
        form.addRow(advanced)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def configuration(self) -> HierarchicalFitConfig:
        parameter = self._parameter
        return HierarchicalFitConfig(
            parameter_name=parameter.name,
            initial_population=float(self._population.text()),
            minimum=float(self._minimum.text()),
            maximum=float(self._maximum.text()),
            log10=self._log10,
            initial_dispersion=float(self._dispersion.text()),
            minimum_dispersion=float(self._dispersion_minimum.text()),
            maximum_dispersion=float(self._dispersion_maximum.text()),
            optimizer_method=str(self._method.currentData() or "trf"),
            max_nfev=self._max_nfev.value(),
            inner_max_nfev=self._inner_max_nfev.value(),
        )


class ParametersIcsTab(QtWidgets.QWidget):
    """Presentation-only parameter, IC-scope, and optimizer panel."""

    setupEditsRequested = QtCore.Signal(object)
    addObservableRequested = QtCore.Signal(object)
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
        self._invalid_numeric_text: dict[tuple[str, ...], str] = {}
        self._invalid_numeric_messages: dict[tuple[str, ...], str] = {}
        self._projecting = False
        self._defaults_backing: FitDefaults | None = None
        self._default_changed_fields: set[str] = set()
        self._running = False
        self._set_parameter_choices(parameter_choices or {})
        self._build_ui()


    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_parameters_panel(), stretch=1)

    def set_origin(self, origin: FitOriginSnapshot) -> None:
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if self._origin is not None and self._origin.capture_id != origin.capture_id:
            self._clear_invalid_numeric_state()
        self._origin = origin


    def set_parameter_choices(self, **choices: object) -> None:
        """Set presentation choices used by the observable page."""

        self._set_parameter_choices(choices)

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
    # Parameter presentation and intents
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
        self.addObservableRequested.emit(
            {
                "name": selection["name"],
                "expr": selection["expr"],
                "scalar_scope": selection["scalar_scope"],
                "dataset_id": selection["dataset_id"],
                "raw_observation_id": selection["raw_observation_id"],
                "raw_column": selection["raw_column"],
            }
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
        self.setupEditsRequested.emit(
            (
                FitInitialScopeSetupEdit(
                    origin_tab_id=setup.origin_tab_id,
                    origin_capture_id=setup.origin_capture_id,
                    species=species_name,
                    scope=scope,
                    global_initial=configured,
                ),
            )
        )

    def _reset_to_initial(self) -> None:
        if not self._can_reset_initial:
            self.statusMessage.emit("Initial Fit setup is not available.")
            return
        self._clear_invalid_numeric_state()
        self.resetRequested.emit("initial")

    def _reset_to_last_fit(self) -> None:
        if not self._can_reset_last_fit:
            self.statusMessage.emit("No completed Fit is available to restore.")
            return
        self._clear_invalid_numeric_state()
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
        "hierarchical": "Related values across datasets (hierarchical / partial pooling)",
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

        self._related_group = QtWidgets.QGroupBox("Related-values configuration", self._selected_editor)
        related_layout = QtWidgets.QFormLayout(self._related_group)
        self._related_population_label = QtWidgets.QLabel("—", self._related_group)
        self._related_range_label = QtWidgets.QLabel("—", self._related_group)
        self._related_variation_label = QtWidgets.QLabel("—", self._related_group)
        self._related_variation_range_label = QtWidgets.QLabel("—", self._related_group)
        self._related_scale_label = QtWidgets.QLabel("—", self._related_group)
        related_layout.addRow("Population starting value:", self._related_population_label)
        related_layout.addRow("Population allowed range:", self._related_range_label)
        related_layout.addRow("Expected between-dataset variation:", self._related_variation_label)
        related_layout.addRow("Variation allowed range:", self._related_variation_range_label)
        related_layout.addRow("Inherited scale:", self._related_scale_label)
        self._configure_related_button = QtWidgets.QPushButton(
            "Configure related-values model", self._related_group
        )
        self._configure_related_button.clicked.connect(self._configure_related_values)
        related_layout.addRow(self._configure_related_button)
        editor_layout.addWidget(self._related_group)
        right_layout.addWidget(self._selected_editor)

        last_group = QtWidgets.QGroupBox("Last Fit", right)
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

        settings_group = QtWidgets.QGroupBox("Fit settings", container)
        settings_group.setObjectName("fit_settings_group")
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
        convergence = QtWidgets.QGroupBox(
            "Advanced convergence settings", settings_group
        )
        convergence.setCheckable(True)
        convergence.setChecked(False)
        convergence_form = QtWidgets.QFormLayout(convergence)
        self._ftol_label = QtWidgets.QLabel("Objective-improvement tolerance:", convergence)
        self._ftol_edit = QtWidgets.QLineEdit(_format_scientific(defaults.ftol), convergence)
        self._xtol_label = QtWidgets.QLabel("Parameter-step tolerance:", convergence)
        self._xtol_edit = QtWidgets.QLineEdit(_format_scientific(defaults.xtol), convergence)
        setup_scientific_validator(self._ftol_edit)
        setup_scientific_validator(self._xtol_edit)
        convergence_form.addRow(self._ftol_label, self._ftol_edit)
        convergence_form.addRow(self._xtol_label, self._xtol_edit)
        settings_form.addRow(convergence)
        layout.addWidget(settings_group)

        uncertainty_group = QtWidgets.QGroupBox(
            "Estimate uncertainty — optional", container
        )
        uncertainty_group.setObjectName("fit_uncertainty_group")
        uncertainty_layout = QtWidgets.QVBoxLayout(uncertainty_group)
        self._profile_enabled = QtWidgets.QCheckBox("Profile likelihood", uncertainty_group)
        uncertainty_layout.addWidget(self._profile_enabled)
        profile_form = QtWidgets.QFormLayout()
        self._profile_coordinates = QtWidgets.QListWidget(uncertainty_group)
        self._profile_coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._profile_confidence = QtWidgets.QLineEdit("0.95", uncertainty_group)
        self._profile_points = QtWidgets.QSpinBox(uncertainty_group)
        self._profile_points.setRange(2, 10000)
        self._profile_points.setValue(8)
        self._profile_budget = QtWidgets.QSpinBox(uncertainty_group)
        self._profile_budget.setRange(1, 2_000_000_000)
        self._profile_budget.setValue(300)
        self._profile_method = QtWidgets.QComboBox(uncertainty_group)
        self._add_method_items(self._profile_method)
        setup_scientific_validator(self._profile_confidence)
        profile_form.addRow("Selected parameters:", self._profile_coordinates)
        profile_form.addRow("Confidence level:", self._profile_confidence)
        profile_form.addRow("Search points on each side:", self._profile_points)
        profile_form.addRow("Maximum evaluations per refit:", self._profile_budget)
        profile_advanced = QtWidgets.QGroupBox("Advanced", uncertainty_group)
        profile_advanced_form = QtWidgets.QFormLayout(profile_advanced)
        profile_advanced_form.addRow("Optimization method:", self._profile_method)
        profile_form.addRow(profile_advanced)
        uncertainty_layout.addLayout(profile_form)

        self._bootstrap_enabled = QtWidgets.QCheckBox(
            "Residual bootstrap", uncertainty_group
        )
        uncertainty_layout.addWidget(self._bootstrap_enabled)
        bootstrap_form = QtWidgets.QFormLayout()
        self._bootstrap_coordinates = QtWidgets.QListWidget(uncertainty_group)
        self._bootstrap_coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._bootstrap_seed = QtWidgets.QSpinBox(uncertainty_group)
        self._bootstrap_seed.setRange(-2_000_000_000, 2_000_000_000)
        self._bootstrap_replicates = QtWidgets.QSpinBox(uncertainty_group)
        self._bootstrap_replicates.setRange(1, 1_000_000)
        self._bootstrap_replicates.setValue(200)
        self._bootstrap_minimum = QtWidgets.QSpinBox(uncertainty_group)
        self._bootstrap_minimum.setRange(1, 1_000_000)
        self._bootstrap_minimum.setValue(50)
        self._bootstrap_confidence = QtWidgets.QLineEdit("0.95", uncertainty_group)
        self._bootstrap_fraction = QtWidgets.QLineEdit("0.8", uncertainty_group)
        setup_scientific_validator(self._bootstrap_confidence)
        setup_scientific_validator(self._bootstrap_fraction)
        bootstrap_form.addRow("Selected parameters:", self._bootstrap_coordinates)
        bootstrap_form.addRow("Number of refits:", self._bootstrap_replicates)
        bootstrap_form.addRow("Minimum successful refits:", self._bootstrap_minimum)
        bootstrap_form.addRow("Required success rate:", self._bootstrap_fraction)
        bootstrap_form.addRow("Confidence level:", self._bootstrap_confidence)
        bootstrap_form.addRow("Random seed:", self._bootstrap_seed)
        uncertainty_layout.addLayout(bootstrap_form)
        self._advanced_apply = QtWidgets.QPushButton(
            "Apply these settings", uncertainty_group
        )
        self._advanced_apply.clicked.connect(self._emit_advanced_analysis_edit)
        uncertainty_layout.addWidget(self._advanced_apply)
        layout.addWidget(uncertainty_group)

        # Signals are attached after construction so projection can use one
        # coherent blocker set and cannot author half a typed intent.
        self._solver_combo.currentIndexChanged.connect(self._emit_solver_edit)
        self._method_combo.currentIndexChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="method")
        )
        self._method_combo.currentIndexChanged.connect(self._sync_method_controls)
        self._max_eval_spin.valueChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="max_nfev")
        )
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        self._seed_check.toggled.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="use_seed")
        )
        self._seed_spin.valueChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="seed")
        )
        self._ftol_edit.editingFinished.connect(
            lambda: self._emit_defaults_edit(changed_field="ftol")
        )
        self._xtol_edit.editingFinished.connect(
            lambda: self._emit_defaults_edit(changed_field="xtol")
        )
        self._ftol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("ftol", text)
        )
        self._xtol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("xtol", text)
        )
        self._vary_check.toggled.connect(self._emit_selected_parameter_edit)
        self._scale_combo.currentIndexChanged.connect(self._emit_selected_parameter_edit)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            edit.editingFinished.connect(self._emit_selected_parameter_edit)
        self._sync_method_controls()
        self._dataset_values_group.setVisible(False)
        self._related_group.setVisible(False)
        self._selected_editor.setEnabled(False)
        return widget

    @staticmethod
    def _add_method_items(combo: QtWidgets.QComboBox) -> None:
        for method in SUPPORTED_FIT_METHODS:
            combo.addItem(FIT_METHOD_LABELS.get(method, method), method)

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if self._origin is not None and (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
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

    def _coordinate_display_name(self, coordinate: str) -> str:
        raw = str(coordinate)
        if "::" not in raw:
            return self._display_parameter_name(raw)
        dataset_id, name = raw.split("::", 1)
        return f"{self._display_parameter_name(name)} — {self._dataset_display_name(dataset_id)}"

    def _selected_coordinate_names(self, widget: QtWidgets.QListWidget) -> tuple[str, ...]:
        return tuple(
            str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text())
            for item in widget.selectedItems()
        )

    def _project_coordinate_list(
        self,
        widget: QtWidgets.QListWidget,
        available: Sequence[str],
        selected: Sequence[str],
    ) -> None:
        blocker = QtCore.QSignalBlocker(widget)
        widget.clear()
        selected_names = set(str(item) for item in selected)
        for coordinate in available:
            raw = str(coordinate)
            item = QtWidgets.QListWidgetItem(self._coordinate_display_name(raw), widget)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, raw)
            item.setToolTip(raw)
            item.setSelected(raw in selected_names)
        del blocker
        widget.setVisible(bool(available))

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self._projected_setup = snapshot
        self._defaults_backing = snapshot.defaults
        self._default_changed_fields.clear()
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

    def project_setup_delta(
        self,
        snapshot: FitSetupSnapshot,
        intents: Sequence[object],
    ) -> None:
        self._check_snapshot(snapshot)
        if not isinstance(intents, tuple):
            raise TypeError("Fit setup delta must carry a tuple of intents.")
        # Owner acknowledgement is the accepted truth.  Reprojecting that
        # truth is atomic for this view and never changes retained Last Fit.
        self.project_setup_snapshot(snapshot)

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
        return "Related values" if last.scope == "hierarchical" else "Per dataset"

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

    def _parameter_row(self, name: str) -> int:
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 1)
            if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == name:
                return row
        return -1

    def _on_roster_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 0:
            return
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if name:
            self._emit_selected_parameter_edit(name=name)

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
        self._related_group.setVisible(False)
        self._last_result_label.setText("—")

    def _render_selected_parameter(self) -> None:  # noqa: C901
        parameter = self._selected_parameter()
        if parameter is None:
            self._clear_selected_editor()
            return
        self._selected_editor.setEnabled(True)
        config = parameter.selected_config
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
            self._render_dataset_values(parameter)
            self._render_related_summary(parameter)
            last = self._last_fit_parameters.get(parameter.name)
            if last is None:
                self._last_result_label.setText("—")
            elif last.scope == "shared":
                self._last_result_label.setText(f"{_format_number(last.selected_config.value)} (Shared)")
            else:
                self._last_result_label.setText(
                    "Related values" if last.scope == "hierarchical" else "Per dataset"
                )
        finally:
            del blockers
        self._clear_parameter_editor_invalid(parameter.name)

    def _render_dataset_values(self, parameter: FitParameterVariable) -> None:
        visible = parameter.scope in {"per_dataset", "hierarchical"}
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
            self._dataset_values_table.setItem(row, 1, value_item)
            last_item = QtWidgets.QTableWidgetItem(
                _format_number(last_values[dataset_id]) if dataset_id in last_values else "—"
            )
            last_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            last_item.setToolTip(f"Dataset ID: {dataset_id}")
            self._dataset_values_table.setItem(row, 2, last_item)
        del blocker

    def _render_related_summary(self, parameter: FitParameterVariable) -> None:
        visible = parameter.scope == "hierarchical"
        self._related_group.setVisible(visible)
        if not visible:
            return
        configuration = next(
            (item for item in (self._projected_setup.hierarchical if self._projected_setup else ())
             if item.parameter_name == parameter.name),
            None,
        )
        if configuration is None:
            self._related_population_label.setText(_format_number(parameter.per_dataset_config.value))
            self._related_range_label.setText(
                f"{_format_number(parameter.per_dataset_config.minimum)} to {_format_number(parameter.per_dataset_config.maximum)}"
            )
            self._related_variation_label.setText("—")
            self._related_variation_range_label.setText("—")
            self._related_scale_label.setText("Logarithmic" if parameter.per_dataset_config.log10 else "Linear")
            return
        self._related_population_label.setText(_format_number(configuration.initial_population))
        self._related_range_label.setText(
            f"{_format_number(configuration.minimum)} to {_format_number(configuration.maximum)}"
        )
        self._related_variation_label.setText(_format_number(configuration.initial_dispersion))
        self._related_variation_range_label.setText(
            f"{_format_number(configuration.minimum_dispersion)} to {_format_number(configuration.maximum_dispersion)}"
        )
        self._related_scale_label.setText("Logarithmic" if configuration.log10 else "Linear")

    def _on_dataset_value_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 1:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        dataset_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        key = ("parameter_dataset", parameter.name, dataset_id, "initial")
        try:
            value = self._parse_number(item.text(), f"{parameter.name} value for dataset {dataset_id}")
            if not math.isfinite(value):
                raise ValueError("value must be finite.")
            self._clear_invalid_numeric(key, edit=None)
            self._emit_selected_parameter_edit()
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_invalid_numeric(key, item.text(), str(exc))
            self.statusMessage.emit(str(exc))

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
            unit=parameter.unit,
            fit_local_provenance=parameter.fit_local_provenance,
        )

    def _emit_selected_parameter_edit(self, *_args: object, name: str | None = None) -> bool:
        if self._projecting or self._running or self._projected_setup is None:
            return False
        parameter = self._parameter_by_name(name or (self._selected_parameter().name if self._selected_parameter() else ""))
        if parameter is None:
            return False
        try:
            edited = self._parameter_projection_for_change(parameter)
            intent = FitParameterSetupEdit(
                origin_tab_id=self._projected_setup.origin_tab_id,
                origin_capture_id=self._projected_setup.origin_capture_id,
                parameter=edited,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit parameter edit is invalid: {exc}")
            self.validityChanged.emit()
            return False
        self.setupEditsRequested.emit((intent,))
        return True

    def _on_scope_changed(self, _index: int) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        desired = str(self._scope_combo.currentData() or "")
        if desired == parameter.scope:
            if desired == "hierarchical":
                self._configure_related_values()
            return
        config = None
        if desired == "hierarchical":
            dialog = _HierarchicalConfigDialog(parameter, parent=self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                self._restore_scope(parameter.scope)
                return
            try:
                config = dialog.configuration()
            except (TypeError, ValueError, OverflowError) as exc:
                self.statusMessage.emit(f"Related-values setup is invalid: {exc}")
                self._restore_scope(parameter.scope)
                return
        self.setupEditsRequested.emit(
            (
                FitParameterScopeSetupEdit(
                    origin_tab_id=self._projected_setup.origin_tab_id,
                    origin_capture_id=self._projected_setup.origin_capture_id,
                    parameter_name=parameter.name,
                    scope=desired,
                    hierarchical_config=config,
                ),
            )
        )

    def _restore_scope(self, scope: str) -> None:
        blocker = QtCore.QSignalBlocker(self._scope_combo)
        self._scope_combo.setCurrentIndex(self._scope_combo.findData(scope))
        del blocker

    def _configure_related_values(self) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        existing = next(
            (item for item in self._projected_setup.hierarchical if item.parameter_name == parameter.name),
            None,
        )
        dialog = _HierarchicalConfigDialog(parameter, configuration=existing, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            configuration = dialog.configuration()
            intent = FitParameterScopeSetupEdit(
                origin_tab_id=self._projected_setup.origin_tab_id,
                origin_capture_id=self._projected_setup.origin_capture_id,
                parameter_name=parameter.name,
                scope="hierarchical",
                hierarchical_config=configuration,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Related-values setup is invalid: {exc}")
            return
        self.setupEditsRequested.emit((intent,))

    @property
    def has_invalid_edits(self) -> bool:
        return bool(self._invalid_numeric_text)

    def _set_invalid_numeric(
        self,
        key: tuple[str, ...],
        raw_text: str,
        message: str,
        *,
        item: QtWidgets.QTableWidgetItem | None = None,
        edit: QtWidgets.QLineEdit | None = None,
    ) -> None:
        self._invalid_numeric_text[key] = str(raw_text)
        self._invalid_numeric_messages[key] = str(message)
        target = edit
        if target is not None:
            target.setStyleSheet("background: rgb(255, 200, 200); color: rgb(80, 0, 0);")
            target.setToolTip(str(message))
        if item is not None:
            item.setBackground(QtGui.QColor(255, 200, 200))
            item.setForeground(QtGui.QColor(80, 0, 0))
            item.setToolTip(str(message))
        self.validityChanged.emit()

    def _clear_invalid_numeric(
        self,
        key: tuple[str, ...],
        *,
        item: QtWidgets.QTableWidgetItem | None = None,
        edit: QtWidgets.QLineEdit | None = None,
    ) -> None:
        changed = key in self._invalid_numeric_text
        self._invalid_numeric_text.pop(key, None)
        self._invalid_numeric_messages.pop(key, None)
        if edit is not None:
            edit.setStyleSheet("")
            edit.setToolTip("")
        if item is not None:
            item.setBackground(QtGui.QBrush())
            item.setForeground(QtGui.QBrush())
        if changed:
            self.validityChanged.emit()

    def _clear_parameter_editor_invalid(self, name: str) -> None:
        for key in tuple(self._invalid_numeric_text):
            if len(key) > 1 and key[0] in {"parameter", "parameter_dataset"} and key[1] == name:
                self._invalid_numeric_text.pop(key, None)
                self._invalid_numeric_messages.pop(key, None)

    def _clear_invalid_numeric_state(self) -> None:
        if self._invalid_numeric_text:
            self._invalid_numeric_text.clear()
            self._invalid_numeric_messages.clear()
            self.validityChanged.emit()

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
        self._defaults_backing = defaults
        self._default_changed_fields.clear()
        blocker = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._method_combo,
                self._max_eval_spin,
                self._ftol_edit,
                self._xtol_edit,
                self._seed_check,
                self._seed_spin,
            )
        ]
        self._method_combo.setCurrentIndex(self._method_combo.findData(defaults.method))
        self._max_eval_spin.setValue(defaults.max_nfev)
        self._ftol_edit.setText(_format_scientific(defaults.ftol))
        self._xtol_edit.setText(_format_scientific(defaults.xtol))
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        del blocker
        self._sync_method_controls()

    def _sync_method_controls(self, *_args: object) -> None:
        is_de = str(self._method_combo.currentData() or "") == DIFFERENTIAL_EVOLUTION_METHOD
        self._ftol_label.setText(
            "Population-improvement tolerance:" if is_de else "Objective-improvement tolerance:"
        )
        self._xtol_edit.setEnabled(not is_de)
        self._xtol_label.setEnabled(not is_de)
        self._xtol_edit.setToolTip(
            "Differential Evolution does not use a parameter-step tolerance."
            if is_de
            else ""
        )

    def _defaults_projection(self) -> FitDefaults:
        if self._defaults_backing is None:
            raise ValueError("Fit defaults are not available.")
        ftol = self._positive_finite_default("ftol", self._ftol_edit.text())
        xtol = self._positive_finite_default("xtol", self._xtol_edit.text())
        return validate_fit_defaults(
            FitDefaults(
                method=str(self._method_combo.currentData() or "trf"),
                max_nfev=self._max_eval_spin.value(),
                ftol=ftol,
                xtol=xtol,
                use_seed=self._seed_check.isChecked(),
                seed=self._seed_spin.value(),
            )
        )

    @staticmethod
    def _positive_finite_default(field: str, raw_text: str) -> float:
        try:
            value = float(raw_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a number.") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be finite and positive.")
        return value

    def _on_default_text_changed(self, field: str, raw_text: str) -> None:
        if self._projecting:
            return
        self._default_changed_fields.add(field)
        edit = self._ftol_edit if field == "ftol" else self._xtol_edit
        try:
            self._positive_finite_default(field, raw_text)
        except ValueError as exc:
            self._set_invalid_numeric(("defaults", field), raw_text, str(exc), edit=edit)
            return
        self._clear_invalid_numeric(("defaults", field), edit=edit)

    def _emit_defaults_edit(self, *_args: object, changed_field: str | None = None) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        if changed_field:
            self._default_changed_fields.add(changed_field)
        try:
            edit = FitDefaultsSetupEdit(
                origin_tab_id=self._projected_setup.origin_tab_id,
                origin_capture_id=self._projected_setup.origin_capture_id,
                defaults=self._defaults_projection(),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit defaults are invalid: {exc}")
            return
        self.setupEditsRequested.emit((edit,))

    def _emit_solver_edit(self, *_args: object) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        try:
            edit = FitSolverSetupEdit(
                origin_tab_id=self._projected_setup.origin_tab_id,
                origin_capture_id=self._projected_setup.origin_capture_id,
                solver=str(self._solver_combo.currentData() or "BDF"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit solver edit is invalid: {exc}")
            return
        self.setupEditsRequested.emit((edit,))

    def _project_advanced_analysis(self, snapshot: FitSetupSnapshot) -> None:
        available = fit_uncertainty_coordinate_names(snapshot.parameters, snapshot.datasets)
        profile = snapshot.profile_likelihood
        bootstrap = snapshot.residual_bootstrap
        controls = (
            self._profile_enabled,
            self._profile_confidence,
            self._profile_points,
            self._profile_budget,
            self._profile_method,
            self._bootstrap_enabled,
            self._bootstrap_seed,
            self._bootstrap_replicates,
            self._bootstrap_minimum,
            self._bootstrap_confidence,
            self._bootstrap_fraction,
        )
        blockers = [QtCore.QSignalBlocker(control) for control in controls]
        try:
            self._profile_enabled.setChecked(profile is not None)
            self._bootstrap_enabled.setChecked(bootstrap is not None)
            self._profile_confidence.setText(_format_scientific(profile.confidence_level if profile else 0.95))
            self._profile_points.setValue(profile.points_per_side if profile else 8)
            self._profile_budget.setValue(profile.max_nfev_per_point if profile else 300)
            self._profile_method.setCurrentIndex(
                self._profile_method.findData(profile.optimizer_method if profile else "trf")
            )
            self._bootstrap_seed.setValue(bootstrap.seed if bootstrap else 0)
            self._bootstrap_replicates.setValue(bootstrap.replicates if bootstrap else 200)
            self._bootstrap_minimum.setValue(bootstrap.minimum_successful_replicates if bootstrap else 50)
            self._bootstrap_confidence.setText(_format_scientific(bootstrap.confidence_level if bootstrap else 0.95))
            self._bootstrap_fraction.setText(_format_scientific(bootstrap.minimum_success_fraction if bootstrap else 0.8))
            self._project_coordinate_list(
                self._profile_coordinates,
                available,
                profile.parameter_names if profile else (),
            )
            self._project_coordinate_list(
                self._bootstrap_coordinates,
                available,
                bootstrap.parameter_names if bootstrap else (),
            )
        finally:
            del blockers

    def _advanced_analysis_intent(self, *, clear_uncertainty: bool = False) -> FitAdvancedAnalysisSetupEdit:
        setup = self._projected_setup
        if setup is None:
            raise ValueError("Fit setup is unavailable.")
        profile = None
        if self._profile_enabled.isChecked() and not clear_uncertainty:
            profile = ProfileLikelihoodConfig(
                parameter_names=self._selected_coordinate_names(self._profile_coordinates),
                confidence_level=float(self._profile_confidence.text()),
                points_per_side=self._profile_points.value(),
                max_nfev_per_point=self._profile_budget.value(),
                optimizer_method=str(self._profile_method.currentData() or "trf"),
            )
        bootstrap = None
        if self._bootstrap_enabled.isChecked() and not clear_uncertainty:
            bootstrap = ResidualBootstrapConfig(
                parameter_names=self._selected_coordinate_names(self._bootstrap_coordinates),
                seed=self._bootstrap_seed.value(),
                replicates=self._bootstrap_replicates.value(),
                confidence_level=float(self._bootstrap_confidence.text()),
                minimum_successful_replicates=self._bootstrap_minimum.value(),
                minimum_success_fraction=float(self._bootstrap_fraction.text()),
            )
        return FitAdvancedAnalysisSetupEdit(
            origin_tab_id=setup.origin_tab_id,
            origin_capture_id=setup.origin_capture_id,
            profile_likelihood=profile,
            residual_bootstrap=bootstrap,
            hierarchical=setup.hierarchical,
            # Results owns Shared-refit authoring.  Parameters only preserves
            # any already accepted owner state while editing uncertainty.
            shared_refit_promotions=setup.shared_refit_promotions,
        )

    def _emit_advanced_analysis_edit(self) -> None:
        if self._projecting or self._running:
            return
        try:
            intent = self._advanced_analysis_intent()
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Uncertainty settings are invalid: {exc}")
            return
        self.setupEditsRequested.emit((intent,))

    def set_running_state(self, running: bool) -> None:
        self._running = bool(running)
        self._add_observable_button.setEnabled(not self._running)
        self._add_button.setEnabled(not self._running)

    def clear_pending_setup_edits(self) -> None:
        self._default_changed_fields.clear()
        self._clear_invalid_numeric_state()
        if self._projected_setup is not None:
            self.project_setup_snapshot(self._projected_setup)
