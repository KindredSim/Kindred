"""Historical Fit parameter presentation with current owner-facing hooks.

The widget construction in this module is intentionally close to the executable
Fit presentation.  The widget is only a view: accepted setup, request
construction, Fit results, defaults persistence, and worker ownership remain
outside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_document import (
    FIT_MAX_NFEV_RANGE,
    FIT_SEED_RANGE,
    FitDefaults,
    SUPPORTED_FIT_METHODS,
    SUPPORTED_SOLVERS,
    default_fit_defaults,
    validate_fit_defaults,
)
from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    STATUS_ITEM_LABEL_MAX_CHARS,
    compact_dataset_label,
    compact_text,
)
from kindred.gui.fitting.constants import (
    DE_POPULATION_TOLERANCE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FUNCTION_TOLERANCE_LABEL,
)
from kindred.gui.fitting.launch import (
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
    """Historical algebraic-observable page used for one Fit-local intent."""

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
        self._numeric_transaction_active = False
        self._parameter_dataset_backing: dict[tuple[str, str], float] = {}
        self._pending_parameter_dataset_values: dict[tuple[str, str], float] = {}
        self._defaults_backing: FitDefaults | None = None
        self._default_changed_fields: set[str] = set()
        self._running = False
        self._set_parameter_choices(parameter_choices or {})
        self._build_ui()

    # ------------------------------------------------------------------
    # Historical widget construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_parameters_panel(), stretch=1)

    def _build_parameters_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        outer_layout = QtWidgets.QVBoxLayout(widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(widget)
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll, stretch=1)

        container = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        scroll.setWidget(container)

        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        self._param_table = QtWidgets.QTableWidget()
        self._param_table.setColumnCount(8)
        self._param_table.setHorizontalHeaderLabels(
            [
                "Fit",
                "Log10",
                "Name",
                "Value",
                "Min",
                "Max",
                "Scope",
                "Last Fit",
            ]
        )
        header = self._param_table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(self._param_table.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        self._param_table.setColumnWidth(2, 140)
        self._param_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._param_table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self._param_table.itemChanged.connect(self._on_param_table_item_changed)
        params_layout.addWidget(self._param_table)

        self._parameter_local_table = QtWidgets.QTableWidget(0, 4, params_group)
        self._parameter_local_table.setHorizontalHeaderLabels(
            ("Parameter", "Dataset", "Per-dataset value", "Last Fit")
        )
        self._parameter_local_table.horizontalHeader().setStretchLastSection(True)
        self._parameter_local_table.itemChanged.connect(
            self._on_parameter_local_item_changed
        )
        params_layout.addWidget(self._parameter_local_table)
        action_row = QtWidgets.QHBoxLayout()
        self._add_button = QtWidgets.QToolButton()
        self._add_button.setText("Add")
        self._add_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QtWidgets.QMenu(self._add_button)
        self._add_initial_action = add_menu.addAction("Initial Concentrations")
        self._add_initial_action.triggered.connect(self._add_initial_concentrations)
        self._add_button.setMenu(add_menu)
        self._add_observable_button = QtWidgets.QPushButton(
            "Define New Fit-local Observable…"
        )
        self._add_observable_button.clicked.connect(self._add_fit_local_observable)
        self._view_steps_button = QtWidgets.QPushButton("View Steps")
        self._view_steps_button.setObjectName("global_fit_parameters_view_steps")
        self._view_steps_button.clicked.connect(self.viewStepsRequested)
        action_row.addWidget(self._add_button)
        action_row.addWidget(self._add_observable_button)
        action_row.addWidget(self._view_steps_button)
        action_row.addStretch()
        params_layout.addLayout(action_row)

        reset_row = QtWidgets.QHBoxLayout()
        self._reset_initial_button = QtWidgets.QPushButton("Reset to Initial")
        self._reset_initial_button.clicked.connect(self._reset_to_initial)
        self._reset_last_button = QtWidgets.QPushButton("Reset to Last Fit")
        self._reset_last_button.clicked.connect(self._reset_to_last_fit)
        reset_row.addWidget(self._reset_initial_button)
        reset_row.addWidget(self._reset_last_button)
        reset_row.addStretch()
        params_layout.addLayout(reset_row)

        algo_form = QtWidgets.QFormLayout()
        self._solver_combo = QtWidgets.QComboBox()
        self._solver_combo.setObjectName("global_fit_solver")
        for solver in SUPPORTED_SOLVERS:
            self._solver_combo.addItem(solver, solver)
        algo_form.addRow("Fit solver:", self._solver_combo)
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(list(SUPPORTED_FIT_METHODS))
        fit_factory = default_fit_defaults()
        self._method_combo.setCurrentText(fit_factory.method)
        self._max_eval_spin = QtWidgets.QSpinBox()
        self._max_eval_spin.setRange(*FIT_MAX_NFEV_RANGE)
        self._max_eval_spin.setValue(fit_factory.max_nfev)
        algo_form.addRow("Method:", self._method_combo)
        algo_form.addRow("Max evaluations:", self._max_eval_spin)

        self._ftol_edit = QtWidgets.QLineEdit(_format_scientific(fit_factory.ftol))
        setup_scientific_validator(self._ftol_edit)
        self._ftol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("ftol", text)
        )
        self._ftol_label = QtWidgets.QLabel("ftol:")
        algo_form.addRow(self._ftol_label, self._ftol_edit)

        self._xtol_edit = QtWidgets.QLineEdit(_format_scientific(fit_factory.xtol))
        setup_scientific_validator(self._xtol_edit)
        self._xtol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("xtol", text)
        )
        self._xtol_label = QtWidgets.QLabel("xtol:")
        algo_form.addRow(self._xtol_label, self._xtol_edit)

        self._seed_check = QtWidgets.QCheckBox("Use fixed random seed")
        self._seed_check.setChecked(fit_factory.use_seed)
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(*FIT_SEED_RANGE)
        self._seed_spin.setValue(fit_factory.seed)
        self._seed_spin.setEnabled(self._seed_check.isChecked())
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        algo_form.addRow(self._seed_check, self._seed_spin)
        params_layout.addLayout(algo_form)

        layout.addWidget(params_group, stretch=1)

        self._solver_combo.currentIndexChanged.connect(self._emit_solver_edit)
        self._method_combo.currentTextChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="method")
        )
        self._method_combo.currentTextChanged.connect(self._sync_method_controls)
        self._max_eval_spin.valueChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="max_nfev")
        )
        self._ftol_edit.editingFinished.connect(
            lambda: self._emit_defaults_edit(changed_field="ftol")
        )
        self._xtol_edit.editingFinished.connect(
            lambda: self._emit_defaults_edit(changed_field="xtol")
        )
        self._seed_check.toggled.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="use_seed")
        )
        self._seed_spin.valueChanged.connect(
            lambda *_args: self._emit_defaults_edit(changed_field="seed")
        )
        self._sync_method_controls()
        return widget

    # ------------------------------------------------------------------
    # Owner projection API
    # ------------------------------------------------------------------

    def set_origin(self, origin: FitOriginSnapshot) -> None:
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if self._origin is not None and self._origin.capture_id != origin.capture_id:
            self._clear_invalid_numeric_state()
        self._origin = origin

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Project owner-accepted setup into the historical controls."""

        self._check_snapshot(snapshot)
        self._projected_setup = snapshot
        self._defaults_backing = snapshot.defaults
        self._default_changed_fields.clear()
        self._parameter_dataset_backing = {
            (parameter.name, dataset_id): float(value)
            for parameter in snapshot.parameters
            for dataset_id, value in parameter.selected_config.dataset_values
        }
        self._pending_parameter_dataset_values.clear()
        self._projecting = True
        projection_blockers = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._param_table,
                self._parameter_local_table,
                self._solver_combo,
                self._method_combo,
                self._max_eval_spin,
                self._ftol_edit,
                self._xtol_edit,
                self._seed_check,
                self._seed_spin,
            )
        ]
        try:
            parameters = tuple(snapshot.parameters)
            self._param_table.setRowCount(len(parameters))
            for row, parameter in enumerate(parameters):
                self._render_parameter_row(
                    row,
                    parameter,
                    self._last_fit_parameters.get(parameter.name),
                )
            self._render_parameter_local_rows(
                parameters,
                self._parameter_dataset_ids(snapshot, parameters),
            )
            self._solver_combo.setCurrentText(snapshot.solver)
            self._project_defaults(snapshot.defaults)
        finally:
            del projection_blockers
            self._projecting = False

    def project_setup_delta(
        self,
        snapshot: FitSetupSnapshot,
        intents: Sequence[object],
    ) -> None:
        """Project only the Parameters-owned fields acknowledged by ``intents``.

        The owner snapshot is always advanced, but unrelated editor overlays and
        Qt rows are left alone.  A full snapshot projection remains the explicit
        path for opening/rebasing/resetting a Fit surface.
        """

        self._check_snapshot(snapshot)
        if not isinstance(intents, tuple):
            raise TypeError("Fit setup delta must carry a tuple of intents.")
        self._projected_setup = snapshot
        parameter_names: set[str] = set()
        local_parameter_names: set[str] = set()
        local_table_needed = False
        defaults_changed = False
        solver_changed = False
        for intent in intents:
            if isinstance(intent, FitParameterSetupEdit):
                parameter_names.add(intent.parameter.name)
                local_parameter_names.add(intent.parameter.name)
                self._pending_parameter_dataset_values = {
                    key: value
                    for key, value in self._pending_parameter_dataset_values.items()
                    if key[0] != intent.parameter.name
                }
                local_table_needed = local_table_needed or (
                    intent.parameter.scope == "per_dataset"
                )
            elif isinstance(intent, FitParameterScopeSetupEdit):
                parameter_names.add(intent.parameter_name)
                local_parameter_names.add(intent.parameter_name)
                local_table_needed = True
            elif isinstance(intent, FitDefaultsSetupEdit):
                defaults_changed = True
                self._default_changed_fields.clear()
            elif isinstance(intent, FitSolverSetupEdit):
                solver_changed = True

        if not parameter_names and not defaults_changed and not solver_changed:
            return

        self._projecting = True
        try:
            for name in parameter_names:
                row = self._parameter_row(name)
                parameter = next(
                    item for item in snapshot.parameters if item.name == name
                )
                if row < 0:
                    # A newly admitted parameter is not a field-local edit; the
                    # caller must use the full projection path for that shape.
                    self.project_setup_snapshot(snapshot)
                    return
                self._parameter_dataset_backing.update(
                    {
                        (parameter.name, dataset_id): float(value)
                        for dataset_id, value in parameter.selected_config.dataset_values
                    }
                )
                self._render_parameter_row(row, parameter, None)
            if local_table_needed:
                self._render_parameter_local_rows(
                    tuple(snapshot.parameters),
                    self._parameter_dataset_ids(snapshot, snapshot.parameters),
                    affected_names=local_parameter_names,
                )
            if solver_changed:
                self._solver_combo.setCurrentText(snapshot.solver)
            if defaults_changed:
                self._defaults_backing = snapshot.defaults
                self._project_defaults(snapshot.defaults)
        finally:
            self._projecting = False

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if self._origin is not None and (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
        ):
            raise ValueError("Fit snapshot does not match this origin.")

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self.set_reset_availability(initial=True, last_fit=self._can_reset_last_fit)

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        if snapshot is not None:
            self._check_snapshot(snapshot)
            self._last_fit_parameters = {
                parameter.name: parameter
                for parameter in snapshot.parameters
            }
        else:
            self._last_fit_parameters.clear()
        self.set_reset_availability(
            initial=self._can_reset_initial,
            last_fit=snapshot is not None,
        )
        self._project_last_fit_columns()

    def set_reset_availability(self, *, initial: bool, last_fit: bool) -> None:
        """Project reset affordances without retaining reset snapshots."""

        self._can_reset_initial = bool(initial)
        self._can_reset_last_fit = bool(last_fit)
        self._reset_initial_button.setEnabled(self._can_reset_initial)
        self._reset_last_button.setEnabled(self._can_reset_last_fit)

    def _project_last_fit_columns(self) -> None:
        """Project retained terminal scalar values into Last Fit columns only."""

        blocker = QtCore.QSignalBlocker(self._param_table)
        try:
            for row in range(self._param_table.rowCount()):
                name_item = self._param_table.item(row, 2)
                last_item = self._param_table.item(row, 7)
                if name_item is None or last_item is None:
                    continue
                name = str(
                    name_item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                )
                last_fit = self._last_fit_parameters.get(name)
                last_item.setText(
                    "—"
                    if last_fit is None
                    else (
                        "Per dataset"
                        if last_fit.scope == "per_dataset"
                    else _format_number(last_fit.selected_config.value)
                    )
                )
            for row in range(self._parameter_local_table.rowCount()):
                parameter_item = self._parameter_local_table.item(row, 0)
                dataset_item = self._parameter_local_table.item(row, 1)
                last_item = self._parameter_local_table.item(row, 3)
                if parameter_item is None or dataset_item is None or last_item is None:
                    continue
                name = str(
                    parameter_item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                )
                dataset_id = str(
                    dataset_item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                )
                last_fit = self._last_fit_parameters.get(name)
                last_values = (
                    dict(last_fit.selected_config.dataset_values)
                    if last_fit is not None and last_fit.scope == "per_dataset"
                    else {}
                )
                last_item.setText(
                    _format_number(last_values[dataset_id])
                    if dataset_id in last_values
                    else "—"
                )
        finally:
            del blocker

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

    def set_running_state(self, running: bool) -> None:
        self._running = bool(running)
        self._add_observable_button.setEnabled(not self._running)
        self._add_button.setEnabled(not self._running)

    def clear_pending_setup_edits(self) -> None:
        """Discard only Parameters-owned raw drafts and validation state."""

        self._pending_parameter_dataset_values.clear()
        self._default_changed_fields.clear()
        self._clear_invalid_numeric_state()

    # ------------------------------------------------------------------
    # Parameter presentation and intents
    # ------------------------------------------------------------------

    def _render_parameter_row(
        self,
        row: int,
        parameter: FitParameterVariable,
        last_fit: FitParameterVariable | None,
    ) -> None:
        config = parameter.selected_config
        last_config = last_fit.selected_config if last_fit is not None else None
        fit_item = QtWidgets.QTableWidgetItem()
        fit_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        fit_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if config.fit
            else QtCore.Qt.CheckState.Unchecked
        )
        self._param_table.setItem(row, 0, fit_item)

        log_item = QtWidgets.QTableWidgetItem()
        log_item.setFlags(fit_item.flags())
        log_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if config.log10
            else QtCore.Qt.CheckState.Unchecked
        )
        self._param_table.setItem(row, 1, log_item)

        name_item = QtWidgets.QTableWidgetItem(parameter.display_name)
        name_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        compact_name = compact_text(
            parameter.display_name,
            max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
        )
        name_item.setText(compact_name.display)
        name_item.setToolTip(
            "\n".join(
                part
                for part in (compact_name.full, f"Parameter ID: {parameter.name}")
                if part
            )
        )
        name_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._param_table.setItem(row, 2, name_item)

        for column, value in (
            (3, config.value),
            (4, config.minimum),
            (5, config.maximum),
        ):
            field = {3: "initial", 4: "minimum", 5: "maximum"}[column]
            key = ("parameter", parameter.name, field)
            item = QtWidgets.QTableWidgetItem(
                self._invalid_numeric_text.get(
                    key,
                    "—"
                    if column == 3 and parameter.scope == "per_dataset"
                    else _format_number(value),
                )
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, float(value))
            if column == 3 and parameter.scope == "per_dataset":
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._apply_numeric_item_state(item, key)
            self._param_table.setItem(row, column, item)

        scope = QtWidgets.QComboBox(self._param_table)
        scope.addItems(("Shared", "Per-dataset"))
        scope.currentTextChanged.connect(
            lambda text, parameter_name=parameter.name: (
                self._on_parameter_scope_changed(parameter_name, text)
            )
        )
        self._param_table.setCellWidget(row, 6, scope)
        scope_blocker = QtCore.QSignalBlocker(scope)
        scope.setCurrentText(
            "Per-dataset" if parameter.scope == "per_dataset" else "Shared"
        )
        del scope_blocker

        last_item = QtWidgets.QTableWidgetItem(
            "—"
            if last_fit is None
            else (
                "Per dataset"
                if last_fit.scope == "per_dataset"
                else _format_number(last_config.value)
            )
        )
        last_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        last_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._param_table.setItem(row, 7, last_item)

        tooltip_lines = [parameter.name]
        if parameter.family:
            tooltip_lines.append(f"Family: {parameter.family}")
        if parameter.role:
            tooltip_lines.append(f"Role: {parameter.role}")
        if parameter.unit:
            tooltip_lines.append(f"Unit: {parameter.unit}")
        if parameter.scope == "per_dataset":
            values = ", ".join(
                f"{dataset_id}: {_format_number(value)}"
                for dataset_id, value in config.dataset_values
            )
            if values:
                tooltip_lines.append(f"Per-dataset values: {values}")
        if compact_name.full and compact_name.full != parameter.name:
            tooltip_lines.insert(0, compact_name.full)
        self._param_table.item(row, 2).setToolTip("\n".join(tooltip_lines))

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
                    return mapped.display_name
        return dataset_id

    def _parameter_local_rows(self, name: str) -> tuple[int, ...]:
        return tuple(
            row
            for row in range(self._parameter_local_table.rowCount())
            if str(
                self._parameter_local_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            == name
        )

    def _render_parameter_local_rows(
        self,
        parameters: Sequence[FitParameterVariable],
        dataset_ids: Sequence[str],
        *,
        affected_names: set[str] | None = None,
    ) -> None:
        if affected_names is not None:
            self._reconcile_parameter_local_rows(
                parameters,
                dataset_ids,
                affected_names,
            )
            return
        self._parameter_local_table.setRowCount(0)
        selected_ids = tuple(dict.fromkeys(str(item) for item in dataset_ids))
        last_fit_by_name = self._last_fit_parameters
        rendered = False
        for parameter in parameters:
            row = self._parameter_row(parameter.name)
            scope = self._param_table.cellWidget(row, 6) if row >= 0 else None
            if (
                not isinstance(scope, QtWidgets.QComboBox)
                or scope.currentText() != "Per-dataset"
            ):
                continue
            values = dict(parameter.per_dataset_config.dataset_values)
            for (parameter_name, dataset_id), value in (
                self._pending_parameter_dataset_values.items()
            ):
                if parameter_name == parameter.name:
                    values[dataset_id] = value
            last_parameter = last_fit_by_name.get(parameter.name)
            last_values = dict(
                last_parameter.selected_config.dataset_values
                if last_parameter is not None
                else ()
            )
            for dataset_id in selected_ids:
                row = self._parameter_local_table.rowCount()
                self._parameter_local_table.insertRow(row)

                compact_parameter = compact_text(
                    parameter.display_name,
                    max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
                )
                parameter_item = QtWidgets.QTableWidgetItem(compact_parameter.display)
                parameter_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                parameter_item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    parameter.name,
                )
                parameter_item.setToolTip(
                    f"{compact_parameter.full}\nParameter ID: {parameter.name}"
                )
                self._parameter_local_table.setItem(row, 0, parameter_item)

                full_dataset_name = self._dataset_display_name(dataset_id)
                compact_dataset = compact_dataset_label(
                    full_dataset_name,
                    max_chars=DATASET_LIST_LABEL_MAX_CHARS,
                )
                dataset_item = QtWidgets.QTableWidgetItem(compact_dataset.display)
                dataset_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                dataset_item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    dataset_id,
                )
                dataset_item.setToolTip(
                    f"{compact_dataset.full}\nDataset ID: {dataset_id}"
                )
                self._parameter_local_table.setItem(row, 1, dataset_item)

                key = ("parameter_dataset", parameter.name, dataset_id, "initial")
                value_item = QtWidgets.QTableWidgetItem(
                    self._invalid_numeric_text.get(
                        key,
                        _format_number(values[dataset_id])
                        if dataset_id in values
                        else "",
                    )
                )
                value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                value_item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    dataset_id,
                )
                self._apply_numeric_item_state(value_item, key)
                self._parameter_local_table.setItem(row, 2, value_item)

                last_item = QtWidgets.QTableWidgetItem(
                    _format_number(last_values[dataset_id])
                    if dataset_id in last_values
                    else "—"
                )
                last_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                last_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                last_item.setToolTip(f"Dataset ID: {dataset_id}")
                self._parameter_local_table.setItem(row, 3, last_item)
                rendered = True
        self._parameter_local_table.setVisible(rendered)

    def _reconcile_parameter_local_rows(
        self,
        parameters: Sequence[FitParameterVariable],
        dataset_ids: Sequence[str],
        affected_names: set[str],
    ) -> None:
        """Replace only the local-row groups acknowledged by one Fit edit."""

        selected_ids = tuple(dict.fromkeys(str(item) for item in dataset_ids))
        desired: dict[str, tuple[tuple[str, float], ...]] = {}
        ordered_names: list[str] = []
        for parameter in parameters:
            row = self._parameter_row(parameter.name)
            scope = self._param_table.cellWidget(row, 6) if row >= 0 else None
            if (
                not isinstance(scope, QtWidgets.QComboBox)
                or scope.currentText() != "Per-dataset"
            ):
                continue
            values = dict(parameter.per_dataset_config.dataset_values)
            values.update(
                {
                    dataset_id: value
                    for (parameter_name, dataset_id), value in (
                        self._pending_parameter_dataset_values.items()
                    )
                    if parameter_name == parameter.name
                }
            )
            desired[parameter.name] = tuple(
                (dataset_id, values[dataset_id])
                for dataset_id in selected_ids
                if dataset_id in values
            )
            ordered_names.append(parameter.name)

        def row_name(row: int) -> str:
            item = self._parameter_local_table.item(row, 0)
            return (
                str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                if item is not None
                else ""
            )

        blockers = QtCore.QSignalBlocker(self._parameter_local_table)
        try:
            for row in range(self._parameter_local_table.rowCount() - 1, -1, -1):
                if row_name(row) in affected_names:
                    self._parameter_local_table.removeRow(row)

            order_index = {name: index for index, name in enumerate(ordered_names)}
            for name in ordered_names:
                if name not in affected_names:
                    continue
                entries = desired.get(name, ())
                target_order = order_index[name]
                insertion = self._parameter_local_table.rowCount()
                for row in range(self._parameter_local_table.rowCount()):
                    current_name = row_name(row)
                    if order_index.get(current_name, target_order) > target_order:
                        insertion = row
                        break
                parameter = next(item for item in parameters if item.name == name)
                last_parameter = self._last_fit_parameters.get(name)
                last_values = dict(
                    last_parameter.selected_config.dataset_values
                    if last_parameter is not None
                    else ()
                )
                for offset, (dataset_id, value) in enumerate(entries):
                    row = insertion + offset
                    self._parameter_local_table.insertRow(row)
                    compact_parameter = compact_text(
                        parameter.display_name,
                        max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
                    )
                    parameter_item = QtWidgets.QTableWidgetItem(
                        compact_parameter.display
                    )
                    parameter_item.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                    parameter_item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        parameter.name,
                    )
                    parameter_item.setToolTip(
                        f"{compact_parameter.full}\nParameter ID: {parameter.name}"
                    )
                    self._parameter_local_table.setItem(row, 0, parameter_item)
                    compact_dataset = compact_dataset_label(
                        self._dataset_display_name(dataset_id),
                        max_chars=DATASET_LIST_LABEL_MAX_CHARS,
                    )
                    dataset_item = QtWidgets.QTableWidgetItem(compact_dataset.display)
                    dataset_item.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                    dataset_item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        dataset_id,
                    )
                    dataset_item.setToolTip(
                        f"{compact_dataset.full}\nDataset ID: {dataset_id}"
                    )
                    self._parameter_local_table.setItem(row, 1, dataset_item)
                    key = ("parameter_dataset", parameter.name, dataset_id, "initial")
                    value_item = QtWidgets.QTableWidgetItem(
                        self._invalid_numeric_text.get(key, _format_number(value))
                    )
                    value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    value_item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset_id)
                    self._apply_numeric_item_state(value_item, key)
                    self._parameter_local_table.setItem(row, 2, value_item)
                    last_item = QtWidgets.QTableWidgetItem(
                        _format_number(last_values[dataset_id])
                        if dataset_id in last_values
                        else "—"
                    )
                    last_item.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                    last_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    last_item.setToolTip(f"Dataset ID: {dataset_id}")
                    self._parameter_local_table.setItem(row, 3, last_item)
        finally:
            del blockers
        self._parameter_local_table.setVisible(bool(self._parameter_local_table.rowCount()))

    def _parameter_row(self, name: str) -> int:
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 2)
            if item is None:
                continue
            if str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == name:
                return row
        return -1

    @property
    def has_invalid_edits(self) -> bool:
        """Whether raw numeric text currently blocks a typed Fit request."""

        return bool(self._invalid_numeric_text)

    def _apply_numeric_item_state(
        self,
        item: QtWidgets.QTableWidgetItem,
        key: tuple[str, ...],
    ) -> None:
        param_blocked = self._param_table.blockSignals(True)
        local_blocked = self._parameter_local_table.blockSignals(True)
        try:
            if key in self._invalid_numeric_text:
                item.setBackground(QtGui.QColor(255, 200, 200))
                item.setForeground(QtGui.QColor(80, 0, 0))
                item.setToolTip(self._invalid_numeric_messages.get(key, "Invalid value"))
                return
            item.setBackground(QtGui.QBrush())
            item.setForeground(QtGui.QBrush())
            item.setToolTip("")
        finally:
            self._param_table.blockSignals(param_blocked)
            self._parameter_local_table.blockSignals(local_blocked)

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
        if item is not None:
            self._apply_numeric_item_state(item, key)
        if edit is not None:
            edit.setStyleSheet("background: rgb(255, 200, 200); color: rgb(80, 0, 0);")
            edit.setToolTip(str(message))
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
        if item is not None:
            self._apply_numeric_item_state(item, key)
        if edit is not None:
            edit.setStyleSheet("")
            edit.setToolTip("")
        if changed:
            self.validityChanged.emit()

    def _parameter_numeric_items(
        self,
        name: str,
    ) -> tuple[tuple[tuple[str, ...], QtWidgets.QTableWidgetItem], ...]:
        row = self._parameter_row(name)
        if row < 0:
            return ()
        return tuple(
            (("parameter", name, field), item)
            for column, field in (
                (3, "initial"), (4, "minimum"), (5, "maximum")
            )
            if (item := self._param_table.item(row, column)) is not None
        )

    def _set_parameter_row_invalid(self, name: str, message: str) -> None:
        for key, item in self._parameter_numeric_items(name):
            self._set_invalid_numeric(key, item.text(), message, item=item)

    def _clear_parameter_row_invalid(self, name: str) -> None:
        for key, item in self._parameter_numeric_items(name):
            self._clear_invalid_numeric(
                key, item=item
            )

    def _clear_invalid_numeric_state(self) -> None:
        if not self._invalid_numeric_text:
            return
        self._invalid_numeric_text.clear()
        self._invalid_numeric_messages.clear()
        self.validityChanged.emit()

    @staticmethod
    def _parse_number(raw_text: str, field: str) -> float:
        try:
            return float(raw_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a number.") from exc

    def _validate_item_text(
        self,
        key: tuple[str, ...],
        item: QtWidgets.QTableWidgetItem,
        field: str,
    ) -> bool:
        try:
            self._parse_number(item.text(), field)
        except ValueError as exc:
            self._set_invalid_numeric(
                key,
                item.text(),
                str(exc),
                item=item,
            )
            self.statusMessage.emit(str(exc))
            return False
        self._clear_invalid_numeric(key, item=item)
        return True

    def _on_parameter_scope_changed(self, name: str, _text: str) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = next(
            item for item in self._projected_setup.parameters if item.name == name
        )
        row = self._parameter_row(name)
        scope = self._param_table.cellWidget(row, 6) if row >= 0 else None
        desired = (
            "per_dataset"
            if isinstance(scope, QtWidgets.QComboBox)
            and scope.currentText() == "Per-dataset"
            else "shared"
        )
        if desired != parameter.scope:
            self.setupEditsRequested.emit(
                (
                    FitParameterScopeSetupEdit(
                        origin_tab_id=self._projected_setup.origin_tab_id,
                        origin_capture_id=self._projected_setup.origin_capture_id,
                        parameter_name=name,
                        scope=desired,
                    ),
                )
            )
        # The owner acknowledgement projects the affected row once.  Rebuilding
        # the local table here would duplicate the accepted scope transition.

    def _on_parameter_local_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._projecting or item.column() != 2:
            return
        parameter_item = self._parameter_local_table.item(item.row(), 0)
        if parameter_item is None:
            return
        name = str(parameter_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if name:
            dataset_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            key = ("parameter_dataset", name, dataset_id, "initial")
            if self._validate_item_text(
                key,
                item,
                f"{name} value for dataset {dataset_id}",
            ):
                self._emit_parameter_edit(name, changed_key=key, changed_item=item)

    def _on_param_table_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._projecting or item.column() not in {0, 1, 3, 4, 5}:
            return
        name_item = self._param_table.item(item.row(), 2)
        if name_item is None or self._projected_setup is None:
            return
        name = str(name_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if not name:
            return
        changed_key = (
            ("parameter", name, {3: "initial", 4: "minimum", 5: "maximum"}[item.column()])
            if item.column() in {3, 4, 5}
            else None
        )
        self._emit_parameter_edit(
            name,
            changed_key=changed_key,
            changed_item=item if changed_key is not None else None,
        )

    def _parameter_projection_for_change(
        self,
        name: str,
        *,
        changed_key: tuple[str, ...] | None,
    ) -> FitParameterVariable:
        if self._projected_setup is None:
            raise RuntimeError("No projected Fit setup is available.")
        parameter = next(
            value for value in self._projected_setup.parameters if value.name == name
        )
        row = next(
            row
            for row in range(self._param_table.rowCount())
            if str(
                self._param_table.item(row, 2).data(QtCore.Qt.ItemDataRole.UserRole)
                or ""
            )
            == name
        )
        scope = self._param_table.cellWidget(row, 6)
        per_dataset = (
            isinstance(scope, QtWidgets.QComboBox)
            and scope.currentText() == "Per-dataset"
        )
        invalid_keys = [
            key
            for key in self._invalid_numeric_text
            if key[0] == "parameter_dataset" and key[1] == name
        ]
        if invalid_keys:
            raise ValueError(f"Correct invalid numeric text for {name} first.")
        selected_config = (
            parameter.per_dataset_config
            if per_dataset
            else parameter.shared_config
        )
        retained_dataset_values = dict(parameter.per_dataset_config.dataset_values)
        for (parameter_name, dataset_id), value in (
            self._pending_parameter_dataset_values.items()
        ):
            if parameter_name == name:
                retained_dataset_values[dataset_id] = value
        if per_dataset:
            for local_row in self._parameter_local_rows(name):
                value_item = self._parameter_local_table.item(local_row, 2)
                dataset_id = str(value_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                key = ("parameter_dataset", name, dataset_id, "initial")
                if changed_key == key:
                    parsed = self._parse_number(
                        value_item.text(),
                        f"{name} value for dataset {dataset_id}",
                    )
                    self._pending_parameter_dataset_values[(name, dataset_id)] = parsed
                    retained_dataset_values[dataset_id] = parsed
        ordered_dataset_ids = tuple(
            dict.fromkeys(
                [
                    dataset_id
                    for dataset_id, _value in parameter.per_dataset_config.dataset_values
                ]
                + [
                    str(
                        self._parameter_local_table.item(local_row, 2).data(
                            QtCore.Qt.ItemDataRole.UserRole
                        )
                        or ""
                    )
                    for local_row in self._parameter_local_rows(name)
                ]
            )
        )
        dataset_values = (
            tuple(
                (dataset_id, retained_dataset_values[dataset_id])
                for dataset_id in ordered_dataset_ids
            )
            if per_dataset
            else tuple(selected_config.dataset_values)
        )
        numeric_values = {
            "initial": selected_config.value,
            "minimum": self._parse_number(
                self._param_table.item(row, 4).text(),
                f"{name} minimum",
            ),
            "maximum": self._parse_number(
                self._param_table.item(row, 5).text(),
                f"{name} maximum",
            ),
        }
        if not per_dataset:
            numeric_values["initial"] = self._parse_number(
                self._param_table.item(row, 3).text(),
                f"{name} value",
            )
        active = FitParameterConfiguration(
            value=float(numeric_values["initial"]),
            minimum=float(numeric_values["minimum"]),
            maximum=float(numeric_values["maximum"]),
            fit=(
                self._param_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            log10=(
                self._param_table.item(row, 1).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            dataset_values=dataset_values,
        )
        shared_config = parameter.shared_config
        per_dataset_config = parameter.per_dataset_config
        if per_dataset:
            per_dataset_config = active
        else:
            shared_config = active
        return FitParameterVariable(
            name=parameter.name,
            display_name=parameter.display_name,
            scope="per_dataset" if per_dataset else "shared",
            shared_config=shared_config,
            per_dataset_config=per_dataset_config,
            family=parameter.family,
            role=parameter.role,
            unit=parameter.unit,
        )

    def _emit_parameter_edit(
        self,
        name: str,
        *,
        changed_key: tuple[str, ...] | None = None,
        changed_item: QtWidgets.QTableWidgetItem | None = None,
    ) -> bool:
        if (
            self._projecting
            or self._numeric_transaction_active
            or self._projected_setup is None
        ):
            return False
        self._numeric_transaction_active = True
        try:
            edited = self._parameter_projection_for_change(
                name,
                changed_key=changed_key,
            )
            intent = FitParameterSetupEdit(
                origin_tab_id=self._projected_setup.origin_tab_id,
                origin_capture_id=self._projected_setup.origin_capture_id,
                parameter=edited,
            )
        except (
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            if changed_key is not None:
                if changed_key[:2] == ("parameter", name):
                    self._set_parameter_row_invalid(name, str(exc))
                else:
                    item = changed_item
                    if item is not None:
                        self._set_invalid_numeric(
                            changed_key,
                            item.text(),
                            str(exc),
                            item=item,
                        )
            self.statusMessage.emit(f"Fit parameter edit is invalid: {exc}")
            return False
        finally:
            self._numeric_transaction_active = False
        if changed_key is not None:
            if changed_key[:2] == ("parameter", name):
                self._clear_parameter_row_invalid(name)
            else:
                self._clear_invalid_numeric(changed_key, item=changed_item)
        self.setupEditsRequested.emit((intent,))
        return True

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

    # ------------------------------------------------------------------
    # Historical dialogs and reset controls
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Optimizer presentation
    # ------------------------------------------------------------------

    def _project_defaults(self, defaults: FitDefaults) -> None:
        self._defaults_backing = defaults
        self._default_changed_fields.clear()
        self._method_combo.setCurrentText(defaults.method)
        self._max_eval_spin.setValue(defaults.max_nfev)
        for field, edit, value in (
            ("ftol", self._ftol_edit, defaults.ftol),
            ("xtol", self._xtol_edit, defaults.xtol),
        ):
            key = ("defaults", field)
            edit.setText(self._invalid_numeric_text.get(key, _format_scientific(value)))
            if key in self._invalid_numeric_text:
                edit.setStyleSheet(
                    "background: rgb(255, 200, 200); color: rgb(80, 0, 0);"
                )
                edit.setToolTip(
                    self._invalid_numeric_messages.get(key, "Invalid value")
                )
            else:
                edit.setStyleSheet("")
                edit.setToolTip("")
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        self._sync_method_controls()

    def _sync_method_controls(self, _method: str = "") -> None:
        is_de = self._method_combo.currentText() == DIFFERENTIAL_EVOLUTION_METHOD
        self._ftol_label.setText(
            f"{DE_POPULATION_TOLERANCE_LABEL if is_de else FUNCTION_TOLERANCE_LABEL}:"
        )
        self._xtol_label.setVisible(not is_de)
        self._xtol_edit.setVisible(not is_de)

    def _emit_solver_edit(self, *_args: object) -> None:
        if self._projecting or self._projected_setup is None:
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

    def _defaults_projection(self) -> FitDefaults:
        if any(key[0] == "defaults" for key in self._invalid_numeric_text):
            raise ValueError("Correct invalid optimizer-default text first.")
        backing = self._defaults_backing
        if backing is None:
            raise ValueError("Fit defaults are not available.")
        ftol = float(backing.ftol)
        xtol = float(backing.xtol)
        if "ftol" in self._default_changed_fields:
            ftol = self._positive_finite_default("ftol", self._ftol_edit.text())
        if "xtol" in self._default_changed_fields:
            xtol = self._positive_finite_default("xtol", self._xtol_edit.text())
        candidate = FitDefaults(
            method=self._method_combo.currentText(),
            max_nfev=self._max_eval_spin.value(),
            ftol=ftol,
            xtol=xtol,
            use_seed=self._seed_check.isChecked(),
            seed=self._seed_spin.value(),
        )
        return validate_fit_defaults(candidate)

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
        key = ("defaults", field)
        try:
            self._positive_finite_default(field, raw_text)
        except ValueError as exc:
            self._set_invalid_numeric(
                key,
                raw_text,
                str(exc),
                edit=edit,
            )
            return
        self._clear_invalid_numeric(key, edit=edit)

    def _emit_defaults_edit(
        self,
        *_args: object,
        changed_field: str | None = None,
    ) -> None:
        if self._projecting or self._projected_setup is None:
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
            self.statusMessage.emit(f"Fit defaults edit is invalid: {exc}")
            return
        self.setupEditsRequested.emit((edit,))

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
