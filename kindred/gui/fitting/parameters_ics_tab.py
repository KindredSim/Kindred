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

from kindred.core.project_document import FitDefaults
from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    STATUS_ITEM_LABEL_MAX_CHARS,
    compact_dataset_label,
    compact_text,
)
from kindred.gui.fitting.constants import (
    FITTING_MAX_NFEV_RANGE,
    FITTING_METHODS,
    FITTING_SEED_RANGE,
)
from kindred.gui.fitting.launch import (
    FitDefaultsSetupEdit,
    FitOriginSnapshot,
    FitParameterSetupEdit,
    FitParameterVariable,
    FitSetupSnapshot,
)
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = ["ParametersIcsTab"]


def _format_number(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _format_scientific(value: object) -> str:
    text = f"{float(value):g}"
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
        available_observations: Mapping[str, object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Define New Fit-local Observable")
        self.setModal(True)
        self.resize(520, 340)

        self._dataset_ids = [str(x) for x in (dataset_ids or []) if str(x)]
        self._available_observations = self._normalize_available_observations(
            available_observations
        )
        self._selection: dict[str, str] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self._observable_tab = QtWidgets.QWidget(self)
        layout.addWidget(self._observable_tab, stretch=1)
        self._init_observable_tab()
        self._connect_observable_signals()
        self._add_button_box(layout)

    @staticmethod
    def _normalize_available_observations(
        available_observations: Mapping[str, object] | None,
    ) -> dict[str, dict[str, object]]:
        normalized: dict[str, dict[str, object]] = {}
        for raw_dataset_id, raw_entry in (available_observations or {}).items():
            dataset_id = str(raw_dataset_id).strip()
            if not dataset_id:
                continue
            if isinstance(raw_entry, Mapping):
                display_name = (
                    str(raw_entry.get("display_name") or dataset_id).strip()
                    or dataset_id
                )
                raw_items = raw_entry.get("observations", ())
            else:
                display_name = dataset_id
                raw_items = raw_entry
            observations: list[tuple[str, str]] = []
            for raw_item in (
                raw_items
                if isinstance(raw_items, Sequence)
                and not isinstance(raw_items, (str, bytes))
                else ()
            ):
                if isinstance(raw_item, Mapping):
                    raw_id = str(
                        raw_item.get("raw_observation_id") or raw_item.get("id") or ""
                    ).strip()
                    raw_column = str(
                        raw_item.get("raw_column") or raw_item.get("column") or raw_id
                    ).strip()
                elif isinstance(raw_item, Sequence) and not isinstance(
                    raw_item, (str, bytes)
                ):
                    values = tuple(raw_item)
                    raw_id = str(values[0]).strip() if values else ""
                    raw_column = str(values[1]).strip() if len(values) > 1 else raw_id
                else:
                    continue
                if raw_id and raw_column:
                    observations.append((raw_id, raw_column))
            normalized[dataset_id] = {
                "display_name": display_name,
                "observations": tuple(observations),
            }
        return normalized

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
                str(entry["display_name"]),
                dataset_id,
            )
        target_form.addRow("Dataset:", self._observable_dataset_combo)
        target_form.addRow(
            "Observed column:",
            self._observable_column_combo,
        )
        observable_layout.addLayout(target_form)

        scope_form = QtWidgets.QFormLayout()
        obs_scope_row = QtWidgets.QHBoxLayout()
        self._observable_shared_radio = QtWidgets.QRadioButton("Shared")
        self._observable_dataset_radio = QtWidgets.QRadioButton("Per-dataset")
        self._observable_shared_radio.setChecked(True)
        obs_scope_row.addWidget(self._observable_shared_radio)
        obs_scope_row.addWidget(self._observable_dataset_radio)
        obs_scope_row.addStretch()
        obs_scope_widget = QtWidgets.QWidget(self._observable_tab)
        obs_scope_widget.setLayout(obs_scope_row)
        scope_form.addRow("Auto-added scalars:", obs_scope_widget)
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
        entry = self._available_observations.get(dataset_id, {})
        for raw_id, raw_column in entry.get("observations", ()):
            self._observable_column_combo.addItem(raw_column, raw_id)

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
        mode = "shared" if self._observable_shared_radio.isChecked() else "per_dataset"
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


class ParametersIcsTab(QtWidgets.QWidget):
    """Presentation-only parameter, optimizer, and execution panel."""

    setupEditsRequested = QtCore.Signal(object)
    addObservableRequested = QtCore.Signal(object)
    restoreSetupRequested = QtCore.Signal(object)
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
        self._initial_snapshot: FitSetupSnapshot | None = None
        self._last_fit_snapshot: FitSetupSnapshot | None = None
        self._execution_presentation_widget: QtWidgets.QWidget | None = None
        self._selected_dataset_ids: tuple[str, ...] = ()
        self._selected_dataset_ids_explicit = False
        self._parameter_choices: dict[str, object] = {}
        self._invalid_numeric_text: dict[tuple[str, ...], str] = {}
        self._invalid_numeric_messages: dict[tuple[str, ...], str] = {}
        self._projecting = False
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
        self._add_observable_button = QtWidgets.QPushButton(
            "Define New Fit-local Observable…"
        )
        self._add_observable_button.clicked.connect(self._add_fit_local_observable)
        self._view_steps_button = QtWidgets.QPushButton("View Steps")
        self._view_steps_button.setObjectName("global_fit_parameters_view_steps")
        self._view_steps_button.clicked.connect(self.viewStepsRequested)
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
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(list(FITTING_METHODS))
        self._method_combo.setCurrentText("trf")
        self._max_eval_spin = QtWidgets.QSpinBox()
        self._max_eval_spin.setRange(*FITTING_MAX_NFEV_RANGE)
        self._max_eval_spin.setValue(1000)
        algo_form.addRow("Method:", self._method_combo)
        algo_form.addRow("Max evaluations:", self._max_eval_spin)

        self._ftol_edit = QtWidgets.QLineEdit("1e-10")
        setup_scientific_validator(self._ftol_edit)
        self._ftol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("ftol", text)
        )
        algo_form.addRow("ftol:", self._ftol_edit)

        self._xtol_edit = QtWidgets.QLineEdit("1e-10")
        setup_scientific_validator(self._xtol_edit)
        self._xtol_edit.textChanged.connect(
            lambda text: self._on_default_text_changed("xtol", text)
        )
        algo_form.addRow("xtol:", self._xtol_edit)

        self._seed_check = QtWidgets.QCheckBox("Use fixed random seed")
        self._seed_check.setChecked(True)
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(*FITTING_SEED_RANGE)
        self._seed_spin.setValue(42)
        self._seed_spin.setEnabled(self._seed_check.isChecked())
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        algo_form.addRow(self._seed_check, self._seed_spin)
        params_layout.addLayout(algo_form)

        # The current execution contract is per dataset.  This host preserves
        # the historical placement while allowing FittingWindow to insert its
        # current execution presentation without this tab owning that policy.
        self._execution_presentation_host = QtWidgets.QWidget(params_group)
        self._execution_presentation_layout = QtWidgets.QVBoxLayout(
            self._execution_presentation_host
        )
        self._execution_presentation_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(self._execution_presentation_host)

        layout.addWidget(params_group, stretch=1)

        self._method_combo.currentTextChanged.connect(self._emit_defaults_edit)
        self._max_eval_spin.valueChanged.connect(self._emit_defaults_edit)
        self._ftol_edit.editingFinished.connect(self._emit_defaults_edit)
        self._xtol_edit.editingFinished.connect(self._emit_defaults_edit)
        self._seed_check.toggled.connect(self._emit_defaults_edit)
        self._seed_spin.valueChanged.connect(self._emit_defaults_edit)
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
        self._projecting = True
        projection_blockers = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._param_table,
                self._parameter_local_table,
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
            last_fit = {
                item.name: item
                for item in (
                    self._last_fit_snapshot.parameters
                    if self._last_fit_snapshot is not None
                    else ()
                )
            }
            for row, parameter in enumerate(parameters):
                self._render_parameter_row(
                    row,
                    parameter,
                    last_fit.get(parameter.name),
                )
            self._render_parameter_local_rows(
                parameters,
                self._parameter_dataset_ids(snapshot, parameters),
            )
            self._project_defaults(snapshot.defaults)
        finally:
            del projection_blockers
            self._projecting = False

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if self._origin is not None and (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
        ):
            raise ValueError("Fit snapshot does not match this origin.")

    def install_initial_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self._initial_snapshot = snapshot
        self._reset_initial_button.setEnabled(True)

    def set_last_fit_snapshot(self, snapshot: FitSetupSnapshot | None) -> None:
        if snapshot is not None:
            self._check_snapshot(snapshot)
        self._last_fit_snapshot = snapshot
        self._reset_last_button.setEnabled(snapshot is not None)
        if self._projected_setup is not None:
            self.project_setup_snapshot(self._projected_setup)

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
        self._selected_dataset_ids_explicit = "dataset_ids" in self._parameter_choices

    def set_selected_dataset_ids(self, dataset_ids: Sequence[str]) -> None:
        self._selected_dataset_ids_explicit = True
        self._selected_dataset_ids = tuple(
            dict.fromkeys(
                str(item).strip() for item in (dataset_ids or ()) if str(item).strip()
            )
        )
        # Selection controls the Fit-local observable target only. Parameter
        # rows always retain every captured dataset value, including excluded
        # datasets.

    def set_execution_presentation_widget(
        self,
        widget: QtWidgets.QWidget | None,
    ) -> None:
        """Install a parent-owned per-dataset execution view in the reserved host."""

        if self._execution_presentation_widget is not None:
            self._execution_presentation_widget.setParent(None)
        self._execution_presentation_widget = widget
        if widget is not None:
            self._execution_presentation_layout.addWidget(widget)

    def set_running_state(self, running: bool) -> None:
        self._running = bool(running)
        self._add_observable_button.setEnabled(not self._running)

    # ------------------------------------------------------------------
    # Parameter presentation and intents
    # ------------------------------------------------------------------

    def _render_parameter_row(
        self,
        row: int,
        parameter: FitParameterVariable,
        last_fit: FitParameterVariable | None,
    ) -> None:
        fit_item = QtWidgets.QTableWidgetItem()
        fit_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        fit_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if parameter.fit
            else QtCore.Qt.CheckState.Unchecked
        )
        self._param_table.setItem(row, 0, fit_item)

        log_item = QtWidgets.QTableWidgetItem()
        log_item.setFlags(fit_item.flags())
        log_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if parameter.log10
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
            (3, parameter.initial),
            (4, parameter.minimum),
            (5, parameter.maximum),
        ):
            field = {3: "initial", 4: "minimum", 5: "maximum"}[column]
            key = ("parameter", parameter.name, field)
            item = QtWidgets.QTableWidgetItem(
                self._invalid_numeric_text.get(key, _format_number(value))
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
                else _format_number(last_fit.initial)
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
                for dataset_id, value in parameter.dataset_values
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
                    for dataset_id, _value in parameter.dataset_values
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
    ) -> None:
        self._parameter_local_table.setRowCount(0)
        selected_ids = tuple(dict.fromkeys(str(item) for item in dataset_ids))
        last_fit_by_name = {
            parameter.name: parameter
            for parameter in (
                self._last_fit_snapshot.parameters
                if self._last_fit_snapshot is not None
                else ()
            )
        }
        rendered = False
        for parameter in parameters:
            row = self._parameter_row(parameter.name)
            scope = self._param_table.cellWidget(row, 6) if row >= 0 else None
            if not (
                parameter.scope == "per_dataset"
                or (
                    isinstance(scope, QtWidgets.QComboBox)
                    and scope.currentText() == "Per-dataset"
                )
            ):
                continue
            values = dict(parameter.dataset_values)
            last_parameter = last_fit_by_name.get(parameter.name)
            last_values = dict(
                last_parameter.dataset_values if last_parameter is not None else ()
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
        if key in self._invalid_numeric_text:
            item.setBackground(QtGui.QColor(255, 200, 200))
            item.setForeground(QtGui.QColor(80, 0, 0))
            item.setToolTip(self._invalid_numeric_messages.get(key, "Invalid value"))
            return
        item.setBackground(QtGui.QBrush())
        item.setForeground(QtGui.QBrush())
        item.setToolTip("")

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
        parameters = tuple(self._projected_setup.parameters)
        self._parameter_local_table.blockSignals(True)
        try:
            self._render_parameter_local_rows(
                parameters,
                self._parameter_dataset_ids(self._projected_setup, parameters),
            )
        finally:
            self._parameter_local_table.blockSignals(False)
        self._emit_parameter_edit(name)

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
        changed_key = None
        if item.column() in {3, 4, 5}:
            field = {3: "initial", 4: "minimum", 5: "maximum"}[item.column()]
            changed_key = ("parameter", name, field)
            if not self._validate_item_text(
                changed_key,
                item,
                f"{name} {field}",
            ):
                return
        self._emit_parameter_edit(
            name,
            changed_key=changed_key,
            changed_item=item if changed_key is not None else None,
        )

    def _parameter_projection(self, name: str) -> FitParameterVariable:
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
            if (key[0] == "parameter" and key[1] == name)
            or (key[0] == "parameter_dataset" and key[1] == name)
        ]
        if invalid_keys:
            raise ValueError(f"Correct invalid numeric text for {name} first.")
        retained_dataset_values = dict(parameter.dataset_values)
        if per_dataset:
            for local_row in self._parameter_local_rows(name):
                value_item = self._parameter_local_table.item(local_row, 2)
                dataset_id = str(value_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                retained_dataset_values[dataset_id] = self._parse_number(
                    value_item.text(),
                    f"{name} value for dataset {dataset_id}",
                )
        ordered_dataset_ids = tuple(
            dict.fromkeys(
                [dataset_id for dataset_id, _value in parameter.dataset_values]
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
        dataset_values = tuple(
            (dataset_id, retained_dataset_values[dataset_id])
            for dataset_id in ordered_dataset_ids
        )
        return FitParameterVariable(
            name=parameter.name,
            display_name=parameter.display_name,
            initial=(
                parameter.initial
                if per_dataset
                else self._parse_number(
                    self._param_table.item(row, 3).text(),
                    f"{name} initial",
                )
            ),
            minimum=self._parse_number(
                self._param_table.item(row, 4).text(),
                f"{name} minimum",
            ),
            maximum=self._parse_number(
                self._param_table.item(row, 5).text(),
                f"{name} maximum",
            ),
            fit=(
                self._param_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            log10=(
                self._param_table.item(row, 1).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            scope="per_dataset" if per_dataset else "shared",
            dataset_values=dataset_values,
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
        if self._projecting or self._projected_setup is None:
            return False
        try:
            edited = self._parameter_projection(name)
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
            if changed_key is not None and changed_item is not None:
                self._set_invalid_numeric(
                    changed_key,
                    changed_item.text(),
                    str(exc),
                    item=changed_item,
                )
            self.statusMessage.emit(f"Fit parameter edit is invalid: {exc}")
            return False
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

    # ------------------------------------------------------------------
    # Historical dialogs and reset controls
    # ------------------------------------------------------------------

    def _reset_to_initial(self) -> None:
        if self._initial_snapshot is None:
            self.statusMessage.emit("Initial Fit setup is not available.")
            return
        self._clear_invalid_numeric_state()
        self.restoreSetupRequested.emit(self._initial_snapshot)

    def _reset_to_last_fit(self) -> None:
        if self._last_fit_snapshot is None:
            self.statusMessage.emit("No completed Fit is available to restore.")
            return
        self._clear_invalid_numeric_state()
        self.restoreSetupRequested.emit(self._last_fit_snapshot)

    # ------------------------------------------------------------------
    # Optimizer presentation
    # ------------------------------------------------------------------

    def _project_defaults(self, defaults: FitDefaults) -> None:
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

    def _defaults_projection(self) -> FitDefaults:
        if any(key[0] == "defaults" for key in self._invalid_numeric_text):
            raise ValueError("Correct invalid optimizer-default text first.")
        ftol = self._positive_finite_default("ftol", self._ftol_edit.text())
        xtol = self._positive_finite_default("xtol", self._xtol_edit.text())
        return FitDefaults(
            method=self._method_combo.currentText(),
            max_nfev=self._max_eval_spin.value(),
            ftol=ftol,
            xtol=xtol,
            use_seed=self._seed_check.isChecked(),
            seed=self._seed_spin.value(),
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

    def _emit_defaults_edit(self, *_args: object) -> None:
        if self._projecting or self._projected_setup is None:
            return
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
    ) -> dict[str, object]:
        selected = {str(dataset_id) for dataset_id in dataset_ids}
        for key in (
            "available_observations",
            "available_observable_observations",
            "available_observation_targets",
        ):
            value = self._parameter_choices.get(key)
            if isinstance(value, Mapping):
                return {
                    str(dataset_id): entry
                    for dataset_id, entry in value.items()
                    if str(dataset_id) in selected
                }
        if self._origin is None:
            return {}
        return {
            mapped.dataset_id: {
                "display_name": mapped.display_name,
                "observations": tuple(
                    {
                        "raw_observation_id": observation.raw_observation_id,
                        "raw_column": observation.raw_column,
                    }
                    for observation in mapped.observations
                ),
            }
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in selected
        }
