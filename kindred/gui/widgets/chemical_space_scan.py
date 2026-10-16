"""Chemical Space Scan window. All generated trajectories stay outside Qt."""
from __future__ import annotations

import html

from kindred.core.quantity_units import format_unit_for_display, presentation_unit

from dataclasses import replace
import os
import secrets

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import canonical_initial_values, parse_points_text, SUPPORTED_SOLVERS
from kindred.core.scipy_integrate import validate_scipy_rtol
from kindred.core.validation import parse_finite_numeric_text
from kindred.gui.controllers.simulation_job_owner import resolve_run_parallel_capacity
from kindred.gui.widgets.scan_range_dialog import ScanRangeDialog, ScanConstraintDialog, entry
from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken
from kindred.gui.ui_helpers import SCIENTIFIC_NAME_HTML_ROLE, SOLVER_TOOLTIP, ScientificNameDelegate, ignore_wheel
from kindred.scan.config import ScanConfig, InputRange, START, END, input_label
from kindred.scan.inputs import input_catalog
from kindred.scan.intervals import Interval


def _whole_number(box: QtWidgets.QLineEdit, label: str) -> int:
    try:
        return int(box.text())
    except ValueError:
        raise ValueError(f"{label} must be a whole number.") from None


class ChemicalSpaceScanDialog(QtWidgets.QDialog):
    def __init__(self, tab, owner, parent=None, *, selected_set_id=None):
        super().__init__(parent)
        self.tab, self.owner = tab, owner
        self.setWindowTitle("Chemical Space Scan")
        self.resize(900, 760)
        self.ranges = {}
        self.constraints = []
        self.rows = {}
        layout = QtWidgets.QVBoxLayout(self)
        self.mechanism = QtWidgets.QLabel()
        layout.addWidget(self.mechanism)
        self.baseline = QtWidgets.QComboBox()
        for initial in tab.initial_condition_sets:
            self.baseline.addItem(initial.name, initial.set_id)
        # Start on the set the main window runs; the first when it is gone.
        self.baseline.setCurrentIndex(max(self.baseline.findData(selected_set_id), 0))
        header = QtWidgets.QFormLayout()
        header.addRow("Initial-condition baseline", self.baseline)
        layout.addLayout(header)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.inputs = QtWidgets.QTableWidget(0, 4)
        self.inputs.setHorizontalHeaderLabels(["Input", "Mode", "Value / Allowed Ranges", "Units"])
        self.inputs.setItemDelegateForColumn(0, ScientificNameDelegate(self.inputs))
        self.inputs.setItemDelegateForColumn(3, ScientificNameDelegate(self.inputs))
        self.inputs.setToolTip("Only independently editable inputs appear here. Calculated parameters keep the values their formulas give.")
        self.inputs.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.inputs.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.inputs, "Inputs")
        self._sampling_tab()
        self._constraints_tab()
        self._execution_tab()
        bottom = QtWidgets.QFormLayout()
        self.requested = entry(100000)
        bottom.addRow("Requested simulations", self.requested)
        path_row = QtWidgets.QWidget()
        path_layout = QtWidgets.QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.destination = QtWidgets.QLineEdit()
        self.browse = QtWidgets.QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        path_layout.addWidget(self.destination, 1)
        path_layout.addWidget(self.browse)
        bottom.addRow("JSON destination", path_row)
        layout.addLayout(bottom)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        layout.addWidget(self.progress_bar)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(155)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("Check Setup")
        self.start_button = QtWidgets.QPushButton("Start")
        self.cancel_button = QtWidgets.QPushButton("Cancel Remaining")
        self.close_button = QtWidgets.QPushButton("Close")
        self.check_button.clicked.connect(lambda: self._start(True))
        self.start_button.clicked.connect(lambda: self._start(False))
        self.cancel_button.clicked.connect(owner.cancel)
        self.close_button.clicked.connect(self.close)
        for button in (self.check_button, self.start_button, self.cancel_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.owner.progress.connect(self._progress)
        self.owner.completed.connect(self._completed)
        self.owner.application_close_settlement_changed.connect(self._busy_state)
        self.baseline.currentIndexChanged.connect(self._baseline_changed)
        self.group.currentIndexChanged.connect(self._counts)
        for edit in (self.requested, self.parameter_count, self.initial_count, self.boundary_count,
                     self.parameter_boundaries, self.initial_boundaries):
            edit.textChanged.connect(self._counts)
        self.boundaries.toggled.connect(self._counts)
        self._load()
        self._busy_state()

    def _sampling_tab(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.sampler = QtWidgets.QComboBox()
        self.sampler.addItem("Sobol", "sobol")
        self.sampler.addItem("Latin hypercube", "lhs")
        form.addRow("Sampling method", self.sampler)
        seed_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(seed_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.seed = entry(1729)
        reseed = QtWidgets.QPushButton("New Seed")
        reseed.clicked.connect(lambda: self.seed.setText(str(secrets.randbelow(2**32))))
        row.addWidget(self.seed)
        row.addWidget(reseed)
        form.addRow("Seed", seed_row)
        self.group = QtWidgets.QComboBox()
        self.group.addItem("Joint sampling", False)
        self.group.addItem("Parameter/time sets × reusable initial-condition sets", True)
        form.addRow("Exploration", self.group)
        self.parameter_count, self.initial_count = entry(1000), entry(100)
        form.addRow("Parameter/time sets", self.parameter_count)
        form.addRow("Initial-condition sets", self.initial_count)
        self.boundaries = QtWidgets.QCheckBox("Deliberately Sample Selected Included Endpoints")
        form.addRow(self.boundaries)
        self.boundary_count, self.parameter_boundaries, self.initial_boundaries = entry(0), entry(0), entry(0)
        form.addRow("Joint endpoint attempts", self.boundary_count)
        form.addRow("Parameter/time endpoint sets", self.parameter_boundaries)
        form.addRow("Initial-condition endpoint sets", self.initial_boundaries)
        self.count_summary = QtWidgets.QLabel()
        self.count_summary.setWordWrap(True)
        form.addRow(self.count_summary)
        note = QtWidgets.QLabel("Every attempt counts, including failures. No extra runs are generated to obtain successes.\n"
                                "Conditional sampling explores allowed intervals. It does not promise uniform coverage of a constrained space.")
        note.setWordWrap(True)
        form.addRow(note)
        self.tabs.addTab(page, "Sampling")

    def _constraints_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        note = QtWidgets.QLabel("Optional conditions on existing, input-only authored quantities. Select an independent input to calculate or sample conditionally.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.constraint_list = QtWidgets.QListWidget()
        self.constraint_list.setItemDelegate(ScientificNameDelegate(self.constraint_list))
        layout.addWidget(self.constraint_list)
        row = QtWidgets.QHBoxLayout()
        add, remove = QtWidgets.QPushButton("Add Constraint…"), QtWidgets.QPushButton("Remove Selected")
        add.clicked.connect(self._add_constraint)
        remove.clicked.connect(self._remove_constraint)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)
        self.tabs.addTab(page, "Constraints")

    def _execution_tab(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.time_form = form
        self.solver = QtWidgets.QComboBox()
        self.solver.addItems(SUPPORTED_SOLVERS)
        self.solver.setToolTip(SOLVER_TOOLTIP)
        self.points = entry(100)
        self.rtol, self.atol = entry(1e-6), entry(1e-12)
        self.jacobian = QtWidgets.QCheckBox("Use Analytic Jacobian")
        self.stops = QtWidgets.QCheckBox("Use Authored Stopping Interventions")
        show = QtWidgets.QPushButton("Show Authored Stopping Declarations")
        show.clicked.connect(self._show_stops)
        form.addRow("Solver", self.solver)
        form.addRow("Requested output points", self.points)
        form.addRow("Relative tolerance", self.rtol)
        form.addRow("Absolute tolerance", self.atol)
        form.addRow(self.jacobian)
        form.addRow(self.stops)
        form.addRow(show)
        note = QtWidgets.QLabel("With authored stops enabled, end time is the maximum endpoint. Otherwise execution uses the configured end time. Boundary rows may supplement the output count.")
        note.setWordWrap(True)
        form.addRow(note)
        self.outputs = QtWidgets.QListWidget()
        form.addRow("Exported quantities (time always included)", self.outputs)
        self.workers = QtWidgets.QSpinBox()
        capacity = resolve_run_parallel_capacity(cpu_count=os.cpu_count(), user_max=None, target_count=2**31 - 1)
        maximum = resolve_run_parallel_capacity(cpu_count=os.cpu_count(), user_max=2**31 - 1, target_count=2**31 - 1)
        self.workers.setRange(1, maximum)
        self.workers.setValue(capacity)
        form.addRow("Local workers", self.workers)
        self.tabs.addTab(page, "Execution and Output")

    def _baseline_config(self):
        tab = self.tab
        set_id = self.baseline.currentData()
        settings, science = tab.resolved_execution_settings(set_id), tab.scientific_settings
        # The values the mechanism resolves, including its T=, C0= and energy= lines.
        resolved = tab.prepared_dsl.analysis
        return ScanConfig(source=tab.mechanism_source.reactions_text, initials=dict(canonical_initial_values(tab, set_id)),
                          parameters={}, ranges={}, outputs=tuple(tab.prepared_dsl.analysis.species_names),
                          t_start=settings.t_start, t_end=settings.t_end, solver=settings.solver,
                          points=settings.integration_points, rtol=science.rtol, atol=science.atol,
                          analytic_jacobian=science.analytic_jacobian_policy, use_authored_stops=science.use_authored_stops,
                          requested=100000, destination="", energy_unit=resolved.energy_unit,
                          temperature_K=resolved.temperature_K, standard_conc_M=resolved.standard_conc_M)

    def _load(self):
        """Build the input rows once; a baseline change only refreshes its values."""
        self.base = self._baseline_config()
        self.catalog = input_catalog(self.tab.prepared_dsl, self.base)
        self.mechanism.setText(f"Mechanism: {self.tab.name}")
        for item in self.catalog:
            mode = QtWidgets.QComboBox()
            ignore_wheel(mode)
            mode.addItems(["Fixed", "Scan", "Calculated"])
            mode.model().item(2).setEnabled(False)
            value = entry(item.value)
            edit = QtWidgets.QPushButton("Edit Ranges…")
            edit.clicked.connect(lambda checked=False, key=item.key: self._edit_range(key))
            holder = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(value, 1)
            row.addWidget(edit)
            self.rows[item.key] = (mode, value, edit, holder)
            mode.currentIndexChanged.connect(lambda index, key=item.key: self._mode_changed(key, index))
            if item.key in (START, END):
                row.insertWidget(0, mode)
                caption = QtWidgets.QLabel(
                    f"{item.label} ({format_unit_for_display(item.unit)})"
                )
                caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
                caption.setAccessibleName(f"{item.label} ({item.unit})")
                self.time_form.insertRow(0 if item.key == START else 1, caption, holder)
            else:
                index = self.inputs.rowCount()
                self.inputs.insertRow(index)
                for column, text in ((0, item.label), (3, item.unit)):
                    cell = QtWidgets.QTableWidgetItem(text)
                    cell.setFlags(cell.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    if column == 0:
                        cell.setData(
                            SCIENTIFIC_NAME_HTML_ROLE,
                            scientific_symbol_html(item.label, family=item.family, role=item.role),
                        )
                        cell.setData(
                            QtCore.Qt.ItemDataRole.AccessibleTextRole,
                            scientific_symbol_spoken(item.label, family=item.family, role=item.role),
                        )
                    else:
                        cell.setData(SCIENTIFIC_NAME_HTML_ROLE, format_unit_for_display(item.unit))
                    self.inputs.setItem(index, column, cell)
                self.inputs.setCellWidget(index, 1, mode)
                self.inputs.setCellWidget(index, 2, holder)
        self.solver.setCurrentText(self.base.solver)
        self.points.setText(str(self.base.points))
        self.rtol.setText(repr(self.base.rtol))
        self.atol.setText(repr(self.base.atol))
        self.jacobian.setChecked(self.base.analytic_jacobian)
        self.stops.setChecked(self.base.use_authored_stops)
        self.outputs.clear()
        names = list(self.base.outputs) + sorted(self.tab.prepared_dsl.parameter_algebra_spec_for_execution().observable_names)
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if name in self.base.outputs else QtCore.Qt.CheckState.Unchecked)
            self.outputs.addItem(item)
        self._counts()

    def _baseline_changed(self):
        """Refresh only the values the baseline set supplies; keep the setup."""
        if not self.rows:
            self._load()
            return
        self.base = self._baseline_config()
        self.catalog = input_catalog(self.tab.prepared_dsl, self.base)
        for item in self.catalog:
            mode, value, _, _ = self.rows[item.key]
            if mode.currentIndex() == 0 and (
                item.key in (START, END) or item.key.startswith("initial:")
            ):
                value.setText(str(item.value))
        self.solver.setCurrentText(self.base.solver)
        self.points.setText(str(self.base.points))

    def _mode_changed(self, key, index):
        mode, value, edit, _ = self.rows[key]
        value.setReadOnly(index != 0)
        edit.setEnabled(index != 0)
        if index and key not in self.ranges:
            fixed = next(item.value for item in self.catalog if item.key == key)
            self.ranges[key] = InputRange((Interval(fixed, fixed),))
        if index:
            value.setText(self._range_label(self.ranges[key]))
        else:
            value.setText(str(next(item.value for item in self.catalog if item.key == key)))

    @staticmethod
    def _range_label(selection):
        text = ", ".join(f"{i.lower:g} to {i.upper:g}" for i in selection.intervals)
        return f"{text} (log)" if selection.log_scale else text

    def _edit_range(self, key):
        item = next(item for item in self.catalog if item.key == key)
        selection = self.ranges.get(key, InputRange((Interval(item.value, item.value),)))
        dialog = ScanRangeDialog(item.label, selection, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.ranges[key] = dialog.selection
            mode, value, _, _ = self.rows[key]
            if mode.currentIndex() == 0:
                mode.setCurrentIndex(1)
            value.setText(self._range_label(dialog.selection))

    def _add_constraint(self):
        spec = self.tab.prepared_dsl.parameter_algebra_spec_for_execution()
        inputs = [item for item in self.catalog if self.rows[item.key][0].currentIndex() != 0]
        dialog = ScanConstraintDialog(spec.observable_block.lines if spec.observable_block else (), inputs, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.constraints.append(dialog.constraint)
            self._render_constraints()

    def _remove_constraint(self):
        index = self.constraint_list.currentRow()
        if index >= 0:
            self.constraints.pop(index)
            self._render_constraints()

    def _render_constraints(self):
        self.constraint_list.clear()
        calculated = {c.dependent for c in self.constraints if c.condition == "equals"}
        statements = {stmt.name: stmt for stmt in self.tab.prepared_dsl.parameter_algebra_spec_for_execution().observable_block.lines}
        for condition in self.constraints:
            statement = statements[condition.observable]
            unit = presentation_unit(statement.expr.quantity_unit, statement.declared_unit)
            unit_label = "units unspecified" if unit is None else unit.symbol
            label = (f"= {condition.lower:g}" if condition.condition == "equals" else
                     f"{'[' if condition.include_lower else '('}{condition.lower:g}, {condition.upper:g}{']' if condition.include_upper else ')'}")
            text = f"{condition.observable} {label} [{unit_label}] — {condition.dependent}"
            row = QtWidgets.QListWidgetItem(text)
            row.setData(
                SCIENTIFIC_NAME_HTML_ROLE,
                f"{html.escape(condition.observable)} {label} [{format_unit_for_display(unit) if unit is not None else 'units unspecified'}] — {html.escape(condition.dependent)}",
            )
            self.constraint_list.addItem(row)
        for key, (mode, _, _, _) in self.rows.items():
            if key in calculated:
                mode.setCurrentIndex(2)
            elif mode.currentIndex() == 2:
                mode.setCurrentIndex(1)
            mode.setEnabled(key not in calculated)

    def _counts(self):
        grouped, boundary = bool(self.group.currentData()), self.boundaries.isChecked()
        for widget in (self.parameter_count, self.initial_count):
            widget.setEnabled(grouped)
        self.boundary_count.setEnabled(boundary and not grouped)
        for widget in (self.parameter_boundaries, self.initial_boundaries):
            widget.setEnabled(boundary and grouped)
        try:
            requested = _whole_number(self.requested, "Requested simulations")
            if grouped:
                p = _whole_number(self.parameter_count, "Parameter/time sets")
                c = _whole_number(self.initial_count, "Initial-condition sets")
                bp, bc = (
                    _whole_number(self.parameter_boundaries, "Parameter/time endpoint sets"),
                    _whole_number(self.initial_boundaries, "Initial-condition endpoint sets"),
                ) if boundary else (0, 0)
                self.count_summary.setText(f"{p} × {c} = {p*c} attempts. Requested: {requested}.\n"
                                           f"{p*c - (p-bp)*(c-bc)} combinations involve deliberately sampled endpoints.")
            else:
                budget = _whole_number(self.boundary_count, "Joint endpoint attempts") if boundary else 0
                self.count_summary.setText(f"{budget} endpoint attempts + {requested-budget} interior attempts = {requested}.")
        except ValueError:
            self.count_summary.setText("Enter whole-number counts.")

    def _configuration(self, check_only):
        base = self._baseline_config()
        initials, parameters, times, ranges = dict(base.initials), {}, {START: base.t_start, END: base.t_end}, {}
        for key, (mode, value, _, _) in self.rows.items():
            if mode.currentIndex():
                selection = self.ranges[key]
                ranges[key] = selection if self.boundaries.isChecked() else replace(selection, endpoints=())
            else:
                numeric = parse_finite_numeric_text(value.text().strip(), field=input_label(key))
                if key.startswith("initial:"):
                    initials[key.removeprefix("initial:")] = numeric
                elif key.startswith("parameter:"):
                    parameters[key.removeprefix("parameter:")] = numeric
                else:
                    times[key] = numeric
        selected = tuple(self.outputs.item(i).text() for i in range(self.outputs.count())
                         if self.outputs.item(i).checkState() == QtCore.Qt.CheckState.Checked)
        destination = self.destination.text().strip()
        if not check_only and not destination:
            raise ValueError("Choose a destination JSON file.")
        grouped, boundary = self.group.currentData(), self.boundaries.isChecked()
        rtol = validate_scipy_rtol(
            parse_finite_numeric_text(self.rtol.text().strip(), field="Relative tolerance"),
            field="Relative tolerance",
        )
        atol = parse_finite_numeric_text(self.atol.text().strip(), field="Absolute tolerance")
        if atol <= 0.0:
            raise ValueError("Absolute tolerance must be finite and positive.")
        return replace(base, initials=initials, parameters=parameters, ranges=ranges, outputs=selected,
                       t_start=times[START], t_end=times[END], solver=self.solver.currentText(),
                       points=parse_points_text(self.points.text(), field="Requested output points"),
                       rtol=rtol, atol=atol, analytic_jacobian=self.jacobian.isChecked(),
                       use_authored_stops=self.stops.isChecked(), requested=_whole_number(self.requested, "Requested simulations"),
                       destination=destination, seed=_whole_number(self.seed, "Seed"), sampler=self.sampler.currentData(),
                       grouped=bool(grouped),
                       parameter_sets=_whole_number(self.parameter_count, "Parameter/time sets") if grouped else 0,
                       initial_sets=_whole_number(self.initial_count, "Initial-condition sets") if grouped else 0,
                       boundary_attempts=_whole_number(self.boundary_count, "Joint endpoint attempts") if boundary and not grouped else 0,
                       parameter_boundaries=_whole_number(self.parameter_boundaries, "Parameter/time endpoint sets") if boundary and grouped else 0,
                       initial_boundaries=_whole_number(self.initial_boundaries, "Initial-condition endpoint sets") if boundary and grouped else 0,
                       workers=self.workers.value(), constraints=tuple(self.constraints))

    def _start(self, check_only):
        try:
            config = self._configuration(check_only)
            if not self.owner.start(config, check_only=check_only):
                raise ValueError("Another computation or application close is active.")
        except ValueError as exc:
            self.status.setPlainText(str(exc))
            return
        self.progress_bar.setValue(0)
        self.status.setPlainText("Checking setup…" if check_only else "Preparing scan…")

    def _busy_state(self):
        busy = self.owner.busy or self.owner.session_closing
        self.tabs.setEnabled(not busy)
        self.baseline.setEnabled(not busy)
        self.requested.setEnabled(not busy)
        self.destination.setEnabled(not busy)
        self.browse.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.check_button.setEnabled(not busy)
        self.cancel_button.setEnabled(self.owner.busy and not self.owner.session_closing)
        self.close_button.setEnabled(not self.owner.session_closing)

    def _progress(self, progress):
        finished = progress.succeeded + progress.failed + progress.cancelled
        self.progress_bar.setValue(int(1000 * finished / max(1, progress.requested)))
        self.status.setPlainText(f"{progress.phase}: attempted {progress.attempted} / {progress.requested}, running {progress.running}\n"
                                 f"Simulation successes: {progress.succeeded}, failures: {progress.failed}, canceled: {progress.cancelled}\n"
                                 f"Saved records: {progress.saved}, export errors: {progress.export_errors}\n"
                                 f"{progress.latest_failure}\n{progress.error}\n{progress.path}")

    def _completed(self, result):
        if isinstance(result, str):
            self.status.setPlainText(result)
        elif result is not None:
            self._progress(result)
            state = "canceled" if result.state == "cancelled" else result.state
            self.status.appendPlainText(f"Scan {state}. Not started: {result.requested-result.attempted}.")

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Chemical Space Scan", self.destination.text(), "JSON files (*.json)", options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.destination.setText(path if path.lower().endswith(".json") else path + ".json")

    def _show_stops(self):
        declarations = self.tab.prepared_dsl.authored_stop_declarations()
        QtWidgets.QMessageBox.information(self, "Authored Stopping Interventions", "\n\n".join(declarations) or "This mechanism has no stopping declarations.")

    def reject(self):
        self.close()

    def closeEvent(self, event):
        event.ignore()
        self.owner.request_close()
