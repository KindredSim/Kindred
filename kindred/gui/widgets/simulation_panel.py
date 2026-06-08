# kindred/gui/widgets/simulation_panel.py
"""Batch initial-conditions table + core simulation controls."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtWidgets

from kindred.core.simulator.solvers import normalize_solver_name
from kindred.gui.project_schema import (
    SIMULATION_NUM_POINTS_RANGE,
    SIMULATION_TEMPERATURE_K_RANGE,
)

# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
)
__all__ = ["SimulationPanel"]


class SimulationPanel(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        batch_model: BatchInitialConditionsTableModel,
        message_parent: QtWidgets.QWidget,
        initial_solver: str,
        on_add_batch_set: Callable[[], None],
        on_move_selected_batch_sets_up: Callable[[], None],
        on_move_selected_batch_sets_down: Callable[[], None],
        on_delete_selected_batch_sets: Callable[[], None],
        on_run_selected: Callable[[], None],
        on_stop: Callable[[], None],
        on_solver_method_changed: Callable[[str], None],
        on_solver_summary_refresh: Callable[[], None],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        batch_widget = QtWidgets.QWidget()
        batch_layout = QtWidgets.QVBoxLayout(batch_widget)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(4)

        self.batch_table = BatchInitialConditionsTableView(batch_widget)
        self.batch_table.setObjectName("batchTable")
        self.batch_table.setModel(batch_model)
        self.batch_table.pasteError.connect(
            lambda msg: QtWidgets.QMessageBox.warning(message_parent, "Paste Error", str(msg))
        )

        controls_row_widget = QtWidgets.QWidget(batch_widget)
        controls_row_widget.setObjectName("batchSolverControlsRow")
        controls_row = QtWidgets.QVBoxLayout(controls_row_widget)
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(4)

        controls_inputs_row = QtWidgets.QHBoxLayout()
        controls_inputs_row.setSpacing(6)

        solver_label = QtWidgets.QLabel("Solver:", controls_row_widget)
        t_end_label = QtWidgets.QLabel("t_end:", controls_row_widget)
        points_label = QtWidgets.QLabel("Points:", controls_row_widget)

        self.solver_method_combo = QtWidgets.QComboBox(controls_row_widget)
        self.solver_method_combo.addItems(["Radau", "BDF"])
        solver_name, _warning = normalize_solver_name(initial_solver)
        self.solver_method_combo.setCurrentText(solver_name)
        self.solver_method_combo.setMaximumWidth(100)
        self.solver_method_combo.currentTextChanged.connect(on_solver_method_changed)

        self.sim_time_spinbox = QtWidgets.QLineEdit(controls_row_widget)
        self.sim_time_spinbox.setText("10.0")
        self.sim_time_spinbox.setToolTip("Total simulation time t_end in seconds (free-form numeric text)")
        self.sim_time_spinbox.setMaximumWidth(140)
        self.sim_time_spinbox.textChanged.connect(lambda _v: on_solver_summary_refresh())

        self.num_points_spinbox = QtWidgets.QSpinBox(controls_row_widget)
        self.num_points_spinbox.setRange(*SIMULATION_NUM_POINTS_RANGE)
        self.num_points_spinbox.setValue(100)
        self.num_points_spinbox.setToolTip("Number of points in the simulation output")
        self.num_points_spinbox.setMaximumWidth(140)
        self.num_points_spinbox.valueChanged.connect(lambda _v: on_solver_summary_refresh())

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setObjectName("runSelectedButton")
        self.run_btn.setToolTip("Run kinetic simulation for the current run target (Ctrl+R or F5)")
        self.run_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.run_btn.clicked.connect(on_run_selected)

        controls_inputs_row.addWidget(solver_label)
        controls_inputs_row.addWidget(self.solver_method_combo)
        controls_inputs_row.addSpacing(6)
        controls_inputs_row.addWidget(t_end_label)
        controls_inputs_row.addWidget(self.sim_time_spinbox)
        controls_inputs_row.addWidget(QtWidgets.QLabel("s", controls_row_widget))
        controls_inputs_row.addSpacing(6)
        controls_inputs_row.addWidget(points_label)
        controls_inputs_row.addWidget(self.num_points_spinbox)
        controls_inputs_row.addSpacing(6)
        self.temperature_label = QtWidgets.QLabel("T:", controls_row_widget)
        self.temperature_label.setVisible(False)
        self.temperature_spinbox = QtWidgets.QDoubleSpinBox(controls_row_widget)
        self.temperature_spinbox.setRange(*SIMULATION_TEMPERATURE_K_RANGE)
        self.temperature_spinbox.setValue(298.15)
        self.temperature_spinbox.setDecimals(2)
        self.temperature_spinbox.setSuffix(" K")
        self.temperature_spinbox.setMaximumWidth(140)
        self.temperature_spinbox.setEnabled(False)
        self.temperature_spinbox.setVisible(False)
        controls_inputs_row.addWidget(self.temperature_label)
        controls_inputs_row.addWidget(self.temperature_spinbox)
        controls_inputs_row.addStretch(1)
        controls_row.addLayout(controls_inputs_row)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setToolTip("Stop running simulation (Esc)")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(on_stop)

        batch_layout.addWidget(controls_row_widget)

        self.solver_summary_label = QtWidgets.QLabel(batch_widget)
        self.solver_summary_label.setObjectName("solverSummaryLabel")
        self.solver_summary_label.setWordWrap(True)
        self.solver_summary_label.setVisible(False)
        batch_layout.addWidget(self.solver_summary_label)

        batch_row_controls = QtWidgets.QHBoxLayout()
        self.add_batch_set_btn = QtWidgets.QPushButton("Add Set")
        self.add_batch_set_btn.setObjectName("addBatchSetButton")
        self.add_batch_set_btn.setToolTip("Add a new initial-condition set")
        self.add_batch_set_btn.clicked.connect(on_add_batch_set)
        self.move_batch_up_btn = QtWidgets.QPushButton("Move Up")
        self.move_batch_up_btn.setObjectName("moveBatchSetUpButton")
        self.move_batch_up_btn.setToolTip("Move the active set up")
        self.move_batch_up_btn.clicked.connect(on_move_selected_batch_sets_up)
        self.move_batch_down_btn = QtWidgets.QPushButton("Move Down")
        self.move_batch_down_btn.setObjectName("moveBatchSetDownButton")
        self.move_batch_down_btn.setToolTip("Move the active set down")
        self.move_batch_down_btn.clicked.connect(on_move_selected_batch_sets_down)
        self.delete_batch_set_btn = QtWidgets.QPushButton("Delete Set(s)…")
        self.delete_batch_set_btn.setObjectName("deleteBatchSetButton")
        self.delete_batch_set_btn.setToolTip("Delete the selected set(s)")
        self.delete_batch_set_btn.clicked.connect(on_delete_selected_batch_sets)
        batch_row_controls.addWidget(self.add_batch_set_btn)
        batch_row_controls.addWidget(self.move_batch_up_btn)
        batch_row_controls.addWidget(self.move_batch_down_btn)
        batch_row_controls.addWidget(self.delete_batch_set_btn)
        batch_row_controls.addWidget(self.run_btn)
        batch_row_controls.addWidget(self.stop_btn)
        batch_row_controls.addStretch(1)
        batch_layout.addLayout(batch_row_controls)

        batch_layout.addWidget(self.batch_table, stretch=1)

        self.sim_progress = QtWidgets.QProgressBar(batch_widget)
        batch_layout.addWidget(self.sim_progress)

        layout.addWidget(batch_widget, stretch=1)
