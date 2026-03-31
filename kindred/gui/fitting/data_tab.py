"""
Data tab widget for the fitting window.

Extracted from FittingWindow — owns the dataset table (with Use/Dataset/Species
columns, Add/Remove buttons) and the sampling configuration panel.
Communicates with FittingWindow via signals (outbound events) and public
methods (inbound queries/commands).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter
from kindred.core.analysis.dataset_sampling import compute_windowed_indices
from kindred.gui.fitting.constants import _SAMPLING_ALL_POINTS_SENTINEL


class DataTab(QtWidgets.QWidget):
    """Drop-in replacement for FittingWindow._create_data_tab().

    Owns the dataset table and sampling configuration panel.  Read/write
    interaction with shared session state goes through callable getters
    (constructor-injected) and outbound signals.
    """

    # ------------------------------------------------------------------
    # Signals (DataTab -> FittingWindow)
    # ------------------------------------------------------------------
    datasetIncludeChanged = Signal(int, str, bool)    # row, ds_id, included
    addDatasetsRequested = Signal()                    # Add button clicked
    removeDatasetsRequested = Signal(list)             # Remove clicked; payload = [ds_id, ...]
    samplingApplied = Signal(str, dict)                # ds_id, config dict
    statusMessage = Signal(str)                        # status bar text

    def __init__(
        self,
        *,
        sampling_applied_config_getter: Callable[[str], Dict[str, float | int | str]],
        sampling_default_config_getter: Callable[[np.ndarray], Dict[str, float | int | str]],
        fit_targets_full_t_getter: Callable[[str], np.ndarray],
        fit_targets_available_getter: Callable[[str], List[str]],
        fit_targets_full_series_getter: Callable[[str], Dict[str, np.ndarray]],
        fit_targets_selection_applied_getter: Callable[[str], List[str]],
        modeled_series_getter: Callable[[], set[str]],
        worker_running_getter: Callable[[], bool],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # Store callable getters for shared state access.
        self._sampling_applied_config_getter = sampling_applied_config_getter
        self._sampling_default_config_getter = sampling_default_config_getter
        self._fit_targets_full_t_getter = fit_targets_full_t_getter
        self._fit_targets_available_getter = fit_targets_available_getter
        self._fit_targets_full_series_getter = fit_targets_full_series_getter
        self._fit_targets_selection_applied_getter = fit_targets_selection_applied_getter
        self._modeled_series_getter = modeled_series_getter
        self._worker_running_getter = worker_running_getter

        # Private UI state.
        self._sampling_current_dataset_id: Optional[str] = None
        self._sampling_is_refreshing = False

        # Build layout.
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # --- Dataset table group ---
        self._dataset_group = QtWidgets.QGroupBox("Datasets")
        dataset_layout = QtWidgets.QVBoxLayout(self._dataset_group)
        self._dataset_table = QtWidgets.QTableWidget()
        self._dataset_table.setColumnCount(3)
        self._dataset_table.setHorizontalHeaderLabels(["Use", "Dataset", "Species"])
        self._dataset_table.horizontalHeader().setStretchLastSection(True)
        self._dataset_table.setAlternatingRowColors(True)
        self._dataset_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._dataset_table.itemChanged.connect(self._on_dataset_table_item_changed)
        self._dataset_table.itemSelectionChanged.connect(self._on_selection_changed)
        self._dataset_table.itemSelectionChanged.connect(self._load_sampling_for_selected_dataset_row)
        dataset_layout.addWidget(self._dataset_table, stretch=1)

        dataset_buttons = QtWidgets.QHBoxLayout()
        self._dataset_add_button = QtWidgets.QPushButton("Add\u2026")
        self._dataset_add_button.setObjectName("global_fit_datasets_add")
        self._dataset_add_button.clicked.connect(self.addDatasetsRequested)
        self._dataset_remove_button = QtWidgets.QPushButton("Remove")
        self._dataset_remove_button.setObjectName("global_fit_datasets_remove")
        self._dataset_remove_button.setEnabled(False)
        self._dataset_remove_button.clicked.connect(self._on_remove_clicked)
        dataset_buttons.addWidget(self._dataset_add_button)
        dataset_buttons.addWidget(self._dataset_remove_button)
        dataset_buttons.addStretch(1)
        dataset_layout.addLayout(dataset_buttons)

        layout.addWidget(self._dataset_group, stretch=3)
        self._sampling_panel_widget = self._create_sampling_panel()
        layout.addWidget(self._sampling_panel_widget, stretch=2)

    # ------------------------------------------------------------------
    # Public API (FittingWindow -> DataTab)
    # ------------------------------------------------------------------

    def select_dataset(self, dataset_id: str) -> None:
        """Load sampling controls for the given dataset (public entry point)."""
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            self._load_sampling_controls_for_dataset("")
            return
        row = None
        for r in range(self._dataset_table.rowCount()):
            item = self._dataset_table.item(r, 0)
            if item is not None and str(item.data(Qt.UserRole) or "").strip() == ds_id:
                row = r
                break
        if row is None:
            return
        self._dataset_table.blockSignals(True)
        self._dataset_table.selectRow(row)
        self._dataset_table.blockSignals(False)
        self._load_sampling_controls_for_dataset(ds_id)

    def populate_table(self, entries: List[Dict[str, Any]]) -> None:
        """Fill the dataset table from the given entry list."""
        self._dataset_table.blockSignals(True)
        self._dataset_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check_item.setCheckState(Qt.Checked if entry.get("include", True) else Qt.Unchecked)
            check_item.setData(Qt.UserRole, entry["id"])
            self._dataset_table.setItem(row, 0, check_item)

            name_item = QtWidgets.QTableWidgetItem(entry["label"])
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._dataset_table.setItem(row, 1, name_item)

            species_text = ", ".join(entry["selected_species"])
            species_item = QtWidgets.QTableWidgetItem(species_text)
            species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._dataset_table.setItem(row, 2, species_item)
        self._dataset_table.blockSignals(False)

    def included_dataset_ids(self) -> List[str]:
        """Return IDs of datasets whose Use checkbox is checked."""
        included: List[str] = []
        for row in range(self._dataset_table.rowCount()):
            item = self._dataset_table.item(row, 0)
            if item is None:
                continue
            ds_id = str(item.data(Qt.UserRole) or "").strip()
            if not ds_id:
                continue
            if item.checkState() == Qt.Checked:
                included.append(ds_id)
        return included

    def dataset_label_for_id(self, dataset_id: str) -> str:
        """Return the human-readable label for a dataset ID from the table."""
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return "dataset"
        for row in range(self._dataset_table.rowCount()):
            item = self._dataset_table.item(row, 0)
            if item is None:
                continue
            if str(item.data(Qt.UserRole) or "").strip() == ds_id:
                name_item = self._dataset_table.item(row, 1)
                if name_item is not None:
                    label = str(name_item.text()).strip()
                    if label:
                        return label
                return ds_id
        return ds_id

    def selected_dataset_id(self) -> Optional[str]:
        """Return the ID of the first selected dataset row, or None."""
        table = self._dataset_table
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()}) if table.selectionModel() else []
        if not rows:
            return None
        item = table.item(int(rows[0]), 0)
        ds_id = str(item.data(Qt.UserRole) or "").strip() if item is not None else ""
        return ds_id or None

    def set_sampling_secondary_error(self, message: Optional[str]) -> None:
        """Set or clear the secondary error message on the sampling footer."""
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_secondary_error(message)

    def load_sampling_for_selected_row(self) -> None:
        """Load sampling controls for whichever dataset row is currently selected."""
        ds_id = self.selected_dataset_id()
        self._load_sampling_controls_for_dataset(ds_id)

    def refresh_remove_button_state(self) -> None:
        """Update the Remove button enabled state from live worker status."""
        self._update_remove_button()

    def sampling_validation_error(self, *, dataset_id: str, config: Dict[str, float | int | str]) -> Optional[str]:
        """Pure-computation validation of a sampling config. No UI side-effects."""
        return self._sampling_pending_validation_error(dataset_id=dataset_id, pending=config)

    # ------------------------------------------------------------------
    # Layout construction (moved from FittingWindow factories)
    # ------------------------------------------------------------------

    def _create_sampling_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setObjectName("global_fit_sampling_panel")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._sampling_header_label = QtWidgets.QLabel("Sampling (selected dataset: \u2014)")
        self._sampling_header_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._sampling_header_label)

        self._sampling_footer = ConfigPanelFooter(
            container,
            show_divider=False,
            show_reset=True,
            show_secondary_error=True,
            messages_position="after_body",
            apply_requires_no_error=True,
            button_order=("reset", "revert", "apply"),
            error_object_name="global_fit_sampling_error",
            secondary_error_object_name="global_fit_sampling_run_blocked",
            apply_object_name="global_fit_sampling_apply",
            revert_object_name="global_fit_sampling_revert",
            reset_object_name="global_fit_sampling_reset",
        )
        layout.addWidget(self._sampling_footer, stretch=1)
        self._sampling_footer.applyRequested.connect(self._apply_sampling_changes)
        self._sampling_footer.revertRequested.connect(self._revert_sampling_changes)
        self._sampling_footer.resetRequested.connect(self._reset_sampling_pending_to_defaults)

        # Stacked label-above-control layout for narrow (~240px) left panel.
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setSpacing(4)

        self._sampling_x_axis_combo = QtWidgets.QComboBox(container)
        self._sampling_x_axis_combo.setObjectName("global_fit_sampling_x_axis")
        self._sampling_x_axis_combo.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("X axis:"))
        controls_layout.addWidget(self._sampling_x_axis_combo)

        self._sampling_x_mode_combo = QtWidgets.QComboBox(container)
        self._sampling_x_mode_combo.setObjectName("global_fit_sampling_x_mode")
        self._sampling_x_mode_combo.setEnabled(False)
        self._sampling_x_mode_combo.setVisible(False)
        self._sampling_x_mode_combo.addItem("Auto", "auto")
        self._sampling_x_mode_combo.addItem("Monotone only (fast)", "monotone")
        self._sampling_x_mode_combo.addItem("Time-guided (non-monotone)", "time_guided")
        self._sampling_x_mode_label = QtWidgets.QLabel("X mapping:")
        self._sampling_x_mode_label.setVisible(False)
        controls_layout.addWidget(self._sampling_x_mode_label)
        controls_layout.addWidget(self._sampling_x_mode_combo)

        self._sampling_t_min_spin = QtWidgets.QDoubleSpinBox(container)
        self._sampling_t_min_spin.setObjectName("global_fit_sampling_t_min")
        self._sampling_t_min_spin.setDecimals(6)
        self._sampling_t_min_spin.setRange(-1e18, 1e18)
        self._sampling_t_min_spin.setSingleStep(1.0)
        self._sampling_t_min_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("t_min:"))
        controls_layout.addWidget(self._sampling_t_min_spin)

        self._sampling_t_max_spin = QtWidgets.QDoubleSpinBox(container)
        self._sampling_t_max_spin.setObjectName("global_fit_sampling_t_max")
        self._sampling_t_max_spin.setDecimals(6)
        self._sampling_t_max_spin.setRange(-1e18, 1e18)
        self._sampling_t_max_spin.setSingleStep(1.0)
        self._sampling_t_max_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("t_max:"))
        controls_layout.addWidget(self._sampling_t_max_spin)

        self._sampling_n_points_spin = QtWidgets.QSpinBox(container)
        self._sampling_n_points_spin.setObjectName("global_fit_sampling_n_points")
        self._sampling_n_points_spin.setRange(0, 0)
        self._sampling_n_points_spin.setSpecialValueText("All")
        self._sampling_n_points_spin.setValue(0)
        self._sampling_n_points_spin.setEnabled(False)
        controls_layout.addWidget(QtWidgets.QLabel("N:"))
        controls_layout.addWidget(self._sampling_n_points_spin)

        self._sampling_footer.body_layout.addLayout(controls_layout)

        self._sampling_used_label = QtWidgets.QLabel("Used: \u2014")
        self._sampling_used_label.setObjectName("global_fit_sampling_used_label")
        self._sampling_used_label.setStyleSheet("font-size: 11px;")
        self._sampling_footer.body_layout.addWidget(self._sampling_used_label)

        self._sampling_t_min_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_t_max_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_n_points_spin.valueChanged.connect(self._on_sampling_controls_changed)
        self._sampling_x_axis_combo.currentIndexChanged.connect(self._on_sampling_controls_changed)
        self._sampling_x_mode_combo.currentIndexChanged.connect(self._on_sampling_controls_changed)

        return container

    # ------------------------------------------------------------------
    # Internal event handlers (boundary — emit signals for FittingWindow)
    # ------------------------------------------------------------------

    def _on_dataset_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        try:
            col = int(item.column())
        except Exception:
            return
        if col == 0:
            row = int(item.row())
            ds_id_item = self._dataset_table.item(row, 0)
            ds_id = str(ds_id_item.data(Qt.UserRole) or "").strip() if ds_id_item is not None else ""
            included = item.checkState() == Qt.Checked
            self.datasetIncludeChanged.emit(row, ds_id, included)
            return

    def _on_selection_changed(self) -> None:
        self._update_remove_button()

    def _on_remove_clicked(self) -> None:
        rows = sorted({item.row() for item in self._dataset_table.selectedItems()})
        if not rows:
            return
        ids: List[str] = []
        for row in rows:
            item = self._dataset_table.item(int(row), 0)
            if item is None:
                continue
            ds_id = str(item.data(Qt.UserRole) or "").strip()
            if ds_id:
                ids.append(ds_id)
        if ids:
            self.removeDatasetsRequested.emit(ids)

    def _update_remove_button(self) -> None:
        if self._worker_running_getter():
            self._dataset_remove_button.setEnabled(False)
            return
        rows = {item.row() for item in self._dataset_table.selectedItems()}
        self._dataset_remove_button.setEnabled(bool(rows))

    # ------------------------------------------------------------------
    # Sampling — internal helpers (moved from FittingWindow)
    # ------------------------------------------------------------------

    def _load_sampling_for_selected_dataset_row(self) -> None:
        ds_id = self.selected_dataset_id()
        self._load_sampling_controls_for_dataset(ds_id)

    def _load_sampling_controls_for_dataset(self, dataset_id: Optional[str]) -> None:
        if not hasattr(self, "_sampling_t_min_spin"):
            return
        if not hasattr(self, "_sampling_footer"):
            return
        ds_id = str(dataset_id or "").strip()
        enabled = bool(ds_id)

        if hasattr(self, "_sampling_header_label") and self._sampling_header_label is not None:
            title = "Sampling (selected dataset: \u2014)" if not enabled else f"Sampling (selected dataset: {self.dataset_label_for_id(ds_id)})"
            self._sampling_header_label.setText(title)

        self._sampling_is_refreshing = True
        try:
            for widget in (
                self._sampling_x_axis_combo,
                self._sampling_x_mode_combo,
                self._sampling_t_min_spin,
                self._sampling_t_max_spin,
                self._sampling_n_points_spin,
            ):
                widget.setEnabled(enabled)
            if self._sampling_footer.reset_button is not None:
                self._sampling_footer.reset_button.setEnabled(enabled)
            if not enabled:
                self._sampling_current_dataset_id = None
                self._sampling_used_label.setText("Used: \u2014")
                self._sampling_footer.set_error(None)
                self._sampling_footer.set_secondary_error(None)
                self._sampling_footer.set_dirty(False)
                self._sampling_x_mode_combo.setVisible(False)
                self._sampling_x_mode_label.setVisible(False)
                try:
                    self._sampling_x_axis_combo.clear()
                    self._sampling_x_mode_combo.setCurrentIndex(0)
                except Exception:
                    return
                return

            full_t = self._fit_targets_full_t_getter(ds_id)
            t_axis = np.asarray(full_t, dtype=float).reshape(-1)
            if t_axis.size:
                t_min_full = float(np.min(t_axis))
                t_max_full = float(np.max(t_axis))
            else:
                t_min_full = 0.0
                t_max_full = 0.0

            cfg = self._sampling_applied_config_getter(ds_id)
            t_min_val = float(cfg.get("t_min", t_min_full))
            t_max_val = float(cfg.get("t_max", t_max_full))
            n_points = int(cfg.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
            x_name = str(cfg.get("x_name") or "t").strip() or "t"
            x_mode = str(cfg.get("x_mapping_mode") or "auto").strip().lower().replace("-", "_").replace(" ", "_") or "auto"
            if x_mode in ("monotone_only", "monotoneonly"):
                x_mode = "monotone"
            if x_mode not in ("auto", "monotone", "time_guided"):
                x_mode = "auto"

            observed = set(self._fit_targets_available_getter(ds_id))
            modeled = self._modeled_series_getter()
            candidates = sorted({name for name in observed & modeled if name})

            self._sampling_x_axis_combo.blockSignals(True)
            try:
                self._sampling_x_axis_combo.clear()
                self._sampling_x_axis_combo.addItem("Time (t)", "t")
                for name in candidates:
                    self._sampling_x_axis_combo.addItem(str(name), str(name))
                idx_x = int(self._sampling_x_axis_combo.findData(x_name))
                if idx_x < 0:
                    idx_x = 0
                self._sampling_x_axis_combo.setCurrentIndex(idx_x)
            finally:
                self._sampling_x_axis_combo.blockSignals(False)

            self._sampling_x_mode_combo.blockSignals(True)
            try:
                idx_mode = int(self._sampling_x_mode_combo.findData(x_mode))
                if idx_mode < 0:
                    idx_mode = 0
                self._sampling_x_mode_combo.setCurrentIndex(idx_mode)
            finally:
                self._sampling_x_mode_combo.blockSignals(False)
            show_mode = x_name != "t"
            self._sampling_x_mode_combo.setVisible(bool(show_mode))
            self._sampling_x_mode_label.setVisible(bool(show_mode))
            self._sampling_x_mode_combo.setEnabled(bool(show_mode))

            self._sampling_t_min_spin.setRange(t_min_full, t_max_full)
            self._sampling_t_max_spin.setRange(t_min_full, t_max_full)
            self._sampling_t_min_spin.setValue(max(t_min_full, min(t_max_full, t_min_val)))
            self._sampling_t_max_spin.setValue(max(t_min_full, min(t_max_full, t_max_val)))

            total = int(t_axis.size)
            self._sampling_n_points_spin.setRange(int(_SAMPLING_ALL_POINTS_SENTINEL), max(int(_SAMPLING_ALL_POINTS_SENTINEL), total))
            self._sampling_n_points_spin.setValue(max(int(_SAMPLING_ALL_POINTS_SENTINEL), min(total, n_points)))

            self._sampling_current_dataset_id = ds_id
            self._sampling_footer.set_error(None)
        finally:
            self._sampling_is_refreshing = False
        self._on_sampling_controls_changed()

    def _sampling_pending_values(self) -> Optional[Dict[str, float | int | str]]:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return None
        return {
            "t_min": float(self._sampling_t_min_spin.value()),
            "t_max": float(self._sampling_t_max_spin.value()),
            "n_points": int(self._sampling_n_points_spin.value()),
            "x_name": str(self._sampling_x_axis_combo.currentData() or "t").strip() or "t",
            "x_mapping_mode": str(self._sampling_x_mode_combo.currentData() or "auto").strip() or "auto",
        }

    def _sampling_pending_validation_error(self, *, dataset_id: str, pending: Dict[str, float | int | str]) -> Optional[str]:
        ds_id = str(dataset_id or "").strip()
        full_t = np.asarray(self._fit_targets_full_t_getter(ds_id), dtype=float).reshape(-1)
        t_min = float(pending.get("t_min", 0.0))
        t_max = float(pending.get("t_max", 0.0))
        n_points = int(pending.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
        if t_min > t_max:
            return "t_min must be \u2264 t_max."
        windowed = compute_windowed_indices(t=full_t, t_min=t_min, t_max=t_max)
        m = int(windowed.size)
        if m < 2:
            return "Sampling window must contain at least 2 points."
        if n_points != int(_SAMPLING_ALL_POINTS_SENTINEL):
            if n_points < 2:
                return "N must be All or \u2265 2."
            if n_points > m:
                return f"N must be \u2264 {m} (windowed points)."

        x_name = str(pending.get("x_name") or "t").strip() or "t"
        if x_name != "t":
            full_series = self._fit_targets_full_series_getter(ds_id) if ds_id else {}
            if not (isinstance(full_series, dict) and x_name in full_series):
                return f"X axis '{x_name}' is not available as an observed column for this dataset."
            modeled = self._modeled_series_getter()
            if modeled and x_name not in modeled:
                return f"X axis '{x_name}' is not a modeled series (species or algebra observable)."
            applied_targets = set(self._fit_targets_selection_applied_getter(ds_id) or [])
            if x_name in applied_targets:
                return f"X axis '{x_name}' cannot also be a fitted series. Remove it from Fit Targets or choose a different X."
        return None

    def _sampling_used_count_for_pending(self, *, dataset_id: str, pending: Dict[str, float | int | str]) -> Tuple[int, int]:
        ds_id = str(dataset_id or "").strip()
        full_t = np.asarray(self._fit_targets_full_t_getter(ds_id), dtype=float).reshape(-1)
        total = int(full_t.size)
        t_min = float(pending.get("t_min", 0.0))
        t_max = float(pending.get("t_max", 0.0))
        n_points = int(pending.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
        if t_min > t_max:
            return 0, total
        windowed = compute_windowed_indices(t=full_t, t_min=t_min, t_max=t_max)
        m = int(windowed.size)
        if m == 0:
            return 0, total
        if n_points == int(_SAMPLING_ALL_POINTS_SENTINEL) or n_points >= m:
            return m, total
        if 2 <= n_points <= m:
            return int(n_points), total
        return 0, total

    def _on_sampling_controls_changed(self) -> None:
        if self._sampling_is_refreshing:
            return
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        pending = self._sampling_pending_values()
        if not isinstance(pending, dict):
            return
        x_name = str(pending.get("x_name") or "t").strip() or "t"
        show_mode = x_name != "t"
        self._sampling_x_mode_combo.setVisible(bool(show_mode))
        self._sampling_x_mode_label.setVisible(bool(show_mode))
        self._sampling_x_mode_combo.setEnabled(bool(show_mode))
        used, total = self._sampling_used_count_for_pending(dataset_id=ds_id, pending=pending)
        self._sampling_used_label.setText(f"Used: {used}/{total} points")

        error = self._sampling_pending_validation_error(dataset_id=ds_id, pending=pending)
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_error(error)

        applied = self._sampling_applied_config_getter(ds_id)
        dirty = (
            (not math.isclose(float(applied.get("t_min", 0.0)), float(pending.get("t_min", 0.0)), rel_tol=1e-9, abs_tol=1e-12))
            or (not math.isclose(float(applied.get("t_max", 0.0)), float(pending.get("t_max", 0.0)), rel_tol=1e-9, abs_tol=1e-12))
            or int(applied.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL))) != int(pending.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
            or str(applied.get("x_name") or "t").strip() != str(pending.get("x_name") or "t").strip()
            or str(applied.get("x_mapping_mode") or "auto").strip() != str(pending.get("x_mapping_mode") or "auto").strip()
        )
        if hasattr(self, "_sampling_footer"):
            self._sampling_footer.set_dirty(bool(dirty))

    def _apply_sampling_changes(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        pending = self._sampling_pending_values()
        if not isinstance(pending, dict):
            return
        error = self._sampling_pending_validation_error(dataset_id=ds_id, pending=pending)
        if error:
            if hasattr(self, "_sampling_footer"):
                self._sampling_footer.set_error(error)
            return

        # ORDERING: samplingApplied must use a direct (synchronous) connection so
        # FittingWindow updates _sampling_applied before _on_sampling_controls_changed
        # reads it back via the getter. Do not use Qt.QueuedConnection.
        self.samplingApplied.emit(ds_id, dict(pending))
        self._on_sampling_controls_changed()

    def _revert_sampling_changes(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        self._load_sampling_controls_for_dataset(ds_id)

    def _reset_sampling_pending_to_defaults(self) -> None:
        ds_id = str(getattr(self, "_sampling_current_dataset_id", "") or "").strip()
        if not ds_id:
            return
        full_t = np.asarray(self._fit_targets_full_t_getter(ds_id), dtype=float).reshape(-1)
        defaults = self._sampling_default_config_getter(full_t)
        self._sampling_is_refreshing = True
        try:
            self._sampling_t_min_spin.setValue(float(defaults.get("t_min", 0.0)))
            self._sampling_t_max_spin.setValue(float(defaults.get("t_max", 0.0)))
            self._sampling_n_points_spin.setValue(int(defaults.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL))))
            idx_mode = int(self._sampling_x_mode_combo.findData("auto"))
            if idx_mode < 0:
                idx_mode = 0
            self._sampling_x_mode_combo.setCurrentIndex(idx_mode)
            idx = int(self._sampling_x_axis_combo.findData("t"))
            if idx < 0:
                idx = 0
            self._sampling_x_axis_combo.setCurrentIndex(idx)
        finally:
            self._sampling_is_refreshing = False
        self._on_sampling_controls_changed()
