"""Standalone Initial Conditions panel for the fitting window."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

logger = logging.getLogger(__name__)


class _ICCol:
    """Column indices for the initial conditions table."""
    FIT = 0
    LOG10 = 1
    SPECIES = 2
    INITIAL = 3
    MIN = 4
    MAX = 5


class InitialConditionsPanel(QtWidgets.QGroupBox):
    """Self-contained IC editor panel with signal-based boundary.

    Emits ``icApplied`` after persisting IC changes to the dataset manager.
    The parent is responsible for applying the updates to parameter state.
    """

    icApplied = Signal(str, dict, dict)
    icReverted = Signal()
    statusMessage = Signal(str)

    def __init__(
        self,
        *,
        dataset_entries: list,
        mechanism_species: list[str],
        dataset_manager_getter: Callable[[], Any],
        worker_running_getter: Callable[[], bool],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Initial Conditions", parent)
        self.setObjectName("global_fit_initial_conditions_panel")

        self._dataset_entries = list(dataset_entries)
        self._mechanism_species = list(mechanism_species)
        self._dataset_manager_getter = dataset_manager_getter
        self._worker_running_getter = worker_running_getter

        # IC editor state
        self._ic_editor_dirty = False
        self._ic_editor_current_dataset_id: Optional[str] = None
        self._ic_editor_is_refreshing = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self._ic_footer = ConfigPanelFooter(
            self,
            show_dirty=True,
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            apply_object_name="global_fit_initial_conditions_apply",
            revert_object_name="global_fit_initial_conditions_revert",
        )
        layout.addWidget(self._ic_footer, stretch=1)
        self._ic_footer.applyRequested.connect(self._apply_initial_conditions_changes)
        self._ic_footer.revertRequested.connect(self._revert_initial_conditions_changes)

        self._ic_dataset_combo = QtWidgets.QComboBox(self)
        self._ic_dataset_combo.setObjectName("global_fit_initial_conditions_dataset_combo")
        self._ic_footer.body_layout.addWidget(self._ic_dataset_combo)

        self._ic_table = QtWidgets.QTableWidget(self)
        self._ic_table.setObjectName("global_fit_initial_conditions_table")
        self._ic_table.setColumnCount(6)
        self._ic_table.setHorizontalHeaderLabels(["Fit", "Log10", "Species", "Initial", "Min", "Max"])
        _ih = self._ic_table.horizontalHeader()
        _ih.setStretchLastSection(False)
        for col in range(self._ic_table.columnCount()):
            _ih.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        _ih.setSectionResizeMode(_ICCol.SPECIES, QtWidgets.QHeaderView.Interactive)
        self._ic_table.setColumnWidth(_ICCol.SPECIES, 140)
        self._ic_table.verticalHeader().setVisible(False)
        self._ic_table.setAlternatingRowColors(True)
        self._ic_table.setMinimumHeight(200)
        self._ic_table.itemChanged.connect(self._on_ic_table_item_changed)
        self._ic_footer.body_layout.addWidget(self._ic_table, stretch=1)

        self._ic_dataset_combo.currentIndexChanged.connect(self._load_initial_conditions_for_current_dataset)
        self._refresh_combo_items()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_dataset_id(self) -> Optional[str]:
        return self._ic_editor_current_dataset_id

    def refresh_dataset_combo(self, dataset_entries: list) -> None:
        self._dataset_entries = list(dataset_entries)
        self._refresh_combo_items()

    def load_for_dataset(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        combo = self._ic_dataset_combo
        found = False
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                if str(combo.itemData(i) or "").strip() == ds_id:
                    combo.setCurrentIndex(i)
                    found = True
                    break
        finally:
            combo.blockSignals(False)
        if not found:
            return
        self._ic_editor_current_dataset_id = ds_id or None
        self._set_ic_editor_dirty_state(False)
        self._populate_initial_conditions_table(ds_id)

    def set_running_state(self, running: bool) -> None:
        if hasattr(self, "_ic_table"):
            self._ic_table.setEnabled(not running)
        if hasattr(self, "_ic_dataset_combo"):
            self._ic_dataset_combo.setEnabled(not running)
        if hasattr(self, "_ic_footer"):
            self._ic_footer.setEnabled(not running)

    def set_mechanism_species(self, species: list[str]) -> None:
        if species == self._mechanism_species:
            return
        self._mechanism_species = list(species)
        self._load_initial_conditions_for_current_dataset()

    def initial_parameter_defaults_for_species(self, dataset_id: str, species: str) -> tuple[bool, dict[str, float]]:
        return self._initial_parameter_defaults_for_species(dataset_id, species)

    # ------------------------------------------------------------------
    # IC editor — dirty state
    # ------------------------------------------------------------------

    def _set_ic_editor_dirty_state(self, dirty: bool) -> None:
        self._ic_editor_dirty = bool(dirty)
        if hasattr(self, "_ic_footer"):
            self._ic_footer.set_dirty(self._ic_editor_dirty)

    def _on_ic_table_item_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if self._ic_editor_is_refreshing:
            return
        self._set_ic_editor_dirty_state(True)

    # ------------------------------------------------------------------
    # IC editor — dataset loading
    # ------------------------------------------------------------------

    def _load_initial_conditions_for_current_dataset(self) -> None:
        if not hasattr(self, "_ic_dataset_combo"):
            return
        if self._ic_editor_dirty:
            self._set_ic_editor_dirty_state(False)
        ds_id = str(self._ic_dataset_combo.currentData() or "").strip()
        self._ic_editor_current_dataset_id = ds_id or None
        self._populate_initial_conditions_table(ds_id)

    def _populate_initial_conditions_table(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        self._ic_editor_is_refreshing = True
        try:
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(None)

            if not self._mechanism_species:
                self._ic_table.setRowCount(0)
                self._ic_table.setEnabled(False)
                return

            settings = None
            dataset_manager = self._dataset_manager_getter()
            if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings") and ds_id:
                try:
                    settings = dataset_manager.get_fit_settings(ds_id)
                except Exception:
                    settings = None

            initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
            fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
            log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
            bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

            self._ic_table.setEnabled(True)
            self._ic_table.setRowCount(len(self._mechanism_species))
            for row, species in enumerate(self._mechanism_species):
                init_val = float(initials.get(species, 0.0))
                fit_flag = bool(fit_flags.get(species, False))
                log10_flag = bool(log10_flags.get(species, False))
                bounds = bounds_map.get(species)
                if not bounds:
                    bounds = (0.0, max(10.0, init_val * 10 or 10.0))
                try:
                    min_val = float(bounds[0])
                    max_val = float(bounds[1])
                except Exception:
                    min_val, max_val = (0.0, max(10.0, init_val * 10 or 10.0))

                species_item = QtWidgets.QTableWidgetItem(str(species))
                species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                species_item.setTextAlignment(Qt.AlignCenter)
                self._ic_table.setItem(row, _ICCol.SPECIES, species_item)

                fit_item = QtWidgets.QTableWidgetItem()
                fit_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                fit_item.setCheckState(Qt.Checked if fit_flag else Qt.Unchecked)
                self._ic_table.setItem(row, _ICCol.FIT, fit_item)

                log_item = QtWidgets.QTableWidgetItem()
                log_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                log_item.setCheckState(Qt.Checked if log10_flag else Qt.Unchecked)
                self._ic_table.setItem(row, _ICCol.LOG10, log_item)

                init_item = QtWidgets.QTableWidgetItem(f"{init_val:.6g}")
                init_item.setTextAlignment(Qt.AlignCenter)
                self._ic_table.setItem(row, _ICCol.INITIAL, init_item)

                min_item = QtWidgets.QTableWidgetItem(f"{min_val:.6g}")
                min_item.setTextAlignment(Qt.AlignCenter)
                max_item = QtWidgets.QTableWidgetItem(f"{max_val:.6g}")
                max_item.setTextAlignment(Qt.AlignCenter)
                self._ic_table.setItem(row, _ICCol.MIN, min_item)
                self._ic_table.setItem(row, _ICCol.MAX, max_item)
        finally:
            self._ic_editor_is_refreshing = False

    # ------------------------------------------------------------------
    # IC editor — collect, apply, revert
    # ------------------------------------------------------------------

    def _collect_initial_conditions_from_table(
        self,
    ) -> Tuple[
        Optional[Dict[str, Dict[str, object]]],
        Optional[Dict[str, bool]],
        Optional[str],
    ]:
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        if not ds_id:
            return None, None, "No dataset selected."
        if not self._mechanism_species:
            return None, None, "No mechanism species available."
        updates: Dict[str, Dict[str, object]] = {}
        fit_flags_updates: Dict[str, bool] = {}
        for row, species in enumerate(self._mechanism_species):
            init_item = self._ic_table.item(row, _ICCol.INITIAL)
            fit_item = self._ic_table.item(row, _ICCol.FIT)
            log_item = self._ic_table.item(row, _ICCol.LOG10)
            min_item = self._ic_table.item(row, _ICCol.MIN)
            max_item = self._ic_table.item(row, _ICCol.MAX)
            try:
                init_val = float(init_item.text())
            except Exception:
                return (
                    None,
                    None,
                    f"Species '{species}' requires a numeric initial concentration.",
                )
            fit_flag = bool(fit_item and fit_item.checkState() == Qt.Checked)
            log10_flag = bool(log_item and log_item.checkState() == Qt.Checked)
            try:
                min_val = float(min_item.text())
                max_val = float(max_item.text())
            except Exception:
                return None, None, f"Species '{species}' requires numeric bounds."
            if fit_flag and not (min_val < max_val):
                return None, None, f"Species '{species}' bounds must satisfy min < max."
            if fit_flag and log10_flag:
                if not (init_val > 0.0 and min_val > 0.0 and max_val > 0.0):
                    return (
                        None,
                        None,
                        f"Species '{species}' requires initial/min/max > 0 when Log10 is enabled.",
                    )
            updates[str(species)] = {
                "initial": float(init_val),
                "log10": bool(log10_flag),
                "min": float(min_val),
                "max": float(max_val),
            }
            fit_flags_updates[str(species)] = bool(fit_flag)
        return updates, fit_flags_updates, None

    def _apply_initial_conditions_changes(self) -> None:
        updates, fit_flags_updates, error = self._collect_initial_conditions_from_table()
        if error:
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(str(error))
            return
        assert updates is not None
        assert fit_flags_updates is not None
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        if not ds_id:
            return
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is None or not hasattr(dataset_manager, "get_fit_settings") or not hasattr(dataset_manager, "update_fit_settings"):
            message = "Dataset manager unavailable; cannot persist Initial Conditions."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return
        try:
            settings = dataset_manager.get_fit_settings(ds_id)
        except Exception:
            message = f"Failed to load fit settings for dataset {ds_id}."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return

        initials = dict(getattr(settings, "initial_conditions", {}) or {})
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {})
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {})
        bounds_map = dict(getattr(settings, "bounds", {}) or {})
        for species, spec in updates.items():
            species_key = str(species)
            initials[str(species)] = float(spec["initial"])
            fit_flags[species_key] = bool(fit_flags_updates.get(species_key, False))
            log10_flags[str(species)] = bool(spec["log10"])
            bounds_map[str(species)] = (float(spec["min"]), float(spec["max"]))

        settings.initial_conditions = initials
        settings.fit_flags = fit_flags
        settings.log10_flags = log10_flags
        settings.bounds = bounds_map
        try:
            dataset_manager.update_fit_settings(ds_id, settings)
        except Exception:
            message = f"Failed to persist fit settings for dataset {ds_id}."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return

        self.icApplied.emit(ds_id, updates, fit_flags_updates)
        self._set_ic_editor_dirty_state(False)
        self.statusMessage.emit("Initial conditions applied")

    def _revert_initial_conditions_changes(self) -> None:
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        self._set_ic_editor_dirty_state(False)
        self._populate_initial_conditions_table(ds_id)
        self.icReverted.emit()

    # ------------------------------------------------------------------
    # IC editor — combo management
    # ------------------------------------------------------------------

    def _refresh_combo_items(self) -> None:
        if not hasattr(self, "_ic_dataset_combo"):
            return
        combo = self._ic_dataset_combo
        current = str(combo.currentData() or "").strip()
        combo.blockSignals(True)
        try:
            combo.clear()
            for entry in self._dataset_entries:
                ds_id = str(entry.get("id") or "").strip()
                if not ds_id:
                    continue
                label = str(entry.get("label") or ds_id)
                combo.addItem(label, ds_id)
        finally:
            combo.blockSignals(False)
        if current:
            for i in range(combo.count()):
                if str(combo.itemData(i) or "").strip() == current:
                    combo.setCurrentIndex(i)
                    break
        if combo.count() and combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        self._load_initial_conditions_for_current_dataset()

    # ------------------------------------------------------------------
    # IC defaults helper
    # ------------------------------------------------------------------

    def _initial_parameter_defaults_for_species(self, dataset_id: str, species: str) -> tuple[bool, dict[str, float]]:
        settings = None
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings"):
            try:
                settings = dataset_manager.get_fit_settings(str(dataset_id))
            except Exception:
                settings = None
        initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
        bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

        init_val = float(initials.get(species, 0.0))
        fit_flag = bool(fit_flags.get(species, False))
        log10_flag = bool(log10_flags.get(species, False))
        bounds = bounds_map.get(species)
        if not bounds:
            bounds = (0.0, max(10.0, init_val * 10 or 10.0))
        try:
            min_val = float(bounds[0])
            max_val = float(bounds[1])
        except Exception:
            min_val = 0.0
            max_val = max(10.0, init_val * 10 or 10.0)
        return fit_flag, {
            "initial": float(init_val),
            "min": float(min_val),
            "max": float(max_val),
            "log10": bool(log10_flag),
        }
