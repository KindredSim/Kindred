"""Unified species table widget for the fitting window.

Provides a single 8-column table that combines target selection, target
weights, and initial-condition editing for all mechanism species per dataset.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPalette

from kindred.gui.ui_helpers import setup_scientific_validator
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

logger = logging.getLogger(__name__)


class _Col:
    """Column indices for the unified species table."""
    INCLUDE = 0
    WEIGHT = 1
    SPECIES = 2
    INITIAL = 3
    FIT_IC = 4
    LOG10 = 5
    MIN = 6
    MAX = 7

    COUNT = 8
    HEADERS = ["Include in Fit", "Weight", "Species", "Initial Value",
               "Fit IC", "Log10", "Min", "Max"]


_INVALID_BG = QBrush(QColor(255, 200, 200))
_INVALID_FG = QBrush(QColor(80, 0, 0))
_DEFAULT_BG = QBrush()
_DEFAULT_FG = QBrush()


class _ValidationCellDelegate(QtWidgets.QStyledItemDelegate):
    """Paints cell text using ForegroundRole color, bypassing stylesheet overrides."""

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        fg_variant = index.data(Qt.ForegroundRole)
        if isinstance(fg_variant, QBrush) and fg_variant.style() != Qt.BrushStyle.NoBrush:
            text = opt.text
            opt.text = ""
            style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
            style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
            painter.save()
            if opt.state & QtWidgets.QStyle.StateFlag.State_Selected:
                painter.setPen(opt.palette.color(QPalette.ColorRole.HighlightedText))
            else:
                painter.setPen(fg_variant.color())
            text_rect = style.subElementRect(QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, option.widget)
            alignment = opt.displayAlignment
            metrics = QFontMetrics(opt.font)
            elided = metrics.elidedText(text, Qt.ElideRight, text_rect.width())
            painter.drawText(text_rect, alignment, elided)
            painter.restore()
        else:
            super().paint(painter, option, index)


class UnifiedSpeciesTable(QtWidgets.QWidget):
    """Combined target-selection, weight, and IC editor for the fitting window."""

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    targetsApplied = Signal()
    validityChanged = Signal()
    icApplied = Signal(str, dict, dict)
    statusMessage = Signal(str)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        dataset_entries: List[Dict[str, Any]],
        mechanism_species: list[str],
        dataset_entries_getter: Callable[[], List[Dict[str, Any]]],
        included_dataset_ids_getter: Callable[[], List[str]],
        dataset_label_getter: Callable[[str], str],
        dataset_weight_getter: Callable[[str], float],
        persist_dataset_weight_callback: Callable[[str, float], None],
        dataset_manager_getter: Callable[[], Any],
        worker_running_getter: Callable[[], bool],
        modeled_series_getter: Optional[Callable[[], set[str]]] = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._dataset_entries_getter = dataset_entries_getter
        self._included_dataset_ids_getter = included_dataset_ids_getter
        self._dataset_label_getter = dataset_label_getter
        self._dataset_weight_getter = dataset_weight_getter
        self._persist_dataset_weight_callback = persist_dataset_weight_callback
        self._dataset_manager_getter = dataset_manager_getter
        self._worker_running_getter = worker_running_getter

        self._mechanism_species: list[str] = list(mechanism_species)
        self._dataset_entries: list = list(dataset_entries)
        self._modeled_series_getter = modeled_series_getter or (
            lambda: {str(species) for species in self._mechanism_species if str(species).strip()}
        )

        self._init_fit_targets_state(dataset_entries)

        # If no mechanism species provided, derive from raw data columns so the
        # table is always populated when datasets have species_data.
        if not self._mechanism_species:
            all_columns: set[str] = set()
            for series_map in self._fit_targets_full_series_by_dataset.values():
                all_columns.update(series_map.keys())
            self._mechanism_species = sorted(all_columns)

        self._current_row_species: list[str] = []
        self._ic_pending: Dict[str, Dict[str, dict[str, object]]] = {}
        self._ic_applied: Dict[str, Dict[str, dict[str, object]]] = {}
        self._ic_error_text: Optional[str] = None
        self._seed_all_ic_state_from_dataset_manager()

        self._ic_editor_dirty = False
        self._ic_editor_is_refreshing = False
        self._current_dataset_id: Optional[str] = None
        self._is_refreshing = False
        self._dataset_weight_is_refreshing = False
        self._cached_modeled_series: frozenset = frozenset()
        self._fit_universe_initialized = False

        self._build_ui()

    # ------------------------------------------------------------------
    # Fit-universe helpers
    # ------------------------------------------------------------------
    def _safe_modeled_series(self) -> set[str]:
        """Return the modeled species set with defensive error handling."""
        try:
            modeled = self._modeled_series_getter()
            result = {str(s) for s in (modeled or set()) if str(s).strip()}
            if result:
                return result
        except Exception:
            pass
        return {str(s) for s in self._mechanism_species if str(s).strip()}

    def _recompute_fit_universe(self) -> None:
        """Recompute available_by_dataset as observed AND modeled for ALL datasets.

        Also prunes pending/applied selections and weights to the new fit-universe.
        Emits targetsApplied if any applied selection was actually pruned so that
        FittingWindow rebuilds its run payloads.
        """
        modeled = self._safe_modeled_series()
        applied_changed = False
        for ds_id in list(self._fit_targets_full_series_by_dataset.keys()):
            all_columns = sorted(self._fit_targets_full_series_by_dataset[ds_id].keys())
            fit_universe = [s for s in all_columns if s in modeled]
            fit_universe_set = set(fit_universe)
            self._fit_targets_available_by_dataset[ds_id] = fit_universe

            if ds_id in self._fit_targets_selection_pending:
                self._fit_targets_selection_pending[ds_id] &= fit_universe_set
            if ds_id in self._fit_targets_selection_applied:
                old_applied = self._fit_targets_selection_applied[ds_id]
                pruned = [s for s in old_applied if s in fit_universe_set]
                if len(pruned) != len(old_applied):
                    applied_changed = True
                self._fit_targets_selection_applied[ds_id] = pruned
            if ds_id in self._fit_target_weights_pending:
                self._fit_target_weights_pending[ds_id] = {
                    k: v for k, v in self._fit_target_weights_pending[ds_id].items()
                    if k in fit_universe_set
                }
            if ds_id in self._fit_target_weights_applied:
                applied_set = set(self._fit_targets_selection_applied.get(ds_id, []))
                self._fit_target_weights_applied[ds_id] = {
                    k: v for k, v in self._fit_target_weights_applied[ds_id].items()
                    if k in applied_set
                }
            if ds_id in self._fit_target_weights_pending_invalid:
                self._fit_target_weights_pending_invalid[ds_id] = {
                    k: v for k, v in self._fit_target_weights_pending_invalid[ds_id].items()
                    if k in fit_universe_set
                }

        if applied_changed:
            self.targetsApplied.emit()

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def _init_fit_targets_state(self, dataset_entries: List[Dict[str, Any]]) -> None:
        full_series: Dict[str, Dict[str, np.ndarray]] = {}
        full_t: Dict[str, np.ndarray] = {}
        available_by_dataset: Dict[str, List[str]] = {}
        applied: Dict[str, List[str]] = {}
        applied_target_weights: Dict[str, Dict[str, float]] = {}

        for entry in dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            t_values = np.asarray(entry.get("t", []), dtype=float).reshape(-1)
            if t_values.size == 0:
                continue
            raw_series = entry.get("species_data") or entry.get("species") or {}
            series_map: Dict[str, np.ndarray] = {}
            parse_failures: List[str] = []
            if isinstance(raw_series, dict):
                for name, values in raw_series.items():
                    key = str(name).strip()
                    if not key:
                        continue
                    try:
                        arr = np.asarray(values, dtype=float).reshape(-1)
                    except Exception as exc:
                        parse_failures.append(key)
                        if len(parse_failures) <= 3:
                            logger.debug(
                                "Skipping invalid fit-target series '%s' for dataset '%s': %s",
                                key, ds_id, exc, exc_info=True,
                            )
                        continue
                    if arr.size != t_values.size:
                        continue
                    series_map[key] = arr

            full_series[ds_id] = series_map
            full_t[ds_id] = t_values
            available = sorted(series_map.keys())
            available_by_dataset[ds_id] = available

            initial_selection = [
                str(x)
                for x in (entry.get("selected_species") or [])
                if str(x).strip() and str(x) in series_map
            ]
            applied[ds_id] = list(initial_selection)
            applied_target_weights[ds_id] = self._normalize_fit_target_weights_for_selection(
                ds_id, initial_selection, entry.get("target_weights"),
            )

        self._fit_targets_full_series_by_dataset = full_series
        self._fit_targets_full_t_by_dataset = full_t
        self._fit_targets_available_by_dataset = available_by_dataset
        self._fit_targets_selection_applied = applied
        self._fit_targets_selection_pending = {ds: set(names) for ds, names in applied.items()}
        self._fit_target_weights_applied = {ds: dict(weights) for ds, weights in applied_target_weights.items()}
        self._fit_target_weights_pending = {ds: dict(weights) for ds, weights in applied_target_weights.items()}
        self._fit_target_weights_pending_invalid: Dict[str, Dict[str, str]] = {}
        self._fit_targets_dirty = False

    def _seed_all_ic_state_from_dataset_manager(self) -> None:
        self._ic_pending = {}
        self._ic_applied = {}
        for ds_id in self._fit_targets_available_by_dataset.keys():
            self._seed_ic_state_for_dataset(ds_id)

    @staticmethod
    def _copy_ic_dataset_state(state: Optional[Dict[str, dict[str, object]]]) -> Dict[str, dict[str, object]]:
        copied: Dict[str, dict[str, object]] = {}
        for species, spec in (state or {}).items():
            copied[str(species)] = {
                "initial": float(spec.get("initial", 0.0)),
                "fit": bool(spec.get("fit", False)),
                "log10": bool(spec.get("log10", False)),
                "min": float(spec.get("min", 0.0)),
                "max": float(spec.get("max", 10.0)),
            }
        return copied

    @staticmethod
    def _ic_entry_from_settings_maps(
        species: str,
        *,
        initials: Dict[str, object],
        fit_flags: Dict[str, object],
        log10_flags: Dict[str, object],
        bounds_map: Dict[str, object],
    ) -> dict[str, object]:
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
        return {
            "initial": float(init_val),
            "fit": bool(fit_flag),
            "log10": bool(log10_flag),
            "min": float(min_val),
            "max": float(max_val),
        }

    def _seed_ic_state_for_dataset(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        settings = None
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings"):
            try:
                settings = dataset_manager.get_fit_settings(ds_id)
            except Exception:
                settings = None
        initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
        bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

        seeded = {
            str(species): self._ic_entry_from_settings_maps(
                str(species),
                initials=initials,
                fit_flags=fit_flags,
                log10_flags=log10_flags,
                bounds_map=bounds_map,
            )
            for species in self._mechanism_species
            if str(species).strip()
        }
        self._ic_applied[ds_id] = self._copy_ic_dataset_state(seeded)
        self._ic_pending[ds_id] = self._copy_ic_dataset_state(seeded)

    def _pending_ic_state_for_dataset(self, dataset_id: str) -> Dict[str, dict[str, object]]:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return {}
        state = self._ic_pending.get(ds_id)
        if state is None:
            self._seed_ic_state_for_dataset(ds_id)
            state = self._ic_pending.get(ds_id, {})
        return state

    def _displayed_row_species_for_dataset(self, dataset_id: str) -> list[str]:
        ds_id = str(dataset_id or "").strip()
        mechanism_species = [str(species) for species in self._mechanism_species if str(species).strip()]
        mechanism_set = set(mechanism_species)
        try:
            modeled = {
                str(species)
                for species in (self._modeled_series_getter() or set())
                if str(species).strip()
            }
        except Exception:
            modeled = set(mechanism_species)
        observed_only = sorted(
            {
                str(species)
                for species in self._fit_targets_available_by_dataset.get(ds_id, [])
                if str(species).strip()
                and str(species) not in mechanism_set
                and str(species) in modeled
            }
        )
        return mechanism_species + observed_only

    def _clear_ic_error(self) -> None:
        self._ic_error_text = None
        if hasattr(self, "_footer"):
            self._footer.set_error(None)

    def _clear_table_for_no_dataset(self) -> None:
        self._current_row_species = []
        if hasattr(self, "_table"):
            self._table.setRowCount(0)
        if hasattr(self, "_context_label"):
            self._context_label.setText("No dataset selected")
        if hasattr(self, "_dataset_weight_edit"):
            self._dataset_weight_edit.clear()
            self._dataset_weight_edit.setEnabled(False)
        self._clear_ic_error()
        self._ic_editor_dirty = False
        self._refresh_dataset_weight_editor_state()

    def _sync_visible_include_states(self) -> None:
        ds_id = str(self._current_dataset_id or "").strip()
        if not ds_id or not hasattr(self, "_table"):
            return
        available = set(self._fit_targets_available_by_dataset.get(ds_id, []))
        pending_sel = self._fit_targets_selection_pending.get(ds_id, set())
        self._is_refreshing = True
        try:
            for row, species in enumerate(self._current_row_species):
                include_item = self._table.item(row, _Col.INCLUDE)
                if include_item is None or species not in available:
                    continue
                include_item.setCheckState(Qt.Checked if species in pending_sel else Qt.Unchecked)
        finally:
            self._is_refreshing = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_fit_target_weight(value: object) -> bool:
        try:
            numeric = float(value)
        except Exception:
            return False
        return bool(np.isfinite(numeric) and numeric > 0.0)

    # ------------------------------------------------------------------
    # State tier helpers
    # ------------------------------------------------------------------
    def _normalize_fit_target_weights_for_selection(
        self, dataset_id: str, selection: Sequence[str], raw_weights: object,
    ) -> Dict[str, float]:
        weights_map = dict(raw_weights) if isinstance(raw_weights, dict) else {}
        normalized: Dict[str, float] = {}
        for name in [str(x) for x in (selection or []) if str(x).strip()]:
            value = weights_map.get(name, 1.0)
            normalized[name] = float(value) if self._is_valid_fit_target_weight(value) else 1.0
        return normalized

    def _pending_fit_target_weights_for_dataset(self, dataset_id: str) -> Dict[str, float]:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return {}
        weights = self._fit_target_weights_pending.setdefault(ds_id, {})
        if not isinstance(weights, dict):
            weights = {}
            self._fit_target_weights_pending[ds_id] = weights
        return weights

    def _pending_fit_target_weight_invalid_text(self, dataset_id: str, target_name: str) -> Optional[str]:
        invalid_map = self._fit_target_weights_pending_invalid.get(str(dataset_id or "").strip(), {})
        if not isinstance(invalid_map, dict):
            return None
        text = invalid_map.get(str(target_name or "").strip())
        return str(text) if text is not None else None

    def _pending_fit_target_weight_value(self, dataset_id: str, target_name: str) -> float:
        ds_id = str(dataset_id or "").strip()
        name = str(target_name or "").strip()
        if not ds_id or not name:
            return 1.0
        weights = self._pending_fit_target_weights_for_dataset(ds_id)
        value = weights.get(name, 1.0)
        if not self._is_valid_fit_target_weight(value):
            value = 1.0
        weights[name] = float(value)
        return float(value)

    def _pending_fit_target_weight_text(self, dataset_id: str, target_name: str) -> str:
        invalid_text = self._pending_fit_target_weight_invalid_text(dataset_id, target_name)
        if invalid_text is not None:
            return invalid_text
        return f"{self._pending_fit_target_weight_value(dataset_id, target_name):.6g}"

    def _set_pending_fit_target_weight_text(self, dataset_id: str, target_name: str, text: str) -> None:
        ds_id = str(dataset_id or "").strip()
        name = str(target_name or "").strip()
        if not ds_id or not name:
            return
        weights = self._pending_fit_target_weights_for_dataset(ds_id)
        invalid_map = self._fit_target_weights_pending_invalid.setdefault(ds_id, {})
        raw_text = str(text or "").strip()
        if not raw_text:
            invalid_map[name] = raw_text
        else:
            try:
                value = float(raw_text)
            except Exception:
                invalid_map[name] = raw_text
            else:
                if self._is_valid_fit_target_weight(value):
                    weights[name] = float(value)
                    invalid_map.pop(name, None)
                else:
                    invalid_map[name] = raw_text
        if not invalid_map:
            self._fit_target_weights_pending_invalid.pop(ds_id, None)

    def _fit_target_weight_is_pending_invalid(self, dataset_id: str, target_name: str) -> bool:
        invalid_map = self._fit_target_weights_pending_invalid.get(str(dataset_id or "").strip(), {})
        return bool(isinstance(invalid_map, dict) and str(target_name or "").strip() in invalid_map)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox("Species Table")
        group.setObjectName("global_fit_unified_species_group")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._footer = ConfigPanelFooter(
            group,
            show_dirty=True,
            show_secondary_error=True,
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            error_object_name="global_fit_species_table_error",
            secondary_error_object_name="global_fit_species_table_run_blocked",
            apply_object_name="global_fit_species_table_apply",
            revert_object_name="global_fit_species_table_revert",
        )
        group_layout.addWidget(self._footer, stretch=1)
        self._footer.applyRequested.connect(self._apply_changes)
        self._footer.revertRequested.connect(self._revert_changes)

        self._context_label = QtWidgets.QLabel("Selected dataset: \u2014", group)
        self._context_label.setStyleSheet("font-weight: bold;")
        self._footer.body_layout.addWidget(self._context_label)

        # Dataset weight controls
        weighting_form = QtWidgets.QFormLayout()
        weighting_form.setContentsMargins(0, 0, 0, 0)

        self._weight_mode_combo = QtWidgets.QComboBox(group)
        self._weight_mode_combo.setObjectName("global_fit_weight_mode_combo")
        self._weight_mode_combo.addItems([
            "Implicit weights (1/N per dataset)",
            "User weights only",
        ])
        weighting_form.addRow("Weight mode:", self._weight_mode_combo)

        self._dataset_weight_edit = QtWidgets.QLineEdit(group)
        self._dataset_weight_edit.setObjectName("global_fit_dataset_weight_edit")
        setup_scientific_validator(self._dataset_weight_edit)
        weighting_form.addRow("Dataset weight:", self._dataset_weight_edit)
        self._footer.body_layout.addLayout(weighting_form)

        self._weight_mode_combo.currentIndexChanged.connect(self._on_weight_mode_changed)
        self._dataset_weight_edit.editingFinished.connect(self._commit_selected_dataset_weight_edit)

        # Bulk buttons
        bulk_row = QtWidgets.QHBoxLayout()
        bulk_label = QtWidgets.QLabel("Bulk:")
        bulk_label.setStyleSheet("font-size: 11px;")
        bulk_row.addWidget(bulk_label)
        self._bulk_all_button = QtWidgets.QPushButton("All", group)
        self._bulk_all_button.setObjectName("global_fit_fit_targets_bulk_all")
        self._bulk_all_button.clicked.connect(lambda: self._apply_bulk_action("all"))
        self._bulk_none_button = QtWidgets.QPushButton("None", group)
        self._bulk_none_button.setObjectName("global_fit_fit_targets_bulk_none")
        self._bulk_none_button.clicked.connect(lambda: self._apply_bulk_action("none"))
        self._bulk_invert_button = QtWidgets.QPushButton("Invert", group)
        self._bulk_invert_button.setObjectName("global_fit_fit_targets_bulk_invert")
        self._bulk_invert_button.clicked.connect(lambda: self._apply_bulk_action("invert"))
        bulk_row.addWidget(self._bulk_all_button)
        bulk_row.addWidget(self._bulk_none_button)
        bulk_row.addWidget(self._bulk_invert_button)
        bulk_row.addStretch(1)
        self._footer.body_layout.addLayout(bulk_row)

        # Table
        self._table = QtWidgets.QTableWidget(group)
        self._table.setObjectName("global_fit_unified_species_table")
        self._table.setItemDelegate(_ValidationCellDelegate(self._table))
        self._table.setColumnCount(_Col.COUNT)
        self._table.setHorizontalHeaderLabels(_Col.HEADERS)
        _ih = self._table.horizontalHeader()
        _ih.setStretchLastSection(False)
        for col in range(_Col.COUNT):
            _ih.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        _ih.setSectionResizeMode(_Col.SPECIES, QtWidgets.QHeaderView.Interactive)
        self._table.setColumnWidth(_Col.SPECIES, 140)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(200)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._footer.body_layout.addWidget(self._table, stretch=1)

        layout.addWidget(group, stretch=1)

    # ------------------------------------------------------------------
    # Dataset weight editor
    # ------------------------------------------------------------------
    def _refresh_dataset_weight_editor_state(self) -> None:
        if not hasattr(self, "_dataset_weight_edit"):
            return
        ds_id = self._current_dataset_id
        label = self._dataset_label_getter(ds_id or "")
        self._dataset_weight_is_refreshing = True
        try:
            if hasattr(self, "_context_label"):
                self._context_label.setText(
                    f"Selected dataset: {label}" if ds_id else "Selected dataset: \u2014"
                )
            enabled = bool(ds_id)
            if hasattr(self, "_weight_mode_combo"):
                self._weight_mode_combo.setEnabled(enabled)
            if ds_id:
                self._dataset_weight_edit.setText(f"{self._dataset_weight_getter(ds_id):.6g}")
            else:
                self._dataset_weight_edit.clear()
            allow_custom = (
                enabled
                and getattr(self, "_weight_mode_combo", None) is not None
                and self._weight_mode_combo.currentIndex() != 0
            )
            self._dataset_weight_edit.setEnabled(bool(allow_custom))
        finally:
            self._dataset_weight_is_refreshing = False

    def _flush_dataset_weight_editor_for_dataset(self, dataset_id: Optional[str]) -> None:
        if self._dataset_weight_is_refreshing or not hasattr(self, "_dataset_weight_edit"):
            return
        ds_id = str(dataset_id or "").strip()
        if not ds_id or not self._dataset_weight_edit.isEnabled():
            return
        text = str(self._dataset_weight_edit.text() or "").strip()
        try:
            weight = float(text)
        except Exception:
            self._refresh_dataset_weight_editor_state()
            return
        if not np.isfinite(weight):
            self._refresh_dataset_weight_editor_state()
            return
        self._persist_dataset_weight_callback(ds_id, weight)

    def _on_weight_mode_changed(self) -> None:
        self._flush_dataset_weight_editor_for_dataset(self._current_dataset_id)
        self._refresh_dataset_weight_editor_state()

    def _commit_selected_dataset_weight_edit(self) -> None:
        if self._dataset_weight_is_refreshing:
            return
        ds_id = self._current_dataset_id
        if not ds_id:
            return
        text = str(self._dataset_weight_edit.text() or "").strip()
        try:
            weight = float(text)
        except Exception:
            self._refresh_dataset_weight_editor_state()
            return
        if not np.isfinite(weight):
            self._refresh_dataset_weight_editor_state()
            return
        self._persist_dataset_weight_callback(ds_id, weight)
        self._refresh_dataset_weight_editor_state()

    # ------------------------------------------------------------------
    # Bulk actions
    # ------------------------------------------------------------------
    def _apply_bulk_action(self, action: str) -> None:
        ds_id = str(self._current_dataset_id or "").strip()
        if not ds_id:
            return
        available_set = set(self._fit_targets_available_by_dataset.get(ds_id, []))
        effective = available_set & set(self._current_row_species)
        if not effective:
            return
        pending = self._fit_targets_selection_pending.setdefault(ds_id, set())
        if action == "all":
            updated = set(effective)
        elif action == "none":
            updated = set()
        elif action == "invert":
            updated = effective - pending
        else:
            return
        self._fit_targets_selection_pending[ds_id] = updated
        self._sync_visible_include_states()
        self._update_combined_dirty_state()

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------
    def _populate_table(self) -> None:
        if not self._fit_universe_initialized:
            self._fit_universe_initialized = True
            self._recompute_fit_universe()
        if not self._cached_modeled_series:
            self._cached_modeled_series = frozenset(self._safe_modeled_series())
        ds_id = str(self._current_dataset_id or "").strip()
        self._is_refreshing = True
        try:
            if not ds_id:
                self._current_row_species = []
                self._table.setRowCount(0)
                return

            available = set(self._fit_targets_available_by_dataset.get(ds_id, []))
            pending_sel = self._fit_targets_selection_pending.get(ds_id, set())
            invalid_weights = self._fit_target_weights_pending_invalid.get(ds_id, {})
            mechanism_set = {str(species) for species in self._mechanism_species if str(species).strip()}
            pending_ic = self._pending_ic_state_for_dataset(ds_id)
            self._current_row_species = self._displayed_row_species_for_dataset(ds_id)

            self._table.setRowCount(len(self._current_row_species))
            for row, species in enumerate(self._current_row_species):
                has_data = species in available
                is_mechanism_species = species in mechanism_set

                # --- Col 0: Include in Fit ---
                include_item = QtWidgets.QTableWidgetItem()
                if has_data:
                    include_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    include_item.setCheckState(Qt.Checked if species in pending_sel else Qt.Unchecked)
                else:
                    include_item.setFlags(Qt.ItemIsSelectable)
                    include_item.setCheckState(Qt.Unchecked)
                self._table.setItem(row, _Col.INCLUDE, include_item)

                # --- Col 1: Weight ---
                weight_text = self._pending_fit_target_weight_text(ds_id, species) if has_data else ""
                if has_data and species in (invalid_weights or {}):
                    weight_text = str(invalid_weights[species])
                weight_item = QtWidgets.QTableWidgetItem(weight_text)
                weight_item.setTextAlignment(Qt.AlignCenter)
                if has_data:
                    weight_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                else:
                    weight_item.setFlags(Qt.ItemIsSelectable)
                if has_data and self._fit_target_weight_is_pending_invalid(ds_id, species):
                    weight_item.setBackground(_INVALID_BG)
                    weight_item.setForeground(_INVALID_FG)
                self._table.setItem(row, _Col.WEIGHT, weight_item)

                # --- Col 2: Species (read-only) ---
                species_item = QtWidgets.QTableWidgetItem(str(species))
                species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                species_item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row, _Col.SPECIES, species_item)

                if is_mechanism_species:
                    spec = dict(pending_ic.get(species) or {})
                    init_val = float(spec.get("initial", 0.0))
                    fit_flag = bool(spec.get("fit", False))
                    log10_flag = bool(spec.get("log10", False))
                    min_val = float(spec.get("min", 0.0))
                    max_val = float(spec.get("max", max(10.0, init_val * 10 or 10.0)))

                    # --- Col 3: Initial Value ---
                    init_item = QtWidgets.QTableWidgetItem(f"{init_val:.6g}")
                    init_item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(row, _Col.INITIAL, init_item)

                    # --- Col 4: Fit IC ---
                    fit_item = QtWidgets.QTableWidgetItem()
                    fit_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    fit_item.setCheckState(Qt.Checked if fit_flag else Qt.Unchecked)
                    self._table.setItem(row, _Col.FIT_IC, fit_item)

                    # --- Col 5: Log10 ---
                    log_item = QtWidgets.QTableWidgetItem()
                    if fit_flag:
                        log_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    else:
                        log_item.setFlags(Qt.ItemIsSelectable)
                    log_item.setCheckState(Qt.Checked if log10_flag else Qt.Unchecked)
                    self._table.setItem(row, _Col.LOG10, log_item)

                    # --- Col 6: Min ---
                    min_item = QtWidgets.QTableWidgetItem(f"{min_val:.6g}")
                    min_item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(row, _Col.MIN, min_item)

                    # --- Col 7: Max ---
                    max_item = QtWidgets.QTableWidgetItem(f"{max_val:.6g}")
                    max_item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(row, _Col.MAX, max_item)
                else:
                    for col in (_Col.INITIAL, _Col.FIT_IC, _Col.LOG10, _Col.MIN, _Col.MAX):
                        disabled_item = QtWidgets.QTableWidgetItem("")
                        disabled_item.setFlags(Qt.ItemIsSelectable)
                        disabled_item.setTextAlignment(Qt.AlignCenter)
                        if col in (_Col.FIT_IC, _Col.LOG10):
                            disabled_item.setCheckState(Qt.Unchecked)
                        self._table.setItem(row, col, disabled_item)
        finally:
            self._is_refreshing = False

    # ------------------------------------------------------------------
    # Table event handling
    # ------------------------------------------------------------------
    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._is_refreshing:
            return
        col = item.column()
        row = item.row()
        if row < 0 or row >= len(self._current_row_species):
            return
        species = self._current_row_species[row]
        ds_id = str(self._current_dataset_id or "").strip()
        mechanism_set = {str(name) for name in self._mechanism_species if str(name).strip()}
        available = set(self._fit_targets_available_by_dataset.get(ds_id, []))

        if col == _Col.INCLUDE:
            if not ds_id or species not in available:
                return
            checked = item.checkState() == Qt.Checked
            pending = self._fit_targets_selection_pending.setdefault(ds_id, set())
            if checked:
                pending.add(species)
            else:
                pending.discard(species)
            self._update_combined_dirty_state()

        elif col == _Col.WEIGHT:
            if not ds_id or species not in available:
                return
            self._set_pending_fit_target_weight_text(ds_id, species, item.text())
            if self._fit_target_weight_is_pending_invalid(ds_id, species):
                item.setBackground(_INVALID_BG)
                item.setForeground(_INVALID_FG)
            else:
                item.setBackground(_DEFAULT_BG)
                item.setForeground(_DEFAULT_FG)
            self._update_combined_dirty_state()

        elif col in (_Col.INITIAL, _Col.MIN, _Col.MAX):
            if not ds_id or species not in mechanism_set:
                return
            self._clear_ic_error()
            self._ic_editor_dirty = True
            field = {
                _Col.INITIAL: "initial",
                _Col.MIN: "min",
                _Col.MAX: "max",
            }[col]
            try:
                value = float(item.text())
            except Exception:
                value = None
            if value is not None:
                pending_ic = self._pending_ic_state_for_dataset(ds_id)
                if species in pending_ic:
                    pending_ic[species][field] = float(value)
            self._update_combined_dirty_state()

        elif col == _Col.FIT_IC:
            if not ds_id or species not in mechanism_set:
                return
            self._clear_ic_error()
            self._ic_editor_dirty = True
            fit_checked = item.checkState() == Qt.Checked
            pending_ic = self._pending_ic_state_for_dataset(ds_id)
            if species in pending_ic:
                pending_ic[species]["fit"] = bool(fit_checked)
            log_item = self._table.item(row, _Col.LOG10)
            if log_item is not None:
                self._is_refreshing = True
                try:
                    if fit_checked:
                        log_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    else:
                        log_item.setFlags(Qt.ItemIsSelectable)
                        log_item.setCheckState(Qt.Unchecked)
                        if species in pending_ic:
                            pending_ic[species]["log10"] = False
                finally:
                    self._is_refreshing = False
            self._update_combined_dirty_state()

        elif col == _Col.LOG10:
            if not ds_id or species not in mechanism_set:
                return
            self._clear_ic_error()
            self._ic_editor_dirty = True
            pending_ic = self._pending_ic_state_for_dataset(ds_id)
            if species in pending_ic:
                pending_ic[species]["log10"] = bool(item.checkState() == Qt.Checked)
            self._update_combined_dirty_state()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_for_dataset(self, ds_id: str) -> None:
        ds_id = str(ds_id or "").strip()
        if not ds_id:
            return
        # Reject unknown dataset IDs
        known_ids = set(self._fit_targets_available_by_dataset.keys())
        entry_ids = {str(e.get("id") or "").strip() for e in self._dataset_entries}
        if ds_id not in known_ids and ds_id not in entry_ids:
            return
        # Flush current state before switching
        old_id = self._current_dataset_id
        if old_id and old_id != ds_id:
            self._flush_visible_weight_edits_internal()
            self._flush_dataset_weight_editor_for_dataset(old_id)
            self._ic_pending[old_id] = self._copy_ic_dataset_state(self._ic_applied.get(old_id, {}))

        self._seed_ic_state_for_dataset(ds_id)
        self._current_dataset_id = ds_id
        self._ic_editor_dirty = False
        self._clear_ic_error()
        self._populate_table()
        self._refresh_dataset_weight_editor_state()
        self._update_combined_dirty_state()

    # ------------------------------------------------------------------
    # Dirty state
    # ------------------------------------------------------------------
    def _update_combined_dirty_state(self) -> None:
        self._update_fit_targets_dirty_state()
        combined = self._fit_targets_dirty or self._ic_editor_dirty
        if hasattr(self, "_footer"):
            self._footer.set_dirty(combined)
        self._refresh_internal_validity_ui()

    def _update_fit_targets_dirty_state(self) -> None:
        all_ids = (
            set(self._fit_targets_selection_applied.keys())
            | set(self._fit_targets_selection_pending.keys())
            | set(self._fit_target_weights_applied.keys())
            | set(self._fit_target_weights_pending.keys())
            | set(self._fit_target_weights_pending_invalid.keys())
        )
        dirty = False
        for ds_id in all_ids:
            if set(self._fit_targets_selection_applied.get(ds_id, []) or []) != set(
                self._fit_targets_selection_pending.get(ds_id, set()) or set()
            ):
                dirty = True
                break
            names = (
                set(self._fit_targets_available_by_dataset.get(ds_id, []) or [])
                | set((self._fit_target_weights_applied.get(ds_id, {}) or {}).keys())
                | set((self._fit_target_weights_pending.get(ds_id, {}) or {}).keys())
                | set((self._fit_target_weights_pending_invalid.get(ds_id, {}) or {}).keys())
            )
            for name in names:
                applied_weight = float((self._fit_target_weights_applied.get(ds_id, {}) or {}).get(name, 1.0))
                pending_weight = float((self._fit_target_weights_pending.get(ds_id, {}) or {}).get(name, 1.0))
                if self._fit_target_weight_is_pending_invalid(ds_id, name) or not math.isclose(
                    applied_weight, pending_weight, rel_tol=1e-12, abs_tol=1e-12,
                ):
                    dirty = True
                    break
            if dirty:
                break
        self._fit_targets_dirty = bool(dirty)

    # ------------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------------
    def _invalid_pending_used_dataset_ids(self) -> List[str]:
        used = set(self._included_dataset_ids_getter())
        invalid = []
        for ds_id in sorted(used):
            pending = self._fit_targets_selection_pending.get(ds_id, set()) if isinstance(self._fit_targets_selection_pending, dict) else set()
            if not pending:
                invalid.append(ds_id)
        return invalid

    def _invalid_pending_target_weight_dataset_ids(self) -> List[str]:
        invalid: List[str] = []
        used = set(self._included_dataset_ids_getter())
        all_ids = set(self._fit_targets_selection_pending.keys()) | set(self._fit_target_weights_pending_invalid.keys())
        for ds_id in sorted(all_ids & used):
            pending_selection = set(self._fit_targets_selection_pending.get(ds_id, set()) or set())
            invalid_map = self._fit_target_weights_pending_invalid.get(ds_id, {})
            if not isinstance(invalid_map, dict):
                continue
            if any(str(name).strip() in pending_selection for name in invalid_map.keys()):
                invalid.append(ds_id)
        return invalid

    def _invalid_applied_used_dataset_ids(self) -> List[str]:
        used = set(self._included_dataset_ids_getter())
        invalid = []
        for ds_id in sorted(used):
            applied = self._fit_targets_selection_applied.get(ds_id, []) if isinstance(self._fit_targets_selection_applied, dict) else []
            if not (applied or []):
                invalid.append(ds_id)
        return invalid

    def _refresh_internal_validity_ui(self) -> None:
        if not hasattr(self, "_footer"):
            return
        invalid_pending = set(self._invalid_pending_used_dataset_ids())
        invalid_pending_weights = set(self._invalid_pending_target_weight_dataset_ids())
        invalid_applied = set(self._invalid_applied_used_dataset_ids())

        current = str(getattr(self, "_current_dataset_id", "") or "").strip()
        target_error: Optional[str] = None
        if current and current in invalid_pending:
            label = self._dataset_label_getter(current)
            target_error = f"Dataset {label} has no fit targets. Select at least one series or uncheck Use."
        elif current and current in invalid_pending_weights:
            label = self._dataset_label_getter(current)
            target_error = f"Dataset {label} has invalid target weights. Use finite values > 0."

        if target_error:
            self._footer.set_error(target_error)
        elif self._ic_error_text:
            self._footer.set_error(self._ic_error_text)
        else:
            self._footer.set_error(None)

        if invalid_applied:
            labels = [self._dataset_label_getter(ds_id) for ds_id in sorted(invalid_applied)]
            joined = ", ".join(labels)
            message = (
                f"Run Fit disabled: {joined} has no applied fit targets. Select targets and Apply, or uncheck Use."
            )
            self._footer.set_secondary_error(message)
        else:
            self._footer.set_secondary_error(None)

        self.validityChanged.emit()

    # ------------------------------------------------------------------
    # Apply / Revert
    # ------------------------------------------------------------------
    def _apply_changes(self) -> None:
        targets_applied = self._apply_targets()
        ic_status = self._apply_ic()

        if targets_applied:
            self.targetsApplied.emit()
        if ic_status != "failed" and (targets_applied or ic_status == "applied"):
            self._populate_table()
        self._update_combined_dirty_state()

        messages = []
        if targets_applied:
            messages.append("Fit targets applied")
        if ic_status == "applied":
            messages.append("Initial conditions applied")
        if not targets_applied and ic_status != "applied":
            messages.append("No changes applied")
        self.statusMessage.emit("; ".join(messages))

    def _apply_targets(self) -> bool:
        self._flush_visible_weight_edits_internal()
        if not self._fit_targets_dirty:
            return False
        used_ids = set(self._included_dataset_ids_getter())
        new_applied = dict(self._fit_targets_selection_applied or {})
        new_applied_target_weights = dict(self._fit_target_weights_applied or {})
        invalid_pending_weights: set[str] = set()
        deferred_excluded_pending_weights: set[str] = set()

        for ds_id in sorted(set(self._fit_targets_selection_pending.keys()) | set(new_applied.keys())):
            available = list(self._fit_targets_available_by_dataset.get(ds_id, []))
            pending_set = self._fit_targets_selection_pending.get(ds_id, set()) or set()
            pending_list = [name for name in available if name in pending_set]
            invalid_map = self._fit_target_weights_pending_invalid.get(ds_id, {})
            has_invalid_pending_weights = isinstance(invalid_map, dict) and any(name in invalid_map for name in pending_list)
            if ds_id in used_ids and has_invalid_pending_weights:
                invalid_pending_weights.add(ds_id)
                continue
            if ds_id not in used_ids and has_invalid_pending_weights:
                deferred_excluded_pending_weights.add(ds_id)
                continue
            new_applied[str(ds_id)] = list(pending_list)
            pending_weights = self._fit_target_weights_pending.get(ds_id, {}) if isinstance(self._fit_target_weights_pending, dict) else {}
            new_applied_target_weights[str(ds_id)] = {
                str(name): float(pending_weights.get(name, 1.0))
                for name in pending_list
            }

        self._fit_targets_selection_applied = {ds: list(v) for ds, v in new_applied.items()}
        self._fit_target_weights_applied = {ds: dict(v) for ds, v in new_applied_target_weights.items()}
        for ds_id in list(self._fit_targets_selection_pending.keys()):
            if (
                ds_id in invalid_pending_weights
                or ds_id in deferred_excluded_pending_weights
            ):
                continue
            self._fit_targets_selection_pending[ds_id] = set(self._fit_targets_selection_applied.get(ds_id, []) or [])
            self._fit_target_weights_pending[ds_id] = dict(self._fit_target_weights_applied.get(ds_id, {}) or {})
            self._fit_target_weights_pending_invalid.pop(ds_id, None)
        return True

    def _apply_ic(self) -> str:
        if not self._ic_editor_dirty:
            return "skipped"
        error = self._validate_ic_cells()
        if error:
            self._ic_error_text = str(error)
            if hasattr(self, "_footer"):
                self._footer.set_error(str(error))
            return "failed"
        ds_id = str(self._current_dataset_id or "").strip()
        if not ds_id:
            self._ic_error_text = "No dataset selected."
            if hasattr(self, "_footer"):
                self._footer.set_error(self._ic_error_text)
            return "failed"
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is None or not hasattr(dataset_manager, "get_fit_settings") or not hasattr(dataset_manager, "update_fit_settings"):
            self._ic_error_text = "Dataset manager unavailable; cannot persist Initial Conditions."
            if hasattr(self, "_footer"):
                self._footer.set_error(self._ic_error_text)
            return "failed"
        try:
            settings = dataset_manager.get_fit_settings(ds_id)
        except Exception:
            self._ic_error_text = f"Failed to load fit settings for dataset {ds_id}."
            if hasattr(self, "_footer"):
                self._footer.set_error(self._ic_error_text)
            return "failed"

        initials = dict(getattr(settings, "initial_conditions", {}) or {})
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {})
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {})
        bounds_map = dict(getattr(settings, "bounds", {}) or {})
        promoted = self._copy_ic_dataset_state(self._pending_ic_state_for_dataset(ds_id))
        updates: Dict[str, Dict[str, object]] = {}
        fit_flags_updates: Dict[str, bool] = {}
        for species, spec in promoted.items():
            species_key = str(species)
            initials[species_key] = float(spec["initial"])
            fit_flags[species_key] = bool(spec["fit"])
            log10_flags[species_key] = bool(spec["log10"])
            bounds_map[species_key] = (float(spec["min"]), float(spec["max"]))
            updates[species_key] = {
                "initial": float(spec["initial"]),
                "log10": bool(spec["log10"]),
                "min": float(spec["min"]),
                "max": float(spec["max"]),
            }
            fit_flags_updates[species_key] = bool(spec["fit"])

        settings.initial_conditions = initials
        settings.fit_flags = fit_flags
        settings.log10_flags = log10_flags
        settings.bounds = bounds_map
        try:
            dataset_manager.update_fit_settings(ds_id, settings)
        except Exception:
            self._ic_error_text = f"Failed to persist fit settings for dataset {ds_id}."
            if hasattr(self, "_footer"):
                self._footer.set_error(self._ic_error_text)
            return "failed"

        self._ic_applied[ds_id] = self._copy_ic_dataset_state(promoted)
        self._ic_pending[ds_id] = self._copy_ic_dataset_state(promoted)
        self._clear_ic_error()
        self.icApplied.emit(ds_id, updates, fit_flags_updates)
        self._ic_editor_dirty = False
        return "applied"

    def _validate_ic_cells(self) -> Optional[str]:
        ds_id = str(self._current_dataset_id or "").strip()
        if not ds_id:
            return "No dataset selected."
        mechanism_set = {str(species) for species in self._mechanism_species if str(species).strip()}
        if not mechanism_set:
            return "No mechanism species available."
        for row, species in enumerate(self._current_row_species):
            if species not in mechanism_set:
                continue
            init_item = self._table.item(row, _Col.INITIAL)
            fit_item = self._table.item(row, _Col.FIT_IC)
            log_item = self._table.item(row, _Col.LOG10)
            min_item = self._table.item(row, _Col.MIN)
            max_item = self._table.item(row, _Col.MAX)
            try:
                init_val = float(init_item.text())
            except Exception:
                return f"Species '{species}' requires a numeric initial concentration."
            fit_flag = bool(fit_item and fit_item.checkState() == Qt.Checked)
            log10_flag = bool(log_item and log_item.checkState() == Qt.Checked)
            try:
                min_val = float(min_item.text())
                max_val = float(max_item.text())
            except Exception:
                return f"Species '{species}' requires numeric bounds."
            if fit_flag and not (min_val < max_val):
                return f"Species '{species}' bounds must satisfy min < max."
            if fit_flag and log10_flag:
                if not (init_val > 0.0 and min_val > 0.0 and max_val > 0.0):
                    return f"Species '{species}' requires initial/min/max > 0 when Log10 is enabled."
        return None

    def _revert_changes(self) -> None:
        # Revert targets
        if self._fit_targets_dirty:
            self._fit_targets_selection_pending = {
                ds: set(v) for ds, v in (self._fit_targets_selection_applied or {}).items()
            }
            self._fit_target_weights_pending = {
                ds: dict(v) for ds, v in (self._fit_target_weights_applied or {}).items()
            }
            self._fit_target_weights_pending_invalid = {}

        self._seed_all_ic_state_from_dataset_manager()
        self._ic_editor_dirty = False
        self._clear_ic_error()
        self._populate_table()
        self._update_combined_dirty_state()

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------
    def _flush_visible_weight_edits_internal(self) -> None:
        ds_id = str(self._current_dataset_id or "").strip()
        if not ds_id or not hasattr(self, "_table"):
            return
        available = set(self._fit_targets_available_by_dataset.get(ds_id, []))
        for row in range(self._table.rowCount()):
            if row >= len(self._current_row_species):
                break
            species = self._current_row_species[row]
            if species not in available:
                continue
            weight_item = self._table.item(row, _Col.WEIGHT)
            if weight_item is None:
                continue
            self._set_pending_fit_target_weight_text(ds_id, species, weight_item.text())

    # ------------------------------------------------------------------
    # IC helpers
    # ------------------------------------------------------------------
    def _initial_parameter_defaults_for_species(self, dataset_id: str, species: str):
        ds_id = str(dataset_id or "").strip()
        settings = None
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings"):
            try:
                settings = dataset_manager.get_fit_settings(ds_id)
            except Exception:
                settings = None
        initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
        bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

        state = self._ic_entry_from_settings_maps(
            str(species),
            initials=initials,
            fit_flags=fit_flags,
            log10_flags=log10_flags,
            bounds_map=bounds_map,
        )
        return bool(state.get("fit", False)), {
            "initial": float(state.get("initial", 0.0)),
            "min": float(state.get("min", 0.0)),
            "max": float(state.get("max", 10.0)),
            "log10": bool(state.get("log10", False)),
        }

    # ------------------------------------------------------------------
    # Public API -- Properties
    # ------------------------------------------------------------------
    @property
    def fit_targets_selection_applied(self) -> Dict[str, List[str]]:
        return {ds: list(v) for ds, v in self._fit_targets_selection_applied.items()}

    def applied_target_weights_for_dataset(self, dataset_id: str) -> Dict[str, float]:
        ds_id = str(dataset_id or "").strip()
        selection = [str(x) for x in (self._fit_targets_selection_applied.get(ds_id, []) or []) if str(x).strip()]
        weights = self._fit_target_weights_applied.get(ds_id, {}) if isinstance(self._fit_target_weights_applied, dict) else {}
        return {
            name: float(weights.get(name, 1.0)) if self._is_valid_fit_target_weight(weights.get(name, 1.0)) else 1.0
            for name in selection
        }

    @property
    def full_series_by_dataset(self) -> dict:
        return self._fit_targets_full_series_by_dataset

    @property
    def full_t_by_dataset(self) -> dict:
        return self._fit_targets_full_t_by_dataset

    @property
    def available_by_dataset(self) -> dict:
        return self._fit_targets_available_by_dataset

    @property
    def _ic_editor_current_dataset_id(self) -> Optional[str]:
        return self._current_dataset_id

    @_ic_editor_current_dataset_id.setter
    def _ic_editor_current_dataset_id(self, value):
        self._current_dataset_id = value

    # ------------------------------------------------------------------
    # Public API -- Flush
    # ------------------------------------------------------------------
    def flush_visible_weight_edits(self) -> None:
        self._flush_visible_weight_edits_internal()

    def flush_dataset_weight_editor(self) -> None:
        self._flush_dataset_weight_editor_for_dataset(self._current_dataset_id)

    # ------------------------------------------------------------------
    # Public API -- Weight mode
    # ------------------------------------------------------------------
    def weight_mode_is_implicit(self) -> bool:
        return bool(getattr(self, "_weight_mode_combo", None) is not None and self._weight_mode_combo.currentIndex() == 0)

    # ------------------------------------------------------------------
    # Public API -- Validity queries
    # ------------------------------------------------------------------
    def invalid_applied_used_dataset_ids(self) -> List[str]:
        return self._invalid_applied_used_dataset_ids()

    def invalid_pending_used_dataset_ids(self) -> List[str]:
        return self._invalid_pending_used_dataset_ids()

    def invalid_pending_target_weight_dataset_ids(self) -> List[str]:
        return self._invalid_pending_target_weight_dataset_ids()

    # ------------------------------------------------------------------
    # Public API -- Dataset mutation
    # ------------------------------------------------------------------
    def add_dataset_state(
        self,
        dataset_id: str,
        *,
        full_series: Dict[str, np.ndarray],
        full_t: np.ndarray,
        available: List[str],
    ) -> None:
        self._fit_targets_full_series_by_dataset[dataset_id] = dict(full_series)
        self._fit_targets_full_t_by_dataset[dataset_id] = full_t
        self._fit_targets_selection_applied[dataset_id] = []
        self._fit_targets_selection_pending[dataset_id] = set()
        self._fit_target_weights_applied[dataset_id] = {}
        self._fit_target_weights_pending[dataset_id] = {}
        self._fit_target_weights_pending_invalid.pop(dataset_id, None)
        self._seed_ic_state_for_dataset(dataset_id)
        self._recompute_fit_universe()

    def remove_dataset_state(self, dataset_ids: set[str]) -> None:
        for ds_id in list(dataset_ids):
            self._fit_targets_selection_applied.pop(ds_id, None)
            self._fit_targets_selection_pending.pop(ds_id, None)
            self._fit_target_weights_applied.pop(ds_id, None)
            self._fit_target_weights_pending.pop(ds_id, None)
            self._fit_target_weights_pending_invalid.pop(ds_id, None)
            self._fit_targets_available_by_dataset.pop(ds_id, None)
            self._fit_targets_full_series_by_dataset.pop(ds_id, None)
            self._fit_targets_full_t_by_dataset.pop(ds_id, None)
            self._ic_pending.pop(ds_id, None)
            self._ic_applied.pop(ds_id, None)
        if self._current_dataset_id in dataset_ids:
            self._current_dataset_id = None
            self._clear_table_for_no_dataset()
        self._update_combined_dirty_state()

    def refresh_dataset_list(self) -> None:
        if self._current_dataset_id and self._current_dataset_id not in self._fit_targets_available_by_dataset:
            self._current_dataset_id = None
        if self._current_dataset_id is None:
            self._clear_table_for_no_dataset()
            self._update_combined_dirty_state()
            return
        self._refresh_dataset_weight_editor_state()
        self._refresh_internal_validity_ui()

    # ------------------------------------------------------------------
    # Public API -- Tab activation
    # ------------------------------------------------------------------
    def on_tab_activated(self, seed_dataset_id: Optional[str] = None) -> None:
        if self._current_dataset_id is None and seed_dataset_id:
            self.load_for_dataset(seed_dataset_id)
        else:
            if self._current_dataset_id:
                if not self._ic_editor_dirty:
                    self._seed_ic_state_for_dataset(self._current_dataset_id)
                self._populate_table()
            self._refresh_dataset_weight_editor_state()
            self._refresh_internal_validity_ui()

    # ------------------------------------------------------------------
    # Public API -- Refresh
    # ------------------------------------------------------------------
    def refresh_validity_ui(self) -> None:
        self._refresh_internal_validity_ui()

    # ------------------------------------------------------------------
    # Public API -- IC interface
    # ------------------------------------------------------------------
    def set_mechanism_species(self, species: list[str]) -> None:
        try:
            current_modeled = frozenset(
                str(s) for s in (self._modeled_series_getter() or set()) if str(s).strip()
            )
        except Exception:
            current_modeled = frozenset()

        mechanism_changed = (species != self._mechanism_species)
        modeled_changed = (current_modeled != self._cached_modeled_series)

        if not mechanism_changed and not modeled_changed:
            return

        self._cached_modeled_series = current_modeled

        if mechanism_changed:
            self._mechanism_species = list(species)
            self._seed_all_ic_state_from_dataset_manager()
            self._ic_editor_dirty = False
            self._clear_ic_error()

        self._recompute_fit_universe()
        self._current_row_species = []
        if self._current_dataset_id:
            self._populate_table()
        self._update_combined_dirty_state()

    def refresh_dataset_entries(self, dataset_entries: list) -> None:
        self._dataset_entries = list(dataset_entries)

    def initial_parameter_defaults_for_species(self, dataset_id: str, species: str) -> tuple[bool, dict[str, float]]:
        return self._initial_parameter_defaults_for_species(dataset_id, species)

    def set_running_state(self, running: bool) -> None:
        if hasattr(self, "_table"):
            self._table.setEnabled(not running)
        if hasattr(self, "_weight_mode_combo"):
            self._weight_mode_combo.setEnabled(not running)
        if hasattr(self, "_dataset_weight_edit"):
            if running:
                self._dataset_weight_edit.setEnabled(False)
            else:
                self._refresh_dataset_weight_editor_state()
        if hasattr(self, "_footer"):
            self._footer.setEnabled(not running)
        for btn in (getattr(self, "_bulk_all_button", None),
                    getattr(self, "_bulk_none_button", None),
                    getattr(self, "_bulk_invert_button", None)):
            if btn is not None:
                btn.setEnabled(not running)
