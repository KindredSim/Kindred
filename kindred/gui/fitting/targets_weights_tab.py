"""
Targets & Weights tab widget for the fitting window.

Extracted from FittingWindow -- owns the fit-target selection checklist,
per-target weight editors, dataset weight editor, bulk actions, and
apply/revert logic.  Communicates with FittingWindow via signals (outbound)
and callable getters/callbacks (inbound).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.ui_helpers import setup_scientific_validator
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

logger = logging.getLogger(__name__)


class TargetsWeightsTab(QtWidgets.QWidget):
    """Standalone Targets & Weights tab for the Global Fit window."""

    # ------------------------------------------------------------------
    # Signals (TargetsWeightsTab -> FittingWindow)
    # ------------------------------------------------------------------
    targetsApplied = Signal()       # emitted at end of apply
    validityChanged = Signal()      # emitted when dirty/validity state changes
    statusMessage = Signal(str)     # for footer status updates

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        # Raw inputs for state initialization
        dataset_entries: List[Dict[str, Any]],
        # Callable getters / callbacks
        dataset_entries_getter: Callable[[], List[Dict[str, Any]]],
        included_dataset_ids_getter: Callable[[], List[str]],
        dataset_label_getter: Callable[[str], str],
        dataset_weight_getter: Callable[[str], float],
        persist_dataset_weight_callback: Callable[[str, float], None],
        worker_running_getter: Callable[[], bool],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # Store callable getters / callbacks
        self._dataset_entries_getter = dataset_entries_getter
        self._included_dataset_ids_getter = included_dataset_ids_getter
        self._dataset_label_getter = dataset_label_getter
        self._dataset_weight_getter = dataset_weight_getter
        self._persist_dataset_weight_callback = persist_dataset_weight_callback
        self._worker_running_getter = worker_running_getter

        # Initialize state from raw dataset entries
        self._init_fit_targets_state(dataset_entries)

        # Build UI
        self._init_ui()

    # ------------------------------------------------------------------
    # State initialization (absorbed from FittingWindow._init_fit_targets_state)
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
                                key,
                                ds_id,
                                exc,
                                exc_info=True,
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
                ds_id,
                initial_selection,
                entry.get("target_weights"),
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
        self._fit_targets_current_dataset_id: Optional[str] = None
        self._fit_targets_is_refreshing = False
        self._fit_targets_local_selection_owned = False
        self._dataset_weight_is_refreshing = False

    # ------------------------------------------------------------------
    # UI construction (merged _create_targets_weights_tab + _create_fit_targets_panel)
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, self)
        layout.addWidget(splitter, stretch=1)

        # -- Left column: dataset list --
        list_column = QtWidgets.QWidget(splitter)
        list_layout = QtWidgets.QVBoxLayout(list_column)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        list_label = QtWidgets.QLabel("Datasets")
        list_label.setStyleSheet("font-weight: bold;")
        list_layout.addWidget(list_label)

        self._fit_targets_dataset_list = QtWidgets.QListWidget(list_column)
        self._fit_targets_dataset_list.setObjectName("global_fit_fit_targets_dataset_list")
        self._fit_targets_dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._fit_targets_dataset_list.currentItemChanged.connect(self._on_fit_targets_dataset_list_selection_changed)
        list_layout.addWidget(self._fit_targets_dataset_list, stretch=1)

        # -- Right column: fit targets panel --
        group = QtWidgets.QGroupBox("Targets & Weights")
        group.setObjectName("global_fit_fit_targets_panel")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._fit_targets_footer = ConfigPanelFooter(
            group,
            show_dirty=True,
            show_secondary_error=True,
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            error_object_name="global_fit_fit_targets_error",
            secondary_error_object_name="global_fit_fit_targets_run_blocked",
            apply_object_name="global_fit_fit_targets_apply",
            revert_object_name="global_fit_fit_targets_revert",
        )
        group_layout.addWidget(self._fit_targets_footer, stretch=1)
        self._fit_targets_footer.applyRequested.connect(self._apply_fit_targets_changes)
        self._fit_targets_footer.revertRequested.connect(self._revert_fit_targets_changes)

        self._fit_targets_context_label = QtWidgets.QLabel("Selected dataset: \u2014", group)
        self._fit_targets_context_label.setStyleSheet("font-weight: bold;")
        self._fit_targets_footer.body_layout.addWidget(self._fit_targets_context_label)

        weighting_form = QtWidgets.QFormLayout()
        weighting_form.setContentsMargins(0, 0, 0, 0)

        self._weight_mode_combo = QtWidgets.QComboBox(group)
        self._weight_mode_combo.setObjectName("global_fit_weight_mode_combo")
        self._weight_mode_combo.addItems(
            [
                "Implicit weights (1/N per dataset)",
                "User weights only",
            ]
        )
        weighting_form.addRow("Weight mode:", self._weight_mode_combo)

        self._dataset_weight_edit = QtWidgets.QLineEdit(group)
        self._dataset_weight_edit.setObjectName("global_fit_dataset_weight_edit")
        setup_scientific_validator(self._dataset_weight_edit)
        weighting_form.addRow("Dataset weight:", self._dataset_weight_edit)
        self._fit_targets_footer.body_layout.addLayout(weighting_form)

        bulk_row = QtWidgets.QHBoxLayout()
        bulk_label = QtWidgets.QLabel("Bulk:")
        bulk_label.setStyleSheet("font-size: 11px;")
        bulk_row.addWidget(bulk_label)
        self._fit_targets_bulk_all_button = QtWidgets.QPushButton("All", group)
        self._fit_targets_bulk_all_button.setObjectName("global_fit_fit_targets_bulk_all")
        self._fit_targets_bulk_all_button.clicked.connect(lambda: self._apply_fit_targets_bulk_action("all"))
        self._fit_targets_bulk_none_button = QtWidgets.QPushButton("None", group)
        self._fit_targets_bulk_none_button.setObjectName("global_fit_fit_targets_bulk_none")
        self._fit_targets_bulk_none_button.clicked.connect(lambda: self._apply_fit_targets_bulk_action("none"))
        self._fit_targets_bulk_invert_button = QtWidgets.QPushButton("Invert", group)
        self._fit_targets_bulk_invert_button.setObjectName("global_fit_fit_targets_bulk_invert")
        self._fit_targets_bulk_invert_button.clicked.connect(lambda: self._apply_fit_targets_bulk_action("invert"))
        bulk_row.addWidget(self._fit_targets_bulk_all_button)
        bulk_row.addWidget(self._fit_targets_bulk_none_button)
        bulk_row.addWidget(self._fit_targets_bulk_invert_button)
        bulk_row.addStretch(1)
        self._fit_targets_footer.body_layout.addLayout(bulk_row)

        self._fit_targets_scroll = QtWidgets.QScrollArea(group)
        self._fit_targets_scroll.setWidgetResizable(True)
        self._fit_targets_scroll.setMinimumHeight(160)
        self._fit_targets_footer.body_layout.addWidget(self._fit_targets_scroll, stretch=1)

        self._fit_targets_checks_container = QtWidgets.QWidget(self._fit_targets_scroll)
        self._fit_targets_checks_layout = QtWidgets.QVBoxLayout(self._fit_targets_checks_container)
        self._fit_targets_checks_layout.setContentsMargins(0, 0, 0, 0)
        self._fit_targets_checks_layout.setSpacing(6)
        self._fit_targets_scroll.setWidget(self._fit_targets_checks_container)

        self._weight_mode_combo.currentIndexChanged.connect(self._on_weight_mode_changed)
        self._dataset_weight_edit.editingFinished.connect(self._commit_selected_dataset_weight_edit)

        # -- Assemble splitter --
        splitter.addWidget(list_column)
        splitter.addWidget(group)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([240, 620])

        # -- Initial refresh --
        self._refresh_fit_targets_dataset_list_items()

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
        self,
        dataset_id: str,
        selection: Sequence[str],
        raw_weights: object,
    ) -> Dict[str, float]:
        weights_map = dict(raw_weights) if isinstance(raw_weights, dict) else {}
        normalized: Dict[str, float] = {}
        for name in [str(x) for x in (selection or []) if str(x).strip()]:
            value = weights_map.get(name, 1.0)
            normalized[name] = float(value) if self._is_valid_fit_target_weight(value) else 1.0
        return normalized

    def _applied_selected_target_weights_for_dataset(self, dataset_id: str) -> Dict[str, float]:
        ds_id = str(dataset_id or "").strip()
        selection = [str(x) for x in (self._fit_targets_selection_applied.get(ds_id, []) or []) if str(x).strip()]
        weights = self._fit_target_weights_applied.get(ds_id, {}) if isinstance(self._fit_target_weights_applied, dict) else {}
        return {
            name: float(weights.get(name, 1.0)) if self._is_valid_fit_target_weight(weights.get(name, 1.0)) else 1.0
            for name in selection
        }

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

    def _set_fit_target_weight_editor_visual_state(
        self,
        editor: QtWidgets.QLineEdit,
        *,
        dataset_id: str,
        target_name: str,
    ) -> None:
        if self._fit_target_weight_is_pending_invalid(dataset_id, target_name):
            editor.setStyleSheet("border: 1px solid rgb(204, 68, 68);")
        else:
            editor.setStyleSheet("")

    # ------------------------------------------------------------------
    # Dataset list
    # ------------------------------------------------------------------
    def _selected_fit_targets_dataset_id(self) -> Optional[str]:
        dataset_list = getattr(self, "_fit_targets_dataset_list", None)
        if dataset_list is None:
            return None
        item = dataset_list.currentItem()
        ds_id = str(item.data(Qt.UserRole) or "").strip() if item is not None else ""
        return ds_id or None

    def _fit_targets_dataset_item_text(self, entry: Dict[str, Any]) -> str:
        ds_id = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or ds_id)
        if not ds_id:
            return label
        included = bool(entry.get("include", True))
        return label if included else f"{label} (not used)"

    def _refresh_fit_targets_dataset_list_items(self) -> None:
        dataset_list = getattr(self, "_fit_targets_dataset_list", None)
        if dataset_list is None:
            return
        current = self._selected_fit_targets_dataset_id()
        dataset_list.blockSignals(True)
        try:
            dataset_list.clear()
            for entry in self._dataset_entries_getter():
                ds_id = str(entry.get("id") or "").strip()
                if not ds_id:
                    continue
                item = QtWidgets.QListWidgetItem(self._fit_targets_dataset_item_text(entry), dataset_list)
                item.setData(Qt.UserRole, ds_id)
        finally:
            dataset_list.blockSignals(False)

        if current:
            for row in range(dataset_list.count()):
                item = dataset_list.item(row)
                if item is not None and str(item.data(Qt.UserRole) or "").strip() == current:
                    dataset_list.setCurrentRow(row)
                    break
        if dataset_list.count() and dataset_list.currentRow() < 0:
            dataset_list.setCurrentRow(0)
        if dataset_list.count() == 0:
            self._fit_targets_current_dataset_id = None
        self._refresh_fit_targets_checklist()
        self._refresh_dataset_weight_editor_state()

    def _select_fit_targets_dataset_by_id(self, dataset_id: Optional[str]) -> bool:
        ds_id = str(dataset_id or "").strip()
        dataset_list = getattr(self, "_fit_targets_dataset_list", None)
        if dataset_list is None or not ds_id:
            return False
        for row in range(dataset_list.count()):
            item = dataset_list.item(row)
            if item is not None and str(item.data(Qt.UserRole) or "").strip() == ds_id:
                dataset_list.setCurrentRow(row)
                return True
        return False

    def _on_fit_targets_dataset_list_selection_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        self._flush_visible_fit_target_weight_edits()
        previous_id = str(previous.data(Qt.UserRole) or "").strip() if previous is not None else ""
        if previous_id:
            self._flush_dataset_weight_editor_for_dataset(previous_id)
        ds_id = str(current.data(Qt.UserRole) or "").strip() if current is not None else ""
        self._fit_targets_current_dataset_id = ds_id or None
        self._refresh_fit_targets_checklist()
        self._refresh_dataset_weight_editor_state()
        self._refresh_internal_validity_ui()

    # ------------------------------------------------------------------
    # Dataset weight editor
    # ------------------------------------------------------------------
    def _refresh_dataset_weight_editor_state(self) -> None:
        if not hasattr(self, "_dataset_weight_edit"):
            return
        ds_id = self._selected_fit_targets_dataset_id()
        label = self._dataset_label_getter(ds_id or "")
        self._dataset_weight_is_refreshing = True
        try:
            if hasattr(self, "_fit_targets_context_label"):
                self._fit_targets_context_label.setText(
                    f"Selected dataset: {label}" if ds_id else "Selected dataset: \u2014"
                )
            enabled = bool(ds_id)
            if hasattr(self, "_weight_mode_combo"):
                self._weight_mode_combo.setEnabled(enabled)
            if ds_id:
                self._dataset_weight_edit.setText(f"{self._dataset_weight_getter(ds_id):.6g}")
            else:
                self._dataset_weight_edit.clear()
            allow_custom = enabled and getattr(self, "_weight_mode_combo", None) is not None and self._weight_mode_combo.currentIndex() != 0
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
        self._flush_dataset_weight_editor_for_dataset(self._selected_fit_targets_dataset_id())
        self._refresh_dataset_weight_editor_state()

    def _commit_selected_dataset_weight_edit(self) -> None:
        if self._dataset_weight_is_refreshing:
            return
        ds_id = self._selected_fit_targets_dataset_id()
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
    def _apply_fit_targets_bulk_action(self, action: str) -> None:
        ds_id = str(self._selected_fit_targets_dataset_id() or "").strip()
        if not ds_id:
            return
        available = list(self._fit_targets_available_by_dataset.get(ds_id, []))
        available_set = {str(x).strip() for x in available if str(x).strip()}
        if not available_set:
            return

        pending = set(self._fit_targets_selection_pending.get(ds_id, set()) or set())
        if str(action).strip().lower() == "all":
            updated = set(available_set)
        elif str(action).strip().lower() == "none":
            updated = set()
        elif str(action).strip().lower() == "invert":
            updated = set(available_set) - pending
        else:
            return
        self._fit_targets_selection_pending[ds_id] = set(updated)
        self._refresh_fit_targets_checklist()
        self._update_fit_targets_dirty_state()

    # ------------------------------------------------------------------
    # Checklist rendering
    # ------------------------------------------------------------------
    def _refresh_fit_targets_checklist(self) -> None:
        ds_id = str(self._selected_fit_targets_dataset_id() or "").strip()
        self._fit_targets_current_dataset_id = ds_id or None

        self._fit_targets_is_refreshing = True
        try:
            while self._fit_targets_checks_layout.count():
                item = self._fit_targets_checks_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            if not ds_id:
                placeholder = QtWidgets.QLabel("No dataset selected.")
                placeholder.setEnabled(False)
                placeholder.setWordWrap(True)
                self._fit_targets_checks_layout.addWidget(placeholder)
                self._fit_targets_checks_layout.addStretch(1)
                return

            available = list(self._fit_targets_available_by_dataset.get(ds_id, []))
            if not available:
                placeholder = QtWidgets.QLabel("No observed series available for this dataset.")
                placeholder.setEnabled(False)
                placeholder.setWordWrap(True)
                self._fit_targets_checks_layout.addWidget(placeholder)
                self._fit_targets_checks_layout.addStretch(1)
                return

            pending = self._fit_targets_selection_pending.get(ds_id, set())
            invalid_weights = self._fit_target_weights_pending_invalid.get(ds_id, {})
            for name in available:
                row = QtWidgets.QWidget(self._fit_targets_checks_container)
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                cb = QtWidgets.QCheckBox(str(name), row)
                cb.setChecked(str(name) in pending)
                cb.toggled.connect(lambda checked, n=str(name): self._on_fit_target_toggled(n, checked))
                row_layout.addWidget(cb, stretch=1)

                weight_edit = QtWidgets.QLineEdit(row)
                weight_edit.setObjectName("global_fit_target_weight_edit")
                weight_edit.setProperty("fitTargetName", str(name))
                weight_edit.setProperty("fitTargetDatasetId", ds_id)
                weight_edit.setMaximumWidth(110)
                setup_scientific_validator(weight_edit)
                weight_edit.setText(self._pending_fit_target_weight_text(ds_id, str(name)))
                weight_edit.textChanged.connect(
                    lambda text, n=str(name): self._on_fit_target_weight_text_edited(n, text)
                )
                weight_edit.editingFinished.connect(
                    lambda n=str(name), editor=weight_edit: self._on_fit_target_weight_editing_finished(n, editor)
                )
                if str(name) in invalid_weights:
                    weight_edit.setText(str(invalid_weights[str(name)]))
                self._set_fit_target_weight_editor_visual_state(
                    weight_edit,
                    dataset_id=ds_id,
                    target_name=str(name),
                )
                row_layout.addWidget(weight_edit)
                self._fit_targets_checks_layout.addWidget(row)
            self._fit_targets_checks_layout.addStretch(1)
        finally:
            self._fit_targets_is_refreshing = False

    # ------------------------------------------------------------------
    # Target event handlers
    # ------------------------------------------------------------------
    def _on_fit_target_toggled(self, series_name: str, checked: bool) -> None:
        if self._fit_targets_is_refreshing:
            return
        ds_id = self._fit_targets_current_dataset_id
        if not ds_id:
            return
        name = str(series_name).strip()
        if not name:
            return
        pending = self._fit_targets_selection_pending.setdefault(ds_id, set())
        if checked:
            pending.add(name)
        else:
            pending.discard(name)
        self._update_fit_targets_dirty_state()

    def _on_fit_target_weight_text_edited(self, series_name: str, text: str) -> None:
        if self._fit_targets_is_refreshing:
            return
        ds_id = self._fit_targets_current_dataset_id
        if not ds_id:
            return
        name = str(series_name).strip()
        if not name:
            return
        self._set_pending_fit_target_weight_text(ds_id, name, text)
        editor = self.sender()
        if isinstance(editor, QtWidgets.QLineEdit):
            self._set_fit_target_weight_editor_visual_state(editor, dataset_id=ds_id, target_name=name)
        self._update_fit_targets_dirty_state()

    def _on_fit_target_weight_editing_finished(
        self,
        series_name: str,
        editor: QtWidgets.QLineEdit,
    ) -> None:
        if self._fit_targets_is_refreshing:
            return
        ds_id = self._fit_targets_current_dataset_id
        if not ds_id:
            return
        name = str(series_name).strip()
        if not name:
            return
        self._set_pending_fit_target_weight_text(ds_id, name, editor.text())
        if not self._fit_target_weight_is_pending_invalid(ds_id, name):
            editor.setText(f"{self._pending_fit_target_weight_value(ds_id, name):.6g}")
        self._set_fit_target_weight_editor_visual_state(editor, dataset_id=ds_id, target_name=name)
        self._update_fit_targets_dirty_state()

    def _flush_visible_fit_target_weight_edits(self) -> None:
        if not hasattr(self, "_fit_targets_checks_container"):
            return
        for editor in self._fit_targets_checks_container.findChildren(QtWidgets.QLineEdit, "global_fit_target_weight_edit"):
            ds_id = str(editor.property("fitTargetDatasetId") or "").strip()
            name = str(editor.property("fitTargetName") or "").strip()
            if not ds_id or not name:
                continue
            self._set_pending_fit_target_weight_text(ds_id, name, editor.text())
            if not self._fit_target_weight_is_pending_invalid(ds_id, name):
                editor.setText(f"{self._pending_fit_target_weight_value(ds_id, name):.6g}")
            self._set_fit_target_weight_editor_visual_state(editor, dataset_id=ds_id, target_name=name)

    # ------------------------------------------------------------------
    # Dirty state
    # ------------------------------------------------------------------
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
                    applied_weight,
                    pending_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    dirty = True
                    break
            if dirty:
                break
        self._fit_targets_dirty = bool(dirty)
        if hasattr(self, "_fit_targets_footer"):
            self._fit_targets_footer.set_dirty(self._fit_targets_dirty)
        self._refresh_internal_validity_ui()

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

    # ------------------------------------------------------------------
    # Validity UI (tab-internal portion)
    # ------------------------------------------------------------------
    def _refresh_internal_validity_ui(self) -> None:
        if not hasattr(self, "_fit_targets_footer"):
            return
        invalid_pending = set(self._invalid_pending_used_dataset_ids())
        invalid_pending_weights = set(self._invalid_pending_target_weight_dataset_ids())
        invalid_applied = set(self._invalid_applied_used_dataset_ids())

        # Inline apply error for currently selected dataset only.
        current = str(getattr(self, "_fit_targets_current_dataset_id", "") or "").strip()
        if current and current in invalid_pending:
            label = self._dataset_label_getter(current)
            message = f"Dataset {label} has no fit targets. Select at least one series or uncheck Use."
            self._fit_targets_footer.set_error(message)
        elif current and current in invalid_pending_weights:
            label = self._dataset_label_getter(current)
            message = f"Dataset {label} has invalid target weights. Use finite values > 0."
            self._fit_targets_footer.set_error(message)
        else:
            self._fit_targets_footer.set_error(None)

        # Run Fit disabling while invalid applied.
        if invalid_applied:
            labels = [self._dataset_label_getter(ds_id) for ds_id in sorted(invalid_applied)]
            joined = ", ".join(labels)
            message = (
                f"Run Fit disabled: {joined} has no applied fit targets. Select targets and Apply, or uncheck Use."
            )
            self._fit_targets_footer.set_secondary_error(message)
        else:
            self._fit_targets_footer.set_secondary_error(None)

        self.validityChanged.emit()

    # ------------------------------------------------------------------
    # Apply / Revert
    # ------------------------------------------------------------------
    def _apply_fit_targets_changes(self) -> None:
        self._flush_visible_fit_target_weight_edits()
        if not self._fit_targets_dirty:
            return
        used_ids = set(self._included_dataset_ids_getter())
        new_applied = dict(self._fit_targets_selection_applied or {})
        new_applied_target_weights = dict(self._fit_target_weights_applied or {})
        invalid_pending_used: set[str] = set()
        invalid_pending_weights: set[str] = set()
        deferred_excluded_pending_weights: set[str] = set()

        for ds_id in sorted(set(self._fit_targets_selection_pending.keys()) | set(new_applied.keys())):
            available = list(self._fit_targets_available_by_dataset.get(ds_id, []))
            pending_set = self._fit_targets_selection_pending.get(ds_id, set()) or set()
            pending_list = [name for name in available if name in pending_set]
            if ds_id in used_ids and not pending_list:
                invalid_pending_used.add(ds_id)
                continue
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

        # Commit: valid datasets update applied; invalid-used datasets keep applied unchanged and keep pending empty.
        self._fit_targets_selection_applied = {ds: list(v) for ds, v in new_applied.items()}
        self._fit_target_weights_applied = {ds: dict(v) for ds, v in new_applied_target_weights.items()}
        for ds_id in list(self._fit_targets_selection_pending.keys()):
            if (
                ds_id in invalid_pending_used
                or ds_id in invalid_pending_weights
                or ds_id in deferred_excluded_pending_weights
            ):
                continue
            self._fit_targets_selection_pending[ds_id] = set(self._fit_targets_selection_applied.get(ds_id, []) or [])
            self._fit_target_weights_pending[ds_id] = dict(self._fit_target_weights_applied.get(ds_id, {}) or {})
            self._fit_target_weights_pending_invalid.pop(ds_id, None)

        # Emit signal so FittingWindow can run post-apply cascade
        self.targetsApplied.emit()

        self._update_fit_targets_dirty_state()
        self._refresh_fit_targets_checklist()
        invalid_pending_any = bool(invalid_pending_used or invalid_pending_weights)
        msg = "Fit targets applied" if not invalid_pending_any else "Fit targets: some changes not applied"
        self.statusMessage.emit(msg)

    def _revert_fit_targets_changes(self) -> None:
        if not self._fit_targets_dirty:
            return
        self._fit_targets_selection_pending = {
            ds: set(v) for ds, v in (self._fit_targets_selection_applied or {}).items()
        }
        self._fit_target_weights_pending = {
            ds: dict(v) for ds, v in (self._fit_target_weights_applied or {}).items()
        }
        self._fit_target_weights_pending_invalid = {}
        self._update_fit_targets_dirty_state()
        self._refresh_fit_targets_checklist()

    # ------------------------------------------------------------------
    # Public API -- Fit execution path
    # ------------------------------------------------------------------
    def flush_visible_weight_edits(self) -> None:
        """Flush in-progress weight edits from visible QLineEdits."""
        self._flush_visible_fit_target_weight_edits()

    def flush_dataset_weight_editor(self) -> None:
        """Flush dataset-level weight editor for the currently selected dataset."""
        self._flush_dataset_weight_editor_for_dataset(self._selected_fit_targets_dataset_id())

    def weight_mode_is_implicit(self) -> bool:
        combo = getattr(self, "_weight_mode_combo", None)
        return combo is None or combo.currentIndex() == 0

    # ------------------------------------------------------------------
    # Public API -- Applied state accessors (return deep copies)
    # ------------------------------------------------------------------
    @property
    def fit_targets_selection_applied(self) -> Dict[str, List[str]]:
        return {ds: list(v) for ds, v in self._fit_targets_selection_applied.items()}

    def applied_target_weights_for_dataset(self, dataset_id: str) -> Dict[str, float]:
        return self._applied_selected_target_weights_for_dataset(dataset_id)

    # ------------------------------------------------------------------
    # Public API -- Full data accessors (for DataTab callable getters)
    # ------------------------------------------------------------------
    @property
    def full_series_by_dataset(self) -> dict:
        return self._fit_targets_full_series_by_dataset

    @property
    def full_t_by_dataset(self) -> dict:
        return self._fit_targets_full_t_by_dataset

    @property
    def available_by_dataset(self) -> dict:
        return self._fit_targets_available_by_dataset

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
        self._fit_targets_available_by_dataset[dataset_id] = list(available)
        self._fit_targets_selection_applied[dataset_id] = []
        self._fit_targets_selection_pending[dataset_id] = set()
        self._fit_target_weights_applied[dataset_id] = {}
        self._fit_target_weights_pending[dataset_id] = {}
        self._fit_target_weights_pending_invalid.pop(dataset_id, None)

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

    def refresh_dataset_list(self) -> None:
        self._refresh_fit_targets_dataset_list_items()
        self._refresh_internal_validity_ui()

    # ------------------------------------------------------------------
    # Public API -- Tab activation
    # ------------------------------------------------------------------
    def on_tab_activated(self, seed_dataset_id: Optional[str] = None) -> None:
        if not self._fit_targets_local_selection_owned:
            seeded = self._select_fit_targets_dataset_by_id(seed_dataset_id)
            if not seeded:
                self._refresh_fit_targets_checklist()
                self._refresh_dataset_weight_editor_state()
            self._fit_targets_local_selection_owned = True
            return
        self._refresh_fit_targets_checklist()
        self._refresh_dataset_weight_editor_state()

    # ------------------------------------------------------------------
    # Public API -- Refresh
    # ------------------------------------------------------------------
    def refresh_validity_ui(self) -> None:
        self._refresh_internal_validity_ui()
