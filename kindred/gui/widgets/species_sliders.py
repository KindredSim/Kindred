from __future__ import annotations

from contextlib import suppress
from functools import partial
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal

from kindred.gui.species_sliders_logic import compute_row_max, compute_slider_max_option_c, try_nonneg_finite
from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel, BatchInitialConditionsTableView
from ..ui_helpers import make_placeholder_label, make_scroll_area

__all__ = ["BatchSpeciesSliders"]

logger = logging.getLogger(__name__)


_SLIDER_STEPS = 10000
_MIXED_VALUE_TEXT = "Multiple values"


def _format_concentration(value: float) -> str:
    v = float(value)
    if v == 0.0:
        return "0"
    av = abs(v)
    if av < 1e-3 or av >= 1e3:
        return f"{v:.3e}"
    if av < 0.1:
        return f"{v:.4f}"
    if av < 10.0:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _safe_object_suffix(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "unnamed"
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^0-9A-Za-z_]+", "", raw)
    return raw or "unnamed"


@dataclass
class _SpeciesSliderRow:
    species: str
    container: QtWidgets.QWidget
    slider: QtWidgets.QSlider
    value_label: QtWidgets.QLabel
    max_label: QtWidgets.QLabel
    slider_max: float
    dragging: bool = False
    mixed: bool = False


class BatchSpeciesSliders(QtWidgets.QWidget):
    """
    Initial concentration slider panel bound to the Batch Initial Conditions table.

    This widget is intentionally GUI-only (Qt), with math delegated to
    `kindred.gui.species_sliders_logic`.
    """

    speciesEdited = Signal(str, float)     # species, value
    speciesDragFinished = Signal(str)      # species
    contentStateChanged = Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, embedded: bool = False) -> None:
        super().__init__(parent)
        self._embedded = bool(embedded)
        self._table: Optional[BatchInitialConditionsTableView] = None
        self._model: Optional[BatchInitialConditionsTableModel] = None
        self._selection_model: Optional[QtCore.QItemSelectionModel] = None

        self._rows: Dict[str, _SpeciesSliderRow] = {}
        self._active: bool = False
        self._suppress_model_updates: int = 0
        self._current_set_id: Optional[str] = None
        self._current_row: Optional[int] = None
        self._slider_callbacks: Dict[str, Dict[str, object]] = {}
        self._transaction_owner: object | None = None
        self._hidden_species: set[str] = set()
        self._visible_species_signature: tuple[str, ...] = ()
        self._selected_rows_signature: tuple[int, ...] = ()
        self._placeholder: QtWidgets.QLabel | None = None
        self._hidden_placeholder: QtWidgets.QLabel | None = None

        layout = QtWidgets.QVBoxLayout(self)
        if self._embedded:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            header = QtWidgets.QLabel("Initial Concentration Sliders")
            header_font = header.font()
            header_font.setBold(True)
            header.setFont(header_font)
            layout.addWidget(header)

            desc = QtWidgets.QLabel("Adjust initial concentrations for the selected set")
            desc.setWordWrap(True)
            desc_font = desc.font()
            desc_font.setPointSize(desc_font.pointSize() - 1)
            desc.setFont(desc_font)
            layout.addWidget(desc)

            self._scroll = make_scroll_area(self)
            layout.addWidget(self._scroll, stretch=1)
        self._sliders_widget = QtWidgets.QWidget(self if self._embedded else self._scroll)
        self._sliders_layout = QtWidgets.QVBoxLayout(self._sliders_widget)
        self._sliders_layout.setContentsMargins(0, 0, 0, 0)
        self._sliders_layout.setSpacing(6 if self._embedded else 8)
        if self._embedded:
            layout.addWidget(self._sliders_widget, stretch=1)
        else:
            self._scroll.setWidget(self._sliders_widget)

            self._placeholder = make_placeholder_label("Select a set row to edit initial concentrations.")
            self._sliders_layout.addWidget(self._placeholder)
            self._hidden_placeholder = make_placeholder_label("All concentration sliders hidden by picker.")
            self._hidden_placeholder.hide()
            self._sliders_layout.addWidget(self._hidden_placeholder)
        self._sliders_layout.addStretch(1)

    def attach(self, *, table: BatchInitialConditionsTableView, model: BatchInitialConditionsTableModel) -> None:
        if self._table is table and self._model is model:
            return
        self.deactivate()
        self._table = table
        self._model = model
        self._selection_model = None

    def set_transaction_owner(self, owner: object) -> None:
        self._transaction_owner = owner

    def slider_picker_entries(self) -> list[tuple[str, str, bool]]:
        return [(species, species, self.species_visible(species)) for species in self.slider_species_names()]

    def slider_species_names(self) -> list[str]:
        if self._rows:
            return [str(species) for species in self._rows]
        model = self._model
        if model is None:
            return []
        try:
            return [str(species) for species in model.store().visible_species()]
        except Exception:
            return []

    def species_visible(self, species: str) -> bool:
        return str(species) not in self._hidden_species

    def set_species_visible(self, species: str, visible: bool) -> None:
        species_s = str(species)
        if visible:
            self._hidden_species.discard(species_s)
        else:
            self._hidden_species.add(species_s)
        self._apply_species_visibility(species_s)
        self._sync_visibility_state()
        self.contentStateChanged.emit(bool(self._rows))

    def activate(self) -> None:
        if self._active:
            self.rebuild_from_current_row()
            return
        if self._table is None or self._model is None:
            return

        self._active = True
        self._model.dataChanged.connect(self._on_model_data_changed)
        self._model.sliderEditTargetsChanged.connect(self._on_slider_edit_targets_changed)
        sel = self._table.selectionModel()
        self._selection_model = sel
        if sel is not None:
            sel.currentChanged.connect(self._on_current_changed)
            sel.selectionChanged.connect(self._on_selection_changed)
        self.rebuild_from_current_row()

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._model is not None:
            with suppress(TypeError, RuntimeError):
                self._model.dataChanged.disconnect(self._on_model_data_changed)
            with suppress(TypeError, RuntimeError):
                self._model.sliderEditTargetsChanged.disconnect(self._on_slider_edit_targets_changed)
        sel = self._selection_model
        self._selection_model = None
        if sel is not None:
            with suppress(TypeError, RuntimeError):
                sel.currentChanged.disconnect(self._on_current_changed)
            with suppress(TypeError, RuntimeError):
                sel.selectionChanged.disconnect(self._on_selection_changed)
        self.contentStateChanged.emit(bool(self._rows))

    def reset_selected_rows_to_snapshot(self) -> bool:
        target_rows = self._target_rows_for_write()
        if not target_rows:
            return False
        try:
            owner = self._transaction_owner
            changed_any = bool(
                owner is not None
                and hasattr(owner, "discard_concentration_overlays_for_rows")
                and owner.discard_concentration_overlays_for_rows(target_rows)
            )
        except Exception:
            changed_any = False
        if bool(changed_any):
            self.rebuild_from_current_row()
        return bool(changed_any)

    def get_current_values(self) -> dict[str, float]:
        """
        Return current slider values keyed by species name.

        This is a public, safe accessor for external callers (e.g. MainWindow)
        that need to commit the current widget state back into the batch model.
        """
        if not self._active or not self._rows:
            return {}
        values: dict[str, float] = {}
        for species, entry in list(self._rows.items()):
            try:
                pos = int(entry.slider.value())
                slider_max = float(entry.slider_max)
                value = self._pos_to_value(pos, slider_max)
            except Exception as exc:
                logger.debug("Failed to compute current slider value for %s: %s", species, exc, exc_info=True)
                continue
            v, ok = try_nonneg_finite(value)
            if not ok:
                logger.debug("Species slider produced invalid value for %s (%r); using 0.0", species, value)
            values[str(species)] = float(v)
        return values

    def clear_snapshot_cache(self) -> None:
        self._current_set_id = None
        self._current_row = None
        self._selected_rows_signature = ()

    def rebase_snapshots_for_rows(self, rows: list[int]) -> None:
        _ = rows

    # ---------------- internal ----------------

    def _on_current_changed(self, current: QtCore.QModelIndex, _prev: QtCore.QModelIndex) -> None:
        if not self._active:
            return
        if not current.isValid():
            return
        self._rebuild_if_primary_row_changed()

    def _on_selection_changed(
        self,
        _selected: QtCore.QItemSelection,
        _deselected: QtCore.QItemSelection,
    ) -> None:
        # Row-highlight selection is a separate group-operation state.
        # It must not retarget the species panel.
        return

    def _on_slider_edit_targets_changed(self) -> None:
        if not self._active:
            return
        self._rebuild_if_primary_row_changed()

    def _on_model_data_changed(
        self,
        top_left: QtCore.QModelIndex,
        bottom_right: QtCore.QModelIndex,
        roles: Optional[list[int]] = None,
    ) -> None:
        if not self._active or self._suppress_model_updates:
            return
        if self._table is None or self._model is None:
            return
        current_row = self._current_row
        if not (top_left.isValid() and bottom_right.isValid()):
            return
        changed_rows = range(int(top_left.row()), int(bottom_right.row()) + 1)
        target_rows = {int(row) for row in self._target_rows_for_write()}
        selection_affected = any(int(row) in target_rows for row in changed_rows)
        current_row_affected = (
            current_row is not None and int(top_left.row()) <= int(current_row) <= int(bottom_right.row())
        )
        if not current_row_affected and not selection_affected:
            return

        any_dragging = any(row.dragging for row in self._rows.values())
        if any_dragging:
            # Do not fight with user gestures. Best-effort refresh of non-dragging
            # readouts (no range recompute).
            self._refresh_from_model(recompute_ranges=False)
            return
        self._refresh_from_model(recompute_ranges=True)

    def rebuild_from_current_row(self) -> None:
        if not self._active or self._table is None or self._model is None:
            self._show_placeholder()
            return
        row = self._primary_source_row()
        if row is None:
            self._show_placeholder()
            return
        if not (0 <= row < self._model.rowCount()):
            self._show_placeholder()
            return

        store = self._model.store()
        try:
            set_id = store.set_id_for_row(int(row))
        except Exception:
            set_id = ""
        self._current_row = int(row)
        self._current_set_id = str(set_id or "") or None
        self._selected_rows_signature = tuple(self._target_rows_for_write())

        self._clear_sliders()
        species = list(store.visible_species())
        self._reset_hidden_species_for_visible_set(species)
        if not species:
            self._show_placeholder()
            return

        parsed, invalid_inputs = self._effective_row_values(int(row), species)
        values = [float(parsed.get(str(sp), 0.0)) for sp in species]
        mixed_species = self._mixed_species_for_rows(self._target_rows_for_write(), species)
        if invalid_inputs:
            logger.debug(
                "Species sliders: %d invalid concentration inputs coerced to 0.0 (first few: %s)",
                len(invalid_inputs),
                invalid_inputs[:5],
            )

        row_max = compute_row_max(values)

        for sp in species:
            v = float(parsed.get(str(sp), 0.0))
            slider_max = compute_slider_max_option_c(v=v, row_max=row_max)
            self._add_species_slider(
                species=str(sp),
                value=v,
                slider_max=float(slider_max),
                mixed=str(sp) in mixed_species,
            )

        if self._placeholder is not None:
            self._placeholder.hide()
        self._sync_visibility_state()
        self.contentStateChanged.emit(True)

    def refresh_current_row_from_model(self, *, recompute_ranges: bool = True) -> None:
        if not self._active or self._table is None or self._model is None:
            return
        if not self._rows:
            self.rebuild_from_current_row()
            return
        self._refresh_from_model(recompute_ranges=bool(recompute_ranges))

    def _refresh_from_model(self, *, recompute_ranges: bool) -> None:
        model = self._model
        if model is None or self._current_row is None:
            return
        store = model.store()
        species = list(store.visible_species())
        if not species:
            return
        row = int(self._current_row)
        self._selected_rows_signature = tuple(self._target_rows_for_write())

        parsed, invalid_inputs = self._effective_row_values(row, species)
        values = [float(parsed.get(str(sp), 0.0)) for sp in species]
        mixed_species = self._mixed_species_for_rows(self._target_rows_for_write(), species)
        if invalid_inputs:
            logger.debug(
                "Species sliders refresh: %d invalid concentration inputs coerced to 0.0 (first few: %s)",
                len(invalid_inputs),
                invalid_inputs[:5],
            )

        any_dragging = any(entry.dragging for entry in self._rows.values())
        effective_recompute = bool(recompute_ranges) and not any_dragging
        row_max = compute_row_max(values) if effective_recompute else None

        for sp in species:
            entry = self._rows.get(str(sp))
            if entry is None:
                continue
            if entry.dragging:
                continue
            v = float(parsed.get(str(sp), 0.0))
            if effective_recompute and row_max is not None:
                new_max = compute_slider_max_option_c(v=v, row_max=float(row_max))
                if math.isfinite(float(new_max)) and float(new_max) > 0.0:
                    entry.slider_max = float(new_max)
                    entry.max_label.setText(_format_concentration(float(entry.slider_max)))
            pos = self._value_to_pos(v, entry.slider_max)
            with QtCore.QSignalBlocker(entry.slider):
                entry.slider.setValue(int(pos))
            entry.mixed = str(sp) in mixed_species
            entry.value_label.setText(_MIXED_VALUE_TEXT if entry.mixed else _format_concentration(v))

    def _current_table_row(self) -> Optional[int]:
        table = self._table
        model = self._model
        if table is None or model is None:
            return None
        index = table.currentIndex()
        if not index.isValid():
            return None
        row = int(index.row())
        if not (0 <= row < int(model.rowCount())):
            return None
        return int(row)

    def _selected_rows(self) -> list[int]:
        table = self._table
        model = self._model
        if table is None or model is None:
            return []
        row_count = int(model.rowCount())
        if row_count <= 0:
            return []
        sel = table.selectionModel()
        if sel is None:
            return []
        rows = sorted({int(idx.row()) for idx in sel.selectedRows(0) if idx.isValid()})
        return [int(row) for row in rows if 0 <= int(row) < row_count]

    def _primary_source_row(self) -> Optional[int]:
        return self._current_table_row()

    def _rebuild_if_primary_row_changed(self) -> None:
        row = self._primary_source_row()
        selected_rows_signature = tuple(self._target_rows_for_write())
        if row is None:
            if self._current_row is not None:
                self.rebuild_from_current_row()
            return
        if self._current_row is not None and int(self._current_row) == int(row):
            current_species = {str(species) for species in self._rows}
            expected_species = {str(species) for species in self.slider_species_names()}
            if (
                current_species == expected_species
                and bool(self._rows)
                and self._selected_rows_signature == selected_rows_signature
            ):
                return
        self.rebuild_from_current_row()

    def _target_rows_for_write(self) -> list[int]:
        model = self._model
        current_row = self._current_table_row()
        if model is None:
            return [int(current_row)] if current_row is not None else []
        target_set_ids: list[str] = []
        owner = self._transaction_owner
        if owner is not None and hasattr(owner, "effective_slider_edit_target_set_ids"):
            try:
                target_set_ids = [
                    str(set_id)
                    for set_id in (owner.effective_slider_edit_target_set_ids() or [])
                    if str(set_id)
                ]
            except Exception:
                target_set_ids = []
        rows: list[int] = []
        seen: set[int] = set()
        source_set_ids = target_set_ids or model.slider_edit_target_set_ids()
        for set_id in source_set_ids:
            row = model.store().row_for_set_id(str(set_id))
            if row is None:
                continue
            row_i = int(row)
            if row_i in seen:
                continue
            rows.append(row_i)
            seen.add(row_i)
        if (not target_set_ids) and current_row is not None and int(current_row) not in seen:
            rows.insert(0, int(current_row))
        return rows

    def _show_placeholder(self) -> None:
        self._clear_sliders()
        self._selected_rows_signature = ()
        if self._placeholder is not None:
            self._placeholder.show()
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.hide()
        self.contentStateChanged.emit(False)

    def has_slider_rows(self) -> bool:
        return bool(self._rows)

    def visible_row_count(self) -> int:
        return sum(1 for species in self._rows if self.species_visible(species))

    def has_visible_entries(self) -> bool:
        return self.visible_row_count() > 0

    def _reset_hidden_species_for_visible_set(self, species_names: Sequence[str]) -> None:
        signature = tuple(str(species) for species in (species_names or []))
        if signature == self._visible_species_signature:
            return
        self._visible_species_signature = signature
        self._hidden_species = set(signature)

    def _clear_sliders(self) -> None:
        for entry in list(self._rows.values()):
            with suppress(RuntimeError, TypeError):
                self._sliders_layout.removeWidget(entry.container)
            w = entry.container
            if w is not None:
                with suppress(RuntimeError, TypeError):
                    w.setParent(None)
                    w.deleteLater()
        self._rows.clear()
        self._slider_callbacks.clear()
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.hide()

    def _effective_row_values(
        self,
        row: int,
        species_names: Sequence[str],
    ) -> tuple[Dict[str, float], list[tuple[str, object]]]:
        model = self._model
        if model is None:
            return {}, []
        store = model.store()
        parsed: Dict[str, float] = {}
        invalid_inputs: list[tuple[str, object]] = []
        for sp in species_names:
            raw = store.get_value(int(row), str(sp))
            value, ok = try_nonneg_finite(str(raw).strip())
            if not ok:
                invalid_inputs.append((str(sp), raw))
            parsed[str(sp)] = float(value)
        owner = self._transaction_owner
        if owner is not None and hasattr(owner, "preview_initials_for_row"):
            try:
                preview = dict(owner.preview_initials_for_row(int(row), parsed))
                parsed = {str(species): float(preview.get(str(species), 0.0)) for species in species_names}
            except Exception:
                logger.debug("Failed to merge staged concentration overlays for row %s", row, exc_info=True)
        return parsed, invalid_inputs

    def _mixed_species_for_rows(self, rows: Sequence[int], species_names: Sequence[str]) -> set[str]:
        candidate_rows = [int(row) for row in (rows or [])]
        if len(candidate_rows) <= 1:
            return set()
        effective_by_row = [self._effective_row_values(int(row), species_names)[0] for row in candidate_rows]
        mixed_species: set[str] = set()
        for sp in species_names:
            values = [float(row_values.get(str(sp), 0.0)) for row_values in effective_by_row]
            first = values[0]
            if any(not math.isclose(float(value), first, rel_tol=1e-12, abs_tol=1e-12) for value in values[1:]):
                mixed_species.add(str(sp))
        return mixed_species

    def _add_species_slider(self, *, species: str, value: float, slider_max: float, mixed: bool = False) -> None:
        container = QtWidgets.QWidget(self._sliders_widget)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        name_label = QtWidgets.QLabel(str(species))
        font = name_label.font()
        font.setBold(True)
        name_label.setFont(font)
        top.addWidget(name_label)
        top.addStretch(1)
        value_label = QtWidgets.QLabel(_MIXED_VALUE_TEXT if mixed else _format_concentration(float(value)))
        value_label.setMinimumWidth(80)
        value_label.setAlignment(QtCore.Qt.AlignRight)
        top.addWidget(value_label)
        layout.addLayout(top)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, container)
        slider.setMinimum(0)
        slider.setMaximum(_SLIDER_STEPS)
        slider.setSingleStep(10)
        slider.setPageStep(200)
        slider.setObjectName(f"speciesSlider_{_safe_object_suffix(species)}")
        slider_pos = self._value_to_pos(float(value), float(slider_max))
        with QtCore.QSignalBlocker(slider):
            slider.setValue(int(slider_pos))
        layout.addWidget(slider)

        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(0)
        range_row.addStretch(1)
        range_row.addWidget(QtWidgets.QLabel("0"))
        range_row.addStretch(1)
        max_label = QtWidgets.QLabel(_format_concentration(float(slider_max)))
        range_row.addWidget(max_label)
        layout.addLayout(range_row)

        self._sliders_layout.insertWidget(self._sliders_layout.count() - 1, container)

        entry = _SpeciesSliderRow(
            species=str(species),
            container=container,
            slider=slider,
            value_label=value_label,
            max_label=max_label,
            slider_max=float(slider_max),
            dragging=False,
            mixed=bool(mixed),
        )
        self._rows[str(species)] = entry
        self._apply_species_visibility(str(species))

        sp = str(species)
        callbacks = self._slider_callbacks.setdefault(sp, {})
        cb_value = partial(self._on_slider_value_changed, sp)
        cb_pressed = partial(self._on_slider_pressed, sp)
        cb_released = partial(self._on_slider_released, sp)
        callbacks["valueChanged"] = cb_value
        callbacks["sliderPressed"] = cb_pressed
        callbacks["sliderReleased"] = cb_released
        slider.valueChanged.connect(cb_value)
        slider.sliderPressed.connect(cb_pressed)
        slider.sliderReleased.connect(cb_released)

    def _on_slider_pressed(self, species: str) -> None:
        entry = self._rows.get(str(species))
        if entry is None:
            return
        entry.dragging = True

    def _on_slider_released(self, species: str) -> None:
        entry = self._rows.get(str(species))
        if entry is None:
            return
        entry.dragging = False
        self.speciesDragFinished.emit(str(species))

    def _on_slider_value_changed(self, species: str, pos: int) -> None:
        if not self._active or self._table is None or self._model is None:
            return
        entry = self._rows.get(str(species))
        if entry is None:
            return
        value = self._pos_to_value(int(pos), float(entry.slider_max))
        value = max(0.0, float(value))
        entry.mixed = False
        entry.value_label.setText(_format_concentration(value))

        target_rows = self._target_rows_for_write()
        if not target_rows:
            return
        try:
            owner = self._transaction_owner
            if owner is not None and hasattr(owner, "stage_concentration_value_for_rows"):
                owner.stage_concentration_value_for_rows(target_rows, species=str(species), value=float(value))
        except Exception:
            logger.debug("Failed to stage concentration overlay for %s", species, exc_info=True)

        self.speciesEdited.emit(str(species), float(value))

    def _apply_species_visibility(self, species: str) -> None:
        entry = self._rows.get(str(species))
        if entry is None:
            return
        entry.container.setVisible(self.species_visible(str(species)))

    def _sync_visibility_state(self) -> None:
        visible_count = 0
        for species, entry in self._rows.items():
            visible = self.species_visible(species)
            entry.container.setVisible(visible)
            if visible:
                visible_count += 1
        if self._hidden_placeholder is not None:
            placeholder_visible = bool(self._placeholder is not None and self._placeholder.isVisible())
            self._hidden_placeholder.setVisible(bool(self._rows) and not placeholder_visible and visible_count == 0)

    @staticmethod
    def _pos_to_value(pos: int, slider_max: float) -> float:
        steps = float(_SLIDER_STEPS)
        p = max(0, min(int(pos), _SLIDER_STEPS))
        mx = float(slider_max) if math.isfinite(float(slider_max)) and float(slider_max) > 0.0 else 1.0
        return (float(p) / steps) * mx

    @staticmethod
    def _value_to_pos(value: float, slider_max: float) -> int:
        steps = float(_SLIDER_STEPS)
        mx = float(slider_max) if math.isfinite(float(slider_max)) and float(slider_max) > 0.0 else 1.0
        v = max(0.0, float(value))
        frac = 0.0 if mx <= 0.0 else min(max(v / mx, 0.0), 1.0)
        return int(round(frac * steps))
