"""Species slider snapshot projection; target selection lives outside the widget."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from PySide6 import QtCore, QtWidgets

from kindred.gui.display_name_policy import CompactText, compact_text
from kindred.gui.ui_helpers import (
    configure_compact_numeric_edit, make_bounded_label, set_bounded_label_text,
)
from kindred.gui.interactive_preview import SliderInteractionKind
from kindred.gui.widgets.fine_drag_slider import FineDragSlider, NumericValueEdit, SliderPress
from kindred.gui.species_sliders_logic import (
    compute_slider_max_option_c,
    _scale_finite_gui_range,
)

__all__ = ["BatchSpeciesSliders"]


def format_slider_value(value: float) -> str:
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    magnitude = abs(value_f)
    if magnitude < 1e-3 or magnitude >= 1e3:
        return f"{value_f:.3e}"
    if magnitude < 0.1:
        return f"{value_f:.4f}"
    if magnitude < 10.0:
        return f"{value_f:.3f}"
    return f"{value_f:.2f}"


@dataclass(slots=True)
class _SpeciesSliderRow:
    container: QtWidgets.QWidget
    name_label: QtWidgets.QLabel
    slider: FineDragSlider
    value_label: QtWidgets.QWidget
    maximum: float
    maximum_label: QtWidgets.QLabel


class BatchSpeciesSliders(QtWidgets.QWidget):
    interactionRequested = QtCore.Signal(str, float, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows: dict[str, _SpeciesSliderRow] = {}
        self._fine_mode = False
        self._values: dict[str, float] = {}
        self._hidden_species: set[str] = set()
        self._scope_signature: object | None = None
        self._target_set_ids: tuple[str, ...] = ()
        self._active_presses: dict[str, SliderPress] = {}
        self._editor_pre_edit_values: dict[str, float] = {}
        self._editor_pre_edit_text: dict[str, str] = {}
        self._last_committed_editor_text: dict[str, str] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        title = QtWidgets.QLabel("Initial Concentrations (M)", self)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        self._content = QtWidgets.QWidget(self)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(14)
        layout.addWidget(self._content)
        layout.addStretch(1)

    def set_fine_mode(self, enabled: bool) -> None:
        """Adjust keyboard steps and mouse sensitivity without changing values."""
        self._fine_mode = bool(enabled)
        for row in self._rows.values():
            row.slider.setSingleStep(1 if self._fine_mode else 50)
            row.slider.setPageStep(50 if self._fine_mode else 1000)
            row.slider.set_fine_mode(self._fine_mode)

    def render_values(
        self,
        values: Mapping[str, float],
        *,
        target_set_ids: Sequence[str],
        mixed_species: Sequence[str] = (),
        preserve_editor_drafts: bool = True,
    ) -> None:
        normalized: dict[str, float] = {}
        # The Slider runtime admitted these names and concentrations.
        for name, value in values.items():
            normalized[name] = float(value)
        self._reconcile_visibility_scope(set(normalized))
        self._target_set_ids = tuple(str(item) for item in target_set_ids)
        self._values = normalized
        mixed = {str(name) for name in mixed_species}
        same_rows = (
            tuple(normalized) == tuple(self._rows)
            and all(
                species in self._rows
                for species in normalized
            )
        )
        if same_rows:
            for species, value in normalized.items():
                self._update_row_projection(
                    species,
                    value,
                    mixed=species in mixed,
                )
        else:
            self._clear_rows()
            for species, value in normalized.items():
                self._add_row(
                    species,
                    value,
                    display_name=species,
                    mixed=species in mixed,
                )
        if not preserve_editor_drafts:
            for species, value in normalized.items():
                text = "Multiple values" if species in mixed else format_slider_value(value)
                self._set_value_projection(
                    species, value, editor_text=text, emit=False
                )
                if species in mixed:
                    editor = self._rows[species].value_label
                    editor.setProperty("invalidNumeric", False)
                    editor.setStyleSheet("")
                self.remember_editor_state(species)
                self._last_committed_editor_text[species] = text
        self._sync_visibility()

    def slider_picker_entries(self) -> list[tuple[str, CompactText, str, bool]]:
        return [
            (
                name,
                compact_text(name, max_chars=48),
                "",
                name not in self._hidden_species,
            )
            for name in self._values
        ]

    def get_current_values(self) -> dict[str, float]:
        return dict(self._values)

    def has_slider_rows(self) -> bool:
        return bool(self._rows)

    def visible_row_count(self) -> int:
        """Return logical picker visibility, independent of ancestor widgets."""

        return sum(
            1
            for species in self._rows
            if species not in self._hidden_species
        )

    def clear(self) -> None:
        self._clear_rows()
        self._values.clear()
        self._target_set_ids = ()
        self._sync_visibility()

    def _clear_rows(self) -> None:
        self._active_presses.clear()
        for row in self._rows.values():
            with QtCore.QSignalBlocker(row.slider):
                row.slider.end_drag()
            self._content_layout.removeWidget(row.container)
            row.container.deleteLater()
        self._rows.clear()
        self._editor_pre_edit_values.clear()
        self._editor_pre_edit_text.clear()
        self._last_committed_editor_text.clear()

    def _add_row(
        self,
        species: str,
        value: float,
        *,
        display_name: str,
        mixed: bool,
    ) -> None:
        container = QtWidgets.QWidget(self._content)
        layout = QtWidgets.QGridLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setColumnStretch(0, 1)
        name_label = make_bounded_label(display_name, container, max_width=16777215)
        if mixed:
            initial_text = "Multiple values"
        else:
            initial_text = format_slider_value(value)
        value_editor = NumericValueEdit(initial_text, container)
        value_editor.setAccessibleName(f"{display_name} initial concentration")
        name_label.setBuddy(value_editor)
        configure_compact_numeric_edit(value_editor, characters=15)
        value_editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        value_editor.editStarted.connect(
            lambda item=species: self._on_editor_started(item)
        )
        value_editor.editEscape.connect(
            lambda item=species: self._on_editor_escape(item)
        )
        value_editor.textChanged.connect(
            lambda _text, item=species: self._decorate_editor(item)
        )
        value_editor.editingFinished.connect(
            lambda item=species: self._commit_editor(item)
        )
        slider = FineDragSlider(
            QtCore.Qt.Orientation.Horizontal,
            container,
        )
        slider.setAccessibleName(f"{display_name} initial concentration slider")
        slider.setRange(0, 10_000)
        slider.setSingleStep(1 if self._fine_mode else 50)
        slider.setPageStep(50 if self._fine_mode else 1000)
        slider.set_fine_mode(self._fine_mode)
        maximum = compute_slider_max_option_c(v=value)
        maximum = max(float(maximum), 1e-12)
        slider.setValue(self._value_to_position(value, maximum))
        slider.setEnabled(bool(self._target_set_ids))
        slider.sliderPressed.connect(
            lambda name=species: self._on_slider_pressed(name)
        )
        slider.sliderReleased.connect(
            lambda name=species: self._on_slider_released(name)
        )
        slider.valueChanged.connect(
            lambda position, name=species: self._on_value_changed(name, position)
        )
        layout.addWidget(name_label, 0, 0)
        layout.addWidget(value_editor, 0, 1)
        layout.addWidget(slider, 1, 0, 1, 2)
        endpoint_row = QtWidgets.QHBoxLayout()
        endpoint_row.setContentsMargins(0, 0, 0, 0)
        endpoint_row.addWidget(QtWidgets.QLabel("0", container))
        endpoint_row.addStretch(1)
        maximum_label = QtWidgets.QLabel(format_slider_value(maximum), container)
        endpoint_row.addWidget(maximum_label)
        layout.addLayout(endpoint_row, 2, 0, 1, 2)
        self._content_layout.addWidget(container)
        self._rows[species] = _SpeciesSliderRow(
            container=container,
            name_label=name_label,
            slider=slider,
            value_label=value_editor,
            maximum=maximum,
            maximum_label=maximum_label,
        )

    def _on_value_changed(self, species: str, position: int) -> None:
        row = self._rows.get(species)
        if row is None or not self._target_set_ids:
            return
        value = float(position) / 10_000.0 * row.maximum
        press = self._active_presses.get(str(species))
        if press is not None and int(position) != press.start_position:
            press.moved = True
        if press is not None and not press.moved:
            self._set_value_projection(
                species,
                value,
                editor_text=None,
                emit=False,
            )
            return
        self._set_value_projection(
            species,
            value,
            editor_text=None,
            emit=True,
            interaction=(
                SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
                if str(species) in self._active_presses
                else SliderInteractionKind.EXACT_DIRECT_CHANGE
            ),
        )

    def _on_slider_pressed(self, species: str) -> None:
        species_s = str(species)
        row = self._rows.get(species_s)
        if row is not None:
            self._active_presses[species_s] = SliderPress(
                start_position=int(row.slider.sliderPosition()),
            )

    def _on_slider_released(self, species: str) -> None:
        species_s = str(species)
        press = self._active_presses.pop(species_s, None)
        value = self._values.get(species_s)
        if value is not None and press is not None and press.moved:
            self.interactionRequested.emit(
                species_s,
                float(value),
                SliderInteractionKind.EXACT_DRAG_RELEASE,
            )

    def _sync_visibility(self) -> None:
        for species, row in self._rows.items():
            row.container.setVisible(species not in self._hidden_species)

    def apply_visibility_projection(
        self, visibility: dict[str, bool] | list[tuple[str, bool]]
    ) -> None:
        """Set supplied picker choices, recalculating ranges on hidden-to-shown changes."""
        projected = dict(visibility)
        for species in tuple(self._rows):
            if species in projected:
                if bool(projected[species]):
                    if species in self._hidden_species:
                        row = self._rows[species]
                        value = self._values[species]
                        row.maximum = compute_slider_max_option_c(v=value)
                        row.maximum_label.setText(format_slider_value(row.maximum))
                        with QtCore.QSignalBlocker(row.slider):
                            row.slider.setValue(
                                self._value_to_position(value, row.maximum)
                            )
                    self._hidden_species.discard(species)
                else:
                    self._hidden_species.add(species)
        self._sync_visibility()

    def _reconcile_visibility_scope(self, names: set[str]) -> None:
        """Reconcile choices by stable canonical species identity."""
        signature = frozenset(names)
        previous = self._scope_signature
        if previous is None:
            self._hidden_species = set(names)
        else:
            preserved = names.intersection(self._hidden_species)
            self._hidden_species = preserved | (names - set(previous))
        self._scope_signature = signature

    def _update_row_projection(
        self,
        species: str,
        value: float,
        *,
        mixed: bool,
    ) -> None:
        row = self._rows.get(str(species))
        if row is None:
            return
        set_bounded_label_text(row.name_label, str(species), max_width=16777215)
        if self._editor_has_focus(species):
            return
        self._set_value_projection(
            str(species),
            value,
            editor_text=("Multiple values" if mixed else None),
            emit=False,
        )

    def _set_value_projection(
        self,
        species: str,
        value: float,
        *,
        editor_text: str | None,
        emit: bool,
        interaction: SliderInteractionKind | None = None,
    ) -> None:
        species_s = str(species)
        row = self._rows.get(species_s)
        if row is None:
            return
        value_f = float(value)
        if not math.isfinite(value_f) or value_f < 0.0:
            return
        if value_f > row.maximum:
            row.maximum = max(
                value_f,
                _scale_finite_gui_range(row.maximum, 2.0),
                1e-12,
            )
            row.maximum_label.setText(format_slider_value(row.maximum))
        self._values[species_s] = value_f
        if species_s not in self._active_presses:
            blocker = QtCore.QSignalBlocker(row.slider)
            try:
                row.slider.setValue(self._value_to_position(value_f, row.maximum))
            finally:
                del blocker
        if editor_text is not None:
            self._set_editor_text(row.value_label, editor_text)
        elif not row.value_label.hasFocus():
            self._set_editor_text(
                row.value_label,
                format_slider_value(value_f),
            )
        self._decorate_editor(species_s)
        if emit:
            self.interactionRequested.emit(species_s, value_f, interaction)

    def _editor_has_focus(self, species: str) -> bool:
        row = self._rows.get(str(species))
        return bool(row is not None and row.value_label.hasFocus())

    def _on_editor_started(self, species: str) -> None:
        row = self._rows.get(str(species))
        if row is None:
            return
        self.remember_editor_state(species)
        self._last_committed_editor_text.pop(str(species), None)
        self._decorate_editor(str(species))

    def remember_editor_state(self, species: str) -> None:
        """Remember the current value and text for subsequent local Escape."""
        self._editor_pre_edit_values[species] = self._values[species]
        self._editor_pre_edit_text[species] = self._rows[species].value_label.text()

    def _on_editor_escape(self, species: str) -> None:
        species_s = str(species)
        row = self._rows.get(species_s)
        if row is None:
            return
        value = self._editor_pre_edit_values.get(
            species_s, self._values.get(species_s, 0.0)
        )
        text = self._editor_pre_edit_text.get(
            species_s, format_slider_value(value)
        )
        self._set_value_projection(
            species_s,
            value,
            editor_text=text,
            emit=False,
        )
        self._last_committed_editor_text[species_s] = text
        self._decorate_editor(species_s)
        if text == "Multiple values":
            row.value_label.setProperty("invalidNumeric", False)
            row.value_label.setStyleSheet("")

    def _decorate_editor(self, species: str) -> None:
        row = self._rows.get(str(species))
        if row is None:
            return
        try:
            value = float(row.value_label.text().strip())
            valid = bool(math.isfinite(value) and value >= 0.0)
        except ValueError:
            valid = bool(
                row.value_label.text() == "Multiple values"
                and not row.value_label.hasFocus()
            )
        row.value_label.setProperty("invalidNumeric", not valid)
        row.value_label.setStyleSheet(
            "QLineEdit { border: 1px solid #c0392b; }"
            if not valid
            else ""
        )

    def _commit_editor(self, species: str) -> None:
        species_s = str(species)
        row = self._rows.get(species_s)
        if row is None:
            return
        raw = row.value_label.text()
        if raw == self._last_committed_editor_text.get(species_s):
            return
        try:
            value = float(raw.strip())
        except ValueError:
            self._decorate_editor(species_s)
            return
        if not math.isfinite(value) or value < 0.0:
            self._decorate_editor(species_s)
            return
        self._last_committed_editor_text[species_s] = raw
        self._set_value_projection(
            species_s,
            value,
            editor_text=raw,
            emit=True,
            interaction=SliderInteractionKind.EXACT_TYPED_COMMIT,
        )

    @staticmethod
    def _set_editor_text(editor: QtWidgets.QLineEdit, text: str) -> None:
        blocker = QtCore.QSignalBlocker(editor)
        try:
            editor.setText(str(text))
        finally:
            del blocker

    @staticmethod
    def _value_to_position(value: float, maximum: float) -> int:
        return max(0, min(10_000, round(float(value) / float(maximum) * 10_000)))
