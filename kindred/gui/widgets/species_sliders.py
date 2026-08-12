"""Species slider snapshot projection; target selection lives outside the widget."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.interactive_preview import SliderInteractionKind
from kindred.gui.species_sliders_logic import (
    compute_slider_max_option_c,
    _scale_finite_gui_range,
)

__all__ = ["BatchSpeciesSliders", "format_slider_value"]


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


class _GrooveJumpSlider(QtWidgets.QSlider):
    """Native slider whose groove accepts an exact click-anywhere jump."""

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.orientation() == QtCore.Qt.Orientation.Horizontal
        ):
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )
            if not handle.contains(event.position().toPoint()):
                span = max(1, groove.width() - handle.width())
                position = int(
                    round(event.position().x() - groove.x() - handle.width() / 2)
                )
                value = QtWidgets.QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    max(0, min(position, span)),
                    span,
                    option.upsideDown,
                )
                self.setValue(value)
                event.accept()
                return
        super().mousePressEvent(event)


@dataclass(slots=True)
class _SpeciesSliderRow:
    container: QtWidgets.QWidget
    name_label: QtWidgets.QLabel
    slider: QtWidgets.QSlider
    value_label: QtWidgets.QWidget
    maximum: float
    maximum_label: QtWidgets.QLabel


class _NumericValueEdit(QtWidgets.QLineEdit):
    """A raw numeric draft editor with local Escape/focus notifications."""

    editStarted = QtCore.Signal()
    editEscape = QtCore.Signal()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self.editStarted.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.editEscape.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class BatchSpeciesSliders(QtWidgets.QWidget):
    interactionRequested = QtCore.Signal(str, float, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._rows: dict[str, _SpeciesSliderRow] = {}
        self._values: dict[str, float] = {}
        self._display_names: dict[str, str] = {}
        self._hidden_species: set[str] = set()
        self._scope_signature: object | None = None
        self._target_set_ids: tuple[str, ...] = ()
        self._active_drag_species: set[str] = set()
        self._editor_pre_edit_values: dict[str, float] = {}
        self._editor_pre_edit_text: dict[str, str] = {}
        self._last_committed_editor_text: dict[str, str] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        if not embedded:
            title = QtWidgets.QLabel("Initial-concentration sliders", self)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            layout.addWidget(title)
        self._target_label = QtWidgets.QLabel("Slider targets: none", self)
        layout.addWidget(self._target_label)
        self._content = QtWidgets.QWidget(self)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content)
        layout.addStretch(1)

    def render_values(
        self,
        values: Mapping[str, float],
        *,
        target_set_ids: Sequence[str],
        target_display_names: Sequence[str] = (),
        mixed_species: Sequence[str] = (),
        display_names: Mapping[str, str] | None = None,
        preserve_visibility: bool = True,
        visibility_scope_signature: object | None = None,
    ) -> None:
        normalized: dict[str, float] = {}
        normalized_display_names: dict[str, str] = {}
        supplied_names = dict(display_names or {})
        for raw_name, raw_value in values.items():
            name = str(raw_name)
            display = str(supplied_names.get(name, name)).strip()
            value = float(raw_value)
            if not name or not display:
                raise ValueError("Species Slider names must be nonblank.")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Species Slider value for {display!r} must be finite "
                    "and nonnegative."
                )
            normalized[name] = value
            normalized_display_names[name] = display
        self._reconcile_visibility_scope(
            set(normalized),
            visibility_scope_signature,
            preserve_visibility=preserve_visibility,
        )
        self._target_set_ids = tuple(str(item) for item in target_set_ids)
        target_names = tuple(str(item).strip() for item in target_display_names)
        if self._target_set_ids and (
            len(target_names) != len(self._target_set_ids)
            or any(not name for name in target_names)
        ):
            raise ValueError(
                "Slider target display names must be nonblank and exact."
            )
        self._values = normalized
        self._display_names = normalized_display_names
        mixed = {str(name) for name in mixed_species}
        target_text = (
            ", ".join(target_names)
            if self._target_set_ids
            else "none"
        )
        self._target_label.setText(f"Slider targets: {target_text}")
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
                    display_name=normalized_display_names[species],
                    mixed=species in mixed,
                )
        self._sync_visibility()

    def slider_picker_entries(self) -> list[tuple[str, str, bool]]:
        return [
            (
                name,
                self._display_names[name],
                name not in self._hidden_species,
            )
            for name in self._values
        ]

    def set_species_visible(self, species: str, visible: bool) -> None:
        name = str(species)
        if name not in self._values:
            return
        if visible:
            self._hidden_species.discard(name)
        else:
            self._hidden_species.add(name)
        self._sync_visibility()

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

    def clear(self, *, preserve_visibility: bool = True) -> None:
        if preserve_visibility:
            self._clear_rows()
            self._values.clear()
            self._display_names.clear()
            self._target_set_ids = ()
            self._target_label.setText("Slider targets: none")
            self._sync_visibility()
            return
        self.render_values({}, target_set_ids=(), preserve_visibility=False)

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._content_layout.removeWidget(row.container)
            row.container.deleteLater()
        self._rows.clear()
        self._active_drag_species.clear()
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
        name_label = QtWidgets.QLabel(display_name, container)
        if mixed:
            initial_text = "Multiple values"
        else:
            initial_text = format_slider_value(value)
        value_editor = _NumericValueEdit(initial_text, container)
        value_editor.setMinimumWidth(100)
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
        slider = _GrooveJumpSlider(
            QtCore.Qt.Orientation.Horizontal,
            container,
        )
        slider.setRange(0, 10_000)
        maximum = compute_slider_max_option_c(
            v=value, row_max=max(self._values.values(), default=value)
        )
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
        interaction = (
            SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
            if str(species) in self._active_drag_species
            else SliderInteractionKind.EXACT_DRAG_RELEASE
        )
        self._set_value_projection(
            species,
            value,
            editor_text=None,
            emit=True,
            interaction=interaction,
        )

    def _on_slider_pressed(self, species: str) -> None:
        self._active_drag_species.add(str(species))

    def _on_slider_released(self, species: str) -> None:
        species_s = str(species)
        self._active_drag_species.discard(species_s)
        value = self._values.get(species_s)
        if value is not None:
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
        """Apply one complete visibility projection without workspace callbacks."""
        projected = dict(visibility)
        for species in tuple(self._rows):
            if species in projected:
                if bool(projected[species]):
                    self._hidden_species.discard(species)
                else:
                    self._hidden_species.add(species)
        self._sync_visibility()

    def _reconcile_visibility_scope(
        self,
        names: set[str],
        supplied_signature: object | None,
        *,
        preserve_visibility: bool,
    ) -> None:
        """Reconcile choices by stable canonical species identity."""
        signature = frozenset(
            str(name)
            for name in (names if supplied_signature is None else supplied_signature)
        )
        if signature != names:
            raise ValueError(
                "Species visibility scope names must match slider values."
            )
        previous = self._scope_signature
        if not preserve_visibility or previous is None:
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
        row.name_label.setText(self._display_names.get(str(species), str(species)))
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
        if species_s not in self._active_drag_species:
            blocker = QtCore.QSignalBlocker(row.slider)
            try:
                row.slider.setValue(self._value_to_position(value_f, row.maximum))
            finally:
                del blocker
        if isinstance(row.value_label, _NumericValueEdit):
            if editor_text is not None:
                self._set_editor_text(row.value_label, editor_text)
            elif not row.value_label.hasFocus():
                self._set_editor_text(
                    row.value_label,
                    format_slider_value(value_f),
                )
            self._decorate_editor(species_s)
        if emit:
            if not isinstance(interaction, SliderInteractionKind):
                raise TypeError("emitted Slider values require an interaction kind.")
            self.interactionRequested.emit(species_s, value_f, interaction)

    def _editor_has_focus(self, species: str) -> bool:
        row = self._rows.get(str(species))
        return bool(
            row is not None
            and isinstance(row.value_label, _NumericValueEdit)
            and row.value_label.hasFocus()
        )

    def _on_editor_started(self, species: str) -> None:
        row = self._rows.get(str(species))
        if row is None or not isinstance(row.value_label, _NumericValueEdit):
            return
        self._editor_pre_edit_values[str(species)] = float(
            self._values.get(str(species), 0.0)
        )
        self._editor_pre_edit_text[str(species)] = row.value_label.text()
        self._last_committed_editor_text.pop(str(species), None)
        self._decorate_editor(str(species))

    def _on_editor_escape(self, species: str) -> None:
        species_s = str(species)
        row = self._rows.get(species_s)
        if row is None or not isinstance(row.value_label, _NumericValueEdit):
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
        if row is None or not isinstance(row.value_label, _NumericValueEdit):
            return
        try:
            value = float(row.value_label.text().strip())
            valid = bool(math.isfinite(value) and value >= 0.0)
        except (TypeError, ValueError, OverflowError):
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
        if row is None or not isinstance(row.value_label, _NumericValueEdit):
            return
        raw = row.value_label.text()
        if raw == self._last_committed_editor_text.get(species_s):
            return
        try:
            value = float(raw.strip())
        except (TypeError, ValueError, OverflowError):
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
