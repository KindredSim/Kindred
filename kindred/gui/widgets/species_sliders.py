"""Species slider snapshot projection; target selection lives outside the widget."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.species_sliders_logic import compute_slider_max_option_c

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
    slider: QtWidgets.QSlider
    value_label: QtWidgets.QLabel
    maximum: float


class BatchSpeciesSliders(QtWidgets.QWidget):
    speciesEdited = QtCore.Signal(str, float)
    speciesDragStarted = QtCore.Signal(str)
    speciesDragFinished = QtCore.Signal(str)

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
        same_scope = (
            preserve_visibility
            and visibility_scope_signature is not None
            and visibility_scope_signature == self._scope_signature
        )
        if same_scope:
            self._hidden_species.intersection_update(normalized)
        else:
            self._hidden_species.clear()
        self._scope_signature = visibility_scope_signature
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
        self._clear_rows()
        target_text = (
            ", ".join(target_names)
            if self._target_set_ids
            else "none"
        )
        self._target_label.setText(f"Slider targets: {target_text}")
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

    def slider_species_names(self) -> list[str]:
        return list(self._values)

    def species_visible(self, species: str) -> bool:
        return str(species) in self._values and str(species) not in self._hidden_species

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
        return sum(row.container.isVisible() for row in self._rows.values())

    def has_visible_entries(self) -> bool:
        return any(name not in self._hidden_species for name in self._rows)

    def clear(self) -> None:
        self.render_values({}, target_set_ids=())

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._content_layout.removeWidget(row.container)
            row.container.deleteLater()
        self._rows.clear()

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
        value_label = QtWidgets.QLabel(
            "Multiple values" if mixed else format_slider_value(value),
            container,
        )
        value_label.setMinimumWidth(80)
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
            lambda name=species: self.speciesDragStarted.emit(name)
        )
        slider.sliderReleased.connect(
            lambda name=species: self.speciesDragFinished.emit(name)
        )
        slider.valueChanged.connect(
            lambda position, name=species: self._on_value_changed(name, position)
        )
        layout.addWidget(name_label, 0, 0)
        layout.addWidget(value_label, 0, 1)
        layout.addWidget(slider, 1, 0, 1, 2)
        endpoint_row = QtWidgets.QHBoxLayout()
        endpoint_row.setContentsMargins(0, 0, 0, 0)
        endpoint_row.addWidget(QtWidgets.QLabel("0", container))
        endpoint_row.addStretch(1)
        endpoint_row.addWidget(
            QtWidgets.QLabel(format_slider_value(maximum), container)
        )
        layout.addLayout(endpoint_row, 2, 0, 1, 2)
        self._content_layout.addWidget(container)
        self._rows[species] = _SpeciesSliderRow(
            container=container,
            slider=slider,
            value_label=value_label,
            maximum=maximum,
        )

    def _on_value_changed(self, species: str, position: int) -> None:
        row = self._rows.get(species)
        if row is None or not self._target_set_ids:
            return
        value = float(position) / 10_000.0 * row.maximum
        self._values[species] = value
        row.value_label.setText(format_slider_value(value))
        self.speciesEdited.emit(species, value)

    def _sync_visibility(self) -> None:
        for species, row in self._rows.items():
            row.container.setVisible(species not in self._hidden_species)

    @staticmethod
    def _value_to_position(value: float, maximum: float) -> int:
        return max(0, min(10_000, round(float(value) / float(maximum) * 10_000)))
