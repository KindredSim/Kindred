"""
Variable sliders widget for interactive parameter adjustment.

Allows real-time modification of mechanism parameters (rate constants,
equilibrium constants, and scalar values) with automatic simulation updates.
"""

from __future__ import annotations

from kindred.core.quantity_units import format_unit_for_display, parse_unit
from kindred.core.validation import parse_finite_numeric_text

from functools import partial
import logging
import math
import sys
from typing import Dict, Iterable, Optional, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal

from kindred.gui.interactive_preview import SliderInteractionKind
from kindred.gui.parameter_enumeration import GuiParameterCoordinate
from kindred.gui.widgets.fine_drag_slider import FineDragSlider, NumericValueEdit, SliderPress
from kindred.gui.widgets.species_sliders import format_slider_value
from html import escape
from kindred.gui.display_name_policy import (
    reaction_text,
    CompactText,
    compact_text,
    literal_tooltip,
    scientific_symbol_html,
    scientific_symbol_spoken,
)

from ..ui_helpers import (
    configure_compact_numeric_edit,
    set_bounded_label_text,
)

logger = logging.getLogger(__name__)

__all__ = ["VariableSliders"]


_LOG_SLIDER_VALUE_MIN = 1e-12
_LOG_SLIDER_EXP_MIN = -12.0
_LOG_SLIDER_EXP_MAX = 12.0
_LOG_DOMAIN_MIN = math.log10(float.fromhex("0x0.0000000000001p-1022"))
_LOG_DOMAIN_MAX = math.log10(sys.float_info.max)


class VariableSliders(QtWidgets.QWidget):
    """
    Interactive sliders for mechanism variables.

    Features:
    - Sliders for rate constants (k1, k2, ...)
    - Sliders for equilibrium constants (Keq1, kf1, kr1, ...)
    - Sliders for scalar parameters defined in mechanism
    - Logarithmic scale for wide range of values
    - Real-time value display
    - Emits signal on value change for automatic simulation update

    Signals
    -------
    interactionRequested : Signal(str, float, object)
        Emitted with one ``SliderInteractionKind`` for every accepted event.
    """

    interactionRequested = Signal(str, float, object)
    editorValidityChanged = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize variable sliders widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        # Storage for variable values and slider metadata
        self._variables: Dict[str, float] = {}  # {name: value}
        self._sliders: Dict[str, FineDragSlider] = {}
        self._value_labels: Dict[str, QtWidgets.QWidget] = {}
        self._slider_ranges: Dict[
            str, Tuple[float, float]
        ] = {}  # log10 min/max per slider
        self._range_labels: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        self._slider_scales: Dict[str, str] = {}  # 'log' (default) or 'linear'
        self._containers: Dict[str, QtWidgets.QWidget] = {}
        self._metadata: Dict[str, GuiParameterCoordinate] = {}
        self._fine_mode: bool = False
        self._hidden_names: set[str] = set()
        self._visibility_scope_signature: object | None = None
        self._active_presses: Dict[str, SliderPress] = {}
        self._editor_pre_edit_state: Dict[str, tuple[float, str, bool]] = {}

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        header = QtWidgets.QLabel("Parameters")
        header.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(header)

        self._sliders_widget = QtWidgets.QWidget()
        self._sliders_layout = QtWidgets.QVBoxLayout(self._sliders_widget)
        self._sliders_layout.setContentsMargins(0, 0, 0, 0)
        self._sliders_layout.setSpacing(14)
        main_layout.addWidget(self._sliders_widget)
        self._sliders_layout.addStretch()

        logger.debug("VariableSliders initialized")

    def set_variables(
        self,
        variables: Dict[str, float],
        *,
        coordinates: Iterable[GuiParameterCoordinate],
    ) -> None:
        """
        Set variables and create sliders.

        Parameters
        ----------
        variables : dict
            Dictionary of {variable_name: value}
            Examples: {'k1': 0.5, 'k2': 1.0, 'Keq1': 10.0, 'kf1': 2.0, 'kr1': 0.2}
        """
        normalized_variables, normalized_metadata = (
            self._normalize_slider_snapshot(variables, coordinates)
        )
        logger.info(f"Setting {len(normalized_variables)} variables")

        current_names = set(normalized_variables)
        self._reconcile_visibility_scope(current_names, metadata=normalized_metadata)

        # Clear existing sliders; Fine mode survives the rebuild.
        self._clear_sliders()

        self._variables = normalized_variables
        self._metadata = normalized_metadata

        for name, value in normalized_variables.items():
            self._create_slider(name, value)

        self._sync_visibility_state()
        logger.info(f"Created {len(self._sliders)} sliders")

    def _clear_sliders(self) -> None:
        """Clear all existing sliders, ending any active drag."""
        for slider in self._sliders.values():
            with QtCore.QSignalBlocker(slider):
                slider.end_drag()
        for container in list(self._containers.values()):
            self._sliders_layout.removeWidget(container)
            container.deleteLater()

        self._containers.clear()
        self._sliders.clear()
        self._value_labels.clear()
        self._slider_ranges.clear()
        self._range_labels.clear()
        self._slider_scales.clear()
        self._metadata.clear()
        self._active_presses.clear()
        self._editor_pre_edit_state.clear()

    def _create_slider(self, name: str, value: float) -> None:
        """
        Create a slider for a variable.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'k1', 'Keq1', 'kf1')
        value : float
            Current value
        """
        # Container for this slider
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        meta = self._metadata[name]
        scale = "log" if meta.log10_allowed else "linear"

        # Top row: name label and editable value projection.
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        title, unit = self._label_parts(name)
        full_label = title + (f" [{unit}]" if unit else "")
        symbol_html = scientific_symbol_html(name, family=meta.family, role=meta.role)
        spoken = scientific_symbol_spoken(name, family=meta.family, role=meta.role)
        context = title
        if symbol_html != escape(name):
            symbol_label = QtWidgets.QLabel(self)
            symbol_label.setTextFormat(QtCore.Qt.RichText)
            symbol_label.setText(symbol_html)
            symbol_label.setAccessibleName(spoken)
            symbol_label.setToolTip(literal_tooltip(full_label))
            symbol_label.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
            )
            top_row.addWidget(symbol_label)
            context = reaction_text(meta.context)
        name_label = QtWidgets.QLabel()
        unit_label = None
        if unit:
            unit_label = QtWidgets.QLabel()
            unit_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            unit_label.setText(
                '<span style="white-space:nowrap">[' + format_unit_for_display(unit)
                + "]</span>"
            )
            unit_label.setAccessibleName(unit)
            unit_label.setToolTip(literal_tooltip(unit))
            unit_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
        set_bounded_label_text(
            name_label,
            context,
            max_width=16777215,
            tooltip_text=full_label,
        )
        name_label.setAccessibleName(context or full_label)
        top_row.addWidget(name_label, 1)
        if unit_label is not None:
            top_row.addWidget(unit_label)

        value_label = NumericValueEdit(format_slider_value(value))
        configure_compact_numeric_edit(value_label, characters=15)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        value_label.editStarted.connect(
            lambda item=name: self.remember_editor_state(item)
        )
        value_label.editEscape.connect(
            lambda item=name: self._on_editor_escape(item)
        )
        value_label.textChanged.connect(
            lambda _text, item=name: self._decorate_editor(item)
        )
        value_label.editingFinished.connect(
            lambda item=name: self._commit_editor(item)
        )
        value_label.setAccessibleName(f"{spoken} value")
        name_label.setBuddy(value_label)
        top_row.addWidget(value_label)

        layout.addLayout(top_row)

        # Slider
        slider = FineDragSlider()
        slider.setAccessibleName(f"{spoken} slider")
        slider.setMinimumHeight(28)
        slider.setMinimum(0)
        # 1000 steps for smooth control, 10000 in Fine mode
        slider.setMaximum(10000 if self._fine_mode else 1000)
        slider.setSingleStep(1 if self._fine_mode else 5)
        slider.setPageStep(50 if self._fine_mode else 100)
        slider.set_fine_mode(self._fine_mode)
        self._slider_scales[name] = scale

        if scale == "linear":
            self._slider_ranges[name] = (meta.minimum, meta.maximum)
        else:
            self._slider_ranges[name] = (_LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX)

        # Store slider reference early so _value_to_slider_pos can access it
        self._sliders[name] = slider

        self._ensure_range_covers(name, value, project_slider=False)
        slider_pos = self._value_to_slider_pos(name, value)
        slider.setValue(slider_pos)

        # Connect slider to update handler
        slider.valueChanged.connect(partial(self._on_slider_changed, name))
        slider.sliderPressed.connect(partial(self._on_slider_pressed, name))
        slider.sliderReleased.connect(partial(self._on_slider_released, name))

        layout.addWidget(slider)

        # Range labels
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.setSpacing(0)
        lo, hi = self._slider_ranges[name]
        if scale == "linear":
            min_label = QtWidgets.QLabel(format_slider_value(lo))
            max_label = QtWidgets.QLabel(format_slider_value(hi))
        else:
            min_label = QtWidgets.QLabel(self._format_power_of_ten(lo))
            max_label = QtWidgets.QLabel(self._format_power_of_ten(hi))
            min_label.setTextFormat(QtCore.Qt.RichText)
            max_label.setTextFormat(QtCore.Qt.RichText)
            min_label.setMinimumHeight(min_label.fontMetrics().height() + 4)
            max_label.setMinimumHeight(max_label.fontMetrics().height() + 4)
        range_layout.addWidget(min_label)
        range_layout.addStretch()
        range_layout.addWidget(max_label)
        layout.addLayout(range_layout)

        # Store references
        self._value_labels[name] = value_label
        self._range_labels[name] = (min_label, max_label)

        # Add to layout (before stretch)
        self._sliders_layout.insertWidget(self._sliders_layout.count() - 1, container)
        self._containers[name] = container
        self._apply_variable_visibility(name)

    def _on_slider_pressed(self, name: str) -> None:
        self._active_presses[name] = SliderPress(
            start_position=self._sliders[name].sliderPosition(),
        )

    def _on_slider_released(self, name: str) -> None:
        press = self._active_presses.pop(name, None)
        if press is not None and press.moved:
            self.interactionRequested.emit(
                name,
                self._variables[name],
                SliderInteractionKind.EXACT_DRAG_RELEASE,
            )

    def _label_parts(self, name: str) -> tuple[str, str]:
        meta = self._metadata[name]
        return name + (" — " + reaction_text(meta.context) if meta.context else ""), meta.unit

    def _on_slider_changed(self, name: str, slider_pos: int) -> None:
        """
        Handle slider value change.

        Parameters
        ----------
        name : str
            Variable name
        slider_pos : int
            Slider position (0-maximum)
        """
        value = self._slider_pos_to_value(name, slider_pos)

        press = self._active_presses.get(name)
        physically_pressed = press is not None
        if press is not None:
            if slider_pos != press.start_position:
                press.moved = True
            if not press.moved:
                self._set_value_projection(
                    name,
                    value,
                    editor_text=None,
                    emit=False,
                )
                return
        interaction = (
            SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
            if physically_pressed
            else SliderInteractionKind.EXACT_DIRECT_CHANGE
        )
        self._set_value_projection(
            name,
            value,
            editor_text=None,
            emit=True,
            interaction=interaction,
        )

    def get_variables(self) -> Dict[str, float]:
        """
        Get current variable values.

        Returns
        -------
        dict
            Dictionary of {variable_name: value}
        """
        return self._variables.copy()

    def render_values(
        self,
        variables: Dict[str, float],
        *,
        coordinates: Iterable[GuiParameterCoordinate],
        preserve_editor_drafts: bool,
    ) -> None:
        """Render a tab snapshot without treating widget values as canonical."""
        normalized_variables, normalized_metadata = (
            self._normalize_slider_snapshot(variables, coordinates)
        )
        same_coordinates = (
            tuple(normalized_variables) == tuple(self._variables)
            and normalized_metadata == self._metadata
        )
        if same_coordinates:
            self._reconcile_visibility_scope(
                set(normalized_variables),
                metadata=normalized_metadata,
            )
            for name, value in normalized_variables.items():
                if not preserve_editor_drafts:
                    self._set_value_projection(
                        name, value, editor_text=format_slider_value(value), emit=False
                    )
                elif name in self._active_presses:
                    # A refresh during a press leaves the handle in place.
                    self._variables[name] = value
                    self._ensure_range_covers(name, value, project_slider=False)
                elif not self._value_labels[name].hasFocus():
                    self._set_value_projection(name, value, editor_text=None, emit=False)
            self._sync_visibility_state()
        else:
            retained_editors = {}
            if preserve_editor_drafts:
                for name, editor in self._value_labels.items():
                    new = normalized_metadata.get(name)
                    # Only rejected text survives a new mechanism; valid text shows the new values.
                    if (
                        editor.property("invalidNumeric")
                        and new is not None
                        and parse_unit(self._metadata[name].unit) == parse_unit(new.unit)
                    ):
                        retained_editors[name] = (editor.text(), self._editor_pre_edit_state.get(name))
            self.set_variables(
                normalized_variables,
                coordinates=normalized_metadata.values(),
            )
            for name, (text, baseline) in retained_editors.items():
                self._set_value_projection(name, normalized_variables[name], editor_text=text, emit=False)
                self._decorate_editor(name, rejected=True)
                if baseline is not None:
                    self._editor_pre_edit_state[name] = baseline
        if not preserve_editor_drafts:
            for name in normalized_variables:
                self.remember_editor_state(name)

    def clear(self) -> None:
        """Clear physical controls, retaining visibility and Fine preferences."""
        self._clear_sliders()
        self._variables.clear()
        logger.debug("Variables cleared")

    def has_slider_entries(self) -> bool:
        return bool(self._variables)

    def visible_slider_count(self) -> int:
        return sum(1 for name in self._containers if self.variable_visible(name))

    def slider_picker_entries(self) -> list[tuple[str, CompactText, str, bool]]:
        entries = []
        for name in self._variables:
            title, unit = self._label_parts(name)
            suffix = f" [{unit}]" if unit else ""
            compact = compact_text(title, max_chars=max(1, 48 - len(suffix)))
            label = CompactText(compact.display + suffix, title + suffix, compact.was_elided)
            meta = self._metadata[name]
            symbol = scientific_symbol_html(name, family=meta.family, role=meta.role)
            markup = (
                symbol + escape(compact.display[len(name):])
                if symbol != escape(name) and compact.display.startswith(name)
                else escape(compact.display)
            )
            markup += f" [{format_unit_for_display(unit)}]" if unit else ""
            entries.append((name, label, markup, self.variable_visible(name)))
        return entries

    def variable_visible(self, name: str) -> bool:
        return name not in self._hidden_names

    def set_variable_visible(self, name: str, visible: bool) -> None:
        name_s = name
        if name_s not in self._containers:
            return
        if visible:
            self._hidden_names.discard(name_s)
        else:
            self._hidden_names.add(name_s)
        self._apply_variable_visibility(name_s)
        self._sync_visibility_state()

    def apply_visibility_projection(self, projected: Dict[str, bool]) -> None:
        """Apply one complete visibility projection without workspace callbacks."""
        for name in tuple(self._containers):
            if name in projected:
                if projected[name]:
                    self._hidden_names.discard(name)
                else:
                    self._hidden_names.add(name)
        self._sync_visibility_state()

    def set_fine_mode(self, enabled: bool) -> None:
        """
        Enable or disable fine mode for all sliders.

        Fine mode increases slider resolution from 1000 to 10000 steps,
        with smaller keyboard steps and tenfold slower mouse dragging. Current
        scientific values are preserved when switching modes.

        Parameters
        ----------
        enabled : bool
            True to enable fine mode, False for normal mode
        """
        self._fine_mode = enabled

        for name, slider in self._sliders.items():
            # Preserve the authoritative scientific value rather than
            # reconstructing it from the quantized slider position.
            val = self._variables[name]

            # Switch resolution without emitting valueChanged (Fine must not trigger simulations).
            new_max = 10000 if enabled else 1000
            with QtCore.QSignalBlocker(slider):
                slider.setMinimum(0)
                slider.setMaximum(new_max)
                slider.setSingleStep(1 if enabled else 5)
                slider.setPageStep(50 if enabled else 100)

                # Restore position for the same numeric value
                slider.setValue(self._value_to_slider_pos(name, val))
                slider.set_fine_mode(enabled)

        logger.info(f"Fine mode {'enabled' if enabled else 'disabled'}")

    def _apply_variable_visibility(self, name: str) -> None:
        self._containers[name].setVisible(self.variable_visible(name))

    def _sync_visibility_state(self) -> None:
        for name, container in self._containers.items():
            container.setVisible(self.variable_visible(name))

    def _reconcile_visibility_scope(
        self,
        names: set[str],
        *,
        metadata: Dict[str, GuiParameterCoordinate],
    ) -> None:
        """Reconcile choices by stable parameter identity, not roster order."""
        signature = tuple(
            (
                name,
                metadata[name].family,
                metadata[name].role,
            )
            for name in names
        )

        previous = self._visibility_scope_signature
        if previous is None:
            self._hidden_names = set(names)
        else:
            previous_by_name = {
                identity[0]: identity
                for identity in previous
            }
            current_by_name = {
                identity[0]: identity
                for identity in signature
            }
            preserved = {
                name
                for name in names
                if previous_by_name.get(name) == current_by_name.get(name)
            }
            self._hidden_names = (
                self._hidden_names.intersection(preserved)
                | (names - preserved)
            )
        self._visibility_scope_signature = signature

    @staticmethod
    def _normalize_slider_snapshot(
        variables: Dict[str, float],
        coordinates: Iterable[GuiParameterCoordinate],
    ) -> tuple[Dict[str, float], Dict[str, GuiParameterCoordinate]]:
        """Copy one render call's trusted projection without changing widget state."""
        normalized_variables = dict(variables)
        normalized_metadata = {item.name: item for item in coordinates}
        return normalized_variables, normalized_metadata

    def remember_editor_state(self, name: str) -> None:
        """Remember the current value and text for subsequent local Escape."""
        editor = self._value_labels[name]
        self._editor_pre_edit_state[name] = (
            self._variables[name],
            editor.text(),
            bool(editor.property("invalidNumeric")),
        )

    def _on_editor_escape(self, name: str) -> None:
        # Focus-in remembers the state before any key reaches the editor.
        value, text, rejected = self._editor_pre_edit_state[name]
        self._set_value_projection(
            name, value, editor_text=text, emit=False
        )
        self._decorate_editor(name, rejected=rejected)

    def reject_editor_value(self, name: str, accepted_value: float) -> None:
        """Restore the accepted slider value while retaining rejected text."""
        editor = self._value_labels[name]
        self._set_value_projection(
            name, accepted_value, editor_text=editor.text(), emit=False
        )
        self._decorate_editor(name, rejected=True)

    def has_rejected_editor_text(self) -> bool:
        return any(
            bool(editor.property("invalidNumeric"))
            for editor in self._value_labels.values()
        )

    def _decorate_editor(self, name: str, *, rejected: bool = False) -> None:
        editor = self._value_labels.get(name)
        if editor is None:
            return
        try:
            parse_finite_numeric_text(editor.text().strip())
            valid = not rejected
        except ValueError:
            valid = False
        was_invalid = bool(editor.property("invalidNumeric"))
        editor.setProperty("invalidNumeric", not valid)
        editor.setStyleSheet(
            "QLineEdit { border: 1px solid #c0392b; }"
            if not valid
            else ""
        )
        if was_invalid != (not valid):
            self.editorValidityChanged.emit()

    def _commit_editor(self, name: str) -> None:
        name_s = name
        editor = self._value_labels.get(name_s)
        if editor is None:
            return
        # Displayed text is a rounded readout; only text the user edited is a
        # value. Setting the text from code clears Qt's modified flag.
        if not editor.isModified():
            return
        raw = editor.text()
        try:
            value = parse_finite_numeric_text(raw.strip())
        except ValueError:
            self._decorate_editor(name_s)
            return
        self._set_value_projection(
            name_s,
            value,
            editor_text=raw,
            emit=True,
            interaction=SliderInteractionKind.EXACT_TYPED_COMMIT,
        )

    def _set_value_projection(
        self,
        name: str,
        value: float,
        *,
        editor_text: str | None,
        emit: bool,
        interaction: SliderInteractionKind | None = None,
    ) -> None:
        name_s = name
        value_f = value
        previous_range = self._slider_ranges[name_s]
        self._ensure_range_covers(name_s, value_f)
        self._variables[name_s] = value_f
        slider = self._sliders[name_s]
        with QtCore.QSignalBlocker(slider):
            slider.setValue(self._value_to_slider_pos(name_s, value_f))
        value_label = self._value_labels[name_s]
        if (
            not emit and editor_text is None
            and value_label.property("invalidNumeric")
        ):
            return
        self._set_editor_text(
            value_label,
            format_slider_value(value_f) if editor_text is None else editor_text,
        )
        self._decorate_editor(name_s)
        logger.debug("Variable projected: %s = %.12g", name_s, value_f)
        if emit:
            self.interactionRequested.emit(name_s, value_f, interaction)
            if (
                interaction is SliderInteractionKind.EXACT_TYPED_COMMIT
                and self._value_labels.get(name_s) is value_label
                and bool(value_label.property("invalidNumeric"))
            ):
                # The synchronous owner restored the accepted value. A rejected
                # typed number must not expand its slider's display range.
                self._slider_ranges[name_s] = previous_range
                self._update_range_labels(name_s)
                with QtCore.QSignalBlocker(slider):
                    slider.setValue(self._value_to_slider_pos(name_s, self._variables[name_s]))

    @staticmethod
    def _set_editor_text(editor: QtWidgets.QLineEdit, text: str) -> None:
        blocker = QtCore.QSignalBlocker(editor)
        try:
            editor.setText(text)
        finally:
            del blocker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_initial_range(self, value: float) -> Tuple[float, float]:
        """Derive an appropriate log range for a slider based on its value."""
        default_min, default_max = _LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX
        if value == 0.0:
            return default_min, default_max

        magnitude = self._log_magnitude(value)
        log_center = math.log10(magnitude)
        span = 4.0  # ±4 orders of magnitude around current value
        log_min = max(log_center - span, _LOG_DOMAIN_MIN)
        log_max = min(log_center + span, _LOG_DOMAIN_MAX)

        return log_min, log_max

    @staticmethod
    def _log_magnitude(value: float) -> float:
        """Place an exact zero on the log handle without changing its value."""
        return _LOG_SLIDER_VALUE_MIN if value == 0.0 else abs(value)

    @staticmethod
    def _linear_fraction(value: float, lower: float, upper: float) -> float:
        """Map a bounded linear value to a fraction."""
        if value <= lower:
            return 0.0
        if value >= upper:
            return 1.0
        return (value - lower) / (upper - lower)

    @staticmethod
    def _linear_value_at_fraction(
        fraction: float,
        lower: float,
        upper: float,
    ) -> float:
        """Map a slider fraction to a bounded linear value."""
        if fraction <= 0.0:
            return lower
        if fraction >= 1.0:
            return upper
        return lower + fraction * (upper - lower)

    def _slider_pos_to_value(self, name: str, slider_pos: int) -> float:
        """Convert a slider position to a concrete variable value."""
        # Qt keeps the position inside the slider's 0..maximum range.
        frac = slider_pos / self._sliders[name].maximum()
        if self._slider_scales[name] == "linear":
            lo, hi = self._slider_ranges[name]
            return self._linear_value_at_fraction(frac, lo, hi)

        log_min, log_max = self._slider_ranges[name]
        return 10 ** (log_min + frac * (log_max - log_min))

    def _value_to_slider_pos(self, name: str, value_abs: float) -> int:
        """Convert a value to slider position using stored range."""
        steps = self._sliders[name].maximum()
        if self._slider_scales[name] == "linear":
            lo, hi = self._slider_ranges[name]
            return round(self._linear_fraction(value_abs, lo, hi) * steps)

        log_min, log_max = self._slider_ranges[name]
        log_span = log_max - log_min

        magnitude = self._log_magnitude(value_abs)
        log_value = math.log10(magnitude)
        log_value = min(max(log_value, log_min), log_max)
        return round((log_value - log_min) / log_span * steps)

    def _ensure_range_covers(
        self,
        name: str,
        value: float,
        *,
        project_slider: bool = True,
    ) -> None:
        """Expand the GUI-only range if needed to include the given value."""
        if self._slider_scales[name] == "linear":
            value_f = value
            linear_min, linear_max = self._slider_ranges[name]
            if value_f < linear_min:
                linear_min = value_f
            elif value_f > linear_max:
                linear_max = value_f

            self._slider_ranges[name] = (linear_min, linear_max)
            self._update_range_labels(name)
            if project_slider:
                slider = self._sliders[name]
                with QtCore.QSignalBlocker(slider):
                    slider.setValue(self._value_to_slider_pos(name, value_f))
            return

        log_min, log_max = self._slider_ranges[name]
        magnitude = self._log_magnitude(value)
        log_value = math.log10(magnitude)

        needs_expand = log_value < log_min or log_value > log_max
        if not needs_expand:
            return

        new_log_min, new_log_max = self._compute_initial_range(value)
        if (new_log_min, new_log_max) == (log_min, log_max):
            return
        self._slider_ranges[name] = (new_log_min, new_log_max)
        self._update_range_labels(name)

        if project_slider:
            slider = self._sliders[name]
            with QtCore.QSignalBlocker(slider):
                slider.setValue(self._value_to_slider_pos(name, value))

    def _update_range_labels(self, name: str) -> None:
        """Refresh the displayed min/max labels for a slider."""
        if name not in self._range_labels:
            # Creation expands the range before its labels exist.
            return
        min_label, max_label = self._range_labels[name]
        if self._slider_scales[name] == "linear":
            linear_min, linear_max = self._slider_ranges[name]
            min_label.setText(format_slider_value(linear_min))
            max_label.setText(format_slider_value(linear_max))
            return
        log_min, log_max = self._slider_ranges[name]
        min_label.setText(self._format_power_of_ten(log_min))
        max_label.setText(self._format_power_of_ten(log_max))

    @staticmethod
    def _format_power_of_ten(log_value: float) -> str:
        """Format a base-10 exponent as a compact string."""
        exponent = int(round(log_value))
        if abs(log_value - exponent) < 1e-6:
            return f"10<sup>{exponent}</sup>"
        value = 10**log_value
        if value >= 1000 or value < 0.001:
            return f"{value:.1e}"
        return f"{value:.3g}"
