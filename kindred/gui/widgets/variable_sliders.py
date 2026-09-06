"""
Variable sliders widget for interactive parameter adjustment.

Allows real-time modification of mechanism parameters (rate constants,
equilibrium constants, and scalar values) with automatic simulation updates.
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
import logging
import math
import sys
from typing import Dict, Iterable, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

from kindred.gui.interactive_preview import SliderInteractionKind
from html import escape
from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken, literal_tooltip

from ..ui_helpers import (
    configure_compact_numeric_edit,
    set_bounded_label_text,
    make_placeholder_label,
    make_scroll_area,
)

logger = logging.getLogger(__name__)

__all__ = ["VariableSliders"]


@dataclass(slots=True)
class _SliderPress:
    """Single physical press record owned by one Slider widget."""

    start_position: int
    moved: bool = False

_LOG_SLIDER_VALUE_MIN = 1e-12
_LOG_SLIDER_EXP_MIN = -12.0
_LOG_SLIDER_EXP_MAX = 12.0
_LOG_DOMAIN_MIN = math.log10(float.fromhex("0x0.0000000000001p-1022"))
_LOG_DOMAIN_MAX = math.log10(sys.float_info.max)


class _NumericValueEdit(QtWidgets.QLineEdit):
    """A raw numeric draft editor with local Escape/focus notifications."""

    editStarted = Signal()
    editEscape = Signal()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self.editStarted.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.editEscape.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _GrooveJumpSlider(QtWidgets.QSlider):
    """Horizontal slider with native click-anywhere groove behavior."""

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation | QtWidgets.QWidget = (
            QtCore.Qt.Orientation.Horizontal
        ),
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        if isinstance(orientation, QtWidgets.QWidget) and parent is None:
            parent = orientation
            orientation = QtCore.Qt.Orientation.Horizontal
        super().__init__(orientation, parent)
        self._fine_drag_mode = False
        self._fine_drag_origin: tuple[float, int, int, int, int] | None = None

    def set_fine_drag_mode(self, enabled: bool) -> None:
        """Reduce handle-drag sensitivity without changing slider semantics."""
        self._fine_drag_mode = bool(enabled)

    def _slider_geometry(
        self,
    ) -> tuple[QtWidgets.QStyleOptionSlider, QtCore.QRect, QtCore.QRect, int]:
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
        return option, groove, handle, max(1, groove.width() - handle.width())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._fine_drag_origin = None
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.orientation() == QtCore.Qt.Orientation.Horizontal
        ):
            option, groove, handle, span = self._slider_geometry()
            if handle.contains(event.position().toPoint()):
                self._fine_drag_origin = (
                    float(event.position().x()),
                    int(self.sliderPosition()),
                    span,
                    int(self.minimum()),
                    int(self.maximum()),
                )
            else:
                position = int(
                    round(event.position().x() - groove.x() - handle.width() / 2)
                )
                self.setValue(
                    QtWidgets.QStyle.sliderValueFromPosition(
                        self.minimum(),
                        self.maximum(),
                        max(0, min(position, span)),
                        span,
                        option.upsideDown,
                    )
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        origin = self._fine_drag_origin
        if (
            self._fine_drag_mode
            and origin is not None
            and self.isSliderDown()
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            origin_x, origin_value, span, minimum, maximum = origin
            option, _, _, _ = self._slider_geometry()
            pointer_delta = float(event.position().x()) - origin_x
            value_delta = pointer_delta * (maximum - minimum) / (span * 10.0)
            slider_position = origin_value + int(
                round(-value_delta if option.upsideDown else value_delta)
            )
            slider_position = max(minimum, min(maximum, slider_position))
            self.setSliderPosition(slider_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._fine_drag_origin = None
        super().mouseReleaseEvent(event)


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
    fineModeReset = Signal(bool)

    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, *, embedded: bool = False
    ):
        """
        Initialize variable sliders widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._embedded = bool(embedded)

        # Storage for variable values and slider metadata
        self._variables: Dict[str, float] = {}  # {name: value}
        self._sliders: Dict[str, QtWidgets.QSlider] = {}
        self._value_labels: Dict[str, QtWidgets.QWidget] = {}
        self._slider_ranges: Dict[
            str, Tuple[float, float]
        ] = {}  # log10 min/max per slider
        self._range_labels: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        self._slider_signs: Dict[str, float] = {}
        self._slider_scales: Dict[str, str] = {}  # 'log' (default) or 'linear'
        self._containers: Dict[str, QtWidgets.QWidget] = {}
        self._metadata: Dict[str, Dict[str, object]] = {}
        self._last_valid_values: Dict[str, float] = {}
        self._fine_mode: bool = False
        self._slider_callbacks: Dict[str, Dict[str, object]] = {}
        self._hidden_names: set[str] = set()
        self._visibility_scope_signature: object | None = None
        self._placeholder: QtWidgets.QLabel | None = None
        self._hidden_placeholder: QtWidgets.QLabel | None = None
        self._active_presses: Dict[str, _SliderPress] = {}
        self._editor_pre_edit_values: Dict[str, float] = {}
        self._editor_pre_edit_text: Dict[str, str] = {}
        self._last_committed_editor_text: Dict[str, str] = {}

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        if self._embedded:
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
        else:
            main_layout.setContentsMargins(4, 4, 4, 4)
            main_layout.setSpacing(4)

            # Header
            header = QtWidgets.QLabel("Variable Sliders")
            header_font = header.font()
            header_font.setBold(True)
            header.setFont(header_font)
            main_layout.addWidget(header)

            # Description
            desc = QtWidgets.QLabel(
                "Adjust mechanism parameters and see simulation update automatically"
            )
            desc.setWordWrap(True)
            desc_font = desc.font()
            desc_font.setPointSize(desc_font.pointSize() - 1)
            desc.setFont(desc_font)
            main_layout.addWidget(desc)

        if self._embedded:
            header = QtWidgets.QLabel("Parameters")
            header.setStyleSheet("font-weight: bold;")
            main_layout.addWidget(header)

        self._sliders_widget = QtWidgets.QWidget()
        self._sliders_layout = QtWidgets.QVBoxLayout(self._sliders_widget)
        self._sliders_layout.setContentsMargins(0, 0, 0, 0)
        self._sliders_layout.setSpacing(14)

        if self._embedded:
            main_layout.addWidget(self._sliders_widget)
        else:
            scroll = make_scroll_area(self)
            scroll.setWidget(self._sliders_widget)
            main_layout.addWidget(scroll)

            # Placeholder message
            self._placeholder = make_placeholder_label(
                "No variables defined.\nParse a mechanism to see sliders."
            )
            self._sliders_layout.addWidget(self._placeholder)
            self._hidden_placeholder = make_placeholder_label(
                "All mechanism sliders hidden by picker."
            )
            self._hidden_placeholder.hide()
            self._sliders_layout.addWidget(self._hidden_placeholder)

        self._sliders_layout.addStretch()

        logger.debug("VariableSliders initialized")

    @property
    def fine_mode_enabled(self) -> bool:
        """Return the current mechanism-slider Fine projection state."""
        return bool(self._fine_mode)

    def set_variables(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = False,
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
            self._normalize_slider_snapshot(variables, metadata)
        )
        logger.info(f"Setting {len(normalized_variables)} variables")

        current_names = set(normalized_variables)
        self._reconcile_visibility_scope(
            current_names,
            metadata=normalized_metadata,
            preserve_visibility=preserve_visibility,
        )

        self._reset_fine_mode_quietly()

        # Clear existing sliders
        self._clear_sliders()

        self._variables = normalized_variables
        self._metadata = normalized_metadata

        if not normalized_variables:
            if self._placeholder is not None:
                self._placeholder.show()
            logger.debug("No variables to display")
            return

        if self._placeholder is not None:
            self._placeholder.hide()

        for name, value in normalized_variables.items():
            self._create_slider(name, value)

        self._sync_visibility_state()
        logger.info(f"Created {len(self._sliders)} sliders")

    def _clear_sliders(self) -> None:
        """Clear all existing sliders."""
        for container in list(self._containers.values()):
            self._sliders_layout.removeWidget(container)
            container.deleteLater()

        self._containers.clear()
        self._sliders.clear()
        self._slider_callbacks.clear()
        self._value_labels.clear()
        self._slider_ranges.clear()
        self._range_labels.clear()
        self._slider_signs.clear()
        self._slider_scales.clear()
        self._metadata.clear()
        self._last_valid_values.clear()
        self._active_presses.clear()
        self._editor_pre_edit_values.clear()
        self._editor_pre_edit_text.clear()
        self._last_committed_editor_text.clear()

        if self._placeholder is not None:
            self._placeholder.show()
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.hide()

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

        meta = self._metadata.get(name) or {}
        scale = meta["scale"]

        # Top row: name label and writable/read-only value projection.
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        full_label = self._format_label_text(name)
        symbol_html = scientific_symbol_html(name, family=meta["family"], role=meta["role"])
        spoken = scientific_symbol_spoken(name, family=meta["family"], role=meta["role"])
        context = full_label
        if symbol_html != escape(name) and full_label.startswith(name):
            symbol_label = QtWidgets.QLabel(self)
            symbol_label.setTextFormat(QtCore.Qt.RichText)
            symbol_label.setText(symbol_html)
            symbol_label.setAccessibleName(spoken)
            symbol_label.setToolTip(literal_tooltip(full_label))
            symbol_label.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
            )
            top_row.addWidget(symbol_label)
            context = full_label[len(name):].removeprefix(" — ").lstrip()
        name_label = QtWidgets.QLabel()
        set_bounded_label_text(
            name_label, context, max_width=16777215, tooltip_text=full_label
        )
        top_row.addWidget(name_label, 1)

        if meta.get("editable") is False:
            value_label: QtWidgets.QWidget = QtWidgets.QLabel(
                self._format_value(value)
            )
            value_label.setAlignment(QtCore.Qt.AlignRight)
        else:
            value_editor = _NumericValueEdit(self._format_value(value))
            configure_compact_numeric_edit(value_editor, characters=15)
            value_editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            value_editor.editStarted.connect(
                lambda item=name: self._on_editor_started(item)
            )
            value_editor.editEscape.connect(
                lambda item=name: self._on_editor_escape(item)
            )
            value_editor.textChanged.connect(
                lambda _text, item=name: self._decorate_editor(item)
            )
            value_editor.editingFinished.connect(
                lambda item=name: self._commit_editor(item)
            )
            value_label = value_editor
        value_label.setAccessibleName(f"{spoken} value")
        if isinstance(value_label, QtWidgets.QLineEdit):
            name_label.setBuddy(value_label)
        top_row.addWidget(value_label)

        layout.addLayout(top_row)

        # Slider
        slider = _GrooveJumpSlider(QtCore.Qt.Horizontal)
        slider.setAccessibleName(f"{spoken} slider")
        slider.setMinimumHeight(28)
        slider.setMinimum(0)
        slider.setMaximum(1000)  # 1000 steps for smooth control
        slider.setSingleStep(5)
        slider.setPageStep(100)
        self._slider_scales[name] = "linear" if scale == "linear" else "log"

        if scale == "linear":
            try:
                lo = (
                    float(meta.get("min"))
                    if meta.get("min") is not None
                    else float(value) - 10.0
                )
            except (ArithmeticError, TypeError, ValueError):
                lo = float(value) - 10.0
            try:
                hi = (
                    float(meta.get("max"))
                    if meta.get("max") is not None
                    else float(value) + 10.0
                )
            except (ArithmeticError, TypeError, ValueError):
                hi = float(value) + 10.0
            self._slider_ranges[name] = (lo, hi)
            self._slider_signs[name] = 1.0
        else:
            sign = 1.0 if value >= 0 else -1.0
            self._slider_signs[name] = sign
            self._slider_ranges[name] = (_LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX)

        # Store slider reference early so _value_to_slider_pos can access it
        self._sliders[name] = slider

        value_f = float(value)
        self._ensure_range_covers(name, value_f, project_slider=False)
        slider_pos = self._value_to_slider_pos(name, value_f)
        slider.setValue(slider_pos)
        self._last_valid_values[name] = value_f

        # Connect slider to update handler
        callbacks = self._slider_callbacks.setdefault(str(name), {})
        cb_value = partial(self._on_slider_changed, str(name))
        cb_pressed = partial(self._on_slider_pressed, str(name))
        cb_released = partial(self._on_slider_released, str(name))
        callbacks["valueChanged"] = cb_value
        callbacks["sliderPressed"] = cb_pressed
        callbacks["sliderReleased"] = cb_released
        slider.valueChanged.connect(cb_value)
        slider.sliderPressed.connect(cb_pressed)
        slider.sliderReleased.connect(cb_released)

        layout.addWidget(slider)

        # Derived / read-only variables (computed from other parameters)
        if meta.get("editable") is False:
            slider.setEnabled(False)
            tip = "Derived parameter (computed). Edit its source parameters instead."
            slider.setToolTip(tip)
            name_label.setToolTip(full_label + "\n" + tip)
            value_label.setToolTip(tip)
            name_label.setStyleSheet("font-style: italic;")

        # Range labels
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.setSpacing(0)
        if scale == "linear":
            lo, hi = self._slider_ranges[name]
            min_label = QtWidgets.QLabel(self._format_value(float(lo)))
            max_label = QtWidgets.QLabel(self._format_value(float(hi)))
        else:
            min_label = QtWidgets.QLabel(self._format_power_of_ten(_LOG_SLIDER_EXP_MIN))
            max_label = QtWidgets.QLabel(self._format_power_of_ten(_LOG_SLIDER_EXP_MAX))
            min_label.setTextFormat(QtCore.Qt.RichText)
            max_label.setTextFormat(QtCore.Qt.RichText)
            min_label.setMinimumHeight(min_label.fontMetrics().height() + 4)
            max_label.setMinimumHeight(max_label.fontMetrics().height() + 4)
        range_layout.addWidget(min_label)
        if scale != "linear":
            for mid_text in ["10<sup>-6</sup>", "1", "10<sup>6</sup>"]:
                range_layout.addStretch()
                mid_label = QtWidgets.QLabel(mid_text)
                mid_label.setTextFormat(QtCore.Qt.RichText)
                mid_label.setMinimumHeight(mid_label.fontMetrics().height() + 4)
                range_layout.addWidget(mid_label)
        range_layout.addStretch()
        range_layout.addWidget(max_label)
        layout.addLayout(range_layout)

        # Store references
        self._sliders[name] = slider
        self._value_labels[name] = value_label
        self._range_labels[name] = (min_label, max_label)

        # Add to layout (before stretch)
        self._sliders_layout.insertWidget(self._sliders_layout.count() - 1, container)
        self._containers[name] = container
        self._apply_variable_visibility(name)

    def _on_slider_pressed(self, name: str) -> None:
        name_s = str(name)
        slider = self._sliders.get(name_s)
        if slider is not None:
            self._active_presses[name_s] = _SliderPress(
                start_position=int(slider.sliderPosition()),
            )

    def _on_slider_released(self, name: str) -> None:
        name_s = str(name)
        press = self._active_presses.pop(name_s, None)
        value = self._variables.get(name_s)
        if value is not None and press is not None and press.moved:
            self.interactionRequested.emit(
                name_s,
                float(value),
                SliderInteractionKind.EXACT_DRAG_RELEASE,
            )

    def _format_label_text(self, name: str) -> str:
        meta = self._metadata.get(name)
        derived_suffix = ""
        if meta and bool(meta.get("derived")):
            derived_suffix = " (derived)"
        unit_suffix = ""
        if meta and meta.get("scale") == "linear" and meta.get("unit"):
            unit_suffix = f" [{meta.get('unit')}]"
        if meta and isinstance(meta.get("label"), str):
            display_name = str(meta["label"]).strip()
            if not display_name:
                raise ValueError("Slider display names must be nonblank.")
            if meta["family"] in {"Constant", "Arrhenius", "Eyring", "reverse"} and meta["role"] in {"k", "kf", "kr", "Keq"}:
                # Only admitted reaction context gets display arrows. Authored
                # scalar labels and canonical metadata remain literal.
                identity, separator, context = display_name.partition(" — ")
                display_name = identity + separator + context.replace(" <-> ", " ⇌ ").replace(" -> ", " → ")
            return f"{display_name}{unit_suffix}{derived_suffix}"
        return f"{name}{unit_suffix}{derived_suffix}"

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
        if name not in self._sliders:
            return

        try:
            value = float(self._slider_pos_to_value(name, slider_pos))
        except (ArithmeticError, TypeError, ValueError) as exc:
            fallback = self._last_valid_values.get(name, self._variables.get(name, 0.0))
            try:
                value = float(fallback)
            except (ArithmeticError, TypeError, ValueError):
                value = 0.0

            if not math.isfinite(value):
                scale = self._slider_scales.get(name, "log")
                if scale == "log":
                    sign = self._slider_signs.get(name, 1.0)
                    value = (1.0 if sign >= 0 else -1.0) * _LOG_SLIDER_VALUE_MIN
                else:
                    value = 0.0

            slider = self._sliders.get(name)
            if slider is not None:
                try:
                    restore_pos = self._value_to_slider_pos(name, value)
                except (ArithmeticError, TypeError, ValueError):
                    restore_pos = max(
                        slider.minimum(), min(int(slider_pos), slider.maximum())
                    )
                with QtCore.QSignalBlocker(slider):
                    slider.setValue(restore_pos)

            self._set_value_projection(name, value, editor_text=None, emit=False)

            logger.warning(
                "Slider conversion failed for %s at pos=%s (%s); restored last valid value %.12g",
                str(name),
                int(slider_pos),
                type(exc).__name__,
                float(value),
            )
            return

        slider = self._sliders.get(name)
        physically_pressed = str(name) in self._active_presses
        if physically_pressed:
            press = self._active_presses.get(str(name))
            if press is not None and int(slider_pos) != press.start_position:
                press.moved = True
            if press is not None and not press.moved:
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

    def _format_value(self, value: float) -> str:
        """
        Format value for display.

        Parameters
        ----------
        value : float
            Value to format

        Returns
        -------
        str
            Formatted string
        """
        if value == 0:
            return "0"
        elif abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.3e}"
        elif abs(value) < 0.1:
            return f"{value:.4f}"
        elif abs(value) < 10:
            return f"{value:.3f}"
        else:
            return f"{value:.2f}"

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
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = True,
    ) -> None:
        """Render a tab snapshot without treating widget values as canonical."""
        normalized_variables, normalized_metadata = (
            self._normalize_slider_snapshot(variables, metadata)
        )
        same_coordinates = (
            tuple(normalized_variables) == tuple(self._variables)
            and normalized_metadata == self._metadata
            and all(name in self._containers for name in normalized_variables)
        )
        if same_coordinates:
            self._reconcile_visibility_scope(
                set(normalized_variables),
                metadata=normalized_metadata,
                preserve_visibility=preserve_visibility,
            )
            for name, value in normalized_variables.items():
                if name in self._active_presses:
                    self.update_variable_readout(name, value)
                elif self._editor_has_focus(name):
                    continue
                else:
                    self.update_variable(name, value)
            self._sync_visibility_state()
            return
        self.set_variables(
            normalized_variables,
            metadata=normalized_metadata,
            preserve_visibility=preserve_visibility,
        )

    def update_variable(self, name: str, value: float) -> None:
        """
        Update a variable value programmatically.

        Parameters
        ----------
        name : str
            Variable name
        value : float
            New value
        """
        if name not in self._variables:
            logger.warning(f"Cannot update unknown variable: {name}")
            return

        if self._editor_has_focus(name):
            return
        self._set_value_projection(
            name,
            float(value),
            editor_text=None,
            emit=False,
        )

    def update_variable_readout(self, name: str, value: float) -> None:
        """
        Update a variable value programmatically without moving the slider handle.

        This is intended for derived/read-only parameter updates during slider-triggered
        refreshes, where changing slider ranges/positions can make handles appear to
        jump/snap.
        """
        if name not in self._variables:
            logger.warning(f"Cannot update unknown variable: {name}")
            return

        value_f = float(value)
        self._variables[name] = value_f
        try:
            self._last_valid_values[name] = value_f
        except (TypeError, ValueError, OverflowError):
            self._last_valid_values.pop(name, None)
        if self._slider_scales.get(name, "log") == "log":
            self._slider_signs[name] = 1.0 if value_f >= 0 else -1.0
        self._ensure_range_covers(name, value_f, project_slider=False)

        value_label = self._value_labels.get(name)
        if isinstance(value_label, QtWidgets.QLabel):
            value_label.setText(self._format_value(value_f))

        logger.debug(f"Variable readout updated: {name} = {value}")

    def clear(self, *, preserve_visibility: bool = True) -> None:
        """Clear physical controls, optionally retaining visibility preference."""
        self._reset_fine_mode_quietly()
        self._clear_sliders()
        self._variables.clear()
        self._slider_ranges.clear()
        self._slider_signs.clear()
        self._slider_scales.clear()
        self._range_labels.clear()
        self._last_valid_values.clear()
        if not preserve_visibility:
            self._hidden_names.clear()
            self._visibility_scope_signature = None
        if self._placeholder is not None:
            self._placeholder.show()
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.hide()
        logger.debug("Variables cleared")

    def has_slider_entries(self) -> bool:
        return bool(self._variables)

    def visible_slider_count(self) -> int:
        return sum(1 for name in self._containers if self.variable_visible(name))

    def slider_picker_entries(self) -> list[tuple[str, str, bool]]:
        entries: list[tuple[str, str, bool]] = []
        for name in self._variables:
            name_s = str(name)
            entries.append(
                (
                    name_s,
                    self._format_label_text(name_s),
                    self.variable_visible(name_s),
                )
            )
        return entries

    def variable_visible(self, name: str) -> bool:
        return str(name) not in self._hidden_names

    def set_variable_visible(self, name: str, visible: bool) -> None:
        name_s = str(name)
        if name_s not in self._containers:
            return
        if visible:
            self._hidden_names.discard(name_s)
        else:
            self._hidden_names.add(name_s)
        self._apply_variable_visibility(name_s)
        self._sync_visibility_state()

    def apply_visibility_projection(
        self, visibility: Dict[str, bool] | Iterable[tuple[str, bool]]
    ) -> None:
        """Apply one complete visibility projection without workspace callbacks."""
        projected = dict(visibility)
        for name in tuple(self._containers):
            if name in projected:
                if bool(projected[name]):
                    self._hidden_names.discard(name)
                else:
                    self._hidden_names.add(name)
        self._sync_visibility_state()

    def set_fine_mode(self, enabled: bool) -> None:
        """
        Enable or disable fine mode for all sliders.

        Fine mode increases slider resolution from 1000 to 10000 steps,
        allowing more precise adjustments while preserving current values.

        Parameters
        ----------
        enabled : bool
            True to enable fine mode, False for normal mode
        """
        enabled = bool(enabled)
        if self._fine_mode == enabled:
            return
        self._fine_mode = enabled

        for name, slider in self._sliders.items():
            # Preserve the authoritative scientific value rather than
            # reconstructing it from the quantized slider position.
            val = float(
                self._variables.get(name, self._last_valid_values.get(name, 0.0))
            )

            # Switch resolution without emitting valueChanged (Fine must not trigger simulations).
            new_max = 10000 if enabled else 1000
            with QtCore.QSignalBlocker(slider):
                slider.setMinimum(0)
                slider.setMaximum(new_max)
                slider.setSingleStep(1 if enabled else 5)
                slider.setPageStep(50 if enabled else 100)
                slider.set_fine_drag_mode(enabled)

                # Restore position for the same numeric value
                slider.setValue(self._value_to_slider_pos(name, float(val)))

        logger.info(f"Fine mode {'enabled' if enabled else 'disabled'}")

    def _reset_fine_mode_quietly(self) -> None:
        """Return child sliders to quiet Normal state before a rebuild/clear."""
        was_fine = bool(self._fine_mode)
        self._fine_mode = False
        for slider in self._sliders.values():
            with QtCore.QSignalBlocker(slider):
                slider.setMinimum(0)
                slider.setMaximum(1000)
                slider.setSingleStep(5)
                slider.setPageStep(100)
                slider.set_fine_drag_mode(False)
        if was_fine:
            self.fineModeReset.emit(False)

    def _apply_variable_visibility(self, name: str) -> None:
        container = self._containers.get(str(name))
        if container is None:
            return
        container.setVisible(self.variable_visible(str(name)))

    def _sync_visibility_state(self) -> None:
        visible_count = 0
        for name, container in self._containers.items():
            visible = self.variable_visible(name)
            container.setVisible(visible)
            if visible:
                visible_count += 1
        has_variables = bool(self._variables)
        if self._placeholder is not None:
            self._placeholder.setVisible(not has_variables)
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.setVisible(
                has_variables and bool(self._containers) and visible_count == 0
            )

    def _reconcile_visibility_scope(
        self,
        names: set[str],
        *,
        metadata: Dict[str, Dict[str, object]],
        preserve_visibility: bool,
    ) -> None:
        """Reconcile choices by stable parameter identity, not roster order."""
        signature = tuple(
            (
                name,
                metadata[name]["family"],
                metadata[name]["role"],
                metadata[name]["scale"],
            )
            for name in names
        )

        previous = self._visibility_scope_signature
        if not preserve_visibility or previous is None:
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
        metadata: Optional[Dict[str, Dict[str, object]]],
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
        """Normalize one render call without changing widget state."""
        try:
            variable_items = variables.items()
        except AttributeError as exc:
            raise ValueError("Slider variables must be a mapping.") from exc

        normalized_variables: Dict[str, float] = {}
        for raw_name, value in variable_items:
            name = str(raw_name)
            if name in normalized_variables:
                raise ValueError(
                    "Slider variables must have unique canonical names."
                )
            try:
                normalized_variables[name] = float(value)
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Slider variable values must be numeric."
                ) from exc

        if metadata is None:
            metadata_items = ()
        else:
            try:
                metadata_items = metadata.items()
            except AttributeError as exc:
                raise ValueError("Slider metadata must be a mapping.") from exc

        normalized_metadata: Dict[str, Dict[str, object]] = {}
        for raw_name, raw_meta in metadata_items:
            name = str(raw_name)
            if name in normalized_metadata:
                raise ValueError(
                    "Slider metadata must have unique canonical names."
                )
            try:
                normalized_metadata[name] = dict(raw_meta)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Slider metadata entries must be mappings."
                ) from exc

        return normalized_variables, normalized_metadata

    def _editor_has_focus(self, name: str) -> bool:
        editor = self._value_labels.get(str(name))
        return isinstance(editor, _NumericValueEdit) and editor.hasFocus()

    def _on_editor_started(self, name: str) -> None:
        editor = self._value_labels.get(str(name))
        if not isinstance(editor, _NumericValueEdit):
            return
        self._editor_pre_edit_values[str(name)] = float(
            self._variables.get(str(name), 0.0)
        )
        self._editor_pre_edit_text[str(name)] = editor.text()
        self._last_committed_editor_text.pop(str(name), None)
        self._decorate_editor(str(name))

    def _on_editor_escape(self, name: str) -> None:
        editor = self._value_labels.get(str(name))
        if not isinstance(editor, _NumericValueEdit):
            return
        value = self._editor_pre_edit_values.get(
            str(name), self._variables.get(str(name), 0.0)
        )
        text = self._editor_pre_edit_text.get(str(name), self._format_value(value))
        self._set_value_projection(
            str(name), value, editor_text=text, emit=False
        )
        self._last_committed_editor_text[str(name)] = text
        self._decorate_editor(str(name))

    def _decorate_editor(self, name: str) -> None:
        editor = self._value_labels.get(str(name))
        if not isinstance(editor, _NumericValueEdit):
            return
        try:
            value = float(editor.text().strip())
            valid = bool(math.isfinite(value))
        except (TypeError, ValueError, OverflowError):
            valid = False
        editor.setProperty("invalidNumeric", not valid)
        editor.setStyleSheet(
            "QLineEdit { border: 1px solid #c0392b; }"
            if not valid
            else ""
        )

    def _commit_editor(self, name: str) -> None:
        name_s = str(name)
        editor = self._value_labels.get(name_s)
        if not isinstance(editor, _NumericValueEdit):
            return
        raw = editor.text()
        if raw == self._last_committed_editor_text.get(name_s):
            return
        try:
            value = float(raw.strip())
        except (TypeError, ValueError, OverflowError):
            self._decorate_editor(name_s)
            return
        if not math.isfinite(value):
            self._decorate_editor(name_s)
            return
        self._last_committed_editor_text[name_s] = raw
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
        name_s = str(name)
        if name_s not in self._sliders:
            return
        value_f = float(value)
        if not math.isfinite(value_f):
            return
        if self._slider_scales.get(name_s, "log") == "log":
            self._slider_signs[name_s] = 1.0 if value_f >= 0 else -1.0
        self._ensure_range_covers(name_s, value_f)
        self._variables[name_s] = value_f
        self._last_valid_values[name_s] = value_f
        slider = self._sliders[name_s]
        with QtCore.QSignalBlocker(slider):
            slider.setValue(self._value_to_slider_pos(name_s, value_f))
        value_label = self._value_labels.get(name_s)
        if isinstance(value_label, _NumericValueEdit):
            self._set_editor_text(
                value_label,
                self._format_value(value_f) if editor_text is None else editor_text,
            )
            self._decorate_editor(name_s)
        elif isinstance(value_label, QtWidgets.QLabel):
            value_label.setText(self._format_value(value_f))
        logger.debug("Variable projected: %s = %.12g", name_s, value_f)
        if emit:
            if not isinstance(interaction, SliderInteractionKind):
                raise TypeError("emitted Slider values require an interaction kind.")
            self.interactionRequested.emit(name_s, value_f, interaction)

    @staticmethod
    def _set_editor_text(editor: QtWidgets.QLineEdit, text: str) -> None:
        blocker = QtCore.QSignalBlocker(editor)
        try:
            editor.setText(str(text))
        finally:
            del blocker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_initial_range(self, value: float) -> Tuple[float, float]:
        """Derive an appropriate log range for a slider based on its value."""
        default_min, default_max = _LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX
        if not math.isfinite(float(value)) or float(value) == 0.0:
            return default_min, default_max

        magnitude = self._clamp_log_magnitude(value)
        log_center = math.log10(magnitude)
        span = 4.0  # ±4 orders of magnitude around current value
        log_min = max(log_center - span, _LOG_DOMAIN_MIN)
        log_max = min(log_center + span, _LOG_DOMAIN_MAX)

        if log_max - log_min < 0.5:
            log_min = max(log_center - 0.25, _LOG_DOMAIN_MIN)
            log_max = min(log_center + 0.25, _LOG_DOMAIN_MAX)

        # Ensure sane ordering
        if log_max <= log_min:
            if log_center <= _LOG_DOMAIN_MIN:
                return _LOG_DOMAIN_MIN, min(
                    _LOG_DOMAIN_MIN + 1.0, _LOG_DOMAIN_MAX
                )
            if log_center >= _LOG_DOMAIN_MAX:
                return max(
                    _LOG_DOMAIN_MAX - 1.0, _LOG_DOMAIN_MIN
                ), _LOG_DOMAIN_MAX
            log_min = max(log_center - 0.5, _LOG_DOMAIN_MIN)
            log_max = min(log_center + 0.5, _LOG_DOMAIN_MAX)
            if log_max <= log_min:
                return default_min, default_max

        return log_min, log_max

    def _clamp_log_magnitude(self, value: float) -> float:
        """Clamp a magnitude to the representable log slider domain."""
        try:
            magnitude = abs(float(value))
        except (ArithmeticError, TypeError, ValueError):
            magnitude = _LOG_SLIDER_VALUE_MIN

        if not math.isfinite(magnitude) or magnitude <= 0.0:
            return _LOG_SLIDER_VALUE_MIN
        return min(max(magnitude, float.fromhex("0x0.0000000000001p-1022")), sys.float_info.max)

    def _validated_log_range(self, name: str) -> Tuple[float, float]:
        """Return a finite, ordered, bounded log10 range for a slider."""
        raw_min, raw_max = self._slider_ranges.get(
            name, (_LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX)
        )
        try:
            log_min = float(raw_min)
        except (ArithmeticError, TypeError, ValueError):
            log_min = _LOG_SLIDER_EXP_MIN
        try:
            log_max = float(raw_max)
        except (ArithmeticError, TypeError, ValueError):
            log_max = _LOG_SLIDER_EXP_MAX

        if not math.isfinite(log_min):
            log_min = _LOG_SLIDER_EXP_MIN
        if not math.isfinite(log_max):
            log_max = _LOG_SLIDER_EXP_MAX

        log_min = min(max(log_min, _LOG_DOMAIN_MIN), _LOG_DOMAIN_MAX)
        log_max = min(max(log_max, _LOG_DOMAIN_MIN), _LOG_DOMAIN_MAX)
        if log_max <= log_min:
            log_min, log_max = _LOG_DOMAIN_MIN, _LOG_DOMAIN_MAX
        return log_min, log_max

    @staticmethod
    def _representable_linear_range(value: float) -> Tuple[float, float]:
        """Return finite ordered neighbors containing one finite value."""
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError("Linear slider values must be finite.")

        lower = math.nextafter(value_f, -math.inf)
        upper = math.nextafter(value_f, math.inf)
        if not math.isfinite(lower):
            lower = value_f
        if not math.isfinite(upper):
            upper = value_f
        if lower < value_f < upper:
            return lower, upper
        if lower < value_f:
            return lower, value_f
        if value_f < upper:
            return value_f, upper
        raise ValueError("Could not construct an ordered linear slider range.")

    @staticmethod
    def _linear_fraction(value: float, lower: float, upper: float) -> float:
        """Map a bounded linear value to a fraction without overflow."""
        value_f = float(value)
        lower_f = float(lower)
        upper_f = float(upper)
        if value_f <= lower_f:
            return 0.0
        if value_f >= upper_f:
            return 1.0

        span = upper_f - lower_f
        if math.isfinite(span):
            fraction = (value_f - lower_f) / span
        else:
            exponent = max(
                math.frexp(abs(endpoint))[1]
                for endpoint in (lower_f, value_f, upper_f)
                if endpoint != 0.0
            )
            scaled_lower = math.ldexp(lower_f, -exponent)
            scaled_value = math.ldexp(value_f, -exponent)
            scaled_upper = math.ldexp(upper_f, -exponent)
            fraction = (scaled_value - scaled_lower) / (
                scaled_upper - scaled_lower
            )
        return min(max(fraction, 0.0), 1.0)

    @staticmethod
    def _linear_value_at_fraction(
        fraction: float,
        lower: float,
        upper: float,
    ) -> float:
        """Map a slider fraction to a bounded linear value without overflow."""
        fraction_f = min(max(float(fraction), 0.0), 1.0)
        if fraction_f <= 0.0:
            return float(lower)
        if fraction_f >= 1.0:
            return float(upper)

        lower_f = float(lower)
        upper_f = float(upper)
        span = upper_f - lower_f
        if math.isfinite(span):
            value = lower_f + fraction_f * span
        else:
            exponent = max(
                math.frexp(abs(endpoint))[1]
                for endpoint in (lower_f, upper_f)
                if endpoint != 0.0
            )
            scaled_lower = math.ldexp(lower_f, -exponent)
            scaled_upper = math.ldexp(upper_f, -exponent)
            value = math.ldexp(
                scaled_lower + fraction_f * (scaled_upper - scaled_lower),
                exponent,
            )
        if not math.isfinite(value):
            return upper_f if fraction_f >= 0.5 else lower_f
        return min(max(value, lower_f), upper_f)

    def _slider_pos_to_value(self, name: str, slider_pos: int) -> float:
        """Convert a slider position to a concrete variable value."""
        scale = self._slider_scales.get(name, "log")
        slider = self._sliders[name]
        steps = max(1, int(slider.maximum()))
        pos = max(slider.minimum(), min(int(slider_pos), slider.maximum()))
        frac = min(max(pos / steps, 0.0), 1.0)

        if scale == "linear":
            lo, hi = self._slider_ranges.get(name, (0.0, 1.0))
            lo = float(lo)
            hi = float(hi)
            if not (
                math.isfinite(lo)
                and math.isfinite(hi)
                and hi > lo
            ):
                reference = self._last_valid_values.get(
                    name,
                    self._variables.get(name, 0.0),
                )
                lo, hi = self._representable_linear_range(float(reference))
                self._slider_ranges[name] = (lo, hi)
            return self._linear_value_at_fraction(frac, lo, hi)

        log_min, log_max = self._validated_log_range(name)
        log_value = log_min + frac * (log_max - log_min)
        log_value = min(max(log_value, log_min), log_max)
        magnitude = 10 ** float(log_value)
        magnitude = min(
            max(magnitude, float.fromhex("0x0.0000000000001p-1022")),
            sys.float_info.max,
        )
        sign = self._slider_signs.get(name, 1.0)
        return magnitude if sign >= 0 else -magnitude

    def _value_to_slider_pos(self, name: str, value_abs: float) -> int:
        """Convert a value to slider position using stored range."""
        scale = self._slider_scales.get(name, "log")
        steps = max(1, self._sliders[name].maximum())
        if scale == "linear":
            lo, hi = self._slider_ranges.get(name, (0.0, 1.0))
            lo = float(lo)
            hi = float(hi)
            if not (
                math.isfinite(lo)
                and math.isfinite(hi)
                and hi > lo
            ):
                lo, hi = self._representable_linear_range(float(value_abs))
                self._slider_ranges[name] = (lo, hi)
            frac = self._linear_fraction(float(value_abs), lo, hi)
            return max(0, min(int(round(frac * steps)), int(steps)))

        log_min, log_max = self._validated_log_range(name)
        log_span = log_max - log_min
        if log_span <= 0:
            log_span = _LOG_DOMAIN_MAX - _LOG_DOMAIN_MIN

        sign = self._slider_signs.get(name, 1.0)
        magnitude = self._clamp_log_magnitude(value_abs)
        log_value = math.log10(magnitude)
        log_value = min(max(log_value, log_min), log_max)
        frac = (log_value - log_min) / log_span
        pos = int(round(frac * steps))
        # Preserve negative direction display: slider handle represents magnitude only.
        self._slider_signs[name] = 1.0 if sign >= 0 else -1.0
        return max(0, min(pos, int(steps)))

    def _ensure_range_covers(
        self,
        name: str,
        value: float,
        *,
        project_slider: bool = True,
    ) -> None:
        """Expand the GUI-only range if needed to include the given value."""
        if name not in self._slider_ranges:
            return

        if self._slider_scales.get(name, "log") == "linear":
            value_f = float(value)
            raw_min, raw_max = self._slider_ranges[name]
            try:
                linear_min = float(raw_min)
                linear_max = float(raw_max)
            except (ArithmeticError, TypeError, ValueError):
                linear_min, linear_max = math.nan, math.nan

            if (
                math.isfinite(linear_min)
                and math.isfinite(linear_max)
                and linear_max > linear_min
            ):
                if value_f < linear_min:
                    linear_min = value_f
                elif value_f > linear_max:
                    linear_max = value_f
            else:
                linear_min, linear_max = self._representable_linear_range(value_f)

            self._slider_ranges[name] = (linear_min, linear_max)
            self._update_range_labels(name)
            if project_slider and name in self._sliders:
                slider = self._sliders[name]
                with QtCore.QSignalBlocker(slider):
                    slider.setValue(self._value_to_slider_pos(name, value_f))
            return

        log_min, log_max = self._validated_log_range(name)
        self._slider_ranges[name] = (log_min, log_max)
        magnitude = self._clamp_log_magnitude(value)
        log_value = math.log10(magnitude)

        needs_expand = log_value < log_min or log_value > log_max
        if not needs_expand:
            return

        new_log_min, new_log_max = self._compute_initial_range(value)
        if (new_log_min, new_log_max) == (log_min, log_max):
            return
        self._slider_ranges[name] = (new_log_min, new_log_max)
        self._update_range_labels(name)

        if project_slider and name in self._sliders:
            slider = self._sliders[name]
            with QtCore.QSignalBlocker(slider):
                slider.setValue(self._value_to_slider_pos(name, value))

    def _update_range_labels(self, name: str) -> None:
        """Refresh the displayed min/max labels for a slider."""
        if name not in self._range_labels or name not in self._slider_ranges:
            return
        min_label, max_label = self._range_labels[name]
        if self._slider_scales.get(name, "log") == "linear":
            linear_min, linear_max = self._slider_ranges[name]
            min_label.setText(self._format_value(float(linear_min)))
            max_label.setText(self._format_value(float(linear_max)))
            return
        log_min, log_max = self._validated_log_range(name)
        self._slider_ranges[name] = (log_min, log_max)
        min_label.setText(self._format_power_of_ten(log_min))
        max_label.setText(self._format_power_of_ten(log_max))

    @staticmethod
    def _format_power_of_ten(log_value: float) -> str:
        """Format a base-10 exponent as a compact string."""
        if not math.isfinite(log_value):
            log_value = _LOG_SLIDER_EXP_MIN
        log_value = min(max(float(log_value), _LOG_DOMAIN_MIN), _LOG_DOMAIN_MAX)
        exponent = int(round(log_value))
        if abs(log_value - exponent) < 1e-6:
            return f"10<sup>{exponent}</sup>"
        value = 10**log_value
        if value >= 1000 or value < 0.001:
            return f"{value:.1e}"
        return f"{value:.3g}"
