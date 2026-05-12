"""
Variable sliders widget for interactive parameter adjustment.

Allows real-time modification of mechanism parameters (rate constants,
equilibrium constants, and scalar values) with automatic simulation updates.
"""

from __future__ import annotations

from functools import partial
import logging
from typing import Dict, Optional, Tuple
import math

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal

from ..ui_helpers import make_bounded_label, make_placeholder_label, make_scroll_area, set_bounded_label_text

logger = logging.getLogger(__name__)

__all__ = ["VariableSliders"]

_LOG_SLIDER_VALUE_MIN = 1e-12
_LOG_SLIDER_VALUE_MAX = 1e12
_LOG_SLIDER_EXP_MIN = -12.0
_LOG_SLIDER_EXP_MAX = 12.0


class VariableSliders(QtWidgets.QWidget):
    """
    Interactive sliders for mechanism variables.

    Features:
    - Sliders for rate constants (k1, k2, ...)
    - Sliders for equilibrium constants (K1, kf1, kr1, ...)
    - Sliders for scalar parameters defined in mechanism
    - Logarithmic scale for wide range of values
    - Real-time value display
    - Emits signal on value change for automatic simulation update

    Signals
    -------
    variableChanged : Signal(str, float)
        Emitted when a variable value changes (name, new_value)
    """

    variableChanged = Signal(str, float)  # (variable_name, new_value)
    sliderDragStarted = Signal(str)       # name
    sliderDragFinished = Signal(str)      # name
    contentStateChanged = Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, embedded: bool = False):
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
        self._labels: Dict[str, QtWidgets.QLabel] = {}
        self._value_labels: Dict[str, QtWidgets.QLabel] = {}
        self._slider_ranges: Dict[str, Tuple[float, float]] = {}  # log10 min/max per slider
        self._range_labels: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        self._slider_signs: Dict[str, float] = {}
        self._slider_scales: Dict[str, str] = {}  # 'log' (default) or 'linear'
        self._containers: Dict[str, QtWidgets.QWidget] = {}
        self._metadata: Dict[str, Dict[str, object]] = {}
        self._last_valid_values: Dict[str, float] = {}
        self._freeze_ranges: bool = False
        self._fine_mode: bool = False
        self._slider_callbacks: Dict[str, Dict[str, object]] = {}
        self._hidden_names: set[str] = set()
        self._visibility_scope_signature: object | None = None
        self._placeholder: QtWidgets.QLabel | None = None
        self._hidden_placeholder: QtWidgets.QLabel | None = None

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
            desc = QtWidgets.QLabel("Adjust mechanism parameters and see simulation update automatically")
            desc.setWordWrap(True)
            desc_font = desc.font()
            desc_font.setPointSize(desc_font.pointSize() - 1)
            desc.setFont(desc_font)
            main_layout.addWidget(desc)

        self._sliders_widget = QtWidgets.QWidget()
        self._sliders_layout = QtWidgets.QVBoxLayout(self._sliders_widget)
        self._sliders_layout.setContentsMargins(0, 0, 0, 0)
        self._sliders_layout.setSpacing(6 if self._embedded else 8)

        if self._embedded:
            main_layout.addWidget(self._sliders_widget)
        else:
            scroll = make_scroll_area(self)
            scroll.setWidget(self._sliders_widget)
            main_layout.addWidget(scroll)

            # Placeholder message
            self._placeholder = make_placeholder_label("No variables defined.\nParse a mechanism to see sliders.")
            self._sliders_layout.addWidget(self._placeholder)
            self._hidden_placeholder = make_placeholder_label("All mechanism sliders hidden by picker.")
            self._hidden_placeholder.hide()
            self._sliders_layout.addWidget(self._hidden_placeholder)

        self._sliders_layout.addStretch()

        logger.debug("VariableSliders initialized")

    def set_variables(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = False,
        visibility_scope_signature: object | None = None,
    ) -> None:
        """
        Set variables and create sliders.

        Parameters
        ----------
        variables : dict
            Dictionary of {variable_name: value}
            Examples: {'k1': 0.5, 'k2': 1.0, 'K1': 10.0, 'kf1': 2.0, 'kr1': 0.2}
        """
        logger.info(f"Setting {len(variables)} variables")

        # Visibility choices are scoped to the currently loaded variable set.
        current_names = {str(name) for name in variables}
        same_visibility_scope = (
            bool(preserve_visibility)
            and visibility_scope_signature is not None
            and self._visibility_scope_signature == visibility_scope_signature
        )
        if same_visibility_scope:
            self._hidden_names.intersection_update(current_names)
        else:
            self._hidden_names = set(current_names)
        self._visibility_scope_signature = visibility_scope_signature

        # Clear existing sliders
        self._clear_sliders()

        self._variables = variables.copy()
        self._metadata = dict(metadata or {})

        if not variables:
            if self._placeholder is not None:
                self._placeholder.show()
            self.contentStateChanged.emit(False)
            logger.debug("No variables to display")
            return

        if self._placeholder is not None:
            self._placeholder.hide()

        for name, value in variables.items():
            self._create_slider(name, value)

        self._sync_visibility_state()
        self.contentStateChanged.emit(True)
        logger.info(f"Created {len(self._sliders)} sliders")

    def _clear_sliders(self) -> None:
        """Clear all existing sliders."""
        for container in list(self._containers.values()):
            self._sliders_layout.removeWidget(container)
            container.setParent(None)
            container.deleteLater()

        self._containers.clear()
        self._sliders.clear()
        self._slider_callbacks.clear()
        self._labels.clear()
        self._value_labels.clear()
        self._slider_ranges.clear()
        self._range_labels.clear()
        self._slider_signs.clear()
        self._slider_scales.clear()
        self._metadata.clear()
        self._last_valid_values.clear()

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
            Variable name (e.g., 'k1', 'K1', 'kf1')
        value : float
            Current value
        """
        # Container for this slider
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        # Top row: name label and value label
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        name_label = make_bounded_label(self._format_label_text(name), max_width=260)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        top_row.addWidget(name_label)

        top_row.addStretch()

        value_label = QtWidgets.QLabel(self._format_value(value))
        value_label.setMinimumWidth(80)
        value_label.setAlignment(QtCore.Qt.AlignRight)
        top_row.addWidget(value_label)

        layout.addLayout(top_row)

        meta = self._metadata.get(name) or {}
        scale = str(meta.get("scale") or "log").strip().lower()

        # Slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimumHeight(28)
        slider.setMinimum(0)
        slider.setMaximum(1000)  # 1000 steps for smooth control
        slider.setSingleStep(5)
        slider.setPageStep(100)
        self._slider_scales[name] = ("linear" if scale == "linear" else "log")

        if scale == "linear":
            try:
                lo = float(meta.get("min")) if meta.get("min") is not None else float(value) - 10.0
            except Exception:
                lo = float(value) - 10.0
            try:
                hi = float(meta.get("max")) if meta.get("max") is not None else float(value) + 10.0
            except Exception:
                hi = float(value) + 10.0
            if not math.isfinite(lo):
                lo = float(value) - 10.0
            if not math.isfinite(hi):
                hi = float(value) + 10.0
            if hi <= lo:
                hi = lo + 1.0
            self._slider_ranges[name] = (lo, hi)
            self._slider_signs[name] = 1.0
        else:
            sign = 1.0 if value >= 0 else -1.0
            self._slider_signs[name] = sign
            self._slider_ranges[name] = (_LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX)

        # Store slider reference early so _value_to_slider_pos can access it
        self._sliders[name] = slider

        slider_pos = self._value_to_slider_pos(name, float(value))
        slider.setValue(slider_pos)
        self._last_valid_values[name] = self._slider_pos_to_value(name, slider_pos)

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
            name_label.setToolTip(tip)
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
        self._labels[name] = name_label
        self._value_labels[name] = value_label
        self._range_labels[name] = (min_label, max_label)

        # Add to layout (before stretch)
        self._sliders_layout.insertWidget(self._sliders_layout.count() - 1, container)
        self._containers[name] = container
        self._apply_variable_visibility(name)

    def _on_slider_pressed(self, name: str) -> None:
        self.sliderDragStarted.emit(name)

    def _on_slider_released(self, name: str) -> None:
        self.sliderDragFinished.emit(name)
    def _format_label_text(self, name: str) -> str:
        meta = self._metadata.get(name)
        derived_suffix = ""
        if meta and bool(meta.get("derived")):
            derived_suffix = " (derived)"
        unit_suffix = ""
        if meta and meta.get("scale") == "linear" and meta.get("unit"):
            unit_suffix = f" [{meta.get('unit')}]"
        if meta and isinstance(meta.get('label'), str):
            return f"{name} ({meta['label']}){unit_suffix}{derived_suffix}"
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
        except Exception as exc:
            fallback = self._last_valid_values.get(name, self._variables.get(name, 0.0))
            try:
                value = float(fallback)
            except Exception:
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
                except Exception:
                    restore_pos = max(slider.minimum(), min(int(slider_pos), slider.maximum()))
                with QtCore.QSignalBlocker(slider):
                    slider.setValue(restore_pos)

            if name in self._value_labels:
                self._value_labels[name].setText(self._format_value(value))
            self._variables[name] = value
            self._last_valid_values[name] = float(value)

            logger.warning(
                "Slider conversion failed for %s at pos=%s (%s); restored last valid value %.12g",
                str(name),
                int(slider_pos),
                type(exc).__name__,
                float(value),
            )
            return

        # Update stored value
        self._variables[name] = value
        self._last_valid_values[name] = float(value)

        # Update display
        if name in self._labels:
            set_bounded_label_text(self._labels[name], self._format_label_text(name), max_width=260)
        if name in self._value_labels:
            self._value_labels[name].setText(self._format_value(value))

        # Emit signal
        logger.debug("Variable changed: %s = %.12g", str(name), float(value))
        self.variableChanged.emit(name, value)

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

    def update_metadata(self, name: str, meta: Dict[str, object]) -> None:
        self._metadata[name] = dict(meta)
        if name in self._labels:
            set_bounded_label_text(self._labels[name], self._format_label_text(name), max_width=260)

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

        value_f = float(value)
        if self._slider_scales.get(name, "log") == "log":
            sign = 1.0 if value_f >= 0 else -1.0
            self._slider_signs[name] = sign
            magnitude = self._clamp_log_magnitude(value_f)
            value_f = magnitude if sign >= 0 else -magnitude
            self._ensure_range_covers(name, value_f)

        self._variables[name] = value_f
        self._last_valid_values[name] = value_f

        # Update slider position
        if name in self._sliders:
            slider_pos = self._value_to_slider_pos(name, value_f)

            slider = self._sliders[name]
            with QtCore.QSignalBlocker(slider):
                slider.setValue(slider_pos)

        # Update display
        if name in self._value_labels:
            self._value_labels[name].setText(self._format_value(value_f))

        logger.debug("Variable updated: %s = %.12g", str(name), float(value_f))

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

        self._variables[name] = value
        try:
            self._last_valid_values[name] = float(value)
        except (TypeError, ValueError, OverflowError):
            self._last_valid_values.pop(name, None)
        if self._slider_scales.get(name, "log") == "log":
            self._slider_signs[name] = 1.0 if value >= 0 else -1.0

        if name in self._value_labels:
            self._value_labels[name].setText(self._format_value(value))

        logger.debug(f"Variable readout updated: {name} = {value}")

    def clear(self) -> None:
        """Clear all variables and sliders."""
        self._clear_sliders()
        self._variables.clear()
        self._slider_ranges.clear()
        self._slider_signs.clear()
        self._slider_scales.clear()
        self._range_labels.clear()
        self._last_valid_values.clear()
        self._hidden_names.clear()
        self._visibility_scope_signature = None
        if self._placeholder is not None:
            self._placeholder.show()
        if self._hidden_placeholder is not None:
            self._hidden_placeholder.hide()
        self.contentStateChanged.emit(False)
        logger.debug("Variables cleared")

    def has_slider_entries(self) -> bool:
        return bool(self._variables)

    def visible_slider_count(self) -> int:
        return sum(1 for name in self._containers if self.variable_visible(name))

    def has_visible_entries(self) -> bool:
        return self.visible_slider_count() > 0

    def has_variable(self, name: str) -> bool:
        """Return True if a slider exists for the given variable."""
        return name in self._sliders

    def slider_picker_entries(self) -> list[tuple[str, str, bool]]:
        entries: list[tuple[str, str, bool]] = []
        for name in self._variables:
            name_s = str(name)
            label = self._labels.get(name_s)
            entries.append((name_s, str(label.text()) if label is not None else name_s, self.variable_visible(name_s)))
        return entries

    def variable_visible(self, name: str) -> bool:
        return str(name) not in self._hidden_names

    def set_variable_visible(self, name: str, visible: bool) -> None:
        name_s = str(name)
        if visible:
            self._hidden_names.discard(name_s)
        else:
            self._hidden_names.add(name_s)
        self._apply_variable_visibility(name_s)
        self._sync_visibility_state()
        self.contentStateChanged.emit(bool(self._variables))

    def begin_live_drag(self) -> None:
        """Prevent range recentering while the user drags a slider."""
        self._freeze_ranges = True

    def end_live_drag(self) -> None:
        """Re-enable automatic range management after dragging."""
        self._freeze_ranges = False

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
        if self._fine_mode == enabled:
            return
        self._fine_mode = enabled

        for name, slider in self._sliders.items():
            # Capture current numeric value
            current_pos = slider.value()
            current_max = max(1, slider.maximum())
            if self._slider_scales.get(name, "log") == "linear":
                lo, hi = self._slider_ranges[name]
                frac = current_pos / current_max
                val = float(lo) + float(frac) * (float(hi) - float(lo))
            else:
                sign = self._slider_signs.get(name, 1.0)
                log_min, log_max = self._slider_ranges[name]
                log_value = log_min + (current_pos / current_max) * (log_max - log_min)
                val = sign * (10 ** log_value)

            # Switch resolution without emitting valueChanged (Fine must not trigger simulations).
            new_max = 10000 if enabled else 1000
            with QtCore.QSignalBlocker(slider):
                slider.setMinimum(0)
                slider.setMaximum(new_max)
                slider.setSingleStep(1 if enabled else 5)
                slider.setPageStep(50 if enabled else 100)

                # Restore position for the same numeric value
                slider.setValue(self._value_to_slider_pos(name, float(val)))

        logger.info(f"Fine mode {'enabled' if enabled else 'disabled'}")

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
            self._hidden_placeholder.setVisible(has_variables and bool(self._containers) and visible_count == 0)

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
        log_min = max(log_center - span, _LOG_SLIDER_EXP_MIN)
        log_max = min(log_center + span, _LOG_SLIDER_EXP_MAX)

        if log_max - log_min < 0.5:
            log_min = max(log_center - 0.25, _LOG_SLIDER_EXP_MIN)
            log_max = min(log_center + 0.25, _LOG_SLIDER_EXP_MAX)

        # Ensure sane ordering
        if log_max <= log_min:
            if log_center <= _LOG_SLIDER_EXP_MIN:
                return _LOG_SLIDER_EXP_MIN, min(_LOG_SLIDER_EXP_MIN + 1.0, _LOG_SLIDER_EXP_MAX)
            if log_center >= _LOG_SLIDER_EXP_MAX:
                return max(_LOG_SLIDER_EXP_MAX - 1.0, _LOG_SLIDER_EXP_MIN), _LOG_SLIDER_EXP_MAX
            log_min = max(log_center - 0.5, _LOG_SLIDER_EXP_MIN)
            log_max = min(log_center + 0.5, _LOG_SLIDER_EXP_MAX)
            if log_max <= log_min:
                return default_min, default_max

        return log_min, log_max

    def _clamp_log_magnitude(self, value: float) -> float:
        """Clamp a magnitude to the representable log slider domain."""
        try:
            magnitude = abs(float(value))
        except Exception:
            magnitude = _LOG_SLIDER_VALUE_MIN

        if not math.isfinite(magnitude) or magnitude <= 0.0:
            return _LOG_SLIDER_VALUE_MIN
        return min(max(magnitude, _LOG_SLIDER_VALUE_MIN), _LOG_SLIDER_VALUE_MAX)

    def _validated_log_range(self, name: str) -> Tuple[float, float]:
        """Return a finite, ordered, bounded log10 range for a slider."""
        raw_min, raw_max = self._slider_ranges.get(name, (_LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX))
        try:
            log_min = float(raw_min)
        except Exception:
            log_min = _LOG_SLIDER_EXP_MIN
        try:
            log_max = float(raw_max)
        except Exception:
            log_max = _LOG_SLIDER_EXP_MAX

        if not math.isfinite(log_min):
            log_min = _LOG_SLIDER_EXP_MIN
        if not math.isfinite(log_max):
            log_max = _LOG_SLIDER_EXP_MAX

        log_min = min(max(log_min, _LOG_SLIDER_EXP_MIN), _LOG_SLIDER_EXP_MAX)
        log_max = min(max(log_max, _LOG_SLIDER_EXP_MIN), _LOG_SLIDER_EXP_MAX)
        if log_max <= log_min:
            log_min, log_max = _LOG_SLIDER_EXP_MIN, _LOG_SLIDER_EXP_MAX
        return log_min, log_max

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
            if not math.isfinite(lo):
                lo = 0.0
            if not math.isfinite(hi):
                hi = 1.0
            if hi <= lo:
                hi = lo + 1.0
            return lo + frac * (hi - lo)

        log_min, log_max = self._validated_log_range(name)
        log_value = log_min + frac * (log_max - log_min)
        log_value = min(max(log_value, log_min), log_max)
        magnitude = 10 ** float(log_value)
        magnitude = min(max(magnitude, _LOG_SLIDER_VALUE_MIN), _LOG_SLIDER_VALUE_MAX)
        sign = self._slider_signs.get(name, 1.0)
        return magnitude if sign >= 0 else -magnitude

    def _value_to_slider_pos(self, name: str, value_abs: float) -> int:
        """Convert a value to slider position using stored range."""
        scale = self._slider_scales.get(name, "log")
        steps = max(1, self._sliders[name].maximum())
        if scale == "linear":
            lo, hi = self._slider_ranges.get(name, (0.0, 1.0))
            span = float(hi) - float(lo)
            if span <= 0:
                span = 1.0
            frac = (float(value_abs) - float(lo)) / span
            frac = min(max(frac, 0.0), 1.0)
            return max(0, min(int(round(frac * steps)), int(steps)))

        log_min, log_max = self._validated_log_range(name)
        log_span = log_max - log_min
        if log_span <= 0:
            log_span = _LOG_SLIDER_EXP_MAX - _LOG_SLIDER_EXP_MIN

        sign = self._slider_signs.get(name, 1.0)
        magnitude = self._clamp_log_magnitude(value_abs)
        log_value = math.log10(magnitude)
        log_value = min(max(log_value, log_min), log_max)
        frac = (log_value - log_min) / log_span
        pos = int(round(frac * steps))
        # Preserve negative direction display: slider handle represents magnitude only.
        self._slider_signs[name] = 1.0 if sign >= 0 else -1.0
        return max(0, min(pos, int(steps)))

    def _ensure_range_covers(self, name: str, value: float) -> None:
        """Expand the slider range if needed to include the given value."""
        if self._slider_scales.get(name, "log") != "log":
            return
        if self._freeze_ranges:
            return
        if name not in self._slider_ranges:
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

        if name in self._sliders:
            slider = self._sliders[name]
            with QtCore.QSignalBlocker(slider):
                slider.setValue(self._value_to_slider_pos(name, value))

    def _update_range_labels(self, name: str) -> None:
        """Refresh the displayed min/max labels for a slider."""
        if name not in self._range_labels or name not in self._slider_ranges:
            return
        log_min, log_max = self._validated_log_range(name)
        self._slider_ranges[name] = (log_min, log_max)
        min_label, max_label = self._range_labels[name]
        min_label.setText(self._format_power_of_ten(log_min))
        max_label.setText(self._format_power_of_ten(log_max))

    @staticmethod
    def _format_power_of_ten(log_value: float) -> str:
        """Format a base-10 exponent as a compact string."""
        if not math.isfinite(log_value):
            log_value = _LOG_SLIDER_EXP_MIN
        log_value = min(max(float(log_value), _LOG_SLIDER_EXP_MIN), _LOG_SLIDER_EXP_MAX)
        exponent = int(round(log_value))
        if abs(log_value - exponent) < 1e-6:
            return f"10<sup>{exponent}</sup>"
        value = 10 ** log_value
        if value >= 1000 or value < 0.001:
            return f"{value:.1e}"
        return f"{value:.3g}"
