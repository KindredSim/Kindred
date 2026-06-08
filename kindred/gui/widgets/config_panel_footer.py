"""
Reusable configuration panel scaffold for FitDialog-like UIs.

Encapsulates the common pattern:
- Hint label (grey)
- Optional "dirty" status label (orange)
- Error label(s) (red)
- Optional divider
- Apply/Revert/(Reset) button row with standard enable/disable logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from PySide6 import QtCore, QtWidgets

from kindred.gui.display_name_policy import INLINE_ERROR_MAX_CHARS
from kindred.gui.ui_helpers import set_compact_label_text


@dataclass(frozen=True, slots=True)
class _ButtonsSpec:
    show_reset: bool
    order: Sequence[str]


class ConfigPanelFooter(QtWidgets.QWidget):
    applyRequested = QtCore.Signal()
    revertRequested = QtCore.Signal()
    resetRequested = QtCore.Signal()

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        hint_text: str = "Changes take effect after Apply.",
        show_hint: bool = True,
        show_dirty: bool = False,
        show_secondary_error: bool = False,
        messages_position: str = "before_body",
        show_divider: bool = False,
        show_reset: bool = False,
        apply_requires_no_error: bool = False,
        button_order: Optional[Sequence[str]] = None,
        hint_object_name: Optional[str] = None,
        dirty_object_name: Optional[str] = None,
        error_object_name: Optional[str] = None,
        secondary_error_object_name: Optional[str] = None,
        apply_object_name: Optional[str] = None,
        revert_object_name: Optional[str] = None,
        reset_object_name: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        self._apply_requires_no_error = bool(apply_requires_no_error)
        self._show_dirty = bool(show_dirty)
        self._dirty = False
        self._error_text = ""
        self._apply_enabled_override: Optional[bool] = None
        self._revert_enabled_override: Optional[bool] = None

        messages_position_norm = str(messages_position or "before_body").strip().lower()
        if messages_position_norm not in ("before_body", "after_body"):
            raise ValueError(
                "ConfigPanelFooter messages_position must be 'before_body' or 'after_body'. "
                f"Got: {messages_position!r}."
            )
        self._messages_position = messages_position_norm

        if button_order is None:
            button_order = ("reset", "revert", "apply") if show_reset else ("apply", "revert")

        normalized_order = self._normalize_button_order_tokens(button_order)
        self._buttons_spec = _ButtonsSpec(show_reset=bool(show_reset), order=normalized_order)
        self._validate_button_order(self._buttons_spec.order, show_reset=self._buttons_spec.show_reset)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.hint_label = QtWidgets.QLabel(str(hint_text))
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("font-size: 11px;")
        if hint_object_name:
            self.hint_label.setObjectName(str(hint_object_name))
        self.hint_label.setVisible(bool(show_hint))
        layout.addWidget(self.hint_label)

        self.dirty_label = QtWidgets.QLabel("Pending changes")
        self.dirty_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        if dirty_object_name:
            self.dirty_label.setObjectName(str(dirty_object_name))
        self.dirty_label.setVisible(False)
        layout.addWidget(self.dirty_label)
        if not self._show_dirty:
            self.dirty_label.hide()

        self._messages = QtWidgets.QWidget(self)
        messages_layout = QtWidgets.QVBoxLayout(self._messages)
        messages_layout.setContentsMargins(0, 0, 0, 0)
        messages_layout.setSpacing(6)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(False)
        self.error_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        if error_object_name:
            self.error_label.setObjectName(str(error_object_name))
        self.error_label.setVisible(False)
        messages_layout.addWidget(self.error_label)

        self.secondary_error_label: Optional[QtWidgets.QLabel]
        if show_secondary_error:
            secondary = QtWidgets.QLabel("")
            secondary.setWordWrap(False)
            secondary.setStyleSheet("font-weight: bold; font-size: 11px;")
            if secondary_error_object_name:
                secondary.setObjectName(str(secondary_error_object_name))
            secondary.setVisible(False)
            messages_layout.addWidget(secondary)
            self.secondary_error_label = secondary
        else:
            self.secondary_error_label = None

        if self._messages_position == "before_body":
            layout.addWidget(self._messages)

        self.body = QtWidgets.QWidget(self)
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        layout.addWidget(self.body, stretch=1)

        if self._messages_position == "after_body":
            layout.addWidget(self._messages)

        self.divider: Optional[QtWidgets.QFrame]
        if show_divider:
            divider = QtWidgets.QFrame(self)
            divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            layout.addWidget(divider)
            self.divider = divider
        else:
            self.divider = None

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)

        self.apply_button = QtWidgets.QPushButton("Apply", self)
        self.apply_button.setEnabled(False)
        if apply_object_name:
            self.apply_button.setObjectName(str(apply_object_name))
        self.apply_button.clicked.connect(self.applyRequested.emit)

        self.revert_button = QtWidgets.QPushButton("Revert", self)
        self.revert_button.setEnabled(False)
        if revert_object_name:
            self.revert_button.setObjectName(str(revert_object_name))
        self.revert_button.clicked.connect(self.revertRequested.emit)

        self.reset_button: Optional[QtWidgets.QPushButton]
        if self._buttons_spec.show_reset:
            reset = QtWidgets.QPushButton("Reset", self)
            reset.setEnabled(False)
            if reset_object_name:
                reset.setObjectName(str(reset_object_name))
            reset.clicked.connect(self.resetRequested.emit)
            self.reset_button = reset
        else:
            self.reset_button = None

        for token in self._buttons_spec.order:
            if token == "reset":
                if self.reset_button is not None:
                    buttons.addWidget(self.reset_button)
            elif token == "revert":
                buttons.addWidget(self.revert_button)
            elif token == "apply":
                buttons.addWidget(self.apply_button)

        layout.addLayout(buttons)

    @staticmethod
    def _normalize_button_order_tokens(order: Sequence[str]) -> tuple[str, ...]:
        return tuple(str(x).strip().lower() for x in order)

    @staticmethod
    def _validate_button_order(order: Sequence[str], *, show_reset: bool) -> None:
        allowed = {"apply", "revert", "reset"}
        normalized = [str(x).strip().lower() for x in order]
        if not normalized:
            raise ValueError("ConfigPanelFooter button_order must not be empty.")
        bad = [x for x in normalized if x not in allowed]
        if bad:
            raise ValueError(f"ConfigPanelFooter button_order has invalid entries: {bad!r}.")
        if "apply" not in normalized or "revert" not in normalized:
            raise ValueError("ConfigPanelFooter button_order must include 'apply' and 'revert'.")
        if show_reset and "reset" not in normalized:
            raise ValueError("ConfigPanelFooter button_order must include 'reset' when show_reset=True.")
        if (not show_reset) and "reset" in normalized:
            raise ValueError("ConfigPanelFooter button_order must not include 'reset' when show_reset=False.")

    def set_hint(self, text: str) -> None:
        self.hint_label.setText(str(text))

    def set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        if self._show_dirty:
            self.dirty_label.setVisible(self._dirty)
        self._refresh_buttons_enabled_state()

    def set_error(self, message: Optional[str], *, tooltip_text: Optional[str] = None) -> None:
        text = str(message or "")
        self._error_text = text
        if text:
            set_compact_label_text(
                self.error_label,
                text,
                max_chars=INLINE_ERROR_MAX_CHARS,
                max_width=520,
                diagnostic=True,
                tooltip_text=tooltip_text,
            )
            self.error_label.setVisible(True)
        else:
            self.error_label.clear()
            self.error_label.setVisible(False)
        self._refresh_buttons_enabled_state()

    def set_secondary_error(self, message: Optional[str], *, tooltip_text: Optional[str] = None) -> None:
        if self.secondary_error_label is None:
            return
        text = str(message or "")
        if text:
            set_compact_label_text(
                self.secondary_error_label,
                text,
                max_chars=INLINE_ERROR_MAX_CHARS,
                max_width=520,
                diagnostic=True,
                tooltip_text=tooltip_text,
            )
            self.secondary_error_label.setVisible(True)
        else:
            self.secondary_error_label.clear()
            self.secondary_error_label.setVisible(False)

    def set_apply_enabled_override(self, enabled: Optional[bool]) -> None:
        """
        Optional escape hatch: allow callers to fully control Apply enabled state.

        When enabled is None, the button state is driven by dirty/error state.
        """
        self._apply_enabled_override = enabled
        self._refresh_buttons_enabled_state()

    def set_revert_enabled_override(self, enabled: Optional[bool]) -> None:
        self._revert_enabled_override = enabled
        self._refresh_buttons_enabled_state()

    def _refresh_buttons_enabled_state(self) -> None:
        computed_revert_enabled = bool(self._dirty)
        if self._apply_requires_no_error:
            computed_apply_enabled = bool(self._dirty) and not bool(self._error_text)
        else:
            computed_apply_enabled = bool(self._dirty)

        self.revert_button.setEnabled(
            computed_revert_enabled if self._revert_enabled_override is None else bool(self._revert_enabled_override)
        )
        self.apply_button.setEnabled(
            computed_apply_enabled if self._apply_enabled_override is None else bool(self._apply_enabled_override)
        )
