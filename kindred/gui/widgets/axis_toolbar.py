"""
Inline Axis Toolbar widget.

Control-surface contract
------------------------
- X dropdown: `t` by default. Any time-varying algebra symbol is allowed.
- Y multi-select: all time-varying species and algebra series. Scalars disabled.
- Add Guide from Scalar (exposed via Options menu stub entry).
- Parametric toggle: plot X(t) vs each Y_i(t). Plotting may resample for
  smoothness, but resampling never affects CTC (enforced in plotting layer).
- Options: sampling choice for plotting, guide management, export scoping.

This widget provides a deterministic, fully functional control strip with:
- A labeled X combobox
- A labeled, checkable Y list
- A Parametric toggle
- An Options toolbutton with a QMenu exposing actions/signals

Signals emitted
--------------
xChanged(name: str)
ySelectionChanged(selected: list[str])
parametricToggled(enabled: bool)
addGuideRequested(from_scalar: str | None)   # None means show chooser upstream
optionsRequested(action: str, data: object)  # generic hook (e.g. "sampling", ...)

Public API
----------
set_x_candidates(names: list[str], default: str | None = None)
set_y_candidates(series: list[tuple[str, bool]], *, disabled: list[str] = [])
select_y(names: list[str])
selected_y() -> list[str]
current_x() -> str
set_parametric(enabled: bool) -> None
parametric_enabled() -> bool
is_auto_range() -> bool
set_auto_range(enabled: bool) -> None
get_manual_ranges() -> tuple[float|None, float|None, float|None, float|None]
set_manual_ranges(x_min, x_max, y_min, y_max) -> None

No filesystem, no network, no cwd usage.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from PySide6.QtCore import Qt

__all__ = ["AxisToolbar"]


class AxisToolbar(QtWidgets.QWidget):
    # High-level signals
    xChanged = QtCore.Signal(str)
    ySelectionChanged = QtCore.Signal(list)
    parametricToggled = QtCore.Signal(bool)
    addGuideRequested = QtCore.Signal(object)          # scalar name or None
    optionsRequested = QtCore.Signal(str, object)      # action, data
    axisRangeChanged = QtCore.Signal()                 # axis range manually changed

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        orientation: str = "horizontal"
    ) -> None:
        super().__init__(parent)
        orientation = orientation.lower()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        self._orientation = orientation

        self._x_combo = QtWidgets.QComboBox(self)
        self._x_combo.setMinimumContentsLength(10)
        self._x_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._x_combo.setMaximumWidth(220)

        self._y_list = QtWidgets.QListWidget(self)
        self._y_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._y_list.setAlternatingRowColors(True)
        self._y_list.setUniformItemSizes(True)
        self._y_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._y_list.setMaximumHeight(160)
        self._y_list.setMinimumWidth(220)

        self._y_selector_btn = QtWidgets.QToolButton(self)
        self._y_selector_btn.setText("Y: none")
        self._y_selector_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._y_selector_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._y_menu = QtWidgets.QMenu(self)
        self._y_popup_container = QtWidgets.QWidget(self._y_menu)
        self._y_popup_container.setObjectName("axisToolbarYSelector")
        y_popup_layout = QtWidgets.QVBoxLayout(self._y_popup_container)
        y_popup_layout.setContentsMargins(6, 6, 6, 6)
        y_popup_layout.setSpacing(4)

        self._y_popup_label = QtWidgets.QLabel("Visible series", self._y_popup_container)
        y_popup_layout.addWidget(self._y_popup_label)
        y_popup_layout.addWidget(self._y_list)

        self._y_menu_action = QtWidgets.QWidgetAction(self._y_menu)
        self._y_menu_action.setDefaultWidget(self._y_popup_container)
        self._y_menu.addAction(self._y_menu_action)
        self._y_selector_btn.setMenu(self._y_menu)

        self._parametric = QtWidgets.QCheckBox("Parametric", self)
        self._parametric.setToolTip("Plot X(t) vs each selected Y(t)")

        self._options_btn = QtWidgets.QToolButton(self)
        self._options_btn.setText("Options")
        self._options_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._options_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._options_btn.setIcon(QtGui.QIcon.fromTheme("preferences-other"))

        self._menu = QtWidgets.QMenu(self)
        self._action_sampling_dense = self._menu.addAction("Sampling: Dense")
        self._action_sampling_coarse = self._menu.addAction("Sampling: Coarse")
        self._menu.addSeparator()
        self._action_add_guide = self._menu.addAction("Add Guide from Scalar…")
        self._menu.addSeparator()
        self._action_export_scope_visible = self._menu.addAction("Export Scope: Visible Series")
        self._action_export_scope_all = self._menu.addAction("Export Scope: All Series")
        self._options_btn.setMenu(self._menu)

        # Labels
        x_label = QtWidgets.QLabel("X:", self)
        y_label = QtWidgets.QLabel("Y:", self)

        # Axis range controls
        self._auto_range = QtWidgets.QCheckBox("Auto", self)
        self._auto_range.setChecked(True)
        self._auto_range.setToolTip("Automatically fit axes to data")

        self._x_min = QtWidgets.QLineEdit(self)
        self._x_min.setPlaceholderText("X min")
        self._x_min.setMaximumWidth(80)
        self._x_min.setEnabled(False)

        self._x_max = QtWidgets.QLineEdit(self)
        self._x_max.setPlaceholderText("X max")
        self._x_max.setMaximumWidth(80)
        self._x_max.setEnabled(False)

        self._y_min = QtWidgets.QLineEdit(self)
        self._y_min.setPlaceholderText("Y min")
        self._y_min.setMaximumWidth(80)
        self._y_min.setEnabled(False)

        self._y_max = QtWidgets.QLineEdit(self)
        self._y_max.setPlaceholderText("Y max")
        self._y_max.setMaximumWidth(80)
        self._y_max.setEnabled(False)

        self._manual_range_row = QtWidgets.QWidget(self)
        self._manual_range_row.setObjectName("axisToolbarManualRangeRow")
        self._manual_range_row.setVisible(False)

        # Layout
        if self._orientation == "vertical":
            self._build_vertical_layout(x_label, y_label)
        else:
            self._build_horizontal_layout(x_label, y_label)

        # Wiring
        self._x_combo.currentTextChanged.connect(self._on_x_changed)
        self._parametric.toggled.connect(self._on_parametric_toggled)
        self._y_list.itemChanged.connect(self._on_y_item_changed)

        self._action_sampling_dense.triggered.connect(lambda: self._emit_option("sampling", "dense"))
        self._action_sampling_coarse.triggered.connect(lambda: self._emit_option("sampling", "coarse"))
        self._action_add_guide.triggered.connect(lambda: self.addGuideRequested.emit(None))
        self._action_export_scope_visible.triggered.connect(lambda: self._emit_option("export_scope", "visible"))
        self._action_export_scope_all.triggered.connect(lambda: self._emit_option("export_scope", "all"))

        # Range control wiring
        self._auto_range.toggled.connect(self._on_auto_range_toggled)
        # Connect both returnPressed (Enter key) and editingFinished (focus loss)
        self._x_min.returnPressed.connect(self._on_range_changed)
        self._x_min.editingFinished.connect(self._on_range_changed)
        self._x_max.returnPressed.connect(self._on_range_changed)
        self._x_max.editingFinished.connect(self._on_range_changed)
        self._y_min.returnPressed.connect(self._on_range_changed)
        self._y_min.editingFinished.connect(self._on_range_changed)
        self._y_max.returnPressed.connect(self._on_range_changed)
        self._y_max.editingFinished.connect(self._on_range_changed)

        # Sensible defaults
        self.set_x_candidates(["t"], default="t")
        self.set_tab_order()

    # ----------------------- public API -----------------------

    def set_x_candidates(self, names: Sequence[str], default: Optional[str] = None) -> None:
        """Populate the X dropdown; if default provided and present, select it."""
        prev = self._x_combo.currentText()
        self._x_combo.blockSignals(True)
        self._x_combo.clear()
        for n in names:
            self._x_combo.addItem(str(n))
        # pick default or keep previous if still present
        if default and default in names:
            self._x_combo.setCurrentText(default)
        elif prev in names:
            self._x_combo.setCurrentText(prev)
        elif names:
            self._x_combo.setCurrentIndex(0)
        self._x_combo.blockSignals(False)
        # Emit if changed
        if self._x_combo.currentText() != prev:
            self.xChanged.emit(self._x_combo.currentText())

    def set_y_candidates(self, series: Sequence[Tuple[str, bool]], *, disabled: Sequence[str] = ()) -> None:
        """
        Populate Y list.

        Parameters
        ----------
        series : list of (name, checked)
            Names of time-varying series and whether they should start checked.
        disabled : list of names
            Items that must be visible but not selectable (e.g., scalars).
        """
        disabled_set = set(map(str, disabled))
        current = set(self.selected_y())

        self._y_list.blockSignals(True)
        self._y_list.clear()
        for name, checked in series:
            it = QtWidgets.QListWidgetItem(str(name))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Disable scalars as Y selections per spec
            if str(name) in disabled_set:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setToolTip("Scalar values cannot be plotted as Y; use 'Add Guide from Scalar' in Options.")
                it.setCheckState(Qt.CheckState.Unchecked)
            else:
                # preserve existing selection if possible
                it.setCheckState(Qt.CheckState.Checked if (str(name) in current or checked) else Qt.CheckState.Unchecked)
            self._y_list.addItem(it)
        self._y_list.blockSignals(False)
        # Emit a consolidated signal after repopulation
        self._update_y_selector_text()
        self.ySelectionChanged.emit(self.selected_y())

    def select_y(self, names: Iterable[str]) -> None:
        target = set(map(str, names))
        self._y_list.blockSignals(True)
        for i in range(self._y_list.count()):
            it = self._y_list.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsEnabled:
                it.setCheckState(Qt.CheckState.Checked if it.text() in target else Qt.CheckState.Unchecked)
        self._y_list.blockSignals(False)
        self._update_y_selector_text()
        self.ySelectionChanged.emit(self.selected_y())

    def selected_y(self) -> List[str]:
        out: List[str] = []
        for i in range(self._y_list.count()):
            it = self._y_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def current_x(self) -> str:
        return self._x_combo.currentText()

    def set_parametric(self, enabled: bool) -> None:
        if self._parametric.isChecked() == bool(enabled):
            return
        self._parametric.setChecked(bool(enabled))

    def parametric_enabled(self) -> bool:
        return self._parametric.isChecked()

    def is_auto_range(self) -> bool:
        """Return True if auto range is enabled."""
        return self._auto_range.isChecked()

    def set_auto_range(self, enabled: bool) -> None:
        """Enable or disable auto range mode."""
        if self._auto_range.isChecked() != enabled:
            self._auto_range.setChecked(enabled)

    def get_manual_ranges(self) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Get manual axis ranges.

        Returns
        -------
        (x_min, x_max, y_min, y_max) : tuple of floats or None
            Returns None for any field that is empty or invalid.
        """
        def parse_float(text: str) -> Optional[float]:
            try:
                return float(text.strip()) if text.strip() else None
            except ValueError:
                return None

        x_min = parse_float(self._x_min.text())
        x_max = parse_float(self._x_max.text())
        y_min = parse_float(self._y_min.text())
        y_max = parse_float(self._y_max.text())

        return (x_min, x_max, y_min, y_max)

    def set_manual_ranges(self, x_min: Optional[float], x_max: Optional[float],
                          y_min: Optional[float], y_max: Optional[float]) -> None:
        """
        Set manual axis ranges.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : float or None
            Axis limits. None means leave field empty.
        """
        self._x_min.setText(f"{x_min:.6g}" if x_min is not None else "")
        self._x_max.setText(f"{x_max:.6g}" if x_max is not None else "")
        self._y_min.setText(f"{y_min:.6g}" if y_min is not None else "")
        self._y_max.setText(f"{y_max:.6g}" if y_max is not None else "")

    # ----------------------- layout helpers -----------------------

    def _build_horizontal_layout(
        self,
        x_label: QtWidgets.QLabel,
        y_label: QtWidgets.QLabel
    ) -> None:
        _ = y_label
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        x_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(x_label, stretch=0)
        lay.addWidget(self._x_combo, stretch=0)
        lay.addWidget(self._y_selector_btn, stretch=0)
        lay.addWidget(self._parametric, stretch=0)
        lay.addWidget(self._auto_range, stretch=0)

        range_layout = QtWidgets.QHBoxLayout(self._manual_range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(4)
        range_layout.addWidget(QtWidgets.QLabel("X:", self._manual_range_row))
        range_layout.addWidget(self._x_min)
        range_layout.addWidget(QtWidgets.QLabel("to", self._manual_range_row))
        range_layout.addWidget(self._x_max)
        range_layout.addSpacing(8)
        range_layout.addWidget(QtWidgets.QLabel("Y:", self._manual_range_row))
        range_layout.addWidget(self._y_min)
        range_layout.addWidget(QtWidgets.QLabel("to", self._manual_range_row))
        range_layout.addWidget(self._y_max)
        lay.addWidget(self._manual_range_row, stretch=0)

        lay.addStretch(1)
        lay.addWidget(self._options_btn, stretch=0)

    def _build_vertical_layout(
        self,
        x_label: QtWidgets.QLabel,
        y_label: QtWidgets.QLabel
    ) -> None:
        max_control_width = 150

        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self._x_combo.setMaximumWidth(max_control_width)
        self._y_list.setMaximumWidth(max_control_width)
        for field in (self._x_min, self._x_max, self._y_min, self._y_max):
            field.setMaximumWidth(70)

        self._options_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._options_btn.setMaximumWidth(max_control_width)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(6)

        x_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(x_label, alignment=Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self._x_combo, alignment=Qt.AlignmentFlag.AlignLeft)

        lay.addWidget(self._parametric, alignment=Qt.AlignmentFlag.AlignLeft)

        lay.addWidget(y_label, alignment=Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self._y_list)

        range_frame = QtWidgets.QFrame(self)
        range_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        range_layout = QtWidgets.QGridLayout(range_frame)
        range_layout.setContentsMargins(0, 4, 0, 0)
        range_layout.setHorizontalSpacing(4)
        range_layout.setVerticalSpacing(2)

        range_layout.addWidget(self._auto_range, 0, 0, 1, 2)

        range_layout.addWidget(QtWidgets.QLabel("X min"), 1, 0)
        range_layout.addWidget(self._x_min, 1, 1)
        range_layout.addWidget(QtWidgets.QLabel("X max"), 2, 0)
        range_layout.addWidget(self._x_max, 2, 1)

        range_layout.addWidget(QtWidgets.QLabel("Y min"), 3, 0)
        range_layout.addWidget(self._y_min, 3, 1)
        range_layout.addWidget(QtWidgets.QLabel("Y max"), 4, 0)
        range_layout.addWidget(self._y_max, 4, 1)

        lay.addWidget(range_frame)
        lay.addStretch()
        lay.addWidget(self._options_btn, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

    def _update_y_selector_text(self) -> None:
        selected = self.selected_y()
        if not selected:
            summary = "Y: none"
            tooltip = "No visible series selected"
        elif len(selected) == 1:
            summary = f"Y: {selected[0]}"
            tooltip = selected[0]
        elif len(selected) == 2:
            summary = f"Y: {selected[0]}, {selected[1]}"
            tooltip = ", ".join(selected)
        else:
            summary = f"Y: {selected[0]} +{len(selected) - 1}"
            tooltip = ", ".join(selected)
        self._y_selector_btn.setText(summary)
        self._y_selector_btn.setToolTip(tooltip)
        self._y_selector_btn.setEnabled(self._y_list.count() > 0)

    # ----------------------- internal wiring -----------------------

    def _emit_option(self, action: str, data: object) -> None:
        self.optionsRequested.emit(action, data)

    def _on_x_changed(self, text: str) -> None:
        self.xChanged.emit(text)

    def _on_parametric_toggled(self, state: bool) -> None:
        self.parametricToggled.emit(bool(state))

    def _on_y_item_changed(self, _item: QtWidgets.QListWidgetItem) -> None:
        self._update_y_selector_text()
        self.ySelectionChanged.emit(self.selected_y())

    def _on_auto_range_toggled(self, checked: bool) -> None:
        """Enable/disable manual range controls based on auto range checkbox."""
        # When auto is checked, disable manual controls
        # When auto is unchecked, enable manual controls
        enabled = not checked
        self._x_min.setEnabled(enabled)
        self._x_max.setEnabled(enabled)
        self._y_min.setEnabled(enabled)
        self._y_max.setEnabled(enabled)
        if self._orientation == "horizontal":
            self._manual_range_row.setVisible(enabled)
            self.updateGeometry()
        # Emit signal to update plot
        self.axisRangeChanged.emit()

    def _on_range_changed(self) -> None:
        """Called when any range text field is edited."""
        # Only emit if auto range is disabled
        if not self._auto_range.isChecked():
            self.axisRangeChanged.emit()

    # ----------------------- usability tweaks -----------------------

    def sizeHint(self) -> QtCore.QSize:
        if self._orientation == "horizontal":
            return QtCore.QSize(960, 56)
        return QtCore.QSize(800, 150)

    def set_tab_order(self) -> None:
        if self._orientation == "horizontal":
            QtWidgets.QWidget.setTabOrder(self._x_combo, self._y_selector_btn)
            QtWidgets.QWidget.setTabOrder(self._y_selector_btn, self._parametric)
            QtWidgets.QWidget.setTabOrder(self._parametric, self._auto_range)
            QtWidgets.QWidget.setTabOrder(self._auto_range, self._x_min)
            QtWidgets.QWidget.setTabOrder(self._x_min, self._x_max)
            QtWidgets.QWidget.setTabOrder(self._x_max, self._y_min)
            QtWidgets.QWidget.setTabOrder(self._y_min, self._y_max)
            QtWidgets.QWidget.setTabOrder(self._y_max, self._options_btn)
            return
        QtWidgets.QWidget.setTabOrder(self._x_combo, self._parametric)
        QtWidgets.QWidget.setTabOrder(self._parametric, self._y_list)
        QtWidgets.QWidget.setTabOrder(self._y_list, self._options_btn)
