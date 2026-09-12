"""
Inline Axis Toolbar widget.

Control-surface contract
------------------------
- X dropdown: typed simulation or experimental identity, with `t` by default.
- Y multi-select: all time-varying species and algebra series. Scalars disabled.
- Display units and a separate row of compact manual axis limits.

This widget provides a deterministic, fully functional control strip with:
- A labeled X combobox
- A labeled, checkable Y list
- Unit selectors and manual range editors

Signals emitted
--------------
xChanged(selection: SimulationSeriesAxisSelection | ExperimentalAxisSelection)
ySelectionChanged(selected: tuple[AxisSeriesSelection, ...])
unitChanged(axis: str, unit: str)

Public API
----------
set_x_candidates(candidates, *, selected)
set_y_candidates(candidates, *, disabled=(), locked_checked=None)
selected_y() -> tuple[AxisSeriesSelection, ...]
current_x() -> AxisSeriesSelection
set_manual_range_controls(*, auto_x: bool, auto_y: bool, enabled: bool) -> None
get_manual_ranges() -> tuple[float|None, float|None, float|None, float|None]
set_manual_ranges(x_min, x_max, y_min, y_max) -> None

Programmatic projection setters update widget state without emitting command
signals; user edits and explicit actions emit the signals above.

No filesystem, no network, no cwd usage.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from PySide6 import QtCore, QtWidgets

from PySide6.QtCore import Qt

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)
from kindred.gui.plot_config import AxisSelectionCandidate
from kindred.gui.ui_helpers import ControlRowsLayout, configure_compact_numeric_edit
from kindred.gui.display_name_policy import compact_text, literal_tooltip

__all__ = ["AxisToolbar"]


class AxisToolbar(QtWidgets.QWidget):
    # High-level signals
    xChanged = QtCore.Signal(object)
    ySelectionChanged = QtCore.Signal(object)
    unitChanged = QtCore.Signal(str, str)
    axisRangeChanged = QtCore.Signal(str)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._x_combo = QtWidgets.QComboBox(self)
        self._x_combo.setMinimumContentsLength(4)
        self._x_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._x_combo.setMaximumWidth(220)
        self._x_combo.setToolTip('Select X-axis Variable (time, species, or observables defined with "let … = …").')

        self._x_unit_combo = QtWidgets.QComboBox(self)
        self._x_unit_combo.setObjectName("axisToolbarXUnit")
        self._x_unit_combo.setMinimumWidth(64)
        self._x_unit_combo.setMaximumWidth(84)
        self._x_unit_combo.setToolTip("Select units for the X axis and copied plot data. Available when plotting time or concentration.")

        self._y_unit_combo = QtWidgets.QComboBox(self)
        self._y_unit_combo.setObjectName("axisToolbarYUnit")
        self._y_unit_combo.setMinimumWidth(64)
        self._y_unit_combo.setMaximumWidth(84)
        self._y_unit_combo.setToolTip("Select units for the Y axis and copied plot data. Available only when all plotted Y variables are concentrations.")

        self._y_list = QtWidgets.QListWidget(self)
        self._y_list.setToolTip('Select Y-axis Variables (species or observables defined with "let … = …").')
        self._y_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._y_list.setAlternatingRowColors(True)
        self._y_list.setUniformItemSizes(True)
        self._y_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._y_list.setMaximumHeight(160)
        self._y_list.setMinimumWidth(220)
        self._y_list.setMaximumWidth(420)
        self._locked_checked_y: frozenset[AxisSeriesSelection] = frozenset()

        self._y_selector_btn = QtWidgets.QToolButton(self)
        self._y_selector_btn.setText("Y: none")
        self._y_selector_btn.setToolTip(self._y_list.toolTip())
        self._y_selector_btn.setMaximumWidth(220)
        self._y_selector_btn.setMinimumWidth(80)
        self._y_selector_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self._y_selector_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._y_selector_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._y_menu = QtWidgets.QMenu(self)
        self._y_popup_container = QtWidgets.QWidget(self._y_menu)
        self._y_popup_container.setObjectName("axisToolbarYSelector")
        y_popup_layout = QtWidgets.QVBoxLayout(self._y_popup_container)
        y_popup_layout.setContentsMargins(6, 6, 6, 6)
        y_popup_layout.setSpacing(4)

        self._y_popup_label = QtWidgets.QLabel("Visible series", self._y_popup_container)
        self._invert_y_selection_btn = QtWidgets.QPushButton("Invert selection", self._y_popup_container)
        self._invert_y_selection_btn.setObjectName("axisToolbarInvertYSelectionButton")
        self._invert_y_selection_btn.setToolTip("Invert the selected Y series")
        y_popup_layout.addWidget(self._y_popup_label)
        y_popup_layout.addWidget(self._invert_y_selection_btn)
        y_popup_layout.addWidget(self._y_list)

        self._y_menu_action = QtWidgets.QWidgetAction(self._y_menu)
        self._y_menu_action.setDefaultWidget(self._y_popup_container)
        self._y_menu.addAction(self._y_menu_action)
        self._y_selector_btn.setMenu(self._y_menu)

        # Labels
        x_label = QtWidgets.QLabel("X:", self)
        x_label.setBuddy(self._x_combo)
        self._x_combo.setAccessibleName("X axis")
        self._x_unit_combo.setAccessibleName("X axis unit")
        self._y_unit_combo.setAccessibleName("Y axis unit")

        # Axis range controls
        self._x_min = QtWidgets.QLineEdit(self)
        self._x_min.setPlaceholderText("X min")
        configure_compact_numeric_edit(self._x_min, characters=11)
        self._x_min.setEnabled(False)

        self._x_max = QtWidgets.QLineEdit(self)
        self._x_max.setPlaceholderText("X max")
        configure_compact_numeric_edit(self._x_max, characters=11)
        self._x_max.setEnabled(False)

        self._y_min = QtWidgets.QLineEdit(self)
        self._y_min.setPlaceholderText("Y min")
        configure_compact_numeric_edit(self._y_min, characters=11)
        self._y_min.setEnabled(False)

        self._y_max = QtWidgets.QLineEdit(self)
        self._y_max.setPlaceholderText("Y max")
        configure_compact_numeric_edit(self._y_max, characters=11)
        self._y_max.setEnabled(False)

        self._manual_range_row = QtWidgets.QWidget(self)
        self._manual_range_row.setObjectName("axisToolbarManualRangeRow")
        self._manual_range_row.setVisible(False)

        # Layout
        self._build_horizontal_layout(x_label)

        # Wiring
        self._x_combo.currentIndexChanged.connect(self._on_x_changed)
        self._x_unit_combo.currentTextChanged.connect(
            lambda unit: self.unitChanged.emit("x", unit)
        )
        self._y_unit_combo.currentTextChanged.connect(
            lambda unit: self.unitChanged.emit("y", unit)
        )
        self._y_list.itemChanged.connect(self._on_y_item_changed)
        self._invert_y_selection_btn.clicked.connect(self.invert_y_selection)

        # Range control wiring
        for axis, edits in (("x", (self._x_min, self._x_max)), ("y", (self._y_min, self._y_max))):
            for edit in edits:
                edit.editingFinished.connect(lambda key=axis: self._on_range_changed(key))

        for edit, name in ((self._x_min, "X minimum"), (self._x_max, "X maximum"),
                           (self._y_min, "Y minimum"), (self._y_max, "Y maximum")):
            edit.setAccessibleName(name)

        # Sensible defaults
        time_selection = SimulationSeriesAxisSelection("t")
        self.set_x_candidates(
            (AxisSelectionCandidate(time_selection, "t"),),
            selected=time_selection,
        )
        self.set_axis_unit_choices(
            x_units=("s",),
            selected_x="s",
            y_units=(),
            selected_y=None,
        )
        self.set_tab_order()

    # ----------------------- public API -----------------------

    def set_x_candidates(
        self,
        candidates: Sequence[AxisSelectionCandidate],
        *,
        selected: AxisSeriesSelection,
    ) -> None:
        """Render one already-reconciled typed X selection exactly."""

        values = tuple(candidates)
        if any(not isinstance(item, AxisSelectionCandidate) for item in values):
            raise TypeError(
                "X candidates must contain AxisSelectionCandidate values."
            )
        identities = tuple(item.selection for item in values)
        if len(identities) != len(set(identities)):
            raise ValueError("X candidate identities must be unique.")
        if not isinstance(
            selected,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError("Selected X must be a typed axis identity.")
        if selected not in identities:
            raise ValueError(
                "Selected X must already be reconciled into the candidate roster."
            )
        self._x_combo.blockSignals(True)
        try:
            self._x_combo.clear()
            for candidate in values:
                self._x_combo.addItem(
                    candidate.label,
                    candidate.selection,
                )
                self._x_combo.setItemData(self._x_combo.count() - 1, literal_tooltip(candidate.label), Qt.ToolTipRole)
            self._x_combo.setCurrentIndex(identities.index(selected))
        finally:
            self._x_combo.blockSignals(False)

    def set_y_candidates(
        self,
        candidates: Sequence[tuple[AxisSelectionCandidate, bool]],
        *,
        disabled: Sequence[AxisSeriesSelection] = (),
        locked_checked: Mapping[AxisSeriesSelection, str] | None = None,
    ) -> None:
        """
        Populate Y list.

        Parameters
        ----------
        candidates
            Typed candidates paired with their projected checked state.
        disabled
            Typed identities that must remain visible but not selectable.
        locked_checked
            Derived overlay identities that remain checked and emit no
            authored Y selection, mapped to their model-series identity.
        """
        values = tuple(candidates)
        if any(
            not isinstance(candidate, AxisSelectionCandidate)
            or type(checked) is not bool
            for candidate, checked in values
        ):
            raise TypeError(
                "Y candidates must pair AxisSelectionCandidate with bool."
            )
        identities = tuple(candidate.selection for candidate, _ in values)
        if len(identities) != len(set(identities)):
            raise ValueError("Y candidate identities must be unique.")
        disabled_set = set(disabled)
        locked_checked_by_identity = dict(locked_checked or {})
        if any(identity not in identities for identity in disabled_set):
            raise ValueError("Disabled Y identities must exist in candidates.")
        if any(
            not isinstance(identity, ExperimentalAxisSelection)
            or identity not in identities
            for identity in locked_checked_by_identity
        ):
            raise ValueError(
                "Locked-checked Y identities must be active experimental "
                "candidates."
            )
        if disabled_set.intersection(locked_checked_by_identity):
            raise ValueError(
                "Disabled and locked-checked Y identities must be distinct."
            )
        for model_identity in locked_checked_by_identity.values():
            if (
                type(model_identity) is not str
                or not model_identity
                or model_identity != model_identity.strip()
            ):
                raise ValueError(
                    "Locked-checked Y model identities must be nonblank "
                    "edge-normalized strings."
                )
        self._locked_checked_y = frozenset(locked_checked_by_identity)

        self._y_list.blockSignals(True)
        self._y_list.clear()
        for candidate, checked in values:
            it = QtWidgets.QListWidgetItem(candidate.label)
            it.setToolTip(literal_tooltip(candidate.label))
            it.setData(
                Qt.ItemDataRole.UserRole,
                candidate.selection,
            )
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Disable scalars as Y selections per spec
            if candidate.selection in disabled_set:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setToolTip(literal_tooltip(candidate.label + "\nScalar values cannot be plotted as Y."))
                it.setCheckState(Qt.CheckState.Unchecked)
            elif candidate.selection in locked_checked_by_identity:
                it.setFlags(
                    it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable
                )
                it.setToolTip(literal_tooltip(
                    candidate.label + "\nShown because this active overlay maps to "
                    f"{locked_checked_by_identity[candidate.selection]}; "
                    "disable it in Plot Overlays."
                ))
                it.setCheckState(Qt.CheckState.Checked)
            else:
                it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._y_list.addItem(it)
        self._y_list.blockSignals(False)
        self._update_y_selector_text()

    def set_axis_unit_choices(
        self,
        *,
        x_units: Sequence[str],
        selected_x: str | None,
        y_units: Sequence[str],
        selected_y: str | None,
        transformed: bool = False,
    ) -> None:
        """Project the available display-only units without emitting intent."""

        self._set_unit_combo(self._x_unit_combo, x_units, selected_x)
        self._set_unit_combo(self._y_unit_combo, y_units, selected_y)
        self._x_unit_label.setText("Source X unit:" if transformed else "Unit:")
        self._y_unit_label.setText("Source Y unit:" if transformed else "Unit:")

    @staticmethod
    def _set_unit_combo(
        combo: QtWidgets.QComboBox,
        units: Sequence[str],
        selected: str | None,
    ) -> None:
        choices = tuple(str(unit) for unit in units)
        blocker = QtCore.QSignalBlocker(combo)
        try:
            combo.clear()
            if not choices:
                combo.addItem("—")
                combo.setEnabled(False)
                return
            combo.addItems(choices)
            combo.setCurrentText(
                selected if selected in choices else choices[0]
            )
            combo.setEnabled(True)
        finally:
            del blocker

    def invert_y_selection(self) -> None:
        """Invert enabled Y-series rows while leaving disabled scalar rows unchecked."""
        self._y_list.blockSignals(True)
        for i in range(self._y_list.count()):
            it = self._y_list.item(i)
            selection = it.data(Qt.ItemDataRole.UserRole)
            if selection in self._locked_checked_y:
                it.setCheckState(Qt.CheckState.Checked)
                continue
            if not (it.flags() & Qt.ItemFlag.ItemIsEnabled):
                it.setCheckState(Qt.CheckState.Unchecked)
                continue
            next_state = (
                Qt.CheckState.Unchecked
                if it.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            it.setCheckState(next_state)
        self._y_list.blockSignals(False)
        self._update_y_selector_text()
        self.ySelectionChanged.emit(self.selected_y())

    def selected_y(self) -> tuple[AxisSeriesSelection, ...]:
        out: list[AxisSeriesSelection] = []
        for i in range(self._y_list.count()):
            it = self._y_list.item(i)
            if (
                it.flags() & Qt.ItemFlag.ItemIsEnabled
                and it.checkState() == Qt.CheckState.Checked
            ):
                selection = it.data(Qt.ItemDataRole.UserRole)
                if not isinstance(
                    selection,
                    (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
                ):
                    raise TypeError(
                        "Y selector item lacks a typed axis identity."
                    )
                if selection in self._locked_checked_y:
                    continue
                out.append(selection)
        return tuple(out)

    def current_x(self) -> AxisSeriesSelection:
        selection = self._x_combo.currentData()
        if not isinstance(
            selection,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError("X selector lacks a typed axis identity.")
        return selection

    def set_manual_range_controls(self, *, auto_x: bool, auto_y: bool, enabled: bool) -> None:
        """Project the plot owner's range mode and gesture restriction."""
        visible = not (auto_x and auto_y)
        self._manual_range_row.setVisible(visible)
        self._manual_range_row.setEnabled(visible and enabled)
        for axis, auto, edits in (("X", auto_x, (self._x_min, self._x_max)), ("Y", auto_y, (self._y_min, self._y_max))):
            self._range_labels[axis].setText(f"{axis} ({'automatic' if auto else 'manual'}):")
            for edit in edits:
                edit.setEnabled(not auto and enabled)
        self.updateGeometry()

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
                value = float(text.strip()) if text.strip() else None
            except ValueError:
                return None
            return value if value is None or math.isfinite(value) else None

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
        def _format(value: Optional[float]) -> str:
            return format(float(value), ".6g") if value is not None else ""

        self._x_min.setText(_format(x_min))
        self._x_max.setText(_format(x_max))
        self._y_min.setText(_format(y_min))
        self._y_max.setText(_format(y_max))

    # ----------------------- layout helpers -----------------------

    def _build_horizontal_layout(
        self,
        x_label: QtWidgets.QLabel,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)
        selectors = QtWidgets.QWidget(self)
        toolbar_rows = ControlRowsLayout(selectors)
        layout.addWidget(selectors)
        layout.addWidget(self._manual_range_row)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._x_combo.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        toolbar_groups = []
        self._x_unit_label = QtWidgets.QLabel("Unit:", self)
        for label, editor in ((x_label, self._x_combo),
                              (self._x_unit_label, self._x_unit_combo)):
            label.setToolTip(editor.toolTip())
            group = QtWidgets.QWidget(self)
            group_layout = QtWidgets.QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(8)
            group_layout.addWidget(label)
            group_layout.addWidget(editor)
            toolbar_groups.append(group)
        toolbar_groups.append(self._y_selector_btn)
        unit_group = QtWidgets.QWidget(self)
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setSpacing(8)
        self._y_unit_label = QtWidgets.QLabel("Unit:", self)
        self._y_unit_label.setToolTip(self._y_unit_combo.toolTip())
        unit_layout.addWidget(self._y_unit_label)
        unit_layout.addWidget(self._y_unit_combo)
        toolbar_groups.append(unit_group)

        range_layout = ControlRowsLayout(self._manual_range_row)
        range_layout.setSpacing(4)
        self._range_labels = {}
        for axis, lower, upper in (("X", self._x_min, self._x_max),
                                   ("Y", self._y_min, self._y_max)):
            group = QtWidgets.QWidget(self._manual_range_row)
            pair = QtWidgets.QHBoxLayout(group)
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            self._range_labels[axis] = QtWidgets.QLabel(axis, group)
            pair.addWidget(self._range_labels[axis])
            pair.addWidget(lower)
            pair.addWidget(QtWidgets.QLabel("to", group))
            pair.addWidget(upper)
            range_layout.addWidget(group)
        for widget in toolbar_groups:
            toolbar_rows.addWidget(widget)

    def _update_y_selector_text(self) -> None:
        selected_labels = [
            self._y_list.item(index).text()
            for index in range(self._y_list.count())
            if (
                self._y_list.item(index).flags()
                & Qt.ItemFlag.ItemIsEnabled
                and self._y_list.item(index).checkState()
                == Qt.CheckState.Checked
            )
        ]
        if not selected_labels:
            summary = "Y: none"
            tooltip = "No visible series selected"
        elif len(selected_labels) == 1:
            summary = f"Y: {selected_labels[0]}"
            tooltip = selected_labels[0]
        elif len(selected_labels) == 2:
            summary = f"Y: {selected_labels[0]}, {selected_labels[1]}"
            tooltip = ", ".join(selected_labels)
        else:
            summary = (
                f"Y: {selected_labels[0]} +{len(selected_labels) - 1}"
            )
            tooltip = ", ".join(selected_labels)
        self._y_selector_btn.setText(compact_text(summary, max_chars=30).display)
        self._y_selector_btn.setAccessibleName(summary)
        self._y_selector_btn.setToolTip(literal_tooltip(self._y_list.toolTip() + "\n\n" + tooltip))
        self._y_selector_btn.setEnabled(self._y_list.count() > 0)

    # ----------------------- internal wiring -----------------------

    def _on_x_changed(self, _index: int) -> None:
        if self._x_combo.count():
            self.xChanged.emit(self.current_x())

    def _on_y_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        selection = item.data(Qt.ItemDataRole.UserRole)
        if selection in self._locked_checked_y:
            blocker = QtCore.QSignalBlocker(self._y_list)
            try:
                item.setCheckState(Qt.CheckState.Checked)
            finally:
                del blocker
            self._update_y_selector_text()
            return
        self._update_y_selector_text()
        self.ySelectionChanged.emit(self.selected_y())

    def _on_range_changed(self, axis: str) -> None:
        if self._manual_range_row.isEnabled():
            self.axisRangeChanged.emit(axis)

    def set_tab_order(self) -> None:
        QtWidgets.QWidget.setTabOrder(self._x_combo, self._x_unit_combo)
        QtWidgets.QWidget.setTabOrder(self._x_unit_combo, self._y_selector_btn)
        QtWidgets.QWidget.setTabOrder(self._y_selector_btn, self._y_unit_combo)
        QtWidgets.QWidget.setTabOrder(self._y_unit_combo, self._x_min)
        QtWidgets.QWidget.setTabOrder(self._x_min, self._x_max)
        QtWidgets.QWidget.setTabOrder(self._x_max, self._y_min)
        QtWidgets.QWidget.setTabOrder(self._y_min, self._y_max)
