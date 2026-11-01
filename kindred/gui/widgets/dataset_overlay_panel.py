"""Snapshot renderer and intent source for active-tab dataset overlays."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.foundation import RawDatasetRecord
from kindred.gui.dataset_payload_snapshot import ActiveDatasetLinkSnapshot
from kindred.gui.display_name_policy import literal_tooltip
from kindred.core.validation import parse_whole_number_text
from kindred.gui.ui_helpers import INVALID_EDITOR_STYLE, ControlRowsLayout, configure_compact_numeric_edit

__all__ = ["DatasetOverlayPanel"]


class _MarkerControls(QtWidgets.QWidget):
    """Keep small marker settings on one row, wrapping when necessary."""

    def __init__(self, fields):
        super().__init__()
        self._rows = ControlRowsLayout(self)
        for title, control, tooltip in fields:
            field = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(field)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            label = QtWidgets.QLabel(title)
            label.setBuddy(control)
            label.setToolTip(tooltip)
            control.setToolTip(tooltip)
            row.addWidget(label)
            row.addWidget(control)
            self._rows.addWidget(field)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setFixedHeight(self._rows.heightForWidth(self.width()))


class DatasetOverlayPanel(QtWidgets.QWidget):
    """Render global records with persisted active-tab overlay membership."""

    overlaySelectionRequested = QtCore.Signal(str, object)
    speciesColorRequested = QtCore.Signal(str, object)
    experimentalPointSizeRequested = QtCore.Signal(int)
    experimentalPointOpacityRequested = QtCore.Signal(float)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        import_action: QtGui.QAction,
        color_for_species: Callable[[str, tuple[str, ...]], QtGui.QColor],
        experimental_point_size: Callable[[], int],
        experimental_point_opacity: Callable[[], float],
    ) -> None:
        super().__init__(parent)
        self._color_for_species = color_for_species
        self._experimental_point_size = experimental_point_size
        self._experimental_point_opacity = experimental_point_opacity
        self._links_by_dataset_id: dict[str, ActiveDatasetLinkSnapshot] = {}
        self._known_species: tuple[str, ...] = ()
        # Only mapped columns have a swatch: the colour is the mapped quantity's.
        self._color_buttons: dict[tuple[str, str], QtWidgets.QPushButton] = {}
        self._color_targets: dict[tuple[str, str], str] = {}
        self._has_rendered_snapshot = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._empty = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self._empty)
        empty_layout.addStretch()
        label = QtWidgets.QLabel("Import experimental data to show it on plots.")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(label)
        import_button = QtWidgets.QPushButton(import_action.text())
        import_button.setToolTip(import_action.toolTip())
        import_button.setEnabled(import_action.isEnabled())
        import_button.clicked.connect(import_action.trigger)
        import_action.changed.connect(lambda: import_button.setEnabled(import_action.isEnabled()))
        empty_layout.addWidget(import_button, 0, Qt.AlignmentFlag.AlignCenter)
        formats = QtWidgets.QLabel("CSV or Excel files")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(formats)
        empty_layout.addStretch()
        layout.addWidget(self._empty, 1)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Show on plot", "Color"])
        self._tree.setAccessibleName("Experimental dataset overlays")
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.hide()
        layout.addWidget(self._tree, 1)

        self._appearance = QtWidgets.QWidget()
        appearance_layout = QtWidgets.QVBoxLayout(self._appearance)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_layout.setSpacing(6)
        title = QtWidgets.QLabel("Marker Appearance")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        appearance_layout.addWidget(title)
        self._size_edit = QtWidgets.QLineEdit()
        self._size_edit.setAccessibleName("Experimental point size")
        self._size_edit.editingFinished.connect(self._on_size_edited)
        self._opacity_edit = QtWidgets.QLineEdit()
        self._opacity_edit.setAccessibleName("Experimental point opacity (%)")
        self._opacity_edit.editingFinished.connect(self._on_opacity_edited)
        for edit in (self._size_edit, self._opacity_edit):
            configure_compact_numeric_edit(edit, characters=4)
        appearance_layout.addWidget(_MarkerControls([
            ("Size", self._size_edit, "Size of Marker (1–20)."),
            ("Opacity (%)", self._opacity_edit, "Opacity of Marker (0–100%)."),
        ]))
        # A box whose text differs from the current setting's value holds a
        # typed draft that projection leaves alone; a rejected one is named here.
        self._shown_marker_text = {self._size_edit: "", self._opacity_edit: ""}
        self._invalid_marker_edits: dict[QtWidgets.QLineEdit, str] = {}
        self._marker_error = QtWidgets.QLabel()
        self._marker_error.setWordWrap(True)
        self._marker_error.hide()
        appearance_layout.addWidget(self._marker_error)
        scope = QtWidgets.QLabel("Applies to experimental data in all plots.")
        scope.setWordWrap(True)
        appearance_layout.addWidget(scope)
        layout.addWidget(self._appearance)
        self._appearance.hide()

    def render_snapshot(
        self,
        *,
        records: Iterable[RawDatasetRecord],
        active_links: Iterable[ActiveDatasetLinkSnapshot],
        known_species: Iterable[str],
    ) -> None:
        """Render one snapshot while retaining expansion and scroll position."""
        rendered_records = tuple(records)
        expanded_dataset_ids = {
            str(item.data(0, Qt.UserRole))
            for index in range(self._tree.topLevelItemCount())
            if (item := self._tree.topLevelItem(index)).isExpanded()
        }
        vertical_scroll = self._tree.verticalScrollBar().value()
        horizontal_scroll = self._tree.horizontalScrollBar().value()
        self._links_by_dataset_id = {
            link.dataset_id: link for link in active_links
        }
        self._known_species = tuple(known_species)
        self._color_buttons.clear()
        self._color_targets.clear()
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for record in rendered_records:
                # The snapshot carries one link for every dataset.
                dataset_id = record.dataset_id
                link = self._links_by_dataset_id[dataset_id]
                enabled_raw_ids = set(link.overlay_raw_observation_ids)
                parent = QtWidgets.QTreeWidgetItem([record.display_name, ""])
                parent.setToolTip(0, literal_tooltip(record.display_name))
                parent.setData(0, Qt.ItemDataRole.AccessibleTextRole, record.display_name)
                parent.setData(0, Qt.UserRole, dataset_id)
                parent.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                self._tree.addTopLevelItem(parent)
                for series in record.observations:
                    column = series.column_name
                    raw_observation_id = series.raw_observation_id
                    mapped_species = dict(link.observation_mapping).get(raw_observation_id)
                    display_name = (
                        f"{column} → {mapped_species}"
                        if mapped_species
                        else column
                    )
                    child = QtWidgets.QTreeWidgetItem([display_name, ""])
                    child.setToolTip(0, literal_tooltip(f"{record.display_name}: {display_name}"))
                    child.setData(0, Qt.ItemDataRole.AccessibleTextRole, display_name)
                    child.setData(0, Qt.ItemDataRole.AccessibleDescriptionRole, record.display_name)
                    child.setData(0, Qt.UserRole, dataset_id)
                    child.setData(
                        0,
                        Qt.UserRole + 1,
                        raw_observation_id,
                    )
                    child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if raw_observation_id in enabled_raw_ids
                        else Qt.CheckState.Unchecked,
                    )
                    parent.addChild(child)
                    if mapped_species:
                        key = (dataset_id, raw_observation_id)
                        button = self._create_color_button(key, mapped_species)
                        self._tree.setItemWidget(child, 1, button)
                        self._color_buttons[key] = button
                        self._color_targets[key] = mapped_species
                parent.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if enabled_raw_ids
                    else Qt.CheckState.Unchecked,
                )
                parent.setExpanded(
                    dataset_id in expanded_dataset_ids
                    if self._has_rendered_snapshot
                    else bool(enabled_raw_ids)
                )
        finally:
            self._tree.blockSignals(False)
        self._has_rendered_snapshot = True
        has_records = bool(rendered_records)
        self._tree.setVisible(has_records)
        self._empty.setVisible(not has_records)
        self._appearance.setVisible(has_records)
        self._render_application_style()
        self._refresh_color_swatches()
        self._tree.verticalScrollBar().setValue(vertical_scroll)
        self._tree.horizontalScrollBar().setValue(horizontal_scroll)

    def render_overlay_selection(
        self,
        *,
        dataset_id: str,
        raw_observation_ids: Iterable[str],
    ) -> None:
        """Project one accepted membership change without rebuilding the tree."""

        target = dataset_id
        link = self._links_by_dataset_id[target]
        selected = tuple(raw_observation_ids)
        self._links_by_dataset_id[target] = replace(
            link,
            overlay_raw_observation_ids=selected,
        )
        self._render_authoritative_overlay_selection()

    def refresh_application_style_controls(self) -> None:
        """Rerender application-wide experimental point style controls."""

        self._render_application_style()

    def refresh_color_swatches(
        self,
        *,
        known_species: Iterable[str],
    ) -> None:
        """Rerender only application-wide color swatches."""

        self._known_species = tuple(known_species)
        self._refresh_color_swatches()

    def _refresh_color_swatches(self) -> None:
        for key, button in self._color_buttons.items():
            color = self._color_for_species(
                self._color_targets[key],
                self._known_species,
            )
            button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                "border: 1px solid; border-radius: 3px;"
            )

    def _create_color_button(
        self,
        key: tuple[str, str],
        target: str,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setFixedSize(30, 20)
        button.setFlat(True)
        button.setAccessibleName(f"Set display color for {target}")
        button.setToolTip(
            literal_tooltip(f"Set the global display color for {target}.")
        )
        button.clicked.connect(
            lambda _checked=False, key=key: self._choose_color(key)
        )
        return button

    def _choose_color(self, key: tuple[str, str]) -> None:
        species_name = self._color_targets[key]
        current = QtGui.QColor(
            self._color_for_species(
                species_name,
                self._known_species,
            )
        )
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Species Color")
        if not color.isValid():
            return
        self.speciesColorRequested.emit(species_name, color)

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        _ = column
        parent = item.parent()
        self._tree.blockSignals(True)
        try:
            if parent is None:
                state = item.checkState(0)
                for index in range(item.childCount()):
                    item.child(index).setCheckState(0, state)
                parent = item
            else:
                any_checked = any(
                    parent.child(index).checkState(0) == Qt.CheckState.Checked
                    for index in range(parent.childCount())
                )
                parent.setCheckState(
                    0,
                    Qt.CheckState.Checked if any_checked else Qt.CheckState.Unchecked,
                )
        finally:
            self._tree.blockSignals(False)
        dataset_id = str(parent.data(0, Qt.UserRole))
        raw_observation_ids = tuple(
            str(parent.child(index).data(0, Qt.UserRole + 1))
            for index in range(parent.childCount())
            if parent.child(index).checkState(0) == Qt.CheckState.Checked
        )
        try:
            self.overlaySelectionRequested.emit(dataset_id, raw_observation_ids)
        finally:
            # Direct handlers update ``_links_by_dataset_id`` on acceptance.
            # Rejection/cancellation leaves the prior authoritative snapshot,
            # so either outcome is rendered without optimistic checkbox drift.
            self._render_authoritative_overlay_selection()

    def _render_authoritative_overlay_selection(self) -> None:
        self._tree.blockSignals(True)
        try:
            for parent_index in range(self._tree.topLevelItemCount()):
                parent = self._tree.topLevelItem(parent_index)
                dataset_id = parent.data(0, Qt.UserRole)
                selected = set(self._links_by_dataset_id[dataset_id].overlay_raw_observation_ids)
                for child_index in range(parent.childCount()):
                    child = parent.child(child_index)
                    raw_observation_id = str(
                        child.data(0, Qt.UserRole + 1)
                    )
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if raw_observation_id in selected
                        else Qt.CheckState.Unchecked,
                    )
                parent.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if selected
                    else Qt.CheckState.Unchecked,
                )
        finally:
            self._tree.blockSignals(False)

    def _on_size_edited(self) -> None:
        value = self._typed_marker_value(self._size_edit, "Size", 1, 20)
        if value is not None:
            self.experimentalPointSizeRequested.emit(value)

    def _on_opacity_edited(self) -> None:
        value = self._typed_marker_value(self._opacity_edit, "Opacity (%)", 0, 100)
        if value is not None:
            self.experimentalPointOpacityRequested.emit(value / 100.0)

    def _typed_marker_value(
        self, edit: QtWidgets.QLineEdit, field: str, minimum: int, maximum: int,
    ) -> int | None:
        """Admit one marker box's draft; None leaves nothing to apply."""
        value = None
        self._invalid_marker_edits.pop(edit, None)
        if edit.text() != self._shown_marker_text[edit]:
            try:
                value = parse_whole_number_text(edit.text(), field=field, minimum=minimum, maximum=maximum)
            except ValueError as exc:
                self._invalid_marker_edits[edit] = str(exc)
            else:
                # Applied text is no draft; the accepted value replaces it.
                self._shown_marker_text[edit] = edit.text()
        edit.setStyleSheet(INVALID_EDITOR_STYLE if edit in self._invalid_marker_edits else "")
        self._marker_error.setText("\n".join(self._invalid_marker_edits.values()))
        self._marker_error.setVisible(bool(self._invalid_marker_edits))
        return value

    def _render_application_style(self) -> None:
        size = int(self._experimental_point_size())
        opacity = float(self._experimental_point_opacity())
        for edit, text in ((self._size_edit, str(size)), (self._opacity_edit, str(round(opacity * 100)))):
            if edit.text() == self._shown_marker_text[edit]:
                edit.setText(text)
            self._shown_marker_text[edit] = text
