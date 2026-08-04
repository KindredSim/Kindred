"""Snapshot renderer and intent source for active-tab dataset overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.project_document import RawDatasetRecord
from kindred.gui.dataset_payload_snapshot import ActiveDatasetLinkSnapshot

__all__ = ["DatasetOverlayPanel", "DatasetStyle", "OverlaySelectionIntent"]


@dataclass(frozen=True)
class DatasetStyle:
    size: int = 6
    opacity: int = 180

    def __post_init__(self) -> None:
        if not 1 <= int(self.size) <= 20:
            raise ValueError("Experimental point size must be between 1 and 20.")
        if not 0 <= int(self.opacity) <= 255:
            raise ValueError("Experimental point opacity must be between 0 and 255.")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "opacity", int(self.opacity))


@dataclass(frozen=True)
class OverlaySelectionIntent:
    dataset_id: str
    raw_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dataset_id, str)
            or not self.dataset_id
            or self.dataset_id != self.dataset_id.strip()
        ):
            raise ValueError(
                "Overlay selection requires a nonblank edge-normalized "
                "dataset ID."
            )
        raw_observation_ids = tuple(self.raw_observation_ids)
        if (
            any(
                not isinstance(raw_observation_id, str)
                or not raw_observation_id
                or raw_observation_id != raw_observation_id.strip()
                for raw_observation_id in raw_observation_ids
            )
            or len(raw_observation_ids) != len(set(raw_observation_ids))
        ):
            raise ValueError(
                "Overlay selection raw observation IDs must be unique "
                "nonblank edge-normalized strings."
            )
        object.__setattr__(
            self,
            "raw_observation_ids",
            raw_observation_ids,
        )


class DatasetOverlayPanel(QtWidgets.QWidget):
    """Render global records with persisted active-tab overlay membership."""

    overlaySelectionRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color_for_species: Callable[[str, tuple[str, ...]], QtGui.QColor],
        set_species_color: Callable[
            [str, QtGui.QColor, tuple[str, ...]],
            None,
        ],
        experimental_style: Callable[[], DatasetStyle],
        set_experimental_style: Callable[[DatasetStyle], None],
    ) -> None:
        super().__init__(parent)
        if not callable(color_for_species) or not callable(set_species_color):
            raise TypeError("DatasetOverlayPanel requires injected species-color callbacks.")
        if not callable(experimental_style) or not callable(set_experimental_style):
            raise TypeError("DatasetOverlayPanel requires injected experimental-style callbacks.")
        self._color_for_species = color_for_species
        self._set_species_color = set_species_color
        self._experimental_style = experimental_style
        self._set_experimental_style = set_experimental_style
        self._links_by_dataset_id: dict[str, ActiveDatasetLinkSnapshot] = {}
        self._known_species: tuple[str, ...] = ()
        self._style = DatasetStyle()
        self._preferences_loaded = False
        self._color_buttons: dict[tuple[str, str], QtWidgets.QPushButton] = {}
        self._raw_columns_by_key: dict[tuple[str, str], str] = {}
        self._has_rendered_snapshot = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        header = QtWidgets.QLabel("Dataset Overlays")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        hint = QtWidgets.QLabel(
            "Experimental overlay membership is independent of row selection, "
            "simulated Plot membership, and Slider pins."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: palette(mid);")
        layout.addWidget(hint)

        self._empty_label = QtWidgets.QLabel("No datasets loaded")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("font-style: italic; padding: 12px;")
        layout.addWidget(self._empty_label)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Dataset / Observation", "Color"])
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.hide()
        layout.addWidget(self._tree)

        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel("Point style"))
        self._size_spin = QtWidgets.QSpinBox()
        self._size_spin.setRange(1, 20)
        self._size_spin.setValue(self._style.size)
        self._size_spin.valueChanged.connect(self._on_style_changed)
        style_row.addWidget(QtWidgets.QLabel("Size"))
        style_row.addWidget(self._size_spin)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setSuffix("%")
        self._opacity_spin.setValue(round(self._style.opacity * 100 / 255))
        self._opacity_spin.valueChanged.connect(self._on_style_changed)
        style_row.addWidget(QtWidgets.QLabel("Opacity"))
        style_row.addWidget(self._opacity_spin)
        style_row.addStretch(1)
        layout.addLayout(style_row)

    def render_snapshot(
        self,
        *,
        records: Iterable[RawDatasetRecord],
        active_links: Iterable[ActiveDatasetLinkSnapshot],
        known_species: Iterable[str] = (),
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
        self._known_species = tuple(str(value) for value in known_species)
        self._color_buttons.clear()
        self._raw_columns_by_key.clear()
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for record in rendered_records:
                dataset_id = str(record.dataset_id)
                link = self._links_by_dataset_id.get(dataset_id)
                enabled_raw_ids = set(
                    link.overlay_raw_observation_ids
                    if link is not None
                    else ()
                )
                parent = QtWidgets.QTreeWidgetItem([record.display_name, ""])
                parent.setData(0, Qt.UserRole, dataset_id)
                parent.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                self._tree.addTopLevelItem(parent)
                for series in record.observations:
                    column = str(series.column_name)
                    raw_observation_id = series.raw_observation_id
                    mapped_species = (
                        dict(link.observation_mapping).get(
                            raw_observation_id
                        )
                        if link is not None
                        else None
                    )
                    display_name = (
                        f"{column} → {mapped_species}"
                        if mapped_species
                        else column
                    )
                    child = QtWidgets.QTreeWidgetItem([display_name, ""])
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
                    button = self._create_color_button(
                        dataset_id,
                        raw_observation_id,
                        column,
                    )
                    self._tree.setItemWidget(child, 1, button)
                    key = (dataset_id, raw_observation_id)
                    self._color_buttons[key] = button
                    self._raw_columns_by_key[key] = column
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
        self._empty_label.setVisible(not has_records)
        self._render_application_style()
        self._refresh_color_swatches()
        self._tree.verticalScrollBar().setValue(vertical_scroll)
        self._tree.horizontalScrollBar().setValue(horizontal_scroll)

    def clear_projection(self) -> None:
        """Clear only rendered Overlay state and revoke Overlay actions."""
        self._links_by_dataset_id.clear()
        self._known_species = ()
        self._color_buttons.clear()
        self._raw_columns_by_key.clear()
        blocker = QtCore.QSignalBlocker(self._tree)
        try:
            self._tree.clear()
        finally:
            del blocker
        self._has_rendered_snapshot = False
        self._tree.hide()
        self._empty_label.show()

    def render_overlay_selection(
        self,
        *,
        active_links: Iterable[ActiveDatasetLinkSnapshot],
    ) -> None:
        """Project authoritative membership without rebuilding the dataset tree."""

        links = tuple(active_links)
        if any(
            not isinstance(link, ActiveDatasetLinkSnapshot)
            for link in links
        ):
            raise TypeError(
                "active_links must contain ActiveDatasetLinkSnapshot values."
            )
        dataset_ids = tuple(link.dataset_id for link in links)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Active overlay dataset IDs must be unique.")
        self._links_by_dataset_id = {
            link.dataset_id: link for link in links
        }
        self._render_authoritative_overlay_selection()

    def refresh_application_style_controls(self) -> None:
        """Rerender only application-wide point-size and opacity controls."""

        self._render_application_style()

    def refresh_color_swatches(
        self,
        *,
        known_species: Iterable[str] | None = None,
    ) -> None:
        """Rerender only application-wide color swatches."""

        self._refresh_color_swatches(known_species=known_species)

    def _refresh_color_swatches(
        self,
        *,
        known_species: Iterable[str] | None = None,
    ) -> None:
        if known_species is not None:
            self._known_species = tuple(str(value) for value in known_species)
        for key, button in self._color_buttons.items():
            color = QtGui.QColor(
                self._color_for_species(
                    self._color_species_name(*key),
                    self._known_species,
                )
            )
            button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                "border: 1px solid; border-radius: 3px;"
            )

    def _create_color_button(
        self,
        dataset_id: str,
        raw_observation_id: str,
        raw_column: str,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setFixedSize(30, 20)
        button.setFlat(True)
        button.setToolTip(
            f"Set the global display color for {raw_column}."
        )
        button.clicked.connect(
            lambda _checked=False, key=(
                dataset_id,
                raw_observation_id,
            ): self._choose_color(key)
        )
        return button

    def _choose_color(self, key: tuple[str, str]) -> None:
        species_name = self._color_species_name(*key)
        current = QtGui.QColor(
            self._color_for_species(
                species_name,
                self._known_species,
            )
        )
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Species Color")
        if not color.isValid():
            return
        self._set_species_color(
            species_name,
            color,
            self._known_species,
        )
        self.refresh_color_swatches()

    def _color_species_name(
        self,
        dataset_id: str,
        raw_observation_id: str,
    ) -> str:
        link = self._links_by_dataset_id.get(str(dataset_id))
        if link is not None:
            mapped = dict(link.observation_mapping).get(
                str(raw_observation_id)
            )
            if mapped:
                return str(mapped)
        return self._raw_columns_by_key[(
            str(dataset_id),
            str(raw_observation_id),
        )]

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
            self.overlaySelectionRequested.emit(
                OverlaySelectionIntent(
                    dataset_id=dataset_id,
                    raw_observation_ids=raw_observation_ids,
                )
            )
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
                dataset_id = str(parent.data(0, Qt.UserRole))
                link = self._links_by_dataset_id.get(dataset_id)
                selected = set(
                    ()
                    if link is None
                    else link.overlay_raw_observation_ids
                )
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

    def _on_style_changed(self) -> None:
        if not self._preferences_loaded:
            return
        requested = DatasetStyle(
            size=self._size_spin.value(),
            opacity=round(self._opacity_spin.value() * 255 / 100),
        )
        self._set_experimental_style(requested)
        self._style = self._read_application_style()
        self._render_application_style()

    def _read_application_style(self) -> DatasetStyle:
        style = self._experimental_style()
        if not isinstance(style, DatasetStyle):
            raise TypeError("experimental_style() must return DatasetStyle.")
        return style

    def _render_application_style(self) -> None:
        self._style = self._read_application_style()
        self._preferences_loaded = True
        self._size_spin.blockSignals(True)
        self._opacity_spin.blockSignals(True)
        try:
            self._size_spin.setValue(self._style.size)
            self._opacity_spin.setValue(round(self._style.opacity * 100 / 255))
        finally:
            self._opacity_spin.blockSignals(False)
            self._size_spin.blockSignals(False)
