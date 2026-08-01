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
    columns: tuple[str, ...]

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
        columns = tuple(self.columns)
        if (
            any(
                not isinstance(column, str)
                or not column
                or column != column.strip()
                for column in columns
            )
            or len(columns) != len(set(columns))
        ):
            raise ValueError(
                "Overlay selection columns must be unique nonblank "
                "edge-normalized names."
            )
        object.__setattr__(self, "columns", columns)


class DatasetOverlayPanel(QtWidgets.QWidget):
    """Render global records with persisted active-tab overlay membership."""

    overlaySelectionRequested = QtCore.Signal(object)
    styleChanged = QtCore.Signal()

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
        """Rebuild physical state directly from one immutable snapshot."""
        rendered_records = tuple(records)
        self._links_by_dataset_id = {
            link.dataset_id: link for link in active_links
        }
        self._known_species = tuple(str(value) for value in known_species)
        self._color_buttons.clear()
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for record in rendered_records:
                dataset_id = str(record.dataset_id)
                link = self._links_by_dataset_id.get(dataset_id)
                enabled_columns = set(link.overlay_columns if link is not None else ())
                parent = QtWidgets.QTreeWidgetItem([record.display_name, ""])
                parent.setData(0, Qt.UserRole, dataset_id)
                parent.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                self._tree.addTopLevelItem(parent)
                for series in record.observations:
                    column = str(series.column_name)
                    mapped_species = (
                        dict(link.observation_mapping).get(column)
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
                    child.setData(0, Qt.UserRole + 1, column)
                    child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked if column in enabled_columns else Qt.CheckState.Unchecked,
                    )
                    parent.addChild(child)
                    button = self._create_color_button(dataset_id, column)
                    self._tree.setItemWidget(child, 1, button)
                    self._color_buttons[(dataset_id, column)] = button
                parent.setCheckState(
                    0,
                    Qt.CheckState.Checked if enabled_columns else Qt.CheckState.Unchecked,
                )
                parent.setExpanded(bool(enabled_columns))
        finally:
            self._tree.blockSignals(False)
        has_records = bool(rendered_records)
        self._tree.setVisible(has_records)
        self._empty_label.setVisible(not has_records)
        self._render_application_style()
        self._refresh_color_swatches()

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

    def _create_color_button(self, dataset_id: str, column: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton()
        button.setFixedSize(30, 20)
        button.setFlat(True)
        button.setToolTip(f"Set the global display color for {column}.")
        button.clicked.connect(
            lambda _checked=False, key=(dataset_id, column): self._choose_color(key)
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
        self._refresh_color_swatches()
        self.styleChanged.emit()

    def _color_species_name(self, dataset_id: str, raw_column: str) -> str:
        link = self._links_by_dataset_id.get(str(dataset_id))
        if link is not None:
            mapped = dict(link.observation_mapping).get(str(raw_column))
            if mapped:
                return str(mapped)
        return str(raw_column)

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
        columns = tuple(
            str(parent.child(index).data(0, Qt.UserRole + 1))
            for index in range(parent.childCount())
            if parent.child(index).checkState(0) == Qt.CheckState.Checked
        )
        self.overlaySelectionRequested.emit(
            OverlaySelectionIntent(dataset_id=dataset_id, columns=columns)
        )

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
        self.styleChanged.emit()

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
