# kindred/gui/widgets/dataset_overlay_panel.py
"""Dataset overlay selector with species-owned color swatches and dataset filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred.gui.color_manager import ColorManager

__all__ = ["DatasetOverlayPanel", "DatasetStyle"]


@dataclass
class DatasetStyle:
    """Configuration for dataset scatter point appearance."""
    size: int = 6
    opacity: int = 180  # 0-255
    color: Optional[QtGui.QColor] = None


class DatasetOverlayPanel(QtWidgets.QWidget):
    """
    Overlay panel with per-column selection and global species-color swatches.

    Features:
    - Dataset-level checkboxes (master enable/disable)
    - Per-species/column checkboxes within each dataset (fine-grained control)
    - Display-only swatches showing globally owned species colors
    - Adjustable point size and opacity for dataset scatter points
    """

    selectionChanged = QtCore.Signal(list)
    styleChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Track dataset selection state
        self._selected: Dict[str, bool] = {}

        # Track per-dataset enabled species/columns
        # Maps: dataset_name -> Set[column_key]
        self._enabled_species: Dict[str, Set[str]] = {}

        # Display-only species swatches.
        # Maps: (dataset_name, column_key) -> QColor
        self._species_colors: Dict[Tuple[str, str], QtGui.QColor] = {}

        # Dataset metadata (for building species lists)
        # Maps: dataset_name -> dataset_payload (with 'species' dict)
        self._datasets: Dict[str, dict] = {}

        # Tree items for UI
        # Maps: dataset_name -> QTreeWidgetItem
        self._tree_items: Dict[str, QtWidgets.QTreeWidgetItem] = {}

        # Maps: (dataset_name, column_key) -> color button widget
        self._color_buttons: Dict[Tuple[str, str], QtWidgets.QPushButton] = {}

        # Styling configuration
        self._dataset_style = DatasetStyle()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        header = QtWidgets.QLabel("Dataset Overlays")
        header.setStyleSheet("font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        hint = QtWidgets.QLabel("Select datasets and species to overlay experimental points.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: palette(mid);")
        hint.setToolTip(
            "Overlay datasets show experimental points on simulation curves.\n"
            "Species colors are owned globally. Dataset identity is shown with markers."
        )
        layout.addWidget(hint)

        self._empty_label = QtWidgets.QLabel("No datasets loaded")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("font-style: italic; padding: 12px;")
        layout.addWidget(self._empty_label)

        # QTreeWidget with 2 columns: [Checkbox+Name, Color Button]
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Dataset / Species", "Color"])
        self._tree.setColumnWidth(0, 200)
        self._tree.setColumnWidth(1, 50)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._tree.setVisible(False)
        layout.addWidget(self._tree)

        self._status_label = QtWidgets.QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        style_row = QtWidgets.QHBoxLayout()
        style_row.setContentsMargins(0, 0, 0, 0)
        style_row.setSpacing(6)
        style_label = QtWidgets.QLabel("Point Style")
        style_label.setStyleSheet("font-weight: bold;")
        style_row.addWidget(style_label)

        # Point size control
        self._size_spin = QtWidgets.QSpinBox()
        self._size_spin.setRange(1, 20)
        self._size_spin.setValue(self._dataset_style.size)
        self._size_spin.setToolTip("Size of dataset scatter points")
        self._size_spin.setMinimumWidth(60)
        self._size_spin.valueChanged.connect(self._on_style_changed)
        size_label = QtWidgets.QLabel("Size")
        size_label.setBuddy(self._size_spin)
        style_row.addWidget(size_label)
        style_row.addWidget(self._size_spin)

        # Opacity control (0-100 for user-friendliness, converted to 0-255 internally)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setValue(int(self._dataset_style.opacity * 100 / 255))
        self._opacity_spin.setSuffix("%")
        self._opacity_spin.setToolTip("Opacity/transparency of dataset points")
        self._opacity_spin.setMinimumWidth(60)
        self._opacity_spin.valueChanged.connect(self._on_style_changed)
        opacity_label = QtWidgets.QLabel("Opacity")
        opacity_label.setBuddy(self._opacity_spin)
        style_row.addWidget(opacity_label)
        style_row.addWidget(self._opacity_spin)
        style_row.addStretch(1)
        layout.addLayout(style_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_datasets(self, datasets: Dict[str, dict]) -> None:
        """
        Refresh dataset tree with per-species checkboxes and color controls.

        Parameters
        ----------
        datasets : dict
            Mapping of dataset_name -> payload with 'species' dict
        """
        incoming = set(datasets.keys())

        # Remove stale datasets
        for name in list(self._tree_items.keys()):
            if name not in incoming:
                item = self._tree_items.pop(name)
                index = self._tree.indexOfTopLevelItem(item)
                self._tree.takeTopLevelItem(index)
                self._selected.pop(name, None)
                self._enabled_species.pop(name, None)
                self._datasets.pop(name, None)
                # Clean up color buttons and colors for this dataset
                keys_to_remove = [k for k in self._color_buttons.keys() if k[0] == name]
                for k in keys_to_remove:
                    self._color_buttons.pop(k, None)
                    self._species_colors.pop(k, None)

        # Add/update datasets
        for name in sorted(incoming):
            payload = datasets[name]
            self._datasets[name] = payload

            if name not in self._tree_items:
                # Create new dataset tree item
                dataset_item = QtWidgets.QTreeWidgetItem([name, ""])
                dataset_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                dataset_item.setCheckState(0, Qt.CheckState.Unchecked)
                self._tree.addTopLevelItem(dataset_item)
                self._tree_items[name] = dataset_item

                # Initialize enabled species (all enabled by default)
                species_dict = payload.get("species", {})
                self._enabled_species[name] = set(species_dict.keys())

                # Add child items for each species/column
                for species_key in sorted(species_dict.keys()):
                    species_item = QtWidgets.QTreeWidgetItem([species_key, ""])
                    species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    species_item.setCheckState(0, Qt.CheckState.Checked)
                    dataset_item.addChild(species_item)

                    # Add color button widget in column 1
                    color_btn = self._create_color_button(name, species_key)
                    self._tree.setItemWidget(species_item, 1, color_btn)
                    self._color_buttons[(name, species_key)] = color_btn
            else:
                # Update existing dataset
                dataset_item = self._tree_items[name]

                # Sync dataset checkbox with internal state
                checked = Qt.CheckState.Checked if self._selected.get(name, False) else Qt.CheckState.Unchecked
                if dataset_item.checkState(0) != checked:
                    dataset_item.setCheckState(0, checked)

                # Update species children if payload changed
                species_dict = payload.get("species", {})
                current_species = set(species_dict.keys())

                # Remove species no longer in payload
                for i in range(dataset_item.childCount() - 1, -1, -1):
                    child = dataset_item.child(i)
                    child_key = child.text(0)
                    if child_key not in current_species:
                        dataset_item.removeChild(child)
                        key = (name, child_key)
                        self._color_buttons.pop(key, None)
                        self._species_colors.pop(key, None)

                # Add new species
                existing_children = {dataset_item.child(i).text(0) for i in range(dataset_item.childCount())}
                for species_key in sorted(current_species - existing_children):
                    species_item = QtWidgets.QTreeWidgetItem([species_key, ""])
                    species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    # Enable if in enabled set, otherwise disable
                    enabled = species_key in self._enabled_species.get(name, set())
                    species_item.setCheckState(0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                    dataset_item.addChild(species_item)

                    # Add color button
                    color_btn = self._create_color_button(name, species_key)
                    self._tree.setItemWidget(species_item, 1, color_btn)
                    self._color_buttons[(name, species_key)] = color_btn

        has_data = bool(incoming)
        self._tree.setVisible(has_data)
        self._empty_label.setVisible(not has_data)
        self.refresh_color_swatches()

    def reconcile_selection(
        self,
        *,
        previous_selected_datasets: Iterable[str],
        previous_enabled_species: Dict[str, Set[str]],
        include_dataset_ids: Iterable[str],
        ordered_dataset_ids: Iterable[str],
        allow_default_include: bool,
        emit: bool = False,
    ) -> Dict[str, Set[str]]:
        """
        Reconcile overlay selection after the dataset catalog/species set changes.

        This is the owner-centered write boundary for programmatic selection updates.
        It preserves still-valid selections, expands to all available species when a
        still-valid dataset loses overlap, and falls back to the first valid included
        dataset when previous selections become invalid.
        """
        available_by_dataset = {
            str(dataset_name): set(((payload or {}).get("species") or {}).keys())
            for dataset_name, payload in self._datasets.items()
        }
        desired: Dict[str, Set[str]] = {}
        prior_ids = [str(x) for x in (previous_selected_datasets or []) if str(x).strip()]
        had_any_selected_before = bool(prior_ids)

        for ds_id in prior_ids:
            available = set(available_by_dataset.get(ds_id, set()) or set())
            if not available:
                continue
            previous_enabled = set((previous_enabled_species or {}).get(ds_id, set()) or set())
            keep = previous_enabled & available
            desired[ds_id] = keep if keep else set(available)

        include_order = [str(x) for x in (include_dataset_ids or []) if str(x).strip()]
        ordered_ids = [str(x) for x in (ordered_dataset_ids or []) if str(x).strip()]

        if not desired and allow_default_include:
            for ds_id in include_order:
                available = set(available_by_dataset.get(ds_id, set()) or set())
                if available:
                    desired[ds_id] = set(available)

        if not desired and had_any_selected_before:
            for ds_id in include_order + [x for x in ordered_ids if x not in include_order]:
                available = set(available_by_dataset.get(ds_id, set()) or set())
                if available:
                    desired[ds_id] = set(available)
                    break

        self._apply_selection_state(desired, emit=emit)
        return {name: set(values) for name, values in self.selected_dataset_species().items()}

    def selected_datasets(self) -> List[str]:
        """Return dataset names currently marked as active overlays."""
        return [name for name, enabled in self._selected.items() if enabled]

    def selected_dataset_species(self) -> Dict[str, Set[str]]:
        """
        Return per-dataset enabled species/columns.

        Returns
        -------
        dict
            Mapping of dataset_name -> set of enabled dataset column keys
        """
        result = {}
        for dataset_name in self.selected_datasets():
            enabled = self._enabled_species.get(dataset_name, set())
            if enabled:
                result[dataset_name] = enabled
        return result

    def species_colors(self) -> Dict[Tuple[str, str], QtGui.QColor]:
        """
        Return display colors resolved from the global species owner.

        Returns
        -------
        dict
            Mapping of (dataset_name, column_key) -> QColor
        """
        self.refresh_color_swatches()
        return {key: QtGui.QColor(color) for key, color in self._species_colors.items()}

    def refresh_color_swatches(self) -> None:
        """Refresh per-dataset swatches from the global species color manager."""
        color_manager = ColorManager.instance()

        refreshed: Dict[Tuple[str, str], QtGui.QColor] = {}
        for dataset_name, payload in self._datasets.items():
            species_dict = (payload or {}).get("species") or {}
            for species_key in species_dict.keys():
                key = (str(dataset_name), str(species_key))
                current_species_color = color_manager.get_current_species_color(str(species_key))
                refreshed[key] = (
                    current_species_color
                    if current_species_color is not None
                    else color_manager.get_non_species_color(str(species_key))
                )
        self._species_colors = refreshed

        for key, button in list(self._color_buttons.items()):
            if button is not None:
                self._update_color_button_appearance(button, key)

    def dataset_style(self) -> DatasetStyle:
        """Return current dataset point styling configuration (size and opacity)."""
        return self._dataset_style

    def set_status_messages(self, messages: Iterable[str]) -> None:
        """Display friendly warnings about overlay availability."""
        filtered = [str(msg) for msg in messages if msg]
        if filtered:
            self._status_label.setText("• " + "\n• ".join(filtered))
            self._status_label.setVisible(True)
        else:
            self._status_label.clear()
            self._status_label.setVisible(False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_selection_state(self, desired_by_dataset: Dict[str, Set[str]], *, emit: bool) -> None:
        previous_state = {name: set(values) for name, values in self.selected_dataset_species().items()}
        normalized: Dict[str, Set[str]] = {}
        for dataset_name in list(self._tree_items.keys()):
            ds_id = str(dataset_name)
            available = set((((self._datasets.get(ds_id) or {}).get("species")) or {}).keys())
            enabled = set((desired_by_dataset or {}).get(ds_id, set()) or set()) & available
            normalized[ds_id] = enabled

        self._tree.blockSignals(True)
        self.blockSignals(True)
        try:
            for dataset_name in list(self._selected.keys()):
                if str(dataset_name) not in normalized:
                    self._selected.pop(dataset_name, None)
            for dataset_name in list(self._enabled_species.keys()):
                if str(dataset_name) not in normalized:
                    self._enabled_species.pop(dataset_name, None)

            for dataset_name, enabled in normalized.items():
                self._selected[dataset_name] = bool(enabled)
                self._enabled_species[dataset_name] = set(enabled)

            for dataset_name, dataset_item in self._tree_items.items():
                ds_id = str(dataset_name)
                enabled = normalized.get(ds_id, set()) or set()
                dataset_item.setCheckState(
                    0,
                    Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked,
                )
                for i in range(dataset_item.childCount()):
                    child = dataset_item.child(i)
                    species_key = child.text(0)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked if species_key in enabled else Qt.CheckState.Unchecked,
                    )
        finally:
            self.blockSignals(False)
            self._tree.blockSignals(False)

        if emit:
            current_state = {name: set(values) for name, values in self.selected_dataset_species().items()}
            if current_state != previous_state:
                self.selectionChanged.emit(self.selected_datasets())

    def _create_color_button(self, dataset_name: str, species_key: str) -> QtWidgets.QPushButton:
        """Create a read-only swatch button for a dataset species."""
        btn = QtWidgets.QPushButton()
        btn.setFixedSize(30, 20)
        btn.setToolTip(f"{species_key} uses a globally owned species color.")
        btn.setFlat(True)

        key = (dataset_name, species_key)
        self._update_color_button_appearance(btn, key)

        return btn

    def _update_color_button_appearance(self, btn: QtWidgets.QPushButton, key: Tuple[str, str]) -> None:
        """Update button to show current color as a swatch."""
        color = self._species_colors.get(key)
        if color:
            btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                f"border: 1px solid; border-radius: 3px;"
            )
        else:
            btn.setStyleSheet("border: 1px solid; border-radius: 3px;")

    def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        """Handle checkbox state changes in the tree."""
        parent = item.parent()

        if parent is None:
            # Top-level dataset item clicked
            dataset_name = item.text(0)
            enabled = item.checkState(0) == Qt.CheckState.Checked

            # Block signals while updating children to prevent recursion
            self._tree.blockSignals(True)
            try:
                # Update all children to match parent state
                for i in range(item.childCount()):
                    child = item.child(i)
                    child.setCheckState(0, item.checkState(0))
                    species_key = child.text(0)
                    if enabled:
                        self._enabled_species.setdefault(dataset_name, set()).add(species_key)
                    else:
                        self._enabled_species.setdefault(dataset_name, set()).discard(species_key)
            finally:
                self._tree.blockSignals(False)

            # Update dataset selection state
            self._selected[dataset_name] = enabled
            self.selectionChanged.emit(self.selected_datasets())
        else:
            # Child species item clicked
            dataset_name = parent.text(0)
            species_key = item.text(0)
            enabled = item.checkState(0) == Qt.CheckState.Checked

            # Update enabled species set
            if enabled:
                self._enabled_species.setdefault(dataset_name, set()).add(species_key)
            else:
                self._enabled_species.setdefault(dataset_name, set()).discard(species_key)

            # Update parent checkbox state based on children
            self._tree.blockSignals(True)
            try:
                # Check if any children are checked
                any_checked = False
                all_checked = True
                for i in range(parent.childCount()):
                    child_state = parent.child(i).checkState(0)
                    if child_state == Qt.CheckState.Checked:
                        any_checked = True
                    else:
                        all_checked = False

                # Set parent state: checked if any child is checked
                if all_checked and parent.childCount() > 0:
                    parent.setCheckState(0, Qt.CheckState.Checked)
                elif any_checked:
                    parent.setCheckState(0, Qt.CheckState.Checked)
                else:
                    parent.setCheckState(0, Qt.CheckState.Unchecked)

                # Update dataset selection
                new_dataset_state = any_checked
                if self._selected.get(dataset_name, False) != new_dataset_state:
                    self._selected[dataset_name] = new_dataset_state
            finally:
                self._tree.blockSignals(False)

            # ALWAYS emit signal when child species toggled (even if parent state unchanged)
            # This ensures plot updates when individual species are checked/unchecked
            self.selectionChanged.emit(self.selected_datasets())

    def _on_style_changed(self) -> None:
        """Handle changes to global styling controls (size and opacity)."""
        self._dataset_style.size = self._size_spin.value()
        # Convert percentage (0-100) to alpha (0-255)
        self._dataset_style.opacity = int(self._opacity_spin.value() * 255 / 100)
        self.styleChanged.emit()
