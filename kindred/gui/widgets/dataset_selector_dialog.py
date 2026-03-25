# kindred/gui/widgets/dataset_selector_dialog.py
"""Dialog for selecting datasets for fitting operations."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6 import QtWidgets


class DatasetSelectorDialog(QtWidgets.QDialog):
    """
    Dialog for selecting one or more datasets for fitting operations.

    Supports both single-selection and multi-selection (for global fit) modes.

    Parameters
    ----------
    datasets : Dict[str, Dict]
        Dictionary mapping dataset names to dataset payloads.
        Each payload should have 't' (time array) and 'species' (dict of arrays).
    multi_select : bool, default=False
        If True, allow multiple dataset selection (for global fit).
        If False, allow only single selection.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        datasets: Dict[str, Dict],
        multi_select: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.datasets = datasets
        self.multi_select = multi_select

        # Set window title based on mode
        if multi_select:
            self.setWindowTitle("Select Datasets for Global Fit")
        else:
            self.setWindowTitle("Select Dataset")
	
        self.resize(500, 400)
	
        self._setup_ui()
        self._populate_datasets()
	
    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Instructions
        if self.multi_select:
            instructions = (
                "Select one or more datasets to include in the global fit.\n"
                "Global fitting will estimate shared parameters across all selected datasets."
            )
        else:
            instructions = (
                "Select a dataset to use for single-dataset operations."
            )

        info_label = QtWidgets.QLabel(instructions)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dataset list
        self.dataset_list = QtWidgets.QListWidget()
        self.dataset_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        if self.multi_select:
            self.dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        else:
            self.dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        # Connect double-click for single selection mode
        if not self.multi_select:
            self.dataset_list.itemDoubleClicked.connect(self.accept)

        layout.addWidget(self.dataset_list)
	
        # Dataset info panel
        info_group = QtWidgets.QGroupBox("Dataset Information")
        info_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        info_layout = QtWidgets.QFormLayout()

        self.info_name = QtWidgets.QLabel("-")
        self.info_points = QtWidgets.QLabel("-")
        self.info_species = QtWidgets.QLabel("-")
        self.info_time_range = QtWidgets.QLabel("-")

        info_layout.addRow("Name:", self.info_name)
        info_layout.addRow("Data Points:", self.info_points)
        info_layout.addRow("Species:", self.info_species)
        info_layout.addRow("Time Range:", self.info_time_range)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Connect selection changed
        self.dataset_list.itemSelectionChanged.connect(self._on_selection_changed)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

    def _populate_datasets(self):
        """Populate the dataset list."""
        for name in sorted(self.datasets.keys()):
            item = QtWidgets.QListWidgetItem(name)
            self.dataset_list.addItem(item)

        # Select first item by default
        if self.dataset_list.count() > 0:
            self.dataset_list.setCurrentRow(0)

    def _on_selection_changed(self):
        """Update the info panel when selection changes."""
        selected_items = self.dataset_list.selectedItems()

        if not selected_items:
            self.info_name.setText("-")
            self.info_points.setText("-")
            self.info_species.setText("-")
            self.info_time_range.setText("-")
            return

        # Show info for the first selected dataset
        name = selected_items[0].text()
        dataset = self.datasets.get(name)

        if not dataset:
            return

        # Extract dataset information
        t = dataset.get("t", [])
        species = dataset.get("species", {})

        num_points = len(t) if t is not None else 0
        species_names = ", ".join(sorted(species.keys())) if species else "None"

        if t is not None and len(t) > 0:
            time_range = f"{min(t):.3g} - {max(t):.3g}"
        else:
            time_range = "N/A"

        # Update info labels
        self.info_name.setText(name)
        self.info_points.setText(str(num_points))
        self.info_species.setText(species_names)
        self.info_time_range.setText(time_range)

        # If multi-select, show count of selected datasets
        if self.multi_select and len(selected_items) > 1:
            self.info_name.setText(f"{len(selected_items)} datasets selected")

    def get_selected_datasets(self) -> List[str]:
        """
        Get the list of selected dataset names.

        Returns
        -------
        List[str]
            List of selected dataset names. Empty if none selected.
        """
        selected_items = self.dataset_list.selectedItems()
        return [item.text() for item in selected_items]

    def get_selected_dataset(self) -> Optional[str]:
        """
        Get the selected dataset name (for single-selection mode).

        Returns
        -------
        Optional[str]
            Selected dataset name, or None if none selected.
        """
        selected = self.get_selected_datasets()
        return selected[0] if selected else None
