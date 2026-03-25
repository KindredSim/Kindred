"""
Test dataset selector dialog for fitting operations.

This module tests the DatasetSelectorDialog widget for:
- Single and multi-selection modes
- Dataset information display
- Selection validation
"""

from __future__ import annotations

import pytest
import numpy as np
from PySide6 import QtWidgets

from kindred.gui.widgets.dataset_selector_dialog import DatasetSelectorDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    return {
        'dataset1.csv': {
            't': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'species': {
                'A': np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
                'B': np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
            }
        },
        'dataset2.csv': {
            't': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            'species': {
                'A': np.array([1.5, 1.2, 0.9, 0.6, 0.3]),
            }
        },
        'dataset3.csv': {
            't': np.array([0.0, 2.0, 4.0]),
            'species': {
                'A': np.array([2.0, 1.0, 0.0]),
                'B': np.array([0.0, 1.0, 2.0]),
                'C': np.array([0.0, 0.5, 1.0]),
            }
        }
    }


class TestDatasetSelectorDialog:
    """Test dataset selector dialog functionality."""

    def test_layout_policies_and_layout_margins(self, qapp, sample_datasets):
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False,
        )

        dataset_policy = dialog.dataset_list.sizePolicy()
        assert (
            dataset_policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
        )
        assert dataset_policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding

        info_groups = [
            g
            for g in dialog.findChildren(QtWidgets.QGroupBox)
            if g.title() == "Dataset Information"
        ]
        assert len(info_groups) == 1
        info_policy = info_groups[0].sizePolicy()
        assert info_policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
        assert info_policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Minimum

        layout = dialog.layout()
        margins = layout.contentsMargins()
        assert (
            margins.left(),
            margins.top(),
            margins.right(),
            margins.bottom(),
        ) == (10, 10, 10, 10)
        assert layout.spacing() == 10

        # Verify we did not hard-lock minimum dimensions (layout still imposes a minimumSizeHint).
        assert dialog.minimumWidth() == 0
        assert dialog.minimumHeight() == 0

    def test_single_selection_mode(self, qapp, sample_datasets):
        """Test single selection mode."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # Verify title
        assert "Select Dataset" == dialog.windowTitle()

        # Verify selection mode
        assert dialog.dataset_list.selectionMode() == QtWidgets.QAbstractItemView.SingleSelection

        # Verify datasets are populated
        assert dialog.dataset_list.count() == 3

    def test_multi_selection_mode(self, qapp, sample_datasets):
        """Test multi-selection mode for global fit."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=True
        )

        # Verify title
        assert "Global Fit" in dialog.windowTitle()

        # Verify selection mode
        assert dialog.dataset_list.selectionMode() == QtWidgets.QAbstractItemView.MultiSelection

    def test_dataset_list_populated_sorted(self, qapp, sample_datasets):
        """Test dataset list is populated and sorted."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # Get all items
        items = [dialog.dataset_list.item(i).text() for i in range(dialog.dataset_list.count())]

        # Should be sorted alphabetically
        assert items == sorted(sample_datasets.keys())

    def test_first_dataset_selected_by_default(self, qapp, sample_datasets):
        """Test first dataset is selected by default."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # First item should be selected
        assert dialog.dataset_list.currentRow() == 0

    def test_dataset_info_display(self, qapp, sample_datasets):
        """Test dataset information is displayed correctly."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # Select first dataset (dataset1.csv)
        dialog.dataset_list.setCurrentRow(0)

        # Verify info is displayed
        assert dialog.info_name.text() == 'dataset1.csv'
        assert dialog.info_points.text() == '5'
        assert 'A, B' in dialog.info_species.text()
        assert '0' in dialog.info_time_range.text() and '4' in dialog.info_time_range.text()

    def test_get_selected_dataset_single_mode(self, qapp, sample_datasets):
        """Test getting selected dataset in single selection mode."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # Select second dataset
        dialog.dataset_list.setCurrentRow(1)

        selected = dialog.get_selected_dataset()
        assert selected == 'dataset2.csv'

    def test_get_selected_datasets_multi_mode(self, qapp, sample_datasets):
        """Test getting selected datasets in multi-selection mode."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=True
        )

        # Select first and third datasets
        dialog.dataset_list.item(0).setSelected(True)
        dialog.dataset_list.item(2).setSelected(True)

        selected = dialog.get_selected_datasets()
        assert len(selected) == 2
        assert 'dataset1.csv' in selected
        assert 'dataset3.csv' in selected

    def test_no_selection_returns_empty(self, qapp, sample_datasets):
        """Test no selection returns empty list."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=True
        )

        # Clear selection
        dialog.dataset_list.clearSelection()

        selected = dialog.get_selected_datasets()
        assert selected == []

        selected_single = dialog.get_selected_dataset()
        assert selected_single is None

    def test_multi_selection_info_shows_count(self, qapp, sample_datasets):
        """Test multi-selection shows count in info panel."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=True
        )

        # Select multiple datasets
        dialog.dataset_list.item(0).setSelected(True)
        dialog.dataset_list.item(1).setSelected(True)

        # Trigger selection changed
        dialog._on_selection_changed()

        # Should show count
        assert '2 datasets' in dialog.info_name.text()

    def test_empty_datasets_dict(self, qapp):
        """Test dialog handles empty datasets dictionary."""
        dialog = DatasetSelectorDialog(
            datasets={},
            multi_select=False
        )

        # Should have no items
        assert dialog.dataset_list.count() == 0

        # Should return None for selection
        assert dialog.get_selected_dataset() is None
        assert dialog.get_selected_datasets() == []

    def test_dataset_with_no_species(self, qapp):
        """Test dataset with missing species data."""
        datasets = {
            'empty.csv': {
                't': np.array([0, 1, 2]),
                'species': {}
            }
        }

        dialog = DatasetSelectorDialog(
            datasets=datasets,
            multi_select=False
        )

        # Select the dataset
        dialog.dataset_list.setCurrentRow(0)

        # Info should handle missing species gracefully
        assert dialog.info_species.text() == 'None'

    def test_double_click_accepts_in_single_mode(self, qapp, sample_datasets):
        """Test double-clicking a dataset accepts the dialog in single mode."""
        dialog = DatasetSelectorDialog(
            datasets=sample_datasets,
            multi_select=False
        )

        # Mock accept method to track if it's called
        accept_called = []

        def mock_accept():
            accept_called.append(True)
            QtWidgets.QDialog.accept(dialog)

        dialog.accept = mock_accept

        # Simulate double-click on first item
        item = dialog.dataset_list.item(0)
        dialog.dataset_list.itemDoubleClicked.emit(item)

        # Accept should have been called
        assert len(accept_called) == 1
