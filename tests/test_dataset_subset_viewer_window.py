import numpy as np
import pytest

from PySide6 import QtCore
from PySide6 import QtWidgets

from kindred.gui.plot_config import is_pyqtgraph_available
from kindred.gui.widgets.dataset_subset_widget import DatasetSubsetWidget


def test_dataset_subset_widget_defaults_to_all_datasets_selected(qt_app):
    if not is_pyqtgraph_available():
        pytest.skip("pyqtgraph not available")

    entries = [
        {"id": "ds1", "t": np.array([0.0, 1.0]), "species_data": {"A": np.array([1.0, 0.5])}, "selected_species": ["A"]},
        {"id": "ds2", "t": np.array([0.0, 1.0]), "species_data": {"A": np.array([2.0, 1.0])}, "selected_species": ["A"]},
    ]
    viewer = DatasetSubsetWidget(dataset_entries=entries)
    try:
        assert len(viewer._grid._datasets) == 2
        # Toggle one dataset off and ensure grid updates.
        item = viewer._selector._tree_items["ds1"]  # type: ignore[attr-defined]
        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        assert len(viewer._grid._datasets) == 1
    finally:
        viewer.close()


@pytest.mark.gui
def test_dataset_subset_widget_does_not_force_large_initial_resize(qt_app, monkeypatch):
    if not is_pyqtgraph_available():
        pytest.skip("pyqtgraph not available")

    entries = [
        {"id": "ds1", "t": np.array([0.0, 1.0]), "species_data": {"A": np.array([1.0, 0.5])}, "selected_species": ["A"]},
        {"id": "ds2", "t": np.array([0.0, 1.0]), "species_data": {"A": np.array([2.0, 1.0])}, "selected_species": ["A"]},
    ]

    resize_calls: list[tuple[int, int]] = []

    def _spy_resize(self, w: int, h: int) -> None:
        resize_calls.append((int(w), int(h)))
        QtWidgets.QWidget.resize(self, w, h)

    monkeypatch.setattr(DatasetSubsetWidget, "resize", _spy_resize, raising=True)

    viewer = DatasetSubsetWidget(dataset_entries=entries)
    try:
        assert (1200, 700) not in resize_calls
        assert viewer.minimumWidth() == 0
        assert viewer.minimumHeight() == 0
    finally:
        viewer.close()
