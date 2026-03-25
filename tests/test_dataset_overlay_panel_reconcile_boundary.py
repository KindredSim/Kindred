from __future__ import annotations

import pytest

from kindred.gui.widgets.dataset_overlay_panel import DatasetOverlayPanel

pytestmark = pytest.mark.gui


def _catalog(**datasets: list[str]) -> dict[str, dict]:
    return {
        str(name): {"species": {str(species): [1.0, 0.5] for species in list(species_names)}}
        for name, species_names in datasets.items()
    }


def test_reconcile_selection_preserves_valid_overlap_after_catalog_change(qt_app):
    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(_catalog(ds1=["A", "B"]))
        panel.reconcile_selection(
            previous_selected_datasets=["ds1"],
            previous_enabled_species={"ds1": {"A"}},
            include_dataset_ids=["ds1"],
            ordered_dataset_ids=["ds1"],
            allow_default_include=False,
        )

        panel.set_datasets(_catalog(ds1=["A", "C"]))
        panel.reconcile_selection(
            previous_selected_datasets=["ds1"],
            previous_enabled_species={"ds1": {"A"}},
            include_dataset_ids=["ds1"],
            ordered_dataset_ids=["ds1"],
            allow_default_include=False,
        )

        assert panel.selected_dataset_species() == {"ds1": {"A"}}
    finally:
        panel.close()
        qt_app.processEvents()


def test_reconcile_selection_expands_to_all_species_when_overlap_disappears(qt_app):
    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(_catalog(ds1=["C", "D"]))
        panel.reconcile_selection(
            previous_selected_datasets=["ds1"],
            previous_enabled_species={"ds1": {"A"}},
            include_dataset_ids=["ds1"],
            ordered_dataset_ids=["ds1"],
            allow_default_include=False,
        )

        assert panel.selected_dataset_species() == {"ds1": {"C", "D"}}
    finally:
        panel.close()
        qt_app.processEvents()


def test_reconcile_selection_falls_back_to_first_valid_included_dataset(qt_app):
    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(
            {
                "empty": {"species": {}},
                **_catalog(ds1=["A"], ds2=["B"]),
            }
        )
        panel.reconcile_selection(
            previous_selected_datasets=["missing"],
            previous_enabled_species={"missing": {"X"}},
            include_dataset_ids=["empty", "ds2", "ds1"],
            ordered_dataset_ids=["empty", "ds1", "ds2"],
            allow_default_include=False,
        )

        assert panel.selected_dataset_species() == {"ds2": {"B"}}
    finally:
        panel.close()
        qt_app.processEvents()


def test_reconcile_selection_is_silent_by_default_and_emits_once_when_requested(qt_app):
    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(_catalog(ds1=["A", "B"]))
        events: list[list[str]] = []
        panel.selectionChanged.connect(lambda datasets: events.append(list(datasets)))

        panel.reconcile_selection(
            previous_selected_datasets=[],
            previous_enabled_species={},
            include_dataset_ids=["ds1"],
            ordered_dataset_ids=["ds1"],
            allow_default_include=True,
        )

        assert events == []

        panel.reconcile_selection(
            previous_selected_datasets=["ds1"],
            previous_enabled_species={"ds1": {"A"}},
            include_dataset_ids=["ds1"],
            ordered_dataset_ids=["ds1"],
            allow_default_include=False,
            emit=True,
        )

        assert events == [["ds1"]]
        assert panel.selected_dataset_species() == {"ds1": {"A"}}
    finally:
        panel.close()
        qt_app.processEvents()
