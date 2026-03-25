from __future__ import annotations

import pytest

from kindred.gui.controllers.dataset_manager import DatasetFitSettings, DatasetManager


@pytest.mark.unit
def test_dataset_manager_batch_set_mapping_api() -> None:
    manager = DatasetManager(plot_tabs=object(), dataset_resolver=lambda _name: None)

    manager.update_fit_settings(
        "ds_by_id",
        DatasetFitSettings(batch_set="Set A", batch_set_id="set-id-a"),
    )
    manager.update_fit_settings(
        "ds_by_name_only",
        DatasetFitSettings(batch_set="Set B", batch_set_id=None),
    )
    manager.update_fit_settings(
        "ds_unmapped",
        DatasetFitSettings(batch_set=None, batch_set_id=None),
    )

    mapped = manager.datasets_mapped_to_batch_sets(
        set_ids=["set-id-a"],
        set_names=["Set B"],
    )
    assert set(mapped) == {"ds_by_id", "ds_by_name_only"}

    affected = manager.unmap_batch_sets(
        set_ids=["set-id-a"],
        set_names=["Set B"],
    )
    assert set(affected) == {"ds_by_id", "ds_by_name_only"}

    assert manager.get_fit_settings("ds_by_id").batch_set is None
    assert manager.get_fit_settings("ds_by_id").batch_set_id is None
    assert manager.get_fit_settings("ds_by_name_only").batch_set is None
    assert manager.get_fit_settings("ds_by_name_only").batch_set_id is None
    assert manager.get_fit_settings("ds_unmapped").batch_set is None
    assert manager.get_fit_settings("ds_unmapped").batch_set_id is None

