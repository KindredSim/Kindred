from __future__ import annotations

import numpy as np
import pytest

from kindred.core.analysis.fit_dataset_payload import normalize_dataset_species_and_y
from kindred.gui.fitting.window import build_selected_fit_dataset_payload


@pytest.mark.unit
def test_normalize_dataset_species_and_y_single_species_flattens() -> None:
    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    species_list, y_mat = normalize_dataset_species_and_y(
        dataset_id="ds1",
        t_values=t,
        species="A",
        y=np.asarray([1.0, 2.0, 3.0], dtype=float),
    )
    assert species_list == ["A"]
    assert y_mat.shape == (1, 3)


@pytest.mark.unit
def test_normalize_dataset_species_and_y_multi_species_matrix() -> None:
    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    species_list, y_mat = normalize_dataset_species_and_y(
        dataset_id="ds1",
        t_values=t,
        species=["A", "B"],
        y=np.asarray([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=float),
    )
    assert species_list == ["A", "B"]
    assert y_mat.shape == (2, 3)


@pytest.mark.unit
def test_gui_payload_builder_roundtrips_into_core_normalizer() -> None:
    payload, err = build_selected_fit_dataset_payload(
        dataset_id="ds1",
        t=np.asarray([0.0, 1.0, 2.0], dtype=float),
        species_data={
            "A": np.asarray([1.0, 2.0, 3.0], dtype=float),
            "B": np.asarray([0.0, 1.0, 0.0], dtype=float),
        },
        selected_species=["A", "B"],
    )
    assert err is None
    assert payload is not None

    species_list, y_mat = normalize_dataset_species_and_y(
        dataset_id=str(payload["id"]),
        t_values=np.asarray(payload["t"], dtype=float),
        species=payload["species"],
        y=payload["y"],
    )
    assert species_list == ["A", "B"]
    assert y_mat.shape == (2, 3)


@pytest.mark.unit
def test_gui_payload_builder_preserves_invalid_x_obs_reason() -> None:
    payload, err = build_selected_fit_dataset_payload(
        dataset_id="ds1",
        t=np.asarray([0.0, 1.0, 2.0], dtype=float),
        species_data={"A": np.asarray([1.0, 2.0, 3.0], dtype=float)},
        selected_species=["A"],
        x_name="X",
        x_obs=object(),
        x_mapping_mode="time_guided",
    )

    assert payload is None
    assert err is not None
    assert "invalid x_obs" in err
    assert "missing x_obs" not in err


@pytest.mark.unit
def test_gui_payload_builder_keeps_missing_x_obs_distinct_from_invalid_x_obs() -> None:
    payload, err = build_selected_fit_dataset_payload(
        dataset_id="ds1",
        t=np.asarray([0.0, 1.0, 2.0], dtype=float),
        species_data={"A": np.asarray([1.0, 2.0, 3.0], dtype=float)},
        selected_species=["A"],
        x_name="X",
        x_obs=None,
        x_mapping_mode="time_guided",
    )

    assert payload is None
    assert err is not None
    assert "missing x_obs" in err
