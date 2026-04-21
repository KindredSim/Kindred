import numpy as np
import pytest

pytestmark = pytest.mark.unit



def test_normalize_fit_dataset_dicts_adds_default_id_without_mutating_input():
    from kindred.core.analysis.fit_dataset_payload import normalize_fit_dataset_dicts

    ds = {
        "t": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 0.5, 0.25]),
        "species": "A",
        "note": "keep-me",
    }

    normalized = normalize_fit_dataset_dicts([ds])[0]

    assert normalized["id"] == "dataset_0"
    assert normalized["species"] == "A"
    assert normalized["note"] == "keep-me"
    assert np.array_equal(normalized["t"], ds["t"])
    assert np.array_equal(normalized["y"], ds["y"])
    assert "id" not in ds


def test_normalize_fit_dataset_dicts_requires_species_field():
    from kindred.core.analysis.fit_dataset_payload import normalize_fit_dataset_dicts

    ds = {
        "t": np.array([0.0, 1.0]),
        "y": np.array([1.0, 0.5]),
    }

    with pytest.raises(ValueError, match="missing required 'species' field"):
        normalize_fit_dataset_dicts([ds])


def test_coerce_fit_dataset_specs_normalizes_single_species_and_x_mode():
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs

    ds = {
        "id": "ds1",
        "t": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 0.5, 0.25]),
        "species": "A",
        "x_name": "X",
        "x_obs": np.array([0.0, 0.1, 0.2]),
        "x_mapping_mode": "monotone-only",
    }
    spec = coerce_fit_dataset_specs([ds])[0]

    assert spec.dataset_id == "ds1"
    assert spec.species_list == ["A"]
    assert spec.y_matrix.shape == (1, 3)
    assert spec.x_name == "X"
    assert spec.x_mode == "monotone"


def test_coerce_fit_dataset_specs_rejects_invalid_x_mapping_mode():
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs

    ds = {
        "id": "ds1",
        "t": np.array([0.0, 1.0]),
        "y": np.array([1.0, 0.5]),
        "species": "A",
        "x_name": "X",
        "x_obs": np.array([0.0, 0.1]),
        "x_mapping_mode": "wat",
    }
    with pytest.raises(ValueError, match="invalid x_mapping_mode"):
        _ = coerce_fit_dataset_specs([ds])
