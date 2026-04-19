import numpy as np
import pytest

from kindred.core.datasets.csv_import import load_csv_dataset

pytestmark = [pytest.mark.integration]


def test_data_manager_explicit_mapping(tmp_path):
    """Explicit column mapping loads correct time/species arrays."""
    csv_path = tmp_path / "custom_dataset.csv"
    csv_path.write_text("stamp,A_rate,B_rate\n0,1.0,2.0\n5,3.0,4.0\n")

    name, payload = load_csv_dataset(
        str(csv_path),
        time_column="stamp",
        species_columns=["B_rate"],
    )

    assert name == "custom_dataset.csv"
    assert np.allclose(payload["t"], [0.0, 5.0])
    assert list(payload["species"].keys()) == ["B_rate"]
    assert payload["metadata"]["time_column"] == "stamp"
    assert payload["metadata"]["mapping_source"] == "explicit"
