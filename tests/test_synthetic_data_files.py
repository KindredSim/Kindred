from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kindred.core.datasets.csv_import import load_csv_dataset

SYNTHETIC_DIR = Path(__file__).parent / "data" / "synthetic"
GLOBAL_DIR = SYNTHETIC_DIR / "first_order_decay_global"
COMPLEX_DIR = SYNTHETIC_DIR / "complex_mechanism_global"
EXPECTED_FILES = [
    SYNTHETIC_DIR / "first_order_decay_single.csv",
    SYNTHETIC_DIR / "consecutive_A_B_C.csv",
    SYNTHETIC_DIR / "parallel_A_to_B_C.csv",
    GLOBAL_DIR / "dataset_01.csv",
    GLOBAL_DIR / "dataset_02.csv",
    GLOBAL_DIR / "dataset_03.csv",
]
EXPECTED_FILES.extend(COMPLEX_DIR / f"dataset_{idx:02d}.csv" for idx in range(1, 7))


@pytest.mark.parametrize("csv_path", EXPECTED_FILES)
def test_synthetic_dataset_files_present(csv_path: Path) -> None:
    assert csv_path.exists(), f"Missing synthetic dataset: {csv_path}"


@pytest.mark.parametrize("csv_path", EXPECTED_FILES)
def test_synthetic_dataset_integrity(csv_path: Path) -> None:
    _, payload = load_csv_dataset(str(csv_path))
    t = payload["t"]
    species = payload["species"]

    assert t.ndim == 1 and t.size > 2
    assert np.all(np.diff(t) > 0), "Time column must be strictly increasing"
    assert not np.isnan(t).any(), "Time column contains NaN"

    for name, values in species.items():
        assert values.shape == t.shape, f"Species {name} length mismatch"
        assert not np.isnan(values).any(), f"Species {name} contains NaN"
        assert np.ptp(values) > 0, f"Species {name} is constant"
        assert np.all(values >= 0), f"Species {name} has negative concentration"
        assert "_conc" not in name, f"Species column '{name}' should use canonical species name"
        assert values.max() < 3.0, f"Species {name} exceeds expected concentration bounds"

    metadata = payload.get("metadata", {})
    assert metadata.get("time_column"), "Metadata missing time column"
    assert metadata.get("species_columns"), "Metadata missing species columns"
