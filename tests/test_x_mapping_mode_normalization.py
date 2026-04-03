from __future__ import annotations

import numpy as np
import pytest

from kindred.core.analysis.x_mapping import parse_x_mapping_mode
from kindred.core.analysis.fit_dataset_payload import build_fit_dataset_payload as _build_selected_fit_dataset_payload


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        (" Auto ", "auto"),
        ("monotone", "monotone"),
        ("monotone only", "monotone"),
        ("monotone_only", "monotone"),
        ("monotoneonly", "monotone"),
        ("time_guided", "time_guided"),
        ("time-guided", "time_guided"),
        ("time guided", "time_guided"),
        ("timeguided", "time_guided"),
    ],
)
def test_parse_x_mapping_mode_accepts_common_spellings(raw: object, expected: str) -> None:
    assert parse_x_mapping_mode(raw) == expected


@pytest.mark.unit
def test_parse_x_mapping_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Invalid x_mapping_mode"):
        _ = parse_x_mapping_mode("wat")


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["timeguided", "time-guided", "time guided"])
def test_gui_payload_builder_accepts_timeguided_variants(raw: str) -> None:
    payload, err = _build_selected_fit_dataset_payload(
        dataset_id="ds1",
        t=np.asarray([0.0, 1.0, 2.0], dtype=float),
        species_data={"A": np.asarray([1.0, 2.0, 3.0], dtype=float)},
        selected_species=["A"],
        x_name="x",
        x_obs=np.asarray([10.0, 11.0, 12.0], dtype=float),
        x_mapping_mode=raw,
    )
    assert err is None
    assert payload is not None
    assert payload.get("x_mapping_mode") == "time_guided"
