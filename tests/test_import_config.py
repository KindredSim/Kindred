from __future__ import annotations

import pytest

from kindred.gui.widgets.import_config import (
    UnitDetection,
    UserImportIntent,
    detect_units_from_row_mapping,
    resolve_import_plans,
)

pytestmark = pytest.mark.unit


def _make_intent(
    time_column="time",
    species_columns=("A",),
    time_unit="s",
    concentration_unit="M",
    override_no_unit_row=False,
    sheet_names=(),
    apply_to_remaining=False,
):
    return UserImportIntent(
        time_column=time_column,
        species_columns=tuple(species_columns),
        time_unit=time_unit,
        concentration_unit=concentration_unit,
        override_no_unit_row=override_no_unit_row,
        sheet_names=tuple(sheet_names),
        apply_to_remaining=apply_to_remaining,
    )


# ---------------------------------------------------------------------------
# resolve_import_plans tests
# ---------------------------------------------------------------------------


def test_csv_has_unit_row_no_override():
    """Detection has_unit_row=True with detected units; resolver uses those units."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_intent(override_no_unit_row=False)
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        intent=intent,
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.skip_unit_row is True
    assert plan.time_factor == pytest.approx(1e-3)
    assert plan.conc_factor == pytest.approx(1e-6)
    assert plan.original_time_unit == "ms"
    assert plan.original_conc_unit == "uM"


def test_csv_has_unit_row_with_override():
    """Detection has_unit_row=True but override=True forces default s/M."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_intent(override_no_unit_row=True)
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        intent=intent,
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.skip_unit_row is True
    assert plan.time_factor == pytest.approx(1.0)
    assert plan.conc_factor == pytest.approx(1.0)
    assert plan.original_time_unit == "s"
    assert plan.original_conc_unit == "M"


def test_csv_no_unit_row_no_override():
    """No unit row detected; resolver falls back to intent units."""
    detection = UnitDetection.empty()
    intent = _make_intent(time_unit="ms", concentration_unit="uM")
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        intent=intent,
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.skip_unit_row is False
    assert plan.time_factor == pytest.approx(1e-3)
    assert plan.conc_factor == pytest.approx(1e-6)


def test_excel_two_sheets_both_have_unit_row():
    """Two Excel sheets, both with unit rows, produce two resolved plans."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_intent(sheet_names=("S1", "S2"))
    plans = resolve_import_plans(
        filepath="data.xlsx",
        file_type="excel",
        intent=intent,
        per_sheet_detections={"S1": detection, "S2": detection},
        per_sheet_columns={"S1": ["time", "A"], "S2": ["time", "A"]},
    )

    assert len(plans) == 2
    assert plans[0].sheet_name == "S1"
    assert plans[1].sheet_name == "S2"
    assert plans[0].skip_unit_row is True
    assert plans[1].skip_unit_row is True


def test_excel_mixed_unit_row_presence_raises():
    """Sheets that disagree on has_unit_row raise ValueError."""
    det_with = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    det_without = UnitDetection.empty()
    intent = _make_intent(sheet_names=("S1", "S2"))

    with pytest.raises(ValueError, match="disagree"):
        resolve_import_plans(
            filepath="data.xlsx",
            file_type="excel",
            intent=intent,
            per_sheet_detections={"S1": det_with, "S2": det_without},
            per_sheet_columns={"S1": ["time", "A"], "S2": ["time", "A"]},
        )


def test_missing_species_column_raises():
    """A species column not present in the sheet columns raises ValueError."""
    detection = UnitDetection.empty()
    intent = _make_intent(species_columns=("A", "MISSING"))

    with pytest.raises(ValueError, match="MISSING"):
        resolve_import_plans(
            filepath="data.csv",
            file_type="csv",
            intent=intent,
            per_sheet_detections={None: detection},
            per_sheet_columns={None: ["time", "A"]},
        )


def test_mixed_concentration_units_raises():
    """Multiple distinct concentration factors among detected units raise ValueError."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM", "nM"),
    )
    intent = _make_intent(override_no_unit_row=False)

    with pytest.raises(ValueError, match="concentration"):
        resolve_import_plans(
            filepath="data.csv",
            file_type="csv",
            intent=intent,
            per_sheet_detections={None: detection},
            per_sheet_columns={None: ["time", "A"]},
        )


def test_override_with_physical_unit_row():
    """Override=True with a physical unit row: skip=True but factors are 1.0."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_intent(override_no_unit_row=True)
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        intent=intent,
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.skip_unit_row is True
    assert plan.time_factor == pytest.approx(1.0)
    assert plan.conc_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_units_from_row_mapping tests
# ---------------------------------------------------------------------------


def test_detect_units_full_row():
    """Full row with time and concentration units is detected correctly."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row)

    assert result.has_unit_row is True
    assert result.detected_time_unit == "ms"
    assert result.detected_conc_unit == "uM"
    assert "uM" in result.detected_conc_units
    assert "nM" in result.detected_conc_units


def test_detect_units_scoped_to_relevant():
    """Scoped extraction limits conc_units but has_unit_row checks all values."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row, relevant_column_names=["time", "A"])

    assert result.has_unit_row is True
    assert result.detected_conc_units == ("uM",)


def test_detect_units_no_unit_row():
    """Numeric values produce an empty detection (no unit row)."""
    row = {"time": "0.0", "A": "1.0"}
    result = detect_units_from_row_mapping(row)

    assert result.has_unit_row is False
    assert result.detected_time_unit is None
