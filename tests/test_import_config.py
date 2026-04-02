from __future__ import annotations

import pytest

from kindred.gui.widgets.import_config import (
    SheetImportIntent,
    UnitDetection,
    UserImportIntent,
    detect_units_from_row_mapping,
    resolve_import_plans,
)

pytestmark = pytest.mark.unit


def _make_intent(sheet_names=(), apply_to_remaining=False):
    return UserImportIntent(
        sheet_names=tuple(sheet_names),
        apply_to_remaining=apply_to_remaining,
    )


def _make_sheet_intent(
    time_column="time",
    species_columns=("A",),
    time_unit="s",
    concentration_unit="M",
    override_no_unit_row=False,
):
    return SheetImportIntent(
        time_column=time_column,
        species_columns=tuple(species_columns),
        time_unit=time_unit,
        concentration_unit=concentration_unit,
        override_no_unit_row=override_no_unit_row,
    )


# ---------------------------------------------------------------------------
# resolve_import_plans tests
# ---------------------------------------------------------------------------


def test_csv_has_unit_row_no_override():
    """Detection has_unit_row=True with detected units; resolver uses intent units."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_sheet_intent(
        time_unit="ms", concentration_unit="uM", override_no_unit_row=False
    )
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
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
    intent = _make_sheet_intent(override_no_unit_row=True)
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
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
    intent = _make_sheet_intent(time_unit="ms", concentration_unit="uM")
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
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
    intent_a = _make_sheet_intent()
    intent_b = _make_sheet_intent()
    plans = resolve_import_plans(
        filepath="data.xlsx",
        file_type="excel",
        per_sheet_intents={"S1": intent_a, "S2": intent_b},
        per_sheet_detections={"S1": detection, "S2": detection},
        per_sheet_columns={"S1": ["time", "A"], "S2": ["time", "A"]},
    )

    assert len(plans) == 2
    assert plans[0].sheet_name == "S1"
    assert plans[1].sheet_name == "S2"
    assert plans[0].skip_unit_row is True
    assert plans[1].skip_unit_row is True


def test_excel_mixed_unit_row_presence_is_resolved_per_sheet():
    """Sheets may disagree on has_unit_row when each sheet has its own intent."""
    det_with = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    det_without = UnitDetection.empty()
    plans = resolve_import_plans(
        filepath="data.xlsx",
        file_type="excel",
        per_sheet_intents={
            "S1": _make_sheet_intent(time_unit="ms", concentration_unit="uM"),
            "S2": _make_sheet_intent(time_unit="s", concentration_unit="M"),
        },
        per_sheet_detections={"S1": det_with, "S2": det_without},
        per_sheet_columns={"S1": ["time", "A"], "S2": ["time", "A"]},
    )

    assert [plan.sheet_name for plan in plans] == ["S1", "S2"]
    assert plans[0].skip_unit_row is True
    assert plans[1].skip_unit_row is False


def test_missing_species_column_raises():
    """A species column not present in the sheet columns raises ValueError."""
    detection = UnitDetection.empty()
    intent = _make_sheet_intent(species_columns=("A", "MISSING"))

    with pytest.raises(ValueError, match="MISSING"):
        resolve_import_plans(
            filepath="data.csv",
            file_type="csv",
            per_sheet_intents={None: intent},
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
    intent = _make_sheet_intent(override_no_unit_row=False)

    with pytest.raises(ValueError, match="concentration"):
        resolve_import_plans(
            filepath="data.csv",
            file_type="csv",
            per_sheet_intents={None: intent},
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
    intent = _make_sheet_intent(override_no_unit_row=True)
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
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
    """Scoped extraction limits detected units to relevant columns."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row, relevant_column_names=["time", "A"])

    assert result.has_unit_row is True
    assert result.detected_conc_units == ("uM",)


def test_detect_units_has_unit_row_uses_full_row_but_extracts_only_relevant_columns():
    row = {"time": "0.5", "A": "1.2", "notes_time": "ms", "notes_conc": "uM"}

    result = detect_units_from_row_mapping(row, relevant_column_names=["time", "A"])

    assert result.has_unit_row is True
    assert result.detected_time_unit is None
    assert result.detected_conc_unit is None
    assert result.detected_conc_units == ()


def test_detect_units_no_unit_row():
    """Numeric values produce an empty detection (no unit row)."""
    row = {"time": "0.0", "A": "1.0"}
    result = detect_units_from_row_mapping(row)

    assert result.has_unit_row is False
    assert result.detected_time_unit is None


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------


def test_intent_units_override_detected_units():
    """Bug 1: When has_unit_row=True, user's intent units must win over detected units."""
    from kindred.core.datasets.units import parse_concentration_unit, parse_time_unit

    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit="uM",
        detected_conc_units=("uM",),
    )
    intent = _make_sheet_intent(
        time_unit="us",
        concentration_unit="nM",
        override_no_unit_row=False,
    )
    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    expected_time_factor = parse_time_unit("us")  # 1e-6
    expected_conc_factor = parse_concentration_unit("nM")  # 1e-9
    assert plan.time_factor == pytest.approx(expected_time_factor)
    assert plan.conc_factor == pytest.approx(expected_conc_factor)
    assert plan.original_time_unit == "us"
    assert plan.original_conc_unit == "nM"


def test_detect_units_scoped_columns_numeric_only():
    """Bug 2: full-row detection still reports a unit row even when selected columns are numeric."""
    row = {"time": "0.5", "A": "1.2", "unit_col1": "ms", "unit_col2": "uM"}
    result = detect_units_from_row_mapping(
        row, relevant_column_names=["time", "A"]
    )
    assert result.has_unit_row is True
    assert result.detected_time_unit is None
    assert result.detected_conc_units == ()


def test_detect_units_scoped_columns_with_units():
    """Bug 2 positive case: selected columns that DO contain unit text
    should yield has_unit_row=True."""
    row = {"time": "ms", "A": "uM", "B": "0.5", "C": "1.2"}
    result = detect_units_from_row_mapping(
        row, relevant_column_names=["time", "A"]
    )
    assert result.has_unit_row is True


def test_resolve_import_plans_uses_each_sheets_independent_intent():
    plans = resolve_import_plans(
        filepath="data.xlsx",
        file_type="excel",
        per_sheet_intents={
            "S1": _make_sheet_intent(
                time_column="time_us",
                species_columns=("A",),
                time_unit="us",
                concentration_unit="nM",
            ),
            "S2": _make_sheet_intent(
                time_column="elapsed_ms",
                species_columns=("B", "C"),
                time_unit="ms",
                concentration_unit="uM",
                override_no_unit_row=True,
            ),
        },
        per_sheet_detections={
            "S1": UnitDetection(
                has_unit_row=True,
                detected_time_unit="us",
                detected_conc_unit="nM",
                detected_conc_units=("nM",),
            ),
            "S2": UnitDetection(
                has_unit_row=True,
                detected_time_unit="ms",
                detected_conc_unit="uM",
                detected_conc_units=("uM",),
            ),
        },
        per_sheet_columns={
            "S1": ["time_us", "A"],
            "S2": ["elapsed_ms", "B", "C"],
        },
    )

    assert [plan.sheet_name for plan in plans] == ["S1", "S2"]
    assert plans[0].time_column == "time_us"
    assert plans[0].species_columns == ("A",)
    assert plans[0].original_time_unit == "us"
    assert plans[0].original_conc_unit == "nM"
    assert plans[1].time_column == "elapsed_ms"
    assert plans[1].species_columns == ("B", "C")
    assert plans[1].original_time_unit == "s"
    assert plans[1].original_conc_unit == "M"
