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
    concentration_units=None,
    override_no_unit_row=False,
):
    if concentration_units is None:
        concentration_units = {col: "M" for col in species_columns}
    return SheetImportIntent(
        time_column=time_column,
        species_columns=tuple(species_columns),
        time_unit=time_unit,
        concentration_units=concentration_units,
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
        detected_conc_unit_by_column={"A": "uM"},
    )
    intent = _make_sheet_intent(
        time_unit="ms", concentration_units={"A": "uM"}, override_no_unit_row=False
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
    assert plan.conc_factors["A"] == pytest.approx(1e-6)
    assert plan.original_time_unit == "ms"
    assert plan.original_conc_units["A"] == "uM"


def test_csv_has_unit_row_with_override():
    """Detection has_unit_row=True but override=True forces default s/M."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit_by_column={"A": "uM"},
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
    assert plan.conc_factors["A"] == pytest.approx(1.0)
    assert plan.original_time_unit == "s"
    assert plan.original_conc_units["A"] == "M"


def test_csv_no_unit_row_no_override():
    """No unit row detected; resolver falls back to intent units."""
    detection = UnitDetection.empty()
    intent = _make_sheet_intent(time_unit="ms", concentration_units={"A": "uM"})
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
    assert plan.conc_factors["A"] == pytest.approx(1e-6)


def test_excel_two_sheets_both_have_unit_row():
    """Two Excel sheets, both with unit rows, produce two resolved plans."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit_by_column={"A": "uM"},
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
        detected_conc_unit_by_column={"A": "uM"},
    )
    det_without = UnitDetection.empty()
    plans = resolve_import_plans(
        filepath="data.xlsx",
        file_type="excel",
        per_sheet_intents={
            "S1": _make_sheet_intent(time_unit="ms", concentration_units={"A": "uM"}),
            "S2": _make_sheet_intent(time_unit="s", concentration_units={"A": "M"}),
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


def test_mixed_concentration_units_resolves_per_column():
    """Multiple distinct concentration units resolve per-column factors."""
    from kindred.core.datasets.units import parse_concentration_unit

    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit_by_column={"A": "uM", "B": "nM"},
    )
    intent = _make_sheet_intent(
        species_columns=("A", "B"),
        concentration_units={"A": "uM", "B": "nM"},
        override_no_unit_row=False,
    )

    plans = resolve_import_plans(
        filepath="data.csv",
        file_type="csv",
        per_sheet_intents={None: intent},
        per_sheet_detections={None: detection},
        per_sheet_columns={None: ["time", "A", "B"]},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.conc_factors["A"] == pytest.approx(parse_concentration_unit("uM"))
    assert plan.conc_factors["B"] == pytest.approx(parse_concentration_unit("nM"))
    assert plan.conc_factors["A"] != plan.conc_factors["B"]


def test_override_with_physical_unit_row():
    """Override=True with a physical unit row: skip=True but factors are 1.0."""
    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit_by_column={"A": "uM"},
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
    assert plan.conc_factors["A"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_units_from_row_mapping tests
# ---------------------------------------------------------------------------


def test_detect_units_full_row():
    """Full row with time and concentration units is detected correctly."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row)

    assert result.has_unit_row is True
    assert result.detected_time_unit == "ms"
    assert result.detected_conc_unit_by_column == {"A": "uM", "B": "nM"}


def test_detect_units_scoped_to_relevant():
    """Scoped extraction limits detected units to relevant columns."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row, relevant_column_names=["time", "A"])

    assert result.has_unit_row is True
    assert result.detected_conc_unit_by_column == {"A": "uM"}


def test_detect_units_has_unit_row_uses_full_row_but_extracts_only_relevant_columns():
    row = {"time": "0.5", "A": "1.2", "notes_time": "ms", "notes_conc": "uM"}

    result = detect_units_from_row_mapping(row, relevant_column_names=["time", "A"])

    assert result.has_unit_row is True
    assert result.detected_time_unit is None
    assert result.detected_conc_unit_by_column == {}


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
    """When has_unit_row=True, user's intent units must win over detected units."""
    from kindred.core.datasets.units import parse_concentration_unit, parse_time_unit

    detection = UnitDetection(
        has_unit_row=True,
        detected_time_unit="ms",
        detected_conc_unit_by_column={"A": "uM"},
    )
    intent = _make_sheet_intent(
        time_unit="us",
        concentration_units={"A": "nM"},
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
    assert plan.conc_factors["A"] == pytest.approx(expected_conc_factor)
    assert plan.original_time_unit == "us"
    assert plan.original_conc_units["A"] == "nM"


def test_detect_units_scoped_columns_numeric_only():
    """Full-row detection still reports a unit row even when selected columns are numeric."""
    row = {"time": "0.5", "A": "1.2", "unit_col1": "ms", "unit_col2": "uM"}
    result = detect_units_from_row_mapping(
        row, relevant_column_names=["time", "A"]
    )
    assert result.has_unit_row is True
    assert result.detected_time_unit is None
    assert result.detected_conc_unit_by_column == {}


def test_detect_units_scoped_columns_with_units():
    """Scoped detection: selected columns with unit text yield has_unit_row=True."""
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
                concentration_units={"A": "nM"},
            ),
            "S2": _make_sheet_intent(
                time_column="elapsed_ms",
                species_columns=("B", "C"),
                time_unit="ms",
                concentration_units={"B": "M", "C": "M"},
                override_no_unit_row=True,
            ),
        },
        per_sheet_detections={
            "S1": UnitDetection(
                has_unit_row=True,
                detected_time_unit="us",
                detected_conc_unit_by_column={"A": "nM"},
            ),
            "S2": UnitDetection(
                has_unit_row=True,
                detected_time_unit="ms",
                detected_conc_unit_by_column={"B": "uM", "C": "uM"},
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
    assert plans[0].original_conc_units["A"] == "nM"
    assert plans[1].time_column == "elapsed_ms"
    assert plans[1].species_columns == ("B", "C")
    assert plans[1].original_time_unit == "s"
    assert plans[1].original_conc_units["B"] == "M"


# ---------------------------------------------------------------------------
# Per-column unit conversion regression tests
# ---------------------------------------------------------------------------


def test_per_column_detection_returns_column_mapping():
    """detect_units_from_row_mapping returns per-column unit mapping."""
    row = {"time": "ms", "A": "uM", "B": "nM"}
    result = detect_units_from_row_mapping(row, relevant_column_names=["A", "B"])
    assert result.detected_conc_unit_by_column == {"A": "uM", "B": "nM"}


# ---------------------------------------------------------------------------
# SheetImportIntent validation tests
# ---------------------------------------------------------------------------


def test_sheet_import_intent_rejects_mismatched_concentration_units_keys():
    """SheetImportIntent must raise when concentration_units keys differ from species_columns."""
    with pytest.raises((ValueError, AssertionError), match="concentration_units keys must match species_columns"):
        SheetImportIntent(
            time_column="time",
            species_columns=("A", "B"),
            time_unit="s",
            concentration_units={"A": "uM"},
            override_no_unit_row=False,
        )


def test_sheet_import_intent_accepts_matching_concentration_units_keys():
    """SheetImportIntent construction succeeds when keys match species_columns."""
    intent = SheetImportIntent(
        time_column="time",
        species_columns=("A", "B"),
        time_unit="s",
        concentration_units={"A": "uM", "B": "nM"},
        override_no_unit_row=False,
    )
    assert intent.species_columns == ("A", "B")
    assert intent.concentration_units == {"A": "uM", "B": "nM"}
