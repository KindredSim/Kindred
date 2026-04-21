import numpy as np
import pytest


@pytest.mark.unit
def test_migrate_initial_concentrations_block_rewrites_stub_and_returns_seed():
    from kindred.core.batch_initial_conditions import migrate_reaction_dsl_initial_concentrations

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Initial concentrations",
            "[A] = 1.0",
            "[B] = 0.0",
            "",
            "let x = 3  # must be preserved",
        ]
    )

    seed, rewritten = migrate_reaction_dsl_initial_concentrations(reaction_text, set_name="set1")

    assert seed == {"A": pytest.approx(1.0), "B": pytest.approx(0.0)}
    assert "[A]" not in rewritten
    assert "[B]" not in rewritten
    assert "let x = 3" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set1)" in rewritten


@pytest.mark.unit
def test_migrate_initial_concentrations_is_one_time_seed_if_stub_present():
    from kindred.core.batch_initial_conditions import migrate_reaction_dsl_initial_concentrations

    stubbed = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there.",
        ]
    )

    seed, rewritten = migrate_reaction_dsl_initial_concentrations(stubbed, set_name="set1")
    assert seed == {}
    assert rewritten == stubbed


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_preserves_imported_names_and_rewrites_each_block():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "randomname3 = {",
            "[A] = 1.0",
            "[B] = 0.0",
            "}",
            "",
            "set-two = {",
            "initial: A=2.5, B=0.5",
            "}",
            "",
            "let x = 3",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {
        "randomname3": {"A": pytest.approx(1.0), "B": pytest.approx(0.0)},
        "set-two": {"A": pytest.approx(2.5), "B": pytest.approx(0.5)},
    }
    assert "randomname3 = {" not in rewritten
    assert "set-two = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (randomname3)" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set-two)" in rewritten
    assert "let x = 3" in rewritten
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "randomname3 = {" not in stripped
    assert "set-two = {" not in stripped
    assert "[A] = 1.0" not in stripped
    assert "initial: A=2.5, B=0.5" not in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_preserves_spaced_names_and_strips_block():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "Set B = {",
            "[A] = 1.5",
            "[B] = 0.25",
            "}",
            "",
            "let x = 3",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {
        "Set B": {"A": pytest.approx(1.5), "B": pytest.approx(0.25)},
    }
    assert "Set B = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (Set B)" in rewritten
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "Set B = {" not in stripped
    assert "[A] = 1.5" not in stripped
    assert "[B] = 0.25" not in stripped
    assert "let x = 3" in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_leaves_empty_algebra_multiword_block_untouched():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Algebra",
            "let config = {",
            "}",
            "",
            "Set B = {",
            "[A] = 1.5",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {"Set B": {"A": pytest.approx(1.5)}}
    assert "let config = {" in rewritten
    assert "\n}\n" in f"\n{rewritten}\n"
    assert "Initial concentrations moved to Batch Initial Conditions table (let config)" not in rewritten
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "let config = {" in stripped
    assert "\n}\n" in f"\n{stripped}\n"
    assert "Set B = {" not in stripped
    assert "[A] = 1.5" not in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_imports_empty_named_blocks_but_leaves_empty_algebra_let_block():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "set2 = {",
            "}",
            "",
            "Set B = {",
            "# comment-only empty import",
            "}",
            "",
            "# Algebra",
            "let config = {",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {
        "set2": {},
        "Set B": {},
    }
    assert "set2 = {" not in rewritten
    assert "Set B = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set2)" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (Set B)" in rewritten
    assert "let config = {" in rewritten
    assert "\n}\n" in f"\n{rewritten}\n"
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "set2 = {" not in stripped
    assert "Set B = {" not in stripped
    assert "let config = {" in stripped
    assert "\n}\n" in f"\n{stripped}\n"


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_merges_empty_default_named_block_with_legacy_initials():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "set1 = {",
            "}",
            "",
            "# Initial concentrations",
            "[A] = 1.0",
            "[B] = 0.5",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(
        reaction_text,
        default_set_name="set1",
    )

    assert seed_sets == {
        "set1": {"A": pytest.approx(1.0), "B": pytest.approx(0.5)},
    }
    assert "set1 = {" not in rewritten
    assert "[A] = 1.0" not in rewritten
    assert "[B] = 0.5" not in rewritten
    assert rewritten.count("Initial concentrations moved to Batch Initial Conditions table (set1)") == 2
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "set1 = {" not in stripped
    assert "[A] = 1.0" not in stripped
    assert "[B] = 0.5" not in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_leaves_outside_algebra_let_and_param_blocks_untouched():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "let baseline = {",
            "}",
            "",
            "param sweep = {",
            "# comment-only empty import",
            "}",
            "",
            "# Algebra",
            "let observable = [A]",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {}
    assert "let baseline = {" in rewritten
    assert "param sweep = {" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (let baseline)" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (param sweep)" not in rewritten
    assert "# Algebra" in rewritten
    assert "let observable = [A]" in rewritten
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "let baseline = {" in stripped
    assert "param sweep = {" in stripped
    assert "# comment-only empty import" in stripped
    assert "# Algebra" in stripped
    assert "let observable = [A]" in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_does_not_harvest_inner_initials_from_unsupported_let_or_param_blocks():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "let baseline = {",
            "[A] = 1.25",
            "}",
            "",
            "param sweep = {",
            "initial: B=0.75",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {}
    assert rewritten == reaction_text


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_does_not_harvest_nested_supported_block_inside_unsupported_outer_block():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "let baseline = {",
            "inner = {",
            "[A] = 1.25",
            "}",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {}
    assert rewritten == reaction_text


@pytest.mark.unit
def test_strip_named_initial_concentration_sets_skips_nested_blocks_inside_unsupported_outer_blocks():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_named_reaction_dsl_initial_concentration_sets,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "let baseline = {",
            "inner = {",
            "[A] = 1.25",
            "}",
            "}",
            "",
            "param sweep = {",
            "other = {",
            "[B] = 0.75",
            "}",
            "}",
            "",
            "set-two = {",
            "[C] = 2.0",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {"set-two": {"C": pytest.approx(2.0)}}
    assert "let baseline = {" in rewritten
    assert "inner = {" in rewritten
    assert "param sweep = {" in rewritten
    assert "other = {" in rewritten
    assert "set-two = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set-two)" in rewritten

    stripped = strip_named_reaction_dsl_initial_concentration_sets(reaction_text)

    assert "let baseline = {" in stripped
    assert "inner = {" in stripped
    assert "[A] = 1.25" in stripped
    assert "param sweep = {" in stripped
    assert "other = {" in stripped
    assert "[B] = 0.75" in stripped
    assert "set-two = {" not in stripped
    assert "[C] = 2.0" not in stripped


@pytest.mark.unit
def test_strip_reaction_dsl_initial_concentrations_preserves_unsupported_outer_blocks():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_named_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "let baseline = {",
            "inner = {",
            "[A] = 1.25",
            "}",
            "}",
            "",
            "param sweep = {",
            "initial: B=0.75",
            "}",
            "",
            "set-two = {",
            "[C] = 2.0",
            "}",
            "",
            "# Initial concentrations",
            "[D] = 3.0",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)
    parse_stripped = strip_named_reaction_dsl_initial_concentration_sets(reaction_text)
    runtime_stripped = strip_reaction_dsl_initial_concentrations(reaction_text)

    assert seed_sets == {
        "set-two": {"C": pytest.approx(2.0)},
        "set1": {"D": pytest.approx(3.0)},
    }
    for text in (rewritten, parse_stripped):
        assert "let baseline = {" in text
        assert "inner = {" in text
        assert "[A] = 1.25" in text
        assert "param sweep = {" in text
        assert "initial: B=0.75" in text

    assert "let baseline = {" in runtime_stripped
    assert "inner = {" in runtime_stripped
    assert "[A] = 1.25" in runtime_stripped
    assert "param sweep = {" in runtime_stripped
    assert "initial: B=0.75" in runtime_stripped

    assert "set-two = {" not in rewritten
    assert "set-two = {" not in parse_stripped
    assert "set-two = {" not in runtime_stripped
    assert "[C] = 2.0" not in rewritten
    assert "[C] = 2.0" not in parse_stripped
    assert "[C] = 2.0" not in runtime_stripped
    assert "# Initial concentrations" not in rewritten.splitlines()
    assert "[D] = 3.0" not in rewritten
    assert "# Initial concentrations" in parse_stripped.splitlines()
    assert "[D] = 3.0" in parse_stripped
    assert "# Initial concentrations" not in runtime_stripped.splitlines()
    assert "[D] = 3.0" not in runtime_stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_does_not_insert_stub_when_nothing_was_imported():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Initial concentrations",
            "let baseline = {",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {}
    assert "Initial concentrations moved to Batch Initial Conditions table" not in rewritten
    assert rewritten == reaction_text


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_leaves_empty_algebra_let_and_param_blocks_untouched():
    from kindred.core.batch_initial_conditions import (
        migrate_reaction_dsl_initial_concentration_sets,
        strip_reaction_dsl_initial_concentrations,
    )

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Algebra",
            "let config = {",
            "}",
            "param sweep = {",
            "}",
            "",
            "set-two = {",
            "[A] = 2.5",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {"set-two": {"A": pytest.approx(2.5)}}
    assert "let config = {" in rewritten
    assert "param sweep = {" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (let config)" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (param sweep)" not in rewritten
    stripped = strip_reaction_dsl_initial_concentrations(reaction_text)
    assert "let config = {" in stripped
    assert "param sweep = {" in stripped
    assert "set-two = {" not in stripped
    assert "[A] = 2.5" not in stripped


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_imports_new_blocks_even_with_existing_stub():
    from kindred.core.batch_initial_conditions import migrate_reaction_dsl_initial_concentration_sets

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there.",
            "",
            "randomname3 = {",
            "[A] = 1.0",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {"randomname3": {"A": pytest.approx(1.0)}}
    assert rewritten.count("Initial concentrations moved to Batch Initial Conditions table") == 2
    assert "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there." in rewritten
    assert "randomname3 = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (randomname3)" in rewritten


@pytest.mark.unit
def test_migrate_named_initial_concentration_sets_ignores_non_import_brace_blocks():
    from kindred.core.batch_initial_conditions import migrate_reaction_dsl_initial_concentration_sets

    reaction_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "let config = {",
            "alpha = 1",
            "}",
            "",
            "set-two = {",
            "[A] = 2.5",
            "}",
        ]
    )

    seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(reaction_text)

    assert seed_sets == {"set-two": {"A": pytest.approx(2.5)}}
    assert "let config = {" in rewritten
    assert "alpha = 1" in rewritten
    assert "set-two = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set-two)" in rewritten


@pytest.mark.unit
def test_batch_store_paste_refuses_when_out_of_bounds():
    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore

    store = BatchInitialConditionsStore()
    store.set_species(["A", "B"])
    store.ensure_set("set1")
    store.ensure_set("set2")

    # Start at row=1 (set2), col=2 (species B), but paste 2 columns => out of bounds.
    with pytest.raises(ValueError, match="exceeds table bounds"):
        store.apply_paste_block(start_row=1, start_col=2, text="1\t2")


@pytest.mark.unit
def test_batch_store_validation_reports_invalid_numeric_cells():
    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore

    store = BatchInitialConditionsStore()
    store.set_species(["A"])
    store.ensure_set("set1")
    store.set_value(0, "A", "not-a-number")

    invalid = store.validate_numeric_cells(rows=[0])
    assert (0, "A") in invalid


@pytest.mark.unit
def test_batch_store_shown_defaults_true_and_round_trips_serialization():
    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore

    store = BatchInitialConditionsStore()
    store.set_species(["A"])
    store.ensure_set("set2")

    assert store.is_shown(0) is True
    assert store.is_shown(1) is True

    store.set_shown(1, False)

    payload = store.as_serializable()
    assert payload["sets"][0]["shown"] is True
    assert payload["sets"][1]["shown"] is False

    restored = BatchInitialConditionsStore.from_serializable(payload)
    assert restored.is_shown(0) is True
    assert restored.is_shown(1) is False

    legacy = {
        "sets": [
            {"set_id": "set-1", "name": "set1", "values": {"A": "1.0"}},
            {"set_id": "set-2", "name": "set2", "values": {"A": "2.0"}},
        ],
        "visible_species": ["A"],
    }
    legacy_restored = BatchInitialConditionsStore.from_serializable(legacy)
    assert legacy_restored.is_shown(0) is True
    assert legacy_restored.is_shown(1) is True


@pytest.mark.unit
def test_run_scope_selected_defaults_to_first_row():
    from kindred.core.batch_initial_conditions import resolve_run_scope

    assert resolve_run_scope(selected_rows=[], total_rows=3, mode="selected") == [0]
    assert resolve_run_scope(selected_rows=[2, 0], total_rows=3, mode="selected") == [2, 0]


@pytest.mark.unit
def test_run_scope_selected_falls_back_to_focused_row_before_first_row():
    from kindred.core.batch_initial_conditions import resolve_run_scope

    assert resolve_run_scope(selected_rows=[], total_rows=3, mode="selected", fallback_row=2) == [2]
    assert resolve_run_scope(selected_rows=[], total_rows=3, mode="selected", fallback_row=None) == [0]
    assert resolve_run_scope(selected_rows=[1], total_rows=3, mode="selected", fallback_row=2) == [1]


@pytest.mark.unit
def test_run_scope_all_includes_every_row():
    from kindred.core.batch_initial_conditions import resolve_run_scope

    assert resolve_run_scope(selected_rows=[1], total_rows=3, mode="all") == [0, 1, 2]


@pytest.mark.unit
def test_dataset_base_label_strips_underscore_suffixes():
    from kindred.core.batch_initial_conditions import dataset_base_label

    assert dataset_base_label("dataset1") == "dataset1"
    assert dataset_base_label("dataset1_1") == "dataset1"
    assert dataset_base_label("dataset1_2") == "dataset1"
    assert dataset_base_label("dataset1_2_3") == "dataset1"
    assert dataset_base_label("dataset1_1.csv") == "dataset1"
    assert dataset_base_label("dataset_01.csv") == "dataset_01"
    assert dataset_base_label("dataset_01_1.csv") == "dataset_01"


@pytest.mark.unit
def test_seed_from_dataset_first_row_only_when_t0_within_tolerance():
    from kindred.core.batch_initial_conditions import seed_batch_set_from_dataset_first_row

    dataset = {
        "t": np.array([0.0, 1.0, 2.0]),
        "species": {"A": np.array([1.0, 0.5, 0.25])},
    }
    mechanism_species = ["A", "B"]
    seeded = seed_batch_set_from_dataset_first_row(dataset, mechanism_species, tol=1e-9)
    assert seeded == {"A": pytest.approx(1.0), "B": pytest.approx(0.0)}

    dataset_offset = {
        "t": np.array([1e-4, 1.0]),
        "species": {"A": np.array([9.0, 8.0])},
    }
    seeded2 = seed_batch_set_from_dataset_first_row(dataset_offset, mechanism_species, tol=1e-9)
    assert seeded2 == {}


@pytest.mark.gui
def test_batch_table_paste_revalidates_invalid_highlighting(main_window):
    from PySide6 import QtCore, QtWidgets

    table = getattr(main_window, "_batch_table", None)
    model = getattr(main_window, "_batch_model", None)
    assert table is not None
    assert model is not None

    model.set_species(["A"])
    idx = model.index(0, 1)
    table.setCurrentIndex(idx)

    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.setText("not-a-number")

    assert model.data(idx, QtCore.Qt.BackgroundRole) is None
    table._handle_paste()

    brush = model.data(idx, QtCore.Qt.BackgroundRole)
    assert brush is not None
    assert brush.color().getRgb()[:3] == (255, 210, 210)


@pytest.mark.gui
def test_batch_table_setdata_and_paste_produce_same_invalid_state(qt_app):
    from PySide6 import QtCore, QtWidgets

    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
    from kindred.gui.widgets.batch_initial_conditions_table import (
        BatchInitialConditionsTableModel,
        BatchInitialConditionsTableView,
    )

    store = BatchInitialConditionsStore()
    store.set_species(["A"])
    store.ensure_set("set1")

    model = BatchInitialConditionsTableModel(store)
    table = BatchInitialConditionsTableView()
    table.setModel(model)

    idx = model.index(0, 1)
    assert model.setData(idx, "not-a-number", QtCore.Qt.EditRole) is True
    setdata_invalid = set(model._invalid)
    setdata_brush = model.data(idx, QtCore.Qt.BackgroundRole)
    assert setdata_brush is not None

    model.setData(idx, "1.0", QtCore.Qt.EditRole)
    assert model._invalid == set()

    table.setCurrentIndex(idx)
    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.setText("not-a-number")
    table._handle_paste()

    paste_invalid = set(model._invalid)
    paste_brush = model.data(idx, QtCore.Qt.BackgroundRole)
    assert paste_brush is not None

    assert setdata_invalid == {(0, "A")}
    assert paste_invalid == setdata_invalid
    assert paste_brush.color().getRgb()[:3] == setdata_brush.color().getRgb()[:3]


@pytest.mark.gui
def test_data_manager_unique_dataset_names_use_underscore_suffix(qt_app):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    panel = DataManagerPanel()
    try:
        panel._datasets.clear()
        panel._datasets["dataset.csv"] = {"t": np.array([0.0]), "species": {"A": np.array([1.0])}}
        assert panel._make_unique_dataset_name("dataset.csv") == "dataset_1.csv"
        panel._datasets["dataset_1.csv"] = {"t": np.array([0.0]), "species": {"A": np.array([1.0])}}
        assert panel._make_unique_dataset_name("dataset.csv") == "dataset_2.csv"
    finally:
        panel.close()


@pytest.mark.gui
def test_display_cached_batch_selection_first_run_is_not_blank(main_window):
    """
    Regression: After the first simulation completes, the plot must render curves
    immediately without requiring a Y-checkbox toggle.

    This reproduces the blank-plot bug by calling the same cached-display path
    used after simulation completion.
    """
    cache_key = "unit-cache"
    t = np.array([0.0, 1.0])
    series = {"A": np.array([1.0, 0.5]), "B": np.array([0.0, 0.5])}
    main_window.simulation_controller.batch_cache.result_cache[f"{cache_key}::set1"] = {
        "t": t,
        "series": series,
        "algebra_scalars": {},
    }

    ok = main_window.display_cached_batch_selection(
        cache_key=cache_key,
        selected_sets=["set1"],
        prefer_set="set1",
    )
    assert ok is True

    plot = main_window._plot_tabs._main_plot
    # Selection and visibility must be consistent: if Y items are checked, the
    # plot should not be empty.
    assert set(plot.selected_series()) == {"A", "B"}
    assert set(plot.visible_series()) == {"A", "B"}


@pytest.mark.gui
def test_batch_table_has_add_and_move_controls(main_window):
    """Batch Initial Conditions table must expose discoverable Add/Up/Down controls."""
    from PySide6 import QtWidgets

    assert main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton") is not None
    assert main_window.findChild(QtWidgets.QPushButton, "moveBatchSetUpButton") is not None
    assert main_window.findChild(QtWidgets.QPushButton, "moveBatchSetDownButton") is not None


@pytest.mark.gui
def test_batch_run_controls_are_condensed_and_run_all_removed(main_window):
    """
    UI condensation: the redundant Run All button must stay removed, primary
    simulation inputs must remain grouped at the top of the panel, and the run
    action must live alongside the batch-set actions instead of in a separate
    control strip.
    """
    from PySide6 import QtWidgets

    assert main_window.findChild(QtWidgets.QPushButton, "runAllSimulationsButton") is None

    run_btn = main_window.findChild(QtWidgets.QPushButton, "runSelectedButton")
    assert run_btn is not None

    controls_row = main_window.findChild(QtWidgets.QWidget, "batchSolverControlsRow")
    assert controls_row is not None
    layout = controls_row.layout()
    assert isinstance(layout, QtWidgets.QVBoxLayout)
    assert layout.count() == 1

    inputs_row = layout.itemAt(0).layout()
    assert isinstance(inputs_row, QtWidgets.QHBoxLayout)

    sim_time = getattr(main_window, "_sim_time_spinbox", None)
    assert isinstance(sim_time, QtWidgets.QLineEdit)
    sim_time.setText("1e12")
    assert sim_time.text() == "1e12"

    input_widgets = []
    for i in range(inputs_row.count()):
        item = inputs_row.itemAt(i)
        if item is None:
            continue
        w = item.widget()
        if w is not None:
            input_widgets.append(w)

    assert getattr(main_window, "_solver_method_combo", None) in input_widgets
    assert getattr(main_window, "_sim_time_spinbox", None) in input_widgets
    assert getattr(main_window, "_num_points_spinbox", None) in input_widgets

    delete_btn = main_window.findChild(QtWidgets.QPushButton, "deleteBatchSetButton")
    assert delete_btn is not None
    main_window.show()
    QtWidgets.QApplication.processEvents()
    assert run_btn.geometry().y() == delete_btn.geometry().y()
    assert run_btn.geometry().x() > delete_btn.geometry().x()


@pytest.mark.gui
def test_add_set_creates_unique_names_and_zeros(main_window):
    from PySide6 import QtWidgets

    # Ensure visible species columns exist.
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None

    add_btn.click()
    add_btn.click()

    names = main_window._batch_store.set_names()
    assert "set2" in names
    assert "set3" in names

    row2 = main_window._batch_store.row_for_set("set2")
    assert row2 is not None
    assert float(main_window._batch_store.get_value(int(row2), "A")) == pytest.approx(0.0)
    assert float(main_window._batch_store.get_value(int(row2), "B")) == pytest.approx(0.0)


@pytest.mark.gui
def test_move_up_down_reorders_sets_and_serializes(main_window, qt_app):
    from PySide6 import QtCore, QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    up_btn = main_window.findChild(QtWidgets.QPushButton, "moveBatchSetUpButton")
    down_btn = main_window.findChild(QtWidgets.QPushButton, "moveBatchSetDownButton")
    assert add_btn is not None and up_btn is not None and down_btn is not None

    # Create set2 and set3.
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()
    assert main_window._batch_store.set_names()[:3] == ["set1", "set2", "set3"]

    # Select set3 row and move it up.
    table = main_window._batch_table
    assert table is not None
    idx = main_window._batch_model.index(2, 0)
    table.setCurrentIndex(idx)
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    up_btn.click()
    qt_app.processEvents()
    assert main_window._batch_store.set_names()[:3] == ["set1", "set3", "set2"]
    payload = main_window._serialize_project_state()
    assert [s["name"] for s in payload["batch_initial_conditions"]["sets"]][:3] == ["set1", "set3", "set2"]

    # Move it back down.
    down_btn.click()
    qt_app.processEvents()
    assert main_window._batch_store.set_names()[:3] == ["set1", "set2", "set3"]


@pytest.mark.gui
def test_move_reorders_cached_main_plot_popup_labels_for_duplicate_names(main_window, qt_app):
    from PySide6 import QtCore, QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    assert first_id and second_id

    main_window._batch_store.set_set_name(0, "dup")
    main_window._batch_store.set_set_name(1, "dup")

    cache = main_window.simulation_controller.batch_cache
    cache.active_batch_set_id = first_id
    cache.active_batch_set = "dup"
    cache.last_display_selection = [first_id, second_id]

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "dup",
                "set_id": second_id,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    main_window.sync_main_plot_copy_labels(first_id, [first_id, second_id])

    plot = main_window._plot_tabs._main_plot
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup (row 1)"
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert overlays
    assert overlays[0]["popup_label"] == "dup (row 2)"

    table = main_window._batch_table
    assert table is not None
    idx = main_window._batch_model.index(1, 0)
    table.setCurrentIndex(idx)
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    main_window._move_selected_batch_sets(delta=-1)
    qt_app.processEvents()

    assert main_window._batch_store.set_names()[:2] == ["dup", "dup"]
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup (row 2)"
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert overlays
    assert overlays[0]["popup_label"] == "dup (row 1)"


@pytest.mark.gui
def test_move_skips_main_plot_popup_resync_for_direct_path_plot(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    assert first_id and second_id

    cache = main_window.simulation_controller.batch_cache
    cache.active_batch_set_id = first_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(first_id) or first_id)
    cache.last_display_selection = [first_id, second_id]

    monkeypatch.setattr(main_window, "display_cached_batch_selection", lambda **_kwargs: False, raising=False)

    main_window.simulation_controller.run_state.latest_sim_request_id = 44
    main_window.simulation_controller.run_state.active_run_id = 44
    main_window.simulation_controller.on_simulation_complete(
        {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.asarray([[2.0, 4.0]], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "reaction: A -> B; k1=1.0",
            "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15},
            "fallback_occurred": False,
            "fallback_message": None,
        },
        run_id=44,
        fast_mode=False,
        request_id=44,
    )
    qt_app.processEvents()

    assert main_window.active_batch_selection() == ("", "")
    assert cache.last_display_selection == []

    sync_calls = []

    def _sync(*args, **kwargs):
        sync_calls.append((args, kwargs))

    monkeypatch.setattr(main_window, "sync_main_plot_copy_labels", _sync, raising=False)

    table = main_window._batch_table
    assert table is not None
    idx = main_window._batch_model.index(1, 0)
    table.setCurrentIndex(idx)
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    main_window._move_selected_batch_sets(delta=-1)
    qt_app.processEvents()

    assert sync_calls == []


@pytest.mark.gui
def test_move_relabels_overlays_when_primary_is_empty_but_display_selection_survives(main_window, qt_app):
    from PySide6 import QtCore, QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    assert first_id and second_id

    main_window._batch_store.set_set_name(0, "dup")
    main_window._batch_store.set_set_name(1, "dup")

    cache = main_window.simulation_controller.batch_cache
    cache.active_batch_set_id = ""
    cache.active_batch_set = ""
    cache.last_display_selection = [first_id, second_id]

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "dup",
                "set_id": second_id,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    main_window.sync_main_plot_copy_labels("", [first_id, second_id])

    plot = main_window._plot_tabs._main_plot
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert overlays
    assert overlays[0]["popup_label"] == "dup (row 2)"

    table = main_window._batch_table
    assert table is not None
    idx = main_window._batch_model.index(1, 0)
    table.setCurrentIndex(idx)
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    main_window._move_selected_batch_sets(delta=-1)
    qt_app.processEvents()

    assert main_window._batch_store.set_names()[:2] == ["dup", "dup"]
    # Without a live active set_id, the primary cannot resolve to a truthful row.
    # Row-qualified disambiguation only applies to entries that still map to a batch row.
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup"
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert overlays
    assert overlays[0]["popup_label"] == "dup (row 1)"


@pytest.mark.gui
def test_build_copy_all_export_plan_uses_live_overlay_for_clean_evicted_visible_set(main_window, qt_app):
    from PySide6 import QtWidgets

    from kindred.core.batch_simulation_cache import BatchSimulationCache

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    first_name = str(main_window.batch_set_name_for_id(first_id) or "")
    second_name = str(main_window.batch_set_name_for_id(second_id) or "")
    assert first_id and second_id

    cache = main_window.simulation_controller.batch_cache
    cache_key = "copy-all-live-overlay-fallback"
    cache.active_cache_key = cache_key
    cache.active_cache_valid_set_ids = (first_id, second_id)
    cache.active_cache_invalidated_set_ids = None
    cache.active_batch_set_id = first_id
    cache.active_batch_set = first_name
    cache.last_display_selection = [first_id, second_id]
    cache.result_cache[BatchSimulationCache.entry_key(cache_key, first_id)] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([1.0, 2.0], dtype=float)},
    }

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label=first_name,
        overlays=[
            {
                "label": second_name,
                "set_id": second_id,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    main_window.sync_main_plot_copy_labels(first_id, [first_id, second_id])

    plan = main_window._build_main_plot_copy_all_export_plan()

    assert [block.set_id for block in plan.shown_blocks] == [first_id, second_id]
    assert plan.missing_items == []
    np.testing.assert_allclose(plan.shown_blocks[1].series["A"], np.asarray([3.0, 4.0], dtype=float))


@pytest.mark.gui
def test_batch_set_rename_resyncs_cached_main_plot_popup_labels(main_window, qt_app):
    from PySide6 import QtCore, QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    assert first_id and second_id

    assert main_window._batch_model.setData(main_window._batch_model.index(0, 0), "dup", QtCore.Qt.EditRole)
    assert main_window._batch_model.setData(main_window._batch_model.index(1, 0), "set2", QtCore.Qt.EditRole)

    cache = main_window.simulation_controller.batch_cache
    cache.active_batch_set_id = first_id
    cache.active_batch_set = "dup"
    cache.last_display_selection = [first_id, second_id]

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "set2",
                "set_id": second_id,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    main_window.sync_main_plot_copy_labels(first_id, [first_id, second_id])

    plot = main_window._plot_tabs._main_plot
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup"
    assert overlays[0].get("popup_label") == "set2"

    assert main_window._batch_model.setData(main_window._batch_model.index(1, 0), "dup", QtCore.Qt.EditRole)
    qt_app.processEvents()

    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup (row 1)"
    assert overlays[0]["popup_label"] == "dup (row 2)"


@pytest.mark.gui
def test_batch_table_paste_rename_resyncs_cached_main_plot_popup_labels(main_window, qt_app):
    from PySide6 import QtWidgets

    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    first_id = str(main_window._batch_set_id_for_row(0) or "")
    second_id = str(main_window._batch_set_id_for_row(1) or "")
    assert first_id and second_id

    main_window._batch_store.set_set_name(0, "dup")
    main_window._batch_store.set_set_name(1, "set2")
    main_window._batch_store.set_value(1, "A", 1.0)

    cache = main_window.simulation_controller.batch_cache
    cache.active_batch_set_id = first_id
    cache.active_batch_set = "dup"
    cache.last_display_selection = [first_id, second_id]

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "set2",
                "set_id": second_id,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    main_window.sync_main_plot_copy_labels(first_id, [first_id, second_id])

    plot = main_window._plot_tabs._main_plot
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup"
    assert overlays[0].get("popup_label") == "set2"

    table = getattr(main_window, "_batch_table", None)
    assert table is not None
    table.setCurrentIndex(main_window._batch_model.index(1, 0))

    clipboard = QtWidgets.QApplication.clipboard()
    assert clipboard is not None
    clipboard.setText("dup\t1.0")
    table._handle_paste()
    qt_app.processEvents()

    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert getattr(plot, "_simulation_set_popup_label", None) == "dup (row 1)"
    assert overlays[0]["popup_label"] == "dup (row 2)"


@pytest.mark.gui
def test_global_fit_creates_and_seeds_new_batch_set_from_dataset_t0(main_window, monkeypatch):
    """
    Regression: starting a global fit should prompt for an unmapped dataset and,
    by default, create a new batch set named after the dataset base label.
    If the dataset starts at t≈0, seed initials from row 0.
    """
    from PySide6 import QtWidgets

    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    data_panel._datasets["dataset1_1"] = {
        "t": np.array([0.0, 1.0]),
        "species": {"A": np.array([1.23, 0.5])},
    }

    monkeypatch.setattr(
        type(main_window),
        "_extract_mechanism_initials",
        lambda _self, _dsl: {"A": 0.0, "B": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = kwargs

        def setWindowTitle(self, *_):
            pass

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._run_global_fit()

    row = main_window._batch_store.row_for_set("dataset1")
    assert row is not None
    assert float(main_window._batch_store.get_value(int(row), "A")) == pytest.approx(1.23)
    assert float(main_window._batch_store.get_value(int(row), "B")) == pytest.approx(0.0)

    settings = main_window._dataset_manager.get_fit_settings("dataset1_1")
    assert settings.batch_set == "dataset1"
    assert "simulation_func" in (captured.get("kwargs") or {})


@pytest.mark.gui
def test_global_fit_creates_new_batch_set_without_seeding_when_t0_not_zero(main_window, monkeypatch):
    """If t0 is not within tolerance, do not seed; default path creates zeros and proceeds."""
    from PySide6 import QtWidgets

    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    data_panel._datasets["dataset2"] = {
        "t": np.array([1e-4, 1.0]),
        "species": {"A": np.array([9.0, 8.0])},
    }

    monkeypatch.setattr(
        type(main_window),
        "_extract_mechanism_initials",
        lambda _self, _dsl: {"A": 0.0, "B": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", lambda *a, **k: QtWidgets.QDialog())

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._run_global_fit()

    row = main_window._batch_store.row_for_set("dataset2")
    assert row is not None
    assert float(main_window._batch_store.get_value(int(row), "A")) == pytest.approx(0.0)
    assert float(main_window._batch_store.get_value(int(row), "B")) == pytest.approx(0.0)


@pytest.mark.gui
def test_global_fit_resolves_each_unmapped_dataset_independently(main_window, monkeypatch):
    """
    Regression: when starting a global fit with multiple datasets, each dataset that
    lacks a saved batch mapping must be resolved independently. It must not silently
    reuse the first dataset's batch set for subsequent datasets.
    """
    from PySide6 import QtWidgets

    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    data_panel._datasets.update(
        {
            "dataset_01.csv": {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.11, 0.5])}},
            "dataset_02.csv": {"t": np.array([0.0, 1.0]), "species": {"A": np.array([2.22, 0.5])}},
            "dataset_03.csv": {"t": np.array([1e-4, 1.0]), "species": {"A": np.array([3.33, 0.5])}},
        }
    )

    monkeypatch.setattr(
        type(main_window),
        "_extract_mechanism_initials",
        lambda _self, _dsl: {"A": 0.0, "B": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", lambda *a, **k: QtWidgets.QDialog())

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._run_global_fit()

    # Each dataset should get its own created set by default.
    for name in ("dataset_01", "dataset_02", "dataset_03"):
        assert main_window._batch_store.row_for_set(name) is not None

    row1 = int(main_window._batch_store.row_for_set("dataset_01"))
    row2 = int(main_window._batch_store.row_for_set("dataset_02"))
    row3 = int(main_window._batch_store.row_for_set("dataset_03"))
    assert float(main_window._batch_store.get_value(row1, "A")) == pytest.approx(1.11)
    assert float(main_window._batch_store.get_value(row2, "A")) == pytest.approx(2.22)
    assert float(main_window._batch_store.get_value(row3, "A")) == pytest.approx(0.0)  # t0 not within tol

    s1 = main_window._dataset_manager.get_fit_settings("dataset_01.csv")
    s2 = main_window._dataset_manager.get_fit_settings("dataset_02.csv")
    s3 = main_window._dataset_manager.get_fit_settings("dataset_03.csv")
    assert s1.batch_set == "dataset_01"
    assert s2.batch_set == "dataset_02"
    assert s3.batch_set == "dataset_03"
