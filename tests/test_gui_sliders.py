from __future__ import annotations

import builtins
import functools
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_initial_conditions import migrate_reaction_dsl_initial_concentrations
import kindred.core.simulator.dsl as dsl
from kindred.core.simulation_preparation import BoundMechanism
from kindred.gui.controllers.simulation_controller import build_fallback_cache_key
from tests.worker_stubs import make_simulation_worker_stub

pytestmark = [pytest.mark.gui, pytest.mark.slow]

def _worker_payload(prepared, mechanism_text):
    t = np.linspace(0.0, 1.0, 4)
    y = np.vstack(
        [
            np.linspace(1.0, 0.4, t.size),
            np.linspace(0.0, 0.6, t.size),
        ]
    )
    return {
        "t": t,
        "Y": y,
        "species_names": ["A", "B"],
        "mechanism": prepared.get("mechanism") if prepared else None,
        "mechanism_text": mechanism_text,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

def _find_slider_visibility_action(main_window, entry_kind: str, name: str):
    picker = main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton")
    assert picker is not None
    menu = picker.menu()
    assert menu is not None
    main_window._mechanism_editor._rebuild_slider_visibility_menu()
    for action in menu.actions():
        if action.data() == (str(entry_kind), str(name)):
            return action
    raise AssertionError(f"Missing visibility action for {(entry_kind, name)!r}")

def _block_insert_dialog_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kindred.gui.widgets.insert_dialog":
            raise AssertionError("preset/example loads should not import insert_dialog")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

def _set_slider_visibility(main_window, entry_kind: str, name: str, *, visible: bool) -> None:
    action = _find_slider_visibility_action(main_window, entry_kind, name)
    if bool(action.isChecked()) != bool(visible):
        action.trigger()
        QtWidgets.QApplication.processEvents()
    assert bool(action.isChecked()) is bool(visible)

def _select_batch_rows(main_window, rows: list[int]) -> None:
    table = main_window._batch_table
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    table.setCurrentIndex(main_window._batch_model.index(int(rows[0]), 0))
    for row in rows:
        idx = main_window._batch_model.index(int(row), 0)
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._refresh_batch_display_from_focus_and_shown()

def _set_shown_rows(main_window, rows: list[int]) -> None:
    model = main_window._batch_model
    shown_rows = {int(row) for row in rows}
    for row in range(model.rowCount()):
        model.set_row_shown(row, row in shown_rows)
        expected = QtCore.Qt.Checked if row in shown_rows else QtCore.Qt.Unchecked
        assert model.data(model.index(row, model.show_column()), QtCore.Qt.CheckStateRole) == expected
    main_window._refresh_batch_display_from_focus_and_shown()

def _set_edit_target_rows(main_window, rows: list[int]) -> None:
    set_ids = [str(main_window._batch_set_id_for_row(int(row)) or "") for row in rows]
    main_window.set_slider_edit_target_set_ids([set_id for set_id in set_ids if set_id])

def _parameter_table_numeric_value(main_window, name: str) -> float:
    table = main_window.main_plot().parameter_table()
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None or item.text() != str(name):
            continue
        value_item = table.item(row, 1)
        assert value_item is not None
        return float(value_item.text())
    raise AssertionError(f"Missing parameter-table row for {name!r}")

def _assert_selection_plot_cleared(main_window) -> None:
    plot = main_window._plot_tabs._main_plot
    assert getattr(plot, "_t", None) is None
    assert dict(getattr(plot, "_series", {}) or {}) == {}

def _current_preview_time_axis(main_window) -> np.ndarray:
    selected_ids = [str(set_id) for set_id in (main_window._batch_set_ids_for_scope("selected") or ()) if str(set_id)]
    target_set_id = selected_ids[0] if selected_ids else str(main_window._preview_session.focused_mechanism_workspace_set_id() or "")
    assert target_set_id
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=target_set_id)
    solver_config, t_end, _ = main_window._current_workspace_preview_context(
        set_id=target_set_id,
        mechanism_text=mechanism_text,
    )
    grid_n = int((solver_config.get("grid") or {}).get("N") or 0)
    return np.linspace(0.0, float(t_end), max(2, grid_n), dtype=float)

def test_slider_changes_use_precompiled_rhs(main_window, monkeypatch):
    """Repeated slider runs should reuse a compiled mechanism instead of re-parsing."""
    parse_calls = 0
    original_parse = dsl.parse_dsl_to_mechanism

    def _counting_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(dsl, "parse_dsl_to_mechanism", _counting_parse)
    monkeypatch.setattr(main_window, "_extract_and_populate_variables", lambda: None)

    prepared_payloads = []

    def _on_init(worker) -> None:
        prepared_payloads.append(worker._prepared)

    def _payload(worker) -> dict:
        return _worker_payload(worker._prepared or {}, worker._mechanism_text)

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_init=_on_init, payload_factory=_payload, emit_progress=(100, "done")),
    )

    main_window._use_sparse_jacobian = False
    main_window._wegscheider_cyclicity_enabled = False

    # Use a DSL without `initial:` lines to avoid the init-migration path
    # rewriting the mechanism text mid-run and invalidating the slider cache.
    main_window._mechanism_editor._reactions_text.setPlainText(
        "A -> B ; k=1.0"
    )
    main_window._preview_session.stage_slider_value("k1", 1.0)

    main_window.simulation_controller.run_simulation_from_slider()
    QtWidgets.QApplication.processEvents()
    parse_after_first = parse_calls
    main_window.simulation_controller.run_simulation_from_slider()
    QtWidgets.QApplication.processEvents()

    assert parse_after_first >= 1, "first slider run should parse the DSL"
    assert parse_calls == parse_after_first, "second slider run should reuse the cached mechanism"
    assert len(prepared_payloads) == 2
    assert all(payload is not None for payload in prepared_payloads)

def test_slider_binding_updates_across_changes(main_window, monkeypatch):
    """Slider-driven runs should reflect the latest parameter values without rebuilds."""
    monkeypatch.setattr(main_window, "_extract_and_populate_variables", lambda: None)

    seen_rates: list[float] = []

    def _on_start(worker) -> None:
        prepared = worker._prepared or {}
        mechanism = prepared.get("mechanism")
        if mechanism and mechanism.reactions:
            rate_obj = mechanism.reactions[0].rate
            try:
                seen_rates.append(float(rate_obj()))
            except Exception:
                seen_rates.append(float("nan"))

    def _payload(worker) -> dict:
        prepared = worker._prepared or {}
        return _worker_payload(prepared, worker._mechanism_text)

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_start=_on_start, payload_factory=_payload),
    )

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )

    for value in (1.0, 0.45, 0.12):
        main_window._preview_session.stage_slider_value("k1", value)
        main_window.simulation_controller.run_simulation_from_slider()
        QtWidgets.QApplication.processEvents()

    assert seen_rates == [1.0, 0.45, 0.12]

def test_slider_move_triggers_fresh_simulation(main_window, qtbot, monkeypatch):
    """Moving a slider should trigger a new simulation using the updated parameter."""
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )

    # Extract sliders and cache overrides for the initial mechanism
    main_window._extract_and_populate_variables()

    # Spy on slider-triggered simulation calls
    started_with: list[str] = []
    seen_rates: list[float] = []

    def _on_start(worker) -> None:
        started_with.append(worker._mechanism_text)
        prepared = worker._prepared or {}
        mechanism = prepared.get("mechanism")
        if mechanism and getattr(mechanism, "reactions", None):
            rate_obj = mechanism.reactions[0].rate
            try:
                seen_rates.append(float(rate_obj()))
            except Exception:
                seen_rates.append(float("nan"))

    def _payload(worker) -> dict:
        payload = worker._prepared or {}
        mech = payload.get("mechanism")
        species_names = payload.get("species_names", ["A", "B"])
        y0 = payload.get("y0")
        if mech is None:
            t = np.array([0.0, 1.0])
            Y = np.vstack([np.linspace(1.0, 0.5, t.size), np.linspace(0.0, 0.5, t.size)])
        else:
            t = np.array([0.0, 1.0])
            Y = np.vstack([y0, y0]) if y0 is not None else np.zeros((len(species_names), t.size))

        return {
            "t": t,
            "Y": Y,
            "species_names": species_names,
            "mechanism": mech,
            "mechanism_text": worker._mechanism_text,
            "solver_config": worker._solver_config,
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_start=_on_start, payload_factory=_payload),
    )

    # Grab the slider widget for k1 and move it to a new value
    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    slider_widget = sliders._sliders["k1"]

    target_pos = sliders._value_to_slider_pos("k1", 2.0)
    slider_widget.setValue(target_pos)

    qtbot.waitUntil(lambda: len(started_with) == 1, timeout=1000)
    expected_override = float(main_window.slider_overrides().get("k1", 0.0))

    # In override mode, the editor DSL stays baseline, but the worker receives an override-rewritten DSL.
    assert any("k=2" in text for text in started_with)
    assert seen_rates and seen_rates[0] == pytest.approx(expected_override)

def test_slider_change_triggers_run(main_window, qtbot, monkeypatch):
    """End-to-end: slider move should schedule a fast simulation run."""
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    calls = []

    def _fake_run(fast_mode=False, **_kwargs):
        calls.append(fast_mode)

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_internal", _fake_run)

    sliders = main_window._mechanism_editor._variable_sliders
    slider_widget = sliders._sliders["k1"]
    slider_widget.setValue(sliders._value_to_slider_pos("k1", 2.0))

    qtbot.waitUntil(lambda: len(calls) == 1, timeout=1000)
    assert calls == [True]
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text

def test_apply_overrides_to_text_reads_metadata_once_per_call(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "reaction: B -> C; k=2.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window.set_variable_metadata(
        {
            "k1": {"type": "reaction", "role": "k"},
            "k2": {"type": "reaction", "role": "k"},
        }
    )
    main_window._preview_session.stage_slider_value("k1", 3.0)
    main_window._preview_session.stage_slider_value("k2", 4.0)

    original_variable_metadata = main_window.variable_metadata
    metadata_calls = 0

    def _counting_variable_metadata():
        nonlocal metadata_calls
        metadata_calls += 1
        return original_variable_metadata()

    monkeypatch.setattr(main_window, "variable_metadata", _counting_variable_metadata)

    updated = main_window._apply_overrides_to_text(
        main_window._mechanism_editor._reactions_text.toPlainText()
    )

    assert metadata_calls == 1
    assert "k=3" in updated
    assert "k=4" in updated

def test_scalar_param_slider_updates_existing_param_in_reactions_algebra(main_window, monkeypatch):
    """
    Regression: scalar slider commits must update an existing `param a = ...`
    definition in the Reactions editor, instead of appending a second
    `param a = ...` line into the Notes tab (which is never parsed).
    """
    from kindred.core.simulator.parameter_algebra import parse_parameter_algebra_spec_from_dsl_text

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "# Algebra",
                "param a = 5  # base scalar",
                "param kf1 = a*k2",
            ]
        )
        + "\n"
    )
    main_window._mechanism_editor._notes_text.setPlainText("keep this note\n")
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    # Mark `a` as a scalar so the slider commit path routes through `_update_scalar_param_in_algebra`.
    main_window.set_variable_metadata({"a": {"type": "scalar"}})

    main_window._commit_slider_value("a", 0.331131)

    # Notes are not a DSL source-of-truth and must not be modified by slider commits.
    assert main_window._mechanism_editor._notes_text.toPlainText() == "keep this note\n"

    # The original inline comment should be preserved in-place.
    reactions_text = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "# base scalar" in reactions_text

    # The effective DSL (built the same way as production) must contain exactly one `param a = ...`.
    full_dsl = main_window._get_mechanism_text()
    from kindred.core.simulator.parameter_namespace import build_flat_compat_namespace

    spec = parse_parameter_algebra_spec_from_dsl_text(
        full_dsl,
        mechanism_namespace=build_flat_compat_namespace({"k2", "kf1"}),
    )
    a_statements = [stmt for stmt in spec.param_statements if stmt.name == "a"]
    assert len(a_statements) == 1
    assert float(a_statements[0].expr_src) == pytest.approx(0.331131)

def test_scalar_param_slider_commit_uses_authoritative_parameter_precision(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "# Algebra",
                "param a = 5",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
    main_window.set_variable_metadata({"a": {"type": "scalar"}})

    main_window._commit_slider_value("a", 1000000.1234567)

    reactions_text = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "param a = 1000000.1234567" in reactions_text
    assert "param a = 1e+06" not in reactions_text

def test_scalar_param_slider_updates_existing_param_without_header(main_window, monkeypatch):
    from kindred.core.simulator.parameter_algebra import parse_parameter_algebra_spec_from_dsl_text
    from kindred.core.simulator.parameter_namespace import build_flat_compat_namespace

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "param a = 5  # base scalar",
                "param kf1 = a*k2",
            ]
        )
        + "\n"
    )
    main_window._mechanism_editor._notes_text.setPlainText("keep this note\n")
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
    main_window.set_variable_metadata({"a": {"type": "scalar"}})

    main_window._commit_slider_value("a", 0.331131)

    reactions_text = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "# Algebra" not in reactions_text
    assert reactions_text.count("param a =") == 1
    assert "# base scalar" in reactions_text
    assert main_window._mechanism_editor._notes_text.toPlainText() == "keep this note\n"

    spec = parse_parameter_algebra_spec_from_dsl_text(
        main_window._get_mechanism_text(),
        mechanism_namespace=build_flat_compat_namespace({"k2", "kf1"}),
    )
    a_statements = [stmt for stmt in spec.param_statements if stmt.name == "a"]
    assert len(a_statements) == 1
    assert float(a_statements[0].expr_src) == pytest.approx(0.331131)

def test_slider_release_does_not_commit_dsl_in_override_mode(main_window, qtbot, monkeypatch):
    """
    Override mode: releasing a slider must not rewrite the DSL automatically.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kf=0.0928966, K=0.00963829\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("Keq1")

    # Prevent background simulation from running in this test.
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    commits: list[tuple[str, float]] = []

    def _spy_commit(name: str, value: float) -> None:
        commits.append((str(name), float(value)))

    monkeypatch.setattr(main_window, "_commit_slider_value", _spy_commit)

    # Simulate a drag gesture and release.
    preview = main_window._preview_session
    main_window._on_slider_drag_started("Keq1")
    slider_widget = sliders._sliders["Keq1"]
    slider_widget.setValue(min(slider_widget.maximum(), slider_widget.value() + 50))

    qtbot.waitUntil(lambda: "Keq1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("Keq1")
    qtbot.waitUntil(lambda: not bool(getattr(preview, "_slider_release_in_progress", False)), timeout=1500)

    assert commits == []
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text

def test_direct_commit_slider_value_materializes_authoritative_text_and_clears_workspace(
    main_window,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_from_slider",
        lambda: (_ for _ in ()).throw(AssertionError("direct authoritative slider commit queued preview rerun")),
    )

    preview = main_window._preview_session
    focused_set_id = str(preview.focused_mechanism_workspace_set_id() or "")
    assert focused_set_id

    main_window._commit_slider_value("k1", 2.0)
    QtWidgets.QApplication.processEvents()

    assert "k=2" in main_window._mechanism_editor._reactions_text.toPlainText().replace(" ", "")
    assert preview.effective_slider_values_for_set(focused_set_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(focused_set_id) == {}
    assert preview.has_local_mechanism_workspaces() is False
    assert preview.slider_overrides() == {}
    assert main_window._mechanism_editor._variable_sliders.get_variables()["k1"] == pytest.approx(2.0)
    assert main_window.simulation_controller.run_state.pending_slider_simulation is False

def test_direct_commit_slider_value_preserves_other_focused_workspace_overrides(
    main_window,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "reaction: B -> C; k=4.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_from_slider",
        lambda: (_ for _ in ()).throw(AssertionError("direct authoritative slider commit queued preview rerun")),
    )

    preview = main_window._preview_session
    focused_set_id = str(preview.focused_mechanism_workspace_set_id() or "")
    assert focused_set_id

    preview.stage_slider_value("k2", 3.0)

    main_window._commit_slider_value("k1", 2.0)
    QtWidgets.QApplication.processEvents()

    reactions_text = main_window._mechanism_editor._reactions_text.toPlainText().replace(" ", "")
    assert "reaction:A->B;k=2" in reactions_text
    assert "reaction:B->C;k=3" in reactions_text
    assert preview.effective_slider_values_for_set(focused_set_id) == {
        "k1": pytest.approx(2.0),
        "k2": pytest.approx(3.0),
    }
    assert preview.local_mechanism_workspace(focused_set_id) == {}
    assert preview.has_local_mechanism_workspaces() is False

def test_direct_scalar_commit_with_colliding_name_preserves_workspace_without_rewriting_state_network(
    main_window,
    monkeypatch,
):
    reactions_before = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Algebra",
            "param energy = 5",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    state_before = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )
    main_window._mechanism_editor._reactions_text.setPlainText(reactions_before)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(state_before)
    main_window._extract_and_populate_variables()
    main_window.set_variable_metadata(
        {
            "k1": {"type": "reaction", "role": "k"},
            "energy": {"type": "scalar", "role": "scalar"},
        }
    )

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_from_slider",
        lambda: (_ for _ in ()).throw(AssertionError("direct authoritative slider commit queued preview rerun")),
    )

    preview = main_window._preview_session
    focused_set_id = str(preview.focused_mechanism_workspace_set_id() or "")
    assert focused_set_id

    preview.stage_slider_value("k1", 2.0)

    main_window._commit_slider_value("energy", 7.0)
    QtWidgets.QApplication.processEvents()

    reactions_text = main_window._mechanism_editor._reactions_text.toPlainText().replace(" ", "")
    assert "reaction:A->B;k=2" in reactions_text
    assert "paramenergy=7" in reactions_text
    assert main_window.mechanism_state_network_dsl_raw() == state_before
    assert preview.effective_slider_values_for_set(focused_set_id) == {
        "energy": pytest.approx(7.0),
        "k1": pytest.approx(2.0),
    }
    assert preview.local_mechanism_workspace(focused_set_id) == {}
    assert preview.has_local_mechanism_workspaces() is False

def test_authoritative_mechanism_editor_write_undo_reverts_reactions_and_state_network_together(
    main_window,
):
    reactions_before = "T=200.0\n# Algebra\nparam a = 5\n"
    state_before = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )
    reactions_after = "T=200.0\n# Algebra\nparam a = 7\n"
    state_after = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=-5, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(reactions_before)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(state_before)

    main_window._set_authoritative_mechanism_editor_texts(
        reactions_text=reactions_after,
        state_network_dsl=state_after,
        description="Commit authoritative editors",
    )

    assert main_window.mechanism_reactions_text_raw() == reactions_after
    assert main_window.mechanism_state_network_dsl_raw() == state_after

    main_window._undo_stack.undo()
    QtWidgets.QApplication.processEvents()

    assert main_window.mechanism_reactions_text_raw() == reactions_before
    assert main_window.mechanism_state_network_dsl_raw() == state_before

def test_authoritative_mechanism_editor_write_undo_reverts_state_network_only_change(
    main_window,
):
    reactions_text = "T=200.0\n"
    state_before = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )
    state_after = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=-5, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(reactions_text)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(state_before)

    main_window._set_authoritative_mechanism_editor_texts(
        reactions_text=reactions_text,
        state_network_dsl=state_after,
        description="Commit state-network editor",
    )

    assert main_window.mechanism_state_network_dsl_raw() == state_after

    main_window._undo_stack.undo()
    QtWidgets.QApplication.processEvents()

    assert main_window.mechanism_reactions_text_raw() == reactions_text
    assert main_window.mechanism_state_network_dsl_raw() == state_before

def test_slider_apply_button_shows_apply_text(main_window):
    """The slider confirmation button must display 'Apply' (not the former 'Commit')."""
    btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    assert btn is not None, "commitSliderOverridesButton not found"
    assert btn.text() == "Apply"

def test_commit_and_reset_buttons_control_dsl_overrides(main_window, qtbot, monkeypatch):
    """
    Override mode UX: sliders change overrides for preview runs; Commit writes to DSL; Reset snaps back.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    # Avoid background simulation from running in this test.
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    slider_widget = sliders._sliders["k1"]
    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False
    slider_widget.setValue(sliders._value_to_slider_pos("k1", 2.0))

    qtbot.waitUntil(lambda: float(main_window.slider_overrides().get("k1", 0.0)) > 1.5, timeout=1000)
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    # Commit writes overrides to the DSL editor.
    commit_btn.click()
    QtWidgets.QApplication.processEvents()
    assert "k=2" in main_window._mechanism_editor._reactions_text.toPlainText().replace(" ", "")
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

    # Reset snaps overrides back to the (now committed) baseline DSL.
    slider_widget.setValue(sliders._value_to_slider_pos("k1", 3.0))
    qtbot.waitUntil(lambda: float(main_window.slider_overrides().get("k1", 0.0)) > 2.5, timeout=1000)
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True
    reset_btn.click()
    QtWidgets.QApplication.processEvents()
    assert float(main_window.slider_overrides().get("k1", 0.0)) == pytest.approx(0.0)
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

def test_commit_and_reset_preserve_mixed_visible_slider_selection(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "0.0")
    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(model.index(0, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    mech_slider = sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    assert mech_slider.isVisible() is False
    assert species_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "k1", visible=True)
    _set_slider_visibility(main_window, "species", "A", visible=True)

    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None

    mech_slider = main_window._mechanism_editor._variable_sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    assert mech_slider.isVisible() is True
    assert species_slider.isVisible() is True

    mech_slider.setValue(sliders._value_to_slider_pos("k1", 2.0))
    qtbot.waitUntil(lambda: float(main_window.slider_overrides().get("k1", 0.0)) > 1.5, timeout=1000)

    commit_btn.click()
    QtWidgets.QApplication.processEvents()

    assert main_window._mechanism_editor._variable_sliders.variable_visible("k1") is True
    assert main_window._mechanism_editor.species_sliders_widget().species_visible("A") is True
    mech_slider = main_window._mechanism_editor._variable_sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    assert mech_slider.isVisible() is True
    assert species_slider.isVisible() is True

    mech_slider.setValue(main_window._mechanism_editor._variable_sliders._value_to_slider_pos("k1", 3.0))
    qtbot.waitUntil(lambda: float(main_window.slider_overrides().get("k1", 0.0)) > 2.5, timeout=1000)

    reset_btn.click()
    QtWidgets.QApplication.processEvents()

    assert main_window._mechanism_editor._variable_sliders.variable_visible("k1") is True
    assert main_window._mechanism_editor.species_sliders_widget().species_visible("A") is True
    mech_slider = main_window._mechanism_editor._variable_sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    assert mech_slider.isVisible() is True
    assert species_slider.isVisible() is True

def test_unified_slider_surface_picker_hides_dirty_mechanism_without_clearing_transaction(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "0.0")
    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(model.index(0, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    picker = main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton")
    assert picker is not None

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    mech_slider = sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert species_slider is not None
    assert mech_slider.isVisible() is False
    assert species_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "k1", visible=True)
    assert mech_slider.isVisible() is True

    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None
    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    mech_slider.setValue(sliders._value_to_slider_pos("k1", 2.0))
    qtbot.waitUntil(lambda: float(main_window.slider_overrides().get("k1", 0.0)) > 1.5, timeout=1000)

    assert main_window._preview_session.has_dirty_transaction() is True
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text

    _set_slider_visibility(main_window, "mechanism", "k1", visible=False)
    assert mech_slider.isVisible() is False
    assert float(main_window.slider_overrides().get("k1", 0.0)) > 1.5
    assert main_window._preview_session.has_dirty_transaction() is True
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text

def test_hidden_mechanism_slider_does_not_leak_into_later_load_with_reused_name(
    main_window, qtbot, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    first_slider = sliders._sliders["k1"]
    assert first_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "k1", visible=False)
    assert first_slider.isVisible() is False

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: C -> D; k=4.0\ninitial: C=3.0\ninitial: D=0.0"
    )
    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(3.0, 2.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": ["C", "D"],
        "mechanism": None,
        "mechanism_text": main_window._mechanism_editor._reactions_text.toPlainText(),
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

    main_window._preview_session._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    assert sliders.has_variable("k1")
    reloaded_slider = sliders._sliders["k1"]
    assert reloaded_slider.isVisible() is False
    assert sliders.variable_visible("k1") is False
    action = _find_slider_visibility_action(main_window, "mechanism", "k1")
    assert action.isChecked() is False

def test_hidden_mechanism_slider_persists_across_same_universe_dsl_text_edit_completion_refresh(
    main_window, qtbot, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    first_slider = sliders._sliders["k1"]
    assert first_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "k1", visible=False)
    assert first_slider.isVisible() is False

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=4.0\ninitial: A=3.0\ninitial: B=0.0"
    )
    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(3.0, 2.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": main_window._mechanism_editor._reactions_text.toPlainText(),
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

    main_window._preview_session._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    assert sliders.has_variable("k1")
    reloaded_slider = sliders._sliders["k1"]
    assert reloaded_slider.isVisible() is False
    assert sliders.variable_visible("k1") is False
    action = _find_slider_visibility_action(main_window, "mechanism", "k1")
    assert action.isChecked() is False

def test_hidden_mechanism_slider_persists_across_same_universe_scalar_reorder_completion_refresh(
    main_window, qtbot, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "# Algebra",
                "param a = 2.0",
                "param b = 3.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("a")
    first_slider = sliders._sliders["a"]
    assert first_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "a", visible=False)
    assert first_slider.isVisible() is False

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "# Algebra",
                "param b = 3.0",
                "param a = 2.0",
            ]
        )
    )
    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(1.0, 0.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": main_window._mechanism_editor._reactions_text.toPlainText(),
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

    main_window._preview_session._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    assert sliders.has_variable("a")
    reloaded_slider = sliders._sliders["a"]
    assert reloaded_slider.isVisible() is False
    assert sliders.variable_visible("a") is False
    action = _find_slider_visibility_action(main_window, "mechanism", "a")
    assert action.isChecked() is False

def test_hidden_mechanism_slider_persists_across_primary_explicit_completion_repopulation(
    main_window, qtbot, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")
    first_slider = sliders._sliders["k1"]
    assert first_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", "k1", visible=False)
    assert first_slider.isVisible() is False

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(1.0, 0.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": main_window._mechanism_editor._reactions_text.toPlainText(),
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

    main_window._preview_session._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    assert sliders.has_variable("k1")
    reloaded_slider = sliders._sliders["k1"]
    assert reloaded_slider.isVisible() is False
    assert sliders.variable_visible("k1") is False
    action = _find_slider_visibility_action(main_window, "mechanism", "k1")
    assert action.isChecked() is False

def test_hidden_mechanism_slider_persists_across_energy_mode_refresh_repopulation(
    main_window, qtbot, monkeypatch
):
    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._reactions_text.setPlainText("T=200.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(
        "\n".join(
            [
                "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        )
    )

    mechanism_text = main_window._get_mechanism_text()
    mechanism = dsl.parse_dsl_to_mechanism(mechanism_text, initials={})

    monkeypatch.setattr(
        main_window._sim_ui_port.provenance,
        "integrate_ctc",
        lambda *args, **kwargs: (0.0, "mock", True, 1e-6, "tail"),
    )

    payload = {
        "t": np.linspace(0.0, 1.0, 5),
        "Y": np.vstack([np.linspace(1.0, 0.5, 5), np.linspace(0.0, 0.5, 5)]),
        "species_names": mechanism.species_names(),
        "mechanism": mechanism,
        "mechanism_text": mechanism_text,
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    }

    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    qtbot.addWidget(main_window)
    main_window.show()

    sliders = main_window._mechanism_editor._variable_sliders
    variable_name = "dG_eq__TS1__A__B"
    assert sliders.has_variable(variable_name)
    first_slider = sliders._sliders[variable_name]
    assert first_slider.isVisible() is False

    _set_slider_visibility(main_window, "mechanism", variable_name, visible=False)
    assert first_slider.isVisible() is False

    main_window._preview_session._slider_triggered_simulation = False
    main_window.simulation_controller.on_simulation_complete(payload, fast_mode=False)
    QtWidgets.QApplication.processEvents()

    assert sliders.has_variable(variable_name)
    reloaded_slider = sliders._sliders[variable_name]
    assert reloaded_slider.isVisible() is False
    assert sliders.variable_visible(variable_name) is False
    action = _find_slider_visibility_action(main_window, "mechanism", variable_name)
    assert action.isChecked() is False

def test_template_load_dirty_slider_cancel_keeps_existing_mechanism_and_transaction(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "cancel",
        raising=False,
    )

    main_window._load_template_from_manager("reaction: A -> B; k=5.0")

    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    assert main_window._preview_session.has_dirty_transaction() is True

def test_template_load_dirty_slider_discard_does_not_trigger_stale_preview_run(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    slider_runs: list[str] = []
    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_from_slider",
        lambda: slider_runs.append("run"),
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "discard",
        raising=False,
    )

    main_window._load_template_from_manager("reaction: A -> B; k=5.0")

    assert slider_runs == []
    assert "k=5.0" in main_window._mechanism_editor._reactions_text.toPlainText()

def test_template_load_dirty_slider_commit_records_post_commit_undo_baseline(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_from_slider",
        lambda: None,
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "commit",
        raising=False,
    )
    monkeypatch.setattr(
        main_window,
        "_on_commit_slider_overrides_clicked",
        lambda: (
            main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=2.0"),
            main_window._preview_session.clear_working_transaction(),
        ),
        raising=False,
    )

    main_window._load_template_from_manager("reaction: A -> B; k=5.0")

    assert "k=5.0" in main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._undo_stack.undo()

    assert "k=2" in main_window._mechanism_editor._reactions_text.toPlainText().replace(" ", "")

def test_release_does_not_recompute_K_when_editing_kr_with_explicit_K(main_window, qtbot, monkeypatch):
    """
    Regression: When an equilibrium explicitly declares K and only kr is user-provided
    (so kf is derived), editing kr must not cause K to be recomputed from a missing kf token.

    Observed bug: on release, K snaps (often to 1.0) because _update_variable_in_mechanism()
    recomputes K from kf/kr even when kf is derived/absent from the DSL.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kr=1.0, K=0.5\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("Keq1")
    assert sliders.has_variable("kr1")

    # Prevent background simulation from running in this test.
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    # Capture the initial K value (should remain stable through a kr edit).
    initial_K = float(sliders.get_variables().get("Keq1"))

    # Simulate a drag gesture for kr1 and release so the release-commit pipeline runs.
    preview = main_window._preview_session
    main_window._on_slider_drag_started("kr1")
    kr_slider = sliders._sliders["kr1"]
    target_pos = sliders._value_to_slider_pos("kr1", 0.4)
    if target_pos == kr_slider.value():
        if target_pos < kr_slider.maximum():
            target_pos += 1
        elif target_pos > kr_slider.minimum():
            target_pos -= 1
    kr_slider.setValue(target_pos)
    qtbot.waitUntil(lambda: "kr1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("kr1")

    qtbot.waitUntil(lambda: not bool(getattr(preview, "_slider_release_in_progress", False)), timeout=1500)

    # K must not change just because kr changed (kf is the derived rate in this case).
    assert float(sliders.get_variables().get("Keq1")) == pytest.approx(initial_K)

def test_fast_worker_completion_never_reextracts_sliders(main_window, qtbot, monkeypatch):
    """
    Regression: even if `_slider_triggered_simulation` is false, a fast-mode worker
    completion must not call `_extract_and_populate_variables()`.

    This can happen if a full simulation finishes, resets the flag, and then a
    pending slider update runs in fast mode; re-extracting sliders can make
    handles appear to move on their own.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kf=0.0928966, K=0.00963829\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()

    main_window._preview_session._slider_triggered_simulation = False

    called = {"extract": 0}

    def _spy_extract():
        called["extract"] += 1

    monkeypatch.setattr(main_window, "_extract_and_populate_variables", _spy_extract)
    monkeypatch.setattr(main_window, "_prepare_slider_runtime", lambda *a, **k: None)

    def _payload(worker) -> dict:
        t = np.array([0.0, 1.0])
        Y = np.array([[1.0, 0.9], [0.0, 0.1]])
        return {
            "t": t,
            "Y": Y,
            "species_names": ["A", "B"],
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": dict(worker._solver_config),
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(payload_factory=_payload),
    )

    # Trigger a fast-mode run; completion should not re-extract sliders.
    main_window.simulation_controller.run_simulation_internal(fast_mode=True)
    QtWidgets.QApplication.processEvents()
    assert called["extract"] == 0

def test_slider_release_never_auto_commits_with_queued_changes(main_window, qtbot, monkeypatch):
    """
    Override mode: Qt can deliver queued valueChanged signals after sliderReleased.
    Ensure we never auto-commit the DSL on release (Commit button is the only writer).
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kf=0.2, kr=0.25\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    commits: list[tuple[str, float]] = []

    def _spy_commit(name: str, value: float) -> None:
        commits.append((str(name), float(value)))

    monkeypatch.setattr(main_window, "_commit_slider_value", _spy_commit)

    sliders = main_window._mechanism_editor._variable_sliders
    preview = main_window._preview_session

    main_window._on_slider_drag_started("kf1")
    sliders._on_slider_changed("kf1", sliders._value_to_slider_pos("kf1", 0.201))
    QtWidgets.QApplication.processEvents()

    main_window._on_slider_drag_finished("kf1")

    # Simulate a queued late update arriving after release.
    main_window._on_variable_changed("kf1", 0.205)
    qtbot.waitUntil(lambda: not bool(getattr(preview, "_slider_release_in_progress", False)), timeout=1500)
    assert commits == []

def test_missing_binding_forces_reparse_with_updated_value(main_window, qtbot, monkeypatch):
    """If a slider target is missing from bindings, fall back to parsing updated DSL."""
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    seen_cache_keys: list[object] = []

    def _spy_complete(_result, *_args, **kwargs) -> None:
        seen_cache_keys.append(kwargs.get("cache_key"))

    monkeypatch.setattr(main_window.simulation_controller, "on_simulation_complete", _spy_complete)

    prepared_args = []
    mechanism_texts = []
    t_ends: list[float] = []
    solver_configs: list[dict] = []

    def _fake_prepare(mechanism_text, param_names, **_kwargs):
        # Return a prepared payload without bindings so the code should skip using it.
        return BoundMechanism(
            mechanism="prepared",
            rhs=lambda t, y: y,
            bindings={},  # Missing k1 binding triggers re-parse
            species_names=["A", "B"],
            y0=np.array([1.0, 0.0]),
            param_names=list(param_names),
            mechanism_text=mechanism_text,
        )

    monkeypatch.setattr("kindred.gui.main_window_variable_runtime.prepare_bound_mechanism", _fake_prepare)

    def _on_init(worker) -> None:
        prepared_args.append(worker._prepared)
        mechanism_texts.append(worker._mechanism_text)
        try:
            t_ends.append(float(worker._t_span[1]))
        except Exception:
            t_ends.append(0.0)
        try:
            solver_configs.append(dict(worker._solver_config or {}))
        except Exception:
            solver_configs.append({})

    def _payload(worker) -> dict:
        t = np.array([0.0, 1.0])
        Y = np.vstack([np.array([1.0, 0.0]), np.array([1.0, 0.0])])
        return {
            "t": t,
            "Y": Y,
            "species_names": ["A", "B"],
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": worker._solver_config,
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_init=_on_init, payload_factory=_payload),
    )

    sliders = main_window._mechanism_editor._variable_sliders
    slider_widget = sliders._sliders["k1"]
    slider_widget.setValue(sliders._value_to_slider_pos("k1", 2.5))

    qtbot.waitUntil(lambda: len(prepared_args) == 1, timeout=1000)
    qtbot.waitUntil(lambda: len(seen_cache_keys) == 1, timeout=1000)

    # When bindings are missing, prepared payload should be skipped so the updated DSL is parsed.
    assert prepared_args[0] is None
    assert any("k=2.5" in text for text in mechanism_texts)
    assert isinstance(seen_cache_keys[0], str)
    assert seen_cache_keys[0] != build_fallback_cache_key(
        str(mechanism_texts[0]),
        float(t_ends[0]),
        dict(solver_configs[0] or {}),
    )

def test_equilibrium_slider_does_not_synthesize_K_when_not_explicit(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, kr=0.25",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("kf1")
    assert sliders.has_variable("kr1")
    assert not sliders.has_variable("Keq1")

    # Changing kf1 should not introduce a K=... token into the DSL.
    main_window._update_variable_in_mechanism("kf1", 1.2)
    updated = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "K=" not in updated

def test_reset_slider_overrides_ends_live_drag(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("k1")

    # Avoid background simulation from running in this test.
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation", lambda: None)

    main_window._on_slider_drag_started("k1")
    assert bool(getattr(sliders, "_freeze_ranges", False)) is True

    main_window._on_reset_slider_overrides_clicked()
    QtWidgets.QApplication.processEvents()
    assert bool(getattr(sliders, "_freeze_ranges", False)) is False

def test_reset_slider_overrides_clears_pending_preview_request_without_forcing_replacement_rerun(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    main_window.simulation_controller.run_state.pending_slider_sim_request_id = 99

    reruns: list[str] = []
    observed_request_ids: list[object] = []

    def _record_rerun() -> None:
        observed_request_ids.append(main_window.simulation_controller.run_state.pending_slider_sim_request_id)
        reruns.append("run")

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", _record_rerun)

    main_window._on_reset_slider_overrides_clicked()

    assert reruns == []
    assert observed_request_ids == []
    assert main_window.simulation_controller.run_state.pending_slider_sim_request_id is None

def test_reset_slider_overrides_clears_all_selected_parameter_workspaces(main_window, monkeypatch, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected")]
    assert len(selected_ids) >= 2
    _set_edit_target_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation", lambda: None)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._on_slider_drag_started("k1")
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    for set_id in selected_ids[:2]:
        assert main_window._preview_session.local_mechanism_workspace(set_id) == {"k1": pytest.approx(2.0)}

    main_window._on_reset_slider_overrides_clicked()
    qt_app.processEvents()

    for set_id in selected_ids[:2]:
        assert main_window._preview_session.local_mechanism_workspace(set_id) == {}
        assert main_window._preview_session.effective_slider_values_for_set(set_id) == {"k1": pytest.approx(1.0)}

def test_parameter_slider_drag_during_active_fast_preview_reserves_one_future_request_without_advancing_latest(
    main_window, qt_app
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    controller = main_window.simulation_controller
    controller.run_state.sim_request_id = 5
    controller.run_state.latest_sim_request_id = 5
    controller.run_state.simulation_running = True
    controller.run_state.slider_simulation_active = True
    controller.batch_run_context = {"active": True, "parallel": True, "fast_mode": True, "request_id": 5}

    main_window._on_slider_drag_started("k1")
    main_window._on_variable_changed("k1", 2.0)
    qt_app.processEvents()

    first_reserved = controller.run_state.pending_slider_sim_request_id
    assert first_reserved == 6
    assert controller.run_state.latest_sim_request_id == 5

    main_window._on_variable_changed("k1", 3.0)
    qt_app.processEvents()

    assert controller.run_state.pending_slider_sim_request_id == 6
    assert controller.run_state.latest_sim_request_id == 5
    main_window._preview_session.stop_variable_update_timer()

def test_parameter_slider_change_after_invalidated_serial_fast_preview_reserves_future_request_without_advancing_latest(
    main_window, qt_app
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    controller = main_window.simulation_controller

    class _RunningFastWorker:
        _fast_mode = True
        _request_id = 5

        def isRunning(self) -> bool:
            return True

    worker = _RunningFastWorker()
    controller._simulation_worker = worker
    controller.batch_run_context = {}
    controller.run_state.sim_request_id = 5
    controller.run_state.latest_sim_request_id = 5
    controller.run_state.pending_slider_sim_request_id = 5
    controller.run_state.simulation_running = True
    controller.run_state.slider_simulation_active = True

    controller.invalidate_slider_preview_work()

    assert controller.run_state.latest_sim_request_id == 6
    assert controller.run_state.pending_slider_sim_request_id == 5
    assert controller.run_state.simulation_running is False
    assert controller.run_state.slider_simulation_active is False

    main_window._on_variable_changed("k1", 2.0)
    qt_app.processEvents()

    assert controller.run_state.pending_slider_sim_request_id == 7
    assert controller.run_state.latest_sim_request_id == 6
    controller._simulation_worker = None
    worker._fast_mode = False
    main_window._preview_session.stop_variable_update_timer()

def test_K_slider_update_writes_consistent_equilibrium_parameters(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=251.189",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("Keq1")

    # Historically, moving K would write a derived kr with too few significant digits
    # and then re-parse would fail the strict K ≈ kf/kr validation.
    main_window._update_variable_in_mechanism("Keq1", 251.189)
    updated = main_window._mechanism_editor._reactions_text.toPlainText()

    # Must be parseable after the GUI rewrites equilibrium tokens.
    dsl.parse_dsl_to_mechanism(updated, initials={})

    # Derived rate must not be rewritten back into the DSL (let the parser derive it).
    assert "kr=" not in updated

def test_step_index_parameter_write_uses_authoritative_parameter_precision(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    main_window._update_variable_in_mechanism("k1", 1000000.1234567)

    updated = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "k=1000000.1234567" in updated
    assert "k=1e+06" not in updated

def test_K_slider_does_not_store_derived_rate_as_override(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=0.109648, K=2.29087e+06",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    main_window._update_variable_in_mechanism("Keq1", 2.29087e6)

    # kr1 is derived/read-only for explicit-K equilibria (default policy), so it must
    # never be persisted into slider overrides (which later rewrite the DSL).
    assert "kr1" not in main_window.slider_overrides()

def test_K_update_uses_current_text_owner_when_cached_metadata_is_stale(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=6, K=3",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kr=2, K=3",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )

    main_window._update_variable_in_mechanism("Keq1", 6.0)

    updated = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "Keq=6" in updated
    assert "kr=2" in updated
    assert "kf=" not in updated

def test_K_update_ignores_stale_cached_constraint_metadata_after_manual_text_edit(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = 4",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=6, K=3",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )

    main_window._update_variable_in_mechanism("Keq1", 8.0)

    updated = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "Keq=8" in updated
    assert "param Keq1 = 4" not in updated

def test_K_update_keeps_current_text_constraint_when_unused_builtin_shadow_scalar_input_present(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = 4",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )

    main_window._update_variable_in_mechanism("Keq1", 8.0)

    updated = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "K=8" not in updated
    assert "param Keq1 = 4" in updated

def test_slider_label_stays_canonical_step_label_after_updates(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    before = dict(main_window.variable_metadata())
    assert before["kf1"]["label"].startswith("Step ")
    assert before["Keq1"]["label"].startswith("Step ")

    main_window._update_variable_in_mechanism("Keq1", 12.0)
    after = dict(main_window.variable_metadata())
    assert after["kf1"]["label"].startswith("Step ")
    assert after["Keq1"]["label"].startswith("Step ")
    assert after["kf1"]["label"] == before["kf1"]["label"]
    assert after["Keq1"]["label"] == before["Keq1"]["label"]

def test_K_slider_uses_longer_debounce_than_other_params(main_window, qtbot, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    # Prevent an actual run; we're only inspecting the timer configuration.
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    sliders = main_window._mechanism_editor._variable_sliders
    preview = main_window._preview_session
    main_window._on_slider_drag_started("k2")
    sliders._on_slider_changed("k2", sliders._value_to_slider_pos("k2", 0.2))
    qtbot.waitUntil(lambda: hasattr(preview, "_variable_update_timer"), timeout=1000)
    short_ms = preview._variable_update_timer.interval()

    main_window._on_slider_drag_started("Keq1")
    sliders._on_slider_changed("Keq1", sliders._value_to_slider_pos("Keq1", 20.0))
    long_ms = preview._variable_update_timer.interval()

    assert long_ms > short_ms

def test_slider_preview_debounce_settings_drive_parameter_and_K_timers(main_window, qtbot, monkeypatch):
    main_window.settings_set_value("simulation/parameter_preview_debounce_ms", 25)
    main_window.settings_set_value("simulation/equilibrium_preview_debounce_ms", 60)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    sliders = main_window._mechanism_editor._variable_sliders
    preview = main_window._preview_session
    main_window._on_slider_drag_started("k2")
    sliders._on_slider_changed("k2", sliders._value_to_slider_pos("k2", 0.2))
    qtbot.waitUntil(lambda: hasattr(preview, "_variable_update_timer"), timeout=1000)
    short_ms = preview._variable_update_timer.interval()

    main_window._on_slider_drag_started("Keq1")
    sliders._on_slider_changed("Keq1", sliders._value_to_slider_pos("Keq1", 20.0))
    long_ms = preview._variable_update_timer.interval()

    assert short_ms == 25
    assert long_ms == 60

def test_K_implied_derived_rate_updates_without_param_block(main_window):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
                "# Algebra",
                "let yield_C = [C] / max([A]_0, 1e-30)",
            ]
        )
    )
    # No `param ...` statements; only post-solve algebra declarations in the Reactions text.
    main_window._extract_and_populate_variables()

    sliders = main_window._mechanism_editor._variable_sliders
    kf = sliders.get_variables()["kf1"]
    sliders.update_variable("Keq1", 20.0)
    main_window._refresh_derived_parameters_display()
    after = sliders.get_variables()
    assert after["kr1"] == pytest.approx(kf / 20.0)

def test_K_drag_uses_preview_t_end_and_release_uses_full_t_end(main_window, monkeypatch, qtbot):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._mechanism_editor.set_slider_points_value(350)
    main_window._sim_time_spinbox.setText("1000.0")

    # Avoid preparing/binding; we only want to inspect the worker inputs.
    monkeypatch.setattr(main_window, "_prepare_slider_runtime", lambda *a, **k: None)

    captured = []

    def _on_init(worker) -> None:
        captured.append((worker._t_span, worker._solver_config))

    def _payload(worker) -> dict:
        return {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.zeros((1, 2), dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "algebra_errors": [],
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": dict(worker._solver_config),
            "provenance": {},
            "fallback_occurred": False,
            "fallback_message": None,
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_init=_on_init, payload_factory=_payload),
    )

    # Drag preview
    preview = main_window._preview_session
    main_window._on_slider_drag_started("Keq1")
    preview._last_slider_change_name = "Keq1"
    main_window.simulation_controller.run_simulation_internal(fast_mode=True)
    qtbot.waitUntil(lambda: len(captured) >= 1, timeout=1000)
    (t_span_preview, cfg_preview) = captured[-1]
    assert t_span_preview[1] == pytest.approx(1000.0)
    assert int(cfg_preview["grid"]["N"]) <= 120

    # Release full fast run
    main_window._on_slider_drag_finished("Keq1")
    preview._last_slider_change_name = "Keq1"
    main_window.simulation_controller.run_simulation_internal(fast_mode=True)
    qtbot.waitUntil(lambda: len(captured) >= 2, timeout=1000)
    (t_span_full, cfg_full) = captured[-1]
    assert t_span_full[1] == pytest.approx(1000.0)
    assert int(cfg_full["grid"]["N"]) > 120

def test_stale_slider_worker_completion_does_not_override_latest(main_window, monkeypatch, qtbot):
    """
    Latest-only slider scheduling should avoid worker churn, while still ensuring a
    pending "latest" request runs after an in-flight fast run completes.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1.5, K=10.0",
                "reaction: B -> C ; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    workers = []

    def _on_init(worker) -> None:
        workers.append(worker)

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(on_init=_on_init, payload_factory=None, stop_after_start=False),
    )

    rid1 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(fast_mode=True, request_id=int(rid1))
    qtbot.waitUntil(lambda: len(workers) >= 1, timeout=1000)
    worker1 = workers[-1]

    rid2 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(fast_mode=True, request_id=int(rid2))
    assert len(workers) == 1  # latest-only: no new worker while fast run in flight

    payload1 = {
        "t": np.array([0.0, 1.0]),
        "Y": np.zeros((1, 2)),
        "species_names": ["A"],
        "mechanism": None,
        "mechanism_text": "old",
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
        "algebra_scalars": {},
        "algebra_errors": [],
        "provenance": {},
        "fallback_occurred": False,
        "fallback_message": None,
    }
    worker1.result_ready.emit(payload1)
    qtbot.waitUntil(lambda: len(workers) >= 2, timeout=1500)

    worker2 = workers[-1]
    payload2 = dict(payload1)
    payload2["mechanism_text"] = "new"
    worker2.result_ready.emit(payload2)
    qtbot.waitUntil(lambda: "Simulation complete" in main_window._status_label.text(), timeout=1500)

def test_variable_sliders_store_callbacks_as_partials(qtbot):
    from kindred.gui.widgets.variable_sliders import VariableSliders

    sliders = VariableSliders()
    qtbot.addWidget(sliders)
    sliders.set_variables({"k1": 1.0})

    callbacks = getattr(sliders, "_slider_callbacks")
    assert "k1" in callbacks
    assert isinstance(callbacks["k1"]["valueChanged"], functools.partial)
    assert isinstance(callbacks["k1"]["sliderPressed"], functools.partial)
    assert isinstance(callbacks["k1"]["sliderReleased"], functools.partial)

def test_simulation_worker_cleanup_does_not_wait():
    from kindred.gui.simulation_worker import SimulationWorker

    class _NoWaitWorker(SimulationWorker):
        def isRunning(self):  # type: ignore[override]
            return True

        def wait(self, *_args, **_kwargs):  # type: ignore[override]
            raise AssertionError("cleanup() must not block by calling wait()")

    worker = _NoWaitWorker(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0",
        {},
        (0.0, 1.0),
        {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
    )
    worker.cleanup()

def test_programmatic_load_clears_stale_parameter_sliders(main_window, monkeypatch):
    sliders = main_window._mechanism_editor._variable_sliders

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )

    sliders.set_variables({"k1": 2.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    main_window._variable_runtime._slider_runtime = object()
    main_window._variable_runtime.set_slider_runtime_dirty(False)
    prompt_calls: list[str] = []
    _block_insert_dialog_import(monkeypatch)
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: prompt_calls.append("prompt") or "discard",
        raising=False,
    )

    main_window._load_preset_mechanism("M9")

    assert main_window.slider_overrides() == {}
    assert sliders.get_variables() == {}
    assert main_window._variable_runtime._slider_runtime is None
    assert main_window._variable_runtime.slider_runtime_dirty() is True
    assert prompt_calls == ["prompt"]

def test_preset_load_is_replace_only_and_does_not_import_insert_dialog(main_window, monkeypatch):
    from kindred.io.resources import get_preset_mechanism

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: OLD -> TEXT; k=1.0")
    _block_insert_dialog_import(monkeypatch)

    main_window._load_preset_mechanism("M1")

    assert main_window._mechanism_editor._reactions_text.toPlainText() == get_preset_mechanism("M1")
    assert "# ===== Appended:" not in main_window._mechanism_editor._reactions_text.toPlainText()

def test_second_preset_load_is_replace_only_and_does_not_import_insert_dialog(main_window, monkeypatch):
    from kindred.io.resources import get_preset_mechanism

    sliders = main_window._mechanism_editor._variable_sliders

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )

    sliders.set_variables({"k1": 2.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    main_window._variable_runtime._slider_runtime = object()
    main_window._variable_runtime.set_slider_runtime_dirty(False)
    prompt_calls: list[str] = []
    _block_insert_dialog_import(monkeypatch)
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: prompt_calls.append("prompt") or "discard",
        raising=False,
    )

    main_window._load_preset_mechanism("M9")

    assert prompt_calls == ["prompt"]
    assert main_window.slider_overrides() == {}
    assert sliders.get_variables() == {}
    assert main_window._variable_runtime._slider_runtime is None
    assert main_window._variable_runtime.slider_runtime_dirty() is True
    assert main_window._mechanism_editor._reactions_text.toPlainText() == get_preset_mechanism("M9")
    assert "# ===== Appended:" not in main_window._mechanism_editor._reactions_text.toPlainText()

def test_selection_change_without_display_retargets_next_slider_edit_to_focused_set(main_window, monkeypatch, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    sliders = main_window._mechanism_editor._variable_sliders
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    set0_id = str(selected_ids[0])
    set1_id = str(main_window._batch_set_id_for_row(1) or "")
    assert set1_id and set1_id != set0_id
    main_window.set_slider_edit_target_set_ids([set0_id])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-selection-no-display-cache-key"
    cache.result_cache[f"{explicit_key}::{set0_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key

    sliders.update_variable("k1", 2.0)
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    assert sliders.get_variables()["k1"] == pytest.approx(2.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(2.0)

    _select_batch_rows(main_window, [1])
    _set_shown_rows(main_window, [1])
    qt_app.processEvents()

    preview = main_window._preview_session
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    assert cache.last_display_selection == []
    _assert_selection_plot_cleared(main_window)
    assert sliders.get_variables()["k1"] == pytest.approx(1.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.0)
    assert preview.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(set1_id) == {}

    sliders.update_variable("k1", 1.5)
    main_window._on_variable_changed("k1", 1.5)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    assert preview.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(1.5)}
    assert preview.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(1.5)}
    assert sliders.get_variables()["k1"] == pytest.approx(1.5)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.5)

def test_multiselect_workspace_retains_staged_values_after_switching_to_fresh_set(main_window, monkeypatch, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    sliders = main_window._mechanism_editor._variable_sliders
    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    set0_id, set1_id = selected_ids
    set2_id = str(main_window._batch_set_id_for_row(2) or "")
    assert set2_id and set2_id not in selected_ids
    _set_edit_target_rows(main_window, [0, 1])
    qt_app.processEvents()

    sliders.update_variable("k1", 2.0)
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    preview = main_window._preview_session
    assert preview.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.0)}

    _select_batch_rows(main_window, [2])
    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert sliders.get_variables()["k1"] == pytest.approx(1.0)
    assert preview.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(set2_id) == {}

    _select_batch_rows(main_window, [0])
    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert sliders.get_variables()["k1"] == pytest.approx(2.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(2.0)
    assert preview.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert preview.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.0)}

def test_replayed_precommit_dirty_overlay_is_not_reaccepted_as_truthful_after_authoritative_change(
    main_window,
    qt_app,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("commit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    primary_set_id = selected_ids[0]
    secondary_set_id = selected_ids[1]

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[primary_set_id])
    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="A", value=2.5) is True

    explicit_key = "gui-slider-live-plot-provenance-followup-explicit-key"
    for set_id, series in (
        (primary_set_id, np.asarray([7.0, 14.0], dtype=float)),
        (secondary_set_id, np.asarray([3.0, 6.0], dtype=float)),
    ):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(selected_ids)
    cache.active_batch_set_id = primary_set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(primary_set_id) or "")
    cache.last_display_selection = list(selected_ids)

    preview_t = _current_preview_time_axis(main_window)
    primary_preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    secondary_preview_series = np.asarray(np.linspace(4.0, 8.0, preview_t.size, dtype=float))
    preview_key = "gui-slider-live-plot-provenance-followup-preview-key"
    for set_id, series in (
        (primary_set_id, primary_preview_series),
        (secondary_set_id, secondary_preview_series),
    ):
        mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=set_id)
        solver_config, _, preview_token = main_window._current_workspace_preview_context(
            set_id=set_id,
            mechanism_text=mechanism_text,
        )
        cache.preview_cache[f"{preview_key}::{set_id}"] = {
            "t": preview_t,
            "series": {"A": series},
            "algebra_scalars": {},
            "mechanism_text": mechanism_text,
            "solver_config": dict(solver_config),
            "preview_batch_cache_token": str(preview_token or ""),
        }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)

    assert main_window.display_cached_batch_selection(
        cache_key=preview_key,
        selected_sets=selected_ids,
        cache_store=cache.preview_cache,
        allow_fallback=False,
    )
    cache.preview_cache.clear()
    cache.active_preview_cache_key = "missing-live-plot-provenance-followup-preview-cache-entry"

    plot = main_window._plot_tabs._main_plot
    preserve_current_display = main_window._active_workspace_preview_display_snapshot()
    assert preserve_current_display is not None

    effective_values = main_window._preview_session.effective_slider_values(set_id=primary_set_id)
    main_window._apply_effective_slider_values_to_mechanism_editors(
        effective_values,
        description="Test stale preview provenance replay",
    )
    main_window._preview_session.commit_current_mechanism_workspace()
    main_window._sync_after_authoritative_slider_materialization(
        preserve_current_display=preserve_current_display
    )
    qt_app.processEvents()

    assert main_window._preview_session.local_mechanism_workspace(primary_set_id) == {}
    assert main_window._preview_session.local_mechanism_workspace(secondary_set_id) == {}
    assert main_window._preview_session.has_dirty_state_for_set(secondary_set_id) is True
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        primary_preview_series,
    )
    replayed_overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(replayed_overlays) == 3
    replayed_primary_canonical_ghost = next(
        entry
        for entry in replayed_overlays
        if str(entry.get("set_id") or "") == primary_set_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((replayed_primary_canonical_ghost.get("series") or {})["A"], dtype=float),
        np.asarray([7.0, 14.0], dtype=float),
    )
    replayed_secondary_preview_overlay = next(
        entry
        for entry in replayed_overlays
        if str(entry.get("set_id") or "") == secondary_set_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((replayed_secondary_preview_overlay.get("series") or {})["A"], dtype=float),
        secondary_preview_series,
    )
    assert main_window._matching_preview_entry_for_workspace_set(set_id=secondary_set_id).entry is None
    assert main_window._displayed_workspace_preview_provenance_matches_current_workspace(
        set_id=secondary_set_id
    ) is False
    assert main_window._active_workspace_preview_display_snapshot() is None

def test_commit_primary_dirty_preview_drops_stale_secondary_explicit_overlay(main_window, qt_app, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("commit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    primary_set_id = selected_ids[0]
    secondary_set_id = selected_ids[1]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-commit-drops-stale-secondary-explicit-overlay-cache-key"
    primary_explicit_series = np.asarray([7.0, 14.0], dtype=float)
    secondary_explicit_series = np.asarray([3.0, 6.0], dtype=float)
    for set_id, series in (
        (primary_set_id, primary_explicit_series),
        (secondary_set_id, secondary_explicit_series),
    ):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(selected_ids)
    cache.active_batch_set_id = primary_set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(primary_set_id) or "")
    cache.last_display_selection = list(selected_ids)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[primary_set_id])

    preview_t = _current_preview_time_axis(main_window)
    primary_preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "gui-slider-commit-drops-stale-secondary-explicit-overlay-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=primary_set_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=primary_set_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{primary_set_id}"] = {
        "t": preview_t,
        "series": {"A": primary_preview_series},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)

    plot = main_window._plot_tabs._main_plot
    primary_label = str(main_window.batch_set_name_for_id(primary_set_id) or primary_set_id)
    secondary_label = str(main_window.batch_set_name_for_id(secondary_set_id) or secondary_set_id)
    plot.set_statistics_results(
        {
            primary_label: {
                "t": np.asarray(preview_t, dtype=float),
                "series": {"A": np.asarray(primary_preview_series, dtype=float)},
            },
            secondary_label: {
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray(secondary_explicit_series, dtype=float)},
            },
        },
        prefer=primary_label,
    )
    main_window.set_data(
        np.asarray(preview_t, dtype=float),
        {"A": np.asarray(primary_preview_series, dtype=float)},
        label=primary_label,
        overlays=[
            {
                "label": secondary_label,
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray(secondary_explicit_series, dtype=float)},
            }
        ],
    )
    assert len(list(getattr(plot, "_simulation_overlays", []) or [])) == 1

    snapshot = main_window._active_workspace_preview_display_snapshot()
    assert snapshot is not None
    assert list(snapshot.get("preserved_overlays") or []) == []

    main_window._on_commit_slider_overrides_clicked()
    qt_app.processEvents()

    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."
    assert cache.last_display_selection == [primary_set_id]
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        primary_preview_series,
    )
    assert list(getattr(plot, "_simulation_overlays", []) or []) == []
    assert plot._stats_result_selector.count() == 1
    assert plot._stats_result_selector.itemText(0) == primary_label

def test_commit_dirty_workspace_does_not_preserve_stale_explicit_plot_without_truthful_preview(
    main_window,
    qt_app,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("commit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    set_id = selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-commit-invalidates-stale-explicit-plot-without-truthful-preview"
    explicit_t = np.asarray([0.0, 1.0], dtype=float)
    explicit_series = np.asarray([7.0, 14.0], dtype=float)
    cache.result_cache[f"{explicit_key}::{set_id}"] = {
        "t": explicit_t,
        "series": {"A": explicit_series},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (set_id,)
    cache.active_batch_set_id = set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(set_id) or "")
    cache.last_display_selection = [set_id]
    cache.active_preview_cache_key = "missing-live-preview-provenance-key"
    cache.active_preview_scope_set_ids = (set_id,)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[set_id])
    main_window._sync_mechanism_controls_to_focused_batch_set(use_workspace=True)
    assert main_window.variable_slider_values()["k1"] == pytest.approx(2.0)

    main_window.set_data(
        np.asarray(explicit_t, dtype=float),
        {"A": np.asarray(explicit_series, dtype=float)},
        label=str(main_window.batch_set_name_for_id(set_id) or set_id),
        overlays=[],
    )

    main_window._on_commit_slider_overrides_clicked()
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (set_id,)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_commit_does_not_preserve_older_active_preview_when_dirty_state_has_advanced(
    main_window,
    qt_app,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("commit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    set_id = selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-commit-drops-older-active-preview-after-dirty-advance-explicit-key"
    cache.result_cache[f"{explicit_key}::{set_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (set_id,)
    cache.active_batch_set_id = set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(set_id) or "")
    cache.last_display_selection = [set_id]

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[set_id])

    preview_t = _current_preview_time_axis(main_window)
    preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "gui-slider-commit-drops-older-active-preview-after-dirty-advance-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=set_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=set_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{set_id}"] = {
        "t": preview_t,
        "series": {"A": preview_series},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (set_id,)
    assert main_window.display_cached_batch_selection(
        cache_key=preview_key,
        selected_sets=[set_id],
        cache_store=cache.preview_cache,
        allow_fallback=False,
    )

    main_window._preview_session.stage_slider_value("k1", 3.0, target_set_ids=[set_id])
    main_window._sim_controller.run_state.pending_slider_simulation = False
    assert main_window._matching_preview_entry_for_workspace_set(set_id=set_id).entry is None

    main_window._on_commit_slider_overrides_clicked()
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (set_id,)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_single_set_stale_preview_cache_display_returns_false_and_defers_to_preview_pending_fallback(
    main_window,
    qt_app,
):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0")
    main_window._extract_and_populate_variables()
    main_window._mechanism_editor.set_slider_points_value(350)
    main_window._batch_model.set_species(["A"])

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    set_id = selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "single-set-stale-preview-cache-display-fallback-explicit-key"
    explicit_t = np.asarray(np.linspace(0.0, 1.0, 100, dtype=float))
    explicit_series = np.asarray(np.linspace(7.0, 14.0, explicit_t.size, dtype=float))
    cache.result_cache[f"{explicit_key}::{set_id}"] = {
        "t": explicit_t,
        "series": {"A": explicit_series},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (set_id,)
    cache.active_batch_set_id = set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(set_id) or "")
    cache.last_display_selection = [set_id]

    main_window.set_data(
        np.asarray(explicit_t, dtype=float),
        {"A": np.asarray(explicit_series, dtype=float)},
        label=str(main_window.batch_set_name_for_id(set_id) or set_id),
        overlays=[],
    )

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[set_id])

    preview_t = _current_preview_time_axis(main_window)
    stale_preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "single-set-stale-preview-cache-display-fallback-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=set_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=set_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{set_id}"] = {
        "t": preview_t,
        "series": {"A": stale_preview_series},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (set_id,)

    main_window._preview_session.stage_slider_value("k1", 3.0, target_set_ids=[set_id])
    main_window._sim_controller.run_state.pending_slider_simulation = False
    assert main_window._matching_preview_entry_for_workspace_set(set_id=set_id).entry is None

    assert (
        main_window.display_cached_batch_selection(
            cache_key=preview_key,
            selected_sets=[set_id],
            cache_store=cache.preview_cache,
            allow_fallback=False,
        )
        is False
    )

    plot = main_window._plot_tabs._main_plot
    displayed_series = np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float)
    assert np.allclose(displayed_series, explicit_series)
    assert displayed_series.size == explicit_series.size
    assert displayed_series.size != stale_preview_series.size

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_commit_drops_stale_secondary_preview_overlay_when_dirty_overlay_has_advanced(
    main_window,
    qt_app,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    assert main_window._batch_model.setData(main_window._batch_model.index(0, 1), "1.0")
    assert main_window._batch_model.setData(main_window._batch_model.index(1, 1), "0.2")

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("commit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    primary_set_id = selected_ids[0]
    secondary_set_id = selected_ids[1]

    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="A", value=2.5) is True

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-species-multiselect-commit-drops-stale-secondary-preview-overlay-explicit-key"
    for set_id, series in (
        (primary_set_id, np.asarray([7.0, 14.0], dtype=float)),
        (secondary_set_id, np.asarray([3.0, 6.0], dtype=float)),
    ):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(selected_ids)
    cache.active_batch_set_id = primary_set_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(primary_set_id) or "")
    cache.last_display_selection = list(selected_ids)

    preview_t = _current_preview_time_axis(main_window)
    primary_preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    secondary_preview_series = np.asarray(np.linspace(4.0, 8.0, preview_t.size, dtype=float))
    preview_key = "gui-species-multiselect-commit-drops-stale-secondary-preview-overlay-preview-key"
    for set_id, series in (
        (primary_set_id, primary_preview_series),
        (secondary_set_id, secondary_preview_series),
    ):
        mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=set_id)
        solver_config, _, preview_token = main_window._current_workspace_preview_context(
            set_id=set_id,
            mechanism_text=mechanism_text,
        )
        cache.preview_cache[f"{preview_key}::{set_id}"] = {
            "t": preview_t,
            "series": {"A": series},
            "algebra_scalars": {},
            "mechanism_text": mechanism_text,
            "solver_config": dict(solver_config),
            "preview_batch_cache_token": str(preview_token or ""),
        }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)
    assert main_window.display_cached_batch_selection(
        cache_key=preview_key,
        selected_sets=selected_ids,
        cache_store=cache.preview_cache,
        allow_fallback=False,
    )

    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="A", value=3.5) is True
    main_window._sim_controller.run_state.pending_slider_simulation = False
    assert main_window._matching_preview_entry_for_workspace_set(set_id=secondary_set_id).entry is None
    assert main_window._matching_preview_entry_for_workspace_set(set_id=primary_set_id).entry is not None

    main_window._on_commit_slider_overrides_clicked()
    qt_app.processEvents()

    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(3.5, rel=1e-6, abs=1e-9)
    assert main_window._preview_session.has_staged_concentration_overlays() is False
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert cache.active_cache_invalidated_set_ids == tuple(selected_ids)
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_workspace_aware_preview_display_clears_stale_pending_status_after_success(
    main_window,
    qt_app,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()

    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    dirty_primary_id = selected_ids[0]
    clean_secondary_id = selected_ids[1]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "workspace-aware-preview-status-success-explicit-key"
    primary_canonical = np.asarray([7.0, 14.0], dtype=float)
    secondary_canonical = np.asarray([3.0, 6.0], dtype=float)
    for set_id, series in (
        (dirty_primary_id, primary_canonical),
        (clean_secondary_id, secondary_canonical),
    ):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(selected_ids)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[dirty_primary_id])

    preview_t = _current_preview_time_axis(main_window)
    dirty_preview = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "workspace-aware-preview-status-success-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=dirty_primary_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=dirty_primary_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{dirty_primary_id}"] = {
        "t": preview_t,
        "series": {"A": dirty_preview},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)

    main_window._status_label.setText("Preview pending for current selection.")
    assert (
        main_window.display_cached_batch_selection(
            cache_key=preview_key,
            selected_sets=selected_ids,
            prefer_set=dirty_primary_id,
            cache_store=cache.preview_cache,
            allow_fallback=False,
        )
        is True
    )
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window._status_label.text() == f"Loaded 1 species, {preview_t.size} timepoints"
    assert main_window.active_batch_selection()[0] == dirty_primary_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        dirty_preview,
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert any(
        str(entry.get("set_id") or "") == clean_secondary_id
        and str(entry.get("curve_role") or "") != "canonical_ghost"
        for entry in overlays
    )
    assert any(
        str(entry.get("set_id") or "") == dirty_primary_id
        and str(entry.get("curve_role") or "") == "canonical_ghost"
        for entry in overlays
    )

@pytest.mark.parametrize(
    ("initial_status", "stage_missing_preview", "seed_missing_explicit", "expected_status"),
    [
        (
            "Result not cached (evicted). Press Run to compute.",
            True,
            False,
            "Preview pending for current selection.",
        ),
        (
            "Preview pending for current selection.",
            False,
            True,
            "Result not cached (evicted). Press Run to compute.",
        ),
    ],
)
def test_workspace_aware_preview_partial_success_uses_current_selection_warning(
    main_window,
    qt_app,
    initial_status,
    stage_missing_preview,
    seed_missing_explicit,
    expected_status,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()

    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    dirty_primary_id = selected_ids[0]
    secondary_id = selected_ids[1]

    cache = main_window.simulation_controller.batch_cache
    if seed_missing_explicit:
        explicit_key = "workspace-aware-preview-partial-status-explicit-key"
        cache.result_cache[f"{explicit_key}::{dirty_primary_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
            "algebra_scalars": {},
        }
        cache.active_cache_key = explicit_key
        cache.active_cache_valid_set_ids = (dirty_primary_id,)
    else:
        cache.active_cache_key = None
        cache.active_cache_valid_set_ids = None

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[dirty_primary_id])
    if stage_missing_preview:
        main_window._preview_session.stage_slider_value("k1", 3.0, target_set_ids=[secondary_id])

    preview_t = _current_preview_time_axis(main_window)
    dirty_preview = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "workspace-aware-preview-partial-status-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=dirty_primary_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=dirty_primary_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{dirty_primary_id}"] = {
        "t": preview_t,
        "series": {"A": dirty_preview},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)

    assert initial_status != expected_status
    main_window._status_label.setText(initial_status)
    assert (
        main_window.display_cached_batch_selection(
            cache_key=preview_key,
            selected_sets=selected_ids,
            prefer_set=dirty_primary_id,
            cache_store=cache.preview_cache,
            allow_fallback=False,
        )
        is True
    )
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window._status_label.text() == expected_status
    assert main_window.active_batch_selection()[0] == dirty_primary_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        dirty_preview,
    )

def test_slider_preview_flush_uses_shown_rows_for_plot_membership_not_only_highlighted_selection(
    main_window,
    qt_app,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [2])
    _set_shown_rows(main_window, [0, 1, 2])
    qt_app.processEvents()

    shown_ids = [str(set_id) for set_id in main_window.shown_batch_set_ids() if str(set_id)]
    assert len(shown_ids) == 3
    clean_shown_ids = shown_ids[:2]
    dirty_primary_id = shown_ids[2]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "slider-preview-flush-shown-membership-explicit-key"
    explicit_series_by_id = {
        clean_shown_ids[0]: np.asarray([2.0, 4.0], dtype=float),
        clean_shown_ids[1]: np.asarray([3.0, 6.0], dtype=float),
        dirty_primary_id: np.asarray([5.0, 10.0], dtype=float),
    }
    for set_id, series in explicit_series_by_id.items():
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(shown_ids)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[dirty_primary_id])

    preview_t = _current_preview_time_axis(main_window)
    dirty_preview = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "slider-preview-flush-shown-membership-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=dirty_primary_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=dirty_primary_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{dirty_primary_id}"] = {
        "t": preview_t,
        "series": {"A": dirty_preview},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (dirty_primary_id,)

    controller = main_window.simulation_controller
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._queue_slider_plot_update(
        set_id=dirty_primary_id,
        cache_key=preview_key,
        request_id=1,
        run_id=2,
        slider_triggered=True,
    )

    assert controller._flush_slider_plot_updates() is True
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window.active_batch_selection()[0] == dirty_primary_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        dirty_preview,
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(overlays) == 3
    for clean_set_id in clean_shown_ids:
        clean_overlay = next(
            entry
            for entry in overlays
            if str(entry.get("set_id") or "") == clean_set_id and str(entry.get("curve_role") or "") != "canonical_ghost"
        )
        assert np.allclose(
            np.asarray((clean_overlay.get("series") or {})["A"], dtype=float),
            explicit_series_by_id[clean_set_id],
        )
    dirty_primary_ghost = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == dirty_primary_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((dirty_primary_ghost.get("series") or {})["A"], dtype=float),
        explicit_series_by_id[dirty_primary_id],
    )

def test_slider_preview_single_shown_dirty_set_flush_displays_canonical_ref(
    main_window,
    qt_app,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [1])
    _set_shown_rows(main_window, [1])
    qt_app.processEvents()

    shown_id = str(main_window.batch_set_id_for_row(1) or "")
    assert shown_id

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "slider-preview-single-shown-flush-explicit-key"
    explicit_series = np.asarray([4.0, 8.0], dtype=float)
    cache.result_cache[f"{explicit_key}::{shown_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": explicit_series},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (shown_id,)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[shown_id])

    preview_t = _current_preview_time_axis(main_window)
    preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    preview_key = "slider-preview-single-shown-flush-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=shown_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=shown_id,
        mechanism_text=mechanism_text,
    )
    cache.preview_cache[f"{preview_key}::{shown_id}"] = {
        "t": preview_t,
        "series": {"A": preview_series},
        "algebra_scalars": {},
        "mechanism_text": mechanism_text,
        "solver_config": dict(solver_config),
        "preview_batch_cache_token": str(preview_token or ""),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (shown_id,)

    controller = main_window.simulation_controller
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._queue_slider_plot_update(
        set_id=shown_id,
        cache_key=preview_key,
        request_id=1,
        run_id=2,
        slider_triggered=True,
    )

    assert controller._flush_slider_plot_updates() is True
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window.active_batch_selection()[0] == shown_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        preview_series,
    )
    canonical_ghost = next(
        entry
        for entry in list(getattr(plot, "_simulation_overlays", []) or [])
        if str(entry.get("set_id") or "") == shown_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((canonical_ghost.get("series") or {})["A"], dtype=float),
        explicit_series,
    )

def test_slider_preview_multiselect_flush_uses_requested_preview_cache_key_for_ghosts(
    main_window,
    qt_app,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_batch_rows(main_window, [0, 1])
    qt_app.processEvents()

    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 2
    dirty_primary_id = selected_ids[0]
    clean_secondary_id = selected_ids[1]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "slider-preview-multiselect-stale-cache-key-explicit-key"
    primary_canonical = np.asarray([7.0, 14.0], dtype=float)
    secondary_canonical = np.asarray([3.0, 6.0], dtype=float)
    for set_id, series in (
        (dirty_primary_id, primary_canonical),
        (clean_secondary_id, secondary_canonical),
    ):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = tuple(selected_ids)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[dirty_primary_id])

    preview_t = _current_preview_time_axis(main_window)
    requested_preview = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    newer_preview = np.asarray(np.linspace(11.0, 22.0, preview_t.size, dtype=float))
    requested_preview_key = "slider-preview-multiselect-requested-preview-key"
    active_preview_key = "slider-preview-multiselect-newer-active-preview-key"
    mechanism_text = main_window._mechanism_text_for_workspace_selection(set_id=dirty_primary_id)
    solver_config, _, preview_token = main_window._current_workspace_preview_context(
        set_id=dirty_primary_id,
        mechanism_text=mechanism_text,
    )
    for preview_key, preview_series in (
        (requested_preview_key, requested_preview),
        (active_preview_key, newer_preview),
    ):
        cache.preview_cache[f"{preview_key}::{dirty_primary_id}"] = {
            "t": preview_t,
            "series": {"A": preview_series},
            "algebra_scalars": {},
            "mechanism_text": mechanism_text,
            "solver_config": dict(solver_config),
            "preview_batch_cache_token": str(preview_token or ""),
        }
    cache.active_preview_cache_key = active_preview_key
    cache.active_preview_scope_set_ids = tuple(selected_ids)

    controller = main_window.simulation_controller
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._queue_slider_plot_update(
        set_id=dirty_primary_id,
        cache_key=requested_preview_key,
        request_id=1,
        run_id=2,
        slider_triggered=True,
    )

    assert controller._flush_slider_plot_updates() is True
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        requested_preview,
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(overlays) == 2
    clean_overlay = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == clean_secondary_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((clean_overlay.get("series") or {})["A"], dtype=float),
        secondary_canonical,
    )
    dirty_primary_ghost = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == dirty_primary_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((dirty_primary_ghost.get("series") or {})["A"], dtype=float),
        primary_canonical,
    )

def test_direct_mechanism_edit_invalidates_displayed_explicit_results(main_window, qt_app, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("edit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    second_set_id = str(main_window._batch_set_id_for_row(1) or "")
    assert second_set_id and second_set_id != selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-direct-edit-invalidates-explicit-results-cache-key"
    for set_id in (selected_ids[0], second_set_id):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (selected_ids[0], second_set_id)
    cache.active_batch_set_id = selected_ids[0]
    cache.active_batch_set = str(main_window.batch_set_name_for_id(selected_ids[0]) or "")
    cache.last_display_selection = list(selected_ids)
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="baseline",
        overlays=[],
    )

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=2.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (selected_ids[0], second_set_id)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_programmatic_mechanism_load_invalidates_displayed_explicit_results(main_window, qt_app, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("programmatic load triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    second_set_id = str(main_window._batch_set_id_for_row(1) or "")
    assert second_set_id and second_set_id != selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-programmatic-load-invalidates-explicit-results-cache-key"
    for set_id in (selected_ids[0], second_set_id):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (selected_ids[0], second_set_id)
    cache.active_batch_set_id = selected_ids[0]
    cache.active_batch_set = str(main_window.batch_set_name_for_id(selected_ids[0]) or "")
    cache.last_display_selection = list(selected_ids)
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="baseline",
        overlays=[],
    )

    reactions_widget = main_window._mechanism_editor._reactions_text
    reactions_widget.blockSignals(True)
    try:
        reactions_widget.setPlainText("reaction: A -> B; k=3.0\ninitial: A=1.0\ninitial: B=0.0")
    finally:
        reactions_widget.blockSignals(False)

    main_window._on_programmatic_mechanism_load()
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (selected_ids[0], second_set_id)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_pending_init_migration_rewrite_does_not_invalidate_displayed_explicit_results(
    main_window, qt_app, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    qt_app.processEvents()

    seed, rewrite = migrate_reaction_dsl_initial_concentrations(
        main_window._mechanism_editor._reactions_text.toPlainText(),
        set_name="set1",
    )
    assert seed == {"A": 1.0, "B": 0.0}
    assert rewrite != main_window._mechanism_editor._reactions_text.toPlainText()

    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = "gui-slider-pending-init-migration-preserves-display-cache-key"
    cache.active_cache_valid_set_ids = ("set1",)
    cache.active_batch_set_id = "set1"
    cache.active_batch_set = "set1"
    cache.last_display_selection = ["set1"]
    cache.result_cache[f"{cache.active_cache_key}::set1"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="set1",
        overlays=[],
    )

    original_set_text = main_window.set_mechanism_reactions_text_with_optional_undo

    def _set_text_and_replay_invalidation(new_text: str, description: str, *, record_undo: bool) -> None:
        original_set_text(new_text, description, record_undo=record_undo)
        QtCore.QTimer.singleShot(0, main_window._on_authoritative_mechanism_input_changed)

    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        _set_text_and_replay_invalidation,
        raising=True,
    )

    assert getattr(main_window._plot_tabs._main_plot, "_t", None) is not None

    applied = main_window.apply_pending_init_migration(seed=seed, rewrite=rewrite)
    qt_app.processEvents()
    qt_app.processEvents()

    assert applied is True
    assert cache.active_cache_invalidated_set_ids is None
    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."
    assert getattr(main_window._plot_tabs._main_plot, "_t", None) is not None
    assert dict(getattr(main_window._plot_tabs._main_plot, "_series", {}) or {}) != {}

def test_pending_init_failed_run_reinvalidates_preserved_result(main_window, qt_app, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    qt_app.processEvents()

    seed, rewrite = migrate_reaction_dsl_initial_concentrations(
        main_window._mechanism_editor._reactions_text.toPlainText(),
        set_name="set1",
    )
    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = "gui-slider-pending-init-failed-run-invalidates-preserved-result"
    cache.active_cache_valid_set_ids = ("set1",)
    cache.active_batch_set_id = "set1"
    cache.active_batch_set = "set1"
    cache.last_display_selection = ["set1"]
    cache.result_cache[f"{cache.active_cache_key}::set1"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="set1",
        overlays=[],
    )

    original_set_text = main_window.set_mechanism_reactions_text_with_optional_undo

    def _set_text_and_replay_invalidation(new_text: str, description: str, *, record_undo: bool) -> None:
        original_set_text(new_text, description, record_undo=record_undo)
        QtCore.QTimer.singleShot(0, main_window._on_authoritative_mechanism_input_changed)

    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        _set_text_and_replay_invalidation,
        raising=True,
    )

    assert main_window.apply_pending_init_migration(seed=seed, rewrite=rewrite) is True
    qt_app.processEvents()
    qt_app.processEvents()
    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."
    assert getattr(main_window._plot_tabs._main_plot, "_t", None) is not None

    main_window.invalidate_pending_init_preserved_results_after_failed_run()
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == ("set1",)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_pending_init_guard_does_not_suppress_next_real_mechanism_edit(
    main_window, qt_app, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("edit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    second_set_id = str(main_window._batch_set_id_for_row(1) or "")
    assert second_set_id and second_set_id != selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-pending-init-guard-next-edit-invalidates-cache-key"
    for set_id in (selected_ids[0], second_set_id):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (selected_ids[0], second_set_id)
    cache.active_batch_set_id = selected_ids[0]
    cache.active_batch_set = str(main_window.batch_set_name_for_id(selected_ids[0]) or "")
    cache.last_display_selection = list(selected_ids)
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="baseline",
        overlays=[],
    )

    seed, rewrite = migrate_reaction_dsl_initial_concentrations(
        main_window._mechanism_editor._reactions_text.toPlainText(),
        set_name="set1",
    )
    assert main_window.apply_pending_init_migration(seed=seed, rewrite=rewrite) is True
    main_window.arm_pending_init_result_invalidation_guard()
    qt_app.processEvents()

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=2.0"
    )
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (selected_ids[0], second_set_id)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_pending_init_guard_does_not_suppress_next_real_state_network_edit(
    main_window, qt_app, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("state-network edit triggered run")),
        raising=True,
    )

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    second_set_id = str(main_window._batch_set_id_for_row(1) or "")
    assert second_set_id and second_set_id != selected_ids[0]

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "gui-slider-pending-init-guard-state-network-edit-invalidates-cache-key"
    for set_id in (selected_ids[0], second_set_id):
        cache.result_cache[f"{explicit_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (selected_ids[0], second_set_id)
    cache.active_batch_set_id = selected_ids[0]
    cache.active_batch_set = str(main_window.batch_set_name_for_id(selected_ids[0]) or "")
    cache.last_display_selection = list(selected_ids)
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([7.0, 14.0], dtype=float)},
        label="baseline",
        overlays=[],
    )

    seed, rewrite = migrate_reaction_dsl_initial_concentrations(
        main_window._mechanism_editor._reactions_text.toPlainText(),
        set_name="set1",
    )
    assert main_window.apply_pending_init_migration(seed=seed, rewrite=rewrite) is True
    main_window.arm_pending_init_result_invalidation_guard()
    qt_app.processEvents()

    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(
        "\n".join(
            [
                "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "edge: A,TS1",
                "edge: TS1,B",
            ]
        )
    )
    qt_app.processEvents()

    assert cache.active_cache_invalidated_set_ids == (selected_ids[0], second_set_id)
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

def test_first_explicit_run_after_example_load_keeps_displayed_result(main_window, qt_app, monkeypatch):
    from PySide6 import QtCore

    class _AsyncWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent=None,
            prepared=None,
            include_mechanism_in_result_payload=True,
        ):
            super().__init__(parent)
            self._running = False
            self._mechanism_text = str(mechanism_text)
            self._prepared = prepared
            self._fast_mode = False

        def start(self) -> None:
            self._running = True

            def _finish() -> None:
                self.progress.emit(100, "done")
                self.result_ready.emit(_worker_payload(self._prepared or {}, self._mechanism_text))
                self._running = False

            QtCore.QTimer.singleShot(0, _finish)

        def cancel(self) -> None:
            self._running = False

        def isRunning(self) -> bool:
            return bool(self._running)

        def wait(self, *_args, **_kwargs) -> bool:
            self._running = False
            return True

        def terminate(self) -> None:
            self._running = False

    monkeypatch.setattr("kindred.gui.simulation_worker.SimulationWorker", _AsyncWorker)

    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    main_window.simulation_controller.run_simulation()
    for _ in range(6):
        qt_app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    plot = main_window._plot_tabs._main_plot
    assert getattr(plot, "_t", None) is not None
    assert dict(getattr(plot, "_series", {}) or {}) != {}
    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."

def test_open_solver_settings_ok_without_changes_preserves_focused_slider_preview_state(
    main_window, qt_app, monkeypatch
):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

    focused_set_id = str(main_window._preview_session.focused_mechanism_workspace_set_id() or "").strip()
    assert focused_set_id

    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[focused_set_id])
    main_window._sync_mechanism_controls_to_focused_batch_set()
    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([3.0, 6.0], dtype=float)},
        label=focused_set_id,
        overlays=[],
    )
    plot = main_window._plot_tabs._main_plot
    before_t = np.asarray(plot._t, dtype=float).copy()
    before_series = np.asarray(plot._series["A"], dtype=float).copy()
    assert main_window.variable_slider_values()["k1"] == pytest.approx(2.0)

    extract_calls: list[bool] = []
    original_extract = main_window._extract_and_populate_variables

    def _spy_extract_and_populate_variables(*, preserve_visibility: bool = False):
        extract_calls.append(bool(preserve_visibility))
        return original_extract(preserve_visibility=bool(preserve_visibility))

    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            return dict(self._settings)

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    monkeypatch.setattr(
        main_window,
        "_extract_and_populate_variables",
        _spy_extract_and_populate_variables,
        raising=True,
    )

    main_window._open_solver_settings()
    qt_app.processEvents()

    assert extract_calls == []
    assert main_window.variable_slider_values()["k1"] == pytest.approx(2.0)
    assert np.array_equal(np.asarray(plot._t, dtype=float), before_t)
    assert np.array_equal(np.asarray(plot._series["A"], dtype=float), before_series)


def test_run_selected_success_refreshes_focused_species_sliders_after_clearing_staged_overlays(
    main_window,
    qt_app,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    assert main_window._batch_model.setData(main_window._batch_model.index(0, 1), "1.0")

    _select_batch_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]
    assert len(selected_ids) == 1
    set_id = selected_ids[0]

    panel = main_window._mechanism_editor.species_sliders_widget()
    assert panel is not None
    panel.rebuild_from_current_row()
    qt_app.processEvents()
    assert panel._rows["A"].value_label.text() == "1.000"

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    panel.rebuild_from_current_row()
    qt_app.processEvents()
    assert panel._rows["A"].value_label.text() == "2.500"

    monkeypatch.setattr(
        main_window.simulation_controller,
        "_start_next_batch_simulation",
        lambda: None,
        raising=True,
    )

    main_window.simulation_controller.run_simulation_internal(fast_mode=False)
    ctx = dict(main_window.simulation_controller._batch_run_context or {})
    mechanism_text_by_set_id = dict(ctx.get("mechanism_text_by_set_id") or {})

    result = {
        "t": np.linspace(0.0, 1.0, 3),
        "Y": np.asarray([[1.0, 0.5, 0.1], [0.0, 0.5, 0.9]], dtype=float),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": str(mechanism_text_by_set_id.get(set_id) or ""),
        "solver_config": {"solver": "Radau", "temperature_K": 298.15},
        "algebra_scalars": {},
        "algebra_errors": [],
        "fallback_occurred": False,
        "fallback_message": None,
    }
    main_window.simulation_controller._on_simulation_complete(
        result,
        run_id=int(main_window.simulation_controller._active_run_id),
        fast_mode=False,
        request_id=int(ctx.get("request_id") or 0),
        batch_set=str(main_window.batch_set_name_for_id(set_id) or ""),
        batch_set_id=set_id,
        cache_key=str(ctx.get("cache_key") or ""),
    )
    qt_app.processEvents()

    assert main_window._preview_session.has_staged_concentration_overlays() is False
    assert panel._rows["A"].value_label.text() == "1.000"


def test_run_selected_after_slider_gesture_does_not_replay_fast_preview(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    _select_batch_rows(main_window, [0])
    qtbot.waitUntil(lambda: len(main_window._batch_set_ids_for_scope("selected")) == 1, timeout=1000)
    display_lengths: list[int] = []
    original_set_data = main_window.results_controller.set_data

    def _recording_set_data(t, series, **kwargs):
        display_lengths.append(int(np.asarray(t, dtype=float).reshape(-1).size))
        return original_set_data(t, series, **kwargs)

    monkeypatch.setattr(main_window.results_controller, "set_data", _recording_set_data, raising=True)

    def _payload(worker) -> dict:
        point_count = 350 if bool(getattr(worker, "_fast_mode", False)) else 100
        t_end = 50.0 if bool(getattr(worker, "_fast_mode", False)) else 1.0
        t = np.asarray(np.linspace(0.0, t_end, point_count, dtype=float))
        y = np.vstack(
            [
                np.asarray(np.linspace(1.0, 0.3, t.size), dtype=float),
                np.asarray(np.linspace(0.0, 0.7, t.size), dtype=float),
            ]
        )
        return {
            "t": t,
            "Y": y,
            "species_names": ["A", "B"],
            "algebra_scalars": {},
            "algebra_errors": [],
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": dict(worker._solver_config),
            "provenance": {},
            "fallback_occurred": False,
            "fallback_message": None,
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.SimulationWorker",
        make_simulation_worker_stub(payload_factory=_payload, emit_progress=(100, "done")),
    )

    sliders = main_window._mechanism_editor._variable_sliders
    sliders.update_variable("k1", 2.0)
    main_window._on_variable_changed("k1", 2.0)
    qtbot.waitUntil(lambda: bool(main_window.simulation_controller.run_state.pending_slider_simulation), timeout=1000)

    display_lengths.clear()
    main_window.simulation_controller._run_simulation()
    qtbot.waitUntil(lambda: 100 in display_lengths, timeout=1000)
    QtWidgets.QApplication.processEvents()

    assert 100 in display_lengths
    assert 350 not in display_lengths
