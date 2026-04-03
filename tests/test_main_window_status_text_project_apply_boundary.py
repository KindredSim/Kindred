import json

import numpy as np
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
from kindred.gui.project_schema import PROJECT_DEFAULTS

pytestmark = pytest.mark.gui


def _load_project_via_file_dialog(main_window, tmp_path, monkeypatch, payload) -> None:
    project_path = tmp_path / "loaded_project.kin"
    project_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )
    main_window.project_controller.load_project()


def test_set_status_text_only_updates_label(main_window):
    main_window.set_status_text("Ready")
    assert "Ready" in main_window._status_label.text()


def test_apply_project_payload_restores_batch_initial_conditions(main_window):
    main_window._batch_store.set_species(["A", "B"])
    row = main_window._batch_store.ensure_set("set2")
    main_window._batch_store.set_value(row, "A", "1.23")
    main_window._batch_store.set_value(row, "B", "0.0")

    payload = main_window._serialize_project_state()

    main_window._batch_store = BatchInitialConditionsStore()
    assert "set2" not in main_window._batch_store.set_names()

    main_window._apply_project_payload(payload, record_undo=False)

    assert "set2" in main_window._batch_store.set_names()
    restored_row = main_window._batch_store.row_for_set("set2")
    assert restored_row is not None
    assert float(main_window._batch_store.get_value(int(restored_row), "A")) == pytest.approx(1.23)


def test_apply_project_payload_legacy_named_inline_initials_materialize_named_batch_sets(main_window):
    payload = main_window._serialize_project_state()
    payload.pop("batch_initial_conditions", None)
    payload["mechanism"] = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "randomname3 = {",
            "[A] = 1.0",
            "[B] = 0.0",
            "}",
            "",
            "set-two = {",
            "[A] = 2.5",
            "[B] = 0.5",
            "}",
        ]
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._batch_store.set_names()[:2] == ["randomname3", "set-two"]
    assert "set1" not in main_window._batch_store.set_names()[:2]
    row_random = main_window._batch_store.row_for_set("randomname3")
    row_two = main_window._batch_store.row_for_set("set-two")
    assert row_random is not None
    assert row_two is not None
    assert float(main_window._batch_store.get_value(int(row_random), "A")) == pytest.approx(1.0)
    assert float(main_window._batch_store.get_value(int(row_two), "A")) == pytest.approx(2.5)
    rewritten = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "randomname3 = {" not in rewritten
    assert "set-two = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (randomname3)" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set-two)" in rewritten


def test_apply_project_payload_named_inline_initials_preserve_spaced_batch_set_name(main_window):
    payload = main_window._serialize_project_state()
    payload.pop("batch_initial_conditions", None)
    payload["mechanism"] = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "Set B = {",
            "[A] = 1.5",
            "[B] = 0.25",
            "}",
        ]
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._batch_store.set_names()[0] == "Set B"
    row = main_window._batch_store.row_for_set("Set B")
    assert row is not None
    assert float(main_window._batch_store.get_value(int(row), "A")) == pytest.approx(1.5)
    rewritten = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "Set B = {" not in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (Set B)" in rewritten


def test_apply_project_payload_empty_algebra_multiword_block_does_not_create_phantom_batch_set(main_window):
    payload = main_window._serialize_project_state()
    payload.pop("batch_initial_conditions", None)
    payload["mechanism"] = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Algebra",
            "let config = {",
            "}",
        ]
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._batch_store.set_names()[0] == "set1"
    assert main_window._batch_store.row_for_set("let config") is None
    rewritten = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "let config = {" in rewritten
    assert "\n}\n" in f"\n{rewritten}\n"
    assert "Initial concentrations moved to Batch Initial Conditions table (let config)" not in rewritten


def test_apply_project_payload_empty_named_inline_initials_materialize_zero_filled_named_batch_sets(main_window):
    payload = main_window._serialize_project_state()
    payload.pop("batch_initial_conditions", None)
    payload["mechanism"] = "\n".join(
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

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._batch_store.set_names()[:2] == ["set2", "Set B"]
    row_set2 = main_window._batch_store.row_for_set("set2")
    row_set_b = main_window._batch_store.row_for_set("Set B")
    assert row_set2 is not None
    assert row_set_b is not None
    assert float(main_window._batch_store.get_value(int(row_set2), "A")) == pytest.approx(0.0)
    assert float(main_window._batch_store.get_value(int(row_set2), "B")) == pytest.approx(0.0)
    assert float(main_window._batch_store.get_value(int(row_set_b), "A")) == pytest.approx(0.0)
    assert float(main_window._batch_store.get_value(int(row_set_b), "B")) == pytest.approx(0.0)
    assert main_window._batch_store.row_for_set("let config") is None
    rewritten = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "set2 = {" not in rewritten
    assert "Set B = {" not in rewritten
    assert "let config = {" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (set2)" in rewritten
    assert "Initial concentrations moved to Batch Initial Conditions table (Set B)" in rewritten


def test_apply_project_payload_merges_empty_default_named_block_with_legacy_initials(main_window):
    payload = main_window._serialize_project_state()
    payload.pop("batch_initial_conditions", None)
    payload["mechanism"] = "\n".join(
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

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._batch_store.set_names()[0] == "set1"
    row = main_window._batch_store.row_for_set("set1")
    assert row is not None
    assert float(main_window._batch_store.get_value(int(row), "A")) == pytest.approx(1.0)
    assert float(main_window._batch_store.get_value(int(row), "B")) == pytest.approx(0.5)
    rewritten = main_window._mechanism_editor._reactions_text.toPlainText()
    assert "set1 = {" not in rewritten
    assert "[A] = 1.0" not in rewritten
    assert "[B] = 0.5" not in rewritten
    assert rewritten.count("Initial concentrations moved to Batch Initial Conditions table (set1)") == 2


def test_load_project_dialog_does_not_push_app_undo_text_entries(main_window, tmp_path, monkeypatch):
    baseline_payload = main_window._serialize_project_state()
    baseline_payload["mechanism"] = "reaction: A -> B; k=1.0"
    baseline_payload["notes"] = "baseline notes"

    loaded_payload = dict(baseline_payload)
    loaded_payload["mechanism"] = "reaction: A -> B; k=2.0"
    loaded_payload["notes"] = "loaded project notes"

    main_window._apply_project_payload(baseline_payload, record_undo=False)
    main_window._undo_stack.clear()

    _load_project_via_file_dialog(main_window, tmp_path, monkeypatch, loaded_payload)

    assert main_window._mechanism_editor._reactions_text.toPlainText() == loaded_payload["mechanism"]
    assert main_window._mechanism_editor._notes_text.toPlainText() == loaded_payload["notes"]
    assert main_window._undo_stack.count() == 0
    assert main_window._undo_stack.canUndo() is False


def test_load_project_dialog_clears_stale_app_undo_and_redo_history(main_window, tmp_path, monkeypatch):
    baseline_payload = main_window._serialize_project_state()
    baseline_payload["mechanism"] = "reaction: A -> B; k=1.0"
    baseline_payload["notes"] = "baseline notes"

    loaded_payload = dict(baseline_payload)
    loaded_payload["mechanism"] = "reaction: A -> B; k=2.0"
    loaded_payload["notes"] = "loaded project notes"

    main_window._apply_project_payload(baseline_payload, record_undo=False)
    main_window._set_text_with_optional_undo(
        main_window._mechanism_editor._notes_text,
        "stale app undo notes",
        "Stale notes edit",
        True,
    )
    assert main_window._undo_stack.canUndo() is True

    main_window._undo_stack.undo()
    assert main_window._undo_stack.canRedo() is True
    assert main_window._mechanism_editor._notes_text.toPlainText() == baseline_payload["notes"]

    _load_project_via_file_dialog(main_window, tmp_path, monkeypatch, loaded_payload)

    assert main_window._mechanism_editor._reactions_text.toPlainText() == loaded_payload["mechanism"]
    assert main_window._mechanism_editor._notes_text.toPlainText() == loaded_payload["notes"]
    assert main_window._undo_stack.count() == 0
    assert main_window._undo_stack.canUndo() is False
    assert main_window._undo_stack.canRedo() is False

    main_window._undo()
    assert main_window._mechanism_editor._notes_text.toPlainText() == loaded_payload["notes"]
    assert main_window._status_label.text() == "Nothing to undo"

    main_window._redo()
    assert main_window._mechanism_editor._notes_text.toPlainText() == loaded_payload["notes"]
    assert main_window._status_label.text() == "Nothing to redo"


def test_load_project_dialog_legacy_initial_migration_keeps_single_source_of_truth(
    main_window,
    tmp_path,
    monkeypatch,
):
    legacy_payload = main_window._serialize_project_state()
    legacy_payload.pop("batch_initial_conditions", None)
    legacy_payload["mechanism"] = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "",
            "# Initial concentrations",
            "[A] = 1.0",
            "[B] = 0.5",
        ]
    )

    _load_project_via_file_dialog(main_window, tmp_path, monkeypatch, legacy_payload)

    assert main_window._undo_stack.count() == 0
    assert main_window._undo_stack.canUndo() is False

    serialized = main_window.serialize_project_state()
    mechanism_text = serialized["mechanism"]
    assert "[A] = 1.0" not in mechanism_text
    assert "[B] = 0.5" not in mechanism_text
    assert "Initial concentrations moved to Batch Initial Conditions table (set1)" in mechanism_text

    batch_payload = serialized["batch_initial_conditions"]
    set1_payload = batch_payload["sets"][0]
    assert set1_payload["name"] == "set1"
    assert float(set1_payload["values"]["A"]) == pytest.approx(1.0)
    assert float(set1_payload["values"]["B"]) == pytest.approx(0.5)


def test_apply_pending_init_migration_emits_rows_inserted_when_named_sets_append(main_window, qt_app):
    main_window._batch_model.set_species(["A"])
    assert main_window._batch_model.rowCount() == 1

    inserted: list[tuple[int, int]] = []
    main_window._batch_model.rowsInserted.connect(
        lambda _parent, first, last: inserted.append((int(first), int(last)))
    )

    rewrite = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Initial concentrations moved to Batch Initial Conditions table (randomname3). Edit there.",
            "# Initial concentrations moved to Batch Initial Conditions table (set-two). Edit there.",
        ]
    )

    applied = main_window.apply_pending_init_migration(
        seed_sets={
            "randomname3": {"A": 1.0},
            "set-two": {"A": 2.5},
        },
        rewrite=rewrite,
    )
    qt_app.processEvents()

    assert applied is True
    assert inserted == [(1, 1)]
    assert main_window._batch_model.rowCount() == 2
    assert main_window._batch_table.model().rowCount() == 2
    assert main_window._batch_store.set_names()[:2] == ["randomname3", "set-two"]


def test_apply_project_payload_syncs_solver_runtime_state_and_visible_combo(main_window):
    payload = main_window._serialize_project_state()
    payload["solver"] = "BDF"
    payload["rtol"] = 1e-5
    payload["atol"] = 1e-9

    main_window._solver_method_combo.setCurrentText("LSODA")
    main_window._initial_rtol = 1e-6
    main_window._initial_atol = 1e-12

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "BDF"
    assert main_window._initial_rtol == pytest.approx(1e-5)
    assert main_window._initial_atol == pytest.approx(1e-9)
    assert main_window._solver_method_combo.currentText() == "BDF"


def test_apply_project_payload_invalid_tolerances_preserve_existing_runtime_values(main_window):
    payload = main_window._serialize_project_state()
    payload["solver"] = "BDF"
    payload["rtol"] = "bad-rtol"
    payload["atol"] = "bad-atol"

    main_window._solver_method_combo.setCurrentText("LSODA")
    main_window._initial_rtol = 1e-6
    main_window._initial_atol = 1e-12

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "BDF"
    assert main_window._initial_rtol == pytest.approx(1e-6)
    assert main_window._initial_atol == pytest.approx(1e-12)
    assert main_window._solver_method_combo.currentText() == "BDF"


def test_apply_project_payload_clears_dirty_session_state(main_window, qt_app):
    payload = main_window._serialize_project_state()

    data_panel = main_window._right_panel._data_manager
    dataset_payload = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "species": {"A": np.asarray([1.0, 0.5], dtype=float)},
    }
    data_panel._on_csv_loaded("dirty.csv", dataset_payload)
    qt_app.processEvents()

    main_window.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([2.0, 1.0], dtype=float)},
        label="dirty-run",
        overlays=[],
    )
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    cache = main_window.simulation_controller.batch_cache
    cache.result_cache["dirty-result::set1"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
    }
    cache.preview_cache["dirty-preview::set1"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([0.8, 0.4], dtype=float)},
    }
    cache.active_cache_key = "dirty-result"
    cache.active_preview_cache_key = "dirty-preview"
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = "set1-id"
    cache.last_display_selection = ["set1-id"]

    assert "dirty.csv" in data_panel.get_datasets()
    assert data_panel._dataset_list.count() == 1
    assert main_window._dataset_manager._dataset_views
    assert main_window._dataset_manager._fit_settings
    assert main_window._dataset_manager._dataset_panel_map
    assert main_window._plot_tabs._dataset_plots
    assert getattr(main_window._plot_tabs._grid_view, "_datasets", [])
    assert plot.export_payload() is not None
    assert getattr(plot, "_overlay_datasets", {})
    assert len(cache.result_cache) == 1
    assert len(cache.preview_cache) == 1

    main_window._apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert data_panel.get_datasets() == {}
    assert data_panel._dataset_list.count() == 0
    assert data_panel.get_selected_dataset() == (None, None)
    assert not data_panel._preview_label.isVisible()
    assert not hasattr(data_panel, "_mapping_widget")

    assert main_window._dataset_manager._dataset_views == {}
    assert main_window._dataset_manager._fit_settings == {}
    assert main_window._dataset_manager._dataset_panel_map == {}
    assert main_window._plot_tabs._dataset_plots == []
    assert getattr(main_window._plot_tabs._grid_view, "_datasets", []) == []

    assert plot.export_payload() is None
    assert getattr(plot, "_overlay_datasets", {}) == {}

    assert len(cache.result_cache) == 0
    assert len(cache.preview_cache) == 0
    assert cache.active_cache_key is None
    assert cache.active_preview_cache_key is None
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None
    assert cache.last_display_selection == []

    current_plot = main_window._plot_tabs.get_current_plot()
    assert current_plot is main_window._plot_tabs._main_plot
    assert isinstance(current_plot, QtWidgets.QWidget)


def test_apply_project_payload_closes_tracked_fit_windows(main_window, qt_app):
    payload = main_window._serialize_project_state()

    class _TrackedFitWindow(QtWidgets.QDialog):
        def __init__(self, parent):
            super().__init__(parent)
            self.close_events = 0
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        def closeEvent(self, event):
            self.close_events += 1
            super().closeEvent(event)

    fit_window = _TrackedFitWindow(main_window)
    main_window._register_fit_window(fit_window)
    assert fit_window.isVisible() is True

    main_window._apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert fit_window.close_events == 1
    qt_app.processEvents()
    assert (not shiboken6.isValid(fit_window)) or (fit_window.isVisible() is False)


def test_apply_project_payload_dirty_slider_cancel_aborts_public_apply(main_window, monkeypatch):
    payload = main_window._serialize_project_state()
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    apply_calls: list[tuple[dict, bool]] = []
    monkeypatch.setattr(main_window, "_apply_project_payload", lambda data, record_undo=True: apply_calls.append((data, record_undo)))
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "cancel",
        raising=False,
    )

    outcome = main_window.apply_project_payload(payload, record_undo=False)

    assert outcome is False
    assert apply_calls == []
    assert main_window._preview_session.has_dirty_transaction() is True


def test_apply_project_payload_dirty_slider_commit_continues_after_transaction_clears(main_window, monkeypatch):
    payload = main_window._serialize_project_state()
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    apply_calls: list[tuple[dict, bool]] = []
    commit_calls: list[str] = []

    monkeypatch.setattr(main_window, "_apply_project_payload", lambda data, record_undo=True: apply_calls.append((data, record_undo)))

    def _commit() -> None:
        commit_calls.append("commit")
        main_window._preview_session.clear_working_transaction()

    monkeypatch.setattr(main_window, "_on_commit_slider_overrides_clicked", _commit)
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "commit",
        raising=False,
    )

    outcome = main_window.apply_project_payload(payload, record_undo=False)

    assert outcome is True
    assert commit_calls == ["commit"]
    assert apply_calls == [(payload, False)]


def test_apply_project_payload_legacy_file_resets_leaky_keys(main_window):
    """Loading a legacy .kin file missing solver/worker keys must reset to schema defaults."""
    # Set non-default values to detect leaks
    main_window._initial_solver = "BDF"
    main_window._initial_rtol = 0.1
    main_window._initial_atol = 0.1
    main_window._use_sparse_jacobian = True
    main_window._wegscheider_cyclicity_enabled = True
    main_window._sim_controller.parallel_batch.max_parallel_workers = 99
    main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker = False

    # Minimal legacy payload — only mechanism and batch
    legacy_payload = {
        "mechanism": "A -> B; k=1",
        "batch_initial_conditions": {"sets": {"set1": {"A": 1.0}}, "species": ["A"]},
    }
    main_window._apply_project_payload(legacy_payload, record_undo=False)

    # Solver state is applied via _apply_solver_runtime_state, which sets
    # _initial_solver. The value sent was PROJECT_DEFAULTS['solver'].
    assert main_window._initial_solver == PROJECT_DEFAULTS["solver"]
    assert main_window._initial_rtol == PROJECT_DEFAULTS["rtol"]
    assert main_window._initial_atol == PROJECT_DEFAULTS["atol"]
    assert main_window._use_sparse_jacobian == PROJECT_DEFAULTS["use_sparse_jacobian"]
    assert main_window._wegscheider_cyclicity_enabled == PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"]
    assert (
        main_window._sim_controller.parallel_batch.max_parallel_workers
        == PROJECT_DEFAULTS["max_parallel_batch_workers"]
    )
    assert (
        main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker
        == PROJECT_DEFAULTS["limit_blas_threads_per_worker"]
    )


# ── Three-tier precedence tests ──────────────────────────────────────


def test_new_project_inherits_user_preferences_not_factory_defaults(main_window, monkeypatch):
    """New Project uses the user's QSettings-based preferences for dual-persisted keys."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/solver", "BDF")
    settings.setValue("simulation/rtol", 1e-5)
    settings.setValue("simulation/temperature", 310.0)
    settings.setValue("simulation/points", 200)
    settings.sync()

    main_window.config_controller.load_settings()

    # Verify user preferences captured
    assert main_window.config_controller._user_preferences["solver"] == "BDF"
    assert main_window.config_controller._user_preferences["temperature_K"] == pytest.approx(310.0)

    # Simulate New Project (mock QMessageBox to auto-discard)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.Discard,
    )
    main_window.project_controller.new_project()

    # Dual-persisted keys should reflect user preferences, not factory defaults
    assert main_window._initial_solver == "BDF"
    assert main_window._initial_rtol == pytest.approx(1e-5)
    assert main_window._temperature_spinbox.value() == pytest.approx(310.0)
    assert main_window._num_points_spinbox.value() == 200


def test_legacy_load_falls_back_to_user_preferences(main_window):
    """A .kin file missing a key falls back to user preferences, not factory defaults."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/solver", "BDF")
    settings.setValue("simulation/temperature", 350.0)
    settings.sync()

    main_window.config_controller.load_settings()

    minimal_payload = {"mechanism": "A -> B ; k1=1"}
    main_window._apply_project_payload(minimal_payload)

    assert main_window._initial_solver == "BDF"
    assert main_window._temperature_spinbox.value() == pytest.approx(350.0)


def test_document_load_does_not_contaminate_user_preferences(main_window):
    """Loading a .kin file changes live state but must not modify _user_preferences."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/solver", "Radau")
    settings.sync()

    main_window.config_controller.load_settings()
    assert main_window.config_controller._user_preferences["solver"] == "Radau"

    payload = {"mechanism": "A -> B ; k1=1", "solver": "BDF", "temperature_K": 500.0}
    main_window._apply_project_payload(payload)

    # Live state reflects document
    assert main_window._initial_solver == "BDF"
    assert main_window._temperature_spinbox.value() == pytest.approx(500.0)

    # User preferences are untouched
    assert main_window.config_controller._user_preferences["solver"] == "Radau"
    assert main_window.config_controller._user_preferences["temperature_K"] == pytest.approx(298.15)


def test_save_settings_writes_user_preferences_not_live_document_state(main_window):
    """save_settings persists user preferences, so loading a .kin cannot leak into QSettings."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/solver", "Radau")
    settings.sync()

    main_window.config_controller.load_settings()

    payload = {"mechanism": "A -> B ; k1=1", "solver": "BDF"}
    main_window._apply_project_payload(payload)
    assert main_window._initial_solver == "BDF"

    main_window.config_controller.save_settings()

    assert settings.value("simulation/solver") == "Radau"


def test_dialog_update_user_preference_roundtrips(main_window):
    """update_user_preference stores values that save_settings persists."""
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.config_controller.load_settings()
    main_window.config_controller.update_user_preference("solver", "LSODA")
    main_window.config_controller.update_user_preference("rtol", 1e-8)

    main_window.config_controller.save_settings()

    assert settings.value("simulation/solver") == "LSODA"
    assert settings.value("simulation/rtol", type=float) == pytest.approx(1e-8)


def test_spinbox_edit_updates_user_preference_when_not_applying_document(main_window):
    """Direct spinbox edits update user preferences when _applying_document is False."""
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.config_controller.load_settings()
    assert main_window._applying_document is False

    main_window._temperature_spinbox.setValue(400.0)
    assert main_window.config_controller._user_preferences["temperature_K"] == pytest.approx(400.0)

    main_window._num_points_spinbox.setValue(500)
    assert main_window.config_controller._user_preferences["num_points"] == 500


def test_spinbox_during_project_apply_does_not_update_user_preferences(main_window):
    """_apply_project_payload sets _applying_document=True so spinbox signals skip preferences."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/temperature", 298.15)
    settings.setValue("simulation/points", 100)
    settings.sync()

    main_window.config_controller.load_settings()
    original_temp = main_window.config_controller._user_preferences["temperature_K"]
    original_points = main_window.config_controller._user_preferences["num_points"]

    payload = {"mechanism": "", "temperature_K": 500.0, "num_points": 999}
    main_window._apply_project_payload(payload)

    # Live state reflects document
    assert main_window._temperature_spinbox.value() == pytest.approx(500.0)
    assert main_window._num_points_spinbox.value() == 999

    # User preferences unchanged
    assert main_window.config_controller._user_preferences["temperature_K"] == pytest.approx(original_temp)
    assert main_window.config_controller._user_preferences["num_points"] == original_points
