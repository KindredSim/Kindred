import json

import numpy as np
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
from kindred.core.simulation_identity import canonical_initials_fingerprint
from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot
from kindred.gui.controllers.simulation_run_state import PreviewOwnershipState
from kindred.gui.ports import SliderReplayIntent
from kindred.gui.project_schema import PROJECT_DEFAULTS

pytestmark = pytest.mark.gui


def _runtime_snapshot(
    *,
    mode: str,
    ready: bool,
    status: str | None = None,
    required: bool = True,
) -> RuntimeReadinessSnapshot:
    ready_value = bool(ready)
    required_value = bool(required)
    status_value = str(status or ("ready" if ready_value else "warming"))
    return RuntimeReadinessSnapshot(
        mode=str(mode),
        status=status_value,
        ready=ready_value,
        generation=1,
        required=required_value,
        controls_ready=bool(ready_value or not required_value),
        polling=bool(required_value and not ready_value and status_value != "failed"),
    )


class _TransitionTestWorker(QtCore.QObject):
    def __init__(self, *, running: bool = True, fast_mode: bool = False, request_id: int = 0) -> None:
        super().__init__()
        self._running = bool(running)
        self._fast_mode = bool(fast_mode)
        self._request_id = int(request_id)
        self.cancel_calls = 0
        self.terminate_calls = 0

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._running = False

    def wait(self, _ms: int | None = None) -> bool:
        return True

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._running = False


def _completion_payload() -> dict[str, object]:
    return {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "Y": np.asarray([[1.0, 0.5], [0.0, 0.5]], dtype=float),
        "species_names": ["A", "B"],
        "algebra_scalars": {},
        "mechanism": None,
        "mechanism_text": "reaction: A -> B; k=1.0",
        "solver_config": {},
        "fallback_occurred": False,
        "fallback_message": None,
    }


def _load_project_via_file_dialog(main_window, tmp_path, monkeypatch, payload) -> None:
    project_path = tmp_path / "loaded_project.kin"
    project_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )
    main_window.project_controller.load_project()


def _arm_pending_preview_state(main_window) -> tuple[str, int]:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    table = main_window._batch_table
    model = main_window._batch_model
    selection = table.selectionModel()
    index = model.index(0, 0)
    table.setCurrentIndex(index)
    selection.clearSelection()
    selection.select(index, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    owner = main_window._preview_session
    controller = main_window.simulation_controller
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    owner.sync_committed_slider_values({"k1": 1.0})
    owner.stage_slider_value("k1", 2.0, target_set_ids=[set0_id])
    owner.submit_slider_replay_intent(
        SliderReplayIntent(target_set_ids=(set0_id,), source="variable_slider"),
        preserve_existing_request=True,
    )

    request_id = controller.run_state.pending_slider_preview_launch.request_id
    assert request_id is not None
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(request_id),
        epoch=1,
        target_set_ids=(set0_id,),
    )
    controller._active_run_id = 11
    controller._latest_sim_request_id = int(request_id)
    controller._queue_slider_plot_update(
        set_id=set0_id,
        cache_key="project-apply-pending-preview-cache",
        request_id=int(request_id),
        run_id=11,
        slider_triggered=True,
    )
    return set0_id, int(request_id)


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

    main_window._solver_method_combo.setCurrentText("BDF")
    main_window._initial_rtol = 1e-6
    main_window._initial_atol = 1e-12

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "BDF"
    assert main_window._initial_rtol == pytest.approx(1e-5)
    assert main_window._initial_atol == pytest.approx(1e-9)
    assert main_window._solver_method_combo.currentText() == "BDF"


def test_apply_project_payload_normalizes_invalid_solver_to_bdf(main_window):
    payload = main_window._serialize_project_state()
    payload["solver"] = "unknown_solver_name"

    main_window._solver_method_combo.setCurrentText("Radau")
    main_window._initial_solver = "Radau"

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "BDF"
    assert main_window._solver_method_combo.currentText() == "BDF"


def test_apply_solver_runtime_state_normalizes_unknown_solver_and_syncs_combo(main_window):
    main_window._solver_method_combo.setCurrentText("Radau")
    main_window._initial_solver = "Radau"

    main_window.apply_solver_runtime_state(solver="unknown_solver_name")

    assert main_window._initial_solver == "BDF"
    assert main_window._solver_method_combo.currentText() == "BDF"


@pytest.mark.parametrize(
    ("invalid_rtol", "invalid_atol"),
    [
        ("garbage", "also-garbage"),
        (None, None),
        ("", ""),
        ("   ", "   "),
        ("nan", "nan"),
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
        (0, 0),
        (-1.0, -1.0),
        (float("nan"), float("nan")),
    ],
)
def test_apply_solver_runtime_state_preserves_active_tolerances_for_invalid_values(
    main_window,
    invalid_rtol,
    invalid_atol,
):
    main_window._initial_rtol = 1e-9
    main_window._initial_atol = 1e-11

    main_window.apply_solver_runtime_state(rtol=invalid_rtol, atol=invalid_atol)

    assert main_window._initial_rtol == pytest.approx(1e-9)
    assert main_window._initial_atol == pytest.approx(1e-11)


def test_apply_solver_runtime_state_accepts_extreme_finite_tolerances(main_window):
    main_window._initial_rtol = 1e-9
    main_window._initial_atol = 1e-11

    main_window.apply_solver_runtime_state(rtol=1e-300, atol=1e300)

    assert main_window._initial_rtol == pytest.approx(1e-300)
    assert main_window._initial_atol == pytest.approx(1e300)


@pytest.mark.parametrize(
    ("kwargs", "expected_rtol", "expected_atol"),
    [
        ({"rtol": "bad-rtol", "atol": 1e-8}, 1e-9, 1e-8),
        ({"rtol": 1e-7, "atol": "bad-atol"}, 1e-7, 1e-11),
    ],
)
def test_apply_solver_runtime_state_mixed_validity_updates_fields_independently(
    main_window,
    kwargs,
    expected_rtol,
    expected_atol,
):
    main_window._initial_rtol = 1e-9
    main_window._initial_atol = 1e-11

    main_window.apply_solver_runtime_state(**kwargs)

    assert main_window._initial_rtol == pytest.approx(expected_rtol)
    assert main_window._initial_atol == pytest.approx(expected_atol)


def test_apply_project_payload_invalid_tolerances_preserve_active_runtime_values(main_window):
    payload = main_window._serialize_project_state()
    payload["solver"] = "BDF"
    payload["rtol"] = "bad-rtol"
    payload["atol"] = "bad-atol"

    main_window._solver_method_combo.setCurrentText("BDF")
    main_window._initial_rtol = 1e-9
    main_window._initial_atol = 1e-11

    main_window._apply_project_payload(payload, record_undo=False)

    assert main_window._initial_solver == "BDF"
    assert main_window._initial_rtol == pytest.approx(1e-9)
    assert main_window._initial_atol == pytest.approx(1e-11)
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
    main_window._sim_controller.batch_runtime_lane_budget = 99
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
        main_window._sim_controller.batch_runtime_lane_budget
        == PROJECT_DEFAULTS["batch_runtime_lane_budget"]
    )
    assert (
        main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker
        == PROJECT_DEFAULTS["limit_blas_threads_per_worker"]
    )


def test_apply_project_payload_missing_simulation_keys_use_user_preferences(main_window):
    main_window.config_controller.update_user_preference("solver", "Radau")
    main_window.config_controller.update_user_preference("rtol", 1e-4)
    main_window.config_controller.update_user_preference("atol", 1e-8)
    main_window.config_controller.update_user_preference("use_sparse_jacobian", False)
    main_window.config_controller.update_user_preference("wegscheider_cyclicity_enabled", False)
    main_window.config_controller.update_user_preference("max_parallel_batch_workers", 7)
    main_window.config_controller.update_user_preference("batch_runtime_lane_budget", 5)
    main_window.config_controller.update_user_preference("limit_blas_threads_per_worker", False)

    legacy_payload = {
        "mechanism": "A -> B; k=1",
        "batch_initial_conditions": {"sets": {"set1": {"A": 1.0}}, "species": ["A"]},
    }

    main_window._apply_project_payload(legacy_payload, record_undo=False)

    assert main_window._initial_solver == "Radau"
    assert main_window._initial_rtol == pytest.approx(1e-4)
    assert main_window._initial_atol == pytest.approx(1e-8)
    assert main_window._use_sparse_jacobian is False
    assert main_window._wegscheider_cyclicity_enabled is False
    assert (
        main_window._sim_controller.parallel_batch.max_parallel_workers
        == 7
    )
    assert main_window._sim_controller.batch_runtime_lane_budget == 5
    assert (
        main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker
        is False
    )


def test_apply_project_payload_invalidates_parallel_pool_when_worker_settings_change(main_window):
    calls: list[None] = []
    main_window._sim_controller.parallel_batch_pool_settings_changed = lambda: calls.append(None)
    main_window._sim_controller.parallel_batch.max_parallel_workers = 2
    main_window._sim_controller.batch_runtime_lane_budget = 2
    main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker = True

    payload = {
        "mechanism": "A -> B; k=1",
        "batch_initial_conditions": {"sets": {"set1": {"A": 1.0}}, "species": ["A"]},
        "max_parallel_batch_workers": 7,
        "batch_runtime_lane_budget": 5,
        "limit_blas_threads_per_worker": False,
    }

    main_window._apply_project_payload(payload, record_undo=False)

    assert int(main_window._sim_controller.parallel_batch.max_parallel_workers) == 7
    assert int(main_window._sim_controller.batch_runtime_lane_budget) == 5
    assert bool(main_window._sim_controller.parallel_batch.limit_blas_threads_per_worker) is False
    assert calls == [None]


def test_apply_project_payload_clamps_parallel_workers_to_shared_ceiling(main_window):
    from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

    payload = {
        "mechanism": "A -> B; k=1",
        "batch_initial_conditions": {"sets": {"set1": {"A": 1.0}}, "species": ["A"]},
        "max_parallel_batch_workers": 200,
        "batch_runtime_lane_budget": 200,
    }

    main_window._apply_project_payload(payload, record_undo=False)

    assert (
        main_window._sim_controller.parallel_batch.max_parallel_workers
        == int(MAX_PARALLEL_WORKERS_CEILING)
    )
    assert (
        main_window._sim_controller.batch_runtime_lane_budget
        == int(MAX_PARALLEL_WORKERS_CEILING)
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


def test_legacy_load_missing_simulation_keys_uses_tier2_preferences(main_window):
    """A .kin file missing simulation keys restores the user's tier-2 preferences."""
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
    main_window.config_controller.update_user_preference("solver", "BDF")
    main_window.config_controller.update_user_preference("rtol", 1e-8)

    main_window.config_controller.save_settings()

    assert settings.value("simulation/solver") == "BDF"
    assert settings.value("simulation/rtol", type=float) == pytest.approx(1e-8)


def test_spinbox_edit_updates_user_preference_when_not_applying_document(main_window):
    """Direct spinbox edits update user preferences when _suppress_preference_updates is False."""
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.config_controller.load_settings()
    assert main_window._suppress_preference_updates is False

    main_window._temperature_spinbox.setValue(400.0)
    assert main_window.config_controller._user_preferences["temperature_K"] == pytest.approx(400.0)

    main_window._num_points_spinbox.setValue(500)
    assert main_window.config_controller._user_preferences["num_points"] == 500


def test_spinbox_during_project_apply_does_not_update_user_preferences(main_window):
    """_apply_project_payload sets _suppress_preference_updates=True so spinbox signals skip preferences."""
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


def test_reset_project_apply_dirty_session_state_clears_pending_preview_state_without_display_state(main_window):
    controller = main_window.simulation_controller
    set0_id, request_id = _arm_pending_preview_state(main_window)

    assert main_window.main_plot_has_data() is False
    assert controller.run_state.pending_slider_preview_launch.active is True
    assert controller.run_state.preview_ownership.request_id == request_id
    assert controller._plot_coalescer.pending.set_ids == {set0_id}

    main_window._reset_project_apply_dirty_session_state()

    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()
    ownership = controller.run_state.preview_ownership
    assert ownership.request_id is None
    assert ownership.target_set_ids == ()
    queued_plot = controller._plot_coalescer.pending
    assert queued_plot.set_ids == set()
    assert queued_plot.request_id is None
    assert queued_plot.cache_key is None


def test_solver_combo_change_persists_to_qsettings(main_window):
    """Changing the solver combo updates user preferences so save_settings persists the value."""
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.config_controller.load_settings()
    original_solver = main_window.config_controller._user_preferences.get("solver")

    # Simulate user changing the solver combo.
    combo = main_window._solver_method_combo
    target_solver = "BDF" if original_solver != "BDF" else "Radau"
    idx = combo.findText(target_solver)
    assert idx >= 0, f"Solver {target_solver!r} not found in combo"
    combo.setCurrentIndex(idx)

    assert main_window.config_controller._user_preferences["solver"] == target_solver

    main_window.config_controller.save_settings()
    assert settings.value("simulation/solver") == target_solver


def test_solver_unchanged_during_document_apply(main_window):
    """Applying a project payload with a different solver must not modify user preferences."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/solver", "Radau")
    settings.sync()

    main_window.config_controller.load_settings()
    assert main_window.config_controller._user_preferences["solver"] == "Radau"

    payload = {"mechanism": "A -> B ; k1=1", "solver": "BDF"}
    main_window._apply_project_payload(payload)

    assert main_window._initial_solver == "BDF"
    assert main_window.config_controller._user_preferences["solver"] == "Radau"


def test_load_settings_suppresses_preference_updates(main_window):
    """load_settings must not trigger update_user_preference via spinbox signals."""
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/temperature", 350.0)
    settings.setValue("simulation/points", 200)
    settings.sync()

    calls = []
    original_update = main_window.config_controller.update_user_preference

    def spy(key, value):
        calls.append((key, value))
        original_update(key, value)

    main_window.config_controller.update_user_preference = spy
    main_window._suppress_preference_updates = True
    try:
        main_window.config_controller.load_settings()
    finally:
        main_window._suppress_preference_updates = False
    main_window.config_controller.update_user_preference = original_update

    assert len(calls) == 0, f"update_user_preference called {len(calls)} times during load_settings: {calls}"


def test_bootstrap_window_state_suppresses_preference_updates(main_window):
    """The production bootstrap path must suppress preference updates during load."""
    calls = []
    original = main_window.config_controller.update_user_preference
    main_window.config_controller.update_user_preference = lambda k, v: calls.append((k, v))
    try:
        main_window._bootstrap_window_state()
    finally:
        main_window.config_controller.update_user_preference = original
    assert len(calls) == 0, f"update_user_preference called during bootstrap: {calls}"


def test_bootstrap_window_state_schedules_runtime_warm_without_blocking_hidden_startup(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    batch_prewarm_calls: list[bool] = []
    interactive_prewarm_calls: list[bool] = []

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_prewarm_calls.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_prewarm_calls.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window._bootstrap_window_state()

    assert scheduled == []
    assert interactive_prewarm_calls == [False]
    assert batch_prewarm_calls == []


def test_project_apply_schedules_runtime_warm_without_blocking_visible_load(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    interactive_prewarm_calls: list[bool] = []
    apply_calls: list[str] = []

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window,
        "_apply_project_payload_inner",
        lambda data, *, record_undo=True: apply_calls.append("apply"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_prewarm_calls.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window.show()
    main_window._apply_project_payload({}, record_undo=False)

    assert apply_calls == ["apply"]
    assert scheduled == []
    assert interactive_prewarm_calls == [False]


def test_startup_runtime_availability_callback_ignores_close_started_window(main_window, monkeypatch):
    calls: list[str] = []
    main_window._simulation_runtime_availability_shutdown_started = True
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: calls.append("interactive"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: calls.append("batch"),
    )

    main_window._make_simulation_runtime_available_after_startup()

    assert calls == []


def test_runtime_availability_refresh_schedules_exact_runtimes_without_blocking(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    interactive_waits: list[bool] = []
    batch_waits: list[bool] = []

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_ready",
        lambda *, fast_mode: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window.show()
    main_window._set_runtime_backed_controls_ready(False)
    main_window._schedule_simulation_runtime_availability_refresh(wait=False)

    assert scheduled == []
    assert interactive_waits
    assert all(wait is False for wait in interactive_waits)
    assert batch_waits
    assert all(wait is False for wait in batch_waits)
    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()


def test_runtime_readiness_does_not_override_non_runtime_run_disabled_state(main_window, monkeypatch):
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_ready",
        lambda *, fast_mode: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window.set_run_button_enabled(False)
    main_window._set_runtime_backed_run_controls_ready(True)

    assert not main_window._run_btn.isEnabled()


def test_draft_reactions_typing_does_not_schedule_runtime_warm(main_window, monkeypatch):
    interactive_warms: list[bool] = []
    batch_warms: list[bool] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    main_window.set_run_button_enabled(True)
    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_warms.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_warms.append(bool(wait)),
    )

    main_window._set_mechanism_edit_locked(False)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=2.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._on_reactions_text_changed_for_main_window()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert interactive_warms == []
    assert batch_warms == []


def test_authoritative_mechanism_commit_schedules_runtime_rewarm_after_invalidating_controls(
    main_window,
    monkeypatch,
):
    interactive_warms: list[bool] = []
    batch_warms: list[bool] = []
    scheduled: list[tuple[int, object]] = []
    readiness_by_mode = {False: False, True: False}

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    main_window.set_run_button_enabled(True)
    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_warms.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_warms.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=bool(readiness_by_mode[bool(fast_mode)]),
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=bool(readiness_by_mode[True])),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=bool(readiness_by_mode[False])),
    )

    main_window._set_mechanism_edit_locked(False)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=2.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    assert main_window._try_lock_mechanism_editor()

    assert not main_window._run_btn.isEnabled()
    assert not main_window._mechanism_editor._variable_sliders.isEnabled()
    assert interactive_warms
    assert all(wait is False for wait in interactive_warms)
    assert batch_warms
    assert all(wait is False for wait in batch_warms)
    assert scheduled == [(50, main_window._poll_interactive_runtime_readiness_after_refresh)]

    readiness_by_mode[False] = True
    readiness_by_mode[True] = True
    scheduled.clear()
    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert main_window._mechanism_editor.species_sliders_widget().isEnabled()
    assert scheduled == []


def test_authoritative_mechanism_commit_invalidates_display_before_scheduling_rewarm(
    main_window,
    monkeypatch,
):
    events: list[str] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    main_window.set_run_button_enabled(True)
    main_window.simulation_controller.batch_cache.active_cache_key = "active-result"

    monkeypatch.setattr(
        main_window,
        "_invalidate_active_results_after_authoritative_mechanism_change",
        lambda **_kwargs: events.append("invalidate_display"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "supersede_active_work_for_authoritative_mechanism_transition",
        lambda *, epoch: events.append(f"supersede:{int(epoch)}"),
    )
    monkeypatch.setattr(
        main_window,
        "_refresh_authoritative_mechanism_derived_ui",
        lambda: events.append("refresh_derived"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: events.append("schedule_rewarm"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )

    main_window._set_mechanism_edit_locked(False)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=2.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    assert main_window._try_lock_mechanism_editor()

    assert events[:4] == ["supersede:1", "invalidate_display", "refresh_derived", "schedule_rewarm"]


def test_programmatic_mechanism_load_invalidates_display_before_scheduling_rewarm(
    main_window,
    monkeypatch,
):
    events: list[str] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    main_window.set_run_button_enabled(True)
    main_window.simulation_controller.batch_cache.active_cache_key = "active-result"

    monkeypatch.setattr(
        main_window,
        "_invalidate_active_results_after_authoritative_mechanism_change",
        lambda **_kwargs: events.append("invalidate_display"),
    )
    monkeypatch.setattr(
        main_window,
        "_refresh_authoritative_mechanism_derived_ui",
        lambda: events.append("refresh_derived"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: events.append("schedule_rewarm"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )

    main_window._on_programmatic_mechanism_load()

    assert events[-1] == "schedule_rewarm"
    assert "invalidate_display" in events[:-1]
    assert "refresh_derived" in events[:-1]
    assert events.index("invalidate_display") < events.index("refresh_derived")
    assert "schedule_rewarm" not in events[:-1]


def test_programmatic_mechanism_load_supersedes_in_flight_work_without_active_display(
    main_window,
    monkeypatch,
):
    controller = main_window.simulation_controller
    events: list[str] = []

    monkeypatch.setattr(main_window, "_authoritative_mechanism_has_active_display", lambda: False)
    monkeypatch.setattr(main_window, "_schedule_simulation_runtime_availability_refresh", lambda *, wait=False: None)
    monkeypatch.setattr(main_window, "_refresh_authoritative_mechanism_derived_ui", lambda: None)
    monkeypatch.setattr(
        controller,
        "supersede_active_work_for_authoritative_mechanism_transition",
        lambda *, epoch: events.append(f"supersede:{int(epoch)}"),
        raising=False,
    )
    monkeypatch.setattr(
        controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )

    main_window._apply_authoritative_mechanism_transition(
        force_runtime_invalidation=True,
    )

    assert events == ["supersede:1"]


def test_direct_authoritative_editor_rewrite_supersedes_no_display_explicit_completion(
    main_window,
    monkeypatch,
):
    controller = main_window.simulation_controller
    worker = _TransitionTestWorker(running=True, fast_mode=False, request_id=21)
    published: list[object] = []
    old_run_id = 7
    old_request_id = 21

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    controller._run_sequence_id = old_run_id
    controller._active_run_id = old_run_id
    controller._latest_sim_request_id = old_request_id
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = worker

    monkeypatch.setattr(main_window, "set_data", lambda *args, **kwargs: published.append((args, kwargs)))
    monkeypatch.setattr(main_window, "_authoritative_mechanism_has_active_display", lambda: False)
    monkeypatch.setattr(main_window, "_schedule_simulation_runtime_availability_refresh", lambda *, wait=False: None)

    main_window._set_authoritative_mechanism_editor_texts(
        reactions_text="reaction: A -> C; k=2.0",
        state_network_dsl="",
        description="Test direct authoritative rewrite",
    )
    controller.on_simulation_complete(
        _completion_payload(),
        run_id=old_run_id,
        fast_mode=False,
        request_id=old_request_id,
    )

    assert worker.cancel_calls == 1
    assert controller._active_run_id != old_run_id
    assert published == []


def test_authoritative_transition_rejects_old_preview_completion_without_active_display(
    main_window,
    monkeypatch,
):
    controller = main_window.simulation_controller
    set_id, old_request_id = _arm_pending_preview_state(main_window)
    old_run_id = int(controller._active_run_id)
    worker = _TransitionTestWorker(running=True, fast_mode=True, request_id=old_request_id)
    published: list[object] = []

    controller._simulation_worker = worker
    controller._simulation_running = True
    controller._slider_simulation_active = True

    monkeypatch.setattr(main_window, "set_data", lambda *args, **kwargs: published.append((args, kwargs)))
    monkeypatch.setattr(main_window, "_authoritative_mechanism_has_active_display", lambda: False)
    monkeypatch.setattr(main_window, "_schedule_simulation_runtime_availability_refresh", lambda *, wait=False: None)

    main_window._apply_authoritative_mechanism_transition(
        force_runtime_invalidation=True,
        transition_source="test_preview_stale_rejection",
    )
    controller.on_simulation_complete(
        _completion_payload(),
        run_id=old_run_id,
        fast_mode=True,
        request_id=old_request_id,
        batch_set_id=set_id,
        cache_key="project-apply-pending-preview-cache",
    )

    assert controller.run_state.latest_sim_request_id > old_request_id
    assert controller.run_state.preview_ownership.request_id is None
    assert controller.run_state.simulation_running is False
    assert controller.run_state.slider_simulation_active is False
    assert published == []


def test_template_load_supersedes_no_display_explicit_completion(
    main_window,
    monkeypatch,
):
    controller = main_window.simulation_controller
    worker = _TransitionTestWorker(running=True, fast_mode=False, request_id=31)
    published: list[object] = []
    old_run_id = 9
    old_request_id = 31

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    controller._run_sequence_id = old_run_id
    controller._active_run_id = old_run_id
    controller._latest_sim_request_id = old_request_id
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = worker

    monkeypatch.setattr(main_window, "set_data", lambda *args, **kwargs: published.append((args, kwargs)))
    monkeypatch.setattr(main_window, "_authoritative_mechanism_has_active_display", lambda: False)
    monkeypatch.setattr(main_window, "_schedule_simulation_runtime_availability_refresh", lambda *, wait=False: None)

    main_window._load_template_from_manager("reaction: A -> C; k=3.0")
    controller.on_simulation_complete(
        _completion_payload(),
        run_id=old_run_id,
        fast_mode=False,
        request_id=old_request_id,
    )

    assert worker.cancel_calls == 1
    assert controller._active_run_id != old_run_id
    assert published == []


def test_slider_materialization_supersedes_no_display_explicit_completion(
    main_window,
    monkeypatch,
):
    controller = main_window.simulation_controller
    worker = _TransitionTestWorker(running=True, fast_mode=False, request_id=41)
    published: list[object] = []
    events: list[str] = []
    old_run_id = 13
    old_request_id = 41
    original_supersede = controller.supersede_active_work_for_authoritative_mechanism_transition

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    controller._run_sequence_id = old_run_id
    controller._active_run_id = old_run_id
    controller._latest_sim_request_id = old_request_id
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = worker

    monkeypatch.setattr(main_window, "set_data", lambda *args, **kwargs: published.append((args, kwargs)))
    monkeypatch.setattr(main_window, "_authoritative_mechanism_has_active_display", lambda: False)
    monkeypatch.setattr(
        controller,
        "supersede_active_work_for_authoritative_mechanism_transition",
        lambda *, epoch: (events.append(f"supersede:{int(epoch)}"), original_supersede(epoch=epoch)),
    )
    monkeypatch.setattr(
        main_window,
        "_schedule_simulation_runtime_availability_refresh",
        lambda *, wait=False: events.append("schedule_rewarm"),
    )

    main_window._apply_effective_slider_values_to_mechanism_editors(
        {"k1": 2.0},
        description="Test slider materialization",
    )
    main_window._sync_after_authoritative_slider_materialization()
    controller.on_simulation_complete(
        _completion_payload(),
        run_id=old_run_id,
        fast_mode=False,
        request_id=old_request_id,
    )

    assert worker.cancel_calls == 1
    assert events[0].startswith("supersede:")
    assert events[-1] == "schedule_rewarm"
    assert published == []


def test_project_apply_state_network_transition_defers_rewarm_until_payload_finishes(
    main_window,
    monkeypatch,
):
    events: list[str] = []
    payload = main_window._serialize_project_state()
    payload["mechanism"] = "reaction: A -> C; k=2.0"
    payload["state_network"] = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: C, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,C",
        ]
    )
    original_programmatic_load = main_window._on_programmatic_mechanism_load

    def _record_programmatic_load(*, schedule_runtime_refresh: bool = True) -> None:
        events.append("programmatic_load_start")
        original_programmatic_load(schedule_runtime_refresh=schedule_runtime_refresh)
        events.append("programmatic_load_done")

    main_window._mechanism_runtime_transition.reset_current_snapshot(
        main_window._authoritative_mechanism_transition_snapshot()
    )
    monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", _record_programmatic_load)
    monkeypatch.setattr(
        main_window.simulation_controller,
        "supersede_active_work_for_authoritative_mechanism_transition",
        lambda *, epoch: events.append(f"supersede:{int(epoch)}"),
    )
    monkeypatch.setattr(
        main_window,
        "_schedule_simulation_runtime_availability_refresh",
        lambda *, wait=False: events.append("schedule_rewarm"),
    )

    main_window._apply_project_payload(payload, record_undo=False)

    assert "programmatic_load_done" in events
    assert "schedule_rewarm" in events
    assert events.index("schedule_rewarm") > events.index("programmatic_load_done")
    assert any(event.startswith("supersede:") for event in events[: events.index("schedule_rewarm")])


def test_programmatic_load_resets_transition_canonical_batch_initials_baseline(
    main_window,
    monkeypatch,
):
    main_window._batch_model.set_species(["A"])
    assert main_window._batch_model.setData(main_window._batch_model.index(0, 1), "2.5")
    set_id = str(main_window._batch_set_id_for_row(0) or "")
    assert set_id

    captured: list[dict[str, object]] = []
    original_apply = main_window._mechanism_runtime_transition.apply_authoritative_transition

    def _capture_transition(snapshot, **kwargs):
        captured.append(dict(kwargs))
        return original_apply(snapshot, **kwargs)

    monkeypatch.setattr(
        main_window._mechanism_runtime_transition,
        "apply_authoritative_transition",
        _capture_transition,
    )

    main_window._on_programmatic_mechanism_load(schedule_runtime_refresh=False)

    assert captured
    canonical_map = captured[-1].get("canonical_batch_initials_by_set_id")
    assert isinstance(canonical_map, dict)
    assert canonical_map[set_id] == canonical_initials_fingerprint({"A": 2.5})


def test_project_apply_defers_authoritative_rewarm_until_payload_finishes(
    main_window,
    monkeypatch,
):
    events: list[str] = []

    main_window.show()
    monkeypatch.setattr(
        main_window,
        "_refresh_authoritative_mechanism_derived_ui",
        lambda: events.append("refresh_derived"),
    )
    monkeypatch.setattr(
        main_window,
        "_apply_project_payload_inner",
        lambda data, *, record_undo=True: (
            events.append("apply_inner_start"),
            main_window._apply_authoritative_mechanism_transition(schedule_runtime_refresh=False),
            events.append("apply_inner_done"),
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: events.append("schedule_rewarm"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: None,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window._apply_project_payload({}, record_undo=False)

    assert events == [
        "apply_inner_start",
        "refresh_derived",
        "apply_inner_done",
        "schedule_rewarm",
    ]


def test_runtime_readiness_poll_enables_run_before_preview_sliders(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    readiness_by_mode = {False: True, True: False}

    main_window._set_runtime_backed_controls_ready(False)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=bool(readiness_by_mode[bool(fast_mode)]),
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=bool(readiness_by_mode[True])),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: True,
    )

    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert not main_window._mechanism_editor._variable_sliders.isEnabled()
    assert not main_window._mechanism_editor.species_sliders_widget().isEnabled()
    assert scheduled == [(50, main_window._poll_interactive_runtime_readiness_after_refresh)]

    readiness_by_mode[True] = True
    scheduled.clear()
    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert main_window._mechanism_editor.species_sliders_widget().isEnabled()
    assert scheduled == []


def test_runtime_readiness_poll_enables_single_set_run_without_batch_runtime(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    batch_ready = {"value": False}
    interactive_waits: list[bool] = []
    batch_waits: list[bool] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(False)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="ordinary", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: bool(batch_ready["value"]),
    )

    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert interactive_waits == []
    assert batch_waits
    assert all(wait is False for wait in batch_waits)
    assert scheduled == []

    scheduled.clear()
    batch_ready["value"] = True
    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert scheduled == []


def test_show_event_keeps_single_set_run_ready_after_hidden_serial_warm_without_batch(main_window, monkeypatch, qtbot):
    scheduled: list[tuple[int, object]] = []
    interactive_waits: list[bool] = []
    batch_waits: list[bool] = []
    batch_ready = {"value": False}

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtimes_ready",
        lambda: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(
            mode="preview" if bool(fast_mode) else "ordinary",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "parallel_batch_runtime_ready",
        lambda: bool(batch_ready["value"]),
    )

    main_window._schedule_simulation_runtime_availability_refresh(wait=False, force_when_hidden=True)

    assert interactive_waits
    assert all(wait is False for wait in interactive_waits)
    assert batch_waits == []
    assert main_window._run_btn.isEnabled()

    main_window.show()
    qtbot.waitUntil(lambda: main_window.isVisible(), timeout=1000)

    assert batch_waits
    assert all(wait is False for wait in batch_waits)
    assert main_window._run_btn.isEnabled()
    assert scheduled == []


def test_multiset_selection_gates_run_and_schedules_runtime_readiness(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []
    batch_ready = {"value": False}
    interactive_waits: list[bool] = []
    batch_waits: list[bool] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_interactive_simulation_runtimes_available",
        lambda *, wait=False: interactive_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "ensure_parallel_batch_pool_eagerly_created",
        lambda *, wait=False: batch_waits.append(bool(wait)),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_ready",
        lambda *, fast_mode: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "interactive_simulation_runtime_snapshot",
        lambda *, fast_mode: _runtime_snapshot(mode="preview", ready=True),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="batch", ready=bool(batch_ready["value"])),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_uses_parallel_batch_runtime",
        lambda: True,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="batch", ready=bool(batch_ready["value"])),
    )

    main_window._on_batch_selection_changed()

    main_window.set_run_button_enabled(True)
    assert not main_window._run_btn.isEnabled()
    assert not main_window._mechanism_editor._variable_sliders.isEnabled()
    assert interactive_waits
    assert batch_waits
    assert all(wait is False for wait in interactive_waits)
    assert all(wait is False for wait in batch_waits)
    assert scheduled == [(50, main_window._poll_interactive_runtime_readiness_after_refresh)]

    scheduled.clear()
    batch_ready["value"] = True
    main_window._poll_interactive_runtime_readiness_after_refresh()

    assert main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert scheduled == []


def test_batch_selection_applies_failed_runtime_status_without_polling(main_window, monkeypatch):
    scheduled: list[tuple[int, object]] = []

    main_window.show()
    main_window._set_runtime_backed_controls_ready(True)
    main_window.set_run_button_enabled(True)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    monkeypatch.setattr(
        main_window.simulation_controller,
        "selected_run_runtime_snapshot",
        lambda: _runtime_snapshot(mode="batch", ready=False, status="failed"),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "slider_preview_runtime_snapshot",
        lambda: _runtime_snapshot(mode="preview", ready=True),
    )

    main_window._on_batch_selection_changed()

    assert not main_window._run_btn.isEnabled()
    assert main_window._mechanism_editor._variable_sliders.isEnabled()
    assert "Simulation runtime failed to prepare" in main_window._status_label.text()
    assert scheduled == []
