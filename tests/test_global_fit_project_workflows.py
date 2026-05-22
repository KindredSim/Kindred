from __future__ import annotations

import numpy as np
import pytest

from kindred.gui.ports import DisplayRefreshSource, DisplayTransitionOutcomeKind
from tests.workflow_helpers import (
    completion_provenance_payload,
    latest_fit_window,
    make_fit_result,
    patch_message_box_exec,
    seed_simple_mechanism,
    seed_two_datasets,
)


pytestmark = [pytest.mark.gui]


def test_global_fit_launch_creates_and_seeds_batch_set_from_dataset_t0(main_window, monkeypatch):
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

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._run_global_fit()
    window = latest_fit_window(main_window)
    try:
        row = main_window._batch_store.row_for_set("dataset1")
        assert row is not None
        assert float(main_window._batch_store.get_value(int(row), "A")) == pytest.approx(1.23)
        assert float(main_window._batch_store.get_value(int(row), "B")) == pytest.approx(0.0)

        settings = main_window._dataset_manager.get_fit_settings("dataset1_1")
        assert settings.batch_set == "dataset1"
        assert window in list(getattr(main_window, "_active_fit_windows", []) or [])
    finally:
        window.close()


def test_global_fit_apply_to_project_parameters_only_respects_dirty_slider_guard(main_window, monkeypatch):
    from PySide6 import QtWidgets

    seed_two_datasets(main_window)
    seed_simple_mechanism(main_window)
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    patch_message_box_exec(monkeypatch)

    prompt_actions: list[str] = []
    programmatic_calls: list[str] = []

    def _prompt(action_text: str, concentration_rows=None):
        prompt_actions.append(str(action_text))
        return "cancel"

    monkeypatch.setattr(main_window, "_prompt_slider_transaction_invalidation", _prompt)
    monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", lambda: programmatic_calls.append("called"))

    main_window._run_global_fit()
    window = latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": make_fit_result(
                    k_value=0.55,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Parameters only")
        button.click()

        assert prompt_actions == ["Applying fitted parameters to the project"]
        assert programmatic_calls == []
        assert main_window._preview_session.has_dirty_transaction() is True
        assert "k=0.2" in main_window._mechanism_editor._reactions_text.toPlainText()
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_conditions_discards_only_affected_dirty_preview(
    main_window,
    monkeypatch,
    qt_app,
):
    from PySide6 import QtWidgets

    seed_two_datasets(main_window)
    seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    patch_message_box_exec(monkeypatch)

    main_window._run_global_fit()
    window = latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None
        ds1_set_id = str(ds1_settings.batch_set_id)
        ds2_set_id = str(ds2_settings.batch_set_id)

        assert main_window._preview_session.stage_concentration_value_for_rows(
            [int(ds1_row)],
            species="A",
            value=6.5,
        ) is True
        assert main_window._preview_session.stage_concentration_value_for_rows(
            [int(ds2_row)],
            species="A",
            value=7.5,
        ) is True
        assert main_window._preview_session.has_dirty_state_for_set(ds1_set_id) is True
        assert main_window._preview_session.has_dirty_state_for_set(ds2_set_id) is True

        window._handle_global_fit_complete(
            {
                "result": make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": 2.5}},
                )
            }
        )

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Initial conditions only")
        button.click()
        qt_app.processEvents()

        assert float(main_window._batch_store.get_value(int(ds1_row), "A")) == pytest.approx(2.5)
        assert main_window._preview_session.has_dirty_state_for_set(ds1_set_id) is False
        assert main_window._preview_session.has_dirty_state_for_set(ds2_set_id) is True
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_conditions_invalidates_only_affected_cached_request_scope(
    main_window,
    monkeypatch,
    qt_app,
):
    from PySide6 import QtWidgets

    seed_two_datasets(main_window)
    seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    patch_message_box_exec(monkeypatch)

    main_window._run_global_fit()
    window = latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None
        ds1_set_id = str(ds1_settings.batch_set_id)
        ds2_set_id = str(ds2_settings.batch_set_id)

        cache = main_window.simulation_controller.batch_cache
        cache_key = "fit-apply-ic-request-cache"
        t = np.asarray([0.0, 1.0], dtype=float)
        ds1_series = {"A": np.asarray([1.0, 0.5], dtype=float)}
        ds2_series = {"A": np.asarray([4.0, 3.5], dtype=float)}
        mechanism_text = main_window._get_mechanism_text()
        cache.put_completion_entry(
            cache_key=cache_key,
            set_id=ds1_set_id,
            is_preview=False,
            t=t,
            series=ds1_series,
            mechanism_text=mechanism_text,
            completion_provenance=completion_provenance_payload(
                t=t,
                series=ds1_series,
                mechanism_text=mechanism_text,
            ),
            owned_species=("A",),
        )
        cache.put_completion_entry(
            cache_key=cache_key,
            set_id=ds2_set_id,
            is_preview=False,
            t=t,
            series=ds2_series,
            mechanism_text=mechanism_text,
            completion_provenance=completion_provenance_payload(
                t=t,
                series=ds2_series,
                mechanism_text=mechanism_text,
            ),
            owned_species=("A",),
        )
        cache.apply_explicit_cache_reconciliation(
            clear_active_cache_identity_state=False,
            active_cache_key=cache_key,
            active_cache_preview_token=None,
            active_cache_preview_scope_set_ids=None,
            active_cache_valid_set_ids=(ds1_set_id, ds2_set_id),
            active_cache_invalidated_set_ids=(),
        )
        for row in range(main_window._batch_model.rowCount()):
            row_set_id = str(main_window._batch_store.set_id_for_row(int(row)))
            main_window._batch_model.set_row_requested_show(
                int(row),
                row_set_id in {ds1_set_id, ds2_set_id},
            )

        outcome = main_window.results_controller.publish_cached_batch_display_scope(
            cache_key=cache_key,
            requested_show_set_ids=(ds1_set_id, ds2_set_id),
            prefer_set=ds1_set_id,
            display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
        )
        assert outcome.transition_outcome is not None
        assert outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

        window._handle_global_fit_complete(
            {
                "result": make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": 2.5}},
                )
            }
        )

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Initial conditions only")
        button.click()
        qt_app.processEvents()

        assert cache.active_cache_key == cache_key
        assert cache.active_cache_invalidated_set_ids == (ds1_set_id,)

        for row in range(main_window._batch_model.rowCount()):
            row_set_id = str(main_window._batch_store.set_id_for_row(int(row)))
            main_window._batch_model.set_row_requested_show(int(row), row_set_id == ds2_set_id)
        ds2_outcome = main_window.results_controller.publish_cached_batch_display_scope(
            cache_key=cache_key,
            requested_show_set_ids=(ds2_set_id,),
            prefer_set=ds2_set_id,
            display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
        )
        assert ds2_outcome.transition_outcome is not None
        assert ds2_outcome.transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED

        active = main_window.results_controller.active_display_transaction()
        assert active is not None
        assert active.display_set_ids == (ds2_set_id,)
        plot = main_window._plot_tabs._main_plot
        assert np.allclose(
            np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
            ds2_series["A"],
        )
    finally:
        window.close()


def test_global_fit_rejects_stale_terminal_result_before_project_apply(main_window, monkeypatch):
    seed_two_datasets(main_window)
    seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    patch_message_box_exec(monkeypatch)

    main_window._run_global_fit()
    window = latest_fit_window(main_window)
    try:
        window._fit_run_state_owner.set_active_run_stamp_hash("current-fit-run")
        before_result = getattr(window, "_last_result", None)

        window._handle_global_fit_complete(
            {
                "result": make_fit_result(
                    k_value=9.0,
                    dataset_initials={"ds1": {"init:A": 9.0}, "ds2": {"init:A": 8.0}},
                ),
                "run_stamp_hash": "old-fit-run",
            }
        )

        assert getattr(window, "_last_result", None) is before_result
        assert window._apply_to_project_button.isEnabled() is False

        current_result = make_fit_result(
            k_value=0.66,
            dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
        )
        window._handle_global_fit_complete(
            {
                "result": current_result,
                "run_stamp_hash": "current-fit-run",
            }
        )

        assert window._last_result is current_result
        assert window._apply_to_project_button.isEnabled() is True
    finally:
        window.close()
