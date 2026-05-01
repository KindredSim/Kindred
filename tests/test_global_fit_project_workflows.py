from __future__ import annotations

import numpy as np
import pytest

from tests.test_global_fit_phase2_defaults_dataset_mgmt_ic_editor import (
    _latest_fit_window,
    _make_fit_result,
    _patch_message_box_exec,
    _seed_simple_mechanism,
    _seed_two_datasets,
    _show_only_batch_set,
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
    window = _latest_fit_window(main_window)
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

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    _patch_message_box_exec(monkeypatch)

    prompt_actions: list[str] = []
    programmatic_calls: list[str] = []

    def _prompt(action_text: str, concentration_rows=None):
        prompt_actions.append(str(action_text))
        return "cancel"

    monkeypatch.setattr(main_window, "_prompt_slider_transaction_invalidation", _prompt)
    monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", lambda: programmatic_calls.append("called"))

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
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


def test_global_fit_apply_to_project_initial_conditions_invalidate_cached_selection(
    main_window,
    monkeypatch,
    qt_app,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    _patch_message_box_exec(monkeypatch)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None
        set_id, set_name = _show_only_batch_set(main_window, row=int(ds1_row), qt_app=qt_app)
        ds2_set_id = str(ds2_settings.batch_set_id)
        ds2_set_name = str(main_window._batch_store.set_name_for_row(int(ds2_row)))

        cache = main_window.simulation_controller.batch_cache
        cache_key = "fit-apply-ic-stale-cache"
        cache.result_cache[f"{cache_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        }
        cache.result_cache[f"{cache_key}::{ds2_set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([4.0, 3.5], dtype=float)},
            "algebra_scalars": {},
        }
        cache.active_cache_key = cache_key
        cache.active_cache_valid_set_ids = (set_id, ds2_set_id)
        cache.active_cache_invalidated_set_ids = None
        main_window._batch_model.set_row_shown(int(ds1_row), True)
        main_window._batch_model.set_row_shown(int(ds2_row), True)
        main_window.set_active_batch_selection(set_id, set_name, [set_id, ds2_set_id])

        main_window._refresh_batch_display_from_focus_and_shown()
        qt_app.processEvents()

        assert main_window.active_batch_selection() == (set_id, set_name)
        assert main_window._plot_tabs._main_plot.export_payload() is not None

        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
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

        assert cache.active_cache_invalidated_set_ids == (set_id,)

        main_window._batch_model.set_row_shown(int(ds1_row), False)
        main_window._batch_model.set_row_shown(int(ds2_row), True)
        main_window.set_active_batch_selection(ds2_set_id, ds2_set_name, [ds2_set_id])
        main_window._refresh_batch_display_from_focus_and_shown()
        qt_app.processEvents()

        plot = main_window._plot_tabs._main_plot
        assert main_window.active_batch_selection()[0] == ds2_set_id
        assert np.allclose(
            np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
            np.asarray([4.0, 3.5], dtype=float),
        )
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_conditions_discards_only_affected_dirty_preview(
    main_window,
    monkeypatch,
    qt_app,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    _patch_message_box_exec(monkeypatch)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
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
                "result": _make_fit_result(
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
