from __future__ import annotations

import hashlib

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def _seed_two_datasets(main_window) -> None:
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    t = np.linspace(0.0, 1.0, 6)
    data_panel._datasets.update(
        {
            "ds1": {"t": t.copy(), "species": {"A": np.linspace(1.0, 0.5, t.size), "B": np.linspace(0.2, 0.9, t.size)}},
            "ds2": {"t": t.copy(), "species": {"A": np.linspace(0.8, 0.1, t.size), "B": np.linspace(0.0, 0.4, t.size)}},
        }
    )


def _seed_simple_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )


def _latest_fit_window(main_window):
    windows = list(getattr(main_window, "_active_fit_windows", []) or [])
    assert windows, "Expected Global Fit window to be registered"
    return windows[-1]


def _show_only_batch_set(main_window, *, row: int, qt_app) -> tuple[str, str]:
    model = main_window._batch_model
    table = main_window._batch_table
    for row_index in range(model.rowCount()):
        model.set_row_shown(row_index, row_index == int(row))
    table.selectRow(int(row))
    qt_app.processEvents()
    set_id = str(main_window._batch_store.set_id_for_row(int(row)))
    set_name = str(main_window._batch_store.set_name_for_row(int(row)))
    return set_id, set_name


def _set_fit_targets_dataset(panel, *, dataset_id: str) -> None:
    from PySide6 import QtCore, QtWidgets

    dataset_list = panel.window().findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
    assert dataset_list is not None
    for i in range(dataset_list.count()):
        item = dataset_list.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            dataset_list.setCurrentRow(i)
            return
    raise AssertionError(f"Dataset id not in list: {dataset_id!r}")


def _set_fit_targets_pending(panel, *, dataset_id: str, enabled_species: set[str], qt_app) -> None:
    from PySide6 import QtWidgets

    _set_fit_targets_dataset(panel, dataset_id=dataset_id)
    for checkbox in panel.findChildren(QtWidgets.QCheckBox):
        label = checkbox.text().strip()
        if label:
            checkbox.setChecked(label in enabled_species)
    qt_app.processEvents()


def _parameter_table_names(window) -> list[str]:
    table = getattr(window, "_param_table", None)
    assert table is not None
    names: list[str] = []
    for row in range(table.rowCount()):
        item = table.item(row, 2)
        if item is not None and item.text().strip():
            names.append(item.text().strip())
    return names


def _parameter_table_rows(window) -> list[dict[str, object]]:
    from PySide6 import QtCore

    table = getattr(window, "_param_table", None)
    assert table is not None
    rows: list[dict[str, object]] = []
    for row in range(table.rowCount()):
        name_item = table.item(row, 2)
        fit_item = table.item(row, 0)
        rows.append(
            {
                "name": "" if name_item is None else name_item.text().strip(),
                "fit": bool(fit_item is not None and fit_item.checkState() == QtCore.Qt.Checked),
            }
        )
    return rows


def _ic_table_species(window) -> list[str]:
    from PySide6 import QtWidgets
    from kindred.gui.fitting.parameters_ics_tab import _ICCol

    table = window.findChild(QtWidgets.QTableWidget, "global_fit_initial_conditions_table")
    assert table is not None
    species: list[str] = []
    for row in range(table.rowCount()):
        item = table.item(row, _ICCol.SPECIES)
        if item is not None and item.text().strip():
            species.append(item.text().strip())
    return species


def _make_fit_result(*, k_value: float, dataset_initials: dict[str, dict[str, float]]):
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

    model = np.linspace(1.0, 0.4, 6)
    residual = np.linspace(0.0, -0.1, 6)
    return GlobalFitResult(
        success=True,
        shared_params={"k1": float(k_value)},
        dataset_params={str(dataset_id): {str(name): float(value) for name, value in values.items()} for dataset_id, values in dataset_initials.items()},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.9,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            ),
            DatasetFitInfo(
                dataset_id="ds2",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            ),
        ],
        nfev=10,
        message="ok",
        covariance=None,
        objective_residuals=np.concatenate([residual.copy(), residual.copy()]),
        model_series={
            "ds1": {"A": model.copy()},
            "ds2": {"A": model.copy()},
        },
        residual_series={
            "ds1": {"A": residual.copy()},
            "ds2": {"A": residual.copy()},
        },
    )


def _make_shared_param_fit_result(*, param_name: str, param_value: float, dataset_initials: dict[str, dict[str, float]]):
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

    model = np.linspace(1.0, 0.4, 6)
    residual = np.linspace(0.0, -0.1, 6)
    return GlobalFitResult(
        success=True,
        shared_params={str(param_name): float(param_value)},
        dataset_params={str(dataset_id): {str(name): float(value) for name, value in values.items()} for dataset_id, values in dataset_initials.items()},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.9,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            ),
            DatasetFitInfo(
                dataset_id="ds2",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=residual.copy(),
                n_points=int(residual.size),
                weight=1.0,
            ),
        ],
        nfev=10,
        message="ok",
        covariance=None,
        objective_residuals=np.concatenate([residual.copy(), residual.copy()]),
        model_series={
            "ds1": {"A": model.copy()},
            "ds2": {"A": model.copy()},
        },
        residual_series={
            "ds1": {"A": residual.copy()},
            "ds2": {"A": residual.copy()},
        },
    )


def test_global_fit_opens_without_config_dialog_and_defaults_targets_none(main_window, monkeypatch):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)

    # Keep setup deterministic and lightweight.
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    class _DialogMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("GlobalFitConfigDialog must not be used in the launch flow")

    monkeypatch.setattr("kindred.gui.fitting.global_fit_config.GlobalFitConfigDialog", _DialogMustNotBeConstructed)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        assert window._dataset_table.rowCount() == 2

        # Default: no fit targets applied for any included dataset.
        assert window._fit_targets_selection_applied.get("ds1") == []
        assert window._fit_targets_selection_applied.get("ds2") == []

        # Default: all datasets included (Use checked), but Run Fit is blocked until Apply.
        assert window._run_button.isEnabled() is False
        footer = window.findChild(QtWidgets.QWidget, "global_fit_footer")
        run_reason = window.findChild(QtWidgets.QLabel, "global_fit_run_block_reason_label")
        assert footer is not None
        assert run_reason is not None
        assert footer.isAncestorOf(run_reason)
        assert run_reason.isVisible()
        data_text = run_reason.text().lower()
        assert "run fit disabled" in data_text
        assert "targets & weights" in data_text

        targets_idx = [window._tabs.tabText(i) for i in range(window._tabs.count())].index("Targets & Weights")
        window._tabs.setCurrentIndex(targets_idx)
        QtWidgets.QApplication.processEvents()
        blocked = window.findChild(QtWidgets.QLabel, "global_fit_fit_targets_run_blocked")
        assert blocked is not None
        assert blocked.isVisible()
        text = blocked.text().lower()
        assert "run fit disabled" in text
        assert "ds1" in text and "ds2" in text
    finally:
        window.close()


def test_global_fit_can_add_remove_datasets_in_window(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    class _DialogMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("GlobalFitConfigDialog must not be used in the launch flow")

    monkeypatch.setattr("kindred.gui.fitting.global_fit_config.GlobalFitConfigDialog", _DialogMustNotBeConstructed)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        assert window._dataset_table.rowCount() == 2
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"B"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        selector = window._subset_widget._selector
        assert selector.selected_dataset_species() == {"ds1": {"A"}, "ds2": {"B"}}

        window._remove_datasets_from_session(["ds2"])
        assert window._dataset_table.rowCount() == 1
        assert [entry["id"] for entry in window._dataset_entries] == ["ds1"]
        assert selector.selected_dataset_species() == {"ds1": {"A"}}

        window._add_datasets_to_session(["ds2"])
        assert window._dataset_table.rowCount() == 2
        ids = [entry["id"] for entry in window._dataset_entries]
        assert set(ids) == {"ds1", "ds2"}
        ds2_entry = next(entry for entry in window._dataset_entries if entry["id"] == "ds2")
        assert ds2_entry.get("include") is True
        assert float(ds2_entry.get("weight")) == pytest.approx(1.0)
        assert window._fit_targets_selection_applied.get("ds2") == []
        assert selector.selected_dataset_species() == {"ds1": {"A"}}
    finally:
        window.close()


def test_global_fit_initial_conditions_editor_apply_persists_to_dataset_manager(main_window, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    class _DialogMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("GlobalFitConfigDialog must not be used in the launch flow")

    monkeypatch.setattr("kindred.gui.fitting.global_fit_config.GlobalFitConfigDialog", _DialogMustNotBeConstructed)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        combo = window.findChild(QtWidgets.QComboBox, "global_fit_initial_conditions_dataset_combo")
        table = window.findChild(QtWidgets.QTableWidget, "global_fit_initial_conditions_table")
        apply_btn = window.findChild(QtWidgets.QPushButton, "global_fit_initial_conditions_apply")
        assert combo is not None
        assert table is not None
        assert apply_btn is not None

        # Select ds1 in the IC editor.
        for i in range(combo.count()):
            if str(combo.itemData(i)) == "ds1":
                combo.setCurrentIndex(i)
                break
        else:
            raise AssertionError("ds1 not present in Initial Conditions dataset combo")

        # Change initial(A) and enable fitting for init:A with bounds.
        from kindred.gui.fitting.parameters_ics_tab import _ICCol
        row_a = None
        for row in range(table.rowCount()):
            if (table.item(row, _ICCol.SPECIES) or QtWidgets.QTableWidgetItem()).text().strip() == "A":
                row_a = row
                break
        assert row_a is not None

        table.item(row_a, _ICCol.INITIAL).setText("2.5")  # Initial
        fit_item = table.item(row_a, _ICCol.FIT)
        assert fit_item is not None
        fit_item.setCheckState(QtCore.Qt.Checked)
        table.item(row_a, _ICCol.MIN).setText("0.1")
        table.item(row_a, _ICCol.MAX).setText("10.0")

        apply_btn.click()

        settings = main_window._dataset_manager.get_fit_settings("ds1")
        assert settings.initial_conditions.get("A") == pytest.approx(2.5)
        assert settings.fit_flags.get("A") is True
        assert settings.bounds.get("A") == pytest.approx((0.1, 10.0))

        # Ensure the fitting window rebuild path now reflects the updated IC settings.
        assert "init:A" in (window._global_dataset_variable_params.get("ds1") or {})
        spec = window._global_dataset_variable_params["ds1"]["init:A"]
        assert float(spec.get("initial")) == pytest.approx(2.5)
    finally:
        window.close()


def test_global_fit_apply_to_project_parameters_only_resyncs_main_window_immediately(main_window, monkeypatch):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        settings = main_window._dataset_manager.get_fit_settings("ds1")
        mapped_row = main_window._batch_store.row_for_set_id(str(settings.batch_set_id))
        assert mapped_row is not None
        batch_a_before = float(main_window._batch_store.get_value(int(mapped_row), "A"))

        programmatic_calls: list[str] = []
        extract_calls: list[bool] = []
        original_programmatic = main_window._on_programmatic_mechanism_load
        original_extract = main_window._extract_and_populate_variables

        def _spy_programmatic() -> None:
            programmatic_calls.append(main_window._mechanism_editor._reactions_text.toPlainText())
            original_programmatic()

        def _spy_extract(*, preserve_visibility: bool = False) -> None:
            extract_calls.append(bool(preserve_visibility))
            original_extract(preserve_visibility=bool(preserve_visibility))

        monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", _spy_programmatic)
        monkeypatch.setattr(main_window, "_extract_and_populate_variables", _spy_extract)

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

        assert "0.55" in main_window._mechanism_editor._reactions_text.toPlainText()
        assert programmatic_calls
        assert extract_calls == [True]
        assert float(main_window._batch_store.get_value(int(mapped_row), "A")) == pytest.approx(batch_a_before)
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_conditions_updates_batch_store_authority(main_window, monkeypatch):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        baseline_payload = main_window.serialize_project_state()
        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )

        serialized_during_review = main_window.serialize_project_state()
        assert serialized_during_review["mechanism"] == baseline_payload["mechanism"]
        assert serialized_during_review["batch_initial_conditions"] == baseline_payload["batch_initial_conditions"]
        assert "fitting" not in serialized_during_review
        assert "fit_results" not in serialized_during_review

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Initial conditions only")
        button.click()

        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None
        assert float(main_window._batch_store.get_value(int(ds1_row), "A")) == pytest.approx(2.5)
        assert float(main_window._batch_store.get_value(int(ds2_row), "A")) == pytest.approx(1.7)
        assert ds1_settings.initial_conditions.get("A") == pytest.approx(2.5)
        assert ds2_settings.initial_conditions.get("A") == pytest.approx(1.7)
        assert "k=0.2" in main_window._mechanism_editor._reactions_text.toPlainText()
    finally:
        window.close()


def test_global_fit_apply_to_project_parameters_scope_guards_dirty_slider_transaction(main_window, monkeypatch):
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
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

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


def test_global_fit_apply_to_project_parameter_noop_skips_slider_guard(main_window, monkeypatch):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.200000",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    info_calls: list[str] = []
    prompt_actions: list[str] = []
    programmatic_calls: list[str] = []
    setter_calls: list[str] = []
    refresh_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    def _prompt(action_text: str, concentration_rows=None):
        prompt_actions.append(str(action_text))
        return "cancel"

    monkeypatch.setattr(main_window, "_prompt_slider_transaction_invalidation", _prompt)
    monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", lambda: programmatic_calls.append("called"))
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: refresh_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=0.2,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Parameters only")
        button.click()

        assert prompt_actions == []
        assert len(info_calls) == info_count_before_apply + 1
        assert programmatic_calls == []
        assert setter_calls == []
        assert refresh_calls == []
        assert main_window._preview_session.has_dirty_transaction() is True
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_stale_authority_already_current_parameter_skips_slider_guard(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.55",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    info_calls: list[str] = []
    prompt_actions: list[str] = []
    programmatic_calls: list[str] = []
    setter_calls: list[str] = []
    refresh_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    def _prompt(action_text: str, concentration_rows=None):
        prompt_actions.append(str(action_text))
        return "cancel"

    monkeypatch.setattr(main_window, "_prompt_slider_transaction_invalidation", _prompt)
    monkeypatch.setattr(main_window, "_on_programmatic_mechanism_load", lambda: programmatic_calls.append("called"))
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: refresh_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

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
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Parameters only")
        button.click()

        assert prompt_actions == []
        assert len(info_calls) == info_count_before_apply + 1
        assert programmatic_calls == []
        assert setter_calls == []
        assert refresh_calls == []
        assert main_window._preview_session.has_dirty_transaction() is True
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_parameter_change_beyond_authoritative_precision_still_guards_dirty_slider_transaction(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=1000000.1234567",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 1000000.1234567, "min": 0.01, "max": 10000000.0}],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

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
                    k_value=1000000.1234568,
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
        assert "k=1000000.1234567" in main_window._mechanism_editor._reactions_text.toPlainText()
    finally:
        window.close()


def test_global_fit_apply_to_project_signed_zero_parameter_noop_skips_slider_guard(main_window, monkeypatch):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.0, "min": -1.0, "max": 1.0}],
    )

    info_calls: list[str] = []
    prompt_actions: list[str] = []
    setter_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    def _prompt(action_text: str, concentration_rows=None):
        prompt_actions.append(str(action_text))
        return "cancel"

    monkeypatch.setattr(main_window, "_prompt_slider_transaction_invalidation", _prompt)
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=-0.0,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Parameters only")
        button.click()

        assert prompt_actions == []
        assert len(info_calls) == info_count_before_apply + 1
        assert setter_calls == []
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_step_parameter_signed_zero_floor_still_guards_dirty_slider_transaction(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1e-300, kr=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("kf1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "kf1", "value": 0.0, "min": 1e-12, "max": 10.0}],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

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
                "result": _make_shared_param_fit_result(
                    param_name="kf1",
                    param_value=-0.0,
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
        assert "kf=1e-300" in main_window._mechanism_editor._reactions_text.toPlainText()
    finally:
        window.close()


def test_global_fit_apply_to_project_missing_step_parameter_warns_instead_of_reporting_success(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1, kr=2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("kf1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "kf1", "value": 1.0, "min": 0.01, "max": 10.0}],
    )

    warning_calls: list[tuple[str, str]] = []
    info_calls: list[str] = []
    prompt_actions: list[str] = []
    setter_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warning_calls.append((str(title), str(text)))
        or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda action_text, concentration_rows=None: prompt_actions.append(str(action_text)) or "cancel",
    )
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_shared_param_fit_result(
                    param_name="kf2",
                    param_value=3.5,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        info_count_before_apply = len(info_calls)
        combo.setCurrentText("Parameters only")
        button.click()

        assert warning_calls
        assert warning_calls[-1][0] == "Apply to Project"
        assert "kf2" in warning_calls[-1][1]
        assert "writable step" in warning_calls[-1][1]
        assert len(info_calls) == info_count_before_apply
        assert prompt_actions == []
        assert setter_calls == []
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_unwritable_derived_step_parameter_warns_instead_of_writing(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; kr=2, K=3",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [
            {"name": "kr1", "value": 2.0, "min": 0.01, "max": 10.0},
            {"name": "K1", "value": 3.0, "min": 0.01, "max": 10.0},
        ],
    )

    warning_calls: list[tuple[str, str]] = []
    info_calls: list[str] = []
    prompt_actions: list[str] = []
    setter_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warning_calls.append((str(title), str(text)))
        or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda action_text, concentration_rows=None: prompt_actions.append(str(action_text)) or "cancel",
    )
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_shared_param_fit_result(
                    param_name="kf1",
                    param_value=9.0,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        info_count_before_apply = len(info_calls)
        combo.setCurrentText("Parameters only")
        button.click()

        assert warning_calls
        assert warning_calls[-1][0] == "Apply to Project"
        assert "kf1" in warning_calls[-1][1]
        assert "no longer writable" in warning_calls[-1][1]
        assert len(info_calls) == info_count_before_apply
        assert prompt_actions == []
        assert setter_calls == []
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_step_canonicalization_only_rewrite_skips_guard_and_write(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "equilibrium: A <-> B ; k=1, kr=2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("kf1", 2.0)
    assert main_window._preview_session.has_dirty_transaction() is True

    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "kf1", "value": 1.0, "min": 0.01, "max": 10.0}],
    )

    warning_calls: list[tuple[str, str]] = []
    info_calls: list[str] = []
    prompt_actions: list[str] = []
    setter_calls: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warning_calls.append((str(title), str(text)))
        or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda action_text, concentration_rows=None: prompt_actions.append(str(action_text)) or "cancel",
    )
    monkeypatch.setattr(
        main_window,
        "set_mechanism_reactions_text_with_optional_undo",
        lambda *_args, **_kwargs: setter_calls.append("called"),
    )

    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        window._handle_global_fit_complete(
            {
                "result": _make_shared_param_fit_result(
                    param_name="kf1",
                    param_value=1.0,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
                )
            }
        )
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Parameters only")
        button.click()

        assert warning_calls == []
        assert prompt_actions == []
        assert setter_calls == []
        assert len(info_calls) == info_count_before_apply + 1
        assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text
    finally:
        window.close()


def test_global_fit_apply_to_project_step_warning_allows_valid_parameter_and_ic_updates(
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    warning_calls: list[tuple[str, str]] = []
    info_calls: list[str] = []
    prompt_actions: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: warning_calls.append((str(title), str(text)))
        or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda action_text, concentration_rows=None: prompt_actions.append(str(action_text)) or "cancel",
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        result = _make_fit_result(
            k_value=0.44,
            dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
        )
        result.shared_params["kf2"] = 3.5
        window._handle_global_fit_complete({"result": result})

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        info_count_before_apply = len(info_calls)
        combo.setCurrentText("Parameters and initial conditions")
        button.click()

        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None
        assert "k=0.44" in main_window._mechanism_editor._reactions_text.toPlainText()
        assert float(main_window._batch_store.get_value(int(ds1_row), "A")) == pytest.approx(2.5)
        assert float(main_window._batch_store.get_value(int(ds2_row), "A")) == pytest.approx(1.7)
        assert ds1_settings.initial_conditions.get("A") == pytest.approx(2.5)
        assert ds2_settings.initial_conditions.get("A") == pytest.approx(1.7)
        assert warning_calls
        assert warning_calls[-1][0] == "Apply to Project"
        assert "kf2" in warning_calls[-1][1]
        assert "writable step" in warning_calls[-1][1]
        assert len(info_calls) == info_count_before_apply
        assert prompt_actions == []
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_conditions_invalidates_stale_cached_result_before_refresh(
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
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        assert ds1_row is not None
        set_id, set_name = _show_only_batch_set(main_window, row=int(ds1_row), qt_app=qt_app)
        ds2_set_id = str(ds2_settings.batch_set_id)

        cache = main_window.simulation_controller.batch_cache
        cache_key = "fit-apply-ic-stale-cache"
        cache.result_cache[f"{cache_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        }
        cache.active_cache_key = cache_key
        cache.active_cache_valid_set_ids = (set_id,)
        cache.active_cache_invalidated_set_ids = None
        main_window.set_active_batch_selection(set_id, set_name, [set_id])

        main_window._refresh_batch_display_from_focus_and_shown()
        qt_app.processEvents()

        assert main_window.active_batch_selection() == (set_id, set_name)
        assert main_window._plot_tabs._main_plot.export_payload() is not None

        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": 2.5}, "ds2": {"init:A": 1.7}},
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

        assert set(cache.active_cache_invalidated_set_ids or ()) == {set_id, ds2_set_id}
        assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
        assert main_window.active_batch_selection() == ("", "")
        assert main_window._plot_tabs._main_plot.export_payload() is None
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_condition_noop_preserves_cached_display(
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

    info_calls: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None

        ds1_value = float(main_window._batch_store.get_value(int(ds1_row), "A"))
        ds2_value = float(main_window._batch_store.get_value(int(ds2_row), "A"))

        set_id, set_name = _show_only_batch_set(main_window, row=int(ds1_row), qt_app=qt_app)

        cache = main_window.simulation_controller.batch_cache
        cache_key = "fit-apply-ic-noop-cache"
        cache.result_cache[f"{cache_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        }
        cache.active_cache_key = cache_key
        cache.active_cache_valid_set_ids = (set_id,)
        cache.active_cache_invalidated_set_ids = None
        main_window.set_active_batch_selection(set_id, set_name, [set_id])

        main_window._refresh_batch_display_from_focus_and_shown()
        qt_app.processEvents()

        baseline_payload = main_window._plot_tabs._main_plot.export_payload()
        assert baseline_payload is not None
        baseline_digest = hashlib.sha256(repr(baseline_payload).encode("utf-8")).hexdigest()
        baseline_status = main_window._status_label.text()

        refresh_calls: list[str] = []
        original_refresh = main_window._refresh_batch_display_from_focus_and_shown

        def _spy_refresh() -> None:
            refresh_calls.append("called")
            original_refresh()

        monkeypatch.setattr(main_window, "_refresh_batch_display_from_focus_and_shown", _spy_refresh)

        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": ds1_value}, "ds2": {"init:A": ds2_value}},
                )
            }
        )
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Initial conditions only")
        button.click()
        qt_app.processEvents()

        current_payload = main_window._plot_tabs._main_plot.export_payload()
        current_digest = hashlib.sha256(repr(current_payload).encode("utf-8")).hexdigest()

        assert len(info_calls) == info_count_before_apply + 1
        assert refresh_calls == []
        assert cache.active_cache_invalidated_set_ids is None
        assert main_window._status_label.text() == baseline_status
        assert main_window.active_batch_selection() == (set_id, set_name)
        assert current_payload is not None
        assert current_digest == baseline_digest
    finally:
        window.close()


def test_global_fit_apply_to_project_initial_condition_settings_sync_without_canonical_change_preserves_cached_display(
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

    info_calls: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: info_calls.append("shown") or QtWidgets.QMessageBox.StandardButton.Ok,
    )

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        ds1_settings = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_settings = main_window._dataset_manager.get_fit_settings("ds2")
        ds1_row = main_window._batch_store.row_for_set_id(str(ds1_settings.batch_set_id))
        ds2_row = main_window._batch_store.row_for_set_id(str(ds2_settings.batch_set_id))
        assert ds1_row is not None
        assert ds2_row is not None

        ds1_value = float(main_window._batch_store.get_value(int(ds1_row), "A"))
        ds2_value = float(main_window._batch_store.get_value(int(ds2_row), "A"))
        ds1_settings.initial_conditions["A"] = ds1_value - 1.0
        ds2_settings.initial_conditions["A"] = ds2_value - 1.0
        main_window._dataset_manager.update_fit_settings("ds1", ds1_settings)
        main_window._dataset_manager.update_fit_settings("ds2", ds2_settings)

        set_id, set_name = _show_only_batch_set(main_window, row=int(ds1_row), qt_app=qt_app)

        cache = main_window.simulation_controller.batch_cache
        cache_key = "fit-apply-ic-settings-sync-only-cache"
        cache.result_cache[f"{cache_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        }
        cache.active_cache_key = cache_key
        cache.active_cache_valid_set_ids = (set_id,)
        cache.active_cache_invalidated_set_ids = None
        main_window.set_active_batch_selection(set_id, set_name, [set_id])

        main_window._refresh_batch_display_from_focus_and_shown()
        qt_app.processEvents()

        baseline_payload = main_window._plot_tabs._main_plot.export_payload()
        assert baseline_payload is not None
        baseline_digest = hashlib.sha256(repr(baseline_payload).encode("utf-8")).hexdigest()
        baseline_status = main_window._status_label.text()

        refresh_calls: list[str] = []
        original_refresh = main_window._refresh_batch_display_from_focus_and_shown

        def _spy_refresh() -> None:
            refresh_calls.append("called")
            original_refresh()

        monkeypatch.setattr(main_window, "_refresh_batch_display_from_focus_and_shown", _spy_refresh)

        window._handle_global_fit_complete(
            {
                "result": _make_fit_result(
                    k_value=0.44,
                    dataset_initials={"ds1": {"init:A": ds1_value}, "ds2": {"init:A": ds2_value}},
                )
            }
        )
        info_count_before_apply = len(info_calls)

        combo = window.findChild(QtWidgets.QComboBox, "global_fit_apply_scope_combo")
        button = window.findChild(QtWidgets.QPushButton, "global_fit_apply_to_project_button")
        assert combo is not None
        assert button is not None
        combo.setCurrentText("Initial conditions only")
        button.click()
        qt_app.processEvents()

        current_payload = main_window._plot_tabs._main_plot.export_payload()
        current_digest = hashlib.sha256(repr(current_payload).encode("utf-8")).hexdigest()
        ds1_after = main_window._dataset_manager.get_fit_settings("ds1")
        ds2_after = main_window._dataset_manager.get_fit_settings("ds2")

        assert len(info_calls) == info_count_before_apply + 1
        assert refresh_calls == []
        assert cache.active_cache_invalidated_set_ids is None
        assert ds1_after.initial_conditions.get("A") == pytest.approx(ds1_value)
        assert ds2_after.initial_conditions.get("A") == pytest.approx(ds2_value)
        assert main_window._status_label.text() == baseline_status
        assert main_window.active_batch_selection() == (set_id, set_name)
        assert current_payload is not None
        assert current_digest == baseline_digest
    finally:
        window.close()


def test_global_fit_rebuilds_live_window_simulation_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)

    def _scan_params(dsl: str):
        text = str(dsl or "")
        if "reaction: B -> C; k=0.4" in text:
            return [
                {"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0},
                {"name": "k2", "value": 0.4, "min": 0.01, "max": 1.0},
            ]
        return [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]

    monkeypatch.setattr(main_window._dataset_manager, "scan_mechanism_parameters", _scan_params)

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1
        first_simulation = captured_runs[0]["kwargs"]["simulation_func"]
        first_result = first_simulation({"k1": 0.2, "init:A": 1.0})
        assert set(first_result["species"]) == {"A", "B"}

        mechanism_b = "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "reaction: B -> C; k=0.4",
                "initial: A=1.0",
                "initial: B=0.0",
                "initial: C=0.0",
            ]
        )
        main_window._mechanism_editor._reactions_text.setPlainText(mechanism_b)
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 2

        second_simulation = captured_runs[1]["kwargs"]["simulation_func"]
        second_result = second_simulation({"k1": 0.2, "k2": 0.4, "init:A": 1.0})
        assert set(second_result["species"]) == {"A", "B", "C"}
        assert "C" in second_result["species"]

        prepared_stamp = window._run_results_tab._last_run_stamp.get("prepared_simulation") or {}
        assert prepared_stamp.get("mechanism_text_sha256") == hashlib.sha256(mechanism_b.encode("utf-8")).hexdigest()
        assert prepared_stamp.get("param_names") == ["k1", "k2"]
    finally:
        window.close()


def test_global_fit_rebuild_refreshes_live_window_parameter_table_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)

    def _scan_params(dsl: str):
        text = str(dsl or "")
        if "reaction: A -> C; k=0.4" in text:
            return [{"name": "k2", "value": 0.4, "min": 0.01, "max": 1.0}]
        return [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]

    monkeypatch.setattr(main_window._dataset_manager, "scan_mechanism_parameters", _scan_params)

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        assert _parameter_table_names(window) == ["k1"]

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1

        mechanism_b = "\n".join(
            [
                "reaction: A -> C; k=0.4",
                "initial: A=1.0",
                "initial: C=0.0",
            ]
        )
        main_window._mechanism_editor._reactions_text.setPlainText(mechanism_b)
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 2
        assert _parameter_table_names(window) == ["k2"]
        assert [str(entry.get("param_name") or "") for entry in window._parameter_state] == ["k2"]
    finally:
        window.close()


def test_global_fit_rebuild_refreshes_live_window_species_editor_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None
        ic_combo = window.findChild(QtWidgets.QComboBox, "global_fit_initial_conditions_dataset_combo")
        assert ic_combo is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        for i in range(ic_combo.count()):
            if str(ic_combo.itemData(i)) == "ds1":
                ic_combo.setCurrentIndex(i)
                break
        qt_app.processEvents()

        assert _ic_table_species(window) == ["A", "B"]

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1

        mechanism_b = "\n".join(
            [
                "reaction: A -> C; k=0.2",
                "initial: A=1.0",
                "initial: C=0.0",
            ]
        )
        main_window._mechanism_editor._reactions_text.setPlainText(mechanism_b)
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 2
        assert _ic_table_species(window) == ["A", "C"]
        assert window._mechanism_species == ["A", "C"]
    finally:
        window.close()


def test_global_fit_rebuild_preserves_shared_initial_rows_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)

    def _scan_params(dsl: str):
        text = str(dsl or "")
        if "reaction: A -> C; k=0.4" in text:
            return [{"name": "k2", "value": 0.4, "min": 0.01, "max": 1.0}]
        return [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]

    monkeypatch.setattr(main_window._dataset_manager, "scan_mechanism_parameters", _scan_params)

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        window._params_ics_tab._add_global_initial_parameter("A", ["ds1", "ds2"])
        window._params_ics_tab._populate_parameter_table()
        qt_app.processEvents()

        assert "Global A_0" in _parameter_table_names(window)

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        assert "init:A" in config["parameters"]
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1

        mechanism_b = "\n".join(
            [
                "reaction: A -> C; k=0.4",
                "initial: A=1.0",
                "initial: C=0.0",
            ]
        )
        main_window._mechanism_editor._reactions_text.setPlainText(mechanism_b)
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 2
        assert "Global A_0" in _parameter_table_names(window)

        config_after = window._params_ics_tab._collect_parameter_config()
        assert config_after is not None
        assert "init:A" in config_after["parameters"]
        assert "k2" in config_after["parameters"]
    finally:
        window.close()


def test_global_fit_rebuild_keeps_fixed_dataset_rows_visible_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        window._params_ics_tab._add_local_initial_parameter("A", ["ds1"])
        window._params_ics_tab._populate_parameter_table()
        qt_app.processEvents()

        table = getattr(window, "_param_table", None)
        assert table is not None
        row_index = next(
            idx for idx, row in enumerate(_parameter_table_rows(window)) if row["name"] == "A_0 (ds1)"
        )
        table.item(row_index, 0).setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()

        rows_before = _parameter_table_rows(window)
        assert any(row["name"] == "A_0 (ds1)" and row["fit"] is False for row in rows_before)

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1

        mechanism_b = "\n".join(
            [
                "reaction: A -> C; k=0.2",
                "initial: A=1.0",
                "initial: C=0.0",
            ]
        )
        main_window._mechanism_editor._reactions_text.setPlainText(mechanism_b)
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 2

        rows_after = _parameter_table_rows(window)
        assert any(row["name"] == "A_0 (ds1)" and row["fit"] is False for row in rows_after)

        overrides = captured_runs[1]["kwargs"]["dataset_overrides"]
        assert overrides[0].fixed_params["init:A"] == pytest.approx(1.0)
    finally:
        window.close()


def test_global_fit_rebuild_handles_scan_failures_with_warning_after_mechanism_edit(main_window, monkeypatch, qt_app):
    from PySide6 import QtCore, QtWidgets

    from kindred.gui.controllers.dataset_manager import DatasetManagerError

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)

    def _scan_params(dsl: str):
        text = str(dsl or "")
        if not text.strip():
            raise DatasetManagerError("Mechanism text is empty.")
        return [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]

    monkeypatch.setattr(main_window._dataset_manager, "scan_mechanism_parameters", _scan_params)

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured_runs.append({"kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

        def pause(self):
            return

        def resume(self):
            return

    warning_calls: list[tuple[str, str]] = []

    def _fake_warning(parent, title, text, *args, **kwargs):
        warning_calls.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    captured_runs: list[dict[str, object]] = []
    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)

    main_window._run_global_fit()
    window = _latest_fit_window(main_window)
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_pending(panel, dataset_id="ds1", enabled_species={"A"}, qt_app=qt_app)
        _set_fit_targets_pending(panel, dataset_id="ds2", enabled_species={"A"}, qt_app=qt_app)
        apply_btn.click()
        qt_app.processEvents()

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1

        main_window._mechanism_editor._reactions_text.setPlainText("")
        qt_app.processEvents()

        window._start_global_fit(config, selection, solver="LSODA", rtol=1e-6, atol=1e-12)
        assert len(captured_runs) == 1
        assert warning_calls
        assert warning_calls[-1] == ("Global Fit", "Mechanism text is empty.")
    finally:
        window.close()
