from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets


def current_preview_time_axis(main_window) -> np.ndarray:
    selected_ids = [
        str(set_id)
        for set_id in (main_window._batch_set_ids_for_scope("selected") or ())
        if str(set_id)
    ]
    target_set_id = (
        selected_ids[0]
        if selected_ids
        else str(main_window._preview_session.focused_mechanism_workspace_set_id() or "")
    )
    assert target_set_id
    mechanism_text = main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(
        set_id=target_set_id
    )
    solver_config, t_end, _ = main_window._simulation_batch_owner.current_workspace_preview_context(
        set_id=target_set_id,
        mechanism_text=mechanism_text,
    )
    grid_n = int((solver_config.get("grid") or {}).get("N") or 0)
    return np.linspace(0.0, float(t_end), max(2, grid_n), dtype=float)


def parameter_table_numeric_value(main_window, name: str) -> float:
    table = main_window.main_plot().parameter_table()
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None or item.text() != str(name):
            continue
        value_item = table.item(row, 1)
        assert value_item is not None
        return float(value_item.text())
    raise AssertionError(f"Missing parameter-table row for {name!r}")


def select_batch_rows(main_window, rows: list[int]) -> QtCore.QItemSelectionModel:
    return set_batch_current_and_selected_rows(
        main_window,
        current_row=int(rows[0]),
        selected_rows=list(rows),
    )


def set_batch_current_and_selected_rows(
    main_window,
    *,
    current_row: int,
    selected_rows: list[int],
) -> QtCore.QItemSelectionModel:
    table = main_window._batch_table
    model = main_window._batch_model
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    assert selected_rows
    current_idx = model.index(int(current_row), 0)
    assert current_idx.isValid()
    table.setCurrentIndex(current_idx)
    sel.clearSelection()
    for row in selected_rows:
        idx = model.index(int(row), 0)
        assert idx.isValid()
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._refresh_batch_display_from_focus_and_shown()
    return sel


def set_shown_rows(main_window, rows: list[int]) -> None:
    model = main_window._batch_model
    shown_rows = {int(row) for row in rows}
    for row in range(model.rowCount()):
        model.set_row_shown(row, row in shown_rows)
        expected = QtCore.Qt.Checked if row in shown_rows else QtCore.Qt.Unchecked
        assert model.data(model.index(row, model.show_column()), QtCore.Qt.CheckStateRole) == expected
    main_window._refresh_batch_display_from_focus_and_shown()


def slider_handle_center(slider: QtWidgets.QSlider) -> QtCore.QPoint:
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider,
        option,
        QtWidgets.QStyle.SC_SliderHandle,
        slider,
    )
    return handle.center()


def seed_two_datasets(main_window) -> None:
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    t = np.linspace(0.0, 1.0, 6)
    data_panel._datasets.update(
        {
            "ds1": {
                "t": t.copy(),
                "species": {
                    "A": np.linspace(1.0, 0.5, t.size),
                    "B": np.linspace(0.2, 0.9, t.size),
                },
            },
            "ds2": {
                "t": t.copy(),
                "species": {
                    "A": np.linspace(0.8, 0.1, t.size),
                    "B": np.linspace(0.0, 0.4, t.size),
                },
            },
        }
    )


def seed_simple_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )


def latest_fit_window(main_window):
    windows = list(getattr(main_window, "_active_fit_windows", []) or [])
    assert windows, "Expected Global Fit window to be registered"
    return windows[-1]


def patch_message_box_exec(monkeypatch, *, info_calls: list[str] | None = None) -> None:
    from PySide6 import QtWidgets

    def _fake_exec(self):
        if info_calls is not None and self.icon() == QtWidgets.QMessageBox.Icon.Information:
            info_calls.append("shown")
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    def _fake_information(*_args, **_kwargs):
        if info_calls is not None:
            info_calls.append("shown")
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", _fake_information)


def show_only_batch_set(main_window, *, row: int, qt_app) -> tuple[str, str]:
    model = main_window._batch_model
    table = main_window._batch_table
    for row_index in range(model.rowCount()):
        model.set_row_shown(row_index, row_index == int(row))
    table.selectRow(int(row))
    qt_app.processEvents()
    set_id = str(main_window._batch_store.set_id_for_row(int(row)))
    set_name = str(main_window._batch_store.set_name_for_row(int(row)))
    return set_id, set_name


def make_fit_result(*, k_value: float, dataset_initials: dict[str, dict[str, float]]):
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
    from kindred.core.fitting_completion import GlobalFitCompletion

    model = np.linspace(1.0, 0.4, 6)
    residual = np.linspace(0.0, -0.1, 6)
    return GlobalFitResult(
        shared_params={"k1": float(k_value)},
        dataset_params={
            str(dataset_id): {str(name): float(value) for name, value in values.items()}
            for dataset_id, values in dataset_initials.items()
        },
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
        completion=GlobalFitCompletion(
            status="ok",
            optimizer_converged=True,
            nonfinite_metrics=False,
        ),
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
