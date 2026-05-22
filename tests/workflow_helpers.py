from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets


def completion_provenance_payload(
    *,
    t: np.ndarray,
    series: dict[str, np.ndarray],
    mechanism_text: str = "A -> B ; k=1.0",
    solver_config: dict | None = None,
    solver_provenance: dict | None = None,
    temperature_K: float = 298.15,
) -> dict:
    solver_config = dict(solver_config or {})
    return {
        "mechanism_text": str(mechanism_text),
        "solver_method": str(solver_config.get("solver") or "RK45"),
        "solver_label": str(solver_config.get("solver_label") or solver_config.get("solver") or "RK45"),
        "solver_warning": None,
        "solver_config": {
            "rtol": solver_config.get("rtol", 1e-6),
            "atol": solver_config.get("atol", 1e-12),
        },
        "temperature_K": float(temperature_K),
        "temperature_source": "test",
        "energy_unit": None,
        "energy_mode": False,
        "simulation_time": float(np.asarray(t, dtype=float).reshape(-1)[-1]),
        "num_points_requested": int(np.asarray(t).size),
        "species_names": [str(name) for name in series],
        "t": t,
        "series": series,
        "algebra_scalars": {},
        "solver_provenance": dict(solver_provenance or {}),
        "warnings": [],
    }


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
    return sel


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
            info_calls.append("information")
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    def _fake_information(*_args, **_kwargs):
        if info_calls is not None:
            info_calls.append("information")
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", _fake_information)


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
