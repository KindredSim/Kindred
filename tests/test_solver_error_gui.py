from unittest.mock import MagicMock

import pytest
from PySide6 import QtWidgets

from kindred.core.exceptions import create_solver_error
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity

pytestmark = [pytest.mark.gui, pytest.mark.slow]


def test_main_window_shows_solver_error(main_window, monkeypatch):
    """Main window should surface solver errors to the user."""

    mock_critical = MagicMock(return_value=QtWidgets.QMessageBox.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", mock_critical)

    mock_plot = MagicMock()
    mock_plot._visible = True
    mock_plot._series = {}
    mock_plot.visible_series.return_value = []
    mock_plot.set_series_visible.return_value = None
    mock_plot.update_statistics.return_value = None
    mock_plot.overlay_snapshot = None

    mock_viewport = MagicMock()
    mock_table = MagicMock()
    mock_table.viewport.return_value = mock_viewport
    mock_plot.stats_table.return_value = mock_table

    main_window.set_data = lambda *args, **kwargs: None
    main_window._plot_tabs = MagicMock()
    main_window._plot_tabs._main_plot = mock_plot

    error = create_solver_error(
        "BDF",
        0.5,
        "forced failure; attempted methods: BDF, Radau",
    )

    main_window.simulation_controller.on_simulation_error(
        str(error),
        callback_identity=SimulationCallbackIdentity.capture(
            run_id=main_window.simulation_controller._active_run_id,
            fast_mode=False,
            request_id=main_window.simulation_controller._latest_sim_request_id,
            owner_epoch=None,
            batch_set=None,
            batch_set_id=None,
            cache_key="",
            callback_context=main_window.simulation_controller.batch_context_owner.callback_context_snapshot(),
            simulation_identity={},
            preview_batch_cache_token="",
        ),
    )

    assert mock_critical.called
    error_text = mock_critical.call_args[0][2]
    assert "solver" in error_text.lower()
    assert "bdf" in error_text.lower()
    assert "attempted methods" in error_text.lower()
