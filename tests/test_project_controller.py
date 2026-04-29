import contextlib
from types import SimpleNamespace

import pytest
from PySide6 import QtCore, QtWidgets
from unittest.mock import MagicMock, mock_open, patch

from kindred.gui.controllers.project_controller import ProjectController


pytestmark = pytest.mark.unit

PROJECT_DIALOG_FILTER = "Kindred Project (*.kin);;JSON Files (*.json);;All Files (*)"


@contextlib.contextmanager
def _null_cursor():
    yield


@pytest.fixture
def controller_and_mw(qt_app):
    parent = QtWidgets.QWidget()
    controller = ProjectController(parent)
    controller._test_parent = parent

    mw = MagicMock(name="MainWindowMock")
    mw.set_status_text = MagicMock(name="SetStatusTextMock")
    mw._plot_tabs = MagicMock(name="PlotTabsMock")
    mw._settings = MagicMock(name="SettingsMock")
    mw.config_controller = MagicMock(name="ConfigControllerMock")
    mw.serialize_project_state = MagicMock(name="SerializeProjectStateMock")
    mw.apply_project_payload = MagicMock(name="ApplyProjectPayloadMock")
    mw.add_to_recent_files = MagicMock(name="AddToRecentFilesMock")
    mw._undo_stack = MagicMock(name="UndoStackMock")
    mw.setWindowTitle = MagicMock(name="SetWindowTitleMock")
    controller.mw = mw

    try:
        yield controller, mw
    finally:
        dialog = getattr(controller, "_export_dialog", None)
        app = QtWidgets.QApplication.instance()
        with contextlib.suppress(RuntimeError, TypeError, AttributeError):
            if dialog is not None and hasattr(dialog, "close"):
                dialog.close()
        with contextlib.suppress(RuntimeError, TypeError, AttributeError):
            if dialog is not None and hasattr(dialog, "deleteLater"):
                dialog.deleteLater()
        with contextlib.suppress(RuntimeError, TypeError, AttributeError):
            controller.deleteLater()
        with contextlib.suppress(RuntimeError, TypeError, AttributeError):
            parent.close()
        with contextlib.suppress(RuntimeError, TypeError, AttributeError):
            parent.deleteLater()
        if app is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            for _ in range(5):
                app.processEvents()


def test_load_project_dialog_cancel_is_noop(controller_and_mw):
    controller, _mw = controller_and_mw
    controller._load_project_from_path = MagicMock()

    with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=("", "")):
        controller.load_project()

    controller._load_project_from_path.assert_not_called()


def test_load_project_dialog_ok_calls_load_from_path(controller_and_mw):
    controller, _mw = controller_and_mw
    controller._load_project_from_path = MagicMock()

    with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=("project.kin", "")) as get_open:
        controller.load_project()

    get_open.assert_called_once_with(
        controller.mw,
        "Load Project",
        "",
        PROJECT_DIALOG_FILTER,
    )
    controller._load_project_from_path.assert_called_once_with(
        "project.kin",
        record_undo=False,
        add_to_recent=True,
        status_path="project.kin",
    )


def test_save_project_no_path_delegates_to_save_as(controller_and_mw):
    """When no project path is set, save_project delegates to save_project_as."""
    controller, mw = controller_and_mw
    assert controller._current_project_path is None

    with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")):
        result = controller.save_project()

    assert result is False
    mw.serialize_project_state.assert_not_called()


def test_save_project_with_path_writes_directly(controller_and_mw):
    """When a project path is already set, save_project writes without dialog."""
    controller, mw = controller_and_mw
    controller._current_project_path = "/existing/project.kin"
    mw.serialize_project_state.return_value = {"ok": True}

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.json.dump") as json_dump,
    ):
        result = controller.save_project()

    assert result is True
    json_dump.assert_called_once()
    mw.set_status_text.assert_called_once_with("Saved project: /existing/project.kin")
    mw.setWindowTitle.assert_called_once_with("Kindred \u2014 project.kin")
    mw.add_to_recent_files.assert_called_once_with("/existing/project.kin")


def test_save_project_as_success(controller_and_mw):
    """save_project_as always shows dialog, writes, and sets path."""
    controller, mw = controller_and_mw
    mw.serialize_project_state.return_value = {"ok": True}

    with (
        patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("out.kin", "")) as get_save,
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.json.dump") as json_dump,
    ):
        result = controller.save_project_as()

    assert result is True
    get_save.assert_called_once_with(
        controller.mw,
        "Save Project As",
        "",
        PROJECT_DIALOG_FILTER,
    )
    json_dump.assert_called_once()
    mw.set_status_text.assert_called_once_with("Saved project: out.kin")
    mw.add_to_recent_files.assert_called_once_with("out.kin")
    assert controller._current_project_path == "out.kin"
    mw.setWindowTitle.assert_called_once_with("Kindred \u2014 out.kin")


def test_save_project_as_cancel_returns_false(controller_and_mw):
    """Cancelling the save-as dialog returns False without writing."""
    controller, mw = controller_and_mw

    with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")):
        result = controller.save_project_as()

    assert result is False
    mw.serialize_project_state.assert_not_called()


def test_save_project_as_failure_returns_false(controller_and_mw):
    """Write failure shows critical dialog and returns False."""
    controller, mw = controller_and_mw
    mw.serialize_project_state.return_value = {"ok": True}

    with (
        patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("out.kin", "")),
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.json.dump", side_effect=ValueError("nope")),
        patch.object(QtWidgets.QMessageBox, "critical") as critical,
    ):
        result = controller.save_project_as()

    assert result is False
    critical.assert_called_once()
    mw.add_to_recent_files.assert_not_called()
    assert controller._current_project_path is None


def test_load_recent_project_missing_file_warns_and_prunes_recent(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    missing = str(tmp_path / "missing.kin")
    other = str(tmp_path / "other.kin")
    mw._settings.value.return_value = [missing, other]

    with (
        patch("kindred.gui.controllers.project_controller.os.path.exists", return_value=False),
        patch.object(QtWidgets.QMessageBox, "warning") as warning,
    ):
        controller.load_recent_project(missing)

    warning.assert_called_once()
    mw._settings.setValue.assert_called_once_with("recent_files", [other])
    mw.config_controller.update_recent_files_menu.assert_called_once()


def test_load_recent_project_existing_file_loads_without_recent_side_effects(controller_and_mw, tmp_path):
    controller, _mw = controller_and_mw
    controller._load_project_from_path = MagicMock()
    project_path = str(tmp_path / "project.kin")

    with patch("kindred.gui.controllers.project_controller.os.path.exists", return_value=True):
        controller.load_recent_project(project_path)

    controller._load_project_from_path.assert_called_once_with(
        project_path,
        record_undo=False,
        add_to_recent=False,
        status_path="project.kin",
    )


def test__load_project_from_path_success_applies_payload_and_updates_status(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    in_path = str(tmp_path / "in.kin")

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("kindred.gui.controllers.project_controller.json.load", return_value={"payload": 1}),
    ):
        controller._load_project_from_path(
            in_path,
            record_undo=True,
            add_to_recent=True,
            status_path="in.kin",
        )

    mw.apply_project_payload.assert_called_once_with({"payload": 1}, record_undo=True)
    mw.set_status_text.assert_called_once_with("Loaded project: in.kin")
    mw.add_to_recent_files.assert_called_once_with(in_path)
    assert controller._current_project_path == in_path
    mw.setWindowTitle.assert_called_once_with("Kindred \u2014 in.kin")


def test__load_project_from_path_non_undoable_load_clears_app_undo_history(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    in_path = str(tmp_path / "in.kin")

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("kindred.gui.controllers.project_controller.json.load", return_value={"payload": 1}),
    ):
        controller._load_project_from_path(
            in_path,
            record_undo=False,
            add_to_recent=False,
            status_path="in.kin",
        )

    mw._undo_stack.clear.assert_called_once_with()
    mw.apply_project_payload.assert_called_once_with({"payload": 1}, record_undo=False)


def test__load_project_from_path_undoable_load_preserves_app_undo_history(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    in_path = str(tmp_path / "in.kin")

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("kindred.gui.controllers.project_controller.json.load", return_value={"payload": 1}),
    ):
        controller._load_project_from_path(
            in_path,
            record_undo=True,
            add_to_recent=False,
            status_path="in.kin",
        )

    mw._undo_stack.clear.assert_not_called()
    mw.apply_project_payload.assert_called_once_with({"payload": 1}, record_undo=True)


def test__load_project_from_path_canceled_apply_does_not_report_success(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    in_path = str(tmp_path / "in.kin")
    mw.apply_project_payload.return_value = False

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("kindred.gui.controllers.project_controller.json.load", return_value={"payload": 1}),
    ):
        controller._load_project_from_path(
            in_path,
            record_undo=True,
            add_to_recent=True,
            status_path="in.kin",
        )

    mw.apply_project_payload.assert_called_once_with({"payload": 1}, record_undo=True)
    mw.set_status_text.assert_not_called()
    mw.add_to_recent_files.assert_not_called()
    assert controller._current_project_path is None
    mw.setWindowTitle.assert_not_called()


def test__load_project_from_path_failure_shows_critical(controller_and_mw, tmp_path):
    controller, _mw = controller_and_mw
    in_path = str(tmp_path / "in.kin")

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open(read_data="{}")),
        patch("kindred.gui.controllers.project_controller.json.load", side_effect=ValueError("bad json")),
        patch.object(QtWidgets.QMessageBox, "critical") as critical,
    ):
        controller._load_project_from_path(
            in_path,
            record_undo=True,
            add_to_recent=False,
            status_path="in.kin",
        )

    critical.assert_called_once()


def test_export_data_preconditions_fail_is_noop(controller_and_mw):
    controller, _mw = controller_and_mw
    controller._validate_export_preconditions = MagicMock(return_value=None)

    controller.export_data()

    assert controller._export_dialog is None


def test_export_data_creates_dialog_connects_and_opens(controller_and_mw):
    controller, mw = controller_and_mw
    controller._validate_export_preconditions = MagicMock(return_value={"t": [0], "series": {"A": [1]}})

    plot = MagicMock()
    plot.get_export_scope_preference.return_value = "axis"
    mw._plot_tabs.get_current_plot.return_value = plot

    dialog = MagicMock()
    dialog.exportAccepted = MagicMock()
    dialog.exportAccepted.connect = MagicMock()

    with patch("kindred.gui.controllers.project_controller.ExportDialog", return_value=dialog) as export_dialog:
        controller.export_data()

    export_dialog.assert_called_once_with(mw)
    dialog.exportAccepted.connect.assert_called_once_with(controller.handle_export_config)
    dialog.set_scope.assert_called_once_with("axis")
    dialog.open.assert_called_once()


def test_export_data_scope_preference_failure_does_not_block(controller_and_mw):
    controller, mw = controller_and_mw
    controller._validate_export_preconditions = MagicMock(return_value={"t": [0], "series": {"A": [1]}})

    plot = MagicMock()
    plot.get_export_scope_preference.return_value = "axis"
    mw._plot_tabs.get_current_plot.return_value = plot

    dialog = MagicMock()
    dialog.exportAccepted = MagicMock()
    dialog.exportAccepted.connect = MagicMock()
    dialog.set_scope.side_effect = RuntimeError("boom")

    with patch("kindred.gui.controllers.project_controller.ExportDialog", return_value=dialog):
        controller.export_data()

    dialog.open.assert_called_once()

def test__warn_no_export_target_sets_status_and_shows_warning(controller_and_mw):
    controller, mw = controller_and_mw

    with patch.object(QtWidgets.QMessageBox, "warning") as warning:
        controller._warn_no_export_target("Nope")

    warning.assert_called_once_with(mw, "Export Unavailable", "Nope")
    mw.set_status_text.assert_called_once_with("Nope")


def test__validate_export_preconditions_rejects_unknown_type(controller_and_mw):
    controller, _mw = controller_and_mw

    with pytest.raises(ValueError, match="Unknown export type"):
        controller._validate_export_preconditions("nope")


def test__validate_export_preconditions_warns_when_no_payload(controller_and_mw):
    controller, mw = controller_and_mw
    controller._resolve_export_payload = MagicMock(return_value=None)
    controller._warn_no_export_target = MagicMock()

    plot = MagicMock()
    mw._plot_tabs.get_current_plot.return_value = plot

    assert controller._validate_export_preconditions("data") is None
    controller._warn_no_export_target.assert_called_once()


def test__validate_export_preconditions_returns_payload_when_available(controller_and_mw):
    controller, mw = controller_and_mw
    controller._resolve_export_payload = MagicMock(return_value={"t": [0], "series": {"A": [1]}})
    controller._warn_no_export_target = MagicMock()

    plot = MagicMock()
    mw._plot_tabs.get_current_plot.return_value = plot

    assert controller._validate_export_preconditions("data") == {"t": [0], "series": {"A": [1]}}
    controller._warn_no_export_target.assert_not_called()


def test__resolve_export_payload_returns_none_for_none_plot(controller_and_mw):
    controller, _mw = controller_and_mw
    assert controller._resolve_export_payload(None) is None


def test__resolve_export_payload_prefers_export_payload_over_dataset_getter(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = {"t": [0.0, 1.0], "series": {"A": [1.0, 2.0]}}
    plot.get_dataset_data.return_value = {"t": [0.0, 1.0], "A": [9.0, 9.0]}

    payload = controller._resolve_export_payload(plot)
    assert payload is not None
    assert list(payload["series"].keys()) == ["A"]
    assert payload["t"].shape == (2,)
    plot.get_dataset_data.assert_not_called()


def test__resolve_export_payload_uses_export_payload_callable(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = {"t": [0], "series": {"A": [1]}}

    payload = controller._resolve_export_payload(plot)
    assert payload is not None
    assert list(payload["series"].keys()) == ["A"]
    assert payload["t"].shape == (1,)


def test__resolve_export_payload_uses_dataset_getter(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = None
    plot.get_dataset_data.return_value = {"t": [0, 1], "A": [1, 2]}

    payload = controller._resolve_export_payload(plot)
    assert payload is not None
    assert list(payload["series"].keys()) == ["A"]
    assert payload["t"].shape == (2,)


def test__resolve_export_payload_returns_none_when_no_sources_available(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = SimpleNamespace()
    plot.export_payload = lambda: None
    plot.get_dataset_data = lambda: {"t": [0, 1]}

    assert controller._resolve_export_payload(plot) is None


def test__handle_export_config_warns_when_payload_gone(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    controller._warn_no_export_target = MagicMock()
    controller._resolve_export_payload = MagicMock(return_value=None)

    mw._plot_tabs.get_current_plot.return_value = MagicMock()
    controller.handle_export_config({"path": str(tmp_path / "out.csv")})

    controller._warn_no_export_target.assert_called_once_with("Simulation data is no longer available for export.")


def test__handle_export_config_writes_csv_default_mode(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    plot = MagicMock()
    mw._plot_tabs.get_current_plot.return_value = plot
    out_path = str(tmp_path / "out.csv")

    controller._resolve_export_payload = MagicMock(return_value={"t": [0], "series": {"A": [1]}})
    controller._prepare_default_export_rows = MagicMock(return_value=(["t", "A"], [[0.0, 1.0], [1.0, 2.0]]))

    writer = MagicMock()
    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.csv.writer", return_value=writer) as csv_writer,
    ):
        controller.handle_export_config({"path": out_path, "mode": "default", "scope": "axis"})

    csv_writer.assert_called_once()
    writer.writerow.assert_any_call(["t", "A"])
    assert writer.writerow.call_count == 1 + 2
    mw.set_status_text.assert_called_once()


def test__handle_export_config_writes_csv_legacy_mode(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    plot = MagicMock()
    mw._plot_tabs.get_current_plot.return_value = plot
    out_path = str(tmp_path / "out.csv")

    controller._resolve_export_payload = MagicMock(return_value={"t": [0], "series": {"A": [1]}})
    controller._prepare_legacy_export_rows = MagicMock(return_value=(["t", "[A]"], [["0.0", "1.0"]]))

    writer = MagicMock()
    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.csv.writer", return_value=writer),
    ):
        controller.handle_export_config({"path": out_path, "mode": "legacy"})

    controller._prepare_legacy_export_rows.assert_called_once_with(plot)
    writer.writerow.assert_any_call(["t", "[A]"])


def test__handle_export_config_value_error_shows_warning(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    mw._plot_tabs.get_current_plot.return_value = MagicMock()
    controller._resolve_export_payload = MagicMock(return_value={"t": [0], "series": {"A": [1]}})
    controller._prepare_default_export_rows = MagicMock(side_effect=ValueError("no series"))
    out_path = str(tmp_path / "out.csv")

    with patch.object(QtWidgets.QMessageBox, "warning") as warning:
        controller.handle_export_config({"path": out_path, "mode": "default", "scope": "axis"})

    warning.assert_called_once_with(mw, "Export Error", "no series")


def test__handle_export_config_unexpected_error_shows_critical(controller_and_mw, tmp_path):
    controller, mw = controller_and_mw
    mw._plot_tabs.get_current_plot.return_value = MagicMock()
    controller._resolve_export_payload = MagicMock(return_value={"t": [0], "series": {"A": [1]}})
    controller._prepare_default_export_rows = MagicMock(return_value=(["t"], [[0]]))
    out_path = str(tmp_path / "out.csv")

    with (
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", side_effect=OSError("disk full")),
        patch.object(QtWidgets.QMessageBox, "critical") as critical,
    ):
        controller.handle_export_config({"path": out_path, "mode": "default", "scope": "axis"})

    critical.assert_called_once()


def test__prepare_legacy_export_rows_success_from_plot_arrays(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = {"t": [0.0, 1.0], "series": {"A": [1.0, 2.0], "B": [3.0, 4.0]}}

    header, rows = controller._prepare_legacy_export_rows(plot)

    assert header == ["t", "[A]", "[B]"]
    assert list(rows) == [["0.000000", "1.000000", "3.000000"], ["1.000000", "2.000000", "4.000000"]]


def test__prepare_legacy_export_rows_rejects_empty_time_axis(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = {"t": [], "series": {"A": []}}

    with pytest.raises(ValueError, match="Time axis has no points"):
        controller._prepare_legacy_export_rows(plot)


def test__prepare_legacy_export_rows_rejects_missing_series(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = SimpleNamespace()
    plot.export_payload = lambda: {"t": [0.0], "series": {}}

    with pytest.raises(ValueError, match="No series data available"):
        controller._prepare_legacy_export_rows(plot)


def test__prepare_legacy_export_rows_rejects_length_mismatch(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = {"t": [0.0, 1.0], "series": {"A": [1.0]}}

    with pytest.raises(ValueError, match="does not match time grid"):
        controller._prepare_legacy_export_rows(plot)


def test__prepare_legacy_export_rows_requires_payload_when_plot_has_no_arrays(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.export_payload.return_value = None
    plot.get_dataset_data.return_value = {"t": []}

    with pytest.raises(ValueError, match="No data available to export"):
        controller._prepare_legacy_export_rows(plot)


def test__prepare_default_export_rows_delegates_to_build_visible_export(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot.build_visible_export.return_value = (["X"], [[1]])

    assert controller._prepare_default_export_rows(plot, scope="axis") == (["X"], [[1]])


def test__prepare_default_export_rows_uses_payload_export_when_available(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = SimpleNamespace()
    plot.export_payload = lambda: {"t": [0.0, 1.0], "series": {"A": [1.0, 2.0]}}

    header, rows = controller._prepare_default_export_rows(plot, scope="all")
    assert header == ["Time", "A"]
    assert list(rows) == [[0.0, 1.0], [1.0, 2.0]]


def test__prepare_default_export_rows_requires_export_interface_when_payload_unavailable(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = SimpleNamespace()
    plot.export_payload = lambda: None

    with pytest.raises(ValueError, match="does not implement the export interface"):
        controller._prepare_default_export_rows(plot, scope="axis")


def test__prepare_payload_export_rows_ignores_checkboxes(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot._species_checkboxes = {"A": MagicMock(isChecked=lambda: False), "B": MagicMock(isChecked=lambda: True)}

    payload = {"t": [0.0, 1.0], "series": {"A": [1.0, 2.0], "B": [3.0, 4.0]}}
    header, rows = controller._prepare_payload_export_rows(payload, plot=plot, scope="axis")

    assert header == ["Time", "A", "B"]
    assert list(rows) == [[0.0, 1.0, 3.0], [1.0, 2.0, 4.0]]


def test__prepare_payload_export_rows_rejects_empty_time_axis(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    with pytest.raises(ValueError, match="Time axis has no points"):
        controller._prepare_payload_export_rows({"t": [], "series": {"A": []}}, plot=plot, scope="all")


def test__prepare_payload_export_rows_rejects_missing_series(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    with pytest.raises(ValueError, match="No series data available"):
        controller._prepare_payload_export_rows({"t": [0], "series": {}}, plot=plot, scope="all")


def test__prepare_payload_export_rows_does_not_require_selected_series(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    plot._species_checkboxes = {"A": MagicMock(isChecked=lambda: False)}
    payload = {"t": [0.0, 1.0], "series": {"A": [1.0, 2.0]}}

    header, rows = controller._prepare_payload_export_rows(payload, plot=plot, scope="axis")
    assert header == ["Time", "A"]
    assert list(rows) == [[0.0, 1.0], [1.0, 2.0]]


def test__prepare_payload_export_rows_rejects_length_mismatch(controller_and_mw):
    controller, _mw = controller_and_mw
    plot = MagicMock()
    payload = {"t": [0.0, 1.0], "series": {"A": [1.0]}}

    with pytest.raises(ValueError, match="does not match time grid"):
        controller._prepare_payload_export_rows(payload, plot=plot, scope="all")


# ------------------------------------------------------------------
# current_project_path property
# ------------------------------------------------------------------

def test_current_project_path_initially_none(controller_and_mw):
    controller, _mw = controller_and_mw
    assert controller.current_project_path is None


# ------------------------------------------------------------------
# _update_window_title
# ------------------------------------------------------------------

def test_update_window_title_with_path(controller_and_mw):
    controller, mw = controller_and_mw
    controller._current_project_path = "/path/to/my_project.kin"
    controller._update_window_title()
    mw.setWindowTitle.assert_called_once_with("Kindred \u2014 my_project.kin")


def test_update_window_title_without_path(controller_and_mw):
    controller, mw = controller_and_mw
    controller._current_project_path = None
    controller._update_window_title()
    mw.setWindowTitle.assert_called_once_with("Kindred")


# ------------------------------------------------------------------
# new_project
# ------------------------------------------------------------------

def test_new_project_cancel_is_noop(controller_and_mw):
    """Cancelling the save prompt does nothing."""
    controller, mw = controller_and_mw
    controller._current_project_path = "/some/file.kin"

    with patch.object(
        QtWidgets.QMessageBox, "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
    ):
        controller.new_project()

    assert controller._current_project_path == "/some/file.kin"
    mw.apply_project_payload.assert_not_called()


def test_new_project_discard_clears_state(controller_and_mw):
    """Choosing Discard clears state without saving."""
    controller, mw = controller_and_mw
    controller._current_project_path = "/some/file.kin"

    with patch.object(
        QtWidgets.QMessageBox, "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Discard,
    ):
        controller.new_project()

    mw.apply_project_payload.assert_called_once()
    payload = mw.apply_project_payload.call_args[0][0]
    assert payload["mechanism"] == ""
    assert payload["notes"] == ""
    assert controller._current_project_path is None
    mw.setWindowTitle.assert_called_with("Kindred")
    mw._undo_stack.clear.assert_called_once()
    mw.set_status_text.assert_called_once_with("New project")


def test_new_project_payload_uses_explicit_user_preference_payload(controller_and_mw):
    """new_project() sends a complete payload with user preferences resolved."""
    from kindred.gui.project_schema import FITTING_DEFAULTS_KEYS, PROJECT_DEFAULTS

    controller, mw = controller_and_mw
    preference_values = {
        "solver": "Radau",
        "use_sparse_jacobian": False,
        "max_parallel_batch_workers": 7,
        "batch_runtime_lane_budget": 5,
        "limit_blas_threads_per_worker": False,
    }
    mw.config_controller.get_user_preference.side_effect = (
        lambda key: preference_values.get(key, PROJECT_DEFAULTS[key])
    )
    with patch.object(
        QtWidgets.QMessageBox, "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Discard,
    ):
        controller.new_project()

    mw.apply_project_payload.assert_called_once()
    payload = mw.apply_project_payload.call_args[0][0]

    assert set(payload.keys()) == (set(PROJECT_DEFAULTS.keys()) - set(FITTING_DEFAULTS_KEYS))
    assert payload["mechanism"] == PROJECT_DEFAULTS["mechanism"]
    assert payload["solver"] == "Radau"
    assert payload["use_sparse_jacobian"] is False
    assert payload["max_parallel_batch_workers"] == 7
    assert payload["batch_runtime_lane_budget"] == 5
    assert payload["limit_blas_threads_per_worker"] is False
    for key in FITTING_DEFAULTS_KEYS:
        assert key not in payload


def test_new_project_uses_live_user_preferences_not_raw_qsettings(controller_and_mw):
    controller, mw = controller_and_mw
    mw.config_controller.get_user_preference.side_effect = lambda key: {
        "solver": "Radau",
        "use_sparse_jacobian": False,
        "max_parallel_batch_workers": 9,
        "batch_runtime_lane_budget": 6,
        "limit_blas_threads_per_worker": False,
    }.get(key)
    mw._settings.value.side_effect = lambda key, default=None, type=None: {
        "simulation/solver": "BDF",
        "simulation/use_sparse_jacobian": True,
        "simulation/max_parallel_batch_workers": 3,
        "simulation/batch_runtime_lane_budget": 2,
        "simulation/limit_blas_threads_per_worker": True,
    }.get(key, default)

    with patch.object(
        QtWidgets.QMessageBox, "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Discard,
    ):
        controller.new_project()

    payload = mw.apply_project_payload.call_args[0][0]
    assert payload["solver"] == "Radau"
    assert payload["use_sparse_jacobian"] is False
    assert payload["max_parallel_batch_workers"] == 9
    assert payload["batch_runtime_lane_budget"] == 6
    assert payload["limit_blas_threads_per_worker"] is False
    assert "fitting_parallel_enabled" not in payload


def test_new_project_save_then_clear(controller_and_mw):
    """Choosing Save triggers save_project, then clears on success."""
    controller, mw = controller_and_mw
    controller._current_project_path = "/existing/project.kin"
    mw.serialize_project_state.return_value = {"ok": True}

    with (
        patch.object(
            QtWidgets.QMessageBox, "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ),
        patch("kindred.gui.controllers.project_controller.BusyCursor", return_value=_null_cursor()),
        patch("builtins.open", mock_open()),
        patch("kindred.gui.controllers.project_controller.json.dump"),
    ):
        controller.new_project()

    mw.serialize_project_state.assert_called_once()
    mw.apply_project_payload.assert_called_once()
    assert controller._current_project_path is None


def test_new_project_save_cancelled_aborts(controller_and_mw):
    """If save_project returns False (user cancelled dialog), new project is aborted."""
    controller, mw = controller_and_mw

    with (
        patch.object(
            QtWidgets.QMessageBox, "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ),
        patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")),
    ):
        controller.new_project()

    mw.apply_project_payload.assert_not_called()


def test_new_project_apply_payload_guard_cancels(controller_and_mw):
    """If the slider transaction guard cancels apply_project_payload, abort."""
    controller, mw = controller_and_mw
    controller._current_project_path = "/some/file.kin"
    mw.apply_project_payload.return_value = False

    with patch.object(
        QtWidgets.QMessageBox, "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Discard,
    ):
        controller.new_project()

    mw.apply_project_payload.assert_called_once()
    assert controller._current_project_path == "/some/file.kin"
    mw._undo_stack.clear.assert_not_called()
