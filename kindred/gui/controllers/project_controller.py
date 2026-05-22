from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.gui.project_schema import (
    FITTING_DEFAULTS_KEYS,
    QSETTINGS_KEY_MAP,
    get_default_project_payload,
)
from kindred.gui.utils import BusyCursor
from kindred.gui.widgets.export_dialog import ExportDialog

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow

logger = logging.getLogger(__name__)

__all__ = ["ProjectController"]


@dataclass(frozen=True, slots=True)
class ExportPrecondition:
    active_display_transaction: bool = False
    payload: Optional[Dict[str, object]] = None


class ProjectController(QtCore.QObject):
    """
    Project + file I/O controller for MainWindow.

    Owns:
    - Project load/save JSON file dialogs
    - CSV export dialog + file writing

    Uses `mw` as the UI routing surface.
    """

    def __init__(self, mw: MainWindow):
        super().__init__(mw)
        self.mw = mw
        self._export_dialog = None
        self._current_project_path: Optional[str] = None

    @property
    def current_project_path(self) -> Optional[str]:
        return self._current_project_path

    def _set_status(self, text: str) -> None:
        self.mw.set_status_text(str(text))

    def _serialize_project_state(self) -> Dict[str, object]:
        return dict(self.mw.serialize_project_state())

    def _apply_project_payload(self, data: Dict[str, object], *, record_undo: bool) -> bool:
        return bool(self.mw.apply_project_payload(data, record_undo=record_undo))

    def _add_to_recent_files(self, filepath: str) -> None:
        self.mw.add_to_recent_files(str(filepath))

    def _clear_app_undo_history(self) -> None:
        stack = getattr(self.mw, "_undo_stack", None)
        if stack is not None and hasattr(stack, "clear"):
            stack.clear()

    def _update_window_title(self) -> None:
        if self._current_project_path is not None:
            self.mw.setWindowTitle(f"Kindred \u2014 {os.path.basename(self._current_project_path)}")
        else:
            self.mw.setWindowTitle("Kindred")

    # ------------------------------------------------------------------
    # Project load/save
    # ------------------------------------------------------------------
    def load_project(self) -> None:
        """Load project from JSON file chosen via QFileDialog."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mw,
            "Load Project",
            "",
            "Kindred Project (*.kin);;JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        self._load_project_from_path(
            filename,
            record_undo=False,
            add_to_recent=True,
            status_path=filename,
        )

    def save_project(self) -> bool:
        """Overwrite the current file, or prompt if no path is known."""
        if self._current_project_path is not None:
            return self._write_project_to_path(self._current_project_path)
        return self.save_project_as()

    def save_project_as(self) -> bool:
        """Always prompt for a file path and save."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw,
            "Save Project As",
            self._current_project_path or "",
            "Kindred Project (*.kin);;JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return False
        return self._write_project_to_path(filename)

    def _write_project_to_path(self, filepath: str) -> bool:
        """Serialize and write the project to the given file path."""
        try:
            with BusyCursor():
                data = self._serialize_project_state()
                with open(filepath, "w") as handle:
                    json.dump(data, handle, indent=2)
        except Exception as exc:
            logger.error("Failed to save project: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.mw,
                "Save Error",
                f"Failed to save project:\n\n{exc}",
            )
            return False

        # File write succeeded — title and recent-files are best-effort.
        self._current_project_path = filepath
        try:
            self._update_window_title()
            self._set_status(f"Saved project: {filepath}")
            logger.info("Saved project to %s", filepath)
            self._add_to_recent_files(filepath)
        except Exception:
            logger.warning("Post-save bookkeeping failed for %s", filepath, exc_info=True)
        return True

    def new_project(self) -> None:
        """Reset the application to an empty project state.

        Prompts the user to save unsaved work before clearing.
        """
        # ── Single-document assumption ──────────────────────────────────
        # Kindred is a single-document application.  This method clears
        # all project state to start fresh.  If multi-document support is
        # added, this flow must be scoped per-document instead.
        # ────────────────────────────────────────────────────────────────
        reply = QtWidgets.QMessageBox.question(
            self.mw,
            "New Project",
            "Do you want to save the current project before starting a new one?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
            return
        if reply == QtWidgets.QMessageBox.StandardButton.Save:
            if not self.save_project():
                return

        payload = get_default_project_payload()
        for key in FITTING_DEFAULTS_KEYS:
            payload.pop(key, None)
        get_user_preference = self.mw.config_controller.get_user_preference
        for key in QSETTINGS_KEY_MAP:
            if key in FITTING_DEFAULTS_KEYS:
                continue
            payload[key] = get_user_preference(key)
        applied = self._apply_project_payload(payload, record_undo=False)
        if not applied:
            return
        self._clear_app_undo_history()
        self._current_project_path = None
        self._update_window_title()
        self._set_status("New project")

    def load_recent_project(self, filepath: str) -> None:
        """Load a project from the Recent Projects menu."""
        if not os.path.exists(filepath):
            QtWidgets.QMessageBox.warning(
                self.mw,
                "File Not Found",
                f"File no longer exists:\n{filepath}",
            )
            recent_files = self.mw._settings_owner.qsettings.value("recent_files", [])
            if filepath in recent_files:
                recent_files.remove(filepath)
                self.mw._settings_owner.qsettings.setValue("recent_files", recent_files)
                self.mw.config_controller.update_recent_files_menu()
            return

        self._load_project_from_path(
            filepath,
            record_undo=False,
            add_to_recent=False,
            status_path=os.path.basename(filepath),
        )

    def _load_project_from_path(
        self,
        filepath: str,
        *,
        record_undo: bool,
        add_to_recent: bool,
        status_path: str,
    ) -> None:
        try:
            with BusyCursor():
                with open(filepath, "r") as handle:
                    data = json.load(handle)

                applied = self._apply_project_payload(data, record_undo=record_undo)
                if not bool(applied):
                    return
                if not record_undo:
                    self._clear_app_undo_history()
                self._current_project_path = filepath
                self._update_window_title()
                self._set_status(f"Loaded project: {status_path}")
            logger.info("Loaded project from %s", filepath)

            if add_to_recent:
                self._add_to_recent_files(filepath)

        except Exception as exc:
            logger.error("Failed to load project: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.mw,
                "Load Error",
                f"Failed to load project:\n\n{exc}",
            )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def export_data(self) -> None:
        """Export simulation/dataset data to a CSV file using ExportDialog."""
        payload = self._validate_export_preconditions("data")
        if payload is None:
            return

        if self._export_dialog is None:
            self._export_dialog = ExportDialog(self.mw)
            self._export_dialog.exportAccepted.connect(self.handle_export_config)

        current_plot = self.mw._plot_tabs.get_current_plot()
        try:
            scope = current_plot.get_export_scope_preference()
        except Exception as exc:
            logger.debug("Failed to read export scope preference: %s", exc)
        else:
            try:
                self._export_dialog.set_scope(scope)
            except Exception as exc:
                logger.debug("Failed to apply export scope preference: %s", exc)

        self._export_dialog.open()

    def _warn_no_export_target(self, message: str) -> None:
        self.mw.set_status_text(message)

    def _validate_export_preconditions(self, export_type: str) -> ExportPrecondition | None:
        if export_type != "data":
            raise ValueError(f"Unknown export type: {export_type}")

        current_plot = self.mw._plot_tabs.get_current_plot()
        if self._active_transaction_export_available(current_plot):
            return ExportPrecondition(active_display_transaction=True)
        if self._is_results_main_plot(current_plot):
            self._warn_no_export_target(
                "No active simulation display transaction is available to export."
            )
            return None
        payload = self._resolve_export_payload(current_plot)
        if payload is None:
            self._warn_no_export_target(
                "No simulation or dataset data available to export.\n\n"
                "Please run a simulation (Simulation → Run) or load a dataset (Data → Load Dataset)."
            )
            return None
        return ExportPrecondition(payload=payload)

    def _active_transaction_export_available(self, plot_widget) -> bool:
        if not self._is_results_main_plot(plot_widget):
            return False
        try:
            return self.mw.results_controller.active_display_transaction() is not None
        except Exception as exc:
            logger.debug("Active display transaction export precondition unavailable: %s", exc, exc_info=True)
            return False

    def _is_results_main_plot(self, plot_widget) -> bool:
        main_plot_getter = getattr(self.mw, "main_plot", None)
        if not callable(main_plot_getter):
            return False
        try:
            return plot_widget is main_plot_getter()
        except Exception as exc:
            logger.debug("Main plot export precondition unavailable: %s", exc, exc_info=True)
            return False

    def _resolve_export_payload(self, plot_widget) -> Optional[Dict[str, object]]:
        if plot_widget is None:
            return None

        try:
            payload = plot_widget.export_payload()
        except Exception:
            payload = None
        if isinstance(payload, dict) and payload:
            normalized = self._normalize_export_payload(payload)
            if normalized is not None:
                return normalized

        try:
            data = plot_widget.get_dataset_data()
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            t_raw = data.get("t")
            if t_raw is None:
                return None
            t_values = np.asarray(t_raw, dtype=float).reshape(-1)
            if "species" in data and isinstance(data.get("species"), dict):
                species_map = dict(data["species"])
            else:
                species_map = {k: v for k, v in data.items() if k != "t"}
            species_map = {str(k): v for k, v in species_map.items() if k and k != "species"}
            if not species_map:
                return None
            return {"t": t_values, "series": species_map}

        return None

    @staticmethod
    def _normalize_export_payload(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        t_raw = payload.get("t")
        if t_raw is None:
            return None
        series = payload.get("series")
        if not isinstance(series, dict) or not series:
            return None
        return {"t": np.asarray(t_raw, dtype=float).reshape(-1), "series": dict(series)}

    def handle_export_config(self, config: dict) -> None:
        try:
            path = config["path"]
            scope = config.get("scope", "axis")

            current_plot = self.mw._plot_tabs.get_current_plot()
            precondition = self._validate_export_preconditions("data")
            if precondition is None:
                return

            header, rows = self._prepare_default_export_rows(current_plot, scope)

            with BusyCursor():
                with open(path, "w", newline="") as handle:
                    writer = csv.writer(handle, delimiter=",")
                    writer.writerow(header)
                    for row in rows:
                        writer.writerow(row)

                self.mw.set_status_text(f"Exported CSV (scope={scope}): {path}")
            logger.info("Exported CSV to %s (scope=%s)", path, scope)

        except ValueError as exc:
            logger.warning("CSV export aborted: %s", exc)
            QtWidgets.QMessageBox.warning(self.mw, "Export Error", str(exc))
        except Exception as exc:
            logger.error("Failed to export CSV: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(self.mw, "Export Error", f"Failed to export CSV:\n\n{exc}")

    def _prepare_default_export_rows(self, plot, scope: str) -> Tuple[List[str], List[List[object]]]:
        active_transaction_rows = self._prepare_active_transaction_export_rows(plot, scope)
        if active_transaction_rows is not None:
            return active_transaction_rows
        if self._is_results_main_plot(plot):
            raise ValueError("No active simulation display transaction is available to export.")

        try:
            export = plot.build_visible_export(scope)
        except AttributeError:
            pass
        else:
            if isinstance(export, tuple) and len(export) == 2:
                return export

        payload = self._resolve_export_payload(plot)
        if payload:
            return self._prepare_payload_export_rows(payload, plot=plot, scope=scope)

        raise ValueError(
            "Current plot does not implement the export interface (expected build_visible_export() or export_payload())."
        )

    def _prepare_active_transaction_export_rows(
        self,
        plot,
        scope: str,
    ) -> Optional[Tuple[List[str], List[List[object]]]]:
        if self._is_results_main_plot(plot):
            try:
                return self.mw.results_controller.build_main_plot_csv_export(scope)
            except ValueError:
                raise
            except Exception as exc:
                logger.debug("Active display transaction export unavailable: %s", exc, exc_info=True)
        return None

    @staticmethod
    def _prepare_payload_export_rows(
        payload: Dict[str, object],
        *,
        plot,
        scope: str,
    ) -> Tuple[List[str], List[List[object]]]:
        # Payload-based export is a generic fallback for plot widgets that do not
        # implement the public `build_visible_export(scope)` interface. This code
        # must not reach into plot-private UI state.
        _ = plot
        scope = str(scope or "")
        t_values = np.asarray(payload.get("t"), dtype=float).reshape(-1)
        if t_values.size == 0:
            raise ValueError("Time axis has no points to export.")

        series = payload.get("series") or {}
        if not isinstance(series, dict) or not series:
            raise ValueError("No series data available to export.")

        species_names = [str(name) for name in series.keys() if str(name)]

        arrays: List[np.ndarray] = []
        for name in species_names:
            arr = np.asarray(series[name], dtype=float).reshape(-1)
            if arr.shape[0] != t_values.shape[0]:
                raise ValueError(
                    f"Series '{name}' length ({arr.shape[0]}) does not match time grid ({t_values.shape[0]})."
                )
            arrays.append(arr)

        header = ["Time"] + species_names

        def row_iter():
            for idx in range(t_values.shape[0]):
                yield [t_values[idx]] + [arr[idx] for arr in arrays]

        return header, row_iter()
