"""Fit-local data-range and observation-count presentation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    compact_dataset_label,
)
from kindred.gui.ui_helpers import setup_scientific_validator
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter


class DataTab(QtWidgets.QWidget):
    """Presentation for one dataset's retained-observation controls."""

    fitSettingsApplied = QtCore.Signal(str, dict)
    statusMessage = QtCore.Signal(str)
    validityChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        settings_getter: Callable[[str], Mapping[str, object]],
        default_settings_getter: Callable[[str], Mapping[str, object]],
        retained_observations_getter: Callable[
            [str, Mapping[str, object]], tuple[object, ...]
        ],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings_getter = settings_getter
        self._default_settings_getter = default_settings_getter
        self._retained_observations_getter = retained_observations_getter
        self._current_dataset_id = ""
        self._display_names: dict[str, str] = {}
        self._pending: dict[str, dict[str, object]] = {}
        self._invalid_raw: dict[tuple[str, str], str] = {}
        self._refreshing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._panel = self._build_panel()
        layout.addWidget(self._panel, stretch=1)

    @property
    def invalid_editor_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._invalid_raw))

    def select_dataset(self, dataset_id: str, display_name: str | None = None) -> None:
        dataset_id = str(dataset_id or "").strip()
        if display_name is not None:
            self._display_names[dataset_id] = str(display_name or dataset_id)
        self._current_dataset_id = dataset_id
        self._project_controls()

    def data_range_validation_error(
        self,
        *,
        dataset_id: str,
        config: Mapping[str, object],
    ) -> str | None:
        return self._validation_error(dataset_id, config)

    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("global_fit_data_range_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._header = QtWidgets.QLabel("Fit data (selected dataset: —)", panel)
        self._header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._header)

        self._footer = ConfigPanelFooter(
            panel,
            show_divider=False,
            show_reset=True,
            show_secondary_error=True,
            messages_position="after_body",
            apply_requires_no_error=True,
            button_order=("reset", "revert", "apply"),
            error_object_name="global_fit_data_range_error",
            apply_object_name="global_fit_data_range_apply",
            revert_object_name="global_fit_data_range_revert",
            reset_object_name="global_fit_data_range_reset",
        )
        layout.addWidget(self._footer, stretch=1)
        controls = QtWidgets.QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(4)
        self._start_edit = self._number_edit("global_fit_simulation_start")
        self._min_edit = self._number_edit("global_fit_data_range_min")
        self._max_edit = self._number_edit("global_fit_data_range_max")
        self._maximum_edit = QtWidgets.QLineEdit(self._footer)
        self._maximum_edit.setObjectName("global_fit_maximum_observations")
        self._maximum_edit.setPlaceholderText("All")
        self._maximum_edit.setValidator(QtGui.QIntValidator(0, 2_000_000_000, self))
        controls.addWidget(
            QtWidgets.QLabel("Simulation start", self._footer), 0, 0
        )
        controls.addWidget(self._start_edit, 0, 1)
        range_widget = QtWidgets.QWidget(self._footer)
        range_layout = QtWidgets.QHBoxLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(self._min_edit)
        range_layout.addWidget(QtWidgets.QLabel("to", range_widget))
        range_layout.addWidget(self._max_edit)
        controls.addWidget(QtWidgets.QLabel("Data Range", self._footer), 1, 0)
        controls.addWidget(range_widget, 1, 1)
        controls.addWidget(
            QtWidgets.QLabel("Maximum observations per target", self._footer),
            2,
            0,
        )
        controls.addWidget(self._maximum_edit, 2, 1)
        self._solver_combo = QtWidgets.QComboBox(self._footer)
        self._solver_combo.setObjectName("global_fit_solver")
        self._solver_combo.addItem("BDF", "BDF")
        self._solver_combo.addItem("Radau", "Radau")
        controls.addWidget(QtWidgets.QLabel("Fit solver", self._footer), 3, 0)
        controls.addWidget(self._solver_combo, 3, 1)
        self._used_label = QtWidgets.QLabel("Used: —", self._footer)
        self._used_label.setObjectName("global_fit_used_count")
        self._used_label.setStyleSheet("font-size: 11px;")
        controls.addWidget(self._used_label, 4, 0, 1, 2)
        self._footer.body_layout.addLayout(controls)

        for editor in (self._start_edit, self._min_edit, self._max_edit, self._maximum_edit):
            editor.textChanged.connect(self._controls_changed)
        self._solver_combo.currentIndexChanged.connect(self._controls_changed)
        self._footer.applyRequested.connect(self._apply)
        self._footer.revertRequested.connect(self._revert)
        self._footer.resetRequested.connect(self._reset)
        return panel

    @staticmethod
    def _number_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setValidator(setup_scientific_validator(edit))
        return edit

    def _project_controls(self) -> None:
        enabled = bool(self._current_dataset_id)
        self._header.setText(
            "Fit data (selected dataset: —)"
            if not enabled
            else "Fit data (selected dataset: "
            + compact_dataset_label(
                self._display_names.get(self._current_dataset_id, self._current_dataset_id),
                max_chars=DATASET_LIST_LABEL_MAX_CHARS,
            ).display
            + ")"
        )
        self._header.setEnabled(enabled)
        for editor in (self._start_edit, self._min_edit, self._max_edit, self._maximum_edit):
            editor.setEnabled(enabled)
        self._solver_combo.setEnabled(enabled)
        if self._footer.reset_button is not None:
            self._footer.reset_button.setEnabled(enabled)
        if not enabled:
            self._used_label.setText("Used: —")
            self._footer.set_error(None)
            self._footer.set_dirty(False)
            return
        config = dict(self._pending.get(self._current_dataset_id, self._settings_getter(self._current_dataset_id)))
        self._refreshing = True
        try:
            self._start_edit.setText(
                self._invalid_raw.get(
                    (self._current_dataset_id, "t_start"),
                    self._format(config.get("t_start", 0.0)),
                )
            )
            self._min_edit.setText(
                self._invalid_raw.get(
                    (self._current_dataset_id, "data_range_min"),
                    self._format(config.get("data_range_min", 0.0)),
                )
            )
            self._max_edit.setText(
                self._invalid_raw.get(
                    (self._current_dataset_id, "data_range_max"),
                    self._format(config.get("data_range_max", 0.0)),
                )
            )
            maximum = config.get("maximum_observations_per_target")
            self._maximum_edit.setText(
                self._invalid_raw.get(
                    (self._current_dataset_id, "maximum_observations_per_target"),
                    "All" if maximum in (None, 0, "") else str(maximum),
                )
            )
            self._solver_combo.setCurrentText(str(config.get("solver", "BDF")))
        finally:
            self._refreshing = False
        self._controls_changed()

    @staticmethod
    def _format(value: object) -> str:
        return format(float(value), ".12g")

    def _capture(self) -> dict[str, object]:
        dataset_id = self._current_dataset_id
        config = dict(self._pending.get(dataset_id, self._settings_getter(dataset_id)))
        controls = (
            ("t_start", self._start_edit),
            ("data_range_min", self._min_edit),
            ("data_range_max", self._max_edit),
        )
        for field, editor in controls:
            raw = editor.text()
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                self._invalid_raw[(dataset_id, field)] = raw
            else:
                self._invalid_raw.pop((dataset_id, field), None)
                config[field] = value
        raw_maximum = self._maximum_edit.text().strip()
        try:
            if not raw_maximum or raw_maximum.casefold() == "all":
                maximum = None
            else:
                maximum = int(raw_maximum)
                if maximum < 2:
                    raise ValueError
        except (TypeError, ValueError, OverflowError):
            self._invalid_raw[(dataset_id, "maximum_observations_per_target")] = raw_maximum
        else:
            self._invalid_raw.pop((dataset_id, "maximum_observations_per_target"), None)
            config["maximum_observations_per_target"] = maximum
        config["solver"] = str(self._solver_combo.currentData() or "BDF")
        self._pending[dataset_id] = config
        return config

    @staticmethod
    def _signature(config: Mapping[str, object]) -> tuple[object, ...]:
        return (
            config.get("solver", "BDF"),
            config.get("t_start"),
            config.get("data_range_min"),
            config.get("data_range_max"),
            config.get("maximum_observations_per_target"),
        )

    def _validation_error(self, dataset_id: str, config: Mapping[str, object]) -> str | None:
        for field in ("t_start", "data_range_min", "data_range_max", "maximum_observations_per_target"):
            if (dataset_id, field) in self._invalid_raw:
                return "Fit data values must be finite; Maximum observations must be All or an integer ≥ 2."
        try:
            if float(config["data_range_min"]) > float(config["data_range_max"]):
                return "Data Range minimum must be ≤ maximum."
        except (KeyError, TypeError, ValueError):
            return "Complete the Fit data values before applying."
        try:
            retained = self._retained_observations_getter(dataset_id, config)
        except ValueError as exc:
            return str(exc)
        if not retained:
            return "Select at least one retained observation target."
        if float(config["data_range_max"]) <= float(config["t_start"]):
            return "Data Range maximum must be greater than Simulation start."
        return None

    def _controls_changed(self) -> None:
        if self._refreshing or not self._current_dataset_id:
            return
        config = self._capture()
        error = self._validation_error(self._current_dataset_id, config)
        try:
            used = sum(len(item.t) for item in self._retained_observations_getter(self._current_dataset_id, config))
            self._used_label.setText(f"Used: {used} observations")
        except ValueError:
            self._used_label.setText("Used: —")
        self._footer.set_error(error)
        applied = self._settings_getter(self._current_dataset_id)
        self._footer.set_dirty(
            self._signature(config) != self._signature(applied)
            or any(key[0] == self._current_dataset_id for key in self._invalid_raw)
        )
        self.validityChanged.emit()

    def _apply(self) -> None:
        if not self._current_dataset_id:
            return
        config = self._capture()
        error = self._validation_error(self._current_dataset_id, config)
        if error:
            self._footer.set_error(error)
            return
        self.fitSettingsApplied.emit(self._current_dataset_id, dict(config))
        self._pending.pop(self._current_dataset_id, None)
        self._project_controls()

    def _revert(self) -> None:
        if not self._current_dataset_id:
            return
        self._pending.pop(self._current_dataset_id, None)
        for key in tuple(self._invalid_raw):
            if key[0] == self._current_dataset_id:
                self._invalid_raw.pop(key, None)
        self._project_controls()

    def _reset(self) -> None:
        if not self._current_dataset_id:
            return
        defaults = dict(self._default_settings_getter(self._current_dataset_id))
        self._pending[self._current_dataset_id] = defaults
        for key in tuple(self._invalid_raw):
            if key[0] == self._current_dataset_id:
                self._invalid_raw.pop(key, None)
        self._project_controls()
