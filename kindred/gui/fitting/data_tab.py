"""Fit-local simulation interval and retained-observation controls."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

from PySide6 import QtCore, QtWidgets

from kindred.core.validation import parse_finite_numeric_text
from kindred.gui.fitting.launch import fit_execution_span
from kindred.gui.ui_helpers import set_bounded_label_text
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter


class DataTab(QtWidgets.QWidget):
    """Presentation for one dataset's retained-observation controls."""

    _OWNED_FIELDS = (
        "t_start",
        "data_range_min",
        "data_range_max",
        "maximum_observations_per_target",
        "parametric_x",
        "simulation_end",
    )

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
        self._changed_fields: dict[str, set[str]] = {}
        self._refreshing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._panel = self._build_panel()
        layout.addWidget(self._panel, stretch=1)

    def select_dataset(self, dataset_id: str, display_name: str | None = None) -> None:
        dataset_id = str(dataset_id or "").strip()
        if display_name is not None:
            self._display_names[dataset_id] = str(display_name or dataset_id)
        self._current_dataset_id = dataset_id
        self._project_controls()

    def has_pending_edits(self, included_dataset_ids: Iterable[str]) -> bool:
        """Return whether an included dataset has an unapplied Data draft."""

        included = {
            dataset_id
            for item in included_dataset_ids
            if (dataset_id := str(item or "").strip())
        }
        return bool(included.intersection(self._pending))

    def data_edit_issues(self, dataset_ids: Iterable[str]) -> dict[str, str | None]:
        """Return errors or None for valid unapplied drafts; omit clean datasets."""

        issues = {}
        for dataset_id in dataset_ids:
            config = self._pending.get(dataset_id)
            if config is None:
                config = self._settings_getter(dataset_id)
            error = self._validation_error(dataset_id, self._owned_settings(config))
            if error:
                issues[dataset_id] = error
            elif dataset_id in self._pending:
                issues[dataset_id] = None
        return issues

    def clear_pending_edits(self) -> None:
        """Discard every Data draft and refresh the visible controls."""

        self._pending.clear()
        self._changed_fields.clear()
        self._invalid_raw.clear()
        self._project_controls()

    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("global_fit_data_range_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._header = QtWidgets.QLabel(panel)
        set_bounded_label_text(self._header, "Fit data", max_width=16777215)
        self._header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._header)

        self._footer = ConfigPanelFooter(
            panel,
            show_dirty=True,
            show_divider=False,
            show_reset=True,
            messages_position="after_body",
            apply_requires_no_error=True,
            button_order=("reset", "revert", "apply"),
            error_object_name="global_fit_data_range_error",
            apply_object_name="global_fit_data_range_apply",
            revert_object_name="global_fit_data_range_revert",
            reset_object_name="global_fit_data_range_reset",
        )
        self._footer.hint_label.setText(
            "Apply and Revert affect data settings for this dataset only."
        )
        layout.addWidget(self._footer, stretch=1)
        controls = QtWidgets.QFormLayout()
        controls.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(4)
        self._min_edit = self._number_edit("global_fit_data_range_min")
        self._max_edit = self._number_edit("global_fit_data_range_max")
        self._start_edit = self._number_edit("global_fit_t_start")
        self._end_edit = self._number_edit("global_fit_simulation_end")
        self.setFocusProxy(self._start_edit)
        self._maximum_edit = QtWidgets.QSpinBox(self._footer)
        self._maximum_edit.setObjectName("global_fit_maximum_observations")
        self._maximum_edit.setRange(0, 0)
        self._maximum_edit.setSpecialValueText("All")
        range_widget = QtWidgets.QWidget(self._footer)
        range_layout = QtWidgets.QHBoxLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(self._min_edit)
        range_layout.addWidget(QtWidgets.QLabel("to", range_widget))
        range_layout.addWidget(self._max_edit)
        controls.addRow("Simulation start", self._start_edit)
        controls.addRow("Simulation end", self._end_edit)
        controls.addRow("Data Range", range_widget)
        self._controls = controls
        self._range_widget = range_widget
        controls.addRow("N:", self._maximum_edit)
        self._x_combo = QtWidgets.QComboBox(self._footer)
        self._x_combo.setObjectName("global_fit_parametric_x")
        controls.addRow("X quantity", self._x_combo)
        self._used_label = QtWidgets.QLabel("Used: —", self._footer)
        self._used_label.setObjectName("global_fit_used_count")
        controls.addRow(self._used_label)
        for row in range(controls.rowCount()):
            item = controls.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            if item is not None:
                item.widget().setWordWrap(True)
        self._used_label.setWordWrap(True)
        for editor in (self._min_edit, self._max_edit, self._start_edit, self._end_edit):
            editor.setMinimumWidth(editor.fontMetrics().horizontalAdvance("0.0000") + 16)
            editor.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        self._footer.body_layout.addLayout(controls)

        self._start_edit.textChanged.connect(
            lambda *_args: self._controls_changed("t_start")
        )
        self._end_edit.textChanged.connect(
            lambda *_args: self._controls_changed("simulation_end")
        )
        self._min_edit.textChanged.connect(
            lambda *_args: self._controls_changed("data_range_min")
        )
        self._max_edit.textChanged.connect(
            lambda *_args: self._controls_changed("data_range_max")
        )
        self._maximum_edit.valueChanged.connect(
            lambda *_args: self._controls_changed("maximum_observations_per_target")
        )
        self._x_combo.currentIndexChanged.connect(
            lambda *_args: self._controls_changed("parametric_x")
        )
        self._footer.applyRequested.connect(self._apply)
        self._footer.revertRequested.connect(self._revert)
        self._footer.resetRequested.connect(self._reset)
        return panel

    @staticmethod
    def _number_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        return edit

    def _project_controls(self) -> None:
        enabled = bool(self._current_dataset_id)
        label = self._display_names.get(self._current_dataset_id, self._current_dataset_id)
        set_bounded_label_text(self._header,
                               f"Fit data — {label}" if enabled else "Fit data — no dataset selected",
                               max_width=16777215)
        self._header.setEnabled(enabled)
        for editor in (
            self._start_edit,
            self._end_edit,
            self._min_edit,
            self._max_edit,
            self._maximum_edit,
        ):
            editor.setEnabled(enabled)
        self._x_combo.setEnabled(enabled)
        if self._footer.reset_button is not None:
            self._footer.reset_button.setEnabled(enabled)
        if not enabled:
            self._used_label.setText("Used: —")
            self._footer.set_error(None)
            self._footer.set_dirty(False)
            return
        accepted = self._owned_settings(
            self._settings_getter(self._current_dataset_id)
        )
        pending = self._pending.get(self._current_dataset_id)
        config = self._owned_settings(pending) if pending is not None else accepted
        self._refreshing = True
        try:
            self._start_edit.setText(
                self._invalid_raw.get(
                    (self._current_dataset_id, "t_start"),
                    self._format(config.get("t_start")),
                )
            )
            self._end_edit.setText(self._invalid_raw.get(
                (self._current_dataset_id, "simulation_end"),
                self._format(config.get("simulation_end")),
            ))
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
            self._set_maximum_range(config)
            selected_x = str(config.get("parametric_x", "t") or "t")
            options = tuple(
                str(item)
                for item in self._settings_getter(
                    self._current_dataset_id
                ).get("eligible_parametric_x", ())
            )
            self._x_combo.clear()
            self._x_combo.addItem("Time", "t")
            for option in options:
                self._x_combo.addItem(option, option)
            selected_index = self._x_combo.findData(selected_x)
            if selected_index < 0 and selected_x != "t":
                # Preserve an accepted-but-no-longer-eligible value as an
                # explicit invalid projection.  Never silently change it to
                # Time; the owner must fail closed until the setup is fixed.
                self._x_combo.addItem(
                    f"{selected_x} (unavailable)",
                    selected_x,
                )
                selected_index = self._x_combo.count() - 1
            self._x_combo.setCurrentIndex(selected_index)
        finally:
            self._refreshing = False
        self._refresh_parametric_presentation()
        self._controls_changed()

    @staticmethod
    def _format(value: object) -> str:
        return "" if value is None else repr(float(value))

    def _available_observation_limit(
        self,
        dataset_id: str,
        config: Mapping[str, object],
    ) -> int:
        """Return the largest actually available per-target observation count."""

        unlimited = dict(config)
        unlimited["maximum_observations_per_target"] = None
        try:
            retained = self._retained_observations_getter(dataset_id, unlimited)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(
            (len(getattr(item, "y", ())) for item in retained),
            default=0,
        )

    def _set_maximum_range(self, config: Mapping[str, object]) -> None:
        maximum = config.get("maximum_observations_per_target")
        maximum_limit = self._available_observation_limit(
            self._current_dataset_id,
            config,
        )
        previous_block = self._maximum_edit.blockSignals(True)
        try:
            self._maximum_edit.setRange(0, maximum_limit)
            self._maximum_edit.setValue(
                0 if maximum in (None, 0) else min(int(maximum), maximum_limit)
            )
        finally:
            self._maximum_edit.blockSignals(previous_block)

    @classmethod
    def _owned_settings(
        cls,
        settings: Mapping[str, object],
    ) -> dict[str, object]:
        return {field: settings.get(field) for field in cls._OWNED_FIELDS}

    def _capture(self, changed_field: str | None = None) -> dict[str, object]:
        dataset_id = self._current_dataset_id
        pending = self._pending.get(dataset_id)
        config = (
            self._owned_settings(pending)
            if pending is not None
            else self._owned_settings(self._settings_getter(dataset_id))
        )
        changed = set(self._changed_fields.get(dataset_id, ()))
        if changed_field is not None:
            changed.add(changed_field)
        controls = (
            ("t_start", self._start_edit),
            ("simulation_end", self._end_edit),
            ("data_range_min", self._min_edit),
            ("data_range_max", self._max_edit),
        )
        for field, editor in controls:
            if field not in changed:
                continue
            raw = editor.text()
            if not raw.strip() and field in {"t_start", "simulation_end"}:
                config[field] = None
                self._invalid_raw.pop((dataset_id, field), None)
                continue
            try:
                value = (
                    parse_finite_numeric_text(raw, field=field)
                    if field in {"t_start", "simulation_end"} else float(raw)
                )
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                self._invalid_raw[(dataset_id, field)] = raw
            else:
                self._invalid_raw.pop((dataset_id, field), None)
                config[field] = value
        if "maximum_observations_per_target" in changed:
            maximum = int(self._maximum_edit.value())
            config["maximum_observations_per_target"] = (
                None if maximum == 0 else maximum
            )
        if "parametric_x" in changed:
            config["parametric_x"] = str(self._x_combo.currentData() or "t")
        self._pending[dataset_id] = config
        return config

    @staticmethod
    def _signature(config: Mapping[str, object]) -> tuple[object, ...]:
        return (
            config.get("t_start"),
            config.get("data_range_min"),
            config.get("data_range_max"),
            config.get("maximum_observations_per_target"),
            config.get("parametric_x", "t"),
            config.get("simulation_end"),
        )

    def _validation_error(self, dataset_id: str, config: Mapping[str, object]) -> str | None:
        accepted = self._settings_getter(dataset_id)
        is_time = str(config.get("parametric_x", "t") or "t") == "t"
        active_fields = ("t_start", "simulation_end")
        if is_time:
            active_fields += ("data_range_min", "data_range_max")
        for field in active_fields:
            if (dataset_id, field) in self._invalid_raw:
                return "Fit data values must be finite."
        try:
            start, end = fit_execution_span(config["t_start"], config["simulation_end"])
            if is_time and float(config["data_range_min"]) > float(config["data_range_max"]):
                return "Data Range minimum must be ≤ maximum."
        except ValueError as exc:
            return str(exc)
        except (KeyError, TypeError):
            return "Complete the Fit data values before applying."
        maximum = config.get("maximum_observations_per_target")
        if maximum not in (None, 0):
            try:
                if int(maximum) < 2:
                    return "Maximum observations must be All or an integer ≥ 2."
            except (TypeError, ValueError, OverflowError):
                return "Maximum observations must be All or an integer ≥ 2."
        try:
            retained = self._retained_observations_getter(dataset_id, config)
        except ValueError as exc:
            return str(exc)
        if not retained:
            return "Select at least one retained observation target."
        if is_time and any(value < start or value > end for item in retained for value in item.t):
            return "Selected experimental times are outside the Simulation interval. Adjust the interval or Data Range."
        selected_x = str(config.get("parametric_x", "t") or "t")
        if selected_x != "t" and selected_x not in tuple(
            str(item) for item in accepted.get("eligible_parametric_x", ())
        ):
            return "Parametric X must be an eligible non-Y experimental series."
        return None

    def _controls_changed(self, changed_field: str | None = None) -> None:
        if self._refreshing or not self._current_dataset_id:
            return
        if changed_field is not None:
            self._changed_fields.setdefault(self._current_dataset_id, set()).add(
                changed_field
            )
        config = self._capture(changed_field)
        self._set_maximum_range(config)
        if changed_field in {
            "t_start",
            "data_range_min",
            "data_range_max",
            "parametric_x",
        }:
            maximum = int(self._maximum_edit.value())
            config["maximum_observations_per_target"] = (
                None if maximum == 0 else maximum
            )
        self._refresh_parametric_presentation()
        error = self._validation_error(self._current_dataset_id, config)
        try:
            used = sum(len(item.y) for item in self._retained_observations_getter(self._current_dataset_id, config))
            self._used_label.setText(f"Used: {used} observations")
        except ValueError:
            self._used_label.setText("Used: —")
        self._footer.set_error(error)
        applied = self._owned_settings(
            self._settings_getter(self._current_dataset_id)
        )
        dirty = (
            self._signature(config) != self._signature(applied)
            or any(
                dataset_id == self._current_dataset_id
                and (
                    config.get("parametric_x", "t") == "t"
                    or field not in {"data_range_min", "data_range_max"}
                )
                for dataset_id, field in self._invalid_raw
            )
        )
        if dirty:
            self._pending[self._current_dataset_id] = config
        else:
            self._pending.pop(self._current_dataset_id, None)
            self._changed_fields.pop(self._current_dataset_id, None)
        self._footer.set_dirty(dirty)
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
        self._changed_fields.pop(self._current_dataset_id, None)
        self._project_controls()

    def _revert(self) -> None:
        if not self._current_dataset_id:
            return
        self._pending.pop(self._current_dataset_id, None)
        self._changed_fields.pop(self._current_dataset_id, None)
        for key in tuple(self._invalid_raw):
            if key[0] == self._current_dataset_id:
                self._invalid_raw.pop(key, None)
        self._project_controls()

    def _reset(self) -> None:
        if not self._current_dataset_id:
            return
        defaults = self._owned_settings(
            self._default_settings_getter(self._current_dataset_id)
        )
        defaults["parametric_x"] = str(self._x_combo.currentData() or "t")
        self._pending[self._current_dataset_id] = defaults
        self._changed_fields[self._current_dataset_id] = set(self._OWNED_FIELDS)
        for key in tuple(self._invalid_raw):
            if key[0] == self._current_dataset_id:
                self._invalid_raw.pop(key, None)
        self._project_controls()

    def _refresh_parametric_presentation(self) -> None:
        is_time = str(self._x_combo.currentData() or "t") == "t"
        self._controls.setRowVisible(self._range_widget, is_time)
