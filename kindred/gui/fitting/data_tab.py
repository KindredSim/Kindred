"""Fit-local simulation interval and retained-observation controls."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.validation import parse_finite_numeric_text, six_digit_text
from kindred.gui.fitting.launch import fit_execution_span
from kindred.gui.ui_helpers import (
    INVALID_EDITOR_STYLE,
    ChoicesComboBox,
    ControlRowsLayout,
    configure_compact_numeric_edit,
    ignore_wheel,
    set_bounded_label_text,
)


_POINTS = "maximum_observations_per_target"
_RANGE_FIELDS = ("data_range_min", "data_range_max")

_FIELD_LABELS = {
    "t_start": "Simulation start",
    "simulation_end": "Simulation end",
    "data_range_min": "Data Range minimum",
    "data_range_max": "Data Range maximum",
}


class DataTab(QtWidgets.QWidget):
    """Presentation for one dataset's retained-observation controls.

    Each valid edit applies at once.  An edit that cannot apply keeps its
    text, blocks Run, and stays until corrected or discarded with Escape.
    """

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
        # Settings that could not apply, by dataset.
        self._held: dict[str, dict[str, object]] = {}
        self._invalid_raw: dict[tuple[str, str], str] = {}
        self._changed_fields: dict[str, set[str]] = {}
        self._refreshing = False

        self._build_ui()

    def select_dataset(self, dataset_id: str, display_name: str | None = None) -> None:
        if display_name is not None:
            self._display_names[dataset_id] = display_name
        self._current_dataset_id = dataset_id
        self._project_controls()

    def data_edit_issues(self, dataset_ids: Iterable[str]) -> dict[str, str]:
        """Return each listed dataset's Fit Data error; omit valid datasets."""

        issues = {}
        for dataset_id in dataset_ids:
            config = self._held.get(dataset_id)
            if config is None:
                config = self._settings_getter(dataset_id)
            error = self._validation_error(dataset_id, self._owned_settings(config))
            if error:
                issues[dataset_id] = error
        return issues

    def clear_pending_edits(self) -> None:
        """Discard every held Fit Data edit and refresh the visible controls."""

        self._held.clear()
        self._changed_fields.clear()
        self._invalid_raw.clear()
        self._project_controls()

    def _build_ui(self) -> None:
        # The title and Reset sit above the wrapping fields and the error
        # below them, so all three stay in view.
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        self._header = QtWidgets.QLabel(self)
        set_bounded_label_text(self._header, "Fit Data", max_width=16777215)
        self._header.setStyleSheet("font-weight: bold;")
        header.addWidget(self._header, 1)
        self._reset_button = QtWidgets.QPushButton("Reset", self)
        self._reset_button.setObjectName("global_fit_data_range_reset")
        self._reset_button.setToolTip(
            "Restore this dataset's default Simulation interval, Data Range "
            "and points per target."
        )
        header.addWidget(self._reset_button)
        layout.addLayout(header)

        # Each setting is one label and its controls; the settings fill each
        # line and wrap onto the next as the panel narrows, as Fit Settings do.
        fields = QtWidgets.QWidget(self)
        fields.setObjectName("global_fit_data_range_panel")
        self._fields = ControlRowsLayout(fields)
        self._start_edit = self._number_edit("global_fit_t_start", 11, "Simulation start")
        self._end_edit = self._number_edit("global_fit_simulation_end", 11, "Simulation end")
        self._min_edit = self._number_edit("global_fit_data_range_min", 11, "Data Range minimum")
        self._max_edit = self._number_edit("global_fit_data_range_max", 11, "Data Range maximum")
        self.setFocusProxy(self._start_edit)
        self._maximum_edit = self._number_edit(
            "global_fit_maximum_observations",
            6,
            "Observations used per target, sampled evenly including both ends. "
            "The full count uses every observation.",
        )
        self._maximum_total = QtWidgets.QLabel("/ 0", fields)
        self._x_combo = ChoicesComboBox(fields)
        self._x_combo.setObjectName("global_fit_parametric_x")
        self._x_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._x_combo.setMinimumContentsLength(12)
        ignore_wheel(self._x_combo)
        self._used_label = QtWidgets.QLabel("Used: —", fields)
        self._used_label.setObjectName("global_fit_used_count")
        self.add_group("Simulation time (s)", self._row(fields, self._start_edit, "to", self._end_edit))
        self._range_group = self.add_group(
            "Data Range (s)", self._row(fields, self._min_edit, "to", self._max_edit),
        )
        self.add_group("Points per target", self._row(fields, self._maximum_edit, self._maximum_total))
        self.add_group("Fit against", self._x_combo)
        self._fields.addWidget(self._used_label)
        layout.addWidget(fields)
        layout.addStretch(1)

        self._error_label = QtWidgets.QLabel(self)
        self._error_label.setObjectName("global_fit_data_range_error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._editors = editors = (
            ("t_start", self._start_edit),
            ("simulation_end", self._end_edit),
            ("data_range_min", self._min_edit),
            ("data_range_max", self._max_edit),
            (_POINTS, self._maximum_edit),
        )
        # Escape discards held edits in these controls only; a row another
        # owner adds keeps its own Escape.
        self._discard_shortcuts = []
        for control in (*(editor for _field, editor in editors), self._x_combo):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), control)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            shortcut.setEnabled(False)
            shortcut.activated.connect(self._discard)
            self._discard_shortcuts.append(shortcut)
        for field, editor in editors:
            editor.textChanged.connect(
                lambda *_args, changed=field: self._mark_changed(changed)
            )
            editor.editingFinished.connect(self._commit)
        self._x_combo.currentIndexChanged.connect(
            lambda *_args: self._commit("parametric_x")
        )
        self._reset_button.clicked.connect(self._reset)

    def add_group(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Add one labelled setting to the wrapping fields, before Used."""

        group = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(group)
        # The right margin separates neighbouring settings on a line.
        form.setContentsMargins(0, 0, 12, 0)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.addRow(label, widget)
        used = self._fields.indexOf(self._used_label)
        self._fields.addWidget(group)
        if used >= 0:
            # Keep Used last.
            self._fields.addItem(self._fields.takeAt(used))
        return group

    def event(self, event: QtCore.QEvent) -> bool:
        # The panel is never shorter than its wrapped settings need, so its
        # divider cannot hide them.
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.LayoutRequest):
            self.setMinimumHeight(self.layout().totalHeightForWidth(self.width()))
        return super().event(event)

    @staticmethod
    def _row(parent: QtWidgets.QWidget, *parts: QtWidgets.QWidget | str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for part in parts:
            layout.addWidget(QtWidgets.QLabel(part, row) if isinstance(part, str) else part)
        layout.addStretch(1)
        return row

    def _number_edit(self, object_name: str, characters: int, tooltip: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(self)
        edit.setObjectName(object_name)
        configure_compact_numeric_edit(edit, characters=characters)
        edit.setToolTip(tooltip)
        return edit

    def _project_controls(self) -> None:
        enabled = bool(self._current_dataset_id)
        label = self._display_names.get(self._current_dataset_id, self._current_dataset_id)
        set_bounded_label_text(self._header,
                               f"Fit Data — {label}" if enabled else "Fit Data — No Dataset Selected",
                               max_width=16777215)
        self._header.setEnabled(enabled)
        for editor in (
            self._start_edit,
            self._end_edit,
            self._min_edit,
            self._max_edit,
        ):
            editor.setEnabled(enabled)
        self._x_combo.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)
        if not enabled:
            self._maximum_edit.setEnabled(False)
            self._maximum_edit.clear()
            self._maximum_total.setText("/ 0")
            self._used_label.setText("Used: —")
            for _field, editor in self._editors:
                editor.setStyleSheet("")
            self._show_error(None)
            return
        held = self._held.get(self._current_dataset_id)
        config = self._owned_settings(
            held if held is not None else self._settings_getter(self._current_dataset_id)
        )
        self._refreshing = True
        try:
            for field, editor in (
                ("t_start", self._start_edit),
                ("simulation_end", self._end_edit),
                ("data_range_min", self._min_edit),
                ("data_range_max", self._max_edit),
            ):
                editor.setText(
                    self._invalid_raw.get(
                        (self._current_dataset_id, field),
                        self._format(config.get(field)),
                    )
                )
            selected_x = config["parametric_x"]
            options = self._settings_getter(self._current_dataset_id)["eligible_parametric_x"]
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
        self._refresh_feedback(config)

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
        except ValueError:
            return 0
        return max((len(item.y) for item in retained), default=0)

    def _show_points(self, config: Mapping[str, object]) -> int:
        """Show the cap beside the available count; the full count means no cap."""

        dataset_id = self._current_dataset_id
        maximum = config.get("maximum_observations_per_target")
        limit = self._available_observation_limit(dataset_id, config)
        self._maximum_edit.setText(self._invalid_raw.get(
            (dataset_id, "maximum_observations_per_target"),
            str(limit if maximum is None else min(int(maximum), limit)),
        ))
        self._maximum_total.setText(f"/ {limit}")
        # Sampling keeps at least two observations per target.
        self._maximum_edit.setEnabled(limit > 2)
        return limit

    @classmethod
    def _owned_settings(
        cls,
        settings: Mapping[str, object],
    ) -> dict[str, object]:
        return {field: settings.get(field) for field in cls._OWNED_FIELDS}

    def _mark_changed(self, field: str) -> None:
        if not self._refreshing and self._current_dataset_id:
            self._changed_fields.setdefault(self._current_dataset_id, set()).add(field)

    def _capture(self) -> dict[str, object]:
        dataset_id = self._current_dataset_id
        held = self._held.get(dataset_id)
        config = self._owned_settings(
            held if held is not None else self._settings_getter(dataset_id)
        )
        changed = self._changed_fields.get(dataset_id, set())
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
            except ValueError:
                self._invalid_raw[(dataset_id, field)] = raw
            else:
                self._invalid_raw.pop((dataset_id, field), None)
                config[field] = value
        if "parametric_x" in changed:
            config["parametric_x"] = str(self._x_combo.currentData() or "t")
        if _POINTS in changed:
            # The available count depends on the other settings just read.
            key = (dataset_id, _POINTS)
            raw = self._maximum_edit.text()
            try:
                config[_POINTS] = self._parse_points(
                    raw, self._available_observation_limit(dataset_id, config)
                )
            except ValueError:
                self._invalid_raw[key] = raw
            else:
                self._invalid_raw.pop(key, None)
        return config

    @staticmethod
    def _parse_points(raw: str, limit: int) -> int | None:
        """Read a typed points count against the available count; all means no cap."""

        value = int(raw.strip())
        if not 2 <= value <= limit:
            raise ValueError
        return None if value == limit else value

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
                return f"{_FIELD_LABELS[field]} must be a finite number."
        raw_points = self._invalid_raw.get((dataset_id, _POINTS))
        if raw_points is not None:
            # Rechecked each time: other targets can change the available count.
            limit = self._available_observation_limit(dataset_id, config)
            try:
                self._parse_points(raw_points, limit)
            except ValueError:
                return f"Points per target must be a whole number from 2 to {limit}."
        try:
            start, end = fit_execution_span(config["t_start"], config["simulation_end"])
        except ValueError as exc:
            return str(exc)
        try:
            retained = self._retained_observations_getter(dataset_id, config)
        except ValueError as exc:
            return str(exc)
        times = [value for item in retained for value in item.t] if is_time else []
        if times and (min(times) < start or max(times) > end):
            return (
                "Simulation start and end must cover the selected data times, "
                f"{six_digit_text(min(times))} to {six_digit_text(max(times))}."
            )
        selected_x = str(config.get("parametric_x", "t") or "t")
        if selected_x != "t" and selected_x not in tuple(
            str(item) for item in accepted.get("eligible_parametric_x", ())
        ):
            return "Fit against must be a measured series of this dataset."
        return None

    def _commit(self, changed_field: str | None = None) -> None:
        """Apply the shown settings when valid; otherwise hold them."""

        if self._refreshing or not self._current_dataset_id:
            return
        dataset_id = self._current_dataset_id
        if changed_field is not None:
            self._mark_changed(changed_field)
        self._apply_or_hold(dataset_id, self._capture())

    def _apply_or_hold(self, dataset_id: str, config: dict[str, object]) -> None:
        if self._validation_error(dataset_id, config) is not None:
            self._held[dataset_id] = config
            self._refresh_feedback(config)
            return
        config = self._accept(dataset_id, config)
        applied = self._signature(config) != self._signature(
            self._owned_settings(self._settings_getter(dataset_id))
        )
        if applied:
            # The owner projects the accepted settings back into the controls.
            self.fitSettingsApplied.emit(dataset_id, dict(config))
        else:
            self._project_controls()

    def apply_valid_held_settings(self) -> None:
        """Apply each held edit that another setup change has made valid."""

        for dataset_id, config in tuple(self._held.items()):
            if self._validation_error(dataset_id, config) is None:
                self.fitSettingsApplied.emit(dataset_id, dict(self._accept(dataset_id, config)))

    def _accept(self, dataset_id: str, config: dict[str, object]) -> dict[str, object]:
        """Settle valid settings for applying; keep text the settings do not use.

        A held points count is read into the settings.  Measured X ignores the
        Data Range, so its held text stays for a return to Time.
        """

        raw_points = self._invalid_raw.get((dataset_id, _POINTS))
        if raw_points is not None:
            config[_POINTS] = self._parse_points(
                raw_points, self._available_observation_limit(dataset_id, config)
            )
        inactive = {} if str(config.get("parametric_x") or "t") == "t" else {
            key: text for key, text in self._invalid_raw.items()
            if key[0] == dataset_id and key[1] in _RANGE_FIELDS
        }
        self._discard_held(dataset_id)
        self._invalid_raw.update(inactive)
        if inactive:
            self._changed_fields[dataset_id] = {field for _dataset_id, field in inactive}
        return config

    def _discard_held(self, dataset_id: str) -> None:
        self._held.pop(dataset_id, None)
        self._changed_fields.pop(dataset_id, None)
        for key in tuple(self._invalid_raw):
            if key[0] == dataset_id:
                self._invalid_raw.pop(key, None)

    def _discard(self) -> None:
        dataset_id = self._current_dataset_id
        if dataset_id not in self._held:
            return
        self._discard_held(dataset_id)
        self._project_controls()
        self.statusMessage.emit(
            f"Invalid Fit Data values for {self._display_names.get(dataset_id, dataset_id)} discarded."
        )

    def _reset(self) -> None:
        if not self._current_dataset_id:
            return
        dataset_id = self._current_dataset_id
        defaults = self._owned_settings(self._default_settings_getter(dataset_id))
        defaults["parametric_x"] = str(self._x_combo.currentData() or "t")
        self._discard_held(dataset_id)
        self._apply_or_hold(dataset_id, defaults)
        if dataset_id in self._held:
            self._project_controls()

    def _refresh_feedback(self, config: Mapping[str, object]) -> None:
        """Show the shown settings' points, invalid fields, error and observation count."""

        dataset_id = self._current_dataset_id
        refreshing, self._refreshing = self._refreshing, True
        try:
            # Held settings change the available count too.
            limit = self._show_points(config)
        finally:
            self._refreshing = refreshing
        for field, editor in self._editors:
            raw = self._invalid_raw.get((dataset_id, field))
            if raw is not None and field == _POINTS:
                try:
                    self._parse_points(raw, limit)
                except ValueError:
                    pass
                else:
                    raw = None
            editor.setStyleSheet(INVALID_EDITOR_STYLE if raw is not None else "")
        self._refresh_parametric_presentation()
        try:
            used = sum(len(item.y) for item in self._retained_observations_getter(dataset_id, config))
            self._used_label.setText(f"Used: {used} observations")
        except ValueError:
            self._used_label.setText("Used: —")
        held = dataset_id in self._held
        error = self._validation_error(dataset_id, config)
        self._show_error(
            f"{error}\nPress Esc in these fields to discard the invalid values."
            if error and held else error
        )
        for shortcut in self._discard_shortcuts:
            shortcut.setEnabled(held)
        self.validityChanged.emit()

    def _show_error(self, text: str | None) -> None:
        self._error_label.setText(text or "")
        self._error_label.setVisible(bool(text))

    def _refresh_parametric_presentation(self) -> None:
        is_time = str(self._x_combo.currentData() or "t") == "t"
        self._range_group.setVisible(is_time)
