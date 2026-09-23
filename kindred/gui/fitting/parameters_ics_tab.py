"""Fit parameter presentation with current owner-facing hooks.

The widget construction in this module is intentionally close to the executable
Fit presentation.  The widget is only a view: accepted setup, request
construction, Fit results, defaults persistence, and worker ownership remain
outside it.
"""

from __future__ import annotations

import html


import sys
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.display_name_policy import compact_text, scientific_symbol_html, scientific_symbol_spoken
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE,
    ScientificNameDelegate,
    ControlRowsLayout,
    configure_compact_numeric_edit,
)
from kindred.core.analysis.fit_uncertainty import (
    BOOTSTRAP_SEED_RANGE,
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
)
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.quantity_units import format_unit_for_display
from kindred.core.validation import parse_finite_numeric_text

from kindred.core.fitting_method_policy import SUPPORTED_FIT_METHODS
from kindred.gui.fitting.constants import FIT_METHOD_LABELS
from kindred.core.project_document import (
    FIT_MAX_NFEV_RANGE,
    FIT_SEED_RANGE,
    FitDefaults,
    SUPPORTED_SOLVERS,
    default_fit_defaults,
    validate_fit_defaults,
)
from kindred.gui.fitting.launch import (
    FitOriginSnapshot,
    FitParameterConfiguration,
    FitParameterVariable,
    FitSetupSnapshot,
    fit_uncertainty_coordinates,
    prune_fit_analysis_coordinates,
    _replace_fit_parameter,
    convert_fit_parameter_scope,
)
from kindred.gui.fitting.constants import (
    CENTRED_FIXED_EFFECTS_SCOPE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
)
from kindred.gui.parameter_enumeration import admit_gui_parameter_value
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = ["ParametersIcsTab"]


@dataclass(frozen=True, slots=True)
class _InvalidNumericDraft:
    raw_fields: tuple[tuple[str, str], ...]
    message: str


def _format_number(value: object) -> str:
    try:
        return repr(float(value))
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _format_scientific(value: object) -> str:
    # Keep the accepted binary64 value intact; this is only a projection.
    text = repr(float(value))
    if "e" not in text:
        return text
    base, exponent = text.split("e", 1)
    sign = ""
    digits = exponent
    if digits and digits[0] in "+-":
        sign = digits[0]
        digits = digits[1:]
    return f"{base}e{sign}{digits.lstrip('0') or '0'}"


def _resolve_coordinate_selection(
    values: Sequence[object],
    available: Sequence[FitCoordinate],
) -> set[FitCoordinate]:
    """Resolve stored coordinate identities without guessing display aliases."""

    roster = tuple(available)
    resolved: set[FitCoordinate] = set()
    for value in values:
        if isinstance(value, FitCoordinate):
            if value not in roster:
                raise ValueError(f"Unavailable Fit coordinate: {value.display_name!r}.")
            resolved.add(value)
            continue
        display = str(value or "").strip()
        matches = tuple(
            coordinate for coordinate in roster if coordinate.display_name == display
        )
        if len(matches) != 1:
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous Fit coordinate display name: {display!r}."
                )
            raise ValueError(f"Unavailable Fit coordinate: {display!r}.")
        resolved.add(matches[0])
    return resolved


class _ProfileLikelihoodDialog(QtWidgets.QDialog):
    """Transactional Profile likelihood configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str] | None = None,
        configuration: ProfileLikelihoodConfig | None,
        default_selection: Sequence[FitCoordinate] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profile likelihood")
        self.setModal(True)
        self.resize(520, 420)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        display_names = dict(display_names or {})
        self._typed_identity_mode = any(
            coordinate.scope in {"centred_overall", "centred_deviation"}
            for coordinate in available
        )
        selected = (
            (
                set(default_selection)
                if configuration is not None
                and not configuration.parameter_names
                and default_selection is not None
                else _resolve_coordinate_selection(
                    configuration.parameter_names,
                    available,
                )
            )
            if configuration
            else set(
                default_selection if default_selection is not None else available
            )
        )
        for coordinate in available:
            if not isinstance(coordinate, FitCoordinate):
                raise TypeError("Uncertainty rosters require typed FitCoordinate values.")
            label = display_names.get(coordinate, coordinate.display_name)
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
            item.setToolTip(coordinate.display_name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if coordinate in selected else QtCore.Qt.Unchecked
            )
        self._confidence = QtWidgets.QLineEdit(
            _format_scientific(configuration.confidence_level if configuration else 0.95),
            self,
        )
        self._points = QtWidgets.QSpinBox(self)
        self._points.setRange(2, 10_000)
        self._points.setValue(
            configuration.max_bracketing_steps_per_side if configuration else 8
        )
        self._budget = QtWidgets.QSpinBox(self)
        self._budget.setRange(1, 100_000)
        self._budget.setValue(configuration.max_nfev_per_point if configuration else 300)
        self._method = QtWidgets.QComboBox(self)
        ParametersIcsTab._add_method_items(self._method)
        method = configuration.optimizer_method if configuration else "trf"
        self._method.setCurrentIndex(max(0, self._method.findData(method)))
        configure_compact_numeric_edit(self._confidence, characters=8)
        setup_scientific_validator(self._confidence)
        form.addRow("Coordinates:", self._coordinates)
        form.addRow("Confidence level:", self._confidence)
        form.addRow("Maximum outward search steps per side", self._points)
        form.addRow("Maximum evaluations per refit:", self._budget)
        form.addRow("Optimization method:", self._method)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def configuration(self) -> ProfileLikelihoodConfig:
        names = tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self._coordinates.count())
            if (item := self._coordinates.item(index)).checkState() == QtCore.Qt.Checked
        )
        if not self._typed_identity_mode:
            self._reject_ambiguous_display_names(names)
        return ProfileLikelihoodConfig(
            parameter_names=names,
            confidence_level=float(self._confidence.text()),
            max_bracketing_steps_per_side=self._points.value(),
            max_nfev_per_point=self._budget.value(),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    @staticmethod
    def _reject_ambiguous_display_names(
        coordinates: Sequence[FitCoordinate],
    ) -> None:
        labels = [coordinate.display_name for coordinate in coordinates]
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Selected uncertainty coordinates have ambiguous display names; "
                "select only one coordinate with each displayed identity."
            )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError):
            return
        super().accept()


class _ResidualBootstrapDialog(QtWidgets.QDialog):
    """Transactional residual Bootstrap configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str] | None = None,
        configuration: ResidualBootstrapConfig | None,
        default_selection: Sequence[FitCoordinate] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Residual bootstrap")
        self.setModal(True)
        self.resize(520, 440)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        display_names = dict(display_names or {})
        self._typed_identity_mode = any(
            coordinate.scope in {"centred_overall", "centred_deviation"}
            for coordinate in available
        )
        selected = (
            _resolve_coordinate_selection(
                configuration.parameter_names,
                available,
            )
            if configuration
            else set(
                default_selection if default_selection is not None else available
            )
        )
        for coordinate in available:
            if not isinstance(coordinate, FitCoordinate):
                raise TypeError("Uncertainty rosters require typed FitCoordinate values.")
            label = display_names.get(coordinate, coordinate.display_name)
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
            item.setToolTip(coordinate.display_name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if coordinate in selected else QtCore.Qt.Unchecked
            )
        self._replicates = QtWidgets.QSpinBox(self)
        self._replicates.setRange(1, 1_000_000)
        self._replicates.setValue(configuration.replicates if configuration else 200)
        self._minimum = QtWidgets.QSpinBox(self)
        self._minimum.setRange(1, 1_000_000)
        self._minimum.setValue(
            configuration.minimum_successful_replicates if configuration else 50
        )
        self._fraction = QtWidgets.QLineEdit(
            _format_scientific(configuration.minimum_success_fraction if configuration else 0.8),
            self,
        )
        self._confidence = QtWidgets.QLineEdit(
            _format_scientific(configuration.confidence_level if configuration else 0.95),
            self,
        )
        self._seed = QtWidgets.QSpinBox(self)
        self._seed.setRange(*BOOTSTRAP_SEED_RANGE)
        bootstrap_seed = (
            configuration.seed
            if configuration is not None
            else int(
                np.random.default_rng().integers(
                    BOOTSTRAP_SEED_RANGE[0],
                    BOOTSTRAP_SEED_RANGE[1],
                    endpoint=True,
                )
            )
        )
        self._seed.setValue(bootstrap_seed)
        self._method = QtWidgets.QComboBox(self)
        ParametersIcsTab._add_method_items(self._method)
        method = configuration.optimizer_method if configuration else "trf"
        self._method.setCurrentIndex(max(0, self._method.findData(method)))
        configure_compact_numeric_edit(self._fraction, characters=8)
        setup_scientific_validator(self._fraction)
        configure_compact_numeric_edit(self._confidence, characters=8)
        setup_scientific_validator(self._confidence)
        form.addRow("Coordinates:", self._coordinates)
        form.addRow("Number of refits:", self._replicates)
        form.addRow("Minimum successful refits:", self._minimum)
        form.addRow("Required success rate:", self._fraction)
        form.addRow("Confidence level:", self._confidence)
        form.addRow("Random seed:", self._seed)
        form.addRow("Optimization method:", self._method)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def configuration(self) -> ResidualBootstrapConfig:
        names = tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self._coordinates.count())
            if (item := self._coordinates.item(index)).checkState() == QtCore.Qt.Checked
        )
        if not self._typed_identity_mode:
            _ProfileLikelihoodDialog._reject_ambiguous_display_names(names)
        return ResidualBootstrapConfig(
            parameter_names=names,
            seed=self._seed.value(),
            replicates=self._replicates.value(),
            confidence_level=float(self._confidence.text()),
            minimum_successful_replicates=self._minimum.value(),
            minimum_success_fraction=float(self._fraction.text()),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError):
            return
        super().accept()


class _ParameterRosterTable(QtWidgets.QTableWidget):
    """Prefer eight visible rows initially; the native splitter owns later sizing."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_columns_initialized", False):
            self._columns_initialized = True
            QtCore.QTimer.singleShot(0, self._initialize_columns)

    def _initialize_columns(self) -> None:
        em = self.fontMetrics().horizontalAdvance("M")
        self.setColumnWidth(2, 12 * em)
        self.setColumnWidth(3, 18 * em)
        self.setColumnWidth(4, 9 * em)
        self.setColumnWidth(1, max(
            18 * em,
            self.viewport().width() - sum(self.columnWidth(i) for i in (0, 2, 3, 4)),
        ))

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        height = self.horizontalHeader().sizeHint().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(row) for row in range(min(8, self.rowCount())))
        scrollbar = self.horizontalScrollBar()
        if self.horizontalHeader().length() > self.viewport().width():
            height += scrollbar.sizeHint().height()
        size.setHeight(height)
        return size


class _DatasetValueDelegate(QtWidgets.QStyledItemDelegate):
    """Paint compact values while native editing retains the complete text."""

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        if index.column() not in (1, 2):
            return
        parameter = self.parent()._selected_parameter()
        dataset_id = index.data(QtCore.Qt.UserRole)
        if parameter is not None and (
            "parameter_dataset", parameter.name, dataset_id, "initial"
        ) in self.parent()._invalid_numeric_drafts and index.column() == 1:
            return
        try:
            option.text = f"{float(option.text):.6g}"
        except ValueError:
            pass

    def eventFilter(self, watched, event) -> bool:
        if (
            event.type() == QtCore.QEvent.ShortcutOverride
            and event.key() == QtCore.Qt.Key_Escape
            and self.parent()._discard_parameter_shortcut.isEnabled()
        ):
            event.ignore()
            return True
        return super().eventFilter(watched, event)


class ParametersIcsTab(QtWidgets.QWidget):
    """Presentation-only parameter, IC-scope, and optimizer panel."""

    setupSnapshotRequested = QtCore.Signal(object)
    resetRequested = QtCore.Signal(str)
    viewStepsRequested = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    validityChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        origin: FitOriginSnapshot | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if origin is not None and not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot or None.")
        self._origin = origin
        self._projected_setup: FitSetupSnapshot | None = None
        self._last_fit_parameters: dict[str, FitParameterVariable] = {}
        self._last_fit_overall_values: dict[str, float] = {}
        self._tolerance_drafts: dict[str, str] = {}
        self._defaults_error_text = ""
        self._roster_height_initialized = False
        self._can_reset_initial = False
        self._can_reset_last_fit = False
        self._invalid_numeric_drafts: dict[
            tuple[str, ...], _InvalidNumericDraft
        ] = {}
        self._projecting = False
        self._running = False
        self._build_ui()


    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._build_parameters_panel(), stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._roster_height_initialized:
            QtCore.QTimer.singleShot(0, self._initialize_roster_height)

    def _on_roster_splitter_moved(self, _position: int, _index: int) -> None:
        self._roster_height_initialized = True

    def _initialize_roster_height(self) -> None:
        if self._roster_height_initialized or not self.isVisible():
            return
        self._param_table.resizeRowsToContents()
        self._param_table.updateGeometry()
        roster = self._left_splitter.widget(0)
        roster.layout().invalidate()
        available = (
            self._left_splitter.contentsRect().height()
            - self._left_splitter.handle(1).height()
        )
        preferred = roster.sizeHint().height()
        self._left_splitter.setSizes([preferred, max(0, available - preferred)])
        self._roster_height_initialized = True
    # ------------------------------------------------------------------
    # Parameter presentation and complete snapshot replacements
    # ------------------------------------------------------------------


    def _reset_to_initial(self) -> None:
        if not self._can_reset_initial:
            self.statusMessage.emit("Initial Fit setup is not available.")
            return
        self.resetRequested.emit("initial")

    def _reset_to_last_fit(self) -> None:
        if not self._can_reset_last_fit:
            self.statusMessage.emit("No completed Fit is available to restore.")
            return
        self.resetRequested.emit("last_fit")


    _SCOPE_LABELS = {
        "shared": "One shared value",
        "per_dataset": "Independent value for each dataset",
        "centred_fixed_effects": CENTRED_FIXED_EFFECTS_SCOPE_LABEL,
    }

    def _build_parameters_panel(self) -> QtWidgets.QWidget:  # noqa: C901
        widget = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(widget)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)
        container = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(container)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, container)
        split.setObjectName("fit_parameter_split")
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 58)
        split.setStretchFactor(1, 42)

        left = QtWidgets.QWidget(split)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        self._left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, left)
        self._left_splitter.setObjectName("fit_parameter_left_split")
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.splitterMoved.connect(self._on_roster_splitter_moved)
        left_layout.addWidget(self._left_splitter)
        roster_pane = QtWidgets.QWidget(self._left_splitter)
        roster_layout = QtWidgets.QVBoxLayout(roster_pane)
        roster_layout.setContentsMargins(0, 0, 0, 6)
        roster_layout.addWidget(QtWidgets.QLabel("Parameter list", roster_pane))
        self._param_table = _ParameterRosterTable(0, 5, roster_pane)
        self._param_table.setObjectName("fit_parameter_roster")
        self._param_table.setWordWrap(False)
        self._param_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self._param_table.setHorizontalHeaderLabels(
            ("Vary", "Parameter", "Mechanism value", "Last result", "Unit")
        )
        self._param_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._param_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._param_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._param_table.horizontalHeader().setStretchLastSection(False)
        self._param_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self._param_table.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._param_table.itemChanged.connect(self._on_roster_item_changed)
        self._param_table.itemSelectionChanged.connect(self._on_parameter_selection_changed)
        self._param_table.setItemDelegateForColumn(1, ScientificNameDelegate(self._param_table))
        self._param_table.setItemDelegateForColumn(4, ScientificNameDelegate(self._param_table))
        self._param_table.installEventFilter(self)
        roster_layout.addWidget(self._param_table, 1)
        split.addWidget(left)

        right = QtWidgets.QWidget(split)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        self._selected_editor = QtWidgets.QGroupBox("Selected parameter", right)
        self._selected_editor.setObjectName("fit_selected_parameter_editor")
        editor_layout = QtWidgets.QVBoxLayout(self._selected_editor)
        form = QtWidgets.QFormLayout()
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self._starting_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._minimum_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._maximum_edit = QtWidgets.QLineEdit(self._selected_editor)
        self._scale_combo = QtWidgets.QComboBox(self._selected_editor)
        self._scale_combo.addItem("Linear", False)
        self._scale_combo.addItem("Logarithmic", True)
        self._scope_combo = QtWidgets.QComboBox(self._selected_editor)
        for scope, label in self._SCOPE_LABELS.items():
            self._scope_combo.addItem(label, scope)
        self._scope_combo.setObjectName("fit_parameter_scope")
        self._starting_edit.setObjectName("fit_parameter_starting_value")
        self._minimum_edit.setObjectName("fit_parameter_minimum")
        self._maximum_edit.setObjectName("fit_parameter_maximum")
        self._scale_combo.setObjectName("fit_parameter_scale")
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            setup_scientific_validator(edit)
            configure_compact_numeric_edit(edit, characters=20)
        for combo in (self._scale_combo, self._scope_combo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._starting_caption = QtWidgets.QLabel(self._selected_editor)
        self._minimum_caption = QtWidgets.QLabel(self._selected_editor)
        self._maximum_caption = QtWidgets.QLabel(self._selected_editor)
        for caption in (self._starting_caption, self._minimum_caption, self._maximum_caption):
            caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow(self._starting_caption, self._starting_edit)
        form.addRow(self._minimum_caption, self._minimum_edit)
        form.addRow(self._maximum_caption, self._maximum_edit)
        form.addRow("Scale:", self._scale_combo)
        form.addRow("Across datasets:", self._scope_combo)
        editor_layout.addLayout(form)

        self._parameter_error = QtWidgets.QLabel(self._selected_editor)
        self._parameter_error.setWordWrap(True)
        self._parameter_error.setVisible(False)
        editor_layout.addWidget(self._parameter_error)
        self._discard_parameter_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Escape), self._selected_editor
        )
        self._discard_parameter_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._discard_parameter_shortcut.setEnabled(False)
        self._discard_parameter_shortcut.activated.connect(self._revert_selected_parameter_draft)

        self._dataset_values_group = QtWidgets.QGroupBox("Values by dataset", self._selected_editor)
        self._dataset_values_group.setObjectName("fit_parameter_values_by_dataset")
        dataset_layout = QtWidgets.QVBoxLayout(self._dataset_values_group)
        self._dataset_values_table = QtWidgets.QTableWidget(0, 4, self._dataset_values_group)
        self._dataset_values_table.setObjectName("fit_parameter_dataset_values")
        self._dataset_values_table.setWordWrap(False)
        self._dataset_values_table.setItemDelegate(_DatasetValueDelegate(self))
        self._dataset_values_table.setItemDelegateForColumn(3, ScientificNameDelegate(self._dataset_values_table))
        self._dataset_values_table.setHorizontalHeaderLabels(
            ("Dataset", "Starting value", "Last result", "Unit")
        )
        self._dataset_values_table.horizontalHeader().setStretchLastSection(True)
        self._dataset_values_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._dataset_values_table.itemChanged.connect(self._on_dataset_value_changed)
        self._dataset_values_table.installEventFilter(self)
        self._dataset_values_table.horizontalScrollBar().rangeChanged.connect(
            lambda *_: self._size_dataset_values_table(self._dataset_values_table)
        )
        dataset_layout.addWidget(self._dataset_values_table)
        editor_layout.addWidget(self._dataset_values_group)

        right_layout.addWidget(self._selected_editor)

        self._last_fit_group = QtWidgets.QGroupBox("Last Fit", right)
        last_group = self._last_fit_group
        last_layout = QtWidgets.QFormLayout(last_group)
        self._last_result_label = QtWidgets.QLabel("—", last_group)
        self._last_result_label.setWordWrap(True)
        self._last_result_caption = QtWidgets.QLabel("Last result:", last_group)
        self._last_result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        last_layout.addRow(self._last_result_caption, self._last_result_label)
        right_layout.addWidget(last_group)
        right_layout.addStretch(1)
        split.addWidget(right)
        split.setSizes([580, 420])
        layout.addWidget(split, stretch=1)

        self._roster_actions_host = QtWidgets.QWidget(roster_pane)
        actions = ControlRowsLayout(self._roster_actions_host)
        self._roster_actions_host.installEventFilter(self)
        self._view_steps_button = QtWidgets.QPushButton("View mechanism steps", left)
        self._view_steps_button.setObjectName("global_fit_parameters_view_steps")
        self._view_steps_button.clicked.connect(self.viewStepsRequested)
        self._reset_initial_button = QtWidgets.QPushButton(
            "Reset to Initial", left
        )
        self._reset_initial_button.setToolTip(
            "Restore the entire Fit setup captured when this window opened, "
            "discarding all later accepted and unapplied changes."
        )
        self._reset_initial_button.clicked.connect(self._reset_to_initial)
        self._reset_last_button = QtWidgets.QPushButton(
            "Reset to Last Fit", left
        )
        self._reset_last_button.setToolTip(
            "Restore the entire Fit setup used by the last completed Fit, "
            "discarding all later accepted and unapplied changes."
        )
        self._reset_last_button.clicked.connect(self._reset_to_last_fit)
        for control in (
            self._view_steps_button,
            self._reset_initial_button,
            self._reset_last_button,
        ):
            actions.addWidget(control)
        roster_layout.addWidget(self._roster_actions_host)
        self._left_splitter.addWidget(roster_pane)
        options_pane = QtWidgets.QWidget(self._left_splitter)
        options_layout = QtWidgets.QVBoxLayout(options_pane)
        options_layout.setContentsMargins(0, 6, 0, 0)
        options_scroll = QtWidgets.QScrollArea(self._left_splitter)
        options_scroll.setObjectName("fit_parameter_options_scroll")
        options_scroll.setStyleSheet(
            "QScrollArea#fit_parameter_options_scroll { border: none; padding: 0; margin: 0; }"
        )
        options_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        options_scroll.setWidgetResizable(True)
        options_scroll.setWidget(options_pane)
        self._left_splitter.addWidget(options_scroll)

        self._analysis_cards_host = QtWidgets.QWidget(options_pane)
        self._analysis_cards_host.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        self._analysis_cards_host.installEventFilter(self)
        self._analysis_cards_layout = QtWidgets.QGridLayout(self._analysis_cards_host)
        self._analysis_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._analysis_cards_layout.setHorizontalSpacing(8)
        self._analysis_cards_layout.setVerticalSpacing(8)

        settings_group = QtWidgets.QGroupBox("Fit settings", options_pane)
        self._settings_group = settings_group
        settings_group.installEventFilter(self)
        settings_group.setObjectName("fit_settings_group")
        settings_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
        self._solver_combo = QtWidgets.QComboBox(settings_group)
        self._solver_combo.setObjectName("global_fit_solver")
        for solver in SUPPORTED_SOLVERS:
            self._solver_combo.addItem(str(solver), str(solver))
        self._method_combo = QtWidgets.QComboBox(settings_group)
        self._method_combo.setObjectName("global_fit_method")
        for method in SUPPORTED_FIT_METHODS:
            self._method_combo.addItem(FIT_METHOD_LABELS.get(method, method), method)
        for combo in (self._method_combo, self._solver_combo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        defaults = default_fit_defaults()
        self._max_eval_spin = QtWidgets.QSpinBox(settings_group)
        self._max_eval_spin.setRange(*FIT_MAX_NFEV_RANGE)
        self._max_eval_spin.setValue(defaults.max_nfev)
        self._max_eval_spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._seed_check = QtWidgets.QCheckBox(
            "Fixed seed", settings_group
        )
        self._seed_spin = QtWidgets.QSpinBox(settings_group)
        self._seed_spin.setRange(*FIT_SEED_RANGE)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        self._ftol_edit = QtWidgets.QLineEdit(settings_group)
        self._xtol_edit = QtWidgets.QLineEdit(settings_group)
        self._ftol_edit.setObjectName("fit_objective_tolerance")
        self._xtol_edit.setObjectName("fit_parameter_tolerance")
        self._ftol_label = QtWidgets.QLabel("Objective-improvement tolerance:", settings_group)
        self._xtol_label = QtWidgets.QLabel("Parameter-step tolerance:", settings_group)
        for label, edit in ((self._ftol_label, self._ftol_edit), (self._xtol_label, self._xtol_edit)):
            configure_compact_numeric_edit(edit, characters=8)
            setup_scientific_validator(edit)
            label.setBuddy(edit)
        # Each native form keeps its label and editor together. The existing
        # wrapping layout places the paired forms on one row when they fit.
        pairs = (
            (("Optimization method:", self._method_combo), ("Maximum evaluations:", self._max_eval_spin)),
            (("Simulation solver:", self._solver_combo), (self._seed_check, self._seed_spin)),
            ((self._ftol_label, self._ftol_edit), (self._xtol_label, self._xtol_edit)),
        )
        for pair in pairs:
            row = QtWidgets.QWidget(settings_group)
            row_layout = ControlRowsLayout(row)
            row_layout.setSpacing(20)
            for label, control in pair:
                group = QtWidgets.QWidget(row)
                form = QtWidgets.QFormLayout(group)
                form.setContentsMargins(0, 0, 0, 0)
                form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
                form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
                form.addRow(label, control)
                row_layout.addWidget(group)
                if control is self._xtol_edit:
                    self._xtol_group = group
            settings_layout.addWidget(row)
        controls = (self._method_combo, self._max_eval_spin, self._solver_combo,
                    self._seed_check, self._seed_spin, self._ftol_edit, self._xtol_edit)
        for first, second in zip(controls, controls[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)
        self._defaults_error = QtWidgets.QLabel(settings_group)
        self._defaults_error.setWordWrap(True)
        self._defaults_error.hide()
        settings_layout.addWidget(self._defaults_error)
        self._discard_tolerances_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Escape), settings_group
        )
        self._discard_tolerances_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._discard_tolerances_shortcut.setEnabled(False)
        self._discard_tolerances_shortcut.activated.connect(self._discard_tolerance_drafts)

        options_layout.addWidget(settings_group)

        profile_card = QtWidgets.QGroupBox("Profile likelihood", self._analysis_cards_host)
        profile_card.setObjectName("fit_profile_card")
        profile_layout = QtWidgets.QVBoxLayout(profile_card)
        self._profile_enabled = QtWidgets.QCheckBox("Enabled", profile_card)
        self._profile_summary = QtWidgets.QLabel("Not configured", profile_card)
        self._profile_summary.setWordWrap(True)
        self._profile_configure = QtWidgets.QPushButton("Configure…", profile_card)
        profile_layout.addWidget(self._profile_enabled)
        profile_layout.addWidget(self._profile_summary)
        profile_layout.addStretch(1)
        profile_layout.addWidget(self._profile_configure)

        bootstrap_card = QtWidgets.QGroupBox("Residual bootstrap", self._analysis_cards_host)
        bootstrap_card.setObjectName("fit_bootstrap_card")
        bootstrap_layout = QtWidgets.QVBoxLayout(bootstrap_card)
        self._bootstrap_enabled = QtWidgets.QCheckBox("Enabled", bootstrap_card)
        self._bootstrap_summary = QtWidgets.QLabel("Not configured", bootstrap_card)
        self._bootstrap_summary.setWordWrap(True)
        self._bootstrap_configure = QtWidgets.QPushButton("Configure…", bootstrap_card)
        bootstrap_layout.addWidget(self._bootstrap_enabled)
        bootstrap_layout.addWidget(self._bootstrap_summary)
        bootstrap_layout.addStretch(1)
        bootstrap_layout.addWidget(self._bootstrap_configure)

        self._analysis_cards = (profile_card, bootstrap_card)
        self._analysis_card_columns = 1
        for row, card in enumerate(self._analysis_cards):
            self._analysis_cards_layout.addWidget(card, row, 0)
        options_layout.addWidget(self._analysis_cards_host)
        options_layout.addStretch(1)

        # Signals are attached after construction so projection can use one
        # coherent blocker set and cannot author a half-formed snapshot.
        self._solver_combo.currentIndexChanged.connect(self._emit_solver_edit)
        self._method_combo.currentIndexChanged.connect(
            self._emit_defaults_edit
        )
        self._max_eval_spin.valueChanged.connect(
            self._emit_defaults_edit
        )
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        self._seed_check.toggled.connect(
            self._emit_defaults_edit
        )
        self._seed_spin.valueChanged.connect(
            self._emit_defaults_edit
        )
        self._profile_enabled.toggled.connect(self._on_profile_toggled)
        self._bootstrap_enabled.toggled.connect(self._on_bootstrap_toggled)
        self._profile_configure.clicked.connect(self._open_profile_dialog)
        self._bootstrap_configure.clicked.connect(self._open_bootstrap_dialog)
        for edit in (self._ftol_edit, self._xtol_edit):
            edit.textChanged.connect(self._on_tolerance_text_changed)
            edit.editingFinished.connect(self._emit_defaults_edit)
        self._scale_combo.currentIndexChanged.connect(self._emit_selected_parameter_edit)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            edit.textChanged.connect(self._on_parameter_numeric_text_changed)
            edit.editingFinished.connect(self._emit_selected_parameter_edit)
        self._dataset_values_group.setVisible(False)
        self._selected_editor.setEnabled(False)
        self._project_defaults(defaults)
        return widget

    @staticmethod
    def _add_method_items(combo: QtWidgets.QComboBox) -> None:
        for method in SUPPORTED_FIT_METHODS:
            combo.addItem(FIT_METHOD_LABELS.get(method, method), method)

    def _parameter_dataset_ids(
        self,
        snapshot: FitSetupSnapshot,
        parameters: Sequence[FitParameterVariable],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [item.dataset_id for item in snapshot.datasets]
                + [
                    dataset_id
                    for parameter in parameters
                    for dataset_id, _value in parameter.per_dataset_config.dataset_values
                ]
            )
        )

    def _dataset_display_name(self, dataset_id: str) -> str:
        if self._origin is not None:
            for mapped in self._origin.mapped_datasets:
                if mapped.dataset_id == dataset_id:
                    return str(mapped.display_name)
        if self._projected_setup is not None:
            for dataset in self._projected_setup.datasets:
                if dataset.dataset_id == dataset_id:
                    return str(dataset_id)
        return str(dataset_id)

    def _parameter_by_name(self, name: str) -> FitParameterVariable | None:
        if self._projected_setup is None:
            return None
        return next((item for item in self._projected_setup.parameters if item.name == name), None)

    def _display_parameter_name(self, name: str) -> str:
        parameter = self._parameter_by_name(name)
        return parameter.display_name if parameter is not None else name

    def _coordinate_display_name(self, coordinate: FitCoordinate) -> str:
        if not isinstance(coordinate, FitCoordinate):
            raise TypeError("Uncertainty roster entries require FitCoordinate values.")
        if coordinate.scope == "shared":
            return self._display_parameter_name(coordinate.name)
        if coordinate.scope == "centred_overall":
            return f"{self._display_parameter_name(coordinate.name)} — overall"
        if coordinate.scope == "dataset":
            return (
                f"{self._display_parameter_name(coordinate.name)} — "
                f"{self._dataset_display_name(str(coordinate.dataset_id))}"
            )
        return coordinate.display_name

    def _available_bootstrap_coordinates(self) -> tuple[FitCoordinate, ...]:
        return self._available_analysis_coordinates()

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._origin = snapshot.origin
        self._projected_setup = snapshot
        blocker = QtCore.QSignalBlocker(self._param_table)
        self._projecting = True
        try:
            self._param_table.setRowCount(len(snapshot.parameters))
            for row, parameter in enumerate(snapshot.parameters):
                self._render_roster_row(row, parameter)
            self._param_table.resizeRowsToContents()
            self._param_table.updateGeometry()
            self._solver_combo.setCurrentText(snapshot.solver)
            self._project_defaults(snapshot.defaults)
            self._project_advanced_analysis(snapshot)
        finally:
            self._projecting = False
            del blocker
        if snapshot.parameters:
            current = self._param_table.currentRow()
            self._param_table.selectRow(
                min(max(current, 0), len(snapshot.parameters) - 1)
            )
        else:
            self._clear_selected_editor()
        self._project_last_fit_columns()

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        if snapshot is not None:
            self._last_fit_parameters = {item.name: item for item in snapshot.parameters}
        else:
            self._last_fit_parameters.clear()
            self._last_fit_overall_values.clear()
        self.set_reset_availability(
            initial=self._can_reset_initial,
            last_fit=snapshot is not None,
        )
        self._project_last_fit_columns()
        self._render_selected_parameter()

    def set_last_fit_presentation(self, overall_values: Mapping[str, float]) -> None:
        """Display retained centred settlements beside the Last Fit dataset values."""

        self._last_fit_overall_values = dict(overall_values)
        self._project_last_fit_columns()

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self.set_reset_availability(initial=True, last_fit=self._can_reset_last_fit)

    def set_reset_availability(self, *, initial: bool, last_fit: bool) -> None:
        self._can_reset_initial = bool(initial)
        self._can_reset_last_fit = bool(last_fit)
        self._reset_initial_button.setEnabled(self._can_reset_initial)
        self._reset_last_button.setEnabled(self._can_reset_last_fit)

    def _render_roster_row(self, row: int, parameter: FitParameterVariable) -> None:
        config = parameter.selected_config
        vary = QtWidgets.QTableWidgetItem()
        vary.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        vary.setCheckState(
            QtCore.Qt.CheckState.Checked if config.fit else QtCore.Qt.CheckState.Unchecked
        )
        vary.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        self._param_table.setItem(row, 0, vary)
        name_item = QtWidgets.QTableWidgetItem(parameter.display_name)
        symbol = scientific_symbol_html(parameter.name, family=parameter.family, role=parameter.role)
        context = parameter.display_name.removeprefix(parameter.name)
        name_item.setData(SCIENTIFIC_NAME_HTML_ROLE,
                          symbol + (" " + html.escape(compact_text(context, max_chars=36).display) if context else ""))
        name_item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole,
                          scientific_symbol_spoken(parameter.name, family=parameter.family, role=parameter.role))
        name_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        name_item.setToolTip(f"{parameter.display_name}\nParameter ID: {parameter.name}")
        self._param_table.setItem(row, 1, name_item)
        origin_parameter = next(item for item in self._origin.parameter_defaults if item.name == parameter.name)
        mechanism_value = origin_parameter.shared_config.value
        mechanism = QtWidgets.QTableWidgetItem(f"{mechanism_value:.6g}")
        mechanism.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        mechanism.setData(QtCore.Qt.AccessibleTextRole, _format_number(mechanism_value))
        mechanism.setToolTip(_format_number(mechanism_value))
        mechanism.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._param_table.setItem(row, 2, mechanism)
        last = QtWidgets.QTableWidgetItem(self._last_fit_summary(parameter.name))
        last.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        last.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._param_table.setItem(row, 3, last)
        unit_item = QtWidgets.QTableWidgetItem(parameter.unit)
        unit_item.setData(SCIENTIFIC_NAME_HTML_ROLE, format_unit_for_display(parameter.unit))
        unit_item.setData(QtCore.Qt.AccessibleTextRole, parameter.unit)
        unit_item.setToolTip(parameter.unit)
        unit_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        self._param_table.setItem(row, 4, unit_item)

    def _last_fit_summary(self, name: str, *, full_precision: bool = False) -> str:
        last = self._last_fit_parameters.get(name)
        if last is None:
            return "—"
        if last.scope == "shared":
            value = last.selected_config.value
            text = _format_number(value) if full_precision else f"{value:.6g}"
            return f"{text} (Shared)"
        if name in self._last_fit_overall_values and last.scope == "centred_fixed_effects":
            value = self._last_fit_overall_values[name]
            text = _format_number(value) if full_precision else f"{value:.6g}"
            return f"{text} (Overall)"
        return (
            CENTRED_FIXED_EFFECTS_SCOPE_LABEL
            if last.scope == "centred_fixed_effects"
            else "Per dataset"
        )

    def _project_last_fit_columns(self) -> None:
        blocker = QtCore.QSignalBlocker(self._param_table)
        try:
            for row in range(self._param_table.rowCount()):
                item = self._param_table.item(row, 1)
                last = self._param_table.item(row, 3)
                if item is not None and last is not None:
                    name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                    last.setText(self._last_fit_summary(name))
                    complete = self._last_fit_summary(name, full_precision=True)
                    last.setData(QtCore.Qt.AccessibleTextRole, complete)
                    last.setToolTip(complete)
        finally:
            del blocker
        self._render_selected_parameter()

    def _on_roster_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 0:
            return
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if name:
            self._emit_vary_edit(
                name,
                item.checkState() == QtCore.Qt.CheckState.Checked,
            )

    def _on_parameter_selection_changed(self) -> None:
        if not self._projecting:
            self._render_selected_parameter()

    def _selected_parameter(self) -> FitParameterVariable | None:
        row = self._param_table.currentRow()
        item = self._param_table.item(row, 1) if row >= 0 else None
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        return self._parameter_by_name(name) if name else None

    def _clear_selected_editor(self) -> None:
        self._selected_editor.setEnabled(False)
        self._dataset_values_group.setVisible(False)
        self._last_result_label.setText("—")
        self._last_result_caption.setText("Last result:")
        self._last_result_label.setAccessibleName("No last result")
        self._last_result_label.setToolTip("")
        self._refresh_parameter_error()

    def _render_selected_parameter(self) -> None:  # noqa: C901
        parameter = self._selected_parameter()
        if parameter is None:
            self._clear_selected_editor()
            return
        self._selected_editor.setEnabled(True)
        display_unit = format_unit_for_display(parameter.unit)
        unit_caption = f" ({display_unit})" if parameter.unit != "1" else ""
        self._starting_caption.setText("Starting value" + unit_caption + ":")
        self._minimum_caption.setText("Finite minimum" + unit_caption + ":")
        self._maximum_caption.setText("Finite maximum" + unit_caption + ":")
        for caption, label in (
            (self._starting_caption, "Starting value"),
            (self._minimum_caption, "Finite minimum"),
            (self._maximum_caption, "Finite maximum"),
        ):
            caption.setAccessibleName(f"{label} {parameter.unit}")
        config = parameter.selected_config
        validator_minimum = (
            -sys.float_info.max
            if parameter.scientific_category == "finite_scalar"
            else 0.0
        )
        for edit in (self._starting_edit, self._minimum_edit, self._maximum_edit):
            setup_scientific_validator(edit, minimum=validator_minimum)
        draft = self._invalid_numeric_drafts.get(("parameter", parameter.name))
        raw_fields = dict(draft.raw_fields) if draft is not None else {}
        blockers = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._starting_edit,
                self._minimum_edit,
                self._maximum_edit,
                self._scale_combo,
                self._scope_combo,
                self._dataset_values_table,
            )
        ]
        try:
            self._project_scope_choices(parameter)
            self._minimum_edit.setText(_format_number(config.minimum))
            self._maximum_edit.setText(_format_number(config.maximum))
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(config.log10))
            self._scope_combo.setCurrentIndex(self._scope_combo.findData(parameter.scope))
            if parameter.scope == "shared":
                self._starting_edit.setReadOnly(False)
                configure_compact_numeric_edit(self._starting_edit, characters=20)
                self._starting_edit.setText(_format_number(config.value))
            else:
                self._starting_edit.setReadOnly(True)
                configure_compact_numeric_edit(self._starting_edit, characters=None)
                self._starting_edit.setText("Values by dataset")
            controls = {
                "starting": self._starting_edit,
                "minimum": self._minimum_edit,
                "maximum": self._maximum_edit,
            }
            for field, edit in controls.items():
                edit.setStyleSheet("")
                edit.setToolTip("")
                if field not in raw_fields:
                    continue
                if field == "starting" and parameter.scope != "shared":
                    continue
                edit.setText(raw_fields[field])
                edit.setStyleSheet(
                    "background: rgb(255, 200, 200); color: rgb(80, 0, 0);"
                )
                edit.setToolTip(draft.message)
            self._render_dataset_values(parameter)
            last = self._last_fit_parameters.get(parameter.name)
            overall = self._last_fit_overall_values.get(parameter.name)
            is_overall = last is not None and last.scope == "centred_fixed_effects" and overall is not None
            self._last_result_caption.setText("Overall fitted value:" if is_overall else "Last result:")
            self._last_result_label.setToolTip("")
            if last is None:
                self._last_result_label.setText("—")
            elif is_overall:
                self._last_result_label.setText(f"{overall:.6g}")
                self._last_result_label.setToolTip(_format_number(overall))
            elif last.scope == "shared":
                self._last_result_label.setText(f"{last.selected_config.value:.6g} (Shared)")
                self._last_result_label.setToolTip(_format_number(last.selected_config.value))
            else:
                self._last_result_label.setText(
                    CENTRED_FIXED_EFFECTS_SCOPE_LABEL
                    if last.scope == "centred_fixed_effects"
                    else "Per dataset"
                )
            self._last_result_label.setAccessibleName(
                self._last_result_caption.text() + " "
                + (self._last_result_label.toolTip() or self._last_result_label.text())
            )
        finally:
            del blockers
        self._refresh_parameter_error()

    def _render_dataset_values(self, parameter: FitParameterVariable) -> None:
        visible = parameter.scope in {"per_dataset", "centred_fixed_effects"}
        self._dataset_values_group.setVisible(visible)
        if not visible:
            self._dataset_values_table.setRowCount(0)
            self._size_dataset_values_table(self._dataset_values_table)
            return
        snapshot = self._projected_setup
        dataset_ids = (
            self._parameter_dataset_ids(snapshot, (parameter,))
            if snapshot is not None
            else tuple(dataset_id for dataset_id, _value in parameter.per_dataset_config.dataset_values)
        )
        values = dict(parameter.per_dataset_config.dataset_values)
        last = self._last_fit_parameters.get(parameter.name)
        last_values = dict(last.selected_config.dataset_values) if last is not None and last.scope != "shared" else {}
        blocker = QtCore.QSignalBlocker(self._dataset_values_table)
        self._dataset_values_table.setRowCount(len(dataset_ids))
        for row, dataset_id in enumerate(dataset_ids):
            dataset_item = QtWidgets.QTableWidgetItem(self._dataset_display_name(dataset_id))
            dataset_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            dataset_item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset_id)
            dataset_item.setToolTip(f"{self._dataset_display_name(dataset_id)}\nDataset ID: {dataset_id}")
            self._dataset_values_table.setItem(row, 0, dataset_item)
            value_item = QtWidgets.QTableWidgetItem(
                _format_number(values[dataset_id]) if dataset_id in values else ""
            )
            value_item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset_id)
            value_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            key = ("parameter_dataset", parameter.name, dataset_id, "initial")
            draft = self._invalid_numeric_drafts.get(key)
            if draft is not None:
                value_item.setText(dict(draft.raw_fields)["initial"])
                value_item.setBackground(QtGui.QColor(255, 200, 200))
                value_item.setForeground(QtGui.QColor(80, 0, 0))
                value_item.setToolTip(draft.message)
            self._dataset_values_table.setItem(row, 1, value_item)
            last_item = QtWidgets.QTableWidgetItem(
                _format_number(last_values[dataset_id]) if dataset_id in last_values else "—"
            )
            last_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            last_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            last_item.setToolTip(f"Dataset ID: {dataset_id}")
            self._dataset_values_table.setItem(row, 2, last_item)
            unit_item = QtWidgets.QTableWidgetItem(parameter.unit)
            unit_item.setData(SCIENTIFIC_NAME_HTML_ROLE, format_unit_for_display(parameter.unit))
            unit_item.setData(QtCore.Qt.AccessibleTextRole, parameter.unit)
            unit_item.setToolTip(parameter.unit)
            unit_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self._dataset_values_table.setItem(row, 3, unit_item)
        del blocker
        self._size_dataset_values_table(self._dataset_values_table)

    def _project_scope_choices(self, parameter: FitParameterVariable) -> None:
        """Offer only scopes supported by the accepted parameter authority."""

        allowed = tuple(self._SCOPE_LABELS)
        current = tuple(
            str(self._scope_combo.itemData(index) or "")
            for index in range(self._scope_combo.count())
        )
        if current == allowed:
            return
        self._scope_combo.clear()
        for scope in allowed:
            self._scope_combo.addItem(self._SCOPE_LABELS[scope], scope)

    def _on_dataset_value_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting or item.column() != 1:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        dataset_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        key = ("parameter_dataset", parameter.name, dataset_id, "initial")
        try:
            parsed = self._parse_number(
                item.text(), f"{parameter.name} value for dataset {dataset_id}"
            )
            admit_gui_parameter_value(
                parsed,
                scientific_category=parameter.scientific_category,
                role=parameter.role,
                unit=parameter.unit,
                allow_exact_zero=parameter.allow_exact_zero,
                field=f"{parameter.name} value for dataset {dataset_id}",
            )
            if key in self._invalid_numeric_drafts:
                self._store_invalid_numeric_draft(
                    key,
                    {"initial": item.text()},
                    "Apply the corrected parameter values or revert this draft.",
                )
                item.setBackground(QtGui.QColor(255, 200, 200))
                item.setForeground(QtGui.QColor(80, 0, 0))
                item.setToolTip(
                    "Apply the corrected parameter values or revert this draft."
                )
            self._emit_selected_parameter_edit()
        except (TypeError, ValueError, OverflowError) as exc:
            self._store_invalid_numeric_draft(
                key,
                {"initial": item.text()},
                str(exc),
            )
            item.setBackground(QtGui.QColor(255, 200, 200))
            item.setForeground(QtGui.QColor(80, 0, 0))
            item.setToolTip(str(exc))
            self.statusMessage.emit(str(exc))

    def _on_parameter_numeric_text_changed(self, _raw_text: str) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        try:
            self._parameter_projection_for_change(parameter)
        except (TypeError, ValueError, OverflowError) as exc:
            self._store_parameter_editor_draft(parameter, str(exc))
            return
        if ("parameter", parameter.name) in self._invalid_numeric_drafts:
            self._store_parameter_editor_draft(
                parameter,
                "Apply the corrected parameter values or revert this draft.",
            )

    def _store_parameter_editor_draft(
        self,
        parameter: FitParameterVariable,
        message: str,
    ) -> None:
        fields = {
            "minimum": self._minimum_edit,
            "maximum": self._maximum_edit,
        }
        if parameter.scope == "shared":
            fields = {"starting": self._starting_edit, **fields}
        self._store_invalid_numeric_draft(
            ("parameter", parameter.name),
            {field: edit.text() for field, edit in fields.items()},
            message,
        )
        for edit in fields.values():
            edit.setStyleSheet(
                "background: rgb(255, 200, 200); color: rgb(80, 0, 0);"
            )
            edit.setToolTip(str(message))

    def _refresh_parameter_error(self) -> None:
        parameter = self._selected_parameter()
        messages = tuple(dict.fromkeys(
            draft.message for key, draft in self._invalid_numeric_drafts.items()
            if parameter is not None and key[1] == parameter.name
        ))
        self._parameter_error.setText(
            "\n".join(messages)
            + "\nPress Esc to discard these invalid edits and restore the last accepted values."
            if messages else ""
        )
        self._parameter_error.setVisible(bool(messages))
        self._discard_parameter_shortcut.setEnabled(bool(messages))

    def _revert_selected_parameter_draft(self) -> None:
        parameter = self._selected_parameter()
        if parameter is None:
            return
        if not self._retire_numeric_drafts({parameter.name}):
            return
        self._render_selected_parameter()
        self.statusMessage.emit(
            f"Invalid values for {parameter.display_name} reverted."
        )

    def _parameter_projection_for_change(self, parameter: FitParameterVariable) -> FitParameterVariable:
        if self._projected_setup is None:
            raise ValueError("Fit setup is unavailable.")
        scope = str(self._scope_combo.currentData() or parameter.scope)
        log10 = bool(self._scale_combo.currentData())
        minimum = self._parse_number(self._minimum_edit.text(), f"{parameter.name} minimum")
        maximum = self._parse_number(self._maximum_edit.text(), f"{parameter.name} maximum")
        config = parameter.shared_config if scope == "shared" else parameter.per_dataset_config
        if scope == "shared":
            value = self._parse_number(self._starting_edit.text(), f"{parameter.name} starting value")
            dataset_values = config.dataset_values
        else:
            value = config.value
            dataset_values_list: list[tuple[str, float]] = []
            for row in range(self._dataset_values_table.rowCount()):
                dataset_item = self._dataset_values_table.item(row, 0)
                value_item = self._dataset_values_table.item(row, 1)
                dataset_id = str(dataset_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if dataset_item else ""
                raw = value_item.text() if value_item is not None else ""
                if not dataset_id or not raw.strip():
                    raise ValueError(f"{parameter.name} requires a starting value for every dataset.")
                dataset_value = self._parse_number(raw, f"{parameter.name} value for dataset {dataset_id}")
                dataset_values_list.append((dataset_id, dataset_value))
            dataset_values = tuple(dataset_values_list)
        active = FitParameterConfiguration(
            value=value,
            minimum=minimum,
            maximum=maximum,
            fit=parameter.selected_config.fit,
            log10=log10,
            dataset_values=dataset_values,
        )
        return FitParameterVariable(
            name=parameter.name,
            display_name=parameter.display_name,
            scope=scope,
            shared_config=active if scope == "shared" else parameter.shared_config,
            per_dataset_config=active if scope != "shared" else parameter.per_dataset_config,
            family=parameter.family,
            role=parameter.role,
            scientific_category=parameter.scientific_category,
            allow_exact_zero=parameter.allow_exact_zero,
            unit=parameter.unit,
        )

    def _uncertainty_effects_for_parameter(
        self,
        parameter: FitParameterVariable,
    ) -> dict[str, tuple[str, ...]]:
        """Return accepted typed uncertainty coordinates affected by Vary off."""

        setup = self._projected_setup
        if setup is None:
            return {}
        available = set(fit_uncertainty_coordinates(setup.parameters, setup.datasets))
        effects: dict[str, tuple[str, ...]] = {}
        for label, configuration in (
            ("Profile", setup.profile_likelihood),
            ("Bootstrap", setup.residual_bootstrap),
        ):
            if configuration is None:
                continue
            coordinates = tuple(
                coordinate
                for coordinate in configuration.parameter_names
                if isinstance(coordinate, FitCoordinate)
                and coordinate in available
                and coordinate.name == parameter.name
            )
            if coordinates:
                effects[label] = tuple(
                    self._coordinate_display_name(coordinate)
                    for coordinate in coordinates
                )
        return effects

    def _confirm_vary_analysis_update(
        self,
        parameter: FitParameterVariable,
    ) -> bool:
        effects = self._uncertainty_effects_for_parameter(parameter)
        if not effects:
            return True
        details = [
            f"Stopping Vary for {parameter.display_name} will update these analyses:"
        ]
        details.extend(
            f"{label}: {', '.join(coordinates)}"
            for label, coordinates in effects.items()
        )
        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Update uncertainty analyses?")
        message.setText("\n".join(details))
        stop_button = message.addButton(
            "Stop varying and update analyses",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = message.addButton(
            "Cancel",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is stop_button

    def _reproject_vary_truth(self, parameter_name: str) -> None:
        """Restore the visible Vary controls from the accepted setup only."""

        setup = self._projected_setup
        parameter = self._parameter_by_name(parameter_name)
        if setup is None or parameter is None:
            return
        blocker = QtCore.QSignalBlocker(self._param_table)
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, 0)
            if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == parameter_name:
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if parameter.selected_config.fit
                    else QtCore.Qt.CheckState.Unchecked
                )
                break
        del blocker

    def _emit_vary_edit(self, name: str, desired_fit: bool) -> bool:
        if self._projecting or self._running or self._projected_setup is None:
            return False
        parameter = self._parameter_by_name(str(name or ""))
        if parameter is None:
            return False
        desired = bool(desired_fit)
        if desired == parameter.selected_config.fit:
            self._reproject_vary_truth(parameter.name)
            return False
        if not desired and not self._confirm_vary_analysis_update(parameter):
            self._reproject_vary_truth(parameter.name)
            return False
        setup = self._projected_setup
        self._reproject_vary_truth(parameter.name)
        replacement = _replace_fit_parameter(parameter, fit=desired)
        profile = setup.profile_likelihood
        bootstrap = setup.residual_bootstrap
        if not desired:
            profile = prune_fit_analysis_coordinates(profile, (parameter.name,))
            bootstrap = prune_fit_analysis_coordinates(bootstrap, (parameter.name,))
        self.setupSnapshotRequested.emit(
            replace(
                setup,
                parameters=tuple(
                    replacement if item.name == replacement.name else item
                    for item in setup.parameters
                ),
                profile_likelihood=profile,
                residual_bootstrap=bootstrap,
            )
        )
        return True

    def _emit_selected_parameter_edit(self, *_args: object, name: str | None = None) -> bool:
        if self._projecting or self._running or self._projected_setup is None:
            return False
        parameter = self._parameter_by_name(name or (self._selected_parameter().name if self._selected_parameter() else ""))
        if parameter is None:
            return False
        try:
            edited = self._parameter_projection_for_change(parameter)
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit parameter edit is invalid: {exc}")
            self._store_parameter_editor_draft(parameter, str(exc))
            return False
        updated = replace(
            self._projected_setup,
            parameters=tuple(
                edited if item.name == edited.name else item
                for item in self._projected_setup.parameters
            ),
        )
        self._retire_numeric_drafts({parameter.name})
        self.setupSnapshotRequested.emit(updated)
        return True

    def _on_scope_changed(self, _index: int) -> None:
        if self._projecting or self._projected_setup is None:
            return
        parameter = self._selected_parameter()
        if parameter is None:
            return
        desired = str(self._scope_combo.currentData() or "")
        if desired == parameter.scope:
            return
        if desired not in {"shared", "per_dataset", "centred_fixed_effects"}:
            self._restore_scope(parameter.scope)
            return
        replacement = convert_fit_parameter_scope(
            parameter, desired, self._projected_setup.datasets
        )
        profile = prune_fit_analysis_coordinates(
            self._projected_setup.profile_likelihood,
            (parameter.name,),
        )
        bootstrap = prune_fit_analysis_coordinates(
            self._projected_setup.residual_bootstrap,
            (parameter.name,),
        )
        # The combo is an immediate projection of the accepted snapshot.  Put
        # it back before publishing the candidate snapshot so a rejected
        # admission cannot leave the control ahead of accepted state.  The
        # owner's synchronous projection restores the accepted new scope.
        self._restore_scope(parameter.scope)
        self.setupSnapshotRequested.emit(
            replace(
                self._projected_setup,
                parameters=tuple(
                    replacement if item.name == replacement.name else item
                    for item in self._projected_setup.parameters
                ),
                profile_likelihood=profile,
                residual_bootstrap=bootstrap,
            )
        )

    def _restore_scope(self, scope: str) -> None:
        blocker = QtCore.QSignalBlocker(self._scope_combo)
        self._scope_combo.setCurrentIndex(self._scope_combo.findData(scope))
        del blocker

    @property
    def has_invalid_edits(self) -> bool:
        return bool(self._invalid_numeric_drafts or self._defaults_error_text)

    def _store_invalid_numeric_draft(
        self,
        key: tuple[str, ...],
        raw_fields: Mapping[str, str],
        message: str,
    ) -> None:
        draft = _InvalidNumericDraft(
            tuple((str(field), str(raw)) for field, raw in raw_fields.items()),
            str(message),
        )
        if self._invalid_numeric_drafts.get(key) != draft:
            self._invalid_numeric_drafts[key] = draft
            self._refresh_parameter_error()
            self.validityChanged.emit()

    def _retire_numeric_drafts(
        self,
        parameter_names: set[str] | None = None,
    ) -> bool:
        keys = tuple(
            key
            for key in self._invalid_numeric_drafts
            if parameter_names is None
            or (len(key) > 1 and key[1] in parameter_names)
        )
        for key in keys:
            self._invalid_numeric_drafts.pop(key)
        if keys:
            self._refresh_parameter_error()
            self.validityChanged.emit()
        return bool(keys)

    @staticmethod
    def _parse_number(raw_text: str, field: str) -> float:
        return parse_finite_numeric_text(raw_text, field=field)

    def _project_defaults(self, defaults: FitDefaults) -> None:
        blocker = [
            QtCore.QSignalBlocker(control)
            for control in (
                self._method_combo,
                self._max_eval_spin,
                self._seed_check,
                self._seed_spin,
                self._ftol_edit,
                self._xtol_edit,
            )
        ]
        self._method_combo.setCurrentIndex(self._method_combo.findData(defaults.method))
        self._max_eval_spin.setValue(defaults.max_nfev)
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        self._ftol_edit.setText(self._tolerance_drafts.get("ftol", _format_scientific(defaults.ftol)))
        self._xtol_edit.setText(self._tolerance_drafts.get("xtol", _format_scientific(defaults.xtol)))
        is_de = defaults.method == DIFFERENTIAL_EVOLUTION_METHOD
        self._ftol_label.setText("Population convergence tolerance:" if is_de else "Objective-improvement tolerance:")
        self._xtol_label.setVisible(not is_de)
        self._xtol_edit.setVisible(not is_de)
        self._xtol_group.setVisible(not is_de)
        del blocker

    def _defaults_projection(self) -> FitDefaults:
        if self._projected_setup is None:
            raise ValueError("Fit defaults are not available.")
        defaults = self._projected_setup.defaults
        ftol = self._parse_number(self._ftol_edit.text(), "Convergence tolerance")
        method = str(self._method_combo.currentData() or "trf")
        xtol = (
            defaults.xtol if method == DIFFERENTIAL_EVOLUTION_METHOD
            else self._parse_number(self._xtol_edit.text(), "Parameter-step tolerance")
        )
        if ftol <= 0.0 or xtol <= 0.0:
            raise ValueError("Convergence tolerances must be positive.")
        return validate_fit_defaults(
            replace(
                defaults,
                method=method,
                max_nfev=self._max_eval_spin.value(),
                ftol=ftol,
                xtol=xtol,
                use_seed=self._seed_check.isChecked(),
                seed=self._seed_spin.value(),
            )
        )

    def _emit_defaults_edit(self, *_args: object) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        try:
            snapshot = replace(
                self._projected_setup,
                defaults=self._defaults_projection(),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_defaults_error(str(exc))
            self._project_defaults(self._projected_setup.defaults)
            self.statusMessage.emit(f"Fit defaults are invalid: {exc}")
            return
        self._tolerance_drafts.clear()
        self._set_defaults_error("")
        self.setupSnapshotRequested.emit(snapshot)
        self._project_defaults(self._projected_setup.defaults)

    def _on_tolerance_text_changed(self, _text: str) -> None:
        if self._projecting or self._projected_setup is None:
            return
        self._tolerance_drafts = {
            "ftol": self._ftol_edit.text(), "xtol": self._xtol_edit.text(),
        }
        try:
            self._defaults_projection()
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_defaults_error(str(exc))
        else:
            self._set_defaults_error("")

    def _set_defaults_error(self, message: str) -> None:
        changed = bool(message) != bool(self._defaults_error_text)
        self._defaults_error_text = message
        self._defaults_error.setText(
            message + "\nPress Esc to restore the accepted tolerances." if message else ""
        )
        self._defaults_error.setVisible(bool(message))
        self._discard_tolerances_shortcut.setEnabled(bool(self._tolerance_drafts))
        if changed:
            self.validityChanged.emit()

    def _discard_tolerance_drafts(self) -> None:
        self._tolerance_drafts.clear()
        self._set_defaults_error("")
        if self._projected_setup is not None:
            self._project_defaults(self._projected_setup.defaults)

    def _emit_solver_edit(self, *_args: object) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        try:
            snapshot = replace(
                self._projected_setup,
                solver=str(self._solver_combo.currentData() or "BDF"),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Fit solver edit is invalid: {exc}")
            return
        self.setupSnapshotRequested.emit(snapshot)

    def _project_advanced_analysis(self, snapshot: FitSetupSnapshot) -> None:
        profile = snapshot.profile_likelihood
        bootstrap = snapshot.residual_bootstrap
        blockers = [
            QtCore.QSignalBlocker(control)
            for control in (self._profile_enabled, self._bootstrap_enabled)
        ]
        try:
            self._profile_enabled.setChecked(profile is not None)
            self._bootstrap_enabled.setChecked(bootstrap is not None)
            self._profile_summary.setText(self._analysis_summary(profile, "Profile"))
            self._bootstrap_summary.setText(
                self._analysis_summary(bootstrap, "Bootstrap")
            )
        finally:
            del blockers

    @staticmethod
    def _analysis_summary(configuration: object | None, label: str) -> str:
        if configuration is None:
            return "Not configured"
        names = tuple(getattr(configuration, "parameter_names", ()))
        method = str(getattr(configuration, "optimizer_method", "trf"))
        if label == "Profile":
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''} · "
                f"{getattr(configuration, 'confidence_level', 0.95):.0%} · "
                f"{getattr(configuration, 'max_bracketing_steps_per_side', 0)} "
                "maximum outward search steps/side"
            )
        else:
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''} · "
                f"{getattr(configuration, 'replicates', 0)} refits · "
                f"{getattr(configuration, 'minimum_success_fraction', 0.0):.0%} minimum"
            )
        return f"Enabled · {detail} · {FIT_METHOD_LABELS.get(method, method)}"

    def _analysis_snapshot(
        self,
        *,
        profile: ProfileLikelihoodConfig | None,
        bootstrap: ResidualBootstrapConfig | None,
    ) -> FitSetupSnapshot:
        setup = self._projected_setup
        if setup is None:
            raise ValueError("Fit setup is unavailable.")
        return replace(
            setup,
            profile_likelihood=profile,
            residual_bootstrap=bootstrap,
        )

    def _available_analysis_coordinates(self) -> tuple[FitCoordinate, ...]:
        setup = self._projected_setup
        if setup is None:
            return ()
        return fit_uncertainty_coordinates(setup.parameters, setup.datasets)

    def _analysis_coordinate_display_names(self) -> dict[FitCoordinate, str]:
        return {
            raw: self._coordinate_display_name(raw)
            for raw in self._available_analysis_coordinates()
        }

    def _open_profile_dialog(self) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        dialog = _ProfileLikelihoodDialog(
            available=self._available_analysis_coordinates(),
            display_names=self._analysis_coordinate_display_names(),
            configuration=setup.profile_likelihood,
            default_selection=None,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            if setup.profile_likelihood is None:
                self._profile_enabled.setChecked(False)
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=dialog.configuration(),
                    bootstrap=setup.residual_bootstrap,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Profile likelihood settings are invalid: {exc}")

    def _open_bootstrap_dialog(self) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        dialog = _ResidualBootstrapDialog(
            available=self._available_bootstrap_coordinates(),
            display_names=self._analysis_coordinate_display_names(),
            configuration=setup.residual_bootstrap,
            default_selection=None,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            if setup.residual_bootstrap is None:
                self._bootstrap_enabled.setChecked(False)
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=setup.profile_likelihood,
                    bootstrap=dialog.configuration(),
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Residual bootstrap settings are invalid: {exc}")

    def _on_profile_toggled(self, enabled: bool) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        # The checkbox is a projection of the accepted owner snapshot.  Keep
        # it there while a dialog is pending; the accepted snapshot reprojects
        # the new value atomically.
        blocker = QtCore.QSignalBlocker(self._profile_enabled)
        self._profile_enabled.setChecked(setup.profile_likelihood is not None)
        del blocker
        if enabled:
            self._open_profile_dialog()
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=None,
                    bootstrap=setup.residual_bootstrap,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Profile likelihood settings are invalid: {exc}")

    def _on_bootstrap_toggled(self, enabled: bool) -> None:
        if self._projecting or self._running:
            return
        setup = self._projected_setup
        if setup is None:
            return
        blocker = QtCore.QSignalBlocker(self._bootstrap_enabled)
        self._bootstrap_enabled.setChecked(setup.residual_bootstrap is not None)
        del blocker
        if enabled:
            self._open_bootstrap_dialog()
            return
        try:
            self.setupSnapshotRequested.emit(
                self._analysis_snapshot(
                    profile=setup.profile_likelihood,
                    bootstrap=None,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.statusMessage.emit(f"Residual bootstrap settings are invalid: {exc}")

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched in (getattr(self, "_dataset_values_table", None), getattr(self, "_param_table", None)) and event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.Copy):
                indexes = watched.selectedIndexes()
                if indexes:
                    selected = {(index.row(), index.column()) for index in indexes}
                    rows = range(min(index.row() for index in indexes), max(index.row() for index in indexes) + 1)
                    columns = range(min(index.column() for index in indexes), max(index.column() for index in indexes) + 1)
                    payload = "\n".join(
                        "\t".join(
                            (
                                watched.item(row, column).data(QtCore.Qt.AccessibleTextRole)
                                if watched is self._param_table and column in (2, 3)
                                else watched.item(row, column).text()
                            ) if (row, column) in selected else ""
                            for column in columns
                        ) for row in rows
                    )
                    mime = QtCore.QMimeData()
                    mime.setText(payload)
                    QtWidgets.QApplication.clipboard().setMimeData(mime)
                return True
        if watched in (getattr(self, "_settings_group", None), getattr(self, "_roster_actions_host", None)):
            if event.type() in {
                QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest, QtCore.QEvent.Show,
                QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange,
            } and watched.layout() is not None:
                required = watched.layout().totalHeightForWidth(watched.width())
                watched.setFixedHeight(max(watched.minimumSizeHint().height(), required))
        if watched is getattr(self, "_analysis_cards_host", None) and event.type() in {
            QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.LayoutRequest,
            QtCore.QEvent.Type.FontChange, QtCore.QEvent.Type.StyleChange,
        }:
            self._relayout_analysis_cards()
        if watched is getattr(self, "_param_table", None) and event.type() in {
            QtCore.QEvent.Show, QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange,
        }:
            watched.resizeRowsToContents()
        if watched is getattr(self, "_dataset_values_table", None) and event.type() in {
            QtCore.QEvent.Type.Show, QtCore.QEvent.Type.FontChange, QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.LayoutRequest,
        }:
            self._size_dataset_values_table(watched)
        return super().eventFilter(watched, event)

    @staticmethod
    def _size_dataset_values_table(table: QtWidgets.QTableWidget) -> None:
        font_key = table.font().toString()
        if getattr(table, "_numeric_column_font", None) != font_key:
            table._numeric_column_font = font_key
            metrics = table.fontMetrics()
            margin = table.style().pixelMetric(QtWidgets.QStyle.PM_HeaderMargin)
            table.horizontalHeader().setMinimumSectionSize(metrics.horizontalAdvance("M") * 8)
            for column in (1, 2):
                table.setColumnWidth(column, max(
                    metrics.horizontalAdvance("-1.23456e+308"),
                    metrics.horizontalAdvance(table.horizontalHeaderItem(column).text()),
                ) + 2 * margin + 8)
        table.resizeRowsToContents()
        height = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()
        height += sum(table.rowHeight(row) for row in range(min(8, table.rowCount())))
        scrollbar = table.horizontalScrollBar()
        if scrollbar.maximum() > scrollbar.minimum():
            height += scrollbar.sizeHint().height()
        table.setFixedHeight(height)

    def _relayout_analysis_cards(self) -> None:
        if not hasattr(self, "_analysis_cards"):
            return
        available = self._analysis_cards_host.contentsRect().width()
        required = sum(card.minimumSizeHint().width() for card in self._analysis_cards)
        required += self._analysis_cards_layout.horizontalSpacing()
        columns = 2 if required <= available else 1
        if columns == self._analysis_card_columns:
            return
        self._analysis_card_columns = columns
        for index, card in enumerate(self._analysis_cards):
            self._analysis_cards_layout.removeWidget(card)
            row, column = divmod(index, columns)
            self._analysis_cards_layout.addWidget(card, row, column)
        for column in range(2):
            self._analysis_cards_layout.setColumnStretch(column, 1 if column < columns else 0)

    def set_running_state(self, running: bool) -> None:
        self._running = bool(running)

    def clear_pending_setup_edits(self) -> None:
        self._retire_numeric_drafts()
        self._discard_tolerance_drafts()
        if self._projected_setup is not None:
            self.project_setup_snapshot(self._projected_setup)
