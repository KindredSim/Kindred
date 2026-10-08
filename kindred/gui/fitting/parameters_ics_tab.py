"""Fit parameter presentation with current owner-facing hooks.

The widget construction in this module is intentionally close to the executable
Fit presentation.  The widget is only a view: accepted setup, request
construction, Fit results, defaults persistence, and worker ownership remain
outside it.
"""

from __future__ import annotations

import html


from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.display_name_policy import compact_text, scientific_symbol_html, scientific_symbol_spoken
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE,
    SOLVER_TOOLTIP,
    ScientificNameDelegate,
    ControlRowsLayout,
    cell_text_as_shown,
    configure_compact_numeric_edit,
    ignore_wheel,
)
from kindred.core.analysis.fit_uncertainty import (
    BOOTSTRAP_SEED_RANGE,
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
)
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.quantity_units import format_unit_for_display
from kindred.core.validation import parse_finite_numeric_text, six_digit_text

from kindred.core.fitting_method_policy import SUPPORTED_FIT_METHODS, UNCERTAINTY_FIT_METHODS
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
    FitParameterVariable,
    FitSetupSnapshot,
    fit_coordinate_datasets_text,
    fit_group_labels,
    fit_uncertainty_coordinates,
    prune_fit_analysis_coordinates,
    skipped_fit_analysis_coordinates,
    _replace_fit_parameter,
    convert_fit_parameter_scope,
)
from kindred.gui.fitting.constants import (
    DE_CONVERGENCE_TOLERANCE_LABEL,
    admit_fit_tolerance,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FUNCTION_TOLERANCE_LABEL,
    STEP_TOLERANCE_LABEL,
)
from kindred.gui.fitting.reporting import fit_identity_text

__all__ = ["ParametersIcsTab"]


@dataclass(frozen=True, slots=True)
class _InvalidRow:
    raw_fields: tuple[tuple[str, str], ...]
    messages: tuple[tuple[str, str], ...]


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


class _ProfileLikelihoodDialog(QtWidgets.QDialog):
    """Transactional Profile likelihood configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str],
        configuration: ProfileLikelihoodConfig | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profile Likelihood")
        self.setModal(True)
        self.resize(520, 420)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        selected = (
            set(configuration.parameter_names) if configuration else set(available)
        )
        for coordinate in available:
            label = display_names[coordinate]
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
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
        return ProfileLikelihoodConfig(
            parameter_names=names,
            confidence_level=parse_finite_numeric_text(
                self._confidence.text(), field="Confidence level"
            ),
            max_bracketing_steps_per_side=self._points.value(),
            max_nfev_per_point=self._budget.value(),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()


class _ResidualBootstrapDialog(QtWidgets.QDialog):
    """Transactional residual Bootstrap configuration."""

    def __init__(
        self,
        *,
        available: Sequence[FitCoordinate],
        display_names: Mapping[FitCoordinate, str],
        configuration: ResidualBootstrapConfig | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Residual Bootstrap")
        self.setModal(True)
        self.resize(520, 440)
        form = QtWidgets.QFormLayout(self)
        self._coordinates = QtWidgets.QListWidget(self)
        self._coordinates.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        selected = (
            set(configuration.parameter_names) if configuration else set(available)
        )
        for coordinate in available:
            label = display_names[coordinate]
            item = QtWidgets.QListWidgetItem(label, self._coordinates)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, coordinate)
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
        configure_compact_numeric_edit(self._confidence, characters=8)
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
        return ResidualBootstrapConfig(
            parameter_names=names,
            seed=self._seed.value(),
            replicates=self._replicates.value(),
            confidence_level=parse_finite_numeric_text(
                self._confidence.text(), field="Confidence level"
            ),
            minimum_successful_replicates=self._minimum.value(),
            minimum_success_fraction=parse_finite_numeric_text(
                self._fraction.text(), field="Required success rate"
            ),
            optimizer_method=str(self._method.currentData() or "trf"),
        )

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.configuration()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()


class _ParameterBoundsDialog(QtWidgets.QDialog):
    """Collect one Minimum/Maximum/scale choice for several Fit parameters."""

    def __init__(
        self,
        parameters: Sequence[FitParameterVariable],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Bounds for Parameters")
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self._minimum = QtWidgets.QLineEdit(self)
        self._maximum = QtWidgets.QLineEdit(self)
        for edit in (self._minimum, self._maximum):
            configure_compact_numeric_edit(edit, characters=20)
        self._log10 = QtWidgets.QCheckBox("Log Scale", self)
        self._parameters = QtWidgets.QTableWidget(len(parameters), 2, self)
        self._parameters.setHorizontalHeaderLabels(("Parameter", "Unit"))
        self._parameters.verticalHeader().setVisible(False)
        self._parameters.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._parameters.setItemDelegateForColumn(1, ScientificNameDelegate(self._parameters))
        for row, parameter in enumerate(parameters):
            item = QtWidgets.QTableWidgetItem(parameter.display_name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self._parameters.setItem(row, 0, item)
            unit = QtWidgets.QTableWidgetItem(parameter.unit)
            unit.setData(SCIENTIFIC_NAME_HTML_ROLE, format_unit_for_display(parameter.unit))
            unit.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self._parameters.setItem(row, 1, unit)
        form.addRow("Minimum:", self._minimum)
        form.addRow("Maximum:", self._maximum)
        form.addRow("", self._log10)
        form.addRow("Parameters:", self._parameters)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def selection(self) -> tuple[tuple[str, ...], str, str, bool]:
        names = tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(self._parameters.rowCount())
            if (item := self._parameters.item(row, 0)).checkState() == QtCore.Qt.CheckState.Checked
        )
        if not names:
            raise ValueError("Select at least one parameter.")
        minimum = parse_finite_numeric_text(self._minimum.text(), field="Minimum")
        maximum = parse_finite_numeric_text(self._maximum.text(), field="Maximum")
        if minimum >= maximum:
            raise ValueError("Minimum must be less than Maximum.")
        if self._log10.isChecked() and minimum <= 0.0:
            raise ValueError("Log Scale requires a positive Minimum.")
        return names, self._minimum.text(), self._maximum.text(), self._log10.isChecked()

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self.selection()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
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
        metrics = self.fontMetrics()
        em = metrics.horizontalAdvance("M")
        self.horizontalHeader().setMinimumSectionSize(3 * em)
        header = self.horizontalHeader()
        # Sized once for realistic values; widths the user drags are kept.
        for column, sample in (
            (_C.START, "-1.23457e-04"), (_C.MIN, "-1.23457e-04"), (_C.MAX, "-1.23457e-04"),
            (_C.LAST, "-1.23457e-04"), (_C.MECH, "-1.23457e-04"),
        ):
            self.setColumnWidth(column, max(header.sectionSizeHint(column), metrics.horizontalAdvance(sample) + em))
        self.fit_name_column()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_name_column()

    def fit_name_column(self) -> None:
        """Stretch Parameter while it stays readable; otherwise keep it and scroll."""

        minimum = 18 * self.fontMetrics().horizontalAdvance("M")
        others = sum(self.columnWidth(column) for column in range(self.columnCount()) if column != _C.NAME)
        header = self.horizontalHeader()
        if self.viewport().width() - others >= minimum:
            header.setSectionResizeMode(_C.NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(_C.NAME, QtWidgets.QHeaderView.ResizeMode.Interactive)
            self.setColumnWidth(_C.NAME, minimum)

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        height = self.horizontalHeader().sizeHint().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(row) for row in range(min(8, self.rowCount())))
        scrollbar = self.horizontalScrollBar()
        if self.horizontalHeader().length() > self.viewport().width():
            height += scrollbar.sizeHint().height()
        size.setHeight(height)
        return size


class _C:
    VARY, LOG, NAME, START, MIN, MAX, SCOPE, LAST, MECH, UNIT, COUNT = range(11)
    HEADERS = ("Vary", "Log Scale", "Parameter", "Initial Guess", "Minimum", "Maximum",
               "Across Datasets", "Last Fit Result", "Mechanism Tab Value", "Unit")


_DATASET_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_COMPACT_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
_INVALID_BG = QtGui.QColor(255, 200, 200)
_INVALID_FG = QtGui.QColor(80, 0, 0)


class _CompactNumberDelegate(QtWidgets.QStyledItemDelegate):
    """Paint compact values while native editing retains the complete text."""

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        if not index.data(_COMPACT_ROLE):
            return
        # The edited cell can repaint during the edit's own resize, before its
        # row re-renders, while it still holds the unparsed text.
        try:
            option.text = six_digit_text(float(option.text))
        except ValueError:
            pass


# Row fields in table order, with the column that shows each.
_FIELD_COLUMNS = {"log10": _C.LOG, "starting": _C.START, "minimum": _C.MIN, "maximum": _C.MAX}


def _field_label(field: str) -> str:
    return _C.HEADERS[_FIELD_COLUMNS["starting" if field.startswith("dataset:") else field]]


def _parse_label(field: str, dataset_name: str) -> str:
    """Name a numeric field in its own parse message; a dataset start names its dataset."""
    if field.startswith("dataset:"):
        return f"Initial guess for dataset {dataset_name}"
    return "Initial guess" if field == "starting" else _field_label(field)


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
        self._last_fit_included: set[str] = set()
        self._tolerance_drafts: dict[str, str] = {}
        self._defaults_error_text = ""
        self._roster_height_initialized = False
        self._can_reset_initial = False
        self._can_reset_last_fit = False
        self._invalid_rows: dict[str, _InvalidRow] = {}
        self._projecting = False
        self._running = False
        self._row_keys: tuple[tuple[str, str | None], ...] = ()
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
        options = self._left_splitter.widget(1).widget().sizeHint().height()
        preferred = max(roster.sizeHint().height(), available - options)
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


    _SCOPE_CHOICES = {
        "shared": ("Shared", "One shared value for all datasets."),
        "per_dataset": ("Per-dataset", "An independent value for each dataset."),
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

        left = QtWidgets.QWidget(container)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, left)
        self._left_splitter.setObjectName("fit_parameter_left_split")
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.splitterMoved.connect(self._on_roster_splitter_moved)
        left_layout.addWidget(self._left_splitter)
        roster_pane = QtWidgets.QWidget(self._left_splitter)
        roster_layout = QtWidgets.QVBoxLayout(roster_pane)
        roster_layout.setContentsMargins(0, 0, 0, 6)
        self._param_table = _ParameterRosterTable(0, _C.COUNT, roster_pane)
        self._param_table.setObjectName("fit_parameter_roster")
        self._param_table.setWordWrap(False)
        self._param_table.setAlternatingRowColors(True)
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self._param_table.setHorizontalHeaderLabels(_C.HEADERS)
        self._param_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ContiguousSelection
        )
        header = self._param_table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(_C.COUNT):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Fixed
                if column in self._CONTENT_COLUMNS
                else QtWidgets.QHeaderView.ResizeMode.Stretch if column == _C.NAME
                else QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
        self._param_table.itemChanged.connect(self._on_roster_item_changed)
        compact = _CompactNumberDelegate(self)
        for column in (_C.MECH, _C.START, _C.MIN, _C.MAX):
            self._param_table.setItemDelegateForColumn(column, compact)
        self._param_table.setItemDelegateForColumn(_C.NAME, ScientificNameDelegate(self._param_table))
        self._param_table.setItemDelegateForColumn(_C.UNIT, ScientificNameDelegate(self._param_table))
        self._param_table.installEventFilter(self)
        roster_layout.addWidget(self._param_table, 1)
        self._parameter_error = QtWidgets.QLabel(roster_pane)
        self._parameter_error.setWordWrap(True)
        self._parameter_error.setVisible(False)
        roster_layout.addWidget(self._parameter_error)
        self._discard_parameter_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Escape), self._param_table
        )
        self._discard_parameter_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._discard_parameter_shortcut.setEnabled(False)
        self._discard_parameter_shortcut.activated.connect(self._revert_selected_parameter_draft)
        layout.addWidget(left, stretch=1)

        self._roster_actions_host = QtWidgets.QWidget(roster_pane)
        actions = ControlRowsLayout(self._roster_actions_host)
        self._roster_actions_host.installEventFilter(self)
        self._view_steps_button = QtWidgets.QPushButton("View Mechanism Steps", left)
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
        self._bounds_button = QtWidgets.QPushButton("Set Bounds for Parameters…", left)
        self._bounds_button.clicked.connect(self._open_bounds_dialog)
        for control in (
            self._bounds_button,
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

        settings_group = QtWidgets.QGroupBox("Fit Settings", options_pane)
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
        self._solver_combo.setToolTip(SOLVER_TOOLTIP)
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
            "Fixed Seed", settings_group
        )
        self._seed_spin = QtWidgets.QSpinBox(settings_group)
        self._seed_spin.setRange(*FIT_SEED_RANGE)
        self._seed_spin.setValue(defaults.seed)
        self._seed_spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        for control in (self._solver_combo, self._method_combo, self._max_eval_spin, self._seed_spin):
            ignore_wheel(control)
        self._seed_check.setChecked(defaults.use_seed)
        self._seed_spin.setEnabled(defaults.use_seed)
        self._ftol_edit = QtWidgets.QLineEdit(settings_group)
        self._xtol_edit = QtWidgets.QLineEdit(settings_group)
        self._ftol_edit.setObjectName("fit_objective_tolerance")
        self._xtol_edit.setObjectName("fit_parameter_tolerance")
        self._ftol_label = QtWidgets.QLabel(f"{FUNCTION_TOLERANCE_LABEL}:", settings_group)
        self._xtol_label = QtWidgets.QLabel(f"{STEP_TOLERANCE_LABEL}:", settings_group)
        for label, edit in ((self._ftol_label, self._ftol_edit), (self._xtol_label, self._xtol_edit)):
            configure_compact_numeric_edit(edit, characters=8)
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

        profile_card = QtWidgets.QGroupBox("Profile Likelihood", self._analysis_cards_host)
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

        bootstrap_card = QtWidgets.QGroupBox("Residual Bootstrap", self._analysis_cards_host)
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
        self._project_defaults(defaults)
        return widget

    @staticmethod
    def _add_method_items(combo: QtWidgets.QComboBox) -> None:
        for method in UNCERTAINTY_FIT_METHODS:
            combo.addItem(FIT_METHOD_LABELS.get(method, method), method)

    def _dataset_display_name(self, dataset_id: str) -> str:
        if self._origin is not None:
            for mapped in self._origin.mapped_datasets:
                if mapped.dataset_id == dataset_id:
                    return str(mapped.display_name)
        return str(dataset_id)

    def _parameter_by_name(self, name: str) -> FitParameterVariable | None:
        if self._projected_setup is None:
            return None
        return next((item for item in self._projected_setup.parameters if item.name == name), None)

    def _display_parameter_name(self, name: str) -> str:
        parameter = self._parameter_by_name(name)
        return parameter.display_name if parameter is not None else name

    def _coordinate_display_name(self, coordinate: FitCoordinate) -> str:
        datasets = fit_coordinate_datasets_text(coordinate, self._projected_setup)
        name = self._display_parameter_name(coordinate.name)
        return f"{name} — {datasets}" if datasets else name

    def _available_bootstrap_coordinates(self) -> tuple[FitCoordinate, ...]:
        return self._available_analysis_coordinates()

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        previous = self._projected_setup
        self._origin = snapshot.origin
        self._projected_setup = snapshot
        if previous is not None:
            # A scope or group change discards that parameter's invalid text.
            structure = {
                item.name: (item.scope, item.per_dataset_config.groups) for item in previous.parameters
            }
            for parameter in snapshot.parameters:
                if (
                    structure[parameter.name] != (parameter.scope, parameter.per_dataset_config.groups)
                    and self._retire_numeric_drafts({parameter.name})
                ):
                    self.statusMessage.emit(f"Invalid values for {parameter.display_name} discarded.")
        self._projecting = True
        try:
            self._render_roster()
            self._solver_combo.setCurrentText(snapshot.solver)
            self._project_defaults(snapshot.defaults)
            self._project_advanced_analysis(snapshot)
        finally:
            self._projecting = False
        self._refresh_parameter_error()

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        if snapshot is not None:
            self._last_fit_parameters = {item.name: item for item in snapshot.parameters}
            self._last_fit_included = {item.dataset_id for item in snapshot.datasets if item.included}
        else:
            self._last_fit_parameters.clear()
            self._last_fit_included = set()
        self.set_reset_availability(
            initial=self._can_reset_initial,
            last_fit=snapshot is not None,
        )
        self._rerender_roster()

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self.set_reset_availability(initial=True, last_fit=self._can_reset_last_fit)

    def set_reset_availability(self, *, initial: bool, last_fit: bool) -> None:
        self._can_reset_initial = bool(initial)
        self._can_reset_last_fit = bool(last_fit)
        self._reset_initial_button.setEnabled(self._can_reset_initial)
        self._reset_last_button.setEnabled(self._can_reset_last_fit)

    def _rerender_roster(self) -> None:
        if self._projected_setup is None:
            return
        self._projecting = True
        try:
            self._render_roster()
        finally:
            self._projecting = False

    def _render_roster(self) -> None:
        snapshot = self._projected_setup
        dataset_ids = tuple(item.dataset_id for item in snapshot.datasets)
        included = {item.dataset_id for item in snapshot.datasets if item.included}
        rows: list[tuple[FitParameterVariable, str | None]] = []
        for parameter in snapshot.parameters:
            rows.append((parameter, None))
            if parameter.scope != "shared":
                rows.extend((parameter, dataset_id) for dataset_id in dataset_ids)
        keys = tuple((parameter.name, dataset_id) for parameter, dataset_id in rows)
        current = self._param_table.currentIndex()
        current_key = (
            self._row_keys[current.row()]
            if current.isValid() and current.row() < len(self._row_keys) else None
        )
        blocker = QtCore.QSignalBlocker(self._param_table)
        try:
            if keys != self._row_keys:
                # Only a structural change (scope, roster) replaces rows; ordinary
                # edits update cells in place so an open editor stays valid.
                self._param_table.setRowCount(0)
                self._param_table.setRowCount(len(rows))
                self._row_keys = keys
            for row, (parameter, dataset_id) in enumerate(rows):
                if dataset_id is None:
                    self._render_parameter_row(row, parameter)
                else:
                    self._render_dataset_row(row, parameter, dataset_id, dataset_id in included)
            self._resize_to_contents()
            self._param_table.updateGeometry()
        finally:
            del blocker
        if current_key in keys:
            self._param_table.setCurrentCell(keys.index(current_key), current.column())
        elif current_key is not None and (current_key[0], None) in keys:
            self._param_table.setCurrentCell(keys.index((current_key[0], None)), current.column())

    _CONTENT_COLUMNS = (_C.VARY, _C.LOG, _C.SCOPE, _C.UNIT)

    def _resize_to_contents(self) -> None:
        """Measure rows and content-sized columns once per render."""

        self._param_table.resizeRowsToContents()
        for column in self._CONTENT_COLUMNS:
            self._param_table.resizeColumnToContents(column)
        self._param_table.fit_name_column()

    @staticmethod
    def _readonly_item(text: str = "") -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        return item

    def _number_item(self, text: str, message: str | None) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setData(_COMPACT_ROLE, message is None)
        item.setData(QtCore.Qt.AccessibleTextRole, text)
        if message is not None:
            item.setBackground(_INVALID_BG)
            item.setForeground(_INVALID_FG)
            item.setToolTip(message)
        return item

    def _unit_item(self, unit: str) -> QtWidgets.QTableWidgetItem:
        item = self._readonly_item(unit)
        item.setData(SCIENTIFIC_NAME_HTML_ROLE, format_unit_for_display(unit))
        item.setData(QtCore.Qt.AccessibleTextRole, unit)
        item.setToolTip(unit)
        return item

    def _render_parameter_row(self, row: int, parameter: FitParameterVariable) -> None:
        config = parameter.selected_config
        draft = self._invalid_rows.get(parameter.name)
        raw = dict(draft.raw_fields) if draft is not None else {}
        messages = dict(draft.messages) if draft is not None else {}
        vary = QtWidgets.QTableWidgetItem()
        vary.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        # Partly ticked when some datasets are varied and some fixed.
        vary.setCheckState(
            QtCore.Qt.CheckState.Unchecked if not config.fit
            else QtCore.Qt.CheckState.PartiallyChecked if config.fixed_dataset_ids
            else QtCore.Qt.CheckState.Checked
        )
        vary.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        self._param_table.setItem(row, _C.VARY, vary)
        name_item = self._readonly_item(parameter.display_name)
        symbol = scientific_symbol_html(parameter.name, family=parameter.family, role=parameter.role)
        context = parameter.display_name.removeprefix(parameter.name)
        name_item.setData(SCIENTIFIC_NAME_HTML_ROLE,
                          symbol + (" " + html.escape(compact_text(context, max_chars=36).display) if context else ""))
        name_item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole,
                          scientific_symbol_spoken(parameter.name, family=parameter.family, role=parameter.role))
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        name_item.setToolTip(f"{parameter.display_name}\nParameter ID: {parameter.name}")
        self._param_table.setItem(row, _C.NAME, name_item)
        origin_parameter = next(item for item in self._origin.parameter_defaults if item.name == parameter.name)
        mechanism = self._readonly_item(_format_number(origin_parameter.shared_config.value))
        mechanism.setTextAlignment(QtCore.Qt.AlignCenter)
        mechanism.setData(_COMPACT_ROLE, True)
        mechanism.setToolTip(
            "The value currently in the mechanism tab: "
            f"{six_digit_text(origin_parameter.shared_config.value)}"
        )
        self._param_table.setItem(row, _C.MECH, mechanism)
        if parameter.scope == "shared":
            start = self._number_item(raw.get("starting", _format_number(config.value)), messages.get("starting"))
        else:
            start = self._readonly_item("Per-dataset")
            start.setTextAlignment(QtCore.Qt.AlignCenter)
            start.setForeground(self.palette().color(QtGui.QPalette.PlaceholderText))
            start.setToolTip("Edit the initial guess of each dataset in the rows below.")
        self._param_table.setItem(row, _C.START, start)
        self._param_table.setItem(row, _C.MIN, self._number_item(raw.get("minimum", _format_number(config.minimum)), messages.get("minimum")))
        self._param_table.setItem(row, _C.MAX, self._number_item(raw.get("maximum", _format_number(config.maximum)), messages.get("maximum")))
        log_item = QtWidgets.QTableWidgetItem()
        log_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        log10 = raw.get("log10", "1" if config.log10 else "") == "1"
        log_item.setCheckState(QtCore.Qt.Checked if log10 else QtCore.Qt.Unchecked)
        log_item.setToolTip("Fit this parameter on a logarithmic scale (requires positive bounds).")
        if "log10" in messages:
            log_item.setBackground(_INVALID_BG)
            log_item.setToolTip(messages["log10"])
        self._param_table.setItem(row, _C.LOG, log_item)
        self._param_table.setItem(row, _C.SCOPE, self._readonly_item())
        combo = self._param_table.cellWidget(row, _C.SCOPE)
        if not isinstance(combo, QtWidgets.QComboBox):
            combo = QtWidgets.QComboBox(self._param_table)
            ignore_wheel(combo)
            combo.setObjectName("fit_parameter_scope")
            combo.setAccessibleName(f"Across Datasets for {parameter.name}")
            for scope, (label, tooltip) in self._SCOPE_CHOICES.items():
                combo.addItem(label, scope)
                combo.setItemData(combo.count() - 1, tooltip, QtCore.Qt.ToolTipRole)
            combo.currentIndexChanged.connect(
                lambda _index, name=parameter.name, box=combo: self._on_scope_changed(name, box)
            )
            self._param_table.setCellWidget(row, _C.SCOPE, combo)
        with QtCore.QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(parameter.scope))
        combo.setToolTip(self._SCOPE_CHOICES[parameter.scope][1])
        last = self._readonly_item(self._last_fit_summary(parameter.name))
        last.setTextAlignment(QtCore.Qt.AlignCenter)
        self._param_table.setItem(row, _C.LAST, last)
        self._param_table.setItem(row, _C.UNIT, self._unit_item(parameter.unit))

    def _render_dataset_row(
        self, row: int, parameter: FitParameterVariable, dataset_id: str, included: bool,
    ) -> None:
        for column in range(_C.COUNT):
            self._param_table.setItem(row, column, self._readonly_item())
        name = self._dataset_display_name(dataset_id)
        name_item = self._readonly_item("    " + name)
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        name_item.setData(_DATASET_ROLE, dataset_id)
        name_item.setToolTip(
            f"{parameter.display_name} — {name}\nDataset ID: {dataset_id}"
            + ("" if included else "\nNot included in this fit")
        )
        if not included:
            name_item.setForeground(self.palette().color(QtGui.QPalette.PlaceholderText))
        self._param_table.setItem(row, _C.NAME, name_item)
        config = parameter.per_dataset_config
        group = next((item for item in config.groups if dataset_id in item), None)
        varied = dataset_id in config.varied_dataset_ids
        checkable = (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        vary = QtWidgets.QTableWidgetItem()
        vary.setFlags(checkable)
        vary.setCheckState(QtCore.Qt.CheckState.Checked if varied else QtCore.Qt.CheckState.Unchecked)
        vary.setToolTip("Fit the value of this dataset. Uncheck to use it as entered.")
        self._param_table.setItem(row, _C.VARY, vary)
        labels = fit_group_labels(parameter, self._projected_setup.datasets, self._origin)
        group_box = QtWidgets.QTableWidgetItem(
            f"Group {labels[group[0]][0]}" if group is not None else "Group —"
        )
        # Only a varied dataset can share a fitted value.
        group_box.setFlags(checkable if varied else QtCore.Qt.ItemFlag.ItemIsSelectable)
        group_box.setCheckState(
            QtCore.Qt.CheckState.Checked if group is not None else QtCore.Qt.CheckState.Unchecked
        )
        group_box.setToolTip(
            "Share one fitted value with other datasets. Uncheck to leave the group."
            if varied else "A fixed dataset cannot share a fitted value."
        )
        self._param_table.setItem(row, _C.SCOPE, group_box)
        draft = self._invalid_rows.get(parameter.name)
        raw = dict(draft.raw_fields) if draft is not None else {}
        values = dict(config.dataset_values)
        text = raw.get(f"dataset:{dataset_id}", _format_number(values[dataset_id]))
        if group is not None and dataset_id != group[0]:
            # A member shows its group's start; only the origin row edits it.
            start = self._readonly_item(text)
            start.setTextAlignment(QtCore.Qt.AlignCenter)
            start.setData(_COMPACT_ROLE, True)
            start.setToolTip(
                f"Shared with {self._dataset_display_name(group[0])}. Edit the initial guess there."
            )
        else:
            start = self._number_item(text, dict(draft.messages).get(f"dataset:{dataset_id}") if draft is not None else None)
            if not varied and draft is None:
                start.setToolTip("Fixed: used as entered.")
        start.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
        start.setData(_DATASET_ROLE, dataset_id)
        self._param_table.setItem(row, _C.START, start)
        last = self._last_fit_parameters.get(parameter.name)
        # A dataset outside the last fit shows no result, even when a group
        # wrote that fit's value into it.
        last_values = (
            dict(last.selected_config.dataset_values)
            if last is not None and last.scope != "shared" and dataset_id in self._last_fit_included
            else {}
        )
        last_item = self._readonly_item(
            six_digit_text(last_values[dataset_id]) if dataset_id in last_values else "—"
        )
        last_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self._param_table.setItem(row, _C.LAST, last_item)
        self._param_table.setItem(row, _C.UNIT, self._unit_item(parameter.unit))

    def _last_fit_summary(self, name: str) -> str:
        last = self._last_fit_parameters.get(name)
        if last is None:
            return "—"
        if last.scope == "shared":
            return six_digit_text(last.selected_config.value)
        return "Per-dataset"

    def _parameter_rows(self, name: str) -> tuple[int, list[tuple[int, str]]]:
        parent = -1
        datasets: list[tuple[int, str]] = []
        for row in range(self._param_table.rowCount()):
            item = self._param_table.item(row, _C.NAME)
            if item is None or item.data(QtCore.Qt.ItemDataRole.UserRole) != name:
                continue
            dataset_id = item.data(_DATASET_ROLE)
            if dataset_id is None:
                parent = row
            else:
                datasets.append((row, str(dataset_id)))
        return parent, datasets

    def _on_roster_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._projecting:
            return
        name_item = self._param_table.item(item.row(), _C.NAME)
        name = str(name_item.data(QtCore.Qt.ItemDataRole.UserRole))
        dataset_id = name_item.data(_DATASET_ROLE)
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if item.column() == _C.VARY and dataset_id is None:
            self._emit_vary_edit(name, checked)
        elif item.column() == _C.VARY:
            self._on_dataset_vary_changed(name, str(dataset_id), checked)
        elif item.column() == _C.SCOPE and dataset_id is not None:
            self._on_group_box_changed(name, str(dataset_id), checked)
        elif item.column() in (_C.START, _C.MIN, _C.MAX, _C.LOG):
            self._emit_parameter_row_edit(name)

    def _selected_parameter(self) -> FitParameterVariable | None:
        row = self._param_table.currentRow()
        item = self._param_table.item(row, _C.NAME) if row >= 0 else None
        name = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        return self._parameter_by_name(name) if name else None

    def _row_raw_fields(self, parameter: FitParameterVariable) -> dict[str, str]:
        parent, datasets = self._parameter_rows(parameter.name)
        table = self._param_table
        raw = {
            "minimum": table.item(parent, _C.MIN).text(),
            "maximum": table.item(parent, _C.MAX).text(),
            "log10": "1" if table.item(parent, _C.LOG).checkState() == QtCore.Qt.Checked else "",
        }
        if parameter.scope == "shared":
            raw["starting"] = table.item(parent, _C.START).text()
        for row, dataset_id in datasets:
            raw[f"dataset:{dataset_id}"] = table.item(row, _C.START).text()
        return raw

    def _invalid_cells(self) -> list[tuple[str, str, str]]:
        """Each invalid cell as (parameter name, field, message) in table order."""

        cells = []
        for parameter in self._projected_setup.parameters if self._projected_setup else ():
            draft = self._invalid_rows.get(parameter.name)
            if draft is None:
                continue
            messages = dict(draft.messages)
            fields = [field for field in _FIELD_COLUMNS if field in messages]
            fields += [field for field, _raw in draft.raw_fields if field.startswith("dataset:") and field in messages]
            cells += [(parameter.name, field, messages[field]) for field in fields]
        return cells

    def _refresh_parameter_error(self) -> None:
        # Every message names its field (and a dataset start its dataset once).
        messages = tuple(dict.fromkeys(
            f"{name}: {message}" for name, _field, message in self._invalid_cells()
        ))
        self._parameter_error.setText(
            "\n".join(messages)
            + "\nSelect a cell of that parameter and press Esc to discard its invalid edits."
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
        self._rerender_roster()
        self.statusMessage.emit(
            f"Invalid values for {parameter.display_name} reverted."
        )

    def _parameter_projection_for_change(
        self, parameter: FitParameterVariable, raw: Mapping[str, str],
    ) -> tuple[FitParameterVariable | None, dict[str, str]]:
        """Build one complete parameter from the row texts, or mark the changed cells.

        A row that is not admissible marks every cell the user changed with
        the row's admission message; unchanged cells are never marked.
        """

        config = parameter.selected_config
        parsed: dict[str, float] = {}
        messages: dict[str, str] = {}
        for field, text in raw.items():
            if field == "log10":
                continue
            try:
                parsed[field] = self._parse_number(text, _parse_label(
                    field, self._dataset_display_name(field.removeprefix("dataset:")),
                ))
            except ValueError as exc:
                messages[field] = str(exc)
        if messages:
            return None, messages
        log10 = raw["log10"] == "1"
        dataset_values = dict(config.dataset_values)
        dataset_values.update(
            (field.removeprefix("dataset:"), value)
            for field, value in parsed.items() if field.startswith("dataset:")
        )
        # Admitting an origin edit copies its value to every group member.
        for group in config.groups:
            for member in group[1:]:
                dataset_values[member] = dataset_values[group[0]]
        try:
            active = replace(
                config,
                value=parsed.get("starting", config.value),
                minimum=parsed["minimum"],
                maximum=parsed["maximum"],
                log10=log10,
                dataset_values=config.dataset_values if parameter.scope == "shared" else tuple(dataset_values.items()),
            )
            return replace(
                parameter,
                **{"shared_config" if parameter.scope == "shared" else "per_dataset_config": active},
            ), {}
        except (TypeError, ValueError, OverflowError) as exc:
            text = fit_identity_text(str(exc), self._origin).removeprefix(f"{parameter.name} ")
            message = text[:1].upper() + text[1:]
        accepted = {
            "starting": config.value, "minimum": config.minimum, "maximum": config.maximum,
            **{f"dataset:{key}": value for key, value in config.dataset_values},
        }
        changed = [field for field, value in parsed.items() if value != accepted[field]]
        if log10 != config.log10:
            changed.append("log10")
        return None, dict.fromkeys(changed, message)

    def _emit_parameter_replacement(
        self, replacement: FitParameterVariable, *, confirm: str,
    ) -> None:
        """Emit the setup with one parameter replaced.

        Profile and Bootstrap keep every selected target that still exists; a
        group whose origin left keeps its targets under its new origin.
        Removing a selected target first asks the user, naming ``confirm`` as
        the change.
        """

        setup = self._projected_setup
        previous = self._parameter_by_name(replacement.name)
        parameters = tuple(
            replacement if item.name == replacement.name else item for item in setup.parameters
        )
        available = set(fit_uncertainty_coordinates(parameters, setup.datasets))
        renamed = {
            FitCoordinate("group", replacement.name, group[0]):
                FitCoordinate("group", replacement.name, remaining[0])
            for group in previous.selected_config.groups
            for remaining in replacement.selected_config.groups
            if remaining[0] != group[0] and set(remaining) <= set(group)
        }
        configurations = {
            label: configuration if configuration is None or not renamed else replace(
                configuration,
                parameter_names=tuple(renamed.get(item, item) for item in configuration.parameter_names),
            )
            for label, configuration in (
                ("Profile", setup.profile_likelihood), ("Bootstrap", setup.residual_bootstrap),
            )
        }
        analyses = {
            label: prune_fit_analysis_coordinates(configuration, available=available, parameters=parameters)
            for label, configuration in configurations.items()
        }
        removed = {
            label: tuple(
                coordinate for coordinate in configuration.parameter_names
                if coordinate not in available
            )
            for label, configuration in configurations.items()
            if configuration is not None
        }
        removed = {label: coordinates for label, coordinates in removed.items() if coordinates}
        turned_off = {label for label in removed if analyses[label] is None}
        if removed and not self._confirm_removed_targets(confirm, removed, turned_off):
            self._rerender_roster()
            return
        self.setupSnapshotRequested.emit(
            replace(
                setup,
                parameters=parameters,
                profile_likelihood=analyses["Profile"],
                residual_bootstrap=analyses["Bootstrap"],
            )
        )
        self._rerender_roster()

    def _confirm_removed_targets(
        self, action: str, removed: Mapping[str, tuple[FitCoordinate, ...]], turned_off: set[str],
    ) -> bool:
        """Ask before a change removes selected Profile or Bootstrap targets.

        An analysis left with no target is named as turned off.
        """

        details = [f"{action} will update these analyses:"]
        details.extend(
            f"{label}: {', '.join(self._coordinate_display_name(item) for item in coordinates)}"
            + (f" (no targets remain, so {label} will be turned off)" if label in turned_off else "")
            for label, coordinates in removed.items()
        )
        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message.setWindowTitle("Update Uncertainty Analyses?")
        message.setText("\n".join(details))
        update_button = message.addButton(
            "Update Analyses",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = message.addButton(
            "Cancel",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is update_button

    def _emit_vary_edit(self, name: str, desired_fit: bool) -> None:
        """Apply the parameter row's Vary: every dataset varied, or every one fixed."""

        if self._running:
            return
        parameter = self._parameter_by_name(name)
        config = parameter.selected_config
        if desired_fit == config.fit and not config.fixed_dataset_ids:
            self._rerender_roster()
            return
        if desired_fit:
            replacement = _replace_fit_parameter(parameter, fit=True, fixed_dataset_ids=())
            self._emit_parameter_replacement(
                replacement, confirm=f"Varying {parameter.display_name}",
            )
            return
        # Vary off fixes every dataset and dissolves the parameter's groups.
        replacement = _replace_fit_parameter(parameter, fit=False, groups=(), fixed_dataset_ids=())
        self._emit_parameter_replacement(
            replacement, confirm=f"Stopping Vary for {parameter.display_name}",
        )

    def _on_dataset_vary_changed(self, name: str, dataset_id: str, checked: bool) -> None:
        """Vary or fix one dataset's value; a dataset that is fixed leaves its group."""

        if self._running:
            return
        parameter = self._parameter_by_name(name)
        config = parameter.per_dataset_config
        all_ids = {item for item, _value in config.dataset_values}
        fixed = set(config.fixed_dataset_ids) if config.fit else set(all_ids)
        if checked == (dataset_id not in fixed):
            self._rerender_roster()
            return
        if checked:
            replacement = _replace_fit_parameter(
                parameter, fit=True, fixed_dataset_ids=tuple(fixed - {dataset_id}),
            )
            self._emit_parameter_replacement(
                replacement,
                confirm=f"Varying {parameter.display_name} on {self._dataset_display_name(dataset_id)}",
            )
            return
        fixed.add(dataset_id)
        groups = self._groups_without(parameter, dataset_id)
        replacement = (
            _replace_fit_parameter(parameter, fit=False, groups=(), fixed_dataset_ids=())
            if fixed == all_ids
            else _replace_fit_parameter(parameter, groups=groups, fixed_dataset_ids=tuple(fixed))
        )
        self._emit_parameter_replacement(
            replacement,
            confirm=f"Stopping Vary for {parameter.display_name} on {self._dataset_display_name(dataset_id)}",
        )

    def _emit_parameter_row_edit(self, name: str) -> None:
        if self._running:
            return
        parameter = self._parameter_by_name(name)
        raw = self._row_raw_fields(parameter)
        edited, messages = self._parameter_projection_for_change(parameter, raw)
        if edited is None:
            self._store_invalid_numeric_draft(parameter.name, raw, messages)
            self._rerender_roster()
            return
        updated = replace(
            self._projected_setup,
            parameters=tuple(
                edited if item.name == edited.name else item
                for item in self._projected_setup.parameters
            ),
        )
        self._retire_numeric_drafts({parameter.name})
        if updated == self._projected_setup:
            self._rerender_roster()
            return
        self.setupSnapshotRequested.emit(updated)

    def _open_bounds_dialog(self) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        setup = self._projected_setup
        dialog = _ParameterBoundsDialog(setup.parameters, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        names, minimum, maximum, log10 = dialog.selection()
        edited: dict[str, FitParameterVariable] = {}
        for parameter in setup.parameters:
            if parameter.name not in names:
                continue
            raw = {
                **self._row_raw_fields(parameter),
                "minimum": minimum,
                "maximum": maximum,
                "log10": "1" if log10 else "",
            }
            replacement, messages = self._parameter_projection_for_change(parameter, raw)
            if replacement is None:
                self._store_invalid_numeric_draft(parameter.name, raw, messages)
            else:
                edited[parameter.name] = replacement
                self._retire_numeric_drafts({parameter.name})
        updated = replace(
            setup,
            parameters=tuple(edited.get(item.name, item) for item in setup.parameters),
        )
        if updated == setup:
            self._rerender_roster()
            return
        self.setupSnapshotRequested.emit(updated)

    def _on_scope_changed(self, name: str, combo: QtWidgets.QComboBox) -> None:
        if self._projecting or self._running or self._projected_setup is None:
            return
        parameter = self._parameter_by_name(name)
        if parameter is None:
            return
        desired = str(combo.currentData() or "")
        if desired == parameter.scope:
            return
        replacement = convert_fit_parameter_scope(
            parameter, desired, self._projected_setup.datasets, self._origin,
        )
        # Restore the projection first; the owner's accepted snapshot reprojects.
        with QtCore.QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(parameter.scope))
        self._emit_parameter_replacement(
            replacement, confirm=f"Changing the scope of {parameter.display_name}",
        )

    def _groups_without(
        self, parameter: FitParameterVariable, dataset_id: str,
    ) -> tuple[tuple[str, ...], ...]:
        """The parameter's groups after one dataset leaves.

        A group left with one member dissolves; when its origin leaves, the
        first remaining member in table order becomes the origin.
        """

        order = tuple(item.dataset_id for item in self._projected_setup.datasets)
        groups = []
        for group in parameter.per_dataset_config.groups:
            remaining = sorted((item for item in group if item != dataset_id), key=order.index)
            if len(remaining) < 2:
                continue
            origin = group[0] if group[0] in remaining else remaining[0]
            groups.append((origin, *(item for item in remaining if item != origin)))
        return tuple(groups)

    def _on_group_box_changed(self, name: str, dataset_id: str, checked: bool) -> None:
        """Unchecking leaves the dataset's group; checking opens the membership dialog."""

        if self._running:
            return
        parameter = self._parameter_by_name(name)
        config = parameter.per_dataset_config
        if not checked:
            self._emit_group_change(parameter, self._groups_without(parameter, dataset_id))
            return
        choice = self._group_membership_dialog(parameter, dataset_id)
        if choice is None:
            self._rerender_roster()
            return
        joined, added = choice
        if joined is None:
            if not added:
                self.statusMessage.emit("A group needs at least one other dataset.")
                self._rerender_roster()
                return
            self._emit_group_change(parameter, (*config.groups, (dataset_id, *added)))
            return
        self._emit_group_change(parameter, tuple(
            (*group, dataset_id, *added) if group[0] == joined else group for group in config.groups
        ))

    def _group_membership_dialog(
        self, parameter: FitParameterVariable, dataset_id: str,
    ) -> tuple[str | None, tuple[str, ...]] | None:
        """Ask which group the dataset joins, or which datasets form a new group with it.

        Returns the joined group's origin (None for a new group) and the other
        datasets added, or None on Cancel.  The dialog only adds: existing
        members stay, and datasets in other groups or fixed are greyed.
        """

        config = parameter.per_dataset_config
        labels = fit_group_labels(parameter, self._projected_setup.datasets, self._origin)
        groups = sorted(config.groups, key=lambda group: labels[group[0]][0])
        included = {item.dataset_id for item in self._projected_setup.datasets if item.included}
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Datasets Sharing {parameter.display_name}")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(f"Datasets sharing {parameter.display_name}:", dialog))
        target = QtWidgets.QComboBox(dialog)
        target.addItem("New Group", None)
        for group in groups:
            target.addItem(labels[group[0]][1], group[0])
        layout.addWidget(target)
        checks: dict[str, QtWidgets.QCheckBox] = {}
        for dataset in self._projected_setup.datasets:
            name = self._dataset_display_name(dataset.dataset_id)
            check = QtWidgets.QCheckBox(
                name if dataset.dataset_id in included else f"{name} (not in fit)", dialog,
            )
            layout.addWidget(check)
            checks[dataset.dataset_id] = check

        def project() -> None:
            """Tick and lock the clicked dataset and the chosen group's members."""

            joined = target.currentData()
            members = next((group for group in groups if group[0] == joined), ())
            occupied = {member for group in groups for member in group}
            for item, check in checks.items():
                locked = item == dataset_id or item in members
                free = not locked and item not in occupied and item in config.varied_dataset_ids
                check.setChecked(locked or (free and check.isChecked()))
                check.setEnabled(free)

        target.currentIndexChanged.connect(project)
        project()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return target.currentData(), tuple(
            item for item, check in checks.items() if check.isChecked() and check.isEnabled()
        )

    def _emit_group_change(
        self, parameter: FitParameterVariable, groups: tuple[tuple[str, ...], ...],
    ) -> None:
        """Emit the setup with new groups; every member stores its origin's value."""

        values = dict(parameter.per_dataset_config.dataset_values)
        for group in groups:
            for member in group[1:]:
                values[member] = values[group[0]]
        self._emit_parameter_replacement(
            _replace_fit_parameter(parameter, dataset_values=tuple(values.items()), groups=groups),
            confirm=f"Changing the groups of {parameter.display_name}",
        )

    @property
    def invalid_edit_locations(self) -> tuple[str, ...]:
        """Name each invalid cell by parameter, dataset and column, in table order."""

        located = [
            f"{self._dataset_display_name(field.removeprefix('dataset:'))}: {name} {_field_label(field)}"
            if field.startswith("dataset:")
            else f"{name} {_field_label(field)}"
            for name, field, _message in self._invalid_cells()
        ]
        if self._defaults_error_text:
            # Only the tolerance pair rule fails without one box failing alone.
            located += [
                f"Fit Settings: {label}" for label in self._invalid_tolerance_labels()
            ] or ["Fit Settings: tolerances"]
        return tuple(located)

    def _store_invalid_numeric_draft(
        self,
        name: str,
        raw_fields: Mapping[str, str],
        messages: Mapping[str, str],
    ) -> None:
        draft = _InvalidRow(tuple(raw_fields.items()), tuple(messages.items()))
        if self._invalid_rows.get(name) != draft:
            self._invalid_rows[name] = draft
            self._refresh_parameter_error()
            self.validityChanged.emit()

    def _retire_numeric_drafts(
        self,
        parameter_names: set[str] | None = None,
    ) -> bool:
        keys = tuple(
            name for name in self._invalid_rows
            if parameter_names is None or name in parameter_names
        )
        for key in keys:
            self._invalid_rows.pop(key)
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
        self._ftol_label.setText(f"{DE_CONVERGENCE_TOLERANCE_LABEL if is_de else FUNCTION_TOLERANCE_LABEL}:")
        self._xtol_label.setVisible(not is_de)
        self._xtol_edit.setVisible(not is_de)
        self._xtol_group.setVisible(not is_de)
        del blocker

    def _defaults_projection(self) -> FitDefaults:
        if self._projected_setup is None:
            raise ValueError("Fit defaults are not available.")
        defaults = self._projected_setup.defaults
        method = str(self._method_combo.currentData() or "trf")
        ftol = self._tolerance_value(self._ftol_edit, self._ftol_label)
        xtol = (
            defaults.xtol if method == DIFFERENTIAL_EVOLUTION_METHOD
            else self._tolerance_value(self._xtol_edit, self._xtol_label)
        )
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

    def _tolerance_value(self, edit: QtWidgets.QLineEdit, label: QtWidgets.QLabel) -> float:
        """Parse one tolerance box, naming it by its visible label."""

        return admit_fit_tolerance(edit.text(), label.text().removesuffix(":"))

    def _invalid_tolerance_labels(self) -> list[str]:
        boxes = [(self._ftol_edit, self._ftol_label)]
        if self._method_combo.currentData() != DIFFERENTIAL_EVOLUTION_METHOD:
            boxes.append((self._xtol_edit, self._xtol_label))
        labels = []
        for edit, label in boxes:
            try:
                self._tolerance_value(edit, label)
            except ValueError:
                labels.append(label.text().removesuffix(":"))
        return labels

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
        # Readiness names each failing box, so every tolerance change refreshes it.
        self._defaults_error_text = message
        self._defaults_error.setText(
            message + "\nPress Esc in a tolerance box to restore the accepted tolerances." if message else ""
        )
        self._defaults_error.setVisible(bool(message))
        self._discard_tolerances_shortcut.setEnabled(bool(self._tolerance_drafts))
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
            excluded = tuple(item.dataset_id for item in snapshot.datasets if not item.included)

            def skipped(configuration: object | None) -> int:
                return 0 if configuration is None else len(skipped_fit_analysis_coordinates(
                    configuration.parameter_names, excluded_dataset_ids=excluded, parameters=snapshot.parameters,
                ))

            self._profile_summary.setText(self._analysis_summary(profile, "Profile", skipped(profile)))
            self._bootstrap_summary.setText(
                self._analysis_summary(bootstrap, "Bootstrap", skipped(bootstrap))
            )
        finally:
            del blockers

    @staticmethod
    def _analysis_summary(configuration: object | None, label: str, skipped: int) -> str:
        """The card text; ``skipped`` counts selected targets whose datasets are all excluded."""
        if configuration is None:
            return "Not configured"
        names = tuple(getattr(configuration, "parameter_names", ()))
        not_in_fit = f" ({skipped} not in fit)" if skipped else ""
        method = str(getattr(configuration, "optimizer_method", "trf"))
        if label == "Profile":
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''}{not_in_fit} · "
                f"{getattr(configuration, 'confidence_level', 0.95):.0%} · "
                f"{getattr(configuration, 'max_bracketing_steps_per_side', 0)} "
                "maximum outward search steps/side"
            )
        else:
            detail = (
                f"{len(names)} coordinate{'s' if len(names) != 1 else ''}{not_in_fit} · "
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
        return fit_uncertainty_coordinates(
            setup.parameters, setup.datasets
        )

    def _analysis_coordinate_display_names(self) -> dict[FitCoordinate, str]:
        """Dialog labels; a target whose datasets are all excluded reads "(not in fit)"."""
        setup = self._projected_setup
        available = self._available_analysis_coordinates()
        skipped = skipped_fit_analysis_coordinates(
            available,
            excluded_dataset_ids=(item.dataset_id for item in setup.datasets if not item.included),
            parameters=setup.parameters,
        )
        return {
            raw: self._coordinate_display_name(raw) + (" (not in fit)" if raw in skipped else "")
            for raw in available
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
        if watched is getattr(self, "_param_table", None) and event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.Copy):
                # Contiguous selection: one rectangle, disabled cells included.
                ranges = watched.selectedRanges()
                if ranges:
                    area = ranges[0]
                    payload = "\n".join(
                        "\t".join(
                            self._copy_text(row, column)
                            for column in range(area.leftColumn(), area.rightColumn() + 1)
                        ) for row in range(area.topRow(), area.bottomRow() + 1)
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
            self._resize_to_contents()
        return super().eventFilter(watched, event)

    def _copy_text(self, row: int, column: int) -> str:
        """Plain copy text: every cell as displayed."""

        combo = self._param_table.cellWidget(row, column)
        if combo is not None:
            return combo.currentText()
        item = self._param_table.item(row, column)
        if item is None:
            return ""
        if column in (_C.VARY, _C.LOG) and item.data(QtCore.Qt.ItemDataRole.CheckStateRole) is not None:
            return {
                QtCore.Qt.CheckState.Checked: "Yes",
                QtCore.Qt.CheckState.PartiallyChecked: "Some datasets",
            }.get(item.checkState(), "No")
        # Copy carries the delegate's display text (numbers at six digits).
        return cell_text_as_shown(self._param_table, self._param_table.indexFromItem(item)).strip()

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
