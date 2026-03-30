"""Parameters & ICs tab for the fitting window."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.ui_helpers import safe_float_parse, setup_scientific_validator
from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter
from kindred.gui.widgets.collapsible_section import CollapsibleSection
from kindred.gui.fitting.constants import INITIAL_PREFIX, DEFAULT_PARALLEL_STARTS

logger = logging.getLogger(__name__)


class _ICCol:
    """Column indices for the initial conditions table."""
    FIT = 0
    LOG10 = 1
    SPECIES = 2
    INITIAL = 3
    MIN = 4
    MAX = 5


class _AddFittableParameterDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        available_rates: Sequence[str],
        available_scalars: Sequence[str],
        available_species: Sequence[str],
        dataset_ids: Sequence[str],
        available_observables: Optional[Dict[str, str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Parameter")
        self.setModal(True)
        self.resize(520, 340)

        self._available_rates = [str(x) for x in (available_rates or []) if str(x)]
        self._available_scalars = [str(x) for x in (available_scalars or []) if str(x)]
        self._available_species = [str(x) for x in (available_species or []) if str(x)]
        self._dataset_ids = [str(x) for x in (dataset_ids or []) if str(x)]
        self._available_observables = self._normalize_available_observables(available_observables)
        self._selection: Optional[Dict[str, str]] = None

        layout = QtWidgets.QVBoxLayout(self)

        self._tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self._tabs, stretch=1)

        self._init_rate_tab()
        self._init_initial_tab()
        self._init_scalar_tab()
        self._init_observable_tab()
        self._connect_observable_signals()

        self._add_button_box(layout)

    @staticmethod
    def _normalize_available_observables(available_observables: Optional[Dict[str, str]]) -> dict[str, str]:
        return {
            str(k): str(v)
            for k, v in (available_observables or {}).items()
            if str(k).strip() and str(v).strip()
        }

    def _init_rate_tab(self) -> None:
        self._rate_tab = QtWidgets.QWidget(self._tabs)
        rate_layout = QtWidgets.QVBoxLayout(self._rate_tab)
        if self._available_rates:
            rate_layout.addWidget(QtWidgets.QLabel("Select a rate constant to add:"))
            self._rate_list = QtWidgets.QListWidget(self._rate_tab)
            self._rate_list.addItems(self._available_rates)
            self._rate_list.setCurrentRow(0)
            rate_layout.addWidget(self._rate_list, stretch=1)
        else:
            self._rate_list = None
            rate_layout.addWidget(QtWidgets.QLabel("No additional rate constants are available to add."))
            rate_layout.addStretch()
        self._tabs.addTab(self._rate_tab, "Rate Constants")

    def _init_initial_tab(self) -> None:
        self._initial_tab = QtWidgets.QWidget(self._tabs)
        init_layout = QtWidgets.QVBoxLayout(self._initial_tab)
        init_layout.addWidget(QtWidgets.QLabel("Select a species initial concentration parameter to add:"))

        form = QtWidgets.QFormLayout()
        self._species_combo = QtWidgets.QComboBox(self._initial_tab)
        self._species_combo.addItems(self._available_species)
        form.addRow("Species:", self._species_combo)

        mode_row = QtWidgets.QHBoxLayout()
        self._global_radio = QtWidgets.QRadioButton("Global")
        self._local_radio = QtWidgets.QRadioButton("Local")
        self._global_radio.setChecked(True)
        mode_row.addWidget(self._global_radio)
        mode_row.addWidget(self._local_radio)
        mode_row.addStretch()
        mode_widget = QtWidgets.QWidget(self._initial_tab)
        mode_widget.setLayout(mode_row)
        form.addRow("Mode:", mode_widget)

        ds_count = len(self._dataset_ids)
        ds_label = f"Applies to {ds_count} selected dataset{'s' if ds_count != 1 else ''}."
        self._dataset_label = QtWidgets.QLabel(ds_label)
        form.addRow("", self._dataset_label)
        init_layout.addLayout(form)

        help_label = QtWidgets.QLabel(
            "Global: one parameter updates the initial value for all selected datasets.\n"
            "Local: a separate parameter is created for each selected dataset."
        )
        help_label.setWordWrap(True)
        init_layout.addWidget(help_label)
        init_layout.addStretch()
        self._tabs.addTab(self._initial_tab, "Initial Concentrations")

    def _init_scalar_tab(self) -> None:
        self._scalar_tab = QtWidgets.QWidget(self._tabs)
        scalar_layout = QtWidgets.QVBoxLayout(self._scalar_tab)
        if self._available_scalars:
            scalar_layout.addWidget(QtWidgets.QLabel("Select an algebra scalar parameter to add:"))
            self._scalar_list = QtWidgets.QListWidget(self._scalar_tab)
            self._scalar_list.addItems(self._available_scalars)
            self._scalar_list.setCurrentRow(0)
            scalar_layout.addWidget(self._scalar_list, stretch=1)
        else:
            self._scalar_list = None
            scalar_layout.addWidget(QtWidgets.QLabel("No algebra scalar parameters are available to add."))
            scalar_layout.addStretch()

        scalar_form = QtWidgets.QFormLayout()
        scalar_mode_row = QtWidgets.QHBoxLayout()
        self._scalar_shared_radio = QtWidgets.QRadioButton("Shared")
        self._scalar_dataset_radio = QtWidgets.QRadioButton("Per-dataset")
        self._scalar_shared_radio.setChecked(True)
        scalar_mode_row.addWidget(self._scalar_shared_radio)
        scalar_mode_row.addWidget(self._scalar_dataset_radio)
        scalar_mode_row.addStretch()
        scalar_mode_widget = QtWidgets.QWidget(self._scalar_tab)
        scalar_mode_widget.setLayout(scalar_mode_row)
        scalar_form.addRow("Scope:", scalar_mode_widget)

        ds_count = len(self._dataset_ids)
        scalar_ds_label = f"Applies to {ds_count} selected dataset{'s' if ds_count != 1 else ''}."
        self._scalar_dataset_label = QtWidgets.QLabel(scalar_ds_label)
        scalar_form.addRow("", self._scalar_dataset_label)
        scalar_layout.addLayout(scalar_form)

        scalar_help = QtWidgets.QLabel(
            "Shared: one scalar value is used for all included datasets.\n"
            "Per-dataset: a separate scalar is created for each selected dataset.\n"
            "A scalar cannot be both shared and per-dataset at the same time."
        )
        scalar_help.setWordWrap(True)
        scalar_layout.addWidget(scalar_help)
        self._tabs.addTab(self._scalar_tab, "Algebra Scalars")

    def _init_observable_tab(self) -> None:
        self._observable_tab = QtWidgets.QWidget(self._tabs)
        observable_layout = QtWidgets.QVBoxLayout(self._observable_tab)
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Select an existing algebraic observable from # Algebra:"))
        header_row.addStretch(1)
        self._define_new_button = QtWidgets.QPushButton("Define new…", self._observable_tab)
        header_row.addWidget(self._define_new_button)
        header_widget = QtWidgets.QWidget(self._observable_tab)
        header_widget.setLayout(header_row)
        observable_layout.addWidget(header_widget)

        self._observable_combo = QtWidgets.QComboBox(self._observable_tab)
        self._observable_combo.addItems(sorted(self._available_observables.keys()))
        observable_layout.addWidget(self._observable_combo)

        self._no_observables_label = QtWidgets.QLabel(
            "No algebraic observables found in # Algebra. Use 'Define new…' to add one.",
            self._observable_tab,
        )
        self._no_observables_label.setWordWrap(True)
        self._no_observables_label.setEnabled(False)
        observable_layout.addWidget(self._no_observables_label)

        observable_layout.addWidget(QtWidgets.QLabel("Expression preview (read-only):"))
        self._observable_expr_preview = QtWidgets.QPlainTextEdit(self._observable_tab)
        self._observable_expr_preview.setReadOnly(True)
        self._observable_expr_preview.setMinimumHeight(70)
        self._observable_expr_preview.setPlainText("")
        observable_layout.addWidget(self._observable_expr_preview)

        scope_form = QtWidgets.QFormLayout()
        obs_scope_row = QtWidgets.QHBoxLayout()
        self._observable_shared_radio = QtWidgets.QRadioButton("Shared")
        self._observable_dataset_radio = QtWidgets.QRadioButton("Per-dataset")
        self._observable_shared_radio.setChecked(True)
        obs_scope_row.addWidget(self._observable_shared_radio)
        obs_scope_row.addWidget(self._observable_dataset_radio)
        obs_scope_row.addStretch()
        obs_scope_widget = QtWidgets.QWidget(self._observable_tab)
        obs_scope_widget.setLayout(obs_scope_row)
        scope_form.addRow("Auto-added scalars:", obs_scope_widget)
        ds_count = len(self._dataset_ids)
        obs_ds_label = f"Applies to {ds_count} selected dataset{'s' if ds_count != 1 else ''}."
        self._observable_dataset_label = QtWidgets.QLabel(obs_ds_label)
        scope_form.addRow("", self._observable_dataset_label)
        observable_layout.addLayout(scope_form)

        self._new_observable_container = QtWidgets.QGroupBox("Define new observable", self._observable_tab)
        self._new_observable_container.setVisible(False)
        self._define_new_mode = False
        new_layout = QtWidgets.QFormLayout(self._new_observable_container)
        self._new_observable_name_edit = QtWidgets.QLineEdit(self._new_observable_container)
        self._new_observable_name_edit.setPlaceholderText("e.g. signal")
        new_layout.addRow("Observable name:", self._new_observable_name_edit)
        self._new_observable_expr_edit = QtWidgets.QLineEdit(self._new_observable_container)
        self._new_observable_expr_edit.setPlaceholderText("e.g. scale * ([A] + [B])")
        new_layout.addRow("Expression:", self._new_observable_expr_edit)
        observable_layout.addWidget(self._new_observable_container)

        observable_help = QtWidgets.QLabel(
            "Species must be referenced using brackets: [A], [A]_0 (bare A is invalid).",
            self._observable_tab,
        )
        observable_help.setWordWrap(True)
        observable_layout.addWidget(observable_help)
        observable_layout.addStretch(1)
        self._tabs.addTab(self._observable_tab, "Algebraic Observables")

    def _connect_observable_signals(self) -> None:
        self._define_new_button.clicked.connect(self._toggle_define_new_observable)
        self._observable_combo.currentTextChanged.connect(self._refresh_observable_preview)
        self._refresh_observable_preview()
        self._refresh_observable_empty_state()

    def _add_button_box(self, layout: QtWidgets.QVBoxLayout) -> None:
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selection(self) -> Optional[Dict[str, str]]:
        return dict(self._selection or {}) if self._selection else None

    def _refresh_observable_empty_state(self) -> None:
        has_obs = bool(self._available_observables)
        self._observable_combo.setVisible(has_obs)
        self._observable_expr_preview.setVisible(has_obs)
        self._no_observables_label.setVisible(not has_obs)

    def _refresh_observable_preview(self) -> None:
        name = str(self._observable_combo.currentText()).strip() if hasattr(self, "_observable_combo") else ""
        expr = str(self._available_observables.get(name, "")).strip()
        try:
            self._observable_expr_preview.setPlainText(expr)
        except Exception:
            return

    def _toggle_define_new_observable(self) -> None:
        self._define_new_mode = not bool(getattr(self, "_define_new_mode", False))
        self._new_observable_container.setVisible(bool(self._define_new_mode))
        self._define_new_button.setText("Hide new…" if self._define_new_mode else "Define new…")

    def accept(self) -> None:  # noqa: D401 - Qt override
        idx = int(self._tabs.currentIndex())
        if idx == 0:
            if self._rate_list is None:
                return
            item = self._rate_list.currentItem()
            if item is None:
                return
            self._selection = {"type": "rate", "name": str(item.text())}
        elif idx == 1:
            species = str(self._species_combo.currentText()).strip()
            if not species:
                return
            mode = "global" if self._global_radio.isChecked() else "local"
            self._selection = {"type": "initial", "species": species, "mode": mode}
        elif idx == 2:
            if self._scalar_list is None:
                return
            item = self._scalar_list.currentItem()
            if item is None:
                return
            name = str(item.text()).strip()
            if not name:
                return
            mode = "shared" if self._scalar_shared_radio.isChecked() else "dataset"
            self._selection = {"type": "scalar", "name": name, "mode": mode}
        else:
            mode = "shared" if self._observable_shared_radio.isChecked() else "dataset"
            if bool(getattr(self, "_define_new_mode", False)):
                name = str(self._new_observable_name_edit.text()).strip()
                expr = str(self._new_observable_expr_edit.text()).strip()
                if not name or not expr:
                    return
                self._selection = {"type": "observable_new", "name": name, "expr": expr, "scalar_scope": mode}
            else:
                name = str(self._observable_combo.currentText()).strip()
                expr = str(self._available_observables.get(name, "")).strip()
                if not name or not expr:
                    return
                self._selection = {"type": "observable_existing", "name": name, "expr": expr, "scalar_scope": mode}
        super().accept()


class ParametersIcsTab(QtWidgets.QWidget):
    addAlgebraicObservableRequested = Signal(dict)
    statusMessage = Signal(str)

    def __init__(
        self,
        *,
        parameter_state: List[Dict[str, Any]],
        initial_parameter_snapshot: List[Dict[str, Any]],
        global_dataset_params: Dict[str, Dict[str, float]],
        global_dataset_variable_params: Dict[str, Dict[str, Dict]],
        fixed_shared_params: Dict[str, float],
        shared_param_definitions: Dict[str, Dict[str, Any]],
        mechanism_species: List[str],
        dataset_entries: List[Dict[str, Any]],
        prepared_param_names: List[str],
        selected_dataset_ids_getter: Callable[[], List[str]],
        dataset_entries_getter: Callable[[], List[Dict[str, Any]]],
        worker_running_getter: Callable[[], bool],
        dataset_manager_getter: Callable[[], Any],
        reactions_text_getter: Callable[[], str],
        integration_defaults: Tuple[str, float, float],
        config_defaults: Dict[str, Any],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Deep-copy transferred state
        self._parameter_state = [dict(row) for row in parameter_state]
        self._initial_parameter_snapshot = [dict(row) for row in initial_parameter_snapshot]
        self._global_dataset_params = {k: dict(v) for k, v in global_dataset_params.items()}
        self._global_dataset_variable_params = {
            k: {kk: dict(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in global_dataset_variable_params.items()
        }
        self._fixed_shared_params = dict(fixed_shared_params)
        self._shared_param_definitions = {k: dict(v) for k, v in shared_param_definitions.items()}
        self._mechanism_species = list(mechanism_species)
        self._dataset_entries = list(dataset_entries)
        self._prepared_param_names = list(prepared_param_names)
        self._last_fit_params: Dict[str, float] = {}
        self._staged_dataset_params: Dict[str, Dict[str, float]] = {}
        # Callable getters
        self._selected_dataset_ids_getter = selected_dataset_ids_getter
        self._dataset_entries_getter = dataset_entries_getter
        self._worker_running_getter = worker_running_getter
        self._dataset_manager_getter = dataset_manager_getter
        self._reactions_text_getter = reactions_text_getter
        # IC editor state
        self._ic_editor_dirty = False
        self._ic_editor_current_dataset_id: Optional[str] = None
        self._ic_editor_is_refreshing = False
        # Build UI
        self._build_ui(integration_defaults)
        self._apply_config_defaults(config_defaults)
        self._populate_parameter_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, integration_defaults: Tuple[str, float, float]) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(Qt.Horizontal, self)
        # --- Parameters panel (left) ---
        params_widget = self._build_parameters_panel(integration_defaults)
        # --- IC panel (right) ---
        ic_panel = self._build_initial_conditions_panel()
        ic_panel.setMinimumWidth(300)
        splitter.addWidget(params_widget)
        splitter.addWidget(ic_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 560])
        outer.addWidget(splitter, stretch=1)

    def _build_parameters_panel(self, integration_defaults: Tuple[str, float, float]) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(widget)
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll, stretch=1)

        container = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        scroll.setWidget(container)

        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        self._param_table = QtWidgets.QTableWidget()
        self._param_table.setColumnCount(7)
        self._param_table.setHorizontalHeaderLabels(["Fit", "Log10", "Name", "Value", "Min", "Max", "Last Fit"])
        _ph = self._param_table.horizontalHeader()
        _ph.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        _ph.setStretchLastSection(True)
        self._param_table.itemChanged.connect(self._on_param_table_item_changed)
        self._param_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._param_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        params_layout.addWidget(self._param_table)

        action_row = QtWidgets.QHBoxLayout()
        self._add_param_button = QtWidgets.QPushButton("Add…")
        self._add_param_button.clicked.connect(self._add_parameter)
        self._remove_param_button = QtWidgets.QPushButton("Remove")
        self._remove_param_button.clicked.connect(self._remove_selected_parameters)
        self._remove_param_button.setEnabled(False)
        self._param_table.itemSelectionChanged.connect(self._update_remove_button_state)
        action_row.addWidget(self._add_param_button)
        action_row.addWidget(self._remove_param_button)
        action_row.addStretch()
        params_layout.addLayout(action_row)

        reset_row = QtWidgets.QHBoxLayout()
        self._reset_initial_button = QtWidgets.QPushButton("Reset to Initial")
        self._reset_initial_button.clicked.connect(self._reset_to_initial)
        self._reset_last_button = QtWidgets.QPushButton("Reset to Last Fit")
        self._reset_last_button.clicked.connect(self._reset_to_last_fit)
        reset_row.addWidget(self._reset_initial_button)
        reset_row.addWidget(self._reset_last_button)
        reset_row.addStretch()
        params_layout.addLayout(reset_row)

        algo_form = QtWidgets.QFormLayout()
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["lm", "trf", "dogbox", "differential_evolution"])
        self._method_combo.setCurrentText("trf")
        self._max_eval_spin = QtWidgets.QSpinBox()
        self._max_eval_spin.setRange(10, 10000)
        self._max_eval_spin.setValue(1000)
        algo_form.addRow("Method:", self._method_combo)
        algo_form.addRow("Max evaluations:", self._max_eval_spin)

        self._ftol_edit = QtWidgets.QLineEdit("1e-10")
        setup_scientific_validator(self._ftol_edit)
        algo_form.addRow("ftol:", self._ftol_edit)

        self._xtol_edit = QtWidgets.QLineEdit("1e-10")
        setup_scientific_validator(self._xtol_edit)
        algo_form.addRow("xtol:", self._xtol_edit)

        self._use_parallel_check = QtWidgets.QCheckBox("Parallel multi-start (DE only)")
        self._seed_check = QtWidgets.QCheckBox("Use fixed random seed")
        self._seed_check.setChecked(True)
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 999_999)
        self._seed_spin.setValue(42)
        self._seed_spin.setEnabled(self._seed_check.isChecked())
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        algo_form.addRow(self._use_parallel_check)
        algo_form.addRow(self._seed_check, self._seed_spin)
        params_layout.addLayout(algo_form)

        integration_section = CollapsibleSection("Advanced Integration Settings", parent=params_group)
        integration_section.setObjectName("global_fit_advanced_integration_section")
        integration_section.set_collapsed(True)
        integration_widget = QtWidgets.QWidget(integration_section)
        integration_form = QtWidgets.QFormLayout(integration_widget)

        default_solver, default_rtol, default_atol = integration_defaults

        def _fmt(value: float) -> str:
            text = f"{float(value):g}"
            if "e" not in text:
                return text
            base, exp = text.split("e", 1)
            sign = ""
            digits = exp
            if digits and digits[0] in "+-":
                sign = digits[0]
                digits = digits[1:]
            digits = digits.lstrip("0") or "0"
            return f"{base}e{sign}{digits}"

        self._integration_solver_combo = QtWidgets.QComboBox(integration_widget)
        self._integration_solver_combo.setObjectName("global_fit_integration_solver")
        self._integration_solver_combo.addItems(["LSODA", "Radau", "BDF"])
        self._integration_solver_combo.setCurrentText(str(default_solver))

        self._integration_rtol_edit = QtWidgets.QLineEdit(_fmt(default_rtol), integration_widget)
        self._integration_rtol_edit.setObjectName("global_fit_integration_rtol")

        self._integration_atol_edit = QtWidgets.QLineEdit(_fmt(default_atol), integration_widget)
        self._integration_atol_edit.setObjectName("global_fit_integration_atol")

        setup_scientific_validator(self._integration_rtol_edit)
        setup_scientific_validator(self._integration_atol_edit)

        integration_form.addRow("Solver:", self._integration_solver_combo)
        integration_form.addRow("rtol:", self._integration_rtol_edit)
        integration_form.addRow("atol:", self._integration_atol_edit)
        integration_section.set_content_widget(integration_widget)
        params_layout.addWidget(integration_section)

        layout.addWidget(params_group, stretch=1)
        return widget

    def _build_initial_conditions_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Initial Conditions")
        group.setObjectName("global_fit_initial_conditions_panel")
        layout = QtWidgets.QVBoxLayout(group)

        self._ic_footer = ConfigPanelFooter(
            group,
            show_dirty=True,
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            apply_object_name="global_fit_initial_conditions_apply",
            revert_object_name="global_fit_initial_conditions_revert",
        )
        layout.addWidget(self._ic_footer, stretch=1)
        self._ic_footer.applyRequested.connect(self._apply_initial_conditions_changes)
        self._ic_footer.revertRequested.connect(self._revert_initial_conditions_changes)

        self._ic_dataset_combo = QtWidgets.QComboBox(group)
        self._ic_dataset_combo.setObjectName("global_fit_initial_conditions_dataset_combo")
        self._ic_footer.body_layout.addWidget(self._ic_dataset_combo)

        self._ic_table = QtWidgets.QTableWidget(group)
        self._ic_table.setObjectName("global_fit_initial_conditions_table")
        self._ic_table.setColumnCount(6)
        self._ic_table.setHorizontalHeaderLabels(["Fit", "Log10", "Species", "Initial", "Min", "Max"])
        _ih = self._ic_table.horizontalHeader()
        _ih.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        _ih.setStretchLastSection(True)
        self._ic_table.verticalHeader().setVisible(False)
        self._ic_table.setAlternatingRowColors(True)
        self._ic_table.setMinimumHeight(200)
        self._ic_table.itemChanged.connect(self._on_ic_table_item_changed)
        self._ic_footer.body_layout.addWidget(self._ic_table, stretch=1)

        self._ic_dataset_combo.currentIndexChanged.connect(self._load_initial_conditions_for_current_dataset)
        self._refresh_initial_conditions_dataset_combo_items()
        return group

    # ------------------------------------------------------------------
    # Config defaults
    # ------------------------------------------------------------------

    def _apply_config_defaults(self, defaults: Dict[str, Any]) -> None:
        if not defaults:
            return

        def _fmt_sci(value: float) -> str:
            text = str(float(value))
            if "e" not in text:
                return text
            base, exp = text.split("e", 1)
            sign = ""
            digits = exp
            if digits and digits[0] in "+-":
                sign = digits[0]
                digits = digits[1:]
            digits = digits.lstrip("0") or "0"
            return f"{base}e{sign}{digits}"

        method = str(defaults.get("method", "")).strip().lower()
        if method in {"lm", "trf", "dogbox", "differential_evolution"}:
            self._method_combo.setCurrentText(method)

        if "max_nfev" in defaults:
            try:
                value = int(defaults.get("max_nfev"))
            except (TypeError, ValueError):
                value = None
            if value is not None:
                value = max(self._max_eval_spin.minimum(), min(self._max_eval_spin.maximum(), value))
                self._max_eval_spin.setValue(value)

        for key, widget in (
            ("ftol", getattr(self, "_ftol_edit", None)),
            ("xtol", getattr(self, "_xtol_edit", None)),
        ):
            if key not in defaults or widget is None:
                continue
            try:
                value = float(defaults.get(key))
            except (TypeError, ValueError):
                value = None
            if value is None:
                continue
            if value > 0.0:
                widget.setText(_fmt_sci(value))

        if "use_parallel" in defaults:
            self._use_parallel_check.setChecked(bool(defaults.get("use_parallel")))

        if "use_seed" in defaults:
            self._seed_check.setChecked(bool(defaults.get("use_seed")))
        if "seed" in defaults:
            try:
                seed_val = int(defaults.get("seed"))
            except (TypeError, ValueError):
                seed_val = None
            if seed_val is not None:
                seed_val = max(self._seed_spin.minimum(), min(self._seed_spin.maximum(), seed_val))
                self._seed_spin.setValue(seed_val)
        self._seed_spin.setEnabled(self._seed_check.isChecked())

    # ------------------------------------------------------------------
    # IC editor
    # ------------------------------------------------------------------

    def _set_ic_editor_dirty_state(self, dirty: bool) -> None:
        self._ic_editor_dirty = bool(dirty)
        if hasattr(self, "_ic_footer"):
            self._ic_footer.set_dirty(self._ic_editor_dirty)

    def _on_ic_table_item_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if self._ic_editor_is_refreshing:
            return
        self._set_ic_editor_dirty_state(True)

    def _load_initial_conditions_for_current_dataset(self) -> None:
        if not hasattr(self, "_ic_dataset_combo"):
            return
        if self._ic_editor_dirty:
            # No modal prompts; discard pending edits when switching datasets.
            self._set_ic_editor_dirty_state(False)
        ds_id = str(self._ic_dataset_combo.currentData() or "").strip()
        self._ic_editor_current_dataset_id = ds_id or None
        self._populate_initial_conditions_table(ds_id)

    def _populate_initial_conditions_table(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        self._ic_editor_is_refreshing = True
        try:
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(None)

            if not self._mechanism_species:
                self._ic_table.setRowCount(0)
                self._ic_table.setEnabled(False)
                return

            settings = None
            dataset_manager = self._dataset_manager_getter()
            if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings") and ds_id:
                try:
                    settings = dataset_manager.get_fit_settings(ds_id)
                except Exception:
                    settings = None

            initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
            fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
            log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
            bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

            self._ic_table.setEnabled(True)
            self._ic_table.setRowCount(len(self._mechanism_species))
            for row, species in enumerate(self._mechanism_species):
                init_val = float(initials.get(species, 0.0))
                fit_flag = bool(fit_flags.get(species, False))
                log10_flag = bool(log10_flags.get(species, False))
                bounds = bounds_map.get(species)
                if not bounds:
                    bounds = (0.0, max(10.0, init_val * 10 or 10.0))
                try:
                    min_val = float(bounds[0])
                    max_val = float(bounds[1])
                except Exception:
                    min_val, max_val = (0.0, max(10.0, init_val * 10 or 10.0))

                species_item = QtWidgets.QTableWidgetItem(str(species))
                species_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ic_table.setItem(row, _ICCol.SPECIES, species_item)

                fit_item = QtWidgets.QTableWidgetItem()
                fit_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                fit_item.setCheckState(Qt.Checked if fit_flag else Qt.Unchecked)
                self._ic_table.setItem(row, _ICCol.FIT, fit_item)

                log_item = QtWidgets.QTableWidgetItem()
                log_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                log_item.setCheckState(Qt.Checked if log10_flag else Qt.Unchecked)
                self._ic_table.setItem(row, _ICCol.LOG10, log_item)

                init_item = QtWidgets.QTableWidgetItem(f"{init_val:.6g}")
                self._ic_table.setItem(row, _ICCol.INITIAL, init_item)

                min_item = QtWidgets.QTableWidgetItem(f"{min_val:.6g}")
                max_item = QtWidgets.QTableWidgetItem(f"{max_val:.6g}")
                self._ic_table.setItem(row, _ICCol.MIN, min_item)
                self._ic_table.setItem(row, _ICCol.MAX, max_item)
        finally:
            self._ic_editor_is_refreshing = False

    def _collect_initial_conditions_from_table(
        self,
    ) -> Tuple[
        Optional[Dict[str, Dict[str, object]]],
        Optional[Dict[str, bool]],
        Optional[str],
    ]:
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        if not ds_id:
            return None, None, "No dataset selected."
        if not self._mechanism_species:
            return None, None, "No mechanism species available."
        updates: Dict[str, Dict[str, object]] = {}
        fit_flags_updates: Dict[str, bool] = {}
        for row, species in enumerate(self._mechanism_species):
            init_item = self._ic_table.item(row, _ICCol.INITIAL)
            fit_item = self._ic_table.item(row, _ICCol.FIT)
            log_item = self._ic_table.item(row, _ICCol.LOG10)
            min_item = self._ic_table.item(row, _ICCol.MIN)
            max_item = self._ic_table.item(row, _ICCol.MAX)
            try:
                init_val = float(init_item.text())
            except Exception:
                return (
                    None,
                    None,
                    f"Species '{species}' requires a numeric initial concentration.",
                )
            fit_flag = bool(fit_item and fit_item.checkState() == Qt.Checked)
            log10_flag = bool(log_item and log_item.checkState() == Qt.Checked)
            try:
                min_val = float(min_item.text())
                max_val = float(max_item.text())
            except Exception:
                return None, None, f"Species '{species}' requires numeric bounds."
            if fit_flag and not (min_val < max_val):
                return None, None, f"Species '{species}' bounds must satisfy min < max."
            if fit_flag and log10_flag:
                if not (init_val > 0.0 and min_val > 0.0 and max_val > 0.0):
                    return (
                        None,
                        None,
                        f"Species '{species}' requires initial/min/max > 0 when Log10 is enabled.",
                    )
            updates[str(species)] = {
                "initial": float(init_val),
                "log10": bool(log10_flag),
                "min": float(min_val),
                "max": float(max_val),
            }
            fit_flags_updates[str(species)] = bool(fit_flag)
        return updates, fit_flags_updates, None

    def _apply_initial_conditions_changes(self) -> None:
        updates, fit_flags_updates, error = self._collect_initial_conditions_from_table()
        if error:
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(str(error))
            return
        assert updates is not None
        assert fit_flags_updates is not None
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        if not ds_id:
            return
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is None or not hasattr(dataset_manager, "get_fit_settings") or not hasattr(dataset_manager, "update_fit_settings"):
            message = "Dataset manager unavailable; cannot persist Initial Conditions."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return
        try:
            settings = dataset_manager.get_fit_settings(ds_id)
        except Exception:
            message = f"Failed to load fit settings for dataset {ds_id}."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return

        initials = dict(getattr(settings, "initial_conditions", {}) or {})
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {})
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {})
        bounds_map = dict(getattr(settings, "bounds", {}) or {})
        for species, spec in updates.items():
            species_key = str(species)
            initials[str(species)] = float(spec["initial"])
            fit_flags[species_key] = bool(fit_flags_updates.get(species_key, False))
            log10_flags[str(species)] = bool(spec["log10"])
            bounds_map[str(species)] = (float(spec["min"]), float(spec["max"]))

        settings.initial_conditions = initials
        settings.fit_flags = fit_flags
        settings.log10_flags = log10_flags
        settings.bounds = bounds_map
        try:
            dataset_manager.update_fit_settings(ds_id, settings)
        except Exception:
            message = f"Failed to persist fit settings for dataset {ds_id}."
            if hasattr(self, "_ic_footer"):
                self._ic_footer.set_error(message)
            return

        self._apply_ic_updates_to_window_state(ds_id, updates, fit_flags_updates)
        self._populate_parameter_table()
        self._set_ic_editor_dirty_state(False)
        self.statusMessage.emit("Initial conditions applied")

    def _apply_ic_updates_to_window_state(
        self,
        dataset_id: str,
        updates: Dict[str, Dict[str, object]],
        fit_flags_updates: Dict[str, bool],
    ) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        fixed = self._global_dataset_params.setdefault(ds_id, {})
        specs = self._global_dataset_variable_params.get(ds_id) if isinstance(self._global_dataset_variable_params, dict) else None
        if not isinstance(specs, dict):
            specs = {}
            self._global_dataset_variable_params[ds_id] = specs

        # Drop existing dataset-local init:* parameter rows.
        self._parameter_state = [
            row
            for row in self._parameter_state
            if not (
                str(row.get("scope") or "") == "dataset"
                and str(row.get("dataset_id") or "") == ds_id
                and str(row.get("param_name") or "").startswith(INITIAL_PREFIX)
            )
        ]

        for species, spec in updates.items():
            param_name = f"{INITIAL_PREFIX}{species}"
            init_val = float(spec["initial"])
            fit_flag = bool(fit_flags_updates.get(str(species), False))
            log10_flag = bool(spec["log10"])
            min_val = float(spec["min"])
            max_val = float(spec["max"])

            if fit_flag:
                specs[param_name] = {"initial": init_val, "min": min_val, "max": max_val, "log10": bool(log10_flag)}
                fixed.pop(param_name, None)
                display = f"{param_name} ({ds_id})"
                self._parameter_state.append(
                    {
                        "scope": "dataset",
                        "name": display,
                        "param_name": param_name,
                        "dataset_id": ds_id,
                        "value": init_val,
                        "min": min_val,
                        "max": max_val,
                        "fit": True,
                        "log10": bool(log10_flag),
                        "last_fit": None,
                    }
                )
            else:
                fixed[param_name] = init_val
                specs.pop(param_name, None)

        if not specs:
            self._global_dataset_variable_params.pop(ds_id, None)

    def _revert_initial_conditions_changes(self) -> None:
        ds_id = str(self._ic_editor_current_dataset_id or "").strip()
        self._set_ic_editor_dirty_state(False)
        self._populate_initial_conditions_table(ds_id)

    def _refresh_initial_conditions_dataset_combo_items(self) -> None:
        if not hasattr(self, "_ic_dataset_combo"):
            return
        combo = self._ic_dataset_combo
        current = str(combo.currentData() or "").strip()
        combo.blockSignals(True)
        try:
            combo.clear()
            for entry in self._dataset_entries:
                ds_id = str(entry.get("id") or "").strip()
                if not ds_id:
                    continue
                label = str(entry.get("label") or ds_id)
                combo.addItem(label, ds_id)
        finally:
            combo.blockSignals(False)
        if current:
            for i in range(combo.count()):
                if str(combo.itemData(i) or "").strip() == current:
                    combo.setCurrentIndex(i)
                    break
        if combo.count() and combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        self._load_initial_conditions_for_current_dataset()

    # ------------------------------------------------------------------
    # Parameter table
    # ------------------------------------------------------------------

    def _populate_parameter_table(self) -> None:
        self._param_table.blockSignals(True)
        self._param_table.setRowCount(len(self._parameter_state))
        for row, entry in enumerate(self._parameter_state):
            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            check_item.setCheckState(Qt.Checked if entry.get("fit", True) else Qt.Unchecked)
            self._param_table.setItem(row, 0, check_item)
            log_item = QtWidgets.QTableWidgetItem()
            log_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            log_item.setCheckState(Qt.Checked if entry.get("log10", False) else Qt.Unchecked)
            self._param_table.setItem(row, 1, log_item)
            name_item = QtWidgets.QTableWidgetItem(str(entry["name"]))
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._param_table.setItem(row, 2, name_item)
            self._param_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{entry['value']:.6g}"))
            self._param_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{entry['min']:.6g}"))
            self._param_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{entry['max']:.6g}"))
            last = entry.get("last_fit")
            display = "—" if last is None else f"{last:.6g}"
            last_item = QtWidgets.QTableWidgetItem(display)
            last_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._param_table.setItem(row, 6, last_item)
        self._param_table.blockSignals(False)
        self._update_remove_button_state()

    def _on_param_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """
        Keep `_parameter_state` as the source of truth for per-parameter flags.

        The table is repopulated during best-updates; without syncing, user toggles
        (Fit/Log10) would be reset on repaint.
        """
        try:
            row = int(item.row())
            col = int(item.column())
        except Exception:
            return
        if not (0 <= row < len(self._parameter_state)):
            return
        entry = self._parameter_state[row]
        if col == 0:
            fit_flag = item.checkState() == Qt.Checked
            entry["fit"] = bool(fit_flag)
            if entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                if ds_id and param_name:
                    if fit_flag:
                        self._global_dataset_variable_params.setdefault(ds_id, {})
                        self._global_dataset_variable_params[ds_id][param_name] = {
                            "initial": float(entry.get("value", 0.0)),
                            "min": float(entry.get("min", -np.inf)),
                            "max": float(entry.get("max", np.inf)),
                            "log10": bool(entry.get("log10", False)),
                        }
                        fixed_map = self._global_dataset_params.get(ds_id)
                        if isinstance(fixed_map, dict):
                            fixed_map.pop(param_name, None)
                    else:
                        self._global_dataset_params.setdefault(ds_id, {})[param_name] = float(entry.get("value", 0.0))
                        spec_map = self._global_dataset_variable_params.get(ds_id)
                        if isinstance(spec_map, dict):
                            spec_map.pop(param_name, None)
                            if not spec_map:
                                self._global_dataset_variable_params.pop(ds_id, None)
        elif col == 1:
            entry["log10"] = item.checkState() == Qt.Checked
            if entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                spec_map = self._global_dataset_variable_params.get(ds_id)
                if isinstance(spec_map, dict) and param_name in spec_map and isinstance(spec_map.get(param_name), dict):
                    spec_map[param_name]["log10"] = bool(entry["log10"])
        elif col in {3, 4, 5}:
            field = {3: "value", 4: "min", 5: "max"}.get(col)
            if not field:
                return
            raw = item.text()
            try:
                value = float(raw)
            except Exception:
                QtWidgets.QMessageBox.warning(self.window(), "Invalid Parameter", "Parameter values must be numeric.")
                self._populate_parameter_table()
                return
            entry[field] = value
            if entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                if entry.get("fit", True):
                    spec_map = self._global_dataset_variable_params.setdefault(ds_id, {})
                    spec = spec_map.setdefault(param_name, {})
                    if isinstance(spec, dict):
                        spec["initial"] = float(entry["value"])
                        spec["min"] = float(entry["min"])
                        spec["max"] = float(entry["max"])
                        spec["log10"] = bool(entry.get("log10", False))
                else:
                    self._global_dataset_params.setdefault(ds_id, {})[param_name] = float(entry.get("value", 0.0))

    def _update_remove_button_state(self) -> None:
        if not hasattr(self, "_remove_param_button"):
            return
        if self._worker_running_getter():
            self._remove_param_button.setEnabled(False)
            return
        rows = {item.row() for item in self._param_table.selectedItems()}
        self._remove_param_button.setEnabled(bool(rows))

    # ------------------------------------------------------------------
    # Add / remove parameters
    # ------------------------------------------------------------------

    def _add_parameter(self) -> None:
        if self._worker_running_getter():
            QtWidgets.QMessageBox.information(self.window(), "Fit Running", "Stop the current fit before editing parameters.")
            return
        dataset_ids = self._selected_dataset_ids_getter()
        if not dataset_ids:
            QtWidgets.QMessageBox.warning(self.window(), "No Datasets", "Select at least one dataset to include before adding parameters.")
            return
        available_observables: Dict[str, str] = {}
        if callable(getattr(self, "_reactions_text_getter", None)):
            try:
                from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text
                from kindred.core.simulator.algebra_section import extract_algebra_section_text

                reactions_text = str(self._reactions_text_getter() or "")
                algebra_text = extract_algebra_section_text(reactions_text)
                available_observables = extract_observables_from_algebra_text(algebra_text)
            except Exception:
                available_observables = {}
        dialog = _AddFittableParameterDialog(
            available_rates=self._available_rate_param_names(),
            available_scalars=self._available_scalar_param_names(),
            available_species=self._available_initial_species(dataset_ids),
            dataset_ids=dataset_ids,
            available_observables=available_observables,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        selection = dialog.selection() or {}
        if selection.get("type") == "rate":
            name = str(selection.get("name") or "").strip()
            if name:
                self._add_rate_parameter(name)
        elif selection.get("type") == "initial":
            species = str(selection.get("species") or "").strip()
            mode = str(selection.get("mode") or "").strip().lower()
            if not species:
                return
            if mode == "local":
                self._add_local_initial_parameter(species, dataset_ids)
            else:
                self._add_global_initial_parameter(species, dataset_ids)
        elif selection.get("type") == "scalar":
            name = str(selection.get("name") or "").strip()
            mode = str(selection.get("mode") or "shared").strip().lower()
            if not name:
                return
            if mode == "dataset":
                self._add_local_scalar_parameter(name, dataset_ids)
            else:
                self._add_shared_scalar_parameter(name, dataset_ids)
        elif selection.get("type") in {"observable_existing", "observable_new"}:
            request = dict(selection)
            request["dataset_ids"] = list(dataset_ids)
            request["persist"] = selection.get("type") == "observable_new"
            self.addAlgebraicObservableRequested.emit(request)
            return
        self._populate_parameter_table()

    def _add_rate_parameter(self, name: str) -> None:
        present = {
            str(entry.get("param_name") or "")
            for entry in self._parameter_state
            if entry.get("scope") == "shared"
        }
        if name in present:
            return
        if name in (self._fixed_shared_params or {}):
            value = float(self._fixed_shared_params.pop(name))
            definition = dict(getattr(self, "_shared_param_definitions", {}).get(name) or {})
            default_min = value * 0.1 if value else -10.0
            default_max = value * 10.0 if value else 10.0
            definition.setdefault("min", default_min)
            definition.setdefault("max", default_max)
            try:
                min_bound = float(definition["min"])
            except Exception:
                min_bound = default_min
                definition["min"] = float(min_bound)
            try:
                max_bound = float(definition["max"])
            except Exception:
                max_bound = default_max
                definition["max"] = float(max_bound)
        else:
            definition = dict(getattr(self, "_shared_param_definitions", {}).get(name) or {})
            definition.setdefault("value", 1.0)
            try:
                value = float(definition["value"])
            except Exception:
                value = 1.0
            definition["value"] = float(value)

            default_min = value * 0.1 if value else -10.0
            default_max = value * 10.0 if value else 10.0
            definition.setdefault("min", default_min)
            definition.setdefault("max", default_max)
            try:
                min_bound = float(definition["min"])
            except Exception:
                min_bound = default_min
                definition["min"] = float(min_bound)
            try:
                max_bound = float(definition["max"])
            except Exception:
                max_bound = default_max
                definition["max"] = float(max_bound)
        self._parameter_state.append(
            {
                "scope": "shared",
                "name": name,
                "param_name": name,
                "dataset_id": None,
                "value": value,
                "min": min_bound,
                "max": max_bound,
                "fit": True,
                "log10": False,
                "last_fit": None,
            }
        )

    def _global_init_param_present(self, param_name: str) -> bool:
        # Initial-condition parameters are fixed per-dataset (dataset_params), not as shared fixed params.
        # Treating "init:*" as present in `_fixed_shared_params` blocks re-adding after removal.
        if (not str(param_name).startswith(INITIAL_PREFIX)) and param_name in (self._fixed_shared_params or {}):
            return True
        for entry in self._parameter_state:
            if entry.get("scope") == "shared" and str(entry.get("param_name") or "") == str(param_name):
                return True
        return False

    def _add_global_initial_parameter(self, species: str, dataset_ids: Sequence[str]) -> None:
        param_name = f"{INITIAL_PREFIX}{species}"
        if self._global_init_param_present(param_name):
            QtWidgets.QMessageBox.information(self.window(), "Add Parameter", f"A global parameter for '{species}_0' already exists.")
            return
        value = None
        if param_name in (self._fixed_shared_params or {}):
            try:
                value = float(self._fixed_shared_params.pop(param_name))
            except Exception:
                value = None
        if value is None:
            ds0 = str(list(dataset_ids)[0])
            try:
                value = float((self._global_dataset_params.get(ds0) or {}).get(param_name))
            except Exception:
                value = 0.0
        if not np.isfinite(float(value)):
            value = 0.0
        min_bound = 0.0
        max_bound = max(10.0, float(value) * 10.0 if float(value) else 10.0)

        # Remove conflicting local specs for selected datasets to avoid overrides.
        remove_rows: List[int] = []
        for idx, entry in enumerate(self._parameter_state):
            if entry.get("scope") != "dataset":
                continue
            if str(entry.get("param_name") or "") != param_name:
                continue
            if str(entry.get("dataset_id") or "") not in {str(x) for x in dataset_ids}:
                continue
            remove_rows.append(idx)
        if remove_rows:
            self._remove_parameter_rows(remove_rows, update_fixed=False)

        for ds_id in dataset_ids:
            spec_map = self._global_dataset_variable_params.get(str(ds_id))
            if isinstance(spec_map, dict):
                spec_map.pop(param_name, None)
                if not spec_map:
                    self._global_dataset_variable_params.pop(str(ds_id), None)

        self._parameter_state.append(
            {
                "scope": "shared",
                "name": f"Global {species}_0",
                "param_name": param_name,
                "dataset_id": None,
                "value": float(value),
                "min": float(min_bound),
                "max": float(max_bound),
                "fit": True,
                "log10": False,
                "last_fit": None,
            }
        )

    def _add_local_initial_parameter(self, species: str, dataset_ids: Sequence[str]) -> None:
        param_name = f"{INITIAL_PREFIX}{species}"
        if self._global_init_param_present(param_name):
            QtWidgets.QMessageBox.warning(
                self.window(),
                "Add Parameter",
                f"Remove the global '{species}_0' parameter before adding local parameters.",
            )
            return
        present = {
            (str(entry.get("dataset_id") or ""), str(entry.get("param_name") or ""))
            for entry in self._parameter_state
            if entry.get("scope") == "dataset"
        }
        for ds_id in dataset_ids:
            ds_id = str(ds_id)
            if (ds_id, param_name) in present:
                continue
            fixed_map = self._global_dataset_params.get(ds_id) if isinstance(self._global_dataset_params, dict) else None
            try:
                value = float((fixed_map or {}).get(param_name, 0.0))
            except Exception:
                value = 0.0
            if not np.isfinite(float(value)):
                value = 0.0
            min_bound = 0.0
            max_bound = max(10.0, float(value) * 10.0 if float(value) else 10.0)
            self._global_dataset_variable_params.setdefault(ds_id, {})
            self._global_dataset_variable_params[ds_id][param_name] = {
                "initial": float(value),
                "min": float(min_bound),
                "max": float(max_bound),
                "log10": False,
            }
            self._parameter_state.append(
                {
                    "scope": "dataset",
                    "name": f"{species}_0 ({ds_id})",
                    "param_name": param_name,
                    "dataset_id": ds_id,
                    "value": float(value),
                    "min": float(min_bound),
                    "max": float(max_bound),
                    "fit": True,
                    "log10": False,
                    "last_fit": None,
                }
            )

    def _add_shared_scalar_parameter(self, name: str, dataset_ids: Sequence[str]) -> None:
        # Remove conflicting per-dataset entries for the selected datasets.
        remove_rows: List[int] = []
        for idx, entry in enumerate(self._parameter_state):
            if entry.get("scope") != "dataset":
                continue
            if str(entry.get("param_name") or "") != str(name):
                continue
            if str(entry.get("dataset_id") or "") not in {str(x) for x in dataset_ids}:
                continue
            remove_rows.append(idx)
        if remove_rows:
            self._remove_parameter_rows(remove_rows, update_fixed=False)

        for ds_id in dataset_ids:
            ds_id = str(ds_id)
            spec_map = self._global_dataset_variable_params.get(ds_id)
            if isinstance(spec_map, dict):
                spec_map.pop(str(name), None)
                if not spec_map:
                    self._global_dataset_variable_params.pop(ds_id, None)
            fixed_map = self._global_dataset_params.get(ds_id)
            if isinstance(fixed_map, dict):
                fixed_map.pop(str(name), None)
                if not fixed_map:
                    self._global_dataset_params.pop(ds_id, None)

        self._add_rate_parameter(str(name))

    def _add_local_scalar_parameter(self, name: str, dataset_ids: Sequence[str]) -> None:
        # Enforce exclusivity: do not allow a scalar to be both shared and per-dataset.
        if any(
            entry.get("scope") == "shared" and str(entry.get("param_name") or "") == str(name)
            for entry in self._parameter_state
        ):
            QtWidgets.QMessageBox.warning(
                self.window(),
                "Add Parameter",
                f"Remove the shared '{name}' parameter before adding per-dataset scalar parameters.",
            )
            return

        definition = dict(getattr(self, "_shared_param_definitions", {}).get(str(name)) or {})
        definition.setdefault("value", 1.0)
        try:
            base_value = float(definition["value"])
        except Exception:
            base_value = 1.0
        if str(name) in (self._fixed_shared_params or {}):
            raw_value = self._fixed_shared_params.pop(str(name))
            try:
                base_value = float(raw_value)
            except (TypeError, ValueError):
                self._fixed_shared_params[str(name)] = raw_value
        definition["value"] = float(base_value)

        default_min = base_value * 0.1 if base_value else -10.0
        default_max = base_value * 10.0 if base_value else 10.0
        definition.setdefault("min", default_min)
        definition.setdefault("max", default_max)
        try:
            min_bound = float(definition["min"])
        except Exception:
            min_bound = default_min
            definition["min"] = float(min_bound)
        try:
            max_bound = float(definition["max"])
        except Exception:
            max_bound = default_max
            definition["max"] = float(max_bound)

        present = {
            (str(entry.get("dataset_id") or ""), str(entry.get("param_name") or ""))
            for entry in self._parameter_state
            if entry.get("scope") == "dataset"
        }
        for ds_id in dataset_ids:
            ds_id = str(ds_id)
            key = (ds_id, str(name))
            if key in present:
                continue
            self._global_dataset_variable_params.setdefault(ds_id, {})
            self._global_dataset_variable_params[ds_id][str(name)] = {
                "initial": float(base_value),
                "min": float(min_bound),
                "max": float(max_bound),
                "log10": False,
            }
            fixed_map = self._global_dataset_params.get(ds_id)
            if isinstance(fixed_map, dict):
                fixed_map.pop(str(name), None)
                if not fixed_map:
                    self._global_dataset_params.pop(ds_id, None)
            self._parameter_state.append(
                {
                    "scope": "dataset",
                    "name": f"{name} ({ds_id})",
                    "param_name": str(name),
                    "dataset_id": ds_id,
                    "value": float(base_value),
                    "min": float(min_bound),
                    "max": float(max_bound),
                    "fit": True,
                    "log10": False,
                    "last_fit": None,
                }
            )

    def _remove_selected_parameters(self) -> None:
        rows = sorted({item.row() for item in self._param_table.selectedItems()})
        if not rows:
            return
        self._remove_parameter_rows(rows)
        self._populate_parameter_table()

    def _remove_parameter_rows(self, rows: Sequence[int], *, update_fixed: bool = True) -> None:
        for row in sorted({int(r) for r in (rows or []) if isinstance(r, int)}, reverse=True):
            if not (0 <= row < len(self._parameter_state)):
                continue
            entry = self._parameter_state[row]
            scope = str(entry.get("scope") or "")
            if scope == "shared":
                param_name = str(entry.get("param_name") or "")
                if update_fixed and param_name:
                    if param_name.startswith(INITIAL_PREFIX):
                        # Shared "init:*" is only used when explicitly fittable; when removed, fall back
                        # to per-dataset fixed initials and allow re-adding.
                        self._fixed_shared_params.pop(param_name, None)
                    else:
                        self._fixed_shared_params[param_name] = float(entry.get("value", 0.0))
                if update_fixed and param_name.startswith(INITIAL_PREFIX):
                    for ds_id in self._selected_dataset_ids_getter():
                        self._global_dataset_params.setdefault(ds_id, {})[param_name] = float(entry.get("value", 0.0))
            elif scope == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                if update_fixed and ds_id and param_name:
                    self._global_dataset_params.setdefault(ds_id, {})[param_name] = float(entry.get("value", 0.0))
                spec_map = self._global_dataset_variable_params.get(ds_id)
                if isinstance(spec_map, dict):
                    spec_map.pop(param_name, None)
                    if not spec_map:
                        self._global_dataset_variable_params.pop(ds_id, None)
            self._parameter_state.pop(row)

    def _available_rate_param_names(self) -> List[str]:
        def _is_scalar(name: str) -> bool:
            definition = dict(getattr(self, "_shared_param_definitions", {}).get(str(name)) or {})
            definition.setdefault("source", "")
            return str(definition["source"] or "").strip().lower() == "scalar parameter"

        present = {
            str(entry.get("param_name") or "")
            for entry in self._parameter_state
            if entry.get("scope") == "shared" and not str(entry.get("param_name") or "").startswith(INITIAL_PREFIX)
            and not _is_scalar(str(entry.get("param_name") or ""))
        }
        candidates = {
            name
            for name in (getattr(self, "_shared_param_definitions", {}) or {}).keys()
            if name and not str(name).startswith(INITIAL_PREFIX) and not _is_scalar(str(name))
        }
        candidates |= {
            str(k)
            for k in (self._fixed_shared_params or {}).keys()
            if str(k) and not str(k).startswith(INITIAL_PREFIX) and not _is_scalar(str(k))
        }
        remaining = sorted({name for name in candidates if name and name not in present})
        return remaining

    def _available_scalar_param_names(self) -> List[str]:
        def _is_scalar(name: str) -> bool:
            definition = dict(getattr(self, "_shared_param_definitions", {}).get(str(name)) or {})
            definition.setdefault("source", "")
            return str(definition["source"] or "").strip().lower() == "scalar parameter"

        present = {
            str(entry.get("param_name") or "")
            for entry in self._parameter_state
            if entry.get("scope") == "shared" and _is_scalar(str(entry.get("param_name") or ""))
        }
        candidates = {
            str(name)
            for name in (getattr(self, "_shared_param_definitions", {}) or {}).keys()
            if _is_scalar(str(name))
        }
        candidates |= {str(k) for k in (self._fixed_shared_params or {}).keys() if _is_scalar(str(k))}
        remaining = sorted({name for name in candidates if name and name not in present})
        return remaining

    def _available_initial_species(self, dataset_ids: Sequence[str]) -> List[str]:
        species: set[str] = set()
        allowed = {str(x) for x in (self._mechanism_species or []) if str(x).strip()}
        for ds_id in dataset_ids or []:
            fixed_map = self._global_dataset_params.get(str(ds_id), {}) if isinstance(self._global_dataset_params, dict) else {}
            if isinstance(fixed_map, dict):
                for key in fixed_map.keys():
                    k = str(key)
                    if k.startswith(INITIAL_PREFIX):
                        candidate = k[len(INITIAL_PREFIX):]
                        if not allowed or candidate in allowed:
                            species.add(candidate)
            var_map = self._global_dataset_variable_params.get(str(ds_id), {}) if isinstance(self._global_dataset_variable_params, dict) else {}
            if isinstance(var_map, dict):
                for key in var_map.keys():
                    k = str(key)
                    if k.startswith(INITIAL_PREFIX):
                        candidate = k[len(INITIAL_PREFIX):]
                        if not allowed or candidate in allowed:
                            species.add(candidate)

        if not species:
            if allowed:
                species |= allowed
            else:
                for entry in self._dataset_entries_getter():
                    for name in (entry.get("selected_species") or []):
                        if str(name).strip():
                            species.add(str(name))
        return sorted(species)

    def _auto_add_missing_scalars_as_parameters(
        self,
        *,
        missing_scalars: Sequence[str],
        dataset_ids: Sequence[str],
        scalar_scope: str,
    ) -> None:
        fallback_scalars: list[str] = []
        requested_scope = "dataset" if str(scalar_scope or "").lower().startswith("d") else "shared"
        selected_dataset_ids = {str(x) for x in dataset_ids}
        for scalar in missing_scalars:
            scalar_name = str(scalar)
            if requested_scope == "dataset":
                shared_present = any(
                    entry.get("scope") == "shared" and str(entry.get("param_name") or "") == scalar_name
                    for entry in (self._parameter_state or [])
                )
                if shared_present:
                    fallback_scalars.append(scalar_name)
                    self._add_shared_scalar_parameter(scalar_name, dataset_ids)
                else:
                    self._add_local_scalar_parameter(scalar_name, dataset_ids)
            else:
                self._add_shared_scalar_parameter(scalar_name, dataset_ids)

            for entry in self._parameter_state:
                if str(entry.get("param_name") or "") != scalar_name:
                    continue
                if entry.get("scope") == "shared":
                    entry["min"] = -np.inf
                    entry["max"] = np.inf
                elif entry.get("scope") == "dataset" and str(entry.get("dataset_id") or "") in selected_dataset_ids:
                    entry["min"] = -np.inf
                    entry["max"] = np.inf

            for ds_id in selected_dataset_ids:
                spec_map = self._global_dataset_variable_params.get(str(ds_id))
                if isinstance(spec_map, dict) and scalar_name in spec_map and isinstance(spec_map.get(scalar_name), dict):
                    spec_map[scalar_name]["min"] = -np.inf
                    spec_map[scalar_name]["max"] = np.inf

        if fallback_scalars:
            QtWidgets.QMessageBox.information(
                self.window(),
                "Algebraic Observables",
                "Some scalars could not be added as per-dataset (existing shared parameter). "
                f"Added as Shared instead: {', '.join(sorted(set(fallback_scalars)))}",
            )

    def _seed_dataset_initial_params_from_fit_settings(self, dataset_id: str) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is None or not hasattr(dataset_manager, "get_fit_settings"):
            return
        try:
            settings = dataset_manager.get_fit_settings(ds_id)
        except Exception:
            return

        fixed = self._global_dataset_params.setdefault(ds_id, {})
        var_specs = self._global_dataset_variable_params.get(ds_id) if isinstance(self._global_dataset_variable_params, dict) else None
        if not isinstance(var_specs, dict):
            var_specs = {}
            self._global_dataset_variable_params[ds_id] = var_specs

        for species in self._mechanism_species:
            key = f"{INITIAL_PREFIX}{species}"
            init_val = float((getattr(settings, "initial_conditions", {}) or {}).get(species, 0.0))
            fit_flag = bool((getattr(settings, "fit_flags", {}) or {}).get(species, False))
            log10_flag = bool((getattr(settings, "log10_flags", {}) or {}).get(species, False))
            bounds = (getattr(settings, "bounds", {}) or {}).get(species)
            if not bounds:
                bounds = (0.0, max(10.0, init_val * 10 or 10.0))
            try:
                min_val = float(bounds[0])
                max_val = float(bounds[1])
            except Exception:
                min_val, max_val = (0.0, max(10.0, init_val * 10 or 10.0))

            if fit_flag:
                var_specs[key] = {"initial": init_val, "min": min_val, "max": max_val, "log10": bool(log10_flag)}
                fixed.pop(key, None)
            else:
                fixed[key] = init_val
                var_specs.pop(key, None)

        if not var_specs:
            self._global_dataset_variable_params.pop(ds_id, None)

    # ------------------------------------------------------------------
    # Reset / config
    # ------------------------------------------------------------------

    def _reset_to_initial(self) -> None:
        self._parameter_state = [dict(row) for row in self._initial_parameter_snapshot]
        self._fixed_shared_params = {}
        self._populate_parameter_table()

    def _reset_to_last_fit(self) -> None:
        if not self._last_fit_params:
            return
        for entry in self._parameter_state:
            if entry.get("scope") == "shared":
                name = entry["param_name"]
                if name in self._last_fit_params:
                    entry["value"] = float(self._last_fit_params[name])
                    entry["fit"] = True
                    entry["last_fit"] = self._last_fit_params[name]
            elif entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                ds_map = self._staged_dataset_params.get(ds_id) if ds_id else None
                if isinstance(ds_map, dict) and param_name in ds_map:
                    entry["value"] = float(ds_map[param_name])
                    entry["last_fit"] = float(ds_map[param_name])
        self._populate_parameter_table()

    def _collect_parameter_config(self) -> Optional[Dict[str, Any]]:
        parameters: Dict[str, float] = {}
        bounds: Dict[str, Tuple[float, float]] = {}
        log10_params: Dict[str, bool] = {}
        fixed_params: Dict[str, float] = dict(self._fixed_shared_params or {})
        updated_state: List[Dict[str, Any]] = []
        for row in range(self._param_table.rowCount()):
            fit_flag = self._param_table.item(row, 0).checkState() == Qt.Checked
            entry = self._parameter_state[row]
            log10_flag = self._param_table.item(row, 1).checkState() == Qt.Checked
            param_name = str(entry.get("param_name") or "")
            try:
                value = float(self._param_table.item(row, 3).text())
                min_val = float(self._param_table.item(row, 4).text())
                max_val = float(self._param_table.item(row, 5).text())
            except (ValueError, AttributeError):
                QtWidgets.QMessageBox.warning(
                    self.window(),
                    "Invalid Parameter",
                    f"Parameter '{self._param_table.item(row, 2).text()}' contains non-numeric values.",
                )
                return None
            if not (min_val < max_val):
                QtWidgets.QMessageBox.warning(
                    self.window(),
                    "Invalid Bounds",
                    f"Parameter '{self._param_table.item(row, 2).text()}' bounds must satisfy min < max.",
                )
                return None
            if log10_flag:
                if not (value > 0.0 and min_val > 0.0 and max_val > 0.0):
                    QtWidgets.QMessageBox.warning(
                        self.window(),
                        "Invalid Log10 Bounds",
                        f"Parameter '{self._param_table.item(row, 2).text()}' requires value/min/max > 0 when Log10 is enabled.",
                    )
                    return None
            scope = str(entry.get("scope") or "shared")
            updated = dict(entry)
            updated["value"] = value
            updated["min"] = min_val
            updated["max"] = max_val
            updated["fit"] = bool(fit_flag)
            updated["log10"] = bool(log10_flag)
            updated_state.append(updated)
            if scope == "shared":
                if fit_flag:
                    parameters[param_name] = value
                    bounds[param_name] = (min_val, max_val)
                    log10_params[param_name] = bool(log10_flag)
                else:
                    fixed_params[param_name] = value
            elif scope == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                if ds_id and param_name:
                    if fit_flag:
                        spec_map = self._global_dataset_variable_params.setdefault(ds_id, {})
                        spec_map[param_name] = {
                            "initial": value,
                            "min": min_val,
                            "max": max_val,
                            "log10": bool(log10_flag),
                        }
                        fixed_map = self._global_dataset_params.get(ds_id)
                        if isinstance(fixed_map, dict):
                            fixed_map.pop(param_name, None)
                    else:
                        self._global_dataset_params.setdefault(ds_id, {})[param_name] = float(value)
                        spec_map = self._global_dataset_variable_params.get(ds_id)
                        if isinstance(spec_map, dict):
                            spec_map.pop(param_name, None)
                            if not spec_map:
                                self._global_dataset_variable_params.pop(ds_id, None)
        self._parameter_state = updated_state
        if not parameters and not any((entry.get("scope") == "dataset" and entry.get("fit", True)) for entry in self._parameter_state):
            QtWidgets.QMessageBox.warning(self.window(), "No Parameters", "Select at least one parameter to fit.")
            return None

        method = self._method_combo.currentText().strip().lower()
        config = {
            "parameters": parameters,
            "bounds": bounds,
            "log10_params": log10_params,
            "fixed_params": fixed_params,
            "method": method,
            "max_nfev": self._max_eval_spin.value(),
            "ftol": max(safe_float_parse(self._ftol_edit.text(), 1e-10), 1e-15),
            "xtol": max(safe_float_parse(self._xtol_edit.text(), 1e-10), 1e-15),
            "seed": self._seed_spin.value() if self._seed_check.isChecked() else None,
            "use_parallel": self._use_parallel_check.isChecked(),
            "parallel_starts": DEFAULT_PARALLEL_STARTS,
        }
        return config

    def _build_parameter_state(self, definitions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        state: List[Dict[str, Any]] = []
        self._shared_param_definitions: Dict[str, Dict[str, Any]] = {}
        missing_name_count = 0
        for definition in definitions or []:
            name_raw = definition.get("name")
            if name_raw is None:
                missing_name_count += 1
                continue
            name = str(name_raw)
            if not name.strip():
                continue
            value = float(definition.get("value", 1.0))
            min_bound = float(definition.get("min", value * 0.1 if value else -10.0))
            max_bound = float(definition.get("max", value * 10.0 if value else 10.0))
            self._shared_param_definitions[name] = dict(definition)
            state.append(
                {
                    "scope": "shared",
                    "name": name,
                    "param_name": name,
                    "dataset_id": None,
                    "value": value,
                    "min": min_bound,
                    "max": max_bound,
                    "fit": True,
                    "log10": False,
                    "last_fit": None,
                }
            )

        for ds_id, specs in (self._global_dataset_variable_params or {}).items():
            if not isinstance(specs, dict):
                continue
            for param_name, spec in specs.items():
                if not isinstance(spec, dict):
                    continue
                param_name = str(param_name)
                if not param_name.strip():
                    continue
                try:
                    init_val = float(spec.get("initial", 0.0))
                    min_val = float(spec.get("min", -np.inf))
                    max_val = float(spec.get("max", np.inf))
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "Skipping invalid global-fit dataset parameter spec '%s' for dataset '%s': %s",
                        param_name,
                        ds_id,
                        exc,
                        exc_info=True,
                    )
                    continue
                log10_flag = bool(spec.get("log10", False))
                if param_name.startswith(INITIAL_PREFIX):
                    species = param_name[len(INITIAL_PREFIX):]
                    display = f"{species}_0 ({ds_id})"
                else:
                    display = f"{param_name} ({ds_id})"
                state.append(
                    {
                        "scope": "dataset",
                        "name": display,
                        "param_name": param_name,
                        "dataset_id": str(ds_id),
                        "value": init_val,
                        "min": min_val,
                        "max": max_val,
                        "fit": True,
                        "log10": log10_flag,
                        "last_fit": None,
                    }
                )

        if not state:
            logger.warning("Fitting window opened with no parameter definitions.")
        if missing_name_count:
            logger.debug("Skipped %d global-fit parameter definitions without a 'name' field.", missing_name_count)
        return state

    def _collect_integration_settings_for_run(self) -> Optional[Tuple[str, float, float]]:
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

        allowed = ("LSODA", "Radau", "BDF")
        combo = getattr(self, "_integration_solver_combo", None)
        solver_label = str(combo.currentText()).strip() if combo is not None else str(DEFAULT_SOLVER_NAME)
        if solver_label not in allowed:
            solver_label = str(DEFAULT_SOLVER_NAME)
        solver_method, solver_warning = normalize_solver_name(solver_label)
        if solver_warning:
            QtWidgets.QMessageBox.information(
                self.window(),
                "Solver Normalization",
                f"{solver_warning}\n\nRequested: {solver_label}\nUsing: {solver_method}",
            )

        rtol_edit = getattr(self, "_integration_rtol_edit", None)
        atol_edit = getattr(self, "_integration_atol_edit", None)
        rtol_text = str(rtol_edit.text()).strip() if rtol_edit is not None else "1e-6"
        atol_text = str(atol_edit.text()).strip() if atol_edit is not None else "1e-12"
        if not rtol_text:
            rtol_text = "1e-6"
        if not atol_text:
            atol_text = "1e-12"

        try:
            rtol = float(rtol_text)
            atol = float(atol_text)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self.window(),
                "Advanced Integration Settings",
                "rtol and atol must be valid floating-point numbers (scientific notation is allowed).",
            )
            return None
        if not (np.isfinite(rtol) and rtol > 0.0):
            QtWidgets.QMessageBox.warning(self.window(), "Advanced Integration Settings", "rtol must be a finite value > 0.")
            return None
        if not (np.isfinite(atol) and atol > 0.0):
            QtWidgets.QMessageBox.warning(self.window(), "Advanced Integration Settings", "atol must be a finite value > 0.")
            return None
        return str(solver_method), float(rtol), float(atol)

    # ------------------------------------------------------------------
    # Mechanism rebuild
    # ------------------------------------------------------------------

    def _scan_parameter_definitions_for_mechanism(self, mechanism_text: str, dataset_manager: Any = None) -> list[dict[str, Any]]:
        if dataset_manager is None:
            dataset_manager = self._dataset_manager_getter()
        scan_params = getattr(dataset_manager, "scan_mechanism_parameters", None)
        if not callable(scan_params):
            return [
                dict(definition)
                for definition in (getattr(self, "_shared_param_definitions", {}) or {}).values()
                if isinstance(definition, dict)
            ]
        param_defs = scan_params(str(mechanism_text or ""))
        return [dict(definition) for definition in (param_defs or []) if isinstance(definition, dict)]

    def _mechanism_species_for_text(self, mechanism_text: str) -> list[str]:
        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism

            mechanism = parse_dsl_to_mechanism(str(mechanism_text or ""), initials={})
        except Exception:
            logger.debug("Failed to parse mechanism species for fitting-window refresh.", exc_info=True)
            return [str(name) for name in (getattr(self, "_mechanism_species", []) or []) if str(name).strip()]
        species_map = getattr(mechanism, "species", {}) or {}
        species_names = [str(name) for name in species_map.keys() if str(name).strip()]
        return list(dict.fromkeys(species_names))

    def _initial_parameter_defaults_for_species(self, dataset_id: str, species: str) -> tuple[bool, dict[str, float]]:
        settings = None
        dataset_manager = self._dataset_manager_getter()
        if dataset_manager is not None and hasattr(dataset_manager, "get_fit_settings"):
            try:
                settings = dataset_manager.get_fit_settings(str(dataset_id))
            except Exception:
                settings = None
        initials = dict(getattr(settings, "initial_conditions", {}) or {}) if settings is not None else {}
        fit_flags = dict(getattr(settings, "fit_flags", {}) or {}) if settings is not None else {}
        log10_flags = dict(getattr(settings, "log10_flags", {}) or {}) if settings is not None else {}
        bounds_map = dict(getattr(settings, "bounds", {}) or {}) if settings is not None else {}

        init_val = float(initials.get(species, 0.0))
        fit_flag = bool(fit_flags.get(species, False))
        log10_flag = bool(log10_flags.get(species, False))
        bounds = bounds_map.get(species)
        if not bounds:
            bounds = (0.0, max(10.0, init_val * 10 or 10.0))
        try:
            min_val = float(bounds[0])
            max_val = float(bounds[1])
        except Exception:
            min_val = 0.0
            max_val = max(10.0, init_val * 10 or 10.0)
        return fit_flag, {
            "initial": float(init_val),
            "min": float(min_val),
            "max": float(max_val),
            "log10": bool(log10_flag),
        }

    @staticmethod
    def _coerce_variable_spec(spec: dict[str, Any]) -> Optional[dict[str, float]]:
        if not isinstance(spec, dict):
            return None
        try:
            initial = float(spec.get("initial", 0.0))
            min_val = float(spec.get("min", -np.inf))
            max_val = float(spec.get("max", np.inf))
        except (TypeError, ValueError):
            return None
        return {
            "initial": float(initial),
            "min": float(min_val),
            "max": float(max_val),
            "log10": bool(spec.get("log10", False)),
        }

    def rebuild_for_mechanism(self, mechanism_text: str, dataset_entries: List[Dict[str, Any]]) -> list[str]:
        param_defs = self._scan_parameter_definitions_for_mechanism(mechanism_text)
        param_names = [str(d.get("name")) for d in (param_defs or []) if d.get("name")]
        self._shared_param_definitions = {str(d.get("name")): dict(d) for d in (param_defs or []) if d.get("name")}
        self._prepared_param_names = list(param_names)

        mechanism_species = self._mechanism_species_for_text(mechanism_text)
        allowed_species = {str(name) for name in mechanism_species if str(name).strip()}
        allowed_param_names = {str(name) for name in param_names if str(name).strip()}

        old_state = [dict(row) for row in (getattr(self, "_parameter_state", []) or []) if isinstance(row, dict)]
        old_shared_rows = {
            str(row.get("param_name") or ""): dict(row)
            for row in old_state
            if str(row.get("scope") or "") == "shared" and str(row.get("param_name") or "").strip()
        }
        old_dataset_rows = {
            (str(row.get("dataset_id") or ""), str(row.get("param_name") or "")): dict(row)
            for row in old_state
            if str(row.get("scope") or "") == "dataset"
            and str(row.get("dataset_id") or "").strip()
            and str(row.get("param_name") or "").strip()
        }

        filtered_fixed_shared: Dict[str, float] = {}
        for name, value in (getattr(self, "_fixed_shared_params", {}) or {}).items():
            key = str(name or "").strip()
            if not key or key not in allowed_param_names:
                continue
            try:
                filtered_fixed_shared[key] = float(value)
            except (TypeError, ValueError):
                continue

        current_fixed = getattr(self, "_global_dataset_params", {}) or {}
        current_variable = getattr(self, "_global_dataset_variable_params", {}) or {}
        refreshed_fixed: Dict[str, Dict[str, float]] = {}
        refreshed_variable: Dict[str, Dict[str, Dict[str, float]]] = {}
        for entry in dataset_entries or []:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            old_fixed = current_fixed.get(ds_id, {}) if isinstance(current_fixed.get(ds_id), dict) else {}
            old_variable = current_variable.get(ds_id, {}) if isinstance(current_variable.get(ds_id), dict) else {}
            fixed_map: Dict[str, float] = {}
            variable_map: Dict[str, Dict[str, float]] = {}

            for key, value in old_fixed.items():
                param_name = str(key or "").strip()
                if not param_name:
                    continue
                if param_name.startswith(INITIAL_PREFIX):
                    species = param_name[len(INITIAL_PREFIX):]
                    if species not in allowed_species:
                        continue
                elif param_name not in allowed_param_names:
                    continue
                try:
                    fixed_map[param_name] = float(value)
                except (TypeError, ValueError):
                    continue

            for key, spec in old_variable.items():
                param_name = str(key or "").strip()
                if not param_name:
                    continue
                if param_name.startswith(INITIAL_PREFIX):
                    species = param_name[len(INITIAL_PREFIX):]
                    if species not in allowed_species:
                        continue
                elif param_name not in allowed_param_names:
                    continue
                cleaned = self._coerce_variable_spec(spec)
                if cleaned is None:
                    continue
                variable_map[param_name] = cleaned

            for species in mechanism_species:
                param_name = f"{INITIAL_PREFIX}{species}"
                if param_name in fixed_map or param_name in variable_map:
                    continue
                fit_flag, default_spec = self._initial_parameter_defaults_for_species(ds_id, species)
                if fit_flag:
                    variable_map[param_name] = dict(default_spec)
                else:
                    fixed_map[param_name] = float(default_spec["initial"])

            if fixed_map:
                refreshed_fixed[ds_id] = fixed_map
            if variable_map:
                refreshed_variable[ds_id] = variable_map

        self._global_dataset_params = refreshed_fixed
        self._global_dataset_variable_params = refreshed_variable
        self._fixed_shared_params = filtered_fixed_shared
        self._mechanism_species = list(mechanism_species)

        rebuilt_state = self._build_parameter_state(param_defs)
        merged_state: List[Dict[str, Any]] = []
        merged_shared_names: set[str] = set()
        merged_dataset_keys: set[tuple[str, str]] = set()
        for row in rebuilt_state:
            scope = str(row.get("scope") or "")
            if scope == "shared":
                param_name = str(row.get("param_name") or "")
                if param_name in self._fixed_shared_params:
                    continue
                previous = old_shared_rows.get(param_name)
                if previous is not None:
                    merged = dict(row)
                    for field in ("value", "min", "max"):
                        try:
                            merged[field] = float(previous.get(field, row.get(field)))
                        except (TypeError, ValueError):
                            merged[field] = float(row.get(field, 0.0))
                    merged["fit"] = bool(previous.get("fit", True))
                    merged["log10"] = bool(previous.get("log10", False))
                    merged["last_fit"] = previous.get("last_fit")
                    row = merged
                merged_shared_names.add(param_name)
            elif scope == "dataset":
                ds_key = (str(row.get("dataset_id") or ""), str(row.get("param_name") or ""))
                previous = old_dataset_rows.get(ds_key)
                if previous is not None:
                    row = dict(row)
                    row["last_fit"] = previous.get("last_fit")
                merged_dataset_keys.add(ds_key)
            merged_state.append(row)

        for param_name, previous in old_shared_rows.items():
            if not param_name.startswith(INITIAL_PREFIX):
                continue
            if param_name in merged_shared_names:
                continue
            species = param_name[len(INITIAL_PREFIX):]
            if species not in allowed_species:
                continue
            restored = dict(previous)
            restored["scope"] = "shared"
            restored["param_name"] = str(param_name)
            restored["dataset_id"] = None
            restored["fit"] = bool(previous.get("fit", True))
            restored["log10"] = bool(previous.get("log10", False))
            restored["name"] = str(previous.get("name") or f"Global {species}_0")
            try:
                shared_value = float(previous.get("value", 0.0))
            except (TypeError, ValueError):
                shared_value = 0.0
            for field, fallback in (
                ("value", shared_value),
                ("min", 0.0),
                ("max", max(10.0, shared_value * 10.0 if shared_value else 10.0)),
            ):
                try:
                    restored[field] = float(previous.get(field, fallback))
                except (TypeError, ValueError):
                    restored[field] = float(fallback)
            merged_state.append(restored)
            merged_shared_names.add(param_name)

        for (ds_id, param_name), previous in old_dataset_rows.items():
            if (ds_id, param_name) in merged_dataset_keys:
                continue
            fixed_map = refreshed_fixed.get(str(ds_id), {})
            if not isinstance(fixed_map, dict) or param_name not in fixed_map:
                continue
            restored = dict(previous)
            restored["scope"] = "dataset"
            restored["dataset_id"] = str(ds_id)
            restored["param_name"] = str(param_name)
            restored["fit"] = False
            restored["value"] = float(fixed_map[param_name])
            restored["log10"] = bool(previous.get("log10", False))
            if str(param_name).startswith(INITIAL_PREFIX):
                species = str(param_name)[len(INITIAL_PREFIX):]
                restored["name"] = str(previous.get("name") or f"{species}_0 ({ds_id})")
            else:
                restored["name"] = str(previous.get("name") or f"{param_name} ({ds_id})")
            for field, fallback in (("min", restored["value"] * 0.1 if restored["value"] else -10.0), ("max", restored["value"] * 10.0 if restored["value"] else 10.0)):
                try:
                    restored[field] = float(previous.get(field, fallback))
                except (TypeError, ValueError):
                    restored[field] = float(fallback)
            merged_state.append(restored)
            merged_dataset_keys.add((str(ds_id), str(param_name)))

        self._parameter_state = merged_state
        self._initial_parameter_snapshot = [dict(row) for row in self._parameter_state]
        self._dataset_entries = list(dataset_entries)
        self._populate_parameter_table()
        self._refresh_initial_conditions_dataset_combo_items()
        return list(self._prepared_param_names)

    # ------------------------------------------------------------------
    # Public API — state getters (return copies)
    # ------------------------------------------------------------------

    def get_parameter_state(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._parameter_state]

    def set_parameter_state(self, value: List[Dict[str, Any]]) -> None:
        self._parameter_state = [dict(row) for row in value]

    def get_initial_parameter_snapshot(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._initial_parameter_snapshot]

    def get_global_dataset_params(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self._global_dataset_params.items()}

    def set_global_dataset_params(self, value: Dict[str, Dict[str, float]]) -> None:
        self._global_dataset_params = {k: dict(v) for k, v in value.items()}

    def get_global_dataset_variable_params(self) -> Dict[str, Dict[str, Dict]]:
        return {
            k: {kk: dict(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in self._global_dataset_variable_params.items()
        }

    def set_global_dataset_variable_params(self, value: Dict[str, Dict[str, Dict]]) -> None:
        self._global_dataset_variable_params = {
            k: {kk: dict(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in value.items()
        }

    def get_fixed_shared_params(self) -> Dict[str, float]:
        return dict(self._fixed_shared_params)

    def set_fixed_shared_params(self, value: Dict[str, float]) -> None:
        self._fixed_shared_params = dict(value)

    def get_last_fit_params(self) -> Dict[str, float]:
        return dict(self._last_fit_params)

    def set_last_fit_params(self, value: Dict[str, float]) -> None:
        self._last_fit_params = dict(value)

    def get_staged_dataset_params(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self._staged_dataset_params.items()}

    def set_staged_dataset_params(self, value: Dict[str, Dict[str, float]]) -> None:
        self._staged_dataset_params = {k: dict(v) for k, v in value.items()}

    def get_mechanism_species(self) -> List[str]:
        return list(self._mechanism_species)

    def set_mechanism_species(self, value: List[str]) -> None:
        self._mechanism_species = list(value)

    def get_prepared_param_names(self) -> List[str]:
        return list(self._prepared_param_names)

    def set_prepared_param_names(self, value: List[str]) -> None:
        self._prepared_param_names = list(value)

    def get_shared_param_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._shared_param_definitions.items()}

    def set_shared_param_definitions(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._shared_param_definitions = {k: dict(v) for k, v in value.items()}

    # ------------------------------------------------------------------
    # Public API — state mutation from fit results
    # ------------------------------------------------------------------

    def push_fit_results(
        self,
        shared_params: Dict[str, float],
        dataset_params: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        self._last_fit_params = dict(shared_params)
        self._staged_dataset_params = {k: dict(v) for k, v in (dataset_params or {}).items()}
        for entry in self._parameter_state:
            if entry.get("scope") == "shared":
                name = entry["param_name"]
                if name in shared_params:
                    entry["value"] = float(shared_params[name])
                    entry["last_fit"] = float(shared_params[name])
            elif entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                if ds_id and param_name:
                    ds_map = (dataset_params or {}).get(ds_id) or {}
                    if isinstance(ds_map, dict) and param_name in ds_map:
                        entry["value"] = float(ds_map[param_name])
                        entry["last_fit"] = float(ds_map[param_name])
        self._populate_parameter_table()

    def push_best_update(
        self,
        shared_params: Dict[str, float],
        dataset_params: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        self._last_fit_params = dict(shared_params)
        if isinstance(dataset_params, dict):
            self._staged_dataset_params = {
                str(ds_id): dict(param_map) for ds_id, param_map in dataset_params.items() if isinstance(param_map, dict)
            }
        for entry in self._parameter_state:
            if entry.get("scope") == "shared":
                name = str(entry.get("param_name") or "")
                if name in shared_params:
                    entry["value"] = float(shared_params[name])
                    entry["last_fit"] = float(shared_params[name])
            elif entry.get("scope") == "dataset":
                ds_id = str(entry.get("dataset_id") or "")
                param_name = str(entry.get("param_name") or "")
                if ds_id and param_name and isinstance(dataset_params, dict):
                    ds_map = dataset_params.get(ds_id) or {}
                    if isinstance(ds_map, dict) and param_name in ds_map:
                        entry["value"] = float(ds_map[param_name])
                        entry["last_fit"] = float(ds_map[param_name])
        self._populate_parameter_table()

    # ------------------------------------------------------------------
    # Public API — running state
    # ------------------------------------------------------------------

    def set_running_state(self, running: bool) -> None:
        if hasattr(self, "_add_param_button"):
            self._add_param_button.setEnabled(not running)
        if hasattr(self, "_remove_param_button"):
            self._remove_param_button.setEnabled(
                (not running) and bool({item.row() for item in self._param_table.selectedItems()})
            )

    # ------------------------------------------------------------------
    # Public API — dataset lifecycle
    # ------------------------------------------------------------------

    def refresh_ic_dataset_combo(self, dataset_entries: List[Dict[str, Any]]) -> None:
        self._dataset_entries = list(dataset_entries)
        self._refresh_initial_conditions_dataset_combo_items()

    def remove_dataset_parameter_rows(self, dataset_ids: Sequence[str]) -> None:
        remove_set = {str(x) for x in dataset_ids}
        for ds_id in remove_set:
            self._global_dataset_params.pop(ds_id, None)
            self._global_dataset_variable_params.pop(ds_id, None)
            self._staged_dataset_params.pop(ds_id, None)
        kept: List[Dict[str, Any]] = []
        for row in self._parameter_state:
            if str(row.get("scope") or "") == "dataset" and str(row.get("dataset_id") or "") in remove_set:
                continue
            kept.append(row)
        self._parameter_state = kept
        if remove_set:
            self._populate_parameter_table()

    def seed_dataset_initial_params(self, dataset_id: str) -> None:
        self._seed_dataset_initial_params_from_fit_settings(dataset_id)

    def repaint_parameter_table(self) -> None:
        self._populate_parameter_table()

    def add_missing_scalars_as_parameters(
        self,
        missing_scalars: Sequence[str],
        dataset_ids: Sequence[str],
        scalar_scope: str,
    ) -> None:
        self._auto_add_missing_scalars_as_parameters(
            missing_scalars=missing_scalars,
            dataset_ids=dataset_ids,
            scalar_scope=scalar_scope,
        )
        self._populate_parameter_table()

    def mirror_staged_ic_values(self) -> int:
        total_updates = 0
        for dataset_id, param_map in self._staged_dataset_params.items():
            if not isinstance(param_map, dict):
                continue
            for key, value in param_map.items():
                key_str = str(key)
                if not key_str.startswith(INITIAL_PREFIX):
                    continue
                self._global_dataset_params.setdefault(dataset_id, {})[key_str] = float(value)
                if dataset_id in self._global_dataset_variable_params:
                    spec = self._global_dataset_variable_params[dataset_id].get(key_str)
                    if isinstance(spec, dict):
                        spec["initial"] = float(value)
                total_updates += 1
        return total_updates

    def collect_integration_settings(self) -> Optional[Tuple[str, float, float]]:
        return self._collect_integration_settings_for_run()
