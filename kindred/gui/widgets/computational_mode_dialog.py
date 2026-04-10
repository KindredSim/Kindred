from __future__ import annotations

from contextlib import suppress
import types
import math
from typing import Optional

from PySide6 import QtCore, QtWidgets

from kindred.core.simulator.computational_mode import (
    COMP_BLOCK_END,
    COMP_BLOCK_START,
    GENERATED_BLOCK_END,
    GENERATED_BLOCK_START,
    compile_comp_spec,
    extract_marked_block,
    parse_comp_block,
    upsert_computational_mode_blocks,
)
from kindred.gui.qt_leak_diagnostics import (
    maybe_log_qt_leak_snapshot,
    schedule_qt_leak_snapshot_after_event_cycles,
)
from kindred.gui.ui_helpers import setup_scientific_validator


class _EnumComboDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, choices: list[str], parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._choices = [str(c) for c in (choices or []) if str(c).strip()]

    def createEditor(self, parent: QtWidgets.QWidget, option, index):  # type: ignore[override]
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(self._choices)
        combo.setEditable(False)
        return combo

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex) -> None:  # type: ignore[override]
        if not isinstance(editor, QtWidgets.QComboBox):
            return
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if value is None:
            value = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        text = str(value or "").strip()
        if text:
            editor.setCurrentText(text)

    def setModelData(self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel, index: QtCore.QModelIndex) -> None:  # type: ignore[override]
        if not isinstance(editor, QtWidgets.QComboBox):
            return
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class ComputationalModeDialog(QtWidgets.QDialog):
    """
    Dialog for authoring Computational Mode inputs (absolute free energies) and generating
    Kindred's existing energy-mode state-network DSL as plain text inside the Reaction DSL.

    This dialog is intentionally separate from the Mechanism panel and does not embed/parent
    the mechanism editor widgets.
    """

    def __init__(self, main_window: object) -> None:
        super().__init__(parent=main_window if isinstance(main_window, QtWidgets.QWidget) else None)
        self._main_window = main_window

        # Canonical/global values (float64) used for serialization.
        self._temperature_K_canonical = float(298.15)
        self._pressure_atm_canonical = float(1.0)
        self._std_default_M_canonical = float(1.0)
        self._kfast_default_canonical = float(1e9)
        self._temperature_display_unit = "K"
        self._pressure_display_unit = "atm"

        self.setWindowTitle("Computational Mode")
        self.setMinimumSize(860, 620)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            "Define absolute computed free energies (DFT-style) and generate a ΔG-based energy-mode mechanism.\n"
            "On Apply/OK, Kindred writes two marked blocks into the Reaction DSL editor."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._error_label = QtWidgets.QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("font-weight: bold;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        max_input_width = 160

        def _make_header(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(str(text))
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            return label

        layout.addWidget(_make_header("Global"))

        globals_section = QtWidgets.QWidget(self)
        globals_section.setObjectName("computationalModeGlobalsSection")
        globals_layout = QtWidgets.QVBoxLayout(globals_section)
        globals_layout.setContentsMargins(0, 0, 0, 0)
        globals_layout.setSpacing(6)

        self._temperature_spin = QtWidgets.QDoubleSpinBox()
        self._temperature_spin.setRange(1e-6, 1e6)
        self._temperature_spin.setDecimals(2)
        self._temperature_spin.setMaximumWidth(max_input_width)
        self._temperature_unit = QtWidgets.QComboBox()
        self._temperature_unit.addItems(["K", "°C"])
        self._temperature_unit.setCurrentText("K")
        self._temperature_unit.setMaximumWidth(80)

        self._pressure_value = QtWidgets.QDoubleSpinBox()
        self._pressure_value.setRange(1e-12, 1e12)
        self._pressure_value.setDecimals(3)
        self._pressure_value.setMaximumWidth(210)
        self._pressure_unit = QtWidgets.QComboBox()
        self._pressure_unit.addItems(["atm", "bar", "Pa"])
        self._pressure_unit.setCurrentText("atm")
        self._pressure_unit.setMaximumWidth(80)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Temperature:"))
        row.addWidget(self._temperature_spin)
        row.addWidget(self._temperature_unit)
        row.addSpacing(14)
        row.addWidget(QtWidgets.QLabel("Pressure:"))
        row.addWidget(self._pressure_value)
        row.addWidget(self._pressure_unit)
        row.addStretch(1)
        globals_layout.addLayout(row)

        self._energy_unit = QtWidgets.QComboBox()
        self._energy_unit.addItems(["hartree", "kJ/mol", "kcal/mol", "J/mol"])
        self._energy_unit.setCurrentText("hartree")
        self._energy_unit.setMaximumWidth(max_input_width)

        self._std_default_spin = QtWidgets.QDoubleSpinBox()
        self._std_default_spin.setRange(1e-12, 1e12)
        self._std_default_spin.setDecimals(3)
        self._std_default_spin.setMaximumWidth(210)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Energy unit (input):"))
        row.addWidget(self._energy_unit)
        row.addSpacing(14)
        row.addWidget(QtWidgets.QLabel("Std default (M):"))
        row.addWidget(self._std_default_spin)
        row.addStretch(1)
        globals_layout.addLayout(row)

        self._output_energy_unit = QtWidgets.QComboBox()
        # Must be compatible with the Reaction DSL `energy=...` directive.
        self._output_energy_unit.addItems(["kJ/mol", "kcal/mol"])
        self._output_energy_unit.setCurrentText("kJ/mol")
        self._output_energy_unit.setMaximumWidth(max_input_width)

        self._kfast_default_edit = QtWidgets.QLineEdit()
        self._kfast_default_edit.setPlaceholderText("e.g. 1e9")
        self._kfast_default_edit.setClearButtonEnabled(True)
        setup_scientific_validator(self._kfast_default_edit)
        self._kfast_default_edit.setMaximumWidth(max_input_width)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Energy unit (output):"))
        row.addWidget(self._output_energy_unit)
        row.addSpacing(14)
        row.addWidget(QtWidgets.QLabel("Fast k default:"))
        row.addWidget(self._kfast_default_edit)
        row.addStretch(1)
        globals_layout.addLayout(row)

        self._std_help = QtWidgets.QLabel(
            "<b>Energy unit (input)</b>: unit for entered absolute <code>G</code> values; converted internally to J/mol.<br>"
            "<b>Energy unit (output)</b>: unit used for <code>energy=...</code> and numeric energies/ΔG values emitted below.<br>"
            "<b>Fast k default</b>: default forward rate for fast GS↔GS equilibria without a TS; reverse is computed from detailed "
            "balance using ΔG°, Std/Cref.<br>"
            "<b>Std (M)</b>: standard-state concentration used for thermodynamic corrections in the generated mechanism.<br>"
            "<b>Cref (M)</b>: reference concentration associated with the absolute <code>G</code> inputs "
            "(for gases, default is derived from <code>p</code>,<code>T</code> unless overridden).<br>"
            "<b>Std default (M)</b>: used when a species omits <b>Std (M)</b>.<br>"
            "<code>G(std) = G(input) + RT ln(std/cref)</code><br>"
            "<b>Units (mass action)</b>: [k<sub>f</sub>] = M<sup>(1−n)</sup> s<sup>−1</sup>, where <i>n</i> is forward molecularity "
            "(sum of reactant stoichiometric coefficients).<br>"
            "n=1: s<sup>−1</sup>; n=2: M<sup>−1</sup> s<sup>−1</sup>; n=3: M<sup>−2</sup> s<sup>−1</sup>."
        )
        self._std_help.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._std_help.setWordWrap(True)
        help_font = self._std_help.font()
        help_font.setPointSize(max(8, int(help_font.pointSize()) - 2))
        self._std_help.setFont(help_font)
        self._std_help.setStyleSheet("")
        globals_layout.addWidget(self._std_help)

        layout.addWidget(globals_section)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        species_panel = QtWidgets.QWidget()
        species_layout = QtWidgets.QVBoxLayout(species_panel)
        species_row = QtWidgets.QHBoxLayout()
        species_row.addWidget(QtWidgets.QLabel("Species (GS/TS):"))
        species_row.addStretch()
        self._add_species_btn = QtWidgets.QPushButton("Add species")
        species_row.addWidget(self._add_species_btn)
        self._remove_species_btn = QtWidgets.QPushButton("Remove selected")
        species_row.addWidget(self._remove_species_btn)
        species_layout.addLayout(species_row)

        self._species_table = QtWidgets.QTableWidget(0, 6)
        self._species_table.setHorizontalHeaderLabels(["Name", "Type", "G", "Std(M)", "Cref(M)", "Degeneracy"])
        self._species_table.horizontalHeader().setStretchLastSection(True)
        self._species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._species_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._species_type_delegate = _EnumComboDelegate(["GS", "TS"], parent=self._species_table)
        self._species_table.setItemDelegateForColumn(1, self._species_type_delegate)
        species_layout.addWidget(self._species_table, stretch=1)
        splitter.addWidget(species_panel)

        reactions_panel = QtWidgets.QWidget()
        reactions_layout = QtWidgets.QVBoxLayout(reactions_panel)
        reactions_row = QtWidgets.QHBoxLayout()
        reactions_row.addWidget(QtWidgets.QLabel("Reactions / Channels:"))
        reactions_row.addStretch()
        self._add_reaction_btn = QtWidgets.QPushButton("Add reaction")
        reactions_row.addWidget(self._add_reaction_btn)
        self._remove_reaction_btn = QtWidgets.QPushButton("Remove selected")
        reactions_row.addWidget(self._remove_reaction_btn)
        reactions_layout.addLayout(reactions_row)

        self._reactions_table = QtWidgets.QTableWidget(0, 3)
        self._reactions_table.setHorizontalHeaderLabels(["Equation (<-> / <=>)", "Via(TS)", "Fast k"])
        self._reactions_table.horizontalHeader().setStretchLastSection(True)
        self._reactions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._reactions_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        reactions_layout.addWidget(self._reactions_table, stretch=1)
        splitter.addWidget(reactions_panel)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        self._apply_btn = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Apply)
        if self._apply_btn is not None:
            self._apply_btn.setText("Apply")
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)
        if self._apply_btn is not None:
            self._apply_btn.clicked.connect(self.apply)
        layout.addWidget(button_box)

        self._add_species_btn.clicked.connect(self._add_species_row)
        self._remove_species_btn.clicked.connect(self._remove_selected_species)
        self._add_reaction_btn.clicked.connect(self._add_reaction_row)
        self._remove_reaction_btn.clicked.connect(self._remove_selected_reaction)
        self._species_table.itemSelectionChanged.connect(self._on_species_selection_changed)

        self._temperature_spin.valueChanged.connect(self._on_temperature_value_changed)
        self._temperature_unit.currentTextChanged.connect(self._on_temperature_unit_changed)
        self._pressure_value.valueChanged.connect(self._on_pressure_value_changed)
        self._pressure_unit.currentTextChanged.connect(self._on_pressure_unit_changed)
        self._std_default_spin.valueChanged.connect(self._on_std_default_changed)
        self._kfast_default_edit.editingFinished.connect(self._on_kfast_default_editing_finished)

        # Expose a minimal UI handle for tests.
        self._cm_ui = types.SimpleNamespace(
            temperature_spin=self._temperature_spin,
            temperature_unit=self._temperature_unit,
            pressure_value=self._pressure_value,
            pressure_unit=self._pressure_unit,
            energy_unit=self._energy_unit,
            output_energy_unit=self._output_energy_unit,
            std_default_spin=self._std_default_spin,
            add_species_btn=self._add_species_btn,
            add_reaction_btn=self._add_reaction_btn,
            species_table=self._species_table,
            reactions_table=self._reactions_table,
            apply_btn=self._apply_btn,
        )

        self._load_from_existing_comp_block()
        self._sync_global_widgets_from_canonical()
        maybe_log_qt_leak_snapshot(
            self,
            milestone="after_open",
            tables=[self._species_table, self._reactions_table],
        )
        schedule_qt_leak_snapshot_after_event_cycles(
            self,
            milestone="after_20_event_cycles",
            cycles=20,
            tables=[self._species_table, self._reactions_table],
        )

    def _show_error(self, message: str) -> None:
        self._error_label.setText(str(message))
        self._error_label.setVisible(True)

    def _clear_error(self) -> None:
        self._error_label.setText("")
        self._error_label.setVisible(False)

    def _add_species_row(self) -> None:
        row = self._species_table.rowCount()
        self._species_table.insertRow(row)
        self._species_table.setItem(row, 1, QtWidgets.QTableWidgetItem("GS"))
        for col, default in [(2, "0.0"), (3, ""), (4, ""), (5, "1")]:
            self._species_table.setItem(row, col, QtWidgets.QTableWidgetItem(default))
        maybe_log_qt_leak_snapshot(
            self,
            milestone="after_add_species",
            tables=[self._species_table, self._reactions_table],
        )

    def _remove_selected_species(self) -> None:
        row = self._species_table.currentRow()
        if row >= 0:
            self._species_table.removeRow(row)

    def _add_reaction_row(self) -> None:
        row = self._reactions_table.rowCount()
        self._reactions_table.insertRow(row)
        for col in range(self._reactions_table.columnCount()):
            self._reactions_table.setItem(row, col, QtWidgets.QTableWidgetItem(""))

    def _remove_selected_reaction(self) -> None:
        row = self._reactions_table.currentRow()
        if row >= 0:
            self._reactions_table.removeRow(row)

    def _on_species_selection_changed(self) -> None:
        if self._species_table.currentRow() < 0:
            return
        maybe_log_qt_leak_snapshot(
            self,
            milestone="after_select_species_row",
            tables=[self._species_table, self._reactions_table],
        )

    def _load_from_existing_comp_block(self) -> None:
        getter = getattr(self._main_window, "mechanism_reactions_text_raw", None)
        if not callable(getter):
            return
        current = str(getter() or "")
        body = extract_marked_block(current, start_marker=COMP_BLOCK_START, end_marker=COMP_BLOCK_END)
        if not body:
            return
        try:
            spec = parse_comp_block(body)
        except Exception:
            return

        # Try to mirror the current generated block unit to avoid unexpected unit changes on Apply.
        generated = extract_marked_block(current, start_marker=GENERATED_BLOCK_START, end_marker=GENERATED_BLOCK_END)
        if generated:
            for raw in str(generated).splitlines():
                stripped = str(raw).strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.lower().startswith("energy="):
                    _, _, unit = stripped.partition("=")
                    unit = unit.strip()
                    if unit and self._output_energy_unit.findText(unit) >= 0:
                        self._output_energy_unit.setCurrentText(unit)
                    break

        self._temperature_K_canonical = float(spec.temperature_K)
        self._temperature_display_unit = "K"
        self._pressure_atm_canonical = float(spec.pressure_Pa) / 101325.0
        self._pressure_display_unit = "atm"
        self._energy_unit.setCurrentText(str(spec.energy_unit))
        self._std_default_M_canonical = float(spec.std_default_M)
        self._kfast_default_canonical = float(spec.kfast_default)

        self._species_table.setRowCount(0)
        for name in sorted(spec.species.keys()):
            sp = spec.species[name]
            self._add_species_row()
            row = self._species_table.rowCount() - 1
            self._species_table.setItem(row, 0, QtWidgets.QTableWidgetItem(sp.name))
            self._species_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(sp.kind)))
            self._species_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(sp.G_value):.17g}"))
            self._species_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem("" if sp.std_M is None else f"{float(sp.std_M):.17g}")
            )
            self._species_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem("" if sp.cref_M is None else f"{float(sp.cref_M):.17g}")
            )
            self._species_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{float(sp.degeneracy):.17g}"))

        self._reactions_table.setRowCount(0)
        for rxn in list(spec.reactions):
            self._add_reaction_row()
            row = self._reactions_table.rowCount() - 1

            def _fmt_side(sto: dict[str, int]) -> str:
                out = []
                for nm in sorted(sto.keys()):
                    coeff = int(sto[nm])
                    out.append(f"{nm}" if coeff == 1 else f"{coeff}{nm}")
                return " + ".join(out)

            eqn = f"{_fmt_side(rxn.reactants)} <-> {_fmt_side(rxn.products)}"
            self._reactions_table.setItem(row, 0, QtWidgets.QTableWidgetItem(eqn))
            self._reactions_table.setItem(row, 1, QtWidgets.QTableWidgetItem("" if rxn.via_ts is None else rxn.via_ts))
            self._reactions_table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem("" if rxn.fast_k is None else f"{float(rxn.fast_k):.17g}"),
            )

    def _read_item_text(self, table: QtWidgets.QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        return "" if item is None else str(item.text()).strip()

    def _serialize_comp_body(self) -> str:
        def _fmt(x: float) -> str:
            return f"{float(x):.17g}"

        T = float(self._temperature_K_canonical)
        P_atm = float(self._pressure_atm_canonical)
        energy_unit = str(self._energy_unit.currentText()).strip()
        std_default = float(self._std_default_M_canonical)
        kfast_default = float(self._read_kfast_default())

        lines: list[str] = []
        lines.append(f"comp: T = {_fmt(T)} K")
        lines.append(f"comp: pressure = {_fmt(P_atm)} atm")
        lines.append(f"comp: energy_unit = {energy_unit}")
        lines.append(f"comp: std_default = {_fmt(std_default)} M")
        lines.append(f"comp: kfast_default = {_fmt(kfast_default)}")

        # Species
        for row in range(self._species_table.rowCount()):
            name = self._read_item_text(self._species_table, row, 0)
            if not name:
                continue
            kind = self._read_item_text(self._species_table, row, 1) or "GS"

            G_text = self._read_item_text(self._species_table, row, 2)
            if not G_text:
                continue
            G_val = float(G_text)

            std_text = self._read_item_text(self._species_table, row, 3)
            cref_text = self._read_item_text(self._species_table, row, 4)
            deg_text = self._read_item_text(self._species_table, row, 5) or "1"

            deg_val = float(deg_text)
            parts = [f"comp: species {name} type={kind} G={_fmt(G_val)}"]
            if std_text:
                parts.append(f"std={_fmt(float(std_text))} M")
            if cref_text:
                parts.append(f"cref={_fmt(float(cref_text))} M")
            parts.append(f"degeneracy={_fmt(deg_val)}")
            lines.append(" ".join(parts))

        # Reactions/channels
        for row in range(self._reactions_table.rowCount()):
            eqn = self._read_item_text(self._reactions_table, row, 0)
            if not eqn:
                continue
            via = self._read_item_text(self._reactions_table, row, 1)
            fast_k = self._read_item_text(self._reactions_table, row, 2)
            head = "channel" if via else "rxn"
            line = f"comp: {head} {eqn}"
            if via:
                line += f" via {via}"
            if fast_k:
                line += f" fast_k={_fmt(float(fast_k))}"
            lines.append(line)

        return "\n".join(lines).rstrip() + "\n"

    def _read_kfast_default(self) -> float:
        text = str(self._kfast_default_edit.text() or "").strip()
        if not text:
            raise ValueError("Fast k default is required.")
        try:
            v = float(text)
        except Exception as exc:
            raise ValueError("Fast k default must be a number (e.g. 1e9).") from exc
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError("Fast k default must be positive and finite.")
        self._kfast_default_canonical = float(v)
        with QtCore.QSignalBlocker(self._kfast_default_edit):
            self._kfast_default_edit.setText(self._format_kfast_display(self._kfast_default_canonical))
        return float(self._kfast_default_canonical)

    def _sync_global_widgets_from_canonical(self) -> None:
        with QtCore.QSignalBlocker(self._temperature_unit):
            self._temperature_unit.setCurrentText(str(self._temperature_display_unit))
        with QtCore.QSignalBlocker(self._pressure_unit):
            self._pressure_unit.setCurrentText(str(self._pressure_display_unit))

        with QtCore.QSignalBlocker(self._temperature_spin):
            self._temperature_spin.setValue(self._temperature_display_from_K(self._temperature_K_canonical))
        with QtCore.QSignalBlocker(self._pressure_value):
            self._pressure_value.setValue(self._pressure_display_from_atm(self._pressure_atm_canonical))
        with QtCore.QSignalBlocker(self._std_default_spin):
            self._std_default_spin.setValue(float(self._std_default_M_canonical))
        with QtCore.QSignalBlocker(self._kfast_default_edit):
            self._kfast_default_edit.setText(self._format_kfast_display(self._kfast_default_canonical))

    def _temperature_display_from_K(self, T_K: float) -> float:
        if str(self._temperature_display_unit) == "°C":
            return float(T_K) - 273.15
        return float(T_K)

    def _temperature_K_from_display(self, T_disp: float, *, unit: str) -> float:
        if str(unit) == "°C":
            return float(T_disp) + 273.15
        return float(T_disp)

    def _pressure_display_from_atm(self, P_atm: float) -> float:
        unit = str(self._pressure_display_unit)
        if unit == "bar":
            return float(P_atm) * 1.01325
        if unit == "Pa":
            return float(P_atm) * 101325.0
        return float(P_atm)

    def _pressure_atm_from_display(self, P_disp: float, *, unit: str) -> float:
        if str(unit) == "bar":
            return float(P_disp) / 1.01325
        if str(unit) == "Pa":
            return float(P_disp) / 101325.0
        return float(P_disp)

    def _on_temperature_value_changed(self, value: float) -> None:
        self._temperature_K_canonical = float(
            self._temperature_K_from_display(float(value), unit=str(self._temperature_display_unit))
        )

    def _on_temperature_unit_changed(self, unit: str) -> None:
        old = str(self._temperature_display_unit)
        # Sync canonical from current display before switching units.
        self._temperature_K_canonical = float(self._temperature_K_from_display(self._temperature_spin.value(), unit=old))
        self._temperature_display_unit = str(unit)
        with QtCore.QSignalBlocker(self._temperature_spin):
            self._temperature_spin.setValue(self._temperature_display_from_K(self._temperature_K_canonical))

    def _on_pressure_value_changed(self, value: float) -> None:
        self._pressure_atm_canonical = float(
            self._pressure_atm_from_display(float(value), unit=str(self._pressure_display_unit))
        )

    def _on_pressure_unit_changed(self, unit: str) -> None:
        old = str(self._pressure_display_unit)
        self._pressure_atm_canonical = float(self._pressure_atm_from_display(self._pressure_value.value(), unit=old))
        self._pressure_display_unit = str(unit)
        with QtCore.QSignalBlocker(self._pressure_value):
            self._pressure_value.setValue(self._pressure_display_from_atm(self._pressure_atm_canonical))

    def _on_std_default_changed(self, value: float) -> None:
        self._std_default_M_canonical = float(value)

    def _format_kfast_display(self, x: float) -> str:
        v = float(x)
        if not math.isfinite(v):
            return str(v)
        if v == 0.0:
            return "0"
        a = abs(v)
        if a >= 1e6 or a < 1e-3:
            mant, exp = f"{v:.16e}".split("e")
            mant = mant.rstrip("0").rstrip(".")
            exp_i = int(exp)
            return mant if exp_i == 0 else f"{mant}e{exp_i}"
        return f"{v:.17g}"

    def _on_kfast_default_editing_finished(self) -> None:
        try:
            self._read_kfast_default()
        except Exception:
            # Leave the in-progress text as-is; Apply will surface an error.
            return

    def apply(self) -> bool:
        self._clear_error()

        get_reactions_text = getattr(self._main_window, "mechanism_reactions_text_raw", None)
        set_reactions_text = getattr(self._main_window, "set_mechanism_reactions_text_with_optional_undo", None)
        if not callable(get_reactions_text) or not callable(set_reactions_text):
            self._show_error("Mechanism editor is unavailable.")
            return False

        # Ensure any in-progress cell edits are committed before reading table contents.
        with suppress(RuntimeError, TypeError):
            focus = QtWidgets.QApplication.focusWidget()
            if focus is not None and self.isAncestorOf(focus):
                for table in (self._species_table, self._reactions_table):
                    with suppress(Exception):
                        table.closeEditor(
                            focus,
                            QtWidgets.QAbstractItemDelegate.EndEditHint.SubmitModelCache,
                        )
                focus.clearFocus()

        try:
            comp_body = self._serialize_comp_body()
            spec = parse_comp_block(comp_body)
            output_unit = str(self._output_energy_unit.currentText()).strip() or "kJ/mol"
            compiled = compile_comp_spec(spec, output_energy_unit=output_unit)
        except Exception as exc:
            self._show_error(str(exc))
            return False

        current_text = str(get_reactions_text() or "")
        updated = upsert_computational_mode_blocks(
            current_text,
            comp_body=comp_body.strip("\n"),
            generated_body=compiled.generated_reaction_dsl.strip("\n"),
        )

        set_reactions_text(updated, "Apply Computational Mode", record_undo=True)
        sync_authoritative_write = getattr(
            self._main_window,
            "finalize_authoritative_mechanism_widget_write",
            None,
        )
        if not callable(sync_authoritative_write):
            self._show_error("Mechanism authoritative write finalizer is unavailable.")
            return False
        sync_authoritative_write(dispatch_consumers=True)

        # Re-parse and refresh energy-mode sliders without running an ODE solve.
        with suppress(ImportError, RuntimeError, TypeError, ValueError):
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            get_full_dsl = getattr(self._main_window, "get_mechanism_text", None)
            get_temperature_k = getattr(self._main_window, "temperature_spinbox_value", None)
            populate = getattr(self._main_window, "populate_energy_mode_variables_from_mechanism", None)
            if not callable(get_full_dsl) or not callable(get_temperature_k) or not callable(populate):
                return True
            full_dsl = str(get_full_dsl() or "")
            units = UnitsModel(temperature_K=float(get_temperature_k()))
            mech = parse_dsl_to_mechanism(full_dsl, initials={}, units=units)
            if callable(populate):
                populate(mech, refresh_sliders=True)
        return True

    def _on_ok(self) -> None:
        if self.apply():
            self.accept()
