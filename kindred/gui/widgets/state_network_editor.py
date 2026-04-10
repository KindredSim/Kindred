"""
State Network Editor widget.

Allows users to define chemical states and transition states with energies,
and connect them with edges to create a state network graph.

The widget has two sections:
1. States table (name, type GS/TS, energy, energy unit, degeneracy)
2. Edges table (state A, state B)

Emits stateNetworkChanged signal when modified.
"""

from __future__ import annotations

from contextlib import suppress
import os
import logging
from typing import List, Dict, Tuple, Optional

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton,
    QComboBox, QHeaderView, QAbstractItemView, QAbstractItemDelegate, QLineEdit, QStyledItemDelegate,
    QSplitter, QGroupBox
)

from kindred.gui.diagnostics import linux_rss_kb, safe_len_find_children

logger = logging.getLogger(__name__)

__all__ = ["StateNetworkEditor"]


_STATE_KIND_OPTIONS = ("GS", "TS")
_ENERGY_UNIT_OPTIONS = ("kJ/mol", "kcal/mol", "J/mol")


class _ComboBoxItemDelegate(QStyledItemDelegate):
    def __init__(self, values: Tuple[str, ...], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._values = tuple(values)

    def createEditor(self, parent: QWidget, option, index):  # type: ignore[override]
        combo = QComboBox(parent)
        combo.setEditable(False)
        combo.addItems(list(self._values))
        return combo

    def setEditorData(self, editor: QWidget, index) -> None:  # type: ignore[override]
        if not isinstance(editor, QComboBox):
            return
        try:
            value = str(index.data() or "")
        except Exception:
            value = ""
        if value and value not in self._values:
            editor.insertItem(0, value)
        if value:
            editor.setCurrentText(value)
        else:
            editor.setCurrentIndex(0)

    def setModelData(self, editor: QWidget, model, index) -> None:  # type: ignore[override]
        if not isinstance(editor, QComboBox):
            return
        model.setData(index, editor.currentText())

    def updateEditorGeometry(self, editor: QWidget, option, index) -> None:  # type: ignore[override]
        editor.setGeometry(option.rect)


class StateNetworkEditor(QWidget):
    """
    State network editor with states and edges tables.

    Signals
    -------
    stateNetworkChanged : emitted when states or edges are modified
    """

    stateNetworkChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Main layout
        layout = QVBoxLayout(self)

        # Splitter for states and edges
        splitter = QSplitter(Qt.Vertical)

        # States section
        states_group = QGroupBox("States")
        states_layout = QVBoxLayout(states_group)

        # States table
        self._states_table = QTableWidget()
        self._states_table.setColumnCount(5)
        self._states_table.setHorizontalHeaderLabels([
            "Name", "Type", "Energy", "Unit", "Degeneracy"
        ])
        self._states_table.horizontalHeader().setStretchLastSection(False)
        self._states_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._states_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._states_table.cellChanged.connect(self._on_states_changed)
        self._states_table.setItemDelegateForColumn(1, _ComboBoxItemDelegate(_STATE_KIND_OPTIONS, self._states_table))
        self._states_table.setItemDelegateForColumn(3, _ComboBoxItemDelegate(_ENERGY_UNIT_OPTIONS, self._states_table))
        self._states_default_edit_triggers = self._states_table.editTriggers()

        # States buttons
        states_btn_layout = QHBoxLayout()
        self._add_state_btn = QPushButton("Add State")
        self._add_state_btn.clicked.connect(self._add_state_row)
        self._remove_state_btn = QPushButton("Remove Selected")
        self._remove_state_btn.clicked.connect(self._remove_selected_states)
        states_btn_layout.addWidget(self._add_state_btn)
        states_btn_layout.addWidget(self._remove_state_btn)
        states_btn_layout.addStretch()

        states_layout.addWidget(self._states_table)
        states_layout.addLayout(states_btn_layout)

        # Edges section
        edges_group = QGroupBox("Edges (Connectivity)")
        edges_layout = QVBoxLayout(edges_group)

        # Edges table
        self._edges_table = QTableWidget()
        self._edges_table.setColumnCount(2)
        self._edges_table.setHorizontalHeaderLabels(["State A", "State B"])
        self._edges_table.horizontalHeader().setStretchLastSection(True)
        self._edges_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._edges_table.cellChanged.connect(self._on_edges_changed)
        self._edges_default_edit_triggers = self._edges_table.editTriggers()

        # Edges buttons
        edges_btn_layout = QHBoxLayout()
        self._add_edge_btn = QPushButton("Add Edge")
        self._add_edge_btn.clicked.connect(self._add_edge_row)
        self._remove_edge_btn = QPushButton("Remove Selected")
        self._remove_edge_btn.clicked.connect(self._remove_selected_edges)
        edges_btn_layout.addWidget(self._add_edge_btn)
        edges_btn_layout.addWidget(self._remove_edge_btn)
        edges_btn_layout.addStretch()

        edges_layout.addWidget(self._edges_table)
        edges_layout.addLayout(edges_btn_layout)

        # Add to splitter
        splitter.addWidget(states_group)
        splitter.addWidget(edges_group)
        splitter.setStretchFactor(0, 2)  # States get more space
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Validation status
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Track if we're programmatically updating (to prevent signal loops)
        self._read_only = False
        self._updating = False
        self._has_validation_errors = False
        self._logged_first_change = False
        self._debug_state_net = bool(os.environ.get("KINDRED_DEBUG_STATE_NET"))
        self._debug_states_changed_depth = 0
        self._debug_edges_changed_depth = 0
        self._debug_validate_depth = 0
        self._debug_states_changed_max_depth = 0
        self._debug_edges_changed_max_depth = 0
        self._debug_validate_max_depth = 0
        self._debug_reentrancy_warned = set()
        self._debug_open_snapshot_done = False
        self._debug_add_state_snapshot_done = False
        self._debug_select_snapshot_done = False
        self._debug_post_select_snapshot_done = False
        self._change_flush_timer = QtCore.QTimer(self)
        self._change_flush_timer.setSingleShot(True)
        self._change_flush_timer.setInterval(0)
        self._change_flush_timer.timeout.connect(self._flush_user_change)
        if self._debug_state_net:
            self._states_table.itemSelectionChanged.connect(self._debug_on_states_selection_changed)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._debug_state_net and not self._debug_open_snapshot_done:
            self._debug_open_snapshot_done = True
            QtCore.QTimer.singleShot(0, lambda: self._debug_snapshot("after_open"))

    def _debug_snapshot(self, tag: str) -> None:
        if not self._debug_state_net:
            return
        widget_total = safe_len_find_children(self, QWidget)
        combo_total = safe_len_find_children(self, QComboBox)
        try:
            states_rows = int(self._states_table.rowCount())
            states_cols = int(self._states_table.columnCount())
        except Exception:
            states_rows = -1
            states_cols = -1
        try:
            edges_rows = int(self._edges_table.rowCount())
            edges_cols = int(self._edges_table.columnCount())
        except Exception:
            edges_rows = -1
            edges_cols = -1

        logger.info(
            "StateNetworkEditor diag (%s): rss_kb=%s widgets=%s combobox=%s items~=(states=%s edges=%s)",
            str(tag),
            linux_rss_kb(),
            widget_total,
            combo_total,
            states_rows * states_cols if states_rows >= 0 and states_cols >= 0 else -1,
            edges_rows * edges_cols if edges_rows >= 0 and edges_cols >= 0 else -1,
        )

    def _debug_on_states_selection_changed(self) -> None:
        if not self._debug_state_net:
            return
        if self._debug_select_snapshot_done:
            return
        if self._states_table.rowCount() <= 0:
            return
        if not self._states_table.selectedIndexes():
            return

        if not self._debug_reentrancy_enter("selection_changed"):
            self._debug_reentrancy_exit("selection_changed")
            return
        try:
            self._debug_select_snapshot_done = True
            self._debug_snapshot("after_select_row")

            if not self._debug_post_select_snapshot_done:
                self._debug_post_select_snapshot_done = True

                def _after_cycles() -> None:
                    def _tick(remaining: int) -> None:
                        if remaining <= 0:
                            self._debug_snapshot("after_select_row+20ticks")
                            return
                        QtCore.QTimer.singleShot(0, lambda: _tick(remaining - 1))

                    _tick(20)

                QtCore.QTimer.singleShot(0, _after_cycles)
        finally:
            self._debug_reentrancy_exit("selection_changed")

    def _debug_reentrancy_enter(self, name: str) -> bool:
        if not self._debug_state_net:
            return True
        depth_attr = f"_debug_{name}_depth"
        max_attr = f"_debug_{name}_max_depth"
        try:
            depth = int(getattr(self, depth_attr, 0)) + 1
        except Exception:
            depth = 1
        setattr(self, depth_attr, depth)
        try:
            current_max = int(getattr(self, max_attr, 0))
        except Exception:
            current_max = 0
        if depth > current_max:
            setattr(self, max_attr, depth)

        if depth <= 3:
            return True

        key = str(name)
        if key not in self._debug_reentrancy_warned:
            self._debug_reentrancy_warned.add(key)
            try:
                import faulthandler

                stack = "".join(faulthandler.format_stack(limit=12))
            except Exception:
                stack = ""
            logger.warning(
                "StateNetworkEditor: suspected re-entrancy in %s (depth=%s)\n%s",
                name,
                depth,
                stack,
            )
        return False

    def _debug_reentrancy_exit(self, name: str) -> None:
        if not self._debug_state_net:
            return
        depth_attr = f"_debug_{name}_depth"
        try:
            depth = int(getattr(self, depth_attr, 0))
        except Exception:
            depth = 0
        if depth <= 1:
            setattr(self, depth_attr, 0)
        else:
            setattr(self, depth_attr, depth - 1)

    def _clear_table_widgets(self, table: QTableWidget) -> None:
        """
        Remove and dispose any QWidget-based cell widgets.

        Note: QTableWidget does not reliably delete cell widgets when rows are removed
        (e.g., via setRowCount(0) / removeRow). Without explicit cleanup, repeated loads
        can leak widgets and cause runaway memory usage.
        """
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                widget = table.cellWidget(row, col)
                if widget is None:
                    continue
                with suppress(RuntimeError, TypeError):
                    table.removeCellWidget(row, col)
                with suppress(RuntimeError, TypeError):
                    widget.setParent(None)
                with suppress(RuntimeError, TypeError):
                    widget.deleteLater()

    def _reset_tables(self) -> None:
        """Clear table contents and dispose cell widgets without emitting change signals."""
        with QtCore.QSignalBlocker(self._states_table), QtCore.QSignalBlocker(self._edges_table):
            self._clear_table_widgets(self._states_table)
            self._clear_table_widgets(self._edges_table)
            self._states_table.clearContents()
            self._edges_table.clearContents()
            self._states_table.setRowCount(0)
            self._edges_table.setRowCount(0)

    # -------------------- Public API --------------------

    def get_state_network_dsl(self) -> str:
        """
        Export state network as DSL text.

        Returns
        -------
        str
            DSL text with state: and edge: lines
        """
        lines = []

        # Add states
        for row in range(self._states_table.rowCount()):
            name = self._states_table.item(row, 0)
            kind_item = self._states_table.item(row, 1)
            energy = self._states_table.item(row, 2)
            unit_item = self._states_table.item(row, 3)
            degeneracy = self._states_table.item(row, 4)

            if name and name.text().strip():
                name_str = name.text().strip()
                kind_str = kind_item.text().strip() if kind_item and kind_item.text().strip() else "GS"
                energy_str = energy.text().strip() if energy else "0"
                unit_str = unit_item.text().strip() if unit_item and unit_item.text().strip() else "kJ/mol"
                deg_str = degeneracy.text().strip() if degeneracy and degeneracy.text().strip() else "1"

                lines.append(
                    f"state: {name_str}, kind={kind_str}, "
                    f"energy={energy_str}, energy_unit={unit_str}, "
                    f"degeneracy={deg_str}"
                )

        # Add edges
        for row in range(self._edges_table.rowCount()):
            state_a = self._edges_table.item(row, 0)
            state_b = self._edges_table.item(row, 1)

            if state_a and state_b and state_a.text().strip() and state_b.text().strip():
                a_str = state_a.text().strip()
                b_str = state_b.text().strip()
                lines.append(f"edge: {a_str},{b_str}")

        return "\n".join(lines)

    def set_state_network_dsl(self, dsl_text: str):
        """
        Load state network from DSL text.

        Parameters
        ----------
        dsl_text : str
            DSL text with state: and edge: lines
        """
        self._change_flush_timer.stop()
        self._updating = True
        try:
            # Clear existing data (must also dispose of cell widgets to avoid leaks)
            self._reset_tables()

            # Parse DSL
            for line in dsl_text.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.lower().startswith('state:'):
                    self._parse_state_line(line)
                elif line.lower().startswith('edge:'):
                    self._parse_edge_line(line)

            self._validate()

        finally:
            self._updating = False

        # Ensure any pending paints flush through without forcing synchronous repaint loops.
        self._states_table.viewport().update()
        self._edges_table.viewport().update()

        self.stateNetworkChanged.emit()

    def clear(self):
        """Clear all states and edges."""
        self._change_flush_timer.stop()
        self._updating = True
        try:
            self._reset_tables()
            self._has_validation_errors = False
            self._status_label.setText("")
        finally:
            self._updating = False

        self.stateNetworkChanged.emit()

    def is_read_only(self) -> bool:
        return bool(getattr(self, "_read_only", False))

    def is_valid(self) -> bool:
        self._change_flush_timer.stop()
        self._commit_active_table_editor(self._states_table)
        self._commit_active_table_editor(self._edges_table)
        self._validate()
        return not bool(self._has_validation_errors)

    def _open_table_editors(self, table: QTableWidget) -> list[QWidget]:
        open_editors = []
        active_editor = QApplication.focusWidget()
        if active_editor is not None and active_editor is not table and table.isAncestorOf(active_editor):
            open_editors.append(active_editor)
        for editor_type in (QLineEdit, QComboBox):
            for editor in table.findChildren(editor_type):
                if editor is table or not table.isAncestorOf(editor) or editor in open_editors:
                    continue
                open_editors.append(editor)
        return open_editors

    def _commit_active_table_editor(self, table: QTableWidget) -> None:
        for editor in self._open_table_editors(table):
            with suppress(RuntimeError, TypeError):
                delegate = table.itemDelegate()
                if delegate is not None:
                    delegate.commitData.emit(editor)
                    delegate.closeEditor.emit(
                        editor,
                        QAbstractItemDelegate.EndEditHint.SubmitModelCache,
                    )
            with suppress(RuntimeError, TypeError):
                table.closeEditor(
                    editor,
                    QAbstractItemDelegate.EndEditHint.SubmitModelCache,
                )

    def _close_active_table_editor(self, table: QTableWidget) -> None:
        open_editors = self._open_table_editors(table)
        if not open_editors:
            return
        for editor in open_editors:
            with suppress(RuntimeError, TypeError):
                delegate = table.itemDelegate()
                if delegate is not None:
                    delegate.closeEditor.emit(
                        editor,
                        QAbstractItemDelegate.EndEditHint.RevertModelCache,
                    )
            with suppress(RuntimeError, TypeError):
                table.closeEditor(
                    editor,
                    QAbstractItemDelegate.EndEditHint.RevertModelCache,
                )
            with suppress(RuntimeError, TypeError):
                editor.blockSignals(True)
                editor.clearFocus()
                editor.hide()
                editor.close()
                editor.setParent(None)
                editor.deleteLater()

    def set_read_only(self, read_only: bool) -> None:
        read_only = bool(read_only)
        self._read_only = read_only

        self._add_state_btn.setEnabled(not read_only)
        self._remove_state_btn.setEnabled(not read_only)
        self._add_edge_btn.setEnabled(not read_only)
        self._remove_edge_btn.setEnabled(not read_only)

        if read_only:
            self._close_active_table_editor(self._states_table)
            self._close_active_table_editor(self._edges_table)
            no_edit = QAbstractItemView.EditTrigger.NoEditTriggers
            self._states_table.setEditTriggers(no_edit)
            self._edges_table.setEditTriggers(no_edit)
            return

        self._states_table.setEditTriggers(self._states_default_edit_triggers)
        self._edges_table.setEditTriggers(self._edges_default_edit_triggers)

    def get_states(self) -> List[Dict]:
        """
        Get states as list of dicts.

        Returns
        -------
        list of dict
            [{"name": str, "kind": "GS"|"TS", "energy": float, "unit": str, "degeneracy": float}, ...]
        """
        states = []
        for row in range(self._states_table.rowCount()):
            name_item = self._states_table.item(row, 0)
            kind_item = self._states_table.item(row, 1)
            energy_item = self._states_table.item(row, 2)
            unit_item = self._states_table.item(row, 3)
            deg_item = self._states_table.item(row, 4)

            if name_item and name_item.text().strip():
                try:
                    states.append({
                        "name": name_item.text().strip(),
                        "kind": kind_item.text().strip() if kind_item and kind_item.text().strip() else "GS",
                        "energy": float(energy_item.text()) if energy_item and energy_item.text().strip() else 0.0,
                        "unit": unit_item.text().strip() if unit_item and unit_item.text().strip() else "kJ/mol",
                        "degeneracy": float(deg_item.text()) if deg_item and deg_item.text().strip() else 1.0
                    })
                except ValueError:
                    continue

        return states

    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Get edges as list of (state_a, state_b) tuples.

        Returns
        -------
        list of tuple
            [(state_a, state_b), ...]
        """
        edges = []
        for row in range(self._edges_table.rowCount()):
            a_item = self._edges_table.item(row, 0)
            b_item = self._edges_table.item(row, 1)

            if a_item and b_item and a_item.text().strip() and b_item.text().strip():
                edges.append((a_item.text().strip(), b_item.text().strip()))

        return edges

    # -------------------- Private Methods --------------------

    def _add_state_row(self):
        """Add a new state row."""
        if self.is_read_only():
            return
        row = self._states_table.rowCount()
        self._states_table.insertRow(row)

        # Name
        self._states_table.setItem(row, 0, QTableWidgetItem(""))

        # Type (GS/TS)
        self._states_table.setItem(row, 1, QTableWidgetItem("GS"))

        # Energy
        self._states_table.setItem(row, 2, QTableWidgetItem("0"))

        # Unit
        self._states_table.setItem(row, 3, QTableWidgetItem("kJ/mol"))

        # Degeneracy
        self._states_table.setItem(row, 4, QTableWidgetItem("1"))

        if not self._updating:
            self._schedule_user_change()
        if self._debug_state_net and not self._debug_add_state_snapshot_done:
            self._debug_add_state_snapshot_done = True
            QtCore.QTimer.singleShot(0, lambda: self._debug_snapshot("after_add_state"))

    def _remove_selected_states(self):
        """Remove selected state rows."""
        if self.is_read_only():
            return
        selected_rows = sorted(set(index.row() for index in self._states_table.selectedIndexes()), reverse=True)
        with QtCore.QSignalBlocker(self._states_table):
            for row in selected_rows:
                self._states_table.removeRow(row)

        if not self._updating:
            self._schedule_user_change()

    def _add_edge_row(self):
        """Add a new edge row."""
        if self.is_read_only():
            return
        row = self._edges_table.rowCount()
        self._edges_table.insertRow(row)

        # State A
        self._edges_table.setItem(row, 0, QTableWidgetItem(""))

        # State B
        self._edges_table.setItem(row, 1, QTableWidgetItem(""))

        if not self._updating:
            self._schedule_user_change()

    def _remove_selected_edges(self):
        """Remove selected edge rows."""
        if self.is_read_only():
            return
        selected_rows = sorted(set(index.row() for index in self._edges_table.selectedIndexes()), reverse=True)
        for row in selected_rows:
            self._edges_table.removeRow(row)

        if not self._updating:
            self._schedule_user_change()

    def _schedule_user_change(self) -> None:
        if self._updating:
            return
        if not self._logged_first_change:
            logger.info("StateNetworkEditor: first user edit event")
            self._logged_first_change = True
        self._change_flush_timer.start()

    def _flush_user_change(self) -> None:
        if self._updating:
            return
        self._validate()
        self.stateNetworkChanged.emit()

    def _on_states_changed(self):
        """Handle state table changes."""
        if not self._updating:
            debug_entered = False
            if self._debug_state_net:
                debug_entered = True
                if not self._debug_reentrancy_enter("states_changed"):
                    self._debug_reentrancy_exit("states_changed")
                    return
            try:
                self._schedule_user_change()
            finally:
                if debug_entered:
                    self._debug_reentrancy_exit("states_changed")

    def _on_edges_changed(self):
        """Handle edge table changes."""
        if not self._updating:
            debug_entered = False
            if self._debug_state_net:
                debug_entered = True
                if not self._debug_reentrancy_enter("edges_changed"):
                    self._debug_reentrancy_exit("edges_changed")
                    return
            try:
                self._schedule_user_change()
            finally:
                if debug_entered:
                    self._debug_reentrancy_exit("edges_changed")

    def _validate(self):
        """Validate state network and show status."""
        debug_entered = False
        if self._debug_state_net:
            debug_entered = True
            if not self._debug_reentrancy_enter("validate"):
                self._debug_reentrancy_exit("validate")
                return
        try:
            states = []
            errors = []
            for row in range(self._states_table.rowCount()):
                name_item = self._states_table.item(row, 0)
                kind_item = self._states_table.item(row, 1)
                energy_item = self._states_table.item(row, 2)
                unit_item = self._states_table.item(row, 3)
                deg_item = self._states_table.item(row, 4)

                name = name_item.text().strip() if name_item and name_item.text().strip() else ""
                if not name:
                    continue

                energy = 0.0
                energy_text = energy_item.text().strip() if energy_item and energy_item.text().strip() else ""
                if energy_text:
                    try:
                        energy = float(energy_text)
                    except ValueError:
                        errors.append(f"State '{name}' has invalid energy")

                degeneracy = 1.0
                degeneracy_text = deg_item.text().strip() if deg_item and deg_item.text().strip() else ""
                if degeneracy_text:
                    try:
                        degeneracy = float(degeneracy_text)
                    except ValueError:
                        errors.append(f"State '{name}' has invalid degeneracy")

                states.append(
                    {
                        "name": name,
                        "kind": kind_item.text().strip() if kind_item and kind_item.text().strip() else "GS",
                        "energy": energy,
                        "unit": unit_item.text().strip() if unit_item and unit_item.text().strip() else "kJ/mol",
                        "degeneracy": degeneracy,
                    }
                )
            edges = self.get_edges()

            # Check for duplicate state names
            names = [s["name"] for s in states]
            if len(names) != len(set(names)):
                errors.append("Duplicate state names")

            # Build adjacency for TS degree check
            state_names = set(names)
            degree = {name: 0 for name in names}

            for a, b in edges:
                if a not in state_names:
                    errors.append(f"Edge references unknown state: {a}")
                if b not in state_names:
                    errors.append(f"Edge references unknown state: {b}")
                if a in degree:
                    degree[a] += 1
                if b in degree:
                    degree[b] += 1

            # Check TS degree = 2
            for state in states:
                if state["kind"] == "TS":
                    name = state["name"]
                    if name in degree and degree[name] != 2:
                        errors.append(f"TS '{name}' has degree {degree[name]}, must be 2")

            self._has_validation_errors = bool(errors)
            if errors:
                self._status_label.setText("⚠️ " + "; ".join(errors))
                self._status_label.setStyleSheet("font-weight: bold;")
            else:
                if states or edges:
                    self._status_label.setText("✓ State network valid")
                    self._status_label.setStyleSheet("")
                else:
                    self._status_label.setText("")

            # Avoid forcing synchronous repaint loops; schedule paints instead.
            self._status_label.update()
            self._states_table.viewport().update()
            self._edges_table.viewport().update()
        finally:
            if debug_entered:
                self._debug_reentrancy_exit("validate")

    def _parse_state_line(self, line: str):
        """Parse 'state: name, kind=GS, energy=0, ...' line."""
        # Simple parser
        _, rest = line.split(':', 1)
        parts = rest.split(',')

        state_data = {"name": "", "kind": "GS", "energy": "0", "unit": "kJ/mol", "degeneracy": "1"}

        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "kind":
                    state_data["kind"] = value.upper()
                elif key == "energy":
                    state_data["energy"] = value
                elif key == "energy_unit":
                    state_data["unit"] = value
                elif key == "degeneracy":
                    state_data["degeneracy"] = value
                elif key in ("name", "state"):
                    state_data["name"] = value
            else:
                # First part without '=' is the name
                if not state_data["name"]:
                    state_data["name"] = part

        # Add row
        row = self._states_table.rowCount()
        self._states_table.insertRow(row)

        self._states_table.setItem(row, 0, QTableWidgetItem(state_data["name"]))

        self._states_table.setItem(row, 1, QTableWidgetItem(state_data["kind"]))

        self._states_table.setItem(row, 2, QTableWidgetItem(state_data["energy"]))

        self._states_table.setItem(row, 3, QTableWidgetItem(state_data["unit"]))

        self._states_table.setItem(row, 4, QTableWidgetItem(state_data["degeneracy"]))

    def _parse_edge_line(self, line: str):
        """Parse 'edge: A,B' or 'edge: A-B' line."""
        _, rest = line.split(':', 1)
        rest = rest.strip()

        # Split by comma or dash
        if ',' in rest:
            parts = rest.split(',')
        elif '-' in rest:
            parts = rest.split('-')
        else:
            return  # Invalid

        if len(parts) != 2:
            return

        state_a = parts[0].strip()
        state_b = parts[1].strip()

        # Add row
        row = self._edges_table.rowCount()
        self._edges_table.insertRow(row)

        self._edges_table.setItem(row, 0, QTableWidgetItem(state_a))
        self._edges_table.setItem(row, 1, QTableWidgetItem(state_b))
