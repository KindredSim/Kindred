# kindred/gui/widgets/mechanism_editor.py
"""Tabbed mechanism editor for reactions, notes, and state networks."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

from kindred.gui.ui_helpers import make_scroll_area

# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.state_network_editor import StateNetworkEditor
from kindred.gui.widgets.mechanism_highlighter import MechanismHighlighter
from kindred.gui.widgets.variable_sliders import VariableSliders
from kindred.gui.widgets.species_sliders import BatchSpeciesSliders

__all__ = ["MechanismEditorTabbed"]


class _PersistentToggleMenu(QtWidgets.QMenu):
    """Keep checkable actions open so users can toggle multiple sliders in one pass."""

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        action = self.actionAt(event.position().toPoint())
        if action is not None and action.isEnabled() and action.isCheckable():
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MechanismEditorTabbed(QtWidgets.QWidget):
    speciesModeChanged = Signal(bool)
    speciesResetRequested = Signal()

    """
    Mechanism editor with Reactions and Notes tabs.

    Features:
    - Reactions tab: DSL text editor (includes `# Algebra` sections for algebraic content)
    - Notes tab: persisted free-form text (never parsed or injected)

    Advanced features (Species Registry and State Network) are accessible via Edit menu.

    The reaction text editor supports debounced updates (500ms) to prevent
    lag during typing while maintaining live preview.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize mechanism editor.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._current_validation_state = "idle"
        self._reactions_edit_action: Optional[QtGui.QAction] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs)

        # Reactions tab - text editor with visual preview
        self._reactions_tab = QtWidgets.QWidget()
        reactions_layout = QtWidgets.QVBoxLayout(self._reactions_tab)

        # Text editor for DSL
        reactions_header_row = QtWidgets.QHBoxLayout()
        self._reactions_edit_btn = QtWidgets.QPushButton()
        self._reactions_edit_btn.setObjectName("reactionsEditGuardButton")
        self._reactions_edit_btn.setCheckable(True)
        self._reactions_edit_btn.setStyleSheet("QPushButton { padding: 2px 8px; }")
        self._reactions_edit_btn.hide()
        reactions_header_row.addWidget(self._reactions_edit_btn)
        reactions_header_row.addStretch()
        self._run_btn = QtWidgets.QPushButton("\u25b6 Run")
        run_font = self._run_btn.font()
        run_font.setPointSize(run_font.pointSize() + 2)
        self._run_btn.setFont(run_font)
        self._run_btn.setStyleSheet("QPushButton { padding: 6px 18px; }")
        self._run_btn.setEnabled(False)
        reactions_header_row.addWidget(self._run_btn)
        reactions_layout.addLayout(reactions_header_row)

        self._reactions_edit_status_label = QtWidgets.QLabel("")
        self._reactions_edit_status_label.setWordWrap(True)
        status_font = self._reactions_edit_status_label.font()
        status_font.setPointSize(max(1, status_font.pointSize() - 1))
        self._reactions_edit_status_label.setFont(status_font)
        reactions_layout.addWidget(self._reactions_edit_status_label)

        self._reactions_text = QtWidgets.QPlainTextEdit()
        self._reactions_text.setPlaceholderText(
            "Example:\n"
            "reaction: 2A + B => C ; kf=1e5\n"
            "equilibrium: C <=> D ; K=2.5\n"
            "reaction: C + E -> F ; kf=k_derived\n"
            "reaction: A -> A_Side ; kf=0.01\n"
            "\n"
            "# algebra\n"
            "param scale = 2.0\n"
            "param k_base = 1.5e3\n"
            "param k_derived = k_base * scale\n"
            "\n"
            "let total_A = [A] + [A_Side]\n"
            "let conversion = 1.0 - ([A] / max([A]_0, 1e-18))\n"
            "\n"
            "# === Computational Mode ===\n"
            "comp: species C G=-450.12\n"
        )
        self._reactions_text.setFont(QtGui.QFont("Courier New", 10))
        self._reactions_text.setUndoRedoEnabled(True)  # Enable undo/redo

        # Apply syntax highlighting
        self._reactions_highlighter = MechanismHighlighter(self._reactions_text.document())

        # Validation indicator (shows DSL parsing status)
        self._validation_label = QtWidgets.QLabel()
        self._validation_label.setStyleSheet("QLabel { padding: 4px; border-radius: 3px; }")
        self._validation_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._set_validation_state("idle")
        reactions_layout.addWidget(self._validation_label)

        # Unified slider surface for interactive parameter adjustment
        self._variable_sliders = VariableSliders(embedded=True)
        self._species_sliders = BatchSpeciesSliders(embedded=True)
        self._variable_sliders.contentStateChanged.connect(self._sync_slider_workspace_state)
        self._species_sliders.contentStateChanged.connect(self._sync_slider_workspace_state)
        self._slider_surface = make_scroll_area(self)
        self._slider_surface.setObjectName("unifiedSliderSurface")
        self._slider_surface_content = QtWidgets.QWidget(self._slider_surface)
        slider_surface_layout = QtWidgets.QVBoxLayout(self._slider_surface_content)
        slider_surface_layout.setContentsMargins(0, 0, 0, 0)
        slider_surface_layout.setSpacing(6)
        slider_surface_layout.addWidget(self._variable_sliders)
        slider_surface_layout.addWidget(self._species_sliders)
        slider_surface_layout.addStretch(1)
        self._slider_surface.setWidget(self._slider_surface_content)

        # Control row for slider settings
        slider_actions_layout = QtWidgets.QHBoxLayout()
        slider_actions_layout.setSpacing(8)

        # Fine mode toggle
        self._fine_btn = QtWidgets.QPushButton("Fine")
        self._fine_btn.setCheckable(True)
        self._fine_btn.toggled.connect(lambda v: self._variable_sliders.set_fine_mode(v))
        slider_actions_layout.addWidget(self._fine_btn)

        # Override mode controls (Commit/Reset)
        self._commit_slider_overrides_btn = QtWidgets.QPushButton("Commit")
        self._commit_slider_overrides_btn.setObjectName("commitSliderOverridesButton")
        slider_actions_layout.addWidget(self._commit_slider_overrides_btn)

        self._reset_slider_overrides_btn = QtWidgets.QPushButton("Reset")
        self._reset_slider_overrides_btn.setObjectName("resetSliderOverridesButton")
        slider_actions_layout.addWidget(self._reset_slider_overrides_btn)

        # Visibility picker for the unified slider surface.
        self._slider_visibility_picker_btn = QtWidgets.QToolButton()
        self._slider_visibility_picker_btn.setText("Visible sliders")
        self._slider_visibility_picker_btn.setObjectName("sliderVisibilityPickerButton")
        self._slider_visibility_picker_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._slider_visibility_menu = _PersistentToggleMenu(self)
        self._slider_visibility_menu.aboutToShow.connect(self._rebuild_slider_visibility_menu)
        self._slider_visibility_picker_btn.setMenu(self._slider_visibility_menu)
        slider_actions_layout.addWidget(self._slider_visibility_picker_btn)
        self._slider_edit_targets_label = QtWidgets.QLabel("Slider edit targets: none")
        target_font = self._slider_edit_targets_label.font()
        target_font.setPointSize(max(1, target_font.pointSize() - 1))
        self._slider_edit_targets_label.setFont(target_font)
        slider_actions_layout.addWidget(self._slider_edit_targets_label)
        slider_actions_layout.addStretch()

        slider_runtime_layout = QtWidgets.QHBoxLayout()
        slider_runtime_layout.setSpacing(8)

        # Slider simulation points
        slider_runtime_layout.addWidget(QtWidgets.QLabel("Points:"))
        self._slider_points_spin = QtWidgets.QSpinBox()
        self._slider_points_spin.setRange(50, 20000)
        self._slider_points_spin.setSingleStep(50)
        self._slider_points_spin.setValue(100)
        self._slider_points_spin.setMinimumWidth(100)
        slider_runtime_layout.addWidget(self._slider_points_spin)

        # Slider solver
        slider_runtime_layout.addWidget(QtWidgets.QLabel("Solver:"))
        self._slider_solver_combo = QtWidgets.QComboBox()
        self._slider_solver_combo.addItems(["LSODA", "Radau", "BDF"])
        self._slider_solver_combo.setCurrentText("LSODA")
        slider_runtime_layout.addWidget(self._slider_solver_combo)
        slider_runtime_layout.addStretch()

        # Bottom pane: slider header + controls + slider surface
        self._slider_pane_container = QtWidgets.QWidget()
        self._slider_pane_container.setObjectName("mechanismSliderPane")
        slider_pane_layout = QtWidgets.QVBoxLayout(self._slider_pane_container)
        slider_pane_layout.setContentsMargins(0, 0, 0, 0)
        slider_pane_layout.setSpacing(6)
        slider_pane_layout.addLayout(slider_actions_layout)
        slider_pane_layout.addLayout(slider_runtime_layout)
        self._slider_empty_state_label = QtWidgets.QLabel("Choose sliders from Visible sliders.")
        self._slider_empty_state_label.setWordWrap(True)
        self._slider_empty_state_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        empty_state_font = self._slider_empty_state_label.font()
        empty_state_font.setPointSize(max(1, empty_state_font.pointSize() - 1))
        self._slider_empty_state_label.setFont(empty_state_font)
        slider_pane_layout.addWidget(self._slider_empty_state_label, stretch=1)
        slider_pane_layout.addWidget(self._slider_surface, stretch=1)
        self._slider_workspace_state = (False, False)
        self._slider_workspace_detached = False
        self._slider_pane_container.hide()

        # Vertical splitter: DSL editor (top) | sliders (bottom)
        self._reactions_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._reactions_splitter.addWidget(self._reactions_text)
        self._reactions_splitter.addWidget(self._slider_pane_container)
        self._reactions_splitter.setCollapsible(0, False)
        self._reactions_splitter.setCollapsible(1, True)
        self._reactions_splitter.setStretchFactor(0, 4)
        self._reactions_splitter.setStretchFactor(1, 2)
        reactions_layout.addWidget(self._reactions_splitter, stretch=1)

        # Debounce timer for DSL validation (prevents lag during typing)
        self._validation_timer = QtCore.QTimer()
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(500)  # 500ms delay after last keystroke
        self._validation_timer.timeout.connect(self._validate_dsl)

        # Connect text editor to validation (debounced updates)
        self._reactions_text.textChanged.connect(self._on_text_changed)

        self._tabs.addTab(self._reactions_tab, "Reactions")

        # Help tab — example + directive/algebra/computational-mode reference
        self._help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(self._help_tab)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_scroll = make_scroll_area(self._help_tab)
        help_label = QtWidgets.QLabel(
            '<pre style="white-space: pre-wrap;"><b>Global DSL Directives &amp; Advanced Features</b>\n'
            "energy=kJ/mol  (Supported: kJ/mol, kcal/mol, J/mol)\n"
            "T=300          (Global isothermal temperature in K)\n"
            "[A]=1.0        (Hardcode initial conditions directly)\n"
            "\n"
            "<b>Algebra &amp; Observables (# algebra)</b>\n"
            "Use 'param name = expr' for static kinetic parameters (e.g., param k2 = k1 * 2).\n"
            "Use 'let name = expr' for observables using species data (e.g., let total = [A] + [B]_0).\n"
            "Math ops: +, -, *, /, ^, min(), max(), exp(), ln(), sin(), piecewise logic, and constants (R, T, N_A).\n"
            "*Note: Bracketed species [A] can ONLY be used in 'let', never in 'param'.*\n"
            "\n"
            "<b>Computational Mode (# === Computational Mode ===)</b>\n"
            "Define advanced species thermodynamics (e.g., comp: species A G=-100).\n"
            "Note: The 'hartree' energy unit is only supported within Computational Mode blocks.\n"
            "\n"
            "Note: Kindred normalizes all energies internally to J/mol.\n"
            "\n"
            "<b>Example</b>\n"
            "reaction: 2A + B =&gt; C ; kf=1e5\n"
            "equilibrium: C &lt;=&gt; D ; K=2.5\n"
            "reaction: C + E -&gt; F ; kf=k_derived\n"
            "reaction: A -&gt; A_Side ; kf=0.01\n"
            "\n"
            "# algebra\n"
            "param scale = 2.0\n"
            "param k_base = 1.5e3\n"
            "param k_derived = k_base * scale\n"
            "\n"
            "let total_A = [A] + [A_Side]\n"
            "let conversion = 1.0 - ([A] / max([A]_0, 1e-18))\n"
            "\n"
            "# === Computational Mode ===\n"
            "comp: species C G=-450.12</pre>"
        )
        help_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        help_label.setStyleSheet("padding: 8px;")
        help_scroll.setWidget(help_label)
        help_scroll.setWidgetResizable(True)
        help_layout.addWidget(help_scroll)
        self._tabs.addTab(self._help_tab, "Help")

        # Notes tab - persisted free-form text (never parsed/injected into DSL)
        self._notes_tab = QtWidgets.QWidget()
        notes_layout = QtWidgets.QVBoxLayout(self._notes_tab)

        notes_header = QtWidgets.QLabel("Notes (not parsed):")
        notes_layout.addWidget(notes_header)

        self._notes_text = QtWidgets.QPlainTextEdit()
        self._notes_text.setPlaceholderText(
            "Free-form notes about this mechanism.\n\n"
            "Important: algebraic scalars and algebraic observables must be defined in the\n"
            "Reactions editor inside a '# Algebra' section. Notes are never parsed or injected."
        )
        self._notes_text.setFont(QtGui.QFont("Courier New", 10))
        self._notes_text.setUndoRedoEnabled(True)
        notes_layout.addWidget(self._notes_text)
        self._tabs.addTab(self._notes_tab, "Notes")

        # Species and State Network tabs moved to Edit menu
        # Keep references for menu access but don't add to main tabs
        self._species_tab = None  # Created on demand from Edit menu
        # Keep for Edit menu access, but ensure it is not visible in the tabbed editor:
        # a visible, non-laid-out child widget will sit at (0, 0) and can intercept
        # clicks on the QTabBar (making it impossible to switch back to "Reactions").
        self._state_network_editor = StateNetworkEditor(parent=self)
        self._state_network_editor.hide()
        QtCore.QTimer.singleShot(0, self._sync_slider_workspace_state)

    def species_sliders_widget(self) -> BatchSpeciesSliders:
        return self._species_sliders

    def set_slider_edit_targets_summary(self, text: str) -> None:
        self._slider_edit_targets_label.setText(str(text or "Slider edit targets: none"))

    def detach_slider_pane_for_dock(self) -> QtWidgets.QWidget:
        if bool(self._slider_workspace_detached):
            return self._slider_pane_container

        splitter = getattr(self, "_reactions_splitter", None)
        if splitter is not None:
            self._slider_pane_container.hide()
            self._slider_pane_container.setParent(None)

        self._slider_workspace_detached = True
        self._sync_slider_workspace_state()
        return self._slider_pane_container

    def _sync_slider_workspace_state(self) -> None:
        has_variable_entries = self._variable_sliders.has_slider_entries()
        has_species_entries = self._species_sliders.has_slider_rows()
        has_available_entries = bool(has_variable_entries or has_species_entries)

        visible_variable_entries = self._variable_sliders.visible_slider_count()
        visible_species_entries = self._species_sliders.visible_row_count()
        has_visible_entries = bool(visible_variable_entries or visible_species_entries)

        self._variable_sliders.setVisible(visible_variable_entries > 0)
        self._species_sliders.setVisible(visible_species_entries > 0)
        self._slider_surface.setVisible(has_visible_entries)
        self._slider_empty_state_label.setVisible(has_available_entries and not has_visible_entries)
        self._slider_visibility_picker_btn.setEnabled(has_available_entries)
        self._fine_btn.setVisible(has_variable_entries)
        self._slider_pane_container.setVisible(has_available_entries)

        if has_available_entries and not has_visible_entries:
            if has_variable_entries and has_species_entries:
                helper = "Choose mechanism or concentration sliders from Visible sliders."
            elif has_variable_entries:
                helper = "Choose mechanism sliders from Visible sliders."
            else:
                helper = "Choose concentration sliders from Visible sliders."
            self._slider_empty_state_label.setText(helper)

        state = (has_available_entries, has_visible_entries)
        if self._slider_workspace_state == state:
            return

        self._slider_workspace_state = state
        if bool(self._slider_workspace_detached):
            return
        if has_visible_entries:
            self._reactions_splitter.setSizes([7, 3])
        elif has_available_entries:
            self._reactions_splitter.setSizes([11, 1])
        else:
            self._reactions_splitter.setSizes([1, 0])

    def _rebuild_slider_visibility_menu(self) -> None:
        menu = self._slider_visibility_menu
        menu.clear()

        mechanism_entries = list(self._variable_sliders.slider_picker_entries())
        species_entries = list(self._species_sliders.slider_picker_entries())
        if not mechanism_entries and not species_entries:
            action = menu.addAction("No sliders available")
            action.setEnabled(False)
            return

        if mechanism_entries:
            self._add_slider_visibility_entries(menu, title="Mechanism", entry_kind="mechanism", entries=mechanism_entries)
        if mechanism_entries and species_entries:
            menu.addSeparator()
        if species_entries:
            self._add_slider_visibility_entries(menu, title="Initial concentrations", entry_kind="species", entries=species_entries)

    def _add_slider_visibility_entries(
        self,
        menu: QtWidgets.QMenu,
        *,
        title: str,
        entry_kind: str,
        entries: list[tuple[str, str, bool]],
    ) -> None:
        header = menu.addAction(str(title))
        header.setEnabled(False)
        for key, label, visible in entries:
            action = menu.addAction(str(label))
            action.setCheckable(True)
            action.setChecked(bool(visible))
            action.setData((str(entry_kind), str(key)))
            action.toggled.connect(
                lambda checked, kind=str(entry_kind), item=str(key): self._set_slider_entry_visible(kind, item, checked)
            )

    def _set_slider_entry_visible(self, entry_kind: str, item_name: str, visible: bool) -> None:
        if str(entry_kind) == "mechanism":
            self._variable_sliders.set_variable_visible(str(item_name), bool(visible))
            return
        if str(entry_kind) == "species":
            self._species_sliders.set_species_visible(str(item_name), bool(visible))

    def _on_text_changed(self):
        """Handle text change event - trigger debounced validation."""
        self._validation_timer.start()
        self._set_validation_state("validating")

    def _validate_dsl(self):
        """Validate DSL text and update validation indicator."""
        text = self._reactions_text.toPlainText()

        # Skip validation if text is empty
        if not text.strip():
            self._set_validation_state("idle")
            return

        # Try to parse DSL
        try:
            from kindred.core.batch_initial_conditions import (
                strip_named_reaction_dsl_initial_concentration_sets,
            )
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism

            parse_text = strip_named_reaction_dsl_initial_concentration_sets(text)

            # Parse with empty initials (will be populated from DSL)
            mechanism = parse_dsl_to_mechanism(parse_text, initials={})

            # Success - show green check
            n_reactions = len(mechanism.reactions)
            n_equilibria = len(mechanism.equilibria)
            n_species = len(mechanism.species)

            msg = f"✓ Valid: {n_species} species, {n_reactions} reactions, {n_equilibria} equilibria"
            self._set_validation_state("valid", msg)

        except Exception as e:
            # Error - show red X with message
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            self._set_validation_state("invalid", f"✗ Error: {error_msg}")

    def _set_validation_state(self, state: str, message: str = ""):
        """
        Set validation indicator state.

        Parameters
        ----------
        state : str
            One of: "idle", "validating", "valid", "invalid"
        message : str
            Status message to display
        """
        self._current_validation_state = state
        self._run_btn.setEnabled(state == "valid")

        if state == "idle":
            self._validation_label.setText("")
            self._validation_label.setStyleSheet("QLabel { padding: 4px; }")

        elif state == "validating":
            self._validation_label.setText("\u23f3 Validating...")
            self._validation_label.setStyleSheet(
                "QLabel { padding: 4px; border-radius: 3px; }"
            )

        elif state == "valid":
            self._validation_label.setText(message)
            self._validation_label.setStyleSheet(
                "QLabel { padding: 4px; border-radius: 3px; }"
            )

        elif state == "invalid":
            self._validation_label.setText(message)
            self._validation_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; }"
            )

    def slider_points_value(self) -> int:
        """Get the current slider simulation points setting."""
        return self._slider_points_spin.value()

    def slider_solver_value(self) -> str:
        """Get the current slider solver setting."""
        return self._slider_solver_combo.currentText()

    def set_slider_points_value(self, n: int) -> None:
        """Set the slider simulation points value."""
        self._slider_points_spin.setValue(n)

    def set_slider_solver_value(self, s: str) -> None:
        """Set the slider solver value."""
        idx = self._slider_solver_combo.findText(s)
        self._slider_solver_combo.setCurrentIndex(idx if idx >= 0 else 0)

    @property
    def run_btn(self) -> QtWidgets.QPushButton:
        return self._run_btn

    def is_mechanism_valid(self) -> bool:
        return self._current_validation_state == "valid"

    def set_reactions_edit_action(self, action: Optional[QtGui.QAction]) -> None:
        if action is None:
            self._reactions_edit_btn.hide()
            return
        btn = self._reactions_edit_btn
        btn.setText(action.text())
        btn.setToolTip(action.toolTip())
        btn.setChecked(action.isChecked())
        if self._reactions_edit_action is not None:
            try:
                btn.toggled.disconnect()
            except (RuntimeError, TypeError):
                pass
            try:
                self._reactions_edit_action.changed.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._reactions_edit_action = action
        btn.toggled.connect(action.trigger)

        def _sync_from_action():
            prev = btn.blockSignals(True)
            try:
                btn.setText(action.text())
                btn.setToolTip(action.toolTip())
                btn.setChecked(action.isChecked())
            finally:
                btn.blockSignals(prev)

        action.changed.connect(_sync_from_action)
        btn.show()

    def set_reactions_edit_status_text(self, text: str) -> None:
        text_s = str(text)
        self._reactions_edit_status_label.setText(text_s)
        self._reactions_edit_status_label.setVisible(bool(text_s.strip()))

    def set_reactions_read_only(self, read_only: bool) -> None:
        self._reactions_text.setReadOnly(bool(read_only))

    def reactions_text(self) -> str:
        """Return the current Reaction DSL text."""
        return str(self._reactions_text.toPlainText())

    def set_reactions_text(self, text: str, *, block_signals: bool = False) -> None:
        """Replace the Reaction DSL text."""
        if not bool(block_signals):
            self._reactions_text.setPlainText(str(text))
            return
        self._reactions_text.blockSignals(True)
        try:
            self._reactions_text.setPlainText(str(text))
        finally:
            self._reactions_text.blockSignals(False)

    def state_network_dsl_raw(self) -> str:
        """Return the current raw state-network DSL (may be empty)."""
        return str(self._state_network_editor.get_state_network_dsl() or "")
