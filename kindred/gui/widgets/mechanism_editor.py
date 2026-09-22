# kindred/gui/widgets/mechanism_editor.py
"""Tabbed mechanism editor for Reactions and Notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

if TYPE_CHECKING:
    from kindred.core.mechanism_source import MechanismAuthoringSource

from kindred.gui.ui_helpers import (
    make_bounded_label,
    make_scroll_area,
    set_bounded_label_text,
)

from kindred.gui.display_name_policy import compact_text, literal_tooltip
from kindred.gui.widgets.mechanism_highlighter import MechanismHighlighter
from kindred.gui.widgets.variable_sliders import VariableSliders
from kindred.gui.widgets.species_sliders import BatchSpeciesSliders
__all__ = [
    "MechanismEditorBinding",
    "MechanismEditorTabbed",
]


@dataclass(frozen=True, slots=True)
class _MechanismEditorRenderProjection:
    """Prepared editor values for one admitted tab snapshot."""

    source: object
    notes: str
    reactions_draft: bool
    accepted_reactions: bool


def _make_sliders_icon(color: QtGui.QColor) -> QtGui.QIcon:
    """Paint the compact three-track glyph used by the Sliders button."""

    size = 18
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(
        QtGui.QPen(
            color,
            1.5,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
    )
    track_start = QtCore.QPointF(2.0, 0.0)
    track_end = QtCore.QPointF(16.0, 0.0)
    for y in (4.0, 9.0, 14.0):
        painter.drawLine(
            QtCore.QPointF(track_start.x(), y),
            QtCore.QPointF(track_end.x(), y),
        )

    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    for y, x in zip((4.0, 9.0, 14.0), (6.0, 12.0, 8.5)):
        painter.drawEllipse(QtCore.QRectF(x - 2.0, y - 2.0, 4.0, 4.0))
    painter.end()
    return QtGui.QIcon(pixmap)


class _PersistentToggleMenu(QtWidgets.QMenu):
    """Keep checkable actions open so users can toggle multiple sliders in one pass."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pressed_checkable_action: Optional[QtGui.QAction] = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        action = self.actionAt(event.position().toPoint())
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and action is not None
            and action.isEnabled()
            and action.isCheckable()
        ):
            self._pressed_checkable_action = action
            event.accept()
            return
        self._pressed_checkable_action = None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        action = self.actionAt(event.position().toPoint())
        pressed_action = self._pressed_checkable_action
        self._pressed_checkable_action = None
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and pressed_action is not None
            and action is pressed_action
            and action.isEnabled()
            and action.isCheckable()
        ):
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MechanismEditorTabbed(QtWidgets.QWidget):
    sourceDraftEdited = Signal(object)
    reactionsRevertRequested = Signal()
    notesEdited = Signal(str)
    sliderApplyRequested = Signal()
    sliderResetRequested = Signal()
    nativeHistoryAvailabilityChanged = Signal(bool, bool)

    """
    Mechanism editor with Reactions and Notes tabs.

    Features:
    - Reactions tab: DSL text editor for reactions and algebra declarations
    - Notes tab: persisted free-form text (never parsed or injected)

    The reaction text editor is directly editable. It emits visible authoring
    drafts; scientific action boundaries own reconciliation.
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
        self._revert_reactions_btn = QtWidgets.QPushButton("Revert", self)
        self._revert_reactions_btn.setObjectName("revertReactionsButton")
        self._revert_reactions_btn.setToolTip(
            "Restore Accepted Reactions as one undoable editor change"
        )
        self._revert_reactions_btn.setEnabled(False)
        self._revert_reactions_btn.clicked.connect(
            lambda _checked=False: self.reactionsRevertRequested.emit()
        )
        reactions_header_row.addWidget(self._revert_reactions_btn)
        self.symbolic_calculator_btn = QtWidgets.QPushButton("Σ", self)
        self.symbolic_calculator_btn.setAccessibleName("Open Symbolic Calculator")
        self.symbolic_calculator_btn.setObjectName("symbolicCalculatorOpenButton")
        self.symbolic_calculator_btn.setToolTip(
            "Show or hide the Symbolic Calculator panel."
        )
        self.symbolic_calculator_btn.setStyleSheet("QPushButton { padding: 6px 12px; }")
        self.symbolic_calculator_btn.setEnabled(False)
        reactions_header_row.addWidget(self.symbolic_calculator_btn)
        self.sliders_btn = QtWidgets.QPushButton(self)
        self.sliders_btn.setAccessibleName("Open Interactive Sliders panel")
        self.sliders_btn.setObjectName("slidersOpenButton")
        self.sliders_btn.setToolTip(
            "Show or hide the Interactive Sliders panel."
        )
        self.sliders_btn.setStyleSheet("QPushButton { padding: 6px 12px; }")
        self.sliders_btn.setIconSize(QtCore.QSize(18, 18))
        self._refresh_sliders_icon()
        reactions_header_row.addWidget(self.sliders_btn)
        self._run_btn = QtWidgets.QPushButton("Run")
        run_font = self._run_btn.font()
        run_font.setPointSize(run_font.pointSize() + 2)
        self._run_btn.setFont(run_font)
        self._run_btn.setStyleSheet("QPushButton { padding: 6px 18px; }")
        self._run_btn.setEnabled(False)
        reactions_header_row.addStretch(1)
        reactions_header_row.addWidget(self._run_btn)
        reactions_layout.addLayout(reactions_header_row)

        self._reactions_text = QtWidgets.QPlainTextEdit()
        self._reactions_text.setPlaceholderText(
            "Example:\n"
            "2A + B => C ; kf=1e5\n"
            "C <=> D ; Keq=2.5 ; kf=10.0\n"
            "C + E -> F ; k=0.5\n"
            "param scale = 2.0\n"
            "A -> A_Side ; kf=0.01\n"
            "let total_A = [A] + [A_Side]\n"
            "param k_base = 1.5e3\n"
            "param floor = 1e-18; unit = M\nlet conversion = 1.0 - ([A] / max([A]_0, floor))\n"
            "\n"
            "param k_derived = k_base * scale\n"
        )
        source_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        source_font.setFamilies(["Cascadia Mono", "Consolas", "DejaVu Sans Mono", source_font.family()])
        source_font.setPointSize(10)
        self._reactions_text.setFont(source_font)
        self._reactions_text.setUndoRedoEnabled(True)  # Enable undo/redo
        self._reactions_text.undoAvailable.connect(
            self._emit_native_history_availability
        )
        self._reactions_text.redoAvailable.connect(
            self._emit_native_history_availability
        )

        # Apply syntax highlighting
        self._reactions_highlighter = MechanismHighlighter(
            self._reactions_text.document()
        )

        # Validation indicator (shows DSL parsing status)
        self._validation_label = QtWidgets.QLabel()
        self._validation_label.setWordWrap(True)
        self._validation_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._validation_label.setStyleSheet(
            "QLabel { padding: 4px; border-radius: 3px; }"
        )
        self._validation_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._set_validation_state("idle")
        reactions_layout.addWidget(self._validation_label)
        # Unified slider surface for interactive parameter adjustment
        self._variable_sliders = VariableSliders(embedded=True)
        self._species_sliders = BatchSpeciesSliders()
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
        slider_actions_layout = QtWidgets.QGridLayout()
        slider_actions_layout.setSpacing(8)
        slider_actions_layout.setAlignment(QtCore.Qt.AlignTop)

        # Fine mode toggle
        self._fine_btn = QtWidgets.QPushButton("Fine")
        self._fine_btn.setCheckable(True)
        self._fine_btn.setToolTip(
            "Drag parameters and initial concentrations at one-tenth normal sensitivity "
            "with the cursor attached to the handle. Also use smaller arrow-key "
            "and Page Up/Down steps."
        )
        self._fine_btn.toggled.connect(
            lambda v: self._variable_sliders.set_fine_mode(v)
        )
        self._fine_btn.toggled.connect(self._species_sliders.set_fine_mode)
        self._variable_sliders.fineModeReset.connect(
            self._sync_fine_button_state
        )
        slider_actions_layout.addWidget(self._fine_btn, 0, 1, alignment=QtCore.Qt.AlignLeft)

        # Override mode controls (Commit/Reset)
        self._commit_runtime_parameters_btn = QtWidgets.QPushButton("Apply")
        self._commit_runtime_parameters_btn.setObjectName(
            "commitRuntimeParametersButton"
        )
        # A mouse Apply accepts the Preview, not the focused editor's draft.
        self._commit_runtime_parameters_btn.setFocusPolicy(QtCore.Qt.TabFocus)
        self._commit_runtime_parameters_btn.setToolTip(
            "Accept the exact Preview, replacing pending Reactions text and "
            "initial-condition drafts for the applied sets. Reactions Undo "
            "can recover replaced text."
        )
        self._commit_runtime_parameters_btn.clicked.connect(self.sliderApplyRequested)
        slider_actions_layout.addWidget(self._commit_runtime_parameters_btn, 0, 2, alignment=QtCore.Qt.AlignLeft)

        self._reset_runtime_parameters_btn = QtWidgets.QPushButton("Reset")
        self._reset_runtime_parameters_btn.setObjectName("resetRuntimeParametersButton")
        self._reset_runtime_parameters_btn.setToolTip(
            "Reset sliders to the accepted model values."
        )
        self._reset_runtime_parameters_btn.clicked.connect(self.sliderResetRequested)
        slider_actions_layout.addWidget(self._reset_runtime_parameters_btn, 0, 3, alignment=QtCore.Qt.AlignLeft)
        self.set_slider_draft_actions_enabled(False, False)

        # Visibility picker for the unified slider surface.
        self._slider_visibility_picker_btn = QtWidgets.QToolButton()
        self._slider_visibility_picker_btn.setText("Visible sliders")
        self._slider_visibility_picker_btn.setObjectName("sliderVisibilityPickerButton")
        self._slider_visibility_picker_btn.setToolTip(
            "Choose which parameters and initial concentrations appear as sliders."
        )
        self._slider_visibility_picker_btn.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self._slider_visibility_menu = _PersistentToggleMenu(self)
        self._slider_visibility_menu.aboutToShow.connect(
            self._rebuild_slider_visibility_menu
        )
        self._slider_visibility_picker_btn.setMenu(self._slider_visibility_menu)
        slider_actions_layout.addWidget(self._slider_visibility_picker_btn, 0, 0, alignment=QtCore.Qt.AlignLeft)
        slider_actions_layout.setColumnStretch(4, 1)
        self._slider_edit_targets_label = make_bounded_label(
            "Initial-condition sets: none",
            max_width=16777215,
        )
        self._slider_edit_targets_label.setToolTip(
            "Concentration sliders affect the current set and any pinned sets."
        )
        target_font = self._slider_edit_targets_label.font()
        target_font.setPointSize(max(1, target_font.pointSize() - 1))
        self._slider_edit_targets_label.setFont(target_font)
        slider_actions_layout.addWidget(self._slider_edit_targets_label, 1, 0, 1, 5)

        # Bottom pane: slider header + controls + slider surface
        self._slider_pane_container = QtWidgets.QWidget()
        self._slider_pane_container.setObjectName("mechanismSliderPane")
        slider_pane_layout = QtWidgets.QVBoxLayout(self._slider_pane_container)
        slider_pane_layout.setContentsMargins(0, 0, 0, 0)
        slider_pane_layout.setSpacing(6)
        slider_pane_layout.addLayout(slider_actions_layout)
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

        # Text changes capture literal drafts only. Scientific admission remains
        # owned by explicit Run/Preview/Fit actions.
        self._reactions_text.textChanged.connect(self._on_text_changed)

        self._tabs.addTab(self._reactions_tab, "Reactions")

        # Help tab — current directive and algebra reference
        self._help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(self._help_tab)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_browser = QtWidgets.QTextBrowser(self._help_tab)
        help_browser.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        help_browser.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        help_browser.document().setDocumentMargin(12)
        help_browser.document().setDefaultStyleSheet(
            "h2 { font-size: 115%; margin-top: 24px; margin-bottom: 10px; }"
            "p { margin-top: 8px; margin-bottom: 8px; }"
            "pre { white-space: pre-wrap; margin: 12px 0px 12px 12px; }"
            "li { margin-top: 4px; margin-bottom: 4px; }"
        )
        help_browser.setHtml(
            "<h2>Writing a Mechanism</h2>"
            "<p>Enter reactions, initial concentrations, parameters, and schedules in <b>Reactions</b>.</p>"
            "<pre>A -&gt; B ; k=0.5</pre>"
            "<h2>Global Settings</h2>"
            "<p>Use these optional settings to specify temperature, reaction energy units, and standard concentration.</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Setting</th><th>Meaning</th></tr>"
            "<tr><td><code>T=</code></td><td>Nominal temperature, in K. Used throughout the simulation unless a temperature schedule is specified.</td></tr>"
            "<tr><td><code>energy=</code></td><td>Unit for <code>Ea</code>, <code>dG_act</code>, and <code>dG_eq</code>. Options: <code>J/mol</code>, <code>kJ/mol</code>, <code>kcal/mol</code>.</td></tr>"
            "<tr><td><code>C0=</code></td><td>Standard concentration, in M. Must be greater than zero.</td></tr>"
            "</table>"
            "<pre>T=298.15\nenergy=kJ/mol\nC0=1.0</pre>"
            "<p>Enter each setting at most once. Omitted settings use the values in <b>Simulation Settings</b>. Enter <code>T=</code> and <code>energy=</code> on separate lines from reactions.</p>"
            "<h2>Initial Conditions</h2>"
            "<p>Enter initial concentrations in M. These forms are equivalent:</p>"
            "<pre>[A] = 1.0, [B] = 0.0\ninit: A=1.0, B=0.0\ninitial: A=1.0, B=0.0</pre>"
            "<p>Concentrations must be finite and nonnegative. To name an initial-condition set, enclose its concentrations in braces:</p>"
            "<pre>start = {\n[A] = 1.0\n[B] = 0.0\n}</pre>"
            "<h2>Reaction Arrows</h2>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Reaction</th><th>Arrow options</th></tr>"
            "<tr><td>Irreversible</td><td><code>-&gt;</code> or <code>=&gt;</code></td></tr>"
            "<tr><td>Reversible</td><td><code>&lt;-&gt;</code> or <code>&lt;=&gt;</code></td></tr>"
            "</table>"
            "<p><code>&lt;-</code> alone is not accepted. Write stoichiometric coefficients as <code>2B</code> or <code>2*B</code>.</p>"
            "<pre>A + 2B -&gt; C ; k=0.5</pre>"
            "<h2>Reaction Parameters</h2>"
            "<p>Add reaction parameters after the semicolon. Choose one forward rate model per reaction:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Forward rate model</th><th>Required fields</th></tr>"
            "<tr><td>Constant</td><td><code>k=</code> or <code>kf=</code></td></tr>"
            "<tr><td>Arrhenius</td><td><code>A=</code> and <code>Ea=</code></td></tr>"
            "<tr><td>Eyring</td><td><code>dG_act=</code></td></tr>"
            "</table>"
            "<p>Constant forward rate constants may be zero. For reversible reactions, the zero-rate restrictions are listed below.</p>"
            "<p>Eyring reactions also accept these optional fields:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Field</th><th>Meaning</th><th>Default</th></tr>"
            "<tr><td><code>kappa=</code></td><td>Transmission coefficient, greater than zero</td><td><code>1</code></td></tr>"
            "<tr><td><code>degeneracy=</code></td><td>Number of equivalent pathways, a positive integer</td><td><code>1</code></td></tr>"
            "</table>"
            "<pre>A -&gt; B ; dG_act=55 ; degeneracy=2</pre>"
            "<p><code>degeneracy</code> multiplies the forward rate. Use it only when the entered barrier does not already account for equivalent pathways.</p>"
            "<h2>Reversible Reactions</h2>"
            "<p>Use a reversible arrow, specify the forward rate under <b>Reaction Parameters</b>, and choose exactly one option:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Option</th><th>Meaning</th></tr>"
            "<tr><td><code>kr=</code></td><td>Reverse rate constant</td></tr>"
            "<tr><td><code>Keq=</code></td><td>Concentration-based equilibrium constant</td></tr>"
            "<tr><td><code>dG_eq=</code></td><td>Reaction Gibbs energy, entered in the global <code>energy=</code> unit</td></tr>"
            "</table>"
            "<pre>A &lt;=&gt; B ; kf=10.0 ; Keq=2.0</pre>"
            "<p>With <code>Keq=</code> or <code>dG_eq=</code>, Kindred calculates the reverse rate constant using:</p>"
            "<p><i>k</i><sub>r</sub> = <i>k</i><sub>f</sub> / <i>K</i><sub>eq</sub></p>"
            "<p>For <code>dG_eq=</code>:</p>"
            "<p><i>K</i><sub>eq</sub> = exp[−Δ<i>G</i><sub>eq</sub> / (<i>R</i><i>T</i>)] <i>C</i><sub>0</sub>^(<i>n</i><sub>r</sub> − <i>n</i><sub>f</sub>)</p>"
            "<p><i>n</i><sub>f</sub> and <i>n</i><sub>r</sub> are the forward and reverse reaction orders. <i>K</i><sub>eq</sub> has units M^(<i>n</i><sub>r</sub> − <i>n</i><sub>f</sub>). <i>C</i><sub>0</sub> is the standard concentration. <i>R</i> is the gas constant and <i>T</i> is the temperature at simulation time <i>t</i>.</p>"
            "<p>With <code>Keq=</code> or <code>dG_eq=</code>, <code>kf=0</code> disables both directions. With an explicit <code>kr=</code>, both directional rate constants must be positive.</p>"
            "<h2>Temperature Schedules</h2>"
            "<p>Use one <code>temp_step</code> or <code>temp_response</code> declaration. Enter times and <code>tau</code> in seconds, and temperatures in kelvin.</p>"
            "<p>List at least two times in increasing order, without repeats. Times must be zero or greater. Supply one temperature for each interval between consecutive times.</p>"
            "<h3>Step schedule</h3>"
            "<pre>temp_step: t=[0,50,100], T=[298,320]</pre>"
            "<p>Each listed temperature applies immediately at the start of its interval:</p>"
            "<p><i>T</i>(<i>t</i>) = <i>T</i><sub><i>i</i></sub>,&nbsp;&nbsp; <i>t</i><sub><i>i</i></sub> ≤ <i>t</i> &lt; <i>t</i><sub><i>i</i>+1</sub></p>"
            "<p><i>t</i><sub><i>i</i></sub> is the interval’s start time and <i>T</i><sub><i>i</i></sub> is its listed temperature.</p>"
            "<h3>Response schedule</h3>"
            "<pre>temp_response: t=[0,50,100], T=[298,320], tau=10</pre>"
            "<p>The temperatures in <code>T=[...]</code> are setpoints. Within each interval:</p>"
            "<p><i>T</i>(<i>t</i>) = <i>T</i><sub>set,<i>i</i></sub> + [<i>T</i>(<i>t</i><sub><i>i</i></sub>) − <i>T</i><sub>set,<i>i</i></sub>] exp[−(<i>t</i> − <i>t</i><sub><i>i</i></sub>)/<i>τ</i>]</p>"
            "<p><i>T</i>(<i>t</i><sub><i>i</i></sub>) is the temperature at the interval’s start and <i>T</i><sub>set,<i>i</i></sub> is its setpoint. The time constant <i>τ</i> is entered as <code>tau</code> and must be positive.</p>"
            "<p>The first interval starts from the nominal temperature specified by <code>T=</code> or <b>Simulation Settings</b>. Each later interval starts from the temperature reached at the end of the previous interval.</p>"
            "<h3>Before and after the listed times</h3>"
            "<p>Before the first listed time, Kindred uses the nominal temperature. After the last listed time, a step schedule keeps its final temperature. A response schedule continues toward its final setpoint.</p>"
            "<h2>Interventions</h2>"
            "<p>Use <code>intervention:</code> to change or maintain a species concentration. Choose an <code>op=</code> option:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Option</th><th>Effect</th></tr>"
            "<tr><td><code>set</code></td><td>Sets the species concentration to <code>value</code> at <code>time</code></td></tr>"
            "<tr><td><code>jump</code></td><td>Adds <code>delta</code> to the species concentration at <code>time</code></td></tr>"
            "<tr><td><code>flux</code></td><td>Adds or removes the species at <code>flux_rate</code>, in M/s, from <code>start_time</code> to <code>end_time</code></td></tr>"
            "<tr><td><code>hold</code></td><td>Maintains the species concentration at <code>value</code> from <code>start_time</code> to <code>end_time</code></td></tr>"
            "</table>"
            "<pre>intervention: op=set; species=A; time=1.0; value=2.0\nintervention: op=jump; species=A; time=2.0; delta=-0.25\nintervention: op=flux; species=B; start_time=0.0; end_time=5.0; flux_rate=-0.1\nintervention: op=hold; species=B; start_time=0.0; end_time=10.0; value=1.0</pre>"
            "<p>Enter times in seconds, concentration values and changes in M, and <code>flux_rate</code> in M/s. Separate intervention fields with semicolons.</p>"
            "<h2>Solutions, Dilution and Aliquots</h2>"
            "<p>Use <code>solution={...}</code> with <code>op=jump</code> to add a solution or with <code>op=flux</code> to supply a continuous inflow. "
            "Use <code>op=scale</code> to change all concentrations by a factor and <code>op=withdraw</code> to remove an aliquot.</p>"
            "<h3>Initial volume</h3>"
            "<p>Solution additions, inflows, and withdrawals require one <code>initial_volume</code> or <code>initial_volume_param</code> declaration. "
            "Place it on one of these actions, outside <code>solution={...}</code>. For a repeating or triggered action, place it inside that action.</p>"
            "<p>The declaration specifies the volume present at simulation start, even if the action containing it occurs later or never occurs. "
            "Kindred evaluates <code>initial_volume_param</code> once at the starting temperature.</p>"
            "<p>Declare the initial volume only once in the mechanism. Multiple declarations are rejected even if their values agree. "
            "A declaration inside a repeating action counts once.</p>"
            "<p>Every action uses the current volume. Scaling before the action containing <code>initial_volume</code> also changes that volume. "
            "Ordinary concentration interventions do not change volume. Models using only concentration interventions do not require a volume declaration.</p>"
            "<h3>Operations</h3>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Operation</th><th>Required fields</th><th>Behavior</th></tr>"
            "<tr><td>Solution addition</td><td><code>op=jump</code>, <code>time</code>, <code>solution={volume=...,stock={...}}</code></td>"
            "<td>Adds the specified volume and mixes the solution with the vessel contents</td></tr>"
            "<tr><td>Solution inflow</td><td><code>op=flux</code>, <code>start_time</code>, <code>end_time</code>, <code>solution={volume_rate=...,stock={...}}</code></td>"
            "<td>Supplies the solution continuously during the interval</td></tr>"
            "<tr><td>Scale</td><td><code>op=scale</code>, <code>time</code>, <code>factor</code></td>"
            "<td>Multiplies concentrations by the factor and divides volume by the same factor</td></tr>"
            "<tr><td>Withdrawal</td><td><code>op=withdraw</code>, <code>time</code>, <code>volume</code></td>"
            "<td>Removes the specified volume without changing concentrations</td></tr>"
            "</table>"
            "<pre>intervention: op=jump; time=1; initial_volume=10 mL; solution={volume=2 mL,stock={species=A,value=3 M}}\nintervention: op=flux; start_time=2; end_time=5; solution={volume_rate=1 mL/min,stock={}}\nintervention: op=scale; time=6; factor=0.5\nintervention: op=withdraw; time=7; volume=2 mL</pre>"
            "<h3>Stock composition</h3>"
            "<p>Include <code>stock={...}</code> in every solution specification. Each stock entry contains <code>species</code> and <code>value</code> or <code>value_param</code>. "
            "Separate multiple entries with semicolons:</p>"
            "<pre>stock={species=A,value=3 M; species=B,value=2 mM}</pre>"
            "<p>An omitted species has zero concentration in the incoming solution. Use <code>stock={}</code> for pure solvent. "
            "With a solution specification, omit the ordinary <code>species</code> and <code>delta</code> or <code>flux_rate</code> fields.</p>"
            "<h3>Scaling and withdrawal</h3>"
            "<p>Use a positive <code>factor</code>. A factor of <code>0.5</code> halves concentrations and doubles volume. "
            "A factor of <code>2</code> doubles concentrations and halves volume. Without an initial-volume declaration, scaling changes concentrations only.</p>"
            "<p>Once a <code>hold</code> takes effect, its prescribed concentration remains unchanged by other interventions until the hold ends. "
            "The order under <b>Actions at the Same Time</b> determines when a new hold takes effect.</p>"
            "<p>A withdrawal must leave a positive volume in the vessel.</p>"
            "<h3>Units and parameters</h3>"
            "<p>Enter volume and flow values with explicit units, such as <code>mL</code>, <code>uL</code>, and <code>mL/min</code>. "
            "Stock concentrations use M when no unit is given. Other concentration units, such as <code>mM</code>, are accepted. Enter <code>factor</code> without a unit.</p>"
            "<p>For an independent parameter, enter its value in the unit declared with <code>unit=</code>. "
            "Use the corresponding <code>_param</code> field in the intervention. Kindred converts the value automatically.</p>"
            "<pre>param dose = 2; unit = mL\nintervention: op=jump; time=1; initial_volume=10 mL; solution={volume_param=dose,stock={}}</pre>"
            "<p>For temperature-dependent parameters, Kindred evaluates addition volumes, withdrawal volumes, scale factors, and stock concentrations when the action occurs. "
            "Inflow rates and stock concentrations follow the temperature throughout the inflow interval. The initial volume is evaluated only at simulation start.</p>"
            "<h2>Using Parameters in Interventions</h2>"
            "<p>To use a parameter instead of entering a number, use the corresponding <code>_param</code> field:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Enter a number</th><th>Use a parameter</th></tr>"
            "<tr><td><code>time</code></td><td><code>time_param</code></td></tr>"
            "<tr><td><code>start_time</code></td><td><code>start_time_param</code></td></tr>"
            "<tr><td><code>end_time</code></td><td><code>end_time_param</code></td></tr>"
            "<tr><td><code>value</code></td><td><code>value_param</code></td></tr>"
            "<tr><td><code>delta</code></td><td><code>delta_param</code></td></tr>"
            "<tr><td><code>flux_rate</code></td><td><code>flux_rate_param</code></td></tr>"
            "<tr><td><code>threshold</code></td><td><code>threshold_param</code></td></tr>"
            "<tr><td><code>volume</code></td><td><code>volume_param</code></td></tr>"
            "<tr><td><code>volume_rate</code></td><td><code>volume_rate_param</code></td></tr>"
            "<tr><td><code>initial_volume</code></td><td><code>initial_volume_param</code></td></tr>"
            "<tr><td><code>factor</code></td><td><code>factor_param</code></td></tr>"
            "</table>"
            "<p>Use one field from each applicable pair. The parameter value must meet the restrictions for that field. "
            "The <code>value</code> and <code>value_param</code> pair also applies to stock concentrations. "
            "Enter <code>every</code>, <code>count</code>, <code>duration</code>, <code>max_count</code>, and <code>min_interval</code> as numbers.</p>"
            "<pre>param pulse_time = 2.0; unit = s\nparam pulse_delta = 0.2; unit = M\nintervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_delta</pre>"
            "<p>Times supplied by parameters are calculated before the simulation using the nominal temperature and remain fixed during the run.</p>"
            "<p>Amounts for <code>set</code> and <code>jump</code> are calculated when the action occurs. Parameter-based <code>flux</code>, <code>hold</code>, and threshold values follow temperature-dependent parameter changes during the simulation.</p>"
            "<h2>Repeating Interventions</h2>"
            "<p>Use <code>op=protocol; kind=repeat</code> to repeat a group of interventions at regular time intervals.</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Field</th><th>Meaning</th></tr>"
            "<tr><td><code>start_time</code> or <code>start_time_param</code></td><td>Start of the first repetition, in seconds</td></tr>"
            "<tr><td><code>every</code></td><td>Time between successive starts, in seconds</td></tr>"
            "<tr><td><code>count</code></td><td>Number of repetitions</td></tr>"
            "<tr><td><code>duration</code></td><td>Duration of each repetition, in seconds</td></tr>"
            "</table>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Group</th><th>When it acts</th><th>Permitted operations</th></tr>"
            "<tr><td><code>before</code></td><td>At the start of each repetition</td><td><code>set</code>, <code>jump</code>, <code>scale</code>, <code>withdraw</code></td></tr>"
            "<tr><td><code>during</code></td><td>Throughout each repetition</td><td><code>flux</code>, <code>hold</code></td></tr>"
            "<tr><td><code>after</code></td><td>At the end of each repetition</td><td><code>set</code>, <code>jump</code>, <code>scale</code>, <code>withdraw</code></td></tr>"
            "</table>"
            "<p>Include at least one group. Include <code>duration</code> when using <code>during</code> or <code>after</code>. Omit it when using only <code>before</code>.</p>"
            "<pre>intervention: op=protocol; kind=repeat; start_time=0.5; every=1.25; count=4; before={op=jump,species=A,delta=0.2}\nintervention: op=protocol; kind=repeat; start_time=0.0; every=2.0; count=3; duration=1.0; during={op=hold,species=B,value=1.0}; after={op=set,species=B,value=0.0}</pre>"
            "<p>Enclose each group in braces. Separate actions with semicolons and fields within an action with commas. Actions run in the order written. "
            "Solution and stock records use nested braces. Do not put timing fields, protocols, or triggers inside an action group.</p>"
            "<pre>before={op=jump,species=A,delta=0.1; op=jump,species=B,delta=0.2}</pre>"
            "<h2>Threshold Triggers</h2>"
            "<p>A trigger performs one action when a watched quantity crosses a threshold. Triggers that perform <code>set</code>, <code>jump</code>, <code>scale</code>, or <code>withdraw</code> watch a species concentration. Stopping triggers can also watch a calculated quantity defined with <code>let</code> that meets the stopping restrictions below.</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Field</th><th>Meaning</th></tr>"
            "<tr><td><code>watch_species</code></td><td>Name of the species to watch</td></tr>"
            "<tr><td><code>threshold</code> or <code>threshold_param</code></td><td>Concentration threshold in M, entered directly or supplied by a parameter</td></tr>"
            "<tr><td><code>max_count</code></td><td>Maximum number of times this trigger can act</td></tr>"
            "<tr><td><code>min_interval</code></td><td>Minimum time between its actions, in seconds. Use <code>0</code> for no additional waiting period.</td></tr>"
            "<tr><td><code>then</code></td><td>One <code>set</code>, <code>jump</code>, <code>scale</code>, or <code>withdraw</code> action, enclosed in braces</td></tr>"
            "</table>"
            "<p><code>max_count</code> and <code>min_interval</code> are required for triggers that perform <code>set</code>, <code>jump</code>, <code>scale</code>, or <code>withdraw</code>.</p>"
            "<pre>intervention: op=trigger; watch_species=A; threshold=0.8; direction=falling; max_count=1; min_interval=0.0; then={op=jump,species=B,delta=0.1}</pre>"
            "<p>The optional <code>direction=</code> field accepts these values for both action and stopping triggers:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Value</th><th>Trigger condition</th></tr>"
            "<tr><td><code>rising</code></td><td>The watched value reaches or crosses the threshold from below</td></tr>"
            "<tr><td><code>falling</code></td><td>The watched value reaches or crosses the threshold from above</td></tr>"
            "<tr><td><code>either</code></td><td>Either direction. Used when <code>direction</code> is omitted.</td></tr>"
            "</table>"
            "<p>Starting at or beyond the threshold does not trigger an action. A crossing must occur during the simulation.</p>"
            "<p>The action occurs at the threshold crossing. Do not include <code>time</code> or <code>time_param</code> inside <code>then</code>. Separate its fields with commas.</p>"
            "<p>Triggers that perform <code>set</code>, <code>jump</code>, <code>scale</code>, or <code>withdraw</code> respond to crossings during the simulated reaction. Intervention actions and temperature steps do not activate these triggers immediately, but they can trigger stopping.</p>"
            "<h2>Stopping Conditions</h2>"
            "<p>Use <code>then={op=stop}</code> to finish the simulation when a threshold is crossed and keep the results up to that time.</p>"
            "<p>Enable <b>Use authored stopping interventions</b> in <b>Simulation Settings</b>. This setting applies to Run and Preview for all initial-condition sets, including those with custom Run settings. When it is off, stopping conditions are ignored. The configured end time remains the maximum simulation time.</p>"
            "<pre>intervention: op=trigger; watch_species=B; threshold=0.9; direction=rising; then={op=stop}</pre>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Field</th><th>Choice</th></tr>"
            "<tr><td>Quantity to watch</td><td><code>watch_species</code> for a species or <code>watch_observable</code> for the name of an existing <code>let</code> definition. Choose one.</td></tr>"
            "<tr><td>Threshold</td><td><code>threshold</code> for a number or <code>threshold_param</code> for a parameter name. Choose one.</td></tr>"
            "<tr><td><code>direction</code></td><td>Optional. Use one of the values listed under <b>Threshold Triggers</b>.</td></tr>"
            "<tr><td><code>then</code></td><td><code>{op=stop}</code>, with no additional fields</td></tr>"
            "</table>"
            "<p>Stopping triggers do not accept <code>max_count</code> or <code>min_interval</code>.</p>"
            "<p>For a concentration threshold, enter <code>threshold</code> in M. For 1 mM, write <code>threshold=0.001</code>, including when the watched quantity has <code>; unit = mM</code>.</p>"
            "<pre>param floor = 1e-12; unit = M\nlet selectivity = [B] / max([A] + [B], floor)\nintervention: op=trigger; watch_observable=selectivity; threshold=0.9; direction=rising; then={op=stop}</pre>"
            "<p>A stopping condition produces a completed result. The toolbar <b>Stop</b> button cancels a Run instead.</p>"
            "<h3>Stopping order and output</h3>"
            "<ul>"
            "<li>When a stopping crossing is detected, the simulation ends before any remaining actions at that time.</li>"
            "<li>Kindred checks stopping conditions after each intervention action. An action that triggers stopping remains applied. Remaining actions are skipped.</li>"
            "<li>A temperature step can trigger stopping. If stopping occurs before the step, the result retains the preceding temperature. If the step triggers stopping, the result retains the changed temperature.</li>"
            "<li>Output samples span the completed simulation interval. Interventions may add rows. Values are not interpolated across concentration jumps. Stopping at the initial time produces one row.</li>"
            "<li>Results record the final time and the reason the simulation ended. Reaching the configured end time also completes normally.</li>"
            "<li>Crossings between solver steps may go undetected.</li>"
            "</ul>"
            "<h3>Calculated quantities in stopping conditions</h3>"
            "<p>A watched <code>let</code> definition may use concentrations at simulation time <i>t</i>, initial concentrations, <code>t</code>, <code>T</code>, parameters, rate constants, and other <code>let</code> definitions. Parts of its formula that change during the simulation have these restrictions:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Operation</th><th>Supported forms</th></tr>"
            "<tr><td>Arithmetic</td><td>Addition (<code>a + b</code>), subtraction (<code>a - b</code>), multiplication (<code>a * b</code>), and changing a value’s sign (<code>-a</code>). A leading <code>+</code> is also accepted.</td></tr>"
            "<tr><td>Powers</td><td><code>a^n</code>, <code>a**n</code>, or <code>pow(a, n)</code>. The exponent must be a whole number, zero or greater, and remain constant during the simulation.</td></tr>"
            "<tr><td>Division</td><td>The denominator must either remain constant and nonzero, or use <code>max(expression, positive_limit)</code> or <code>min(expression, negative_limit)</code>. The limit must remain constant and appear second. A separate <code>let</code> definition may supply the denominator.</td></tr>"
            "<tr><td>Functions</td><td><code>abs</code>, <code>min</code>, <code>max</code>, <code>clip</code>, <code>exp</code>, <code>expm1</code>, <code>sin</code>, <code>cos</code>, <code>erf</code></td></tr>"
            "</table>"
            "<p>Changing parts of a stopping formula cannot use comparisons, <code>&amp;&amp;</code>, <code>||</code>, <code>!</code>, <code>ifelse</code>, <code>heaviside</code>, <code>sqrt</code>, <code>ln</code>, <code>log10</code>, <code>log1p</code>, or <code>tan</code>.</p>"
            "<p>Parts of the formula that remain constant during the simulation may use the functions listed under <b>Functions</b>. All calculated values must be finite. These restrictions apply to quantities used in stopping conditions, including any other <code>let</code> definitions they use.</p>"
            "<h2>Intervention Input Limits</h2>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Input</th><th>Allowed values</th></tr>"
            "<tr><td>Times and <code>min_interval</code></td><td>Finite and nonnegative. <code>end_time</code> must be greater than <code>start_time</code>.</td></tr>"
            "<tr><td>Species thresholds, <code>set</code> values, and <code>hold</code> values</td><td>Finite and nonnegative, within the species concentration limits</td></tr>"
            "<tr><td>Calculated-quantity thresholds, concentration changes (<code>delta</code>), and concentration rates (<code>flux_rate</code>)</td><td>Finite positive, negative, or zero values</td></tr>"
            "<tr><td><code>initial_volume</code></td><td>10⁻³⁰ through 10³⁰ m³, inclusive</td></tr>"
            "<tr><td>Added or withdrawn volume</td><td>Zero or 10⁻³⁰ through 10³⁰ m³, inclusive</td></tr>"
            "<tr><td>Solution inflow rate</td><td>Zero or 10⁻³⁰ through 10³⁰ m³/s, inclusive</td></tr>"
            "<tr><td><code>factor</code></td><td>Finite and greater than zero</td></tr>"
            "<tr><td>Stock concentration</td><td>Finite and nonnegative</td></tr>"
            "<tr><td><code>every</code> and <code>duration</code></td><td>Greater than zero</td></tr>"
            "<tr><td><code>count</code> and <code>max_count</code></td><td>Positive integers, entered without a decimal point</td></tr>"
            "</table>"
            "<p>Kindred checks volume and inflow limits after converting your input units, including values supplied by parameters. "
            "The vessel volume may grow beyond these input limits during the simulation, but it must remain positive and finite. "
            "A zero inflow adds no solution. Reactions continue normally.</p>"
            "<p>A negative <code>jump</code> cannot reduce the concentration below zero. A negative <code>flux</code> removes the species without producing a negative concentration.</p>"
            "<h2>Actions at the Same Time</h2>"
            "<p>When interventions act at the same time, Kindred applies this order:</p>"
            "<ol>"
            "<li>End the <code>flux</code> and <code>hold</code> intervals that finish at this time.</li>"
            "<li>Apply scheduled <code>set</code>, <code>jump</code>, <code>scale</code>, and <code>withdraw</code> actions, including repeating <code>before</code> actions, in the order written in the mechanism.</li>"
            "<li>Start new <code>hold</code> intervals.</li>"
            "<li>Apply triggered <code>set</code>, <code>jump</code>, <code>scale</code>, and <code>withdraw</code> actions in the order their triggers are written.</li>"
            "<li>Apply repeating <code>after</code> actions in the order written.</li>"
            "</ol>"
            "<p>A detected stopping crossing interrupts this order as described under <b>Stopping order and output</b>.</p>"
            "<p>Only one <code>hold</code> per species may be active at a time. Overlapping holds are rejected before the simulation, even if their values agree or an earlier stop would skip them. One hold may start when another ends. Different species may be held simultaneously.</p>"
            "<p>For the same species at the same time, Kindred rejects different <code>set</code> values, a <code>set</code> combined with a <code>jump</code>, a <code>jump</code> when a <code>hold</code> starts, or a <code>set</code> that disagrees with a starting <code>hold</code>.</p>"
            "<p>Kindred also rejects a <code>set</code> at the same time as a solution addition or <code>scale</code> action.</p>"
            "<h2>Inspecting Interventions</h2>"
            "<p>Open <b>Simulation → Inspect Mechanism…</b> to view reaction equations and intervention details.</p>"
            "<p>To change volume display units, right-click the plot and choose <b>Appearance → Volume Unit</b>. "
            "The default, <b>Authored</b>, uses the unit in <code>initial_volume</code> or its parameter declaration. "
            "Kindred uses L if that declaration has no compatible volume unit. "
            "Your selection applies to volume axes and intervention labels. Inflow labels use the same volume unit divided by the selected time unit. "
            "When volume shares an axis with concentrations, only the volume curve changes units. Its curve label shows the volume unit. "
            "The control is unavailable when the plot has no volume result. Display-unit choices reset when you reopen the project.</p>"
            "<h2>Parameters and Observables</h2>"
            "<p>Use <code>param</code> to define an adjustable input or a value calculated from other parameters.</p>"
            "<pre>param dose = 5; unit = mM\nparam duration = 2; unit = min\nparam doubled = 2*dose</pre>"
            "<p>An independent parameter without <code>unit=</code> is dimensionless. Kindred determines the units of calculated parameters and <code>let</code> quantities from their formulas.</p>"
            "<p>Use <code>let</code> to define a quantity calculated from the model.</p>"
            "<pre>A -&gt; B ; k=0.5\n[A] = 1.0, [B] = 0.0\nlet conversion = 1 - [A]/[A]_0\nlet rate = k1*[A]</pre>"
            "<p><code>[A]</code> means the concentration of A at simulation time <i>t</i>. <code>[A]_0</code> means the initial concentration of A supplied for the run.</p>"
            "<h3>Entering units</h3>"
            "<p>Add <code>; unit = ...</code> after a parameter or calculated quantity:</p>"
            "<pre>param dose = 5; unit = mM\nparam duration = 2; unit = min\nparam volume = 2; unit = mL\nparam doubled = 2*dose; unit = mM</pre>"
            "<p>For an independent input, enter its value in the declared unit. For a calculated quantity, a compatible unit declaration specifies a preferred output unit and leaves the formula unchanged.</p>"
            "<h3>Supported units</h3>"
            "<p>Use the spellings shown below after <code>; unit =</code>. The table includes unit symbols and useful compound forms.</p>"
            '<table border="1" cellpadding="5" cellspacing="0">'
            "<tr><th>Quantity</th><th>Units</th></tr>"
            "<tr><td>Concentration</td><td><code>fM</code>, <code>pM</code>, <code>nM</code>, <code>uM</code>, <code>mM</code>, <code>M</code>, <code>kM</code>, <code>mol/L</code>, <code>mmol/L</code>, <code>umol/L</code>, <code>mol/m^3</code></td></tr>"
            "<tr><td>Amount</td><td><code>fmol</code>, <code>pmol</code>, <code>nmol</code>, <code>umol</code>, <code>mmol</code>, <code>mol</code>, <code>kmol</code></td></tr>"
            "<tr><td>Length</td><td><code>fm</code>, <code>pm</code>, <code>nm</code>, <code>um</code>, <code>mm</code>, <code>cm</code>, <code>m</code>, <code>km</code></td></tr>"
            "<tr><td>Area</td><td>Examples: <code>m^2</code>, <code>cm^2</code></td></tr>"
            "<tr><td>Volume</td><td><code>fL</code>, <code>pL</code>, <code>nL</code>, <code>uL</code>, <code>mL</code>, <code>L</code>, <code>kL</code>, <code>nm^3</code>, <code>um^3</code>, <code>mm^3</code>, <code>cm^3</code>, <code>m^3</code>, <code>km^3</code></td></tr>"
            "<tr><td>Time</td><td><code>fs</code>, <code>ps</code>, <code>ns</code>, <code>us</code>, <code>ms</code>, <code>s</code>, <code>ks</code>, <code>min</code>, <code>h</code>, <code>d</code>, <code>yr</code></td></tr>"
            "<tr><td>Energy</td><td><code>fJ</code>, <code>pJ</code>, <code>nJ</code>, <code>uJ</code>, <code>mJ</code>, <code>J</code>, <code>kJ</code>, <code>kcal</code></td></tr>"
            "<tr><td>Energy per mole</td><td><code>fJ/mol</code>, <code>pJ/mol</code>, <code>nJ/mol</code>, <code>uJ/mol</code>, <code>mJ/mol</code>, <code>J/mol</code>, <code>kJ/mol</code>, <code>kcal/mol</code></td></tr>"
            "<tr><td>Temperature</td><td><code>K</code></td></tr>"
            "<tr><td>Dimensionless</td><td><code>1</code></td></tr>"
            "<tr><td>Concentration per time</td><td>Examples: <code>M/s</code>, <code>mM/s</code></td></tr>"
            "<tr><td>Rate constants</td><td>Examples: <code>s^-1</code>, <code>1/(M*s)</code></td></tr>"
            "</table>"
            "<p>Symbols are case-sensitive. <code>M</code> means molar concentration, while <code>m</code> means meter. Use capital <code>L</code> for liters and <code>u</code> for micro.</p>"
            "<p>The symbols <code>min</code>, <code>h</code>, <code>d</code>, and <code>yr</code> mean minutes, hours, days, and years. Kindred treats one year as 365.25 days.</p>"
            '<table border="1" cellpadding="5" cellspacing="0">'
            "<tr><th>Prefix</th><th>Meaning</th><th>Factor</th></tr>"
            "<tr><td><code>f</code></td><td>femto</td><td>10⁻¹⁵</td></tr>"
            "<tr><td><code>p</code></td><td>pico</td><td>10⁻¹²</td></tr>"
            "<tr><td><code>n</code></td><td>nano</td><td>10⁻⁹</td></tr>"
            "<tr><td><code>u</code></td><td>micro</td><td>10⁻⁶</td></tr>"
            "<tr><td><code>m</code></td><td>milli</td><td>10⁻³</td></tr>"
            "<tr><td><code>c</code></td><td>centi, available in <code>cm</code></td><td>10⁻²</td></tr>"
            "<tr><td><code>k</code></td><td>kilo</td><td>10³</td></tr>"
            "</table>"
            "<h3>Writing compound units</h3>"
            "<p>Combine unit symbols using products, quotients, and powers:</p>"
            '<table border="1" cellpadding="5" cellspacing="0">'
            "<tr><th>Operation</th><th>Write</th></tr>"
            "<tr><td>Product</td><td><code>M*s</code></td></tr>"
            "<tr><td>Quotient</td><td><code>mol/L</code></td></tr>"
            "<tr><td>Multiple factors in the denominator</td><td><code>1/(M*s)</code></td></tr>"
            "<tr><td>Integer power</td><td><code>m^3</code> or <code>m**3</code></td></tr>"
            "<tr><td>Negative power</td><td><code>s^-1</code></td></tr>"
            "<tr><td>Fractional power</td><td><code>M^(1/2)</code> or <code>M^0.5</code></td></tr>"
            "</table>"
            "<p>Use <code>*</code> for multiplication. Spaces can surround an operator, but a space does not replace <code>*</code>.</p>"
            "<p>Put fractional exponents written as fractions in parentheses. Type powers with ordinary keyboard characters, such as <code>m^3</code>.</p>"
            "<p>Other products, quotients, and powers are supported within the limits listed under <b>Supported input ranges</b>.</p>"
            "<h3>Units of calculated quantities</h3>"
            "<p>If a declared unit does not match the formula, Kindred keeps the units determined from the formula.</p>"
            "<p>If Kindred cannot determine consistent units, it still calculates the value and leaves its units unspecified. Invalid or unsupported unit spellings produce an error.</p>"
            "<h3>Names you can use</h3>"
            "<p>A <code>param</code> expression can refer to other parameters and available quantities from the reaction mechanism:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Name</th><th>Quantity</th></tr>"
            "<tr><td><code>k1</code>, <code>kf1</code>, <code>kr1</code></td><td>Rate constants</td></tr>"
            "<tr><td><code>Keq1</code></td><td>Equilibrium constant</td></tr>"
            "<tr><td><code>A1</code></td><td>Arrhenius prefactor</td></tr>"
            "<tr><td><code>Ea1</code></td><td>Arrhenius activation energy</td></tr>"
            "<tr><td><code>dG_act1</code></td><td>Eyring activation Gibbs energy</td></tr>"
            "<tr><td><code>dG_eq1</code></td><td>Reaction Gibbs energy</td></tr>"
            "<tr><td><code>kappa1</code></td><td>Eyring transmission coefficient</td></tr>"
            "</table>"
            "<p>Names such as <code>kf1</code> and <code>Keq1</code> refer to reaction parameters. The number identifies the reaction step. Available names depend on the model and parameters used for that step. Choose a different name, such as <code>pulse_time</code>, for an independent input.</p>"
            "<p>A <code>param</code> expression cannot directly use species concentrations, <code>t</code>, or <code>T</code>. It can depend on temperature indirectly through a temperature-dependent rate constant. Referring to an energy parameter alone does not make an expression temperature-dependent.</p>"
            "<p>A <code>let</code> expression can use species concentrations, parameters, other <code>let</code> definitions, simulation time <code>t</code>, and temperature <code>T</code>.</p>"
            "<p>Begin each definition with <code>param</code> or <code>let</code>. A line containing only <code>name = expression</code> is not accepted. <code>observable</code> is not a keyword.</p>"
            "<h3>Supported input ranges</h3>"
            "<p>Kindred uses M for concentration, s for time, K for temperature, J/mol for energy parameters, and m³ for volume in calculations.</p>"
            "<p>An ordinary adjustable parameter can be zero, or a positive or negative value with magnitude between 10⁻³⁰ and 10³⁰, inclusive, after conversion to these units.</p>"
            "<p>These limits apply to ordinary independent parameter values. Calculated expressions and simulation results are not restricted to this range. Species concentrations and physical-model parameters have their own input limits.</p>"
            "<p>The factor used to convert a declared unit to Kindred’s calculation units must be between 10⁻³⁰ and 10³⁰, inclusive. For example, mM has a factor of 0.001 relative to M. Kindred checks unit declarations even when the starting value is zero and reports unsupported declarations in <b>Reactions</b>.</p>"
            "<p>Kindred reports an error if an entered nonzero number is so small that it would be read as zero.</p>"
            "<h2>Functions</h2>"
            "<p>Use these functions in <code>param</code> and <code>let</code> expressions:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Write</th><th>Meaning</th></tr>"
            "<tr><td><code>sqrt(x)</code></td><td>Square root</td></tr>"
            "<tr><td><code>ln(x)</code>, <code>log10(x)</code></td><td>Natural and base-10 logarithms</td></tr>"
            "<tr><td><code>log1p(x)</code></td><td>ln(1 + <i>x</i>)</td></tr>"
            "<tr><td><code>exp(x)</code></td><td>e<sup><i>x</i></sup></td></tr>"
            "<tr><td><code>expm1(x)</code></td><td>e<sup><i>x</i></sup> − 1</td></tr>"
            "<tr><td><code>pow(x, y)</code></td><td><i>x</i> raised to the power <i>y</i></td></tr>"
            "<tr><td><code>abs(x)</code></td><td>Absolute value</td></tr>"
            "<tr><td><code>sin(x)</code>, <code>cos(x)</code>, <code>tan(x)</code></td><td>Trigonometric functions, with angles in radians</td></tr>"
            "<tr><td><code>erf(x)</code></td><td>Error function</td></tr>"
            "<tr><td><code>min(a, b)</code>, <code>max(a, b)</code></td><td>Smallest or largest supplied value. More than two values are accepted.</td></tr>"
            "<tr><td><code>clip(x, lower, upper)</code></td><td>Limits <i>x</i> to the specified range</td></tr>"
            "<tr><td><code>heaviside(x)</code></td><td>0 when <i>x</i> ≤ 0, otherwise 1</td></tr>"
            "<tr><td><code>ifelse(condition, a, b)</code></td><td>Selects <i>a</i> when the condition is true, otherwise <i>b</i></td></tr>"
            "</table>"
            "<p>Both values in <code>ifelse</code> are calculated, so both expressions must be valid.</p>"
            "<h3>Arithmetic and comparisons</h3>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Write</th><th>Meaning</th></tr>"
            "<tr><td><code>+</code>, <code>-</code>, <code>*</code>, <code>/</code></td><td>Addition, subtraction, multiplication, division</td></tr>"
            "<tr><td><code>-a</code></td><td>Changes the sign of <i>a</i>. A leading <code>+</code> is also accepted and leaves the value unchanged.</td></tr>"
            "<tr><td><code>a^b</code> or <code>a**b</code></td><td><i>a</i> raised to the power <i>b</i></td></tr>"
            "<tr><td><code>==</code>, <code>!=</code></td><td>Equal to, not equal to</td></tr>"
            "<tr><td><code>&lt;</code>, <code>&lt;=</code></td><td>Less than, less than or equal to</td></tr>"
            "<tr><td><code>&gt;</code>, <code>&gt;=</code></td><td>Greater than, greater than or equal to</td></tr>"
            "<tr><td><code>&amp;&amp;</code>, <code>||</code>, <code>!</code></td><td>And, or, not</td></tr>"
            "</table>"
            "<h2>Constants, Time, and Temperature</h2>"
            "<p>Use these names directly in expressions:</p>"
            "<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">"
            "<tr><th>Name to enter</th><th>Meaning</th></tr>"
            "<tr><td><code>t</code></td><td>Simulation time <i>t</i>, in seconds</td></tr>"
            "<tr><td><code>T</code></td><td>Temperature <i>T</i> at simulation time <i>t</i>, in kelvin</td></tr>"
            "<tr><td><code>k_B</code></td><td>Boltzmann constant, in J/K</td></tr>"
            "<tr><td><code>h</code></td><td>Planck constant, in J·s</td></tr>"
            "<tr><td><code>h_bar</code></td><td>Reduced Planck constant, in J·s</td></tr>"
            "<tr><td><code>N_A</code></td><td>Avogadro constant, in mol⁻¹</td></tr>"
            "<tr><td><code>R</code></td><td>Gas constant, in J/(mol·K)</td></tr>"
            "</table>"
            "<p>Enter <code>t</code> and <code>T</code> directly in <code>let</code> expressions. For temperature-dependent <code>param</code> expressions, use the reaction parameters described under <b>Parameters and Observables</b>.</p>"
            "<p>A global <code>T=</code> setting supplies the nominal temperature. When a temperature schedule is present, <code>T</code> follows that schedule during the run.</p>"
            "<p>The names <code>t</code> and <code>T</code> cannot be redefined.</p>"
            "<h2>Complete Example</h2>"
            "<p>Paste this example into <b>Reactions</b>.</p>"
            "<pre>energy=kJ/mol\nC0=1.0\n[A] = 1.0, [B] = 0.0, [C] = 0.0\n\nparam pulse_time = 1.5; unit = s\nparam pulse_amount = 0.2; unit = M\n\nA -&gt; B ; dG_act=55 ; degeneracy=2\nB &lt;=&gt; C ; kf=1.0 ; Keq=3.0\nintervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_amount\nlet product = [C]</pre>"
        )
        copy_help_btn = QtWidgets.QPushButton("Copy Plain Text", self._help_tab)
        copy_help_btn.setToolTip("Copy the entire Help reference as plain text, including examples.")
        copy_help_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(help_browser.toPlainText())
        )
        help_layout.addWidget(copy_help_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        help_layout.addWidget(help_browser)
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
            "Reactions text. Notes are never parsed or injected."
        )
        self._notes_text.setFont(QtGui.QFont("Courier New", 10))
        self._notes_text.setUndoRedoEnabled(True)
        self._notes_text.undoAvailable.connect(
            self._emit_native_history_availability
        )
        self._notes_text.redoAvailable.connect(
            self._emit_native_history_availability
        )
        self._notes_text.textChanged.connect(
            lambda: self.notesEdited.emit(str(self._notes_text.toPlainText()))
        )
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.focusChanged.connect(
                self._emit_native_history_availability
            )
        notes_layout.addWidget(self._notes_text)
        self._tabs.addTab(self._notes_tab, "Notes")

        self._initial_slider_sync_timer = QtCore.QTimer(self)
        self._initial_slider_sync_timer.setSingleShot(True)
        self._initial_slider_sync_timer.timeout.connect(
            self._sync_slider_workspace_state
        )
        self._initial_slider_sync_timer.start(0)

    def species_sliders_widget(self) -> BatchSpeciesSliders:
        return self._species_sliders

    def set_slider_edit_targets_summary(self, text: str) -> None:
        summary = str(text or "Initial-condition sets: none")
        base_tooltip = (
            "Concentration sliders affect the current set and any pinned sets."
        )
        tooltip = (
            base_tooltip
            if summary == "Initial-condition sets: none"
            else f"{base_tooltip}\n\n{summary}"
        )
        set_bounded_label_text(
            self._slider_edit_targets_label,
            summary,
            max_width=16777215,
            tooltip_text=tooltip,
        )

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
        self._sync_fine_button_state(self._variable_sliders.fine_mode_enabled)
        has_variable_entries = self._variable_sliders.has_slider_entries()
        has_species_entries = self._species_sliders.has_slider_rows()
        has_available_entries = bool(has_variable_entries or has_species_entries)

        visible_variable_entries = self._variable_sliders.visible_slider_count()
        visible_species_entries = self._species_sliders.visible_row_count()
        has_visible_entries = bool(visible_variable_entries or visible_species_entries)

        self._variable_sliders.setVisible(visible_variable_entries > 0)
        self._species_sliders.setVisible(visible_species_entries > 0)
        self._slider_surface.setVisible(has_visible_entries)
        self._slider_visibility_picker_btn.setEnabled(has_available_entries)
        self._fine_btn.setEnabled(has_available_entries)
        self._slider_pane_container.setVisible(has_available_entries)

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

    def _sync_fine_button_state(self, enabled: bool) -> None:
        """Project child Fine state without re-entering the toggle handler."""
        self._species_sliders.set_fine_mode(enabled)
        blocker = QtCore.QSignalBlocker(self._fine_btn)
        try:
            self._fine_btn.setChecked(bool(enabled))
        finally:
            del blocker

    def render_slider_workspace_presentation(self) -> None:
        """Project Slider child content without publishing user intent."""
        self._sync_slider_workspace_state()

    def _rebuild_slider_visibility_menu(self) -> None:
        menu = self._slider_visibility_menu
        menu.clear()
        menu.setToolTipsVisible(True)

        mechanism_entries = list(self._variable_sliders.slider_picker_entries())
        species_entries = list(self._species_sliders.slider_picker_entries())
        invert_action = menu.addAction("Invert selection")
        invert_action.setToolTip("Show the hidden sliders and hide the visible ones.")
        invert_action.setEnabled(bool(mechanism_entries or species_entries))
        invert_action.triggered.connect(
            lambda _checked=False: self._invert_slider_visibility()
        )
        if not mechanism_entries and not species_entries:
            action = menu.addAction("No sliders available")
            action.setToolTip("No adjustable parameter or concentration sliders are available.")
            action.setEnabled(False)
            return

        if mechanism_entries:
            self._add_slider_visibility_entries(
                menu,
                title="Mechanism",
                tooltip="Adjustable inputs in the mechanism.",
                entry_kind="mechanism",
                entries=mechanism_entries,
            )
        if mechanism_entries and species_entries:
            menu.addSeparator()
        if species_entries:
            self._add_slider_visibility_entries(
                menu,
                title="Initial concentrations",
                tooltip="Starting concentrations for species in the mechanism.",
                entry_kind="species",
                entries=species_entries,
            )

    def _invert_slider_visibility(self) -> None:
        """Invert both complete child rosters in one workspace transaction."""
        mechanism_entries = tuple(
            self._variable_sliders.slider_picker_entries()
        )
        species_entries = tuple(
            self._species_sliders.slider_picker_entries()
        )
        mechanism_projection = {
            name: not bool(visible)
            for name, _label, visible in mechanism_entries
        }
        species_projection = {
            name: not bool(visible)
            for name, _label, visible in species_entries
        }
        self._variable_sliders.apply_visibility_projection(
            mechanism_projection
        )
        self._species_sliders.apply_visibility_projection(species_projection)
        self._sync_slider_workspace_state()

    def _add_slider_visibility_entries(
        self,
        menu: QtWidgets.QMenu,
        *,
        title: str,
        tooltip: str,
        entry_kind: str,
        entries: list[tuple[str, str, bool]],
    ) -> None:
        header = menu.addAction(str(title))
        header.setToolTip(tooltip)
        header.setEnabled(False)
        for key, label, visible in entries:
            action = menu.addAction(compact_text(label, max_chars=48).display)
            action.setToolTip(literal_tooltip(str(label)))
            action.setStatusTip(str(label))
            action.setCheckable(True)
            action.setChecked(bool(visible))
            action.setData((str(entry_kind), str(key)))
            action.toggled.connect(
                lambda checked,
                kind=str(entry_kind),
                item=str(key): self._set_slider_entry_visible(kind, item, checked)
            )

    def _set_slider_entry_visible(
        self, entry_kind: str, item_name: str, visible: bool
    ) -> None:
        if str(entry_kind) == "mechanism":
            self._variable_sliders.set_variable_visible(str(item_name), bool(visible))
        elif str(entry_kind) == "species":
            self._species_sliders.apply_visibility_projection(
                {str(item_name): bool(visible)}
            )
        self._sync_slider_workspace_state()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange):
            self._refresh_sliders_icon()
            if hasattr(self, "_reactions_highlighter"):
                with QtCore.QSignalBlocker(self._reactions_text):
                    self._reactions_highlighter.set_theme(
                        dark_mode=self.palette().color(QtGui.QPalette.Text).lightness() > 128
                    )

    def _refresh_sliders_icon(self) -> None:
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        self.sliders_btn.setIcon(_make_sliders_icon(color))

    def _on_text_changed(self):
        """Capture literal source edits without running scientific preparation."""
        self._on_source_draft_edited()

    def _on_source_draft_edited(self) -> None:
        from kindred.core.mechanism_source import MechanismAuthoringSource

        self.sourceDraftEdited.emit(
            MechanismAuthoringSource.from_parts(
                reactions_text=self.reactions_text(),
            )
        )

    def _set_validation_state(self, state: str, message: str = ""):
        """
        Set validation indicator state.

        Parameters
        ----------
        state : str
            One of: "idle", "draft", "valid", "invalid"
        message : str
            Status message to display
        """
        if state == "valid":
            run_tooltip = (
                "Run simulation for the current run target from Initial Conditions"
            )
        elif state == "draft":
            run_tooltip = "Visible Reactions will be validated before Run."
        else:
            run_tooltip = (
                "No valid mechanism — enter a valid reaction mechanism to enable"
            )
        self._run_btn.setToolTip(run_tooltip)

        if state == "idle":
            self._set_validation_label_text("")
            self._validation_label.setStyleSheet("QLabel { padding: 4px; }")
        elif state == "draft":
            self._set_validation_label_text(
                message
                or "Visible Reactions have changed; scientific actions will validate them."
            )
            self._validation_label.setStyleSheet(
                "QLabel { padding: 4px; border-radius: 3px; }"
            )

        elif state == "valid":
            self._set_validation_label_text(message)
            self._validation_label.setStyleSheet(
                "QLabel { padding: 4px; border-radius: 3px; }"
            )

        elif state == "invalid":
            self._set_validation_label_text(message)
            self._validation_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; }"
            )

    def _set_validation_label_text(self, text: str) -> None:
        full_text = str(text or "")
        self._validation_label.setMaximumWidth(520)
        self._validation_label.setMaximumHeight(96)
        self._validation_label.setWordWrap(True)
        self._validation_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self._validation_label.setVisible(bool(full_text))
        self._validation_label.setText(full_text)
        self._validation_label.setToolTip(full_text if full_text else "")

    @property
    def run_btn(self) -> QtWidgets.QPushButton:
        return self._run_btn

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(bool(enabled))

    def set_slider_draft_actions_enabled(
        self,
        dirty: bool,
        reset_enabled: bool,
    ) -> None:
        enabled = bool(dirty)
        self._commit_runtime_parameters_btn.setEnabled(enabled)
        self._reset_runtime_parameters_btn.setEnabled(reset_enabled)
        self._commit_runtime_parameters_btn.setText(
            "Apply \N{BULLET}" if dirty else "Apply"
        )

    def render_reactions_acceptance_status(
        self,
        *,
        has_draft: bool,
        has_accepted_reactions: bool,
    ) -> None:
        """Render explicit draft/accepted truth supplied by the binding."""

        draft = bool(has_draft)
        self._revert_reactions_btn.setEnabled(draft)
        if draft:
            self._set_validation_state(
                "draft",
                "Visible Reactions are Draft. Current work and results use "
                "Accepted Reactions.",
            )
        elif bool(has_accepted_reactions):
            self._set_validation_state("valid", "Accepted Reactions.")
        else:
            self._set_validation_state("idle")

    def reactions_text(self) -> str:
        """Return the current Reaction DSL text."""
        return str(self._reactions_text.toPlainText())

    def set_reactions_text(self, text: str, *, block_signals: bool = False) -> None:
        """Replace the Reaction DSL text."""
        replacement = str(text)
        if self._reactions_text.toPlainText() == replacement:
            return
        if not bool(block_signals):
            self._reactions_text.setPlainText(replacement)
            return
        blocker = QtCore.QSignalBlocker(self._reactions_text)
        try:
            self._reactions_text.setPlainText(replacement)
        finally:
            del blocker

    def apply_accepted_reactions_rewrite(
        self,
        *,
        expected_visible_text: str,
        accepted_text: str,
    ) -> bool:
        """Install parser-authored provenance as one undoable native edit."""

        if self.reactions_text() != str(expected_visible_text):
            return False
        replacement = str(accepted_text)
        if replacement == self.reactions_text():
            return False
        cursor = self._reactions_text.textCursor()
        cursor.beginEditBlock()
        try:
            cursor.select(QtGui.QTextCursor.SelectionType.Document)
            cursor.insertText(replacement)
        finally:
            cursor.endEditBlock()
        return True

    def render_authoring_snapshot(
        self,
        *,
        source,
        notes: str,
    ) -> None:
        """Render one tab snapshot without retaining canonical state."""
        from kindred.core.mechanism_source import MechanismAuthoringSource

        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be MechanismAuthoringSource.")
        editor_blocker = QtCore.QSignalBlocker(self)
        reactions_blocker = QtCore.QSignalBlocker(self._reactions_text)
        notes_blocker = QtCore.QSignalBlocker(self._notes_text)
        try:
            self.set_reactions_text(source.reactions_text)
            notes_text = str(notes)
            if self._notes_text.toPlainText() != notes_text:
                self._notes_text.setPlainText(notes_text)
        finally:
            del notes_blocker
            del reactions_blocker
            del editor_blocker

    def variable_sliders_widget(self) -> VariableSliders:
        return self._variable_sliders

    def focused_native_editor(self) -> str | None:
        focused = QtWidgets.QApplication.focusWidget()
        if focused is self._reactions_text:
            return "reactions"
        if focused is self._notes_text:
            return "notes"
        return None

    def native_undo_available(self) -> bool:
        editor = self._focused_text_editor()
        return bool(editor is not None and editor.document().isUndoAvailable())

    def native_redo_available(self) -> bool:
        editor = self._focused_text_editor()
        return bool(editor is not None and editor.document().isRedoAvailable())

    def clear_native_text_history(self) -> None:
        """Clear only Reactions/Notes native history on document replacement."""

        self._reactions_text.document().clearUndoRedoStacks()
        self._notes_text.document().clearUndoRedoStacks()
        self._emit_native_history_availability()

    def undo_focused_text(self) -> bool:
        editor = self._focused_text_editor()
        if editor is None or not editor.document().isUndoAvailable():
            return False
        editor.undo()
        return True

    def redo_focused_text(self) -> bool:
        editor = self._focused_text_editor()
        if editor is None or not editor.document().isRedoAvailable():
            return False
        editor.redo()
        return True

    def _focused_text_editor(self) -> QtWidgets.QPlainTextEdit | None:
        return {
            "reactions": self._reactions_text,
            "notes": self._notes_text,
        }.get(self.focused_native_editor())

    @QtCore.Slot()
    def _emit_native_history_availability(self, *_args) -> None:
        self.nativeHistoryAvailabilityChanged.emit(
            self.native_undo_available(),
            self.native_redo_available(),
        )

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._emit_native_history_availability()


class MechanismEditorBinding(QtCore.QObject):
    """Live binding for one published mechanism-tab editor."""

    draftStateChanged = QtCore.Signal(str, bool)

    def __init__(
        self,
        *,
        tab_id: str,
        editor: MechanismEditorTabbed,
        registry,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        if not isinstance(editor, MechanismEditorTabbed):
            raise TypeError("editor must be MechanismEditorTabbed.")
        if not callable(getattr(registry, "tab", None)):
            raise TypeError("registry must provide tab(tab_id).")
        self._tab_id = target
        self._editor = editor
        self._registry = registry
        self._rendering = False
        editor.sourceDraftEdited.connect(self._on_source_draft)
        editor.reactionsRevertRequested.connect(self.revert_reactions_draft)
        editor.notesEdited.connect(self._on_notes_edited)

    @property
    def tab_id(self) -> str:
        return self._tab_id

    def prepare_render(self) -> _MechanismEditorRenderProjection:
        """Capture the current published document into a view projection."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        source = runtime.reactions_draft or tab.mechanism_source
        return _MechanismEditorRenderProjection(
            source=source,
            notes=(
                tab.notes
                if runtime.notes_draft is None
                else runtime.notes_draft
            ),
            reactions_draft=runtime.reactions_draft is not None,
            accepted_reactions=tab.prepared_dsl is not None,
        )

    def apply_render_projection(
        self,
        projection: _MechanismEditorRenderProjection,
    ) -> None:
        if not isinstance(projection, _MechanismEditorRenderProjection):
            raise TypeError(
                "projection must be _MechanismEditorRenderProjection."
            )
        self._rendering = True
        try:
            self._editor.render_authoring_snapshot(
                source=projection.source,
                notes=projection.notes,
            )
            self._editor.render_reactions_acceptance_status(
                has_draft=projection.reactions_draft,
                has_accepted_reactions=projection.accepted_reactions,
            )
        finally:
            self._rendering = False

    def render(self) -> None:
        self.apply_render_projection(self.prepare_render())

    @QtCore.Slot(object)
    def _on_source_draft(self, source: object) -> None:
        from kindred.core.mechanism_source import MechanismAuthoringSource

        if self._rendering or not isinstance(source, MechanismAuthoringSource):
            return
        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        runtime.reactions_draft = None if source == tab.mechanism_source else source
        self._editor.render_reactions_acceptance_status(
            has_draft=runtime.reactions_draft is not None,
            has_accepted_reactions=tab.prepared_dsl is not None,
        )
        self.draftStateChanged.emit(tab.tab_id, runtime.has_dirty_drafts)

    @QtCore.Slot(str)
    def _on_notes_edited(self, notes: str) -> None:
        if self._rendering:
            return
        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        visible_notes = str(notes)
        runtime.notes_draft = (
            None if visible_notes == tab.notes else visible_notes
        )
        self.draftStateChanged.emit(tab.tab_id, runtime.has_dirty_drafts)

    def apply_accepted_source_rewrite(
        self,
        visible: MechanismAuthoringSource,
        accepted: MechanismAuthoringSource,
    ) -> bool:
        """Keep deterministic parser rewrites inside native editor history."""

        from kindred.core.mechanism_source import MechanismAuthoringSource

        if not isinstance(visible, MechanismAuthoringSource):
            raise TypeError("visible must be MechanismAuthoringSource.")
        if not isinstance(accepted, MechanismAuthoringSource):
            raise TypeError("accepted must be MechanismAuthoringSource.")
        return self._editor.apply_accepted_reactions_rewrite(
            expected_visible_text=visible.reactions_text,
            accepted_text=accepted.reactions_text,
        )

    @QtCore.Slot()
    def revert_reactions_draft(self) -> bool:
        """Restore canonical source as one native undoable edit."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        draft = runtime.reactions_draft
        if draft is None:
            return False
        return self._editor.apply_accepted_reactions_rewrite(
            expected_visible_text=draft.reactions_text,
            accepted_text=tab.mechanism_source.reactions_text,
        )
