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
            "reaction: 2A + B => C ; kf=1e5\n"
            "equilibrium: C <=> D ; Keq=2.5 ; kf=10.0\n"
            "reaction: C + E -> F ; k=0.5\n"
            "param scale = 2.0\n"
            "reaction: A -> A_Side ; kf=0.01\n"
            "let total_A = [A] + [A_Side]\n"
            "param k_base = 1.5e3\n"
            "let conversion = 1.0 - ([A] / max([A]_0, 1e-18))\n"
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
            "<h2>Quick Start</h2>"
            "<p>A mechanism is a readable list of settings, starting concentrations, reactions, and optional schedules.</p>"
            "<pre>reaction: A -&gt; B ; k=0.5</pre>"
            "<p>Put one directive or declaration on each line in Reactions, then use Run, Preview, or Fit. The same accepted source drives each action.</p>"
            "<h2>Global Settings</h2>"
            "<p>These lines override the mechanism&#x27;s saved defaults for nominal temperature, input energy units, and standard concentration.</p>"
            "<pre>T=298.15\nenergy=kJ/mol\nC0=1.0</pre>"
            "<p>Supported energy units are kJ/mol, kcal/mol, and J/mol. T is in K; C0 is positive and in M.</p>"
            "<p>Each global setting may appear once anywhere in the source; duplicate definitions reject. Omitted settings use this mechanism&#x27;s defaults in Simulation Settings, which can also be saved as defaults for new projects and tabs.</p>"
            "<p>Per-reaction T= and energy= are rejected.</p>"
            "<h2>Initial Conditions</h2>"
            "<p>Initial conditions are the concentrations at the start of a run, before reactions or interventions act.</p>"
            "<pre>[A] = 1.0, [B] = 0.0</pre>"
            "<p>You can also write init: A=1.0, B=0.0 or initial: A=1.0, B=0.0. Values must be finite and nonnegative. A named set uses one braced block, for example:</p>"
            "<pre>start = {\n[A] = 1.0\n[B] = 0.0\n}</pre>"
            "<h2>Reaction Arrows</h2>"
            "<p>The arrow tells Kindred whether material can move only forward or in both directions.</p>"
            "<pre>reaction: A -&gt; B ; k=0.5\nreaction: A &lt;-&gt; B ; kf=1.0 ; Keq=2.0</pre>"
            "<p>Use -&gt; or =&gt; for irreversible steps, and &lt;-&gt; or &lt;=&gt; for reversible steps. &lt;- alone is not accepted. Stoichiometric terms may be written as A + 2B or 2*B.</p>"
            "<h2>Reaction Parameters</h2>"
            "<p>These fields choose the forward-rate model and its authored scientific inputs.</p>"
            "<pre>reaction: A -&gt; B ; dG_act=55 ; degeneracy=2</pre>"
            "<p>Use k= or kf= for a positive constant rate; A= with Ea= selects Arrhenius; dG_act= selects Eyring. kappa= is an optional positive local transmission coefficient on that Eyring step and defaults to 1.</p>"
            "<p>degeneracy= is the positive integer count of equivalent Eyring pathways and defaults to 1. It multiplies the effective forward Eyring rate, but it is distinct from kappa, the transmission coefficient; it does not change the authored barrier.</p>"
            "<p>Do not encode the same pathway count in both a lowered barrier and degeneracy, or multiply the resulting rate again: that double counts.</p>"
            "<p>For reversible steps, kr= is the explicit concentration-form reverse rate when that is the selected authority; Keq= or dG_eq= can instead derive the reverse rate.</p>"
            "<h2>Equilibrium Lines</h2>"
            "<p>An equilibrium line declares a reversible pair whose forward rate and thermodynamic relation determine the reverse rate.</p>"
            "<pre>equilibrium: A &lt;=&gt; B ; kf=10.0 ; Keq=2.0</pre>"
            "<p>Use a reversible arrow and kf= plus exactly one of kr=, Keq=, or dG_eq=. Keq= is concentration-form with units M^(nr-nf). dG_eq= is entered in the selected global energy unit and determines Keq=exp(-dG_eq/(R*T))*C0^(nr-nf), using consistent energy units. Here, T is the current execution temperature: it follows the temperature schedule when one is present and otherwise stays at the resolved nominal temperature.</p>"
            "<p>A thermodynamic kf=0 disables both directions; explicit kf+kr remains strictly positive. Supplying incompatible rate authorities, such as kr= with Keq=, is rejected.</p>"
            "<h2>Temperature Schedules</h2>"
            "<p>A schedule makes the external temperature T(t) change during the run instead of staying at the resolved nominal temperature.</p>"
            "<pre>temp_step: t=[0,50,100], T=[298,320]</pre>"
            "<p>As an alternative, use temp_response: t=[0,50,100], T=[298,320], tau=10. Use at most one schedule. Times are finite and nonnegative; a step schedule has one temperature per interval, and tau is positive.</p>"
            "<h2>Interventions</h2>"
            "<p>Interventions change or constrain a species during the run at an exact time or over an interval.</p>"
            "<pre>intervention: op=set; species=A; time=1.0; value=2.0\nintervention: op=jump; species=A; time=2.0; delta=-0.25\nintervention: op=flux; species=B; start_time=0.0; end_time=5.0; flux_rate=-0.1\nintervention: op=hold; species=B; start_time=0.0; end_time=10.0; value=1.0</pre>"
            "<ul>"
            "<li>set assigns a species concentration at one exact time.</li>"
            "<li>jump adds delta at one exact time.</li>"
            "<li>flux adds flux_rate to the derivative between start_time and end_time.</li>"
            "<li>hold fixes a species at value throughout that interval.</li>"
            "<li>Top-level fields use semicolons.</li>"
            "</ul>"
            "<h2>Parameterized Interventions</h2>"
            "<p>A parameterized field keeps the schedule shape fixed while a named scalar supplies its value.</p>"
            "<pre>param pulse_time = 2.0\nparam pulse_delta = 0.2\nintervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_delta</pre>"
            "<p>For each pair, provide exactly one literal field or its matching *_param field. Parameter values must satisfy the same finite-value and domain requirements as literal values.</p>"
            "<p>Timing parameters (time_param, start_time_param, and end_time_param) are evaluated once before execution at the resolved nominal temperature. Their resulting times remain fixed throughout that execution, even when temperature changes.</p>"
            "<p>For set and jump, value_param and delta_param are evaluated when the action executes. During an active flux or hold, flux_rate_param and value_param follow the current temperature-dependent parameter values. Trigger thresholds supplied through threshold_param also follow those values as execution proceeds.</p>"
            "<h2>Repeat Protocols</h2>"
            "<p>A repeat protocol applies the same authored phase operations at regularly spaced starts.</p>"
            "<pre>intervention: op=protocol; kind=repeat; start_time=0.5; every=1.25; count=4; before={op=jump,species=A,delta=0.2}</pre>"
            "<p>before and after permit set or jump; during permits flux or hold. A protocol with during or after also needs a positive duration:</p>"
            "<pre>intervention: op=protocol; kind=repeat; start_time=0.0; every=2.0; count=3; duration=1.0; during={op=hold,species=B,value=1.0}; after={op=set,species=B,value=0.0}</pre>"
            "<p>Use one braced envelope per phase. A pipe separates ordered records inside an envelope, and commas separate their fields. Braces do not nest; semicolons are not allowed inside an envelope; nested timing, protocol, and trigger operations are not allowed.</p>"
            "<h2>State Triggers</h2>"
            "<p>A trigger performs one action when its watched quantity crosses a threshold. Chemical-action triggers watch a species concentration; stopping triggers can also watch an eligible authored let.</p>"
            "<ul>"
            "<li>then={op=set,species=B,value=1.0} assigns a concentration.</li>"
            "<li>then={op=jump,species=B,delta=0.1} changes a concentration by delta.</li>"
            "<li>then={op=stop} ends the simulation successfully at the triggering time. See Optional Stopping Interventions below.</li>"
            "</ul>"
            "<pre>intervention: op=trigger; watch_species=A; threshold=0.8; direction=falling; max_count=1; min_interval=0.0; then={op=jump,species=B,delta=0.1}</pre>"
            "<p>direction is rising, falling, or either (the default). A chemical-action trigger’s then contains exactly one set or jump and has implicit trigger time, so it cannot contain time or time_param. Chemical-action triggers fire at threshold crossings located by the solver.</p>"
            "<p>Initial equality is neutral; the trajectory must first occupy a strict side and then make a genuine direction-compatible crossing. A chemical action cannot recursively fire another chemical-action trigger at the same boundary, but it can cause a stopping intervention to fire.</p>"
            "<h2>Optional Stopping Interventions</h2>"
            "<p><b>stop ends integration successfully when its trigger fires.</b> Kindred retains the trajectory through that time and produces a result. Remaining actions at that time and all later interventions do not execute. An action that caused the crossing remains applied. This is successful early completion, distinct from manually cancelling a Run with the GUI Stop button.</p>"
            "<p>Stopping declarations are optional. To use them, enable <b>Use authored stopping interventions</b> in Simulation Settings. This mechanism setting applies to Run and Preview for every initial-condition set, including sets with custom Run settings. The configured end time is then the maximum allowed time. With the checkbox off, stopping declarations are bypassed and the run uses its fixed end time. Fitting always ignores stopping interventions while keeping chemical interventions active.</p>"
            "<pre>intervention: op=trigger; watch_species=B; threshold=0.9; direction=rising; then={op=stop}</pre>"
            "<p>This ends the run when B crosses 0.9 upward. Starting at or above 0.9 does not immediately stop; a rising crossing must come from below. A falling crossing must come from above; either accepts either direction.</p>"
            "<p>Use exactly one watch_species or watch_observable and exactly one threshold or threshold_param. watch_observable names an existing authored LET; it does not define an expression here. direction is rising, falling, or either (the default). Stop actions accept no fields and stopping triggers do not accept max_count or min_interval.</p>"
            "<pre>let selectivity = [B] / max([A] + [B], 1e-12)\nintervention: op=trigger; watch_observable=selectivity; threshold=0.9; direction=rising; then={op=stop}</pre>"
            "<p>This watches the existing selectivity definition and ends the run when its value crosses 0.9 upward.</p>"
            "<pre>param target = 0.9\nintervention: op=trigger; watch_species=B; threshold_param=target; direction=rising; then={op=stop}</pre>"
            "<p>Here, target supplies a constant threshold of 0.9. Changing that parameter changes the stopping threshold without rewriting the trigger. If the parameter instead depends on a temperature-sensitive rate, its threshold follows that dependence during execution.</p>"
            "<h2>Stopping Order and Output</h2>"
            "<p>An eligible stop takes priority over pending chemical actions. Otherwise, each chemical action is followed by a stopping check before the next action, including actions at the same time. The relative order of chemical actions that execute stays unchanged.</p>"
            "<p>For example, B is 0.8 just before two same-time jumps: the first adds 0.2 and the second subtracts 0.2. With a rising stop threshold of 0.9, the first jump raises B to 1.0 and stops the run. The second jump does not execute; the result ends with B at 1.0. Skipped actions do not get intervention markers.</p>"
            "<p>A scheduled temperature change can also trigger a stop through an eligible observable. For example, let heat = T with watch_observable=heat and a rising threshold of 310 stops when a temperature step changes T from 298 K to 320 K.</p>"
            "<p>If a stopping crossing occurs at the same time as a scheduled temperature step, the result reflects whether that step executed before stopping. A stop before the step retains the preceding temperature; a stop caused by the step retains the changed temperature. In both cases, the final concentrations, reported T, and temperature-dependent parameters used by let outputs describe the same executed state. The reported stopping time is unchanged.</p>"
            "<p>Requested output samples cover the actual simulated interval. For example, 1,000 requested samples and a stop at 12 seconds from a start at zero produce samples across 0–12 seconds. Intervention boundaries may add rows; interpolation does not cross intervention jumps. A stop at the initial time produces a single row.</p>"
            "<p>Reaching the configured end time without a stopping crossing is also normal completion. Results record the actual endpoint and termination cause. Stopping does not guarantee detection of every crossing between solver steps.</p>"
            "<h2>Eligible Observables for Stopping</h2>"
            "<p>Stopping uses the existing let definition, including any other lets it references. Current concentrations, initial concentrations, t, T, parameters, and available effective rate coefficients can be used. The parts that vary during a run have these restrictions:</p>"
            "<ul>"
            "<li>Arithmetic: +, -, *, unary + and -, and powers written with **, ^, or pow. A power’s exponent must be a nonnegative integer determined before the run.</li>"
            "<li>Functions: abs, min, max, clip, exp, expm1, sin, cos, and erf.</li>"
            "<li>Division: the denominator must be a finite nonzero value determined before the run, or use max(expression, positive_bound) or min(expression, negative_bound). The bound must be determined before the run and must be the second argument. A denominator may reference a let containing one of these forms.</li>"
            "<li>Comparisons, logical operators, ifelse, heaviside, sqrt, ln, log10, log1p, and tan are not supported when they depend on quantities that vary during the run.</li>"
            "</ul>"
            "<p>Parts determined entirely before integration, such as sqrt([A]_0), may use the ordinary Algebra functions if their values are finite and valid. These rules also apply through referenced lets. All evaluated stopping quantities must remain finite.</p>"
            "<pre>let guarded_ratio = [B] / max([A], 1e-12)\nlet squared = [B] ** 2</pre>"
            "<p>These are supported stopping targets. In contrast, [B] / [A] has an unguarded changing denominator, sqrt([B]) uses an unsupported changing function, and [B] &gt; 0.9 is a comparison. For a threshold, watch B or a numeric let and put 0.9 in the trigger’s threshold field.</p>"
            "<p>An unsupported active target is rejected with its stopping declaration and quantity identified. These stopping restrictions do not restrict ordinary output observables. Bypassed stopping targets are not evaluated for stopping.</p>"
            "<h2>Intervention Value Rules</h2>"
            "<p>These domains keep every requested change physically representable and keep parameterized schedules safe after resolution.</p>"
            "<pre>intervention: op=jump; species=A; time=2.0; delta=-0.25</pre>"
            "<ul>"
            "<li>Times, species thresholds, and min_interval are finite and nonnegative; observable thresholds are finite signed values; end_time must be greater than start_time.</li>"
            "<li>set and hold values are finite and nonnegative.</li>"
            "<li>jump deltas and flux rates are finite signed values, including zero.</li>"
            "<li>Negative jumps saturate at zero, and negative flux never creates a negative state.</li>"
            "<li>every and required duration are strictly positive; count and max_count are positive non-boolean integers.</li>"
            "<li>Unknown, duplicate, missing, empty, or malformed fields reject.</li>"
            "</ul>"
            "<h2>Operations at the Same Time</h2>"
            "<p>When several scheduled or triggered changes share a boundary, authored order and conflict checks determine one unambiguous result.</p>"
            "<pre>intervention: op=set; species=A; time=1.0; value=2.0</pre>"
            "<p>Without active stops, Kindred validates all static and exact-root trigger candidates before mutation. Ending intervals end, ordinary and protocol-before operations run in authored order, entering holds are established, trigger actions run in declaration order, and protocol-after operations run in authored order.</p>"
            "<p>With active stops, an eligible stop interrupts this order immediately. Each executed chemical action can make a stop eligible before the next action; the causing action remains applied. See Stopping Order and Output above.</p>"
            "<p>Only one hold per species may be active at a time. After resolving time parameters, Kindred rejects overlapping hold intervals before integration, even if their values agree or an earlier stop would skip them. Adjacent intervals may share an endpoint, and different species may be held simultaneously.</p>"
            "<p>Other conflicts reject unequal absolute sets, set plus jump, and a hold start that is not an equal set.</p>"
            "<p>Scheduled changes, holds, trigger actions, sampled output, and manual sign checks do not create chemical-trigger roots. Active stopping interventions also check crossings immediately after each chemical action.</p>"
            "<h2>Intervention Parameters and Protected Names</h2>"
            "<p>Only the documented parameter fields may be replaced by named values, and indexed mechanism names retain their step meaning.</p>"
            "<pre>param pulse_time = 2.0\nintervention: op=set; species=A; time_param=pulse_time; value=1.0</pre>"
            "<p>The complete *_param set is time_param, start_time_param, end_time_param, value_param, delta_param, flux_rate_param, and threshold_param. every, count, duration, max_count, and min_interval remain literal fields. Exact indexed names such as k1, kf1, kr1, and Keq1 resolve through the mechanism parameter namespace; use a longer name such as K1_test for an independent scalar or schedule parameter.</p>"
            "<h2>Inspecting Interventions</h2>"
            "<p>Simulation → Inspect Mechanism... shows what an admitted schedule will do without editing it.</p>"
            "<pre>intervention: op=hold; species=B; start_time=0.0; end_time=5.0; value=1.0</pre>"
            "<p>Open Simulation → Inspect Mechanism... to read step numbers, symbolic right-hand-side equations, the authored schedule, typed operations, discontinuities, and repeat/trigger lineage. Plot annotations for executed schedule operations and exact trigger firings come from the run provenance and can be toggled in the plot context menu.</p>"
            "<h2>Parameters and Observables</h2>"
            "<p>param defines scalar inputs or relationships between parameters. let defines calculated quantities that can be plotted or matched to observations.</p>"
            "<pre>reaction: A -&gt; B ; k=0.5\n[A] = 1.0, [B] = 0.0\nparam scale = 2.0\nlet conversion = 1.0 - ([A] / max([A]_0, 1e-18))</pre>"
            "<p>Use [A] for the current concentration and [A]_0 for its initial value. A param expression can reference other parameters and available mechanism quantities, including rate and equilibrium symbols such as k1, kf1, kr1, and Keq1, and physical model parameters such as A1, Ea1, dG_act1, dG_eq1, and kappa1. The numerical suffix identifies the mechanism step; available names depend on that step's rate model and authored parameters.</p>"
            "<p>A param expression cannot directly reference concentrations, t, or T. It can nevertheless change with temperature when it depends on a temperature-sensitive quantity, such as an Arrhenius or Eyring rate. Referencing an energy parameter alone does not make an expression temperature-dependent.</p>"
            "<p>A let expression can use species concentrations, parameters, and the time t and temperature T at the current result row. Bare name = expr is not accepted, and observable is not a keyword.</p>"
            "<h2>Built-in Functions</h2>"
            "<p>Built-ins provide safe, copyable math inside param and let expressions.</p>"
            "<pre>reaction: A -&gt; B ; k=0.5\nlet bounded = clip([A], 0, 1)</pre>"
            "<ul>"
            "<li>Math: sqrt, ln, log10, log1p, exp, expm1, pow, abs, sin, cos, tan, and erf.</li>"
            "<li>Aggregation: min and max.</li>"
            "<li>Helpers: heaviside(x), clip(x, lo, hi), and ifelse(cond, a, b).</li>"
            "<li>Arithmetic, comparisons, &amp;&amp;, ||, !, and ** or ^ are also supported by the algebra language.</li>"
            "</ul>"
            "<h2>Constants, Time, and Temperature</h2>"
            "<p>These names supply physical constants, simulation time, and external temperature to expressions.</p>"
            "<pre>reaction: A -&gt; B ; k=0.5\nlet thermal = R * T\nlet elapsed = t\nlet decaying = exp(-0.5 * t)</pre>"
            "<ul>"
            "<li><code>t</code>: simulation time in seconds at the current result row.</li>"
            "<li><code>T</code>: external temperature in kelvin at the current result row.</li>"
            "<li><code>k_B</code>: Boltzmann's constant.</li>"
            "<li><code>h</code>: Planck's constant.</li>"
            "<li><code>h_bar</code>: reduced Planck constant.</li>"
            "<li><code>N_A</code>: Avogadro's constant.</li>"
            "<li><code>R</code>: gas constant in J/mol/K.</li>"
            "<li><code>Rkcal</code>: gas constant in kcal/mol/K.</li>"
            "</ul>"
            "<p>Both <code>t</code> and <code>T</code> can be referenced directly in <code>let</code> expressions, but not in <code>param</code> expressions. Parameters can depend indirectly on temperature through mechanism quantities, as described under Parameters and Observables. The names <code>t</code> and <code>T</code> cannot be redefined by Algebra.</p>"
            "<p>A global <code>T=</code> directive sets the nominal temperature. Energies are normalized internally to J/mol.</p>"
            "<h2>Complete Example</h2>"
            "<p>This complete source combines global settings, initial conditions, scalar params, both reaction families, an intervention, and an observable.</p>"
            "<pre>energy=kJ/mol\nC0=1.0\n[A] = 1.0, [B] = 0.0, [C] = 0.0\n\nparam pulse_time = 1.5\nparam pulse_amount = 0.2\n\nreaction: A -&gt; B ; dG_act=55 ; degeneracy=2\nreaction: B &lt;=&gt; C ; kf=1.0 ; Keq=3.0\nintervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_amount\nlet product = [C]</pre>"
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
