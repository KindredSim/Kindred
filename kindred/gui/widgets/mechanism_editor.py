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
    mechanismInspectRequested = Signal()
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
        reactions_header_row.addStretch()
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
        self.symbolic_calculator_btn.setObjectName("symbolicCalculatorOpenButton")
        self.symbolic_calculator_btn.setToolTip("Open the Symbolic Calculator panel")
        self.symbolic_calculator_btn.setStyleSheet("QPushButton { padding: 6px 12px; }")
        self.symbolic_calculator_btn.setEnabled(False)
        reactions_header_row.addWidget(self.symbolic_calculator_btn)
        self.inspect_mechanism_btn = QtWidgets.QPushButton("Inspect...", self)
        self.inspect_mechanism_btn.setObjectName("mechanismInspectorOpenButton")
        self.inspect_mechanism_btn.setToolTip("Open a read-only Mechanism Inspector")
        self.inspect_mechanism_btn.setStyleSheet("QPushButton { padding: 2px 8px; }")
        self.inspect_mechanism_btn.clicked.connect(
            lambda _checked=False: self.mechanismInspectRequested.emit()
        )
        reactions_header_row.addWidget(self.inspect_mechanism_btn)
        self._run_btn = QtWidgets.QPushButton("Run")
        run_font = self._run_btn.font()
        run_font.setPointSize(run_font.pointSize() + 2)
        self._run_btn.setFont(run_font)
        self._run_btn.setStyleSheet("QPushButton { padding: 6px 18px; }")
        self._run_btn.setEnabled(False)
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
        self._reactions_text.setFont(QtGui.QFont("Courier New", 10))
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
        self._species_sliders = BatchSpeciesSliders(embedded=True)
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
        self._fine_btn.setToolTip(
            "Toggle fine adjustment mode for more precise slider control"
        )
        self._fine_btn.toggled.connect(
            lambda v: self._variable_sliders.set_fine_mode(v)
        )
        self._variable_sliders.fineModeReset.connect(
            self._sync_fine_button_state
        )
        slider_actions_layout.addWidget(self._fine_btn)

        # Override mode controls (Commit/Reset)
        self._commit_runtime_parameters_btn = QtWidgets.QPushButton("Apply")
        self._commit_runtime_parameters_btn.setObjectName(
            "commitRuntimeParametersButton"
        )
        self._commit_runtime_parameters_btn.setToolTip(
            "Apply current slider values to the canonical mechanism"
        )
        self._commit_runtime_parameters_btn.clicked.connect(self.sliderApplyRequested)
        slider_actions_layout.addWidget(self._commit_runtime_parameters_btn)

        self._reset_runtime_parameters_btn = QtWidgets.QPushButton("Reset")
        self._reset_runtime_parameters_btn.setObjectName("resetRuntimeParametersButton")
        self._reset_runtime_parameters_btn.setToolTip(
            "Revert sliders to canonical mechanism values"
        )
        self._reset_runtime_parameters_btn.clicked.connect(self.sliderResetRequested)
        slider_actions_layout.addWidget(self._reset_runtime_parameters_btn)
        self.set_slider_draft_actions_enabled(False)

        # Visibility picker for the unified slider surface.
        self._slider_visibility_picker_btn = QtWidgets.QToolButton()
        self._slider_visibility_picker_btn.setText("Visible sliders")
        self._slider_visibility_picker_btn.setObjectName("sliderVisibilityPickerButton")
        self._slider_visibility_picker_btn.setToolTip(
            "Choose which mechanism parameters to show as interactive sliders"
        )
        self._slider_visibility_picker_btn.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self._slider_visibility_menu = _PersistentToggleMenu(self)
        self._slider_visibility_menu.aboutToShow.connect(
            self._rebuild_slider_visibility_menu
        )
        self._slider_visibility_picker_btn.setMenu(self._slider_visibility_menu)
        slider_actions_layout.addWidget(self._slider_visibility_picker_btn)
        self._slider_edit_targets_label = make_bounded_label(
            "Slider edit targets: none",
            max_width=240,
        )
        self._slider_edit_targets_label.setToolTip(
            "The set whose initial conditions are controlled by concentration sliders"
        )
        target_font = self._slider_edit_targets_label.font()
        target_font.setPointSize(max(1, target_font.pointSize() - 1))
        self._slider_edit_targets_label.setFont(target_font)
        slider_actions_layout.addWidget(self._slider_edit_targets_label)
        slider_actions_layout.addStretch()

        # Bottom pane: slider header + controls + slider surface
        self._slider_pane_container = QtWidgets.QWidget()
        self._slider_pane_container.setObjectName("mechanismSliderPane")
        slider_pane_layout = QtWidgets.QVBoxLayout(self._slider_pane_container)
        slider_pane_layout.setContentsMargins(0, 0, 0, 0)
        slider_pane_layout.setSpacing(6)
        slider_pane_layout.addLayout(slider_actions_layout)
        self._slider_empty_state_label = QtWidgets.QLabel(
            "Choose sliders from Visible sliders."
        )
        self._slider_empty_state_label.setWordWrap(True)
        self._slider_empty_state_label.setAlignment(
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        )
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

        # Text changes capture literal drafts only. Scientific admission remains
        # owned by explicit Run/Preview/Fit actions.
        self._reactions_text.textChanged.connect(self._on_text_changed)

        self._tabs.addTab(self._reactions_tab, "Reactions")

        # Help tab — current directive and algebra reference
        self._help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(self._help_tab)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_scroll = make_scroll_area(self._help_tab)
        help_label = QtWidgets.QLabel(
            '<pre style="white-space: pre-wrap;"><b>Quick Start</b>\n'
            "Meaning: a mechanism is a readable list of settings, starting concentrations, reactions, and optional schedules.\n"
            "reaction: A -&gt; B ; k=0.5\n"
            "Put one directive or declaration on each line in Reactions, then use Run, Preview, or Fit. The same accepted source drives each action.\n"
            "\n"
            "<b>Global Settings</b>\n"
            "Meaning: these lines choose the source-wide energy unit and standard concentration.\n"
            "energy=kJ/mol\n"
            "C0=1.0\n"
            "Supported energy units are kJ/mol, kcal/mol, and J/mol. C0 is positive and in M. Each global setting may appear once anywhere in the source; duplicate definitions reject. Nominal temperature belongs to the mechanism tab. A source-level T= or per-reaction energy= is rejected.\n"
            "\n"
            "<b>Initial Conditions</b>\n"
            "Meaning: initial conditions are the concentrations at the start of a run, before reactions or interventions act.\n"
            "[A] = 1.0, [B] = 0.0\n"
            "You can also write init: A=1.0, B=0.0 or initial: A=1.0, B=0.0. Values must be finite and nonnegative. A named set uses one braced block, for example:\n"
            "start = {\n"
            "[A] = 1.0\n"
            "[B] = 0.0\n"
            "}\n"
            "\n"
            "<b>Reaction Arrows</b>\n"
            "Meaning: the arrow tells Kindred whether material can move only forward or in both directions.\n"
            "reaction: A -&gt; B ; k=0.5\n"
            "reaction: A &lt;-&gt; B ; kf=1.0 ; Keq=2.0\n"
            "Use -&gt; or =&gt; for irreversible steps, and &lt;-&gt; or &lt;=&gt; for reversible steps. &lt;- alone is not accepted. Stoichiometric terms may be written as A + 2B or 2*B.\n"
            "\n"
            "<b>Reaction Parameters</b>\n"
            "Meaning: these fields choose the forward-rate model and its authored scientific inputs.\n"
            "reaction: A -&gt; B ; dG_act=55 ; degeneracy=2\n"
            "Use k= or kf= for a positive constant rate; A= with Ea= selects Arrhenius; dG_act= selects Eyring. kappa= is an optional positive local transmission coefficient on that Eyring step and defaults to 1.\n"
            "degeneracy= is the positive integer count of equivalent Eyring pathways and defaults to 1. It multiplies the effective forward Eyring rate, but it is distinct from kappa, the transmission coefficient; it does not change the authored barrier. Do not encode the same pathway count in both a lowered barrier and degeneracy, or multiply the resulting rate again: that double counts.\n"
            "For reversible steps, kr= is the explicit concentration-form reverse rate when that is the selected authority; Keq= or dG_eq= can instead derive the reverse rate.\n"
            "\n"
            "<b>Equilibrium Lines</b>\n"
            "Meaning: an equilibrium line declares a reversible pair whose forward rate and thermodynamic relation determine the reverse rate.\n"
            "equilibrium: A &lt;=&gt; B ; kf=10.0 ; Keq=2.0\n"
            "Use a reversible arrow and kf= plus exactly one of kr=, Keq=, or dG_eq=. Keq= is concentration-form with units M^(nr-nf); dG_eq= uses Keq=exp(-dG_eq/(R*T))*C0^(nr-nf) in the selected global energy unit and nominal temperature. A thermodynamic kf=0 disables both directions; explicit kf+kr remains strictly positive. Supplying incompatible rate authorities, such as kr= with Keq=, is rejected.\n"
            "\n"
            "<b>Temperature Schedules</b>\n"
            "Meaning: a schedule makes the external temperature T(t) change during the run instead of staying at the nominal tab value.\n"
            "temp_step: t=[0,50,100], T=[298,320]\n"
            "As an alternative, use temp_response: t=[0,50,100], T=[298,320], tau=10. Use at most one schedule. Times are finite and nonnegative; a step schedule has one temperature per interval, and tau is positive.\n"
            "\n"
            "<b>Interventions</b>\n"
            "Meaning: interventions change or constrain a species during the run at an exact time or over an interval.\n"
            "intervention: op=set; species=A; time=1.0; value=2.0\n"
            "intervention: op=jump; species=A; time=2.0; delta=-0.25\n"
            "intervention: op=flux; species=B; start_time=0.0; end_time=5.0; flux_rate=-0.1\n"
            "intervention: op=hold; species=B; start_time=0.0; end_time=10.0; value=1.0\n"
            "set assigns a species concentration at one exact time. jump adds delta at one exact time. flux adds flux_rate to the derivative between start_time and end_time. hold fixes a species at value throughout that interval. Top-level fields use semicolons.\n"
            "\n"
            "<b>Parameterized Interventions</b>\n"
            "Meaning: a parameterized field keeps the schedule shape fixed while a named scalar supplies its value.\n"
            "param pulse_time = 2.0\n"
            "param pulse_delta = 0.2\n"
            "intervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_delta\n"
            "For each pair, provide exactly one literal field or its matching *_param field. The named scalar is resolved before execution and is checked against the same finite/domain rules as a literal.\n"
            "\n"
            "<b>Repeat Protocols</b>\n"
            "Meaning: a repeat protocol applies the same authored phase operations at regularly spaced starts.\n"
            "intervention: op=protocol; kind=repeat; start_time=0.5; every=1.25; count=4; before={op=jump,species=A,delta=0.2}\n"
            "before and after permit set or jump; during permits flux or hold. A protocol with during or after also needs a positive duration:\n"
            "intervention: op=protocol; kind=repeat; start_time=0.0; every=2.0; count=3; duration=1.0; during={op=hold,species=B,value=1.0}; after={op=set,species=B,value=0.0}\n"
            "Use one braced envelope per phase. A pipe separates ordered records inside an envelope, and commas separate their fields. Braces do not nest; semicolons are not allowed inside an envelope; nested timing, protocol, and trigger operations are not allowed.\n"
            "\n"
            "<b>State Triggers</b>\n"
            "Meaning: a trigger watches one simulated concentration and performs one action when the trajectory crosses its threshold.\n"
            "intervention: op=trigger; watch_species=A; threshold=0.8; direction=falling; max_count=1; min_interval=0.0; then={op=jump,species=B,delta=0.1}\n"
            "direction is rising, falling, or either (the default). then contains exactly one set or jump and has implicit trigger time, so it cannot contain time or time_param. Only an exact terminal solver root fires a trigger. Initial equality is neutral; the trajectory must first occupy a strict side and then make a genuine direction-compatible crossing. A firing cannot recursively fire another trigger at the same boundary.\n"
            "\n"
            "<b>Intervention Value Rules</b>\n"
            "Meaning: these domains keep every requested change physically representable and keep parameterized schedules safe after resolution.\n"
            "intervention: op=jump; species=A; time=2.0; delta=-0.25\n"
            "Times, thresholds, and min_interval are finite and nonnegative; end_time must be greater than start_time. set and hold values are finite and nonnegative. jump deltas and flux rates are finite signed values, including zero. Negative jumps saturate at zero, and negative flux never creates a negative state. every and required duration are strictly positive; count and max_count are positive non-boolean integers. Unknown, duplicate, missing, empty, or malformed fields reject.\n"
            "\n"
            "<b>Operations at the Same Time</b>\n"
            "Meaning: when several scheduled or triggered changes share a boundary, authored order and conflict checks determine one unambiguous result.\n"
            "intervention: op=set; species=A; time=1.0; value=2.0\n"
            "Kindred validates all static and exact-root trigger candidates before mutation. Then ending intervals end, ordinary and protocol-before operations run in authored order, entering holds are established, trigger actions run in declaration order, and protocol-after operations run in authored order.\n"
            "Conflicts reject unequal absolute sets, set plus jump, unequal overlapping holds for one species, and a hold start that is not an equal set. Equal overlapping holds are compatible. Scheduled changes, holds, trigger actions, sampled output, and manual sign checks do not create trigger roots.\n"
            "\n"
            "<b>Intervention Parameters and Protected Names</b>\n"
            "Meaning: only the documented parameter fields may be replaced by named values, and indexed mechanism names retain their step meaning.\n"
            "param pulse_time = 2.0\n"
            "intervention: op=set; species=A; time_param=pulse_time; value=1.0\n"
            "The complete *_param set is time_param, start_time_param, end_time_param, value_param, delta_param, flux_rate_param, and threshold_param. every, count, duration, max_count, and min_interval remain literal fields. Exact indexed names such as k1, kf1, kr1, and Keq1 resolve through the mechanism parameter namespace; use a longer name such as K1_test for an independent scalar or schedule parameter.\n"
            "\n"
            "<b>Inspecting Interventions</b>\n"
            "Meaning: Inspect... shows what an admitted schedule will do without editing it.\n"
            "intervention: op=hold; species=B; start_time=0.0; end_time=5.0; value=1.0\n"
            "Open Inspect... to read step numbers, symbolic right-hand-side equations, the authored schedule, typed operations, discontinuities, and repeat/trigger lineage. Plot annotations for executed schedule operations and exact trigger firings come from the run provenance and can be toggled in the plot context menu.\n"
            "\n"
            "<b>Parameters and Observables</b>\n"
            "Meaning: params are static scalar inputs; lets are calculated quantities that can be plotted or matched to observations.\n"
            "reaction: A -&gt; B ; k=0.5\n"
            "[A] = 1.0, [B] = 0.0\n"
            "param scale = 2.0\n"
            "let conversion = 1.0 - ([A] / max([A]_0, 1e-18))\n"
            "Use [A] for the current concentration and [A]_0 for its initial value. A param expression is static and cannot use concentrations, t, or T; a let can use species, row-local t, and row-local T. Bare name = expr is not accepted, and observable is not a keyword.\n"
            "\n"
            "<b>Built-in Functions</b>\n"
            "Meaning: built-ins provide safe, copyable math inside param and let expressions.\n"
            "reaction: A -&gt; B ; k=0.5\n"
            "let bounded = clip([A], 0, 1)\n"
            "Math: sqrt, ln, log10, log1p, exp, expm1, pow, abs, sin, cos, tan, and erf. Aggregation: min and max. Helpers: heaviside(x), clip(x, lo, hi), and ifelse(cond, a, b). Arithmetic, comparisons, &&, ||, !, and ** or ^ are also supported by the algebra language.\n"
            "\n"
            "<b>Protected Constants</b>\n"
            "Meaning: these read-only names supply physical constants or the current external temperature to expressions.\n"
            "reaction: A -&gt; B ; k=0.5\n"
            "let thermal = R * T\n"
            "k_B is Boltzmann's constant, h is Planck's constant, h_bar is the reduced Planck constant, N_A is Avogadro's constant, R is the gas constant in J/mol/K, and Rkcal is the gas constant in kcal/mol/K. T is the row-local external temperature in K for let observables. None may be redefined; T is not allowed in param expressions or as a source-level T= directive. Energies are normalized internally to J/mol.\n"
            "\n"
            "<b>Complete Example</b>\n"
            "Meaning: this complete source combines global settings, initial conditions, scalar params, both reaction families, an intervention, and an observable.\n"
            "energy=kJ/mol\n"
            "C0=1.0\n"
            "[A] = 1.0, [B] = 0.0, [C] = 0.0\n"
            "\n"
            "param pulse_time = 1.5\n"
            "param pulse_amount = 0.2\n"
            "\n"
            "reaction: A -&gt; B ; dG_act=55 ; degeneracy=2\n"
            "reaction: B &lt;=&gt; C ; kf=1.0 ; Keq=3.0\n"
            "intervention: op=jump; species=A; time_param=pulse_time; delta_param=pulse_amount\n"
            "let product = [C]</pre>"
        )
        help_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
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
        summary = str(text or "Slider edit targets: none")
        base_tooltip = (
            "The set whose initial conditions are controlled by concentration sliders"
        )
        tooltip = (
            base_tooltip
            if summary == "Slider edit targets: none"
            else f"{base_tooltip}\n\n{summary}"
        )
        set_bounded_label_text(
            self._slider_edit_targets_label,
            summary,
            max_width=240,
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
        self._slider_empty_state_label.setVisible(
            has_available_entries and not has_visible_entries
        )
        self._slider_visibility_picker_btn.setEnabled(has_available_entries)
        self._fine_btn.setVisible(has_variable_entries)
        self._slider_pane_container.setVisible(has_available_entries)

        if has_available_entries and not has_visible_entries:
            if has_variable_entries and has_species_entries:
                helper = (
                    "Choose mechanism or concentration sliders from Visible sliders."
                )
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

    def _sync_fine_button_state(self, enabled: bool) -> None:
        """Project child Fine state without re-entering the toggle handler."""
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

        mechanism_entries = list(self._variable_sliders.slider_picker_entries())
        species_entries = list(self._species_sliders.slider_picker_entries())
        invert_action = menu.addAction("Invert selection")
        invert_action.setEnabled(bool(mechanism_entries or species_entries))
        invert_action.triggered.connect(
            lambda _checked=False: self._invert_slider_visibility()
        )
        if not mechanism_entries and not species_entries:
            action = menu.addAction("No sliders available")
            action.setEnabled(False)
            return

        if mechanism_entries:
            self._add_slider_visibility_entries(
                menu,
                title="Mechanism",
                entry_kind="mechanism",
                entries=mechanism_entries,
            )
        if mechanism_entries and species_entries:
            menu.addSeparator()
        if species_entries:
            self._add_slider_visibility_entries(
                menu,
                title="Initial concentrations",
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
            self._species_sliders.set_species_visible(str(item_name), bool(visible))
        self._sync_slider_workspace_state()

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
    ) -> None:
        enabled = bool(dirty)
        self._commit_runtime_parameters_btn.setEnabled(enabled)
        self._reset_runtime_parameters_btn.setEnabled(enabled)
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
