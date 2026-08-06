# kindred/gui/widgets/mechanism_editor.py
"""Tabbed mechanism editor for Reactions and Notes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable, Optional

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
from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport

__all__ = [
    "DetailedBalanceCorrectionOutcome",
    "DetailedBalanceCorrectionRequest",
    "MechanismEditorBinding",
    "MechanismEditorTabbed",
    "open_detailed_balance_correction_modal",
]


@dataclass(frozen=True, slots=True)
class _MechanismEditorRenderProjection:
    source: object
    notes: str
    slider_dirty: bool
    document_identity: tuple[int, str]


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionRequest:
    request_id: str
    origin: str
    tab_id: str
    source: str
    issue: str
    temperature_K: float = 298.15
    report: WegscheiderCyclicityReport | None = None
    source_energy_unit: str = "kJ/mol"

    def __post_init__(self) -> None:
        for field_name in ("request_id", "origin", "tab_id", "issue"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonblank.")
        if str(self.origin).strip().casefold() not in {"run", "fit", "preview"}:
            raise ValueError("origin must be Run, Fit, or Preview.")
        if not str(self.source or "").strip():
            raise ValueError("source must be nonblank.")
        if not isinstance(self.report, WegscheiderCyclicityReport):
            raise TypeError(
                "Detailed-balance correction requires the accepted cyclicity report."
            )
        if not self.report.unresolved_cycles:
            raise ValueError("Detailed-balance report has no unresolved cycles.")
        temperature = float(self.temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_K must be positive and finite.")
        source_energy_unit = str(self.source_energy_unit or "").strip()
        if not source_energy_unit:
            raise ValueError("source_energy_unit must be nonblank.")
        object.__setattr__(self, "temperature_K", temperature)
        object.__setattr__(self, "source_energy_unit", source_energy_unit)


@dataclass(frozen=True, slots=True)
class DetailedBalanceCorrectionOutcome:
    request_id: str
    origin: str
    tab_id: str
    source: str
    issue: str
    corrected_source: str | None
    cancelled: bool

    def __post_init__(self) -> None:
        for field_name in ("request_id", "origin", "tab_id", "source", "issue"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be nonblank.")
        if str(self.origin).strip().casefold() not in {"run", "fit", "preview"}:
            raise ValueError("origin must be Run, Fit, or Preview.")
        if type(self.cancelled) is not bool:
            raise TypeError("cancelled must be a bool.")
        corrected = self.corrected_source
        if corrected is not None and not str(corrected).strip():
            raise ValueError("corrected_source must be nonblank when provided.")
        if self.cancelled == (corrected is not None):
            raise ValueError(
                "Detailed-balance outcome must be either cancelled or corrected."
            )


class _DetailedBalanceCorrectionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        request: DetailedBalanceCorrectionRequest,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._request = request
        self._corrected_source: str | None = None
        self.setWindowTitle("Fix Detailed Balance")
        self.setModal(True)
        if not isinstance(request.report, WegscheiderCyclicityReport):
            raise TypeError("Detailed-balance request has no accepted report.")
        self._report = request.report
        if not self._report.unresolved_cycles:
            raise ValueError("The accepted source has no unresolved detailed-balance cycle.")

        layout = QtWidgets.QVBoxLayout(self)
        issue = QtWidgets.QLabel(request.issue, self)
        issue.setWordWrap(True)
        layout.addWidget(issue)
        form = QtWidgets.QFormLayout()
        self._selectors: dict[str, QtWidgets.QComboBox] = {}
        for cycle in self._report.unresolved_cycles:
            selector = QtWidgets.QComboBox(self)
            selector.addItems(list(cycle.parameter_names))
            self._selectors[cycle.cycle_id] = selector
            label = f"{cycle.cycle_id} dependent parameter"
            if any(name.startswith("dG_eq") for name in cycle.parameter_names):
                label += f" ({request.source_energy_unit})"
            form.addRow(
                label,
                selector,
            )
        layout.addLayout(form)
        explanation = QtWidgets.QLabel(
            "Kindred will add one symbolic dependency for each cycle. The "
            "detached Reactions edit is accepted and reverified by the owning "
            "Run or Fit workflow before execution continues.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_correction)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def corrected_source(self) -> str | None:
        return self._corrected_source

    @QtCore.Slot()
    def _accept_correction(self) -> None:
        from kindred.core.simulator.wegscheider_symbolic import (
            apply_wegscheider_resolution_to_reactions_text,
            build_wegscheider_resolution_updates,
        )

        selected = {
            cycle_id: selector.currentText()
            for cycle_id, selector in self._selectors.items()
        }
        try:
            updates = build_wegscheider_resolution_updates(
                self._report,
                selected,
            )
            corrected = apply_wegscheider_resolution_to_reactions_text(
                self._request.source,
                updates,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Detailed Balance Not Corrected",
                str(exc),
            )
            return
        self._corrected_source = corrected
        self.accept()


def open_detailed_balance_correction_modal(
    parent: QtWidgets.QWidget | None,
    request: DetailedBalanceCorrectionRequest,
) -> DetailedBalanceCorrectionOutcome:
    """Run the sole detailed-balance modal and return typed truth."""

    if not isinstance(request, DetailedBalanceCorrectionRequest):
        raise TypeError("request must be DetailedBalanceCorrectionRequest.")
    dialog = _DetailedBalanceCorrectionDialog(request, parent)
    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    corrected = dialog.corrected_source if accepted else None
    return DetailedBalanceCorrectionOutcome(
        request_id=request.request_id,
        origin=request.origin,
        tab_id=request.tab_id,
        source=request.source,
        issue=request.issue,
        corrected_source=corrected,
        cancelled=not accepted,
    )


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
    speciesModeChanged = Signal(bool)
    speciesResetRequested = Signal()
    mechanismInspectRequested = Signal()
    sourceDraftEdited = Signal(object)
    notesEdited = Signal(str)
    sliderApplyRequested = Signal()
    sliderResetRequested = Signal()
    validationStateChanged = Signal(str)
    nativeHistoryAvailabilityChanged = Signal(bool, bool)
    detailedBalanceCorrectionCompleted = Signal(object)

    """
    Mechanism editor with Reactions and Notes tabs.

    Features:
    - Reactions tab: DSL text editor for reactions and algebra declarations
    - Notes tab: persisted free-form text (never parsed or injected)

    The reaction text editor is directly editable. It emits visible authoring
    and validation intent; scientific action boundaries own reconciliation.
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
        self._validation_temperature_provider: Callable[[], float] = (
            lambda: 298.15
        )

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
        self._detailed_balance_fix_button = QtWidgets.QPushButton(
            "Fix Detailed Balance…",
            self,
        )
        self._detailed_balance_fix_button.setObjectName(
            "detailedBalanceFixButton"
        )
        self._detailed_balance_fix_button.hide()
        self._pending_detailed_balance_request: (
            DetailedBalanceCorrectionRequest | None
        ) = None
        self._detailed_balance_context_provider: (
            Callable[[], tuple[str, str]] | None
        ) = None
        self._detailed_balance_fix_button.clicked.connect(
            self._run_pending_detailed_balance_fix
        )
        reactions_layout.addWidget(self._detailed_balance_fix_button)

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

        # Reserved for explicit validation requests; text edits do not parse drafts.
        self._validation_timer = QtCore.QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(100)
        self._validation_timer.timeout.connect(self._validate_dsl)

        # Text changes mark drafts only. MainWindow owns dirty equality refresh.
        self._reactions_text.textChanged.connect(self._on_text_changed)

        self._tabs.addTab(self._reactions_tab, "Reactions")

        # Help tab — current directive and algebra reference
        self._help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(self._help_tab)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_scroll = make_scroll_area(self._help_tab)
        help_label = QtWidgets.QLabel(
            '<pre style="white-space: pre-wrap;"><b>Global Directives</b>\n'
            "energy=kJ/mol       (Supported: kJ/mol, kcal/mol, J/mol)\n"
            "Nominal temperature is owned by the mechanism tab; a source-level T= directive is rejected.\n"
            "C0=1.0              (Standard concentration in M)\n"
            "kappa=1.0           (Transmission coefficient; also κ=)\n"
            "\n"
            "<b>Initial Conditions</b>\n"
            "Ordinary source, presets, and intervention examples use the same typed [A]=, init:, initial:, and named-block admission. Acceptance materializes canonical sets and leaves deterministic commented provenance in Reactions.\n"
            "\n"
            "<b>Arrows</b>\n"
            "Irreversible: -&gt; or =&gt;    Reversible: &lt;-&gt; or &lt;=&gt;\n"
            "\n"
            "<b>Reaction Parameters</b>\n"
            "k= or kf=  (forward rate)    kr=  (reverse rate)\n"
            "Keq=        (equilibrium constant)\n"
            "dG_act=     (Eyring activation free energy)\n"
            "dG_eq=      (equilibrium free energy)\n"
            "A=, Ea=     (Arrhenius pre-exponential and activation energy)\n"
            "Per-step: κ= overrides global kappa for Eyring computation.\n"
            "\n"
            "<b>Equilibrium Lines</b>\n"
            "equilibrium: A &lt;=&gt; B ; Keq=2.5 ; kf=10\n"
            "Requires kf= plus exactly one of kr=, Keq=, or dG_eq=.\n"
            "Keq= without kf=, kr= plus Keq=, and all-three rate forms are rejected.\n"
            "\n"
            "<b>Temperature Schedules</b>\n"
            "temp_step: t=[0,50,100], T=[298,320]\n"
            "temp_response: t=[0,50,100], T=[298,320], tau=10\n"
            "\n"
            "<b>Species Intervention Schedules</b>\n"
            "The public operations are set, jump, flux, hold, protocol, and trigger. Top-level fields are separated by semicolons.\n"
            "intervention: op=set; species=A; time=1.0; value=2.0\n"
            "intervention: op=jump; species=A; time=2.0; delta=-0.25\n"
            "intervention: op=flux; species=B; start_time=0.0; end_time=5.0; flux_rate=-0.1\n"
            "intervention: op=hold; species=O2; start_time=0.0; end_time=10.0; value=1.0\n"
            "set requires species, exactly one time/time_param, and exactly one value/value_param. jump replaces the last pair with exactly one delta/delta_param.\n"
            "flux requires species and exactly one each of start_time/start_time_param, end_time/end_time_param, and flux_rate/flux_rate_param. hold uses the same timing fields and exactly one value/value_param.\n"
            "Each ordinary numeric field above is replaced, not supplemented, by its matching parameter field when parameterized.\n"
            "The exhaustive parameter fields are time_param, start_time_param, end_time_param, value_param, delta_param, flux_rate_param, and threshold_param.\n"
            "\n"
            "<b>Repeat Protocols</b>\n"
            "intervention: op=protocol; kind=repeat; start_time=0.5; every=1.25; count=4; before={op=jump,species=Oxidant,delta=0.2}\n"
            "intervention: op=protocol; kind=repeat; start_time=0.0; every=2.0; count=3; duration=1.0; during={op=hold,species=Light,value=1.0}; after={op=set,species=Light,value=0.0}\n"
            "before={op=set,species=A,value=1|op=jump,species=B,delta=0.2}\n"
            "during={op=flux,species=C,flux_rate=-0.1|op=hold,species=D,value=4}\n"
            "before, during, and after each contain one braced envelope. A pipe separates ordered operation records; a comma separates fields.\n"
            "before and after allow set or jump. during allows flux or hold. Nested operations declare op and species exactly once and have implicit protocol timing; nested timing, protocol, and trigger operations are forbidden.\n"
            "Nested set/hold use exactly one value/value_param, jump uses exactly one delta/delta_param, and flux uses exactly one flux_rate/flux_rate_param.\n"
            "Protocol requires kind=repeat, exactly one start_time/start_time_param, a positive finite literal every, a positive integer literal count, and at least one nonempty phase.\n"
            "A positive finite literal duration is required exactly when during or after is present. Braces do not nest; semicolons are forbidden inside an envelope; brace, pipe, and comma are forbidden in values.\n"
            "Empty records or fields and unmatched braces reject.\n"
            "\n"
            "<b>State Triggers</b>\n"
            "intervention: op=trigger; watch_species=A; threshold=0.8; direction=falling; max_count=1; min_interval=0.0; then={op=jump,species=B,delta=0.1}\n"
            "Trigger requires watch_species, exactly one threshold/threshold_param, a positive integer literal max_count, a finite nonnegative literal min_interval, and one braced then action.\n"
            "If supplied, direction is rising, falling, or either; omission defaults to either. then allows exactly one set or jump with implicit trigger time, so time/time_param is not used inside then.\n"
            "Only an exact terminal solver root can fire a trigger. After firing at T, it rearms no earlier than max(T + min_interval, nextafter(T, +infinity)); a firing cannot recursively trigger another firing at the same boundary.\n"
            "Initial equality is neutral. A trigger becomes eligible only after the trajectory occupies a strict side and then makes a genuine direction-compatible crossing. Count and firing provenance advance atomically only after the action succeeds.\n"
            "\n"
            "<b>Intervention Domains and Ordering</b>\n"
            "Literal and resolved time, start_time, end_time, threshold, and min_interval are finite and nonnegative; resolved end_time is greater than start_time.\n"
            "Set/hold value is finite and nonnegative. Jump delta and flux flux_rate are finite signed values, including zero. Negative jumps saturate at zero; negative flux never creates negative state.\n"
            "every and required duration are finite and strictly positive. count and max_count are positive non-boolean integers. Resolved parameters and derived times are validated under the same domains and must remain finite.\n"
            "Unknown, duplicate, missing, or empty fields reject at top level and inside protocol/trigger records.\n"
            "Declaration order and nested authored-operation order are semantic. At one boundary, all static and exact-root trigger candidates are validated before mutation; ending intervals end; ordinary and protocol-before operations run in authored order; entering holds are established; trigger actions run in declaration order; then protocol-after operations run in authored order.\n"
            "That one conflict check covers static/static, trigger/trigger, and trigger/static candidates. It rejects unequal absolute sets, set plus jump, unequal overlapping holds for one species, and a hold-start instant other than an equal set. Equal overlapping holds are compatible and keep complete lineage.\n"
            "Scheduled changes, holds, trigger actions, sampled output, and manual sign checks do not create trigger roots.\n"
            "Protected indexed names such as K1, k1, kf1, kr1, and Keq1 resolve through the mechanism parameter namespace; use a longer ordinary name such as K1_test for an independent schedule parameter.\n"
            "Use Examples &gt; Intervention Examples for runnable final-schema examples.\n"
            "Use Inspect... to open the Mechanism Inspector for read-only step numbers, symbolic RHS equations, declarative payload, typed executable operations, discontinuities, and protocol lineage from core authorities.\n"
            "Executed schedule operations and exact-root trigger firings appear as main-plot annotations from solver provenance by default and can be toggled from the plot context menu.\n"
            "\n"
            "<b>Algebra Declarations and Observables</b>\n"
            "Write algebra declarations directly in the Reactions text. They may appear before, after, or between reaction lines.\n"
            "param name = expr      Scalar solver parameter. Use for rate constants, energies, and other fittable inputs.\n"
            "let name = expr        Algebraic observable. Use for derived quantities to plot or fit against data.\n"
            "Bare name = expr       Not supported. Use let or param explicitly.\n"
            "Bracketed species like [A] and [A]_0 are valid in observables, not in param declarations.\n"
            "'observable' is not a keyword. Use let for observables.\n"
            "\n"
            "<b>Built-in Functions</b>\n"
            "Math: sqrt(x), ln(x), log10(x), log1p(x), exp(x), expm1(x), pow(x, y), abs(x)\n"
            "Trigonometry: sin(x), cos(x), tan(x)\n"
            "Aggregation: min(...), max(...)\n"
            "Special: erf(x), heaviside(x), clip(x, lo, hi), ifelse(cond, a, b)\n"
            "\n"
            "<b>Protected Constants</b>\n"
            "k_B   Boltzmann constant (read-only)\n"
            "h     Planck constant (read-only)\n"
            "N_A   Avogadro constant (read-only)\n"
            "R     Gas constant in J/mol/K (read-only)\n"
            "Rkcal Gas constant in kcal/mol/K (read-only)\n"
            "T     Actual row-local external temperature T(t) in K for let observables (read-only; not allowed in param)\n"
            "\n"
            "Note: Kindred normalizes all energies internally to J/mol.\n"
            "\n"
            "<b>Examples</b>\n"
            "Grouped declarations:\n"
            "reaction: 2A + B =&gt; C ; kf=1e5\n"
            "equilibrium: C &lt;=&gt; D ; Keq=2.5 ; kf=10.0\n"
            "reaction: C + E -&gt; F ; k=0.5\n"
            "reaction: A -&gt; A_Side ; kf=0.01\n"
            "\n"
            "param scale = 2.0\n"
            "param k_base = 1.5e3\n"
            "param k_derived = k_base * scale\n"
            "let total_A = [A] + [A_Side]\n"
            "let conversion = 1.0 - ([A] / max([A]_0, 1e-18))\n"
            "\n"
            "Interleaved declarations:\n"
            "reaction: A -&gt; B ; k=1.0\n"
            "param scale = 2.0\n"
            "reaction: B -&gt; C ; k=0.5\n"
            "let yield_C = [C] / max([A]_0, 1e-18)\n"
            "equilibrium: C &lt;=&gt; D ; Keq=2.5 ; kf=10.0\n"
            "let total_CD = [C] + [D]\n"
            "\n"
            "Reaction arrows: -&gt; for irreversible, &lt;-&gt; or &lt;=&gt; for reversible. &lt;- is not accepted.</pre>"
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
        """Handle text change event."""
        self._validation_timer.stop()
        self._set_validation_state(
            "draft",
            "Visible Reactions have changed; scientific actions will validate them.",
        )
        self._on_source_draft_edited()
        self._validation_timer.start()

    def _on_source_draft_edited(self) -> None:
        from kindred.core.mechanism_source import MechanismAuthoringSource

        if self._current_validation_state != "draft":
            self._set_validation_state(
                "draft",
                "Visible mechanism source has changed; scientific actions will validate it.",
            )
            self._validation_timer.start()
        self.sourceDraftEdited.emit(
            MechanismAuthoringSource.from_parts(
                reactions_text=self.reactions_text(),
            )
        )

    def _validate_dsl(self):
        """Validate DSL text and update validation indicator."""
        text = self._reactions_text.toPlainText()

        # Skip validation if text is empty
        if not text.strip():
            self._set_validation_state("idle")
            return

        # Try to parse DSL
        try:
            from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
            from kindred.core.simulator.parameter_algebra import (
                apply_parameter_algebra_spec_to_mechanism,
            )
            from kindred.core.mechanism_source import MechanismAuthoringSource
            from kindred.core.units import UnitsModel

            validation_text = MechanismAuthoringSource.from_parts(
                reactions_text=text,
            ).reactions_text

            prepared_dsl = prepare_dsl_request_set(
                validation_text,
                initials={},
                units=UnitsModel(
                    temperature_K=float(
                        self._validation_temperature_provider()
                    )
                ),
            )
            mechanism = prepared_dsl.mechanism_for_execution()
            apply_parameter_algebra_spec_to_mechanism(
                prepared_dsl.parameter_algebra_spec_for_execution(),
                mechanism=mechanism,
                require_mutable=False,
            )

            # Success - show green check
            n_reactions = len(mechanism.reactions)
            n_equilibria = len(mechanism.equilibria)
            n_species = len(mechanism.species)

            msg = f"✓ Valid: {n_species} species, {n_reactions} reactions, {n_equilibria} equilibria"
            spec = prepared_dsl.parameter_algebra_spec_for_execution()
            for warning in getattr(spec, "override_warnings", ()) or ():
                msg += f"\nWarning: {warning.message}"
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
            One of: "idle", "draft", "validating", "valid", "invalid"
        message : str
            Status message to display
        """
        self._current_validation_state = state
        self.validationStateChanged.emit(str(state))
        if state == "valid":
            run_tooltip = (
                "Run simulation for the current run target from Initial Conditions"
            )
        elif state == "draft":
            run_tooltip = "Visible Reactions will be validated before Run."
        elif state == "validating":
            run_tooltip = "Mechanism validation is still running."
        else:
            run_tooltip = (
                "No valid mechanism — enter a valid reaction mechanism to enable"
            )
        self._run_btn.setToolTip(run_tooltip)

        if state == "idle":
            self._set_validation_label_text("")
            self._validation_label.setStyleSheet("QLabel { padding: 4px; }")

        elif state == "validating":
            self._set_validation_label_text("\u23f3 Validating...")
            self._validation_label.setStyleSheet(
                "QLabel { padding: 4px; border-radius: 3px; }"
            )

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

    def reset_slider_session_presentation(self) -> None:
        """Restore the owned Slider widgets to a neutral project session."""

        variable_blocker = QtCore.QSignalBlocker(self._variable_sliders)
        species_blocker = QtCore.QSignalBlocker(self._species_sliders)
        try:
            self._variable_sliders.clear(preserve_visibility=False)
            self._species_sliders.clear(preserve_visibility=False)
            self._fine_btn.setChecked(False)
        finally:
            del species_blocker
            del variable_blocker
        self.clear_preview_detailed_balance_block()
        self.set_slider_draft_actions_enabled(False)
        self.set_slider_edit_targets_summary("Slider edit targets: none")
        self.render_slider_workspace_presentation()

    def is_mechanism_valid(self) -> bool:
        return self._current_validation_state == "valid"

    def set_reactions_status_text(self, text: str, *, state: str) -> None:
        self._set_validation_state(str(state or "idle"), str(text or ""))

    def present_detailed_balance_correction(
        self,
        request: DetailedBalanceCorrectionRequest,
    ) -> DetailedBalanceCorrectionOutcome:
        """Present the one modal used by Run/Fit correction continuation."""

        if not isinstance(request, DetailedBalanceCorrectionRequest):
            raise TypeError("request must be DetailedBalanceCorrectionRequest.")
        if request.origin.strip().casefold() not in {"run", "fit"}:
            raise ValueError(
                "Immediate detailed-balance correction requires Run or Fit origin."
            )
        if not self._detailed_balance_request_is_current(request):
            outcome = self._cancelled_detailed_balance_outcome(request)
            self.detailedBalanceCorrectionCompleted.emit(outcome)
            return outcome
        outcome = open_detailed_balance_correction_modal(self, request)
        if (
            outcome.corrected_source is not None
            and not self._detailed_balance_request_is_current(request)
        ):
            outcome = self._cancelled_detailed_balance_outcome(request)
        if outcome.corrected_source is not None:
            self.apply_detailed_balance_rewrite(outcome.corrected_source)
        self.detailedBalanceCorrectionCompleted.emit(outcome)
        return outcome

    def show_preview_detailed_balance_block(
        self,
        request: DetailedBalanceCorrectionRequest,
    ) -> None:
        """Show Preview's clear blocked message and explicit Fix action."""

        if not isinstance(request, DetailedBalanceCorrectionRequest):
            raise TypeError("request must be DetailedBalanceCorrectionRequest.")
        if request.origin.strip().casefold() != "preview":
            raise ValueError("Preview Fix requires a Preview-origin request.")
        self._pending_detailed_balance_request = request
        self.set_reactions_status_text(
            f"Preview blocked: {request.issue}",
            state="invalid",
        )
        self._detailed_balance_fix_button.show()

    def clear_preview_detailed_balance_block(self) -> None:
        self._pending_detailed_balance_request = None
        self._detailed_balance_fix_button.hide()

    @QtCore.Slot()
    def _run_pending_detailed_balance_fix(self) -> None:
        request = self._pending_detailed_balance_request
        if request is None:
            return
        if not self._detailed_balance_request_is_current(request):
            self.detailedBalanceCorrectionCompleted.emit(
                self._cancelled_detailed_balance_outcome(request)
            )
            return
        outcome = open_detailed_balance_correction_modal(self, request)
        if (
            outcome.corrected_source is not None
            and not self._detailed_balance_request_is_current(request)
        ):
            outcome = self._cancelled_detailed_balance_outcome(request)
        if outcome.corrected_source is not None:
            self.apply_detailed_balance_rewrite(outcome.corrected_source)
            self.clear_preview_detailed_balance_block()
            self.set_reactions_status_text(
                "Detailed-balance correction applied. Preview was not "
                "retried; adjust a Slider to request a new Preview.",
                state="draft",
            )
        self.detailedBalanceCorrectionCompleted.emit(outcome)

    def set_detailed_balance_context_provider(
        self,
        provider: Callable[[], tuple[str, str]],
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable.")
        self._detailed_balance_context_provider = provider

    def set_validation_temperature_provider(
        self,
        provider: Callable[[], float],
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable.")
        self._validation_temperature_provider = provider

    def _detailed_balance_request_is_current(
        self,
        request: DetailedBalanceCorrectionRequest,
    ) -> bool:
        provider = self._detailed_balance_context_provider
        if provider is None:
            return False
        try:
            tab_id, source = provider()
        except (KeyError, RuntimeError, ValueError):
            return False
        return str(tab_id) == request.tab_id and str(source) == request.source

    @staticmethod
    def _cancelled_detailed_balance_outcome(
        request: DetailedBalanceCorrectionRequest,
    ) -> DetailedBalanceCorrectionOutcome:
        return DetailedBalanceCorrectionOutcome(
            request_id=request.request_id,
            origin=request.origin,
            tab_id=request.tab_id,
            source=request.source,
            issue=request.issue,
            corrected_source=None,
            cancelled=True,
        )

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

    def apply_detailed_balance_rewrite(self, corrected_text: str) -> bool:
        """Install a correction as exactly one native Reactions edit."""

        replacement = str(corrected_text)
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
        slider_dirty: bool,
        refresh_validation: bool = True,
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
            self.set_slider_draft_actions_enabled(
                bool(slider_dirty),
            )
            if bool(refresh_validation):
                self.refresh_validation_indicator()
        finally:
            del notes_blocker
            del reactions_blocker
            del editor_blocker

    def refresh_validation_indicator(self) -> None:
        """Refresh validation from the literal visible authoring snapshot."""
        self._validation_timer.stop()
        self._validate_dsl()

    def variable_sliders_widget(self) -> VariableSliders:
        return self._variable_sliders

    def notes_text(self) -> str:
        return str(self._notes_text.toPlainText())

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
    """Bind one retained editor to one immutable mechanism-tab identity."""

    draftStateChanged = QtCore.Signal(str, bool)
    errorRaised = QtCore.Signal(str)

    def __init__(
        self,
        *,
        tab_id: str,
        editor: MechanismEditorTabbed,
        registry,
        simulation_jobs,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        registry.tab(target)
        self._tab_id = target
        self._editor = editor
        self._registry = registry
        self._simulation_jobs = simulation_jobs
        self._rendering = False
        self._rendered_document_identity: tuple[int, str] | None = None
        editor.set_detailed_balance_context_provider(
            self._detailed_balance_context
        )
        editor.set_validation_temperature_provider(
            lambda: self._registry.tab(self._tab_id).scientific_settings.temperature
        )
        editor.sourceDraftEdited.connect(self._on_source_draft)
        editor.notesEdited.connect(self._on_notes_edited)

    @property
    def tab_id(self) -> str:
        return self._tab_id

    def _detailed_balance_context(self) -> tuple[str, str]:
        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        source = runtime.reactions_draft or tab.mechanism_source
        return tab.tab_id, source.reactions_text

    def prepare_render(self) -> _MechanismEditorRenderProjection:
        """Prepare every fallible authoring projection before widget mutation."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        source = runtime.reactions_draft or tab.mechanism_source
        document_identity = self._registry.document_identity(self._tab_id)
        if (
            type(document_identity) is not tuple
            or len(document_identity) != 2
        ):
            raise TypeError(
                "document_identity must be (generation, tab_id)."
            )
        return _MechanismEditorRenderProjection(
            source=source,
            notes=(
                tab.notes
                if runtime.notes_draft is None
                else runtime.notes_draft
            ),
            slider_dirty=bool(
                runtime.slider_parameter_draft
                or runtime.slider_species_drafts_by_set_id
            ),
            document_identity=document_identity,
        )

    def apply_render_projection(
        self,
        projection: _MechanismEditorRenderProjection,
        *,
        refresh_validation: bool = True,
    ) -> None:
        """Apply one fully prepared authoring projection."""

        if not isinstance(projection, _MechanismEditorRenderProjection):
            raise TypeError(
                "projection must be _MechanismEditorRenderProjection."
            )
        identity_changed = (
            projection.document_identity != self._rendered_document_identity
        )
        self._rendering = True
        try:
            self._editor.render_authoring_snapshot(
                source=projection.source,
                notes=projection.notes,
                slider_dirty=projection.slider_dirty,
                refresh_validation=bool(refresh_validation),
            )
            if identity_changed:
                self._editor.clear_native_text_history()
            self._rendered_document_identity = projection.document_identity
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
        was_canonical = runtime.reactions_draft is None
        runtime.reactions_draft = None if source == tab.mechanism_source else source
        if was_canonical and runtime.reactions_draft is not None:
            self._simulation_jobs.deauthorize_targets(
                tab.tab_id,
                (item.set_id for item in tab.initial_condition_sets),
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

    def has_reactions_draft(self) -> bool:
        return self._registry.tab_runtime(self._tab_id).reactions_draft is not None

    def apply_accepted_source_rewrite(
        self,
        visible: MechanismAuthoringSource,
        accepted: MechanismAuthoringSource,
    ) -> bool:
        """Keep deterministic parser rewrites inside native Reactions history."""

        from kindred.core.mechanism_source import MechanismAuthoringSource

        if not isinstance(visible, MechanismAuthoringSource):
            raise TypeError("visible must be MechanismAuthoringSource.")
        if not isinstance(accepted, MechanismAuthoringSource):
            raise TypeError("accepted must be MechanismAuthoringSource.")
        return self._editor.apply_accepted_reactions_rewrite(
            expected_visible_text=visible.reactions_text,
            accepted_text=accepted.reactions_text,
        )

    def discard_reactions_draft(self) -> bool:
        runtime = self._registry.tab_runtime(self._tab_id)
        changed = runtime.reactions_draft is not None
        runtime.reactions_draft = None
        self.render()
        return changed
