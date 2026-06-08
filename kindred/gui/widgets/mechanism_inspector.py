"""Read-only mechanism inspector backed by core parsing authorities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable, Mapping

from PySide6 import QtWidgets

from kindred.core.batch_initial_conditions import strip_named_reaction_dsl_initial_concentration_sets
from kindred.core.intervention_schedule_compiler import compile_intervention_schedule
from kindred.core.intervention_schedule import InterventionScheduleError, intervention_schedule_parameter_names
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map, iter_canonical_parameters
from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

__all__ = ["MechanismInspectorDialog", "MechanismInspectorSections", "build_mechanism_inspector_sections"]


@dataclass(frozen=True, slots=True)
class MechanismInspectorSections:
    steps_text: str
    equations_text: str
    interventions_text: str


def build_mechanism_inspector_sections(dsl_text: str) -> MechanismInspectorSections:
    """Build read-only inspector sections from the same core authorities used at runtime."""
    parse_text = strip_named_reaction_dsl_initial_concentration_sets(str(dsl_text or ""))
    mechanism = parse_dsl_to_mechanism(parse_text, initials={})
    apply_parameter_algebra_to_mechanism(parse_text, mechanism=mechanism, require_mutable=False)
    return MechanismInspectorSections(
        steps_text=_safe_format_section("steps", lambda: _format_steps(mechanism)),
        equations_text=_safe_format_section("equations", lambda: _format_equations(mechanism)),
        interventions_text=_safe_format_section("interventions", lambda: _format_interventions(mechanism)),
    )


def _safe_format_section(label: str, formatter: Callable[[], str]) -> str:
    try:
        return formatter()
    except Exception as exc:
        return f"Unable to inspect {label}:\n{exc}"


def _format_steps(mechanism: object) -> str:
    entries = get_step_index_map(mechanism)
    parameter_names_by_step_index: dict[int, list[str]] = {}
    for param_name, entry, _role in iter_canonical_parameters(mechanism):
        parameter_names_by_step_index.setdefault(int(entry["step_index"]), []).append(param_name)

    lines: list[str] = []
    if entries:
        for entry in entries:
            step_index = int(entry["step_index"])
            kind = str(entry["kind"])
            context = str(entry["context"])
            lines.append(f"Step {step_index} {kind}: {context}")
            parameter_names = parameter_names_by_step_index.get(step_index, [])
            if parameter_names:
                lines.append(f"  Parameters: {', '.join(parameter_names)}")
            authority = entry.get("equilibrium_authority")
            if authority:
                lines.append(f"  Equilibrium authority: {json.dumps(authority, sort_keys=True)}")
    else:
        lines.append("No indexed DSL reaction or equilibrium steps.")

    state_network_summary = _format_state_network_summary(mechanism)
    if state_network_summary:
        if lines:
            lines.append("")
        lines.extend(state_network_summary)
    return "\n".join(lines)


def _format_equations(mechanism: object) -> str:
    if _state_network_metadata(mechanism):
        return "\n".join(
            [
                "Symbolic RHS equations are unavailable for mechanisms with state-network definitions.",
                "State-network generated reactions/equilibria are included in simulation, but they are not represented in indexed symbolic RHS output.",
            ]
        )

    structure = build_symbolic_jacobian_structure(mechanism)
    schedule = getattr(mechanism, "metadata", {}).get(MechanismMetadataKeys.INTERVENTION_SCHEDULE)
    lines = ["Base symbolic RHS equations:"]
    if schedule is not None:
        lines.append("Note: intervention schedules are compiled execution inputs; equations here are the base unscheduled mechanism RHS.")
        lines.append("")
    for species_name, rhs_expression in zip(
        structure.species_names,
        structure.rhs_expressions,
        strict=True,
    ):
        lines.append(f"d[{species_name}]/dt = {rhs_expression}")
    lines.append("")
    lines.append("Jacobian rows:")
    for species_name, row in zip(structure.species_names, structure.jacobian_expressions, strict=True):
        lines.append(f"{species_name}: {', '.join(row)}")
    lines.append("")
    lines.append(f"Structure fingerprint: {structure.structure_fingerprint}")
    lines.append(f"Parameter symbols: {', '.join(structure.parameter_symbols) or 'none'}")
    return "\n".join(lines)


def _format_interventions(mechanism: object) -> str:
    schedule = getattr(mechanism, "metadata", {}).get(MechanismMetadataKeys.INTERVENTION_SCHEDULE)
    if schedule is None:
        return "No intervention schedule."

    lines = [
        "Declarative schedule payload:",
        json.dumps(schedule.to_payload(), indent=2, sort_keys=True),
        "",
        f"Declarative schedule fingerprint: {schedule.fingerprint or 'none'}",
    ]
    if schedule.is_parameterized:
        parameter_names = sorted(intervention_schedule_parameter_names(schedule))
        lines.extend(
            [
                "",
                "Executable schedule payload is unavailable until schedule parameters are resolved.",
            ]
        )
        if parameter_names:
            lines.append(f"Schedule parameters: {', '.join(parameter_names)}")
        return "\n".join(lines)

    try:
        compiled = compile_intervention_schedule(schedule)
    except InterventionScheduleError as exc:
        lines.extend(
            [
                "",
                "Executable schedule payload is unavailable:",
                str(exc),
            ]
        )
        return "\n".join(lines)

    lines = [
        "Declarative schedule payload:",
        json.dumps(compiled.normalized_declarative_payload, indent=2, sort_keys=True),
        "",
        "Executable schedule payload:",
        json.dumps(compiled.executable_payload, indent=2, sort_keys=True),
        "",
        "executable intervals:",
        json.dumps(compiled.executable_payload.get("intervals", []), indent=2, sort_keys=True),
        "",
        "Lineage:",
        json.dumps(list(compiled.lineage), indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def _state_network_metadata(mechanism: object) -> Mapping[str, object]:
    metadata = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get(MechanismMetadataKeys.STATE_NETWORK)
    if isinstance(raw, Mapping):
        return raw
    return {}


def _format_state_network_summary(mechanism: object) -> list[str]:
    state_network = _state_network_metadata(mechanism)
    if not state_network:
        return []
    raw_states = state_network.get("states")
    raw_edges = state_network.get("edges")
    state_count = len(raw_states) if isinstance(raw_states, Mapping) else 0
    edge_count = len(raw_edges) if isinstance(raw_edges, list) else 0
    generated_reaction_count = sum(
        1
        for rxn in getattr(mechanism, "reactions", []) or []
        if _is_state_network_generated_metadata(getattr(rxn, "metadata", {}) or {})
    )
    generated_equilibrium_count = sum(
        1
        for eq in getattr(mechanism, "equilibria", []) or []
        if _is_state_network_generated_metadata(getattr(eq, "metadata", {}) or {})
    )
    return [
        "State-network generated content:",
        f"  states={state_count}, edges={edge_count}",
        f"  generated reactions={generated_reaction_count}, generated equilibria={generated_equilibrium_count}",
        "  Generated steps are simulation inputs but do not consume indexed DSL step numbers.",
    ]


def _is_state_network_generated_metadata(metadata: object) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return str(metadata.get("source") or "") in {"state_network", "state_network_direct"}


class MechanismInspectorDialog(QtWidgets.QDialog):
    """Read-only dialog for current mechanism steps, equations, and interventions."""

    def __init__(self, dsl_text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mechanismInspectorDialog")
        self.setWindowTitle("Mechanism Inspector")
        self.resize(760, 560)

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget(self)
        tabs.setObjectName("mechanismInspectorTabs")
        layout.addWidget(tabs)

        try:
            sections = build_mechanism_inspector_sections(dsl_text)
        except Exception as exc:
            sections = MechanismInspectorSections(
                steps_text=f"Unable to inspect mechanism:\n{exc}",
                equations_text=f"Unable to inspect mechanism:\n{exc}",
                interventions_text=f"Unable to inspect mechanism:\n{exc}",
            )

        for title, text, object_name in (
            ("Steps", sections.steps_text, "mechanismInspectorStepsText"),
            ("Equations", sections.equations_text, "mechanismInspectorEquationsText"),
            ("Interventions", sections.interventions_text, "mechanismInspectorInterventionsText"),
        ):
            editor = QtWidgets.QPlainTextEdit(self)
            editor.setObjectName(object_name)
            editor.setReadOnly(True)
            editor.setPlainText(text)
            tabs.addTab(editor, title)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
