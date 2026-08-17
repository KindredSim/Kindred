"""Read-only mechanism inspector backed by core parsing authorities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable, Mapping

from PySide6 import QtWidgets

from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    intervention_schedule_parameter_names,
)
from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulator.step_indexing import get_step_index_map, iter_canonical_parameters
from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

__all__ = ["MechanismInspectorDialog", "MechanismInspectorSections", "build_mechanism_inspector_sections"]


@dataclass(frozen=True, slots=True)
class MechanismInspectorSections:
    steps_text: str
    equations_text: str
    interventions_text: str


def build_mechanism_inspector_sections(
    prepared_dsl: PreparedDslRequestSet,
) -> MechanismInspectorSections:
    """Build read-only inspector sections from the same core authorities used at runtime."""
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    mechanism = prepared_dsl.mechanism_for_execution()
    apply_parameter_algebra_spec_to_mechanism(
        prepared_dsl.parameter_algebra_spec_for_execution(),
        mechanism=mechanism,
        require_mutable=False,
    )
    parameterized_schedule = prepared_dsl.parameterized_intervention_schedule_for_binding()
    intervention_execution = prepared_dsl.intervention_execution_contract()
    return MechanismInspectorSections(
        steps_text=_safe_format_section(
            "steps",
            lambda: _format_steps_from_authority(mechanism),
        ),
        equations_text=_safe_format_section(
            "equations",
            lambda: _format_equations(
                mechanism,
                has_interventions=(
                    parameterized_schedule is not None
                    or bool(intervention_execution and intervention_execution.declarative_payload.get("operations"))
                ),
            ),
        ),
        interventions_text=_safe_format_section(
            "interventions",
            lambda: _format_interventions(
                parameterized_schedule=parameterized_schedule,
                intervention_execution=intervention_execution,
            ),
        ),
    )


def _safe_format_section(label: str, formatter: Callable[[], str]) -> str:
    try:
        return formatter()
    except Exception as exc:
        return f"Unable to inspect {label}:\n{exc}"


def _format_steps_from_authority(mechanism: object) -> str:
    coefficient_authority = build_effective_rate_coefficient_evaluator(mechanism)
    return _format_steps(
        mechanism,
        authority_snapshot=coefficient_authority.authority_snapshot,
    )


def _format_steps(
    mechanism: object,
    *,
    authority_snapshot: tuple[Mapping[str, object], ...],
) -> str:
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
            rate_authority = _rate_authority_for_step(
                mechanism,
                entry,
                authority_snapshot,
            )
            if rate_authority is not None:
                lines.append(
                    "  Effective-rate authority: "
                    + json.dumps(
                        _plain_json_value(rate_authority),
                        sort_keys=True,
                    )
                )
    else:
        lines.append("No indexed DSL reaction or equilibrium steps.")

    return "\n".join(lines)


def _rate_authority_for_step(
    mechanism: object,
    entry: Mapping[str, object],
    authority_snapshot: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    """Select the evaluator-owned snapshot for one indexed mechanism step."""
    kind = str(entry.get("kind") or "")
    try:
        if kind == "reaction":
            index = int(entry["reaction_index"])
        elif kind == "equilibrium":
            index = len(tuple(getattr(mechanism, "reactions", ()) or ())) + int(
                entry["equilibrium_index"]
            )
        else:
            return None
        snapshot = authority_snapshot[index]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return snapshot if isinstance(snapshot, Mapping) else None


def _format_equations(mechanism: object, *, has_interventions: bool) -> str:
    structure = build_symbolic_jacobian_structure(mechanism)
    lines = ["Canonical symbolic RHS equations:"]
    if has_interventions:
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
    lines.append(
        "Canonical parameter symbols: "
        + (", ".join(structure.parameter_symbols) or "none")
    )
    return "\n".join(lines)


def _format_interventions(
    *,
    parameterized_schedule: InterventionSchedule | None,
    intervention_execution: InterventionExecutionContract | None,
) -> str:
    if parameterized_schedule is not None:
        parameter_names = sorted(intervention_schedule_parameter_names(parameterized_schedule))
        lines = [
            "Declarative schedule payload:",
            json.dumps(parameterized_schedule.to_payload(), indent=2, sort_keys=True),
            "",
            "Typed executable schedule is unavailable until schedule parameters are resolved.",
        ]
        if parameter_names:
            lines.append(f"Schedule parameters: {', '.join(parameter_names)}")
        return "\n".join(lines)
    if intervention_execution is None:
        return "No intervention schedule."
    return _format_intervention_execution(intervention_execution)


def _format_intervention_execution(
    contract: InterventionExecutionContract,
) -> str:
    """Project the compiler-owned typed contract without reconstructing it."""

    typed_sections = (
        ("Instants", contract.instants),
        ("Fluxes", contract.fluxes),
        ("Holds", contract.holds),
        ("Triggers", contract.triggers),
        ("Trigger firings", contract.trigger_firings),
    )
    lines = [
        "Declarative schedule payload:",
        json.dumps(
            _plain_json_value(contract.declarative_payload),
            indent=2,
            sort_keys=True,
        ),
        "",
    ]
    for label, values in typed_sections:
        lines.extend(
            [
                "",
                f"Typed {label.lower()}:",
                json.dumps(
                    [value.to_payload() for value in values],
                    indent=2,
                    sort_keys=True,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Discontinuities:",
            json.dumps(list(contract.discontinuities), indent=2),
            "",
            "Protocol lineage:",
            json.dumps(
                _plain_json_value(contract.protocol_lineage),
                indent=2,
                sort_keys=True,
            ),
        ]
    )
    return "\n".join(lines)


def _plain_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


class MechanismInspectorDialog(QtWidgets.QDialog):
    """Read-only dialog for current mechanism steps, equations, and interventions."""

    def __init__(
        self,
        prepared_dsl: PreparedDslRequestSet,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("mechanismInspectorDialog")
        self.setWindowTitle("Mechanism Inspector")
        self.resize(760, 560)

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget(self)
        tabs.setObjectName("mechanismInspectorTabs")
        layout.addWidget(tabs)

        try:
            sections = build_mechanism_inspector_sections(prepared_dsl)
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
