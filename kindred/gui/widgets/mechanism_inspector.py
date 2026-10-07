"""Read-only mechanism inspector backed by core parsing authorities."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from fractions import Fraction
from collections.abc import Callable, Mapping

from PySide6 import QtWidgets

from kindred.core.algebra.printer import expression_to_source
from kindred.core.equilibrium_rate_authority import EquilibriumRateAuthorityKind
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    intervention_schedule_parameter_names,
)
from kindred.core.kinetics import ArrheniusRateModel, EyringRateModel
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
from kindred.core.quantity_units import Unit, format_unit_for_display
from kindred.core.simulator.dsl_parse import DslAnalysis, PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
)
from kindred.core.simulator.parameter_coordinates import build_parameter_target_map
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.step_indexing import (
    get_step_index_map,
    iter_canonical_parameters,
)
from kindred.core.stoichiometry import directional_order_fraction

__all__ = [
    "MechanismInspectorDialog",
    "MechanismInspectorSections",
    "build_mechanism_inspector_sections",
]


@dataclass(frozen=True, slots=True)
class MechanismInspectorSections:
    mechanism_text: str
    steps_text: str
    equations_text: str
    interventions_text: str


def build_mechanism_inspector_sections(
    prepared_dsl: PreparedDslRequestSet,
) -> MechanismInspectorSections:
    """Build read-only inspector sections from the same core authorities used at runtime."""
    accepted_mechanism = prepared_dsl.accepted_mechanism_for_structure()
    parameter_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    if parameter_spec.param_statements:
        mechanism = prepared_dsl.mutable_mechanism_shell()
        parameter_target_map = build_parameter_target_map(mechanism)
        apply_parameter_algebra_spec_to_mechanism(
            parameter_spec,
            mechanism=mechanism,
            parameter_target_map=parameter_target_map,
        )
    else:
        mechanism = accepted_mechanism
    parameterized_schedule = (
        prepared_dsl.parameterized_intervention_schedule_for_binding()
    )
    intervention_execution = prepared_dsl.intervention_execution_contract()
    return MechanismInspectorSections(
        mechanism_text=_safe_format_section(
            "mechanism",
            lambda: _format_mechanism_reference(
                accepted_mechanism,
                parameter_spec=parameter_spec,
                analysis=prepared_dsl.analysis,
            ),
        ),
        steps_text=_safe_format_section(
            "steps",
            lambda: _format_steps_from_authority(mechanism, parameter_spec=parameter_spec),
        ),
        equations_text=_safe_format_section(
            "equations",
            lambda: _format_equations(
                prepared_dsl.symbolic_jacobian_structure(),
                has_interventions=(
                    parameterized_schedule is not None
                    or bool(
                        intervention_execution
                        and intervention_execution.declarative_payload.get("operations")
                    )
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


def _format_mechanism_reference(
    mechanism: object,
    *,
    parameter_spec: ParameterAlgebraSpec,
    analysis: DslAnalysis,
) -> str:
    """Project the accepted symbolic mechanism without evaluating coordinates."""

    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)
    parameter_provenance = {
        str(statement.name): expression_to_source(statement.expr)
        for statement in parameter_spec.parameter_block.lines
    }
    step_lines: list[str] = []
    uses_standard_concentration = False
    for entry in get_step_index_map(mechanism):
        line, step_uses_standard_concentration = _format_mechanism_step(
            mechanism,
            entry,
            parameter_provenance=parameter_provenance,
            units=dict(parameter_spec.input_units),
        )
        step_lines.append(line)
        uses_standard_concentration |= step_uses_standard_concentration

    temperature_note = (
        " (nominal, runtime schedule defined)"
        if metadata.temperature_schedule is not None
        else ""
    )
    lines = [
        f"Temperature: {_format_number(analysis.temperature_K)} K{temperature_note}"
    ]
    if uses_standard_concentration:
        lines.append(f"C0: {_format_number(metadata.standard_conc_M)} M")
    lines.extend(("", *(step_lines or ("No reaction steps.",))))

    algebra_lines = [
        html.escape(
            f"param {statement.name} = {expression_to_source(statement.expr)}"
            + (f"; unit={statement.declared_unit.symbol}" if statement.declared_unit is not None else "")
        )
        for statement in parameter_spec.parameter_block.lines
    ]
    if parameter_spec.observable_block is not None:
        algebra_lines.extend(
            html.escape(
                f"let {statement.name} = {expression_to_source(statement.expr)}"
                + (f"; unit={statement.declared_unit.symbol}" if statement.declared_unit is not None else "")
            )
            for statement in parameter_spec.observable_block.lines
        )
    if algebra_lines:
        lines.extend(("", "Algebra", *algebra_lines))
    return "<pre>" + "\n".join(lines) + "</pre>"


def _format_mechanism_step(
    mechanism: object,
    entry: Mapping[str, object],
    *,
    parameter_provenance: Mapping[str, str],
    units: Mapping[str, Unit],
) -> tuple[str, bool]:
    """Project one step; units come from the admitted parameter spec."""
    step_index = int(entry["step_index"])
    kind = str(entry["kind"])
    context = html.escape(str(entry["context"]).replace(" <-> ", " ⇌ ").replace(" -> ", " → "))
    source_coordinates = {
        str(role): str(name)
        for role, name in dict(entry.get("source_coordinates") or {}).items()
    }

    if kind == "reaction":
        reaction = mechanism.reactions[
            int(entry["reaction_index"])
        ]
        forward_model = reaction.forward_model
        equilibrium = None
    else:
        equilibrium = mechanism.equilibria[
            int(entry["equilibrium_index"])
        ]
        forward_model = equilibrium.forward_model

    items = [
        _format_coordinate(
            name,
            unit=units.get(name),
            provenance=parameter_provenance.get(name),
        )
        for role, name in source_coordinates.items()
        if role not in {"k", "kf", "Keq", "kr"}
    ]

    forward_name = f"k{step_index}" if kind == "reaction" else f"kf{step_index}"
    forward_provenance = parameter_provenance.get(forward_name)
    uses_standard_concentration = False
    if forward_provenance is None and isinstance(forward_model, ArrheniusRateModel):
        forward_provenance = _family_provenance(
            "Arrhenius",
            source_coordinates,
            roles=("A", "Ea"),
            uses_standard_concentration=False,
        )
    elif forward_provenance is None and isinstance(forward_model, EyringRateModel):
        uses_standard_concentration = Fraction(str(forward_model.molecularity)) != 1
        forward_provenance = _family_provenance(
            "Eyring",
            source_coordinates,
            roles=("dG_act", "kappa", "degeneracy"),
            uses_standard_concentration=uses_standard_concentration,
        )
    items.append(
        _format_coordinate(
            forward_name,
            unit=units.get(forward_name),
            provenance=forward_provenance,
        )
    )

    if equilibrium is not None:
        keq_name = f"Keq{step_index}"
        reverse_name = f"kr{step_index}"
        authority_kind = equilibrium.authority.kind
        keq_provenance = parameter_provenance.get(keq_name)
        if keq_provenance is None and authority_kind == EquilibriumRateAuthorityKind.KR:
            keq_provenance = f"{forward_name} / {reverse_name}"
        elif keq_provenance is None and "dG_eq" in source_coordinates:
            delta_n = directional_order_fraction(
                equilibrium.stoich_back,
            ) - directional_order_fraction(
                equilibrium.stoich_forward,
            )
            thermo_inputs = [source_coordinates["dG_eq"], "T"]
            if delta_n:
                thermo_inputs.append("C0")
                uses_standard_concentration = True
            keq_provenance = f"thermodynamic({', '.join(thermo_inputs)})"
        items.append(
            _format_coordinate(
                keq_name,
                unit=units.get(keq_name),
                provenance=keq_provenance,
            )
        )

        reverse_provenance = parameter_provenance.get(reverse_name)
        if (
            reverse_provenance is None
            and authority_kind == EquilibriumRateAuthorityKind.KEQ
        ):
            reverse_provenance = f"{forward_name} / {keq_name}"
        items.append(
            _format_coordinate(
                reverse_name,
                unit=units.get(reverse_name),
                provenance=reverse_provenance,
            )
        )

    return (
        f"{step_index}  {context}    {' · '.join(items)}",
        uses_standard_concentration,
    )


def _format_coordinate(
    name: str,
    *,
    unit: Unit | None,
    provenance: str | None,
) -> str:
    unit_text = (
        "" if unit is None or unit.symbol == "1"
        else f" [{format_unit_for_display(unit)}]"
    )
    provenance_text = "" if provenance is None else f" (derived from {html.escape(provenance)})"
    return f"{html.escape(name)}{unit_text}{provenance_text}"


def _family_provenance(
    family: str,
    source_coordinates: Mapping[str, str],
    *,
    roles: tuple[str, ...],
    uses_standard_concentration: bool,
) -> str:
    inputs = [source_coordinates[role] for role in roles if role in source_coordinates]
    inputs.append("T")
    if uses_standard_concentration:
        inputs.append("C0")
    return f"{family}({', '.join(inputs)})"


def _format_number(value: object) -> str:
    return format(float(value), ".15g")


def _format_steps_from_authority(mechanism: object, *, parameter_spec: ParameterAlgebraSpec) -> str:
    coefficient_authority = build_effective_rate_coefficient_evaluator(mechanism)
    return _format_steps(
        mechanism,
        authority_snapshot=coefficient_authority.authority_snapshot,
        parameter_spec=parameter_spec,
    )


def _format_steps(
    mechanism: object,
    *,
    authority_snapshot: tuple[Mapping[str, object], ...],
    parameter_spec: ParameterAlgebraSpec,
) -> str:
    entries = get_step_index_map(mechanism)
    parameter_names_by_step_index: dict[int, list[str]] = {}
    for param_name, entry, _role in iter_canonical_parameters(mechanism):
        parameter_names_by_step_index.setdefault(int(entry["step_index"]), []).append(
            param_name
        )

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
            if kind == "equilibrium":
                lines.append(
                    _equilibrium_authority_line(
                        mechanism.equilibria[int(entry["equilibrium_index"])],
                        step_index,
                    )
                )
            rate_authority = _rate_authority_for_step(
                mechanism,
                entry,
                authority_snapshot,
            )
            if rate_authority is not None:
                lines.append(
                    "  Immutable model/default inputs (before retained assignments): "
                    + json.dumps(
                        _plain_json_value(rate_authority),
                        sort_keys=True,
                    )
                )
            coordinate_names = set((entry.get("source_coordinates") or {}).values()) | set(parameter_names)
            for statement in parameter_spec.parameter_block.lines:
                if statement.name in coordinate_names:
                    kind = parameter_spec.parameter_assignment_kind(statement.name)
                    declaration = f"{statement.name} = {expression_to_source(statement.expr)}"
                    if statement.declared_unit is not None:
                        declaration += f"; unit={statement.declared_unit.symbol}"
                    lines.append(f"  Retained {kind}: {declaration}")
    else:
        lines.append("No indexed DSL reaction or equilibrium steps.")

    return "\n".join(lines)


def _equilibrium_authority_line(equilibrium: object, step_index: int) -> str:
    """Describe the one authored reverse authority of an equilibrium step."""
    authority = equilibrium.authority
    kf, kr, keq = f"kf{step_index}", f"kr{step_index}", f"Keq{step_index}"
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        return f"  Equilibrium authority: {kr} authored, {keq} = {kf} / {kr} derived"
    if authority.has_explicit_keq_param:
        return f"  Equilibrium authority: {keq} authored, {kr} = {kf} / {keq} derived"
    return (
        f"  Equilibrium authority: dG_eq{step_index} authored, "
        f"{keq} from dG_eq{step_index}, T and C0, {kr} = {kf} / {keq} derived"
    )


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
            index = len(tuple(mechanism.reactions)) + int(
                entry["equilibrium_index"]
            )
        else:
            return None
        snapshot = authority_snapshot[index]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return snapshot if isinstance(snapshot, Mapping) else None


def _format_equations(structure: object, *, has_interventions: bool) -> str:
    lines = ["Canonical symbolic RHS equations:"]
    if has_interventions:
        lines.append(
            "Note: intervention schedules are compiled execution inputs. Equations here are the base unscheduled mechanism RHS."
        )
        lines.append("")
    for species_name, rhs_expression in zip(
        structure.species_names,
        structure.rhs_expressions,
        strict=True,
    ):
        lines.append(f"d[{species_name}]/dt = {rhs_expression}")
    lines.append("")
    lines.append("Jacobian rows:")
    for species_name, row in zip(
        structure.species_names, structure.jacobian_expressions, strict=True
    ):
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
        parameter_names = sorted(
            intervention_schedule_parameter_names(parameterized_schedule)
        )
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
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


class MechanismInspectorDialog(QtWidgets.QDialog):
    """Read-only dialog for the accepted mechanism and technical authorities."""

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
                mechanism_text=f"Unable to inspect mechanism:\n{exc}",
                steps_text=f"Unable to inspect mechanism:\n{exc}",
                equations_text=f"Unable to inspect mechanism:\n{exc}",
                interventions_text=f"Unable to inspect mechanism:\n{exc}",
            )

        for title, text, object_name, no_wrap, tooltip in (
            (
                "Mechanism",
                sections.mechanism_text,
                "mechanismInspectorMechanismText",
                True,
                "Accepted symbolic mechanism. Fit values remain in Parameters & ICs.",
            ),
            ("Steps", sections.steps_text, "mechanismInspectorStepsText", False, ""),
            (
                "Equations",
                sections.equations_text,
                "mechanismInspectorEquationsText",
                False,
                "",
            ),
            (
                "Interventions",
                sections.interventions_text,
                "mechanismInspectorInterventionsText",
                False,
                "",
            ),
        ):
            editor = (
                QtWidgets.QTextBrowser(self)
                if title == "Mechanism"
                else QtWidgets.QPlainTextEdit(self)
            )
            editor.setObjectName(object_name)
            editor.setReadOnly(True)
            if no_wrap and isinstance(editor, QtWidgets.QPlainTextEdit):
                editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            if isinstance(editor, QtWidgets.QTextBrowser):
                editor.setOpenLinks(False)
                editor.setOpenExternalLinks(False)
                editor.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
                editor.setHtml(text)
            else:
                editor.setPlainText(text)
            tab_index = tabs.addTab(editor, title)
            if tooltip:
                tabs.setTabToolTip(tab_index, tooltip)
        tabs.setCurrentIndex(0)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, self
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
