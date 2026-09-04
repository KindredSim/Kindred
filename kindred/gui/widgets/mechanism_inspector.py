"""Read-only mechanism inspector backed by core parsing authorities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from fractions import Fraction
from collections.abc import Callable, Mapping

from PySide6 import QtWidgets

from kindred.core.algebra.printer import expression_to_source
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    authority_fields_from_step_entry,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.intervention_schedule import (
    InterventionSchedule,
    intervention_schedule_parameter_names,
)
from kindred.core.kinetics import (
    ArrheniusRateModel,
    EyringRateModel,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
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
from kindred.core.units import equilibrium_constant_unit, rate_constant_unit
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
            require_mutable=False,
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
            lambda: _format_steps_from_authority(mechanism),
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

    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    parameter_provenance = {
        str(statement.name): str(statement.expr_src)
        for statement in parameter_spec.param_statements
    }
    step_lines: list[str] = []
    uses_standard_concentration = False
    for entry in get_step_index_map(mechanism):
        line, step_uses_standard_concentration = _format_mechanism_step(
            mechanism,
            entry,
            metadata=metadata,
            parameter_provenance=parameter_provenance,
        )
        step_lines.append(line)
        uses_standard_concentration |= step_uses_standard_concentration

    temperature_note = (
        " (nominal; runtime schedule defined)"
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
        f"param {statement.name} = {statement.expr_src}"
        for statement in parameter_spec.param_statements
    ]
    if parameter_spec.observable_block is not None:
        algebra_lines.extend(
            f"let {statement.name} = {expression_to_source(statement.expr)}"
            for statement in parameter_spec.observable_block.lines
        )
    if algebra_lines:
        lines.extend(("", "Algebra", *algebra_lines))
    return "\n".join(lines)


def _format_mechanism_step(
    mechanism: object,
    entry: Mapping[str, object],
    *,
    metadata: MechanismMetadataView,
    parameter_provenance: Mapping[str, str],
) -> tuple[str, bool]:
    step_index = int(entry["step_index"])
    kind = str(entry["kind"])
    context = str(entry["context"]).replace(" <-> ", " ⇌ ").replace(" -> ", " → ")
    source_coordinates = {
        str(role): str(name)
        for role, name in dict(entry.get("source_coordinates") or {}).items()
    }

    if kind == "reaction":
        reaction = (getattr(mechanism, "reactions", ()) or ())[
            int(entry["reaction_index"])
        ]
        forward_order = float(
            directional_order_fraction(
                reaction.rate_orders,
            )
        )
        forward_model = forward_rate_model_for_reaction(
            reaction,
            default_standard_conc_M=float(metadata.standard_conc_M),
        )
        equilibrium = None
        reverse_order = None
    else:
        equilibrium = (getattr(mechanism, "equilibria", ()) or ())[
            int(entry["equilibrium_index"])
        ]
        forward_order = float(
            directional_order_fraction(
                equilibrium.stoich_forward,
            )
        )
        reverse_order = float(
            directional_order_fraction(
                equilibrium.stoich_back,
            )
        )
        forward_model = forward_rate_model_for_equilibrium(
            equilibrium,
            temperature_K=None,
            default_standard_conc_M=float(metadata.standard_conc_M),
        )

    items = [
        _format_coordinate(
            name,
            unit=_source_coordinate_unit(
                role,
                forward_order=forward_order,
                energy_unit=metadata.energy_unit,
            ),
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
            unit=rate_constant_unit(forward_order),
            provenance=forward_provenance,
        )
    )

    if equilibrium is not None and reverse_order is not None:
        keq_name = f"Keq{step_index}"
        reverse_name = f"kr{step_index}"
        authority = authority_fields_from_step_entry(entry)
        authority_kind = str(authority.get("kind") or "")
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
                unit=equilibrium_constant_unit(reverse_order - forward_order),
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
                unit=rate_constant_unit(reverse_order),
                provenance=reverse_provenance,
            )
        )

    return (
        f"{step_index}  {context}    {' · '.join(items)}",
        uses_standard_concentration,
    )


def _source_coordinate_unit(
    role: str,
    *,
    forward_order: float,
    energy_unit: str,
) -> str:
    if role == "A":
        return rate_constant_unit(forward_order)
    if role in {"Ea", "dG_act", "dG_eq"}:
        return energy_unit
    return "1"


def _format_coordinate(
    name: str,
    *,
    unit: str,
    provenance: str | None,
) -> str:
    unit_text = "" if unit == "1" else f" [{unit}]"
    provenance_text = "" if provenance is None else f" (derived from {provenance})"
    return f"{name}{unit_text}{provenance_text}"


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
            authority = entry.get("equilibrium_authority")
            if authority:
                lines.append(
                    f"  Equilibrium authority: {json.dumps(authority, sort_keys=True)}"
                )
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


def _format_equations(structure: object, *, has_interventions: bool) -> str:
    lines = ["Canonical symbolic RHS equations:"]
    if has_interventions:
        lines.append(
            "Note: intervention schedules are compiled execution inputs; equations here are the base unscheduled mechanism RHS."
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
            editor = QtWidgets.QPlainTextEdit(self)
            editor.setObjectName(object_name)
            editor.setReadOnly(True)
            if no_wrap:
                editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
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
