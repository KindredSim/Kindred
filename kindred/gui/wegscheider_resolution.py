from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Mapping

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.wegscheider_symbolic import (
    WegscheiderCyclicityReport,
    WegscheiderResolutionUpdate,
    analyze_wegscheider_cyclicity,
    apply_wegscheider_resolution_to_reactions_text,
    build_wegscheider_resolution_updates,
)

__all__ = [
    "GuiWegscheiderResolution",
    "WegscheiderResolutionUnavailable",
    "resolve_wegscheider_cyclicity_for_gui",
]


@dataclass(frozen=True)
class GuiWegscheiderResolution:
    report: WegscheiderCyclicityReport
    updates: tuple[WegscheiderResolutionUpdate, ...]
    rewritten_reactions_text: str


class WegscheiderResolutionUnavailable(DSLError):
    def __init__(self, message: str) -> None:
        super().__init__(
            str(message),
            suggestion=(
                "Add an explicit symbolic parameter dependency for the unresolved "
                "Wegscheider cycle before running."
            ),
        )
        self.stage = "wegscheider_gui_resolution"


def _default_dependent_parameters(report: WegscheiderCyclicityReport) -> dict[str, str]:
    selected: dict[str, str] = {}
    used_parameters: set[str] = set()
    for cycle in report.unresolved_cycles:
        choice = None
        for candidate in reversed(cycle.parameter_names):
            key = str(candidate).lower()
            if key not in used_parameters:
                choice = str(candidate)
                used_parameters.add(key)
                break
        if choice is None:
            raise WegscheiderResolutionUnavailable(
                f"{cycle.cycle_id} shares all available parameters with another unresolved cycle."
            )
        selected[str(cycle.cycle_id)] = choice
    return selected


def _format_resolution_message(
    report: WegscheiderCyclicityReport,
    choices: Mapping[str, tuple[WegscheiderResolutionUpdate, ...]],
) -> str:
    cycle_lines = [
        f"{cycle.cycle_id}: {', '.join(cycle.parameter_names)}"
        for cycle in report.unresolved_cycles
    ]
    update_lines: list[str] = []
    for cycle_id, options in choices.items():
        update_lines.append(f"{cycle_id}:")
        update_lines.extend(f"  {update.line}" for update in options)
    return (
        "This mechanism has unresolved Wegscheider cyclicity. "
        "Choose the dependent symbolic Keq parameter for each cycle; Kindred will "
        "add durable parameter algebra to the Reactions text.\n\n"
        "Unresolved cycles:\n"
        + "\n".join(cycle_lines)
        + "\n\nAvailable Reactions updates:\n"
        + "\n".join(update_lines)
    )


def _resolution_choices(
    report: WegscheiderCyclicityReport,
) -> dict[str, tuple[WegscheiderResolutionUpdate, ...]]:
    choices: dict[str, tuple[WegscheiderResolutionUpdate, ...]] = {}
    for cycle in report.unresolved_cycles:
        updates: list[WegscheiderResolutionUpdate] = []
        for parameter_name in cycle.parameter_names:
            update = build_wegscheider_resolution_updates(
                report,
                {str(cycle.cycle_id): str(parameter_name)},
            )[0]
            updates.append(update)
        choices[str(cycle.cycle_id)] = tuple(updates)
    return choices


def _choice_payload(
    choices: Mapping[str, tuple[WegscheiderResolutionUpdate, ...]],
) -> dict[str, list[dict[str, str]]]:
    return {
        str(cycle_id): [
            {
                "cycle_id": str(update.cycle_id),
                "parameter_name": str(update.parameter_name),
                "expr_src": str(update.expr_src),
                "line": str(update.line),
            }
            for update in options
        ]
        for cycle_id, options in choices.items()
    }


def resolve_wegscheider_cyclicity_for_gui(
    reactions_text: str,
    *,
    enabled: bool,
    choose_resolution: Callable[[str, str, Mapping[str, list[dict[str, str]]]], Mapping[str, str] | None] | None = None,
    ask_user: object | None = None,
    dependent_parameters: Mapping[str, str] | None = None,
) -> GuiWegscheiderResolution | None:
    if not bool(enabled):
        return None
    if not str(reactions_text or "").strip():
        return None
    mechanism = parse_dsl_to_mechanism(str(reactions_text or ""), initials={})
    if isinstance(getattr(mechanism, "metadata", None), dict):
        mechanism.metadata["wegscheider_cyclicity_enabled"] = True
    report = analyze_wegscheider_cyclicity(mechanism)
    if report.is_resolved:
        return None

    available_choices = _resolution_choices(report)
    message = _format_resolution_message(report, available_choices)
    if dependent_parameters is not None:
        choices = dict(dependent_parameters)
    elif choose_resolution is not None:
        selected = choose_resolution(
            "Resolve Wegscheider Cyclicity",
            message,
            _choice_payload(available_choices),
        )
        if selected is None:
            return None
        choices = dict(selected)
    elif ask_user is not None:
        choices = _default_dependent_parameters(report)
        if not bool(ask_user("Resolve Wegscheider Cyclicity", message, accept_label="Apply Resolution")):
            return None
    else:
        raise WegscheiderResolutionUnavailable("No Wegscheider resolution chooser is available.")
    updates = build_wegscheider_resolution_updates(report, choices)
    rewritten = apply_wegscheider_resolution_to_reactions_text(str(reactions_text or ""), updates)

    reparsed = parse_dsl_to_mechanism(rewritten, initials={})
    if isinstance(getattr(reparsed, "metadata", None), dict):
        reparsed.metadata["wegscheider_cyclicity_enabled"] = True
    verification = analyze_wegscheider_cyclicity(reparsed)
    if not verification.is_resolved:
        raise WegscheiderResolutionUnavailable(
            "Generated Wegscheider source update did not resolve every cycle."
        )
    return GuiWegscheiderResolution(
        report=report,
        updates=updates,
        rewritten_reactions_text=rewritten,
    )
