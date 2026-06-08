from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_STATE_NETWORK_DSL = "\n".join(
    [
        "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
        "state: B, kind=GS, energy=1, energy_unit=kJ/mol, degeneracy=1",
        "edge: A,B",
    ]
)


def test_mechanism_inspector_sections_use_complete_source_with_state_network() -> None:
    from kindred.core.mechanism_source import MechanismAuthoringSource
    from kindred.gui.widgets.mechanism_inspector import build_mechanism_inspector_sections

    source = MechanismAuthoringSource.from_parts(
        reactions_text="\n".join(
            [
                "reaction: X -> Y; k=0.1",
                "initial: X=1.0",
                "initial: Y=0.0",
            ]
        ),
        state_network_dsl=_STATE_NETWORK_DSL,
    )

    sections = build_mechanism_inspector_sections(source.full_dsl)

    assert "Step 1 reaction: X -> Y" in sections.steps_text
    assert "State-network generated content:" in sections.steps_text
    assert "Symbolic RHS equations are unavailable for mechanisms with state-network definitions." in (
        sections.equations_text
    )
    assert "No intervention schedule." in sections.interventions_text
