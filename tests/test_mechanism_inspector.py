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
    assert "Generated mechanism content:" in sections.steps_text
    assert "source-owned conversion outputs" in sections.steps_text
    assert "do not consume authored DSL step indices" in sections.steps_text
    assert "Generated steps use source-order step numbers" not in sections.steps_text
    assert "State-network generated content:" not in sections.steps_text
    assert "Symbolic RHS equations are unavailable for mechanisms with generated equilibria." in (
        sections.equations_text
    )
    assert "No intervention schedule." in sections.interventions_text


def test_mechanism_inspector_generated_fast_uses_same_generated_classifier_for_equations() -> None:
    from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
    from kindred.gui.widgets.mechanism_inspector import build_mechanism_inspector_sections

    source = "\n".join(
        [
            "energy=kJ/mol",
            GENERATED_BLOCK_START,
            "equilibrium: A <-> B; kf=10.0; dG_eq=1.0; cm_id=feq__A__B; cm_std_ratio=1.0",
            GENERATED_BLOCK_END,
            "init: A=1.0, B=0.0",
        ]
    )

    sections = build_mechanism_inspector_sections(source)

    assert "Generated mechanism content:" in sections.steps_text
    assert "generated equilibria=1" in sections.steps_text
    assert "source-owned conversion outputs" in sections.steps_text
    assert "Symbolic RHS equations are unavailable for mechanisms with generated equilibria." in (
        sections.equations_text
    )
