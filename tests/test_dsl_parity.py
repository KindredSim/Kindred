import numpy as np
import pytest

from kindred.core.simulator.dsl import parse_and_preview, parse_dsl_to_mechanism


DSL_CASES = [
    (
        "irreversible",
        "\n".join(
            [
                "reaction: A -> B; k=1.5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
    ),
    (
        "reversible_explicit",
        "\n".join(
            [
                "equilibrium: A <-> B; kf=2.0; kr=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
    ),
    (
        "reversible_eyring",
        "\n".join(
            [
                "energy=kJ/mol",
                "reaction: A <-> B; dG_act=75.0; dG_eq=-10.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
    ),
    (
        "temperature_schedule",
        "\n".join(
            [
                "# Algebra",
                "let x = 1",
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
                "temp_step: t=[0,10], T=[300]",
            ]
        ),
    ),
    (
        "temperature_response",
        "\n".join(
            [
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
                "temp_response: t=[0,5,10], T=[300,600], tau=2.0",
            ]
        ),
    ),
    (
        "algebra_block",
        "\n".join(
            [
                "# Algebra",
                "let c1 = 2.0",
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
    ),
]


@pytest.mark.parametrize("name,dsl", DSL_CASES)
def test_preview_and_mechanism_parity(name, dsl):
    previews = parse_and_preview(dsl)
    mech = parse_dsl_to_mechanism(dsl, initials={})

    # Preview count should match total step count
    total_steps = len(mech.reactions) + len(mech.equilibria)
    assert len(previews) == total_steps

    if name == "irreversible":
        assert len(mech.reactions) == 1 and len(mech.equilibria) == 0
        assert "->" in previews[0] and "<->" not in previews[0]
    elif name == "reversible_explicit":
        assert len(mech.equilibria) == 1
        eq = mech.equilibria[0]
        assert np.isclose(float(eq.kf), 2.0)
        assert np.isclose(float(eq.kr), 1.0)
    elif name == "reversible_eyring":
        assert len(mech.equilibria) == 1
        eq = mech.equilibria[0]
        assert float(eq.kf) > 0 and float(eq.kr) > 0
        assert "<->" in previews[0]
    elif name == "temperature_schedule":
        from kindred.core.temperature_dsl import parse_temperature_schedule

        schedule = parse_temperature_schedule(dsl)
        assert (schedule is None) == ("temperature_schedule" not in mech.metadata)
        if schedule is not None:
            sched_mech = mech.metadata.get("temperature_schedule")
            assert sched_mech is not None
            for t_sample in (0.0, 5.0, 9.9):
                assert pytest.approx(schedule(t_sample)) == pytest.approx(sched_mech(t_sample))
    elif name == "temperature_response":
        from kindred.core.temperature_dsl import parse_temperature_schedule

        schedule = parse_temperature_schedule(dsl)
        assert schedule is not None
        assert schedule.schedule_type == "response"
        sched_mech = mech.metadata.get("temperature_schedule")
        assert sched_mech is not None
        assert getattr(sched_mech, "schedule_type", None) == "response"
        for t_sample in (0.0, 5.0, 7.0, 12.0):
            assert pytest.approx(schedule(t_sample)) == pytest.approx(sched_mech(t_sample))
    elif name == "algebra_block":
        # Algebra lines should be preserved in metadata for downstream
        assert "algebra_text" in mech.metadata or mech.metadata.get("algebra_text") is not None


def test_invalid_dsl_errors_align():
    bad = "reaction: A B; k=1.0"
    with pytest.raises(Exception) as exc_preview:
        parse_and_preview(bad)
    with pytest.raises(Exception) as exc_mech:
        parse_dsl_to_mechanism(bad, initials={})

    assert type(exc_preview.value) is type(exc_mech.value)
