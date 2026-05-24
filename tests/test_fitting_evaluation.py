from __future__ import annotations

from typing import Any

import pytest

from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

pytestmark = pytest.mark.unit


def _protocol_process_payload() -> dict[str, Any]:
    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=0.0",
                "initial: B=0.0",
                "intervention: op=protocol; kind=repeat; name=feed; start=0.0; every=1.0; duration=0.5; count=2; during=source:A:rate=1.0",
            ]
        ),
        param_names=[],
        t_end=2.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    return SerialFittingEvaluator(context).to_process_payload()


@pytest.mark.parametrize(
    "field_name",
    [
        "intervention_schedule_declarative_fingerprint",
        "intervention_schedule_executable_fingerprint",
    ],
)
def test_process_payload_rejects_mismatched_intervention_schedule_fingerprint(field_name: str) -> None:
    payload = _protocol_process_payload()
    payload["prepared_metadata"][field_name] = "wrong-fingerprint"

    with pytest.raises(ValueError, match="intervention_schedule"):
        SerialFittingEvaluator.from_process_payload(payload)
