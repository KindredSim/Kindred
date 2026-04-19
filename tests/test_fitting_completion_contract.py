from __future__ import annotations

import pytest

from kindred.core.fitting_completion import FitDetailSection, FitDiagnostic, GlobalFitCompletion
from kindred.core.simulation_failure import build_simulation_failure


pytestmark = [pytest.mark.unit]


def _diagnostic(*, remediation: str = "generic_retry") -> FitDiagnostic:
    return FitDiagnostic(
        phase="fatal",
        dataset_id=None,
        failure=build_simulation_failure(kind="simulation_error", message="fatal failure"),
        remediation=remediation,
    )


def test_completion_rejects_ok_when_optimizer_not_converged() -> None:
    with pytest.raises(ValueError):
        GlobalFitCompletion(
            status="ok",
            optimizer_converged=False,
            nonfinite_metrics=False,
        )


def test_completion_rejects_ok_when_metrics_nonfinite() -> None:
    with pytest.raises(ValueError):
        GlobalFitCompletion(
            status="ok",
            optimizer_converged=True,
            nonfinite_metrics=True,
        )


def test_completion_rejects_warn_when_metrics_nonfinite() -> None:
    with pytest.raises(ValueError):
        GlobalFitCompletion(
            status="warn",
            optimizer_converged=False,
            nonfinite_metrics=True,
        )


def test_completion_rejects_semantically_empty_warn() -> None:
    with pytest.raises(ValueError):
        GlobalFitCompletion(
            status="warn",
            optimizer_converged=True,
            nonfinite_metrics=False,
        )


def test_completion_rejects_semantically_empty_fail() -> None:
    with pytest.raises(ValueError):
        GlobalFitCompletion(
            status="fail",
            optimizer_converged=False,
            nonfinite_metrics=False,
        )


def test_completion_accepts_fail_with_converged_optimizer_when_failure_source_exists() -> None:
    completion = GlobalFitCompletion(
        status="fail",
        optimizer_converged=True,
        nonfinite_metrics=False,
        optimizer_diagnostic=_diagnostic(),
    )

    assert completion.status == "fail"
    assert completion.optimizer_converged is True
    assert completion.optimizer_diagnostic is not None
    assert completion.optimizer_diagnostic.remediation == "generic_retry"


def test_completion_coerces_optimizer_diagnostic_to_typed_contract() -> None:
    completion = GlobalFitCompletion(
        status="fail",
        optimizer_converged=False,
        nonfinite_metrics=False,
        optimizer_diagnostic={
            "phase": "fatal",
            "dataset_id": "ds1",
            "failure": build_simulation_failure(kind="simulation_error", message="fatal failure"),
            "remediation": "generic_retry",
        },
    )

    assert isinstance(completion.optimizer_diagnostic, FitDiagnostic)
    assert completion.optimizer_diagnostic.dataset_id == "ds1"
    assert completion.optimizer_diagnostic.remediation == "generic_retry"


def test_completion_coerces_detail_sections_from_mapping_shape() -> None:
    completion = GlobalFitCompletion(
        status="fail",
        optimizer_converged=True,
        nonfinite_metrics=False,
        optimizer_diagnostic=_diagnostic(),
        detail_sections=[
            {
                "dataset_id": "ds1",
                "failure": build_simulation_failure(kind="simulation_error", message="detail failure"),
            }
        ],
    )

    assert completion.detail_sections == [
        FitDetailSection(
            dataset_id="ds1",
            failure=build_simulation_failure(kind="simulation_error", message="detail failure"),
        )
    ]
