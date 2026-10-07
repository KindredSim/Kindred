"""Small Fit-local UI constants; simulation science comes from each mapped set."""

from kindred.core.fitting_method_policy import validate_fit_method
from kindred.core.validation import parse_finite_numeric_text

INITIAL_PREFIX = "init:"
DIFFERENTIAL_EVOLUTION_METHOD = "differential_evolution"
# Tolerance labels shared by the Fit window and the Fitting Defaults dialog.
FUNCTION_TOLERANCE_LABEL = "Objective-improvement tolerance"
STEP_TOLERANCE_LABEL = "Parameter-step tolerance"
DE_CONVERGENCE_TOLERANCE_LABEL = "Population convergence tolerance"
FIT_DATASET_WEIGHT_MODE_LABELS = {
    "implicit": "Implicit weights (1/N per dataset)",
    "user": "User weights only",
}
FIT_METHOD_LABELS = {
    "trf": "Trust Region Reflective",
    "dogbox": "Dogbox",
    "differential_evolution": "Differential Evolution",
}


def fit_method_label(value: object) -> str:
    return FIT_METHOD_LABELS[validate_fit_method(value)]


def admit_fit_tolerance(text: str, label: str) -> float:
    """Admit one optimizer tolerance box, named by its visible label."""
    value = parse_finite_numeric_text(text, field=label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return value
