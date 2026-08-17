"""Small Fit-local UI constants; simulation science comes from each mapped set."""

from kindred.core.fitting_method_policy import validate_fit_method

INITIAL_PREFIX = "init:"
DIFFERENTIAL_EVOLUTION_METHOD = "differential_evolution"
FUNCTION_TOLERANCE_LABEL = "Function tolerance"
DE_CONVERGENCE_TOLERANCE_LABEL = "Convergence tolerance"
CENTRED_FIXED_EFFECTS_SCOPE_LABEL = "Overall + centred dataset deviations"
FIT_METHOD_LABELS = {
    "trf": "Trust Region Reflective",
    "dogbox": "Dogbox",
    "differential_evolution": "Differential Evolution",
}


def fit_method_label(value: object) -> str:
    return FIT_METHOD_LABELS[validate_fit_method(value)]
