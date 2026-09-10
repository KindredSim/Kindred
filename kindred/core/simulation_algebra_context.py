"""Selected scientific symbols shared by execution and result Algebra."""
from __future__ import annotations

def runtime_algebra_symbol_names(parameter_algebra_spec: object) -> frozenset[str]:
    """Classify row dependence without materializing parameter values."""
    indexed = {
        str(item.canonical_name)
        for item in parameter_algebra_spec.mechanism_namespace.ordered_items
        if item.info.role in {"k", "kf", "kr", "Keq"}
    }
    return frozenset(indexed | parameter_algebra_spec.temperature_dependent_names)


def prepare_simulation_algebra_symbols(*, temperature_schedule, coefficient_evaluator):
    """Share the candidate's bound graph without binding or evaluating it again."""
    nominal = coefficient_evaluator.nominal_temperature_K

    def symbols(time: float, requested_names: tuple[str, ...]) -> dict[str, float]:
        temperature = float(temperature_schedule(time)) if temperature_schedule is not None else nominal
        return coefficient_evaluator.parameter_values(temperature, requested_names)

    return symbols
