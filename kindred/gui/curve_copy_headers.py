"""Qt-free labels for main and Fit convenience copies."""

from kindred.core.algebra.printer import expression_to_source
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet


def observable_copy_definitions(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[tuple[str, str], ...]:
    """Project direct authored expressions from the result's admitted model."""

    spec = prepared_dsl.parameter_algebra_spec_for_execution()
    block = spec.observable_block
    return () if block is None else tuple(
        (statement.name, expression_to_source(statement.expr))
        for statement in block.lines
    )


def copy_series_label(
    name: str,
    unit: str | None = None,
    definition: str | None = None,
) -> str:
    label = name if definition is None else f"{name} (= {definition})"
    return label if unit is None else f"{label} ({unit})"


def copy_column_header(
    label: str,
    name: str,
    unit: str | None = None,
    definition: str | None = None,
) -> str:
    """Format the header; the clipboard owner protects and encodes the cell."""

    return f"{label}::{copy_series_label(name, unit, definition)}"
