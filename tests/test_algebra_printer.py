from __future__ import annotations

import math

import pytest

from kindred.core.algebra.parser import NumberNode, parse_algebra

pytestmark = pytest.mark.unit


def _expr(source: str):
    return parse_algebra(f"# Algebra\nlet value = {source}\n").lines[0].expr


@pytest.mark.parametrize(
    "source",
    [
        "(k1 ^ 2) ^ 3",
        "k1 ^ (2 ^ 3)",
        "k1 && (k2 || k3)",
        "(k1 || 0) && 0",
        "a - (b - c)",
        "a / (b / c)",
        "max([A], [B]_0 + [C](T0))",
        "0.12345678901234567",
    ],
)
def test_algebra_printer_preserves_ast_when_source_is_reparsed(source: str) -> None:
    from kindred.core.algebra.printer import expression_to_source

    emitted = expression_to_source(_expr(source))

    assert _expr(emitted) == _expr(source)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_algebra_printer_rejects_nonfinite_number_nodes(value: float) -> None:
    from kindred.core.algebra.printer import expression_to_source

    with pytest.raises(ValueError, match="non-finite"):
        expression_to_source(NumberNode(value))
