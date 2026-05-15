from __future__ import annotations

import importlib.resources
import pytest

pytestmark = pytest.mark.unit



def test_dsl_module_is_a_thin_facade_over_dsl_parse() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl.py")
        .read_text(encoding="utf-8")
    )

    assert "from .dsl_parse import (" in source
    assert "class DSLResult" not in source
    assert "class DSLIR" not in source
    assert "class ParsedStep" not in source
    assert "def _parse_dsl_ir(" not in source


def test_dsl_facade_does_not_keep_stale_parameter_definition_lazy_alias() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl.py")
        .read_text(encoding="utf-8")
    )
    parse_source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl_parse.py")
        .read_text(encoding="utf-8")
    )

    assert "ParameterDefinition" not in source
    assert "ParameterDefinition" not in parse_source
