from __future__ import annotations

import importlib.resources

import pytest


@pytest.mark.unit
def test_dsl_module_does_not_directly_depend_on_parameter_algebra() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl.py")
        .read_text(encoding="utf-8")
    )

    assert "parameter_algebra" not in source
