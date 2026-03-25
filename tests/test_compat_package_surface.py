from __future__ import annotations

import importlib


def test_compat_package_root_stays_importable_without_re_exporting_shims() -> None:
    compat = importlib.import_module("kindred.compat")

    assert getattr(compat, "__all__", []) == []
    assert not hasattr(compat, "fit_global")
    assert not hasattr(compat, "parse_dsl_to_mechanism")
    assert not hasattr(compat, "solve_ode")
