from __future__ import annotations

import importlib
import importlib.resources

import pytest


@pytest.mark.unit
def test_fit_global_authoritative_api_module_exports_core_contract() -> None:
    api = importlib.import_module("kindred.core.api.fitting")

    assert hasattr(api, "fit_global")
    assert hasattr(api, "GlobalFitResult")
    assert hasattr(api, "DatasetFitInfo")


@pytest.mark.unit
def test_gui_global_fit_code_imports_from_core_api_not_gui_shim() -> None:
    targets = (
        ("kindred.gui.fitting", "window.py"),
        ("kindred.gui.fitting", "worker.py"),
    )

    for package, filename in targets:
        source = importlib.resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        assert "from kindred.gui.compat.shims import fit_global" not in source
        assert "from kindred.core.api.fitting import fit_global" in source
