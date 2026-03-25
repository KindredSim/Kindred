from __future__ import annotations

import importlib.resources
import re

import pytest


TARGETS = (
    ("kindred.gui", "simulation_worker.py"),
    ("kindred.gui.widgets", "shortcuts_dialog.py"),
)


@pytest.mark.unit
def test_gui_logging_uses_parameterized_style_in_target_modules() -> None:
    pattern = re.compile(r"logger\.(?:debug|info|warning|error|exception|critical)\(\s*f[\"']")

    offenders: list[str] = []
    for package, filename in TARGETS:
        source = importlib.resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        if pattern.search(source):
            offenders.append(f"{package}:{filename}")

    assert offenders == []
