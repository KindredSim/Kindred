import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys


def test_importing_dsl_does_not_eagerly_import_builder_dependencies():
    code = r"""
import sys
import kindred.core.simulator.dsl  # noqa: F401

blocked = [
    "kindred.core.mechanism",
    "kindred.core.simulator.kinetics",
    "kindred.core.simulator.state_model",
    "kindred.core.simulator.common",
    "kindred.core.simulator.dsl_preview",
    "kindred.core.simulator.dsl_build",
]

imported = [m for m in blocked if m in sys.modules]
print(imported)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()  # nosec B603 - controlled args
    assert out == "[]", f"unexpected eager imports: {out}"
