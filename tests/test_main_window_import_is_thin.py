import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys


def test_importing_main_window_does_not_eagerly_import_heavy_gui_modules():
    code = r"""
import sys
import kindred.gui.main_window  # noqa: F401

blocked = [
    "kindred.gui.widgets.plot_tabs",
    "kindred.gui.widgets.right_panel",
    "kindred.gui.controllers.simulation_controller",
    "kindred.gui.controllers.dataset_manager",
    "kindred.gui.controllers.results_controller",
    "kindred.core.simulation_preparation",
]
imported = [m for m in blocked if m in sys.modules]
print(imported)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()  # nosec B603 - controlled args
    assert out == "[]", f"unexpected eager imports: {out}"
