"""
Kindred GUI package.

Contracts
---------
- Contains all PySide6 UI wiring: main window, widgets, dialogs, controllers.
- Console-script GUI entry point is `kindred.gui_entrypoint:main`; `python -m
  kindred` uses `kindred.__main__:main`. Both boot QApplication and show
  `kindred.gui.main_window.MainWindow`.
- Feature set: simulation + plotting, sliders, statistics/CTC, global fit,
  solver settings, temperature schedules, species registry, templates/profiles,
  and CSV export (only).
- This __init__ does not import PySide6 at module import time to avoid
  side-effects in import-sensitive contexts (for example tests).
- Bundled `kindred/data` resources are accessed through `kindred.io.resources`
  (no cwd reliance).

Structure
---------
main_window.py   : QMainWindow, menus, docks, signals and slots
controllers/     : auxiliary controllers such as DatasetManager
widgets/         : editors, dialogs, and PyQtGraph-backed plot widgets
"""

__all__: list[str] = []
