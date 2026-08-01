"""
Kindred GUI package.

Contracts
---------
- Contains all PySide6 UI wiring: main window, widgets, dialogs, controllers.
- Console scripts and `python -m kindred` share the canonical
  `kindred.__main__:main` route, which boots QApplication and shows
  `kindred.gui.main_window.MainWindow`.
- Feature set: simulation + plotting, sliders, statistics/CTC, global fit,
  solver settings, State Network editing, templates, dataset import,
  original-data export, and generated XLSX result export.
- This __init__ does not import PySide6 at module import time to avoid
  side-effects in import-sensitive contexts (for example tests).
- Bundled `kindred/data` resources are accessed through `kindred.io.resources`
  (no cwd reliance).

Structure
---------
main_window.py   : QMainWindow, menus, docks, signals and slots
controllers/     : configuration, project-disposition, and simulation-job ownership
widgets/         : editors, dialogs, and PyQtGraph-backed plot widgets
"""

__all__: list[str] = []
