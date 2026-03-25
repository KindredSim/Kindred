import csv

import pytest

from tools import gui_static_scan

pytestmark = [pytest.mark.unit]


def test_gui_static_scanner_finds_simple_qaction(tmp_path):
    repo_root = tmp_path / "repo"
    gui_dir = repo_root / "kindred" / "gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    (gui_dir / "__init__.py").write_text("", encoding="utf-8")
    (gui_dir / "sample.py").write_text(
        "from PySide6.QtGui import QAction\n"
        "action = QAction('Hello')\n",
        encoding="utf-8",
    )

    scanner = gui_static_scan.GUIStaticScanner(str(repo_root))
    result = scanner.scan()

    assert len(result.controls) == 1
    control = result.controls[0]
    assert control.kind == "QAction"
    assert control.text == "Hello"


def test_write_csv_writes_header_and_rows(tmp_path):
    control = gui_static_scan.Control(
        kind="QAction",
        var_name="action",
        object_name="",
        text="Hello",
        shortcut="",
        enabled_default="True",
        visible_default="True",
        created_file="kindred/gui/sample.py",
        created_line=1,
        parent_var="",
    )
    result = gui_static_scan.ScanResult(
        controls=[control],
        duplicate_shortcuts={},
        orphans=[],
        never_added_actions=[],
        disabled_permanently=[],
    )

    out_path = tmp_path / "out.csv"
    gui_static_scan.write_csv(result, str(out_path))

    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0][0] == "kind"
    assert rows[1][0] == "QAction"
