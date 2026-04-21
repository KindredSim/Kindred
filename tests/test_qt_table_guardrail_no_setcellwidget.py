import re
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit



def test_guardrail_no_setcellwidget_in_gui_sources():
    """
    Guardrail: prevent reintroducing persistent per-cell widgets in Qt tables.

    `QTableWidget.setCellWidget(...)` is a common source of widget leaks and UI freezes when
    tables are rebuilt or rows are removed. Kindred uses delegate-based editors instead.
    """

    gui_root = Path(__file__).resolve().parent.parent / "kindred" / "gui"
    assert gui_root.exists()

    # Disallow Qt item-view widget embedding APIs (leak-prone in table-driven dialogs).
    pattern = re.compile(r"\bset(?:CellWidget|IndexWidget)\s*\(")
    allowlist: set[Path] = set()

    violations: list[str] = []
    for path in sorted(gui_root.rglob("*.py")):
        # Backups are excluded from pytest collection, but keep scan defensive.
        if any(part.startswith("_backup_before_") for part in path.parts):
            continue
        if path in allowlist:
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            violations.append(f"{path.relative_to(gui_root.parent)}:{line}")
            break

    assert not violations, (
        "setCellWidget/setIndexWidget usage is disallowed in kindred/gui:\n" + "\n".join(violations)
    )
