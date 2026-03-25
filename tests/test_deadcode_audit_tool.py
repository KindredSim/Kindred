import ast

import pytest

from tools.audit import deadcode_audit

pytestmark = [pytest.mark.unit]


def test_has_main_guard_detects_dunder_main():
    guarded = ast.parse("if __name__ == '__main__':\n    print('x')\n")
    assert deadcode_audit._has_main_guard(guarded) is True

    unguarded = ast.parse("print('x')\n")
    assert deadcode_audit._has_main_guard(unguarded) is False


def test_deadcode_audit_main_smoke_no_candidates(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred").mkdir(parents=True, exist_ok=True)
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)

    (repo_root / "kindred" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "kindred" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "kindred" / "__main__.py").write_text(
        "from kindred import core\nprint(core.VALUE)\n",
        encoding="utf-8",
    )

    output_path = repo_root / "C_deadcode.txt"
    rc = deadcode_audit.main(["--root", str(repo_root), "--output", str(output_path)])
    assert rc == 0

    report = output_path.read_text(encoding="utf-8")
    assert "Kindred Audit C: Dead-code candidates" in report

    captured = capsys.readouterr()
    assert "DEADCODE_AUDIT_COUNTS" in captured.out


def test_pyproject_script_entrypoint_modules_recognizes_gui_scripts(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "kindred"',
                "[project.gui-scripts]",
                'kindred = "kindred.gui_entrypoint:main"',
                'kindred-gui = "kindred.gui_entrypoint:main"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    modules = deadcode_audit._pyproject_script_entrypoint_modules(repo_root)

    assert modules == [
        ("kindred.gui_entrypoint", 4, 'kindred = "kindred.gui_entrypoint:main"'),
        ("kindred.gui_entrypoint", 5, 'kindred-gui = "kindred.gui_entrypoint:main"'),
    ]
