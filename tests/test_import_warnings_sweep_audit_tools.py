from __future__ import annotations

import os
import stat
import subprocess  # nosec B404 - controlled local test subprocesses
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.audit import import_sweep_audit, warnings_sweep_audit
from tools.audit.import_sweep_targets import iter_module_targets

pytestmark = [pytest.mark.unit]


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_repo_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _write(repo_root / "kindred" / "__init__.py")
    _write(repo_root / "kindred" / "alpha.py", "VALUE = 1\n")
    _write(repo_root / "kindred" / "pkg" / "__init__.py")
    _write(repo_root / "kindred" / "pkg" / "beta.py", "VALUE = 2\n")
    _write(repo_root / "tools" / "__init__.py")
    _write(repo_root / "tools" / "runner.py", "VALUE = 3\n")
    _write(repo_root / "tools" / "audit" / "skip_me.py", "VALUE = 4\n")
    _write(repo_root / "_audit_reports" / "old" / "ignored.py", "VALUE = 5\n")
    _write(repo_root / "kindred" / "_backup_before_old" / "ignored.py", "VALUE = 6\n")
    return repo_root


def _make_fake_python3(fake_dir: Path, *, banner: str, counts_line: str) -> Path:
    script = fake_dir / "python3"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            out=""
            while [[ "$#" -gt 0 ]]; do
              if [[ "$1" == "--out" ]]; then
                out="$2"
                shift 2
                continue
              fi
              shift
            done
            if [[ -z "$out" ]]; then
              echo "missing --out" >&2
              exit 2
            fi
            cat > "$out" <<'EOF'
            {banner}
            Timestamp (UTC): 20260101T000000Z
            Repo root: /tmp/fake

            {counts_line}
            EOF
            """
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_iter_module_targets_preserves_exclusions_and_deterministic_order(tmp_path):
    repo_root = _build_repo_tree(tmp_path)

    without_tools = iter_module_targets(repo_root, include_tools=False)
    assert [(target.module, target.relpath, target.kind) for target in without_tools] == [
        ("kindred", "kindred/__init__.py", "package"),
        ("kindred.alpha", "kindred/alpha.py", "module"),
        ("kindred.pkg", "kindred/pkg/__init__.py", "package"),
        ("kindred.pkg.beta", "kindred/pkg/beta.py", "module"),
    ]

    with_tools = iter_module_targets(repo_root, include_tools=True)
    assert [(target.module, target.relpath, target.kind) for target in with_tools] == [
        ("kindred", "kindred/__init__.py", "package"),
        ("kindred.alpha", "kindred/alpha.py", "module"),
        ("kindred.pkg", "kindred/pkg/__init__.py", "package"),
        ("kindred.pkg.beta", "kindred/pkg/beta.py", "module"),
        ("tools", "tools/__init__.py", "package"),
        ("tools.runner", "tools/runner.py", "module"),
    ]


def test_import_sweep_main_emits_stable_counts_and_stdout(tmp_path, monkeypatch, capsys):
    repo_root = _build_repo_tree(tmp_path)
    out_path = tmp_path / "E_import_sweep.txt"
    seen_modules: list[str] = []

    def fake_run(cmd, **_kwargs):  # noqa: ANN001
        seen_modules.append(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(import_sweep_audit.subprocess, "run", fake_run)

    rc = import_sweep_audit.main(
        ["--root", str(repo_root), "--out", str(out_path), "--include-tools"]
    )
    assert rc == 0
    assert seen_modules == [
        "kindred",
        "kindred.alpha",
        "kindred.pkg",
        "kindred.pkg.beta",
        "tools",
        "tools.runner",
    ]

    report = out_path.read_text(encoding="utf-8")
    assert "Kindred Audit E: Import sweep (report-only)" in report
    assert "IMPORT_SWEEP_COUNTS|modules=6|failures=0|timeouts=0" in report
    captured = capsys.readouterr()
    assert "IMPORT_SWEEP_COUNTS|modules=6|failures=0|timeouts=0" in captured.out


def test_warnings_sweep_main_preserves_warning_normalization_and_counts(tmp_path, monkeypatch, capsys):
    repo_root = _build_repo_tree(tmp_path)
    out_path = tmp_path / "H_warnings.txt"
    warning_path = repo_root / "kindred" / "alpha.py"
    seen_modules: list[str] = []

    def fake_run(cmd, **_kwargs):  # noqa: ANN001
        module = cmd[-1]
        seen_modules.append(module)
        if module == "kindred.alpha":
            stderr = (
                f"{warning_path}:3: UserWarning: alpha warning\n"
                f"{warning_path}:3: UserWarning: alpha warning\n"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr=stderr)
        if module == "kindred.pkg.beta":
            return SimpleNamespace(returncode=1, stdout="bad stdout\n", stderr="bad stderr\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(warnings_sweep_audit.subprocess, "run", fake_run)

    rc = warnings_sweep_audit.main(["--root", str(repo_root), "--out", str(out_path)])
    assert rc == 0
    assert seen_modules == ["kindred", "kindred.alpha", "kindred.pkg", "kindred.pkg.beta"]

    report = out_path.read_text(encoding="utf-8")
    assert "Kindred Audit H: Import-time Python warnings sweep (report-only)" in report
    assert "kindred/alpha.py:3: UserWarning: alpha warning" in report
    assert "WARNINGS_SWEEP_COUNTS|modules=4|modules_with_warnings=1|warnings=1|unique_warning_lines=1|failures=1|timeouts=0" in report
    assert "  - returncode: 1" in report
    captured = capsys.readouterr()
    assert "WARNINGS_SWEEP_COUNTS|modules=4|modules_with_warnings=1|warnings=1|unique_warning_lines=1|failures=1|timeouts=0" in captured.out


def test_audit_e_wrapper_preserves_filename_and_summary_contract(tmp_path):
    fake_dir = tmp_path / "fake-bin"
    fake_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_python3(
        fake_dir,
        banner="Kindred Audit E: Import sweep (report-only)",
        counts_line="IMPORT_SWEEP_COUNTS|modules=3|failures=0|timeouts=0",
    )

    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "tools" / "audit" / "audit_E_import_sweep.sh"
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = report_dir / "SUMMARY.txt"
    env = dict(os.environ)
    env["PATH"] = str(fake_dir) + os.pathsep + env["PATH"]

    res = subprocess.run(  # nosec B603 - controlled local wrapper invocation
        ["bash", str(wrapper), str(report_dir), str(summary)],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0, res.stderr or res.stdout
    assert (report_dir / "E_import_sweep.txt").exists()
    assert "Audit E: PASS (modules=3 failures=0 timeouts=0)" in summary.read_text(encoding="utf-8")


def test_audit_h_wrapper_preserves_filename_and_summary_contract(tmp_path):
    fake_dir = tmp_path / "fake-bin"
    fake_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_python3(
        fake_dir,
        banner="Kindred Audit H: Import-time Python warnings sweep (report-only)",
        counts_line="WARNINGS_SWEEP_COUNTS|modules=4|modules_with_warnings=1|warnings=1|unique_warning_lines=1|failures=0|timeouts=0",
    )

    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "tools" / "audit" / "audit_H_warnings.sh"
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = report_dir / "SUMMARY.txt"
    env = dict(os.environ)
    env["PATH"] = str(fake_dir) + os.pathsep + env["PATH"]

    res = subprocess.run(  # nosec B603 - controlled local wrapper invocation
        ["bash", str(wrapper), str(report_dir), str(summary)],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0, res.stderr or res.stdout
    assert (report_dir / "H_warnings.txt").exists()
    assert (
        "Audit H: WARN (modules=4 modules_with_warnings=1 warnings=1 "
        "unique_warning_lines=1 failures=0 timeouts=0)"
    ) in summary.read_text(encoding="utf-8")
