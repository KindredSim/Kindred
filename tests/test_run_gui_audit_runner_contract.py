from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.integration



def _load_run_gui_audit_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "tools" / "run_gui_audit.py"
    spec = importlib.util.spec_from_file_location("run_gui_audit", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_gui_audit_defaults_to_report_dir_and_ascii_status(tmp_path, monkeypatch, capsys):
    mod = _load_run_gui_audit_module()
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True)
    for name in ("gui_static_scan.py", "gui_dynamic_probe.py", "gui_correlate.py"):
        (tools_dir / name).write_text("print('ok')\n", encoding="utf-8")

    seen: dict[str, object] = {}

    def fake_run_step(*, step_name, script_path, cwd, env, timeout_seconds, tail_lines):  # noqa: ANN001
        artifacts_dir = Path(env["KINDRED_GUI_AUDIT_ARTIFACTS_DIR"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "gui_wiring_audit.md").write_text("markdown\n", encoding="utf-8")
        (artifacts_dir / "gui_wiring_audit.csv").write_text("csv\n", encoding="utf-8")
        seen["artifacts_dir"] = artifacts_dir
        seen["step_name"] = step_name
        seen["script_path"] = Path(script_path)
        seen["cwd"] = Path(cwd)
        seen["timeout_seconds"] = timeout_seconds
        seen["tail_lines"] = tail_lines
        print(f"GUI_AUDIT_STEP|step={step_name}|status=ok|returncode=0")
        return mod.StepResult(status="ok", returncode=0)

    monkeypatch.setattr(mod, "run_step", fake_run_step)

    rc = mod.main(["--repo-root", str(repo_root)])

    out = capsys.readouterr().out
    assert rc == 0
    report_dir_lines = [line for line in out.splitlines() if line.startswith("AUDIT_REPORT_DIR|")]
    assert len(report_dir_lines) == 1
    report_dir = Path(report_dir_lines[0].split("|", 1)[1])
    assert report_dir.parent == repo_root / "_audit_reports"
    assert seen["artifacts_dir"] == report_dir / "G_gui_audit_artifacts"
    assert seen["cwd"] == repo_root
    assert "[OK] Markdown report:" in out
    assert "[OK] CSV report:" in out
    assert "GUI_AUDIT_STEP|step=Static Scan|status=ok|returncode=0" in out
    assert "✓" not in out
    assert "✗" not in out
