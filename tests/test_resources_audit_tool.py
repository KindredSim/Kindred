import pytest

from tools.audit import resources_audit

pytestmark = [pytest.mark.unit]


def test_resources_audit_main_smoke_no_findings(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred").mkdir(parents=True, exist_ok=True)
    (repo_root / "kindred" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "kindred" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    (repo_root / "kindred" / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "hello.txt").write_text("hi\n", encoding="utf-8")

    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rc = resources_audit.main(["--report-dir", str(report_dir), "--root", str(repo_root)])
    assert rc == 0

    report_path = report_dir / "I_resources.txt"
    report = report_path.read_text(encoding="utf-8")
    assert "Kindred Audit I: Resources + Windows packaging compatibility" in report
    assert "I_RESOURCES_COUNTS" in report
    assert "missing: 0" in report
