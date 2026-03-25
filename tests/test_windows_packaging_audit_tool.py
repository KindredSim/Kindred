import pytest

from tools.audit import windows_packaging_audit

pytestmark = [pytest.mark.unit]


def test_windows_packaging_audit_main_smoke(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred").mkdir(parents=True, exist_ok=True)
    (repo_root / "kindred" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "kindred" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    report_dir = tmp_path / "reports"
    output = report_dir / "L_windows_packaging.txt"
    rc = windows_packaging_audit.main(
        [
            "--repo-root",
            str(repo_root),
            "--report-dir",
            str(report_dir),
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    report = output.read_text(encoding="utf-8")
    assert "Kindred Audit L: Windows packaging readiness" in report
    assert "WINDOWS_PACKAGING_COUNTS" in report
