import pytest

from tools.audit import deps_audit

pytestmark = [pytest.mark.unit]


def test_dep_base_name_parses_common_specs():
    assert deps_audit._dep_base_name("numpy==2.0.0") == "numpy"
    assert deps_audit._dep_base_name("PySide6>=6.7.2; platform_system=='Windows'") == "PySide6"
    assert deps_audit._dep_base_name("scipy[extra]==1.13.1") == "scipy"
    assert deps_audit._dep_base_name("") == ""


def test_module_root_to_dist_name_handles_known_mappings():
    assert deps_audit._module_root_to_dist_name("pytest") == "pytest"
    assert deps_audit._module_root_to_dist_name("_pytest") == "pytest"
    assert deps_audit._module_root_to_dist_name("qdarktheme") == "pyqtdarktheme-fork"


def test_deps_audit_main_smoke_no_third_party(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred").mkdir(parents=True, exist_ok=True)
    (repo_root / "kindred" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "kindred" / "foo.py").write_text("import os\n\nVALUE = os.name\n", encoding="utf-8")

    out_file = repo_root / "D_deps.txt"
    rc = deps_audit.main(["--root", str(repo_root), "--out", str(out_file)])
    assert rc == 0

    report = out_file.read_text(encoding="utf-8")
    assert "Kindred Audit D: Deps vs imports" in report
    assert "COUNTS|" in report
    assert "third_party_modules=0" in report
