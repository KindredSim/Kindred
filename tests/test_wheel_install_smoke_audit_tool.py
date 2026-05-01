import pytest

from tools.audit import wheel_install_smoke_audit

pytestmark = [pytest.mark.unit]


def test_tail_text_applies_line_and_char_limits():
    text = "\n".join([f"line-{i}" for i in range(10)])
    tailed = wheel_install_smoke_audit._tail_text(text, max_chars=100, max_lines=2)
    assert tailed.splitlines() == ["line-8", "line-9"]

    tailed_chars = wheel_install_smoke_audit._tail_text(text, max_chars=8, max_lines=50)
    assert tailed_chars.endswith("line-9")


def test_sanitized_subprocess_env_sets_safe_defaults_and_strips_pythonpath(tmp_path):
    base = {
        "PYTHONPATH": str(tmp_path / "sneaky"),
        "PYTHONHOME": str(tmp_path / "home"),
        "OTHER": "1",
    }
    env = wheel_install_smoke_audit._sanitized_subprocess_env(base)
    assert env["OTHER"] == "1"
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_subprocess_cwd_rejects_tmp_dir_inside_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    tmp_run_dir = repo_root / "run"
    tmp_run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        wheel_install_smoke_audit._subprocess_cwd(tmp_run_dir=tmp_run_dir, repo_root=repo_root)


def test_scan_case_conflicts_detects_casefold_duplicates(tmp_path, monkeypatch):
    purelib = tmp_path / "purelib"
    pkg = purelib / "kindred"
    pkg.mkdir(parents=True, exist_ok=True)

    (pkg / "x.py").write_text("# lower\n", encoding="utf-8")

    def fake_walk(root):
        assert root == pkg
        yield str(pkg), [], ["x.py", "X.py"]

    monkeypatch.setattr(wheel_install_smoke_audit.os, "walk", fake_walk)

    count, groups = wheel_install_smoke_audit._scan_case_conflicts(purelib)
    assert count == 1
    assert len(groups) == 1
    assert groups[0].casefold_key == "kindred/x.py"
    assert groups[0].members == ("kindred/X.py", "kindred/x.py")
