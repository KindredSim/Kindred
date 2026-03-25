import pytest

from tools.audit import wheel_audit

pytestmark = [pytest.mark.unit]


def test_read_tail_lines_returns_last_n_lines():
    text = "a\nb\nc\nd\ne\n"
    assert wheel_audit._read_tail_lines(text, max_lines=3) == "c\nd\ne"


def test_iter_repo_resource_files_lists_caldata(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred" / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "kindred" / "data" / "hello.txt").write_text("hi\n", encoding="utf-8")

    files = wheel_audit._iter_repo_resource_files(repo_root)
    assert files == ["kindred/data/hello.txt"]


def test_scan_windows_hygiene_detects_common_problems():
    long_name = "a" * 241
    paths = [
        "A.txt",
        "a.TXT",
        "CON.txt",
        "bad./file.txt",
        "a/../b.txt",
        f"{long_name}.txt",
        "pkg/__pycache__/x.pyc",
        "bad<name>.txt",
    ]
    findings = wheel_audit._scan_windows_hygiene(paths)

    assert findings["case_conflict_groups"]
    assert findings["reserved_names"]
    assert findings["trailing_dot_space"]
    assert findings["invalid_names"]
    assert findings["long_paths"]
    assert findings["pyc_files"]
