from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 - controlled local audit script probes
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = REPO_ROOT / "tools" / "audit" / "run_all.sh"
RUN_STRICT = REPO_ROOT / "tools" / "audit" / "run_strict.sh"
RUN_CI = REPO_ROOT / "tools" / "audit" / "run_ci.sh"

STRICT_STAGES = "A B C D E F G H"
EXHAUSTIVE_STAGES = "A B C D E F G H I J K L"


def _bash_path(path: Path) -> str:
    resolved = Path(path).resolve()
    text = str(resolved)
    if os.name != "nt":
        return text
    normalized = text.replace("\\", "/")
    lower = normalized.lower()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if lower.startswith(prefix):
            parts = normalized.split("/")
            if len(parts) >= 5:
                return "/" + "/".join(parts[4:])
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return normalized


def _quote_bash_path(path: Path) -> str:
    return shlex.quote(_bash_path(path))


def _list_stages(*args: str) -> list[str]:
    result = subprocess.run(  # nosec B603 - controlled local script invocation
        ["bash", _bash_path(RUN_ALL), *args, "--list-stages"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    return result.stdout.splitlines()


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - controlled local script invocation
        ["bash", _bash_path(script), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )


def test_run_all_list_stages_exposes_strict_and_exhaustive_contracts():
    assert _list_stages("--strict") == ["AUDIT_MODE|strict", f"AUDIT_STAGES|{STRICT_STAGES}"]
    assert _list_stages("--exhaustive") == [
        "AUDIT_MODE|exhaustive",
        f"AUDIT_STAGES|{EXHAUSTIVE_STAGES}",
    ]


def test_run_all_default_remains_exhaustive_for_release_audits():
    assert _list_stages() == ["AUDIT_MODE|exhaustive", f"AUDIT_STAGES|{EXHAUSTIVE_STAGES}"]


def test_run_all_rejects_conflicting_modes():
    result = _run_script(RUN_ALL, "--strict", "--exhaustive", "--list-stages")

    assert result.returncode == 2
    assert "Choose only one audit mode." in result.stderr
    assert "AUDIT_STAGES|" not in result.stdout


def test_run_strict_executes_run_all_strict_and_accepts_clean_summary(tmp_path):
    audit_dir = tmp_path / "tools" / "audit"
    audit_dir.mkdir(parents=True)

    run_strict = audit_dir / "run_strict.sh"
    run_strict.write_text(RUN_STRICT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    run_strict.chmod(0o755)

    report_dir = tmp_path / "fake_report"
    args_file = tmp_path / "run_all_args.txt"
    fake_run_all = audit_dir / "run_all.sh"
    fake_run_all.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"args_file={_quote_bash_path(args_file)}",
                f"report_dir={_quote_bash_path(report_dir)}",
                'printf "%s\\n" "$@" > "${args_file}"',
                'mkdir -p "${report_dir}"',
                'printf "%s\\n" "Kindred Audit Summary" "Audit A: PASS" > "${report_dir}/SUMMARY.txt"',
                'echo "AUDIT_REPORT_DIR|${report_dir}"',
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    fake_run_all.chmod(0o755)

    result = _run_script(run_strict)

    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == ["--strict"]
    assert "STRICT GATE PASS: no WARN/FAIL/TIMEOUT in SUMMARY.txt." in result.stdout


def test_run_strict_list_stages_is_strict_and_non_reporting():
    result = _run_script(RUN_STRICT, "--list-stages")

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["AUDIT_MODE|strict", f"AUDIT_STAGES|{STRICT_STAGES}"]
    assert "AUDIT_REPORT_DIR|" not in result.stdout


def test_run_strict_rejects_exhaustive_mode():
    result = _run_script(RUN_STRICT, "--exhaustive")

    assert result.returncode == 2
    assert "Unknown argument: --exhaustive" in result.stderr
    assert "AUDIT_STAGES|" not in result.stdout


def test_run_ci_source_preserves_audit_first_and_pytest_skip_on_audit_failure():
    text = RUN_CI.read_text(encoding="utf-8")

    strict_index = text.index('bash "${SCRIPT_DIR}/run_strict.sh"')
    skip_index = text.index('echo "CI (pytest): SKIP (reason=strict_failed exit=${strict_rc})"')
    pytest_index = text.index('(cd -- "${REPO_ROOT}" && pytest -q)')

    assert strict_index < skip_index < pytest_index
    assert "run_all.sh" not in text


def test_run_ci_skips_pytest_when_strict_gate_fails(tmp_path):
    audit_dir = tmp_path / "tools" / "audit"
    audit_dir.mkdir(parents=True)

    run_ci = audit_dir / "run_ci.sh"
    run_ci.write_text(RUN_CI.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    run_ci.chmod(0o755)

    report_dir = tmp_path / "fake_report"
    fake_strict = audit_dir / "run_strict.sh"
    fake_strict.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"report_dir={_quote_bash_path(report_dir)}",
                'mkdir -p "${report_dir}"',
                'printf "%s\\n" "Kindred Audit Summary" > "${report_dir}/SUMMARY.txt"',
                'echo "AUDIT_REPORT_DIR|${report_dir}"',
                "exit 7",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    fake_strict.chmod(0o755)

    pytest_marker = tmp_path / "pytest_was_called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pytest = bin_dir / "pytest"
    fake_pytest.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"touch {_quote_bash_path(pytest_marker)}",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    fake_pytest.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{_bash_path(bin_dir)}{os.pathsep}{env['PATH']}"
    result = subprocess.run(  # nosec B603 - controlled local script invocation
        ["bash", _bash_path(run_ci)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 7
    assert not pytest_marker.exists()
    assert "CI (pytest): SKIP (reason=strict_failed exit=7)" in (
        report_dir / "SUMMARY.txt"
    ).read_text(encoding="utf-8")
    ci_pytest = (report_dir / "CI_pytest.txt").read_text(encoding="utf-8")
    assert "status: SKIPPED" in ci_pytest
    assert "strict_exit: 7" in ci_pytest
