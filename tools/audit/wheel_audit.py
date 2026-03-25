#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import tempfile
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


INVALID_WIN_CHARS = set('<>:"\\|?*')
RESERVED_DEVICE_BASES = {
    "con",
    "prn",
    "aux",
    "nul",
    *{f"com{i}" for i in range(1, 10)},
    *{f"lpt{i}" for i in range(1, 10)},
}


@dataclass(frozen=True)
class BuildResult:
    attempted: int
    ok: int
    wheel_filenames: list[str]
    selected_wheel_name: str | None
    selected_sha256: str | None
    used_no_build_isolation: int
    failure_reason: str | None
    failure_tail: str | None
    zip_files: list[str]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _read_tail_lines(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _iter_repo_resource_files(repo_root: Path) -> list[str]:
    data_root = (repo_root / "kindred" / "data").resolve()
    if not data_root.exists():
        return []
    out: list[str] = []
    for dirpath_str, dirnames, filenames in os.walk(data_root):
        dirpath = Path(dirpath_str)
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            p = dirpath / name
            try:
                rel_under_data = p.relative_to(repo_root / "kindred")
            except ValueError:
                continue
            rel_posix = str(rel_under_data).replace("\\", "/")
            out.append(f"kindred/{rel_posix}")
    return sorted(out)


def _is_reserved_device_segment(segment: str) -> bool:
    trimmed = segment.rstrip(" .")
    if not trimmed:
        return False
    base = trimmed.split(".", 1)[0]
    return base.casefold() in RESERVED_DEVICE_BASES


def _scan_windows_hygiene(zip_paths: list[str]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {
        "case_conflict_groups": [],
        "reserved_names": [],
        "invalid_names": [],
        "trailing_dot_space": [],
        "long_paths": [],
        "pyc_files": [],
    }

    files = [p for p in zip_paths if p and not p.endswith("/")]
    lower_map: dict[str, set[str]] = {}
    for p in files:
        key = p.casefold()
        lower_map.setdefault(key, set()).add(p)
    conflict_groups = []
    for key in sorted(lower_map):
        members = sorted(lower_map[key])
        if len(members) > 1:
            conflict_groups.append((key, members))
    for key, members in conflict_groups:
        findings["case_conflict_groups"].append(
            f"casefold_key={key} members={members!r}"
        )

    for p in sorted(files):
        if len(p) > 240:
            findings["long_paths"].append(f"{p} | len={len(p)}")
        p_lower = p.casefold()
        if p_lower.endswith(".pyc") or "/__pycache__/" in p_lower:
            findings["pyc_files"].append(p)

        segments = p.split("/")
        for seg in segments:
            if seg in {".", ".."}:
                findings["invalid_names"].append(f"{p} | segment={seg!r} | dot_segment")
                continue
            if seg.endswith(" ") or seg.endswith("."):
                findings["trailing_dot_space"].append(
                    f"{p} | segment={seg!r} | trailing_dot_or_space"
                )
            if any(ch in INVALID_WIN_CHARS for ch in seg):
                bad = "".join(sorted({ch for ch in seg if ch in INVALID_WIN_CHARS}))
                findings["invalid_names"].append(
                    f"{p} | segment={seg!r} | invalid_chars={bad!r}"
                )
            if _is_reserved_device_segment(seg):
                findings["reserved_names"].append(
                    f"{p} | segment={seg!r} | reserved_device_name"
                )

    return findings


def _pip_wheel_build(
    *,
    repo_root: Path,
    report_dir: Path,
    timeout_seconds: float,
    tail_lines: int,
) -> BuildResult:
    build_attempted = 1
    used_no_build_isolation = 0

    with tempfile.TemporaryDirectory(dir=report_dir, prefix="_tmp_wheel_build_") as tmp:
        tmp_dir = Path(tmp).resolve()
        wheelhouse = tmp_dir / "wheelhouse"
        tmp_work = tmp_dir / "tmp"
        src_root = tmp_dir / "src"
        wheelhouse.mkdir(parents=True, exist_ok=True)
        tmp_work.mkdir(parents=True, exist_ok=True)

        def _ignore_copy(dirpath: str, names: list[str]) -> set[str]:
            out: set[str] = set()
            for name in names:
                if name in {
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    ".tox",
                    ".nox",
                    ".venv",
                    "venv",
                    "node_modules",
                    "build",
                    "dist",
                }:
                    out.add(name)
                if name == "_audit_reports":
                    out.add(name)
            return out

        shutil.copytree(
            repo_root,
            src_root,
            symlinks=True,
            ignore=_ignore_copy,
        )

        env = os.environ.copy()
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1",
                "TMPDIR": str(tmp_work),
                "TEMP": str(tmp_work),
                "TMP": str(tmp_work),
                "PYTHONPYCACHEPREFIX": str(tmp_dir / "pycache"),
            }
        )

        try:
            pip_probe = subprocess.run(  # nosec B603 - controlled local interpreter args, shell=False
                [sys.executable, "-m", "pip", "--version"],
                cwd=str(src_root),
                env=env,
                text=True,
                capture_output=True,
                timeout=10.0,
            )
        except Exception as e:
            return BuildResult(
                attempted=build_attempted,
                ok=0,
                wheel_filenames=[],
                selected_wheel_name=None,
                selected_sha256=None,
                used_no_build_isolation=0,
                failure_reason=f"pip_probe_failed:{type(e).__name__}",
                failure_tail=str(e),
                zip_files=[],
            )
        if pip_probe.returncode != 0:
            combined = (pip_probe.stdout or "") + ("\n" + pip_probe.stderr if pip_probe.stderr else "")
            return BuildResult(
                attempted=build_attempted,
                ok=0,
                wheel_filenames=[],
                selected_wheel_name=None,
                selected_sha256=None,
                used_no_build_isolation=0,
                failure_reason="pip_unavailable",
                failure_tail=_read_tail_lines(combined, max_lines=tail_lines),
                zip_files=[],
            )

        base_cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "-w",
            str(wheelhouse),
        ]

        def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(  # nosec B603 - controlled local pip args, shell=False
                cmd,
                cwd=str(src_root),
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )

        try:
            p1 = _run(base_cmd)
        except subprocess.TimeoutExpired as e:
            combined = ""
            if e.stdout:
                combined += str(e.stdout)
            if e.stderr:
                combined += "\n" + str(e.stderr)
            return BuildResult(
                attempted=build_attempted,
                ok=0,
                wheel_filenames=[],
                selected_wheel_name=None,
                selected_sha256=None,
                used_no_build_isolation=0,
                failure_reason="build_timeout",
                failure_tail=_read_tail_lines(combined, max_lines=tail_lines),
                zip_files=[],
            )
        except Exception as e:
            return BuildResult(
                attempted=build_attempted,
                ok=0,
                wheel_filenames=[],
                selected_wheel_name=None,
                selected_sha256=None,
                used_no_build_isolation=0,
                failure_reason=f"build_exception:{type(e).__name__}",
                failure_tail=str(e),
                zip_files=[],
            )

        combined1 = (p1.stdout or "") + ("\n" + p1.stderr if p1.stderr else "")
        if p1.returncode != 0:
            retry_cmd = base_cmd + ["--no-build-isolation"]
            try:
                p2 = _run(retry_cmd)
            except subprocess.TimeoutExpired as e:
                used_no_build_isolation = 1
                combined = ""
                if e.stdout:
                    combined += str(e.stdout)
                if e.stderr:
                    combined += "\n" + str(e.stderr)
                return BuildResult(
                    attempted=build_attempted,
                    ok=0,
                    wheel_filenames=[],
                    selected_wheel_name=None,
                    selected_sha256=None,
                    used_no_build_isolation=used_no_build_isolation,
                    failure_reason="build_timeout_no_build_isolation",
                    failure_tail=_read_tail_lines(combined, max_lines=tail_lines),
                    zip_files=[],
                )
            combined2 = (p2.stdout or "") + ("\n" + p2.stderr if p2.stderr else "")
            used_no_build_isolation = 1
            if p2.returncode != 0:
                tail = _read_tail_lines(
                    combined1 + "\n\n--- retry --no-build-isolation ---\n\n" + combined2,
                    max_lines=tail_lines,
                )
                return BuildResult(
                    attempted=build_attempted,
                    ok=0,
                    wheel_filenames=[],
                    selected_wheel_name=None,
                    selected_sha256=None,
                    used_no_build_isolation=used_no_build_isolation,
                    failure_reason="pip_wheel_failed",
                    failure_tail=tail,
                    zip_files=[],
                )

        wheel_paths = sorted(wheelhouse.glob("*.whl"), key=lambda p: p.name)
        if not wheel_paths:
            return BuildResult(
                attempted=build_attempted,
                ok=0,
                wheel_filenames=[],
                selected_wheel_name=None,
                selected_sha256=None,
                used_no_build_isolation=used_no_build_isolation,
                failure_reason="no_wheel_files_produced",
                failure_tail=_read_tail_lines(combined1, max_lines=tail_lines),
                zip_files=[],
            )

        selected_wheel_path = wheel_paths[0]
        wheel_bytes = selected_wheel_path.read_bytes()
        selected_sha = hashlib.sha256(wheel_bytes).hexdigest()
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
            zip_paths = [p.replace("\\", "/") for p in zf.namelist()]
            zip_files = sorted([p for p in zip_paths if p and not p.endswith("/")])

        return BuildResult(
            attempted=build_attempted,
            ok=1,
            wheel_filenames=[p.name for p in wheel_paths],
            selected_wheel_name=selected_wheel_path.name,
            selected_sha256=selected_sha,
            used_no_build_isolation=used_no_build_isolation,
            failure_reason=None,
            failure_tail=None,
            zip_files=zip_files,
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Audit J: Wheel contents + Windows filename/case hygiene (stdlib-only)."
    )
    ap.add_argument("--report-dir", required=True, help="Report directory (_audit_reports/<timestamp>).")
    ap.add_argument("--output", required=True, help="Output report path (J_wheel.txt).")
    ap.add_argument("--repo-root", default=None, help="Repo root (default: derive from __file__).")
    ap.add_argument("--timeout-seconds", type=float, default=180.0, help="Wheel build timeout seconds.")
    ap.add_argument("--build-tail-lines", type=int, default=60, help="Tail lines of pip output on failure.")
    args = ap.parse_args(argv)

    repo_root = (Path(args.repo_root).resolve() if args.repo_root else _default_repo_root()).resolve()
    report_dir = Path(args.report_dir).resolve()
    out_path = Path(args.output).resolve()

    timestamp_utc = report_dir.name
    if not re.fullmatch(r"\d{8}T\d{6}Z", timestamp_utc or ""):
        timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    out_lines: list[str] = []
    out_lines.append(
        "Kindred Audit J: Wheel contents + Windows filename/case hygiene (report-only)"
    )
    out_lines.append(f"Timestamp (UTC): {timestamp_utc}")
    out_lines.append(f"Repo root: {repo_root}")
    out_lines.append(f"Report dir: {report_dir}")
    out_lines.append("")

    build_attempted = 0
    build_ok = 0
    wheel_files_count = 0
    wheel_filenames: list[str] = []
    selected_wheel_name: str | None = None
    selected_wheel_sha256: str | None = None
    retry_no_build_isolation = 0
    missing_resources: list[str] = []
    case_conflict_groups: list[str] = []
    reserved_names: list[str] = []
    invalid_names: list[str] = []
    trailing_dot_space: list[str] = []
    long_paths: list[str] = []
    pyc_files: list[str] = []

    status = "SKIP"
    skip_reason: str | None = None

    try:
        build = _pip_wheel_build(
            repo_root=repo_root,
            report_dir=report_dir,
            timeout_seconds=float(args.timeout_seconds),
            tail_lines=int(args.build_tail_lines),
        )
        build_attempted = build.attempted
        build_ok = build.ok
        wheel_filenames = list(build.wheel_filenames)
        wheel_files_count = len(wheel_filenames)
        selected_wheel_name = build.selected_wheel_name
        selected_wheel_sha256 = build.selected_sha256
        retry_no_build_isolation = build.used_no_build_isolation

        out_lines.append("Build:")
        out_lines.append(f"- build_attempted: {build_attempted}")
        out_lines.append(f"- build_ok: {build_ok}")
        out_lines.append("- build_tool: python -m pip wheel")
        out_lines.append(f"- retry_no_build_isolation: {retry_no_build_isolation}")
        out_lines.append(f"- wheel_files: {wheel_files_count}")
        if wheel_filenames:
            out_lines.append("- wheel_filenames:")
            for name in sorted(wheel_filenames):
                out_lines.append(f"  - {name}")
        if selected_wheel_name and selected_wheel_sha256:
            out_lines.append(f"- selected_wheel: {selected_wheel_name}")
            out_lines.append(f"- selected_wheel_sha256: {selected_wheel_sha256}")
        if build_ok != 1:
            skip_reason = build.failure_reason or "wheel_build_failed"
            out_lines.append(f"- failure_reason: {skip_reason}")
            out_lines.append("")
            out_lines.append("Build output tail (most recent lines):")
            tail = build.failure_tail or ""
            if tail.strip():
                out_lines.extend([f"  {line}" for line in tail.splitlines()])
            else:
                out_lines.append("  (no output captured)")
            out_lines.append("")
            status = "SKIP"
    except ModuleNotFoundError as e:
        skip_reason = f"pip_missing:{e.name}"
        out_lines.append("Build:")
        out_lines.append("- build_attempted: 0")
        out_lines.append("- build_ok: 0")
        out_lines.append(f"- failure_reason: {skip_reason}")
        out_lines.append("")
        status = "SKIP"
    except Exception as e:
        skip_reason = f"audit_exception:{type(e).__name__}"
        out_lines.append("Build:")
        out_lines.append(f"- build_attempted: {build_attempted}")
        out_lines.append("- build_ok: 0")
        out_lines.append(f"- failure_reason: {skip_reason}")
        out_lines.append("")
        out_lines.append("Exception:")
        out_lines.append(textwrap.indent(str(e), "  "))
        out_lines.append("")
        status = "SKIP"

    if build_ok == 1:
        expected_resources = _iter_repo_resource_files(repo_root)
        wheel_files = build.zip_files
        wheel_set = set(wheel_files)
        missing_resources = sorted([p for p in expected_resources if p not in wheel_set])

        hygiene = _scan_windows_hygiene(wheel_files)
        case_conflict_groups = sorted(hygiene["case_conflict_groups"])
        reserved_names = sorted(hygiene["reserved_names"])
        invalid_names = sorted(hygiene["invalid_names"])
        trailing_dot_space = sorted(hygiene["trailing_dot_space"])
        long_paths = sorted(hygiene["long_paths"])
        pyc_files = sorted(hygiene["pyc_files"])

        out_lines.append("Checks:")
        out_lines.append(f"- wheel_zip_files: {len(wheel_files)}")
        out_lines.append(f"- expected_repo_resources: {len(expected_resources)}")
        out_lines.append(f"- missing_resources: {len(missing_resources)}")
        out_lines.append(f"- case_conflict_groups: {len(case_conflict_groups)}")
        out_lines.append(f"- reserved_names: {len(reserved_names)}")
        out_lines.append(f"- invalid_names: {len(invalid_names)}")
        out_lines.append(f"- trailing_dot_space: {len(trailing_dot_space)}")
        out_lines.append(f"- long_path_candidates: {len(long_paths)}")
        out_lines.append(f"- pyc_files: {len(pyc_files)}")
        out_lines.append("")

        out_lines.append("=== Missing Repo Resources (kindred/data/**) ===")
        if not missing_resources:
            out_lines.append("- (none)")
        else:
            for p in missing_resources:
                out_lines.append(f"- {p}")
        out_lines.append("")

        out_lines.append("=== Windows Case-Conflict Groups (case-insensitive collisions) ===")
        if not case_conflict_groups:
            out_lines.append("- (none)")
        else:
            for g in case_conflict_groups:
                out_lines.append(f"- {g}")
        out_lines.append("")

        out_lines.append("=== Reserved Device Names (Windows) ===")
        if not reserved_names:
            out_lines.append("- (none)")
        else:
            for r in reserved_names:
                out_lines.append(f"- {r}")
        out_lines.append("")

        out_lines.append("=== Invalid Names / Characters (Windows) ===")
        if not invalid_names:
            out_lines.append("- (none)")
        else:
            for r in invalid_names:
                out_lines.append(f"- {r}")
        out_lines.append("")

        out_lines.append("=== Trailing Dot / Space (Windows) ===")
        if not trailing_dot_space:
            out_lines.append("- (none)")
        else:
            for r in trailing_dot_space:
                out_lines.append(f"- {r}")
        out_lines.append("")

        out_lines.append("=== Long Path Candidates (>240 chars; informational) ===")
        if not long_paths:
            out_lines.append("- (none)")
        else:
            for r in long_paths:
                out_lines.append(f"- {r}")
        out_lines.append("")

        out_lines.append("=== .pyc Files In Wheel (should not ship) ===")
        if not pyc_files:
            out_lines.append("- (none)")
        else:
            for r in pyc_files:
                out_lines.append(f"- {r}")
        out_lines.append("")

        out_lines.append("=== Wheel Zip File List ===")
        for p in wheel_files:
            out_lines.append(p)
        out_lines.append("")

        status = "PASS"
        if (
            missing_resources
            or case_conflict_groups
            or reserved_names
            or invalid_names
            or trailing_dot_space
            or long_paths
            or pyc_files
        ):
            status = "WARN"

    if build_attempted == 0 and skip_reason:
        out_lines.append(f"Skip reason: {skip_reason}")
        out_lines.append("")

    out_lines.append(
        "WHEEL_AUDIT_COUNTS"
        f"|build_attempted={int(build_attempted)}"
        f"|build_ok={int(build_ok)}"
        f"|wheel_files={int(wheel_files_count)}"
        f"|missing_resources={len(missing_resources)}"
        f"|case_conflicts={len(case_conflict_groups)}"
        f"|reserved_names={len(reserved_names)}"
        f"|invalid_names={len(invalid_names)}"
        f"|trailing_dot_space={len(trailing_dot_space)}"
        f"|long_paths={len(long_paths)}"
        f"|pyc_files={len(pyc_files)}"
        f"|status={status}"
    )
    out_lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
