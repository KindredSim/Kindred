#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.audit.import_sweep_targets import (
    ModuleTarget,
    iter_module_targets as _iter_module_targets,
)

WARNING_LINE_RE = re.compile(
    r"^(?P<path>.*?):(?P<lineno>[0-9]+):\s*(?P<category>[A-Za-z_][A-Za-z0-9_]*Warning):\s*(?P<message>.*)$"
)


@dataclass(frozen=True)
class ImportWarningResult:
    target: ModuleTarget
    ok: bool
    timeout: bool
    returncode: int | None
    duration_ms: int
    warning_lines: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cap_tail(text: str | None, *, max_lines: int, max_chars: int) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if max_lines > 0:
        lines = lines[-max_lines:]
    out = "\n".join(lines)
    if max_chars > 0 and len(out) > max_chars:
        out = out[-max_chars:]
    return out

def _normalize_warning_line(repo_root: Path, line: str) -> str | None:
    line = line.strip("\n")
    m = WARNING_LINE_RE.match(line)
    if not m:
        return None
    raw_path = m.group("path").strip()
    lineno = m.group("lineno").strip()
    category = m.group("category").strip()
    message = m.group("message").strip()

    norm_path = raw_path.replace("\\", "/")
    p = Path(raw_path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(repo_root)
        except (OSError, ValueError):
            pass
        else:
            norm_path = str(rel).replace("\\", "/")

    return f"{norm_path}:{lineno}: {category}: {message}"


def _extract_warning_lines(repo_root: Path, stderr_text: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in stderr_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw:
            continue
        norm = _normalize_warning_line(repo_root, raw)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return tuple(sorted(out))


def _import_one(
    repo_root: Path,
    target: ModuleTarget,
    *,
    timeout_seconds: float,
    max_tail_lines: int,
    max_tail_chars: int,
) -> ImportWarningResult:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONWARNINGS"] = "default"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "-W",
        "default",
        "-c",
        "import importlib,sys; importlib.import_module(sys.argv[1])",
        target.module,
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(  # nosec B603 - controlled local interpreter args, shell=False
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        ok = proc.returncode == 0
        warning_lines = _extract_warning_lines(repo_root, proc.stderr) if ok else ()
        return ImportWarningResult(
            target=target,
            ok=ok,
            timeout=False,
            returncode=proc.returncode,
            duration_ms=duration_ms,
            warning_lines=warning_lines,
            stdout_tail=_cap_tail(proc.stdout, max_lines=max_tail_lines, max_chars=max_tail_chars),
            stderr_tail=_cap_tail(proc.stderr, max_lines=max_tail_lines, max_chars=max_tail_chars),
        )
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        stdout = getattr(e, "stdout", None)
        stderr = getattr(e, "stderr", None)
        return ImportWarningResult(
            target=target,
            ok=False,
            timeout=True,
            returncode=None,
            duration_ms=duration_ms,
            warning_lines=(),
            stdout_tail=_cap_tail(stdout, max_lines=max_tail_lines, max_chars=max_tail_chars),
            stderr_tail=_cap_tail(stderr, max_lines=max_tail_lines, max_chars=max_tail_chars),
        )


def _write_report(
    *,
    out_path: Path,
    repo_root: Path,
    timestamp_utc: str,
    timeout_seconds: float,
    max_tail_lines: int,
    max_tail_chars: int,
    results: list[ImportWarningResult],
) -> None:
    modules = len(results)
    failures = len([r for r in results if not r.ok])
    timeouts = len([r for r in results if r.timeout])

    ok_results = [r for r in results if r.ok]
    modules_with_warnings = len([r for r in ok_results if r.warning_lines])
    warnings = sum(len(r.warning_lines) for r in ok_results)
    unique_warning_lines = len({w for r in ok_results for w in r.warning_lines})

    out: list[str] = []
    out.append("Kindred Audit H: Import-time Python warnings sweep (report-only)")
    out.append(f"Timestamp (UTC): {timestamp_utc}")
    out.append(f"Repo root: {repo_root}")
    out.append("")
    out.append("Scope:")
    out.append("- kindred/** (all importable packages/modules)")
    out.append("")
    out.append("Policy:")
    out.append(f"- Each module import runs in an isolated subprocess with timeout {timeout_seconds:.1f}s.")
    out.append("- Warnings are enabled via PYTHONWARNINGS=default and -W default.")
    out.append("- Imports run with PYTHONDONTWRITEBYTECODE=1 to avoid writing __pycache__.")
    out.append("- Imports run with QT_QPA_PLATFORM=offscreen and MPLBACKEND=Agg for offscreen safety.")
    out.append(f"- Failure stdout/stderr are captured as deterministic tails: last {max_tail_lines} lines, last {max_tail_chars} chars.")
    out.append("")
    out.append("Counts:")
    out.append(f"- modules: {modules}")
    out.append(f"- modules_with_warnings: {modules_with_warnings}")
    out.append(f"- warnings: {warnings}")
    out.append(f"- unique_warning_lines: {unique_warning_lines}")
    out.append(f"- failures: {failures}")
    out.append(f"- timeouts: {timeouts}")
    out.append("")
    out.append(
        "WARNINGS_SWEEP_COUNTS"
        f"|modules={modules}"
        f"|modules_with_warnings={modules_with_warnings}"
        f"|warnings={warnings}"
        f"|unique_warning_lines={unique_warning_lines}"
        f"|failures={failures}"
        f"|timeouts={timeouts}"
    )
    out.append("")

    out.append("=== Modules With Warnings (deterministic order) ===")
    if modules_with_warnings == 0:
        out.append("- (none)")
    else:
        for r in ok_results:
            if not r.warning_lines:
                continue
            t = r.target
            out.append(f"- {t.module}  ({t.relpath})")
            for w in r.warning_lines:
                out.append(f"  - {w}")
    out.append("")

    out.append("=== Incomplete Sweep: Import Failures / Timeouts (deterministic order) ===")
    if failures == 0:
        out.append("- (none)")
    else:
        for r in results:
            if r.ok:
                continue
            t = r.target
            status = "TIMEOUT" if r.timeout else "ERROR"
            out.append(f"- {t.module}  [{status}]  ({t.relpath})  duration_ms={r.duration_ms}")
            if r.returncode is not None:
                out.append(f"  - returncode: {r.returncode}")
            if r.stdout_tail:
                out.append("  - stdout_tail:")
                for line in r.stdout_tail.split("\n"):
                    out.append(f"    {line}")
            if r.stderr_tail:
                out.append("  - stderr_tail:")
                for line in r.stderr_tail.split("\n"):
                    out.append(f"    {line}")
            out.append("")

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit H: import-time warnings sweep (stdlib-only).")
    ap.add_argument("--out", required=True, help="Output report path (H_warnings.txt).")
    ap.add_argument("--root", default=None, help="Repo root (default: derive from __file__).")
    ap.add_argument("--timeout-seconds", type=float, default=10.0, help="Timeout per module import.")
    ap.add_argument("--max-tail-lines", type=int, default=25, help="Max stdout/stderr tail lines per failure/timeout.")
    ap.add_argument("--max-tail-chars", type=int, default=4000, help="Max stdout/stderr tail chars per failure/timeout.")
    args = ap.parse_args(argv)

    repo_root = (Path(args.root).resolve() if args.root else _default_repo_root()).resolve()
    out_path = Path(args.out).resolve()
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    targets = _iter_module_targets(repo_root, include_tools=False)
    results: list[ImportWarningResult] = []
    for t in targets:
        results.append(
            _import_one(
                repo_root,
                t,
                timeout_seconds=float(args.timeout_seconds),
                max_tail_lines=int(args.max_tail_lines),
                max_tail_chars=int(args.max_tail_chars),
            )
        )

    _write_report(
        out_path=out_path,
        repo_root=repo_root,
        timestamp_utc=timestamp_utc,
        timeout_seconds=float(args.timeout_seconds),
        max_tail_lines=int(args.max_tail_lines),
        max_tail_chars=int(args.max_tail_chars),
        results=results,
    )

    counts_line = (
        "WARNINGS_SWEEP_COUNTS"
        f"|modules={len(results)}"
        f"|modules_with_warnings={len([r for r in results if r.ok and r.warning_lines])}"
        f"|warnings={sum(len(r.warning_lines) for r in results if r.ok)}"
        f"|unique_warning_lines={len({w for r in results if r.ok for w in r.warning_lines})}"
        f"|failures={len([r for r in results if not r.ok])}"
        f"|timeouts={len([r for r in results if r.timeout])}"
    )
    print(counts_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
