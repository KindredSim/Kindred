#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.audit.import_sweep_targets import (
    ModuleTarget,
    iter_module_targets as _iter_module_targets,
)

@dataclass(frozen=True)
class ImportResult:
    target: ModuleTarget
    ok: bool
    timeout: bool
    returncode: int | None
    duration_ms: int
    stdout_snip: str
    stderr_snip: str


def _cap_text(text: str | None, *, max_lines: int, max_chars: int) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    lines = lines[:max_lines]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out

def _import_one(
    repo_root: Path,
    target: ModuleTarget,
    *,
    timeout_seconds: float,
    max_output_lines: int,
    max_output_chars: int,
) -> ImportResult:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
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
        return ImportResult(
            target=target,
            ok=ok,
            timeout=False,
            returncode=proc.returncode,
            duration_ms=duration_ms,
            stdout_snip=_cap_text(proc.stdout, max_lines=max_output_lines, max_chars=max_output_chars),
            stderr_snip=_cap_text(proc.stderr, max_lines=max_output_lines, max_chars=max_output_chars),
        )
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return ImportResult(
            target=target,
            ok=False,
            timeout=True,
            returncode=None,
            duration_ms=duration_ms,
            stdout_snip=_cap_text(getattr(e, "stdout", None), max_lines=max_output_lines, max_chars=max_output_chars),
            stderr_snip=_cap_text(getattr(e, "stderr", None), max_lines=max_output_lines, max_chars=max_output_chars),
        )


def _write_report(
    *,
    out_path: Path,
    repo_root: Path,
    timestamp_utc: str,
    include_tools: bool,
    timeout_seconds: float,
    results: list[ImportResult],
) -> None:
    modules = len(results)
    failures = len([r for r in results if not r.ok])
    timeouts = len([r for r in results if r.timeout])

    out: list[str] = []
    out.append("Kindred Audit E: Import sweep (report-only)")
    out.append(f"Timestamp (UTC): {timestamp_utc}")
    out.append(f"Repo root: {repo_root}")
    out.append("")
    out.append("Scope:")
    out.append("- kindred/** (always)")
    out.append(f"- tools/**: {'included' if include_tools else 'not included'} (excludes tools/audit/**)")
    out.append("")
    out.append("Policy:")
    out.append(f"- Each module import runs in an isolated subprocess with timeout {timeout_seconds:.1f}s.")
    out.append("- Imports run with PYTHONDONTWRITEBYTECODE=1 to avoid writing __pycache__.")
    out.append("- Imports run with QT_QPA_PLATFORM=offscreen and MPLBACKEND=Agg for offscreen safety.")
    out.append("")
    out.append("Counts:")
    out.append(f"- modules: {modules}")
    out.append(f"- failures: {failures}")
    out.append(f"- timeouts: {timeouts}")
    out.append("")
    out.append(f"IMPORT_SWEEP_COUNTS|modules={modules}|failures={failures}|timeouts={timeouts}")
    out.append("")

    if failures == 0:
        out.append("=== Failures ===")
        out.append("- (none)")
        out.append("")
        out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return

    out.append("=== Failures (deterministic order) ===")
    for r in results:
        if r.ok:
            continue
        t = r.target
        status = "TIMEOUT" if r.timeout else "ERROR"
        out.append(f"- {t.module}  [{status}]  ({t.relpath})  duration_ms={r.duration_ms}")
        if r.returncode is not None:
            out.append(f"  - returncode: {r.returncode}")
        if r.stdout_snip:
            out.append("  - stdout:")
            for line in r.stdout_snip.split("\n"):
                out.append(f"    {line}")
        if r.stderr_snip:
            out.append("  - stderr:")
            for line in r.stderr_snip.split("\n"):
                out.append(f"    {line}")
        out.append("")

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit E: import sweep (stdlib-only).")
    ap.add_argument("--root", required=True, help="Repo root.")
    ap.add_argument("--out", required=True, help="Output report path (E_import_sweep.txt).")
    ap.add_argument("--timeout-seconds", type=float, default=10.0, help="Timeout per module import.")
    ap.add_argument("--include-tools", action="store_true", help="Also sweep importable modules under tools/**.")
    ap.add_argument("--max-output-lines", type=int, default=25, help="Max stdout/stderr lines per failure.")
    ap.add_argument("--max-output-chars", type=int, default=4000, help="Max stdout/stderr chars per failure.")
    args = ap.parse_args(argv)

    repo_root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    targets = _iter_module_targets(repo_root, include_tools=bool(args.include_tools))
    results: list[ImportResult] = []
    for t in targets:
        results.append(
            _import_one(
                repo_root,
                t,
                timeout_seconds=float(args.timeout_seconds),
                max_output_lines=int(args.max_output_lines),
                max_output_chars=int(args.max_output_chars),
            )
        )

    _write_report(
        out_path=out_path,
        repo_root=repo_root,
        timestamp_utc=timestamp_utc,
        include_tools=bool(args.include_tools),
        timeout_seconds=float(args.timeout_seconds),
        results=results,
    )

    counts_line = f"IMPORT_SWEEP_COUNTS|modules={len(results)}|failures={len([r for r in results if not r.ok])}|timeouts={len([r for r in results if r.timeout])}"
    print(counts_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
