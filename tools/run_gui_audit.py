#!/usr/bin/env python3
"""Orchestrator script to run GUI wiring audit.

This runner is intentionally usable from any working directory.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _tail_lines(text: str, *, max_lines: int) -> str:
    max_lines = int(max_lines)
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail)


def _default_report_dir(repo_root: Path) -> Path:
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return repo_root / "_audit_reports" / timestamp_utc


@dataclass(frozen=True)
class StepResult:
    status: str
    returncode: int
    detail: str | None = None


def _emit_step_status(*, step_name: str, result: StepResult) -> None:
    line = f"GUI_AUDIT_STEP|step={step_name}|status={result.status}|returncode={int(result.returncode)}"
    if result.detail:
        line += f"|detail={result.detail}"
    print(line)


def run_step(
    *,
    step_name: str,
    script_path: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float | None,
    tail_lines: int,
) -> StepResult:
    """Run a single audit step."""
    try:
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(  # noqa: S603,S607 - command is local + controlled
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

        if result.returncode != 0:
            combined = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")
            if combined.strip():
                print("\n--- Captured output (tail) ---")
                print(_tail_lines(combined, max_lines=tail_lines))
            step_result = StepResult(status="failed", returncode=int(result.returncode))
            _emit_step_status(step_name=step_name, result=step_result)
            return step_result

        step_result = StepResult(status="ok", returncode=0)
        _emit_step_status(step_name=step_name, result=step_result)
        return step_result

    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")
        if combined.strip():
            print("\n--- Captured output (tail) ---")
            print(_tail_lines(combined, max_lines=tail_lines))
        step_result = StepResult(status="timeout", returncode=-1, detail=f"timeout={timeout_seconds}")
        _emit_step_status(step_name=step_name, result=step_result)
        return step_result
    except (OSError, subprocess.SubprocessError) as exc:
        step_result = StepResult(
            status="error",
            returncode=-1,
            detail=f"{exc.__class__.__name__}:{exc}",
        )
        _emit_step_status(step_name=step_name, result=step_result)
        return step_result


def main(argv: list[str]) -> int:
    """Main entry point."""
    print("GUI Control Wiring Audit")
    print("=" * 60)

    ap = argparse.ArgumentParser(description="Run the GUI wiring/plumbing audit.")
    ap.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (defaults to the parent of this script's directory).",
    )
    ap.add_argument(
        "--report-dir",
        default=None,
        help="Audit report directory (defaults to <repo-root>/_audit_reports/<timestamp>).",
    )
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for audit artifacts (defaults to <report-dir>/G_gui_audit_artifacts).",
    )
    ap.add_argument(
        "--step-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-step timeout in seconds (0 disables).",
    )
    ap.add_argument(
        "--tail-lines",
        type=int,
        default=80,
        help="How many lines of captured output to show on failure/timeout.",
    )
    args = ap.parse_args(argv)

    default_repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else default_repo_root
    tools_dir = repo_root / "tools"
    report_dir = Path(args.report_dir).resolve() if args.report_dir else _default_report_dir(repo_root)
    artifacts_dir = (
        Path(args.artifacts_dir).resolve() if args.artifacts_dir else (report_dir / "G_gui_audit_artifacts")
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"AUDIT_REPORT_DIR|{report_dir}")

    env = dict(os.environ)
    env["KINDRED_GUI_AUDIT_ARTIFACTS_DIR"] = str(artifacts_dir)

    steps = [
        ("Static Scan", tools_dir / "gui_static_scan.py"),
        ("Dynamic Probe", tools_dir / "gui_dynamic_probe.py"),
        ("Correlate Results", tools_dir / "gui_correlate.py"),
    ]

    failed_steps = []

    timeout_seconds = float(args.step_timeout_seconds)
    if timeout_seconds <= 0:
        timeout_seconds = None

    for step_name, script_path in steps:
        if not script_path.exists():
            _emit_step_status(
                step_name=step_name,
                result=StepResult(status="missing_script", returncode=-1, detail=str(script_path)),
            )
            failed_steps.append(step_name)
            continue

        step_result = run_step(
            step_name=step_name,
            script_path=script_path,
            cwd=repo_root,
            env=env,
            timeout_seconds=timeout_seconds,
            tail_lines=int(args.tail_lines),
        )
        if step_result.status != "ok":
            failed_steps.append(step_name)

            # For dynamic probe, we can continue with static-only results
            if step_name == "Dynamic Probe":
                print("  Continuing with static-only results...")
                continue

    print(f"\n{'='*60}")
    print("Audit Complete")
    print(f"{'='*60}")

    if failed_steps:
        print(f"\nWarning: Some steps failed: {', '.join(failed_steps)}")
        print("Check the output above for details.")

    # Check for output files
    report_md = artifacts_dir / "gui_wiring_audit.md"
    report_csv = artifacts_dir / "gui_wiring_audit.csv"

    if report_md.exists():
        print(f"\n[OK] Markdown report: {report_md}")
    else:
        print(f"\n[MISSING] Markdown report not found: {report_md}")

    if report_csv.exists():
        print(f"[OK] CSV report: {report_csv}")
    else:
        print(f"[MISSING] CSV report not found: {report_csv}")

    print("\nDone!")

    return 0 if not failed_steps else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
