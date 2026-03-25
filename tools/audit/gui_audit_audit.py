#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import selectors
import shlex
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.audit.tail_utils import TailBuffer as _TailBuffer, tail_text as _tail_text


def _render_cmd(cmd: list[str]) -> str:
    try:
        return shlex.join(cmd)
    except Exception:
        return " ".join(cmd)


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    timeout: bool
    returncode: int | None
    duration_ms: int
    stdout_tail: str
    stderr_tail: str
    stdout_dropped_bytes: int
    stderr_dropped_bytes: int


def _run_capped(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    max_stream_bytes: int,
    max_tail_lines: int,
    max_tail_chars: int,
) -> CommandResult:
    start = time.monotonic()
    stdout_buf = _TailBuffer(max_bytes=max_stream_bytes)
    stderr_buf = _TailBuffer(max_bytes=max_stream_bytes)

    proc = subprocess.Popen(  # nosec B603 - controlled local subprocess args, shell=False
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

    timed_out = False
    cleanup_errors: list[str] = []

    def _read_ready(fileobj, target: _TailBuffer) -> bool:
        try:
            chunk = fileobj.read(4096)
        except Exception:
            chunk = b""
        if not chunk:
            try:
                sel.unregister(fileobj)
            except Exception as exc:
                cleanup_errors.append(f"selector unregister failed: {exc}")
            return False
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        target.add(chunk)
        return True

    try:
        while True:
            now = time.monotonic()
            remaining = float(timeout_seconds) - (now - start)
            if remaining <= 0:
                timed_out = True
                break

            if proc.poll() is not None and not sel.get_map():
                break

            events = sel.select(timeout=min(0.1, remaining))
            if not events:
                continue

            for key, _mask in events:
                stream = key.data
                if stream == "stdout":
                    _read_ready(key.fileobj, stdout_buf)
                else:
                    _read_ready(key.fileobj, stderr_buf)
    finally:
        if timed_out and proc.poll() is None:
            try:
                proc.terminate()
            except Exception as exc:
                cleanup_errors.append(f"proc.terminate failed: {exc}")
            try:
                proc.wait(timeout=0.75)
            except Exception:
                try:
                    proc.kill()
                except Exception as exc:
                    cleanup_errors.append(f"proc.kill failed: {exc}")

        try:
            proc.wait(timeout=1.0)
        except Exception as exc:
            cleanup_errors.append(f"proc.wait failed: {exc}")

        drain_deadline = time.monotonic() + 0.25
        while sel.get_map() and time.monotonic() < drain_deadline:
            events = sel.select(timeout=0.0)
            if not events:
                break
            for key, _mask in events:
                stream = key.data
                if stream == "stdout":
                    _read_ready(key.fileobj, stdout_buf)
                else:
                    _read_ready(key.fileobj, stderr_buf)

        try:
            sel.close()
        except Exception as exc:
            cleanup_errors.append(f"selector close failed: {exc}")
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception as exc:
            cleanup_errors.append(f"stdout close failed: {exc}")
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception as exc:
            cleanup_errors.append(f"stderr close failed: {exc}")

    duration_ms = int((time.monotonic() - start) * 1000)
    rc = proc.returncode
    ok = (not timed_out) and (rc == 0)

    stdout_text = stdout_buf.to_bytes().decode("utf-8", errors="replace")
    stderr_text = stderr_buf.to_bytes().decode("utf-8", errors="replace")
    if cleanup_errors:
        stderr_text += "\n[cleanup]\n" + "\n".join(cleanup_errors) + "\n"

    return CommandResult(
        ok=ok,
        timeout=timed_out,
        returncode=rc if not timed_out else None,
        duration_ms=duration_ms,
        stdout_tail=_tail_text(stdout_text, max_lines=max_tail_lines, max_chars=max_tail_chars),
        stderr_tail=_tail_text(stderr_text, max_lines=max_tail_lines, max_chars=max_tail_chars),
        stdout_dropped_bytes=stdout_buf.dropped_bytes,
        stderr_dropped_bytes=stderr_buf.dropped_bytes,
    )


def _write_report(
    *,
    out_path: Path,
    repo_root: Path,
    report_dir: Path,
    timestamp_utc: str,
    artifacts_dir: Path,
    timeout_seconds: float,
    max_stream_bytes: int,
    env_overrides: dict[str, str],
    cmd: list[str],
    result: CommandResult,
) -> None:
    status = "PASS" if result.ok else ("TIMEOUT" if result.timeout else "FAIL")
    rc_text = "-" if result.timeout or result.returncode is None else str(result.returncode)
    timeout_flag = 1 if result.timeout else 0
    exit_for_counts = -1 if result.timeout else int(result.returncode or 0)
    seconds = f"{(result.duration_ms / 1000.0):.3f}"

    out: list[str] = []
    out.append("Kindred Audit G: GUI wiring/plumbing audit (report-only)")
    out.append(f"Timestamp (UTC): {timestamp_utc}")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Report dir: {report_dir}")
    out.append(f"Artifacts dir: {artifacts_dir}")
    out.append("")
    out.append("Policy:")
    out.append(f"- Command runs in an isolated subprocess with timeout {timeout_seconds:.1f}s.")
    out.append(f"- stdout/stderr capture is capped (tail buffer) at {max_stream_bytes} bytes per stream.")
    out.append("- Environment overrides:")
    for k in sorted(env_overrides):
        out.append(f"  - {k}={env_overrides[k]}")
    out.append("")
    out.append("Command:")
    out.append(f"- {_render_cmd(cmd)}")
    out.append("")
    out.append("Result:")
    out.append(f"- status: {status}")
    out.append(f"- exit: {rc_text}")
    out.append(f"- timeout: {timeout_flag}")
    out.append(f"- seconds: {seconds}")
    out.append("")

    out.append("=== stdout (tail) ===")
    out.append(f"dropped_bytes={result.stdout_dropped_bytes}")
    if result.stdout_tail:
        out.extend(result.stdout_tail.split("\n"))
    out.append("")

    out.append("=== stderr (tail) ===")
    out.append(f"dropped_bytes={result.stderr_dropped_bytes}")
    if result.stderr_tail:
        out.extend(result.stderr_tail.split("\n"))
    out.append("")

    out.append(f"GUI_AUDIT_COUNTS|exit={exit_for_counts}|timeout={timeout_flag}|seconds={seconds}")
    out.append("")
    out_path.write_text("\n".join(out), encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit G: GUI wiring/plumbing audit (report-only).")
    ap.add_argument("--root", required=True, help="Repo root.")
    ap.add_argument("--report-dir", required=True, help="Report directory (_audit_reports/<timestamp>).")
    ap.add_argument("--out", required=True, help="Output report path (G_gui_audit.txt).")
    ap.add_argument("--timeout-seconds", type=float, default=90.0, help="Timeout for GUI audit runner.")
    ap.add_argument("--max-stream-bytes", type=int, default=16384, help="Tail-buffer cap per stream (stdout/stderr).")
    ap.add_argument("--max-tail-lines", type=int, default=80, help="Max stdout/stderr lines in report.")
    ap.add_argument("--max-tail-chars", type=int, default=8000, help="Max stdout/stderr chars in report.")
    args = ap.parse_args(argv)

    repo_root = Path(args.root).resolve()
    report_dir = Path(args.report_dir).resolve()
    out_path = Path(args.out).resolve()
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    artifacts_dir = report_dir / "G_gui_audit_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sandbox_root = report_dir / "G_gui_audit_sandbox"
    (sandbox_root / "xdg_cache").mkdir(parents=True, exist_ok=True)
    (sandbox_root / "xdg_config").mkdir(parents=True, exist_ok=True)
    (sandbox_root / "xdg_data").mkdir(parents=True, exist_ok=True)
    (sandbox_root / "mpl_config").mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    env_overrides: dict[str, str] = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "QT_QPA_PLATFORM": "offscreen",
        "MPLBACKEND": "Agg",
        "XDG_CACHE_HOME": str((sandbox_root / "xdg_cache").resolve()),
        "XDG_CONFIG_HOME": str((sandbox_root / "xdg_config").resolve()),
        "XDG_DATA_HOME": str((sandbox_root / "xdg_data").resolve()),
        "MPLCONFIGDIR": str((sandbox_root / "mpl_config").resolve()),
        "KINDRED_GUI_AUDIT_ARTIFACTS_DIR": str(artifacts_dir.resolve()),
    }
    env = dict(env_base)
    env.update(env_overrides)

    runner = (repo_root / "tools" / "run_gui_audit.py").resolve()
    cmd = [sys.executable, str(runner), "--repo-root", str(repo_root), "--report-dir", str(report_dir)]

    result = _run_capped(
        cmd,
        cwd=repo_root,
        env=env,
        timeout_seconds=float(args.timeout_seconds),
        max_stream_bytes=int(args.max_stream_bytes),
        max_tail_lines=int(args.max_tail_lines),
        max_tail_chars=int(args.max_tail_chars),
    )

    _write_report(
        out_path=out_path,
        repo_root=repo_root,
        report_dir=report_dir,
        timestamp_utc=timestamp_utc,
        artifacts_dir=artifacts_dir,
        timeout_seconds=float(args.timeout_seconds),
        max_stream_bytes=int(args.max_stream_bytes),
        env_overrides=env_overrides,
        cmd=cmd,
        result=result,
    )

    exit_for_counts = -1 if result.timeout else int(result.returncode or 0)
    timeout_flag = 1 if result.timeout else 0
    seconds = f"{(result.duration_ms / 1000.0):.3f}"
    print(f"GUI_AUDIT_COUNTS|exit={exit_for_counts}|timeout={timeout_flag}|seconds={seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
