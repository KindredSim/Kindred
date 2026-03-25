#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import selectors
import shlex
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import tempfile
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
class CommandSpec:
    raw: str
    argv: list[str]


@dataclass(frozen=True)
class CommandResult:
    spec: CommandSpec
    ok: bool
    timeout: bool
    returncode: int | None
    duration_ms: int
    stdout_tail: str
    stderr_tail: str
    stdout_dropped_bytes: int
    stderr_dropped_bytes: int


def _load_allowlist(path: Path) -> list[CommandSpec]:
    lines = path.read_text(encoding="utf-8").splitlines()
    specs: list[CommandSpec] = []
    seen: set[str] = set()
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace("{PY}", shlex.quote(sys.executable))
        argv = shlex.split(s, posix=True)
        if not argv:
            continue
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        key = _render_cmd(argv)
        if key in seen:
            continue
        seen.add(key)
        specs.append(CommandSpec(raw=raw.rstrip("\n"), argv=argv))
    return specs


def _run_capped(
    spec: CommandSpec,
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

    proc = subprocess.Popen(  # nosec B603 - command specs come from repo-controlled allowlist
        spec.argv,
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
        spec=spec,
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
    timestamp_utc: str,
    timeout_seconds: float,
    max_stream_bytes: int,
    env_overrides: dict[str, str],
    specs: list[CommandSpec],
    results: list[CommandResult],
) -> None:
    commands = len(specs)
    failures = len([r for r in results if (not r.ok) and (not r.timeout)])
    timeouts = len([r for r in results if r.timeout])

    out: list[str] = []
    out.append("Kindred Audit F: Runtime entrypoint sweep (report-only)")
    out.append(f"Timestamp (UTC): {timestamp_utc}")
    out.append(f"Repo root: {repo_root}")
    out.append("")
    out.append("Policy:")
    out.append(f"- Each command runs in an isolated subprocess with timeout {timeout_seconds:.1f}s.")
    out.append(f"- stdout/stderr capture is capped (tail buffer) at {max_stream_bytes} bytes per stream.")
    out.append("- Environment overrides:")
    for k in sorted(env_overrides):
        out.append(f"  - {k}={env_overrides[k]}")
    out.append("")
    out.append("Counts:")
    out.append(f"- commands: {commands}")
    out.append(f"- failures: {failures}")
    out.append(f"- timeouts: {timeouts}")
    out.append("")
    out.append(f"ENTRYPOINT_SWEEP_COUNTS|commands={commands}|failures={failures}|timeouts={timeouts}")
    out.append("")

    out.append("=== Commands (deterministic order) ===")
    for idx, spec in enumerate(specs, start=1):
        out.append(f"{idx:02d}. {_render_cmd(spec.argv)}")
    out.append("")

    out.append("=== Results ===")
    for idx, r in enumerate(results, start=1):
        status = "PASS" if r.ok else ("TIMEOUT" if r.timeout else "FAIL")
        rc_text = "-" if r.timeout or r.returncode is None else str(r.returncode)
        out.append(f"{idx:02d}. {status} rc={rc_text} duration_ms={r.duration_ms} | {_render_cmd(r.spec.argv)}")
        if status != "PASS":
            if r.stdout_dropped_bytes:
                out.append(f"  - stdout: (tail; dropped_bytes={r.stdout_dropped_bytes})")
            elif r.stdout_tail:
                out.append("  - stdout:")
            if r.stdout_tail:
                for line in r.stdout_tail.split("\n"):
                    out.append(f"    {line}")

            if r.stderr_dropped_bytes:
                out.append(f"  - stderr: (tail; dropped_bytes={r.stderr_dropped_bytes})")
            elif r.stderr_tail:
                out.append("  - stderr:")
            if r.stderr_tail:
                for line in r.stderr_tail.split("\n"):
                    out.append(f"    {line}")
        out.append("")

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit F: runtime entrypoint sweep (stdlib-only).")
    ap.add_argument("--root", required=True, help="Repo root.")
    ap.add_argument("--out", required=True, help="Output report path (F_entrypoints.txt).")
    ap.add_argument(
        "--allowlist",
        default=None,
        help="Allowlist file path (defaults to tools/audit/entrypoints_allowlist.txt under --root).",
    )
    ap.add_argument("--timeout-seconds", type=float, default=10.0, help="Timeout per entrypoint command.")
    ap.add_argument("--max-stream-bytes", type=int, default=16384, help="Tail-buffer cap per stream (stdout/stderr).")
    ap.add_argument("--max-tail-lines", type=int, default=25, help="Max stdout/stderr lines in report for failures.")
    ap.add_argument("--max-tail-chars", type=int, default=4000, help="Max stdout/stderr chars in report for failures.")
    args = ap.parse_args(argv)

    repo_root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    allowlist_path = (
        Path(args.allowlist).resolve()
        if args.allowlist
        else (repo_root / "tools" / "audit" / "entrypoints_allowlist.txt").resolve()
    )

    env_base = dict(os.environ)
    env_base.setdefault("QT_QPA_PLATFORM", "offscreen")
    env_base.setdefault("MPLBACKEND", "Agg")
    env_overrides: dict[str, str] = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "QT_QPA_PLATFORM": env_base["QT_QPA_PLATFORM"],
        "MPLBACKEND": env_base["MPLBACKEND"],
    }

    with tempfile.TemporaryDirectory(prefix="auditF_tmp_", dir=str(out_path.parent)) as tmpdir:
        tmp = Path(tmpdir)
        env_overrides.update(
            {
                "XDG_CACHE_HOME": str(tmp / "xdg_cache"),
                "XDG_CONFIG_HOME": str(tmp / "xdg_config"),
                "XDG_DATA_HOME": str(tmp / "xdg_data"),
                "MPLCONFIGDIR": str(tmp / "mpl_config"),
            }
        )

        env = dict(env_base)
        env.update(env_overrides)

        try:
            specs = _load_allowlist(allowlist_path)
        except OSError:
            specs = [
                CommandSpec(
                    raw="{PY} -c \"import kindred; print('OK: import kindred')\"",
                    argv=[sys.executable, "-c", "import kindred; print('OK: import kindred')"],
                )
            ]

        results: list[CommandResult] = []
        for spec in specs:
            results.append(
                _run_capped(
                    spec,
                    cwd=repo_root,
                    env=env,
                    timeout_seconds=float(args.timeout_seconds),
                    max_stream_bytes=int(args.max_stream_bytes),
                    max_tail_lines=int(args.max_tail_lines),
                    max_tail_chars=int(args.max_tail_chars),
                )
            )

        _write_report(
            out_path=out_path,
            repo_root=repo_root,
            timestamp_utc=timestamp_utc,
            timeout_seconds=float(args.timeout_seconds),
            max_stream_bytes=int(args.max_stream_bytes),
            env_overrides=env_overrides,
            specs=specs,
            results=results,
        )

        commands = len(specs)
        failures = len([r for r in results if (not r.ok) and (not r.timeout)])
        timeouts = len([r for r in results if r.timeout])
        print(f"ENTRYPOINT_SWEEP_COUNTS|commands={commands}|failures={failures}|timeouts={timeouts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
