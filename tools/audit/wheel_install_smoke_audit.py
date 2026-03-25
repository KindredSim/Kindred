#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess  # nosec B404 - audit runner uses controlled subprocess arguments
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager, suppress
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path


KNOWN_RESOURCE_REL = "presets/M1.txt"


@dataclass(frozen=True)
class CmdResult:
    ok: bool
    returncode: int | None
    timed_out: bool
    seconds: float
    stdout_tail: str
    stderr_tail: str


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_for_containment(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except Exception:
        return path.absolute()


def _is_within(child: Path, parent: Path) -> bool:
    child_r = _resolve_for_containment(child)
    parent_r = _resolve_for_containment(parent)
    return child_r == parent_r or child_r.is_relative_to(parent_r)


def _subprocess_cwd(*, tmp_run_dir: Path, repo_root: Path) -> Path:
    tmp_run_dir_r = _resolve_for_containment(tmp_run_dir)
    repo_root_r = _resolve_for_containment(repo_root)
    if _is_within(tmp_run_dir_r, repo_root_r):
        raise ValueError(f"Audit K isolation failure: tmp_run_dir is within repo_root: {tmp_run_dir_r}")
    return tmp_run_dir_r


def _sanitized_subprocess_env(base_env: Mapping[str, str]) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        env.pop(key, None)
    return env


@contextmanager
def _choose_tmp_run_dir(*, repo_root: Path) -> Iterator[Path]:
    repo_root_r = _resolve_for_containment(repo_root)

    candidates: list[Path] = []
    with suppress(OSError):
        candidates.append(Path(tempfile.gettempdir()))

    for base_dir in candidates:
        base_dir_r = _resolve_for_containment(base_dir)
        if _is_within(base_dir_r, repo_root_r):
            continue
        if not base_dir_r.exists():
            continue
        try:
            with tempfile.TemporaryDirectory(dir=str(base_dir_r), prefix="audit_k_run_") as td:
                tmp_run_dir = _resolve_for_containment(Path(td))
                if _is_within(tmp_run_dir, repo_root_r):
                    continue
                yield tmp_run_dir
                return
        except OSError:
            continue

    with tempfile.TemporaryDirectory(prefix="audit_k_run_") as td:
        tmp_run_dir = _resolve_for_containment(Path(td))
        if _is_within(tmp_run_dir, repo_root_r):
            raise RuntimeError(f"Audit K isolation failure: no temp dir outside repo_root: {tmp_run_dir}")
        yield tmp_run_dir


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _tail_text(text: str, *, max_chars: int, max_lines: int) -> str:
    text = text or ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str],
    timeout_seconds: float,
    max_tail_chars: int,
    max_tail_lines: int,
) -> CmdResult:
    started = time.monotonic()
    try:
        p = subprocess.run(  # nosec B603 - command is constructed by this audit with shell=False
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        seconds = time.monotonic() - started
        return CmdResult(
            ok=(p.returncode == 0),
            returncode=int(p.returncode),
            timed_out=False,
            seconds=seconds,
            stdout_tail=_tail_text(p.stdout, max_chars=max_tail_chars, max_lines=max_tail_lines),
            stderr_tail=_tail_text(p.stderr, max_chars=max_tail_chars, max_lines=max_tail_lines),
        )
    except subprocess.TimeoutExpired as e:
        seconds = time.monotonic() - started
        stdout = ""
        stderr = ""
        if e.stdout:
            stdout = str(e.stdout)
        if e.stderr:
            stderr = str(e.stderr)
        return CmdResult(
            ok=False,
            returncode=None,
            timed_out=True,
            seconds=seconds,
            stdout_tail=_tail_text(stdout, max_chars=max_tail_chars, max_lines=max_tail_lines),
            stderr_tail=_tail_text(stderr, max_chars=max_tail_chars, max_lines=max_tail_lines),
        )
    except Exception as e:
        seconds = time.monotonic() - started
        return CmdResult(
            ok=False,
            returncode=None,
            timed_out=False,
            seconds=seconds,
            stdout_tail="",
            stderr_tail=f"{type(e).__name__}: {e}",
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_repo_to_temp(*, repo_root: Path, dest: Path) -> None:
    def _ignore_copy(_dirpath: str, names: list[str]) -> set[str]:
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
        dest,
        symlinks=True,
        ignore=_ignore_copy,
    )


@dataclass(frozen=True)
class WheelBuild:
    attempted: int
    ok: int
    used_no_build_isolation: int
    wheel_path: Path | None
    wheel_sha256: str | None
    wheel_files: list[str]
    failure_reason: str | None
    failure_tail: str | None


def _build_wheel(
    *,
    repo_root: Path,
    tmp_run_dir: Path,
    timeout_seconds: float,
    max_tail_chars: int,
    max_tail_lines: int,
) -> WheelBuild:
    attempted = 1
    used_no_build_isolation = 0
    tmp_dir = tmp_run_dir.resolve()
    src_root = tmp_dir / "src"
    wheelhouse = tmp_dir / "wheelhouse"
    tmp_work = tmp_dir / "tmp"
    pycache = tmp_dir / "pycache"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    tmp_work.mkdir(parents=True, exist_ok=True)
    pycache.mkdir(parents=True, exist_ok=True)

    if src_root.exists():
        shutil.rmtree(src_root, ignore_errors=True)
    _copy_repo_to_temp(repo_root=repo_root, dest=src_root)

    env = _sanitized_subprocess_env(os.environ.copy())
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_PROGRESS_BAR": "off",
            "TMPDIR": str(tmp_work),
            "TEMP": str(tmp_work),
            "TMP": str(tmp_work),
            "PYTHONPYCACHEPREFIX": str(pycache),
        }
    )

    pip_probe = _run_cmd(
        [sys.executable, "-m", "pip", "--version"],
        cwd=src_root,
        env=env,
        timeout_seconds=10.0,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if not pip_probe.ok:
        combined = (pip_probe.stdout_tail + "\n" + pip_probe.stderr_tail).strip()
        return WheelBuild(
            attempted=attempted,
            ok=0,
            used_no_build_isolation=0,
            wheel_path=None,
            wheel_sha256=None,
            wheel_files=[],
            failure_reason="pip_unavailable",
            failure_tail=combined,
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
    p1 = _run_cmd(
        base_cmd,
        cwd=src_root,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if not p1.ok:
        used_no_build_isolation = 1
        p2 = _run_cmd(
            base_cmd + ["--no-build-isolation"],
            cwd=src_root,
            env=env,
            timeout_seconds=timeout_seconds,
            max_tail_chars=max_tail_chars,
            max_tail_lines=max_tail_lines,
        )
        if not p2.ok:
            combined = "\n\n".join(
                [
                    "--- pip wheel (first attempt) ---",
                    (p1.stdout_tail + "\n" + p1.stderr_tail).strip(),
                    "--- pip wheel --no-build-isolation (retry) ---",
                    (p2.stdout_tail + "\n" + p2.stderr_tail).strip(),
                ]
            ).strip()
            reason = "build_timeout" if (p1.timed_out or p2.timed_out) else "build_failed"
            return WheelBuild(
                attempted=attempted,
                ok=0,
                used_no_build_isolation=used_no_build_isolation,
                wheel_path=None,
                wheel_sha256=None,
                wheel_files=[],
                failure_reason=reason,
                failure_tail=_tail_text(combined, max_chars=max_tail_chars, max_lines=max_tail_lines),
            )

    wheels = sorted([p for p in wheelhouse.glob("*.whl") if p.is_file()], key=lambda p: p.name)
    wheel_files = [p.name for p in wheels]
    if not wheels:
        combined = (p1.stdout_tail + "\n" + p1.stderr_tail).strip()
        return WheelBuild(
            attempted=attempted,
            ok=0,
            used_no_build_isolation=used_no_build_isolation,
            wheel_path=None,
            wheel_sha256=None,
            wheel_files=wheel_files,
            failure_reason="wheel_missing",
            failure_tail=combined,
        )

    wheel_path = wheels[0].resolve()
    sha256 = _sha256_file(wheel_path)
    return WheelBuild(
        attempted=attempted,
        ok=1,
        used_no_build_isolation=used_no_build_isolation,
        wheel_path=wheel_path,
        wheel_sha256=sha256,
        wheel_files=wheel_files,
        failure_reason=None,
        failure_tail=None,
    )


@dataclass(frozen=True)
class VenvInfo:
    ok: int
    venv_dir: Path | None
    python_path: Path | None
    system_site_packages: int
    pip_version: str | None
    failure_reason: str | None
    failure_tail: str | None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_venv(
    *,
    tmp_run_dir: Path,
    timeout_seconds: float,
    max_tail_chars: int,
    max_tail_lines: int,
) -> VenvInfo:
    system_site_packages = 0

    venv_dir = (tmp_run_dir.resolve() / "venv").resolve()
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    env = _sanitized_subprocess_env(os.environ.copy())
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_PROGRESS_BAR": "off",
        }
    )
    cmd = [sys.executable, "-m", "venv"]
    cmd.append(str(venv_dir))
    venv_res = _run_cmd(
        cmd,
        cwd=None,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if not venv_res.ok:
        combined = (venv_res.stdout_tail + "\n" + venv_res.stderr_tail).strip()
        return VenvInfo(
            ok=0,
            venv_dir=None,
            python_path=None,
            system_site_packages=system_site_packages,
            pip_version=None,
            failure_reason="venv_create_timeout" if venv_res.timed_out else "venv_create_failed",
            failure_tail=combined,
        )

    python_path = _venv_python(venv_dir)
    ensurepip_res = _run_cmd(
        [str(python_path), "-m", "ensurepip", "--upgrade"],
        cwd=None,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if not ensurepip_res.ok:
        combined = (ensurepip_res.stdout_tail + "\n" + ensurepip_res.stderr_tail).strip()
        return VenvInfo(
            ok=0,
            venv_dir=None,
            python_path=None,
            system_site_packages=system_site_packages,
            pip_version=None,
            failure_reason="ensurepip_timeout" if ensurepip_res.timed_out else "ensurepip_failed",
            failure_tail=combined,
        )

    pipv = _run_cmd(
        [str(python_path), "-m", "pip", "--version"],
        cwd=None,
        env=env,
        timeout_seconds=10.0,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    pip_version = None
    if pipv.stdout_tail.strip():
        pip_version = pipv.stdout_tail.strip().splitlines()[-1]

    return VenvInfo(
        ok=1,
        venv_dir=venv_dir,
        python_path=python_path,
        system_site_packages=system_site_packages,
        pip_version=pip_version,
        failure_reason=None,
        failure_tail=None,
    )


@dataclass(frozen=True)
class InstallInfo:
    ok: int
    attempted_no_deps_fallback: int
    failure_reason: str | None
    failure_tail: str | None


def _install_wheel(
    *,
    python_path: Path,
    wheel_path: Path,
    cwd: Path | None,
    timeout_seconds: float,
    max_tail_chars: int,
    max_tail_lines: int,
) -> InstallInfo:
    env = _sanitized_subprocess_env(os.environ.copy())
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_PROGRESS_BAR": "off",
        }
    )
    base_cmd = [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--no-index",
        str(wheel_path),
    ]
    p1 = _run_cmd(
        base_cmd,
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if p1.ok:
        return InstallInfo(ok=1, attempted_no_deps_fallback=0, failure_reason=None, failure_tail=None)

    p2 = _run_cmd(
        base_cmd[:]
        + [
            "--no-deps",
        ],
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=max_tail_chars,
        max_tail_lines=max_tail_lines,
    )
    if p2.ok:
        return InstallInfo(ok=1, attempted_no_deps_fallback=1, failure_reason=None, failure_tail=None)

    combined = "\n\n".join(
        [
            "--- pip install (first attempt; deps enforced; --no-index) ---",
            (p1.stdout_tail + "\n" + p1.stderr_tail).strip(),
            "--- pip install --no-deps (fallback; --no-index) ---",
            (p2.stdout_tail + "\n" + p2.stderr_tail).strip(),
        ]
    ).strip()
    reason = "install_timeout" if (p1.timed_out or p2.timed_out) else "install_failed"
    return InstallInfo(
        ok=0,
        attempted_no_deps_fallback=1,
        failure_reason=reason,
        failure_tail=_tail_text(combined, max_chars=max_tail_chars, max_lines=max_tail_lines),
    )


def _get_purelib(*, python_path: Path, timeout_seconds: float) -> Path | None:
    env = _sanitized_subprocess_env(os.environ.copy())
    res = _run_cmd(
        [str(python_path), "-c", "import sysconfig; print(sysconfig.get_paths().get('purelib',''))"],
        cwd=None,
        env=env,
        timeout_seconds=timeout_seconds,
        max_tail_chars=4000,
        max_tail_lines=25,
    )
    if not res.ok:
        return None
    line = (res.stdout_tail or "").strip().splitlines()[-1].strip()
    if not line:
        return None
    return Path(line).resolve()


def _scan_case_conflicts(purelib: Path) -> tuple[int, list[str]]:
    candidates: list[Path] = []
    pkg = purelib / "kindred"
    if pkg.exists():
        candidates.append(pkg)
    if purelib.exists():
        for child in sorted(purelib.iterdir(), key=lambda p: p.name.casefold()):
            n = child.name
            if n.casefold().startswith("kindred-") and n.casefold().endswith(".dist-info"):
                candidates.append(child)

    # Fall back: include anything starting with "kindred" (case-insensitive).
    if not candidates and purelib.exists():
        for child in sorted(purelib.iterdir(), key=lambda p: p.name.casefold()):
            if child.name.casefold().startswith("kindred"):
                candidates.append(child)

    seen: dict[str, set[str]] = {}
    for root in candidates:
        if root.is_file():
            rel = root.name
            seen.setdefault(rel.casefold(), set()).add(rel)
            continue
        for dirpath_str, _dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath_str)
            for fname in filenames:
                full = dirpath / fname
                try:
                    rel_under = full.relative_to(purelib)
                except ValueError:
                    continue
                rel_posix = str(rel_under).replace("\\", "/")
                seen.setdefault(rel_posix.casefold(), set()).add(rel_posix)

    groups: list[str] = []
    for key in sorted(seen):
        members = sorted(seen[key])
        if len(members) > 1:
            groups.append(f"casefold_key={key} members={members!r}")
    return (len(groups), groups)


def _smoke_checks(
    *,
    python_path: Path,
    cwd: Path | None,
    repo_root: Path,
    purelib: Path | None,
    timeout_seconds: float,
    max_tail_chars: int,
    max_tail_lines: int,
) -> tuple[list[tuple[str, CmdResult]], int, int]:
    env = _sanitized_subprocess_env(os.environ.copy())
    env.update(
        {
            "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen"),
            "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_PROGRESS_BAR": "off",
        }
    )

    repo_root_str = str(_resolve_for_containment(repo_root))
    path_guard_prelude = textwrap.dedent(
        f"""
        import sys
        import sysconfig
        from pathlib import Path

        repo_root = Path({str(_resolve_for_containment(repo_root))!r}).resolve()

        while "" in sys.path:
            sys.path.remove("")

        cleaned = []
        for p in sys.path:
            if not p:
                continue
            try:
                rp = Path(p).resolve()
            except Exception:
                cleaned.append(p)
                continue
            try:
                rp.relative_to(repo_root)
                continue
            except Exception:
                cleaned.append(p)
        sys.path[:] = cleaned

        venv_paths = sysconfig.get_paths()
        venv_purelib = venv_paths.get("purelib", "")
        venv_platlib = venv_paths.get("platlib", "")
        preferred = [p for p in [venv_purelib, venv_platlib] if p]
        for p in reversed(preferred):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        """
    ).strip()

    def _script(*lines: str) -> str:
        return "\n".join([path_guard_prelude, *lines]).strip() + "\n"

    checks: list[tuple[str, list[str]]] = [
        (
            "K0 ImportOriginGuard",
            [
                str(python_path),
                "-c",
                _script(
                    "import kindred",
                    "from pathlib import Path",
                    "",
                    "pkg = Path(kindred.__file__).resolve()",
                    f"repo = Path({repo_root_str!r}).resolve()",
                    "purelib_raw = sysconfig.get_paths().get('purelib', '')",
                    "purelib = Path(purelib_raw).resolve() if purelib_raw else None",
                    "",
                    "in_repo = 0",
                    "try:",
                    "    pkg.relative_to(repo)",
                    "    in_repo = 1",
                    "except Exception:",
                    "    in_repo = 0",
                    "",
                    "in_purelib = 0",
                    "if purelib is not None:",
                    "    try:",
                    "        pkg.relative_to(purelib)",
                    "        in_purelib = 1",
                    "    except Exception:",
                    "        in_purelib = 0",
                    "print(str(pkg))",
                    "print(int(in_repo))",
                    "print(int(in_purelib))",
                    "if in_repo or purelib is None or (not in_purelib):",
                    "    raise SystemExit(2)",
                    "raise SystemExit(0)",
                ),
            ],
        ),
        (
            "K1 ImportKindred",
            [
                str(python_path),
                "-c",
                _script(
                    "import kindred",
                    "print(kindred.get_version())",
                ),
            ],
        ),
        (
            "K2 ImportResourcesAndResolvePath",
            [
                str(python_path),
                "-c",
                _script(
                    "from kindred.io.resources import get_resource_path",
                    f"p = get_resource_path({KNOWN_RESOURCE_REL!r})",
                    "print(str(p))",
                    "print(int(p.exists()))",
                ),
            ],
        ),
        (
            "K3 ImportMainOnly",
            [
                str(python_path),
                "-c",
                _script(
                    "import kindred.__main__",
                ),
            ],
        ),
        (
            "K4 ImportlibResourcesProbe",
            [
                str(python_path),
                "-c",
                _script(
                    "import importlib.resources as ir",
                    f'f = ir.files("kindred").joinpath("data", *{KNOWN_RESOURCE_REL.split("/")!r})',
                    "print(int(f.is_file()))",
                    'data = f.read_text(encoding="utf-8")',
                    "print(len(data))",
                ),
            ],
        ),
    ]

    results: list[tuple[str, CmdResult]] = []
    failures = 0
    timeouts = 0
    for name, cmd in checks:
        res = _run_cmd(
            cmd,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            max_tail_chars=max_tail_chars,
            max_tail_lines=max_tail_lines,
        )
        if not res.ok:
            failures += 1
        if res.timed_out:
            timeouts += 1
        results.append((name, res))
    return results, failures, timeouts


def _render_report(
    *,
    out_file: Path,
    repo_root: Path,
    timestamp_utc: str,
    build: WheelBuild,
    venv: VenvInfo,
    install: InstallInfo | None,
    purelib: Path | None,
    resource_on_disk_ok: int,
    case_conflicts: int,
    case_conflict_groups: list[str],
    smoke: list[tuple[str, CmdResult]],
    smoke_failures: int,
    smoke_timeouts: int,
    status: str,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def _yn(val: bool) -> str:
        return "yes" if val else "no"

    lines: list[str] = []
    lines.append("Kindred Audit K: Wheel build + install + smoke checks (report-only)")
    lines.append(f"Timestamp (UTC): {timestamp_utc}")
    lines.append(f"Repo root: {repo_root}")
    lines.append("")

    lines.append("== Wheel build ==")
    lines.append(f"build_attempted: {build.attempted}")
    lines.append(f"build_ok: {build.ok}")
    lines.append(f"used_no_build_isolation: {build.used_no_build_isolation}")
    lines.append(f"wheel_files: {build.wheel_files!r}")
    lines.append(f"wheel_selected: {build.wheel_path.name if build.wheel_path else '-'}")
    lines.append(f"wheel_sha256: {build.wheel_sha256 or '-'}")
    if build.failure_reason:
        lines.append(f"build_failure_reason: {build.failure_reason}")
    if build.failure_tail:
        lines.append("build_failure_tail:")
        lines.append(textwrap.indent(build.failure_tail, prefix="  "))
    lines.append("")

    lines.append("== Venv ==")
    lines.append(f"venv_ok: {venv.ok}")
    lines.append(f"system_site_packages: {venv.system_site_packages}")
    lines.append(f"venv_python: {_posix(venv.python_path) if venv.python_path else '-'}")
    lines.append(f"pip_version: {venv.pip_version or '-'}")
    if venv.failure_reason:
        lines.append(f"venv_failure_reason: {venv.failure_reason}")
    if venv.failure_tail:
        lines.append("venv_failure_tail:")
        lines.append(textwrap.indent(venv.failure_tail, prefix="  "))
    lines.append("")

    lines.append("== Install ==")
    if install is None:
        lines.append("install_attempted: 0")
        lines.append("install_ok: 0")
    else:
        lines.append("install_attempted: 1")
        lines.append(f"install_ok: {install.ok}")
        lines.append(f"install_no_deps_fallback_used: {install.attempted_no_deps_fallback}")
        if install.failure_reason:
            lines.append(f"install_failure_reason: {install.failure_reason}")
        if install.failure_tail:
            lines.append("install_failure_tail:")
            lines.append(textwrap.indent(install.failure_tail, prefix="  "))
    lines.append("")

    lines.append("== Installed layout ==")
    lines.append(f"purelib: {_posix(purelib) if purelib else '-'}")
    lines.append(f"known_resource_on_disk_ok: {resource_on_disk_ok}")
    lines.append(f"case_conflict_groups: {case_conflicts}")
    if case_conflict_groups:
        lines.append("case_conflict_details:")
        for item in case_conflict_groups[:50]:
            lines.append(f"  - {item}")
        if len(case_conflict_groups) > 50:
            lines.append(f"  ... truncated (total groups={len(case_conflict_groups)})")
    lines.append("")

    lines.append("== Smoke checks (subprocess imports) ==")
    for check_name, res in smoke:
        status_text = "PASS" if res.ok else ("TIMEOUT" if res.timed_out else "FAIL")
        lines.append(
            f"{check_name}: {status_text} (seconds={res.seconds:.3f} returncode={res.returncode if res.returncode is not None else '-'})"
        )
        if res.stdout_tail.strip():
            lines.append("  stdout_tail:")
            lines.append(textwrap.indent(res.stdout_tail, prefix="    "))
        if res.stderr_tail.strip():
            lines.append("  stderr_tail:")
            lines.append(textwrap.indent(res.stderr_tail, prefix="    "))
    lines.append("")

    lines.append("== Summary ==")
    lines.append(f"status: {status}")
    lines.append(
        "WHEEL_INSTALL_SMOKE_COUNTS"
        f"|build_attempted={build.attempted}"
        f"|build_ok={build.ok}"
        f"|venv_ok={venv.ok}"
        f"|install_ok={(install.ok if install is not None else 0)}"
        f"|smoke_checks={len(smoke)}"
        f"|smoke_failures={smoke_failures}"
        f"|timeouts={smoke_timeouts}"
        f"|case_conflicts={case_conflicts}"
        f"|known_resource_on_disk_ok={resource_on_disk_ok}"
        f"|status={status}"
    )
    lines.append("")
    lines.append("Notes:")
    lines.append(
        f"- Known resource probed: kindred/data/{KNOWN_RESOURCE_REL} (helper + importlib.resources)."
    )
    lines.append(
        "- Wheel build runs in a temporary repo copy; venv is temporary and removed after the audit run."
    )
    lines.append(f"- Subprocess output tails are capped; timeouts enforced: {_yn(True)}.")

    payload = "\n".join(lines).rstrip() + "\n"
    tmp_out = out_file.with_name(out_file.name + ".tmp")
    tmp_out.write_text(payload, encoding="utf-8")
    os.replace(tmp_out, out_file)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Audit K: wheel build+install smoke checks (report-only).")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--venv-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--install-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--smoke-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-tail-chars", type=int, default=4000)
    parser.add_argument("--max-tail-lines", type=int, default=60)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report_dir = args.report_dir.resolve()
    out_file = args.output.resolve()
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    build = WheelBuild(
        attempted=0,
        ok=0,
        used_no_build_isolation=0,
        wheel_path=None,
        wheel_sha256=None,
        wheel_files=[],
        failure_reason="not_run",
        failure_tail=None,
    )
    venv = VenvInfo(
        ok=0,
        venv_dir=None,
        python_path=None,
        system_site_packages=1,
        pip_version=None,
        failure_reason="not_run",
        failure_tail=None,
    )
    install: InstallInfo | None = None
    purelib: Path | None = None
    resource_on_disk_ok = 0
    case_conflicts = 0
    case_conflict_groups: list[str] = []
    smoke: list[tuple[str, CmdResult]] = []
    smoke_failures = 0
    smoke_timeouts = 0

    status = "WARN"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        with _choose_tmp_run_dir(repo_root=repo_root) as tmp_run_dir:
            work_cwd = _subprocess_cwd(tmp_run_dir=tmp_run_dir, repo_root=repo_root)
            build = _build_wheel(
                repo_root=repo_root,
                tmp_run_dir=(tmp_run_dir / "wheel_build"),
                timeout_seconds=float(args.timeout_seconds),
                max_tail_chars=int(args.max_tail_chars),
                max_tail_lines=int(args.max_tail_lines),
            )
            if not build.ok or build.wheel_path is None:
                status = "SKIP"
                _render_report(
                    out_file=out_file,
                    repo_root=repo_root,
                    timestamp_utc=timestamp_utc,
                    build=build,
                    venv=venv,
                    install=install,
                    purelib=purelib,
                    resource_on_disk_ok=resource_on_disk_ok,
                    case_conflicts=case_conflicts,
                    case_conflict_groups=case_conflict_groups,
                    smoke=smoke,
                    smoke_failures=smoke_failures,
                    smoke_timeouts=smoke_timeouts,
                    status=status,
                )
                return 0

            venv = _create_venv(
                tmp_run_dir=tmp_run_dir,
                timeout_seconds=float(args.venv_timeout_seconds),
                max_tail_chars=int(args.max_tail_chars),
                max_tail_lines=int(args.max_tail_lines),
            )
            if not venv.ok or venv.python_path is None:
                status = "WARN"
                _render_report(
                    out_file=out_file,
                    repo_root=repo_root,
                    timestamp_utc=timestamp_utc,
                    build=build,
                    venv=venv,
                    install=install,
                    purelib=purelib,
                    resource_on_disk_ok=resource_on_disk_ok,
                    case_conflicts=case_conflicts,
                    case_conflict_groups=case_conflict_groups,
                    smoke=smoke,
                    smoke_failures=smoke_failures,
                    smoke_timeouts=smoke_timeouts,
                    status=status,
                )
                return 0

            install = _install_wheel(
                python_path=venv.python_path,
                wheel_path=build.wheel_path,
                cwd=work_cwd,
                timeout_seconds=float(args.install_timeout_seconds),
                max_tail_chars=int(args.max_tail_chars),
                max_tail_lines=int(args.max_tail_lines),
            )
            if not install.ok:
                status = "WARN"
                _render_report(
                    out_file=out_file,
                    repo_root=repo_root,
                    timestamp_utc=timestamp_utc,
                    build=build,
                    venv=venv,
                    install=install,
                    purelib=purelib,
                    resource_on_disk_ok=resource_on_disk_ok,
                    case_conflicts=case_conflicts,
                    case_conflict_groups=case_conflict_groups,
                    smoke=smoke,
                    smoke_failures=smoke_failures,
                    smoke_timeouts=smoke_timeouts,
                    status=status,
                )
                return 0

            purelib = _get_purelib(python_path=venv.python_path, timeout_seconds=10.0)
            if purelib is not None:
                resource_on_disk_ok = int(
                    (purelib / "kindred" / "data" / Path(KNOWN_RESOURCE_REL)).exists()
                )
                case_conflicts, case_conflict_groups = _scan_case_conflicts(purelib)

            smoke, smoke_failures, smoke_timeouts = _smoke_checks(
                python_path=venv.python_path,
                cwd=work_cwd,
                repo_root=repo_root,
                purelib=purelib,
                timeout_seconds=float(args.smoke_timeout_seconds),
                max_tail_chars=int(args.max_tail_chars),
                max_tail_lines=int(args.max_tail_lines),
            )

        status = "PASS"
        if smoke_failures or smoke_timeouts or case_conflicts or not resource_on_disk_ok:
            status = "WARN"

        _render_report(
            out_file=out_file,
            repo_root=repo_root,
            timestamp_utc=timestamp_utc,
            build=build,
            venv=venv,
            install=install,
            purelib=purelib,
            resource_on_disk_ok=resource_on_disk_ok,
            case_conflicts=case_conflicts,
            case_conflict_groups=case_conflict_groups,
            smoke=smoke,
            smoke_failures=smoke_failures,
            smoke_timeouts=smoke_timeouts,
            status=status,
        )
        return 0
    except Exception as e:
        status = "WARN"
        failure_tail = f"{type(e).__name__}: {e}"
        build = WheelBuild(
            attempted=build.attempted,
            ok=build.ok,
            used_no_build_isolation=build.used_no_build_isolation,
            wheel_path=build.wheel_path,
            wheel_sha256=build.wheel_sha256,
            wheel_files=build.wheel_files,
            failure_reason=build.failure_reason or "exception",
            failure_tail=(build.failure_tail or "") + ("\n" + failure_tail if failure_tail else ""),
        )
        _render_report(
            out_file=out_file,
            repo_root=repo_root,
            timestamp_utc=timestamp_utc,
            build=build,
            venv=venv,
            install=install,
            purelib=purelib,
            resource_on_disk_ok=resource_on_disk_ok,
            case_conflicts=case_conflicts,
            case_conflict_groups=case_conflict_groups,
            smoke=smoke,
            smoke_failures=smoke_failures,
            smoke_timeouts=smoke_timeouts,
            status=status,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
