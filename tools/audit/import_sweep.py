"""Import every kindred module in its own interpreter, in parallel.

A fresh process per module catches import errors and circular imports that only
appear when that module is imported first. Warnings are errors, so an
import-time warning fails the sweep too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 60


def _modules() -> list[str]:
    modules = []
    for path in sorted((REPO_ROOT / "kindred").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(parts))
    return modules


def _import(module: str, env: dict[str, str]) -> str | None:
    """Return the failure output, or None when the import is clean."""
    try:
        proc = subprocess.run(
            [sys.executable, "-W", "error", "-c", "import importlib, sys; importlib.import_module(sys.argv[1])", module],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {TIMEOUT_SECONDS} s"
    if proc.returncode == 0:
        return None
    return (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"


def main() -> int:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    modules = _modules()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(lambda module: (module, _import(module, env)), modules))
    failures = [(module, output) for module, output in results if output is not None]
    for module, output in failures:
        print(f"--- {module}\n{output}\n")
    print(f"import sweep: {len(modules)} modules, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
