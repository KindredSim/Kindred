"""Bring the running interpreter's packages in line with pyproject.toml.

Compares the exact pins with what is installed and runs pip only when something
differs, so an up-to-date environment costs no network and almost no time.

    python tools/audit/sync_env.py --group dev                     # Kindred + dev tools
    python tools/audit/sync_env.py                                 # Kindred only
    python -E -s tools/audit/sync_env.py --no-project --group fitting-runtime
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
import sysconfig
import tomllib
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Only what pip installed into this interpreter counts. PYTHONPATH or the build's
# kindred.egg-info in the checkout must not stand in for an installation.
SITE_PACKAGES = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})


def _pins(requirements: list[str]) -> dict[str, str]:
    pins = {}
    for requirement in requirements:
        name, separator, version = requirement.partition("==")
        if not separator:
            raise SystemExit(f"sync_env: expected an exact `name==version` pin, got {requirement!r}")
        pins[name.strip()] = version.strip()
    return pins


def _installed(name: str) -> metadata.Distribution | None:
    return next(iter(metadata.distributions(name=name, path=SITE_PACKAGES)), None)


def _differences(pins: dict[str, str], project: bool) -> list[str]:
    differences = []
    for name, version in pins.items():
        dist = _installed(name)
        found = dist.version if dist else None
        if found != version:
            differences.append(f"{name}: installed {found}, pinned {version}")
    if project:
        dist = _installed("kindred")
        editable = bool(dist and json.loads(dist.read_text("direct_url.json") or "{}").get("dir_info", {}).get("editable"))
        expected = runpy.run_path(str(REPO_ROOT / "kindred" / "_version.py"))["__version__"]
        if dist is None or dist.version != expected or not editable:
            found = f"{dist.version}{' editable' if editable else ''}" if dist else None
            differences.append(f"kindred: installed {found}, expected {expected} editable")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", action="append", default=[], help="dependency group to sync (repeatable)")
    parser.add_argument("--no-project", action="store_true", help="sync only the groups, not Kindred and its pins")
    args = parser.parse_args()

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = pyproject.get("dependency-groups", {})
    pins = {} if args.no_project else _pins(pyproject["project"]["dependencies"])
    for group in args.group:
        if group not in groups:
            raise SystemExit(f"sync_env: expected one of the dependency groups {sorted(groups)}, got {group!r}")
        pins |= _pins(groups[group])

    differences = _differences(pins, project=not args.no_project)
    if not differences:
        print(f"sync_env: {sys.executable} is up to date")
        return 0
    for difference in differences:
        print(f"sync_env: {difference}")

    flags = (["-E"] if sys.flags.ignore_environment else []) + (["-s"] if sys.flags.no_user_site else [])
    command = [sys.executable, *flags, "-m", "pip", "install"]
    if not args.no_project:
        command += ["-e", "."]
    for group in args.group:
        command += ["--group", group]
    print(f"sync_env: running {' '.join(command)}", flush=True)
    if subprocess.run(command, cwd=REPO_ROOT).returncode != 0:
        return 1
    remaining = _differences(pins, project=not args.no_project)
    for difference in remaining:
        print(f"sync_env: still differs after install: {difference}")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
