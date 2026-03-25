from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EXCLUDE_DIR_NAMES = {
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
    "_audit_reports",
}


def _should_skip_dirname(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES or name.startswith(".") or name.startswith("_")


@dataclass(frozen=True)
class ModuleTarget:
    module: str
    relpath: str
    kind: str  # "package" | "module"


def iter_module_targets(repo_root: Path, *, include_tools: bool) -> list[ModuleTarget]:
    targets: list[ModuleTarget] = []

    scan_roots = [repo_root / "kindred"]
    if include_tools:
        scan_roots.append(repo_root / "tools")

    for scan_root in scan_roots:
        if not scan_root.exists():
            continue

        for dirpath_str, dirnames, filenames in os.walk(scan_root):
            dirpath = Path(dirpath_str)
            try:
                rel_dir = dirpath.relative_to(repo_root)
            except ValueError:
                continue

            dirnames[:] = sorted(
                [
                    d
                    for d in dirnames
                    if not _should_skip_dirname(d)
                ]
            )

            if "_audit_reports" in rel_dir.parts:
                continue

            filenames_sorted = sorted(filenames)
            if "__init__.py" in filenames_sorted:
                if str(rel_dir) == "kindred":
                    mod = "kindred"
                else:
                    mod = str(rel_dir).replace("\\", "/").replace("/", ".")
                targets.append(
                    ModuleTarget(
                        module=mod,
                        relpath=str(rel_dir).replace("\\", "/") + "/__init__.py",
                        kind="package",
                    )
                )

            for name in filenames_sorted:
                if not name.endswith(".py"):
                    continue
                if name == "__init__.py":
                    continue
                path = dirpath / name
                try:
                    rel = path.relative_to(repo_root)
                except ValueError:
                    continue
                rel_posix = str(rel).replace("\\", "/")
                if "_audit_reports" in rel.parts:
                    continue
                if rel_posix.startswith("tools/audit/"):
                    continue
                mod = rel_posix[:-3].replace("/", ".")
                targets.append(ModuleTarget(module=mod, relpath=rel_posix, kind="module"))

    uniq: dict[str, ModuleTarget] = {}
    for target in targets:
        if target.module not in uniq:
            uniq[target.module] = target
    return sorted(uniq.values(), key=lambda target: (target.module, target.relpath, target.kind))
