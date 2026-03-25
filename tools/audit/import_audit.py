#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModuleInfo:
    module: str
    relpath: str
    is_init: bool


@dataclass(frozen=True)
class ImportOccurrence:
    src_module: str
    src_relpath: str
    lineno: int
    line_text: str
    dst_module: str


def _should_skip_dir(dirname: str) -> bool:
    if dirname in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
        return True
    if dirname == "_audit_reports":
        return True
    return False


def iter_py_files(package_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            files.append(Path(dirpath) / filename)
    files.sort(key=lambda p: p.as_posix())
    return files


def module_name_for_path(*, repo_root: Path, package: str, path: Path) -> tuple[str, bool]:
    rel = path.relative_to(repo_root).with_suffix("")
    parts = list(rel.parts)
    if not parts or parts[0] != package:
        raise ValueError(f"path is not under package {package!r}: {path}")
    is_init = parts[-1] == "__init__"
    if is_init:
        parts = parts[:-1]
    return ".".join(parts), is_init


def build_module_index(*, repo_root: Path, package: str, package_dir: Path) -> dict[str, ModuleInfo]:
    index: dict[str, ModuleInfo] = {}
    for path in iter_py_files(package_dir):
        module, is_init = module_name_for_path(repo_root=repo_root, package=package, path=path)
        relpath = path.relative_to(repo_root).as_posix()
        if module not in index:
            index[module] = ModuleInfo(module=module, relpath=relpath, is_init=is_init)
    return index


def _current_package(src_module: str, src_is_init: bool) -> str:
    if src_is_init:
        return src_module
    parts = src_module.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _resolve_importfrom_base(
    *, src_module: str, src_is_init: bool, level: int, module: str | None
) -> str:
    if level == 0:
        return module or ""
    current_pkg = _current_package(src_module, src_is_init)
    parts = current_pkg.split(".") if current_pkg else []
    if level == 1:
        base_pkg = current_pkg
    else:
        base_pkg = ".".join(parts[: max(0, len(parts) - (level - 1))])
    if module:
        return f"{base_pkg}.{module}" if base_pkg else module
    return base_pkg


def _read_line_text(path: Path, lineno: int) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i == lineno:
                    return line.rstrip("\n")
    except OSError:
        return ""
    return ""


def iter_internal_imports(
    *, repo_root: Path, module_index: dict[str, ModuleInfo], package: str
) -> tuple[list[ImportOccurrence], list[tuple[str, str]]]:
    """
    Returns:
      - occurrences: import edges with file:line evidence when resolvable
      - parse_warnings: list of (relpath, message) for files that could not be parsed
    """
    occurrences: list[ImportOccurrence] = []
    parse_warnings: list[tuple[str, str]] = []

    for src_module, info in sorted(module_index.items(), key=lambda kv: kv[0]):
        src_path = repo_root / info.relpath
        try:
            text = src_path.read_text(encoding="utf-8")
        except OSError as e:
            parse_warnings.append((info.relpath, f"read failed: {e}"))
            continue

        try:
            tree = ast.parse(text, filename=info.relpath)
        except SyntaxError as e:
            msg = f"syntax error: {e.msg} (line {e.lineno})"
            parse_warnings.append((info.relpath, msg))
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == package or name.startswith(f"{package}."):
                        if name in module_index:
                            dst = name
                        else:
                            # Importing a package namespace or a module not in index.
                            continue
                        lineno = getattr(node, "lineno", 0) or 0
                        line_text = _read_line_text(src_path, lineno) if lineno else ""
                        occurrences.append(
                            ImportOccurrence(
                                src_module=src_module,
                                src_relpath=info.relpath,
                                lineno=lineno,
                                line_text=line_text.strip(),
                                dst_module=dst,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_importfrom_base(
                    src_module=src_module,
                    src_is_init=info.is_init,
                    level=node.level or 0,
                    module=node.module,
                )
                if not (base == package or base.startswith(f"{package}.")):
                    continue

                lineno = getattr(node, "lineno", 0) or 0
                line_text = _read_line_text(src_path, lineno) if lineno else ""

                # Prefer the most specific internal module when possible: base.name
                for alias in node.names:
                    if alias.name == "*":
                        candidates = [base]
                    else:
                        candidates = [f"{base}.{alias.name}", base]
                    dst: str | None = None
                    for cand in candidates:
                        if cand in module_index:
                            dst = cand
                            break
                    if dst is None:
                        continue
                    occurrences.append(
                        ImportOccurrence(
                            src_module=src_module,
                            src_relpath=info.relpath,
                            lineno=lineno,
                            line_text=line_text.strip(),
                            dst_module=dst,
                        )
                    )

    occurrences.sort(key=lambda o: (o.src_relpath, o.lineno, o.dst_module))
    return occurrences, parse_warnings


def build_graph(occurrences: Iterable[ImportOccurrence]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for occ in occurrences:
        graph.setdefault(occ.src_module, set()).add(occ.dst_module)
    return graph


def strongly_connected_components(graph: dict[str, set[str]], nodes: list[str]) -> list[list[str]]:
    """
    Tarjan SCC. Returns SCCs as lists of nodes.
    """
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    sys.setrecursionlimit(max(2000, sys.getrecursionlimit()))

    def visit(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in sorted(graph.get(v, set())):
            if w not in indices:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comp.sort()
            sccs.append(comp)

    for v in nodes:
        if v not in indices:
            visit(v)

    # Deterministic ordering: by size desc, then lexical by joined name.
    sccs.sort(key=lambda comp: (-len(comp), ",".join(comp)))
    return sccs


def indegree_counts(nodes: list[str], graph: dict[str, set[str]]) -> dict[str, int]:
    indeg = {n: 0 for n in nodes}
    for src, dsts in graph.items():
        for dst in dsts:
            if dst in indeg:
                indeg[dst] += 1
    return indeg


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit A helper: internal import graph checks for kindred/")
    parser.add_argument("--repo-root", required=True, help="Path to repository root")
    parser.add_argument("--package", default="kindred", help="Top-level package to scan (default: kindred)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    package = args.package
    package_dir = repo_root / package
    if not package_dir.is_dir():
        print(f"A0 ParseError | WARN | - | package dir not found: {package_dir} | Ensure --repo-root points at the repository root", flush=True)
        return 0

    module_index = build_module_index(repo_root=repo_root, package=package, package_dir=package_dir)
    occurrences, parse_warnings = iter_internal_imports(repo_root=repo_root, module_index=module_index, package=package)
    graph = build_graph(occurrences)
    nodes = sorted(module_index.keys())
    indeg = indegree_counts(nodes, graph)

    hard_fail = 0

    # A0 ParseError (WARN): best-effort; does not fail runner yet.
    for relpath, msg in sorted(parse_warnings):
        print(f"A0 ParseError | WARN | {relpath} | {msg} | Fix syntax/read errors so audits can analyze imports")

    # A1 CoreGUIImport (FAIL)
    core_gui: list[ImportOccurrence] = []
    for occ in occurrences:
        if occ.src_module.startswith("kindred.core") and occ.dst_module.startswith("kindred.gui"):
            core_gui.append(occ)
    if core_gui:
        hard_fail = 1
        for occ in core_gui:
            loc = f"{occ.src_relpath}:{occ.lineno}" if occ.lineno else occ.src_relpath
            msg = f"{occ.src_module} imports {occ.dst_module}"
            evidence = occ.line_text or "-"
            print(
                f"A1 CoreGUIImport | FAIL | {loc} | {msg} (evidence: {evidence}) | Move shared logic down or invert dependency"
            )
    else:
        print("A1 CoreGUIImport | INFO | - | No kindred.core.* -> kindred.gui.* imports detected | -")

    # A2 ForbiddenDslFastEqCycle (FAIL)
    dsl = "kindred.core.simulator.dsl"
    fast_eq = "kindred.core.simulator.fast_eq"
    forbidden: list[ImportOccurrence] = []
    for occ in occurrences:
        if (occ.src_module == dsl and occ.dst_module == fast_eq) or (occ.src_module == fast_eq and occ.dst_module == dsl):
            forbidden.append(occ)
    if forbidden:
        hard_fail = 1
        for occ in forbidden:
            loc = f"{occ.src_relpath}:{occ.lineno}" if occ.lineno else occ.src_relpath
            direction = f"{occ.src_module} -> {occ.dst_module}"
            evidence = occ.line_text or "-"
            print(
                f"A2 ForbiddenDslFastEqCycle | FAIL | {loc} | Forbidden import edge {direction} (evidence: {evidence}) | Move shared logic to kindred.core.simulator.common"
            )
    else:
        print("A2 ForbiddenDslFastEqCycle | INFO | - | No dsl<->fast_eq import edges detected | -")

    # A4 ImportCycles (WARN)
    sccs = [c for c in strongly_connected_components(graph, nodes) if len(c) > 1]
    if sccs:
        for comp in sccs:
            print(f"A4 ImportCycles | WARN | - | SCC size={len(comp)}: {', '.join(comp)} | Break cycle by extracting shared lower-level module")
    else:
        print("A4 ImportCycles | INFO | - | No internal import SCCs (size>1) detected | -")

    # A5 OrphanModules (SUSPECT)
    orphan_modules: list[str] = []
    for mod, info in module_index.items():
        if info.is_init:
            continue
        if indeg.get(mod, 0) == 0:
            orphan_modules.append(mod)
    orphan_modules.sort()
    if orphan_modules:
        print(f"A5 OrphanModules | SUSPECT | - | indegree==0 module count={len(orphan_modules)} (may be false positives) | Verify dynamic imports/entrypoints before deleting")
        for mod in orphan_modules:
            relpath = module_index[mod].relpath
            print(f"A5 OrphanModules | SUSPECT | {relpath} | indegree==0: {mod} | -")
    else:
        print("A5 OrphanModules | INFO | - | No non-__init__ indegree==0 modules detected | -")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

