#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "node_modules",
    "venv",
    ".venv",
}

SCAN_DIR_RELPATHS = [
    Path("kindred"),
    Path("tools"),
    Path("tests"),
]

PATH_CLASS_PROD = "PROD"
PATH_CLASS_TEST = "TEST"
PATH_CLASS_TOOLS = "TOOLS"
PATH_CLASS_OTHER = "OTHER"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_NORMALIZE_SEP_RE = re.compile(r"[-_.]+")
_DEP_NAME_SPLIT_RE = re.compile(r"[ \t\[\];<>=!~]")


def _normalize_name(name: str) -> str:
    return _NORMALIZE_SEP_RE.sub("-", name.strip().lower())


def _dep_base_name(dep_spec: str) -> str:
    s = dep_spec.strip()
    if not s:
        return ""
    return _DEP_NAME_SPLIT_RE.split(s, maxsplit=1)[0].strip()


def _classify_relpath(relpath: str) -> str:
    rp = relpath.replace("\\", "/")
    if rp.startswith("kindred/"):
        return PATH_CLASS_PROD
    if rp.startswith("tests/"):
        return PATH_CLASS_TEST
    if rp.startswith("tools/"):
        return PATH_CLASS_TOOLS
    return PATH_CLASS_OTHER


def _module_root_to_dist_name(module_root: str) -> str:
    """
    Conservative module -> distribution mapping.
    """
    if module_root == "pytest" or module_root.startswith("_pytest"):
        return "pytest"
    if module_root == "qdarktheme":
        return "pyqtdarktheme-fork"
    return module_root


def _module_root_to_dist_norm(module_root: str) -> str:
    dist_name = _module_root_to_dist_name(module_root).lstrip("_")
    return _normalize_name(dist_name)


def _safe_relpath(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _is_excluded_relpath(rel: Path) -> bool:
    if "_audit_reports" in rel.parts:
        return True
    if len(rel.parts) >= 2 and rel.parts[0] == "tools" and rel.parts[1] == "audit":
        return True
    if any(p in EXCLUDE_DIR_NAMES for p in rel.parts):
        return True
    return False


def _iter_py_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel_scan_root in SCAN_DIR_RELPATHS:
        scan_root = repo_root / rel_scan_root
        if not scan_root.exists():
            continue
        for dirpath_str, dirnames, filenames in os.walk(scan_root):
            dirpath = Path(dirpath_str)
            try:
                rel_dir = dirpath.relative_to(repo_root)
            except ValueError:
                dirnames[:] = []
                continue

            if _is_excluded_relpath(rel_dir):
                dirnames[:] = []
                continue

            next_dirnames: list[str] = []
            for d in dirnames:
                rel_child = rel_dir / d
                if _is_excluded_relpath(rel_child):
                    continue
                if d in EXCLUDE_DIR_NAMES or d == "_audit_reports":
                    continue
                next_dirnames.append(d)
            dirnames[:] = sorted(set(next_dirnames))

            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                p = dirpath / name
                try:
                    rel_file = p.relative_to(repo_root)
                except ValueError:
                    continue
                if _is_excluded_relpath(rel_file):
                    continue
                files.append(p)
    return sorted(set(files), key=lambda p: _safe_relpath(p, root=repo_root))


def _discover_local_roots(repo_root: Path) -> set[str]:
    roots = {"kindred"}
    for child in sorted(repo_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or name.startswith("_"):
            continue
        if name in EXCLUDE_DIR_NAMES or name in {"tools", "tests"}:
            continue
        init_py = child / "__init__.py"
        if init_py.exists():
            roots.add(name)
    return roots


def _stdlib_names() -> tuple[set[str], bool]:
    """
    Return (stdlib_module_names, used_fallback).
    """
    names = getattr(sys, "stdlib_module_names", None)
    if isinstance(names, (set, frozenset)):
        return set(names), False
    fallback = set(getattr(sys, "builtin_module_names", ()))
    fallback |= {"__future__", "typing", "types", "collections", "pathlib", "re"}
    return fallback, True


def _try_parse_pyproject(repo_root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], list[str]]:
    """
    Return (core_deps, optional_deps_by_group, warnings).

    core_deps: norm_name -> base_name
    optional_deps_by_group: group -> dict(norm_name -> base_name)
    """
    warnings: list[str] = []
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        warnings.append("pyproject.toml missing; cannot read declared dependencies")
        return {}, {}, warnings

    try:
        import tomllib as toml_loader  # pyright: ignore[reportMissingImports]  # stdlib in py3.11+
    except Exception:
        try:
            import tomli as toml_loader  # type: ignore[import-not-found]
        except Exception:
            warnings.append("tomllib/tomli not available; cannot parse pyproject.toml dependencies")
            return {}, {}, warnings

    try:
        raw = pyproject.read_bytes()
    except Exception as e:
        warnings.append(f"failed to read pyproject.toml: {e}")
        return {}, {}, warnings

    try:
        data = toml_loader.loads(raw.decode("utf-8"))
    except Exception as e:
        warnings.append(f"failed to parse pyproject.toml: {e}")
        return {}, {}, warnings

    project = data.get("project")
    if not isinstance(project, dict):
        warnings.append("[project] table missing; cannot read declared dependencies")
        return {}, {}, warnings

    core_deps_specs = project.get("dependencies")
    core_deps: dict[str, str] = {}
    if core_deps_specs is None:
        warnings.append("[project].dependencies missing; treating as empty")
    elif not isinstance(core_deps_specs, list):
        warnings.append("[project].dependencies not a list; treating as empty")
    else:
        for s in core_deps_specs:
            if not isinstance(s, str):
                continue
            base = _dep_base_name(s)
            if not base:
                continue
            core_deps[_normalize_name(base)] = base

    optional_by_group: dict[str, dict[str, str]] = {}
    opt = project.get("optional-dependencies")
    if opt is None:
        return core_deps, optional_by_group, warnings
    if not isinstance(opt, dict):
        warnings.append("[project].optional-dependencies not a table; ignoring")
        return core_deps, optional_by_group, warnings
    for group, specs in opt.items():
        if not isinstance(group, str):
            continue
        if not isinstance(specs, list):
            continue
        s_norms: dict[str, str] = {}
        for s in specs:
            if not isinstance(s, str):
                continue
            base = _dep_base_name(s)
            if not base:
                continue
            s_norms[_normalize_name(base)] = base
        if s_norms:
            optional_by_group[group] = {k: s_norms[k] for k in sorted(s_norms.keys())}
    return core_deps, optional_by_group, warnings


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return False
    if isinstance(t, ast.Name):
        return t.id in {"ImportError", "ModuleNotFoundError"}
    if isinstance(t, ast.Tuple):
        for elt in t.elts:
            if isinstance(elt, ast.Name) and elt.id in {"ImportError", "ModuleNotFoundError"}:
                return True
    return False


def _is_type_checking_test(test: ast.AST) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        if isinstance(test.value, ast.Name) and test.value.id == "typing":
            return True
    return False


@dataclass(frozen=True)
class ImportSite:
    module_root: str
    module_full: str
    relpath: str
    lineno: int
    unguarded_module_level: bool


@dataclass(frozen=True)
class ScanResult:
    files_scanned: int
    parse_failures: int
    import_sites: list[ImportSite]


def _scan_imports(*, repo_root: Path, py_files: list[Path]) -> ScanResult:
    local_roots = _discover_local_roots(repo_root)
    stdlib, used_fallback = _stdlib_names()

    import_sites: list[ImportSite] = []
    parse_failures = 0

    def record_site(*, root: str, full: str, relpath: str, lineno: int, unguarded: bool) -> None:
        if not root:
            return
        # Filter to third-party later; record everything except stdlib/local/relative here for evidence grouping.
        import_sites.append(
            ImportSite(
                module_root=root,
                module_full=full,
                relpath=relpath,
                lineno=lineno,
                unguarded_module_level=unguarded,
            )
        )

    def is_local_root(root: str) -> bool:
        if root in local_roots:
            return True
        return False

    def is_stdlib_root(root: str) -> bool:
        return root in stdlib

    for path in py_files:
        relpath = _safe_relpath(path, root=repo_root)
        try:
            with tokenize.open(path) as f:
                src = f.read()
        except Exception:
            parse_failures += 1
            continue

        try:
            tree = ast.parse(src, filename=relpath)
        except Exception:
            parse_failures += 1
            continue

        def walk(node: ast.AST, *, nest: int, in_import_try: bool, in_type_checking: bool) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                # Nested imports are considered guarded for optional dependency checks.
                for child in ast.iter_child_nodes(node):
                    walk(child, nest=nest + 1, in_import_try=in_import_try, in_type_checking=in_type_checking)
                return

            if isinstance(node, ast.Try):
                guarded = any(_handler_catches_import_error(h) for h in node.handlers)
                for child in node.body:
                    walk(child, nest=nest, in_import_try=in_import_try or guarded, in_type_checking=in_type_checking)
                for child in node.handlers:
                    walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking)
                for child in node.orelse:
                    walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking)
                for child in node.finalbody:
                    walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking)
                return

            if isinstance(node, ast.If):
                is_tc = _is_type_checking_test(node.test)
                for child in node.body:
                    walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking or is_tc)
                for child in node.orelse:
                    walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking)
                return

            if isinstance(node, ast.Import):
                unguarded = nest == 0 and not in_import_try and not in_type_checking
                for alias in node.names:
                    full = alias.name
                    root = full.split(".", 1)[0]
                    if is_stdlib_root(root) or is_local_root(root):
                        continue
                    record_site(
                        root=root,
                        full=full,
                        relpath=relpath,
                        lineno=getattr(node, "lineno", 0) or 0,
                        unguarded=unguarded,
                    )
                return

            if isinstance(node, ast.ImportFrom):
                unguarded = nest == 0 and not in_import_try and not in_type_checking
                if getattr(node, "level", 0):
                    return
                mod = node.module or ""
                if not mod:
                    return
                root = mod.split(".", 1)[0]
                if is_stdlib_root(root) or is_local_root(root):
                    return
                record_site(
                    root=root,
                    full=mod,
                    relpath=relpath,
                    lineno=getattr(node, "lineno", 0) or 0,
                    unguarded=unguarded,
                )
                return

            for child in ast.iter_child_nodes(node):
                walk(child, nest=nest, in_import_try=in_import_try, in_type_checking=in_type_checking)

        walk(tree, nest=0, in_import_try=False, in_type_checking=False)

    import_sites = sorted(import_sites, key=lambda s: (s.module_root.lower(), s.relpath, s.lineno, s.module_full))
    return ScanResult(files_scanned=len(py_files), parse_failures=parse_failures, import_sites=import_sites)


def _candidate_dep_matches(*, module_norm: str, declared_norms: list[str]) -> list[str]:
    """
    Best-effort heuristic candidates for reporting under MAPPING_UNCERTAIN.
    """
    module_flat = module_norm.replace("-", "")
    candidates: list[str] = []
    for dep_norm in declared_norms:
        dep_flat = dep_norm.replace("-", "")
        if module_norm and (module_norm in dep_norm or dep_norm in module_norm):
            candidates.append(dep_norm)
        elif module_flat and (module_flat == dep_flat):
            candidates.append(dep_norm)
    return sorted(set(candidates))


def _write_report(*, repo_root: Path, out_file: Path) -> int:
    timestamp_utc = _utc_timestamp()
    stdlib, used_fallback = _stdlib_names()
    local_roots = _discover_local_roots(repo_root)

    core_deps, optional_by_group, dep_warnings = _try_parse_pyproject(repo_root)
    declared_norm_to_base: dict[str, str] = {}
    declared_norm_to_origin: dict[str, str] = {}

    for norm, base in sorted(core_deps.items()):
        declared_norm_to_base[norm] = base
        declared_norm_to_origin[norm] = "core"

    for group, deps in sorted(optional_by_group.items()):
        for norm, base in sorted(deps.items()):
            if norm not in declared_norm_to_base:
                declared_norm_to_base[norm] = base
            if norm in declared_norm_to_origin:
                declared_norm_to_origin[norm] = f"{declared_norm_to_origin[norm]}+optional:{group}"
            else:
                declared_norm_to_origin[norm] = f"optional:{group}"

    all_declared_norms = sorted(declared_norm_to_base.keys())
    core_norms = set(core_deps.keys())
    optional_norms_by_group: dict[str, set[str]] = {g: set(d.keys()) for g, d in optional_by_group.items()}
    optional_test_norms = optional_norms_by_group.get("test", set())
    optional_non_test_norms: set[str] = set()
    for group, norms in optional_norms_by_group.items():
        if group == "test":
            continue
        optional_non_test_norms |= norms

    declared_for_prod = core_norms | optional_non_test_norms
    declared_for_tools = declared_for_prod
    declared_for_test = declared_for_prod | optional_test_norms

    py_files = _iter_py_files(repo_root)
    scan = _scan_imports(repo_root=repo_root, py_files=py_files)

    sites_by_root: dict[str, list[ImportSite]] = {}
    for site in scan.import_sites:
        sites_by_root.setdefault(site.module_root, []).append(site)

    third_party_roots = sorted(sites_by_root.keys(), key=lambda s: s.lower())

    sites_by_class_and_root: dict[str, dict[str, list[ImportSite]]] = {
        PATH_CLASS_PROD: {},
        PATH_CLASS_TEST: {},
        PATH_CLASS_TOOLS: {},
        PATH_CLASS_OTHER: {},
    }
    for site in scan.import_sites:
        cls = _classify_relpath(site.relpath)
        sites_by_class_and_root.setdefault(cls, {}).setdefault(site.module_root, []).append(site)

    declared_sets_by_class: dict[str, set[str]] = {
        PATH_CLASS_PROD: set(declared_for_prod),
        PATH_CLASS_TOOLS: set(declared_for_tools),
        PATH_CLASS_TEST: set(declared_for_test),
        PATH_CLASS_OTHER: set(declared_for_prod),
    }

    undeclared_roots_by_class: dict[str, set[str]] = {
        PATH_CLASS_PROD: set(),
        PATH_CLASS_TEST: set(),
        PATH_CLASS_TOOLS: set(),
        PATH_CLASS_OTHER: set(),
    }
    mapping_uncertain: dict[str, list[str]] = {}

    for cls, roots_to_sites in sites_by_class_and_root.items():
        declared_norms = declared_sets_by_class.get(cls, set(declared_for_prod))
        for root in sorted(roots_to_sites.keys(), key=lambda s: s.lower()):
            dist_norm = _module_root_to_dist_norm(root)
            if dist_norm in declared_norms:
                continue
            undeclared_roots_by_class[cls].add(root)
            candidates = _candidate_dep_matches(module_norm=dist_norm, declared_norms=all_declared_norms)
            if candidates:
                mapping_uncertain.setdefault(root, candidates)

    undeclared_roots_any: set[str] = set()
    for roots in undeclared_roots_by_class.values():
        undeclared_roots_any |= roots

    observed_dep_norms: set[str] = set()
    for root in third_party_roots:
        observed_dep_norms.add(_module_root_to_dist_norm(root))
    unused_declared_norms = [n for n in all_declared_norms if n not in observed_dep_norms]

    optional_only_norms: set[str] = set()
    for deps in optional_by_group.values():
        for norm in deps.keys():
            if norm not in core_norms:
                optional_only_norms.add(norm)

    optional_only_norms_for_guard = optional_only_norms - optional_test_norms

    unguarded_optional_sites: list[ImportSite] = []
    for root in third_party_roots:
        dist_norm = _module_root_to_dist_norm(root)
        if dist_norm not in optional_only_norms_for_guard:
            continue
        for site in sites_by_root.get(root, []):
            cls = _classify_relpath(site.relpath)
            if cls not in {PATH_CLASS_PROD, PATH_CLASS_TOOLS}:
                continue
            if site.unguarded_module_level:
                unguarded_optional_sites.append(site)

    unguarded_optional_sites = sorted(
        unguarded_optional_sites, key=lambda s: (s.module_root.lower(), s.relpath, s.lineno, s.module_full)
    )

    undeclared_prod = sorted(undeclared_roots_by_class[PATH_CLASS_PROD], key=lambda s: s.lower())
    undeclared_test = sorted(undeclared_roots_by_class[PATH_CLASS_TEST], key=lambda s: s.lower())
    undeclared_tools = sorted(undeclared_roots_by_class[PATH_CLASS_TOOLS], key=lambda s: s.lower())
    undeclared_other = sorted(undeclared_roots_by_class[PATH_CLASS_OTHER], key=lambda s: s.lower())

    counts = {
        "files_scanned": scan.files_scanned,
        "parse_failures": scan.parse_failures,
        "third_party_modules": len(third_party_roots),
        "third_party_import_sites": len(scan.import_sites),
        "undeclared_imports": len(sorted(undeclared_roots_any, key=lambda s: s.lower())),
        "undeclared_imports_prod": len(undeclared_prod),
        "undeclared_imports_test": len(undeclared_test),
        "undeclared_imports_tools": len(undeclared_tools),
        "unused_declared_deps": len(unused_declared_norms),
        "unguarded_optional_imports": len(unguarded_optional_sites),
        "pyproject_parse_warnings": len(dep_warnings),
        "stdlib_fallback": 1 if used_fallback else 0,
    }

    lines: list[str] = []
    lines.append("Kindred Audit D: Deps vs imports (report-only)")
    lines.append(f"Timestamp (UTC): {timestamp_utc}")
    lines.append(f"Repo root: {repo_root}")
    lines.append(f"Python: {sys.version.split()[0]}")
    if used_fallback:
        lines.append("Stdlib detection: sys.stdlib_module_names unavailable; using conservative fallback")
    else:
        lines.append("Stdlib detection: sys.stdlib_module_names")
    lines.append(f"Local roots: {', '.join(sorted(local_roots))}")
    lines.append("")
    lines.append(
        "COUNTS|"
        + "|".join(
            f"{k}={counts[k]}"
            for k in [
                "files_scanned",
                "parse_failures",
                "third_party_modules",
                "third_party_import_sites",
                "undeclared_imports",
                "undeclared_imports_prod",
                "undeclared_imports_test",
                "undeclared_imports_tools",
                "unused_declared_deps",
                "unguarded_optional_imports",
                "pyproject_parse_warnings",
                "stdlib_fallback",
            ]
        )
    )
    lines.append("")

    if dep_warnings:
        lines.append("PYPROJECT_WARNINGS:")
        for w in dep_warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("DECLARED_DEPENDENCIES:")
    if core_deps:
        lines.append("core:")
        for norm in sorted(core_deps.keys()):
            lines.append(f"- {core_deps[norm]}")
    else:
        lines.append("core: (none)")
    if optional_by_group:
        lines.append("optional-groups:")
        for group, deps in sorted(optional_by_group.items()):
            names = [deps[n] for n in sorted(deps.keys())]
            joined = ", ".join(names)
            lines.append(f"- {group}: {joined}")
    else:
        lines.append("optional-groups: (none)")
    lines.append("")

    lines.append("SCAN_SCOPE:")
    lines.append("- kindred/**, tools/**, tests/** (Python files only)")
    lines.append("- excludes: _audit_reports/**,**, tools/audit/**, .venv/**, __pycache__/**, build/dist")
    lines.append("")

    lines.append("UNDECLARED_IMPORTS_PROD (WARN):")
    if undeclared_prod:
        for root in undeclared_prod:
            lines.append(f"- {root}")
            for site in sorted(
                [s for s in sites_by_class_and_root[PATH_CLASS_PROD].get(root, [])],
                key=lambda s: (s.relpath, s.lineno, s.module_full),
            ):
                lines.append(f"  - {site.relpath}:{site.lineno} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("UNDECLARED_IMPORTS_TEST (INFO):")
    if undeclared_test:
        for root in undeclared_test:
            lines.append(f"- {root}")
            for site in sorted(
                [s for s in sites_by_class_and_root[PATH_CLASS_TEST].get(root, [])],
                key=lambda s: (s.relpath, s.lineno, s.module_full),
            ):
                lines.append(f"  - {site.relpath}:{site.lineno} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("UNDECLARED_IMPORTS_TOOLS (INFO):")
    if undeclared_tools:
        for root in undeclared_tools:
            lines.append(f"- {root}")
            for site in sorted(
                [s for s in sites_by_class_and_root[PATH_CLASS_TOOLS].get(root, [])],
                key=lambda s: (s.relpath, s.lineno, s.module_full),
            ):
                lines.append(f"  - {site.relpath}:{site.lineno} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    if undeclared_other:
        lines.append("UNDECLARED_IMPORTS_OTHER (INFO):")
        for root in undeclared_other:
            lines.append(f"- {root}")
            for site in sorted(
                [s for s in sites_by_class_and_root[PATH_CLASS_OTHER].get(root, [])],
                key=lambda s: (s.relpath, s.lineno, s.module_full),
            ):
                lines.append(f"  - {site.relpath}:{site.lineno} | {site.module_full}")
        lines.append("")

    lines.append("UNDECLARED_IMPORTS (INFO):")
    if undeclared_roots_any:
        lines.append(
            f"- total={counts['undeclared_imports']} (prod={counts['undeclared_imports_prod']} test={counts['undeclared_imports_test']} tools={counts['undeclared_imports_tools']})"
        )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("OPTIONAL_DEP_IMPORTED_UNGUARDED (WARN):")
    if unguarded_optional_sites:
        for site in unguarded_optional_sites:
            lines.append(f"- {site.module_root} | {site.relpath}:{site.lineno} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("UNUSED_DECLARED_DEPS (WARN):")
    if unused_declared_norms:
        for norm in unused_declared_norms:
            base = declared_norm_to_base.get(norm, norm)
            origin = declared_norm_to_origin.get(norm, "unknown")
            lines.append(f"- {base} ({origin})")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("MAPPING_UNCERTAIN (INFO):")
    if mapping_uncertain:
        for root in sorted(mapping_uncertain.keys(), key=lambda s: s.lower()):
            candidates = mapping_uncertain[root]
            pretty = ", ".join(declared_norm_to_base.get(c, c) for c in candidates)
            lines.append(f"- {root} -> candidates: {pretty}")
            for site in sites_by_root.get(root, []):
                lines.append(f"  - {site.relpath}:{site.lineno} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("THIRD_PARTY_IMPORT_INVENTORY (INFO):")
    if third_party_roots:
        for root in third_party_roots:
            lines.append(f"- {root}")
            for site in sites_by_root.get(root, []):
                tag = "UNGUARDED" if site.unguarded_module_level else "GUARDED"
                lines.append(f"  - {site.relpath}:{site.lineno} | {tag} | {site.module_full}")
    else:
        lines.append("- (none)")
    lines.append("")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit D: compare declared deps vs third-party imports (stdlib-only)")
    parser.add_argument("--root", required=True, help="Repo root (absolute or relative)")
    parser.add_argument("--out", required=True, help="Output file path (D_deps.txt)")
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve()
    out_file = Path(args.out)
    return _write_report(repo_root=repo_root, out_file=out_file)


if __name__ == "__main__":
    raise SystemExit(main())
