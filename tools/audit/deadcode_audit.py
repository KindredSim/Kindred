#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleInfo:
    relpath: str
    module: str
    is_init: bool
    is_main_module: bool
    has_main_guard: bool


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

REFERENCE_SEARCH_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".sh",
    ".bash",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".json",
    ".ui",
    ".qss",
}

REFERENCE_SEARCH_SPECIAL_BASENAMES = {
    "Makefile",
    "Dockerfile",
}

REFERENCE_SEARCH_MAX_FILE_BYTES = 2_000_000
REFERENCE_SEARCH_MAX_HITS_PER_SCRIPT = 5
REFERENCE_SEARCH_SNIPPET_MAX_CHARS = 160
PACKAGE_EVIDENCE_MAX_HITS_PER_CANDIDATE = 20
PACKAGE_EVIDENCE_EXCLUDE_RELPATHS = {
    "PROJECT_DNA.txt",
    "PLUMBING_REPORT.md",
}
PACKAGE_EVIDENCE_BUCKET_ORDER = [
    "IMPORT_EVIDENCE",
    "PY_RUNTIME_REFERENCE",
    "NON_PY_REFERENCE",
    "NO_EVIDENCE",
]
PACKAGE_HIGH_CONFIDENCE_EXCLUDE_PRIVATE_PREFIX = "_"
_IMPORT_SYNTAX_RE = re.compile(r"^\s*(from|import)\s+")

PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE = 10
PACKAGE_EVIDENCE_V25_SHORTLIST_MAX = 20
PACKAGE_EVIDENCE_V25_BUCKET_ORDER = [
    "IMPORT_EVIDENCE",
    "DYNAMIC_EVIDENCE",
    "REEXPORT_ONLY",
    "NO_EVIDENCE",
]
_DYNAMIC_IMPORT_MODULE_RE = re.compile(r"kindred(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

PACKAGE_EVIDENCE_V26_MAX_HITS_PER_CLASS = 10
PACKAGE_EVIDENCE_V26_SHORTLIST_MAX = 20

TEST_ONLY_KEEP_ALLOWLIST_DEFAULT = Path("tools/audit/deadcode_test_only_keep_allowlist.txt")


def _iter_py_files(root: Path, *, scan_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for scan_root in scan_dirs:
        if not scan_root.exists():
            continue
        for dirpath_str, dirnames, filenames in os.walk(scan_root):
            dirpath = Path(dirpath_str)
            dirnames[:] = sorted(
                [
                    d
                    for d in dirnames
                    if d not in EXCLUDE_DIR_NAMES
                    and d != "_audit_reports"
                   
                ]
            )
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                path = dirpath / name
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                if "_audit_reports" in rel.parts:
                    continue
                files.append(path)
    files = sorted(set(files), key=lambda p: str(p.relative_to(root)))
    return files


def _iter_top_level_py(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix == ".py"], key=lambda p: p.name)


def _iter_reference_search_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(repo_root):
        dirpath = Path(dirpath_str)
        dirnames[:] = sorted(
            [
                d
                for d in dirnames
                if d not in EXCLUDE_DIR_NAMES
                and d != "_audit_reports"
               
            ]
        )
        for name in sorted(filenames):
            path = dirpath / name
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                continue
            if "_audit_reports" in rel.parts:
                continue
            if str(rel).replace("\\", "/") == "tools/audit/deadcode_allowlist.txt":
                continue
            if str(rel).replace("\\", "/") == "tools/audit/deadcode_test_only_keep_allowlist.txt":
                continue
            if path.is_symlink():
                continue
            if not path.is_file():
                continue
            if path.suffix and path.suffix.lower() not in REFERENCE_SEARCH_TEXT_SUFFIXES:
                continue
            if not path.suffix and path.name not in REFERENCE_SEARCH_SPECIAL_BASENAMES:
                continue
            try:
                if path.stat().st_size > REFERENCE_SEARCH_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return sorted(set(files), key=lambda p: str(p.relative_to(repo_root)))


def _iter_text_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    try:
        if path.suffix == ".py":
            with tokenize.open(path) as f:
                for i, line in enumerate(f, 1):
                    lines.append((i, line.rstrip("\n")))
        else:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    lines.append((i, line.rstrip("\n")))
    except Exception:
        return []
    return lines


def _script_reference_tokens(script_relpath: str) -> set[str]:
    p = Path(script_relpath)
    tokens = {script_relpath.replace("\\", "/"), p.name}
    if script_relpath.startswith("tools/") and script_relpath.endswith(".py"):
        tokens.add(script_relpath[:-3].replace("/", "."))
    return {t for t in tokens if t}


def _find_script_reference_hits(
    *,
    repo_root: Path,
    scripts: list[str],
    search_files: list[Path],
) -> dict[str, list[tuple[str, int, str]]]:
    """
    Return up to REFERENCE_SEARCH_MAX_HITS_PER_SCRIPT evidence hits for each script.

    Evidence tuple: (relpath:line, token, snippet)
    """
    scripts = sorted(dict.fromkeys([s.replace("\\", "/").lstrip("./") for s in scripts]))
    hits: dict[str, list[tuple[str, int, str]]] = {s: [] for s in scripts}

    token_to_scripts: dict[str, set[str]] = defaultdict(set)
    for s in scripts:
        for tok in _script_reference_tokens(s):
            token_to_scripts[tok].add(s)

    if not token_to_scripts:
        return hits

    tokens = sorted(token_to_scripts.keys(), key=lambda t: (-len(t), t))
    token_re = re.compile("|".join(re.escape(t) for t in tokens))

    seen: dict[str, set[tuple[str, int, str]]] = {s: set() for s in scripts}

    for path in sorted(search_files, key=lambda p: str(p.relative_to(repo_root))):
        try:
            relfile = str(path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            continue
        file_lines = _iter_text_lines(path)
        if not file_lines:
            continue
        for lineno, line in file_lines:
            for m in token_re.finditer(line):
                tok = m.group(0)
                for s in token_to_scripts.get(tok, ()):
                    if relfile == s:
                        continue
                    key = (relfile, lineno, tok)
                    if key in seen[s]:
                        continue
                    if len(hits[s]) >= REFERENCE_SEARCH_MAX_HITS_PER_SCRIPT:
                        continue
                    snippet = line.strip()
                    if len(snippet) > REFERENCE_SEARCH_SNIPPET_MAX_CHARS:
                        snippet = snippet[: REFERENCE_SEARCH_SNIPPET_MAX_CHARS - 3] + "..."
                    hits[s].append((relfile, lineno, f"{tok} | {snippet}"))
                    seen[s].add(key)
    return hits


def _module_reference_tokens(module: str, relpath: str) -> set[str]:
    rel = relpath.replace("\\", "/").lstrip("./")
    tokens = {t for t in (module, rel) if t}
    # Also match "from <parent> import <leaf>" references for submodules, which
    # do not include the full dotted module name (e.g., "from kindred.io import paths").
    if module and "." in module:
        parent, leaf = module.rsplit(".", 1)
        if parent and leaf:
            tokens.add(f"from {parent} import {leaf}")
    return tokens


def _find_module_reference_hits(
    *,
    repo_root: Path,
    candidates: list[tuple[str, str]],
    search_files: list[Path],
) -> dict[str, list[tuple[str, int, str]]]:
    """
    Return up to PACKAGE_EVIDENCE_MAX_HITS_PER_CANDIDATE evidence hits for each candidate module.

    Candidate tuple: (module_dotted_name, repo_relative_path)
    Evidence tuple: (relfile, lineno, token | snippet)
    """
    candidates = [(m, p.replace("\\", "/").lstrip("./")) for (m, p) in candidates if m and p]
    candidates = sorted(dict.fromkeys(candidates), key=lambda t: (t[1], t[0]))
    module_to_relpath = {m: p for (m, p) in candidates}
    modules = [m for (m, _) in candidates]
    hits: dict[str, list[tuple[str, int, str]]] = {m: [] for m in modules}

    token_to_modules: dict[str, set[str]] = defaultdict(set)
    for module, relpath in candidates:
        for tok in _module_reference_tokens(module, relpath):
            token_to_modules[tok].add(module)

    if not token_to_modules:
        return hits

    tokens = sorted(token_to_modules.keys(), key=lambda t: (-len(t), t))
    token_re = re.compile("|".join(re.escape(t) for t in tokens))

    seen: dict[str, set[tuple[str, int, str]]] = {m: set() for m in modules}

    for path in sorted(search_files, key=lambda p: str(p.relative_to(repo_root))):
        try:
            relfile = str(path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            continue
        file_lines = _iter_text_lines(path)
        if not file_lines:
            continue
        for lineno, line in file_lines:
            for m in token_re.finditer(line):
                tok = m.group(0)
                for module in token_to_modules.get(tok, ()):
                    if relfile == module_to_relpath.get(module):
                        continue
                    key = (relfile, lineno, tok)
                    if key in seen[module]:
                        continue
                    if len(hits[module]) >= PACKAGE_EVIDENCE_MAX_HITS_PER_CANDIDATE:
                        continue
                    snippet = line.strip()
                    if len(snippet) > REFERENCE_SEARCH_SNIPPET_MAX_CHARS:
                        snippet = snippet[: REFERENCE_SEARCH_SNIPPET_MAX_CHARS - 3] + "..."
                    hits[module].append((relfile, lineno, f"{tok} | {snippet}"))
                    seen[module].add(key)

    for module in hits:
        hits[module] = sorted(hits[module], key=lambda t: (t[0], t[1], t[2]))
    return hits


def _collect_package_v25_evidence_hits(
    *,
    repo_root: Path,
    module_index: dict[str, ModuleInfo],
    search_py_files: list[Path],
) -> tuple[
    dict[str, list[tuple[str, int, str, str, bool]]],
    dict[str, list[tuple[str, int, str, str, bool]]],
    dict[str, list[tuple[str, int, str, str, bool]]],
]:
    """
    Collect v2.5 package evidence hits by parsing repo-wide Python files in the
    reference-search scope (including tests), without modifying candidate selection.

    Returns (import_stmt_hits, dynamic_import_hits, reexport_hits).
    """
    import_stmt_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)
    dynamic_import_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)
    reexport_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)

    dummy_src = ModuleInfo(relpath="-", module="", is_init=False, is_main_module=False, has_main_guard=False)

    for path in sorted(search_py_files, key=lambda p: str(p.relative_to(repo_root))):
        if path.suffix != ".py":
            continue
        try:
            relfile = str(path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            continue
        try:
            text = _read_text(path)
            tree = ast.parse(text, filename=relfile)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        text_lines = text.splitlines()

        src_info: ModuleInfo = dummy_src
        if relfile.startswith("kindred/"):
            modname, is_init, is_main = _module_name_for_file(repo_root, path)
            src_info = module_index.get(
                modname,
                ModuleInfo(
                    relpath=relfile,
                    module=modname,
                    is_init=is_init,
                    is_main_module=is_main,
                    has_main_guard=False,
                ),
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lineno = int(getattr(node, "lineno", 1) or 1)
                snippet = _line_snippet(text_lines, lineno)
                targets = _import_targets_from_node(node, src=src_info, module_index=module_index)
                for tgt in targets:
                    if tgt not in module_index:
                        continue
                    if not module_index[tgt].relpath.startswith("kindred/"):
                        continue
                    if module_index[tgt].relpath == relfile:
                        continue
                    import_stmt_hits[tgt].append((relfile, lineno, "IMPORT", snippet, src_info.is_init))
                continue

            if isinstance(node, ast.Call):
                if _is_importlib_import_module_call(node) or _is_dunder_import_call(node):
                    if not node.args:
                        continue
                    s = _joinedstr_constant_prefix(node.args[0])
                    if not s:
                        continue
                    lineno = int(getattr(node, "lineno", 1) or 1)
                    snippet = _line_snippet(text_lines, lineno)
                    for m in _DYNAMIC_IMPORT_MODULE_RE.finditer(s):
                        name = m.group(0)
                        if name in module_index and module_index[name].relpath.startswith("kindred/"):
                            if module_index[name].relpath == relfile:
                                continue
                            dynamic_import_hits[name].append(
                                (relfile, lineno, "DYNAMIC", snippet, src_info.is_init)
                            )
                continue

            if src_info.is_init and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                    value = node.value
                else:
                    targets = [node.target]
                    value = node.value
                if value is None:
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                    continue
                lineno = int(getattr(node, "lineno", 1) or 1)
                snippet = _line_snippet(text_lines, lineno)
                for name in _extract_all_strings(value):
                    sub = f"{src_info.module}.{name}"
                    if sub in module_index and module_index[sub].relpath.startswith("kindred/"):
                        if module_index[sub].relpath == relfile:
                            continue
                        reexport_hits[sub].append((relfile, lineno, "__ALL__", snippet, True))

    for mod, lineno, snippet in _pyproject_script_entrypoint_modules(repo_root):
        if mod in module_index and module_index[mod].relpath.startswith("kindred/"):
            dynamic_import_hits[mod].append(("pyproject.toml", lineno, "DYNAMIC", snippet, False))

    return import_stmt_hits, dynamic_import_hits, reexport_hits


def _build_script_inventory_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    scripts: list[str],
    reference_hits: dict[str, list[tuple[str, int, str]]],
) -> str:
    scripts = sorted(dict.fromkeys([s.replace("\\", "/").lstrip("./") for s in scripts]))
    referenced = [s for s in scripts if reference_hits.get(s)]
    unreferenced = [s for s in scripts if not reference_hits.get(s)]

    out: list[str] = []
    out.append("Kindred Audit C: Script inventory (reference-search heuristic, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- scripts_total: {len(scripts)}")
    out.append(f"- scripts_referenced: {len(referenced)}")
    out.append(f"- scripts_unreferenced: {len(unreferenced)}")
    out.append(f"- evidence_hits_capped_at: {REFERENCE_SEARCH_MAX_HITS_PER_SCRIPT}")
    out.append("")
    out.append("Reference search scope:")
    out.append("- Repo-wide text scan (excluding `_audit_reports/` and ).")
    out.append(f"- File types: {', '.join(sorted(REFERENCE_SEARCH_TEXT_SUFFIXES))} (+ {', '.join(sorted(REFERENCE_SEARCH_SPECIAL_BASENAMES))})")
    out.append("")

    out.append("=== Scripts ===")
    if not scripts:
        out.append("- (none)")
        return "\n".join(out) + "\n"

    for s in scripts:
        hits = reference_hits.get(s, [])
        if hits:
            out.append(f"- {s} | REFERENCED | hits={len(hits)}")
            for relfile, lineno, evidence in hits:
                out.append(f"  - {relfile}:{lineno} | {evidence}")
        else:
            out.append(f"- {s} | UNREFERENCED | hits=0")
    out.append("")
    return "\n".join(out) + "\n"


def _build_unreferenced_scripts_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    scripts_total: int,
    candidates: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    out: list[str] = []
    out.append("Kindred Audit C: Unreferenced scripts (reference-search heuristic, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Finding format:")
    out.append("RULE-ID | SEVERITY | file:line | message | suggested-fix")
    out.append("")
    out.append("Counts:")
    out.append(f"- scripts_total: {scripts_total}")
    out.append(f"- raw_candidates: {len(candidates)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist summary ===")
    if allowlist_missing:
        out.append("C5 Allowlist | INFO | - | Allowlist missing; treating as empty | -")
    else:
        out.append("C5 Allowlist | INFO | - | Allowlist loaded | -")
    if allowlist_unmatched:
        for rel in allowlist_unmatched:
            out.append(f"C5 AllowlistEntry | INFO | {rel}:1 | Allowlist entry did not match any scanned .py file | -")
    out.append("")

    out.append("=== Unreferenced script candidates (C6) ===")
    if candidates:
        for rel in sorted(candidates):
            out.append(
                f"C6 UnreferencedScript | SUSPECT | {rel}:1 | No references found in repo search (excluding self) | Allowlist if intentional standalone tool/script"
            )
    else:
        out.append("C6 UnreferencedScript | INFO | - | No unreferenced script candidates found | -")
    out.append("")
    out.append("=== Notes (limitations) ===")
    out.append("- Heuristic signal only; false positives expected (manual invocation, external tooling, packaging entrypoints).")
    out.append("- References are detected via string search for path/basename/module-like tokens in repo text files.")
    return "\n".join(out) + "\n"


def _build_unreferenced_scripts_filtered_report(
    *,
    repo_root: Path,
    raw_report_path: Path,
    filtered_output_path: Path,
    report_class: str,
    scripts_total: int,
    raw_candidates: list[str],
    filtered_candidates: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    out: list[str] = []
    out.append("Kindred Audit C: Unreferenced scripts (filtered by allowlist)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw report: {raw_report_path}")
    out.append(f"Filtered report: {filtered_output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- scripts_total: {scripts_total}")
    out.append(f"- raw_candidates: {len(raw_candidates)}")
    out.append(f"- filtered_candidates: {len(filtered_candidates)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Unmatched allowlist entries ===")
    if allowlist_unmatched:
        for rel in allowlist_unmatched:
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered unreferenced script candidates (C6) ===")
    if filtered_candidates:
        for rel in sorted(filtered_candidates):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")
    return "\n".join(out) + "\n"


def _module_name_for_file(root: Path, path: Path) -> tuple[str, bool, bool]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    is_init = parts[-1] == "__init__.py"
    is_main_module = parts[-1] == "__main__.py"
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if is_init:
        parts = parts[:-1]
    module = ".".join(parts) if parts else rel.stem
    return module, is_init, is_main_module


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        comp = test.comparators[0]
        if isinstance(comp, ast.Constant) and comp.value == "__main__":
            return True
    return False


def _read_text(path: Path) -> str:
    with tokenize.open(path) as f:
        return f.read()


_PYPROJECT_ENTRYPOINTS_HEADER_RE = re.compile(r"^\s*\[project\.(?:scripts|gui-scripts)\]\s*$")
_PYPROJECT_SECTION_RE = re.compile(r"^\s*\[.*\]\s*$")
_PYPROJECT_SCRIPT_LINE_RE = re.compile(
    r'^\s*([A-Za-z0-9_.-]+)\s*=\s*"([A-Za-z0-9_.]+):([A-Za-z0-9_]+)"\s*$'
)


def _pyproject_script_entrypoint_modules(repo_root: Path) -> list[tuple[str, int, str]]:
    """
    Best-effort extraction of GUI/runtime entrypoints from `pyproject.toml`.

    This audit runs stdlib-only, so we avoid TOML parsing and rely on a conservative
    line-based parse that matches common `project.scripts` / `project.gui-scripts`
    formatting:

        [project.scripts]
        kindred = "kindred.cli:main"
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        text = pyproject.read_text(encoding="utf-8")
    except Exception:
        return []

    in_scripts = False
    modules: list[tuple[str, int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _PYPROJECT_ENTRYPOINTS_HEADER_RE.match(line):
            in_scripts = True
            continue
        if in_scripts and _PYPROJECT_SECTION_RE.match(line):
            in_scripts = False
            continue
        if not in_scripts:
            continue
        m = _PYPROJECT_SCRIPT_LINE_RE.match(raw)
        if not m:
            continue
        # script name is m.group(1); currently unused
        module = m.group(2)
        modules.append((module, lineno, raw.strip()))
    return modules


def _line_snippet(text_lines: list[str], lineno: int) -> str:
    if lineno < 1 or lineno > len(text_lines):
        return ""
    snippet = text_lines[lineno - 1].strip()
    if len(snippet) > REFERENCE_SEARCH_SNIPPET_MAX_CHARS:
        snippet = snippet[: REFERENCE_SEARCH_SNIPPET_MAX_CHARS - 3] + "..."
    return snippet


def _joinedstr_constant_prefix(node: ast.AST) -> str | None:
    """
    Best-effort extraction of literal content from an f-string / JoinedStr.
    Only constant string parts are concatenated; formatted values are ignored.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        s = "".join(parts)
        return s if s else None
    return None


def _is_importlib_import_module_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    )


def _is_dunder_import_call(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Name) and func.id == "__import__"


def _extract_all_strings(value: ast.AST) -> list[str]:
    """
    Extract constant string elements from simple list/tuple/set literals.
    """
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
        return out
    return []


def _dedupe_and_cap_hits(
    hits: list[tuple[str, int, str, str, bool]],
    *,
    cap: int,
) -> list[tuple[str, int, str, str, bool]]:
    # key: (relfile, lineno, kind, snippet, src_is_init)
    deduped = sorted(set(hits), key=lambda t: (t[0], t[1], t[2], t[3], int(t[4])))
    return deduped[:cap]


def _source_class_for_relfile(relfile: str) -> str:
    rel = relfile.replace("\\", "/").lstrip("./")
    if rel == "pyproject.toml":
        return "PROD"
    if rel.startswith("kindred/"):
        return "PROD"
    if rel.startswith("tests/"):
        return "TEST"
    if rel.startswith("tools/"):
        if rel.startswith("tools/audit/"):
            return "OTHER"
        return "TOOLS"
    if "/" not in rel and rel.endswith(".py"):
        return "ROOT"
    return "OTHER"


def _hits_by_source_class(
    hits: list[tuple[str, int, str, str, bool]],
) -> dict[str, list[tuple[str, int, str, str, bool]]]:
    by: dict[str, list[tuple[str, int, str, str, bool]]] = {
        "PROD": [],
        "TEST": [],
        "TOOLS": [],
        "ROOT": [],
        "OTHER": [],
    }
    for h in hits:
        by[_source_class_for_relfile(h[0])].append(h)
    for k in by:
        by[k] = _dedupe_and_cap_hits(by[k], cap=PACKAGE_EVIDENCE_V26_MAX_HITS_PER_CLASS)
    return by


def _package_v26_bucket_for_candidate(
    *,
    hits_by_class: dict[str, list[tuple[str, int, str, str, bool]]],
) -> tuple[str, str]:
    """
    v2.6 buckets: classify based on PROD evidence only.

    - PROD_IMPORT_EVIDENCE: PROD contains import evidence from a non-__init__.py module.
    - PROD_DYNAMIC_EVIDENCE: PROD contains dynamic import evidence.
    - PROD_REEXPORT_ONLY: PROD contains only __init__.py re-exports (no other PROD evidence).
    - NO_PROD_EVIDENCE: no PROD evidence.
    """
    prod_hits = hits_by_class.get("PROD", [])
    if not prod_hits:
        return "NO_PROD_EVIDENCE", "no PROD evidence found"

    prod_import_non_init = [h for h in prod_hits if h[2] == "IMPORT" and (not h[4])]
    if prod_import_non_init:
        relfile, lineno, *_ = prod_import_non_init[0]
        return "PROD_IMPORT_EVIDENCE", f"prod import evidence in {relfile}:{lineno}"

    prod_dynamic = [h for h in prod_hits if h[2] == "DYNAMIC"]
    if prod_dynamic:
        relfile, lineno, *_ = prod_dynamic[0]
        return "PROD_DYNAMIC_EVIDENCE", f"prod dynamic import evidence in {relfile}:{lineno}"

    # If all PROD hits originate from __init__.py, treat as re-export-only.
    if all(h[4] for h in prod_hits):
        relfile, lineno, *_ = prod_hits[0]
        return "PROD_REEXPORT_ONLY", f"prod re-export-only evidence in {relfile}:{lineno}"

    relfile, lineno, *_ = prod_hits[0]
    return "PROD_IMPORT_EVIDENCE", f"prod evidence in {relfile}:{lineno}"


def _build_package_prod_evidence_buckets_v26_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    rationale_for: dict[str, str],
    hits_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    bucket_order = [
        "PROD_IMPORT_EVIDENCE",
        "PROD_DYNAMIC_EVIDENCE",
        "PROD_REEXPORT_ONLY",
        "NO_PROD_EVIDENCE",
    ]
    counts = {b: 0 for b in bucket_order}
    for m in candidates_sorted:
        counts[bucket_for[m]] = counts.get(bucket_for[m], 0) + 1

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (production-vs-tests evidence buckets v2.6, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Source classes:")
    out.append("- PROD: kindred/** (excluding tests/, tools/, repo-root scripts)")
    out.append("- TEST: tests/**")
    out.append("- TOOLS: tools/** (excluding tools/audit/**)")
    out.append("- ROOT: repo-root *.py")
    out.append("")
    out.append("Buckets (PROD-only, deterministic precedence):")
    out.append("- PROD_IMPORT_EVIDENCE: PROD contains import evidence from a non-__init__.py module.")
    out.append("- PROD_DYNAMIC_EVIDENCE: PROD contains dynamic import evidence.")
    out.append("- PROD_REEXPORT_ONLY: PROD evidence exists only in __init__.py (imports/__all__).")
    out.append("- NO_PROD_EVIDENCE: zero PROD evidence (may still have TEST/TOOLS/ROOT evidence).")
    out.append("")
    out.append("Counts:")
    out.append(f"- candidates: {len(candidates_sorted)}")
    for b in bucket_order:
        out.append(f"- bucket_{b.lower()}: {counts.get(b, 0)}")
    out.append(f"- evidence_hit_cap_per_class: {PACKAGE_EVIDENCE_V26_MAX_HITS_PER_CLASS}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    for b in bucket_order:
        mods = [m for m in candidates_sorted if bucket_for[m] == b]
        out.append(f"=== {b} (count={len(mods)}) ===")
        if not mods:
            out.append("- (none)")
            out.append("")
            continue
        for m in mods:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
            out.append(f"  - rationale: {rationale_for[m]}")
            by = hits_for.get(m, {"PROD": [], "TEST": [], "TOOLS": [], "ROOT": [], "OTHER": []})
            for cls in ("PROD", "TEST", "TOOLS", "ROOT"):
                out.append(f"  - {cls}:")
                hits = by.get(cls, [])
                if hits:
                    for relfile, lineno, kind, snippet, _src_is_init in hits:
                        out.append(f"    - {relfile}:{lineno} [{kind}] {snippet}")
                else:
                    out.append("    - (none)")
        out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Buckets are derived from PROD evidence only; TEST-only references do not prove production usage.")
    out.append("- Report-only: this does not authorize deletions.")
    return "\n".join(out) + "\n"


def _build_package_prod_evidence_buckets_v26_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    bucket_for: dict[str, str],
    rationale_for: dict[str, str],
    hits_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))
    bucket_order = [
        "PROD_IMPORT_EVIDENCE",
        "PROD_DYNAMIC_EVIDENCE",
        "PROD_REEXPORT_ONLY",
        "NO_PROD_EVIDENCE",
    ]

    def _counts(mods: list[str]) -> dict[str, int]:
        c = {b: 0 for b in bucket_order}
        for m in mods:
            c[bucket_for[m]] = c.get(bucket_for[m], 0) + 1
        return c

    raw_counts = _counts(raw_sorted)
    filtered_counts = _counts(filtered_sorted)

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (production-vs-tests evidence buckets v2.6, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw buckets report: {base_report_path}")
    out.append(f"Filtered buckets report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_candidates: {len(raw_sorted)}")
    for b in bucket_order:
        out.append(f"- raw_bucket_{b.lower()}: {raw_counts.get(b, 0)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_candidates: {len(filtered_sorted)}")
    for b in bucket_order:
        out.append(f"- filtered_bucket_{b.lower()}: {filtered_counts.get(b, 0)}")
    out.append(f"- evidence_hit_cap_per_class: {PACKAGE_EVIDENCE_V26_MAX_HITS_PER_CLASS}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Unmatched allowlist entries ===")
    if allowlist_unmatched:
        for rel in sorted(allowlist_unmatched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered candidates (by bucket) ===")
    for b in bucket_order:
        mods = [m for m in filtered_sorted if bucket_for[m] == b]
        out.append(f"--- {b} (count={len(mods)}) ---")
        if not mods:
            out.append("- (none)")
            continue
        for m in mods:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
            out.append(f"  - rationale: {rationale_for[m]}")
            by = hits_for.get(m, {"PROD": [], "TEST": [], "TOOLS": [], "ROOT": [], "OTHER": []})
            for cls in ("PROD", "TEST", "TOOLS", "ROOT"):
                out.append(f"  - {cls}:")
                hits = by.get(cls, [])
                if hits:
                    for relfile, lineno, kind, snippet, _src_is_init in hits:
                        out.append(f"    - {relfile}:{lineno} [{kind}] {snippet}")
                else:
                    out.append("    - (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; bucket assignment is unchanged.")
    return "\n".join(out) + "\n"


def _build_package_no_prod_evidence_shortlist_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    hits_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    no_prod = [m for m in candidates_sorted if bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")]
    shortlist = no_prod[:PACKAGE_EVIDENCE_V26_SHORTLIST_MAX]

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Definition:")
    out.append("- Candidate is in the package C1 set (import-graph indegree=0 heuristic).")
    out.append("- Candidate has zero PROD import/dynamic evidence; may have TEST/TOOLS/ROOT evidence.")
    out.append(f"- Ranked by repo-relative path; shortlist shows first {PACKAGE_EVIDENCE_V26_SHORTLIST_MAX}.")
    out.append("")
    out.append("Counts:")
    out.append(f"- no_prod_evidence_candidates_total: {len(no_prod)}")
    out.append(f"- shortlist_count: {len(shortlist)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== No-PROD-evidence shortlist (raw) ===")
    if not shortlist:
        out.append("- (none)")
        out.append("")
        return "\n".join(out) + "\n"

    for m in shortlist:
        rel = module_index[m].relpath
        out.append(f"- {rel}  ({m})  [{bucket_for[m]}]")
        by = hits_for.get(m, {})
        for cls in ("TEST", "TOOLS", "ROOT"):
            hits = by.get(cls, [])
            if hits:
                relfile, lineno, kind, snippet, _ = hits[0]
                out.append(f"  - example_{cls}: {relfile}:{lineno} [{kind}] {snippet}")
    out.append("")
    out.append("=== Notes (limitations) ===")
    out.append("- TEST-only evidence is not proof of production usage; treat these as triage candidates.")
    return "\n".join(out) + "\n"


def _build_package_no_prod_evidence_shortlist_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    bucket_for: dict[str, str],
    hits_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))
    raw_no = [m for m in raw_sorted if bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")]
    filtered_no = [m for m in filtered_sorted if bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")]
    filtered_shortlist = filtered_no[:PACKAGE_EVIDENCE_V26_SHORTLIST_MAX]

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw shortlist report: {base_report_path}")
    out.append(f"Filtered shortlist report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_no_prod_evidence_candidates_total: {len(raw_no)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_no_prod_evidence_candidates_total: {len(filtered_no)}")
    out.append(f"- filtered_shortlist_count: {len(filtered_shortlist)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== No-PROD-evidence shortlist (filtered) ===")
    if not filtered_shortlist:
        out.append("- (none)")
        out.append("")
        return "\n".join(out) + "\n"

    for m in filtered_shortlist:
        rel = module_index[m].relpath
        out.append(f"- {rel}  ({m})  [{bucket_for[m]}]")
        by = hits_for.get(m, {})
        for cls in ("TEST", "TOOLS", "ROOT"):
            hits = by.get(cls, [])
            if hits:
                relfile, lineno, kind, snippet, _ = hits[0]
                out.append(f"  - example_{cls}: {relfile}:{lineno} [{kind}] {snippet}")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; shortlist criteria are otherwise unchanged.")
    return "\n".join(out) + "\n"


def _build_package_no_prod_evidence_test_only_keep_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_shortlist_path: Path,
    module_index: dict[str, ModuleInfo],
    shortlist: list[str],
    accepted: list[str],
    remaining: list[str],
    bucket_for: dict[str, str],
    hits_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]],
    keep_allowlist_path: str | None,
    keep_allowlist_missing: bool,
    keep_allowlist_entries: list[str],
    keep_allowlist_unmatched_entries: list[str],
    title: str,
) -> str:
    accepted_set = set(accepted)
    remaining_set = set(remaining)

    out: list[str] = []
    out.append(title)
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Base shortlist (filtered by deadcode_allowlist): {base_shortlist_path}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- shortlist_total: {len(shortlist)}")
    out.append(f"- accepted_test_only_keep: {len(accepted)}")
    out.append(f"- remaining_after_keep: {len(remaining)}")
    out.append(f"- keep_allowlist_path: {keep_allowlist_path or '-'}")
    out.append(f"- keep_allowlist_missing: {1 if keep_allowlist_missing else 0}")
    out.append(f"- keep_allowlist_entries: {len(keep_allowlist_entries)}")
    out.append(f"- keep_allowlist_unmatched_entries: {len(keep_allowlist_unmatched_entries)}")
    out.append("")

    out.append("=== Unmatched keep-allowlist entries ===")
    if keep_allowlist_unmatched_entries:
        for rel in sorted(keep_allowlist_unmatched_entries):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Shortlist items (marked) ===")
    if not shortlist:
        out.append("- (none)")
        out.append("")
        return "\n".join(out) + "\n"

    for m in shortlist:
        rel = module_index[m].relpath
        if m in accepted_set:
            tag = "ACCEPTED_TEST_ONLY_KEEP"
        elif m in remaining_set:
            tag = "REMAINING"
        else:
            tag = "UNCLASSIFIED"
        out.append(f"- {rel}  ({m})  [{bucket_for[m]}]  [{tag}]")
        by = hits_for.get(m, {})
        for cls in ("TEST", "TOOLS", "ROOT"):
            hits = by.get(cls, [])
            if hits:
                relfile, lineno, kind, snippet, _ = hits[0]
                out.append(f"  - example_{cls}: {relfile}:{lineno} [{kind}] {snippet}")
        if tag == "ACCEPTED_TEST_ONLY_KEEP":
            out.append("  - note: accepted as intentionally TEST-only (see keep allowlist entry comments)")
    out.append("")

    out.append("=== Notes ===")
    out.append("- This report does not change the candidate set; it only marks v2.6 shortlist items as accepted vs remaining.")
    out.append("- Status/WARN semantics should be driven by remaining_after_keep for the v2.6 no-production-evidence shortlist.")
    return "\n".join(out) + "\n"


def _package_v25_bucket_for_candidate(
    *,
    module: str,
    relpath: str,
    import_hits: list[tuple[str, int, str, str, bool]],
    dynamic_hits: list[tuple[str, int, str, str, bool]],
    reexport_hits: list[tuple[str, int, str, str, bool]],
) -> tuple[str, str]:
    """
    v2.5 evidence bucketing (package candidates only).

    Buckets are conservative and deterministic:
    - IMPORT_EVIDENCE: imported by a non-__init__.py module (Import/ImportFrom; incl. relative + submodule from-import forms).
    - DYNAMIC_EVIDENCE: referenced by importlib.import_module/__import__ with a literal containing the dotted name.
    - REEXPORT_ONLY: evidence exists only in package __init__.py files (imports or __all__ mentions).
    - NO_EVIDENCE: none of the above.
    """
    import_hits = _dedupe_and_cap_hits(import_hits, cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE)
    dynamic_hits = _dedupe_and_cap_hits(dynamic_hits, cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE)
    reexport_hits = _dedupe_and_cap_hits(reexport_hits, cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE)

    non_init_import = [h for h in import_hits if not h[4]]
    if non_init_import:
        relfile, lineno, _, _, _ = non_init_import[0]
        return "IMPORT_EVIDENCE", f"import evidence in {relfile}:{lineno}"

    if dynamic_hits:
        relfile, lineno, _, _, _ = dynamic_hits[0]
        return "DYNAMIC_EVIDENCE", f"dynamic import evidence in {relfile}:{lineno}"

    init_only = [h for h in import_hits if h[4]] + reexport_hits
    if init_only:
        relfile, lineno, _, _, _ = init_only[0]
        return "REEXPORT_ONLY", f"re-export evidence in {relfile}:{lineno}"

    return "NO_EVIDENCE", "no AST/dynamic/re-export evidence found"


def _build_package_evidence_buckets_v25_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    rationale_for: dict[str, str],
    hits_for: dict[str, list[tuple[str, int, str, str, bool]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    counts = {b: 0 for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER}
    for m in candidates_sorted:
        counts[bucket_for[m]] = counts.get(bucket_for[m], 0) + 1

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (evidence buckets v2.5, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Bucket definitions (deterministic precedence):")
    out.append("- IMPORT_EVIDENCE: imported by a non-__init__.py module (Import/ImportFrom; incl. relative and submodule from-import resolution).")
    out.append("- DYNAMIC_EVIDENCE: importlib.import_module/__import__ with a literal containing the dotted module name.")
    out.append("- REEXPORT_ONLY: evidence exists only in package __init__.py (imports or __all__ mentions); treat as not-safe-to-delete without deeper review.")
    out.append("- NO_EVIDENCE: no evidence found by the v2.5 AST/dynamic/re-export heuristics.")
    out.append("")
    out.append("Counts:")
    out.append(f"- candidates: {len(candidates_sorted)}")
    for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER:
        out.append(f"- bucket_{b.lower()}: {counts.get(b, 0)}")
    out.append(f"- evidence_hit_cap_per_candidate: {PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist summary ===")
    if allowlist_missing:
        out.append("C5 Allowlist | INFO | - | Allowlist missing; treating as empty | -")
    else:
        out.append("C5 Allowlist | INFO | - | Allowlist loaded | -")
    out.append("")

    for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER:
        bucket_modules = [m for m in candidates_sorted if bucket_for[m] == b]
        out.append(f"=== {b} (count={len(bucket_modules)}) ===")
        if not bucket_modules:
            out.append("- (none)")
            out.append("")
            continue
        for m in bucket_modules:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
            out.append(f"  - rationale: {rationale_for[m]}")
            hits = _dedupe_and_cap_hits(hits_for.get(m, []), cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE)
            if hits:
                for relfile, lineno, kind, snippet, _src_is_init in hits:
                    out.append(f"  - {relfile}:{lineno} [{kind}] {snippet}")
            else:
                out.append("  - (none)")
        out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- v2.5 evidence is heuristic and conservative; absence of evidence is not proof of dead code.")
    out.append("- This report does not change candidate selection; it only enriches evidence for the existing C1 candidate set.")
    return "\n".join(out) + "\n"


def _build_package_evidence_buckets_v25_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    bucket_for: dict[str, str],
    rationale_for: dict[str, str],
    hits_for: dict[str, list[tuple[str, int, str, str, bool]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))

    def _counts(mods: list[str]) -> dict[str, int]:
        c = {b: 0 for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER}
        for m in mods:
            c[bucket_for[m]] = c.get(bucket_for[m], 0) + 1
        return c

    raw_counts = _counts(raw_sorted)
    filtered_counts = _counts(filtered_sorted)

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (evidence buckets v2.5, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw buckets report: {base_report_path}")
    out.append(f"Filtered buckets report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_candidates: {len(raw_sorted)}")
    for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER:
        out.append(f"- raw_bucket_{b.lower()}: {raw_counts.get(b, 0)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_candidates: {len(filtered_sorted)}")
    for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER:
        out.append(f"- filtered_bucket_{b.lower()}: {filtered_counts.get(b, 0)}")
    out.append(f"- evidence_hit_cap_per_candidate: {PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Unmatched allowlist entries ===")
    if allowlist_unmatched:
        for rel in sorted(allowlist_unmatched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered candidates (by bucket) ===")
    for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER:
        bucket_modules = [m for m in filtered_sorted if bucket_for[m] == b]
        out.append(f"--- {b} (count={len(bucket_modules)}) ---")
        if not bucket_modules:
            out.append("- (none)")
            continue
        for m in bucket_modules:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
            out.append(f"  - rationale: {rationale_for[m]}")
            hits = _dedupe_and_cap_hits(hits_for.get(m, []), cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE)
            if hits:
                for relfile, lineno, kind, snippet, _src_is_init in hits:
                    out.append(f"  - {relfile}:{lineno} [{kind}] {snippet}")
            else:
                out.append("  - (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; bucket assignment is unchanged.")
    return "\n".join(out) + "\n"


def _build_package_no_evidence_shortlist_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    no_evidence = [m for m in candidates_sorted if bucket_for[m] == "NO_EVIDENCE"]
    shortlist = no_evidence[:PACKAGE_EVIDENCE_V25_SHORTLIST_MAX]

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (NO_EVIDENCE shortlist v2.5, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Definition:")
    out.append("- Candidate is in the package C1 set (import-graph indegree=0 heuristic).")
    out.append("- Candidate is bucketed as NO_EVIDENCE by v2.5 evidence bucketing.")
    out.append(f"- Ranked by repo-relative path; shortlist shows first {PACKAGE_EVIDENCE_V25_SHORTLIST_MAX}.")
    out.append("")
    out.append("Counts:")
    out.append(f"- no_evidence_candidates_total: {len(no_evidence)}")
    out.append(f"- shortlist_count: {len(shortlist)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== NO_EVIDENCE shortlist (raw) ===")
    if shortlist:
        for m in shortlist:
            out.append(f"- {module_index[m].relpath}  ({m})")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- This shortlist is heuristic; verify before deleting (dynamic imports and registries may be missed).")
    return "\n".join(out) + "\n"


def _build_package_no_evidence_shortlist_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    bucket_for: dict[str, str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))
    raw_no = [m for m in raw_sorted if bucket_for[m] == "NO_EVIDENCE"]
    filtered_no = [m for m in filtered_sorted if bucket_for[m] == "NO_EVIDENCE"]
    filtered_shortlist = filtered_no[:PACKAGE_EVIDENCE_V25_SHORTLIST_MAX]

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (NO_EVIDENCE shortlist v2.5, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw shortlist report: {base_report_path}")
    out.append(f"Filtered shortlist report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_no_evidence_candidates_total: {len(raw_no)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_no_evidence_candidates_total: {len(filtered_no)}")
    out.append(f"- filtered_shortlist_count: {len(filtered_shortlist)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== NO_EVIDENCE shortlist (filtered) ===")
    if filtered_shortlist:
        for m in filtered_shortlist:
            out.append(f"- {module_index[m].relpath}  ({m})")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; shortlist criteria are otherwise unchanged.")
    return "\n".join(out) + "\n"


def _normalize_allowlist_path(raw: str) -> str | None:
    s = raw.strip()
    if not s:
        return None
    if s.startswith("./"):
        s = s[2:]
    s = s.replace("\\", "/")
    s = s.strip("/")
    if not s.endswith(".py"):
        return None
    return s


def _load_allowlist(repo_root: Path, allowlist_path: Path | None) -> tuple[set[str], bool, list[str]]:
    if allowlist_path is None:
        return set(), True, []
    path = allowlist_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        return set(), True, []

    entries: set[str] = set()
    raw_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw_lines.append(line)
        left = line.split("#", 1)[0]
        norm = _normalize_allowlist_path(left)
        if norm:
            entries.add(norm)
    return entries, False, sorted(entries)


def _longest_internal_prefix(module_index: dict[str, ModuleInfo], name: str) -> str | None:
    if not name:
        return None
    parts = name.split(".")
    for k in range(len(parts), 0, -1):
        cand = ".".join(parts[:k])
        if cand in module_index:
            return cand
    return None


def _current_package_name(mod: ModuleInfo) -> str:
    if mod.is_init:
        return mod.module
    if "." in mod.module:
        return mod.module.rsplit(".", 1)[0]
    return ""


def _resolve_relative(module: str | None, level: int, *, src: ModuleInfo) -> str | None:
    # Absolute import-from: "from X import Y" (level == 0) should resolve to X.
    if level <= 0:
        return module or None

    # Relative import-from: "from .X import Y" (level > 0) resolves from the
    # current package of the source module.
    pkg = _current_package_name(src)
    base_parts = pkg.split(".") if pkg else []
    if len(base_parts) < (level - 1):
        return None
    base = ".".join(base_parts[: len(base_parts) - (level - 1)])
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def _import_targets_from_node(
    node: ast.AST, *, src: ModuleInfo, module_index: dict[str, ModuleInfo]
) -> set[str]:
    targets: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            resolved = _longest_internal_prefix(module_index, alias.name)
            if resolved:
                targets.add(resolved)
    elif isinstance(node, ast.ImportFrom):
        level = int(getattr(node, "level", 0) or 0)
        base = _resolve_relative(node.module, level, src=src)
        if base:
            resolved_base = _longest_internal_prefix(module_index, base)
            if resolved_base:
                targets.add(resolved_base)
        for alias in node.names:
            if alias.name == "*":
                continue
            if base:
                sub = f"{base}.{alias.name}"
                # Treat "from X import Y" as importing X.Y only when X.Y is a
                # real module/package in the scanned module index. This avoids
                # incorrectly treating attribute imports as module edges.
                if sub in module_index:
                    targets.add(sub)
    return targets


def _build_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str | None,
    scan_files: list[Path],
    module_index: dict[str, ModuleInfo],
    parsed_ok: set[str],
    parse_failures: list[tuple[str, str]],
    entrypoints: list[str],
    init_modules: list[str],
    unreferenced: list[str],
    incoming_sources: dict[str, set[str]],
    edge_count: int,
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    scan_roots = ["kindred/", "tools/", "<repo_root>/*.py"]
    excludes = sorted(
        [
            "_audit_reports/",
                        ".git/",
            ".venv/",
            "venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            ".tox/",
            ".nox/",
            ".ipynb_checkpoints/",
            "build/",
            "dist/",
            "node_modules/",
        ]
    )

    out: list[str] = []
    out.append("Kindred Audit C: Dead-code candidates (heuristic, report-only)")
    out.append(f"Repo root: {repo_root}")
    if report_class:
        out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Finding format:")
    out.append("RULE-ID | SEVERITY | file:line | message | suggested-fix")
    out.append("")
    out.append("Scan scope:")
    for r in scan_roots:
        out.append(f"- {r}")
    out.append("")
    out.append("Excludes:")
    for e in excludes:
        out.append(f"- {e}")
    out.append("")
    out.append("Counts:")
    out.append(f"- python_files_scanned: {len(scan_files)}")
    out.append(f"- modules_indexed: {len(module_index)}")
    out.append(f"- parsed_ok: {len(parsed_ok)}")
    out.append(f"- parse_failures: {len(parse_failures)}")
    out.append(f"- internal_import_edges: {edge_count}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist summary ===")
    if allowlist_missing:
        out.append("C5 Allowlist | INFO | - | Allowlist missing; treating as empty | -")
    else:
        out.append("C5 Allowlist | INFO | - | Allowlist loaded | -")
    if allowlist_unmatched:
        for rel in allowlist_unmatched:
            out.append(f"C5 AllowlistEntry | INFO | {rel}:1 | Allowlist entry did not match any scanned .py file | -")
    out.append("")

    if parse_failures:
        out.append("=== Parse failures (C3) ===")
        for rel, err in sorted(parse_failures):
            out.append(
                f"C3 ParseFailure | WARN | {rel}:1 | AST parse failed; graph may be incomplete ({err}) | Fix SyntaxError/encoding issue"
            )
        out.append("")

    out.append("=== Entrypoints (C2) ===")
    for mod in entrypoints:
        rel = module_index[mod].relpath
        out.append(f"C2 Entrypoint | INFO | {rel}:1 | Entrypoint (main-guard or __main__.py) | -")
    if not entrypoints:
        out.append("C2 Entrypoint | INFO | - | No entrypoints detected in scan scope | -")
    out.append("")

    out.append("=== Package init modules (C4) ===")
    for mod in init_modules:
        rel = module_index[mod].relpath
        out.append(f"C4 PackageInit | INFO | {rel}:1 | Package __init__.py module | -")
    if not init_modules:
        out.append("C4 PackageInit | INFO | - | No __init__.py modules detected in scan scope | -")
    out.append("")

    out.append("=== Unreferenced module candidates (C1) ===")
    if unreferenced:
        for mod in unreferenced:
            rel = module_index[mod].relpath
            out.append(
                f"C1 UnreferencedModule | SUSPECT | {rel}:1 | No inbound imports from scanned set (in-degree=0) | Verify via runtime usage, entrypoints, dynamic imports, or tests before deleting"
            )
    else:
        out.append("C1 UnreferencedModule | INFO | - | No unreferenced module candidates found | -")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Heuristic signal only; false positives expected (dynamic imports, plugins, tests, optional deps).")
    out.append("- Inbound imports are computed only from parsed AST import/from-import nodes within the scan scope.")
    out.append("- TYPE_CHECKING-only imports count as inbound references in this audit.")
    return "\n".join(out) + "\n"


def _build_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    filtered_output_path: Path,
    report_class: str | None,
    module_index: dict[str, ModuleInfo],
    parse_failures: list[tuple[str, str]],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    out: list[str] = []
    out.append("Kindred Audit C: Dead-code candidates (filtered by allowlist)")
    out.append(f"Repo root: {repo_root}")
    if report_class:
        out.append(f"Candidate class: {report_class}")
    out.append(f"Raw report: {output_path}")
    out.append(f"Filtered report: {filtered_output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_candidates: {len(raw_candidates)}")
    out.append(f"- filtered_candidates: {len(filtered_candidates)}")
    out.append(f"- parse_failures: {len(parse_failures)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in allowlist_matched:
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Unmatched allowlist entries ===")
    if allowlist_unmatched:
        for rel in allowlist_unmatched:
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered unreferenced module candidates (C1) ===")
    if filtered_candidates:
        for mod in filtered_candidates:
            out.append(f"- {module_index[mod].relpath}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- This file lists C1 candidates after removing paths present in the allowlist.")
    out.append("- Review before deleting: dynamic imports, entrypoints, optional deps may be missed.")
    return "\n".join(out) + "\n"


def _build_package_usage_evidence_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    reference_hits: dict[str, list[tuple[str, int, str]]],
    search_files: list[Path],
    excluded_relpaths: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    has_evidence = sum(1 for m in candidates_sorted if reference_hits.get(m))
    no_evidence = len(candidates_sorted) - has_evidence

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (usage evidence triage, heuristic, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- candidates: {len(candidates_sorted)}")
    out.append(f"- has_textual_references: {has_evidence}")
    out.append(f"- no_textual_references: {no_evidence}")
    out.append(f"- evidence_hit_cap_per_candidate: {PACKAGE_EVIDENCE_MAX_HITS_PER_CANDIDATE}")
    out.append(f"- reference_search_files: {len(search_files)}")
    out.append(f"- reference_search_suffixes: {', '.join(sorted(REFERENCE_SEARCH_TEXT_SUFFIXES))}")
    out.append(f"- reference_search_excluded_relpaths: {', '.join(excluded_relpaths) if excluded_relpaths else '-'}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Evidence buckets (conservative) ===")
    out.append("- HAS_EVIDENCE: one or more textual references found (not proof of runtime import).")
    out.append("- NO_EVIDENCE: no textual references found in the scanned text files (may still be used).")
    out.append("")

    out.append("=== Candidates with usage evidence ===")
    if not candidates_sorted:
        out.append("- (none)")
    for module in candidates_sorted:
        rel = module_index[module].relpath
        hits = reference_hits.get(module, [])
        bucket = "HAS_EVIDENCE" if hits else "NO_EVIDENCE"
        out.append(f"- {rel}  ({module})  [{bucket}]")
        if hits:
            for relfile, lineno, detail in hits:
                out.append(f"  - {relfile}:{lineno} {detail}")
        else:
            out.append("  - (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Textual references are heuristic only; they can be false positives (docs/examples) or false negatives.")
    out.append("- Scan excludes `_audit_reports/` and  and skips binary/large files.")
    out.append("- Self-references (within the candidate module file itself) are ignored.")
    return "\n".join(out) + "\n"


def _build_package_usage_evidence_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    reference_hits: dict[str, list[tuple[str, int, str]]],
    search_files: list[Path],
    excluded_relpaths: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))
    raw_has = sum(1 for m in raw_sorted if reference_hits.get(m))
    raw_none = len(raw_sorted) - raw_has
    filtered_has = sum(1 for m in filtered_sorted if reference_hits.get(m))
    filtered_none = len(filtered_sorted) - filtered_has

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (usage evidence triage, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw evidence report: {base_report_path}")
    out.append(f"Filtered evidence report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_candidates: {len(raw_sorted)}")
    out.append(f"- raw_has_textual_references: {raw_has}")
    out.append(f"- raw_no_textual_references: {raw_none}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_candidates: {len(filtered_sorted)}")
    out.append(f"- filtered_has_textual_references: {filtered_has}")
    out.append(f"- filtered_no_textual_references: {filtered_none}")
    out.append(f"- evidence_hit_cap_per_candidate: {PACKAGE_EVIDENCE_MAX_HITS_PER_CANDIDATE}")
    out.append(f"- reference_search_files: {len(search_files)}")
    out.append(f"- reference_search_suffixes: {', '.join(sorted(REFERENCE_SEARCH_TEXT_SUFFIXES))}")
    out.append(f"- reference_search_excluded_relpaths: {', '.join(excluded_relpaths) if excluded_relpaths else '-'}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Unmatched allowlist entries ===")
    if allowlist_unmatched:
        for rel in sorted(allowlist_unmatched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered candidates with usage evidence ===")
    if not filtered_sorted:
        out.append("- (none)")
    for module in filtered_sorted:
        rel = module_index[module].relpath
        hits = reference_hits.get(module, [])
        bucket = "HAS_EVIDENCE" if hits else "NO_EVIDENCE"
        out.append(f"- {rel}  ({module})  [{bucket}]")
        if hits:
            for relfile, lineno, detail in hits:
                out.append(f"  - {relfile}:{lineno} {detail}")
        else:
            out.append("  - (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Textual references are heuristic only; they can be false positives (docs/examples) or false negatives.")
    out.append("- Self-references are ignored; a hit means the module/path name appears elsewhere in the repo text.")
    return "\n".join(out) + "\n"


def _split_evidence_detail(detail: str) -> tuple[str, str]:
    if " | " in detail:
        tok, snippet = detail.split(" | ", 1)
        return tok.strip(), snippet.strip()
    return detail.strip(), ""


def _package_evidence_bucket_for_candidate(
    *,
    module: str,
    relpath: str,
    hits: list[tuple[str, int, str]],
) -> tuple[str, str]:
    """
    Deterministically bucket a candidate based on evidence hits collected by textual reference search.

    Returns: (bucket_label, rationale)
    """
    if not hits:
        return "NO_EVIDENCE", "no references found in reference-search scope"

    hits_sorted = sorted(hits, key=lambda t: (t[0], t[1], t[2]))
    py_hits = [(f, n, d) for (f, n, d) in hits_sorted if f.endswith(".py")]
    non_py_hits = [(f, n, d) for (f, n, d) in hits_sorted if not f.endswith(".py")]

    from_tok: str | None = None
    if "." in module:
        parent, leaf = module.rsplit(".", 1)
        if parent and leaf:
            from_tok = f"from {parent} import {leaf}"

    for relfile, lineno, detail in py_hits:
        tok, snippet = _split_evidence_detail(detail)
        if (tok == module or (from_tok and tok == from_tok)) and _IMPORT_SYNTAX_RE.match(snippet):
            return "IMPORT_EVIDENCE", f"import syntax in {relfile}:{lineno}"

    if py_hits:
        relfile, lineno, _ = py_hits[0]
        return "PY_RUNTIME_REFERENCE", f"python reference in {relfile}:{lineno}"

    relfile, lineno, _ = non_py_hits[0]
    return "NON_PY_REFERENCE", f"non-python reference in {relfile}:{lineno}"


def _is_private_package_candidate(*, module: str, relpath: str) -> bool:
    name = Path(relpath).name
    if name.startswith(PACKAGE_HIGH_CONFIDENCE_EXCLUDE_PRIVATE_PREFIX):
        return True
    last = module.rsplit(".", 1)[-1]
    return last.startswith(PACKAGE_HIGH_CONFIDENCE_EXCLUDE_PRIVATE_PREFIX)


def _build_package_evidence_buckets_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    rationale_for: dict[str, str],
    reference_hits: dict[str, list[tuple[str, int, str]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))

    bucket_counts: dict[str, int] = {b: 0 for b in PACKAGE_EVIDENCE_BUCKET_ORDER}
    for m in candidates_sorted:
        b = bucket_for[m]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (evidence buckets, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Bucket definitions (deterministic precedence):")
    out.append("- IMPORT_EVIDENCE: evidence includes a Python import/from statement importing the dotted module name.")
    out.append("- PY_RUNTIME_REFERENCE: evidence exists in .py files but not as import/from syntax.")
    out.append("- NON_PY_REFERENCE: evidence exists only in non-.py files in the reference-search scope.")
    out.append("- NO_EVIDENCE: no evidence hits found in the reference-search scope.")
    out.append("")
    out.append("Counts:")
    out.append(f"- candidates: {len(candidates_sorted)}")
    for b in PACKAGE_EVIDENCE_BUCKET_ORDER:
        out.append(f"- bucket_{b.lower()}: {bucket_counts.get(b, 0)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    for b in PACKAGE_EVIDENCE_BUCKET_ORDER:
        bucket_modules = [m for m in candidates_sorted if bucket_for[m] == b]
        out.append(f"=== {b} (count={len(bucket_modules)}) ===")
        if not bucket_modules:
            out.append("- (none)")
            out.append("")
            continue
        for m in bucket_modules:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})  # {rationale_for[m]}")
            hits = reference_hits.get(m, [])
            if hits:
                relfile, lineno, detail = sorted(hits, key=lambda t: (t[0], t[1], t[2]))[0]
                out.append(f"  - {relfile}:{lineno} {detail}")
            else:
                out.append("  - (none)")
        out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Bucket assignment is based on evidence hits captured (capped) by the reference-search heuristic.")
    out.append("- Buckets indicate textual references, not definitive runtime imports.")
    return "\n".join(out) + "\n"


def _build_package_evidence_buckets_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_candidates: list[str],
    filtered_candidates: list[str],
    bucket_for: dict[str, str],
    reference_hits: dict[str, list[tuple[str, int, str]]],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_candidates, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_candidates, key=lambda m: (module_index[m].relpath, m))

    def _count(mods: list[str]) -> dict[str, int]:
        d: dict[str, int] = {b: 0 for b in PACKAGE_EVIDENCE_BUCKET_ORDER}
        for m in mods:
            d[bucket_for[m]] = d.get(bucket_for[m], 0) + 1
        return d

    raw_counts = _count(raw_sorted)
    filtered_counts = _count(filtered_sorted)

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (evidence buckets, filtered by allowlist)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw buckets report: {base_report_path}")
    out.append(f"Filtered buckets report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_candidates: {len(raw_sorted)}")
    for b in PACKAGE_EVIDENCE_BUCKET_ORDER:
        out.append(f"- raw_bucket_{b.lower()}: {raw_counts.get(b, 0)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_candidates: {len(filtered_sorted)}")
    for b in PACKAGE_EVIDENCE_BUCKET_ORDER:
        out.append(f"- filtered_bucket_{b.lower()}: {filtered_counts.get(b, 0)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Filtered candidates (by bucket) ===")
    for b in PACKAGE_EVIDENCE_BUCKET_ORDER:
        bucket_modules = [m for m in filtered_sorted if bucket_for[m] == b]
        out.append(f"--- {b} (count={len(bucket_modules)}) ---")
        if not bucket_modules:
            out.append("- (none)")
            continue
        for m in bucket_modules:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
            hits = reference_hits.get(m, [])
            if hits:
                relfile, lineno, detail = sorted(hits, key=lambda t: (t[0], t[1], t[2]))[0]
                out.append(f"  - {relfile}:{lineno} {detail}")
            else:
                out.append("  - (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; bucket assignment is unchanged.")
    return "\n".join(out) + "\n"


def _build_package_high_confidence_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    module_index: dict[str, ModuleInfo],
    candidates: list[str],
    bucket_for: dict[str, str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    candidates_sorted = sorted(candidates, key=lambda m: (module_index[m].relpath, m))
    shortlist = [
        m
        for m in candidates_sorted
        if bucket_for[m] == "NO_EVIDENCE"
        and not _is_private_package_candidate(module=m, relpath=module_index[m].relpath)
    ]

    out: list[str] = []
    out.append("Kindred Audit C: Package dead-code candidates (high-confidence shortlist, report-only)")
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Output: {output_path}")
    out.append("")
    out.append("Definition (strict, conservative):")
    out.append("- Candidate is in the package C1 set (import-graph indegree=0 heuristic).")
    out.append("- Bucket is NO_EVIDENCE in the evidence bucketing report.")
    out.append(f"- Excludes private/underscore modules (basename or module segment starts with `{PACKAGE_HIGH_CONFIDENCE_EXCLUDE_PRIVATE_PREFIX}`).")
    out.append("")
    out.append("Counts:")
    out.append(f"- high_confidence_candidates: {len(shortlist)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== High-confidence candidates (raw) ===")
    if shortlist:
        for m in shortlist:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- This shortlist is heuristic; verify before deleting.")
    out.append("- The filtered counterpart removes allowlisted paths.")
    return "\n".join(out) + "\n"


def _build_package_high_confidence_filtered_report(
    *,
    repo_root: Path,
    output_path: Path,
    report_class: str,
    base_report_path: Path,
    module_index: dict[str, ModuleInfo],
    raw_shortlist: list[str],
    filtered_shortlist: list[str],
    allowlist_path: str | None,
    allowlist_missing: bool,
    allowlist_entries: list[str],
    allowlist_matched: list[str],
    allowlist_unmatched: list[str],
) -> str:
    raw_sorted = sorted(raw_shortlist, key=lambda m: (module_index[m].relpath, m))
    filtered_sorted = sorted(filtered_shortlist, key=lambda m: (module_index[m].relpath, m))

    out: list[str] = []
    out.append(
        "Kindred Audit C: Package dead-code candidates (high-confidence shortlist, filtered by allowlist)"
    )
    out.append(f"Repo root: {repo_root}")
    out.append(f"Candidate class: {report_class}")
    out.append(f"Raw report: {base_report_path}")
    out.append(f"Filtered report: {output_path}")
    out.append("")
    out.append("Counts:")
    out.append(f"- raw_high_confidence_candidates: {len(raw_sorted)}")
    out.append(f"- allowlist_matched_candidates: {len(allowlist_matched)}")
    out.append(f"- filtered_high_confidence_candidates: {len(filtered_sorted)}")
    out.append(f"- allowlist_path: {allowlist_path or '-'}")
    out.append(f"- allowlist_missing: {1 if allowlist_missing else 0}")
    out.append(f"- allowlist_entries: {len(allowlist_entries)}")
    out.append(f"- allowlist_unmatched_entries: {len(allowlist_unmatched)}")
    out.append("")

    out.append("=== Allowlist matched candidates ===")
    if allowlist_matched:
        for rel in sorted(allowlist_matched):
            out.append(f"- {rel}")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== High-confidence candidates (filtered) ===")
    if filtered_sorted:
        for m in filtered_sorted:
            rel = module_index[m].relpath
            out.append(f"- {rel}  ({m})")
    else:
        out.append("- (none)")
    out.append("")

    out.append("=== Notes (limitations) ===")
    out.append("- Filtered output removes allowlisted paths; shortlist criteria are otherwise unchanged.")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit C: dead-code candidates (report-only, stdlib-only).")
    ap.add_argument("--root", "--repo-root", dest="root", required=True, help="Repo root directory.")
    out_group = ap.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--report-dir",
        help="Report directory for Audit C v2 outputs (writes per-class reports and legacy aliases).",
    )
    out_group.add_argument("--output", help="Legacy raw output report path (C_deadcode.txt).")
    ap.add_argument("--filtered-output", help="Legacy filtered output report path (C_deadcode_filtered.txt).")
    ap.add_argument("--allowlist", help="Allowlist file path (repo-relative or absolute).")
    args = ap.parse_args(argv)

    repo_root = Path(args.root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else None
    output_path = Path(args.output).resolve() if args.output else None
    filtered_output_path = Path(args.filtered_output).resolve() if args.filtered_output else None
    allowlist_path = Path(args.allowlist) if args.allowlist else None

    scan_dirs = [repo_root / "kindred", repo_root / "tools"]
    scan_files = _iter_py_files(repo_root, scan_dirs=scan_dirs) + _iter_top_level_py(repo_root)
    scan_files = sorted(set(scan_files), key=lambda p: str(p.relative_to(repo_root)))
    scan_files = [
        p for p in scan_files if not str(p.relative_to(repo_root)).startswith("tools/audit/")
    ]

    allowlist_set, allowlist_missing, allowlist_entries = _load_allowlist(repo_root, allowlist_path)
    keep_allowlist_set, keep_allowlist_missing, keep_allowlist_entries = _load_allowlist(
        repo_root, TEST_ONLY_KEEP_ALLOWLIST_DEFAULT
    )

    module_index: dict[str, ModuleInfo] = {}
    for path in scan_files:
        rel = str(path.relative_to(repo_root))
        module, is_init, is_main = _module_name_for_file(repo_root, path)
        module_index[module] = ModuleInfo(
            relpath=rel,
            module=module,
            is_init=is_init,
            is_main_module=is_main,
            has_main_guard=False,
        )

    incoming_sources: dict[str, set[str]] = defaultdict(set)
    edge_count = 0
    parse_failures: list[tuple[str, str]] = []
    parsed_ok: set[str] = set()
    entrypoints: set[str] = set()

    # v2.5 package evidence inputs (does not affect candidate selection).
    # hit tuple: (relfile, lineno, kind, snippet, src_is_init)
    import_stmt_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)
    dynamic_import_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)
    reexport_hits: dict[str, list[tuple[str, int, str, str, bool]]] = defaultdict(list)

    for module in sorted(module_index):
        info = module_index[module]
        path = repo_root / info.relpath
        try:
            text = _read_text(path)
            tree = ast.parse(text, filename=info.relpath)
        except Exception as exc:
            parse_failures.append((info.relpath, f"{type(exc).__name__}: {exc}"))
            continue

        parsed_ok.add(module)
        text_lines = text.splitlines()
        has_guard = info.is_main_module or _has_main_guard(tree)
        if has_guard:
            entrypoints.add(module)
        module_index[module] = ModuleInfo(
            relpath=info.relpath,
            module=info.module,
            is_init=info.is_init,
            is_main_module=info.is_main_module,
            has_main_guard=has_guard,
        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                snippet = _line_snippet(text_lines, int(getattr(node, "lineno", 1) or 1))
                for tgt in _import_targets_from_node(
                    node, src=module_index[module], module_index=module_index
                ):
                    if tgt == module:
                        continue
                    incoming_sources[tgt].add(module)
                    edge_count += 1
                    import_stmt_hits[tgt].append(
                        (info.relpath, int(getattr(node, "lineno", 1) or 1), "IMPORT", snippet, info.is_init)
                    )
                continue

            if isinstance(node, ast.Call):
                if _is_importlib_import_module_call(node) or _is_dunder_import_call(node):
                    if not node.args:
                        continue
                    s = _joinedstr_constant_prefix(node.args[0])
                    if not s:
                        continue
                    snippet = _line_snippet(text_lines, int(getattr(node, "lineno", 1) or 1))
                    for m in _DYNAMIC_IMPORT_MODULE_RE.finditer(s):
                        name = m.group(0)
                        if name in module_index and module_index[name].relpath.startswith("kindred/"):
                            dynamic_import_hits[name].append(
                                (
                                    info.relpath,
                                    int(getattr(node, "lineno", 1) or 1),
                                    "DYNAMIC",
                                    snippet,
                                    info.is_init,
                                )
                            )
                continue

            if info.is_init and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                    value = node.value
                else:
                    targets = [node.target]
                    value = node.value
                if value is None:
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                    continue
                for name in _extract_all_strings(value):
                    sub = f"{module}.{name}"
                    if sub in module_index and module_index[sub].relpath.startswith("kindred/"):
                        snippet = _line_snippet(text_lines, int(getattr(node, "lineno", 1) or 1))
                        reexport_hits[sub].append(
                            (info.relpath, int(getattr(node, "lineno", 1) or 1), "__ALL__", snippet, True)
                        )

    for mod, lineno, snippet in _pyproject_script_entrypoint_modules(repo_root):
        if mod in module_index and module_index[mod].relpath.startswith("kindred/"):
            entrypoints.add(mod)

    init_modules = sorted([m for m, inf in module_index.items() if inf.is_init])
    entrypoints_sorted = sorted(entrypoints)
    unreferenced: list[str] = []
    for mod, inf in sorted(module_index.items(), key=lambda kv: kv[0]):
        if mod not in parsed_ok:
            continue
        if inf.is_init:
            continue
        if inf.has_main_guard:
            continue
        if inf.relpath.startswith("tools/audit/"):
            continue
        if len(incoming_sources.get(mod, set())) == 0:
            unreferenced.append(mod)

    scanned_relpaths = {inf.relpath for inf in module_index.values()}
    allowlist_unmatched = sorted([p for p in allowlist_entries if p not in scanned_relpaths])

    def _candidate_class_for_relpath(rel: str) -> str:
        if rel.startswith("kindred/"):
            return "package"
        if rel.startswith("tools/"):
            return "tools"
        if "/" not in rel and rel.endswith(".py"):
            return "root"
        return "other"

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "package": {
                "raw": report_dir / "C_deadcode.txt",
                "filtered": report_dir / "C_deadcode_filtered.txt",
                "raw_alias": report_dir / "C_deadcode_package.txt",
                "filtered_alias": report_dir / "C_deadcode_package_filtered.txt",
                "label": "package (kindred/**)",
            },
            "tools": {
                "raw": report_dir / "C_deadcode_tools.txt",
                "filtered": report_dir / "C_deadcode_tools_filtered.txt",
                "label": "tools scripts (tools/**, excluding tools/audit/**)",
                "inventory": report_dir / "C_deadcode_tools_inventory.txt",
            },
            "root": {
                "raw": report_dir / "C_deadcode_root.txt",
                "filtered": report_dir / "C_deadcode_root_filtered.txt",
                "label": "repo-root scripts (<repo_root>/*.py)",
                "inventory": report_dir / "C_deadcode_root_inventory.txt",
            },
        }

        # Partition global scan results by candidate class.
        files_by_class: dict[str, list[Path]] = {"package": [], "tools": [], "root": []}
        module_by_class: dict[str, dict[str, ModuleInfo]] = {"package": {}, "tools": {}, "root": {}}
        parsed_ok_by_class: dict[str, set[str]] = {"package": set(), "tools": set(), "root": set()}
        parse_failures_by_class: dict[str, list[tuple[str, str]]] = {"package": [], "tools": [], "root": []}

        for p in scan_files:
            rel = str(p.relative_to(repo_root))
            cls = _candidate_class_for_relpath(rel)
            if cls in files_by_class:
                files_by_class[cls].append(p)

        for mod, info in module_index.items():
            cls = _candidate_class_for_relpath(info.relpath)
            if cls in module_by_class:
                module_by_class[cls][mod] = info

        for mod in parsed_ok:
            info = module_index[mod]
            cls = _candidate_class_for_relpath(info.relpath)
            if cls in parsed_ok_by_class:
                parsed_ok_by_class[cls].add(mod)

        for rel, err in parse_failures:
            cls = _candidate_class_for_relpath(rel)
            if cls in parse_failures_by_class:
                parse_failures_by_class[cls].append((rel, err))

        init_by_class: dict[str, list[str]] = {"package": [], "tools": [], "root": []}
        for mod in init_modules:
            cls = _candidate_class_for_relpath(module_index[mod].relpath)
            if cls in init_by_class:
                init_by_class[cls].append(mod)
        for cls in init_by_class:
            init_by_class[cls] = sorted(init_by_class[cls])

        entrypoints_by_class: dict[str, list[str]] = {"package": [], "tools": [], "root": []}
        for mod in entrypoints_sorted:
            cls = _candidate_class_for_relpath(module_index[mod].relpath)
            if cls in entrypoints_by_class:
                entrypoints_by_class[cls].append(mod)
        for cls in entrypoints_by_class:
            entrypoints_by_class[cls] = sorted(entrypoints_by_class[cls])

        unreferenced_by_class: dict[str, list[str]] = {"package": [], "tools": [], "root": []}
        for mod in unreferenced:
            cls = _candidate_class_for_relpath(module_index[mod].relpath)
            if cls in unreferenced_by_class:
                unreferenced_by_class[cls].append(mod)

        allowlisted_by_class: dict[str, list[str]] = {"package": [], "tools": [], "root": []}
        filtered_by_class: dict[str, list[str]] = {"package": [], "tools": [], "root": []}
        for cls, mods in unreferenced_by_class.items():
            for mod in mods:
                rel = module_index[mod].relpath
                if rel in allowlist_set:
                    allowlisted_by_class[cls].append(rel)
                else:
                    filtered_by_class[cls].append(mod)
            allowlisted_by_class[cls] = sorted(set(allowlisted_by_class[cls]))

        # Package class: keep the existing import-graph-based heuristic and report formats.
        pkg_cfg = outputs["package"]
        pkg_report_text = _build_report(
            repo_root=repo_root,
            output_path=pkg_cfg["raw"],
            report_class=pkg_cfg["label"],
            scan_files=sorted(files_by_class["package"], key=lambda p: str(p.relative_to(repo_root))),
            module_index=module_by_class["package"],
            parsed_ok=parsed_ok_by_class["package"],
            parse_failures=sorted(parse_failures_by_class["package"]),
            entrypoints=entrypoints_by_class["package"],
            init_modules=init_by_class["package"],
            unreferenced=unreferenced_by_class["package"],
            incoming_sources=incoming_sources,
            edge_count=edge_count,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_cfg["raw"].write_text(pkg_report_text, encoding="utf-8")

        pkg_filtered_text = _build_filtered_report(
            repo_root=repo_root,
            output_path=pkg_cfg["raw"],
            filtered_output_path=pkg_cfg["filtered"],
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            parse_failures=sorted(parse_failures_by_class["package"]),
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_cfg["filtered"].write_text(pkg_filtered_text, encoding="utf-8")
        pkg_cfg["raw_alias"].write_text(pkg_report_text, encoding="utf-8")
        pkg_cfg["filtered_alias"].write_text(pkg_filtered_text, encoding="utf-8")

        search_files = _iter_reference_search_files(repo_root)
        pkg_excluded_relpaths = sorted(PACKAGE_EVIDENCE_EXCLUDE_RELPATHS)
        pkg_search_files: list[Path] = []
        for p in search_files:
            try:
                rel = str(p.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                continue
            if rel in PACKAGE_EVIDENCE_EXCLUDE_RELPATHS:
                continue
            pkg_search_files.append(p)

        # Package class: usage-evidence triage (textual reference search), without changing candidate selection.
        pkg_evidence_raw_path = report_dir / "C_deadcode_package_evidence.txt"
        pkg_evidence_filtered_path = report_dir / "C_deadcode_package_evidence_filtered.txt"
        pkg_candidate_tuples = [
            (m, module_by_class["package"][m].relpath) for m in unreferenced_by_class["package"]
        ]
        pkg_reference_hits = _find_module_reference_hits(
            repo_root=repo_root,
            candidates=pkg_candidate_tuples,
            search_files=pkg_search_files,
        )

        pkg_evidence_raw_text = _build_package_usage_evidence_report(
            repo_root=repo_root,
            output_path=pkg_evidence_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            reference_hits=pkg_reference_hits,
            search_files=pkg_search_files,
            excluded_relpaths=pkg_excluded_relpaths,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_evidence_raw_path.write_text(pkg_evidence_raw_text, encoding="utf-8")

        pkg_evidence_filtered_text = _build_package_usage_evidence_filtered_report(
            repo_root=repo_root,
            output_path=pkg_evidence_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_evidence_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            reference_hits=pkg_reference_hits,
            search_files=pkg_search_files,
            excluded_relpaths=pkg_excluded_relpaths,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_evidence_filtered_path.write_text(pkg_evidence_filtered_text, encoding="utf-8")

        # Package class: deterministic evidence bucketing + high-confidence shortlist (report-only).
        pkg_bucket_raw_path = report_dir / "C_deadcode_package_evidence_buckets.txt"
        pkg_bucket_filtered_path = report_dir / "C_deadcode_package_evidence_buckets_filtered.txt"
        pkg_high_raw_path = report_dir / "C_deadcode_package_high_confidence.txt"
        pkg_high_filtered_path = report_dir / "C_deadcode_package_high_confidence_filtered.txt"

        pkg_bucket_for: dict[str, str] = {}
        pkg_rationale_for: dict[str, str] = {}
        for m in unreferenced_by_class["package"]:
            rel = module_by_class["package"][m].relpath
            b, why = _package_evidence_bucket_for_candidate(
                module=m, relpath=rel, hits=pkg_reference_hits.get(m, [])
            )
            pkg_bucket_for[m] = b
            pkg_rationale_for[m] = why

        def _bucket_counts(mods: list[str]) -> dict[str, int]:
            counts = {b: 0 for b in PACKAGE_EVIDENCE_BUCKET_ORDER}
            for m in mods:
                counts[pkg_bucket_for[m]] = counts.get(pkg_bucket_for[m], 0) + 1
            return counts

        pkg_bucket_counts_raw = _bucket_counts(unreferenced_by_class["package"])
        pkg_bucket_counts_filtered = _bucket_counts(filtered_by_class["package"])

        pkg_bucket_raw_text = _build_package_evidence_buckets_report(
            repo_root=repo_root,
            output_path=pkg_bucket_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_bucket_for,
            rationale_for=pkg_rationale_for,
            reference_hits=pkg_reference_hits,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_bucket_raw_path.write_text(pkg_bucket_raw_text, encoding="utf-8")

        pkg_bucket_filtered_text = _build_package_evidence_buckets_filtered_report(
            repo_root=repo_root,
            output_path=pkg_bucket_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_bucket_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            bucket_for=pkg_bucket_for,
            reference_hits=pkg_reference_hits,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_bucket_filtered_path.write_text(pkg_bucket_filtered_text, encoding="utf-8")

        pkg_high_raw = [
            m
            for m in sorted(unreferenced_by_class["package"], key=lambda x: module_by_class["package"][x].relpath)
            if pkg_bucket_for[m] == "NO_EVIDENCE"
            and not _is_private_package_candidate(module=m, relpath=module_by_class["package"][m].relpath)
        ]
        pkg_high_allowlisted = sorted(
            {
                module_by_class["package"][m].relpath
                for m in pkg_high_raw
                if module_by_class["package"][m].relpath in allowlist_set
            }
        )
        pkg_high_filtered = [
            m for m in pkg_high_raw if module_by_class["package"][m].relpath not in allowlist_set
        ]

        pkg_high_raw_text = _build_package_high_confidence_report(
            repo_root=repo_root,
            output_path=pkg_high_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_bucket_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=pkg_high_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_high_raw_path.write_text(pkg_high_raw_text, encoding="utf-8")

        pkg_high_filtered_text = _build_package_high_confidence_filtered_report(
            repo_root=repo_root,
            output_path=pkg_high_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_high_raw_path,
            module_index=module_by_class["package"],
            raw_shortlist=pkg_high_raw,
            filtered_shortlist=pkg_high_filtered,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=pkg_high_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_high_filtered_path.write_text(pkg_high_filtered_text, encoding="utf-8")

        # Package class: v2.5 strengthened evidence detection (AST imports + dynamic imports + re-export-only).
        pkg_v25_bucket_raw_path = report_dir / "C_deadcode_package_evidence_buckets_v2_5.txt"
        pkg_v25_bucket_filtered_path = report_dir / "C_deadcode_package_evidence_buckets_v2_5_filtered.txt"
        pkg_v25_short_raw_path = report_dir / "C_deadcode_package_no_evidence_shortlist.txt"
        pkg_v25_short_filtered_path = report_dir / "C_deadcode_package_no_evidence_shortlist_filtered.txt"

        pkg_v25_import_stmt_hits, pkg_v25_dynamic_import_hits, pkg_v25_reexport_hits = (
            _collect_package_v25_evidence_hits(
                repo_root=repo_root,
                module_index=module_index,
                search_py_files=pkg_search_files,
            )
        )

        pkg_v25_bucket_for: dict[str, str] = {}
        pkg_v25_rationale_for: dict[str, str] = {}
        pkg_v25_hits_for: dict[str, list[tuple[str, int, str, str, bool]]] = {}

        for m in unreferenced_by_class["package"]:
            relpath = module_by_class["package"][m].relpath
            import_hits = [h for h in pkg_v25_import_stmt_hits.get(m, []) if h[0] != relpath]
            dyn_hits = [h for h in pkg_v25_dynamic_import_hits.get(m, []) if h[0] != relpath]
            rex_hits = [h for h in pkg_v25_reexport_hits.get(m, []) if h[0] != relpath]

            bucket, why = _package_v25_bucket_for_candidate(
                module=m, relpath=relpath, import_hits=import_hits, dynamic_hits=dyn_hits, reexport_hits=rex_hits
            )
            pkg_v25_bucket_for[m] = bucket
            pkg_v25_rationale_for[m] = why
            pkg_v25_hits_for[m] = _dedupe_and_cap_hits(
                import_hits + dyn_hits + rex_hits, cap=PACKAGE_EVIDENCE_V25_MAX_HITS_PER_CANDIDATE
            )

        def _bucket_counts_v25(mods: list[str]) -> dict[str, int]:
            counts = {b: 0 for b in PACKAGE_EVIDENCE_V25_BUCKET_ORDER}
            for m in mods:
                counts[pkg_v25_bucket_for[m]] = counts.get(pkg_v25_bucket_for[m], 0) + 1
            return counts

        pkg_v25_bucket_counts_raw = _bucket_counts_v25(unreferenced_by_class["package"])
        pkg_v25_bucket_counts_filtered = _bucket_counts_v25(filtered_by_class["package"])

        pkg_v25_bucket_raw_text = _build_package_evidence_buckets_v25_report(
            repo_root=repo_root,
            output_path=pkg_v25_bucket_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_v25_bucket_for,
            rationale_for=pkg_v25_rationale_for,
            hits_for=pkg_v25_hits_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v25_bucket_raw_path.write_text(pkg_v25_bucket_raw_text, encoding="utf-8")

        pkg_v25_bucket_filtered_text = _build_package_evidence_buckets_v25_filtered_report(
            repo_root=repo_root,
            output_path=pkg_v25_bucket_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_v25_bucket_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            bucket_for=pkg_v25_bucket_for,
            rationale_for=pkg_v25_rationale_for,
            hits_for=pkg_v25_hits_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v25_bucket_filtered_path.write_text(pkg_v25_bucket_filtered_text, encoding="utf-8")

        pkg_v25_short_raw_text = _build_package_no_evidence_shortlist_report(
            repo_root=repo_root,
            output_path=pkg_v25_short_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_v25_bucket_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v25_short_raw_path.write_text(pkg_v25_short_raw_text, encoding="utf-8")

        pkg_v25_short_filtered_text = _build_package_no_evidence_shortlist_filtered_report(
            repo_root=repo_root,
            output_path=pkg_v25_short_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_v25_short_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            bucket_for=pkg_v25_bucket_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v25_short_filtered_path.write_text(pkg_v25_short_filtered_text, encoding="utf-8")

        # Package class: v2.6 production-vs-tests evidence buckets (derived from v2.5 evidence hits).
        pkg_v26_bucket_raw_path = report_dir / "C_deadcode_package_prod_evidence_buckets.txt"
        pkg_v26_bucket_filtered_path = report_dir / "C_deadcode_package_prod_evidence_buckets_filtered.txt"
        pkg_v26_short_raw_path = report_dir / "C_deadcode_package_no_prod_evidence_shortlist.txt"
        pkg_v26_short_filtered_path = report_dir / "C_deadcode_package_no_prod_evidence_shortlist_filtered.txt"
        pkg_v26_keep_accepted_path = report_dir / "C_deadcode_package_no_prod_evidence_shortlist_test_only_keep_accepted.txt"
        pkg_v26_keep_remaining_path = report_dir / "C_deadcode_package_no_prod_evidence_shortlist_test_only_keep_remaining.txt"

        pkg_v26_bucket_for: dict[str, str] = {}
        pkg_v26_rationale_for: dict[str, str] = {}
        pkg_v26_hits_by_class_for: dict[str, dict[str, list[tuple[str, int, str, str, bool]]]] = {}

        for m in unreferenced_by_class["package"]:
            relpath = module_by_class["package"][m].relpath
            import_hits = [h for h in pkg_v25_import_stmt_hits.get(m, []) if h[0] != relpath]
            dyn_hits = [h for h in pkg_v25_dynamic_import_hits.get(m, []) if h[0] != relpath]
            rex_hits = [h for h in pkg_v25_reexport_hits.get(m, []) if h[0] != relpath]
            combined = _dedupe_and_cap_hits(
                import_hits + dyn_hits + rex_hits,
                cap=(PACKAGE_EVIDENCE_V26_MAX_HITS_PER_CLASS * 5),
            )
            by_class = _hits_by_source_class(combined)
            bucket, why = _package_v26_bucket_for_candidate(hits_by_class=by_class)
            pkg_v26_bucket_for[m] = bucket
            pkg_v26_rationale_for[m] = why
            pkg_v26_hits_by_class_for[m] = by_class

        def _bucket_counts_v26(mods: list[str]) -> dict[str, int]:
            bucket_order = [
                "PROD_IMPORT_EVIDENCE",
                "PROD_DYNAMIC_EVIDENCE",
                "PROD_REEXPORT_ONLY",
                "NO_PROD_EVIDENCE",
            ]
            counts = {b: 0 for b in bucket_order}
            for mod in mods:
                counts[pkg_v26_bucket_for[mod]] = counts.get(pkg_v26_bucket_for[mod], 0) + 1
            return counts

        pkg_v26_bucket_counts_filtered = _bucket_counts_v26(filtered_by_class["package"])

        pkg_v26_bucket_raw_text = _build_package_prod_evidence_buckets_v26_report(
            repo_root=repo_root,
            output_path=pkg_v26_bucket_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_v26_bucket_for,
            rationale_for=pkg_v26_rationale_for,
            hits_for=pkg_v26_hits_by_class_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v26_bucket_raw_path.write_text(pkg_v26_bucket_raw_text, encoding="utf-8")

        pkg_v26_bucket_filtered_text = _build_package_prod_evidence_buckets_v26_filtered_report(
            repo_root=repo_root,
            output_path=pkg_v26_bucket_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_v26_bucket_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            bucket_for=pkg_v26_bucket_for,
            rationale_for=pkg_v26_rationale_for,
            hits_for=pkg_v26_hits_by_class_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v26_bucket_filtered_path.write_text(pkg_v26_bucket_filtered_text, encoding="utf-8")

        pkg_v26_short_raw_text = _build_package_no_prod_evidence_shortlist_report(
            repo_root=repo_root,
            output_path=pkg_v26_short_raw_path,
            report_class=pkg_cfg["label"],
            module_index=module_by_class["package"],
            candidates=unreferenced_by_class["package"],
            bucket_for=pkg_v26_bucket_for,
            hits_for=pkg_v26_hits_by_class_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v26_short_raw_path.write_text(pkg_v26_short_raw_text, encoding="utf-8")

        pkg_v26_short_filtered_text = _build_package_no_prod_evidence_shortlist_filtered_report(
            repo_root=repo_root,
            output_path=pkg_v26_short_filtered_path,
            report_class=pkg_cfg["label"],
            base_report_path=pkg_v26_short_raw_path,
            module_index=module_by_class["package"],
            raw_candidates=unreferenced_by_class["package"],
            filtered_candidates=filtered_by_class["package"],
            bucket_for=pkg_v26_bucket_for,
            hits_for=pkg_v26_hits_by_class_for,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlisted_by_class["package"],
            allowlist_unmatched=allowlist_unmatched,
        )
        pkg_v26_short_filtered_path.write_text(pkg_v26_short_filtered_text, encoding="utf-8")

        filtered_sorted_v26 = sorted(
            filtered_by_class["package"], key=lambda m: (module_by_class["package"][m].relpath, m)
        )
        filtered_no_prod_v26 = [
            m
            for m in filtered_sorted_v26
            if pkg_v26_bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")
        ]
        filtered_shortlist_v26 = filtered_no_prod_v26[:PACKAGE_EVIDENCE_V26_SHORTLIST_MAX]

        keep_allowlist_unmatched_in_shortlist = sorted(
            [rel for rel in keep_allowlist_entries if rel not in {module_by_class["package"][m].relpath for m in filtered_shortlist_v26}]
        )
        keep_accepted_v26 = [
            m for m in filtered_shortlist_v26 if module_by_class["package"][m].relpath in keep_allowlist_set
        ]
        keep_remaining_v26 = [m for m in filtered_shortlist_v26 if m not in set(keep_accepted_v26)]

        pkg_v26_keep_accepted_text = _build_package_no_prod_evidence_test_only_keep_report(
            repo_root=repo_root,
            output_path=pkg_v26_keep_accepted_path,
            report_class=pkg_cfg["label"],
            base_shortlist_path=pkg_v26_short_filtered_path,
            module_index=module_by_class["package"],
            shortlist=keep_accepted_v26,
            accepted=keep_accepted_v26,
            remaining=[],
            bucket_for=pkg_v26_bucket_for,
            hits_for=pkg_v26_hits_by_class_for,
            keep_allowlist_path=str(TEST_ONLY_KEEP_ALLOWLIST_DEFAULT),
            keep_allowlist_missing=keep_allowlist_missing,
            keep_allowlist_entries=keep_allowlist_entries,
            keep_allowlist_unmatched_entries=keep_allowlist_unmatched_in_shortlist,
            title="Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, accepted as TEST-only keep)",
        )
        pkg_v26_keep_accepted_path.write_text(pkg_v26_keep_accepted_text, encoding="utf-8")

        pkg_v26_keep_remaining_text = _build_package_no_prod_evidence_test_only_keep_report(
            repo_root=repo_root,
            output_path=pkg_v26_keep_remaining_path,
            report_class=pkg_cfg["label"],
            base_shortlist_path=pkg_v26_short_filtered_path,
            module_index=module_by_class["package"],
            shortlist=keep_remaining_v26,
            accepted=[],
            remaining=keep_remaining_v26,
            bucket_for=pkg_v26_bucket_for,
            hits_for=pkg_v26_hits_by_class_for,
            keep_allowlist_path=str(TEST_ONLY_KEEP_ALLOWLIST_DEFAULT),
            keep_allowlist_missing=keep_allowlist_missing,
            keep_allowlist_entries=keep_allowlist_entries,
            keep_allowlist_unmatched_entries=keep_allowlist_unmatched_in_shortlist,
            title="Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, remaining after TEST-only keep allowlist)",
        )
        pkg_v26_keep_remaining_path.write_text(pkg_v26_keep_remaining_text, encoding="utf-8")

        # Tools/root classes: analyze as scripts via reference search (inventory + candidates).
        tools_scripts = sorted(
            [str(p.relative_to(repo_root)).replace("\\", "/") for p in files_by_class["tools"]]
        )
        root_scripts = sorted(
            [str(p.relative_to(repo_root)).replace("\\", "/") for p in files_by_class["root"]]
        )
        all_scripts = sorted(dict.fromkeys(tools_scripts + root_scripts))
        reference_hits = _find_script_reference_hits(
            repo_root=repo_root,
            scripts=all_scripts,
            search_files=search_files,
        )

        tools_unreferenced = sorted([s for s in tools_scripts if not reference_hits.get(s)])
        tools_allowlisted = sorted([s for s in tools_unreferenced if s in allowlist_set])
        tools_filtered = sorted([s for s in tools_unreferenced if s not in allowlist_set])

        root_unreferenced = sorted([s for s in root_scripts if not reference_hits.get(s)])
        root_allowlisted = sorted([s for s in root_unreferenced if s in allowlist_set])
        root_filtered = sorted([s for s in root_unreferenced if s not in allowlist_set])

        tools_cfg = outputs["tools"]
        tools_inv_text = _build_script_inventory_report(
            repo_root=repo_root,
            output_path=tools_cfg["inventory"],
            report_class=tools_cfg["label"],
            scripts=tools_scripts,
            reference_hits=reference_hits,
        )
        tools_cfg["inventory"].write_text(tools_inv_text, encoding="utf-8")

        tools_raw_text = _build_unreferenced_scripts_report(
            repo_root=repo_root,
            output_path=tools_cfg["raw"],
            report_class=tools_cfg["label"],
            scripts_total=len(tools_scripts),
            candidates=tools_unreferenced,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=tools_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        tools_cfg["raw"].write_text(tools_raw_text, encoding="utf-8")

        tools_filtered_text = _build_unreferenced_scripts_filtered_report(
            repo_root=repo_root,
            raw_report_path=tools_cfg["raw"],
            filtered_output_path=tools_cfg["filtered"],
            report_class=tools_cfg["label"],
            scripts_total=len(tools_scripts),
            raw_candidates=tools_unreferenced,
            filtered_candidates=tools_filtered,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=tools_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        tools_cfg["filtered"].write_text(tools_filtered_text, encoding="utf-8")

        root_cfg = outputs["root"]
        root_inv_text = _build_script_inventory_report(
            repo_root=repo_root,
            output_path=root_cfg["inventory"],
            report_class=root_cfg["label"],
            scripts=root_scripts,
            reference_hits=reference_hits,
        )
        root_cfg["inventory"].write_text(root_inv_text, encoding="utf-8")

        root_raw_text = _build_unreferenced_scripts_report(
            repo_root=repo_root,
            output_path=root_cfg["raw"],
            report_class=root_cfg["label"],
            scripts_total=len(root_scripts),
            candidates=root_unreferenced,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=root_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        root_cfg["raw"].write_text(root_raw_text, encoding="utf-8")

        root_filtered_text = _build_unreferenced_scripts_filtered_report(
            repo_root=repo_root,
            raw_report_path=root_cfg["raw"],
            filtered_output_path=root_cfg["filtered"],
            report_class=root_cfg["label"],
            scripts_total=len(root_scripts),
            raw_candidates=root_unreferenced,
            filtered_candidates=root_filtered,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=root_allowlisted,
            allowlist_unmatched=allowlist_unmatched,
        )
        root_cfg["filtered"].write_text(root_filtered_text, encoding="utf-8")

        pkg_evidence_raw_has = len([m for m in unreferenced_by_class["package"] if pkg_reference_hits.get(m)])
        pkg_evidence_raw_none = len(unreferenced_by_class["package"]) - pkg_evidence_raw_has
        pkg_evidence_filtered_has = len([m for m in filtered_by_class["package"] if pkg_reference_hits.get(m)])
        pkg_evidence_filtered_none = len(filtered_by_class["package"]) - pkg_evidence_filtered_has

        pkg_v25_no_evidence_raw = len([m for m in unreferenced_by_class["package"] if pkg_v25_bucket_for[m] == "NO_EVIDENCE"])
        pkg_v25_no_evidence_filtered = len([m for m in filtered_by_class["package"] if pkg_v25_bucket_for[m] == "NO_EVIDENCE"])
        pkg_v25_short_raw_count = min(pkg_v25_no_evidence_raw, PACKAGE_EVIDENCE_V25_SHORTLIST_MAX)
        pkg_v25_short_filtered_count = min(pkg_v25_no_evidence_filtered, PACKAGE_EVIDENCE_V25_SHORTLIST_MAX)

        pkg_v26_no_prod_raw = len(
            [
                m
                for m in unreferenced_by_class["package"]
                if pkg_v26_bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")
            ]
        )
        pkg_v26_no_prod_filtered = len(
            [
                m
                for m in filtered_by_class["package"]
                if pkg_v26_bucket_for[m] in ("NO_PROD_EVIDENCE", "PROD_REEXPORT_ONLY")
            ]
        )
        pkg_v26_short_raw_count = min(pkg_v26_no_prod_raw, PACKAGE_EVIDENCE_V26_SHORTLIST_MAX)
        pkg_v26_short_filtered_count = min(pkg_v26_no_prod_filtered, PACKAGE_EVIDENCE_V26_SHORTLIST_MAX)
        pkg_v26_short_filtered_keep_accepted = len(keep_accepted_v26)
        pkg_v26_short_filtered_keep_remaining = len(keep_remaining_v26)

        print(
            "DEADCODE_AUDIT_COUNTS_V2"
            f" package_raw={len(unreferenced_by_class['package'])}"
            f" package_filtered={len(filtered_by_class['package'])}"
            f" package_allowlisted={len(allowlisted_by_class['package'])}"
            f" package_parse_failures={len(parse_failures_by_class['package'])}"
            f" package_evidence_raw_has={pkg_evidence_raw_has}"
            f" package_evidence_raw_none={pkg_evidence_raw_none}"
            f" package_evidence_filtered_has={pkg_evidence_filtered_has}"
            f" package_evidence_filtered_none={pkg_evidence_filtered_none}"
            f" package_bucket_raw_import={pkg_bucket_counts_raw.get('IMPORT_EVIDENCE', 0)}"
            f" package_bucket_raw_py_runtime={pkg_bucket_counts_raw.get('PY_RUNTIME_REFERENCE', 0)}"
            f" package_bucket_raw_non_py={pkg_bucket_counts_raw.get('NON_PY_REFERENCE', 0)}"
            f" package_bucket_raw_no_evidence={pkg_bucket_counts_raw.get('NO_EVIDENCE', 0)}"
            f" package_bucket_filtered_import={pkg_bucket_counts_filtered.get('IMPORT_EVIDENCE', 0)}"
            f" package_bucket_filtered_py_runtime={pkg_bucket_counts_filtered.get('PY_RUNTIME_REFERENCE', 0)}"
            f" package_bucket_filtered_non_py={pkg_bucket_counts_filtered.get('NON_PY_REFERENCE', 0)}"
            f" package_bucket_filtered_no_evidence={pkg_bucket_counts_filtered.get('NO_EVIDENCE', 0)}"
            f" package_high_confidence_raw={len(pkg_high_raw)}"
            f" package_high_confidence_filtered={len(pkg_high_filtered)}"
            f" package_v25_bucket_raw_import={pkg_v25_bucket_counts_raw.get('IMPORT_EVIDENCE', 0)}"
            f" package_v25_bucket_raw_dynamic={pkg_v25_bucket_counts_raw.get('DYNAMIC_EVIDENCE', 0)}"
            f" package_v25_bucket_raw_reexport_only={pkg_v25_bucket_counts_raw.get('REEXPORT_ONLY', 0)}"
            f" package_v25_bucket_raw_no_evidence={pkg_v25_bucket_counts_raw.get('NO_EVIDENCE', 0)}"
            f" package_v25_bucket_filtered_import={pkg_v25_bucket_counts_filtered.get('IMPORT_EVIDENCE', 0)}"
            f" package_v25_bucket_filtered_dynamic={pkg_v25_bucket_counts_filtered.get('DYNAMIC_EVIDENCE', 0)}"
            f" package_v25_bucket_filtered_reexport_only={pkg_v25_bucket_counts_filtered.get('REEXPORT_ONLY', 0)}"
            f" package_v25_bucket_filtered_no_evidence={pkg_v25_bucket_counts_filtered.get('NO_EVIDENCE', 0)}"
            f" package_v25_shortlist_raw={pkg_v25_short_raw_count}"
            f" package_v25_shortlist_filtered={pkg_v25_short_filtered_count}"
            f" package_v26_bucket_filtered_prod_import={pkg_v26_bucket_counts_filtered.get('PROD_IMPORT_EVIDENCE', 0)}"
            f" package_v26_bucket_filtered_prod_dynamic={pkg_v26_bucket_counts_filtered.get('PROD_DYNAMIC_EVIDENCE', 0)}"
            f" package_v26_bucket_filtered_prod_reexport_only={pkg_v26_bucket_counts_filtered.get('PROD_REEXPORT_ONLY', 0)}"
            f" package_v26_bucket_filtered_no_prod_evidence={pkg_v26_bucket_counts_filtered.get('NO_PROD_EVIDENCE', 0)}"
            f" package_v26_shortlist_raw={pkg_v26_short_raw_count}"
            f" package_v26_shortlist_filtered={pkg_v26_short_filtered_count}"
            f" package_v26_shortlist_filtered_keep_accepted={pkg_v26_short_filtered_keep_accepted}"
            f" package_v26_shortlist_filtered_keep_remaining={pkg_v26_short_filtered_keep_remaining}"
            f" package_v26_keep_allowlist_missing={1 if keep_allowlist_missing else 0}"
            f" package_v26_keep_allowlist_entries={len(keep_allowlist_entries)}"
            f" tools_total_scripts={len(tools_scripts)}"
            f" tools_referenced_scripts={len([s for s in tools_scripts if reference_hits.get(s)])}"
            f" tools_raw={len(tools_unreferenced)}"
            f" tools_filtered={len(tools_filtered)}"
            f" tools_allowlisted={len(tools_allowlisted)}"
            f" tools_parse_failures={len(parse_failures_by_class['tools'])}"
            f" root_total_scripts={len(root_scripts)}"
            f" root_referenced_scripts={len([s for s in root_scripts if reference_hits.get(s)])}"
            f" root_raw={len(root_unreferenced)}"
            f" root_filtered={len(root_filtered)}"
            f" root_allowlisted={len(root_allowlisted)}"
            f" root_parse_failures={len(parse_failures_by_class['root'])}"
            f" allowlist_entries={len(allowlist_entries)}"
            f" allowlist_missing={1 if allowlist_missing else 0}"
        )

        if (
            filtered_by_class["package"]
            or parse_failures_by_class["package"]
            or tools_filtered
            or parse_failures_by_class["tools"]
            or root_filtered
            or parse_failures_by_class["root"]
        ):
            return 3
        return 0

    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowlist_matched: list[str] = []
    filtered_candidates: list[str] = []
    for mod in unreferenced:
        rel = module_index[mod].relpath
        if rel in allowlist_set:
            allowlist_matched.append(rel)
        else:
            filtered_candidates.append(mod)
    allowlist_matched = sorted(set(allowlist_matched))

    report_text = _build_report(
        repo_root=repo_root,
        output_path=output_path,
        report_class=None,
        scan_files=scan_files,
        module_index=module_index,
        parsed_ok=parsed_ok,
        parse_failures=parse_failures,
        entrypoints=entrypoints_sorted,
        init_modules=init_modules,
        unreferenced=unreferenced,
        incoming_sources=incoming_sources,
        edge_count=edge_count,
        allowlist_path=str(allowlist_path) if allowlist_path else None,
        allowlist_missing=allowlist_missing,
        allowlist_entries=allowlist_entries,
        allowlist_matched=allowlist_matched,
        allowlist_unmatched=allowlist_unmatched,
    )
    output_path.write_text(report_text, encoding="utf-8")

    if filtered_output_path is not None:
        filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_text = _build_filtered_report(
            repo_root=repo_root,
            output_path=output_path,
            filtered_output_path=filtered_output_path,
            report_class=None,
            module_index=module_index,
            parse_failures=parse_failures,
            raw_candidates=unreferenced,
            filtered_candidates=filtered_candidates,
            allowlist_path=str(allowlist_path) if allowlist_path else None,
            allowlist_missing=allowlist_missing,
            allowlist_entries=allowlist_entries,
            allowlist_matched=allowlist_matched,
            allowlist_unmatched=allowlist_unmatched,
        )
        filtered_output_path.write_text(filtered_text, encoding="utf-8")

    print(
        "DEADCODE_AUDIT_COUNTS"
        f" raw={len(unreferenced)}"
        f" filtered={len(filtered_candidates)}"
        f" allowlisted={len(allowlist_matched)}"
        f" allowlist_entries={len(allowlist_entries)}"
        f" allowlist_missing={1 if allowlist_missing else 0}"
        f" parse_failures={len(parse_failures)}"
    )

    if filtered_candidates or parse_failures:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
