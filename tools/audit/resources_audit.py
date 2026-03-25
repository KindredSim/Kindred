#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.audit.path_filters import is_excluded_rel as _is_excluded_rel


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

RESOURCE_ROOT_CANDIDATES = (
    "kindred/data",
    "data",
    "benchmarks/regression_suite",
)

RESOURCE_EXTS = {
    ".ico",
    ".png",
    ".svg",
    ".ui",
    ".qss",
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".dsl",
}

POSIX_LITERAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("POSIX_TMP", "/tmp"),  # nosec B108 - literal pattern for path scanning, not temp file creation
    ("POSIX_VAR", "/var/"),
    ("POSIX_USR", "/usr/"),
    ("POSIX_HOME", "/home/"),
    ("POSIX_ETC", "/etc/"),
    ("XDG_CONFIG", "~/.config"),
    ("XDG_CACHE", "~/.cache"),
    ("XDG_DATA", "~/.local"),
)

RESOURCE_HINT_PARTS = {
    "assets",
    "ui",
    "data",
    "presets",
    "dsl",
    "benchmarks",
    "resources",
}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    file_line: str
    message: str
    suggested_fix: str

    def render(self) -> str:
        return (
            f"{self.rule_id} | {self.severity} | {self.file_line} | {self.message} | {self.suggested_fix}"
        )


@dataclass(frozen=True)
class RefEvidence:
    rel_file: str
    lineno: int
    kind: str
    ref: str
    candidates: tuple[str, ...]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_python_files(repo_root: Path) -> list[Path]:
    scan_roots = [repo_root / "kindred", repo_root / "tools"]
    out: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for dirpath_str, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath_str)
            try:
                rel_dir = dirpath.relative_to(repo_root)
            except ValueError:
                continue
            rel_dir_posix = str(rel_dir).replace("\\", "/")
            if _is_excluded_rel(rel_dir_posix + "/"):
                dirnames[:] = []
                continue

            dirnames[:] = sorted(
                [
                    d
                    for d in dirnames
                    if d not in EXCLUDE_DIR_NAMES
                    and d != "_audit_reports"
                   
                ]
            )
            filenames_sorted = sorted(filenames)
            for name in filenames_sorted:
                if not name.endswith(".py"):
                    continue
                path = dirpath / name
                try:
                    rel = path.relative_to(repo_root)
                except ValueError:
                    continue
                rel_posix = str(rel).replace("\\", "/")
                if _is_excluded_rel(rel_posix):
                    continue
                out.append(path)
    uniq = sorted({p.resolve() for p in out}, key=lambda p: str(p).replace("\\", "/"))
    return uniq


def _discover_resource_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for rel in RESOURCE_ROOT_CANDIDATES:
        p = (repo_root / rel).resolve()
        try:
            p.relative_to(repo_root)
        except ValueError:
            continue
        if p.exists() and p.is_dir():
            roots.append(p)
    return sorted(roots, key=lambda p: str(p).replace("\\", "/"))


def _iter_files_under(repo_root: Path, root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        try:
            rel_dir = dirpath.relative_to(repo_root)
        except ValueError:
            continue
        rel_dir_posix = str(rel_dir).replace("\\", "/")
        if _is_excluded_rel(rel_dir_posix + "/"):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            [
                d
                for d in dirnames
                if d not in EXCLUDE_DIR_NAMES
                and d != "_audit_reports"
               
            ]
        )
        for name in sorted(filenames):
            p = dirpath / name
            try:
                rel = p.relative_to(repo_root)
            except ValueError:
                continue
            rel_posix = str(rel).replace("\\", "/")
            if _is_excluded_rel(rel_posix):
                continue
            out.append(p)
    return sorted({p.resolve() for p in out}, key=lambda p: str(p).replace("\\", "/"))


def _rel_posix(repo_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root)
    except Exception:
        return str(path).replace("\\", "/")
    return str(rel).replace("\\", "/")


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _compute_docstring_const_ids(tree: ast.AST) -> set[int]:
    doc_ids: set[int] = set()

    def _maybe_mark_docstring(body: list[ast.stmt]) -> None:
        if not body:
            return
        first = body[0]
        if not isinstance(first, ast.Expr):
            return
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            doc_ids.add(id(value))

    if isinstance(tree, ast.Module):
        _maybe_mark_docstring(tree.body)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _maybe_mark_docstring(node.body)

    return doc_ids


def _contains_name(node: ast.AST, *, name: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == name:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == name:
            return True
    return False


def _platform_guard_kind(test: ast.AST) -> str | None:
    def _is_attr(n: ast.AST, base: str, attr: str) -> bool:
        return (
            isinstance(n, ast.Attribute)
            and n.attr == attr
            and isinstance(n.value, ast.Name)
            and n.value.id == base
        )

    def _is_platform_expr(n: ast.AST) -> bool:
        if _is_attr(n, "sys", "platform"):
            return True
        if _is_attr(n, "os", "name"):
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if isinstance(n.func.value, ast.Name) and n.func.value.id == "platform":
                if n.func.attr in {"system", "platform"}:
                    return True
        return False

    def _kind_from_compare(cmp: ast.Compare) -> str | None:
        if not _is_platform_expr(cmp.left):
            return None
        if len(cmp.comparators) != 1 or len(cmp.ops) != 1:
            return "platform_guard"
        op = cmp.ops[0]
        rhs = cmp.comparators[0]
        if not isinstance(rhs, ast.Constant) or not isinstance(rhs.value, str):
            return "platform_guard"
        v = rhs.value.lower()
        windows_vals = {"win32", "windows", "nt"}
        if isinstance(op, ast.Eq):
            return "windows_only" if v in windows_vals else "platform_guard"
        if isinstance(op, ast.NotEq):
            return "non_windows_only" if v in windows_vals else "platform_guard"
        if isinstance(op, ast.In):
            return "platform_guard"
        if isinstance(op, ast.NotIn):
            return "platform_guard"
        return "platform_guard"

    if isinstance(test, ast.Compare):
        return _kind_from_compare(test)
    if isinstance(test, ast.BoolOp):
        for v in test.values:
            k = _platform_guard_kind(v)
            if k:
                return k
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _platform_guard_kind(test.operand) or None
    if isinstance(test, ast.Call) and _contains_name(test, name="sys") and _contains_name(test, name="platform"):
        return "platform_guard"
    if _is_platform_expr(test):
        return "platform_guard"
    return None


def _string_looks_like_resource_path(s: str) -> bool:
    if not s:
        return False
    if "*" in s or "?" in s:
        return False
    if "(*" in s or ";;" in s:
        return False
    if s.startswith(":/"):
        return False
    if s.startswith(("http://", "https://")):
        return False
    norm = s.replace("\\", "/").lstrip("/")
    ext = Path(norm).suffix.lower()
    parts = tuple(p for p in norm.split("/") if p)
    if norm.startswith("kindred/data/"):
        return True
    if ext in RESOURCE_EXTS and any(p.lower() in RESOURCE_HINT_PARTS for p in parts):
        return True
    if norm.startswith(("assets/", "ui/", "presets/", "dsl/", "data/", "benchmarks/")):
        return True
    return False


def _candidate_paths_for_get_resource(repo_root: Path, ref: str) -> tuple[str, ...]:
    r = ref.replace("\\", "/").lstrip("/")
    if r.startswith("kindred/data/"):
        r = r[len("kindred/data/") :]
    candidates = []
    candidates.append(_rel_posix(repo_root, repo_root / "kindred" / "data" / r))
    candidates.append(_rel_posix(repo_root, repo_root / r))
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return tuple(uniq)


def _candidate_paths_for_relative(
    repo_root: Path, *, ref: str, module_dir: Path | None
) -> tuple[str, ...]:
    r = ref.replace("\\", "/").lstrip("/")
    candidates: list[Path] = []
    if module_dir is not None:
        candidates.append(module_dir / r)
    candidates.append(repo_root / r)
    if r.startswith(("assets/", "ui/", "presets/", "dsl/")):
        candidates.append(repo_root / "kindred" / "data" / r)
    uniq: list[str] = []
    seen: set[str] = set()
    for p in candidates:
        s = _rel_posix(repo_root, p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return tuple(uniq)


def _exists_any(repo_root: Path, candidates: tuple[str, ...]) -> bool:
    for rel in candidates:
        p = Path(rel)
        if p.is_absolute():
            if p.exists():
                return True
        else:
            if (repo_root / rel).exists():
                return True
    return False


class _PathEnv:
    def __init__(self, *, file_path: Path):
        self._file_path = file_path
        self._vars: dict[str, Path] = {}

    def set(self, name: str, value: Path) -> None:
        self._vars[name] = value

    def get(self, name: str) -> Path | None:
        return self._vars.get(name)

    @property
    def file_path(self) -> Path:
        return self._file_path


def _eval_path_expr(node: ast.AST, env: _PathEnv) -> Path | None:
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return env.file_path
        v = env.get(node.id)
        if v is not None:
            return v
        return None

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return Path(node.value)
        return None

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            if not node.args:
                return None
            base = _eval_path_expr(node.args[0], env)
            if base is None:
                return None
            return Path(str(base))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            base = _eval_path_expr(node.func.value, env)
            return base.resolve() if base is not None else None
        return None

    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _eval_path_expr(node.value, env)
        return base.parent if base is not None else None

    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "parents":
            base = _eval_path_expr(node.value.value, env)
            if base is None:
                return None
            idx = None
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                idx = int(node.slice.value)
            if idx is None:
                return None
            try:
                return base.resolve().parents[idx]
            except Exception:
                return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_path_expr(node.left, env)
        right = _eval_path_expr(node.right, env)
        if left is None or right is None:
            return None
        return Path(str(left)) / str(right)

    return None


def _index_simple_path_assignments(tree: ast.AST, env: _PathEnv) -> None:
    if not isinstance(tree, ast.Module):
        return
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _eval_path_expr(stmt.value, env)
        if value is None:
            continue
        env.set(target.id, value)


def _iter_string_constants(tree: ast.AST, *, skip_const_ids: set[int]) -> list[ast.Constant]:
    out: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if id(node) in skip_const_ids:
            continue
        if not isinstance(node.value, str):
            continue
        if getattr(node, "lineno", None) is None:
            continue
        out.append(node)
    return out


def _extract_get_resource_refs(tree: ast.AST) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name not in {"get_resource_path", "get_resource_text"}:
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and getattr(arg, "lineno", None):
            out.append((int(arg.lineno), func_name, arg.value))
    return out


def _extract_open_relative_refs(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "open":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and getattr(arg, "lineno", None):
            s = arg.value
            if not s or s.startswith(("/", "://")):
                continue
            if _string_looks_like_resource_path(s):
                out.append((int(arg.lineno), s))
    return out


def _extract_path_read_calls(tree: ast.AST) -> list[tuple[int, ast.AST, str]]:
    out: list[tuple[int, ast.AST, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"read_text", "read_bytes", "open"}:
            continue
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        out.append((int(lineno), node.func.value, node.func.attr))
    return out


def _render_guard_context(guard_stack: tuple[str, ...]) -> str:
    if not guard_stack:
        return "guarded=no"
    uniq = []
    seen: set[str] = set()
    for g in guard_stack:
        if g in seen:
            continue
        seen.add(g)
        uniq.append(g)
    return "guarded=yes(" + ",".join(uniq) + ")"


def _collect_case_collisions(repo_root: Path, resource_roots: list[Path]) -> list[list[str]]:
    seen: dict[str, set[str]] = {}
    for root in resource_roots:
        for p in _iter_files_under(repo_root, root):
            rel = _rel_posix(repo_root, p)
            key = rel.lower()
            seen.setdefault(key, set()).add(rel)
    groups: list[list[str]] = []
    for _k, paths in seen.items():
        if len(paths) <= 1:
            continue
        groups.append(sorted(paths))
    return sorted(groups, key=lambda g: (g[0], len(g)))


def _inventory_assets(repo_root: Path, resource_roots: list[Path]) -> list[str]:
    assets: list[str] = []
    for root in resource_roots:
        for p in _iter_files_under(repo_root, root):
            if p.is_file() and p.suffix.lower() in RESOURCE_EXTS:
                assets.append(_rel_posix(repo_root, p))
    return sorted(set(assets))


def _scan_one_python_file(
    *,
    repo_root: Path,
    path: Path,
    resource_roots: list[Path],
) -> tuple[list[RefEvidence], list[Finding], list[Finding], list[Finding], list[Finding]]:
    rel_file = _rel_posix(repo_root, path)
    text = _safe_read_text(path)
    if not text.strip():
        return ([], [], [], [], [])
    try:
        tree = ast.parse(text, filename=rel_file)
    except SyntaxError:
        return ([], [], [], [], [])

    doc_ids = _compute_docstring_const_ids(tree)
    env = _PathEnv(file_path=path.resolve())
    _index_simple_path_assignments(tree, env)

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    missing_refs: list[RefEvidence] = []
    packaging_risks: list[Finding] = []
    posix_risks: list[Finding] = []
    qt_refs: list[Finding] = []
    info_findings: list[Finding] = []

    string_occurrences: list[tuple[int, str, tuple[str, ...]]] = []
    file_usage_linenos: set[int] = set()

    def _invert_guard(kind: str) -> str:
        if kind == "windows_only":
            return "non_windows_only"
        if kind == "non_windows_only":
            return "windows_only"
        return kind

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self._guard_stack: list[str] = []

        def visit_If(self, node: ast.If) -> None:
            kind = _platform_guard_kind(node.test)
            if not kind:
                self.generic_visit(node)
                return

            self._guard_stack.append(kind)
            for child in node.body:
                self.visit(child)
            self._guard_stack.pop()

            else_kind = _invert_guard(kind)
            self._guard_stack.append(else_kind)
            for child in node.orelse:
                self.visit(child)
            self._guard_stack.pop()

        def visit_Constant(self, node: ast.Constant) -> None:
            if id(node) not in doc_ids and isinstance(node.value, str) and getattr(node, "lineno", None):
                string_occurrences.append(
                    (int(node.lineno), str(node.value), tuple(self._guard_stack))
                )
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == "__file__" and getattr(node, "lineno", None):
                file_usage_linenos.add(int(node.lineno))
            self.generic_visit(node)

    Collector().visit(tree)

    # Qt refs + POSIX literals from string constants (docstrings excluded), with platform-guard context.
    for lineno, s, guards in sorted(string_occurrences, key=lambda t: (t[0], t[1])):
        if s.startswith(":/"):
            src = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
            qt_refs.append(
                Finding(
                    rule_id="I5 QtResourceRef",
                    severity="INFO",
                    file_line=f"{rel_file}:{lineno}",
                    message=f"qt_resource_ref={s!r} {_render_guard_context(guards)} context={src!r}",
                    suggested_fix="If this must map to disk assets, ensure a Qt .qrc/.rcc build pipeline exists.",
                )
            )
            continue

        for label, needle in POSIX_LITERAL_PATTERNS:
            if s == needle or s.startswith(needle) or needle in s:
                src = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
                posix_risks.append(
                    Finding(
                        rule_id="I4 PosixPathLiteral",
                        severity="INFO",
                        file_line=f"{rel_file}:{lineno}",
                        message=f"{label} literal={s!r} {_render_guard_context(guards)} context={src!r}",
                        suggested_fix="Use platform-specific user dirs (Qt QStandardPaths or kindred.io.paths helpers) and guard OS-specific paths.",
                    )
                )
                break

    # get_resource_* refs (high-confidence packaged resources under kindred/data).
    for lineno, func_name, ref in _extract_get_resource_refs(tree):
        candidates = _candidate_paths_for_get_resource(repo_root, ref)
        if not _exists_any(repo_root, candidates):
            missing_refs.append(
                RefEvidence(
                    rel_file=rel_file,
                    lineno=lineno,
                    kind=func_name,
                    ref=ref,
                    candidates=candidates,
                )
            )

    # open("relative/path.ext") refs (packaging risk + missing check, conservative).
    module_dir = path.resolve().parent
    for lineno, ref in _extract_open_relative_refs(tree):
        candidates = _candidate_paths_for_relative(repo_root, ref=ref, module_dir=module_dir)
        packaging_risks.append(
            Finding(
                rule_id="I3 PackagingRisk",
                severity="INFO",
                file_line=f"{rel_file}:{lineno}",
                message=f"open() with relative path literal ref={ref!r} candidates={list(candidates)!r}",
                suggested_fix="Prefer importlib.resources (kindred.io.resources) or absolute user-selected paths; avoid cwd-relative opens in packaged apps.",
            )
        )
        if not _exists_any(repo_root, candidates):
            missing_refs.append(
                RefEvidence(
                    rel_file=rel_file,
                    lineno=lineno,
                    kind="open",
                    ref=ref,
                    candidates=candidates,
                )
            )

    # __file__ usage signals packaging risk; record first occurrence.
    if file_usage_linenos:
        lineno = min(file_usage_linenos)
        src = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
        packaging_risks.append(
            Finding(
                rule_id="I3 PackagingRisk",
                severity="INFO",
                file_line=f"{rel_file}:{lineno}",
                message=f"uses __file__ for filesystem access context={src!r}",
                suggested_fix="Prefer importlib.resources for packaged assets; if using __file__, ensure behavior is correct under frozen builds.",
            )
        )

    for lineno, base_expr, attr in _extract_path_read_calls(tree):
        p = _eval_path_expr(base_expr, env)
        if p is None:
            continue
        src = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
        # Treat repo-internal file reads as packaging risks unless clearly user-supplied paths.
        if str(p).startswith(str(repo_root)):
            packaging_risks.append(
                Finding(
                    rule_id="I3 PackagingRisk",
                    severity="INFO",
                    file_line=f"{rel_file}:{lineno}",
                    message=f"pathlib.{attr}() on repo-relative path target={_rel_posix(repo_root, p)!r} context={src!r}",
                    suggested_fix="Prefer importlib.resources for packaged assets; avoid relying on repo layout at runtime.",
                )
            )
        # Missing check for resolved repo-internal paths with resource-like extensions.
        rel_target = _rel_posix(repo_root, p)
        if (
            str(p).startswith(str(repo_root))
            and Path(rel_target).suffix.lower() in RESOURCE_EXTS
            and not (repo_root / rel_target).exists()
        ):
            missing_refs.append(
                RefEvidence(
                    rel_file=rel_file,
                    lineno=lineno,
                    kind=f"pathlib.{attr}",
                    ref=rel_target,
                    candidates=(rel_target,),
                )
            )

    # De-dup and sort.
    missing_refs = sorted(
        {
            (r.rel_file, r.lineno, r.kind, r.ref, r.candidates): r
            for r in missing_refs
        }.values(),
        key=lambda r: (r.rel_file, r.lineno, r.kind, r.ref),
    )
    packaging_risks = sorted(
        {f.render(): f for f in packaging_risks}.values(), key=lambda f: f.file_line
    )
    posix_risks = sorted({f.render(): f for f in posix_risks}.values(), key=lambda f: f.file_line)
    qt_refs = sorted({f.render(): f for f in qt_refs}.values(), key=lambda f: f.file_line)
    info_findings = sorted({f.render(): f for f in info_findings}.values(), key=lambda f: f.file_line)

    return (missing_refs, packaging_risks, posix_risks, qt_refs, info_findings)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Audit I: Resources + Windows packaging compatibility (stdlib-only)."
    )
    ap.add_argument("--report-dir", required=True, help="Report directory (_audit_reports/<timestamp>).")
    ap.add_argument("--root", default=None, help="Repo root (default: derive from __file__).")
    args = ap.parse_args(argv)

    repo_root = (Path(args.root).resolve() if args.root else _default_repo_root()).resolve()
    report_dir = Path(args.report_dir).resolve()
    out_path = report_dir / "I_resources.txt"

    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    resource_roots = _discover_resource_roots(repo_root)
    asset_inventory = _inventory_assets(repo_root, resource_roots)
    case_conflicts = _collect_case_collisions(repo_root, resource_roots)

    py_files = _iter_python_files(repo_root)
    missing: list[RefEvidence] = []
    packaging: list[Finding] = []
    posix: list[Finding] = []
    qt: list[Finding] = []

    for p in py_files:
        m, pack, pos, q, _info = _scan_one_python_file(
            repo_root=repo_root, path=p, resource_roots=resource_roots
        )
        missing.extend(m)
        packaging.extend(pack)
        posix.extend(pos)
        qt.extend(q)

    missing = sorted(
        {
            (r.rel_file, r.lineno, r.kind, r.ref, r.candidates): r
            for r in missing
        }.values(),
        key=lambda r: (r.rel_file, r.lineno, r.kind, r.ref),
    )
    packaging = sorted({f.render(): f for f in packaging}.values(), key=lambda f: f.file_line)
    posix = sorted({f.render(): f for f in posix}.values(), key=lambda f: f.file_line)
    qt = sorted({f.render(): f for f in qt}.values(), key=lambda f: f.file_line)

    out: list[str] = []
    out.append("Kindred Audit I: Resources + Windows packaging compatibility (report-only)")
    out.append(f"Timestamp (UTC): {timestamp_utc}")
    out.append(f"Repo root: {repo_root}")
    out.append("")
    out.append("Scope:")
    out.append("- Python scan: kindred/** and tools/** (excluding tools/audit/**)")
    out.append("- Resource roots (derived from repo structure):")
    if not resource_roots:
        out.append("  - (none discovered)")
    else:
        for r in resource_roots:
            out.append(f"  - {_rel_posix(repo_root, r)}")
    out.append("")
    out.append("Counts:")
    out.append(f"- scanned_py: {len(py_files)}")
    out.append(f"- scanned_assets: {len(asset_inventory)}")
    out.append(f"- missing: {len(missing)}")
    out.append(f"- case_conflicts: {len(case_conflicts)}")
    out.append(f"- packaging_risks: {len(packaging)}")
    out.append(f"- posix_path_risks: {len(posix)}")
    out.append(f"- qt_refs: {len(qt)}")
    out.append("")
    out.append(
        "I_RESOURCES_COUNTS"
        f"|missing={len(missing)}"
        f"|case_conflicts={len(case_conflicts)}"
        f"|packaging_risks={len(packaging)}"
        f"|posix_path_risks={len(posix)}"
        f"|qt_refs={len(qt)}"
        f"|scanned_py={len(py_files)}"
        f"|scanned_assets={len(asset_inventory)}"
    )
    out.append("")

    out.append("=== Missing Resources (filesystem) ===")
    if not missing:
        out.append("- (none)")
    else:
        for r in missing:
            file_line = f"{r.rel_file}:{r.lineno}"
            out.append(
                Finding(
                    rule_id="I1 MissingResource",
                    severity="WARN",
                    file_line=file_line,
                    message=f"kind={r.kind} ref={r.ref!r} candidates={list(r.candidates)!r}",
                    suggested_fix="Add the resource file/dir or update the reference to a packaged resource loader (importlib.resources / kindred.io.resources).",
                ).render()
            )
    out.append("")

    out.append("=== Windows Case-Collision Risks (resource roots) ===")
    if not case_conflicts:
        out.append("- (none)")
    else:
        for group in case_conflicts:
            out.append(
                Finding(
                    rule_id="I2 CaseCollision",
                    severity="WARN",
                    file_line="-",
                    message=f"paths differ only by case: {group!r}",
                    suggested_fix="Rename files to be unique under case-insensitive filesystems (Windows, default macOS).",
                ).render()
            )
    out.append("")

    out.append("=== Packaging-Risk Patterns (informational) ===")
    if not packaging:
        out.append("- (none)")
    else:
        for f in packaging:
            out.append(f.render())
    out.append("")

    out.append("=== POSIX Path Literals (informational) ===")
    if not posix:
        out.append("- (none)")
    else:
        for f in posix:
            out.append(f.render())
    out.append("")

    out.append("=== Qt Resource References (:/...) (informational) ===")
    if not qt:
        out.append("- (none)")
    else:
        for f in qt:
            out.append(f.render())
    out.append("")

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
