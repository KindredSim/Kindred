#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
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


FROZEN_GUARD_RE = re.compile(
    r"(sys\.(?:frozen|_MEIPASS))"
    r"|(_MEIPASS)"
    r"|(getattr\s*\(\s*sys\s*,\s*['\"]frozen['\"])"
    r"|(getattr\s*\(\s*sys\s*,\s*['\"]_MEIPASS['\"])"
    r"|(hasattr\s*\(\s*sys\s*,\s*['\"]_MEIPASS['\"])"
    r"|(_is_frozen\s*\()"
    r"|(PyInstaller)",
    re.IGNORECASE,
)


QT_ENV_RE = re.compile(
    r"\b(QT_PLUGIN_PATH|QT_QPA_PLATFORM_PLUGIN_PATH|QT_QPA_PLATFORM)\b", re.IGNORECASE
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel_posix(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _iter_python_files(repo_root: Path, include_tools: bool) -> list[Path]:
    scan_roots = [repo_root / "kindred"]
    if include_tools:
        scan_roots.append(repo_root / "tools")

    out: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            )
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                rel = _rel_posix(repo_root, path)
                if _is_excluded_rel(rel):
                    continue
                out.append(path)
    return sorted(out, key=lambda p: _rel_posix(repo_root, p))


def _iter_tree_files(repo_root: Path, root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = _rel_posix(repo_root, path)
            if _is_excluded_rel(rel):
                continue
            out.append(path)
    return sorted(out, key=lambda p: _rel_posix(repo_root, p))


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        if base:
            return f"{base}.{func.attr}"
        return func.attr
    return None


def _expr_contains_name(expr: ast.AST, name: str) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id == name:
            return True
    return False


def _const_str(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    file_line: str
    message: str
    suggested_fix: str
    likely_break: bool

    def render(self) -> str:
        return (
            f"{self.rule_id} | {self.severity} | {self.file_line} | {self.message} | {self.suggested_fix}"
        )


@dataclass(frozen=True)
class _ScanResult:
    scanned_py_files: int
    file_usage_total: int
    file_usage_unguarded: int
    posix_literal_hits: int
    qt_resource_literal_hits: int
    dynamic_import_hits: int
    fs_resource_hits: int
    qt_env_hits: int
    case_conflicts_groups: int
    likely_break_total: int
    findings: list[Finding]
    case_conflict_groups: list[list[str]]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _line_window(lines: list[str], lineno_1based: int, radius: int = 8) -> str:
    idx = max(lineno_1based - 1, 0)
    start = max(0, idx - radius)
    end = min(len(lines), idx + radius + 1)
    return "".join(lines[start:end])


def _scan_case_conflicts(repo_root: Path) -> tuple[list[list[str]], int]:
    kindred_root = repo_root / "kindred"
    files = _iter_tree_files(repo_root, kindred_root)
    case_map: dict[str, set[str]] = {}
    for path in files:
        rel = _rel_posix(repo_root, path)
        key = rel.lower()
        case_map.setdefault(key, set()).add(rel)

    groups: list[list[str]] = []
    for key in sorted(case_map):
        variants = sorted(case_map[key])
        if len(variants) > 1:
            groups.append(variants)
    return groups, len(groups)


def _scan_file(repo_root: Path, path: Path) -> tuple[list[Finding], dict[str, int]]:
    rel = _rel_posix(repo_root, path)
    text = _read_text(path)
    lines = text.splitlines(keepends=True)

    findings: list[Finding] = []
    counts = {
        "file_usage_total": 0,
        "file_usage_unguarded": 0,
        "posix_literal_hits": 0,
        "qt_resource_literal_hits": 0,
        "dynamic_import_hits": 0,
        "fs_resource_hits": 0,
        "qt_env_hits": 0,
        "likely_break_total": 0,
    }

    def add_finding(f: Finding) -> None:
        findings.append(f)
        if f.likely_break:
            counts["likely_break_total"] += 1

    for i, line in enumerate(lines, start=1):
        if QT_ENV_RE.search(line):
            counts["qt_env_hits"] += 1
            add_finding(
                Finding(
                    rule_id="L6 QtEnvVar",
                    severity="INFO",
                    file_line=f"{rel}:{i}",
                    message=f"Qt env var referenced in code: {line.strip()}",
                    suggested_fix="Prefer packaging/runtime defaults; if needed, set env vars in launcher/installer with clear docs.",
                    likely_break=False,
                )
            )

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        add_finding(
            Finding(
                rule_id="L0 ParseError",
                severity="WARN",
                file_line=f"{rel}:{getattr(exc, 'lineno', 1) or 1}",
                message=f"Python parse failed: {exc.msg}",
                suggested_fix="Fix syntax so audits can analyze this file deterministically.",
                likely_break=False,
            )
        )
        return sorted(findings, key=lambda f: f.file_line), counts

    path_context_names = {
        "open",
        "Path",
        "pathlib.Path",
        "os.path.join",
        "os.path.exists",
        "os.path.isfile",
        "os.path.isdir",
        "os.path.isabs",
        "os.path.abspath",
        "os.path.realpath",
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copytree",
        "shutil.move",
    }
    path_context_attrs = {
        "open",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "exists",
        "is_file",
        "is_dir",
        "glob",
        "rglob",
        "mkdir",
        "unlink",
        "rename",
        "replace",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__file__":
            counts["file_usage_total"] += 1
            window = _line_window(lines, getattr(node, "lineno", 1))
            guarded = bool(FROZEN_GUARD_RE.search(window))
            kind = "GUARDED_FROZEN" if guarded else "UNGUARDED"
            severity = "INFO" if guarded else "WARN"
            likely = not guarded
            if not guarded:
                counts["file_usage_unguarded"] += 1
            add_finding(
                Finding(
                    rule_id="L1 FileMagic",
                    severity=severity,
                    file_line=f"{rel}:{getattr(node, 'lineno', 1)}",
                    message=f"__file__ used ({kind})",
                    suggested_fix="Prefer importlib.resources for packaged resources; for path resolution, guard frozen builds (sys.frozen/_MEIPASS) and avoid assuming on-disk layout.",
                    likely_break=likely,
                )
            )

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name is None:
                continue

            base_call = call_name
            if base_call in path_context_names or (
                isinstance(node.func, ast.Attribute) and node.func.attr in path_context_attrs
            ):
                for arg in list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]:
                    lit = _const_str(arg)
                    if lit is None:
                        continue
                    if "://" in lit:
                        continue
                    if ":/" in lit:
                        counts["qt_resource_literal_hits"] += 1
                        add_finding(
                            Finding(
                                rule_id="L2 QtResourceRef",
                                severity="INFO",
                                file_line=f"{rel}:{getattr(arg, 'lineno', getattr(node, 'lineno', 1))}",
                                message=f'Qt resource-like literal used in path context: "{lit}"',
                                suggested_fix="Qt resource refs (:/...) are fine, but ensure the .qrc/.rcc pipeline is correct for packaging.",
                                likely_break=False,
                            )
                        )
                        continue
                    if "/" in lit:
                        counts["posix_literal_hits"] += 1
                        add_finding(
                            Finding(
                                rule_id="L2 PosixLiteral",
                                severity="WARN",
                                file_line=f"{rel}:{getattr(arg, 'lineno', getattr(node, 'lineno', 1))}",
                                message=f'POSIX-style "/" literal in path context for {base_call}: "{lit}"',
                                suggested_fix="Use Path(...)/... or os.path.join; avoid hard-coded '/' in Windows-targeted code paths.",
                                likely_break=True,
                            )
                        )

            if call_name in {"importlib.import_module", "__import__"}:
                mod_arg = node.args[0] if node.args else None
                lit = _const_str(mod_arg) if mod_arg is not None else None
                if lit is None:
                    counts["dynamic_import_hits"] += 1
                    add_finding(
                        Finding(
                            rule_id="L3 DynamicImport",
                            severity="SUSPECT",
                            file_line=f"{rel}:{getattr(node, 'lineno', 1)}",
                            message=f"Dynamic import via {call_name} with non-literal module name",
                            suggested_fix="PyInstaller may need hiddenimports; prefer explicit imports or enumerate possible modules.",
                            likely_break=True,
                        )
                    )

            if call_name == "pkgutil.iter_modules":
                counts["dynamic_import_hits"] += 1
                add_finding(
                    Finding(
                        rule_id="L3 DynamicImport",
                        severity="SUSPECT",
                        file_line=f"{rel}:{getattr(node, 'lineno', 1)}",
                        message="Dynamic module discovery via pkgutil.iter_modules",
                        suggested_fix="PyInstaller may miss discovered modules; ensure packaging hooks/hiddenimports cover these.",
                        likely_break=True,
                    )
                )

            if call_name == "importlib.metadata.entry_points":
                counts["dynamic_import_hits"] += 1
                add_finding(
                    Finding(
                        rule_id="L3 EntryPoints",
                        severity="SUSPECT",
                        file_line=f"{rel}:{getattr(node, 'lineno', 1)}",
                        message="Entrypoint discovery via importlib.metadata.entry_points (plugin-style loading)",
                        suggested_fix="Frozen builds may need explicit plugin inclusion; document/validate plugin loading strategy.",
                        likely_break=True,
                    )
                )

            if call_name.startswith("importlib.resources."):
                pkg_arg = node.args[0] if node.args else None
                if pkg_arg is not None and _const_str(pkg_arg) is None and not isinstance(
                    pkg_arg, (ast.Name, ast.Attribute)
                ):
                    counts["dynamic_import_hits"] += 1
                    add_finding(
                        Finding(
                            rule_id="L3 ImportlibResources",
                            severity="SUSPECT",
                            file_line=f"{rel}:{getattr(node, 'lineno', 1)}",
                            message=f"importlib.resources called with computed package reference: {call_name}",
                            suggested_fix="Prefer passing a module/package object or a literal package name; avoid computed package strings.",
                            likely_break=True,
                        )
                    )

            if call_name == "open" or call_name.endswith(".open"):
                file_arg = node.args[0] if node.args else None
                if file_arg is None:
                    continue
                lit = _const_str(file_arg)
                lineno = getattr(node, "lineno", 1)
                if _expr_contains_name(file_arg, "__file__"):
                    counts["fs_resource_hits"] += 1
                    add_finding(
                        Finding(
                            rule_id="L4 FsResource",
                            severity="WARN",
                            file_line=f"{rel}:{lineno}",
                            message="File opened via a path derived from __file__ (on-disk resource access)",
                            suggested_fix="Use importlib.resources for packaged data; avoid relying on package file layout at runtime.",
                            likely_break=True,
                        )
                    )
                if lit is not None:
                    normalized = lit.replace("\\", "/")
                    if normalized.startswith("data/") or "/data/" in normalized or normalized.startswith(
                        "kindred/data/"
                    ):
                        counts["fs_resource_hits"] += 1
                        add_finding(
                            Finding(
                                rule_id="L4 FsResource",
                                severity="WARN",
                                file_line=f"{rel}:{lineno}",
                                message=f'On-disk resource-like path opened directly: "{lit}"',
                                suggested_fix="Use importlib.resources.files(...).joinpath(...)/as_file(...) or ship as external user data (not package data).",
                                likely_break=True,
                            )
                        )

    for i, line in enumerate(lines, start=1):
        if "__file__" in line and ("'data'" in line or '"data"' in line) and "importlib.resources" not in line:
            counts["fs_resource_hits"] += 1
            add_finding(
                Finding(
                    rule_id="L4 FsResource",
                    severity="SUSPECT",
                    file_line=f"{rel}:{i}",
                    message="Line references __file__ and 'data' (heuristic on-disk packaged resource access)",
                    suggested_fix="Prefer importlib.resources for packaged data; if using __file__, ensure frozen/Windows behavior is explicitly guarded.",
                    likely_break=True,
                )
            )

    return sorted(findings, key=lambda f: (f.rule_id, f.file_line, f.message)), counts


def _run_scan(repo_root: Path, include_tools: bool) -> _ScanResult:
    py_files = _iter_python_files(repo_root, include_tools=include_tools)
    all_findings: list[Finding] = []
    agg = {
        "file_usage_total": 0,
        "file_usage_unguarded": 0,
        "posix_literal_hits": 0,
        "qt_resource_literal_hits": 0,
        "dynamic_import_hits": 0,
        "fs_resource_hits": 0,
        "qt_env_hits": 0,
        "likely_break_total": 0,
    }

    for path in py_files:
        findings, counts = _scan_file(repo_root, path)
        all_findings.extend(findings)
        for k in agg:
            agg[k] += int(counts.get(k, 0))

    case_groups, case_group_count = _scan_case_conflicts(repo_root)
    if case_group_count:
        for group in case_groups:
            rep = group[0]
            all_findings.append(
                Finding(
                    rule_id="L5 CaseConflict",
                    severity="WARN",
                    file_line=f"{rep}:1",
                    message=f"Case-insensitive collision group: {', '.join(group)}",
                    suggested_fix="Rename or remove conflicting paths so that lowercased paths are unique on Windows filesystems.",
                    likely_break=False,
                )
            )

    likely_break_total = (
        agg["file_usage_unguarded"]
        + agg["posix_literal_hits"]
        + agg["dynamic_import_hits"]
        + agg["fs_resource_hits"]
    )

    return _ScanResult(
        scanned_py_files=len(py_files),
        file_usage_total=agg["file_usage_total"],
        file_usage_unguarded=agg["file_usage_unguarded"],
        posix_literal_hits=agg["posix_literal_hits"],
        qt_resource_literal_hits=agg["qt_resource_literal_hits"],
        dynamic_import_hits=agg["dynamic_import_hits"],
        fs_resource_hits=agg["fs_resource_hits"],
        qt_env_hits=agg["qt_env_hits"],
        case_conflicts_groups=case_group_count,
        likely_break_total=likely_break_total,
        findings=sorted(all_findings, key=lambda f: (f.rule_id, f.file_line, f.message)),
        case_conflict_groups=case_groups,
    )


def _write_report(
    *,
    repo_root: Path,
    report_dir: Path,
    output: Path,
    include_tools: bool,
    result: _ScanResult,
) -> None:
    timestamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    scope = "kindred/** + tools/**" if include_tools else "kindred/**"

    counts_line = (
        "WINDOWS_PACKAGING_COUNTS"
        f"|scanned_py_files={result.scanned_py_files}"
        f"|file_usage_total={result.file_usage_total}"
        f"|file_usage_unguarded={result.file_usage_unguarded}"
        f"|posix_literal_hits={result.posix_literal_hits}"
        f"|dynamic_import_hits={result.dynamic_import_hits}"
        f"|fs_resource_hits={result.fs_resource_hits}"
        f"|case_conflicts_groups={result.case_conflicts_groups}"
        f"|likely_break_total={result.likely_break_total}"
    )

    out_lines: list[str] = []
    out_lines.append("Kindred Audit L: Windows packaging readiness (report-only)")
    out_lines.append(f"Timestamp (UTC): {timestamp_utc}")
    out_lines.append(f"Repo root: {repo_root}")
    out_lines.append(f"Report dir: {report_dir}")
    out_lines.append(f"Scan scope: {scope}")
    out_lines.append("")
    out_lines.append(counts_line)
    out_lines.append("")
    out_lines.append("Notes:")
    out_lines.append("- This audit detects common Windows/PyInstaller packaging hazards; it does not build an executable.")
    out_lines.append("- Summary status is derived from counts: WARN iff likely_break_total>0 OR case_conflicts_groups>0.")
    out_lines.append("")

    out_lines.append("Findings (paste-ready):")
    if not result.findings and result.case_conflicts_groups == 0:
        out_lines.append("L0 NoFindings | INFO | - | No findings in scan scope | -")
    else:
        out_lines.extend([f.render() for f in result.findings])

    out_lines.append("")
    out_lines.append("Case-conflict groups (repo tree under kindred/**):")
    if not result.case_conflict_groups:
        out_lines.append("- (none)")
    else:
        for group in result.case_conflict_groups:
            out_lines.append(f"- {', '.join(group)}")

    out_lines.append("")
    out_lines.append("Counts (informational):")
    out_lines.append(f"- scanned_py_files: {result.scanned_py_files}")
    out_lines.append(f"- file_usage_total: {result.file_usage_total}")
    out_lines.append(f"- file_usage_unguarded: {result.file_usage_unguarded}")
    out_lines.append(f"- posix_literal_hits: {result.posix_literal_hits}")
    out_lines.append(f"- qt_resource_literal_hits: {result.qt_resource_literal_hits}")
    out_lines.append(f"- dynamic_import_hits: {result.dynamic_import_hits}")
    out_lines.append(f"- fs_resource_hits: {result.fs_resource_hits}")
    out_lines.append(f"- qt_env_hits: {result.qt_env_hits}")
    out_lines.append(f"- case_conflicts_groups: {result.case_conflicts_groups}")
    out_lines.append(f"- likely_break_total: {result.likely_break_total}")

    output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Audit L: Windows packaging readiness hazards scanner")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="Also scan tools/** (excluding tools/audit/**). Default: kindred/** only.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report_dir = args.report_dir.resolve()
    output = args.output.resolve()
    include_tools = bool(args.include_tools)

    report_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = _run_scan(repo_root, include_tools=include_tools)
    _write_report(
        repo_root=repo_root,
        report_dir=report_dir,
        output=output,
        include_tools=include_tools,
        result=result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
