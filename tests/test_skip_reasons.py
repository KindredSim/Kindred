import ast
from pathlib import Path


GENERIC_REASONS = {
    "",
    "skip",
    "skipped",
    "not supported",
    "unsupported",
    "broken",
    "flaky",
    "todo",
    "tbd",
    "na",
    "n/a",
    "none",
    "unknown",
    "not implemented",
}


def _collect_string_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                constants[node.targets[0].id] = node.value.value
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                constants[node.target.id] = node.value.value
    return constants


def _resolve_reason(expr: ast.AST | None, constants: dict[str, str]) -> str | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in constants:
        return constants[expr.id]
    if isinstance(expr, ast.JoinedStr):
        parts: list[str] = []
        for value in expr.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = value.value
                if isinstance(inner, ast.Name) and inner.id in constants:
                    parts.append(constants[inner.id])
        if parts:
            return "".join(parts)
    return None


def _is_pytest_mark(call: ast.Call, attr: str) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == attr
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "pytest"
    )


def _is_pytest_call(call: ast.Call, name: str) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == name
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _extract_reason_data(
    node: ast.Call, kind: str, constants: dict[str, str]
) -> tuple[bool, str | None]:
    reason_expr = None
    if kind in {"pytest.mark.skip", "pytest.skip"}:
        if node.args:
            reason_expr = node.args[0]
        for kw in node.keywords or []:
            if kw.arg == "reason":
                reason_expr = kw.value
    elif kind == "pytest.mark.skipif":
        for kw in node.keywords or []:
            if kw.arg == "reason":
                reason_expr = kw.value
        if reason_expr is None and len(node.args) >= 2:
            reason_expr = node.args[1]
    elif kind == "pytest.importorskip":
        for kw in node.keywords or []:
            if kw.arg == "reason":
                reason_expr = kw.value
        if reason_expr is None and len(node.args) >= 2:
            reason_expr = node.args[1]
    provided = reason_expr is not None
    return provided, _resolve_reason(reason_expr, constants)


def _iter_skips():
    for path in Path("tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _collect_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kind = None
            if _is_pytest_mark(node, "skip"):
                kind = "pytest.mark.skip"
            elif _is_pytest_mark(node, "skipif"):
                kind = "pytest.mark.skipif"
            elif _is_pytest_call(node, "skip"):
                kind = "pytest.skip"
            elif _is_pytest_call(node, "importorskip"):
                kind = "pytest.importorskip"

            if kind:
                provided, reason = _extract_reason_data(node, kind, constants)
                yield path, node.lineno, kind, provided, reason


def test_skip_reasons_are_specific():
    issues: list[str] = []
    for path, lineno, kind, provided, reason in _iter_skips():
        location = f"{path}:{lineno}"
        if not provided:
            issues.append(f"{location} ({kind}) missing explicit reason")
            continue
        if reason is None:
            # Dynamic reason; assume it is meaningful because an explicit argument was provided.
            continue
        normalized = reason.strip()
        if not normalized:
            issues.append(f"{location} ({kind}) has empty reason")
            continue
        if normalized.lower() in GENERIC_REASONS:
            issues.append(f"{location} ({kind}) reason too generic: {reason!r}")

    assert not issues, "Skip reasons must be explicit and actionable:\n" + "\n".join(
        sorted(issues)
    )
