from __future__ import annotations


def is_excluded_rel(rel_posix: str) -> bool:
    if rel_posix.startswith("_audit_reports/"):
        return True
    if rel_posix.startswith("tools/audit/"):
        return True
    if "/_audit_reports/" in rel_posix:
        return True
    if "/tools/audit/" in rel_posix:
        return True
    return False

