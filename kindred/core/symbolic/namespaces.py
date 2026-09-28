from __future__ import annotations


def symbolic_status_payload(
    *,
    kind: str,
    state: str,
    code: str,
    reason: str,
) -> dict[str, str]:
    return {
        "kind": str(kind),
        "state": str(state),
        "code": str(code),
        "reason": str(reason),
    }
