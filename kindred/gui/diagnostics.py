from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def linux_rss_kb() -> Optional[int]:
    """Return RSS (kB) on Linux via /proc, otherwise None."""
    if not sys.platform.startswith("linux"):
        return None
    status_path = Path(os.sep) / "proc" / "self" / "status"
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def safe_len_find_children(root: object, child_type: type[_T]) -> int:
    """Best-effort len(root.findChildren(child_type)); returns -1 on error."""
    try:
        find_children = getattr(root, "findChildren", None)
        if not callable(find_children):
            return -1
        return int(len(find_children(child_type)))
    except Exception:
        return -1


def record_best_effort_failure(
    owner: object,
    key: str,
    *,
    message: str,
    exc: Optional[BaseException] = None,
    log: logging.Logger = logger,
    max_logs: int = 3,
    failures_attr: str = "_best_effort_failures",
    counts_attr: str = "_best_effort_failure_counts",
) -> None:
    failures = getattr(owner, failures_attr, None)
    if not isinstance(failures, set):
        failures = set()
        setattr(owner, failures_attr, failures)
    failures.add(str(key))

    counts = getattr(owner, counts_attr, None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(owner, counts_attr, counts)
    count = int(counts.get(key, 0)) + 1
    counts[key] = count

    if count <= int(max_logs):
        if exc is None:
            log.debug("%s (key=%s count=%d)", message, key, count)
        else:
            log.debug("%s (key=%s count=%d): %s", message, key, count, exc, exc_info=True)
