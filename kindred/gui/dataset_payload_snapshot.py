"""Dataset payload snapshot helpers for GUI dataset ownership boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from kindred.core.datasets.observation_payload import canonicalize_dataset_payload


def copy_dataset_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return an owned dataset payload snapshot with copied numeric arrays."""
    source = canonicalize_dataset_payload(payload)
    copied = {
        "observations": {
            str(name): {
                "t": source["observations"][str(name)]["t"].copy(),
                "y": source["observations"][str(name)]["y"].copy(),
            }
            for name in dict(source.get("observations") or {}).keys()
        },
        "metadata": deepcopy(dict(source.get("metadata") or {})),
    }
    for key, value in dict(source).items():
        if key in copied:
            continue
        copied[key] = deepcopy(value)
    return copied
