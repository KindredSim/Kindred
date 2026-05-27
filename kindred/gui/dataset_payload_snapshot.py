"""Dataset payload snapshot helpers for GUI dataset ownership boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


def copy_dataset_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return an owned dataset payload snapshot with copied numeric arrays."""
    source = dict(payload or {})
    species = source.get("species") if isinstance(source.get("species"), Mapping) else {}
    return {
        "t": np.asarray(source.get("t", []), dtype=float).copy(),
        "species": {
            str(name): np.asarray(values, dtype=float).copy()
            for name, values in species.items()
        },
        "metadata": deepcopy(dict(source.get("metadata") or {})),
    }
