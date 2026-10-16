"""Deterministic selection of existing experimental observation rows."""

from __future__ import annotations

__all__ = ["compute_sampled_indices"]


def compute_sampled_indices(*, count: int, n_points: int) -> tuple[int, ...]:
    """Select ``n_points`` of ``count`` rows evenly, including both endpoints.

    Existing rows are selected; nothing is interpolated, averaged, or
    fabricated. ``n_points`` is at least 2.
    """
    if n_points >= count:
        return tuple(range(count))
    denominator = n_points - 1
    last = count - 1
    return tuple(
        (2 * index * last + denominator) // (2 * denominator)
        for index in range(n_points)
    )
