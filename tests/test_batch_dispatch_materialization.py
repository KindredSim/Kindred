from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kindred.gui.controllers.batch_dispatch_materialization import BatchDispatchMaterializationOwner


pytestmark = pytest.mark.unit


def test_materialize_initials_applies_pending_seed_before_preview_overlay() -> None:
    batch = MagicMock()
    slider = MagicMock()
    batch.batch_initials_for_row.return_value = {"A": 1.0, "B": 2.0}
    slider.preview_initials_for_row.return_value = {"A": 9.0, "B": 3.0}
    owner = BatchDispatchMaterializationOwner(batch=batch, slider=slider)

    initials = owner.materialize_initials(
        row=4,
        set_name="Set 1",
        fast_mode=True,
        pending_init_seed={"Set 1": {"A": 5.0}},
        pending_init_applied=False,
    )

    assert initials == {"A": 9.0, "B": 3.0}
    slider.preview_initials_for_row.assert_called_once_with(4, {"A": 5.0, "B": 2.0})


def test_materialize_initials_applies_pending_seed_without_preview_overlay_for_explicit_runs() -> None:
    batch = MagicMock()
    slider = MagicMock()
    batch.batch_initials_for_row.return_value = {"A": 1.0}
    owner = BatchDispatchMaterializationOwner(batch=batch, slider=slider)

    initials = owner.materialize_initials(
        row=0,
        set_name="Set 1",
        fast_mode=False,
        pending_init_seed={"Set 1": {"A": 5.0}},
        pending_init_applied=False,
    )

    assert initials == {"A": 5.0}
    slider.preview_initials_for_row.assert_not_called()
