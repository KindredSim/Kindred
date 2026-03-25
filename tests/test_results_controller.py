from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.gui.controllers.results_controller import (
    CachedBatchSelectionDisplayOutcome,
    ResultsController,
)


pytestmark = pytest.mark.unit


def _make_results_ui(plot: object) -> SimpleNamespace:
    return SimpleNamespace(
        parent=QtCore.QObject(),
        main_plot=lambda: plot,
        main_plot_has_data=lambda: True,
        main_plot_selected_series=lambda: ["A"],
        set_main_plot_selected_series=lambda _series: None,
        batch_name_for_id=lambda set_id: {"set1": "set1"}.get(str(set_id)),
        batch_id_for_name=lambda name: {"set1": "set1"}.get(str(name)),
        shown_batch_set_ids=lambda: ["set1"],
        focused_batch_set_id=lambda: "set1",
        selected_batch_set_ids=lambda: ["set1"],
        current_batch_row=lambda: 0,
        batch_set_id_for_row=lambda row: "set1" if int(row) == 0 else None,
        active_batch_cache_key=lambda: "cache-key",
        active_batch_valid_set_ids=lambda: None,
        active_batch_invalidated_set_ids=lambda: None,
        active_batch_selection=lambda: ("set1", "set1"),
        set_active_batch_selection=lambda *_args, **_kwargs: None,
        result_cache_store=lambda: {},
        set_main_plot_scalar_values=lambda _scalars: None,
        update_main_plot_statistics=lambda **_kwargs: None,
        main_plot_stats_table=lambda: MagicMock(),
        set_results_table=lambda _table: None,
        set_main_plot_data=lambda *_args, **_kwargs: None,
        show_simulation_tab=lambda: None,
        refresh_simulation_plot_views=lambda: None,
        schedule_main_plot_refresh=lambda _delays: None,
        set_status_text=lambda _text: None,
    )


def test_refresh_batch_plot_after_set_mutation_reuses_visible_plot_when_cache_is_missing() -> None:
    plot = MagicMock()
    plot._t = np.asarray([0.0, 1.0], dtype=float)
    plot._series = {"A": np.asarray([1.0, 0.5], dtype=float)}
    controller = ResultsController(_make_results_ui(plot))
    controller.display_cached_batch_selection_outcome = MagicMock(  # type: ignore[method-assign]
        return_value=CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")
    )

    controller.refresh_batch_plot_after_set_mutation()

    plot.set_data.assert_called_once()


def test_refresh_batch_plot_after_set_mutation_prefers_cached_selection_when_available() -> None:
    plot = MagicMock()
    plot._t = np.asarray([0.0, 1.0], dtype=float)
    plot._series = {"A": np.asarray([1.0, 0.5], dtype=float)}
    controller = ResultsController(_make_results_ui(plot))
    controller.display_cached_batch_selection_outcome = MagicMock(  # type: ignore[method-assign]
        return_value=CachedBatchSelectionDisplayOutcome(True)
    )

    controller.refresh_batch_plot_after_set_mutation()

    plot.set_data.assert_not_called()


def test_refresh_batch_plot_after_set_mutation_forwards_narrowed_valid_subset_without_fallback() -> None:
    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.active_batch_valid_set_ids = lambda: ("set2",)
    ui.active_batch_invalidated_set_ids = lambda: ("set1",)
    controller = ResultsController(ui)
    controller.display_cached_batch_selection_outcome = MagicMock(  # type: ignore[method-assign]
        return_value=CachedBatchSelectionDisplayOutcome(True, reason=None)
    )

    controller.refresh_batch_plot_after_set_mutation()

    controller.display_cached_batch_selection_outcome.assert_called_once_with(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=None,
        valid_set_ids=("set2",),
        invalidated_set_ids=("set1",),
        allow_fallback=False,
    )


def test_refresh_batch_plot_after_set_mutation_does_not_fallback_to_valid_sibling_outside_narrowed_subset() -> None:
    plot = MagicMock()
    plot._t = np.asarray([0.0, 1.0], dtype=float)
    plot._series = {"A": np.asarray([1.0, 0.5], dtype=float)}
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {"set1": "set1", "set2": "set2"}.get(str(set_id))
    ui.batch_id_for_name = lambda name: {"set1": "set1", "set2": "set2"}.get(str(name))
    ui.selected_batch_set_ids = lambda: ["set1"]
    ui.current_batch_row = lambda: 0
    ui.batch_set_id_for_row = lambda row: "set1" if int(row) == 0 else None
    ui.active_batch_valid_set_ids = lambda: ("set2",)
    ui.active_batch_selection = lambda: ("set2", "set2")
    ui.result_cache_store = lambda: {
        "cache-key::set2": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
            "algebra_scalars": {},
        }
    }
    ui.set_main_plot_data = MagicMock()
    controller = ResultsController(ui)

    controller.refresh_batch_plot_after_set_mutation()

    ui.set_main_plot_data.assert_not_called()
    plot.set_data.assert_not_called()


def test_display_cached_batch_selection_outcome_distinguishes_invalid_cache_entry_from_cache_miss() -> None:
    plot = MagicMock()
    store = {"cache-key::set1": {"t": np.asarray([0.0, 1.0], dtype=float), "series": object()}}
    controller = ResultsController(_make_results_ui(plot))

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=store,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(False, reason="invalid_cache_entry")


def test_display_cached_batch_selection_outcome_reports_cache_miss_when_entry_is_absent() -> None:
    plot = MagicMock()
    controller = ResultsController(_make_results_ui(plot))

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store={},
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")


def test_display_cached_batch_selection_outcome_still_displays_valid_cache_entry() -> None:
    plot = MagicMock()
    controller = ResultsController(_make_results_ui(plot))
    store = {
        "cache-key::set1": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        }
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=store,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(True, reason=None)


def test_display_cached_batch_selection_outcome_rejects_selected_cached_row_outside_latest_valid_subset() -> None:
    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {"set1": "set1", "set2": "set2"}.get(str(set_id))
    ui.batch_id_for_name = lambda name: {"set1": "set1", "set2": "set2"}.get(str(name))
    ui.active_batch_selection = lambda: ("set1", "set1")
    ui.set_main_plot_data = MagicMock()
    controller = ResultsController(ui)
    store = {
        "cache-key::set1": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
        },
        "cache-key::set2": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
            "algebra_scalars": {},
        },
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=store,
        valid_set_ids=("set2",),
        invalidated_set_ids=(),
        allow_fallback=False,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")
    ui.set_main_plot_data.assert_not_called()


def test_display_cached_batch_selection_outcome_reports_cache_miss_for_selected_row_outside_latest_valid_subset() -> None:
    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {"set1": "set1", "set2": "set2"}.get(str(set_id))
    ui.batch_id_for_name = lambda name: {"set1": "set1", "set2": "set2"}.get(str(name))
    ui.active_batch_selection = lambda: ("set2", "set2")
    controller = ResultsController(ui)
    store = {
        "cache-key::set2": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
            "algebra_scalars": {},
        },
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=store,
        valid_set_ids=("set2",),
        invalidated_set_ids=(),
        allow_fallback=False,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")


def test_display_cached_batch_selection_outcome_does_not_resurrect_selected_invalidated_row_with_stale_cache_entry() -> None:
    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {"set1": "set1", "set2": "set2"}.get(str(set_id))
    ui.batch_id_for_name = lambda name: {"set1": "set1", "set2": "set2"}.get(str(name))
    controller = ResultsController(ui)
    store = {
        "cache-key::set1": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([9.0, 9.0], dtype=float)},
            "algebra_scalars": {},
        },
        "cache-key::set2": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
            "algebra_scalars": {},
        },
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store=store,
        valid_set_ids=("set2",),
        invalidated_set_ids=("set1",),
        allow_fallback=False,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")


def test_display_cached_batch_selection_outcome_excludes_invalidated_selected_overlay_from_mixed_selection() -> None:
    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {"set1": "set1", "set2": "set2"}.get(str(set_id))
    ui.batch_id_for_name = lambda name: {"set1": "set1", "set2": "set2"}.get(str(name))
    ui.set_main_plot_data = MagicMock()
    controller = ResultsController(ui)
    store = {
        "cache-key::set1": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([9.0, 9.0], dtype=float)},
            "algebra_scalars": {},
        },
        "cache-key::set2": {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
            "algebra_scalars": {},
        },
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["set1", "set2"],
        prefer_set="set2",
        cache_store=store,
        valid_set_ids=("set2",),
        invalidated_set_ids=("set1",),
        allow_fallback=False,
    )

    assert outcome == CachedBatchSelectionDisplayOutcome(True, reason=None)
    assert ui.set_main_plot_data.call_args.kwargs["label"] == "set2"
    assert ui.set_main_plot_data.call_args.kwargs["overlays"] == []


def test_display_cached_batch_selection_forwards_valid_subset_without_fallback() -> None:
    plot = MagicMock()
    controller = ResultsController(_make_results_ui(plot))
    controller.display_cached_batch_selection_outcome = MagicMock(  # type: ignore[method-assign]
        return_value=CachedBatchSelectionDisplayOutcome(True, reason=None)
    )

    ok = controller.display_cached_batch_selection(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store={},
        valid_set_ids=("set2",),
        invalidated_set_ids=("set1",),
        allow_fallback=False,
    )

    assert ok is True
    controller.display_cached_batch_selection_outcome.assert_called_once_with(
        cache_key="cache-key",
        selected_sets=["set1"],
        prefer_set="set1",
        cache_store={},
        valid_set_ids=("set2",),
        invalidated_set_ids=("set1",),
        allow_fallback=False,
    )


def test_fallback_path_must_not_return_invalidated_cached_result_as_available() -> None:
    """The fallback active-batch-selection path must respect invalidated_ids.

    When no selected IDs produce an available cache entry, the fallback
    consults ``active_batch_selection()``.  If that fallback ID is in
    ``invalidated_set_ids``, the fallback must NOT return it as available,
    even if a cache entry exists.
    """
    from kindred.core.batch_simulation_cache import BatchSimulationCache

    plot = MagicMock()
    ui = _make_results_ui(plot)
    ui.batch_name_for_id = lambda set_id: {
        "id-fallback": "fallback-name",
        "id-selected": "selected-name",
    }.get(str(set_id))
    ui.batch_id_for_name = lambda name: {
        "fallback-name": "id-fallback",
        "selected-name": "id-selected",
    }.get(str(name))
    ui.selected_batch_set_ids = lambda: ["id-selected"]
    ui.active_batch_selection = lambda: ("id-fallback", "fallback-name")
    controller = ResultsController(ui)

    # Populate the store so the fallback ID has a valid cache entry.
    entry_key = BatchSimulationCache.entry_key("cache-key", "id-fallback")
    store = {
        entry_key: {
            "version": 1,
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "",
            "simulation_identity": {},
            "solver_config": {},
            "preview_batch_cache_token": "",
            "fallback_occurred": False,
            "fallback_message": None,
        },
    }

    outcome = controller.display_cached_batch_selection_outcome(
        cache_key="cache-key",
        selected_sets=["id-selected"],
        prefer_set="id-selected",
        cache_store=store,
        valid_set_ids=None,
        invalidated_set_ids=("id-fallback",),
        allow_fallback=True,
    )

    # The fallback ID is invalidated, so it must not be treated as available.
    assert outcome.displayed is False, (
        "Fallback path returned an invalidated cache entry as available"
    )
