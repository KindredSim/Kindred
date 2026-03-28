from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtGui

from kindred.gui.plot_config import get_plot_panel_class, is_pyqtgraph_available
from kindred.gui.widgets.dataset_plot_panel import DatasetPlotPanel
from kindred.gui.widgets.dataset_overlay_panel import DatasetOverlayPanel
from kindred.gui.widgets.grid_plot_view import GridPlotView

pytestmark = pytest.mark.gui


def _rgb(color: QtGui.QColor) -> tuple[int, int, int]:
    return (int(color.red()), int(color.green()), int(color.blue()))


def _brush_rgb(brush: object) -> tuple[int, int, int]:
    if isinstance(brush, QtGui.QBrush):
        return _rgb(brush.color())
    color = getattr(brush, "color", lambda: None)()
    if isinstance(color, QtGui.QColor):
        return _rgb(color)
    raise AssertionError(f"Unsupported brush value: {type(brush)!r}")


def _pen_rgb(pen: object) -> tuple[int, int, int]:
    if isinstance(pen, QtGui.QPen):
        return _rgb(pen.color())
    if isinstance(pen, tuple) and len(pen) >= 3:
        return (int(pen[0]), int(pen[1]), int(pen[2]))
    color = getattr(pen, "color", lambda: None)()
    if isinstance(color, QtGui.QColor):
        return _rgb(color)
    raise AssertionError(f"Unsupported pen value: {type(pen)!r}")


def test_color_manager_exposes_24_slot_palette_with_deterministic_overflow() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    species_names = [f"S{i:02d}" for i in range(26)]
    manager.set_species_roster(species_names)

    palette = manager.species_palette()
    assert len(palette) == 24
    assert len({_rgb(color) for color in palette}) == 24
    assert (0, 0, 0) not in {_rgb(color) for color in palette}

    first_cycle = [_rgb(manager.get_species_color(name)) for name in species_names[:24]]
    assert len(set(first_cycle)) == 24

    assert _rgb(manager.get_species_color("S24")) == _rgb(manager.get_species_color("S00"))
    assert _rgb(manager.get_species_color("S25")) == _rgb(manager.get_species_color("S01"))


def test_color_manager_alias_uses_registered_real_species_instead_of_raw_dataset_suffix() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A"])

    assert manager.resolve_species_key("A_conc", known_species=["A_conc"]) == "A"
    assert _rgb(manager.get_species_color("A_conc", known_species=["A_conc"])) == _rgb(manager.get_species_color("A"))


def test_color_manager_preserves_real_species_names_that_end_with_conc_suffix() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A", "A_conc"])

    assert manager.resolve_species_key("A_conc") == "A_conc"
    assert _rgb(manager.get_species_color("A")) != _rgb(manager.get_species_color("A_conc"))


def test_color_manager_lookup_does_not_mutate_registered_species_roster() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A", "B"])
    before = manager.registered_species_names()

    _ = manager.get_species_color("A_conc", known_species=["A_conc"])
    _ = manager.get_species_color("C", known_species=["C"])

    assert manager.registered_species_names() == before
    assert _rgb(manager.get_species_color("A_conc", known_species=["A_conc"])) == _rgb(manager.get_species_color("A"))


def test_color_manager_current_roster_preview_colors_do_not_drift_after_unrelated_slot_growth() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_current_species_roster(["A", "B"])

    preview_before = {
        name: _rgb(manager.get_current_species_color(name))
        for name in ("A", "B")
    }

    manager.seed_species(["X", "Y"])

    preview_after = {
        name: _rgb(manager.get_current_species_color(name))
        for name in ("A", "B")
    }

    assert preview_after == preview_before


def test_color_manager_repeated_current_roster_refresh_avoids_seeded_visible_color_collisions() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_current_species_roster(["A", "B"])

    manager.seed_species(["X", "Y"])
    seeded = {
        name: _rgb(manager.get_species_color(name, known_species=[name]))
        for name in ("X", "Y")
    }

    manager.set_current_species_roster(["A", "B"])
    preview = {
        name: _rgb(manager.get_current_species_color(name))
        for name in ("A", "B")
    }

    assert len(set(preview.values())) == 2
    assert preview["A"] not in seeded.values()
    assert preview["B"] not in seeded.values()


def test_color_manager_first_commit_avoids_visible_collision_after_unrelated_slot_growth() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_current_species_roster(["A"])

    manager.seed_species(["X"])
    seeded = _rgb(manager.get_species_color("X", known_species=["X"]))

    manager.set_species_roster(["A"])

    assert _rgb(manager.get_species_color("A")) != seeded


def test_color_manager_first_committed_roster_matches_refreshed_preview_after_unrelated_slot_growth() -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_current_species_roster(["A", "B"])

    manager.seed_species(["X", "Y"])
    manager.set_current_species_roster(["A", "B"])

    preview = {
        name: _rgb(manager.get_current_species_color(name))
        for name in ("A", "B")
    }

    manager.set_species_roster(["A", "B"])

    committed = {
        name: _rgb(manager.get_species_color(name))
        for name in ("A", "B")
    }

    assert committed == preview


def test_main_window_overlay_swatches_use_current_mechanism_species_before_first_result_draw(main_window) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    payload = {
        "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
        "species": {"A_conc": np.asarray([1.0, 0.7, 0.3], dtype=float)},
    }
    main_window._right_panel._data_manager._datasets["exp.csv"] = payload
    main_window._on_dataset_loaded("exp.csv", payload)

    colors = main_window.main_plot()._overlay_panel.species_colors()
    assert manager.registered_species_names() == ("A", "B")
    assert _rgb(colors[("exp.csv", "A_conc")]) == _rgb(manager.get_species_color("A"))


def test_main_window_mechanism_edit_refreshes_overlay_swatches_without_dataset_reload_or_result_draw(
    main_window,
    qt_app,
) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    payload = {
        "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
        "species": {"X_conc": np.asarray([1.0, 0.7, 0.3], dtype=float)},
    }

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._right_panel._data_manager._datasets["exp.csv"] = payload
    main_window._on_dataset_loaded("exp.csv", payload)
    qt_app.processEvents()

    colors_before = main_window.main_plot()._overlay_panel.species_colors()
    assert manager.registered_species_names() == ("A", "B")
    assert _rgb(colors_before[("exp.csv", "X_conc")]) == _rgb(manager.get_non_species_color("X_conc"))

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: X -> Y; k=0.4")
    qt_app.processEvents()

    colors_after = main_window.main_plot()._overlay_panel.species_colors()
    assert manager.registered_species_names() == ("X", "Y")
    assert _rgb(colors_after[("exp.csv", "X_conc")]) == _rgb(manager.get_species_color("X"))


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_main_window_mechanism_edit_keeps_visible_overlay_markers_in_sync_with_swatches(
    main_window,
    qt_app,
) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    simulated = np.asarray([1.0, 0.7, 0.3], dtype=float)
    observed = np.asarray([1.1, 0.8, 0.4], dtype=float)
    payload = {
        "t": t,
        "species": {"A_conc": observed},
    }

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    main_window._right_panel._data_manager._datasets["exp.csv"] = payload
    main_window._on_dataset_loaded("exp.csv", payload)

    plot = main_window.main_plot()
    plot.set_data(t, {"A": simulated}, label="set1", overlays=[], owned_species=["A"])
    plot._overlay_panel.reconcile_selection(
        previous_selected_datasets=[],
        previous_enabled_species={},
        include_dataset_ids=["exp.csv"],
        ordered_dataset_ids=["exp.csv"],
        allow_default_include=True,
        emit=True,
    )
    qt_app.processEvents()

    assert _brush_rgb(plot._overlay_items[("exp.csv", "A")].opts["brush"]) == _rgb(manager.get_species_color("A"))

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: X -> Y; k=0.4")
    qt_app.processEvents()

    swatch_color = _rgb(plot._overlay_panel.species_colors()[("exp.csv", "A_conc")])

    assert swatch_color == _rgb(manager.get_non_species_color("A_conc"))
    assert ("exp.csv", "A") not in plot._overlay_items


def test_main_window_transient_parseable_mechanism_edits_do_not_shift_final_species_colors(
    main_window,
    qt_app,
) -> None:
    from kindred.gui.color_manager import ColorManager

    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    final_series = {
        "A": np.asarray([1.0, 0.8, 0.4], dtype=float),
        "B": np.asarray([0.2, 0.4, 0.6], dtype=float),
    }

    ColorManager.reset_for_tests()
    fresh_manager = ColorManager.instance()
    fresh_manager.set_species_roster(["A", "B"])
    expected = {
        name: _rgb(fresh_manager.get_species_color(name))
        for name in ("A", "B")
    }

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: X -> Y; k=0.1")
    qt_app.processEvents()
    assert manager.registered_species_names() == ("X", "Y")

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=0.2")
    qt_app.processEvents()

    plot = main_window.main_plot()
    plot.set_data(t, final_series, label="final", overlays=[], owned_species=["A", "B"])
    qt_app.processEvents()

    assert _rgb(manager.get_species_color("A")) == expected["A"]
    assert _rgb(manager.get_species_color("B")) == expected["B"]
    assert plot._colors["A"] == expected["A"]
    assert plot._colors["B"] == expected["B"]


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_dataset_plot_panel_uses_global_species_colors_for_data_and_model(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A", "B"])

    panel = DatasetPlotPanel(dataset_name="ds1")
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        a = np.asarray([1.0, 0.5, 0.2], dtype=float)
        b = np.asarray([0.4, 0.3, 0.1], dtype=float)

        panel.set_data(t, a, xlabel="Time", ylabel="A", all_species={"A": a, "B": b})
        panel.plot_simulation_results(t, {"A": a * 0.9, "B": b * 0.8})
        qt_app.processEvents()

        backend = panel._plot_panel
        assert _brush_rgb(backend._dataset_scatter_items["A"].opts["brush"]) == _rgb(manager.get_species_color("A"))
        assert _brush_rgb(backend._dataset_scatter_items["B"].opts["brush"]) == _rgb(manager.get_species_color("B"))
        assert _pen_rgb(backend._dataset_model_items["A"].opts["pen"]) == _rgb(manager.get_species_color("A"))
        assert _pen_rgb(backend._dataset_model_items["B"].opts["pen"]) == _rgb(manager.get_species_color("B"))
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_main_plot_owned_species_roster_shrink_preserves_surviving_species_colors(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    panel_cls = get_plot_panel_class()
    panel = panel_cls()
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        series_full = {
            "A": np.asarray([1.0, 0.8, 0.4], dtype=float),
            "B": np.asarray([0.6, 0.5, 0.3], dtype=float),
            "C": np.asarray([0.2, 0.3, 0.7], dtype=float),
        }
        series_shrunk = {
            "B": np.asarray([0.7, 0.55, 0.35], dtype=float),
            "C": np.asarray([0.25, 0.35, 0.8], dtype=float),
        }

        panel.set_data(t, series_full, label="set1", overlays=[], owned_species=["A", "B", "C"])
        qt_app.processEvents()

        before_b = _rgb(manager.get_species_color("B"))
        before_c = _rgb(manager.get_species_color("C"))
        assert manager.registered_species_names() == ("A", "B", "C")

        panel.set_data(t, series_shrunk, label="set2", overlays=[], owned_species=["B", "C"])
        qt_app.processEvents()

        assert manager.registered_species_names() == ("B", "C")
        assert _rgb(manager.get_species_color("B")) == before_b
        assert _rgb(manager.get_species_color("C")) == before_c
        assert panel._colors["B"] == before_b
        assert panel._colors["C"] == before_c
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_overlay_selector_swatches_do_not_present_removed_exact_species_as_current_owned_colors(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A", "B", "C"])
    historical_a = _rgb(manager.get_species_color("A"))
    manager.set_species_roster(["B", "C"])

    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(
            {
                "ds1": {
                    "species": {
                        "A": np.asarray([1.0, 0.5], dtype=float),
                        "B": np.asarray([0.2, 0.4], dtype=float),
                    }
                }
            }
        )

        colors = panel.species_colors()
        assert _rgb(colors[("ds1", "B")]) == _rgb(manager.get_species_color("B"))
        assert _rgb(colors[("ds1", "A")]) == _rgb(manager.get_non_species_color("A"))
        assert _rgb(colors[("ds1", "A")]) != historical_a
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_overlay_selector_swatches_canonicalize_dataset_aliases_to_species_colors(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A"])

    panel = DatasetOverlayPanel()
    try:
        panel.set_datasets(
            {
                "ds1": {"species": {"A": np.asarray([1.0, 0.5], dtype=float)}},
                "ds2": {"species": {"A_conc": np.asarray([0.9, 0.4], dtype=float)}},
            }
        )

        colors = panel.species_colors()
        expected = _rgb(manager.get_species_color("A"))
        assert _rgb(colors[("ds1", "A")]) == expected
        assert _rgb(colors[("ds2", "A_conc")]) == expected
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_main_plot_overlay_aliases_keep_species_color_and_vary_marker_by_dataset(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A"])

    panel_cls = get_plot_panel_class()
    panel = panel_cls()
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        simulated = np.asarray([3.0, 3.0, 3.0], dtype=float)
        observed_a = np.asarray([1.0, 0.6, 0.2], dtype=float)
        observed_alias = np.asarray([1.2, 0.8, 0.4], dtype=float)

        panel.set_data(t, {"A": simulated}, label="set1", overlays=[], owned_species=["A"])
        panel.set_overlay_catalog(
            {
                "ds1": {"t": t, "species": {"A": observed_a}},
                "ds2": {"t": t, "species": {"A_conc": observed_alias}},
            }
        )
        panel._overlay_panel.reconcile_selection(
            previous_selected_datasets=[],
            previous_enabled_species={},
            include_dataset_ids=["ds1", "ds2"],
            ordered_dataset_ids=["ds1", "ds2"],
            allow_default_include=True,
            emit=True,
        )
        qt_app.processEvents()

        expected = _rgb(manager.get_species_color("A"))
        item_one = panel._overlay_items[("ds1", "A")]
        item_two = panel._overlay_items[("ds2", "A")]

        assert _brush_rgb(item_one.opts["brush"]) == expected
        assert _brush_rgb(item_two.opts["brush"]) == expected
        assert item_one.opts["symbol"] != item_two.opts["symbol"]
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_main_plot_overlay_enabled_alias_preserves_resolved_dataset_column_color_with_exact_column_present(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()

    panel_cls = get_plot_panel_class()
    panel = panel_cls()
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        x_axis_species = np.asarray([1.0, 2.0, 3.0], dtype=float)
        simulated = np.asarray([3.0, 3.0, 3.0], dtype=float)
        observed_exact = np.asarray([1.0, 0.6, 0.2], dtype=float)
        observed_alias = np.asarray([1.2, 0.8, 0.4], dtype=float)

        panel.set_data(
            t,
            {"A": simulated, "B": x_axis_species},
            label="set1",
            owned_species=["A", "A_conc", "B"],
        )
        manager.set_species_roster(["A", "A_conc", "B"])
        panel.set_overlay_catalog(
            {
                "ds1": {
                    "t": t,
                    "species": {
                        "A": observed_exact,
                        "A_conc": observed_alias,
                        "B": x_axis_species,
                    },
                },
            }
        )
        panel._overlay_panel._selected["ds1"] = True
        panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
        panel.set_selected_series(["A"])
        panel._on_x_axis_changed("B")
        qt_app.processEvents()

        overlay = panel._active_overlay_series[0]
        expected = _rgb(manager.get_species_color("A_conc"))
        item = panel._overlay_items[("ds1", "A")]

        assert overlay.species == "A"
        assert overlay.resolved_x_column == "B"
        assert overlay.resolved_y_column == "A_conc"
        np.testing.assert_allclose(overlay.x, x_axis_species)
        np.testing.assert_allclose(overlay.y, observed_alias)
        assert _brush_rgb(item.opts["brush"]) == expected
    finally:
        panel.close()
        qt_app.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_grid_plot_view_keeps_species_color_stable_when_selected_subset_changes(qt_app) -> None:
    from kindred.gui.color_manager import ColorManager

    ColorManager.reset_for_tests()
    manager = ColorManager.instance()
    manager.set_species_roster(["A", "B"])

    view = GridPlotView()
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        a = np.asarray([1.0, 0.5, 0.2], dtype=float)
        b = np.asarray([0.2, 0.4, 0.9], dtype=float)

        view.set_datasets(
            [
                {
                    "name": "ds1",
                    "data_x": t,
                    "data_y": a,
                    "model_x": t,
                    "model_y": a * 0.9,
                    "model_series": {"A": a * 0.9, "B": b * 0.8},
                    "all_species": {"A": a, "B": b},
                    "current_species": "A",
                }
            ]
        )
        qt_app.processEvents()

        view.set_species_selection(["A", "B"])
        qt_app.processEvents()
        before = _pen_rgb(view._plot_series_items[0]["B::model"].opts["pen"])

        view.set_species_selection(["B"])
        qt_app.processEvents()
        after = _pen_rgb(view._plot_series_items[0]["B::model"].opts["pen"])

        expected = _rgb(manager.get_species_color("B"))
        assert before == expected
        assert after == expected
    finally:
        view.close()
        qt_app.processEvents()
