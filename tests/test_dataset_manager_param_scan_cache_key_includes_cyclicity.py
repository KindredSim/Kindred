from __future__ import annotations

from kindred.gui.controllers.dataset_manager import DatasetManager, DatasetManagerError
import pytest

pytestmark = pytest.mark.unit



def test_param_scan_cache_key_separates_wegscheider_cyclicity_mode():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = 1 / (Keq1 * Keq2)",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )

    flag = {"enabled": True}

    def get_cfg() -> dict[str, object]:
        return {"wegscheider_cyclicity_enabled": bool(flag["enabled"])}

    dm = DatasetManager(plot_tabs=None, dataset_resolver=lambda name: None, solver_settings_getter=get_cfg)

    flag["enabled"] = True
    on = dm.scan_mechanism_parameters(dsl)
    on_names = sorted(p["name"] for p in on)
    assert "Keq3" not in on_names

    flag["enabled"] = False
    off = dm.scan_mechanism_parameters(dsl)
    off_names = sorted(p["name"] for p in off)
    assert "Keq3" not in off_names


def test_param_scan_cache_does_not_reuse_off_mode_for_unresolved_on_mode():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )

    flag = {"enabled": False}

    def get_cfg() -> dict[str, object]:
        return {"wegscheider_cyclicity_enabled": bool(flag["enabled"])}

    dm = DatasetManager(plot_tabs=None, dataset_resolver=lambda name: None, solver_settings_getter=get_cfg)

    off = dm.scan_mechanism_parameters(dsl)
    assert {p["name"] for p in off} >= {"Keq1", "Keq2", "Keq3"}

    flag["enabled"] = True
    with pytest.raises(DatasetManagerError, match="unresolved Wegscheider cyclicity"):
        dm.scan_mechanism_parameters(dsl)


def test_param_scan_rejects_invalid_algebra_instead_of_reaction_only_fallback():
    from kindred.core.simulator.errors import DSLError

    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = Keq1",
            "param Keq3 = 1 / (Keq1 * Keq2)",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    dm = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda name: None,
        solver_settings_getter=lambda: {"wegscheider_cyclicity_enabled": True},
    )

    with pytest.raises(DSLError, match="Duplicate derived parameter definition"):
        dm.scan_mechanism_parameters(dsl)
