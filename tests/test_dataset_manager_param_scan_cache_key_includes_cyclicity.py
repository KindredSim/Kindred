from __future__ import annotations

from kindred.gui.controllers.dataset_manager import DatasetManager


def test_param_scan_cache_key_separates_wegscheider_cyclicity_mode():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
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
    assert "kr3" not in on_names

    flag["enabled"] = False
    off = dm.scan_mechanism_parameters(dsl)
    off_names = sorted(p["name"] for p in off)
    assert "kr3" in off_names
