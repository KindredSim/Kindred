from kindred.gui.controllers.dataset_manager import DatasetManager
import pytest

pytestmark = pytest.mark.unit



def test_parameter_scan_excludes_wegscheider_derived_parameters():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "equilibrium: B <-> C; kf=3.0; kr=1.0",
            "equilibrium: C <-> A; kf=1.0; kr=1.0",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )

    mgr = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda _name: None,
        mechanism_getter=None,
        simulation_runner=None,
        solver_settings_getter=lambda: {"wegscheider_cyclicity_enabled": True},
    )
    params = mgr.scan_mechanism_parameters(dsl)
    names = {p["name"] for p in params}
    assert "kr3" not in names

