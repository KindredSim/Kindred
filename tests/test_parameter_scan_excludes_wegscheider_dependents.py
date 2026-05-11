from kindred.gui.controllers.dataset_manager import DatasetManager
import pytest

pytestmark = pytest.mark.unit



def test_parameter_scan_rejects_unresolved_wegscheider_cyclicity():
    from kindred.gui.controllers.dataset_manager import DatasetManagerError

    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
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

    with pytest.raises(DatasetManagerError, match="unresolved Wegscheider cyclicity"):
        mgr.scan_mechanism_parameters(dsl)


def test_parameter_scan_excludes_symbolic_wegscheider_dependent_keq():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = 1 / (Keq1 * Keq2)",
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
    assert "Keq1" in names
    assert "Keq2" in names
    assert "Keq3" not in names


def test_parameter_scan_excludes_reverse_rate_derived_by_explicit_keq():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0; K=2.0",
            "init: A=1.0, B=0.0",
        ]
    )

    mgr = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda _name: None,
        mechanism_getter=None,
        simulation_runner=None,
        solver_settings_getter=lambda: {"wegscheider_cyclicity_enabled": False},
    )

    params = mgr.scan_mechanism_parameters(dsl)
    names = {p["name"] for p in params}

    assert "kf1" in names
    assert "Keq1" in names
    assert "kr1" not in names
