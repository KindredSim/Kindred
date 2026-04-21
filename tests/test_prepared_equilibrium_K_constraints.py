import pytest

from kindred.core.simulation_preparation import prepare_bound_mechanism
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

pytestmark = pytest.mark.unit



def test_prepared_mode_binds_equilibrium_derived_kr_when_K_is_explicit():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kf=1, K=4",
            "reaction: B -> C ; k=0.1",
            "init: A=1, B=0, C=0",
            "",
            "# Algebra",
            "param a = 4",
            "param k2 = kf1*a",
            "",
        ]
    )

    # Prepared/bound runs (slider fast-path) bind only what the UI declares mutable.
    # For explicit K equilibria, one of {kfN, krN} is derived and still must be mutable
    # so parameter algebra can enforce the equilibrium constraint each run.
    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kf1", "Keq1", "k2", "a"],
        temperature_K=300.0,
        initials={},
        use_advanced_dsl=True,
    )

    assert "kr1" in bound.bindings
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)
    assert bound.bindings["kr1"]() == pytest.approx(bound.bindings["kf1"]() / bound.bindings["Keq1"]())

    bound.bindings["kf1"].set(8.0)
    bound.bindings["Keq1"].set(4.0)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)
    assert bound.bindings["kr1"]() == pytest.approx(2.0)


def test_prepared_mode_binds_equilibrium_derived_kf_when_K_is_explicit():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kr=2, K=5",
            "reaction: B -> C ; k=0.1",
            "init: A=1, B=0, C=0",
        ]
    )

    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kr1", "Keq1"],
        temperature_K=300.0,
        initials={},
        use_advanced_dsl=True,
    )

    assert "kf1" in bound.bindings
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)
    assert bound.bindings["kf1"]() == pytest.approx(bound.bindings["kr1"]() * bound.bindings["Keq1"]())


def test_prepared_rhs_uses_updated_equilibrium_binding_values():
    """
    Regression: slider fast-path uses a prepared RHS with RateBinding objects.

    For explicit-K equilibria, one of kf/kr is derived and updated via parameter algebra
    each run; the RHS must read the latest binding values (not a cached constant), or
    slider-driven simulations will appear "stuck" until a full re-parse/run.
    """
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kf=1, K=4",
            "init: A=1, B=1",
        ]
    )

    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kf1", "Keq1"],
        temperature_K=300.0,
        initials={},
        use_advanced_dsl=True,
    )

    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)
    dy0 = bound.rhs(0.0, bound.y0)

    # Change K -> derived kr1 must update, and RHS must reflect it.
    bound.bindings["Keq1"].set(2.0)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)
    dy1 = bound.rhs(0.0, bound.y0)

    assert dy0[0] == pytest.approx(-(1.0 * 1.0 - 0.25 * 1.0))
    assert dy1[0] == pytest.approx(-(1.0 * 1.0 - 0.5 * 1.0))


def test_prepared_energy_fast_equilibrium_keeps_explicit_K_mutable():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kf=10, K=5",
            "init: A=1, B=0",
            "",
            "# Algebra",
            "param Keq1 = 4",
        ]
    )

    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["dG_eq_fast__feq__A__B"],
        temperature_K=300.0,
        initials={},
        use_advanced_dsl=True,
    )

    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)

    eq = bound.mechanism.equilibria[0]
    meta = getattr(eq, "metadata", {}) or {}

    assert float(eq.kf) == pytest.approx(10.0)
    assert float(eq.kr()) == pytest.approx(2.5)
    assert float(meta["Keq_input"]()) == pytest.approx(4.0)


def test_prepared_energy_fast_equilibrium_preserves_explicit_K_when_kf_changes():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B ; kf=10, K=5",
            "init: A=1, B=0",
        ]
    )

    bound = prepare_bound_mechanism(
        mechanism_text=dsl,
        param_names=["kf1", "dG_eq_fast__feq__A__B"],
        temperature_K=300.0,
        initials={},
        use_advanced_dsl=True,
    )

    eq = bound.mechanism.equilibria[0]
    meta = getattr(eq, "metadata", {}) or {}

    assert float(eq.kf()) == pytest.approx(10.0)
    assert float(eq.kr()) == pytest.approx(2.0)
    assert float(meta["Keq_input"]()) == pytest.approx(5.0)

    bound.bindings["kf1"].set(20.0)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)

    assert float(eq.kf()) == pytest.approx(20.0)
    assert float(eq.kr()) == pytest.approx(4.0)
    assert float(meta["Keq_input"]()) == pytest.approx(5.0)

    bound.bindings["kf1"].set(30.0)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=bound.mechanism, require_mutable=True)

    assert float(eq.kf()) == pytest.approx(30.0)
    assert float(eq.kr()) == pytest.approx(6.0)
    assert float(meta["Keq_input"]()) == pytest.approx(5.0)
