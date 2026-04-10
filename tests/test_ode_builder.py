import numpy as np
import pytest

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.mechanism import Equilibrium, Mechanism
from kindred.core.simulator.common import derive_equilibrium_rates
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


pytestmark = pytest.mark.unit


def test_equilibrium_net_rates_avoid_inf_minus_inf_cancellation():
    kf = 1.0e308
    kr = 9.9996e307
    a0 = 10.0
    b0 = 10.0

    dsl = "\n".join(
        [
            f"equilibrium: A <-> B; kf={kf}; kr={kr}",
            f"initial: A={a0}",
            f"initial: B={b0}",
        ]
    )

    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names], dtype=float)

    dy = np.asarray(rhs(0.0, y0), dtype=float)
    assert dy.shape == y0.shape
    assert np.all(np.isfinite(dy))

    expected_r = np.longdouble(kf) * np.longdouble(a0) - np.longdouble(kr) * np.longdouble(b0)
    expected = np.array([-expected_r, expected_r], dtype=np.longdouble)
    assert np.allclose(dy, expected.astype(float), rtol=1e-9, atol=0.0)


def test_fast_equilibrium_K_only_derives_rates_via_common_policy():
    Keq = 2.0
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=Keq,
        kf=None,
        kr=None,
        fast=True,
        metadata={},
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    A_idx = species_names.index("A")
    B_idx = species_names.index("B")
    T = float(mechanism.metadata["temperature_K"])

    fe = derive_equilibrium_rates(Keq=Keq, T=T, explicit_rates=None)

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 1.0
    y[B_idx] = 0.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[B_idx]) == pytest.approx(float(fe.kf), rel=1e-12, abs=0.0)

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 0.0
    y[B_idx] = 1.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[A_idx]) == pytest.approx(float(fe.kr), rel=1e-12, abs=0.0)


def test_programmatic_nonfast_K_only_equilibrium_builds_and_evaluates():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=2.0,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    A_idx = species_names.index("A")
    B_idx = species_names.index("B")

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 1.0
    y[B_idx] = 0.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[B_idx]) == pytest.approx(1.0, rel=0, abs=0.0)

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 0.0
    y[B_idx] = 1.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[A_idx]) == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_programmatic_reverse_anchor_equilibrium_builds_and_evaluates():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=5.0,
        kf=None,
        kr=2.0,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    A_idx = species_names.index("A")
    B_idx = species_names.index("B")

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 1.0
    y[B_idx] = 0.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[B_idx]) == pytest.approx(10.0, rel=1e-12, abs=0.0)

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 0.0
    y[B_idx] = 1.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[A_idx]) == pytest.approx(2.0, rel=1e-12, abs=0.0)


def test_programmatic_forward_anchor_equilibrium_builds_and_evaluates():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=5.0,
        kf=10.0,
        kr=None,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    A_idx = species_names.index("A")
    B_idx = species_names.index("B")

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 1.0
    y[B_idx] = 0.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[B_idx]) == pytest.approx(10.0, rel=1e-12, abs=0.0)

    y = np.zeros(len(species_names), dtype=float)
    y[A_idx] = 0.0
    y[B_idx] = 1.0
    dy = np.asarray(rhs(0.0, y), dtype=float)
    assert float(dy[A_idx]) == pytest.approx(2.0, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("K_value", "label"),
    [
        (0.0, "zero"),
        (-1.0, "negative"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_programmatic_nonfast_K_only_equilibrium_rejects_invalid_K(K_value, label):
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=K_value,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    y = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match="Keq must be positive and finite"):
        rhs(0.0, y)


@pytest.mark.parametrize(
    ("K_value", "label"),
    [
        (0.0, "zero"),
        (-1.0, "negative"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_programmatic_forward_anchor_equilibrium_rejects_invalid_K(K_value, label):
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=K_value,
        kf=2.0,
        kr=None,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    y = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match="Keq must be positive and finite"):
        rhs(0.0, y)


@pytest.mark.parametrize(
    ("K_value", "label"),
    [
        (0.0, "zero"),
        (-1.0, "negative"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_programmatic_reverse_anchor_equilibrium_rejects_invalid_K(K_value, label):
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15

    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=K_value,
        kf=None,
        kr=2.0,
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    y = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match="Keq must be positive and finite"):
        rhs(0.0, y)


def test_runtime_equilibrium_without_usable_anchor_data_raises():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15
    mechanism.equilibria.append(
        Equilibrium(
            stoich_forward={"A": 1.0},
            stoich_back={"B": 1.0},
            kf=None,
            kr=None,
            Keq=None,
            fast=True,
            metadata={},
        )
    )

    rhs = build_ode_rhs_from_mechanism(mechanism)
    y = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match="usable kinetic and thermodynamic data"):
        rhs(0.0, y)
