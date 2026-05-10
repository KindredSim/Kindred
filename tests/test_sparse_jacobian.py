import inspect

import numpy as np
import pytest

from kindred.core import sparse_jacobian
from kindred.core.mechanism import Mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.sparse_jacobian import HAS_SCIPY_SPARSE, build_sparse_jacobian, detect_sparsity_pattern
from kindred.core.simulator.dsl import parse_dsl_to_mechanism

pytestmark = pytest.mark.unit



def test_sparse_jacobian_runtime_callback_avoids_numpy_native_update_boundaries():
    source = inspect.getsource(sparse_jacobian.build_sparse_jacobian)
    helper_source = source.split("    def _monomial_derivatives_inplace", 1)[1].split(
        "    # Build list", 1
    )[0]
    jacobian_source = source.split("    def jacobian", 1)[1].split("    return jacobian", 1)[0]

    for token in ("np.take", "np.logical_and", "np.power", "np.prod"):
        assert token not in helper_source
    for token in ("np.any(", "np.add.at"):
        assert token not in jacobian_source


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
def test_sparse_jacobian_returns_stable_matrix_snapshots():
    mech = Mechanism()
    mech.add_species("A", 1.0)
    mech.add_species("B", 2.0)
    mech.add_species("C", 0.0)

    # A + B -> C; r = k * A * B
    mech.add_reaction(reactants={"A": 1.0, "B": 1.0}, products={"C": 1.0}, rate=2.0)

    info = detect_sparsity_pattern(mech)
    jac = build_sparse_jacobian(mech, info)

    species_names = mech.species_names()
    idx = {name: i for i, name in enumerate(species_names)}

    y1 = np.zeros(len(species_names), dtype=float)
    y1[idx["A"]] = 1.0
    y1[idx["B"]] = 2.0
    y1[idx["C"]] = 0.0

    J1 = jac(0.0, y1)
    J1_snapshot = J1.toarray().copy()

    y2 = np.zeros(len(species_names), dtype=float)
    y2[idx["A"]] = 2.0
    y2[idx["B"]] = 1.0
    y2[idx["C"]] = 0.0

    J2 = jac(0.0, y2)
    J2_snapshot = J2.toarray().copy()

    assert J2 is not J1
    assert J2.data is not J1.data
    np.testing.assert_allclose(J1.toarray(), J1_snapshot)
    assert not np.allclose(J1_snapshot, J2_snapshot)


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
def test_sparse_jacobian_overflowing_power_returns_inf_derivatives():
    mech = Mechanism()
    mech.add_species("A", 0.0)
    mech.add_species("B", 0.0)
    mech.add_reaction(reactants={"A": 2.0}, products={"B": 1.0}, rate=1.0)

    jac = build_sparse_jacobian(mech, detect_sparsity_pattern(mech))
    species_idx = {name: idx for idx, name in enumerate(mech.species_names())}

    y = np.zeros(len(species_idx), dtype=float)
    y[species_idx["A"]] = 1.0e200

    J = jac(0.0, y).toarray()

    assert np.isneginf(J[species_idx["A"], species_idx["A"]])
    assert np.isposinf(J[species_idx["B"], species_idx["A"]])


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
def test_sparse_jacobian_numerical_accuracy():
    dsl = "\n".join(
        [
            "reaction: A + B -> C; k=0.7",
            "reaction: C + D -> E; k=0.3",
            "reaction: E -> F; k=0.2",
            "equilibrium: B <-> D; kf=1.1; kr=0.4",
            "[A] = 1.2",
            "[B] = 0.9",
            "[C] = 0.1",
            "[D] = 1.1",
            "[E] = 0.2",
            "[F] = 0.0",
        ]
    )

    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    assert len(mechanism.reactions) >= 1
    assert len(mechanism.equilibria) >= 1

    rhs = build_ode_rhs_from_mechanism(mechanism)
    jac = build_sparse_jacobian(mechanism)

    species_names = mechanism.species_names()
    n = len(species_names)
    rng = np.random.default_rng(0)
    y = rng.uniform(0.2, 2.0, size=n).astype(float, copy=False)

    J_analytical = np.asarray(jac(0.0, y).toarray(), dtype=float)
    assert J_analytical.shape == (n, n)

    J_fd = np.zeros((n, n), dtype=float)
    eps_base = 1e-7
    for j in range(n):
        eps = eps_base * max(1.0, abs(float(y[j])))
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += eps
        y_minus[j] -= eps

        f_plus = np.asarray(rhs(0.0, y_plus), dtype=float)
        f_minus = np.asarray(rhs(0.0, y_minus), dtype=float)
        J_fd[:, j] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(J_analytical, J_fd, rtol=5e-7, atol=1e-10)


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
def test_sparse_jacobian_programmatic_nonfast_k_only_equilibrium_preserves_valid_anchor():
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15
    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=2.0,
    )

    jac = build_sparse_jacobian(mechanism)
    J = np.asarray(jac(0.0, np.array([1.0, 1.0], dtype=float)).toarray(), dtype=float)

    np.testing.assert_allclose(
        J,
        np.array([[-1.0, 0.5], [1.0, -0.5]], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
def test_sparse_jacobian_programmatic_reverse_anchor_equilibrium_preserves_valid_behavior():
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

    jac = build_sparse_jacobian(mechanism)
    J = np.asarray(jac(0.0, np.array([1.0, 1.0], dtype=float)).toarray(), dtype=float)

    np.testing.assert_allclose(
        J,
        np.array([[-10.0, 2.0], [10.0, -2.0]], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
@pytest.mark.parametrize(
    "K_value",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_sparse_jacobian_programmatic_nonfast_k_only_equilibrium_rejects_invalid_K(K_value):
    mechanism = Mechanism()
    mechanism.add_species("A", 0.0)
    mechanism.add_species("B", 0.0)
    mechanism.metadata["temperature_K"] = 298.15
    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        Keq=K_value,
    )

    with pytest.raises(ValueError, match="Keq must be positive and finite"):
        build_sparse_jacobian(mechanism)


@pytest.mark.skipif(not HAS_SCIPY_SPARSE, reason="scipy.sparse is required")
@pytest.mark.parametrize(
    "K_value",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_sparse_jacobian_programmatic_reverse_anchor_equilibrium_rejects_invalid_K(K_value):
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

    with pytest.raises(ValueError, match="Keq must be positive and finite"):
        build_sparse_jacobian(mechanism)
