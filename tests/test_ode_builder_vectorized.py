import numpy as np

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
import pytest

pytestmark = pytest.mark.unit



def _build_reference_rhs(mechanism):
    """Scalar reference implementation mirroring the pre-vectorized RHS."""
    species_names = mechanism.species_names()
    species_index = {name: idx for idx, name in enumerate(species_names)}

    steps = []
    for rxn in mechanism.reactions:
        steps.append(("reaction", rxn))
    for eq in mechanism.equilibria:
        steps.append(("equilibrium", eq))

    S = np.zeros((len(species_names), len(steps)))
    rate_funcs = []

    def _evaluate_scalar(value):
        if value is None:
            return None
        return float(value()) if callable(value) else float(value)

    for i_step, (step_type, step_obj) in enumerate(steps):
        if step_type == "reaction":
            S[:, i_step] = step_obj.net_stoich_vector(species_names)
            rate_obj = step_obj.rate
            k = float(rate_obj()) if callable(rate_obj) else float(rate_obj)
            reactant_info = [
                (species_index[name], order)
                for name, order in step_obj.rate_orders.items()
            ]

            def make_rate(constant, reactants):
                def rate(y):
                    r = constant
                    for idx, order in reactants:
                        r *= y[idx] ** order
                    return r

                return rate

            rate_funcs.append(make_rate(k, reactant_info))

        elif step_type == "equilibrium":
            fwd_vec = np.array(step_obj.forward_vector(species_names))
            back_vec = np.array(step_obj.back_vector(species_names))
            S[:, i_step] = back_vec - fwd_vec

            kf = _evaluate_scalar(step_obj.kf)
            kr = _evaluate_scalar(step_obj.kr)
            K = _evaluate_scalar(step_obj.Keq)

            if kf is None and kr is None:
                if K is None:
                    raise ValueError("Equilibrium missing kinetic parameters (need K or rates)")
                kf = 1.0
                kr = kf / K
            elif kf is None:
                if K is None:
                    raise ValueError("Equilibrium missing kf and equilibrium information to derive kf")
                kf = kr * K
            elif kr is None:
                if K is None:
                    raise ValueError("Equilibrium missing kr and equilibrium information to derive kr")
                kr = kf / K

            forward_terms = [
                (species_index[name], order) for name, order in step_obj.stoich_forward.items()
            ]
            reverse_terms = [
                (species_index[name], order) for name, order in step_obj.stoich_back.items()
            ]

            def make_eq_rate(kf_val, kr_val, forward_terms_val, reverse_terms_val):
                def rate(y):
                    r_fwd = kf_val
                    for idx, order in forward_terms_val:
                        r_fwd *= y[idx] ** order
                    r_rev = kr_val
                    for idx, order in reverse_terms_val:
                        r_rev *= y[idx] ** order
                    return r_fwd - r_rev

                return rate

            rate_funcs.append(make_eq_rate(kf, kr, forward_terms, reverse_terms))
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def rhs(t, y):
        rates = np.array([rf(y) for rf in rate_funcs])
        return S @ rates

    return rhs


def test_vectorized_rhs_matches_reference_irreversible():
    dsl = "\n".join(
        [
            "reaction: 2*A + B -> C; k=0.3",
            "reaction: C -> D; k=0.1",
            "initial: A=1.0",
            "initial: B=0.5",
            "initial: C=0.25",
            "initial: D=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs_vectorized = build_ode_rhs_from_mechanism(mechanism)
    rhs_reference = _build_reference_rhs(mechanism)

    species_names = mechanism.species_names()
    base_state = np.array([mechanism.species[name].initial_conc for name in species_names])
    test_states = [
        base_state,
        base_state * 1.5 + 0.1,
        np.linspace(0.1, 1.2, len(species_names)),
    ]

    for state in test_states:
        expected = rhs_reference(0.0, state)
        actual = rhs_vectorized(0.0, state)
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_vectorized_rhs_matches_reference_equilibrium():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=0.5",
            "reaction: B -> C; k=0.7",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.2",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs_vectorized = build_ode_rhs_from_mechanism(mechanism)
    rhs_reference = _build_reference_rhs(mechanism)

    species_names = mechanism.species_names()
    base_state = np.array([mechanism.species[name].initial_conc for name in species_names])
    test_states = [
        base_state,
        base_state + np.array([0.2, 0.4, 0.1]),
        np.array([0.8, 0.3, 0.6]),
    ]

    for state in test_states:
        expected = rhs_reference(0.0, state)
        actual = rhs_vectorized(0.0, state)
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
