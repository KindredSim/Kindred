import numpy as np
import pytest

from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block
from kindred.core.algebra.errors import AlgebraNameError
from kindred.core.algebra.parser import parse_algebra
from kindred.core.algebra.symbol_table import build_algebra_symbol_table
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _context_from_mechanism(mech):
    t = np.linspace(0.0, 1.0, 5)
    species_names = mech.species_names()
    species_series = {sp: np.zeros_like(t) for sp in species_names}
    initials = {sp: mech.species[sp].initial_conc for sp in species_names}
    symtab = build_algebra_symbol_table(mech)
    return EvaluationContext(
        t=t,
        species_series=species_series,
        initials=initials,
        species_names=set(species_names),
        symtab=symtab,
        baseline=None,
    )


def test_algebra_uses_explicit_kf_kr_and_K():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=2.5",
            "equilibrium: B <-> C; kf=4.0; kr=2.0; K=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    ctx = _context_from_mechanism(mech)

    algebra = parse_algebra(
        "\n".join(
            [
                "# Algebra",
                "let ratio = kf2 / kr2",
                "let checkK = ratio - Keq2",
            ]
        )
    )

    series, scalars = evaluate_block(algebra, ctx)

    assert "ratio" in series and "checkK" in series
    assert series["ratio"].shape == ctx.t.shape
    assert np.allclose(series["ratio"], 2.0)
    assert np.allclose(series["checkK"], 0.0)
    assert scalars == {}


def test_algebra_sees_derived_kr_from_K():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=10.0; K=5.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    ctx = _context_from_mechanism(mech)

    algebra = parse_algebra(
        "\n".join(
            [
                "# Algebra",
                "let derived = kr1 - (kf1 / Keq1)",
                "let forward_ok = kf1 - 10.0",
            ]
        )
    )

    series, scalars = evaluate_block(algebra, ctx)

    assert np.allclose(series["derived"], 0.0)
    assert np.allclose(series["forward_ok"], 0.0)
    assert scalars == {}


def test_algebra_unknown_symbol_raises():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mech = parse_dsl_to_mechanism(dsl, initials={})
    ctx = _context_from_mechanism(mech)

    algebra = parse_algebra("# Algebra\nlet bad = missing_symbol + 1")

    with pytest.raises(AlgebraNameError):
        evaluate_block(algebra, ctx)
