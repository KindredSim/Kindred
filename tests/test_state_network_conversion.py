import math

import pytest

from kindred.core.simulator.state_model import StateNetwork, StateType
from kindred.core.simulator.state_network_converter import (
    StateNetworkConverter,
    convert_state_network_to_mechanism,
)


def _make_unimolecular_network():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"))
    net.add_state("TS1", kind=StateType.TS, energy=(50.0, "kJ/mol"))
    net.add_state("B", kind=StateType.GS, energy=(-10.0, "kJ/mol"))
    net.add_edge("A", "TS1")
    net.add_edge("TS1", "B")
    return net


def test_unimolecular_conversion_rates_and_species():
    net = _make_unimolecular_network()
    converter = StateNetworkConverter(temperature_K=300.0, kappa=1.0, C0_M=1.0)
    mech = converter.convert(net, initials={"A": 1.0, "B": 0.0})

    assert len(mech.species) == 2  # A, B only (TS not a species)
    assert len(mech.reactions) == 0
    assert len(mech.equilibria) == 1

    eq = mech.equilibria[0]
    kf = float(eq.kf)
    kr = float(eq.kr)

    dG_forward = net.get("TS1").energy_jmol - net.get("A").energy_jmol
    dG_reverse = net.get("TS1").energy_jmol - net.get("B").energy_jmol
    exp_kf = converter._eyring_rate(dG_forward, net.get("TS1").degeneracy / net.get("A").degeneracy)
    exp_kr = converter._eyring_rate(dG_reverse, net.get("TS1").degeneracy / net.get("B").degeneracy)

    assert math.isfinite(kf) and math.isfinite(kr)
    assert pytest.approx(kf, rel=1e-9) == exp_kf
    assert pytest.approx(kr, rel=1e-9) == exp_kr


def test_branching_ts_network_converts_multiple_paths():
    net = StateNetwork()
    net.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"))
    net.add_state("B", kind=StateType.GS, energy=(5.0, "kJ/mol"))
    net.add_state("C", kind=StateType.GS, energy=(10.0, "kJ/mol"))
    net.add_state("TS1", kind=StateType.TS, energy=(40.0, "kJ/mol"))
    net.add_state("TS2", kind=StateType.TS, energy=(55.0, "kJ/mol"))
    net.add_edge("A", "TS1")
    net.add_edge("TS1", "B")
    net.add_edge("B", "TS2")
    net.add_edge("TS2", "C")

    mech = convert_state_network_to_mechanism(net, initials={"A": 1.0, "B": 0.0, "C": 0.0}, temperature_K=298.15)

    assert len(mech.equilibria) == 2
    stoich_pairs = [{**eq.stoich_forward, **eq.stoich_back} for eq in mech.equilibria]
    assert any("A" in sp and "B" in sp for sp in stoich_pairs)
    assert any("B" in sp and "C" in sp for sp in stoich_pairs)
    for eq in mech.equilibria:
        assert math.isfinite(float(eq.kf))
        assert math.isfinite(float(eq.kr))


def test_bimolecular_members_convert_to_mechanism_with_std_product():
    net = StateNetwork()
    net.add_state("AB", kind=StateType.GS, energy=(0.0, "kJ/mol"), members=("A", "B"), std_conc_product_M=4.0)
    net.add_state("TS1", kind=StateType.TS, energy=(40.0, "kJ/mol"), std_conc_product_M=1.0)
    net.add_state("C", kind=StateType.GS, energy=(-5.0, "kJ/mol"), members=("C",), std_conc_product_M=1.0)
    net.add_edge("AB", "TS1")
    net.add_edge("TS1", "C")

    converter = StateNetworkConverter(temperature_K=300.0, kappa=1.0, C0_M=1.0)
    mech = converter.convert(net, initials={})

    assert sorted(mech.species_names()) == ["A", "B", "C"]
    assert len(mech.equilibria) == 1
    eq = mech.equilibria[0]

    assert eq.stoich_forward == {"A": 1.0, "B": 1.0}
    assert eq.stoich_back == {"C": 1.0}

    meta = getattr(eq, "metadata", {}) or {}
    assert int(meta.get("molecularity_fwd", 0)) == 2
    assert int(meta.get("molecularity_rev", 0)) == 1
    assert float(meta.get("std_conc_product_reactant", 0.0)) == pytest.approx(4.0, rel=0, abs=0.0)
    assert float(meta.get("std_conc_product_product", 0.0)) == pytest.approx(1.0, rel=0, abs=0.0)
    assert float(meta.get("std_conc_product_ts", 0.0)) == pytest.approx(1.0, rel=0, abs=0.0)

    dG_forward = net.get("TS1").energy_jmol - net.get("AB").energy_jmol
    dG_reverse = net.get("TS1").energy_jmol - net.get("C").energy_jmol
    exp_kf = converter._eyring_rate(dG_forward, 1.0, std_ratio=(1.0 / 4.0))
    exp_kr = converter._eyring_rate(dG_reverse, 1.0, std_ratio=1.0)

    assert float(eq.kf) == pytest.approx(exp_kf, rel=1e-12)
    assert float(eq.kr) == pytest.approx(exp_kr, rel=1e-12)
