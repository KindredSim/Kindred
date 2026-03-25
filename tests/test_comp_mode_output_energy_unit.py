from __future__ import annotations

import re

import pytest

from kindred.core.simulator.computational_mode import compile_comp_spec, parse_comp_block


pytestmark = [pytest.mark.unit]


def _extract_global_energy_unit(generated_dsl: str) -> str:
    for raw in str(generated_dsl or "").splitlines():
        stripped = str(raw).strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("energy="):
            _, _, unit = stripped.partition("=")
            return unit.strip()
    raise AssertionError("Missing global energy=... directive in generated DSL")


def _extract_float(pattern: str, text: str, *, label: str) -> float:
    m = re.search(pattern, str(text), flags=re.MULTILINE)
    assert m is not None, f"Missing {label}"
    return float(m.group(1))


def test_compiled_generated_block_converts_energies_to_selected_output_unit():
    comp_body = "\n".join(
        [
            "comp: T = 298.15 K",
            "comp: pressure = 1 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0.0 std=1 M cref=1 M degeneracy=1",
            "comp: species B type=GS G=0.01 std=1 M cref=1 M degeneracy=1",
            "comp: species C type=GS G=-0.01 std=1 M cref=1 M degeneracy=1",
            "comp: species TS1 type=TS G=0.02 std=1 M cref=1 M degeneracy=1",
            "comp: channel A <-> C via TS1",
            "comp: rxn A <-> B",
        ]
    )
    spec = parse_comp_block(comp_body)

    compiled_kj = compile_comp_spec(spec, output_energy_unit="kJ/mol")
    compiled_kcal = compile_comp_spec(spec, output_energy_unit="kcal/mol")

    assert _extract_global_energy_unit(compiled_kj.generated_reaction_dsl) == "kJ/mol"
    assert _extract_global_energy_unit(compiled_kcal.generated_reaction_dsl) == "kcal/mol"

    dG_kj = _extract_float(r"^equilibrium:.*\bdG_eq=([-\d.+eE]+)\b", compiled_kj.generated_reaction_dsl, label="dG_eq (kJ/mol)")
    dG_kcal = _extract_float(r"^equilibrium:.*\bdG_eq=([-\d.+eE]+)\b", compiled_kcal.generated_reaction_dsl, label="dG_eq (kcal/mol)")
    assert (dG_kj / dG_kcal) == pytest.approx(4.184, rel=1e-6, abs=1e-9)

    ts_energy_kj = _extract_float(r"^state:\s*TS1\b.*\benergy=([-\d.+eE]+)\b", compiled_kj.generated_reaction_dsl, label="TS1 energy (kJ/mol)")
    ts_energy_kcal = _extract_float(r"^state:\s*TS1\b.*\benergy=([-\d.+eE]+)\b", compiled_kcal.generated_reaction_dsl, label="TS1 energy (kcal/mol)")
    assert (ts_energy_kj / ts_energy_kcal) == pytest.approx(4.184, rel=1e-6, abs=1e-9)

