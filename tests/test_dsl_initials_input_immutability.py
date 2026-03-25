from __future__ import annotations

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism


@pytest.mark.unit
def test_parse_dsl_to_mechanism_does_not_mutate_initials_mapping() -> None:
    initials = {"A": 1.0}
    dsl = """
    init: A=2.0
    reaction: A -> B; k=1.0
    """

    mechanism = parse_dsl_to_mechanism(dsl, initials=initials)

    assert initials == {"A": 1.0}
    assert mechanism.species["A"].initial_conc == 2.0

