"""
Computational Simulator DSL facade.

This module is intentionally thin. Parsing and IR modeling live in
`dsl_parse.py`, preview assembly lives in `dsl_preview.py`, mechanism building
is delegated through `dsl_build.py`, and parameter extraction lives in
`dsl_parameter_scan.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .dsl_parse import (
    _ARROW_RE as _ARROW_RE,
    _bool_from_str as _bool_from_str,
    _extract_numeric_value as _extract_numeric_value,
    _parse_dsl_ir,
    _parse_kappa_directive as _parse_kappa_directive,
    _parse_keyvals as _parse_keyvals,
    _parse_members_expr as _parse_members_expr,
    _parse_standard_conc_directive as _parse_standard_conc_directive,
    _parse_stoich as _parse_stoich,
    _split_stoich_and_params as _split_stoich_and_params,
    DSLIR as DSLIR,
    DSLResult,
    ParsedStep as ParsedStep,
    parse_and_preview,
    parse_dsl,
)
from .dsl_types import StepPreview
from .errors import DSLError

if TYPE_CHECKING:
    from ..mechanism import Mechanism
    from ..units import UnitsModel


__all__ = [
    "DSLError",
    "StepPreview",
    "DSLResult",
    "parse_and_preview",
    "parse_dsl",
    "parse_dsl_to_mechanism",
    "extract_parameters_from_dsl",
    "extract_parameter_names_from_dsl",
]


def extract_parameters_from_dsl(text: str) -> list[object]:
    from .dsl_parameter_scan import extract_parameters_from_dsl as _impl

    return _impl(text)


def extract_parameter_names_from_dsl(text: str) -> set[str]:
    from .dsl_parameter_scan import extract_parameter_names_from_dsl as _impl

    return _impl(text)


def parse_dsl_to_mechanism(
    text: str,
    initials: dict[str, float] | None = None,
    *,
    units: "UnitsModel | None" = None,
) -> "Mechanism":
    ir = _parse_dsl_ir(text, units=units)
    from .dsl_build import build_mechanism_from_ir

    return build_mechanism_from_ir(ir, initials=initials)
