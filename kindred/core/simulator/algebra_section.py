from __future__ import annotations

from kindred.core.simulator.parameter_algebra_spec import classify_parameter_algebra_declaration

__all__ = ["is_algebra_line"]

def is_algebra_line(line: str) -> bool:
    return classify_parameter_algebra_declaration(line).kind in {
        "param",
        "let",
        "invalid_step_key_identifier",
        "unsupported_bare_assignment",
    }
