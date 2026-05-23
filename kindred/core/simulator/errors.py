"""
Enhanced error messages for DSL parsing with helpful suggestions.

Provides user-friendly error messages that include:
- Clear problem description
- Contextual information
- Suggestions for common mistakes
- Examples of correct syntax
"""

from __future__ import annotations

from typing import Optional, List

from kindred.core.exceptions import ErrorContext, ReactionValidationError


class DSLError(ReactionValidationError, ValueError):
    """
    Enhanced DSL validation or parsing error with helpful context.

    Attributes
    ----------
    message : str
        Main error message
    suggestion : str, optional
        Suggested fix or common solution
    examples : list of str, optional
        Valid syntax examples
    line_number : int, optional
        Line number where error occurred
    line_content : str, optional
        Content of the problematic line
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        examples: Optional[List[str]] = None,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None
    ):
        self.line_number = line_number
        self.line_content = line_content
        context = None
        if line_number is not None or line_content is not None:
            context = ErrorContext(line=line_number, line_text=line_content)
        super().__init__(
            message,
            code="E203",
            suggestion=suggestion,
            examples=examples,
            context=context,
        )

    def _build_message(self) -> str:
        """Build user-friendly error message."""
        parts = [self.message]

        # Add line context if available
        if self.line_number is not None and self.line_content is not None:
            parts.append("")
            parts.append(f"  Line {self.line_number}: {self.line_content}")
            parts.append("  " + "^" * min(len(self.line_content), 60))

        # Add suggestion
        if self.suggestion:
            parts.append("")
            parts.append(f"Suggestion: {self.suggestion}")

        # Add examples
        if self.examples:
            parts.append("")
            parts.append("Valid examples:")
            for example in self.examples:
                parts.append(f"  • {example}")

        return "\n".join(parts)


# Common error factories for frequently encountered issues

def invalid_temperature_error(value: str) -> DSLError:
    """Error for invalid temperature value."""
    return DSLError(
        f"Invalid temperature: T={value}",
        suggestion="Temperature must be positive (in Kelvin)",
        examples=[
            "T=298.15  (room temperature, 25°C)",
            "T=310.0   (body temperature, 37°C)",
            "T=373.15  (boiling point of water)"
        ]
    )
def invalid_number_error(value: str, field: str) -> DSLError:
    """Error for non-numeric value where number expected."""
    return DSLError(
        f"Invalid number for '{field}': {value}",
        suggestion="Provide a valid numeric value",
        examples=[
            f"{field}=1.5",
            f"{field}=2.5e-3",
            f"{field}=0.01"
        ]
    )


def missing_arrow_error(line: str) -> DSLError:
    """Error for reaction line missing arrow."""
    return DSLError(
        "Reaction/equilibrium must contain '->', '<->', or '<=>'",
        suggestion="Use '->' for irreversible or '<->' / '<=>' for reversible reactions",
        examples=[
            "reaction: A -> B; k=1.0",
            "reaction: A + B -> C; dG_act=75.5",
            "equilibrium: A <-> B; kf=1.0; Keq=2.0",
            "equilibrium: A <=> B; kf=1.0; Keq=2.0",
        ],
        line_content=line
    )


def missing_rate_parameters_error(reaction_type: str) -> DSLError:
    """Error for reaction missing rate constants or thermodynamic parameters."""
    if reaction_type == "Eyring":
        return DSLError(
            "Eyring reaction requires either 'k' or 'dG_act' parameter",
            suggestion="Provide explicit rate constant or activation free energy",
            examples=[
                "reaction: A -> B; k=1.5",
                "reaction: A -> B; dG_act=75.5",
                "reaction: A -> B; dG_act=75.5; T=310.0"
            ]
        )
    elif reaction_type == "Arrhenius":
        return DSLError(
            "Arrhenius reaction requires both 'A' and 'Ea' parameters",
            suggestion="Provide pre-exponential factor and activation energy",
            examples=[
                "reaction: A -> B; A=1.5e10; Ea=65.0",
                "reaction: A -> B; A=1e13; Ea=50.0; energy=kJ/mol"
            ]
        )
    else:
        return DSLError(
            "Reaction requires rate constant or thermodynamic parameters",
            examples=[
                "reaction: A -> B; k=1.5",
                "reaction: A -> B; dG_act=75.5"
            ]
        )


def missing_equilibrium_parameters_error() -> DSLError:
    """Error for equilibrium missing required parameters."""
    return DSLError(
        "Equilibrium requires kf and exactly one of kr or Keq/dG_eq",
        suggestion="Provide kf with either kr, Keq, or dG_eq, but not more than one reverse authority",
        examples=[
            "equilibrium: A <-> B; kf=1.5; Keq=2.0",
            "equilibrium: A <-> B; kf=1.5; dG_eq=-8.5",
            "equilibrium: A <-> B; kf=1.5; kr=0.75",
            "equilibrium: A <=> B; kf=1.5; Keq=2.0",
        ]
    )


def invalid_species_term_error(term: str) -> DSLError:
    """Error for malformed species term."""
    suggestion = (
        "Use format: 'A' or '2A' or '0.5A' (optionally with '*', e.g. '2*A'). "
        "Species names must start with a letter/underscore and contain only letters, digits, and underscores."
    )

    return DSLError(
        f"Invalid species term: {term}",
        suggestion=suggestion,
        examples=[
            "A",
            "2A",
            "0.5B",
            "A + 2 B -> 3 C",
            "A + 2*B -> 3*C",
        ]
    )


def empty_stoichiometry_error() -> DSLError:
    """Error for empty reactants or products."""
    return DSLError(
        "Empty stoichiometry - reactants or products missing",
        suggestion="Both sides of the reaction arrow must have at least one species",
        examples=[
            "A -> B",
            "A + B -> C",
            "2*A -> B + C"
        ]
    )


def non_positive_stoichiometry_error() -> DSLError:
    """Error for non-positive stoichiometric coefficients."""
    return DSLError(
        "Stoichiometric coefficients must be positive",
        suggestion="Use positive numbers for all coefficients",
        examples=[
            "A -> B",
            "2*A -> B",
            "0.5*A + B -> C"
        ]
    )


def non_integer_molecularity_error() -> DSLError:
    """Error for fractional molecularity."""
    return DSLError(
        "Molecularity must be an integer (sum of reactant coefficients)",
        suggestion="Ensure reactant stoichiometry sums to a whole number",
        examples=[
            "A -> B            (molecularity = 1)",
            "A + B -> C        (molecularity = 2)",
            "2*A -> B          (molecularity = 2)",
            "A + 2*B -> C      (molecularity = 3)"
        ]
    )


def reversible_arrhenius_missing_kr_error() -> DSLError:
    """Error for reversible Arrhenius without reverse rate."""
    return DSLError(
        "Reversible Arrhenius reaction needs exactly one of kr or Keq/dG_eq",
        suggestion="For reversible Arrhenius reactions, specify one reverse authority",
        examples=[
            "reaction: A <-> B; A=1e10; Ea=50.0; kr=0.5",
            "reaction: A <-> B; A=1e10; Ea=50.0; Keq=2.0",
            "reaction: A <-> B; A=1e10; Ea=50.0; dG_eq=-8.5",
            "reaction: A <=> B; A=1e10; Ea=50.0; Keq=2.0",
        ]
    )


def invalid_keyvalue_pair_error(chunk: str) -> DSLError:
    """Error for malformed key=value pair."""
    return DSLError(
        f"Expected key=value pair, got: {chunk}",
        suggestion="Use format: key=value with no spaces around '='",
        examples=[
            "k=1.5",
            "dG_act=75.5",
            "T=310.0",
            "energy=kJ/mol"
        ]
    )


def invalid_boolean_error(value: str, field: str) -> DSLError:
    """Error for invalid boolean value."""
    return DSLError(
        f"Invalid boolean for '{field}': {value}",
        suggestion="Use: 1/true/yes/on or 0/false/no/off",
        examples=[
            f"{field}=true",
            f"{field}=false",
            f"{field}=1",
            f"{field}=0"
        ]
    )
