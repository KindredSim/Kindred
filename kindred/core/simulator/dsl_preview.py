from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

from .common import molecularity
from .dsl_format import format_stoichiometry_side
from .dsl_types import StepPreview
from .kinetics import preview_line, rate_units


class _StepLike(Protocol):
    reactants: Dict[str, float]
    products: Dict[str, float]
    reversible: bool
    kf: float
    kr: Optional[float]
    model: str
    kappa: Optional[float]
    is_equilibrium: bool
    dG_eq_J_per_mol: Optional[float]
    Keq_input: Optional[float]
    explicit_rates: List[float]
    user_kf_explicit: bool
    user_kr_explicit: bool


def _preview_source_for_step(step: object) -> str:
    if bool(getattr(step, "is_equilibrium", False)):
        user_kf = bool(getattr(step, "user_kf_explicit", False))
        user_kr = bool(getattr(step, "user_kr_explicit", False))
        if user_kf and user_kr:
            return "explicit"
        if getattr(step, "Keq_input", None) is not None or getattr(step, "dG_eq_J_per_mol", None) is not None:
            return "mixed(kr=derived(K))"
        if list(getattr(step, "explicit_rates", []) or []):
            return "derived(k_fast)"
        return "derived"
    return "explicit"


def build_step_previews(
    steps: Sequence[object],
    *,
    temperature_K: float,
    kappa_global: float,
) -> List[StepPreview]:
    previews: List[StepPreview] = []
    for step in list(steps or []):
        react = dict(getattr(step, "reactants", {}) or {})
        prod = dict(getattr(step, "products", {}) or {})
        reversible = bool(getattr(step, "reversible", False))
        kf = float(getattr(step, "kf", 0.0))
        kr = getattr(step, "kr", None)
        kr_val = float(kr) if kr is not None else None
        n = molecularity(react)
        unit = rate_units(n)
        model = str(getattr(step, "model", "Eyring") or "Eyring")
        kappa = getattr(step, "kappa", None)
        kappa_val = float(kappa) if kappa is not None else float(kappa_global)
        previews.append(
            StepPreview(
                preview_line(
                    format_stoichiometry_side(react),
                    format_stoichiometry_side(prod),
                    reversible=reversible,
                    kf=kf,
                    kr=kr_val,
                    model=model,
                    unit=unit,
                    kappa=kappa_val,
                    T=float(temperature_K),
                    source=_preview_source_for_step(step),
                )
            )
        )
    return previews


def build_preview_lines(
    steps: Sequence[object],
    *,
    temperature_K: float,
    kappa_global: float,
) -> List[str]:
    return [p.text for p in build_step_previews(steps, temperature_K=temperature_K, kappa_global=kappa_global)]
