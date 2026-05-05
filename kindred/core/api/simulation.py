from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import numpy as np

from kindred.core.exceptions import SimulationBuilderContractError

if TYPE_CHECKING:
    from kindred.core.simulation_preparation import BoundMechanism

__all__ = [
    "SimulationBuilder",
    "SimulationBuilderContractError",
    "coerce_simulation_builder",
    "prepare_bound_mechanism",
]

_REQUIRED_BUILDER_KWARGS = ("solver", "rtol", "atol")
_OPTIONAL_BUILDER_KWARGS = (
    "temperature_K",
    "use_sparse_jacobian",
    "wegscheider_cyclicity_enabled",
)


class SimulationBuilder(Protocol):
    def __call__(
        self,
        mechanism_text: str,
        param_names: list[str],
        *,
        solver: str,
        rtol: float,
        atol: float,
        temperature_K: Optional[float] = None,
        use_sparse_jacobian: Optional[bool] = None,
        wegscheider_cyclicity_enabled: Optional[bool] = None,
    ) -> Callable[[dict[str, float]], dict[str, np.ndarray]]: ...


def coerce_simulation_builder(builder: Callable[..., Any]) -> SimulationBuilder:
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        signature = None

    def _accepts_optional_kwarg(name: str) -> bool:
        if signature is None:
            return False
        if name in signature.parameters:
            return True
        return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    accepted_optional_kwargs = {
        name
        for name in _OPTIONAL_BUILDER_KWARGS
        if _accepts_optional_kwarg(name)
    }

    def _is_required_kwarg_contract_error(exc: TypeError) -> bool:
        msg = str(exc)
        if "unexpected keyword argument" not in msg:
            return False
        if not any(f"'{name}'" in msg for name in _REQUIRED_BUILDER_KWARGS):
            return False
        tb = exc.__traceback__
        return tb is not None and tb.tb_next is None

    def _wrapped(
        mechanism_text: str,
        param_names: list[str],
        *,
        solver: str,
        rtol: float,
        atol: float,
        temperature_K: Optional[float] = None,
        use_sparse_jacobian: Optional[bool] = None,
        wegscheider_cyclicity_enabled: Optional[bool] = None,
    ) -> Callable[[dict[str, float]], dict[str, np.ndarray]]:
        optional_kwargs: dict[str, object] = {}
        if "temperature_K" in accepted_optional_kwargs and temperature_K is not None:
            optional_kwargs["temperature_K"] = float(temperature_K)
        if "use_sparse_jacobian" in accepted_optional_kwargs and use_sparse_jacobian is not None:
            optional_kwargs["use_sparse_jacobian"] = bool(use_sparse_jacobian)
        if "wegscheider_cyclicity_enabled" in accepted_optional_kwargs and wegscheider_cyclicity_enabled is not None:
            optional_kwargs["wegscheider_cyclicity_enabled"] = bool(wegscheider_cyclicity_enabled)
        try:
            return builder(
                mechanism_text,
                list(param_names),
                solver=str(solver),
                rtol=float(rtol),
                atol=float(atol),
                **optional_kwargs,
            )
        except TypeError as exc:
            if _is_required_kwarg_contract_error(exc):
                raise SimulationBuilderContractError() from exc
            raise

    return _wrapped


def prepare_bound_mechanism(*args, **kwargs) -> "BoundMechanism":
    from kindred.core.simulation_preparation import prepare_bound_mechanism as _impl

    return _impl(*args, **kwargs)
