"""Immutable contracts for transient Interactive Slider Preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections.abc import Mapping
from typing import Iterable

from kindred.core.result_repository import PreviewResultKey
from kindred.core.simulation_preparation import SimulationExecutionRequest
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.solver_types import SimulationPresentationBoundary

__all__ = [
    "InteractivePreviewFrame",
    "InteractivePreviewBatchIdentity",
    "InteractivePreviewRequest",
    "InteractivePreviewSampling",
    "InteractivePreviewSeries",
    "InteractivePreviewTarget",
    "InteractivePreviewTerminalOutcome",
    "InteractivePreviewTerminalState",
    "SliderInteractionKind",
]


def _required_text(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def _finite_tuple(values: Iterable[float], *, field: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values.")
    return normalized


class SliderInteractionKind(str, Enum):
    """The sole classification carried by a physical Slider value event."""

    INTERACTIVE_DRAG_SAMPLE = "interactive_drag_sample"
    EXACT_DRAG_RELEASE = "exact_drag_release"
    EXACT_TYPED_COMMIT = "exact_typed_commit"


class InteractivePreviewTerminalState(str, Enum):
    """Terminal facts emitted for an exact Preview settlement.

    These states are deliberately independent of repository entry presence.
    In particular, ``OBSOLETE`` means that native work returned after its
    generation had been fenced and must never become display authority.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    STOPPED = "stopped"
    OBSOLETE = "obsolete"


@dataclass(frozen=True, slots=True)
class InteractivePreviewBatchIdentity:
    """Stable identity shared by approximate frames and exact settlement."""

    tab_id: str
    generation: int
    batch_id: str
    target_set_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        tab_id = _required_text(self.tab_id, "batch tab_id")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("batch generation must be a positive int.")
        batch_id = _required_text(self.batch_id, "batch_id")
        targets = tuple(
            _required_text(value, "batch target_set_id")
            for value in self.target_set_ids
        )
        if not targets:
            raise ValueError("a Preview batch requires at least one target.")
        if len(set(targets)) != len(targets):
            raise ValueError("Preview batch targets must be unique.")
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "target_set_ids", targets)

    @property
    def key(self) -> tuple[str, int, str, tuple[str, ...]]:
        """Return the complete immutable identity used for display binding."""

        return (
            self.tab_id,
            self.generation,
            self.batch_id,
            self.target_set_ids,
        )


@dataclass(frozen=True, slots=True)
class InteractivePreviewTerminalOutcome:
    """Explicit exact Preview terminal fact for one batch target.

    The display consumes this contract instead of inferring completion from a
    repository entry changing or disappearing.  ``result_key`` is supplied
    only for a successful exact publication and remains the repository's
    immutable completed-result identity.
    """

    batch: InteractivePreviewBatchIdentity
    target_set_id: str
    state: InteractivePreviewTerminalState
    result_key: PreviewResultKey | None = None
    result_published: bool = False
    message: str = ""
    settles_display: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.batch, InteractivePreviewBatchIdentity):
            raise TypeError("batch must be InteractivePreviewBatchIdentity.")
        target = _required_text(self.target_set_id, "target_set_id")
        if target not in self.batch.target_set_ids:
            raise ValueError("terminal target must belong to its Preview batch.")
        state = self.state
        if not isinstance(state, InteractivePreviewTerminalState):
            try:
                state = InteractivePreviewTerminalState(str(state))
            except ValueError as exc:
                raise ValueError("invalid Interactive Preview terminal state.") from exc
        result_key = self.result_key
        if result_key is not None and not isinstance(result_key, PreviewResultKey):
            raise TypeError("result_key must be a PreviewResultKey or None.")
        if result_key is not None and (
            result_key.tab_id != self.batch.tab_id
            or result_key.set_id != target
        ):
            raise ValueError(
                "published exact result identity must match its batch target."
            )
        published = bool(self.result_published)
        settles_display = bool(self.settles_display)
        if state is InteractivePreviewTerminalState.SUCCESS:
            if result_key is None or not published:
                raise ValueError(
                    "successful exact Preview settlement requires a published result key."
                )
            if not settles_display:
                raise ValueError("successful Preview settlement must settle display.")
        elif result_key is not None or published:
            raise ValueError(
                "unsuccessful Preview settlement cannot publish a result key."
            )
        object.__setattr__(self, "target_set_id", target)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "result_published", published)
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "settles_display", settles_display)


@dataclass(frozen=True, slots=True)
class InteractivePreviewSampling:
    """Owner-scoped lightweight output policy for transient calculations."""

    output_points: int = 48

    def __post_init__(self) -> None:
        if type(self.output_points) is not int:
            raise TypeError("output_points must be an int.")
        if not 10 <= self.output_points <= 250:
            raise ValueError("output_points must be between 10 and 250.")

    def lightweight_integration_points(self, configured_points: object) -> int:
        """Resolve one target's lightweight solver grid from this policy."""

        if type(configured_points) is not int:
            raise TypeError("configured integration points must be an int.")
        if configured_points < 10:
            raise ValueError("configured integration points must be >= 10.")
        return max(10, min(self.output_points, configured_points // 2))

    def validate_execution_grid(self, grid: object) -> None:
        """Validate the grid a worker will consume for one preview target."""

        if not isinstance(grid, Mapping):
            raise TypeError("Interactive Preview solver grid must be a mapping.")
        points = grid.get("N")
        if type(points) is not int:
            raise TypeError("Interactive Preview solver grid N must be an int.")
        if not 10 <= points <= self.output_points:
            raise ValueError(
                "Interactive Preview solver grid N must be between 10 and "
                f"{self.output_points}."
            )


@dataclass(frozen=True, slots=True)
class InteractivePreviewTarget:
    """One immutable transient execution target; it has no repository key."""

    set_id: str
    set_name: str
    prepared_dsl: PreparedDslRequestSet
    execution_request: SimulationExecutionRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", _required_text(self.set_id, "set_id"))
        object.__setattr__(
            self,
            "set_name",
            _required_text(self.set_name, "set_name"),
        )
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be PreparedDslRequestSet.")
        if not isinstance(self.execution_request, SimulationExecutionRequest):
            raise TypeError("execution_request must be SimulationExecutionRequest.")
        object.__setattr__(
            self,
            "execution_request",
            self.execution_request.isolated_copy(),
        )


@dataclass(frozen=True, slots=True)
class InteractivePreviewRequest:
    """Newest-only transient request with exact document and Draft identity."""

    tab_id: str
    generation: int
    batch_id: str
    document_identity: tuple[int, str]
    science_identity: str
    draft_identity: str
    targets: tuple[InteractivePreviewTarget, ...]
    sampling: InteractivePreviewSampling

    def __post_init__(self) -> None:
        tab_id = _required_text(self.tab_id, "tab_id")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive int.")
        batch_id = _required_text(self.batch_id, "batch_id")
        identity = tuple(self.document_identity)
        if (
            len(identity) != 2
            or type(identity[0]) is not int
            or identity[0] < 0
            or _required_text(identity[1], "document identity tab") != tab_id
        ):
            raise ValueError("document_identity must exactly identify the request tab.")
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("an Interactive Preview requires at least one target.")
        if any(not isinstance(target, InteractivePreviewTarget) for target in targets):
            raise TypeError("targets must contain InteractivePreviewTarget values.")
        set_ids = tuple(target.set_id for target in targets)
        if len(set(set_ids)) != len(set_ids):
            raise ValueError("Interactive Preview target set IDs must be unique.")
        if not isinstance(self.sampling, InteractivePreviewSampling):
            raise TypeError("sampling must be InteractivePreviewSampling.")
        for target in targets:
            self.sampling.validate_execution_grid(
                target.execution_request.solver_config.grid
            )
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "document_identity", (identity[0], tab_id))
        object.__setattr__(
            self,
            "science_identity",
            _required_text(self.science_identity, "science_identity"),
        )
        object.__setattr__(
            self,
            "draft_identity",
            _required_text(self.draft_identity, "draft_identity"),
        )
        object.__setattr__(self, "targets", targets)

    @property
    def target_set_ids(self) -> tuple[str, ...]:
        """Return the immutable target roster captured by this batch."""

        return tuple(target.set_id for target in self.targets)

    @property
    def batch_identity(self) -> InteractivePreviewBatchIdentity:
        """Return the request's batch identity for exact settlement binding."""

        return InteractivePreviewBatchIdentity(
            tab_id=self.tab_id,
            generation=self.generation,
            batch_id=self.batch_id,
            target_set_ids=self.target_set_ids,
        )

@dataclass(frozen=True, slots=True)
class InteractivePreviewSeries:
    """One immutable named Y series in a transient frame."""

    series_id: str
    label: str
    values: tuple[float, ...]
    color: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "series_id",
            _required_text(self.series_id, "series_id"),
        )
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(
            self,
            "values",
            _finite_tuple(self.values, field=f"series {self.series_id!r}"),
        )
        if self.color is not None:
            object.__setattr__(self, "color", _required_text(self.color, "color"))


@dataclass(frozen=True, slots=True)
class InteractivePreviewFrame:
    """Immutable transient display data; ``x_values`` are canonical times."""

    tab_id: str
    set_id: str
    generation: int
    batch_id: str
    document_identity: tuple[int, str]
    draft_identity: str
    target_set_ids: tuple[str, ...]
    x_values: tuple[float, ...]
    segment_ids: tuple[int, ...]
    segment_keys: tuple[str, ...]
    discontinuity_ids: tuple[int, ...]
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...]
    series: tuple[InteractivePreviewSeries, ...]

    def __post_init__(self) -> None:
        tab_id = _required_text(self.tab_id, "tab_id")
        set_id = _required_text(self.set_id, "set_id")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive int.")
        batch_id = _required_text(self.batch_id, "batch_id")
        identity = tuple(self.document_identity)
        if (
            len(identity) != 2
            or type(identity[0]) is not int
            or identity[0] < 0
            or _required_text(identity[1], "document identity tab") != tab_id
        ):
            raise ValueError("document_identity must exactly identify the frame tab.")
        target_set_ids = tuple(
            _required_text(value, "target_set_id") for value in self.target_set_ids
        )
        if not target_set_ids:
            raise ValueError("an Interactive Preview frame requires a target roster.")
        if len(set(target_set_ids)) != len(target_set_ids):
            raise ValueError("Interactive Preview target roster must be unique.")
        if set_id not in target_set_ids:
            raise ValueError("frame set_id must belong to its target roster.")
        x_values = _finite_tuple(self.x_values, field="x_values")
        if len(x_values) > 1 and any(
            current <= previous
            for previous, current in zip(x_values, x_values[1:])
        ):
            raise ValueError("x_values must be strictly increasing.")
        segment_ids = tuple(int(value) for value in self.segment_ids)
        segment_keys = tuple(
            _required_text(value, "segment_key") for value in self.segment_keys
        )
        discontinuity_ids = tuple(int(value) for value in self.discontinuity_ids)
        if not (
            len(segment_ids)
            == len(segment_keys)
            == len(discontinuity_ids)
            == len(x_values)
        ):
            raise ValueError(
                "Interactive Preview topology must align with x_values."
            )
        if any(value < 0 for value in segment_ids + discontinuity_ids):
            raise ValueError("Interactive Preview topology IDs must be nonnegative.")
        if any(
            current < previous
            for previous, current in zip(segment_ids, segment_ids[1:])
        ) or any(
            current < previous
            for previous, current in zip(
                discontinuity_ids,
                discontinuity_ids[1:],
            )
        ):
            raise ValueError("Interactive Preview topology IDs must be ordered.")
        boundaries = tuple(self.presentation_boundaries)
        if any(
            not isinstance(boundary, SimulationPresentationBoundary)
            for boundary in boundaries
        ):
            raise TypeError(
                "presentation_boundaries must contain only "
                "SimulationPresentationBoundary values."
            )
        series = tuple(self.series)
        if not series:
            raise ValueError("an Interactive Preview frame requires series.")
        if any(not isinstance(item, InteractivePreviewSeries) for item in series):
            raise TypeError("series must contain InteractivePreviewSeries values.")
        if any(len(item.values) != len(x_values) for item in series):
            raise ValueError("every series must align with x_values.")
        series_ids = tuple(item.series_id for item in series)
        if len(set(series_ids)) != len(series_ids):
            raise ValueError("Interactive Preview series IDs must be unique.")
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "set_id", set_id)
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "document_identity", (identity[0], tab_id))
        object.__setattr__(
            self,
            "draft_identity",
            _required_text(self.draft_identity, "draft_identity"),
        )
        object.__setattr__(self, "target_set_ids", target_set_ids)
        object.__setattr__(self, "x_values", x_values)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "presentation_boundaries", boundaries)
        object.__setattr__(self, "series", series)

    @property
    def batch_identity(self) -> InteractivePreviewBatchIdentity:
        """Return the physically displayable frame's batch identity."""

        return InteractivePreviewBatchIdentity(
            tab_id=self.tab_id,
            generation=self.generation,
            batch_id=self.batch_id,
            target_set_ids=self.target_set_ids,
        )
