"""Single owner for the current immutable project and archive baseline."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace

from kindred.core.project_document import (
    ProjectDocument,
    validate_project_document,
)
from kindred.core.project_presentation_state import (
    WorkspacePresentationState,
    default_workspace_presentation_state,
    validate_workspace_presentation_state,
)

__all__ = [
    "PreparedProjectInstall",
    "ProjectSession",
    "SavedArchiveBaseline",
]


@dataclass(frozen=True, slots=True)
class SavedArchiveBaseline:
    document: ProjectDocument
    presentation: WorkspacePresentationState


@dataclass(frozen=True, slots=True)
class PreparedProjectInstall:
    """Fully validated values for one fallible-free session install."""

    document: ProjectDocument
    path: str | None
    saved_baseline: SavedArchiveBaseline


class ProjectSession:
    """Own the current document, path, and last saved archive baseline."""

    def __init__(
        self,
        document: ProjectDocument,
        *,
        presentation: WorkspacePresentationState | None = None,
    ) -> None:
        validated_document = validate_project_document(document)
        initial_presentation = (
            default_workspace_presentation_state(validated_document)
            if presentation is None
            else validate_workspace_presentation_state(
                validated_document,
                presentation,
            )
        )
        self._state = PreparedProjectInstall(
            document=validated_document,
            path=None,
            saved_baseline=SavedArchiveBaseline(
                document=validated_document,
                presentation=initial_presentation,
            ),
        )
        self._identity_generation = 0

    @property
    def document(self) -> ProjectDocument:
        return self._state.document

    @property
    def current_path(self) -> str | None:
        return self._state.path

    @property
    def saved_baseline(self) -> SavedArchiveBaseline:
        return self._state.saved_baseline

    @property
    def identity_generation(self) -> int:
        """Session-only generation for real project identity replacement."""

        return self._identity_generation

    @property
    def is_document_dirty(self) -> bool:
        return self.document != self.saved_baseline.document

    def is_presentation_dirty(
        self,
        presentation: WorkspacePresentationState,
    ) -> bool:
        candidate = validate_workspace_presentation_state(
            self.document,
            presentation,
        )
        return candidate != self.saved_baseline.presentation

    def is_dirty(
        self,
        presentation: WorkspacePresentationState,
    ) -> bool:
        return self.is_document_dirty or self.is_presentation_dirty(
            presentation
        )

    def replace_document(
        self,
        document: ProjectDocument,
        *,
        fence_deletions: Callable[
            [ProjectDocument, ProjectDocument],
            None,
        ],
    ) -> bool:
        """Validate, fence deletions, then commit one canonical replacement."""

        if not callable(fence_deletions):
            raise TypeError("fence_deletions must be callable.")
        candidate = validate_project_document(document)
        if candidate == self.document:
            return False
        fence_deletions(self.document, candidate)
        self._state = replace(self._state, document=candidate)
        return True

    def prepare_install(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | os.PathLike[str] | None,
    ) -> PreparedProjectInstall:
        """Validate every incoming session value without mutating live state."""

        candidate = validate_project_document(document)
        candidate_presentation = validate_workspace_presentation_state(
            candidate,
            presentation,
        )
        normalized_path = (
            None
            if path is None
            else os.path.abspath(os.fsdecode(os.fspath(path)))
        )
        return PreparedProjectInstall(
            document=candidate,
            path=normalized_path,
            saved_baseline=SavedArchiveBaseline(
                document=candidate,
                presentation=candidate_presentation,
            ),
        )

    def commit_install(self, prepared: PreparedProjectInstall) -> None:
        """Commit one already-prepared replacement using assignments only."""

        if not isinstance(prepared, PreparedProjectInstall):
            raise TypeError("prepared must be PreparedProjectInstall.")
        self._state = prepared
        self._identity_generation += 1

    def mark_saved(
        self,
        path: str | os.PathLike[str],
        presentation: WorkspacePresentationState,
    ) -> None:
        candidate_presentation = validate_workspace_presentation_state(
            self.document,
            presentation,
        )
        normalized_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        self._state = PreparedProjectInstall(
            document=self.document,
            path=normalized_path,
            saved_baseline=SavedArchiveBaseline(
                document=self.document,
                presentation=candidate_presentation,
            ),
        )
