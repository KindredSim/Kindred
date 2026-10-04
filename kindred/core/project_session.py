"""Single owner for the current immutable project and archive baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import (
    WorkspacePresentationState,
    default_workspace_presentation_state,
)

__all__ = [
    "ProjectContentToken",
    "ProjectSession",
    "SavedArchiveBaseline",
]


@dataclass(frozen=True, slots=True)
class SavedArchiveBaseline:
    document: ProjectDocument
    presentation: WorkspacePresentationState


@dataclass(frozen=True, slots=True)
class ProjectContentToken:
    """Opaque session-local identity for one canonical content revision."""

    _owner: object
    revision: int


class ProjectSession:
    """Own the current document, path, and last saved archive baseline."""

    def __init__(
        self,
        document: ProjectDocument,
    ) -> None:
        initial_presentation = default_workspace_presentation_state(document)
        self._document = document
        self._current_path: str | None = None
        self._saved_baseline = SavedArchiveBaseline(
            document=document,
            presentation=initial_presentation,
        )
        self._content_revision = 0
        self._content_token_owner = object()

    @property
    def document(self) -> ProjectDocument:
        return self._document

    @property
    def current_path(self) -> str | None:
        return self._current_path

    @property
    def saved_baseline(self) -> SavedArchiveBaseline:
        return self._saved_baseline

    def capture_content_token(self) -> ProjectContentToken:
        """Capture the exact current revision for conditional publication."""

        return ProjectContentToken(
            self._content_token_owner,
            self._content_revision,
        )

    def is_content_token_current(self, token: ProjectContentToken) -> bool:
        """Return whether ``token`` belongs to this session's current content."""

        if not isinstance(token, ProjectContentToken):
            raise TypeError("token must be ProjectContentToken.")
        return bool(
            token._owner is self._content_token_owner
            and token.revision == self._content_revision
        )

    def _advance_content_revision(self) -> ProjectContentToken:
        self._content_revision += 1
        return self.capture_content_token()

    @property
    def is_document_dirty(self) -> bool:
        return self.document != self.saved_baseline.document

    def is_presentation_dirty(
        self,
        presentation: WorkspacePresentationState,
    ) -> bool:
        return presentation != self.saved_baseline.presentation

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
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Publish one admitted document directly under the current token.

        ``None`` means a newer canonical document or presentation superseded
        the caller.  An unchanged document is an accepted no-op and returns
        the still-current token.
        """

        if not self.is_content_token_current(expected_token):
            return None
        if document == self.document:
            return self.capture_content_token()
        self._document = document
        return self._advance_content_revision()

    def replace_project(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | os.PathLike[str] | None,
        baseline_document: ProjectDocument,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Publish one admitted project identity and its saved baseline."""

        if not self.is_content_token_current(expected_token):
            return None
        normalized_path = (
            None
            if path is None
            else os.path.abspath(os.fsdecode(os.fspath(path)))
        )
        self._document = document
        self._current_path = normalized_path
        self._saved_baseline = SavedArchiveBaseline(
            document=baseline_document,
            presentation=presentation,
        )
        return self._advance_content_revision()

    def mark_saved(
        self,
        path: str | os.PathLike[str],
        presentation: WorkspacePresentationState,
        *,
        expected_token: ProjectContentToken,
    ) -> bool:
        """Conditionally publish path and baseline without changing content."""

        if not self.is_content_token_current(expected_token):
            return False
        normalized_path = os.path.abspath(os.fsdecode(os.fspath(path)))
        self._current_path = normalized_path
        self._saved_baseline = SavedArchiveBaseline(
            document=self.document,
            presentation=presentation,
        )
        return True
