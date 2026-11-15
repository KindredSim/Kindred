"""Single owner for the current immutable project and archive baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace

from kindred.core.project_document import ProjectDocument
from kindred.core.project_presentation_state import (
    WorkspacePresentationState,
    default_workspace_presentation_state,
)


def _saved_choices(presentation: WorkspacePresentationState) -> tuple:
    # The active tab and each tab's current row are saved navigation, so
    # changing them alone is not an unsaved change.
    return tuple(
        (tab_id, replace(state, current_set_id=None))
        for tab_id, state in presentation.presentation_by_tab_id
    )


@dataclass(frozen=True, slots=True)
class SavedArchiveBaseline:
    document: ProjectDocument
    presentation: WorkspacePresentationState


@dataclass(frozen=True, slots=True)
class ProjectContentToken:
    """Opaque identity for one canonical content revision of the one session."""

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

        return ProjectContentToken(self._content_revision)

    def is_content_token_current(self, token: ProjectContentToken) -> bool:
        """Return whether ``token`` belongs to this session's current content."""

        return token.revision == self._content_revision

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
        return _saved_choices(presentation) != _saved_choices(
            self.saved_baseline.presentation
        )

    def is_dirty(
        self,
        presentation: WorkspacePresentationState,
    ) -> bool:
        return self.is_document_dirty or self.is_presentation_dirty(
            presentation
        )

    def replace_document(self, document: ProjectDocument) -> ProjectContentToken:
        """Publish one admitted document directly.

        A caller that showed a dialog between reading the project and
        publishing checks its content token first; the registry does not.  An
        unchanged document is an accepted no-op and returns the still-current
        token.
        """

        if document == self.document:
            return self.capture_content_token()
        self._document = document
        return self._advance_content_revision()

    def replace_project(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | None,
        baseline_document: ProjectDocument,
    ) -> ProjectContentToken:
        """Publish one admitted project identity and its saved baseline.

        The project controller checked the caller's token immediately before;
        ``path`` is already absolute.
        """

        self._document = document
        self._current_path = path
        self._saved_baseline = SavedArchiveBaseline(
            document=baseline_document,
            presentation=presentation,
        )
        return self._advance_content_revision()

    def mark_saved(
        self,
        path: str,
        presentation: WorkspacePresentationState,
    ) -> None:
        """Publish path and baseline without changing content."""

        self._current_path = path
        self._saved_baseline = SavedArchiveBaseline(
            document=self.document,
            presentation=presentation,
        )
