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
    reconcile_workspace_presentation_state,
    validate_workspace_presentation_state,
)

__all__ = [
    "ProjectContentToken",
    "PreparedDocumentMutation",
    "PreparedProjectInstall",
    "ProjectSession",
    "SavedArchiveBaseline",
]


def _validate_canonical_document(document: ProjectDocument) -> ProjectDocument:
    """Validate project truth and canonicalize initial-condition zero."""

    candidate = validate_project_document(document)
    tabs = tuple(
        replace(
            tab,
            initial_condition_sets=tuple(
                replace(
                    initial_set,
                    values=tuple(
                        (
                            species,
                            0.0 if value == 0.0 else value,
                        )
                        for species, value in initial_set.values
                    ),
                )
                for initial_set in tab.initial_condition_sets
            ),
        )
        for tab in candidate.tabs
    )
    return replace(candidate, tabs=tabs)


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


@dataclass(frozen=True, slots=True)
class PreparedDocumentMutation:
    """A fully validated canonical document ready for non-fallible commit."""

    document: ProjectDocument


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
        *,
        presentation: WorkspacePresentationState | None = None,
    ) -> None:
        validated_document = _validate_canonical_document(document)
        initial_presentation = (
            default_workspace_presentation_state(validated_document)
            if presentation is None
            else reconcile_workspace_presentation_state(
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
        self._content_revision = 0
        self._content_token_owner = object()

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

    def prepare_document_mutation(
        self,
        document: ProjectDocument,
    ) -> PreparedDocumentMutation:
        """Validate one proposed canonical document without publishing it."""

        return PreparedDocumentMutation(
            document=_validate_canonical_document(document),
        )

    def commit_document_mutation(
        self,
        prepared: PreparedDocumentMutation,
        *,
        expected_token: ProjectContentToken,
        fence_deletions: Callable[
            [ProjectDocument, ProjectDocument],
            None,
        ],
    ) -> ProjectContentToken | None:
        """Conditionally fence and publish one prepared document value.

        ``None`` means a newer canonical document or presentation superseded
        the caller.  An unchanged document is an accepted no-op and returns
        the still-current token.
        """

        if not isinstance(prepared, PreparedDocumentMutation):
            raise TypeError("prepared must be PreparedDocumentMutation.")
        if not callable(fence_deletions):
            raise TypeError("fence_deletions must be callable.")
        if not self.is_content_token_current(expected_token):
            return None
        candidate = prepared.document
        if candidate == self.document:
            return self.capture_content_token()
        fence_deletions(self.document, candidate)
        if not self.is_content_token_current(expected_token):
            return None
        self._state = replace(self._state, document=candidate)
        return self._advance_content_revision()

    def record_presentation_change(
        self,
        previous: WorkspacePresentationState,
        current: WorkspacePresentationState,
    ) -> ProjectContentToken:
        """Advance currentness after one accepted presentation mutation."""

        before = validate_workspace_presentation_state(
            self.document,
            previous,
        )
        after = validate_workspace_presentation_state(
            self.document,
            current,
        )
        if before == after:
            return self.capture_content_token()
        return self._advance_content_revision()

    def prepare_install(
        self,
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
        *,
        path: str | os.PathLike[str] | None,
    ) -> PreparedProjectInstall:
        """Validate every incoming session value without mutating live state."""

        candidate = _validate_canonical_document(document)
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

    def commit_install(
        self,
        prepared: PreparedProjectInstall,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Conditionally commit one already-prepared project replacement."""

        if not isinstance(prepared, PreparedProjectInstall):
            raise TypeError("prepared must be PreparedProjectInstall.")
        if not self.is_content_token_current(expected_token):
            return None
        self._state = prepared
        self._identity_generation += 1
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
        return True
