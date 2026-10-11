"""Current-schema mechanism template persistence and management."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from PySide6.QtCore import QStandardPaths

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.io.paths import file_error_text

logger = logging.getLogger(__name__)

__all__ = ["Template", "TemplateManager"]


_TEMPLATE_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "category",
        "tags",
        "reactions_text",
        "created_at",
        "modified_at",
        "author",
    }
)
_TEMPLATE_STRING_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "category",
        "reactions_text",
        "created_at",
        "modified_at",
    }
)
_UNSET = object()


def _template_name(value: object) -> str:
    if type(value) is not str:
        raise TypeError("template name must be an exact str.")
    if not value or value != value.strip():
        raise ValueError("template name must be nonblank and edge-normalized.")
    return value


def _nonblank_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact str.")
    if not value.strip():
        raise ValueError(f"{label} must be nonblank.")
    return value


def _require_template_fields(data: Mapping[str, Any]) -> None:
    missing_fields = sorted(_TEMPLATE_FIELDS - set(data.keys()))
    if missing_fields:
        raise ValueError(
            "template payload is missing required field(s): "
            + ", ".join(missing_fields)
        )
    unknown_fields = sorted(set(data.keys()) - _TEMPLATE_FIELDS)
    if unknown_fields:
        raise ValueError(
            "template payload has unknown field(s): "
            + ", ".join(unknown_fields)
        )


def _validate_template_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    for field in sorted(_TEMPLATE_STRING_FIELDS):
        if type(data[field]) is not str:
            raise TypeError(f"template payload field {field!r} must be an exact str.")
    tags = data["tags"]
    if not isinstance(tags, list) or any(type(tag) is not str for tag in tags):
        raise TypeError("template payload field 'tags' must be a list of exact str.")
    author = data["author"]
    if author is not None and type(author) is not str:
        raise TypeError("template payload field 'author' must be an exact str or None.")
    source = MechanismAuthoringSource.from_parts(reactions_text=data["reactions_text"])
    return {
        "id": _nonblank_string(data["id"], "template ID"),
        "name": _template_name(data["name"]),
        "description": str(data["description"]),
        "category": _nonblank_string(data["category"], "template category"),
        "tags": list(tags),
        "reactions_text": source.reactions_text,
        "created_at": _nonblank_string(data["created_at"], "template created_at"),
        "modified_at": _nonblank_string(data["modified_at"], "template modified_at"),
        "author": author,
    }


def _decode_template_json(path: Path) -> Mapping[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate template payload field {key!r}.")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=no_duplicate_keys)
    if not isinstance(data, Mapping):
        raise TypeError("template JSON root must be an object.")
    return data


@dataclass
class Template:
    """One complete current-schema template and its stable identity."""

    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    reactions_text: str
    created_at: str
    modified_at: str
    author: Optional[str] = None

    def __post_init__(self) -> None:
        validated = _validate_template_payload(asdict(self))
        for field_name, value in validated.items():
            setattr(self, field_name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached persistence payload."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Template":
        """Construct one template only from the strict current schema."""

        _require_template_fields(data)
        return cls(**data)

    @property
    def source(self) -> MechanismAuthoringSource:
        """Return the structurally admitted current authoring source."""

        return MechanismAuthoringSource.from_parts(reactions_text=self.reactions_text)

    @staticmethod
    def generate_id() -> str:
        """Allocate an opaque ID independent of the display name."""

        return uuid.uuid4().hex


class TemplateManager:
    """Own current-only template identity, collection admission, and writes."""

    def __init__(self):
        self._templates: Dict[str, Template] = {}
        self._templates_dir = self._get_templates_directory()
        self._write_disabled_reason: str | None = (
            "Template persistence is disabled until the stored collection "
            "has been admitted."
        )

    @property
    def templates_directory(self) -> Path:
        """The folder that holds the stored template files."""

        return self._templates_dir

    @property
    def write_disabled_reason(self) -> str | None:
        """Explain why persistence is unavailable for the admitted collection."""

        return self._write_disabled_reason

    def _require_writable_collection(self) -> None:
        reason = self._write_disabled_reason
        if reason is not None:
            raise RuntimeError(reason)

    def _record_storage_failure(self, action: str, exc: Exception) -> None:
        detail = file_error_text(exc, self._templates_dir).strip() or type(exc).__name__
        self._write_disabled_reason = (
            f"Template changes are disabled because {action}: {detail}"
        )

    def _get_templates_directory(self) -> Path:
        config_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        if not config_dir:
            config_dir = str(Path.home() / ".kindred")
        return (Path(config_dir) / "templates").resolve()

    def _templates_root(self, *, create: bool = False) -> Path:
        root = Path(self._templates_dir).resolve()
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"Template path is not a directory: {root}.")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _template_path(self, template_id: str, *, create_root: bool = False) -> Path:
        # Stored IDs come from file names in the directory or from uuid4.
        return self._templates_root(create=create_root) / f"{template_id}.json"

    @staticmethod
    def _name_key(name: str) -> str:
        return name.casefold()

    def _assert_available_name(
        self,
        name: str,
        *,
        except_template_id: str | None = None,
    ) -> None:
        candidate = self._name_key(name)
        for current_id, template in self._templates.items():
            if current_id != except_template_id and self._name_key(template.name) == candidate:
                raise ValueError(f"Template name already exists: {name!r}.")

    def _write_template(self, template: Template) -> None:
        self._require_writable_collection()
        payload = template.to_dict()
        target = self._template_path(payload["id"], create_root=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{payload['id']}.",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                handle.write("\n")
            os.replace(temporary_path, target)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def load_templates(self) -> None:
        """Validate the entire collection before publishing any of it."""

        try:
            root = self._templates_root()
            staged: Dict[str, Template] = {}
            names: set[str] = set()
            if root.exists():
                for template_file in sorted(
                    root.glob("*.json"), key=lambda item: item.name
                ):
                    if template_file.is_symlink() or not template_file.is_file():
                        raise ValueError(
                            "Template file is not a contained regular file: "
                            f"{template_file}"
                        )
                    template = Template.from_dict(
                        _decode_template_json(template_file)
                    )
                    # The file name owns the ID, so every ID is one safe file name.
                    if template_file.name != f"{template.id}.json":
                        raise ValueError(
                            "Template file name does not match its stable ID: "
                            f"{template_file.name}"
                        )
                    name_key = self._name_key(template.name)
                    if name_key in names:
                        raise ValueError(
                            "Duplicate case-insensitive template name: "
                            f"{template.name!r}."
                        )
                    staged[template.id] = template
                    names.add(name_key)
        except Exception as exc:
            self._record_storage_failure(
                "the stored templates could not be read",
                exc,
            )
            raise
        self._templates = staged
        self._write_disabled_reason = None
        logger.info("Loaded %d template(s)", len(staged))

    def delete_template(self, template_id: str) -> None:
        self._require_writable_collection()
        if template_id not in self._templates:
            raise KeyError(f"Unknown template ID: {template_id!r}.")
        template_file = self._template_path(template_id)
        if template_file.exists():
            template_file.unlink()
        del self._templates[template_id]
        logger.info("Deleted template: %s", template_id)

    def get_template(self, template_id: str) -> Optional[Template]:
        return self._templates.get(template_id)

    def get_all_templates(self) -> List[Template]:
        return list(self._templates.values())

    def get_categories(self) -> List[str]:
        return sorted({template.category for template in self._templates.values()})

    def search_templates(self, query: str) -> List[Template]:
        query_lower = str(query).casefold()
        return [
            template
            for template in self._templates.values()
            if (
                query_lower in template.name.casefold()
                or query_lower in template.description.casefold()
                or any(query_lower in tag.casefold() for tag in template.tags)
            )
        ]

    def create_template(
        self,
        name: str,
        source: MechanismAuthoringSource,
        description: str = "",
        category: str = "General",
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
    ) -> Template:
        """Create a new template; never select or overwrite an existing one."""

        self._require_writable_collection()
        normalized_name = _template_name(name)
        self._assert_available_name(normalized_name)
        template = Template(
            id=Template.generate_id(),
            name=normalized_name,
            description=description,
            category=category,
            tags=[] if tags is None else list(tags),
            reactions_text=source.reactions_text,
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
            author=author,
        )
        self._write_template(template)
        self._templates[template.id] = template
        logger.info("Created template: %s", template.id)
        return template

    def update_template(
        self,
        template_id: str,
        *,
        name: object = _UNSET,
        source: object = _UNSET,
        description: object = _UNSET,
        category: object = _UNSET,
        tags: object = _UNSET,
        author: object = _UNSET,
    ) -> Template:
        """Update fields of one existing template while preserving its ID."""

        self._require_writable_collection()
        current = self._templates.get(template_id)
        if current is None:
            raise KeyError(f"Unknown template ID: {template_id!r}.")
        candidate = Template(
            id=current.id,
            name=current.name if name is _UNSET else name,
            description=current.description if description is _UNSET else description,
            category=current.category if category is _UNSET else category,
            tags=current.tags if tags is _UNSET else tags,
            reactions_text=(
                current.reactions_text
                if source is _UNSET
                else source.reactions_text
            ),
            created_at=current.created_at,
            modified_at=datetime.now().isoformat(),
            author=current.author if author is _UNSET else author,
        )
        self._assert_available_name(candidate.name, except_template_id=current.id)
        self._write_template(candidate)
        self._templates[candidate.id] = candidate
        logger.info("Updated template: %s", candidate.id)
        return candidate

    def replace_template(self, template_id: str, template: Template) -> Template:
        """Replace one existing template with a complete same-ID value."""

        self._require_writable_collection()
        if template_id not in self._templates:
            raise KeyError(f"Unknown template ID: {template_id!r}.")
        candidate = template
        if candidate.id != template_id:
            raise ValueError("Replacement template must retain the target stable ID.")
        self._assert_available_name(candidate.name, except_template_id=template_id)
        self._write_template(candidate)
        self._templates[template_id] = candidate
        logger.info("Replaced template: %s", template_id)
        return candidate
