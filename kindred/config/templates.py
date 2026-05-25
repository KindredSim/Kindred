"""
Template manager for Kindred mechanism templates.

Allows users to save, load, and manage mechanism templates with metadata.
Templates are stored in the user's configuration directory as JSON files.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Mapping

from PySide6.QtCore import QStandardPaths

from kindred.core.mechanism_source import MechanismAuthoringSource

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
        "state_network_dsl",
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
        "state_network_dsl",
        "created_at",
        "modified_at",
    }
)


def _validate_template_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("template payload must be a mapping.")
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
    for field in sorted(_TEMPLATE_STRING_FIELDS):
        if not isinstance(data[field], str):
            raise TypeError(f"template payload field {field!r} must be a str.")
    tags = data["tags"]
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise TypeError("template payload field 'tags' must be a list of str.")
    author = data["author"]
    if author is not None and not isinstance(author, str):
        raise TypeError("template payload field 'author' must be a str or None.")
    return dict(data)


@dataclass
class Template:
    """
    Mechanism template with metadata.

    Attributes
    ----------
    id : str
        Unique template identifier (generated from name)
    name : str
        Human-readable template name
    description : str
        Template description
    category : str
        Template category (e.g., "Kinetics", "Atmospheric", "Polymer")
    tags : list of str
        Searchable tags
    reactions_text : str
        Reaction-layer DSL text
    state_network_dsl : str
        State Network DSL text
    created_at : str
        ISO 8601 timestamp
    modified_at : str
        ISO 8601 timestamp
    author : str, optional
        Template author name
    """
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    reactions_text: str
    state_network_dsl: str
    created_at: str
    modified_at: str
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Template:
        """Create template from dictionary."""
        return cls(**_validate_template_payload(data))

    @property
    def source(self) -> MechanismAuthoringSource:
        """Return the template as a complete authoring source."""
        return MechanismAuthoringSource.from_parts(
            reactions_text=self.reactions_text,
            state_network_dsl=self.state_network_dsl,
        )

    @staticmethod
    def generate_id(name: str) -> str:
        """Generate template ID from name."""
        # Convert to lowercase, replace spaces with underscores
        return name.lower().replace(' ', '_').replace('-', '_')


class TemplateManager:
    """
    Manager for user-defined mechanism templates.

    Templates are stored as JSON files in the user's configuration directory:
    - Linux: ~/.config/Kindred/templates/
    - Windows: C:/Users/<user>/AppData/Local/Kindred/templates/
    - macOS: ~/Library/Application Support/Kindred/templates/
    """

    def __init__(self):
        """Initialize template manager."""
        self._templates: Dict[str, Template] = {}
        self._templates_dir = self._get_templates_directory()

    def _get_templates_directory(self) -> Path:
        """Get templates directory path."""
        override = os.environ.get("KINDRED_TEMPLATES_DIR")
        if override:
            return Path(str(override)).expanduser().resolve()

        # Get platform-specific config directory
        config_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )

        if not config_dir:
            # Fallback to home directory
            config_dir = str(Path.home() / ".kindred")

        templates_dir = Path(config_dir) / "templates"
        return templates_dir

    def _ensure_directory_exists(self) -> None:
        """Ensure templates directory exists."""
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Templates directory: {self._templates_dir}")

    def load_templates(self) -> None:
        """Load all templates from disk."""
        self._templates.clear()

        if not self._templates_dir.exists():
            logger.debug("Templates directory does not exist yet: %s", self._templates_dir)
            return

        for template_file in self._templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                template = Template.from_dict(data)
                self._templates[template.id] = template
                logger.debug(f"Loaded template: {template.name}")

            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

        logger.info(f"Loaded {len(self._templates)} template(s)")

    def save_template(self, template: Template) -> None:
        """
        Save template to disk.

        Parameters
        ----------
        template : Template
            Template to save
        """
        template_file = self._templates_dir / f"{template.id}.json"

        try:
            self._ensure_directory_exists()
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, indent=2)

            self._templates[template.id] = template
            logger.info(f"Saved template: {template.name} -> {template_file}")

        except Exception as e:
            logger.error(f"Failed to save template {template.name}: {e}")
            raise

    def delete_template(self, template_id: str) -> None:
        """
        Delete template from disk.

        Parameters
        ----------
        template_id : str
            Template ID to delete
        """
        template_file = self._templates_dir / f"{template_id}.json"

        try:
            if template_file.exists():
                template_file.unlink()

            if template_id in self._templates:
                del self._templates[template_id]

            logger.info(f"Deleted template: {template_id}")

        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            raise

    def get_template(self, template_id: str) -> Optional[Template]:
        """
        Get template by ID.

        Parameters
        ----------
        template_id : str
            Template ID

        Returns
        -------
        Template or None
            Template if found, None otherwise
        """
        return self._templates.get(template_id)

    def get_all_templates(self) -> List[Template]:
        """
        Get all templates.

        Returns
        -------
        list of Template
            All loaded templates
        """
        return list(self._templates.values())

    def get_templates_by_category(self, category: str) -> List[Template]:
        """
        Get templates by category.

        Parameters
        ----------
        category : str
            Category name

        Returns
        -------
        list of Template
            Templates in the specified category
        """
        return [t for t in self._templates.values() if t.category == category]

    def get_categories(self) -> List[str]:
        """
        Get all template categories.

        Returns
        -------
        list of str
            Unique categories
        """
        categories = set(t.category for t in self._templates.values())
        return sorted(categories)

    def search_templates(self, query: str) -> List[Template]:
        """
        Search templates by name, description, or tags.

        Parameters
        ----------
        query : str
            Search query (case-insensitive)

        Returns
        -------
        list of Template
            Matching templates
        """
        query_lower = query.lower()
        results = []

        for template in self._templates.values():
            # Search in name, description, and tags
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)

        return results

    def create_template(
        self,
        name: str,
        source: MechanismAuthoringSource,
        description: str = "",
        category: str = "General",
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
    ) -> Template:
        """
        Create a new template.

        Parameters
        ----------
        name : str
            Template name
        source : MechanismAuthoringSource
            Complete mechanism source
        description : str, optional
            Template description
        category : str, optional
            Template category (default: "General")
        tags : list of str, optional
            Searchable tags
        author : str, optional
            Template author name

        Returns
        -------
        Template
            Created template
        """
        template_id = Template.generate_id(name)
        now = datetime.now().isoformat()
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")

        template = Template(
            id=template_id,
            name=name,
            description=description,
            category=category,
            tags=tags or [],
            reactions_text=source.reactions_text,
            state_network_dsl=source.state_network_dsl,
            created_at=now,
            modified_at=now,
            author=author,
        )

        self.save_template(template)
        return template

    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        source: Optional[MechanismAuthoringSource] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
    ) -> Optional[Template]:
        """
        Update an existing template.

        Parameters
        ----------
        template_id : str
            Template ID to update
        name : str, optional
            New name
        source : MechanismAuthoringSource, optional
            New complete mechanism source
        description : str, optional
            New description
        category : str, optional
            New category
        tags : list of str, optional
            New tags
        author : str, optional
            New author

        Returns
        -------
        Template or None
            Updated template if found, None otherwise
        """
        template = self.get_template(template_id)
        if not template:
            return None

        # Update fields
        if name is not None:
            template.name = name
            # Regenerate ID if name changed
            new_id = Template.generate_id(name)
            if new_id != template_id:
                # Delete old template file
                self.delete_template(template_id)
                template.id = new_id

        if source is not None:
            if not isinstance(source, MechanismAuthoringSource):
                raise TypeError("source must be a MechanismAuthoringSource.")
            template.reactions_text = source.reactions_text
            template.state_network_dsl = source.state_network_dsl
        if description is not None:
            template.description = description
        if category is not None:
            template.category = category
        if tags is not None:
            template.tags = tags
        if author is not None:
            template.author = author

        template.modified_at = datetime.now().isoformat()

        self.save_template(template)
        return template
