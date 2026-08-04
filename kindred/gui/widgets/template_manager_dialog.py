"""
Template Manager Dialog for Kindred.

Provides a GUI for browsing, creating, updating, replacing, and applying
mechanism templates.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.config.templates import Template, TemplateManager
from kindred.core.mechanism_source import MechanismAuthoringSource

logger = logging.getLogger(__name__)

__all__ = ["TemplateManagerDialog", "TemplateEditorDialog"]


class TemplateEditorDialog(QtWidgets.QDialog):
    """
    Dialog for creating or editing a template.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        template: Optional[Template] = None,
        mechanism_source: Optional[MechanismAuthoringSource] = None,
        operation: str = "create",
    ):
        """
        Initialize template editor dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        template : Template, optional
            Existing template to edit (None for new template)
        mechanism_source : MechanismAuthoringSource, optional
            Complete mechanism source for new templates.
        """
        super().__init__(parent)
        self._template = template
        self._is_edit_mode = template is not None
        normalized_operation = str(operation or "").strip().casefold()
        if normalized_operation not in {"create", "update"}:
            raise ValueError("Template editor operation must be create or update.")
        if normalized_operation == "update" and not self._is_edit_mode:
            raise ValueError("Template update requires an existing template.")
        if normalized_operation == "create" and self._is_edit_mode:
            raise ValueError("Template create cannot target an existing template.")
        self._operation = normalized_operation

        self.setWindowTitle(
            "Update Template" if self._is_edit_mode else "Create Template"
        )
        self.resize(700, 600)

        layout = QtWidgets.QVBoxLayout(self)

        # Form layout for metadata
        form_layout = QtWidgets.QFormLayout()

        # Name
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("e.g., Enzyme Kinetics")
        if template:
            self._name_edit.setText(template.name)
        form_layout.addRow("Name*:", self._name_edit)

        # Category
        self._category_combo = QtWidgets.QComboBox()
        self._category_combo.setEditable(True)
        self._category_combo.addItems([
            "General",
            "Kinetics",
            "Atmospheric",
            "Polymer",
            "Photochemistry",
            "Electrochemistry",
            "Combustion",
            "Biochemistry",
        ])
        if template:
            self._category_combo.setCurrentText(template.category)
        else:
            self._category_combo.setCurrentText("General")
        form_layout.addRow("Category*:", self._category_combo)

        # Tags
        self._tags_edit = QtWidgets.QLineEdit()
        self._tags_edit.setPlaceholderText("e.g., enzyme, michaelis-menten, steady-state")
        if template:
            self._tags_edit.setText(", ".join(template.tags))
        form_layout.addRow("Tags:", self._tags_edit)

        # Author
        self._author_edit = QtWidgets.QLineEdit()
        self._author_edit.setPlaceholderText("Your name (optional)")
        if template:
            self._author_edit.setText(template.author or "")
        form_layout.addRow("Author:", self._author_edit)

        # Description
        self._description_edit = QtWidgets.QTextEdit()
        self._description_edit.setPlaceholderText(
            "Describe the template, its purpose, and any key features..."
        )
        self._description_edit.setMaximumHeight(100)
        if template:
            self._description_edit.setPlainText(template.description)
        form_layout.addRow("Description*:", self._description_edit)

        layout.addLayout(form_layout)

        # Mechanism text editor
        layout.addWidget(QtWidgets.QLabel("Mechanism DSL*:"))
        self._mechanism_edit = QtWidgets.QPlainTextEdit()
        self._mechanism_edit.setFont(QtGui.QFont("Courier New", 10))
        if template:
            self._mechanism_edit.setPlainText(template.source.reactions_text)
        else:
            source = mechanism_source or MechanismAuthoringSource()
            self._mechanism_edit.setPlainText(source.reactions_text)
        layout.addWidget(self._mechanism_edit, stretch=1)

        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        save_button = button_box.button(QtWidgets.QDialogButtonBox.Save)
        if save_button is not None:
            save_button.setText(
                "Update" if self._operation == "update" else "Create"
            )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _validate_and_accept(self):
        """Validate inputs and accept dialog."""
        # Check required fields
        if not self._name_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                "Template name is required."
            )
            self._name_edit.setFocus()
            return

        if not self._mechanism_edit.toPlainText().strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                "Mechanism text is required."
            )
            self._mechanism_edit.setFocus()
            return

        try:
            MechanismAuthoringSource.from_parts(
                reactions_text=self._mechanism_edit.toPlainText().strip()
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                str(exc),
            )
            self._mechanism_edit.setFocus()
            return

        self.accept()

    def get_template_data(self) -> Dict[str, Any]:
        """
        Get template data from form.

        Returns
        -------
        dict
            Template data
        """
        tags_text = self._tags_edit.text().strip()
        tags = [t.strip() for t in tags_text.split(',') if t.strip()]

        source = MechanismAuthoringSource.from_parts(
            reactions_text=self._mechanism_edit.toPlainText().strip()
        )
        return {
            'name': self._name_edit.text().strip(),
            'category': self._category_combo.currentText().strip(),
            'tags': tags,
            'author': self._author_edit.text().strip() or None,
            'description': self._description_edit.toPlainText().strip(),
            'source': source,
        }


class TemplateManagerDialog(QtWidgets.QDialog):
    """
    Template Manager Dialog.

    Provides explicit Create, Update, Replace, and Apply operations.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        template_manager: Optional[TemplateManager] = None,
        current_mechanism_source: Optional[MechanismAuthoringSource] = None,
        source_preflight: Callable[
            [MechanismAuthoringSource], MechanismAuthoringSource
        ] | None = None,
        destination_apply: (
            Callable[[MechanismAuthoringSource], bool] | None
        ) = None,
    ):
        """
        Initialize template manager dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        template_manager : TemplateManager, optional
            Template manager instance (creates new if None)
        current_mechanism_source : MechanismAuthoringSource, optional
            Current complete mechanism source for creating templates.
        source_preflight : callable
            Source-tab structural/scientific admission for persistence edits.
        destination_apply : callable
            Destination-tab owner returning whether Apply was accepted.
        """
        super().__init__(parent)
        self._template_manager = template_manager or TemplateManager()
        self._current_mechanism_source = current_mechanism_source or MechanismAuthoringSource()
        if not callable(source_preflight):
            raise TypeError("source_preflight must be callable.")
        if not callable(destination_apply):
            raise TypeError("destination_apply must be callable.")
        self._source_preflight = source_preflight
        self._destination_apply = destination_apply
        self._current_template: Optional[Template] = None

        self.setWindowTitle("Template Manager")
        self.resize(1000, 700)

        self._setup_ui()
        self._render_collection_state()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QtWidgets.QVBoxLayout(self)

        # Top toolbar
        toolbar_layout = QtWidgets.QHBoxLayout()

        # Search bar
        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Search templates...")
        self._search_edit.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(QtWidgets.QLabel("Search:"))
        toolbar_layout.addWidget(self._search_edit, stretch=1)

        # Category filter
        self._category_combo = QtWidgets.QComboBox()
        self._category_combo.addItem("All Categories", None)
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        toolbar_layout.addWidget(QtWidgets.QLabel("Category:"))
        toolbar_layout.addWidget(self._category_combo)

        layout.addLayout(toolbar_layout)

        self._storage_state_label = QtWidgets.QLabel()
        self._storage_state_label.setObjectName("templateStorageStateLabel")
        self._storage_state_label.setWordWrap(True)
        self._storage_state_label.setVisible(False)
        layout.addWidget(self._storage_state_label)

        # Splitter for list and preview
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: Template list
        list_widget = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_layout.addWidget(QtWidgets.QLabel("Templates:"))
        self._template_list = QtWidgets.QListWidget()
        self._template_list.currentItemChanged.connect(self._on_template_selected)
        self._template_list.itemDoubleClicked.connect(self._on_apply_clicked)
        list_layout.addWidget(self._template_list)

        splitter.addWidget(list_widget)

        # Right: Preview and details
        preview_widget = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Template details
        details_group = QtWidgets.QGroupBox("Template Details")
        details_layout = QtWidgets.QFormLayout(details_group)

        self._name_label = QtWidgets.QLabel()
        self._name_label.setWordWrap(True)
        details_layout.addRow("Name:", self._name_label)

        self._category_label = QtWidgets.QLabel()
        details_layout.addRow("Category:", self._category_label)

        self._tags_label = QtWidgets.QLabel()
        self._tags_label.setWordWrap(True)
        details_layout.addRow("Tags:", self._tags_label)

        self._author_label = QtWidgets.QLabel()
        details_layout.addRow("Author:", self._author_label)

        self._modified_label = QtWidgets.QLabel()
        details_layout.addRow("Modified:", self._modified_label)

        self._description_text = QtWidgets.QTextEdit()
        self._description_text.setReadOnly(True)
        self._description_text.setMaximumHeight(80)
        details_layout.addRow("Description:", self._description_text)

        preview_layout.addWidget(details_group)

        # Preview text
        preview_layout.addWidget(QtWidgets.QLabel("Mechanism Preview:"))
        self._preview_text = QtWidgets.QPlainTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setFont(QtGui.QFont("Courier New", 9))
        preview_layout.addWidget(self._preview_text, stretch=1)

        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

        # Bottom button bar
        button_layout = QtWidgets.QHBoxLayout()

        self._create_btn = QtWidgets.QPushButton("Create from Current")
        self._create_btn.setToolTip(
            "Create a new template from the current mechanism"
        )
        self._create_btn.clicked.connect(self._on_create_clicked)
        button_layout.addWidget(self._create_btn)

        self._update_btn = QtWidgets.QPushButton("Update")
        self._update_btn.clicked.connect(self._on_update_clicked)
        button_layout.addWidget(self._update_btn)

        self._replace_btn = QtWidgets.QPushButton("Replace from Current")
        self._replace_btn.setToolTip(
            "Replace the selected template's mechanism with the current mechanism"
        )
        self._replace_btn.clicked.connect(self._on_replace_clicked)
        button_layout.addWidget(self._replace_btn)

        self._delete_btn = QtWidgets.QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        button_layout.addWidget(self._delete_btn)

        button_layout.addStretch()

        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        button_layout.addWidget(self._apply_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _render_collection_state(
        self,
        *,
        selected_template_id: str | None = None,
    ) -> None:
        """Render only the collection state admitted before dialog creation."""

        self._update_category_filter()
        self._refresh_template_list()
        if selected_template_id is not None:
            self._select_template(selected_template_id)
        self._render_storage_state()

    def _render_storage_state(self) -> None:
        reason = self._template_manager.write_disabled_reason
        if reason is None:
            self._storage_state_label.clear()
            self._storage_state_label.setVisible(False)
        else:
            self._storage_state_label.setText(reason)
            self._storage_state_label.setVisible(True)
        self._update_action_availability()

    def _update_action_availability(self) -> None:
        selected = self._current_template is not None
        write_disabled_reason = self._template_manager.write_disabled_reason

        write_actions = (
            (
                self._create_btn,
                "Create a new template from the current mechanism.",
                False,
            ),
            (
                self._update_btn,
                "Update the selected template.",
                True,
            ),
            (
                self._replace_btn,
                "Replace the selected template's mechanism with the current mechanism.",
                True,
            ),
            (
                self._delete_btn,
                "Delete the selected template.",
                True,
            ),
        )
        for button, normal_tooltip, requires_selection in write_actions:
            if write_disabled_reason is not None:
                button.setEnabled(False)
                button.setToolTip(write_disabled_reason)
            elif requires_selection and not selected:
                button.setEnabled(False)
                button.setToolTip("Select a template first.")
            else:
                button.setEnabled(True)
                button.setToolTip(normal_tooltip)

        self._apply_btn.setEnabled(selected)
        self._apply_btn.setToolTip(
            "Apply the selected template to its captured destination tab."
            if selected
            else "Select a template first."
        )

    def _refresh_template_list(self):
        """Refresh template list widget."""
        self._template_list.clear()

        # Get filtered templates
        search_query = self._search_edit.text().strip()
        category_filter = self._category_combo.currentData()

        if search_query:
            templates = self._template_manager.search_templates(search_query)
        elif category_filter:
            templates = self._template_manager.get_templates_by_category(category_filter)
        else:
            templates = self._template_manager.get_all_templates()

        # Sort by name
        templates.sort(key=lambda t: t.name.lower())

        # Add to list
        for template in templates:
            item = QtWidgets.QListWidgetItem(f"{template.name} ({template.category})")
            item.setData(QtCore.Qt.UserRole, template.id)
            self._template_list.addItem(item)

        # Update status
        count = self._template_list.count()
        self.setWindowTitle(f"Template Manager ({count} template{'s' if count != 1 else ''})")
        self._update_action_availability()

    def _update_category_filter(self):
        """Update category filter combo box."""
        current_category = self._category_combo.currentData()

        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItem("All Categories", None)

        categories = self._template_manager.get_categories()
        for category in categories:
            self._category_combo.addItem(category, category)

        # Restore selection
        if current_category:
            index = self._category_combo.findData(current_category)
            if index >= 0:
                self._category_combo.setCurrentIndex(index)

        self._category_combo.blockSignals(False)

    def _on_search_changed(self, text: str):
        """Handle search text changed."""
        self._refresh_template_list()

    def _on_category_changed(self, index: int):
        """Handle category filter changed."""
        self._refresh_template_list()

    def _on_template_selected(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        """Handle template selection."""
        if current is None:
            self._current_template = None
            self._clear_preview()
            self._update_action_availability()
            return

        template_id = current.data(QtCore.Qt.UserRole)
        self._current_template = self._template_manager.get_template(template_id)

        if self._current_template:
            self._show_preview(self._current_template)
        else:
            self._clear_preview()
        self._update_action_availability()

    def _show_preview(self, template: Template):
        """Show template preview."""
        self._name_label.setText(template.name)
        self._category_label.setText(template.category)
        self._tags_label.setText(", ".join(template.tags) if template.tags else "(none)")
        self._author_label.setText(template.author or "(anonymous)")

        # Format modified date
        try:
            modified_dt = datetime.fromisoformat(template.modified_at)
            modified_str = modified_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError):
            modified_str = template.modified_at

        self._modified_label.setText(modified_str)
        self._description_text.setPlainText(template.description)
        self._preview_text.setPlainText(template.source.reactions_text)

    def _clear_preview(self):
        """Clear template preview."""
        self._name_label.clear()
        self._category_label.clear()
        self._tags_label.clear()
        self._author_label.clear()
        self._modified_label.clear()
        self._description_text.clear()
        self._preview_text.clear()

    def _on_create_clicked(self):
        """Create one new template from an admitted source candidate."""
        dialog = TemplateEditorDialog(
            parent=self,
            mechanism_source=self._current_mechanism_source,
            operation="create",
        )

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            data = dialog.get_template_data()

            try:
                data["source"] = self._preflight_source(data["source"])
                template = self._template_manager.create_template(**data)
                self._render_collection_state(
                    selected_template_id=template.id
                )

                QtWidgets.QMessageBox.information(
                    self,
                    "Template Created",
                    f"Template '{template.name}' created successfully."
                )

            except Exception as e:
                logger.error(f"Failed to create template: {e}", exc_info=True)
                self._render_storage_state()
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to create template:\n\n{e}"
                )

    def _on_update_clicked(self):
        """Update one selected template from an admitted source candidate."""
        if not self._current_template:
            return

        dialog = TemplateEditorDialog(
            parent=self,
            template=self._current_template,
            operation="update",
        )

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            data = dialog.get_template_data()

            try:
                data["source"] = self._preflight_source(data["source"])
                template = self._template_manager.update_template(
                    self._current_template.id,
                    **data
                )
                self._render_collection_state(
                    selected_template_id=template.id
                )

                QtWidgets.QMessageBox.information(
                    self,
                    "Template Updated",
                    f"Template '{data['name']}' updated successfully."
                )

            except Exception as e:
                logger.error(f"Failed to update template: {e}", exc_info=True)
                self._render_storage_state()
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to update template:\n\n{e}"
                )

    def _on_replace_clicked(self):
        """Explicitly replace one selected template from the source tab."""
        current = self._current_template
        if current is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Replace Template",
            f"Replace template '{current.name}' with the current mechanism?\n\n"
            "Its stable identity and metadata will be preserved.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            admitted_source = self._preflight_source(
                self._current_mechanism_source
            )
            candidate = replace(
                current,
                reactions_text=admitted_source.reactions_text,
                modified_at=datetime.now().isoformat(),
            )
            template = self._template_manager.replace_template(
                current.id,
                candidate,
            )
            self._render_collection_state(
                selected_template_id=template.id
            )
            QtWidgets.QMessageBox.information(
                self,
                "Template Replaced",
                f"Template '{template.name}' replaced successfully.",
            )
        except Exception as exc:
            logger.error("Failed to replace template: %s", exc, exc_info=True)
            self._render_storage_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to replace template:\n\n{exc}",
            )

    def _on_delete_clicked(self):
        """Handle Delete button clicked."""
        if not self._current_template:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete template '{self._current_template.name}'?\n\n"
            "This action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                template_name = self._current_template.name
                self._template_manager.delete_template(self._current_template.id)
                self._current_template = None
                self._render_collection_state()

                QtWidgets.QMessageBox.information(
                    self,
                    "Template Deleted",
                    f"Template '{template_name}' deleted successfully."
                )

            except Exception as e:
                logger.error(f"Failed to delete template: {e}", exc_info=True)
                self._render_storage_state()
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete template:\n\n{e}"
                )

    def _on_apply_clicked(self, *_args: object) -> None:
        """Apply the selected source through the destination-tab owner."""
        if not self._current_template:
            return
        try:
            accepted = self._destination_apply(
                self._current_template.source
            )
            if type(accepted) is not bool:
                raise TypeError("destination_apply must return bool.")
        except Exception as exc:
            logger.error(
                "Template destination rejected Apply: %s",
                exc,
                exc_info=True,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Template Apply",
                f"The template could not be applied:\n\n{exc}",
            )
            return
        if accepted:
            self.accept()

    def _preflight_source(
        self,
        source: MechanismAuthoringSource,
    ) -> MechanismAuthoringSource:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("template source must be MechanismAuthoringSource.")
        admitted = self._source_preflight(source)
        if not isinstance(admitted, MechanismAuthoringSource):
            raise TypeError(
                "source_preflight must return MechanismAuthoringSource."
            )
        return admitted

    def _select_template(self, template_id: str) -> None:
        for index in range(self._template_list.count()):
            item = self._template_list.item(index)
            if item.data(QtCore.Qt.UserRole) == str(template_id):
                self._template_list.setCurrentItem(item)
                return
