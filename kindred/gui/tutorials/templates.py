"""Create a reusable mechanism template and load it in a separate tab."""

from dataclasses import replace

from PySide6 import QtCore, QtWidgets

from kindred.gui.tutorials.controller import Step
from kindred.gui.tutorials.overlay import Highlight, menu_target
from kindred.gui.tutorials.practice_lesson import PracticeLesson
from kindred.gui.widgets.template_manager_dialog import (
    TemplateEditorDialog,
    TemplateManagerDialog,
)


class CreateAndReuseTemplates(PracticeLesson):
    title = "Create and Reuse Templates"
    description = "Create a template with comments and starting values, describe it, and reuse it from the Examples menu."
    _preset_name = "M3"
    _reuse_steps = frozenset({"create_reuse_tab", "apply_template"})
    _editor_steps = frozenset(
        {"template_values", "template_comments", "template_name", "template_description", "template_tags"}
    )

    def blocked(self):
        missing = super().blocked()
        if missing is not None:
            return missing
        if self.step.name in self._editor_steps and self._dialog(TemplateEditorDialog) is None:
            return self._missing("Open the template editor to continue this exercise.", "open_templates")
        if self.step.name == "create_reuse_tab" and self._dialog((TemplateManagerDialog, TemplateEditorDialog)) is not None:
            return self._missing("Finish creating the template and close Template Manager.", "save_template")
        return None

    def _values(self, set_id, amount: float) -> bool:
        item = next((item for item in self.tab.initial_condition_sets if item.set_id == set_id), None)
        return (
            item is not None
            and not self.registry.tab_runtime(self.tab_id).initial_text_by_set_id.get(set_id)
            and dict(item.values) == {"A": amount, "B": 1.0, "C": 0.0}
        )

    def _template_loaded(self) -> bool:
        if self.tab.prepared_dsl is None:
            return False
        imported = next((item for item in self.tab.initial_condition_sets if item.name == "preset_set1"), None)
        if imported is None:
            return False
        self.reuse_set = imported.set_id
        return True

    def _creation_target(self):
        editor = self._dialog(TemplateEditorDialog)
        if editor is not None:
            return Highlight(editor)
        manager = self._dialog(TemplateManagerDialog)
        if manager is not None:
            button = next(button for button in manager.findChildren(QtWidgets.QPushButton) if button.text() == "Create from Current")
            return Highlight(button)
        return menu_target(self.examples_menu, lambda action: action.objectName() == "templateManagerAction")

    def _source_editor_target(self):
        return Highlight(self._dialog(TemplateEditorDialog).findChild(QtWidgets.QPlainTextEdit))

    def _metadata_targets(self, *rows: int):
        form = self._dialog(TemplateEditorDialog).layout().itemAt(0).layout()
        return tuple(Highlight(form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole).widget()) for row in rows)

    def _save_template_target(self):
        editor = self._dialog(TemplateEditorDialog)
        if editor is not None:
            buttons = editor.findChild(QtWidgets.QDialogButtonBox)
            return Highlight(buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save))
        manager = self._dialog(TemplateManagerDialog)
        if manager is not None:
            close = next(button for button in manager.findChildren(QtWidgets.QPushButton) if button.text() == "Close")
            return (Highlight(manager.findChild(QtWidgets.QListWidget)), Highlight(close))
        return None

    def _user_template_target(self):
        menu = next(
            menu for menu in self.examples_menu.findChildren(QtWidgets.QMenu, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
            if menu.title() == "User-Created Templates"
        )
        if menu.isVisible():
            return Highlight(menu)
        return menu_target(self.examples_menu, lambda action: action.text() == "User-Created Templates")

    def _steps(self) -> tuple[Step, ...]:
        create_tab, load_preset = self._setup_steps()
        return (
            replace(
                create_tab,
                instruction="<p>A template lets you reuse a mechanism, its comments, and its initial-condition declarations.</p>"
                "<p>Click <b>+</b> beside the mechanism tabs to create a practice tab. "
                "We will use M3 to create a template, then load it into another tab.</p>",
            ),
            replace(
                load_preset,
                title="Load M3",
                instruction="<p>Open <b>Examples → Preset Mechanisms → M3</b>.</p>"
                "<p>We will use this example to explore what Kindred includes in a template. "
                "It supplies <b>preset_set1</b> with <b>A = 1, B = 1, and C = 0</b>.</p>",
                success="M3 and the starting values are ready.",
            ),
            Step(
                "open_templates", "Create a Template from This Tab",
                "<p>Open <b>Examples → Template Manager</b>, then click <b>Create from Current</b>.</p>"
                "<p>The template editor opens with a copy of the mechanism text from the selected tab. "
                "Changes made here affect the template being created.</p>",
                self._creation_target,
                lambda: self._dialog(TemplateEditorDialog) is not None,
            ),
            Step(
                "template_values", "Find the Starting Values",
                "<p>In <b>Mechanism DSL</b>, find the <b>preset_set1</b> declaration. It contains the starting concentrations from M3.</p>"
                "<p>After loading M3, Kindred recorded that declaration as comments in Reactions. "
                "Here, it has been restored as a declaration so the template can supply those starting values when loaded.</p>"
                "<p>Check <b>A = 1, B = 1, and C = 0</b>. These are the recorded values. "
                "Changes made only in the Initial Conditions table are not copied into the template.</p>",
                self._source_editor_target,
            ),
            Step(
                "template_comments", "Comments Are Included Too",
                "<p>Find the comments beginning with <b>#</b>, including the headings and the comment after <b>let mass_A</b>.</p>"
                "<p>Ordinary comments are copied into the template and remain comments. You can use them to explain the mechanism "
                "or leave instructions for someone reusing it.</p>"
                "<p>The reaction and observable definitions are included too.</p>",
                self._source_editor_target,
            ),
            Step(
                "template_name", "Name and Categorize the Template",
                "<p>Enter <b>Association model</b> in <b>Name</b>. If you already have a template with that name, choose a different name.</p>"
                "<p>Set <b>Category</b> to <b>Kinetics</b>. Categories let you filter templates in Template Manager. "
                "You can also type a category of your own.</p>",
                lambda: self._metadata_targets(0, 1),
            ),
            Step(
                "template_description", "Add a Description",
                "<p>In <b>Description</b>, enter:</p>"
                "<p>Association of A and B to form C, with starting values for one simulation.</p>"
                "<p>Use this field to explain what the template contains and when someone might use it. "
                "Descriptions are also included when searching in Template Manager.</p>",
                lambda: self._metadata_targets(4),
            ),
            Step(
                "template_tags", "Add Tags and an Author",
                "<p>In <b>Tags</b>, enter <b>association, example</b>. Separate tags with commas. "
                "Tags provide additional words you can use to find the template in Template Manager.</p>"
                "<p>Enter your name in <b>Author</b>, or leave it blank.</p>",
                lambda: self._metadata_targets(2, 3),
            ),
            Step(
                "save_template", "Create the Template",
                "<p>Click <b>Create</b> and dismiss the confirmation message.</p>"
                "<p>Find the saved template in Template Manager, then close the manager. "
                "Next, we will load it into a fresh mechanism tab.</p>",
                self._save_template_target,
                manual=True,
            ),
            Step(
                "create_reuse_tab", "Create a Tab for Reuse",
                "<p>Click <b>+</b> beside the mechanism tabs to create another tab.</p>"
                "<p>Keep this new tab selected. The template will load into it.</p>",
                self._tab_target,
                lambda: self._new_tab(reuse=True),
            ),
            Step(
                "apply_template", "Load Your Template",
                "<p>Open <b>Examples → User-Created Templates</b> and choose <b>Association model</b>, or the name you entered.</p>"
                "<p>The mechanism text, comments, and starting-value declarations load into the selected tab.</p>",
                self._user_template_target,
                self._template_loaded,
                "The template and its starting values are available in this tab.",
            ),
            Step(
                "finish", "Lesson Complete",
                "<p>You have created a template containing a mechanism, comments, and starting-value declarations, "
                "added information to help identify it, and loaded it into another tab.</p>"
                "<p>Use <b>Examples → User-Created Templates</b> to reuse it again. "
                "Use <b>Template Manager</b> when you want to edit or organize your templates.</p>"
                "<p>Choose <b>Finish</b> to close this lesson.</p>",
                lambda: None,
            ),
        )
