"""Tutorial catalog and the application's only tutorial construction entry point."""

from functools import partial

from kindred.gui.tutorials.controller import TutorialController
from kindred.gui.tutorials.explore_plot import ExplorePlot
from kindred.gui.tutorials.interactive_sliders import InteractiveSliders
from kindred.gui.tutorials.quick_start import QuickStart
from kindred.gui.tutorials.reactions_and_initials import ReactionsAndInitialConditions
from kindred.gui.tutorials.templates import CreateAndReuseTemplates

LESSONS = (
    QuickStart,
    ReactionsAndInitialConditions,
    ExplorePlot,
    InteractiveSliders,
    CreateAndReuseTemplates,
)


def create_controller(window, **existing_owners) -> TutorialController:
    controller = TutorialController(
        window,
        tuple(
            (lesson.title, lesson.description, partial(lesson, **existing_owners))
            for lesson in LESSONS
        ),
    )
    existing_owners["project_controller"].nativeCloseAuthorized.connect(controller.stop)
    return controller
