"""Tutorial catalog and the application's only tutorial construction entry point."""

from functools import partial

from kindred.gui.tutorials.controller import TutorialController
from kindred.gui.tutorials.getting_started import GettingStarted

LESSONS = (GettingStarted,)


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
