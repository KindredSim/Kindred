from __future__ import annotations

import pytest


pytestmark = [pytest.mark.gui]


def test_notes_text_is_not_injected_into_mechanism_dsl(main_window):
    """
    Regression for the Notes hard-break:

    The GUI Notes tab must never be parsed/injected/concatenated into the mechanism
    DSL used for simulation/fitting. Algebraic content must come only from `# Algebra`
    inside the Reactions DSL text.
    """
    reactions = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    main_window._mechanism_editor._reactions_text.setPlainText(reactions)

    # Locate the legacy "Algebra" tab text box (which will become "Notes") without
    # relying on private attribute names.
    tabs = main_window._mechanism_editor._tabs
    notes_editor = None
    for i in range(tabs.count()):
        if str(tabs.tabText(i)).strip().lower() in {"algebra", "notes"}:
            tab = tabs.widget(i)
            notes_editor = tab.findChild(type(main_window._mechanism_editor._reactions_text))
            break
    assert notes_editor is not None

    notes_editor.setPlainText("param injected = 123\nlet bogus = [A]\n")

    full_dsl = str(main_window._get_mechanism_text() or "")
    assert "param injected" not in full_dsl
    assert "let bogus" not in full_dsl

