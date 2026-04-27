from __future__ import annotations


def main() -> int:
    """
    GUI-only console entry point.

    Importing PySide6 here makes the GUI runtime dependency contract explicit.
    """
    from multiprocessing import freeze_support

    freeze_support()
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PySide6 is required to launch the Kindred GUI.\n\n"
            "Install or reinstall Kindred with its standard GUI dependencies.\n"
        ) from exc

    from kindred.__main__ import main as gui_main

    return gui_main()
