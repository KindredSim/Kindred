#!/usr/bin/env python3
"""Dynamic runtime introspection for GUI controls."""

import json
import os
import sys
from typing import Any, Dict

# Set offscreen platform before importing Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from PySide6 import QtWidgets, QtGui
    from PySide6.QtCore import QTimer, QObject, QEvent
except ImportError as e:
    print(f"ERROR: Cannot import PySide6: {e}")
    sys.exit(1)


class DialogAutoAcceptor(QObject):
    """Event filter that auto-accepts modal dialogs to prevent hangs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seen_dialogs = []

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter events and auto-accept dialogs."""
        if isinstance(obj, QtWidgets.QDialog) and event.type() == QEvent.Show:
            dialog = obj
            title = dialog.windowTitle()
            self.seen_dialogs.append({
                "title": title,
                "class": dialog.__class__.__name__
            })

            # Schedule auto-accept after 250ms
            QTimer.singleShot(250, lambda: self._accept_dialog(dialog))

        return False  # Don't block the event

    def _accept_dialog(self, dialog):
        """Accept a dialog if it's still visible."""
        if dialog.isVisible():
            try:
                dialog.accept()
            except Exception:
                pass


class GUIDynamicProbe:
    """Runtime introspection of GUI controls."""

    def __init__(self):
        self.app = None
        self.main_window = None
        self.results = {
            "buttons": [],
            "actions": [],
            "shortcuts": [],
            "dialogs_seen": [],
            "errors": []
        }
        self.dialog_filter = None

    def probe(self) -> Dict[str, Any]:
        """Run the dynamic probe."""
        try:
            print("Starting dynamic probe...")

            # Create application
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                self.app = QtWidgets.QApplication(sys.argv)

            # Install dialog auto-acceptor
            self.dialog_filter = DialogAutoAcceptor()
            self.app.installEventFilter(self.dialog_filter)

            # Import and create main window
            print("  Importing main window...")
            try:
                from kindred.gui.main_window import MainWindow
            except ImportError as e:
                self.results["errors"].append(f"Cannot import MainWindow: {e}")
                return self.results

            print("  Creating main window...")
            try:
                self.main_window = MainWindow()
            except Exception as e:
                self.results["errors"].append(f"Cannot create MainWindow: {e}")
                return self.results

            # Process events to let window initialize
            self.app.processEvents()

            # Scan the widget tree
            print("  Scanning widget tree...")
            self._scan_tree(self.main_window)

            # Probe controls (carefully)
            print("  Probing controls...")
            self._probe_actions()

            # Collect dialog info
            self.results["dialogs_seen"] = self.dialog_filter.seen_dialogs

            print(f"  Found {len(self.results['buttons'])} buttons")
            print(f"  Found {len(self.results['actions'])} actions")
            print(f"  Found {len(self.results['shortcuts'])} shortcuts")

        except Exception as e:
            self.results["errors"].append(f"Probe failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            if self.main_window:
                try:
                    self.main_window.close()
                except Exception:
                    pass

        return self.results

    def _scan_tree(self, root: QObject):
        """Recursively scan the QObject tree."""
        # Process the root object
        self._process_object(root)

        # Recurse to children
        for child in root.findChildren(QObject):
            self._process_object(child)

    def _process_object(self, obj: QObject):
        """Process a single QObject."""
        # Check for buttons
        if isinstance(obj, QtWidgets.QAbstractButton):
            self._record_button(obj)

        # Check for actions
        elif isinstance(obj, QtGui.QAction):
            self._record_action(obj)

        # Check for shortcuts
        elif isinstance(obj, QtGui.QShortcut):
            self._record_shortcut(obj)

    def _record_button(self, button: QtWidgets.QAbstractButton):
        """Record a button."""
        self.results["buttons"].append({
            "class": button.__class__.__name__,
            "objectName": button.objectName(),
            "text": button.text(),
            "enabled": button.isEnabled(),
            "visible": button.isVisible(),
            "checkable": button.isCheckable(),
            "checked": button.isChecked() if button.isCheckable() else False,
            "parent": self._get_parent_path(button)
        })

    def _record_action(self, action: QtGui.QAction):
        """Record an action."""
        shortcut = ""
        try:
            shortcut = action.shortcut().toString()
        except Exception:
            pass

        self.results["actions"].append({
            "objectName": action.objectName(),
            "text": action.text(),
            "enabled": action.isEnabled(),
            "visible": action.isVisible(),
            "checkable": action.isCheckable(),
            "checked": action.isChecked() if action.isCheckable() else False,
            "shortcut": shortcut,
            "parent": self._get_parent_path(action)
        })

    def _record_shortcut(self, shortcut: QtGui.QShortcut):
        """Record a shortcut."""
        key_seq = ""
        try:
            key_seq = shortcut.key().toString()
        except Exception:
            pass

        self.results["shortcuts"].append({
            "objectName": shortcut.objectName(),
            "key": key_seq,
            "enabled": shortcut.isEnabled(),
            "parent": self._get_parent_path(shortcut)
        })

    def _get_parent_path(self, obj: QObject) -> str:
        """Get parent path for an object."""
        parts = []
        current = obj.parent()
        while current and len(parts) < 5:
            name = current.objectName()
            if not name:
                name = current.__class__.__name__
            parts.append(name)
            current = current.parent()

        return " > ".join(reversed(parts)) if parts else "no_parent"

    def _probe_actions(self):
        """Safely probe actions by triggering them."""
        # For safety, we'll just record current state without triggering
        # Triggering could have side effects we want to avoid
        pass


def write_json(results: Dict[str, Any], output_path: str):
    """Write results to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def main():
    """Main entry point."""
    probe = GUIDynamicProbe()

    try:
        results = probe.probe()
    except Exception as e:
        print(f"ERROR: Dynamic probe failed: {e}")
        import traceback
        traceback.print_exc()
        results = {"errors": [str(e)]}

    artifacts_dir = os.environ.get("KINDRED_GUI_AUDIT_ARTIFACTS_DIR") or os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    output_path = os.path.join(artifacts_dir, "gui_dynamic_probe.json")
    write_json(results, output_path)
    print(f"  Wrote {output_path}")

    return results


if __name__ == "__main__":
    main()
