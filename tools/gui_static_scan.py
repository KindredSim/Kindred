#!/usr/bin/env python3
"""Static AST-based scanner for GUI control wiring audit."""

import ast
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Control:
    """Represents a GUI control (button, action, shortcut, etc.)."""
    kind: str  # QPushButton, QAction, QToolButton, QShortcut, QMenu
    var_name: str
    object_name: str
    text: str
    shortcut: str
    enabled_default: str  # "True", "False", "Unknown"
    visible_default: str  # "True", "False", "Unknown"
    created_file: str
    created_line: int
    parent_var: str
    added_to_parent_file: str = ""
    added_to_parent_line: int = 0
    connections: List[str] = field(default_factory=list)
    connection_lines: List[int] = field(default_factory=list)
    slot_targets: List[str] = field(default_factory=list)
    is_orphan: bool = False
    is_never_added: bool = False
    is_noop_connection: bool = False
    static_confidence: str = "high"
    risks: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Results from static scanning."""
    controls: List[Control]
    duplicate_shortcuts: Dict[str, List[Tuple[str, int]]]
    orphans: List[Control]
    never_added_actions: List[Control]
    disabled_permanently: List[Control]


class GUIStaticScanner:
    """Static scanner for GUI controls using AST."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.controls: Dict[str, Control] = {}  # var_name -> Control
        self.control_list: List[Control] = []
        self.shortcuts: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # shortcut -> [(file, line)]
        self.connections: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)  # var -> [(slot, line, file)]
        self.added_to_parent: Dict[str, Tuple[str, int]] = {}  # var -> (file, line)
        self.disabled_vars: Set[str] = set()
        self.hidden_vars: Set[str] = set()
        self.enabled_vars: Set[str] = set()
        self.shown_vars: Set[str] = set()

    def scan(self) -> ScanResult:
        """Run the full static scan."""
        gui_dir = self.root_dir / "kindred" / "gui"

        # First pass: collect all controls
        for py_file in gui_dir.rglob("*.py"):
            self._scan_file(py_file)

        # Second pass: analyze connections and flags
        self._analyze_controls()

        return self._build_result()

    def _scan_file(self, filepath: Path):
        """Scan a single Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(filepath))
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
            return

        # Also use regex for patterns AST might miss
        self._regex_scan(filepath, content)

        # Walk AST
        for node in ast.walk(tree):
            self._process_node(node, filepath, content)

    def _process_node(self, node: ast.AST, filepath: Path, content: str):
        """Process an AST node looking for GUI controls."""
        rel_path = str(filepath.relative_to(self.root_dir))

        # Look for assignments creating controls
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id

                if isinstance(node.value, ast.Call):
                    self._process_call_assignment(node.value, var_name, rel_path, node.lineno, content)

                    # Check if this is assigning result of menu.addAction()
                    if isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr == "addAction" and node.value.args:
                            if isinstance(node.value.args[0], ast.Constant):
                                # This assigns an action created by addAction
                                text = str(node.value.args[0].value)
                                callback = self._get_slot_name(node.value.args[1] if len(node.value.args) > 1 else None)

                                control = Control(
                                    kind="QAction",
                                    var_name=var_name,
                                    object_name="",
                                    text=text,
                                    shortcut="",
                                    enabled_default="True",
                                    visible_default="True",
                                    created_file=rel_path,
                                    created_line=node.lineno,
                                    parent_var="",
                                    added_to_parent_file=rel_path,
                                    added_to_parent_line=node.lineno
                                )

                                if callback and callback != "unknown":
                                    control.connections.append(f"{rel_path}:{node.lineno}")
                                    control.connection_lines.append(node.lineno)
                                    control.slot_targets.append(callback)

                                self.controls[var_name] = control
                                self.control_list.append(control)

        # Look for method calls (connections, setEnabled, etc.)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._process_method_call(node.value, rel_path, node.lineno, content)

    def _process_call_assignment(self, call: ast.Call, var_name: str, filepath: str, lineno: int, content: str):
        """Process assignment of a GUI control."""
        func_name = self._get_func_name(call.func)

        if func_name in ["QPushButton", "QToolButton", "QAction", "QShortcut", "QMenu"]:
            control = Control(
                kind=func_name,
                var_name=var_name,
                object_name="",
                text="",
                shortcut="",
                enabled_default="True",
                visible_default="True",
                created_file=filepath,
                created_line=lineno,
                parent_var=""
            )

            # Extract arguments
            if call.args:
                if func_name in ["QPushButton", "QToolButton"]:
                    # First arg is usually text or icon
                    if isinstance(call.args[0], ast.Constant):
                        control.text = str(call.args[0].value)
                elif func_name == "QAction":
                    # First arg might be text or icon
                    if isinstance(call.args[0], ast.Constant):
                        control.text = str(call.args[0].value)
                elif func_name == "QShortcut":
                    # First arg is key sequence
                    if isinstance(call.args[0], ast.Constant):
                        control.shortcut = str(call.args[0].value)

            # Check for parent in args or kwargs
            if len(call.args) > 1:
                if isinstance(call.args[-1], ast.Name):
                    control.parent_var = call.args[-1].id

            for kw in call.keywords:
                if kw.arg == "text" and isinstance(kw.value, ast.Constant):
                    control.text = str(kw.value.value)
                elif kw.arg == "parent" and isinstance(kw.value, ast.Name):
                    control.parent_var = kw.value.id

            self.controls[var_name] = control
            self.control_list.append(control)

    def _process_method_call(self, call: ast.Call, filepath: str, lineno: int, content: str):
        """Process method calls for connections, setEnabled, etc."""
        if isinstance(call.func, ast.Attribute):
            method_name = call.func.attr

            # Get the object this method is called on
            obj_name = self._get_obj_name(call.func.value)

            # Track connections
            if method_name == "connect":
                # This might be a signal connection
                signal_obj = self._get_signal_obj(call.func.value)
                if signal_obj:
                    slot_name = self._get_slot_name(call.args[0] if call.args else None)
                    self.connections[signal_obj].append((slot_name, lineno, filepath))

            # Track addAction, addMenu, etc.
            elif method_name in ["addAction", "addMenu", "addWidget", "addSeparator"]:
                if method_name == "addAction" and call.args:
                    # Pattern: menu.addAction("text", callback) - creates inline action
                    if isinstance(call.args[0], ast.Constant):
                        # This is an inline action creation
                        text = str(call.args[0].value)
                        callback = self._get_slot_name(call.args[1] if len(call.args) > 1 else None)

                        # Create a control for this inline action
                        var_name = f"inline_action_{lineno}"
                        control = Control(
                            kind="QAction",
                            var_name=var_name,
                            object_name="",
                            text=text,
                            shortcut="",
                            enabled_default="True",
                            visible_default="True",
                            created_file=filepath,
                            created_line=lineno,
                            parent_var=obj_name,
                            added_to_parent_file=filepath,
                            added_to_parent_line=lineno
                        )

                        # Record the connection
                        if callback and callback != "unknown":
                            control.connections.append(f"{filepath}:{lineno}")
                            control.connection_lines.append(lineno)
                            control.slot_targets.append(callback)

                        self.controls[var_name] = control
                        self.control_list.append(control)

                # Also track when a variable is added
                if call.args and isinstance(call.args[0], ast.Name):
                    action_var = call.args[0].id
                    self.added_to_parent[action_var] = (filepath, lineno)

            # Track setEnabled/setDisabled
            elif method_name == "setEnabled":
                if obj_name and call.args:
                    if isinstance(call.args[0], ast.Constant) and call.args[0].value is False:
                        self.disabled_vars.add(obj_name)
                    elif isinstance(call.args[0], ast.Constant) and call.args[0].value is True:
                        self.enabled_vars.add(obj_name)
            elif method_name == "setDisabled":
                if obj_name and call.args:
                    if isinstance(call.args[0], ast.Constant) and call.args[0].value is True:
                        self.disabled_vars.add(obj_name)

            # Track setVisible/setHidden
            elif method_name == "setVisible":
                if obj_name and call.args:
                    if isinstance(call.args[0], ast.Constant) and call.args[0].value is False:
                        self.hidden_vars.add(obj_name)
                    elif isinstance(call.args[0], ast.Constant) and call.args[0].value is True:
                        self.shown_vars.add(obj_name)
            elif method_name == "setHidden":
                if obj_name and call.args:
                    if isinstance(call.args[0], ast.Constant) and call.args[0].value is True:
                        self.hidden_vars.add(obj_name)

            # Track setShortcut
            elif method_name == "setShortcut":
                if obj_name and call.args:
                    if isinstance(call.args[0], ast.Constant):
                        shortcut = str(call.args[0].value)
                        self.shortcuts[shortcut].append((filepath, lineno))
                        # Update control if we have it
                        if obj_name in self.controls:
                            self.controls[obj_name].shortcut = shortcut

    def _get_func_name(self, node: ast.AST) -> str:
        """Extract function name from call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _get_obj_name(self, node: ast.AST) -> str:
        """Extract object name from expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For self.something, return "something"
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return node.attr
            return self._get_obj_name(node.value)
        return ""

    def _get_signal_obj(self, node: ast.AST) -> Optional[str]:
        """Extract object name from signal (e.g., btn.clicked -> btn)."""
        if isinstance(node, ast.Attribute):
            # node.attr is the signal name (clicked, triggered, etc.)
            return self._get_obj_name(node.value)
        return None

    def _get_slot_name(self, node: Optional[ast.AST]) -> str:
        """Extract slot/callback name."""
        if node is None:
            return "unknown"
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return f"self.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Lambda):
            return "lambda"
        return "unknown"

    def _regex_scan(self, filepath: Path, content: str):
        """Use regex to catch patterns AST might miss."""
        rel_path = str(filepath.relative_to(self.root_dir))
        lines = content.split('\n')

        # Pattern for signal connections
        conn_pattern = re.compile(r'(\w+)\.(clicked|triggered|activated|pressed|released|toggled)\.connect\((.+?)\)')
        for i, line in enumerate(lines, 1):
            matches = conn_pattern.finditer(line)
            for match in matches:
                obj_name = match.group(1)
                _signal = match.group(2)
                slot = match.group(3).strip()

                # Extract just the function name
                if '.' in slot:
                    slot = slot.split('(')[0].strip()
                elif 'lambda' in slot:
                    slot = 'lambda'
                else:
                    slot = slot.split('(')[0].strip()

                self.connections[obj_name].append((slot, i, rel_path))

        # Pattern for setObjectName
        name_pattern = re.compile(r'(\w+)\.setObjectName\(["\'](.+?)["\']\)')
        for i, line in enumerate(lines, 1):
            matches = name_pattern.finditer(line)
            for match in matches:
                var_name = match.group(1)
                obj_name = match.group(2)
                if var_name in self.controls:
                    self.controls[var_name].object_name = obj_name

        # Pattern for shortcuts
        shortcut_pattern = re.compile(r'setShortcut\(["\'](.+?)["\']\)|QShortcut\(["\'](.+?)["\']')
        for i, line in enumerate(lines, 1):
            matches = shortcut_pattern.finditer(line)
            for match in matches:
                shortcut = match.group(1) or match.group(2)
                if shortcut:
                    self.shortcuts[shortcut].append((rel_path, i))

    def _analyze_controls(self):
        """Analyze controls for orphans, connections, etc."""
        for control in self.control_list:
            # Check for connections
            if control.var_name in self.connections:
                conns = self.connections[control.var_name]
                for slot, line, file in conns:
                    control.connections.append(f"{file}:{line}")
                    control.connection_lines.append(line)
                    control.slot_targets.append(slot)

                    # Check for noop lambdas
                    if slot == "lambda" or "pass" in slot or "logger.debug" in slot:
                        control.is_noop_connection = True
                        control.risks.append("Noop connection (lambda pass or logger only)")

            # Check if added to parent
            if control.var_name in self.added_to_parent:
                file, line = self.added_to_parent[control.var_name]
                control.added_to_parent_file = file
                control.added_to_parent_line = line

            # Determine if orphan
            if control.kind in ["QPushButton", "QToolButton"]:
                if not control.connections:
                    control.is_orphan = True
                    control.risks.append("No signal connections found")
            elif control.kind == "QAction":
                if not control.connections:
                    control.is_orphan = True
                    control.risks.append("No triggered connection")
                if not control.added_to_parent_file:
                    control.is_never_added = True
                    control.risks.append("Created but never added to menu/toolbar")
            elif control.kind == "QShortcut":
                if not control.connections:
                    control.is_orphan = True
                    control.risks.append("No activated connection")

            # Check enabled/visible state
            if control.var_name in self.disabled_vars:
                control.enabled_default = "False"
                if control.var_name not in self.enabled_vars:
                    control.risks.append("Disabled with no code path to enable")

            if control.var_name in self.hidden_vars:
                control.visible_default = "False"
                if control.var_name not in self.shown_vars:
                    control.risks.append("Hidden with no code path to show")

    def _build_result(self) -> ScanResult:
        """Build the final scan result."""
        orphans = [c for c in self.control_list if c.is_orphan]
        never_added = [c for c in self.control_list if c.is_never_added]
        disabled = [c for c in self.control_list if "Disabled with no code path" in c.risks]

        # Find duplicate shortcuts
        dup_shortcuts = {k: v for k, v in self.shortcuts.items() if len(v) > 1}

        return ScanResult(
            controls=self.control_list,
            duplicate_shortcuts=dup_shortcuts,
            orphans=orphans,
            never_added_actions=never_added,
            disabled_permanently=disabled
        )


def write_csv(result: ScanResult, output_path: str):
    """Write results to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "kind", "var_name", "object_name", "text", "shortcut",
            "enabled_default", "visible_default", "created_file", "created_line",
            "parent_var", "added_to_parent_file", "added_to_parent_line",
            "connections", "slot_targets", "is_orphan", "is_never_added",
            "static_confidence", "risks"
        ])

        for control in result.controls:
            writer.writerow([
                control.kind,
                control.var_name,
                control.object_name,
                control.text,
                control.shortcut,
                control.enabled_default,
                control.visible_default,
                control.created_file,
                control.created_line,
                control.parent_var,
                control.added_to_parent_file,
                control.added_to_parent_line,
                "; ".join(control.connections),
                "; ".join(control.slot_targets),
                control.is_orphan,
                control.is_never_added,
                control.static_confidence,
                "; ".join(control.risks)
            ])


def main():
    """Main entry point."""
    root_dir = os.getcwd()
    scanner = GUIStaticScanner(root_dir)

    print("Running static GUI wiring scan...")
    result = scanner.scan()

    print(f"  Found {len(result.controls)} controls")
    print(f"  Orphans: {len(result.orphans)}")
    print(f"  Never added: {len(result.never_added_actions)}")
    print(f"  Duplicate shortcuts: {len(result.duplicate_shortcuts)}")

    artifacts_dir = os.environ.get("KINDRED_GUI_AUDIT_ARTIFACTS_DIR") or os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    csv_path = os.path.join(artifacts_dir, "gui_static_scan.csv")
    write_csv(result, csv_path)
    print(f"  Wrote {csv_path}")

    return result


if __name__ == "__main__":
    main()
