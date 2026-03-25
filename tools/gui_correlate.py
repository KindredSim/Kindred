#!/usr/bin/env python3
"""Correlate static and dynamic scan results into final report."""

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


class ControlReport:
    """Combined report for a control."""

    def __init__(self, static_row: Optional[Dict] = None, dynamic_data: Optional[Dict] = None):
        self.static_row = static_row or {}
        self.dynamic_data = dynamic_data or {}
        self.match_quality = "none"

        if static_row and dynamic_data:
            self.match_quality = "good"
        elif static_row:
            self.match_quality = "static_only"
        elif dynamic_data:
            self.match_quality = "dynamic_only"

    def get_kind(self) -> str:
        """Get control kind."""
        if self.static_row:
            return self.static_row.get("kind", "Unknown")
        if self.dynamic_data:
            return self.dynamic_data.get("class", "Unknown")
        return "Unknown"

    def get_text(self) -> str:
        """Get control text/label."""
        text = self.static_row.get("text", "") if self.static_row else ""
        if not text and self.dynamic_data:
            text = self.dynamic_data.get("text", "")
        return text

    def get_object_name(self) -> str:
        """Get object name."""
        name = self.static_row.get("object_name", "") if self.static_row else ""
        if not name and self.dynamic_data:
            name = self.dynamic_data.get("objectName", "")
        return name

    def get_var_name(self) -> str:
        """Get variable name."""
        return self.static_row.get("var_name", "") if self.static_row else ""

    def get_enabled(self) -> str:
        """Get enabled status."""
        if self.dynamic_data and "enabled" in self.dynamic_data:
            return str(self.dynamic_data["enabled"])
        if self.static_row:
            return self.static_row.get("enabled_default", "Unknown")
        return "Unknown"

    def get_visible(self) -> str:
        """Get visible status."""
        if self.dynamic_data and "visible" in self.dynamic_data:
            return str(self.dynamic_data["visible"])
        if self.static_row:
            return self.static_row.get("visible_default", "Unknown")
        return "Unknown"

    def get_created_location(self) -> str:
        """Get creation location."""
        if self.static_row:
            file = self.static_row.get("created_file", "")
            line = self.static_row.get("created_line", "")
            if file and line:
                return f"{file}:{line}"
        return "Unknown"

    def get_added_location(self) -> str:
        """Get added-to-parent location."""
        if self.static_row:
            file = self.static_row.get("added_to_parent_file", "")
            line = self.static_row.get("added_to_parent_line", "")
            if file and line:
                return f"{file}:{line}"
        return ""

    def get_connections(self) -> str:
        """Get connection info."""
        if self.static_row:
            return self.static_row.get("connections", "")
        return ""

    def get_slots(self) -> str:
        """Get slot targets."""
        if self.static_row:
            return self.static_row.get("slot_targets", "")
        return ""

    def get_risks(self) -> str:
        """Get risk assessment."""
        if self.static_row:
            return self.static_row.get("risks", "")
        return ""

    def is_orphan(self) -> bool:
        """Check if orphan."""
        if self.static_row:
            return self.static_row.get("is_orphan", "False") == "True"
        return False

    def is_never_added(self) -> bool:
        """Check if never added."""
        if self.static_row:
            return self.static_row.get("is_never_added", "False") == "True"
        return False

    def get_severity(self) -> str:
        """Determine severity."""
        kind = self.get_kind()
        if self.is_orphan():
            if kind in ["QPushButton", "QToolButton"]:
                return "Critical"
            else:
                return "High"
        if self.is_never_added():
            return "High"
        if "duplicate" in self.get_risks().lower():
            return "Medium"
        if "disabled" in self.get_risks().lower() or "hidden" in self.get_risks().lower():
            return "Low"
        return "Info"


class GUICorrelator:
    """Correlate static and dynamic results."""

    def __init__(self, static_csv: str, dynamic_json: str):
        self.static_csv = static_csv
        self.dynamic_json = dynamic_json
        self.static_data: List[Dict] = []
        self.dynamic_data: Dict[str, Any] = {}
        self.reports: List[ControlReport] = []
        self.duplicate_shortcuts: Dict[str, List[str]] = {}

    def correlate(self):
        """Run correlation."""
        print("Correlating static and dynamic results...")

        # Load static data
        self._load_static()

        # Load dynamic data
        self._load_dynamic()

        # Match controls
        self._match_controls()

        # Analyze for duplicates and issues
        self._analyze_duplicates()

        print(f"  Correlated {len(self.reports)} controls")

    def _load_static(self):
        """Load static CSV."""
        if not os.path.exists(self.static_csv):
            print(f"  Warning: Static CSV not found: {self.static_csv}")
            return

        with open(self.static_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.static_data = list(reader)

        print(f"  Loaded {len(self.static_data)} static controls")

    def _load_dynamic(self):
        """Load dynamic JSON."""
        if not os.path.exists(self.dynamic_json):
            print(f"  Warning: Dynamic JSON not found: {self.dynamic_json}")
            return

        with open(self.dynamic_json, 'r', encoding='utf-8') as f:
            self.dynamic_data = json.load(f)

        total = (len(self.dynamic_data.get("buttons", [])) +
                 len(self.dynamic_data.get("actions", [])) +
                 len(self.dynamic_data.get("shortcuts", [])))
        print(f"  Loaded {total} dynamic controls")

    def _match_controls(self):
        """Match static and dynamic controls."""
        # For now, create reports from static data
        # Matching by objectName would require dynamic data to have it set
        for static_row in self.static_data:
            report = ControlReport(static_row=static_row)
            self.reports.append(report)

        # Add dynamic-only controls
        # (In practice, most controls should be in static scan)

    def _analyze_duplicates(self):
        """Analyze for duplicate shortcuts."""
        shortcut_map = defaultdict(list)

        for report in self.reports:
            if report.static_row:
                shortcut = report.static_row.get("shortcut", "")
                if shortcut:
                    loc = report.get_created_location()
                    shortcut_map[shortcut].append(loc)

        self.duplicate_shortcuts = {k: v for k, v in shortcut_map.items() if len(v) > 1}

    def write_csv(self, output_path: str):
        """Write final CSV report."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "kind", "objectName", "label_text", "enabled_default", "visible_default",
                "created_at", "added_to_parent_at", "connection_lines", "slot_target_name",
                "risks", "severity", "match_quality"
            ])

            for report in self.reports:
                writer.writerow([
                    report.get_kind(),
                    report.get_object_name(),
                    report.get_text(),
                    report.get_enabled(),
                    report.get_visible(),
                    report.get_created_location(),
                    report.get_added_location(),
                    report.get_connections(),
                    report.get_slots(),
                    report.get_risks(),
                    report.get_severity(),
                    report.match_quality
                ])

        print(f"  Wrote {output_path}")

    def write_markdown(self, output_path: str):
        """Write final markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# GUI Control Wiring Audit Report\n\n")

            # Executive summary
            f.write("## Executive Summary\n\n")
            orphans = [r for r in self.reports if r.is_orphan()]
            never_added = [r for r in self.reports if r.is_never_added()]
            critical = [r for r in self.reports if r.get_severity() == "Critical"]
            high = [r for r in self.reports if r.get_severity() == "High"]

            f.write(f"- **Total controls scanned:** {len(self.reports)}\n")
            f.write(f"- **Orphaned controls (no connections):** {len(orphans)}\n")
            f.write(f"- **Actions never added to UI:** {len(never_added)}\n")
            f.write(f"- **Duplicate shortcuts:** {len(self.duplicate_shortcuts)}\n")
            f.write(f"- **Critical issues:** {len(critical)}\n")
            f.write(f"- **High priority issues:** {len(high)}\n")
            f.write("\n")

            # Counts by type
            f.write("## Controls by Type\n\n")
            type_counts = defaultdict(int)
            for report in self.reports:
                type_counts[report.get_kind()] += 1

            for kind, count in sorted(type_counts.items()):
                f.write(f"- **{kind}:** {count}\n")
            f.write("\n")

            # Main table
            f.write("## Control Details\n\n")
            f.write("| Kind | ObjectName | Label/Text | Enabled | Visible | Created At | Added To Parent | Connections | Slot Target | Risks | Severity |\n")
            f.write("|------|------------|------------|---------|---------|------------|-----------------|-------------|-------------|-------|----------|\n")

            for report in sorted(self.reports, key=lambda r: (r.get_severity(), r.get_kind())):
                kind = report.get_kind()
                obj_name = report.get_object_name() or "(unnamed)"
                text = report.get_text()[:40]  # Truncate long text
                enabled = report.get_enabled()
                visible = report.get_visible()
                created = report.get_created_location()
                added = report.get_added_location() or "N/A"
                conns = report.get_connections() or "NONE"
                slots = report.get_slots() or "NONE"
                risks = report.get_risks()[:60] or "None"
                severity = report.get_severity()

                f.write(f"| {kind} | {obj_name} | {text} | {enabled} | {visible} | {created} | {added} | {conns} | {slots} | {risks} | **{severity}** |\n")

            f.write("\n")

            # Orphans section
            if orphans:
                f.write("## Orphaned Controls\n\n")
                f.write("Controls with no signal connections:\n\n")
                for report in orphans:
                    f.write(f"- **{report.get_kind()}** `{report.get_object_name() or report.get_var_name()}` ")
                    f.write(f"at `{report.get_created_location()}` - {report.get_text()}\n")
                f.write("\n")

            # Never added actions
            if never_added:
                f.write("## Actions Never Added to UI\n\n")
                f.write("QAction instances created but never added to menus/toolbars:\n\n")
                for report in never_added:
                    f.write(f"- `{report.get_var_name()}` at `{report.get_created_location()}` - {report.get_text()}\n")
                f.write("\n")

            # Duplicate shortcuts
            if self.duplicate_shortcuts:
                f.write("## Duplicate Shortcuts\n\n")
                for shortcut, locations in self.duplicate_shortcuts.items():
                    f.write(f"- **{shortcut}** defined at:\n")
                    for loc in locations:
                        f.write(f"  - `{loc}`\n")
                f.write("\n")

            # Prioritized fix list
            f.write("## Prioritized Fix List\n\n")
            f.write("### Critical Priority\n\n")
            for report in critical:
                f.write(f"1. Fix orphan button `{report.get_var_name()}` at `{report.get_created_location()}`\n")
            f.write("\n")

            f.write("### High Priority\n\n")
            for report in high:
                f.write(f"1. {report.get_risks()} - `{report.get_created_location()}`\n")
            f.write("\n")

        print(f"  Wrote {output_path}")


def main():
    """Main entry point."""
    root_dir = os.getcwd()
    artifacts_dir = os.environ.get("KINDRED_GUI_AUDIT_ARTIFACTS_DIR") or os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    static_csv = os.path.join(artifacts_dir, "gui_static_scan.csv")
    dynamic_json = os.path.join(artifacts_dir, "gui_dynamic_probe.json")

    correlator = GUICorrelator(static_csv, dynamic_json)
    correlator.correlate()

    # Write outputs
    csv_path = os.path.join(artifacts_dir, "gui_wiring_audit.csv")
    md_path = os.path.join(artifacts_dir, "gui_wiring_audit.md")

    correlator.write_csv(csv_path)
    correlator.write_markdown(md_path)

    print("\nAudit complete!")
    print(f"  Total controls: {len(correlator.reports)}")
    print(f"  Orphans: {len([r for r in correlator.reports if r.is_orphan()])}")
    print(f"  Duplicate shortcuts: {len(correlator.duplicate_shortcuts)}")


if __name__ == "__main__":
    main()
