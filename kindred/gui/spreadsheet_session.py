"""The temporary workbook, used only by the spreadsheet calculation child."""
from __future__ import annotations

import csv
import io
import math
import re

import ironcalc

from kindred.core.datasets.foundation import RawDatasetRecord, validate_raw_dataset_record
from kindred.core.datasets.observation_payload import created_dataset_payload
from kindred.core.project_document import new_dataset_id
from kindred.core.quantity_units import scientific_unit


class SpreadsheetSession:
    """One live UserModel and the column settings accompanying its native history."""

    def __init__(self, records, output_names):
        raw = ironcalc.create("Kindred temporary data")
        self.sheets = []
        used = set()
        for sheet, (record, output) in enumerate(zip(records, output_names, strict=True)):
            base = re.sub(r"[\[\]:*?/\\]", "_", record.display_name).strip("'")[:25] or "Data"
            name, suffix = base, 1
            while name.casefold() in used:
                name = f"{base}_{suffix}"
                suffix += 1
            used.add(name.casefold())
            if sheet:
                raw.add_sheet(name)
            else:
                raw.rename_sheet(0, name)
            rows = sorted({row for series in record.observations for row in series.source_rows})
            positions = {row: index for index, row in enumerate(rows, 1)}
            timed = bool(record.observations[0].t)
            columns = [("Time", "s")] if timed else []
            for column, series in enumerate(record.observations, 1 + timed):
                columns.append((series.column_name, scientific_unit(series.unit.dimensions).symbol))
                for index, row in enumerate(series.source_rows):
                    raw.update_cell_with_number(sheet, positions[row], column, series.y[index])
                    if timed:
                        raw.update_cell_with_number(sheet, positions[row], 1, series.t[index])
            self.sheets.append({"sheet": name, "source": record.display_name, "name": output,
                                "timed": timed, "columns": columns, "rows": len(rows)})
        raw.evaluate()
        self.model = ironcalc.UserModel.from_bytes(raw.to_bytes())
        self.actions = []
        self.position = 0
        self.clipboard = None

    def state(self):
        return self.sheets, self.position > 0, self.position < len(self.actions)

    def _metadata(self, changes, sheet, key, index, before, after):
        change = (sheet, key, index, before, after)
        self._restore(change, False)
        changes.append(change)

    def _restore(self, change, reverse):
        sheet, key, index, before, after = change
        old, new = (after, before) if reverse else (before, after)
        if key == "columns":
            self.sheets[sheet][key][index:index + len(old)] = new
        else:
            self.sheets[sheet][key] = new

    def _grow(self, sheet, rows, columns, changes):
        info = self.sheets[sheet]
        if rows > info["rows"]:
            self._metadata(changes, sheet, "rows", None, info["rows"], rows)
        added = []
        names = {name for name, _unit in info["columns"]}
        for index in range(len(info["columns"]), columns):
            name, suffix = f"Column {index + 1}", 1
            while name in names:
                name = f"Column {index + 1}_{suffix}"
                suffix += 1
            names.add(name)
            added.append((name, None))
        if added:
            self._metadata(changes, sheet, "columns", len(info["columns"]), (), tuple(added))

    def _content(self, raw, sheet, row, column):
        value = raw.get_cell_value(sheet, row, column)
        if raw.get_cell_formula(sheet, row, column) is None and type(value) in (int, float):
            return repr(value)
        return self.model.get_cell_content(sheet, row, column)

    def view(self, sheet, area):
        raw = ironcalc.load_from_bytes(self.model.to_bytes())
        top, left, bottom, right = area
        return {(row, column): (
            format(value, ".6g") if type(value := raw.get_cell_value(sheet, row, column)) in (int, float)
            else self.model.get_formatted_cell_value(sheet, row, column),
            self._content(raw, sheet, row, column),
            self.model.get_cell_type(sheet, row, column) == ironcalc.CellType.ErrorValue,
        ) for row in range(top, bottom + 1) for column in range(left, right + 1)}

    def copy(self, sheet, area):
        raw = ironcalc.load_from_bytes(self.model.to_bytes())
        self.model.set_selected_sheet(sheet)
        self.model.set_selected_cell(area[0], area[1])
        self.model.set_selected_range(*area)
        self.clipboard = self.model.copy_to_clipboard()
        top, left, bottom, right = area
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for row in range(top, bottom + 1):
            cells = self.clipboard["data"].setdefault(row, {})
            texts = []
            for column in range(left, right + 1):
                if column not in cells:
                    cells[column] = {"is_spill": False, "style": self.model.get_cell_style(sheet, row, column)}
                cells[column]["text"] = self._content(raw, sheet, row, column)
                if cells[column]["is_spill"]:
                    structure = self.model.get_cell_array_structure(sheet, row, column)
                    anchor_row, anchor_column, _width, _height = next(iter(structure.values()))
                    # Without its formula, a selected spill result is a value.
                    if not (top <= anchor_row <= bottom and left <= anchor_column <= right):
                        cells[column]["is_spill"] = False
                value = raw.get_cell_value(sheet, row, column)
                texts.append(repr(value) if type(value) in (int, float)
                             else self.model.get_formatted_cell_value(sheet, row, column))
            writer.writerow(texts)
        # Both clipboard forms include the selected trailing blanks.
        self.clipboard["range"] = area
        return output.getvalue()

    def edit(self, operation, sheet, args):
        if operation in ("undo", "redo"):
            reverse = operation == "undo"
            if reverse and self.position == 0 or not reverse and self.position == len(self.actions):
                return self.state()
            native_count, changes = self.actions[self.position - 1 if reverse else self.position]
            if native_count:
                self.model.pause_evaluation()
                try:
                    for _ in range(native_count):
                        (self.model.undo if reverse else self.model.redo)()
                finally:
                    self.model.resume_evaluation()
                self.model.evaluate()
            for change in reversed(changes) if reverse else changes:
                self._restore(change, reverse)
            self.position += -1 if reverse else 1
            self.model.flush_send_queue()
            return self.state()

        info, changes = self.sheets[sheet], []
        native_count = 1
        if operation == "set":
            row, column, text = args
            raw = ironcalc.load_from_bytes(self.model.to_bytes())
            if text == self._content(raw, sheet, row, column):
                return self.state()
            self.model.set_user_input(sheet, row, column, text)
            self._grow(sheet, row, column, changes)
        elif operation == "clear":
            self.model.range_clear_contents(sheet, *args)
        elif operation == "paste":
            area, text, native_copy = args
            records = list(csv.reader(io.StringIO(text), delimiter="\t", strict=True))
            if not records:
                return self.state()
            width = next((len(row) for row in records if row), 1)
            if any(row and len(row) != width for row in records):
                raise csv.Error("Paste requires the same number of tab-separated columns in each row.")
            self.model.set_selected_sheet(sheet)
            self.model.set_selected_cell(area[0], area[1])
            self.model.set_selected_range(*area)
            if native_copy:
                height = self.clipboard["range"][2] - self.clipboard["range"][0] + 1
                width = self.clipboard["range"][3] - self.clipboard["range"][1] + 1
                self.model.pause_evaluation()
                try:
                    if any(cell["is_spill"] for row in self.clipboard["data"].values() for cell in row.values()):
                        # Native paste records only styles at spill children.
                        # Record every overwritten value before it clears them.
                        self.model.range_clear_contents(sheet, area[0], area[1],
                                                        area[0] + height - 1, area[1] + width - 1)
                        native_count = 2
                    self.model.paste_from_clipboard(self.clipboard["sheet"], self.clipboard["range"],
                                                    self.clipboard["data"], False)
                finally:
                    self.model.resume_evaluation()
                self.model.evaluate()
            else:
                # IronCalc skips empty physical lines. Encode blank records as
                # empty fields so its single grouped paste retains their rows.
                normalized = io.StringIO()
                csv.writer(normalized, delimiter="\t", lineterminator="\n").writerows(
                    row or [""] * width for row in records)
                self.model.paste_csv_string(sheet, *area, normalized.getvalue())
                height = len(records)
            self._grow(sheet, area[0] + height - 1, area[1] + width - 1, changes)
        elif operation == "fill":
            top, left, bottom, right = args
            if top == bottom:
                return self.state()
            self.model.auto_fill_rows(sheet, top, left, top, right, bottom)
            self._grow(sheet, bottom, right, changes)
        elif operation in ("insert_rows", "delete_rows"):
            start, count = args
            getattr(self.model, operation)(sheet, start, count)
            size = info["rows"] + (count if operation == "insert_rows" else -count)
            self._metadata(changes, sheet, "rows", None, info["rows"], max(0, size))
        elif operation in ("insert_columns", "delete_columns"):
            start, count = args
            if info["timed"] and start == 1:
                raise ValueError("Recorded time remains the first coordinate column.")
            getattr(self.model, operation)(sheet, start, count)
            if operation == "insert_columns":
                names = {name for name, _unit in info["columns"]}
                new = []
                number = 1
                while len(new) < count:
                    name = f"New column {number}"
                    number += 1
                    if name not in names:
                        names.add(name)
                        new.append((name, None))
                self._metadata(changes, sheet, "columns", start - 1, (), tuple(new))
            else:
                old = tuple(info["columns"][start - 1:start - 1 + count])
                self._metadata(changes, sheet, "columns", start - 1, old, ())
        elif operation == "column":
            index, name, unit = args
            before, after = (info["columns"][index],), ((name, unit),)
            native_count = 0
            if before != after:
                self._metadata(changes, sheet, "columns", index, before, after)
        elif operation == "name":
            native_count = 0
            if info["name"] != args[0]:
                self._metadata(changes, sheet, "name", None, info["name"], args[0])
        else:
            raise ValueError(f"Unknown spreadsheet operation: {operation}")
        if native_count:
            # Formula spills can enlarge another opened worksheet too.
            for index, sheet_info in enumerate(self.sheets):
                top, bottom, _left, right = self.model.get_sheet_dimensions(index)
                if bottom <= sheet_info["rows"] and right <= len(sheet_info["columns"]):
                    continue
                # Dimensions retain empty cells after undo; only live cells
                # and formula spill results warrant automatic growth.
                live_bottom = live_right = 0
                for row in range(top, bottom + 1):
                    column = self.model.get_last_non_empty_in_row_before_column(index, row, right + 1)
                    if column is not None:
                        live_bottom = row
                        live_right = max(live_right, column)
                self._grow(index, live_bottom, live_right, changes)
            self.model.flush_send_queue()
        if native_count or changes:
            self.actions[self.position:] = [(native_count, tuple(changes))]
            self.position += 1
        return self.state()

    def outputs(self):
        raw = ironcalc.load_from_bytes(self.model.to_bytes())
        records = []
        for sheet, info in enumerate(self.sheets):
            columns = []
            for column, (name, unit) in enumerate(info["columns"], 1):
                if unit is None:
                    raise ValueError(f"{info['sheet']}: choose a unit for {name}.")
                values = []
                for row in range(1, info["rows"] + 1):
                    value = raw.get_cell_value(sheet, row, column)
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                        cell = f"{ironcalc.column_name_from_number(column)}{row}"
                        raise ValueError(f"{info['sheet']}!{cell}: a finite number or blank is required ({value!r}).")
                    values.append(value)
                columns.append((name, unit, values))
            time = (columns[0][1], columns[0][2]) if info["timed"] else None
            try:
                payload = created_dataset_payload(columns[1:] if info["timed"] else columns, time=time)
                record = RawDatasetRecord(new_dataset_id(), info["name"].strip(), None, None, payload.observations)
                records.append(validate_raw_dataset_record(record))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{info['sheet']}: {exc}") from exc
        return tuple(records)


def run_spreadsheet_child(pipe, records, names):
    """Perform editor requests against one workbook until the pipe is closed."""
    try:
        session = SpreadsheetSession(records, names)
        pipe.send(("ready", session.state(), ""))
        while True:
            operation, sheet, args = pipe.recv()
            if operation == "outputs":
                try:
                    pipe.send((operation, session.outputs(), ""))
                except (ValueError, TypeError) as exc:
                    pipe.send((operation, None, str(exc)))
            elif operation == "view":
                pipe.send((operation, session.view(sheet, args), ""))
            elif operation == "copy":
                pipe.send((operation, session.copy(sheet, args), ""))
            else:
                try:
                    pipe.send((operation, session.edit(operation, sheet, args), ""))
                except csv.Error as exc:
                    pipe.send((operation, session.state(), str(exc)))
    except EOFError:
        pass
    except Exception as exc:
        pipe.send(("failed", None, str(exc)))
    finally:
        pipe.close()
