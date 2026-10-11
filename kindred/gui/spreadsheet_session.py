"""The temporary scientific workbook, owned by the calculation child."""
from __future__ import annotations

import csv
import io
import math
import re
from decimal import Decimal

import ironcalc

from kindred.core.datasets.foundation import RawDatasetRecord, thaw_json_value
from kindred.core.datasets.observation_payload import created_dataset_payload
from kindred.core.datasets.units import parse_measurement_unit, parse_time_unit
from kindred.core.project_document import new_dataset_id
from kindred.core.quantity_units import ONE_SCALE, conversion_scale, parse_unit
from kindred.gui.display_name_policy import and_join
from kindred.gui.spreadsheet_formulas import compile_columns, formula_at_row, pattern_from_row, remap_pattern


def _number(text):
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
        if not math.isfinite(value) or value == 0 and Decimal(text) != 0:
            raise ValueError
    except (ValueError, ArithmeticError):
        raise ValueError(f"A finite number or blank is required, not {text!r}.") from None
    return value


# IronCalc's worksheet grid; operations beyond it are refused before any change.
_MAX_ROWS, _MAX_COLUMNS = 1_048_576, 16_384


def _within_grid(bottom, right):
    if bottom > _MAX_ROWS or right > _MAX_COLUMNS:
        raise ValueError(f"A worksheet holds at most {_MAX_ROWS:,} rows and {_MAX_COLUMNS:,} columns.")


def _new_columns(columns, count):
    names = {column["name"] for column in columns}
    result, number = [], 1
    while len(result) < count:
        name = f"New column {number}"
        number += 1
        if name not in names:
            names.add(name)
            result.append({"name": name, "unit": None, "declared": False, "formula": None})
    return result


def _calculated(columns):
    return any(column["formula"] is not None for column in columns)


def _prepared_columns(columns):
    compiled = compile_columns(columns)
    columns = list(columns)
    for column, pattern in compiled.items():
        columns[column - 1] = {**columns[column - 1], "unit": pattern.unit.symbol if pattern.unit else None,
                               "declared": True}
    return columns, compiled


class SpreadsheetSession:
    """One live UserModel with column definitions and associated native history."""

    def __init__(self, records, output_names):
        raw = ironcalc.create("Kindred temporary data")
        self.sheets = []
        used = set()
        for sheet, (record, output) in enumerate(zip(records, output_names, strict=True)):
            metadata = thaw_json_value(record.source_metadata)
            source_name = metadata.get("sheet_name") if metadata is not None else None
            base = re.sub(r"[\[\]:*?/\\]", "_", source_name or record.display_name)[:31]
            name, suffix = base, 1
            while name.casefold() in used:
                ending = f"_{suffix}"
                name = f"{base[:31 - len(ending)]}{ending}"
                suffix += 1
            used.add(name.casefold())
            if sheet:
                raw.add_sheet(name)
            else:
                raw.rename_sheet(0, name)
            rows = sorted({row for series in record.observations for row in series.source_rows})
            positions = {row: index for index, row in enumerate(rows, 1)}
            timed = bool(record.observations[0].t)
            columns = ([{"name": record.time_column_name, "unit": "s", "declared": True, "formula": None}]
                       if timed else [])
            for column, series in enumerate(record.observations, 1 + timed):
                columns.append({"name": series.column_name, "unit": series.unit.symbol if series.unit else None,
                                "declared": True, "formula": None})
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
        self._read_model = None

    def state(self, shown=None):
        """Sheets, Undo/Redo availability, and the sheet an Undo/Redo changed."""
        return self.sheets, self.position > 0, self.position < len(self.actions), shown

    def _raw(self):
        # Typed reads need the raw API; this snapshot never calculates or mutates.
        if self._read_model is None:
            self._read_model = ironcalc.load_from_bytes(self.model.to_bytes())
        return self._read_model

    def _metadata(self, changes, sheet, key, before, after):
        change = (sheet, key, before, after)
        self._restore(change, False)
        changes.append(change)

    def _restore(self, change, reverse):
        sheet, key, before, after = change
        value = before if reverse else after
        self.sheets[sheet][key] = list(value) if key == "columns" else value

    def view(self, sheet, area):
        raw = self._raw()
        top, left, bottom, right = area
        return {(row, column): (
            format(value, ".6g") if type(value := raw.get_cell_value(sheet, row, column)) in (int, float)
            else self.model.get_formatted_cell_value(sheet, row, column),
            repr(value) if type(value) in (int, float) else "" if value is None else str(value),
            self.model.get_cell_type(sheet, row, column) == ironcalc.CellType.ErrorValue,
        ) for row in range(top, bottom + 1) for column in range(left, right + 1)}

    def copy(self, sheet, area):
        raw = self._raw()
        top, left, bottom, right = area
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for row in range(top, bottom + 1):
            writer.writerow("" if (value := raw.get_cell_value(sheet, row, column)) is None else
                            repr(value) if type(value) in (int, float) else str(value)
                            for column in range(left, right + 1))
        return output.getvalue()

    def _write(self, sheet, area, records):
        top, left, bottom, _right = area
        if bottom < top:
            return 0
        output = io.StringIO()
        csv.writer(output, delimiter="\t", lineterminator="\n").writerows(records)
        self.model.set_selected_sheet(sheet)
        self.model.set_selected_cell(top, left)
        self.model.set_selected_range(*area)
        self.model.paste_csv_string(sheet, *area, output.getvalue())
        return 1

    def _calculate(self, sheet, initial, compiled, top, bottom):
        if not compiled or bottom < top:
            return 0
        rows = range(top, bottom + 1)

        def filled(row, column):
            # Input columns hold numbers or blanks and formula columns formulas
            # or blanks, so the live cell content shows which are present.
            return self.model.get_cell_content(sheet, row, column) != ""

        references = {ref for pattern in compiled.values() for ref in pattern.references}
        present = {ref: bytearray(filled(row, ref) for row in rows) for ref in references if ref not in compiled}
        count = 0
        for column, pattern in compiled.items():
            mask = present[column] = bytearray(all(present[ref][offset] for ref in pattern.references)
                                               for offset in range(len(rows)))
            first, last = 0, len(rows) - 1
            if column in initial and initial[column].formula(1) == pattern.formula(1):
                # An unchanged pattern keeps its cells except in rows that gained or lost inputs.
                changed = [offset for offset, row in enumerate(rows) if mask[offset] != filled(row, column)]
                if not changed:
                    continue
                first, last = changed[0], changed[-1]
            count += self._write(sheet, (top + first, column, top + last, column),
                                 ((pattern.formula(row) if mask[offset] else "",)
                                  for offset, row in enumerate(rows[first:last + 1], first)))
        return count

    @staticmethod
    def _input_destination(columns, left, right):
        for column in range(left, min(right, len(columns)) + 1):
            if columns[column - 1]["formula"] is not None:
                raise ValueError(f"Column {ironcalc.column_name_from_number(column)} is calculated by its formula. "
                                 "Edit the formula instead.")

    def edit(self, operation, sheet, args):
        if operation in ("undo", "redo"):
            reverse = operation == "undo"
            if reverse and self.position == 0 or not reverse and self.position == len(self.actions):
                return self.state()
            shown, native_count, changes = self.actions[self.position - 1 if reverse else self.position]
            calculated = _calculated(self.sheets[shown]["columns"])
            if native_count:
                self.model.pause_evaluation()
                try:
                    for _ in range(native_count):
                        (self.model.undo if reverse else self.model.redo)()
                finally:
                    self.model.resume_evaluation()
                self._read_model = None
            for change in reversed(changes) if reverse else changes:
                self._restore(change, reverse)
            # Formulas use only their own worksheet, so other results stay current.
            if native_count and (calculated or _calculated(self.sheets[shown]["columns"])):
                self.model.evaluate()
            self.position += -1 if reverse else 1
            self.model.flush_send_queue()
            return self.state(shown)

        info, changes = self.sheets[sheet], []
        columns, rows = list(info["columns"]), info["rows"]
        native_count, recalculate = 0, None
        compiled = initial = compile_columns(columns)
        self.model.pause_evaluation()
        try:
            if operation == "formula":
                # Every cell and formula-bar entry arrives here.
                row, column, text = args
                text = text.strip()
                spec = columns[column - 1]
                letter = ironcalc.column_name_from_number(column)
                if text.startswith("="):
                    if info["timed"] and column == 1:
                        raise ValueError("Recorded time holds typed times, not formulas.")
                    source = pattern_from_row(text, row)
                    if source == spec["formula"]:
                        return self.state()
                    columns[column - 1] = {**spec, "formula": source}
                    columns, compiled = _prepared_columns(columns)
                    recalculate = 1, rows
                elif spec["formula"] is not None:
                    if text:
                        raise ValueError(f"Column {letter} is calculated by {formula_at_row(spec['formula'], row)}. "
                                         "Edit it, or delete it including = to keep the values.")
                    # Removing the formula keeps its current results as numbers.
                    raw = self._raw()
                    values = []
                    for result_row in range(1, rows + 1):
                        value = raw.get_cell_value(sheet, result_row, column)
                        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                            raise ValueError(f"Column {letter} has an error in row {result_row}. "
                                             "Fix the error before removing the formula.")
                        values.append(("" if value is None else repr(value),))
                    columns[column - 1] = {**spec, "formula": None}
                    columns, compiled = _prepared_columns(columns)
                    native_count = self._write(sheet, (1, column, rows, column), values)
                    recalculate = 1, rows
                else:
                    _within_grid(row, column)
                    value = _number(text)
                    if value == self._raw().get_cell_value(sheet, row, column):
                        return self.state()
                    native_count = self._write(sheet, (row, column, row, column), [("" if value is None else repr(value),)])
                    recalculate = min(row, rows + 1), row
                    rows = max(rows, row)
            elif operation == "clear":
                if not columns:
                    return self.state()
                top, left, bottom, right = args
                selected = [column for column in range(left, right + 1) if columns[column - 1]["formula"] is None]
                if not selected:
                    raise ValueError("Formula cells cannot be cleared. Clear the cells the formula uses, or edit or delete the formula.")
                raw = self._raw()
                if all(raw.get_cell_value(sheet, row, column) is None
                       for column in selected for row in range(top, bottom + 1)):
                    return self.state()
                for column in selected:
                    self.model.range_clear_contents(sheet, top, column, bottom, column)
                    native_count += 1
                recalculate = top, min(bottom, rows)
            elif operation == "paste":
                area, text = args
                records = list(csv.reader(io.StringIO(text), delimiter="\t", strict=True))
                if not records:
                    return self.state()
                width = next((len(row) for row in records if row), 1)
                if any(row and len(row) != width for row in records):
                    raise ValueError("Paste requires the same number of tab-separated columns in each row.")
                top, left = area[:2]
                bottom, right = top + len(records) - 1, left + width - 1
                _within_grid(bottom, right)
                self._input_destination(columns, left, right)
                values = [[_number(text) for text in row or [""] * width] for row in records]
                raw = self._raw()
                if right <= len(columns) and bottom <= rows and all(
                        raw.get_cell_value(sheet, top + i, left + j) == value
                        for i, row in enumerate(values) for j, value in enumerate(row)):
                    return self.state()
                if right > len(columns):
                    columns.extend(_new_columns(columns, right - len(columns)))
                native_count = self._write(sheet, (top, left, bottom, right),
                                           (("" if value is None else repr(value) for value in row) for row in values))
                recalculate = min(top, rows + 1), bottom
                rows = max(rows, bottom)
            elif operation == "fill":
                source, bottom = args
                if bottom <= source[2]:
                    return self.state()
                self._input_destination(columns, source[1], source[3])
                if bottom > rows:
                    raise ValueError("Fill through existing worksheet rows only.")
                self.model.auto_fill_rows(sheet, *source, bottom)
                native_count = 1
                recalculate = source[2] + 1, bottom
            elif operation in ("move_rows", "move_columns"):
                start, count, delta = args
                if not delta:
                    return self.state()
                if operation == "move_columns":
                    if info["timed"] and min(start, start + delta) == 1:
                        raise ValueError("Recorded time remains the first coordinate column.")
                    order = list(range(1, len(columns) + 1))
                    moved = order[start - 1:start - 1 + count]
                    del order[start - 1:start - 1 + count]
                    order[start + delta - 1:start + delta - 1] = moved
                    positions = {old: new for new, old in enumerate(order, 1)}
                    columns = [columns[old - 1] for old in order]
                    columns = self._remap(columns, positions)
                getattr(self.model, operation)(sheet, start, count, delta)
                native_count = 1
            elif operation in ("insert_rows", "delete_rows"):
                start, count = args
                if operation == "insert_rows":
                    _within_grid(rows + count, 1)
                getattr(self.model, operation)(sheet, start, count)
                native_count = 1
                rows += count if operation == "insert_rows" else -count
                if operation == "insert_rows":
                    recalculate = start, start + count - 1
            elif operation in ("insert_columns", "delete_columns"):
                start, count = args
                if info["timed"] and start == 1:
                    raise ValueError("Recorded time remains the first coordinate column.")
                if operation == "insert_columns":
                    _within_grid(1, len(columns) + count)
                    positions = {old: old + count if old >= start else old for old in range(1, len(columns) + 1)}
                    columns[start - 1:start - 1] = _new_columns(columns, count)
                else:
                    if start + count - 1 > len(columns):
                        return self.state()
                    deleted = set(range(start, start + count))
                    users = [columns[column - 1]["name"] for column, pattern in compiled.items()
                             if column not in deleted and pattern.references & deleted]
                    if users:
                        raise ValueError(f"Cannot delete columns used by the formulas of {and_join(users)}. Delete those columns too, or edit their formulas first.")
                    positions = {old: old - count if old >= start + count else old
                                 for old in range(1, len(columns) + 1) if old not in deleted}
                    del columns[start - 1:start - 1 + count]
                columns = self._remap(columns, positions)
                getattr(self.model, operation)(sheet, start, count)
                native_count = 1
            elif operation == "column":
                index, field, value, correction = args
                before = columns[index]
                if field == "name":
                    value = value.strip()
                    if not value or any(other != index and column["name"] == value for other, column in enumerate(columns)):
                        raise ValueError("Column names must be nonempty and unique.")
                    columns[index] = {**before, "name": value}
                else:
                    if before["formula"] is not None:
                        raise ValueError("This unit follows from the column formula.")
                    unit = (parse_time_unit if info["timed"] and index == 0 else parse_measurement_unit)(value)
                    columns[index] = {**before, "unit": unit.symbol, "declared": True}
                    columns, compiled = _prepared_columns(columns)
                    if before["declared"] and not correction and before["unit"] != unit.symbol:
                        if before["unit"] is None:
                            raise ValueError("This column has Unspecified units. Use Correct Unit Declaration… to assign its meaning.")
                        previous = parse_unit(before["unit"])
                        if not previous.compatible(unit):
                            raise ValueError(f"Incompatible units: {before['unit']} and {unit.symbol}. Use Correct Unit "
                                             "Declaration… to change what the numbers mean.")
                        scale = conversion_scale(previous, unit)
                        if scale != ONE_SCALE:
                            raw = self._raw()
                            values = [raw.get_cell_value(sheet, row, index + 1) for row in range(1, rows + 1)]
                            converted = [("" if value is None else repr(scale.apply(value)),) for value in values]
                            native_count = self._write(sheet, (1, index + 1, rows, index + 1), converted)
                    if columns != info["columns"]:
                        recalculate = 1, rows
            elif operation == "name":
                name = args[0].strip()
                if not name:
                    raise ValueError("New dataset name must not be blank.")
                if name != info["name"]:
                    self._metadata(changes, sheet, "name", info["name"], name)
            else:
                raise ValueError(f"Unknown spreadsheet operation: {operation}")
            if native_count:
                self._read_model = None
            if recalculate is not None:
                native_count += self._calculate(sheet, initial, compiled, *recalculate)
        finally:
            self.model.resume_evaluation()
            if native_count:
                # Formulas use only their own worksheet, so other results stay current.
                if _calculated(info["columns"]) or _calculated(columns):
                    self.model.evaluate()
                self._read_model = None
                self.model.flush_send_queue()
        if columns != info["columns"]:
            self._metadata(changes, sheet, "columns", tuple(info["columns"]), tuple(columns))
        if rows != info["rows"]:
            self._metadata(changes, sheet, "rows", info["rows"], rows)
        if native_count or changes:
            self.actions[self.position:] = [(sheet, native_count, tuple(changes))]
            self.position += 1
        return self.state()

    @staticmethod
    def _remap(columns, positions):
        return [{**column, "formula": remap_pattern(column["formula"], positions)}
                if column["formula"] is not None else column for column in columns]

    def outputs(self):
        raw = self._raw()
        records = []
        for sheet, info in enumerate(self.sheets):
            columns = []
            for column, spec in enumerate(info["columns"], 1):
                if not spec["declared"]:
                    raise ValueError(f"{info['sheet']}: choose a unit for {spec['name']}.")
                values = []
                for row in range(1, info["rows"] + 1):
                    value = raw.get_cell_value(sheet, row, column)
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                        cell = f"{ironcalc.column_name_from_number(column)}{row}"
                        raise ValueError(f"{info['sheet']}!{cell}: a finite number or blank is required ({value!r}).")
                    values.append(value)
                unit = parse_unit(spec["unit"]) if spec["unit"] is not None else None
                columns.append((spec["name"], unit, values))
            time = (columns[0][1], columns[0][2]) if info["timed"] else None
            try:
                payload = created_dataset_payload(columns[1:] if info["timed"] else columns, time=time)
                record = RawDatasetRecord(dataset_id=new_dataset_id(), display_name=info["name"].strip(),
                                          time_column_name=columns[0][0] if info["timed"] else None,
                                          source_id=None, source_metadata=None, observations=payload.observations)
                records.append(record)
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
                except (csv.Error, ValueError) as exc:
                    pipe.send((operation, session.state(), str(exc)))
    except EOFError:
        pass
    except Exception as exc:
        pipe.send(("failed", None, str(exc)))
    finally:
        pipe.close()
