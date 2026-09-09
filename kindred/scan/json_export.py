"""Incremental JSON export; no result-repository or scientific evaluation ownership."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy

from kindred._version import __version__ as KINDRED_VERSION
from kindred.scan.config import START, END


def plain(value):
    """Project existing scientific facts into JSON-supported values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    return value


def input_record(run):
    return {
        "run_id": run.run_id, "parameter_set": run.parameter_set, "initial_set": run.initial_set,
        "inputs": {
            "initials": {k.removeprefix("initial:"): v for k, v in run.values.items() if k.startswith("initial:")},
            "parameters": {k.removeprefix("parameter:"): v for k, v in run.values.items() if k.startswith("parameter:")},
            "t_start": run.values.get(START), "t_end": run.values.get(END),
        },
    }


def result_record(run, result, outputs):
    record = input_record(run)
    data, scalars, missing = {"t": result.t.tolist()}, {}, []
    for name in outputs:
        if name in result.series:
            data[name] = result.series[name].tolist()
        elif name in result.scalars:
            scalars[name] = result.scalars[name]
        else:
            missing.append(name)
    record.update({
        "simulation_status": "succeeded", "export_status": "failed" if missing else "succeeded",
        "missing_outputs": missing, "data": data, "scalars": scalars,
        "termination": plain(result.termination), "solver_provenance": plain(result.solver_provenance),
        "segment_ids": result.segment_ids.tolist(), "segment_keys": list(result.segment_keys),
        "discontinuity_ids": result.discontinuity_ids.tolist(),
        "parameters": plain(result.parameters),
        "algebra_errors": [error.to_payload() for error in result.algebra_errors],
        "boundaries": [{"time": b.time,
                        "pre": {k: v for k, v in b.pre_state if k in outputs},
                        "post": {k: v for k, v in b.post_state if k in outputs}}
                       for b in result.presentation_boundaries],
        "interventions": result.intervention_execution.provenance_projection(),
    })
    return record


class JsonScanWriter:
    def __init__(self, config):
        self.destination = os.path.abspath(config.destination)
        self.path = ""
        self.stream = None
        self.count = 0

    def initialize(self, config, design):
        directory = os.path.dirname(self.destination)
        descriptor, self.path = tempfile.mkstemp(prefix=f".{Path(self.destination).name}.",
                                                suffix=".partial.json", dir=directory)
        try:
            self.stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(descriptor)
            raise
        versions = {"kindred": KINDRED_VERSION, "numpy": np.__version__,
                    "scipy": scipy.__version__}
        header = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "versions": versions, "configuration": asdict(config),
                  "effective_ranges": plain(design.effective_ranges),
                  "sampling_note": "Conditional interval sampling; constrained physical coordinates are not guaranteed uniform or Latin-hypercube stratified."}
        encoded = json.dumps(header, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        self.stream.write(encoded[:-1] + ',"runs":[\n')

    def write(self, record):
        if self.count:
            self.stream.write(",\n")
        json.dump(record, self.stream, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        self.stream.flush()
        self.count += 1

    def finish(self, progress):
        summary = asdict(progress)
        summary["path"] = self.destination if progress.state == "complete" else self.path
        summary["not_started"] = progress.requested - progress.attempted
        self.stream.write('],"summary":')
        json.dump(summary, self.stream, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        self.stream.write("}\n")
        self.stream.close()
        if progress.state == "complete":
            os.replace(self.path, self.destination)
            self.path = self.destination
        return self.path

    def close(self):
        if self.stream is not None:
            self.stream.close()
