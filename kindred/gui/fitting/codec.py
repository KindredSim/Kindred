"""Protocol-5 fitting payloads and binary pipe framing."""

import copyreg
import io
import pickle
from types import MappingProxyType

def _immutable_mapping(values):
    return MappingProxyType(values)


def _reduce_mapping(values):
    return _immutable_mapping, (dict(values),)


class _Writer(pickle.Pickler):
    def __init__(self, stream, references):
        super().__init__(stream, protocol=5)
        self.references = references
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table[MappingProxyType] = _reduce_mapping

    def persistent_id(self, value):
        return next((name for name, reference in self.references.items()
                     if value is reference), None)


class _Reader(pickle.Unpickler):
    def __init__(self, stream, references):
        super().__init__(stream)
        self.references = references

    def persistent_load(self, name):
        return self.references[name]


def _encode_fit(value, references):
    stream = io.BytesIO()
    _Writer(stream, references).dump(value)
    return stream.getvalue()


def _decode_fit(payload, references):
    return _Reader(io.BytesIO(payload), references).load()


def read_frame(stream):
    """Return None only for clean EOF between frames."""
    header = stream.read(8)
    if not header:
        return None
    while len(header) < 8:
        chunk = stream.read(8 - len(header))
        if not chunk:
            raise EOFError("Truncated Fit frame header.")
        header += chunk
    remaining = int.from_bytes(header, "big")
    payload = bytearray()
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise EOFError("Truncated Fit frame payload.")
        payload.extend(chunk)
        remaining -= len(chunk)
    return bytes(payload)


def write_frame(stream, payload):
    """Complete and flush one frame, including short pipe writes."""
    for part in (len(payload).to_bytes(8, "big"), payload):
        pending = memoryview(part)
        while pending:
            written = stream.write(pending)
            if not written:
                raise BrokenPipeError("Fit pipe did not accept the frame.")
            pending = pending[written:]
    stream.flush()
