from __future__ import annotations

from collections import deque


class TailBuffer:
    def __init__(self, *, max_bytes: int):
        self._max_bytes = int(max_bytes)
        self._chunks: deque[bytes] = deque()
        self._total = 0
        self._dropped = 0

    def add(self, data: bytes) -> None:
        if not data:
            return
        self._chunks.append(data)
        self._total += len(data)
        while self._total > self._max_bytes and self._chunks:
            overflow = self._total - self._max_bytes
            left = self._chunks[0]
            if overflow >= len(left):
                self._chunks.popleft()
                self._total -= len(left)
                self._dropped += len(left)
                continue
            self._chunks[0] = left[overflow:]
            self._total -= overflow
            self._dropped += overflow
            break

    def to_bytes(self) -> bytes:
        return b"".join(self._chunks)

    @property
    def dropped_bytes(self) -> int:
        return self._dropped


def tail_text(text: str, *, max_lines: int, max_chars: int) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    lines = lines[-int(max_lines) :]
    out = "\n".join(lines)
    if len(out) > int(max_chars):
        out = out[-int(max_chars) :]
    return out
