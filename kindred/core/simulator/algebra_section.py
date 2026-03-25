from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "extract_algebra_section_text",
    "upsert_lines_into_algebra_section",
]


def extract_algebra_section_text(dsl_text: str) -> str:
    """
    Return the raw text inside all `# Algebra...` sections of a DSL text.

    - Includes lines between a `# Algebra`/`# Algebraic ...` header and the next `# ` header.
    - Excludes the `# Algebra...` header lines themselves.
    - Preserves original line ordering.
    """
    lines = str(dsl_text or "").splitlines()
    out: list[str] = []
    in_algebra = False
    for raw in lines:
        stripped = raw.strip()
        lower = stripped.lower()
        if lower.startswith("# algebra"):
            in_algebra = True
            continue
        if lower.startswith("# ") and in_algebra and not lower.startswith("# algebra"):
            in_algebra = False
        if in_algebra:
            out.append(raw.rstrip("\n"))
    return "\n".join(out).rstrip("\n")


def upsert_lines_into_algebra_section(
    dsl_text: str,
    lines_to_add: Sequence[str],
    *,
    header: str = "# Algebra",
) -> str:
    """
    Ensure the DSL text has an `# Algebra...` section, and append the given lines into it.

    If an algebra section exists, inserts just before the section terminator (next `# ` header
    that is not `# Algebra...`). Otherwise, appends a new `# Algebra` section at the end.
    """
    additions = [str(x).rstrip("\n") for x in (lines_to_add or []) if str(x).strip()]
    if not additions:
        return str(dsl_text or "")

    lines = str(dsl_text or "").splitlines()

    header_idx = None
    in_algebra = False
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        lower = stripped.lower()
        if lower.startswith("# algebra"):
            header_idx = idx
            in_algebra = True
            continue
        if lower.startswith("# ") and in_algebra and not lower.startswith("# algebra"):
            in_algebra = False
            break

    if header_idx is None:
        new_lines = list(lines)
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(str(header).strip() or "# Algebra")
        new_lines.extend(additions)
        return "\n".join(new_lines).rstrip("\n") + "\n"

    # Find insertion point at the end of the first algebra section.
    insert_at = len(lines)
    in_algebra = False
    for idx in range(header_idx, len(lines)):
        stripped = lines[idx].strip()
        lower = stripped.lower()
        if lower.startswith("# algebra"):
            in_algebra = True
            continue
        if lower.startswith("# ") and in_algebra and not lower.startswith("# algebra"):
            insert_at = idx
            break

    new_lines = list(lines)
    # Keep a blank line separator when appending into a non-empty section.
    if insert_at > header_idx + 1 and new_lines[insert_at - 1].strip() and additions[0].strip():
        additions = [""] + additions
    new_lines[insert_at:insert_at] = additions
    return "\n".join(new_lines).rstrip("\n") + "\n"

