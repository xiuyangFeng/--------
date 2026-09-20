"""Locate an owned tracker section by its heading.

The tracker is edited by hand in a rich-text Markdown editor, so the section
boundaries cannot be HTML comments (raw HTML) or `[//]: # (...)` lines
(reference-style link definitions) — both force the editor into code mode. The
heading itself is the anchor instead: a section runs from its heading line to
just before the next heading of the same or shallower level.
"""
from __future__ import annotations

import re


def heading_level(line: str) -> int:
    match = re.match(r"(#{1,6}) \S", line)
    return len(match.group(1)) if match else 0


def section_bounds(lines: list[str], heading: str) -> tuple[int, int]:
    """Return the [start, end) line span owned by `heading`; the heading must be unique."""
    level = heading_level(heading)
    if not level:
        raise ValueError(f"anchor is not a Markdown heading: {heading!r}")
    hits = [i for i, line in enumerate(lines) if line == heading]
    if len(hits) != 1:
        raise ValueError(f"anchor heading occurs {len(hits)} times, expected exactly 1: {heading!r}")
    start = hits[0]
    for i in range(start + 1, len(lines)):
        nested = heading_level(lines[i])
        if nested and nested <= level:
            return start, i
    return start, len(lines)


def get_section(text: str, heading: str) -> str:
    lines = text.split("\n")
    start, end = section_bounds(lines, heading)
    return "\n".join(lines[start:end])


def split_section(text: str, heading: str) -> tuple[str, str, str]:
    """Split into (before, section, after); `before + section + after` rebuilds the text."""
    lines = text.split("\n")
    start, end = section_bounds(lines, heading)
    before = "".join(line + "\n" for line in lines[:start])
    section = "".join(line + "\n" for line in lines[start:end])
    after = "\n".join(lines[end:])
    if end == len(lines) and section.endswith("\n"):
        section, after = section[:-1], after
    return before, section, after


def replace_or_append(text: str, heading: str, section: str) -> str:
    """Swap the section owned by `heading` for `section`, or append it when absent."""
    if not section.startswith(heading + "\n") and section.rstrip("\n") != heading:
        raise ValueError("replacement section must start with its own anchor heading")
    if text.count(heading + "\n") == 0:
        return text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"
    before, _, after = split_section(text, heading)
    separator = "\n" if after else ""
    return before + section.rstrip("\n") + "\n" + separator + after
