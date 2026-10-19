"""Filename inference (MD9; TST-M-05). Pure: no file access, no tag library.

``compile_pattern`` turns a ``--infer-from-filename`` pattern into a
:class:`Pattern`; ``infer`` applies it to one C-01 relative path and returns
the typed values as ``ManifestCell`` opinions (operation ``set``) plus at most
one informational note. Whether an inferred value is used (``fill`` or
``override``, MD12) is the resolver's decision, not this module's.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from norma.metadata import model
from norma.models import ManifestCell

TEXT_PLACEHOLDERS = ("title", "artist", "album", "genre")
PLACEHOLDERS = (*TEXT_PLACEHOLDERS, "track", "disc", "date", "year", "*")

_REGEX = {
    "track": r"[0-9]+",
    "disc": r"[0-9]+",
    "year": r"[0-9]{4}",
    "date": r"[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?",
    "*": r"[^/]*?",
}
_TEXT_REGEX = r"[^/]+?"
_TOKEN = re.compile(r"\{([^{}]*)\}")


class InvalidPattern(ValueError):
    """The pattern itself is unusable (a usage error for the caller, never a per-file note)."""


@dataclass(frozen=True)
class Pattern:
    source: str
    separators: int
    """*k*: the number of ``/`` in the pattern; it is matched against the last *k* + 1 components."""
    placeholders: tuple[str, ...]
    """Capturing placeholders in order of appearance, repeats included."""
    loose: re.Pattern[str]
    """Independent group per placeholder; repeats are compared after processing."""
    strict: re.Pattern[str]
    """Repeats are back-references: the engine backtracks to find equal captures."""


def compile_pattern(pattern: str) -> Pattern:
    """Compile ``pattern``; raises :class:`InvalidPattern` for an empty pattern, an unknown
    placeholder, an unbalanced brace or an empty path component."""
    text = unicodedata.normalize("NFC", pattern)
    if not text.strip():
        raise InvalidPattern("the pattern is empty")
    if "\x00" in text:
        raise InvalidPattern("the pattern contains NUL")
    if any(not part for part in text.split("/")):
        raise InvalidPattern("the pattern has an empty path component")
    names: list[str] = []
    loose: list[str] = []
    strict: list[str] = []
    seen: dict[str, int] = {}
    position = 0
    for token in _TOKEN.finditer(text):
        literal = text[position:token.start()]
        if "{" in literal or "}" in literal:
            raise InvalidPattern("unbalanced brace in the pattern")
        loose.append(re.escape(literal))
        strict.append(re.escape(literal))
        name = token.group(1)
        if name not in PLACEHOLDERS:
            raise InvalidPattern(f"unknown placeholder {{{name}}} — use " + " ".join(f"{{{p}}}" for p in PLACEHOLDERS))
        body = _TEXT_REGEX if name in TEXT_PLACEHOLDERS else _REGEX[name]
        if name == "*":
            loose.append(f"(?:{body})")
            strict.append(f"(?:{body})")
        else:
            count = seen.get(name, 0)
            seen[name] = count + 1
            loose.append(f"(?P<{name}_{count}>{body})")
            strict.append(f"(?P<{name}_{count}>{body})" if count == 0 else f"(?P={name}_0)")
            names.append(name)
        position = token.end()
    tail = text[position:]
    if "{" in tail or "}" in tail:
        raise InvalidPattern("unbalanced brace in the pattern")
    loose.append(re.escape(tail))
    strict.append(re.escape(tail))
    return Pattern(text, text.count("/"), tuple(names),
                   re.compile("".join(loose) + r"\Z", re.DOTALL), re.compile("".join(strict) + r"\Z", re.DOTALL))


@dataclass(frozen=True)
class Inference:
    cells: Mapping[str, ManifestCell]
    """Typed opinions (``set``) by field, provenance ``filename``; ``track``/``disc`` are ints."""
    matched: bool
    notes: tuple[str, ...]
    """At most one informational note (no match, or invalid captures); never an error."""


def target_string(relative_path: PurePosixPath, separators: int) -> str | None:
    """The last *k* + 1 components of ``relative_path``, extension removed, NFC; ``None`` when shorter."""
    parts = list(relative_path.parts)
    if not parts or len(parts) < separators + 1:
        return None
    parts[-1] = PurePosixPath(parts[-1]).with_suffix("").name or parts[-1]
    return unicodedata.normalize("NFC", "/".join(parts[-(separators + 1):]))


def _process(raw: str, underscores: bool) -> str:
    text = raw.replace("_", " ") if underscores else raw
    return unicodedata.normalize("NFC", text.strip())


def _comparable(name: str, raw: str, underscores: bool) -> str:
    text = _process(raw, underscores)
    return str(int(text)) if name in ("track", "disc") else text


def _groups(match: re.Match[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for key, value in match.groupdict().items():
        grouped.setdefault(key.rsplit("_", 1)[0], []).append(value)
    return grouped


def infer(pattern: Pattern, relative_path: PurePosixPath, *, underscores: bool = False) -> Inference:
    """Apply ``pattern`` to ``relative_path`` (MD9)."""
    no_match = Inference({}, False, (f"filename does not match pattern {pattern.source!r}",))
    target = target_string(relative_path, pattern.separators)
    match = pattern.loose.match(target) if target is not None else None
    if match is None:
        return no_match
    groups = _groups(match)

    def repeats_equal(found: dict[str, list[str]]) -> bool:
        return all(len({_comparable(name, v, underscores) for v in values}) == 1 for name, values in found.items())

    if not repeats_equal(groups):
        match = pattern.strict.match(target or "")
        if match is None:
            return no_match
        groups = _groups(match)
    cells: dict[str, ManifestCell] = {}
    problems: list[str] = []
    blocked: set[str] = set()

    def accept(field: str, value: str | int, shown: str) -> None:
        if field in blocked:
            return
        if field in cells and cells[field].value != value:
            problems.append(f"{field} {shown!r} disagrees with another capture")
            del cells[field]
            blocked.add(field)
            return
        cells[field] = ManifestCell("set", value)

    for name, captured in groups.items():
        text = _process(captured[0], underscores)
        if name in TEXT_PLACEHOLDERS:
            try:
                value = model.canonical_text(text, name)
            except model.InvalidValue as exc:
                problems.append(f"{name} {text!r}: {exc}")
                continue
            if not value:
                problems.append(f"{name} is empty")
                continue
            accept(name, value, text)
        elif name in ("track", "disc"):
            number = int(text)
            if number < 1:
                problems.append(f"{name} {text!r} is not 1 or more")
                continue
            accept(name, number, text)
        else:  # date or year: both set the date field
            try:
                accept("date", model.canonical_date(text), text)
            except model.InvalidValue as exc:
                problems.append(f"{name} {text!r}: {exc}")
    notes = (f"filename pattern {pattern.source!r}: " + "; ".join(problems),) if problems else ()
    return Inference(cells, True, notes)
