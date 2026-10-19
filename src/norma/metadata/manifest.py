"""Manifest v1 loading, matching and template content (C-11; schema MS1–MS13; MD28).

``load_manifest`` reads one local CSV or JSON file and returns the C-11
``ManifestData``; ``parse_manifest`` does the same from bytes and is pure.
Whole-manifest problems raise :class:`ManifestError` (a usage error, exit 2).
Row problems become ``ManifestIssue`` records and the invalid row keeps no
cells. ``match_manifest`` applies the lexical matching of MS5 and MS13 to
discovered inputs; it never opens a selector. ``build_template`` renders the
``tags template`` content as text for the CLI to print or write.

No network, no remote reference, no tag library. The only file access is the
single read in :func:`load_manifest`.
"""

from __future__ import annotations

import csv
import io
import json
import posixpath
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from norma.errors import UsageError
from norma.metadata import model
from norma.models import InputItem, ManifestCell, ManifestData, ManifestIssue, ManifestRow, MetadataValue

# --- Limits and vocabularies (MS6, MS8, MS9) --------------------------------------

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10_000
MAX_COLUMNS = 200
MAX_ISSUES = 50
MAX_ARTWORK_PATH = 1024

DEFAULTS_ROW = "*"
CLEAR_TEXT = "[clear]"

ROW_DETAILS: frozenset[str] = frozenset({
    "date_invalid", "date_spreadsheet", "number_invalid", "number_spreadsheet", "pair_inconsistent",
    "boolean_invalid", "text_invalid", "artwork_path_invalid", "illegal_extra_name",
    "reserved_extra_name", "unknown_field", "cell_count", "json_structure",
})
"""Details of a row-level problem inside ``ManifestData.issues``: the matched input fails."""

WARNING_DETAILS: frozenset[str] = frozenset({"likely_typo"})
MORE_ISSUES = "more_issues"
"""Final record when more than :data:`MAX_ISSUES` issues were found; ``field`` holds the count."""

ISSUE_TEXT: dict[str, str] = {
    "unreadable": "cannot read the manifest",
    "too_large": "over the manifest size limits (10 MiB, 10,000 rows, 200 columns)",
    "encoding": "not UTF-8 — save the manifest as UTF-8 (CSV UTF-8 in Excel)",
    "delimiter": 'cannot find a "file" column with comma or semicolon separators',
    "csv_syntax": "invalid CSV quoting",
    "missing_file_column": 'no "file" column — add one naming each audio file',
    "header_duplicate": "two columns are the same field",
    "header_empty_name": "a column has values but no name",
    "illegal_extra_name": "not a valid extra name — letters, digits and _ only, starting with a letter",
    "reserved_extra_name": "reserved for names Norma writes or removes",
    "defaults_duplicate": 'a second "*" row — only one defaults row is allowed',
    "defaults_invalid": "invalid value in the defaults",
    "file_value_invalid": "use a path relative to the manifest's folder, naming a file",
    "json_syntax": "invalid JSON",
    "version": "write the integer 1",
    "json_structure": "does not fit manifest version 1",
    "date_invalid": "not a real date (YYYY, YYYY-MM or YYYY-MM-DD)",
    "date_spreadsheet": "looks reformatted by a spreadsheet; format the column as Text and enter YYYY-MM-DD",
    "number_invalid": "write a whole number of 1 or more, without a decimal point",
    "number_spreadsheet": "a spreadsheet may have turned n/N into a date; use separate number and total columns",
    "pair_inconsistent": "the number is greater than the total, or the pair disagrees",
    "boolean_invalid": "true or false",
    "text_invalid": "not valid text — omit the key for no opinion, or use null to clear",
    "artwork_path_invalid": "empty or over 1024 characters",
    "unknown_field": "unknown field — extras go under \"extras\"",
    "cell_count": "more values than columns",
    "duplicate_match": "several rows match one input — keep one",
    "ambiguous_match": "matches several inputs — row not applied; use the relative path",
    "unmatched_row": "matches no input — row ignored",
    "likely_typo": "is used as an extra tag — did you mean a canonical field?",
    MORE_ISSUES: "more issues not shown",
}


def format_issue(issue: ManifestIssue, value: object | None = None) -> str:
    """MS8 message form ``<locator> <field>: "<value>" — <expected form>``."""
    if issue.detail == MORE_ISSUES:
        return f"{issue.locator}: {issue.field} more issues not shown"
    where = issue.locator + (f" {issue.field}" if issue.field else "")
    text = ISSUE_TEXT.get(issue.detail, issue.detail)
    return f'{where}: "{value}" — {text}' if value is not None else f"{where}: {text}"


class ManifestError(UsageError):
    """A whole-manifest error (MS8): the run stops before any file is processed (exit 2)."""

    def __init__(self, issues: Sequence[ManifestIssue], messages: Sequence[str]) -> None:
        self.issues: tuple[ManifestIssue, ...] = tuple(issues)
        self.messages: tuple[str, ...] = tuple(messages)
        super().__init__("\n".join(self.messages))

    @property
    def detail(self) -> str:
        return self.issues[0].detail


# --- Problem collection ---------------------------------------------------------------


@dataclass
class _Problem:
    locator: str
    field: str | None
    detail: str
    value: object | None = None

    def issue(self) -> ManifestIssue:
        return ManifestIssue(self.locator, self.field, self.detail, ())

    def message(self) -> str:
        if self.detail in ("csv_syntax", "json_syntax") and self.value is not None:
            return f"{self.locator}: {self.value}"
        return format_issue(self.issue(), self.value)


class _Fail(Exception):
    """Internal: one cell's problem, ``detail`` per MS9."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass
class _Collector:
    name: str
    whole: list[_Problem] = field(default_factory=list)
    rows: list[ManifestRow] = field(default_factory=list)
    row_problems: list[_Problem] = field(default_factory=list)
    warnings: list[_Problem] = field(default_factory=list)
    defaults: Mapping[str, ManifestCell] | None = None

    def raise_whole(self, problems: Sequence[_Problem]) -> None:
        shown = list(problems[:MAX_ISSUES])
        issues = [p.issue() for p in shown]
        messages = [p.message() for p in shown]
        if len(problems) > MAX_ISSUES:
            issues.append(ManifestIssue(self.name, str(len(problems) - MAX_ISSUES), MORE_ISSUES, ()))
            messages.append(f"{self.name}: {len(problems) - MAX_ISSUES} more issues not shown")
        raise ManifestError(issues, messages)


def _raise_one(name: str, locator: str, field_: str | None, detail: str,
               value: object | None = None) -> NoReturn:
    _Collector(name).raise_whole([_Problem(locator, field_, detail, value)])
    raise AssertionError("unreachable")  # pragma: no cover


# --- Selectors (MS2, MS13) -------------------------------------------------------------


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def comparison_key(lexical: str) -> str:
    """``NFC(casefold(NFC(key)))`` (MS13)."""
    return _nfc(_nfc(lexical).casefold())


def selector_problem(selector: str) -> bool:
    """True when ``selector`` is invalid as a ``file`` value (MS2). ``*`` is handled by callers."""
    if not selector.strip() or "\x00" in selector:
        return True
    if selector.startswith(("/", "~")) or selector.endswith("/"):
        return True
    normal = posixpath.normpath(selector)
    return normal == "." or all(part == ".." for part in normal.split("/"))


def _make_row(selector: str, cells: Mapping[str, ManifestCell], locator: str) -> ManifestRow:
    lexical = PurePosixPath(posixpath.normpath(selector))
    return ManifestRow(selector, lexical, comparison_key(str(lexical)), cells, locator)


# --- Cell values (MS1, MS4) -----------------------------------------------------------

_INT_CSV = re.compile(r"^[0-9]+(\.0+)?$")
_DATE_SHEET = (
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{1,2}-[A-Za-z]{3,9}-\d{2,4}$"),
    re.compile(r"^[A-Za-z]{3,9}-\d{2,4}$"),
    re.compile(r"^\d{1,2} [A-Za-z]{3,9} \d{2,4}$"),
    re.compile(r"^\d{5}$"),
)
_NUMBER_SHEET = (
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{1,2}-[A-Za-z]{3,9}$"),
    re.compile(r"^[A-Za-z]{3,9}-\d{1,2}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{1,2} [A-Za-z]{3,9}( \d{2,4})?$"),
)
_BOOL_CSV = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


def _csv_int(text: str) -> int:
    if not _INT_CSV.match(text):
        raise _Fail("number_invalid")
    number = int(text.split(".", 1)[0])
    if number < 1:
        raise _Fail("number_invalid")
    return number


def _text_cell(raw: str, field_: str) -> str:
    try:
        return model.canonical_text(raw, field_)
    except model.InvalidValue:
        raise _Fail("text_invalid") from None


def _date_value(raw: str, *, sheet_hint: bool) -> str:
    try:
        return model.canonical_date(raw)
    except model.InvalidValue:
        if sheet_hint and any(p.match(raw.strip()) for p in _DATE_SHEET):
            raise _Fail("date_spreadsheet") from None
        raise _Fail("date_invalid") from None


def _artwork_cell(raw: str, folder: Path) -> Path:
    text = _nfc(raw.strip())
    if not text or len(text) > MAX_ARTWORK_PATH or "\x00" in text:
        raise _Fail("artwork_path_invalid")
    path = Path(text)
    return path if path.is_absolute() else folder / path


def _csv_cells(values: Mapping[str, str], folder: Path) -> dict[str, ManifestCell]:
    """Convert one CSV record (canonical columns by field name, extras by name)."""
    cells: dict[str, ManifestCell] = {}
    problems: list[tuple[str, str, str]] = []

    def put(key: str, cell: ManifestCell) -> None:
        cells[key] = cell

    for key, raw in values.items():
        text = raw.strip()
        if not text:
            continue
        try:
            if text.casefold() == CLEAR_TEXT:
                put(key, ManifestCell("clear", None))
                continue
            if key in model.TEXT_FIELDS:
                value = _text_cell(raw, key)
                if value:
                    put(key, ManifestCell("set", value))
            elif key == "date":
                put(key, ManifestCell("set", _date_value(text, sheet_hint=True)))
            elif key in ("track", "disc"):
                if "/" in text or any(p.match(text) for p in _NUMBER_SHEET):
                    _pair_cell(key, text, values, cells)
                else:
                    put(key, ManifestCell("set", _csv_int(text)))
            elif key in ("track_total", "disc_total"):
                put(key, ManifestCell("set", _csv_int(text)))
            elif key == "compilation":
                if text.casefold() not in _BOOL_CSV:
                    raise _Fail("boolean_invalid")
                put(key, ManifestCell("set", _BOOL_CSV[text.casefold()]))
            elif key == "artwork":
                put(key, ManifestCell("set", _artwork_cell(raw, folder)))
            else:  # an extra
                value = _text_cell(raw, model.EXTRA)
                if value:
                    put(key, ManifestCell("set", value))
        except _Fail as exc:
            problems.append((key, exc.detail, raw))
    for number, total in (("track", "track_total"), ("disc", "disc_total")):
        low, high = cells.get(number), cells.get(total)
        if (low and high and low.operation == high.operation == "set"
                and type(low.value) is int and type(high.value) is int and low.value > high.value
                and not any(p[0] == number for p in problems)):
            problems.append((number, "pair_inconsistent", f"{low.value}/{high.value}"))
    if problems:
        raise _RowFailure(problems)
    return cells


def _pair_cell(key: str, text: str, values: Mapping[str, str], cells: dict[str, ManifestCell]) -> None:
    """A compound ``n/N`` cell sets the number and its total (MS4)."""
    if any(p.match(text) for p in _NUMBER_SHEET):
        raise _Fail("number_spreadsheet")
    parts = text.split("/")
    if len(parts) != 2:
        raise _Fail("number_invalid")
    number, total = _csv_int(parts[0].strip()), _csv_int(parts[1].strip())
    if number > total:
        raise _Fail("pair_inconsistent")
    total_key = key + "_total"
    other = values.get(total_key, "").strip()
    if other and (other.casefold() == CLEAR_TEXT or _csv_int(other) != total):
        raise _Fail("pair_inconsistent")
    cells[key] = ManifestCell("set", number)
    cells[total_key] = ManifestCell("set", total)


class _RowFailure(Exception):
    def __init__(self, problems: list[tuple[str, str, str]]) -> None:
        super().__init__("row failure")
        self.problems = problems


# --- CSV (MS6, MS12) -------------------------------------------------------------------


def _lex_csv(text: str, delimiter: str, *, header_only: bool = False) -> list[tuple[int, list[str]]]:
    """MS12 state machine. Returns ``(start line, cells)`` per record; raises ``_Fail('csv_syntax')``
    carrying the line in ``.line``."""
    records: list[tuple[int, list[str]]] = []
    cells: list[str] = []
    buf: list[str] = []
    line = 1
    start = 1
    state = "start"
    quote_line = 1
    i, n = 0, len(text)

    def syntax(message: str, at: int) -> _CsvSyntax:
        return _CsvSyntax(at, message)

    def end_record() -> None:
        nonlocal cells, buf, start
        cells.append("".join(buf))
        records.append((start, cells))
        cells, buf = [], []

    while i < n:
        ch = text[i]
        newline = ch in "\r\n"
        step = 2 if ch == "\r" and text[i + 1:i + 2] == "\n" else 1
        if state == "start":
            if ch == '"':
                state, quote_line = "quoted", line
            elif ch == delimiter:
                cells.append("")
            elif newline:
                end_record()
                line += 1
                start = line
                if header_only:
                    return records
                i += step
                continue
            else:
                state = "unquoted"
                buf.append(ch)
        elif state == "unquoted":
            if ch == '"':
                raise syntax("quote inside an unquoted cell — quote the whole cell (\"ab\"\"cd\") or remove the quote"
                             " (a space before an opening quote counts as cell text)", line)
            if ch == delimiter:
                cells.append("".join(buf))
                buf = []
                state = "start"
            elif newline:
                end_record()
                line += 1
                start = line
                state = "start"
                if header_only:
                    return records
                i += step
                continue
            else:
                buf.append(ch)
        elif state == "quoted":
            if ch == '"':
                state = "quote_seen"
            else:
                buf.append(text[i:i + step])
                if newline:
                    line += 1
        elif state == "quote_seen":
            if ch == '"':
                buf.append('"')
                state = "quoted"
            elif ch == delimiter:
                cells.append("".join(buf))
                buf = []
                state = "start"
            elif newline:
                end_record()
                line += 1
                start = line
                state = "start"
                if header_only:
                    return records
                i += step
                continue
            else:
                raise syntax("text after a closing quote", line)
        i += step
    if state == "quoted":
        raise syntax("unterminated quoted field", quote_line)
    if state != "start" or cells or buf:
        end_record()
    return records


class _CsvSyntax(Exception):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(message)
        self.line = line
        self.message = message


def _decode(data: bytes, name: str) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        _raise_one(name, name, None, "encoding", "UTF-16 byte-order mark")
    offset = 0
    if data.startswith(b"\xef\xbb\xbf"):
        data, offset = data[3:], 3
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _raise_one(name, name, None, "encoding", f"byte {exc.start + offset}")


_DISTANCE_ALIASES = {
    "year": "date", "albumartist": "album_artist", "tracknumber": "track", "track_no": "track",
    "trackno": "track", "totaltracks": "track_total", "tracktotal": "track_total", "discnumber": "disc",
    "disk": "disc", "totaldiscs": "disc_total", "disctotal": "disc_total", "cover": "artwork",
    "image": "artwork", "comments": "comment", "description": "comment", "filename": "file",
    "path": "file",
}


def _osa_distance(left: str, right: str) -> int:
    rows = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        rows[i][0] = i
    for j in range(len(right) + 1):
        rows[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[len(left)][len(right)]


def likely_typo(name: str) -> str | None:
    """The canonical field (or ``file``) an extra name probably meant (MS3), else ``None``."""
    folded = name.casefold()
    if folded in _DISTANCE_ALIASES:
        return _DISTANCE_ALIASES[folded]
    limit = 2 if len(folded) >= 6 else 1
    for target in (*model.CANONICAL_FIELDS, "file"):
        if folded != target and _osa_distance(folded, target) <= limit:
            return target
    return None


def _parse_csv(text: str, path: Path, size: int) -> ManifestData:
    name = path.name
    headers: dict[str, list[str]] = {}
    syntax_errors: dict[str, _CsvSyntax] = {}
    for candidate in (",", ";"):
        try:
            first = _lex_csv(text, candidate, header_only=True)
        except _CsvSyntax as exc:
            syntax_errors[candidate] = exc
            continue
        if first and any(cell.strip().casefold() == "file" for cell in first[0][1]):
            headers[candidate] = first[0][1]
    if len(headers) == 2 and headers[","] == headers[";"]:
        headers.pop(";")  # a single-column header names the same file column under both
    if len(headers) != 1:
        if headers:
            _raise_one(name, f"{name}:1", None, "delimiter")
        for candidate, exc in syntax_errors.items():
            if any(c.strip().casefold() == "file" for c in first_split(text, candidate)):
                _raise_one(name, f"{name}:{exc.line}", None, "csv_syntax", exc.message)
        head = re.split(r"\r\n|\r|\n", text, maxsplit=1)[0]
        _raise_one(name, f"{name}:1", None, "delimiter" if "\t" in head or not head.strip()
                   else "missing_file_column")
    delimiters = list(headers)
    delimiter = delimiters[0]
    try:
        records = _lex_csv(text, delimiter)
    except _CsvSyntax as exc:
        _raise_one(name, f"{name}:{exc.line}", None, "csv_syntax", exc.message)
        raise  # pragma: no cover
    header_line, header = records[0]
    collector = _Collector(name)
    columns: list[str | None] = []
    seen: dict[str, str] = {}
    body = [(line, cells) for line, cells in records[1:] if any(cell.strip() for cell in cells)]
    if len(body) > MAX_ROWS or len(header) > MAX_COLUMNS or size > MAX_BYTES:
        _raise_one(name, name, None, "too_large")
    file_column = -1
    for index, raw in enumerate(header):
        column = _nfc(raw.strip())
        locator = f"{name}:{header_line}"
        if not column:
            if any(index < len(cells) and cells[index].strip() for _, cells in body):
                collector.whole.append(_Problem(locator, f"column {index + 1}", "header_empty_name"))
            columns.append(None)
            continue
        folded = column.casefold()
        if folded in seen:
            collector.whole.append(_Problem(locator, column, "header_duplicate", column))
            columns.append(None)
            continue
        seen[folded] = column
        if folded == "file":
            file_column = index
            columns.append("file")
        elif folded in model.CANONICAL_FIELDS:
            columns.append(folded)
        else:
            problem = model.extra_name_problem(column)
            if problem is not None:
                collector.whole.append(_Problem(locator, column, problem, column))
                columns.append(None)
                continue
            columns.append(column)
            suggestion = likely_typo(column)
            if suggestion is not None:
                collector.warnings.append(_Problem(locator, column, "likely_typo", suggestion))
    if file_column < 0:
        collector.whole.append(_Problem(f"{name}:{header_line}", None, "missing_file_column"))
    defaults_seen = False
    for line, cells in body:
        locator = f"{name}:{line}"
        values: dict[str, str] = {}
        extra_cells = [c for c in cells[len(columns):] if c.strip()]
        for index, column in enumerate(columns):
            if column is not None and column != "file" and index < len(cells):
                values[column] = cells[index]
        selector = cells[file_column].strip() if 0 <= file_column < len(cells) else ""
        if file_column >= 0 and selector == DEFAULTS_ROW:
            if defaults_seen:
                collector.whole.append(_Problem(locator, None, "defaults_duplicate"))
                continue
            defaults_seen = True
            try:
                collector.defaults = _csv_cells(values, path.parent)
            except _RowFailure as failure:
                for key, _detail, raw in failure.problems:
                    collector.whole.append(_Problem(locator, key, "defaults_invalid", raw))
            if extra_cells:
                collector.whole.append(_Problem(locator, None, "defaults_invalid", "extra values"))
            continue
        if selector_problem(_nfc(selector)):
            collector.whole.append(_Problem(locator, "file", "file_value_invalid", selector))
            continue
        selector = _nfc(selector)
        try:
            row_cells = _csv_cells(values, path.parent)
            failures: list[tuple[str, str, str]] = []
        except _RowFailure as failure:
            row_cells, failures = {}, failure.problems
        if extra_cells:
            failures.append(("", "cell_count", f"{len(cells)} values but {len(columns)} columns"))
        for key, detail, raw in failures:
            collector.row_problems.append(_Problem(locator, key or None, detail, raw))
        collector.rows.append(_make_row(selector, {} if failures else row_cells, locator))
    return _finish(collector, path)


def first_split(text: str, delimiter: str) -> list[str]:
    return re.split(r"\r\n|\r|\n", text, maxsplit=1)[0].split(delimiter)


# --- JSON (MS7, MS11) ---------------------------------------------------------------------


class _NonInteger:
    """A JSON number token with a fraction or exponent (MS11)."""

    def __init__(self, token: str) -> None:
        self.token = token

    def __repr__(self) -> str:
        return self.token


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _JsonDuplicate(key)
        result[key] = value
    return result


class _JsonDuplicate(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _json_constant(token: str) -> Any:
    raise ValueError(f"{token} is not allowed")


def _pointer(*parts: str) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def _json_int(value: object) -> int:
    if type(value) is not int or value < 1:
        raise _Fail("number_invalid")
    return value


def _json_cell(key: str, value: object, folder: Path) -> ManifestCell:
    if value is None:
        return ManifestCell("clear", None)
    if key in model.TEXT_FIELDS:
        if not isinstance(value, str) or not value.strip():
            raise _Fail("text_invalid")
        text = _text_cell(value, key)
        if not text:
            raise _Fail("text_invalid")
        return ManifestCell("set", text)
    if key == "date":
        if not isinstance(value, str):
            raise _Fail("date_invalid")
        return ManifestCell("set", _date_value(value, sheet_hint=False))
    if key in model.INTEGER_FIELDS:
        return ManifestCell("set", _json_int(value))
    if key == "compilation":
        if type(value) is not bool:
            raise _Fail("boolean_invalid")
        return ManifestCell("set", value)
    if key == "artwork":
        if not isinstance(value, str):
            raise _Fail("artwork_path_invalid")
        return ManifestCell("set", _artwork_cell(value, folder))
    raise _Fail("unknown_field")  # pragma: no cover


def _json_fields(obj: object, folder: Path, *, locator: str,
                 warnings: list[_Problem]) -> dict[str, ManifestCell]:
    """Convert one fields object; raises :class:`_RowFailure` with every problem."""
    if not isinstance(obj, dict):
        raise _RowFailure([("", "json_structure", "not an object")])
    cells: dict[str, ManifestCell] = {}
    problems: list[tuple[str, str, str]] = []
    for key, value in obj.items():
        if key == "extras":
            if not isinstance(value, dict):
                problems.append(("extras", "text_invalid", repr(value)))
                continue
            folded: dict[str, str] = {}
            for extra, raw in value.items():
                issue = model.extra_name_problem(extra)
                if issue is None and extra.casefold() in folded:
                    issue = "illegal_extra_name"
                if issue is not None:
                    problems.append((extra, issue, extra))
                    continue
                folded[extra.casefold()] = extra
                suggestion = likely_typo(extra)
                if suggestion is not None:
                    warnings.append(_Problem(locator + _pointer("extras", extra), extra, "likely_typo", suggestion))
                try:
                    if raw is None:
                        cells[extra] = ManifestCell("clear", None)
                    elif not isinstance(raw, str) or not raw.strip():
                        raise _Fail("text_invalid")
                    else:
                        text = _text_cell(raw, model.EXTRA)
                        if not text:
                            raise _Fail("text_invalid")
                        cells[extra] = ManifestCell("set", text)
                except _Fail as exc:
                    problems.append((extra, exc.detail, str(raw)))
            continue
        if key not in model.CANONICAL_FIELDS:
            problems.append((key, "unknown_field", key))
            continue
        try:
            cells[key] = _json_cell(key, value, folder)
        except _Fail as exc:
            problems.append((key, exc.detail, repr(value) if not isinstance(value, str) else value))
    for number, total in (("track", "track_total"), ("disc", "disc_total")):
        low, high = cells.get(number), cells.get(total)
        if (low and high and low.operation == high.operation == "set" and low.value > high.value  # type: ignore[operator]
                and not any(p[0] == number for p in problems)):
            problems.append((number, "pair_inconsistent", f"{low.value}/{high.value}"))
    if problems:
        raise _RowFailure(problems)
    return cells


def _contains_non_integer(value: object) -> bool:
    if isinstance(value, _NonInteger):
        return True
    if isinstance(value, dict):
        return any(_contains_non_integer(v) for v in value.values())
    return False


def _parse_json(text: str, path: Path, size: int) -> ManifestData:
    name = path.name
    if size > MAX_BYTES:
        _raise_one(name, name, None, "too_large")
    try:
        document = json.loads(text, parse_float=_NonInteger, parse_constant=_json_constant,
                              object_pairs_hook=_json_pairs)
    except _JsonDuplicate as exc:
        _raise_one(name, name, None, "json_syntax", f"duplicate key {exc.key!r}")
    except json.JSONDecodeError as exc:
        _raise_one(name, f"{name}:{exc.lineno}:{exc.colno}", None, "json_syntax", exc.msg)
    except ValueError as exc:
        _raise_one(name, name, None, "json_syntax", str(exc))
    collector = _Collector(name)
    if not isinstance(document, dict):
        _raise_one(name, f"{name}#", None, "json_structure", "top level is not an object")
    for key in document:
        if key not in ("version", "defaults", "files"):
            collector.whole.append(_Problem(f"{name}#" + _pointer(key), None, "json_structure", key))
    version = document.get("version")
    if "version" not in document or type(version) is not int or version != 1:
        collector.whole.append(_Problem(f"{name}#/version", None, "version", repr(version) if "version" in document else None))
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        collector.whole.append(_Problem(f"{name}#/files", None, "json_structure", "must contain at least one file"))
        files = {}
    defaults = document.get("defaults", {})
    if "defaults" in document:
        try:
            if _contains_non_integer(defaults) and isinstance(defaults, dict):
                raise _RowFailure([(k, "number_invalid", repr(v)) for k, v in defaults.items()
                                   if isinstance(v, _NonInteger)])
            collector.defaults = _json_fields(defaults, path.parent, locator=f"{name}#/defaults",
                                               warnings=collector.warnings)
        except _RowFailure as failure:
            for key, _detail, raw in failure.problems:
                collector.whole.append(_Problem(f"{name}#/defaults" + (_pointer(key) if key else ""),
                                                key or None, "defaults_invalid", raw))
    for selector, entry in files.items():
        locator = f"{name}#" + _pointer("files", selector)
        if selector == DEFAULTS_ROW or selector_problem(_nfc(selector)):
            collector.whole.append(_Problem(locator, "file", "file_value_invalid", selector))
            continue
        try:
            row_cells = _json_fields(entry, path.parent, locator=locator, warnings=collector.warnings)
            failures: list[tuple[str, str, str]] = []
        except _RowFailure as failure:
            row_cells, failures = {}, failure.problems
        for key, detail, raw in failures:
            collector.row_problems.append(_Problem(locator, key or None, detail, raw))
        collector.rows.append(_make_row(_nfc(selector), {} if failures else row_cells, locator))
    return _finish(collector, path)


# --- Entry points -----------------------------------------------------------------------


def _finish(collector: _Collector, path: Path) -> ManifestData:
    if collector.whole:
        collector.raise_whole(collector.whole)
    issues = [p.issue() for p in collector.row_problems[:MAX_ISSUES]]
    if len(collector.row_problems) > MAX_ISSUES:
        issues.append(ManifestIssue(path.name, str(len(collector.row_problems) - MAX_ISSUES), MORE_ISSUES, ()))
    issues.extend(p.issue() for p in collector.warnings)
    return ManifestData(path, 1, collector.defaults or {}, tuple(collector.rows), tuple(issues))


def parse_manifest(data: bytes, path: Path) -> ManifestData:
    """Parse manifest ``data`` as the form named by ``path``'s suffix (``.csv`` or ``.json``).

    Pure: nothing is read from disk. ``path`` is absolute and only supplies the
    file name for locators, the folder for artwork paths and the suffix.
    """
    suffix = path.suffix.casefold()
    if suffix not in (".csv", ".json"):
        _raise_one(path.name, path.name, None, "unreadable", "not a .csv or .json file")
    if len(data) > MAX_BYTES:
        _raise_one(path.name, path.name, None, "too_large", f"{len(data)} bytes")
    text = _decode(data, path.name)
    return _parse_csv(text, path, len(data)) if suffix == ".csv" else _parse_json(text, path, len(data))


def load_manifest(path: Path | str) -> ManifestData:
    """Read and parse one local manifest file; a read failure is ``unreadable`` (exit 2)."""
    given = Path(path)
    try:
        resolved = given.resolve(strict=True)
        data = resolved.read_bytes()
    except OSError as exc:
        _raise_one(given.name or str(given), given.name or str(given), None, "unreadable", exc.strerror or str(exc))
        raise  # pragma: no cover
    return parse_manifest(data, resolved)


def row_issues(data: ManifestData) -> Mapping[str, tuple[ManifestIssue, ...]]:
    """Row-level problems of ``data`` by row locator."""
    grouped: dict[str, list[ManifestIssue]] = {}
    for issue in data.issues:
        if issue.detail in ROW_DETAILS:
            grouped.setdefault(issue.locator, []).append(issue)
    return {locator: tuple(items) for locator, items in grouped.items()}


# --- Matching (MS5, MS13) ----------------------------------------------------------------


@dataclass(frozen=True)
class InputMatch:
    """What the manifest says about one input."""

    input_index: int
    row: ManifestRow | None
    """The single applied row, if any (defaults are separate: they apply to every input)."""
    failures: tuple[ManifestIssue, ...] = ()
    """``manifest_row_invalid`` causes: the matched row's own issues or ``duplicate_match``."""
    warnings: tuple[ManifestIssue, ...] = ()
    """``ambiguous_match`` issues, one per ambiguous row that touches this input."""


@dataclass(frozen=True)
class ManifestMatching:
    inputs: Mapping[int, InputMatch]
    unmatched: tuple[ManifestIssue, ...]
    """``unmatched_row`` run-level warnings, for rows that matched no input or are invalid and not applied."""


def input_key(source_path: Path, folder: Path) -> str:
    """The lexical comparison key of an input relative to the manifest folder (MS13)."""
    return comparison_key(posixpath.relpath(str(source_path), str(folder)))


def manifest_relative(source_path: Path, folder: Path) -> str:
    """The manifest-relative lexical path used for ``file`` values (NFC, spelling kept)."""
    return _nfc(posixpath.relpath(str(source_path), str(folder)))


def match_manifest(data: ManifestData, inputs: Sequence[InputItem]) -> ManifestMatching:
    """Match rows to inputs lexically; never opens a selector or resolves a link."""
    folder = data.path.parent
    keys = {item.input_index: input_key(item.source_path, folder) for item in inputs}
    invalid = row_issues(data)
    candidates: dict[int, list[ManifestRow]] = {item.input_index: [] for item in inputs}
    ambiguous: dict[int, list[ManifestIssue]] = {item.input_index: [] for item in inputs}
    unmatched: list[ManifestIssue] = []
    for row in data.rows:
        if "/" in row.match_key:
            hits = [i for i, key in keys.items() if key == row.comparison_key]
        else:
            hits = [i for i, key in keys.items() if key == row.comparison_key]
            if not hits:
                hits = [i for i, key in keys.items() if key.rsplit("/", 1)[-1] == row.comparison_key]
        if len(hits) == 1:
            candidates[hits[0]].append(row)
        elif len(hits) > 1:
            issue = ManifestIssue(row.locator, "file", "ambiguous_match", tuple(sorted(hits)))
            for index in hits:
                ambiguous[index].append(issue)
        else:
            unmatched.append(ManifestIssue(row.locator, "file", "unmatched_row", ()))
    result: dict[int, InputMatch] = {}
    for item in inputs:
        index = item.input_index
        rows = candidates[index]
        failures: list[ManifestIssue] = []
        applied: ManifestRow | None = None
        bad = [issue for row in rows for issue in invalid.get(row.locator, ())]
        if bad:
            failures.extend(bad)
        if len(rows) > 1:
            failures.append(ManifestIssue(", ".join(row.locator for row in rows), "file", "duplicate_match",
                                          (index,)))
        if len(rows) == 1 and not failures:
            applied = rows[0]
        result[index] = InputMatch(index, applied, tuple(failures), tuple(ambiguous[index]))
    return ManifestMatching(result, tuple(unmatched))


# --- Template content (MD28) ---------------------------------------------------------------

TEMPLATE_FIELDS: tuple[str, ...] = (
    "title", "artist", "album_artist", "album", "date", "track", "track_total", "disc", "disc_total", "genre",
    "composer", "comment", "compilation", "artwork",
)
"""MD28 column order after ``file`` (``compilation`` before ``artwork``)."""
TEMPLATE_COLUMNS: tuple[str, ...] = ("file", *TEMPLATE_FIELDS)
TEMPLATE_DEFAULT_FIELDS: tuple[str, ...] = (
    "artist", "album_artist", "album", "genre", "composer", "track_total", "disc", "disc_total", "compilation",
)


@dataclass(frozen=True)
class TemplateEntry:
    """One input's canonical projection (after reading and optional inference)."""

    source_path: Path
    fields: Mapping[str, MetadataValue]
    """Canonical fields; ``artwork`` is ignored because embedded pictures have no path."""
    extras: Mapping[str, str] = field(default_factory=dict)


def _cell_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _usable(key: str, value: object) -> bool:
    if key == "artwork" or value is None:
        return False
    if isinstance(value, str) and (not value.strip() or value.strip().casefold() == CLEAR_TEXT):
        return False
    return True


def build_template(entries: Sequence[TemplateEntry], folder: Path, form: str = "csv") -> str:
    """Template text for ``entries``, ``file`` values relative to ``folder`` (the template's folder).

    ``form`` is ``csv`` or ``json``. Defaults hold values identical across every entry.
    """
    if form not in ("csv", "json"):
        raise ValueError(f"unknown template form: {form!r}")
    if not entries:
        raise ValueError("a template needs at least one input")
    projections = [{k: v for k, v in entry.fields.items() if _usable(k, v)} for entry in entries]
    defaults: dict[str, MetadataValue] = {}
    for key in TEMPLATE_DEFAULT_FIELDS:
        first = projections[0].get(key)
        if first is not None and all(key in p and model.values_equal(p[key], first) for p in projections):
            defaults[key] = first
    extras: dict[str, str] = {}
    for entry in entries:
        for name, value in entry.extras.items():
            if model.extra_name_problem(name) is None and _usable(name, value):
                extras.setdefault(name.casefold(), name)
    extra_names = sorted(extras.values(), key=lambda n: (n.casefold(), n))
    rows: list[tuple[str, dict[str, MetadataValue], dict[str, str]]] = []
    for entry, projection in zip(entries, projections, strict=True):
        own = {k: v for k, v in projection.items() if k not in defaults}
        own_extras = {n: v for n, v in entry.extras.items()
                      if model.extra_name_problem(n) is None and _usable(n, v)}
        rows.append((manifest_relative(entry.source_path, folder), own, own_extras))
    if form == "json":
        def fields_of(values: Mapping[str, MetadataValue], ex: Mapping[str, str]) -> dict[str, Any]:
            out: dict[str, Any] = {k: values[k] for k in TEMPLATE_FIELDS if k in values}
            if ex:
                out["extras"] = {n: ex[n] for n in extra_names if n in ex}
            return out
        document: dict[str, Any] = {"version": 1}
        if defaults:
            document["defaults"] = fields_of(defaults, {})
        document["files"] = {file: fields_of(own, ex) for file, own, ex in rows}
        return json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow([*TEMPLATE_COLUMNS, *extra_names])

    def line(selector: str, values: Mapping[str, MetadataValue], ex: Mapping[str, str]) -> list[str]:
        return [selector,
                *(_cell_text(values[k]) if k in values else "" for k in TEMPLATE_FIELDS),
                *(ex.get(n, "") for n in extra_names)]

    if defaults:
        writer.writerow(line(DEFAULTS_ROW, defaults, {}))
    for file, own, ex in rows:
        writer.writerow(line(file, own, ex))
    return stream.getvalue()
