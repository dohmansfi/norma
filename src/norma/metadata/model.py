"""Canonical metadata fields: validation and tag-system mappings (WS-06, pure).

Field rules follow the metadata schema (MS1, MS3, MS4) and the
source-reading rules (MD5); native
store priority and key mappings follow MD3 and the format tables; the
processing-mark grammar and validity classes follow MD26 and MD37.

Pure domain module: standard library and ``norma.models`` only. It defines no
competing boundary record; C-06 records live in ``norma.models`` (CONTRACTS
§ Change control).
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import replace as _evolve
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

from norma.models import (
    Diagnostic,
    FieldStatus,
    GainPlan,
    LoudnessMeasurement,
    MetadataPlan,
    NativeItemStatus,
    ProcessingMark,
    RawNativeItem,
)

# --- Canonical fields (MS1) ------------------------------------------------------

CANONICAL_FIELDS: tuple[str, ...] = (
    "title", "artist", "album_artist", "album", "date", "track", "track_total",
    "disc", "disc_total", "genre", "composer", "comment", "artwork", "compilation",
)
TEXT_FIELDS: frozenset[str] = frozenset(
    {"title", "artist", "album_artist", "album", "genre", "composer", "comment"})
INTEGER_FIELDS: frozenset[str] = frozenset({"track", "track_total", "disc", "disc_total"})
EXTRA = "extra"
"""Pseudo-field name selecting the extras' text rules in :func:`canonical_text`."""

TEXT_LIMITS = {"comment": 4000, EXTRA: 4000}
DEFAULT_TEXT_LIMIT = 1000
_MULTILINE = frozenset({"comment", EXTRA})

MARK_NAME = "NORMA"
"""The reserved processing-mark extra name (MD26; D-041)."""

_RESERVED_NAMES = frozenset({MARK_NAME.casefold(), "itunnorm", "itunsmpb", "encoder"})
_RESERVED_PREFIXES = ("replaygain_", "r128_")
_EXTRA_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_CANONICAL_OR_FILE = frozenset(CANONICAL_FIELDS) | {"file"}

MULTI_VALUE_SEPARATOR = "; "
"""Join for several source values of one text field (MD5)."""


class InvalidValue(ValueError):
    """A source or input value that fails the canonical rules; ``detail`` names the rule."""

    def __init__(self, detail: str, message: str | None = None) -> None:
        super().__init__(message or detail)
        self.detail = detail


# --- Text --------------------------------------------------------------------------


def normalise_text(raw: str) -> str:
    """CR LF and lone CR become LF; trimmed; NFC (MS4). Never validates."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text.strip())


def canonical_text(raw: str, field: str) -> str:
    """The valid canonical text for ``field`` (a text field or :data:`EXTRA`).

    Returns ``""`` when nothing remains after trimming: the caller treats that
    as no value. Raises :class:`InvalidValue` for a control character, a line
    break outside ``comment``/extras, or a length over the limit (MS1, MS4).
    """
    if field not in TEXT_FIELDS and field != EXTRA:
        raise ValueError(f"{field!r} is not a text field")
    text = normalise_text(raw)
    if not text:
        return ""
    for char in text:
        if char == "\n":
            if field not in _MULTILINE:
                raise InvalidValue("line_break_not_allowed", f"{field} cannot contain a line break")
        elif char != "\t" and unicodedata.category(char) == "Cc":
            raise InvalidValue("text_invalid", f"{field} contains control character U+{ord(char):04X}")
    limit = TEXT_LIMITS.get(field, DEFAULT_TEXT_LIMIT)
    if len(text) > limit:
        raise InvalidValue("too_long", f"{field} has {len(text)} characters, at most {limit} allowed")
    return text


def join_text(values: Sequence[str]) -> str:
    """Join several already normalised values with ``; `` (MD5), skipping empty ones."""
    return MULTI_VALUE_SEPARATOR.join(value for value in values if value)


# --- Dates -------------------------------------------------------------------------

_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2})(?::(\d{2})(?::(\d{2})(?:\.\d+)?)?)?"
    r"(Z|[+-](\d{2})(?::?(\d{2}))?)?$"
)


def canonical_date(text: str) -> str:
    """A C-11 date: ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``, a real calendar date."""
    value = text.strip()
    match = _DATE_RE.match(value)
    if match is None:
        raise InvalidValue("date_invalid", f"{value!r} is not YYYY, YYYY-MM or YYYY-MM-DD")
    year, month, day = (int(part) if part else None for part in match.groups())
    assert year is not None
    try:
        _dt.date(year, 1 if month is None else month, 1 if day is None else day)
    except ValueError:
        raise InvalidValue("date_invalid", f"{value!r} is not a real calendar date") from None
    return value


def is_source_timestamp(value: str) -> bool:
    """True for MD5's source timestamp form (ISO 8601 date, ``T``, time, optional offset)."""
    match = _TIMESTAMP_RE.match(value)
    if match is None:
        return False
    date, hour, minute, second, _, off_hour, off_minute = match.groups()
    try:
        canonical_date(date)
    except InvalidValue:
        return False
    return (int(hour) < 24 and int(minute or 0) < 60 and int(second or 0) < 60
            and int(off_hour or 0) < 24 and int(off_minute or 0) < 60)


def source_date(text: str) -> str:
    """A valid source date value, kept as stored (trimmed): a C-11 date or a timestamp.

    MD5: a timestamp is valid only from a source store, and the stored text is
    what a roll-up's ``requested`` keeps; :func:`project_date` gives the
    canonical date part.
    """
    value = text.strip()
    if is_source_timestamp(value):
        return value
    return canonical_date(value)


def project_date(value: str) -> str:
    """The canonical C-11 date of a valid source date (a timestamp's date part)."""
    if is_source_timestamp(value):
        return value.split("T", 1)[0]
    return canonical_date(value)


def time_of_day_dropped(value: str) -> bool:
    """True when projecting ``value`` drops a time of day (MD5 detail ``time_of_day_dropped``)."""
    return is_source_timestamp(value)


def id3v23_date(year: str, ddmm: str | None) -> str:
    """ID3v2.3 ``TYER`` (+ ``TDAT`` as ``DDMM``) as one date value (MD3, MD5)."""
    base = canonical_date(year)
    if len(base) != 4:
        raise InvalidValue("date_invalid", f"TYER {year!r} is not a four-digit year")
    if ddmm is None:
        return base
    text = ddmm.strip()
    if not re.fullmatch(r"\d{4}", text):
        raise InvalidValue("date_invalid", f"TDAT {ddmm!r} is not DDMM")
    return canonical_date(f"{base}-{text[2:]}-{text[:2]}")


# --- Numbers and booleans ----------------------------------------------------------

_PAIR_RE = re.compile(r"^\s*(\d*)\s*(?:/\s*(\d*)\s*)?$")


def parse_integer(text: str) -> int | None:
    """A source number or total: digits, zero padding allowed; 0 or empty is absent (MD5)."""
    value = text.strip()
    if not re.fullmatch(r"\d*", value):
        raise InvalidValue("number_invalid", f"{text!r} is not a number")
    number = int(value) if value else 0
    return number or None


def parse_number_pair(text: str) -> tuple[int | None, int | None, bool]:
    """``n``, ``n/N`` or ``/N`` as ``(number, total, total_invalid)`` (MD5).

    A component that is 0 or empty is absent. When ``n > N`` the number is
    kept and the total is invalid (``total_invalid`` true, total ``None``).
    """
    match = _PAIR_RE.match(text)
    if match is None:
        raise InvalidValue("number_invalid", f"{text!r} is not n or n/N")
    number = int(match.group(1)) if match.group(1) else None
    total = int(match.group(2)) if match.group(2) else None
    return check_pair(number or None, total or None)


def check_pair(number: int | None, total: int | None) -> tuple[int | None, int | None, bool]:
    """Apply ``number <= total``: an inconsistent total is invalid, the number kept (MD5)."""
    if number is not None and total is not None and number > total:
        return number, None, True
    return number, total, False


_TRUE = frozenset({"1", "true"})
_FALSE = frozenset({"0", "false"})


def parse_compilation(text: str) -> bool:
    """``1``, ``0``, ``true`` or ``false``, case-insensitive (MD5)."""
    value = text.strip().casefold()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise InvalidValue("boolean_invalid", f"{text!r} is not 1, 0, true or false")


def values_equal(left: object, right: object) -> bool:
    """Canonical equality without Python's ``True == 1`` shortcut (C-06 equality)."""
    return type(left) is type(right) and left == right


# --- Extras (MS3) ------------------------------------------------------------------


def is_reserved_name(name: str) -> bool:
    """A reserved name: generated or removed by design (MS3), case-insensitive."""
    folded = name.casefold()
    return folded in _RESERVED_NAMES or folded.startswith(_RESERVED_PREFIXES)


def extra_name_problem(name: str) -> str | None:
    """``None`` for a usable extra name, else ``illegal_extra_name`` or ``reserved_extra_name``."""
    if not _EXTRA_NAME_RE.match(name) or name.casefold() in _CANONICAL_OR_FILE:
        return "illegal_extra_name"
    if is_reserved_name(name):
        return "reserved_extra_name"
    return None


# --- Stores and native keys (MD3, MD4; formats § 8) ---------------------------------

CONTAINER_STORES: dict[str, tuple[str, ...]] = {
    "mp4": ("mp4",),
    "flac": ("vorbis", "id3v2"),
    "mp3": ("id3v2", "apev2", "id3v1"),
    "aiff": ("id3v2", "aiff-text"),
    "wav": ("id3v2", "riff-info"),
}
"""Stores inventoried per source container, in canonical priority order (MD3)."""

_CONTAINER_ALIASES = {"mov": "mp4", "m4a": "mp4", "mpeg": "mp3", "aifc": "aiff", "rf64": "wav", "bw64": "wav"}

STORES: tuple[str, ...] = ("id3v2", "vorbis", "mp4", "riff-info", "aiff-text", "apev2", "id3v1")


def store_family(container: str) -> str:
    """The MD3 container row for a C-02 container name; ``ValueError`` if MD3 has none."""
    name = _CONTAINER_ALIASES.get(container.casefold(), container.casefold())
    if name not in CONTAINER_STORES:
        raise ValueError(f"no MD3 store row for container {container!r}")
    return name


def container_stores(container: str) -> tuple[str, ...]:
    """Stores for ``container`` in canonical priority order (MD3)."""
    return CONTAINER_STORES[store_family(container)]


def native_identity(store: str, native_key: str, occurrence: int | None = None) -> str:
    """``store:<store>:<native key>[#<n>]`` (MD4)."""
    identity = f"store:{store}:{native_key}"
    return identity if occurrence is None else f"{identity}#{occurrence}"


def nontag_identity(kind: str, locator: str | None = None, occurrence: int | None = None) -> str:
    """``item:<kind>[:<locator>][#<n>]`` (MD4)."""
    identity = f"item:{kind}" if locator is None else f"item:{kind}:{locator}"
    return identity if occurrence is None else f"{identity}#{occurrence}"


# Frame IDs per canonical field; ID3v2.2 three-character IDs are read through
# their ID3v2.3/2.4 equivalents for the projection only (MD3 version translation).
_ID3_FIELDS: dict[str, tuple[str, ...]] = {
    "TIT2": ("title",), "TT2": ("title",),
    "TPE1": ("artist",), "TP1": ("artist",),
    "TPE2": ("album_artist",), "TP2": ("album_artist",),
    "TALB": ("album",), "TAL": ("album",),
    "TDRC": ("date",), "TYER": ("date",), "TDAT": ("date",), "TYE": ("date",), "TDA": ("date",),
    "TRCK": ("track", "track_total"), "TRK": ("track", "track_total"),
    "TPOS": ("disc", "disc_total"), "TPA": ("disc", "disc_total"),
    "TCON": ("genre",), "TCO": ("genre",),
    "TCOM": ("composer",), "TCM": ("composer",),
    "TCMP": ("compilation",), "TCP": ("compilation",),
    "APIC": ("artwork",), "PIC": ("artwork",),
}
_VORBIS_FIELDS: dict[str, tuple[str, ...]] = {
    "TITLE": ("title",), "ARTIST": ("artist",), "ALBUMARTIST": ("album_artist",), "ALBUM": ("album",),
    "DATE": ("date",), "TRACKNUMBER": ("track", "track_total"), "TRACKTOTAL": ("track_total",),
    "TOTALTRACKS": ("track_total",), "DISCNUMBER": ("disc", "disc_total"), "DISCTOTAL": ("disc_total",),
    "TOTALDISCS": ("disc_total",), "GENRE": ("genre",), "COMPOSER": ("composer",),
    "COMMENT": ("comment",), "COMPILATION": ("compilation",), "PICTURE": ("artwork",),
}
_MP4_FIELDS: dict[str, tuple[str, ...]] = {
    "©nam": ("title",), "©ART": ("artist",), "aART": ("album_artist",), "©alb": ("album",),
    "©day": ("date",), "trkn": ("track", "track_total"), "disk": ("disc", "disc_total"),
    "©gen": ("genre",), "©wrt": ("composer",), "©cmt": ("comment",), "cpil": ("compilation",),
    "covr": ("artwork",),
}
_RIFF_FIELDS: dict[str, tuple[str, ...]] = {
    "INAM": ("title",), "IART": ("artist",), "IPRD": ("album",), "ICRD": ("date",),
    "IPRT": ("track",), "IGNR": ("genre",), "ICMT": ("comment",),
}
_AIFF_FIELDS: dict[str, tuple[str, ...]] = {"NAME": ("title",), "AUTH": ("artist",), "ANNO": ("comment",)}
_APE_FIELDS: dict[str, tuple[str, ...]] = {
    "TITLE": ("title",), "ARTIST": ("artist",), "ALBUM ARTIST": ("album_artist",),
    "ALBUMARTIST": ("album_artist",), "ALBUM": ("album",), "YEAR": ("date",),
    "TRACK": ("track", "track_total"), "DISC": ("disc", "disc_total"), "GENRE": ("genre",),
    "COMPOSER": ("composer",), "COMMENT": ("comment",), "COMPILATION": ("compilation",),
}
_ID3V1_FIELDS: dict[str, tuple[str, ...]] = {
    "title": ("title",), "artist": ("artist",), "album": ("album",), "year": ("date",),
    "comment": ("comment",), "track": ("track",), "genre": ("genre",),
}
_RIFF_NOT_EXTRA = frozenset({"ISFT"})
"""INFO items on the MD6 list are never carried as extras."""

_VORBIS_CHAPTER_RE = re.compile(r"^CHAPTER\d+(NAME|URL)?$")
_VORBIS_NOT_EXTRA = frozenset({"METADATA_BLOCK_PICTURE", "COVERART", "COVERARTMIME"})
_ITUNES_FREEFORM = "----:com.apple.iTunes:"


def _id3_frame_and_parts(native_key: str) -> tuple[str, list[str]]:
    frame, _, rest = native_key.partition(":")
    return frame, rest.split(":") if rest else []


def native_fields(store: str, native_key: str) -> tuple[str, ...]:
    """Canonical fields an occurrence with this native key feeds (MD3; formats § 8).

    ``native_key`` is the MD4 native key without ``#n``. The mapping is by key
    only; whether the stored value is valid is a reading question (MD5).
    """
    if store == "id3v2":
        frame, parts = _id3_frame_and_parts(native_key)
        if frame in ("COMM", "COM"):
            return ("comment",) if parts and parts[0] == "" else ()
        return _ID3_FIELDS.get(frame, ())
    if store == "vorbis":
        return _VORBIS_FIELDS.get(native_key.upper(), ())
    if store == "mp4":
        return _MP4_FIELDS.get(native_key, ())
    if store == "riff-info":
        return _RIFF_FIELDS.get(native_key, ())
    if store == "aiff-text":
        return _AIFF_FIELDS.get(native_key, ())
    if store == "apev2":
        return _APE_FIELDS.get(native_key.upper(), ())
    if store == "id3v1":
        return _ID3V1_FIELDS.get(native_key, ())
    raise ValueError(f"unknown store {store!r}")


def native_extra(store: str, native_key: str) -> str | None:
    """The extra name a text occurrence with this native key carries, or ``None`` (MD19).

    Only names that pass MS3 (legal, not canonical, not reserved) are extras;
    the processing mark is therefore never an extra.
    """
    if native_fields(store, native_key):
        return None
    name: str | None = None
    if store == "id3v2":
        frame, parts = _id3_frame_and_parts(native_key)
        if frame in ("TXXX", "TXX") and parts:
            name = ":".join(parts)
    elif store == "vorbis":
        upper = native_key.upper()
        if not _VORBIS_CHAPTER_RE.match(upper) and upper not in _VORBIS_NOT_EXTRA:
            name = native_key
    elif store == "mp4":
        if native_key.startswith(_ITUNES_FREEFORM):
            name = native_key[len(_ITUNES_FREEFORM):]
    elif store == "riff-info":
        if native_key not in _RIFF_NOT_EXTRA:
            name = native_key
    elif store == "apev2":
        name = native_key
    if name is None or extra_name_problem(name) is not None:
        return None
    return name


def is_mark_key(store: str, native_key: str) -> bool:
    """True when the occurrence holds the reserved mark name (MD26 detection, any store)."""
    folded = MARK_NAME.casefold()
    if store == "id3v2":
        frame, parts = _id3_frame_and_parts(native_key)
        return frame in ("TXXX", "TXX") and ":".join(parts).casefold() == folded
    if store in ("vorbis", "apev2"):
        return native_key.casefold() == folded
    if store == "mp4":
        if not native_key.startswith("----:"):
            return False
        return native_key.rsplit(":", 1)[-1].casefold() == folded
    return False


def is_chapter_key(store: str, native_key: str) -> bool:
    """Native chapter occurrences that feed ``item:chapters`` (MD7)."""
    if store == "id3v2":
        return _id3_frame_and_parts(native_key)[0] in ("CHAP", "CTOC")
    if store == "vorbis":
        return _VORBIS_CHAPTER_RE.match(native_key.upper()) is not None
    return False


# --- Processing mark (MD26, MD37) ---------------------------------------------------

MARK_STAGE = "tags_read"
_MARK_MAX = 255
_KEY_RE = re.compile(r"^[a-z_]+$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+(\.\d{1,2})?$")
_TWO_PLACES_RE = re.compile(r"^[+-]?\d+\.\d{2}$")
_SIGNED_TWO_RE = re.compile(r"^[+-]\d+\.\d{2}$")
_UNSIGNED_TWO_RE = re.compile(r"^\d+\.\d{2}$")
_MODES = ("copy", "linear", "limit", "reduce", "dynamic")
_REQUIRED = ("v", "tool", "target", "tp", "mode", "gain", "out_i", "out_tp", "origin", "ffmpeg", "date")


def _valid_mark_date(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        canonical_date(value)
    except InvalidValue:
        return False
    return True


_MARK_FORMATS = {
    "tool": _TOKEN_RE.match,
    "target": _DECIMAL_RE.match,
    "tp": _DECIMAL_RE.match,
    "mode": lambda value: value in _MODES,
    "gain": _SIGNED_TWO_RE.match,
    "limited": _UNSIGNED_TWO_RE.match,
    "out_i": _TWO_PLACES_RE.match,
    "out_tp": _TWO_PLACES_RE.match,
    "origin": re.compile(r"^[a-z0-9_]+$").match,
    "ffmpeg": _TOKEN_RE.match,
    "date": _valid_mark_date,
}


def _split_pairs(value: str) -> list[tuple[str, str]] | None:
    pairs = []
    for part in value.split(";"):
        key, sep, item = part.partition("=")
        if not sep:
            return None
        pairs.append((key, item))
    return pairs


def _mark_problem(value: str) -> tuple[str, str | None]:
    """``("valid", None)``, ``("unsupported", <v>)`` or ``("malformed", <reason>)``."""
    pairs = _split_pairs(value)
    if pairs is not None and pairs and pairs[0][0] == "v" and re.fullmatch(r"[1-9]\d*", pairs[0][1]) \
            and pairs[0][1] != "1":
        return "unsupported", pairs[0][1]
    if not value.isascii() or not value.isprintable() or " " in value:
        return "malformed", "not_printable_ascii"
    if len(value) > _MARK_MAX:
        return "malformed", "too_long"
    if pairs is None:
        return "malformed", "not_key_value_pairs"
    keys = [key for key, _ in pairs]
    if not keys or keys[0] != "v":
        return "malformed", "v_not_first"
    if pairs[0][1] != "1":
        return "malformed", "invalid_value:v"
    for key, item in pairs:
        if not _KEY_RE.match(key):
            return "malformed", f"invalid_key:{key}"
        if "=" in item:
            return "malformed", f"invalid_value:{key}"
    if len(set(keys)) != len(keys):
        duplicate = next(key for key in keys if keys.count(key) > 1)
        return "malformed", f"duplicate_key:{duplicate}"
    values = dict(pairs)
    for key in _REQUIRED:
        if key not in values:
            return "malformed", f"missing_key:{key}"
    if values["mode"] == "limit" and "limited" not in values:
        return "malformed", "missing_key:limited"
    if "limited" in values and values["mode"] != "limit":
        return "malformed", "limited_without_limit"
    for key, check in _MARK_FORMATS.items():
        if key in values and not check(values[key]):
            return "malformed", f"invalid_value:{key}"
    return "valid", None


def _mark_number(value: str) -> float:
    try:
        number = float(Decimal(value))
    except (InvalidOperation, ValueError):  # pragma: no cover - guarded by the grammar
        raise InvalidValue("number_invalid") from None
    return number


def parse_processing_mark(occurrences: Sequence[tuple[str, str | None]]) -> ProcessingMark:
    """Classify the mark (MD37) from ``(identity, value)`` pairs in store order.

    ``value`` is the decoded mark text, or ``None`` when the stored bytes could
    not be decoded as text (then the mark is present and malformed). Presence
    is independent of validity: any occurrence makes the mark present (MD26).
    """
    if not occurrences:
        return ProcessingMark("absent", None, None, None, None)
    locator = occurrences[0][0]
    values = [value for _, value in occurrences]

    def invalid(validity: str, detail: str, message: str) -> ProcessingMark:
        diagnostic = Diagnostic("processing_mark_invalid", message, MARK_STAGE, detail)
        return ProcessingMark(validity, None, None, None, locator, (diagnostic,))  # type: ignore[arg-type]

    if any(value is None for value in values):
        return invalid("malformed", "malformed:undecodable", "processing mark is not readable text")
    if len(set(values)) > 1:
        return invalid("malformed", "malformed:mark_inconsistent", "processing mark occurrences disagree")
    value = values[0]
    assert value is not None
    validity, reason = _mark_problem(value)
    if validity == "unsupported":
        return invalid("unsupported", f"unsupported_version:{reason}",
                       f"processing mark version {reason} is not supported")
    if validity == "malformed":
        return invalid("malformed", f"malformed:{reason}", f"processing mark is malformed ({reason})")
    fields = dict(pair.split("=", 1) for pair in value.split(";"))
    return ProcessingMark("valid", _mark_number(fields["target"]), fields["mode"],  # type: ignore[arg-type]
                          fields["origin"], locator)


# --- Processing mark: text and finalisation (MD26, MD34; CR-086) --------------------


def format_mark_number(value: float | None, *, signed: bool = False, strip: bool = False) -> str:
    """MD26 rounding: shortest round-trip decimal, two places, half-even; no negative zero.

    ``strip`` drops trailing zeros and a trailing point (``target``, ``tp``); ``signed`` forces a
    leading sign (``gain``). A missing or non-finite value is a bug: ``ValueError``, never written.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) \
            or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"a processing-mark number must be finite, got {value!r}")
    number = Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    if number == 0:
        number = Decimal("0.00")
    text = f"{number:.2f}"
    if strip:
        text = text.rstrip("0").rstrip(".")
    if signed and not text.startswith("-"):
        text = "+" + text
    return text


def ffmpeg_version_token(version: str) -> str:
    """MD26 ``ffmpeg``: the version string up to, not including, its second ``-``."""
    parts = version.split("-")
    return "-".join(parts[:2]) if len(parts) > 2 else version


def processing_mark_value(gain: GainPlan, measurement: LoudnessMeasurement, *, origin: str, ffmpeg_version: str,
                          run_date: str, tool_version: str) -> str:
    """The MD26 mark text from the last verified audio plan and measurement.

    ``limited`` is C-09 ``limited_db`` (the policy bound), present only for ``mode=limit``.
    ``tool_version`` is the bare version (``0.1.0``); the caller passes ``norma.__version__``, because
    this module imports only ``norma.models``. The text is checked against the MD37 grammar before it
    is returned, so a value that would read back as invalid (a non-finite number, a bad token, a text
    over 255 characters) is a bug, ``ValueError``, never written.
    """
    pairs = [("v", "1"), ("tool", f"norma-{tool_version}"),
             ("target", format_mark_number(gain.target_lufs, strip=True)),
             ("tp", format_mark_number(gain.ceiling_dbtp, strip=True)), ("mode", gain.action),
             ("gain", format_mark_number(gain.gain_db or 0.0, signed=True))]
    if gain.action == "limit":
        pairs.append(("limited", format_mark_number(gain.limiter_reduction_db)))
    pairs += [("out_i", format_mark_number(measurement.integrated_lufs)),
              ("out_tp", format_mark_number(measurement.true_peak_dbtp)),
              ("origin", origin), ("ffmpeg", ffmpeg_version_token(ffmpeg_version)), ("date", run_date)]
    text = ";".join(f"{key}={value}" for key, value in pairs)
    validity, reason = _mark_problem(text)
    if validity != "valid":
        raise ValueError(f"the processing mark would not be valid ({validity}: {reason}): {text!r}")
    return text


def finalise_mark(plan: MetadataPlan, mark: str, items: Sequence[RawNativeItem]) -> MetadataPlan:
    """``mark_finalised`` (MD26, MD34): a new revision whose only change is the mark.

    ``items`` are the mark's expected native items, one per target store that can hold it, prepared
    by the write edge (``mapping.native_recipes`` with ``mark``, then ``write.prepare_store``) so each
    carries the exact payload read-back produces; this module is pure and cannot prepare them. Each
    native row is ``output:…`` ``added`` and the roll-up ``extra:NORMA`` is ``added``; user values,
    fates, the loss-policy result and sidecar requests are unchanged. A plan whose mark is not
    ``pending`` (``final``, ``omitted``, ``not_storable``) is returned unchanged.
    """
    if plan.mark_state != "pending":
        return plan
    if _mark_problem(mark)[0] != "valid":
        raise ValueError(f"the planned processing mark is not valid: {mark!r}")
    items = tuple(items)
    if not items:
        raise ValueError("a pending mark needs a store that can hold it")
    ids = tuple(item.identity for item in items)
    if len(set(ids)) != len(ids) or set(ids) & {item.identity for item in plan.expected_native}:
        raise ValueError("the expected inventory already holds a processing mark")
    rows = tuple(NativeItemStatus("output:" + identity, "added", (), (identity,), "none", "") for identity in ids)
    rollup = FieldStatus("extra:" + MARK_NAME, "added", mark, mark, "generated", "none", (), ids, "")
    return _evolve(plan, plan_revision=plan.plan_revision + 1, expected_native=plan.expected_native + items,
                   native_statuses=plan.native_statuses + rows, statuses=plan.statuses + (rollup,),
                   mark_state="final")
