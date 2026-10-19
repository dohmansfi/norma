"""Pure canonical-field helpers and mark parsing (metadata/model.py; MS1, MS3, MS4, MD3–MD5, MD26, MD37)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from norma.metadata import model
from norma.metadata.model import InvalidValue

FIXTURE_MARK = (
    "v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
    "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05"
)


def test_model_is_pure() -> None:
    tree = ast.parse(Path(model.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    assert "mutagen" not in roots
    assert {name for name in roots if name.startswith("norma")} == {"norma.models"}


def test_fourteen_canonical_fields() -> None:
    assert len(model.CANONICAL_FIELDS) == 14
    assert model.TEXT_FIELDS | model.INTEGER_FIELDS | {"date", "artwork", "compilation"} == set(model.CANONICAL_FIELDS)


@pytest.mark.parametrize(("raw", "field", "expected"), [
    ("  Night Bus  ", "title", "Night Bus"),
    ("Cafe\u0301", "title", "Café"),
    ("a\r\nb\rc", "comment", "a\nb\nc"),
    ("tab\there", "title", "tab\there"),
    ("   ", "title", ""),
    ("x" * 1000, "title", "x" * 1000),
    ("x" * 4000, "comment", "x" * 4000),
    ("line\nbreak", model.EXTRA, "line\nbreak"),
])
def test_canonical_text_accepts(raw: str, field: str, expected: str) -> None:
    assert model.canonical_text(raw, field) == expected


@pytest.mark.parametrize(("raw", "field", "detail"), [
    ("a\nb", "title", "line_break_not_allowed"),
    ("bell\x07", "artist", "text_invalid"),
    ("nul\x00x", "comment", "text_invalid"),
    ("x" * 1001, "title", "too_long"),
    ("x" * 4001, "comment", "too_long"),
    ("x" * 4001, model.EXTRA, "too_long"),
])
def test_canonical_text_refuses(raw: str, field: str, detail: str) -> None:
    with pytest.raises(InvalidValue) as error:
        model.canonical_text(raw, field)
    assert error.value.detail == detail


def test_canonical_text_rejects_non_text_field() -> None:
    with pytest.raises(ValueError):
        model.canonical_text("3", "track")


@pytest.mark.parametrize("value", ["2026", "2026-10", "2026-10-05", "2028-02-29"])
def test_dates_keep_precision(value: str) -> None:
    assert model.canonical_date(value) == value
    assert model.source_date(value) == value
    assert model.project_date(value) == value
    assert not model.time_of_day_dropped(value)


@pytest.mark.parametrize("value", ["2026-02-30", "2026-13", "Spring 2026", "26", "2026/10/05", "0000", "2027-02-29"])
def test_invalid_dates(value: str) -> None:
    with pytest.raises(InvalidValue):
        model.source_date(value)


@pytest.mark.parametrize(("value", "date"), [
    ("2026-10-05T07:00:00Z", "2026-10-05"),
    ("2026-10-05T07:00", "2026-10-05"),
    ("2026-10-05T07", "2026-10-05"),
    ("2026-10-05T23:59:59.123+01:00", "2026-10-05"),
    ("2026-10-05T07:00:00-0530", "2026-10-05"),
])
def test_source_timestamps_project_to_their_date(value: str, date: str) -> None:
    assert model.source_date(value) == value  # the stored text is kept (MD5, MD40 requested)
    assert model.project_date(value) == date
    assert model.time_of_day_dropped(value)


@pytest.mark.parametrize("value", ["2026-10-05T25:00", "2026-02-30T07:00", "2026-10-05 07:00", "2026-10-05T07:61"])
def test_invalid_timestamps(value: str) -> None:
    assert not model.is_source_timestamp(value)
    with pytest.raises(InvalidValue):
        model.canonical_date(value)


def test_timestamp_never_enters_canonical_date() -> None:
    with pytest.raises(InvalidValue):
        model.canonical_date("2026-10-05T07:00:00Z")


@pytest.mark.parametrize(("year", "ddmm", "expected"), [
    ("2026", None, "2026"), ("2026", "0510", "2026-10-05"), ("2026", " 0510 ", "2026-10-05"),
])
def test_id3v23_date(year: str, ddmm: str | None, expected: str) -> None:
    assert model.id3v23_date(year, ddmm) == expected


@pytest.mark.parametrize(("year", "ddmm"), [("2026", "3102"), ("2026", "5"), ("26", None), ("2026-10", None)])
def test_id3v23_date_invalid(year: str, ddmm: str | None) -> None:
    with pytest.raises(InvalidValue):
        model.id3v23_date(year, ddmm)


@pytest.mark.parametrize(("text", "expected"), [
    ("3", (3, None, False)), ("03/12", (3, 12, False)), ("0/12", (None, 12, False)), ("/12", (None, 12, False)),
    ("3/0", (3, None, False)), ("3/", (3, None, False)), ("", (None, None, False)), (" 3 / 12 ", (3, 12, False)),
    ("13/12", (13, None, True)),
])
def test_number_pairs(text: str, expected: tuple) -> None:
    assert model.parse_number_pair(text) == expected


@pytest.mark.parametrize("text", ["A1", "3/12/1", "-3", "3.0", "2026-03-12"])
def test_number_pairs_invalid(text: str) -> None:
    with pytest.raises(InvalidValue):
        model.parse_number_pair(text)


def test_parse_integer() -> None:
    assert model.parse_integer("012") == 12
    assert model.parse_integer("0") is None
    assert model.parse_integer("") is None
    with pytest.raises(InvalidValue):
        model.parse_integer("twelve")


@pytest.mark.parametrize(("text", "expected"), [("1", True), ("0", False), ("TRUE", True), (" false ", False)])
def test_compilation(text: str, expected: bool) -> None:
    assert model.parse_compilation(text) is expected


@pytest.mark.parametrize("text", ["yes", "2", ""])
def test_compilation_invalid(text: str) -> None:
    with pytest.raises(InvalidValue):
        model.parse_compilation(text)


def test_values_equal_does_not_confuse_bool_and_int() -> None:
    assert model.values_equal(True, True)
    assert not model.values_equal(True, 1)
    assert not model.values_equal(1, "1")
    assert model.values_equal("2026", "2026")


@pytest.mark.parametrize(("name", "problem"), [
    ("BPM", None), ("MOOD", None), ("demo_note", None), ("YEAR", None),
    ("Title", "illegal_extra_name"), ("FILE", "illegal_extra_name"), ("bad key", "illegal_extra_name"),
    ("1BPM", "illegal_extra_name"), ("A" * 65, "illegal_extra_name"), ("", "illegal_extra_name"),
    ("NORMA", "reserved_extra_name"), ("norma", "reserved_extra_name"), ("iTunNORM", "reserved_extra_name"),
    ("iTunSMPB", "reserved_extra_name"), ("Encoder", "reserved_extra_name"),
    ("REPLAYGAIN_TRACK_GAIN", "reserved_extra_name"), ("r128_track_gain", "reserved_extra_name"),
])
def test_extra_names(name: str, problem: str | None) -> None:
    assert model.extra_name_problem(name) == problem


@pytest.mark.parametrize(("container", "stores"), [
    ("mp4", ("mp4",)), ("flac", ("vorbis", "id3v2")), ("mp3", ("id3v2", "apev2", "id3v1")),
    ("aiff", ("id3v2", "aiff-text")), ("wav", ("id3v2", "riff-info")), ("rf64", ("id3v2", "riff-info")),
])
def test_store_priority_follows_md3(container: str, stores: tuple[str, ...]) -> None:
    assert model.container_stores(container) == stores


@pytest.mark.parametrize("container", ["caf", "adts", "ogg"])
def test_containers_without_md3_row_are_refused(container: str) -> None:
    with pytest.raises(ValueError, match="no MD3 store row"):
        model.store_family(container)


def test_identities_follow_md4() -> None:
    assert model.native_identity("riff-info", "INAM") == "store:riff-info:INAM"
    assert model.native_identity("vorbis", "ARTIST", 2) == "store:vorbis:ARTIST#2"
    assert model.nontag_identity("chunk", "bext") == "item:chunk:bext"
    assert model.nontag_identity("chapters") == "item:chapters"


@pytest.mark.parametrize(("store", "key", "fields"), [
    ("id3v2", "TIT2", ("title",)), ("id3v2", "TT2", ("title",)), ("id3v2", "TRCK", ("track", "track_total")),
    ("id3v2", "COMM::eng", ("comment",)), ("id3v2", "COMM:iTunNORM:eng", ()), ("id3v2", "TYER", ("date",)),
    ("id3v2", "APIC:", ("artwork",)), ("id3v2", "TXXX:BPM", ()),
    ("vorbis", "tracktotal", ("track_total",)), ("vorbis", "ALBUMARTIST", ("album_artist",)),
    ("mp4", "trkn", ("track", "track_total")), ("mp4", "©day", ("date",)), ("mp4", "covr", ("artwork",)),
    ("riff-info", "IPRT", ("track",)), ("riff-info", "ICOP", ()), ("aiff-text", "NAME", ("title",)),
    ("apev2", "Album Artist", ("album_artist",)), ("id3v1", "year", ("date",)),
])
def test_native_fields(store: str, key: str, fields: tuple[str, ...]) -> None:
    assert model.native_fields(store, key) == fields


@pytest.mark.parametrize(("store", "key", "extra"), [
    ("id3v2", "TXXX:BPM", "BPM"), ("id3v2", "TXXX:REPLAYGAIN_TRACK_GAIN", None), ("id3v2", "TXXX:NORMA", None),
    ("id3v2", "TXXX:TITLE", None), ("id3v2", "PRIV:com.example", None), ("vorbis", "Mood", "Mood"),
    ("vorbis", "CHAPTER001", None), ("vorbis", "ENCODER", None), ("mp4", "----:com.apple.iTunes:BPM", "BPM"),
    ("mp4", "----:com.example:BPM", None), ("mp4", "tmpo", None), ("riff-info", "ICOP", "ICOP"),
    ("riff-info", "ISFT", None), ("apev2", "Custom", "Custom"), ("apev2", "Cover Art (Front)", None),
    ("aiff-text", "(c) ", None), ("id3v1", "comment", None),
])
def test_native_extra(store: str, key: str, extra: str | None) -> None:
    assert model.native_extra(store, key) == extra


@pytest.mark.parametrize(("store", "key", "expected"), [
    ("id3v2", "TXXX:NORMA", True), ("id3v2", "TXXX:norma", True), ("vorbis", "Norma", True),
    ("mp4", "----:com.apple.iTunes:NORMA", True), ("mp4", "----:org.example:norma", True), ("apev2", "NORMA", True),
    ("id3v2", "TXXX:NORMAL", False), ("riff-info", "NORM", False), ("mp4", "©nam", False),
])
def test_mark_detection_is_case_insensitive(store: str, key: str, expected: bool) -> None:
    assert model.is_mark_key(store, key) is expected


# --- Processing mark (MD37) ---------------------------------------------------------------


def test_mark_absent() -> None:
    mark = model.parse_processing_mark([])
    assert (mark.validity, mark.locator, mark.diagnostics) == ("absent", None, ())


def test_fixture_mark_is_valid() -> None:
    mark = model.parse_processing_mark([("store:id3v2:TXXX:NORMA", FIXTURE_MARK)])
    assert mark.validity == "valid"
    assert (mark.target_lufs, mark.mode, mark.origin_codec) == (-16.0, "limit", "pcm_s24le")
    assert mark.locator == "store:id3v2:TXXX:NORMA" and mark.diagnostics == ()


def test_unknown_keys_are_ignored() -> None:
    mark = model.parse_processing_mark([("x", FIXTURE_MARK + ";future_key=abc")])
    assert mark.validity == "valid"


def _mark(**changes: str | None) -> str:
    pairs = dict(pair.split("=", 1) for pair in FIXTURE_MARK.split(";"))
    for key, value in changes.items():
        if value is None:
            pairs.pop(key)
        else:
            pairs[key] = value
    return ";".join(f"{key}={value}" for key, value in pairs.items())


@pytest.mark.parametrize(("value", "detail"), [
    ("v=1;target=-14;tp=-1;gain=0;mode=linear", "malformed:missing_key:tool"),  # the F-18 mark-invalid value
    (_mark(limited=None), "malformed:missing_key:limited"),
    (_mark(mode="linear"), "malformed:limited_without_limit"),
    (_mark(gain="5.80"), "malformed:invalid_value:gain"),
    (_mark(target="-16.125"), "malformed:invalid_value:target"),
    (_mark(out_i="-16.0"), "malformed:invalid_value:out_i"),
    (_mark(mode="loud", limited=None), "malformed:invalid_value:mode"),
    (_mark(date="2026-02-30"), "malformed:invalid_value:date"),
    (_mark(origin="PCM"), "malformed:invalid_value:origin"),
    ("tool=x;" + _mark(), "malformed:v_not_first"),
    (_mark() + ";tool=again", "malformed:duplicate_key:tool"),
    (_mark().replace("v=1", "v=one"), "malformed:invalid_value:v"),
    ("v=1;tool", "malformed:not_key_value_pairs"),
    (_mark() + ";Bad=1", "malformed:invalid_key:Bad"),
    (_mark(tool="norma 0.1"), "malformed:not_printable_ascii"),
    (_mark(tool="x" * 300), "malformed:too_long"),
    (_mark(tool="a=b"), "malformed:invalid_value:tool"),
])
def test_malformed_marks(value: str, detail: str) -> None:
    mark = model.parse_processing_mark([("store:vorbis:NORMA", value)])
    assert mark.validity == "malformed"
    assert (mark.target_lufs, mark.mode, mark.origin_codec) == (None, None, None)
    assert mark.locator == "store:vorbis:NORMA"
    assert [(d.code, d.stage, d.detail) for d in mark.diagnostics] == [("processing_mark_invalid", "tags_read", detail)]


def test_unsupported_version() -> None:
    mark = model.parse_processing_mark([("x", "v=2;anything=goes")])
    assert mark.validity == "unsupported"
    assert mark.diagnostics[0].detail == "unsupported_version:2"


def test_inconsistent_and_undecodable_marks_are_present_but_malformed() -> None:
    inconsistent = model.parse_processing_mark([("a", FIXTURE_MARK), ("b", _mark(target="-14"))])
    assert inconsistent.validity == "malformed" and inconsistent.locator == "a"
    assert inconsistent.diagnostics[0].detail == "malformed:mark_inconsistent"
    undecodable = model.parse_processing_mark([("a", None)])
    assert undecodable.validity == "malformed"
    assert undecodable.diagnostics[0].detail == "malformed:undecodable"


def test_equal_occurrences_are_one_valid_mark() -> None:
    mark = model.parse_processing_mark([("a", FIXTURE_MARK), ("b", FIXTURE_MARK)])
    assert mark.validity == "valid" and mark.locator == "a"
