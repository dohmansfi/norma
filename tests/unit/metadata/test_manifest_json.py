"""JSON manifest loading (MS7, MS11; TST-M-04, F-17). Hermetic: bytes in, records out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from norma.metadata import manifest as m
from norma.models import ManifestCell

PATH = Path("/music/demos.json")


def load(document: object | str | bytes, path: Path = PATH):
    if isinstance(document, bytes):
        data = document
    else:
        data = (document if isinstance(document, str) else json.dumps(document)).encode("utf-8")
    return m.parse_manifest(data, path)


def whole(document: object | str | bytes) -> m.ManifestError:
    with pytest.raises(m.ManifestError) as caught:
        load(document)
    return caught.value


def files(**entry: object) -> dict[str, object]:
    return {"version": 1, "files": {"a.wav": entry}}


def test_valid_document() -> None:
    data = load({"version": 1, "defaults": {"artist": "Me", "extras": {"LABEL": "X"}},
                 "files": {"a.wav": {"title": "T", "track": 3, "compilation": False, "date": "2026-04",
                                     "comment": None, "extras": {"BPM": "92", "MOOD": None}}}})
    assert data.version == 1 and data.issues == ()
    assert data.defaults == {"artist": ManifestCell("set", "Me"), "LABEL": ManifestCell("set", "X")}
    assert data.rows[0].cells == {
        "title": ManifestCell("set", "T"), "track": ManifestCell("set", 3), "compilation": ManifestCell("set", False),
        "date": ManifestCell("set", "2026-04"), "comment": ManifestCell("clear", None),
        "BPM": ManifestCell("set", "92"), "MOOD": ManifestCell("clear", None)}
    assert data.rows[0].locator == "demos.json#/files/a.wav"


def test_bom_is_accepted_and_locator_escapes_pointer_characters() -> None:
    data = load(b"\xef\xbb\xbf" + json.dumps({"version": 1, "files": {"sub/a~b.wav": {}}}).encode())
    assert data.rows[0].locator == "demos.json#/files/sub~1a~0b.wav"


def test_absent_key_is_no_opinion() -> None:
    assert load(files(title="T")).rows[0].cells.keys() == {"title"}


@pytest.mark.parametrize("name", ["titel", "genres", "YEAR", "albumartist", "track_no", "cover", "path"])
@pytest.mark.parametrize("value", ["alternate", None])
@pytest.mark.parametrize("layer", ["defaults", "files"])
def test_json_extra_typo_warns_without_changing_set_or_clear(name, value, layer) -> None:
    document = {"version": 1, "files": {"sub/a~b.wav": {}}}
    if layer == "defaults":
        document["defaults"] = {"extras": {name: value}}
        locator = "demos.json#/defaults/extras/" + name
    else:
        document["files"]["sub/a~b.wav"] = {"extras": {name: value}}
        locator = "demos.json#/files/sub~1a~0b.wav/extras/" + name
    data = load(document)
    assert [(issue.detail, issue.field, issue.locator, issue.affected_indices) for issue in data.issues] == [
        ("likely_typo", name, locator, ())]
    assert m.row_issues(data) == {}
    cells = data.defaults if layer == "defaults" else data.rows[0].cells
    assert cells == {name: ManifestCell("clear", None) if value is None else ManifestCell("set", value)}


def test_json_typo_warnings_keep_each_locator_and_do_not_invalidate_bad_rows() -> None:
    data = load({"version": 1, "defaults": {"extras": {"titel": "default"}}, "files": {
        "a.wav": {"extras": {"titel": "row"}},
        "b.wav": {"track": 0, "extras": {"titel": "row"}}}})
    assert [(issue.locator, issue.detail) for issue in data.issues] == [
        ("demos.json#/files/b.wav", "number_invalid"),
        ("demos.json#/defaults/extras/titel", "likely_typo"),
        ("demos.json#/files/a.wav/extras/titel", "likely_typo"),
        ("demos.json#/files/b.wav/extras/titel", "likely_typo")]
    assert list(m.row_issues(data)) == ["demos.json#/files/b.wav"]
    assert data.rows[0].cells == {"titel": ManifestCell("set", "row")}
    assert data.rows[1].cells == {}


@pytest.mark.parametrize("name", ["MOOD", "BPM", "DEMO", "CustomLabel"])
def test_json_unrelated_extra_names_do_not_warn(name) -> None:
    assert load(files(extras={name: "value"})).issues == ()


@pytest.mark.parametrize("token", ["3.0", "3e0", "1.0E+2", "3.5"])
def test_integer_fields_refuse_fraction_and_exponent_tokens(token: str) -> None:
    data = load('{"version": 1, "files": {"a.wav": {"track": %s}}}' % token)
    assert [(i.detail, i.field) for i in data.issues] == [("number_invalid", "track")]
    assert data.rows[0].cells == {}


@pytest.mark.parametrize("value", ['"3"', "true", "false", "0", "-2", "[]", "{}"])
def test_integer_fields_refuse_other_types_and_values(value: str) -> None:
    data = load('{"version": 1, "files": {"a.wav": {"disc_total": %s}}}' % value)
    assert [i.detail for i in data.issues] == ["number_invalid"]


def test_integer_cell_is_a_plain_int() -> None:
    cell = load(files(track=3)).rows[0].cells["track"]
    assert cell.value == 3 and type(cell.value) is int


@pytest.mark.parametrize("version", ["2", "1.0", "true", '"1"', "null", "0", "1e0"])
def test_version_must_be_the_integer_one(version: str) -> None:
    error = whole('{"version": %s, "files": {"a.wav": {}}}' % version)
    assert error.detail == "version" and error.issues[0].locator == "demos.json#/version"


def test_version_absent() -> None:
    assert whole({"files": {"a.wav": {}}}).detail == "version"


@pytest.mark.parametrize("document, locator", [
    ({"version": 1}, "demos.json#/files"),
    ({"version": 1, "files": {}}, "demos.json#/files"),
    ({"version": 1, "files": []}, "demos.json#/files"),
    ({"version": 1, "files": {"a.wav": {}}, "other": 1}, "demos.json#/other"),
    ([1], "demos.json#"),
])
def test_json_structure(document: object, locator: str) -> None:
    error = whole(document)
    assert error.detail == "json_structure" and error.issues[0].locator == locator


@pytest.mark.parametrize("text", [
    "{", "", '{"version": 1, "version": 1, "files": {"a.wav": {}}}',
    '{"version": 1, "files": {"a.wav": {"title": "x", "title": "y"}}}',
    '{"version": NaN, "files": {"a.wav": {}}}', '{"version": 1, "files": {"a.wav": {"track": Infinity}}}',
])
def test_json_syntax(text: str) -> None:
    assert whole(text).detail == "json_syntax"


def test_utf8_only() -> None:
    assert whole(b'{"version": 1, "files": {"a.wav": {"title": "\xff"}}}').detail == "encoding"


@pytest.mark.parametrize("key", ["", "  ", "*", "/abs.wav", "~/a.wav", "sub/", ".", "..", "a/..", "a\u0000b"])
def test_file_keys_that_are_not_selectors(key: str) -> None:
    error = whole({"version": 1, "files": {key: {}}})
    assert error.detail == "file_value_invalid"


def test_dot_segment_keys_are_valid_selectors() -> None:
    data = load({"version": 1, "files": {"sub/../a.wav": {}, "../Other/x.wav": {}}})
    assert [str(r.lexical_key) for r in data.rows] == ["a.wav", "../Other/x.wav"]


@pytest.mark.parametrize("fields, field, detail", [
    ({"title": ""}, "title", "text_invalid"),
    ({"title": "   "}, "title", "text_invalid"),
    ({"title": 3}, "title", "text_invalid"),
    ({"title": "a\nb"}, "title", "text_invalid"),
    ({"title": "x" * 1001}, "title", "text_invalid"),
    ({"comment": "x" * 4001}, "comment", "text_invalid"),
    ({"date": "2026-02-30"}, "date", "date_invalid"),
    ({"date": 2026}, "date", "date_invalid"),
    ({"date": "10/12/2026"}, "date", "date_invalid"),
    ({"compilation": "true"}, "compilation", "boolean_invalid"),
    ({"compilation": 1}, "compilation", "boolean_invalid"),
    ({"artwork": ""}, "artwork", "artwork_path_invalid"),
    ({"artwork": "a" * 1025}, "artwork", "artwork_path_invalid"),
    ({"artwork": 3}, "artwork", "artwork_path_invalid"),
    ({"titel": "x"}, "titel", "unknown_field"),
    ({"extras": {"bad key": "x"}}, "bad key", "illegal_extra_name"),
    ({"extras": {"title": "x"}}, "title", "illegal_extra_name"),
    ({"extras": {"NORMA": "x"}}, "NORMA", "reserved_extra_name"),
    ({"extras": {"ReplayGain_Track_Gain": "x"}}, "ReplayGain_Track_Gain", "reserved_extra_name"),
    ({"extras": {"BPM": ""}}, "BPM", "text_invalid"),
    ({"extras": {"BPM": 92}}, "BPM", "text_invalid"),
    ({"extras": {"BPM": "x", "bpm": "y"}}, "bpm", "illegal_extra_name"),
    ({"extras": ["x"]}, "extras", "text_invalid"),
    ({"track": 13, "track_total": 12}, "track", "pair_inconsistent"),
    ({"disc": 3, "disc_total": 2}, "disc", "pair_inconsistent"),
])
def test_row_problems_fail_that_row_only(fields: dict[str, object], field: str, detail: str) -> None:
    data = load({"version": 1, "files": {"a.wav": fields, "b.wav": {"title": "ok"}}})
    assert [(i.locator, i.field, i.detail) for i in data.issues] == [("demos.json#/files/a.wav", field, detail)]
    assert data.rows[0].cells == {} and data.rows[1].cells == {"title": ManifestCell("set", "ok")}


def test_entry_that_is_not_an_object_is_a_row_problem() -> None:
    data = load({"version": 1, "files": {"a.wav": "x"}})
    assert [i.detail for i in data.issues] == ["json_structure"] and data.rows[0].cells == {}


def test_number_may_equal_total() -> None:
    assert load(files(track=12, track_total=12)).issues == ()


def test_text_is_trimmed_and_nfc() -> None:
    assert load(files(title="  Cafe\u0301 ")).rows[0].cells["title"].value == "Caf\u00e9"


@pytest.mark.parametrize("fields", [{"date": "2026-13"}, {"track": 3.0}, {"title": ""}, {"nope": 1},
                                    {"track": 3, "track_total": 2}, {"extras": {"NORMA": "x"}}])
def test_defaults_problems_are_whole_manifest_errors(fields: dict[str, object]) -> None:
    error = whole({"version": 1, "defaults": fields, "files": {"a.wav": {}}})
    assert error.detail == "defaults_invalid" and error.issues[0].locator.startswith("demos.json#/defaults")


def test_defaults_must_be_an_object() -> None:
    assert whole({"version": 1, "defaults": 3, "files": {"a.wav": {}}}).detail == "defaults_invalid"


def test_artwork_relative_to_manifest_folder() -> None:
    assert load(files(artwork="art/c.png")).rows[0].cells["artwork"].value == Path("/music/art/c.png")


def test_issue_cap() -> None:
    document = {"version": 1, "files": {f"a{i}.wav": {"track": 0} for i in range(60)}}
    data = load(document)
    assert len(data.issues) == m.MAX_ISSUES + 1 and data.issues[-1].field == "10"


def test_every_row_is_kept_even_when_invalid() -> None:
    data = load({"version": 1, "files": {"a.wav": {"track": 0}, "b.wav": {}}})
    assert [r.match_key for r in data.rows] == ["a.wav", "b.wav"]


# --- CSV/JSON semantic equivalence (TST-M-04) -------------------------------------------


def test_equivalent_csv_and_json_load_to_the_same_records() -> None:
    csv_text = ('file,title,artist,date,track,track_total,disc,compilation,comment,artwork,BPM\n'
                '*,,Me,2026,,4,1,false,,cover.jpg,\n'
                'a.wav,One,,2026-03-14,1,,,,[clear],,92\n'
                'sub/b.wav,"Two, ""II""",,,2/4,,,,,,\n')
    json_doc = {"version": 1,
                "defaults": {"artist": "Me", "date": "2026", "track_total": 4, "disc": 1,
                             "compilation": False, "artwork": "cover.jpg"},
                "files": {"a.wav": {"title": "One", "date": "2026-03-14", "track": 1, "comment": None,
                                    "extras": {"BPM": "92"}},
                          "sub/b.wav": {"title": 'Two, "II"', "track": 2, "track_total": 4}}}
    left = m.parse_manifest(csv_text.encode(), Path("/music/x.csv"))
    right = m.parse_manifest(json.dumps(json_doc).encode(), Path("/music/x.json"))
    assert left.defaults == right.defaults
    assert [(r.match_key, r.lexical_key, r.comparison_key, dict(r.cells)) for r in left.rows] == \
           [(r.match_key, r.lexical_key, r.comparison_key, dict(r.cells)) for r in right.rows]
