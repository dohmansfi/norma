"""CSV manifest loading (MS1–MS6, MS8, MS9, MS12; TST-M-04, F-17). Hermetic: bytes in, records out."""

from __future__ import annotations

from pathlib import Path

import pytest

from norma.errors import UsageError
from norma.metadata import manifest as m
from norma.models import ManifestCell

PATH = Path("/music/demos.csv")
HEADER = "file,title,artist,date,track,track_total,disc,disc_total,compilation,comment,artwork,BPM\n"


def load(text: str | bytes, path: Path = PATH):
    data = text.encode("utf-8") if isinstance(text, str) else text
    return m.parse_manifest(data, path)


def whole(text: str | bytes) -> m.ManifestError:
    with pytest.raises(m.ManifestError) as caught:
        load(text)
    return caught.value


COLUMNS = HEADER.strip().split(",")[1:]


def one_row(**columns: str):
    """One row under HEADER; each keyword is a column, its value the raw cell text (quoted here)."""
    unknown = set(columns) - set(COLUMNS)
    assert not unknown, unknown
    cells = ['"' + columns.get(name, "").replace('"', '""') + '"' for name in COLUMNS]
    data = load(HEADER + "a.wav," + ",".join(cells) + "\n")
    return data.rows[0], data


def test_example_manifest_loads_and_equals_the_json_example() -> None:
    root = Path(__file__).parents[3] / "specs" / "examples"
    csv_data = m.load_manifest(root / "manifest.csv")
    json_data = m.load_manifest(root / "manifest.json")
    assert csv_data.issues == () and json_data.issues == ()
    assert csv_data.defaults == json_data.defaults
    assert [(r.match_key, dict(r.cells)) for r in csv_data.rows] == [(r.match_key, dict(r.cells)) for r in json_data.rows]
    assert csv_data.defaults["track_total"] == ManifestCell("set", 4)
    assert csv_data.defaults["compilation"] == ManifestCell("set", False)
    assert csv_data.rows[2].cells["comment"] == ManifestCell("clear", None)


def test_multiline_quoted_cell_and_line_numbers() -> None:
    data = load('file,comment,title\n"a.wav","x\r\n""y""\nz",T\nb.wav,,U\n')
    assert data.rows[0].cells["comment"].value == 'x\n"y"\nz'
    assert [r.locator for r in data.rows] == ["demos.csv:2", "demos.csv:5"]


@pytest.mark.parametrize("prefix", ["", "\ufeff"])
@pytest.mark.parametrize("delimiter", [",", ";"])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_valid_variants_load_equal(prefix: str, delimiter: str, newline: str) -> None:
    text = prefix + newline.join([f"file{delimiter}title{delimiter}track", f"a.wav{delimiter}One{delimiter}3", ""])
    data = load(text.encode("utf-8"))
    assert data.rows[0].cells == {"title": ManifestCell("set", "One"), "track": ManifestCell("set", 3)}


def test_single_column_header_is_comma() -> None:
    assert load("file\na.wav\n").rows[0].cells == {}


def test_header_order_case_and_extras() -> None:
    data = load("BPM,Title,FILE\n92,One,a.wav\n")
    assert data.rows[0].cells == {"BPM": ManifestCell("set", "92"), "title": ManifestCell("set", "One")}


def test_empty_cell_is_no_opinion_and_clear_is_case_insensitive() -> None:
    row, _ = one_row(title="T", artist=" ", comment="[CLEAR] ")
    assert row.cells == {"title": ManifestCell("set", "T"), "comment": ManifestCell("clear", None)}


def test_text_is_trimmed_nfc_and_line_endings_normalised() -> None:
    data = load('file,comment\na.wav,"  Cafe\u0301 \r\n x"\n')
    assert data.rows[0].cells["comment"].value == "Caf\u00e9 \n x"


def test_defaults_row_and_blank_records() -> None:
    data = load("file,artist,track_total\n\n*,Me,4\n,,\na.wav,,\n")
    assert data.defaults == {"artist": ManifestCell("set", "Me"), "track_total": ManifestCell("set", 4)}
    assert [r.match_key for r in data.rows] == ["a.wav"]


def test_short_record_gives_missing_cells_no_opinion() -> None:
    data = load(HEADER + "a.wav,T\n")
    assert data.rows[0].cells == {"title": ManifestCell("set", "T")}


@pytest.mark.parametrize("cell, expected", [("3", 3), ("03", 3), ("3.0", 3), ("3.00", 3)])
def test_integer_forms(cell: str, expected: int) -> None:
    row, _ = one_row(track=cell)
    assert row.cells["track"] == ManifestCell("set", expected)


@pytest.mark.parametrize("cell", ["0", "-1", "3.5", "1e1", "x", "3,0"])
def test_integer_invalid(cell: str) -> None:
    _, data = one_row(track=cell)
    assert [i.detail for i in data.issues] == ["number_invalid"]


def test_compound_track_sets_total_and_disc() -> None:
    row, _ = one_row(track="3/12", disc="2/3")
    assert row.cells["track"].value == 3 and row.cells["track_total"].value == 12
    assert row.cells["disc"].value == 2 and row.cells["disc_total"].value == 3


def test_compound_agrees_with_equal_total_cell() -> None:
    row, _ = one_row(track="3/12", track_total="12")
    assert row.cells["track_total"].value == 12


@pytest.mark.parametrize("columns", [
    {"track": "13/12"}, {"track": "3/12", "track_total": "11"}, {"track": "13", "track_total": "12"},
    {"track": "3/12", "track_total": "[clear]"},
])
def test_pair_inconsistent(columns: dict[str, str]) -> None:
    row, data = one_row(**columns)
    assert row.cells == {}
    assert [i.detail for i in data.issues] == ["pair_inconsistent"]
    assert data.issues[0].field == "track" and data.issues[0].locator == "demos.csv:2"


def test_total_without_number_is_valid() -> None:
    row, data = one_row(track_total="12")
    assert row.cells == {"track_total": ManifestCell("set", 12)} and data.issues == ()


@pytest.mark.parametrize("cell", ["2026-03-12", "12-Mar", "Mar-12", "10/12/2026"])
def test_track_cell_that_looks_like_a_date(cell: str) -> None:
    _, data = one_row(track=cell)
    assert [i.detail for i in data.issues] == ["number_spreadsheet"]


@pytest.mark.parametrize("cell, detail", [
    ("2026", None), ("2026-04", None), ("2028-02-29", None),
    ("2026-02-30", "date_invalid"), ("2026-13", "date_invalid"), ("2027-02-29", "date_invalid"),
    ("26", "date_invalid"), ("2026/03/14", "date_invalid"),
    ("10/12/2026", "date_spreadsheet"), ("12-Mar-26", "date_spreadsheet"), ("Mar-26", "date_spreadsheet"),
    ("5 Oct 2026", "date_spreadsheet"), ("46000", "date_spreadsheet"),
])
def test_dates(cell: str, detail: str | None) -> None:
    row, data = one_row(date=cell)
    assert [i.detail for i in data.issues] == ([detail] if detail else [])
    if detail is None:
        assert row.cells["date"].value == cell


@pytest.mark.parametrize("cell, value", [("true", True), ("FALSE", False), ("1", True), ("0", False),
                                         ("Yes", True), ("no", False)])
def test_compilation_forms(cell: str, value: bool) -> None:
    row, _ = one_row(compilation=cell)
    assert row.cells["compilation"].value is value


def test_compilation_invalid() -> None:
    _, data = one_row(compilation="maybe")
    assert [i.detail for i in data.issues] == ["boolean_invalid"]


@pytest.mark.parametrize("cell, detail", [
    ("T\x07x", "text_invalid"), ("x" * 1001, "text_invalid"),
])
def test_text_problems(cell: str, detail: str) -> None:
    _, data = one_row(title=cell)
    assert [i.detail for i in data.issues] == [detail]


def test_line_break_in_title_is_invalid_but_allowed_in_comment_and_extras() -> None:
    _, data = one_row(title="a\nb", comment="c\nd", BPM="e\nf")
    assert [(i.field, i.detail) for i in data.issues] == [("title", "text_invalid")]
    row, data = one_row(comment="c\nd", BPM="e\nf")
    assert row.cells["comment"].value == "c\nd" and row.cells["BPM"].value == "e\nf" and data.issues == ()


def test_comment_limit_is_4000() -> None:
    row, _ = one_row(comment="x" * 4000)
    assert len(row.cells["comment"].value) == 4000  # type: ignore[arg-type]
    _, data = one_row(comment="x" * 4001)
    assert [i.detail for i in data.issues] == ["text_invalid"]


def test_artwork_resolves_against_the_manifest_folder_and_is_not_opened() -> None:
    row, _ = one_row(artwork="art/cover.png")
    assert row.cells["artwork"] == ManifestCell("set", Path("/music/art/cover.png"))
    row, _ = one_row(artwork="/abs/cover.png")
    assert row.cells["artwork"].value == Path("/abs/cover.png")


def test_artwork_too_long() -> None:
    _, data = one_row(artwork="a" * 1025)
    assert [i.detail for i in data.issues] == ["artwork_path_invalid"]


def test_all_row_problems_reported_and_row_keeps_no_cells() -> None:
    row, data = one_row(title="T", date="2026-02-30", track="x", compilation="maybe")
    assert row.cells == {}
    assert sorted(i.detail for i in data.issues) == ["boolean_invalid", "date_invalid", "number_invalid"]
    assert m.row_issues(data)["demos.csv:2"] == data.issues


def test_cell_count() -> None:
    data = load("file,title\na.wav,T,extra\nb.wav,U,\n")
    assert [(i.locator, i.detail) for i in data.issues] == [("demos.csv:2", "cell_count")]
    assert data.rows[0].cells == {} and data.rows[1].cells == {"title": ManifestCell("set", "U")}


def test_extra_column_with_clear() -> None:
    row, _ = one_row(BPM="[clear]")
    assert row.cells == {"BPM": ManifestCell("clear", None)}


def test_extras_are_not_unknown_column_errors() -> None:
    data = load("file,MOOD,Key_1\na.wav,calm,Am\n")
    assert data.issues == () and set(data.rows[0].cells) == {"MOOD", "Key_1"}


# --- Whole-manifest errors ---------------------------------------------------------------


def test_whole_errors_are_usage_errors_with_exit_2() -> None:
    error = whole("title\nx\n")
    assert isinstance(error, UsageError) and error.exit_code == 2
    assert error.detail == "missing_file_column"


@pytest.mark.parametrize("text, detail, locator", [
    ("file\ttitle\na.wav\tx\n", "delimiter", "demos.csv:1"),
    ("", "delimiter", "demos.csv:1"),
    ('file,title\na.wav,ab"cd\n', "csv_syntax", "demos.csv:2"),
    ('file,title\na.wav, "abc"\n', "csv_syntax", "demos.csv:2"),
    ('file,title\na.wav,"abc"d\n', "csv_syntax", "demos.csv:2"),
    ('file,title\na.wav,"abc\nb.wav,x\n', "csv_syntax", "demos.csv:2"),
    ("file,Title,title\na.wav,x,y\n", "header_duplicate", "demos.csv:1"),
    ("file,title,\na.wav,x,y\n", "header_empty_name", "demos.csv:1"),
    ("file,bad key\na.wav,x\n", "illegal_extra_name", "demos.csv:1"),
    ("file,NORMA\na.wav,x\n", "reserved_extra_name", "demos.csv:1"),
    ("file,replaygain_track_gain\na.wav,x\n", "reserved_extra_name", "demos.csv:1"),
    ("file,title\n*,x\n*,y\n", "defaults_duplicate", "demos.csv:3"),
    ("file,date\n*,2026-13\n", "defaults_invalid", "demos.csv:2"),
    ("file,title\n,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\n/abs/a.wav,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\n~/a.wav,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\nsub/,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\n.,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\n../..,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\nsub/..,x\n", "file_value_invalid", "demos.csv:2"),
    ("file,title\n\x00a.wav,x\n", "file_value_invalid", "demos.csv:2"),
])
def test_whole_manifest_errors(text: str, detail: str, locator: str) -> None:
    error = whole(text)
    assert error.issues[0].detail == detail and error.issues[0].locator == locator
    assert detail in error.issues[0].detail and error.messages[0].startswith(locator)


def test_unnamed_empty_column_is_ignored() -> None:
    assert load("file,title,\na.wav,x,\n").rows[0].cells == {"title": ManifestCell("set", "x")}


def test_encoding_errors() -> None:
    error = whole(b"file,title\na.wav,\xff\n")
    assert error.detail == "encoding" and "byte 17" in error.messages[0]
    assert whole("\ufeff".encode("utf-16") + "file\n".encode("utf-16-le")).detail == "encoding"
    assert "byte 20" in whole(b"\xef\xbb\xbffile,title\na.wav,\xff\n").messages[0]


def test_limits() -> None:
    rows = "".join(f"a{i}.wav\n" for i in range(m.MAX_ROWS + 1))
    assert whole("file\n" + rows).detail == "too_large"
    assert whole("file," + ",".join(f"c{i}" for i in range(m.MAX_COLUMNS)) + "\n").detail == "too_large"
    assert load("file\n" + rows[: -len(f"a{m.MAX_ROWS}.wav\n")]).rows[-1].match_key == f"a{m.MAX_ROWS - 1}.wav"
    with pytest.raises(m.ManifestError) as caught:
        m.parse_manifest(b"file\n" + b"x" * m.MAX_BYTES, PATH)
    assert caught.value.detail == "too_large"


def test_only_csv_and_json_suffixes() -> None:
    with pytest.raises(m.ManifestError) as caught:
        m.parse_manifest(b"file\n", Path("/music/demos.txt"))
    assert caught.value.detail == "unreadable"


def test_whole_errors_report_every_issue_up_to_fifty_and_count_the_rest() -> None:
    body = "".join(f"/abs{i}.wav,x\n" for i in range(60))
    error = whole("file,title\n" + body)
    assert len(error.issues) == m.MAX_ISSUES + 1
    assert error.issues[-1].detail == m.MORE_ISSUES and error.issues[-1].field == "10"


def test_row_issues_capped_at_fifty_with_a_count() -> None:
    body = "".join(f"a{i}.wav,x\n" for i in range(55))
    data = load("file,track\n" + body)
    assert len(data.issues) == m.MAX_ISSUES + 1 and data.issues[-1].detail == m.MORE_ISSUES
    assert data.issues[-1].field == "5"


def test_load_manifest_reads_a_local_file_and_reports_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "demos.csv"
    path.write_text("file,title\na.wav,One\n", encoding="utf-8")
    data = m.load_manifest(path)
    assert data.path == path.resolve() and data.rows[0].cells["title"].value == "One"
    with pytest.raises(m.ManifestError) as caught:
        m.load_manifest(tmp_path / "missing.csv")
    assert caught.value.detail == "unreadable"


# --- Typo warnings (MS3) ----------------------------------------------------------------


@pytest.mark.parametrize("column, suggestion", [
    ("titel", "title"), ("genres", "genre"), ("year", "date"), ("albumartist", "album_artist"),
    ("TrackNo", "track"), ("cover", "artwork"), ("description", "comment"), ("path", "file"),
    ("comments", "comment"),
])
def test_likely_typo_warns_but_accepts_the_extra(column: str, suggestion: str) -> None:
    assert m.likely_typo(column) == suggestion
    data = load(f"file,{column}\na.wav,x\n")
    assert [(i.detail, i.field, i.locator) for i in data.issues] == [("likely_typo", column, "demos.csv:1")]
    assert data.rows[0].cells == {column: ManifestCell("set", "x")}
    assert m.row_issues(data) == {}


@pytest.mark.parametrize("column", ["BPM", "MOOD", "Key_1", "ISRC", "label"])
def test_ordinary_extras_do_not_warn(column: str) -> None:
    assert m.likely_typo(column) is None
