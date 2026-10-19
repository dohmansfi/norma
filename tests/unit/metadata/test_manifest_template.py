"""Template content (MD28; TST-M-04): round trip through the loader and matcher."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path, PurePosixPath

from norma.metadata import manifest as m
from norma.models import InputItem, ManifestCell

FOLDER = Path("/m/Demos")


def entry(name: str, extras: dict[str, str] | None = None, **fields: object) -> m.TemplateEntry:
    return m.TemplateEntry(FOLDER / name, fields, extras or {})  # type: ignore[arg-type]


ENTRIES = [
    entry("Spring/01 One.wav", {"BPM": "92", "mood": "calm"}, title="One", artist="Me", album="A", date="2026-03",
          track=1, track_total=2, disc=1, disc_total=1, genre="Indie", compilation=False, comment="a, \"b\"\nc"),
    entry("Spring/02 Two.mp3", {"BPM": "104", "Key": "Am"}, title="Two", artist="Me", album="A", date="2026",
          track=2, track_total=2, disc=1, disc_total=1, genre="Indie", compilation=False),
]


def rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


def test_csv_columns_and_order() -> None:
    table = rows(m.build_template(ENTRIES, FOLDER))
    assert table[0] == ["file", "title", "artist", "album_artist", "album", "date", "track", "track_total", "disc",
                        "disc_total", "genre", "composer", "comment", "compilation", "artwork",
                        "BPM", "Key", "mood"]


def test_defaults_row_holds_identical_values_and_file_rows_leave_them_empty() -> None:
    table = rows(m.build_template(ENTRIES, FOLDER))
    header = table[0]
    defaults = dict(zip(header, table[1], strict=True))
    assert defaults["file"] == "*"
    assert {k: v for k, v in defaults.items() if v} == {
        "file": "*", "artist": "Me", "album": "A", "disc": "1", "disc_total": "1", "genre": "Indie",
        "track_total": "2", "compilation": "false"}
    first = dict(zip(header, table[2], strict=True))
    assert first["file"] == "Spring/01 One.wav" and first["artist"] == "" and first["track"] == "1"
    assert first["date"] == "2026-03" and first["comment"] == 'a, "b"\nc' and first["artwork"] == ""
    assert first["BPM"] == "92" and first["mood"] == "calm" and first["Key"] == ""


def test_no_compound_numbers_and_dates_are_text() -> None:
    text = m.build_template(ENTRIES, FOLDER)
    assert "1/2" not in text and "2026-03" in text


def test_template_loads_and_matches_its_own_inputs() -> None:
    for form, name in (("csv", "t.csv"), ("json", "t.json")):
        text = m.build_template(ENTRIES, FOLDER, form)
        data = m.parse_manifest(text.encode("utf-8"), FOLDER / name)
        assert data.issues == ()
        items = [InputItem(e.source_path, e.source_path.parent, PurePosixPath(e.source_path.name), (1, i), i, False)
                 for i, e in enumerate(ENTRIES)]
        result = m.match_manifest(data, items)
        assert [result.inputs[i].row.match_key for i in (0, 1)] == ["Spring/01 One.wav", "Spring/02 Two.mp3"]  # type: ignore[union-attr]
        assert result.unmatched == ()
        assert data.defaults["artist"] == ManifestCell("set", "Me")
        assert result.inputs[0].row.cells["comment"].value == 'a, "b"\nc'  # type: ignore[union-attr]
        assert result.inputs[0].row.cells["date"].value == "2026-03"  # type: ignore[union-attr]
        assert result.inputs[0].row.cells["track"] == ManifestCell("set", 1)  # type: ignore[union-attr]


def test_csv_and_json_templates_are_equivalent() -> None:
    left = m.parse_manifest(m.build_template(ENTRIES, FOLDER).encode(), FOLDER / "t.csv")
    right = m.parse_manifest(m.build_template(ENTRIES, FOLDER, "json").encode(), FOLDER / "t.json")
    assert left.defaults == right.defaults
    assert [(r.match_key, dict(r.cells)) for r in left.rows] == [(r.match_key, dict(r.cells)) for r in right.rows]


def test_json_form_uses_numbers_booleans_and_extras() -> None:
    document = json.loads(m.build_template(ENTRIES, FOLDER, "json"))
    assert document["version"] == 1 and document["defaults"]["compilation"] is False
    assert document["defaults"]["track_total"] == 2
    assert document["files"]["Spring/01 One.wav"]["track"] == 1
    assert document["files"]["Spring/01 One.wav"]["extras"] == {"BPM": "92", "mood": "calm"}


def test_input_outside_the_template_folder_uses_dot_dot() -> None:
    text = m.build_template([entry("../Other/x.wav", title="X")], FOLDER)
    assert rows(text)[-1][0] == "../Other/x.wav"


def test_artwork_reserved_and_unusable_extras_and_clear_text_are_left_out() -> None:
    odd = entry("a.wav", {"NORMA": "x", "bad key": "y", "OK": "z", "EMPTY": " "}, title="[clear]", artwork="x.png")  # type: ignore[arg-type]
    table = rows(m.build_template([odd], FOLDER))
    assert table[0][-1] == "OK" and "NORMA" not in table[0] and "bad key" not in table[0]
    assert dict(zip(table[0], table[-1], strict=True))["title"] == ""


def test_defaults_need_every_input_to_agree() -> None:
    a = entry("a.wav", artist="Me", genre="Indie")
    b = entry("b.wav", artist="Me")
    table = rows(m.build_template([a, b], FOLDER))
    header = table[0]
    assert dict(zip(header, table[1], strict=True))["artist"] == "Me"
    assert dict(zip(header, table[1], strict=True))["genre"] == ""
    assert dict(zip(header, table[2], strict=True))["genre"] == "Indie"


def test_false_compilation_is_a_value_not_an_absence() -> None:
    table = rows(m.build_template([entry("a.wav", compilation=False)], FOLDER))
    assert dict(zip(table[0], table[1], strict=True))["compilation"] == "false"


def test_no_defaults_row_when_nothing_is_shared() -> None:
    table = rows(m.build_template([entry("a.wav", title="A"), entry("b.wav", title="B")], FOLDER))
    assert [r[0] for r in table[1:]] == ["a.wav", "b.wav"]


def test_unknown_form_and_empty_input_are_refused() -> None:
    import pytest

    with pytest.raises(ValueError):
        m.build_template(ENTRIES, FOLDER, "xml")
    with pytest.raises(ValueError):
        m.build_template([], FOLDER)
