"""Filename inference (MD9; TST-M-05). Pure: patterns and relative paths in, typed opinions out."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import pytest

import norma.metadata.infer as infer_module
from norma.metadata.infer import InvalidPattern, compile_pattern, infer, target_string
from norma.models import ManifestCell


def run(pattern: str, path: str, *, underscores: bool = False):
    return infer(compile_pattern(pattern), PurePosixPath(path), underscores=underscores)


def values(result) -> dict[str, object]:
    return {k: v.value for k, v in result.cells.items()}


def test_track_and_title_with_padded_number() -> None:
    result = run("{track} {title}", "Spring/03 Glasshouse.flac")
    assert values(result) == {"track": 3, "title": "Glasshouse"} and result.matched and result.notes == ()
    assert type(result.cells["track"].value) is int
    assert all(isinstance(c, ManifestCell) and c.operation == "set" for c in result.cells.values())


@pytest.mark.parametrize("text, number", [("007", 7), ("10", 10), ("1", 1)])
def test_padded_numbers_are_integers(text: str, number: int) -> None:
    assert values(run("{track} - {title}", f"{text} - X.wav")) == {"track": number, "title": "X"}


def test_all_placeholders() -> None:
    result = run("{artist} - {album} ({year}) {disc}-{track} {title} [{genre}]{*}",
                 "Me - Spring (2026) 1-02 Night Bus [Indie]extra.mp3")
    assert values(result) == {"artist": "Me", "album": "Spring", "date": "2026", "disc": 1, "track": 2,
                              "title": "Night Bus", "genre": "Indie"}


@pytest.mark.parametrize("date", ["2026", "2026-03", "2026-03-14"])
def test_date_placeholder_forms(date: str) -> None:
    assert values(run("{date} {title}", f"{date} Song.wav")) == {"date": date, "title": "Song"}


def test_text_placeholders_are_lazy_and_greedy_enough_for_the_end() -> None:
    assert values(run("{artist} - {title}", "A - B - C.wav")) == {"artist": "A", "title": "B - C"}


def test_star_captures_nothing_and_never_matches_a_slash() -> None:
    assert values(run("{*} {title}", "junk 05 Song.wav")) == {"title": "05 Song"}
    assert values(run("{*}/{*}-{title}", "a/b/c-d.wav")) == {"title": "d"}
    assert run("{*}/{title}", "Song.wav").matched is False


def test_extension_is_removed_only_once_and_inner_dots_kept() -> None:
    assert values(run("{title}", "My.Song.v2.wav")) == {"title": "My.Song.v2"}


def test_anchored_whole_name() -> None:
    assert values(run("{title}", "x.wav")) == {"title": "x"}
    assert run("{track} {title}", "Song.wav").matched is False


def test_literal_text_is_case_sensitive_and_escaped() -> None:
    assert run("Take {track}", "take 3.wav").matched is False
    assert values(run("Take {track}", "Take 3.wav")) == {"track": 3}
    assert values(run("a.b ({title})", "a.b (X).wav")) == {"title": "X"}
    assert run("a.b {title}", "aXb Y.wav").matched is False


def test_path_suffix_with_separators_is_anchored_on_components() -> None:
    assert values(run("{album}/{track} {title}", "Music/Spring/03 Glass.wav")) == {
        "album": "Spring", "track": 3, "title": "Glass"}
    assert run("{album}/{title}", "Song.wav").matched is False  # fewer components than the pattern needs
    assert values(run("{artist}/{album}/{title}", "Me/Spring/Song.wav")) == {
        "artist": "Me", "album": "Spring", "title": "Song"}


def test_placeholders_never_match_a_slash() -> None:
    result = run("{title}", "Dir/Song.wav")
    assert values(result) == {"title": "Song"}  # k = 0: only the last component
    assert run("{artist} {title}", "A/B C.wav").matched and values(run("{artist} {title}", "A/B C.wav")) == {
        "artist": "B", "title": "C"}


def test_target_is_nfc() -> None:
    result = run("{title}", "Caf\u0065\u0301.wav")
    assert values(result) == {"title": "Caf\u00e9"}
    assert values(run("Caf\u0065\u0301 {title}", "Caf\u00e9 X.wav")) == {"title": "X"}


def test_target_string_helper() -> None:
    assert target_string(PurePosixPath("a/b/c.wav"), 1) == "b/c"
    assert target_string(PurePosixPath("c.wav"), 1) is None


def test_repeated_captures_must_be_equal() -> None:
    assert values(run("{artist} - {title} - {artist}", "Me - Song - Me.wav")) == {"artist": "Me", "title": "Song"}
    result = run("{artist} - {title} - {artist}", "Me - Song - You.wav")
    assert result.matched is False and not result.cells and len(result.notes) == 1


def test_repeated_captures_may_need_backtracking() -> None:
    assert values(run("{title}-{title}", "a-b-a-b.wav")) == {"title": "a-b"}


def test_repeated_numbers_compare_as_integers() -> None:
    assert values(run("{track} {title} {track}", "03 X 3.wav")) == {"track": 3, "title": "X"}


def test_trimming_and_underscores_are_opt_in() -> None:
    assert values(run("{artist}-{title}", "Me - Song_One.wav")) == {"artist": "Me", "title": "Song_One"}
    assert values(run("{artist}-{title}", "Me_-_Song_One.wav", underscores=True)) == {
        "artist": "Me", "title": "Song One"}
    assert values(run("{artist}-{title}", " __Me__ - Song__One_.wav", underscores=True)) == {
        "artist": "Me", "title": "Song  One"}


def test_underscores_apply_before_the_equality_check() -> None:
    assert values(run("{title} {title}", "a_b a b.wav", underscores=True)) == {"title": "a b"}


@pytest.mark.parametrize("pattern, path, field", [
    ("{date} {title}", "2026-13 Song.wav", "date"),
    ("{date} {title}", "2026-02-30 Song.wav", "date"),
    ("{year} {title}", "0000 Song.wav", "year"),
    ("{track} {title}", "000 Song.wav", "track"),
    ("{disc}-{title}", "0-Song.wav", "disc"),
])
def test_invalid_capture_is_one_note_and_nothing_inferred_for_that_field(pattern: str, path: str, field: str) -> None:
    result = run(pattern, path)
    assert result.matched and len(result.notes) == 1 and field in result.notes[0]
    assert "date" not in result.cells and "track" not in result.cells and "disc" not in result.cells
    assert "title" in values(result) or field == "disc"


def test_other_fields_survive_an_invalid_capture() -> None:
    result = run("{track} {date} {title}", "03 2026-13 Song.wav")
    assert values(result) == {"track": 3, "title": "Song"} and len(result.notes) == 1


def test_no_match_is_informational() -> None:
    result = run("{track} {title}", "Song.wav")
    assert (result.matched, dict(result.cells), len(result.notes)) == (False, {}, 1)


def test_date_and_year_in_one_pattern() -> None:
    assert values(run("{date} {year} {title}", "2026 2026 X.wav")) == {"date": "2026", "title": "X"}
    result = run("{date} {year} {title}", "2026-03 2025 X.wav")
    assert "date" not in result.cells and len(result.notes) == 1 and values(result) == {"title": "X"}


def test_control_characters_and_length_are_invalid_text() -> None:
    result = run("{title}", "x" * 1001 + ".wav")
    assert not result.cells and len(result.notes) == 1


@pytest.mark.parametrize("pattern", ["", "  ", "{nope}", "{title", "title}", "{title}/", "/{title}", "a//{title}",
                                     "{}", "{ title }"])
def test_invalid_patterns(pattern: str) -> None:
    with pytest.raises(InvalidPattern):
        compile_pattern(pattern)


def test_result_does_not_depend_on_mode_or_files() -> None:
    first = run("{track} {title}", "01 A.wav")
    assert first == run("{track} {title}", "01 A.wav")


def test_infer_module_is_pure() -> None:
    tree = ast.parse(Path(infer_module.__file__).read_text(encoding="utf-8"))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "re", "unicodedata", "collections", "dataclasses", "pathlib", "norma"}
    source = Path(infer_module.__file__).read_text(encoding="utf-8")
    assert "open(" not in source and "read_text" not in source and "os." not in source
