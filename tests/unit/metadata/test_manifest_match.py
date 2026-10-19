"""Manifest matching (MS5, MS13; TST-M-04, X1–X6)."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path, PurePosixPath

import pytest

from norma.metadata import manifest as m
from norma.models import InputItem

FOLDER = Path("/m/Demos")


def inputs(*names: str) -> list[InputItem]:
    """Inputs given relative to /m/Demos (``../`` allowed)."""
    items = []
    for index, name in enumerate(names):
        source = Path(__import__("os").path.normpath(FOLDER / name))
        root = source.parent
        items.append(InputItem(source, root, PurePosixPath(source.name), (1, index), index, False))
    return items


def manifest(*selectors: str, form: str = "json"):
    if form == "json":
        raw = json.dumps({"version": 1, "files": {s: {"title": f"t{n}"} for n, s in enumerate(selectors)}})
        return m.parse_manifest(raw.encode(), FOLDER / "demos.json")
    raw = "file,title\n" + "".join(f"{s},t{n}\n" for n, s in enumerate(selectors))
    return m.parse_manifest(raw.encode(), FOLDER / "demos.csv")


def run(selectors: list[str], names: list[str], form: str = "json"):
    data = manifest(*selectors, form=form)
    return data, m.match_manifest(data, inputs(*names))


def matched(result: m.ManifestMatching, index: int) -> str | None:
    row = result.inputs[index].row
    return row.match_key if row else None


def test_x1_dot_dot_segment_is_normalised() -> None:
    _, result = run(["sub/../a.wav"], ["a.wav"])
    assert matched(result, 0) == "sub/../a.wav"


def test_x2_dot_segment_equivalents_are_duplicates() -> None:
    _, result = run(["a.wav", "./a.wav"], ["a.wav"], form="csv")
    match = result.inputs[0]
    assert match.row is None
    assert [(i.detail, i.locator, i.affected_indices) for i in match.failures] == \
           [("duplicate_match", "demos.csv:2, demos.csv:3", (0,))]


def test_x3_input_outside_the_folder() -> None:
    _, result = run(["../Other/x.wav"], ["../Other/x.wav"])
    assert matched(result, 0) == "../Other/x.wav" and not result.unmatched


def test_x4_bare_name_without_local_candidate_is_ambiguous_by_base_name() -> None:
    _, result = run(["Song.wav"], ["Spring/Song.wav", "Autumn/song.WAV"], form="csv")
    for index in (0, 1):
        match = result.inputs[index]
        assert match.row is None and match.failures == ()
        assert [(i.detail, i.affected_indices) for i in match.warnings] == [("ambiguous_match", (0, 1))]
    assert result.unmatched == ()


def test_x5_two_local_candidates_are_ambiguous() -> None:
    _, result = run(["song.wav"], ["Song.wav", "song.wav"])
    assert all(r.row is None and len(r.warnings) == 1 for r in result.inputs.values())


def test_x6_path_selector_does_not_broaden() -> None:
    _, result = run(["Spring/./Song.wav"], ["Spring/Song.wav"])
    assert matched(result, 0) == "Spring/./Song.wav"
    _, result = run(["./Song.wav"], ["Spring/Song.wav"])
    assert matched(result, 0) is None and [i.detail for i in result.unmatched] == ["unmatched_row"]


def test_bare_name_prefers_the_manifest_folder() -> None:
    _, result = run(["song.wav"], ["sub/song.wav", "Song.wav"])
    assert matched(result, 1) == "song.wav" and matched(result, 0) is None
    assert result.unmatched == ()


def test_bare_name_falls_back_to_a_single_base_name() -> None:
    _, result = run(["song.wav"], ["sub/Song.wav", "other/x.wav"])
    assert matched(result, 0) == "song.wav" and matched(result, 1) is None


def test_path_selector_never_falls_back_to_base_names() -> None:
    _, result = run(["x/song.wav"], ["sub/song.wav"])
    assert matched(result, 0) is None and len(result.unmatched) == 1


def test_unmatched_rows_are_run_warnings_with_locator() -> None:
    _, result = run(["a.wav", "gone.wav"], ["a.wav"], form="csv")
    assert [(i.locator, i.detail) for i in result.unmatched] == [("demos.csv:3", "unmatched_row")]


def test_nfc_and_case_folding_are_a_matching_policy() -> None:
    decomposed = unicodedata.normalize("NFD", "Perché no.aif")
    _, result = run(["PERCH\u00c9 NO.AIF"], [decomposed])
    assert matched(result, 0) == "PERCH\u00c9 NO.AIF"
    _, result = run(["Straße.wav"], ["STRASSE.wav"])
    assert matched(result, 0) == "Straße.wav"


def test_comparison_key_matches_ms13() -> None:
    row = manifest("Sub/../Perch\u00e9.wav").rows[0]
    assert row.lexical_key == PurePosixPath("Perch\u00e9.wav")
    assert row.comparison_key == unicodedata.normalize("NFC", "perch\u00e9.wav")


def test_invalid_row_fails_only_its_matched_input() -> None:
    raw = json.dumps({"version": 1, "files": {"a.wav": {"track": 0}, "b.wav": {"title": "ok"}}})
    data = m.parse_manifest(raw.encode(), FOLDER / "demos.json")
    result = m.match_manifest(data, inputs("a.wav", "b.wav", "c.wav"))
    assert [i.detail for i in result.inputs[0].failures] == ["number_invalid"]
    assert result.inputs[0].row is None
    assert result.inputs[1].failures == () and result.inputs[1].row is not None
    assert result.inputs[2] == m.InputMatch(2, None)


def test_invalid_row_that_matches_nothing_is_only_a_warning() -> None:
    raw = json.dumps({"version": 1, "files": {"gone.wav": {"track": 0}}})
    data = m.parse_manifest(raw.encode(), FOLDER / "demos.json")
    result = m.match_manifest(data, inputs("a.wav"))
    assert result.inputs[0].failures == () and len(result.unmatched) == 1


def test_ambiguous_invalid_row_is_not_applied() -> None:
    raw = json.dumps({"version": 1, "files": {"song.wav": {"track": 0}}})
    data = m.parse_manifest(raw.encode(), FOLDER / "demos.json")
    result = m.match_manifest(data, inputs("a/song.wav", "b/song.wav"))
    assert all(r.failures == () and len(r.warnings) == 1 for r in result.inputs.values())


def test_duplicate_through_mixed_path_and_base_name() -> None:
    _, result = run(["a.wav", "sub/../a.wav", "sub/b.wav", "b.wav"], ["a.wav", "sub/b.wav"], form="csv")
    assert [i.locator for i in result.inputs[0].failures] == ["demos.csv:2, demos.csv:3"]
    # the path selector and the bare name (no local b.wav, so the base name applies) reach the same input
    assert [i.locator for i in result.inputs[1].failures] == ["demos.csv:4, demos.csv:5"]
    assert result.inputs[0].row is None and result.inputs[1].row is None


def test_ambiguous_row_does_not_count_towards_duplicates() -> None:
    _, result = run(["song.wav", "x/song.wav"], ["x/song.wav", "y/song.wav"], form="csv")
    assert matched(result, 0) == "x/song.wav" and result.inputs[0].failures == ()
    assert len(result.inputs[0].warnings) == 1 and len(result.inputs[1].warnings) == 1


def test_defaults_apply_regardless_of_matching() -> None:
    data = m.parse_manifest(b"file,artist\n*,Me\ngone.wav,X\n", FOLDER / "demos.csv")
    result = m.match_manifest(data, inputs("a.wav"))
    assert data.defaults["artist"].value == "Me" and result.inputs[0].row is None


def test_matching_never_touches_the_filesystem(tmp_path: Path) -> None:
    # inputs that do not exist on disk still match: matching is lexical (MS13)
    _, result = run(["not/on/disk.wav"], ["not/on/disk.wav"])
    assert matched(result, 0) == "not/on/disk.wav"


@pytest.mark.parametrize("selector, expected", [
    ("a.wav", True), ("./a.wav", True), ("sub/../a.wav", True), ("sub//../a.wav", True),
])
def test_selector_forms_against_a_local_input(selector: str, expected: bool) -> None:
    _, result = run([selector], ["a.wav"])
    assert (matched(result, 0) is not None) is expected


def test_manifest_relative_paths_for_templates() -> None:
    assert m.manifest_relative(Path("/m/Demos/Spring/a.wav"), FOLDER) == "Spring/a.wav"
    assert m.manifest_relative(Path("/m/Other/x.wav"), FOLDER) == "../Other/x.wav"
    assert m.manifest_relative(Path("/m/Demos/Cafe\u0301.wav"), FOLDER) == "Caf\u00e9.wav"
