"""TST-M-04: real CLI manifest parsing, matching, limits and template reload."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
from pathlib import Path

import pytest

from tests.e2e.test_metadata_support import details, document, ffmpeg_trap, plan, tag_wav


@pytest.mark.parametrize("variant", ["csv", "csv-semicolon", "json", "csv-extra", "json-extra",
                                     "csv-typo", "json-typo", "csv-clear", "json-clear", "csv-blank", "json-blank"])
def test_m04_f17_valid_forms_clear_blank_and_extras(norma, tools, fixture_path, fixture_module,
                                                   tmp_path, monkeypatch, variant):
    source = tools.tone(tmp_path / "space name.wav", seconds=3)
    tag_wav(source, title="Source")
    manifest = fixture_path("F-17", variant)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source, "--manifest", manifest)
    for _, _, body in details(run, "PLAN"):
        values = {key: value["value"] for key, value in body["resolved"].items()}
        assert values["artist"] == "Test Artist" and values["date"] == "2026-10-05"
        assert values["track"] == 3 and values["track_total"] == 12
        assert values["comment"] == fixture_module.COMMENT
        assert values.get("title") == (None if variant.endswith("clear") else "Source" if variant.endswith("blank") else fixture_module.TITLE)
        assert body["resolved"]["artist"]["provenance"] == "manifest_default"
    tags = [r for _, _, r in details(run, "TAG")]
    if variant.endswith("extra"):
        assert any(r["key"] == "extra:MOOD" and r["effective"] == "hopeful" for r in tags)
    if variant.endswith("typo"):
        assert any(r["key"] == "extra:titel" and r["effective"] == "alternate" for r in tags)
        if variant == "csv-typo":
            assert "did you mean" in run.stderr and "titel" in run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("variant,code,fragment", [
    ("csv-bad-date", 2, "date"), ("json-bad-date", 2, "date"),  # fixture mutates defaults
    ("csv-missing-file", 2, "file"), ("json-version", 2, "version"),
    ("csv-bad-extra", 2, "bad key"), ("json-bad-extra", 1, "bad key"),
    ("csv-bad-track", 1, "track"), ("json-bad-track", 1, "track"),
    ("csv-duplicate", 1, "match"), ("json-duplicate", 2, "duplicate"),
])
def test_m04_f17_whole_versus_matched_row_errors(norma, tools, fixture_path, tmp_path,
                                               monkeypatch, variant, code, fragment):
    source = tools.tone(tmp_path / "space name.wav", seconds=3)
    manifest = fixture_path("F-17", variant)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--manifest", manifest)
    assert run.code == code and fragment in run.stderr + run.stdout
    if code == 2:
        assert run.stdout == ""
    else:
        assert run.files[0][0] == "FAIL" and "manifest_row_invalid" in run.stdout
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("variant", ["csv-ambiguous", "json-ambiguous"])
def test_m04_ambiguous_f15_row_warns_every_candidate_without_applying(norma, tools, fixture_path,
                                                                   tmp_path, monkeypatch, variant):
    tree = fixture_path("F-15")
    manifest = fixture_path("F-17", variant)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", tree / "a/Song.wav", tree / "b/song.WAV", "--config", "/dev/null",
                "--manifest", manifest, "-v")
    assert run.code == 0 and len(run.files) == 2
    assert all(row[0] == "WARN" and "manifest_row_ambiguous" in row[-1] for row in run.files)
    assert all(body["resolved"]["title"]["value"] == "shared" for _, _, body in details(run, "PLAN"))
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


WHOLE = [
    ("csv", b"file\ttitle\na.wav\tx\n", "comma or semicolon"),
    ("csv", b"file,title\na.wav,ab\"cd\n", "quote"),
    ("csv", b"file,title\na.wav, \"abc\"\n", "quote"),
    ("csv", b"file,title\na.wav,\"ab\"cd\n", "quote"),
    ("csv", b"file,title\na.wav,\"abc", "quote"),
    ("csv", b"file,Title,title\na.wav,x,y\n", "same"),
    ("csv", b"file,\na.wav,x\n", "name"),
    ("csv", b"file,NORMA\na.wav,x\n", "reserved"),
    ("csv", b"file,title\n*,one\n*,two\n", "default"),
    ("csv", b"file,title\n/absolute.wav,x\n", "relative"),
    ("csv", b"file,title\n~/a.wav,x\n", "relative"),
    ("csv", b"file,title\nfolder/,x\n", "file"),
    ("csv", b"file,title\nsub/..,x\n", "file"),
    ("csv", b"file,title\na\x00.wav,x\n", "file"),
    ("csv", b"file,title\na.wav,\xff\n", "UTF-8"),
    ("json", b'{"version":1,"version":1,"files":{"a.wav":{}}}', "duplicate"),
    ("json", b'{"version":1,"files":{"a.wav":{"track":NaN}}}', "NaN"),
    ("json", b'{"version":1,"files":{"a.wav":{"track":Infinity}}}', "Infinity"),
    ("json", b'{"version":true,"files":{"a.wav":{}}}', "version"),
    ("json", b'{"version":1.0,"files":{"a.wav":{}}}', "version"),
    ("json", b'{"version":1,"files":[]}', "files"),
    ("json", b'{"version":1,"defaults":{"date":"2026-02-30"},"files":{"a.wav":{}}}', "date"),
]


@pytest.mark.parametrize("suffix,content,fragment", WHOLE, ids=[f"MS-whole-{i:02}" for i in range(len(WHOLE))])
def test_m04_ms6_ms7_ms11_ms12_whole_errors_write_nothing(norma, tools, tmp_path, monkeypatch,
                                                        suffix, content, fragment):
    source = tools.tone(tmp_path / "a.wav", seconds=3)
    manifest = tmp_path / ("invalid." + suffix)
    manifest.write_bytes(content)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--manifest", manifest,
                "--report", tmp_path / "never.json")
    assert run.code == 2 and run.stdout == "" and fragment.casefold() in run.stderr.casefold()
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


ROW_VALUES = [
    ('"track":3.0', "track"), ('"track":3e0', "track"), ('"track":true', "track"),
    ('"track":"3"', "track"), ('"date":"2026-02-30"', "date"), ('"date":"10/12/2026"', "date"),
    ('"track":13,"track_total":12', "track"), ('"compilation":"false"', "compilation"),
    ('"title":""', "title"), ('"title":"bad\\u0001"', "title"),
    ('"extras":{"NORMA":"x"}', "NORMA"), ('"titel":"x"', "titel"),
]


@pytest.mark.parametrize("content,field", ROW_VALUES)
def test_m04_matched_bad_row_fails_only_that_file_with_escaped_locator(norma, tools, tmp_path,
                                                                     monkeypatch, content, field):
    first = tools.tone(tmp_path / "a" / "one.wav", seconds=3)
    other = tools.tone(tmp_path / "good.wav", seconds=3)
    manifest = tmp_path / "rows.json"
    manifest.write_text('{"version":1,"files":{"a/one.wav":{' + content + '},"good.wav":{"title":"Good"}}}')
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    run = norma("tags", "plan", first, other, "--config", "/dev/null", "--manifest", manifest, "-v")
    assert run.code == 1 and len(run.files) == 2
    assert run.row(first.name)[0] == "FAIL" and "manifest_row_invalid" in run.stdout
    assert run.row(other.name)[0] == "OK"
    assert field in run.stderr and "a~1one.wav" in run.stderr
    assert calls.read_text() == ""


@pytest.mark.parametrize("ending", ["\n", "\r", "\r\n"])
@pytest.mark.parametrize("delimiter,bom", [(";", True), (",", False)])
def test_m04_csv_physical_lines_and_normalised_text(norma, tools, tmp_path, ending, delimiter, bom):
    source = tools.tone(tmp_path / "a.wav", seconds=3)
    stream = io.StringIO(newline="")
    csv.writer(stream, delimiter=delimiter, lineterminator=ending).writerows([
        ["file", "title", "comment", "track", "compilation", "MOOD"],
        [source.name, "  Cafe\u0301  ", 'line one\r\nline two; comma, quote "', "03.00", "no", "bright"],
    ])
    manifest = tmp_path / "dialect.csv"
    manifest.write_bytes((b"\xef\xbb\xbf" if bom else b"") + stream.getvalue().encode())
    run = plan(norma, source, "--manifest", manifest)
    for _, _, body in details(run, "PLAN"):
        resolved = body["resolved"]
        assert resolved["title"]["value"] == "Café" and resolved["track"]["value"] == 3
        assert resolved["compilation"]["value"] is False
        assert resolved["comment"]["value"] == 'line one\nline two; comma, quote "'
        assert resolved["title"]["locator"].endswith(":2")


@pytest.mark.parametrize("boundary", ["bytes", "rows", "columns"])
@pytest.mark.parametrize("over", [False, True])
def test_m04_manifest_size_boundaries(norma, tools, tmp_path, monkeypatch, boundary, over):
    source = tools.tone(tmp_path / "a.wav", seconds=3)
    manifest = tmp_path / "limit.csv"
    if boundary == "bytes":
        manifest = manifest.with_suffix(".json")
        prefix = b'{"version":1,"files":{"a.wav":{"title":"Limit"}}}'
        content = prefix + b" " * (10485760 + int(over) - len(prefix))
    elif boundary == "rows":
        content = ("file,title\na.wav,Limit\n" + "".join(f"missing-{i}.wav,\n" for i in range(9999 + int(over)))).encode()
    else:
        content = ("file," + ",".join(f"EXTRA{i}" for i in range(199 + int(over))) + "\na.wav\n").encode()
    manifest.write_bytes(content)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--manifest", manifest)
    assert run.code == (2 if over else 0), run.stderr
    assert (run.stdout == "") == over
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("selectors,names,matched,status", [
    (["sub/../a.wav"], ["a.wav"], True, "OK"),  # X1, non-existent sub is never opened
    (["a.wav", "./a.wav"], ["a.wav"], False, "FAIL"),  # X2
    (["Spring/./Song.wav"], ["Spring/Song.wav"], True, "OK"),  # X6
    (["./Song.wav"], ["Spring/Song.wav"], False, "OK"),  # path has no basename fallback
    (["Song.wav"], ["Spring/Song.wav", "Autumn/song.WAV"], False, "WARN"),  # X4
    (["Song.wav"], ["Song.wav", "Autumn/song.WAV"], True, "OK"),  # local wins
    (["PERCHÉ.WAV"], ["Perche\u0301.wav"], True, "OK"),
    (["Perche.wav"], ["Perché.wav"], False, "OK"),  # accents are not stripped
])
def test_m04_ms13_lexical_matching(norma, tools, tmp_path, monkeypatch, selectors, names, matched, status):
    sources = [tools.tone(tmp_path / name, seconds=3) for name in names]
    manifest = tmp_path / "match.csv"
    manifest.write_text("file,title\n" + "".join(f"{selector},Manifest\n" for selector in selectors))
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", *sources, "--config", "/dev/null", "--manifest", manifest, "-v")
    assert run.code == int(status == "FAIL") and run.files[0][0] == status
    if status != "FAIL":
        first_plans = [body for name, _, body in details(run, "PLAN") if Path(name) == sources[0].resolve()]
        assert first_plans
        assert all((body["resolved"]["title"]["value"] == "Manifest") == matched for body in first_plans)
    if not matched and status == "OK":
        assert "match" in run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


def test_m04_out_of_folder_and_symlink_keys_are_lexical(norma, tools, tmp_path, monkeypatch):
    source = tools.tone(tmp_path / "Other/x.wav", seconds=3)
    folder = tmp_path / "Manifests"
    folder.mkdir()
    alias = folder / "alias.wav"
    alias.symlink_to(source)
    manifest = folder / "matching.json"
    manifest.write_text(json.dumps({"version": 1, "files": {"../Other/x.wav": {"artist": "Matched"},
                                                            "./alias.wav": {"title": "Never opened"}}}))
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    run = plan(norma, alias, "--manifest", manifest)
    for _, _, body in details(run, "PLAN"):
        assert body["resolved"]["artist"]["value"] == "Matched"
        assert body["resolved"]["title"]["value"] == "x"
    assert "alias.wav" in run.stderr and "match" in run.stderr and calls.read_text() == ""


@pytest.mark.parametrize("form", ["csv", "json", "stdout"])
def test_m04_tags_template_reloads_without_ffmpeg(norma, tools, tmp_path, monkeypatch, form):
    source = tools.tone(tmp_path / "demos/03 Perché.wav", seconds=3)
    tag_wav(source, title="Perché", artist="Test Artist", track="3/12", comment='line one\nline two, "quote"')
    folder = tmp_path / "manifests"
    folder.mkdir()
    destination = "-" if form == "stdout" else folder / ("template." + form)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "template", source, "--config", "/dev/null", "-o", destination)
    assert run.code == 0 and calls.read_text() == ""
    if form == "stdout":
        assert tools.snapshot(tmp_path) == before and run.stdout.startswith("file,")
        destination = tmp_path / "stdout.csv"
        destination.write_text(run.stdout)
        rows = list(csv.DictReader(io.StringIO(run.stdout)))
        assert [r["file"] for r in rows if r["file"] != "*"] == ["demos/03 Perché.wav"]
    else:
        assert tools.difference(before, tools.snapshot(tmp_path)) == {
            "added": {str(destination.relative_to(tmp_path))}, "removed": set(), "changed": set()}
        if form == "json":
            assert set(document(destination)["files"]) == {"../demos/03 Perché.wav"}
        else:
            rows = list(csv.DictReader(io.StringIO(destination.read_text())))
            assert [r["file"] for r in rows if r["file"] != "*"] == ["../demos/03 Perché.wav"]
    reload = plan(norma, source, "--manifest", destination)
    for _, _, body in details(reload, "PLAN"):
        assert body["resolved"]["title"]["value"] == "Perché"
        assert body["resolved"]["title"]["provenance"] == "manifest"
        assert body["resolved"]["track"]["value"] == 3 and body["resolved"]["track_total"]["value"] == 12
    assert calls.read_text() == ""


def test_m04_json_extra_typo_has_the_required_run_warning(norma, tools, tmp_path, monkeypatch):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    manifest = tmp_path / "typo.json"
    manifest.write_text(json.dumps({"version": 1, "files": {source.name: {"extras": {"titel": "alternate"}}}}))
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source, "--manifest", manifest)
    assert any(row["key"] == "extra:titel" and row["effective"] == "alternate"
               for _, _, row in details(run, "TAG"))
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before
    assert "did you mean" in run.stderr and "titel" in run.stderr


def test_m04_issue_cap_keeps_first_fifty_and_an_exact_remainder(norma, tools, tmp_path, monkeypatch):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    manifest = tmp_path / "many.json"
    manifest.write_text(json.dumps({"version": 1, "files": {source.name: {}}, "defaults": {"extras": {
        f"EXTRA{number}": "\u0001" for number in range(57)}}}))
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--manifest", manifest)
    assert run.code == 2 and run.stdout == ""
    notes = run.stderr.splitlines()
    number_issues = [line for line in notes if "EXTRA" in line]
    assert len(number_issues) == 50, run.stderr
    assert any("7" in line and "more issues" in line for line in notes), run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before
