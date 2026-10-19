"""TST-M-03/M-05: normative resolution tables and anchored filename inference."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.test_metadata_support import (
    assert_accounted, assert_no_render, branch_rows, details, ffmpeg_trap,
    normalise, plan, readback, tag_wav,
)


# MD14's first matching row, including absence rather than a fabricated empty value.
R_CASES = [
    ("R1", "source", {"title": "Source"}, ["--clear-tag", "title", "--infer-from-filename", "{track} {title}"], None, "cli"),
    ("R2", "explicit", {"title": "Source"}, ["--title", "Explicit"], "Explicit", "cli"),
    ("R3", "explicit", {"title": "Source"}, ["--infer-from-filename", "{track} {title}", "--infer-mode", "override"], "Glasshouse", "filename"),
    ("R4", "explicit", {"title": "Source"}, [], "Source", "source"),
    ("R5", "explicit", {}, ["--infer-from-filename", "{track} {title}"], "Glasshouse", "filename"),
    ("R6", "source", {"title": "Source"}, ["--title", "Explicit"], "Source", "source"),
    ("R7", "source", {}, ["--title", "Explicit"], "Explicit", "cli"),
    ("R8", "source", {}, ["--infer-from-filename", "{track} {title}"], "Glasshouse", "filename"),
    ("R9", "explicit", {}, [], "03 Glasshouse", "fallback"),
    ("R10", "explicit", {}, ["--no-title-fallback"], None, None),
]


@pytest.mark.parametrize("case,prefer,source_fields,options,value,provenance", R_CASES,
                         ids=[r[0] for r in R_CASES])
def test_m03_r1_to_r10(norma, tools, tmp_path, monkeypatch, case, prefer, source_fields, options, value, provenance):
    source = tools.tone(tmp_path / "03 Glasshouse.wav", seconds=3)
    if source_fields:
        tag_wav(source, **source_fields)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source, "--prefer", prefer, *options)
    assert_accounted(run, source)
    for _, branch, body in details(run, "PLAN"):
        resolved = body["resolved"].get("title")
        assert (resolved["value"] if resolved else None) == value
        if provenance is not None and value is not None:
            assert resolved["provenance"] == provenance
        fate, = [row for row in branch_rows(run, "TAG", branch) if row["key"] == "field:title"] if case != "R10" else [None]
        if case == "R1":
            assert fate["status"] == "removed_by_request" and fate["effective"] is None
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


W_CASES = [
    ("W1", {"artist": "Dave G"}, {"artist": "Test Artist"}, {}, {}, [], "artist", "Test Artist", "manifest_default", ["artist"]),
    ("W2", {"artist": "Dave G"}, {"artist": "Test Artist"}, {}, {}, ["--prefer", "source"], "artist", "Dave G", "source", ["artist"]),
    ("W3", {"comment": "old note"}, {}, {"comment": None}, {}, ["--prefer", "source"], "comment", None, "manifest", []),
    ("W4", {}, {}, {}, {}, ["--infer-from-filename", "{track} {title}"], "title", "Glasshouse", "filename", []),
    ("W5", {"title": "Bounce 7"}, {}, {}, {}, ["--infer-from-filename", "{track} {title}"], "title", "Bounce 7", "source", []),
    ("W6", {"title": "Bounce 7"}, {}, {}, {}, ["--infer-from-filename", "{track} {title}", "--infer-mode", "override"], "title", "Glasshouse", "filename", ["title"]),
    ("W7", {}, {"comment": "Demos 2026"}, {"comment": None}, {}, ["--comment", "Mastered"], "comment", "Mastered", "cli", []),
    ("W8", {"track": "3/12"}, {}, {"track": 13}, {}, [], "track", 13, "manifest", ["track", "track_total"]),
    ("W9", {"artist": "Dave G"}, {}, {"artist": "Anna Rossi"}, {"artist": "Test Artist"}, [], "artist", "Anna Rossi", "manifest", ["artist"]),
    ("W10", {"title": "Bounce 7", "REPLAYGAIN_TRACK_GAIN": "-3 dB"}, {}, {}, {}, ["--no-source-tags"], "title", "03 Glasshouse", "fallback", []),
    ("W12", {"artist": "Anna Rossi"}, {}, {"artist": "Anna Rossi"}, {"artist": "Test Artist"}, ["--strict-conflicts"], "artist", "Anna Rossi", "manifest", []),
    ("W13", {"artist": "Anna Rossi"}, {}, {}, {"artist": "Test Artist"}, [], "artist", "Test Artist", "config", ["artist"]),
    ("W14", {"artist": "Anna Rossi"}, {}, {}, {"artist": "Test Artist"}, ["--artist", "Anna Rossi"], "artist", "Anna Rossi", "cli", []),
]


@pytest.mark.parametrize("case,source_fields,defaults,row_fields,config_fields,options,field,value,provenance,conflicts",
                         W_CASES, ids=[r[0] for r in W_CASES])
def test_m03_worked_examples_w1_to_w14(norma, tools, tmp_path, case, source_fields, defaults,
                                      row_fields, config_fields, options, field, value, provenance, conflicts):
    source = tools.tone(tmp_path / "03 Glasshouse.wav", seconds=3)
    if source_fields:
        tag_wav(source, **source_fields)
    options = list(options)
    if defaults or row_fields:
        manifest = tmp_path / "opinions.json"
        manifest.write_text(json.dumps({"version": 1, "defaults": defaults, "files": {source.name: row_fields}}))
        options += ["--manifest", manifest]
    if config_fields:
        config = tmp_path / "prefs.toml"
        config.write_text("\n".join(f"{key}={json.dumps(val)}" for key, val in config_fields.items()))
        options += ["--config", config]
    preview = plan(norma, source, *options)
    assert_accounted(preview, source)
    for _, branch, body in details(preview, "PLAN"):
        resolved = body["resolved"].get(field)
        if value is None:
            assert resolved is None
            cleared, = [r for r in branch_rows(preview, "TAG", branch) if r["key"] == "field:" + field]
            assert cleared["status"] == "removed_by_request"
        else:
            assert resolved["value"] == value and resolved["provenance"] == provenance
        actual = [c["key"].removeprefix("field:") for _, label, c in details(preview, "CONFLICT") if label == branch]
        assert sorted(actual) == sorted(conflicts)
        statuses = {row["key"]: row for row in branch_rows(preview, "TAG", branch)}
        if case == "W8":
            assert statuses["field:track_total"]["effective"] is None
            assert statuses["field:track_total"]["detail"] == "pair_inconsistent"
        if case == "W10":
            assert statuses["item:native:id3v2:TXXX:REPLAYGAIN_TRACK_GAIN"]["status"] == "removed_by_design"
            native_title = next(row for row in branch_rows(preview, "NATIVE_FATE", branch)
                                if "store:id3v2:TIT2" in row["source_ids"])
            assert native_title["status"] == "removed_by_request"
    before = tools.snapshot(source.parent)[source.name]
    _, record, output = normalise(norma, source, tmp_path / "run", "--no-mark-processed", *options)
    assert readback(output)[0].get(field) == value
    if value is not None:
        assert record["tag_sources"][field] == provenance
    assert sorted(c["key"].removeprefix("field:") for c in record["conflicts"]) == sorted(conflicts)
    assert tools.snapshot(source.parent)[source.name] == before
    if case in ("W4", "W8"):
        assert readback(output)[0]["track"] == (3 if case == "W4" else 13)
        assert "track_total" not in readback(output)[0]


@pytest.mark.parametrize("conflict", ["source-explicit", "source-inference", "pair"])
def test_m03_strict_conflicts_fail_before_render(norma, tools, tmp_path, monkeypatch, conflict):
    source = tools.tone(tmp_path / "03 Glasshouse.wav", seconds=3)
    tag_wav(source, artist="Anna Rossi", title="Bounce 7", track="3/12")
    options = {"source-explicit": ["--artist", "Test Artist"],
               "source-inference": ["--infer-from-filename", "{track} {title}", "--infer-mode", "override"],
               "pair": ["--track", "13"]}[conflict]
    calls = ffmpeg_trap(tmp_path, monkeypatch, delegate=True)
    _, row, output = normalise(norma, source, tmp_path / "run", "--strict-conflicts", *options, code=1)
    assert row["reason"] == "metadata_conflict_strict" and row["execution"] == "not_written"
    assert row["published_paths"] == [] and (output is None or not output.exists())
    assert_no_render(calls, source)


INVALID = [
    ["--title", "X", "--clear-tag", "title"],  # W11
    ["--track", "0"], ["--track", "4/3"], ["--disc", "/3"], ["--date", "2026-02-30"],
    ["--artist", "line\nbreak"], ["--comment", "x" * 4001], ["--tag", "NORMA=x"],
    ["--tag", "rEpLaYgAiN_TRACK_GAIN=x"], ["--tag", "ITUNNORM=x"], ["--tag", "ITUNSMPB=x"],
    ["--tag", "ENCODER=x"], ["--tag", "R128_TRACK_GAIN=x"], ["--tag", "title=x"],
    ["--tag", "1bad=x"], ["--tag", "DEMO="], ["--tag", "compilation=1"],
    ["--tag", "DEMO=x", "--clear-tag", "demo"], ["--clear-tag", "artwork"],
    ["--prefer", "source", "--no-source-tags"], ["--infer-mode", "fill"], ["--infer-underscores"],
    ["--infer-from-filename", "literal"], ["--infer-from-filename", "{unknown}"],
    ["--infer-from-filename", "{title"], ["--on-metadata-loss", "bad"], ["--id3-version", "2.5"],
]


@pytest.mark.parametrize("command", [("tags", "plan"), ("normalise",)])
@pytest.mark.parametrize("options", INVALID, ids=[f"U{i:02}" for i in range(len(INVALID))])
def test_m03_usage_errors_precede_all_ffmpeg_and_writes(norma, tools, tmp_path, monkeypatch, command, options):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma(*command, source, "--config", "/dev/null", *options)
    assert run.code == 2 and run.stdout == "" and "norma: error:" in run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("relative,pattern,options,expected", [
    ("Artist/Album/03 - Perché.wav", "{artist}/{album}/{track} - {title}", [],
     dict(artist="Artist", album="Album", track=3, title="Perché")),
    ("Artist/Album/004_Not_a_match.wav", "{track}_{title}", [], dict(track=4, title="Not_a_match")),
    ("Artist/Album/004_Not_a_match.wav", "{track}_{title}", ["--infer-underscores"], dict(track=4, title="Not a match")),
    ("Artist/Album/004_Not_a_match.wav", "{track} - {title}", [], {}),
    ("unicode/Perche\u0301.wav", "{title}", [], dict(title="Perché")),
    ("東京 🎵.wav", "{title}", [], dict(title="東京 🎵")),
])
def test_m05_inference_on_f15_is_anchored_and_typed(norma, tools, fixture_path, tmp_path,
                                                  monkeypatch, relative, pattern, options, expected):
    tree = fixture_path("F-15")
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, tree, "--recursive", "--infer-from-filename", pattern, "--no-title-fallback", *options)
    selected = [body for name, _, body in details(run, "PLAN") if Path(name).samefile(tree / relative)]
    assert selected
    for body in selected:
        inferred = {k: v["value"] for k, v in body["resolved"].items() if v["provenance"] == "filename"}
        assert inferred == expected
        if "track" in inferred:
            assert type(inferred["track"]) is int
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("name,pattern,expected", [
    ("Anna - Anna.wav", "{artist} - {artist}", {"artist": "Anna"}),
    ("Anna - Bob.wav", "{artist} - {artist}", {}),
    ("2026-13 title.wav", "{date} {title}", {"title": "title"}),
    ("2026 Song.wav", "{year} {title}", {"date": "2026", "title": "Song"}),
    ("x/Anna/Song.wav", "{artist}/{title}", {"artist": "Anna", "title": "Song"}),
    ("x/Anna/Song.wav", "{title}", {"title": "Song"}),
])
def test_m05_repeated_captures_invalid_dates_and_path_suffix(norma, tools, tmp_path, name, pattern, expected):
    source = tools.tone(tmp_path / name, seconds=3)
    run = plan(norma, tmp_path, "--recursive", "--infer-from-filename", pattern, "--no-title-fallback")
    for _, _, body in details(run, "PLAN"):
        inferred = {k: v["value"] for k, v in body["resolved"].items() if v["provenance"] == "filename"}
        assert inferred == expected
    if "2026-13" in name:
        assert "date" in run.stderr and run.code == 0


def test_m03_w3_clear_provenance_is_the_winning_explicit_layer(norma, tools, tmp_path):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tag_wav(source, comment="old note")
    manifest = tmp_path / "clear.json"
    manifest.write_text(json.dumps({"version": 1, "files": {source.name: {"comment": None}}}))
    run = plan(norma, source, "--manifest", manifest, "--prefer", "source")
    for _, branch, _ in details(run, "PLAN"):
        cleared, = [row for row in branch_rows(run, "TAG", branch) if row["key"] == "field:comment"]
        assert cleared["effective"] is None and cleared["status"] == "removed_by_request"
        assert cleared["provenance"] == "manifest"
