"""TST-M-06/M-07 and catalogue accounting: artwork and genuine read-only plans."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest
from mutagen.id3 import APIC, PRIV
from mutagen.wave import WAVE

from tests.e2e.test_metadata_support import (
    CATALOGUE_AUDIO, F13, TARGETS, assert_accounted, branch_rows, chunks, details, document,
    ffmpeg_trap, info_items, normalise, plan, readback, tag_wav,
)


def png(width, height):
    def chunk(key, body):
        return struct.pack(">I", len(body)) + key + body + struct.pack(">I", zlib.crc32(key + body))
    scanline = b"\0" + b"\x28\x78\xc8" * width
    # Compress incrementally: even a decode-boundary image is not a huge in-memory array.
    compressor = zlib.compressobj(9)
    data = b"".join(compressor.compress(scanline) for _ in range(height)) + compressor.flush()
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", data) + chunk(b"IEND", b"")


def image_facts(path):
    completed = subprocess.run(["/usr/bin/sips", "-g", "format", "-g", "space", "-g", "pixelWidth",
                                "-g", "pixelHeight", str(path)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    return {key.strip(): val.strip() for line in completed.stdout.splitlines()[1:]
            if ":" in line for key, val in [line.split(":", 1)]}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("variant", [None, "heic", "large", "cmyk"])
def test_m06_f16_artwork_preparation_and_native_bytes(norma, tools, fixture_path, tmp_path, target, variant):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    cover = fixture_path("F-16", variant)
    cover_before = tools.snapshot(cover.parent)
    before = tools.snapshot(tmp_path)
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target, "--artwork", cover,
                              "--no-mark-processed")
    pictures = readback(output)[2]
    picture, = pictures
    extracted = tmp_path / ("readback.png" if variant is None else "readback.jpg")
    extracted.write_bytes(picture)
    facts = image_facts(extracted)
    assert facts["format"] == ("png" if variant is None else "jpeg") and facts["space"] == "RGB"
    assert (int(facts["pixelWidth"]), int(facts["pixelHeight"])) == ((3000, 3000) if variant == "large" else (64, 64))
    if variant is None:
        assert picture == cover.read_bytes()  # supported supplied PNG needs no rewrite
    assert tools.snapshot(tmp_path)["take.wav"] == before["take.wav"]
    assert tools.snapshot(cover.parent) == cover_before
    assert row["dropped"] == [] and row["execution"] == "committed"


@pytest.mark.parametrize("maximum", [3000, 1000])
def test_m06_non_square_scaling_preserves_aspect_and_png(norma, tools, tmp_path, maximum):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    cover = tmp_path / "rectangular.png"
    cover.write_bytes(png(6001, 3001))
    _, _, output = normalise(norma, source, tmp_path / "run", "--artwork", cover,
                            "--artwork-max-size", maximum)
    image = tmp_path / "readback.png"
    image.write_bytes(readback(output)[2][0])
    facts = image_facts(image)
    width, height = int(facts["pixelWidth"]), int(facts["pixelHeight"])
    assert facts["format"] == "png" and max(width, height) == maximum
    assert abs(height - width * 3001 / 6001) <= 1


@pytest.mark.parametrize("size", [16384, 16385])
def test_m06_pixel_decode_limit(norma, tools, tmp_path, size):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    cover = tmp_path / "wide.png"
    cover.write_bytes(png(size, 16))
    run = norma("tags", "plan", source, "--config", "/dev/null", "--artwork", cover)
    assert run.code == int(size > 16384), run.stdout + run.stderr
    if size > 16384:
        assert "artwork_invalid" in run.stdout and "too large" in run.stdout + run.stderr


@pytest.mark.parametrize("over", [False, True])
def test_m06_byte_decode_limit(norma, tools, fixture_module, tmp_path, over):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    cover = tmp_path / "boundary.png"
    header = fixture_module.cover_png()
    cover.write_bytes(header + bytes(52428800 + int(over) - len(header)))
    run = norma("tags", "plan", source, "--config", "/dev/null", "--artwork", cover)
    assert run.code == int(over), run.stdout + run.stderr
    if over:
        assert "artwork_invalid" in run.stdout and "too large" in run.stdout + run.stderr


@pytest.mark.parametrize("origin", ["cli-corrupt", "manifest-missing",
    "folder-corrupt"])
def test_m06_bad_content_fails_before_render(norma, tools, fixture_path, tmp_path, monkeypatch, origin):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    if origin == "cli-corrupt":
        options = ["--artwork", fixture_path("F-16", "corrupt")]
    elif origin == "manifest-missing":
        manifest = tmp_path / "art.json"
        manifest.write_text(json.dumps({"version": 1, "files": {source.name: {"artwork": "missing.jpg"}}}))
        options = ["--manifest", manifest]
    else:
        (tmp_path / "cover.jpg").write_bytes(b"not an image\n")
        options = ["--folder-artwork"]
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", *options)
    assert run.code == 1 and "artwork_invalid" in run.stdout, run.stdout + run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


def test_m06_sips_missing_is_preflight_exit_three(norma, tools, fixture_module, tmp_path, monkeypatch):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    cover = tmp_path / "cover.png"
    cover.write_bytes(fixture_module.cover_png())
    folder = tmp_path / "tools"
    folder.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        assert executable
        (folder / name).symlink_to(executable)
    monkeypatch.setenv("PATH", str(folder))
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--artwork", cover)
    assert run.code == 3 and "sips" in run.stderr and run.stdout == ""
    assert tools.snapshot(tmp_path) == before


@pytest.mark.parametrize("report_form", [None, "csv", "json"])
@pytest.mark.parametrize("artwork", [False, True])
def test_m07_executable_trap_proves_zero_ffmpeg_and_only_explicit_report(
        norma, tools, fixture_module, tmp_path, monkeypatch, report_form, artwork):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tag_wav(source, title="Source", artist="Artist", iTunSMPB="gapless")
    options = ["--peak-mode", "dynamic", "--dry-run", "--log", tmp_path / "never.log", "--id3-version", "2.4"]
    if artwork:
        cover = tmp_path / "cover.png"
        cover.write_bytes(png(4000, 2000))  # requires real sips conversion, still no persistent temp
        options += ["--artwork", cover]
    report = tmp_path / ("plan." + report_form) if report_form else None
    if report:
        options += ["--report", report]
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source, *options)
    assert_accounted(run, source)
    assert calls.read_text() == ""
    after = tools.snapshot(tmp_path)
    assert tools.difference(before, after) == {
        "added": {report.name} if report else set(), "removed": set(), "changed": set()}
    assert all(row[3:10] == [""] * 7 for row in run.files)
    assert "writes no log" in run.stderr and "no audio is processed" in run.stderr
    branches = {branch for _, branch, _ in details(run, "PLAN")}
    assert branches == {"if stream-copied", "if encoded"}
    for _, branch, body in details(run, "PLAN"):
        assert body["mark"] == "pending"
        gapless, = [r for r in branch_rows(run, "TAG", branch) if r["key"] == "item:native:id3v2:TXXX:iTunSMPB"]
        assert gapless["status"] == ("kept" if branch == "if stream-copied" else "removed_by_design")
    if report_form == "json":
        row, = document(report)["outcomes"]
        assert row["execution"] == "plan" and row["published_paths"] == []
        assert len(row["conditional_plans"]) == 2
        assert row["in_lufs"] is row["out_lufs"] is row["mode"] is None
        for branch in row["conditional_plans"]:
            assert branch["metadata"]["plan_revision"] == branch["output"]["plan_revision"]
            assert all(member["temporary_path"] is None for member in branch["output"]["members"])


@pytest.mark.parametrize("case", ["copy-refused", "both-refused", "encode-only"])
def test_m07_natural_conditional_refusals_are_not_dropped(norma, tools, tmp_path, monkeypatch, case):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tag_wav(source, title="Storable", iTunSMPB="gapless")
    options = ["--wav-tags", "info", "--on-metadata-loss", "error"]
    if case == "both-refused":
        audio = WAVE(source)
        audio.tags.add(PRIV(owner="com.example", data=b"binary"))
        audio.save()
    elif case == "encode-only":
        options = ["-f", "flac", "--on-metadata-loss", "error"]
    report = tmp_path / "plan.json"
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "-v", "--report", report, *options)
    assert run.code == int(case == "both-refused"), run.stdout + run.stderr
    row, = document(report)["outcomes"]
    assert row["published_paths"] == [] and row["execution"] == "plan"
    if case == "encode-only":
        assert len(details(run, "PLAN")) == 1 and not details(run, "REFUSAL")
    else:
        branches = {b["assumption"]: b for b in row["conditional_plans"]}
        assert set(branches) == {"stream_copy", "encode"}
        refused = {basis for basis, b in branches.items() if "refusal" in b}
        assert refused == ({"stream_copy", "encode"} if case == "both-refused" else {"stream_copy"})
        assert len(details(run, "REFUSAL")) == len(refused)
        for basis in refused:
            assert branches[basis]["metadata"] is branches[basis]["output"] is None
            assert branches[basis]["refusal"]["reason_code"] == "metadata_loss_error"
        assert (row["out_path"] is None) == (case == "both-refused")
    assert calls.read_text() == ""
    assert tools.difference(before, tools.snapshot(tmp_path)) == {"added": {report.name}, "removed": set(), "changed": set()}


@pytest.mark.parametrize("fixture,variant", CATALOGUE_AUDIO,
                         ids=[f"{fixture}/{variant or 'default'}" for fixture, variant in CATALOGUE_AUDIO])
def test_m02_catalogue_has_zero_unaccounted_supported_source_occurrences(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, monkeypatch, fixture, variant):
    cached = fixture_path(fixture, variant)
    source = demos({"take" + cached.suffix: cached}) / ("take" + cached.suffix)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = plan(norma, source, "--strict-conflicts", "--force")
    assert_accounted(run, source)
    if fixture == "F-18" and variant == "chunks":
        expected = {"item:chunk:" + key.decode("latin-1") for key, _ in chunks(source.read_bytes())
                    if key in (b"bext", b"cue ", b"iXML")}
        for _, branch, _ in details(run, "PLAN"):
            items = [r for r in branch_rows(run, "TAG", branch) if r["key"] in expected]
            assert {r["key"] for r in items} == expected and len(items) == 3
            assert all(r["status"] == "dropped_format_limit" and r["loss_kind"] == "whole_value" for r in items)
    if variant == "conflict":
        assert all(body["resolved"]["title"]["value"] == fixture_module.TITLE for _, _, body in details(run, "PLAN"))
        assert dict(info_items(source))["INAM"] == b"INFO title\0"
        assert not details(run, "CONFLICT")  # internal-store disagreement is not a strict conflict
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before


def test_m06_missing_explicit_path_is_usage_error_before_tools(norma, tools, tmp_path, monkeypatch):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    calls = ffmpeg_trap(tmp_path, monkeypatch)
    before = tools.snapshot(tmp_path)
    run = norma("tags", "plan", source, "--config", "/dev/null", "--artwork", tmp_path / "missing.png")
    assert run.code == 2 and run.stdout == "" and "missing.png" in run.stderr
    assert calls.read_text() == "" and tools.snapshot(tmp_path) == before
