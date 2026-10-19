"""TST-M-01/M-02 extended native evidence, chapters and precision cases."""
from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path

import pytest
from mutagen.flac import FLAC
from mutagen.id3 import CHAP, CTOC, EQU2, ID3, PRIV, RVA2, TIT2, TRCK
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

from tests.e2e.test_metadata_support import (
    assert_accounted, branch_rows, chunks, details, document, info_items,
    normalise, plan, readback, sha,
)


@pytest.mark.parametrize("target", ["wav", "flac", "alac"])
def test_m02_repeated_vorbis_values_each_have_one_fate(norma, fixture_path, demos, tmp_path, target):
    source = demos({"take.flac": fixture_path("F-13", "flac24")}) / "take.flac"
    tags = FLAC(source)
    tags["ARTIST"] = ["Anna", "Bob"]
    tags["MOOD"] = ["calm", "bright"]
    tags.save()
    preview = plan(norma, source, "-f", target, "--no-mark-processed")
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        native = branch_rows(preview, "NATIVE_FATE", branch)
        artist = [row for row in native if any(i.startswith("store:vorbis:ARTIST#") for i in row["source_ids"])]
        assert len(artist) == 2 and {row["source_ids"][0] for row in artist} == {
            "store:vorbis:ARTIST#1", "store:vorbis:ARTIST#2"}
        assert all(row["status"] == "changed" and row["loss_kind"] == "none" for row in artist)
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target, "--no-mark-processed")
    fields, extras, _ = readback(output)
    assert fields["artist"] == "Anna; Bob" and extras["MOOD"] == "calm; bright"
    assert row["dropped"] == []


@pytest.mark.parametrize("request_date", [None, "2026", "2026-10"])
def test_m02_loss_compares_winning_request_not_superseded_source(norma, fixture_path, demos,
                                                               tmp_path, request_date):
    source = demos({"take.flac": fixture_path("F-13", "flac24")}) / "take.flac"
    tags = FLAC(source)
    tags["DATE"] = ["2026-03"]
    tags.save()
    options = ["-f", "aiff", "--no-mark-processed"]
    if request_date is not None:
        options += ["--date", request_date]
    preview = plan(norma, source, *options)
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        date, = [r for r in branch_rows(preview, "TAG", branch) if r["key"] == "field:date"]
        assert date["requested"] == (request_date or "2026-03") and date["effective"] == "2026"
        assert date["loss_kind"] == ("none" if request_date == "2026" else "precision")
    _, row, output = normalise(norma, source, tmp_path / "run", *options)
    assert readback(output)[0]["date"] == "2026"
    assert [loss["loss_kind"] for loss in row["dropped"]] == ([] if request_date == "2026" else ["precision"])


@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_m02_mp4_timestamp_projects_only_the_date_and_sidecar_keeps_original(
        norma, tools, fixture_path, demos, tmp_path, policy):
    source = demos({"take.m4a": fixture_path("F-13", "alac24")}) / "take.m4a"
    tags = MP4(source)
    timestamp = "2026-10-05T07:00:00Z"
    tags["©day"] = [timestamp]
    tags["cpil"] = False
    tags.save()
    _, row, output = normalise(norma, source, tmp_path / "run", "--on-metadata-loss", policy,
                              "--no-mark-processed", code=int(policy == "error"))
    if policy == "error":
        assert row["reason"] == "metadata_loss_error" and row["published_paths"] == []
        return
    fields, _, pictures = readback(output)
    assert fields["date"] == "2026-10-05" and fields["compilation"] is False
    assert pictures == readback(source)[2]
    loss, = row["dropped"]
    assert loss["key"] == "field:date" and loss["requested"] == timestamp and loss["effective"] == "2026-10-05"
    assert loss["detail"] == "time_of_day_dropped" and loss["loss_kind"] == "precision"
    if policy == "sidecar":
        sidecar = document(Path(row["sidecar"]))
        subset = sidecar["manifest"]["files"][output.name]
        assert subset["date"] == "2026-10-05" and subset["compilation"] is False
        assert "artwork" not in subset  # retained cover has no unreserved export
        assert sidecar["losses"][0]["requested"] == timestamp
        assert set(row["published_paths"]) == {str(output), row["sidecar"]}


@pytest.mark.parametrize("repair", [False, True])
def test_m02_invalid_source_number_is_evidence_not_a_stringified_field(norma, tools, tmp_path, repair):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    tags = WAVE(source)
    tags.add_tags()
    tags.tags.add(TRCK(encoding=3, text=["A1"]))
    tags.save()
    options = ["--no-mark-processed", "--on-metadata-loss", "sidecar"]
    if repair:
        options += ["--track", "3"]
    preview = plan(norma, source, *options)
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        track, = [r for r in branch_rows(preview, "TAG", branch) if r["key"] == "field:track"]
        assert track["effective"] == (3 if repair else None)
        assert track["status"] == ("changed" if repair else "dropped_format_limit")
        if not repair:
            assert track["requested"]["kind"] == "inventory_ref"
    _, row, output = normalise(norma, source, tmp_path / "run", *options)
    assert readback(output)[0].get("track") == (3 if repair else None)
    if repair:
        assert row["sidecar"] is None and row["dropped"] == []
    else:
        sidecar = document(Path(row["sidecar"]))
        assert "track" not in sidecar["manifest"]["files"][output.name]
        assert sidecar["losses"][0]["requested"]["items"][0]["payload_b64"]


@pytest.mark.parametrize("target", ["wav", "flac", "mp3"])
def test_m02_chapters_have_one_rollup_not_one_per_native_frame(norma, tools, tmp_path, target):
    source = tools.tone(tmp_path / "chapters.wav", seconds=3)
    tags = WAVE(source)
    tags.add_tags()
    for n, start, end, title in [(1, 0, 1500, "Intro"), (2, 1500, 3000, "Outro")]:
        tags.tags.add(CHAP(element_id=f"ch{n}", start_time=start, end_time=end,
                           start_offset=0xFFFFFFFF, end_offset=0xFFFFFFFF,
                           sub_frames=[TIT2(encoding=3, text=[title])]))
    tags.tags.add(CTOC(element_id="toc", flags=3, child_element_ids=["ch1", "ch2"], sub_frames=[]))
    tags.save(v2_version=4)
    preview = plan(norma, source, "-f", target, "--on-metadata-loss", "sidecar")
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        rollups = [r for r in branch_rows(preview, "TAG", branch) if r["key"].startswith("item:")]
        assert [r["key"] for r in rollups] == ["item:chapters"]
        assert len(rollups[0]["source_native_ids"]) == 3
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target, "--on-metadata-loss", "sidecar")
    assert [r["key"] for r in row["dropped"]] == ["item:chapters"]
    sidecar = document(Path(row["sidecar"]))
    item, = sidecar["items"]
    payload = base64.b64decode(item["payload_b64"])
    assert sha(payload) == item["payload_sha256"]
    evidence = json.loads(payload)
    chapters = evidence["sources"][0]["chapters"]
    assert [c["start"] for c in chapters] == [[0, 1000], [1500, 1000]]
    assert [c["end"] for c in chapters] == [[1500, 1000], [3000, 1000]]
    assert [c["title"] for c in chapters] == ["Intro", "Outro"]
    assert len(sidecar["native_losses"]) == 3


@pytest.mark.parametrize("frame_id", ["RVA2", "EQU2"])
def test_m01_fractional_raw_frames_are_accounted_without_float_coercion(norma, tools, tmp_path, frame_id):
    source = tools.tone(tmp_path / "raw.wav", seconds=3)
    tags = WAVE(source)
    tags.add_tags()
    frame = RVA2(desc="track", channel=1, gain=-3.25, peak=.875) if frame_id == "RVA2" else EQU2(
        method=0, desc="eq", adjustments=[(500, -.5), (1000, 2)])
    tags.tags.add(frame)
    tags.save(v2_version=4)
    body = next(b for k, b in chunks(source.read_bytes()) if k.lower() == b"id3 ")
    stored = body[10:10 + 10 + sum((body[14 + i] & 127) << (21 - i * 7) for i in range(4))]
    preview = plan(norma, source, "--id3-version", "2.4", "--on-metadata-loss", "sidecar")
    assert_accounted(preview, source)
    for _, branch, _ in details(preview, "PLAN"):
        row, = [r for r in branch_rows(preview, "NATIVE_FATE", branch) if r["source_ids"]]
        item, = row["source_items"]
        assert item["payload_encoding"] == "dn-raw-1" and item["payload_sha256"] == sha(stored)
        assert row["status"] == ("removed_by_design" if frame_id == "RVA2" else "dropped_format_limit")
    _, row, output = normalise(norma, source, tmp_path / "run", "--id3-version", "2.4", "--on-metadata-loss", "sidecar")
    assert not WAVE(output).tags.getall(frame_id)
    if frame_id == "RVA2":
        assert row["sidecar"] is None and row["dropped"] == []
    else:
        item, = document(Path(row["sidecar"]))["native_losses"][0]["items"]
        assert base64.b64decode(item["payload_b64"]) == stored


@pytest.mark.parametrize("payload_size", [1048576, 1048577])
def test_m02_sidecar_evidence_payload_at_and_over_one_mib(norma, tools, tmp_path, payload_size):
    source = tools.tone(tmp_path / "chunk.wav", seconds=3)
    data = source.read_bytes()
    payload = b"x" * payload_size
    data += b"iXML" + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
    source.write_bytes(data[:4] + struct.pack("<I", len(data) - 8) + data[8:])
    _, row, _ = normalise(norma, source, tmp_path / "run", "--on-metadata-loss", "sidecar")
    item, = document(Path(row["sidecar"]))["items"]
    assert item["payload_size"] == payload_size and item["payload_sha256"] == sha(payload)
    if payload_size > 1048576:
        assert item["payload_omitted"] == "over_limit" and "payload_b64" not in item
    else:
        assert "payload_omitted" not in item and base64.b64decode(item["payload_b64"]) == payload


def test_m01_wav_info_native_utf8_nul_without_cset(norma, tools, tmp_path):
    source = tools.tone(tmp_path / "take.wav", seconds=3)
    _, _, output = normalise(norma, source, tmp_path / "run", "--title", "Perché 東京 🎵", "--artist", "Test Artist",
                            "--comment", 'line one\nline two; comma, "quote"', "--wav-tags", "info")
    native = dict(info_items(output))
    assert native["INAM"] == "Perché 東京 🎵".encode() + b"\0"
    assert native["IART"] == b"Test Artist\0" and native["ICMT"].endswith(b"\0")
    assert b"CSET" not in [key for key, _ in chunks(output.read_bytes())]
    assert "ISFT" not in native
