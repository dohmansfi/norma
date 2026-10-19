"""TST-M-01/M-02: MD3 priority, legacy-store accounting and output absence."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest
from mutagen.apev2 import APEv2, APENoHeaderError

from tests.e2e.test_metadata_support import (
    assert_accounted, assert_no_render, branch_rows, chunks, details, document, ffmpeg_trap,
    native_ids, normalise, plan, readback,
)


@pytest.mark.parametrize("target", ["mp3", "flac", "alac"])
def test_m01_legacy_mp3_stores_are_accounted_but_never_written(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, target):
    source = demos({"take.mp3": fixture_path("F-13", "mp3-v0")}) / "take.mp3"
    ape = APEv2()
    ape["Title"] = "APE title"
    ape["Artist"] = "APE artist"
    ape.save(source)
    # Complete ID3v1.1 in native bytes; no library translation can mask its priority.
    tag = (b"TAG" + b"v1 title".ljust(30, b"\0") + b"v1 artist".ljust(30, b"\0")
           + b"v1 album".ljust(30, b"\0") + b"2026" + b"v1 comment".ljust(28, b"\0")
           + bytes([0, 7, 255]))
    assert len(tag) == 128
    source.write_bytes(source.read_bytes() + tag)
    before = tools.snapshot(source.parent)
    preview = plan(norma, source, "-f", target, "--strict-conflicts", "--no-mark-processed")
    assert_accounted(preview, source)
    ids = native_ids(source)
    assert {"store:apev2:Title", "store:apev2:Artist", "store:id3v1:title", "store:id3v1:track"} <= ids
    for _, _, body in details(preview, "PLAN"):
        assert body["resolved"]["title"]["value"] == fixture_module.TITLE
        assert body["resolved"]["artist"]["value"] == "Test Artist"
        assert body["resolved"]["track"]["value"] == 3
    assert not details(preview, "CONFLICT")  # disagreements within S are not strict conflicts
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target,
                              "--strict-conflicts", "--no-mark-processed")
    assert readback(output)[0]["title"] == fixture_module.TITLE
    assert row["dropped"] == []
    assert not any(identity.startswith(("store:apev2:", "store:id3v1:")) for identity in native_ids(output))
    if target == "mp3":
        with pytest.raises(APENoHeaderError):
            APEv2(output)
        assert output.read_bytes()[-128:-125] != b"TAG"
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize("target", ["aiff", "wav", "flac"])
def test_m01_aiff_text_is_inventoried_under_id3_priority_and_removed_from_outputs(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, target):
    source = demos({"take.aif": fixture_path("F-13", "aiff24")}) / "take.aif"
    data = source.read_bytes()
    for key, payload in ((b"NAME", b"Legacy title"), (b"AUTH", b"Legacy artist"), (b"ANNO", b"Legacy note")):
        data += key + struct.pack(">I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
    source.write_bytes(data[:4] + struct.pack(">I", len(data) - 8) + data[8:])
    before = tools.snapshot(source.parent)
    preview = plan(norma, source, "-f", target, "--strict-conflicts", "--no-mark-processed")
    assert_accounted(preview, source)
    for _, branch, body in details(preview, "PLAN"):
        assert body["resolved"]["title"]["value"] == fixture_module.TITLE
        assert len([row for row in branch_rows(preview, "NATIVE_FATE", branch)
                    if row["key"].startswith("source:store:aiff-text:")]) == 3
    _, row, output = normalise(norma, source, tmp_path / "run", "-f", target,
                              "--strict-conflicts", "--no-mark-processed")
    assert readback(output)[0]["title"] == fixture_module.TITLE and row["dropped"] == []
    if target == "aiff":
        assert not any(key in (b"NAME", b"AUTH", b"ANNO", b"(c) ") for key, _ in chunks(output.read_bytes(), big=True))
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize("legacy", ["ape-binary", "ape-illegal", "aiff-copyright"])
@pytest.mark.parametrize("policy", ["warn", "sidecar", "error"])
def test_m02_unprojected_legacy_loss_plan_and_run_agree(
        norma, tools, fixture_path, demos, tmp_path, monkeypatch, legacy, policy):
    aiff = legacy == "aiff-copyright"
    cached = fixture_path("F-13", "aiff24" if aiff else "mp3-v0")
    source = demos({"take.aif" if aiff else "take.mp3": cached}) / ("take.aif" if aiff else "take.mp3")
    if aiff:
        payload = b"Aiff copyright"
        data = source.read_bytes() + b"(c) " + struct.pack(">I", len(payload)) + payload
        source.write_bytes(data[:4] + struct.pack(">I", len(data) - 8) + data[8:])
        identity = "store:aiff-text:(c) "
    else:
        tags = APEv2()
        key = "Cover Art (Front)" if legacy == "ape-binary" else "bad name!"
        tags[key] = b"cover.jpg\0\xff\xd8\xff" if legacy == "ape-binary" else "unprojected text"
        tags.save(source)
        identity = "store:apev2:" + key
    before = tools.snapshot(source.parent)
    calls = ffmpeg_trap(tmp_path, monkeypatch, delegate=True)
    options = ("-f", "flac", "--on-metadata-loss", policy, "--no-mark-processed")
    preview = norma("tags", "plan", source, "--config", "/dev/null", "-v", *options)
    assert calls.read_text() == ""
    run, row, output = normalise(norma, source, tmp_path / "run", *options, code=int(policy == "error"))
    assert preview.code == int(policy == "error")
    if policy == "error":
        assert "metadata_loss_error" in preview.stdout
        assert row["reason"] == "metadata_loss_error" and not row["published_paths"]
        assert output is None or not output.exists()
        assert_no_render(calls, source)
    else:
        assert_accounted(preview, source)
        native, = [r for _, _, r in details(preview, "NATIVE_FATE") if r["key"] == "source:" + identity]
        assert (native["status"], native["detail"], native["expected_ids"]) == (
            "dropped_format_limit", "legacy_store_not_carried", [])
        lost = next(r for r in row["dropped"] if identity in r["source_native_ids"])
        assert lost["detail"] == "legacy_store_not_carried" and lost["loss_kind"] == "whole_value"
        assert "metadata_loss_warn" in run.stdout and output.exists()
        if policy == "sidecar":
            sidecar = document(Path(row["sidecar"]))
            evidence = next(r for r in sidecar["losses"] if r["key"] == lost["key"])
            assert evidence["requested"]["items"][0]["identity"] == identity
            assert evidence["requested"]["items"][0]["payload_b64"]
        else:
            assert not row["sidecar"]
    assert tools.snapshot(source.parent) == before
