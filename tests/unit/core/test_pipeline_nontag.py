"""C-06 non-tag items through ``normalise`` (IC-2 F-1, D-072 (1), D-074; MD6, MD7; TK-0415, TK-0427).

S2 refused every source carrying a non-tag item (D-066). From S4 (TK-0427) a non-tag item
is planned, never carried in v1: its roll-up is ``dropped_format_limit`` with detail
``not_carried_v1`` (MD7), so the default ``warn`` policy writes the audio with
``metadata_loss_warn`` and ``--on-metadata-loss error`` fails the file before any render.
The D-074 filler and ``PEAK`` chunks are ``removed_by_design`` and are not a loss. The
real-tool cases are IC-2 Appendix A probe P1: FFmpeg-made untagged files, one item added by
hand each, run through the whole pipeline.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from norma import batch, pipeline
from norma.adapters import ffmpeg
from norma.models import RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
BITEXACT = ("-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1")
SECONDS = 4
RATE = 48000

real = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evidence folder is a ``TemporaryDirectory``; see test_pipeline_doctor.py for why."""
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


@pytest.fixture(autouse=True)
def no_process_left() -> None:
    yield
    assert ffmpeg._ACTIVE == {}, "a tool call outlived its test"


# --- Real tools: IC-2 probe P1 ------------------------------------------------------------------


def tone(path: Path, *codec: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=440:sample_rate={RATE}:duration={SECONDS}",
                    "-af", "volume=-24dB,aformat=channel_layouts=stereo", *codec, *BITEXACT, str(path)],
                   check=True, timeout=60)
    return path


def riff_append(path: Path, chunks: list[tuple[bytes, bytes]], *, big: bool = False) -> None:
    raw = bytearray(path.read_bytes())
    fmt = ">I" if big else "<I"
    for cid, body in chunks:
        raw += cid + struct.pack(fmt, len(body)) + body + (b"\0" if len(body) & 1 else b"")
    raw[4:8] = struct.pack(fmt, len(raw) - 8)
    path.write_bytes(bytes(raw))


def flac_insert(path: Path, kind: int, body: bytes) -> None:
    raw = path.read_bytes()
    assert raw[:4] == b"fLaC"
    header, length = raw[4], int.from_bytes(raw[5:8], "big")
    end = 8 + length
    block = bytes([kind | (header & 0x80)]) + len(body).to_bytes(3, "big") + body
    path.write_bytes(b"fLaC" + bytes([header & 0x7F]) + raw[5:end] + block + raw[end:])


def cuesheet() -> bytes:
    body = b"\0" * 128 + (0).to_bytes(8, "big") + b"\0" + b"\0" * 258 + bytes([2])
    body += (0).to_bytes(8, "big") + bytes([1]) + b"\0" * 12 + b"\0" + b"\0" * 13 + bytes([1])
    body += (0).to_bytes(8, "big") + bytes([1]) + b"\0" * 3
    body += (SECONDS * RATE).to_bytes(8, "big") + bytes([170]) + b"\0" * 12 + b"\0" + b"\0" * 13 + bytes([0])
    return body


def build_p1(folder: Path) -> dict[str, str]:
    """IC-2 P1 inputs, each with the words its refusal must name."""
    path = tone(folder / "cue.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"cue ", struct.pack("<I", 1) + struct.pack("<II4sIII", 1, 96000, b"data", 0, 0, 96000)),
                       (b"LIST", b"adtl" + b"labl" + struct.pack("<I", 10) + struct.pack("<I", 1) + b"Verse\0")])
    path = tone(folder / "bext.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"bext", b"Demo take 3, Studio B".ljust(256, b"\0") + b"\0" * 346)])
    path = tone(folder / "ixml.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"iXML", b"<?xml version='1.0'?><BWFXML><SCENE>12</SCENE></BWFXML>")])
    path = tone(folder / "mark.aiff", "-c:a", "pcm_s24be")
    riff_append(path, [(b"MARK", struct.pack(">HHI", 1, 1, 96000) + bytes([5]) + b"Verse")], big=True)
    path = tone(folder / "cue.flac", "-c:a", "flac")
    flac_insert(path, 5, cuesheet())
    meta = folder.parent / "chapters.txt"
    meta.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=Intro\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Verse\n", encoding="utf-8")
    plain = tone(folder.parent / "chapters-source.wav", "-c:a", "pcm_s24le")
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(plain),
                    "-f", "ffmetadata", "-i", str(meta), "-map", "0:a", "-map_metadata", "-1", "-map_chapters", "1",
                    "-c:a", "alac", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(folder / "chap.m4a")],
                   check=True, timeout=60)
    return {"cue.wav": "cue, LIST/adtl chunks", "bext.wav": "bext chunk", "ixml.wav": "iXML chunk",
            "mark.aiff": "MARK chunk", "cue.flac": "chapters", "chap.m4a": "chapters"}


def executor(request: pipeline.BatchRequest) -> batch.BatchResult:
    control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                 force_hook=request.force_hook)
    return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                               jobs=request.jobs, control=control, handle_sigint=False)


def run_normalise(tmp_path: Path, folder: Path, **changes) -> dict[str, pipeline.Normalised]:
    run_options = replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=(folder,),
                                     jobs=2), **changes)
    tools = pipeline.locate_normalise_tools(run_options)
    items = pipeline.discover_inputs(run_options).items
    run = pipeline.normalise_items(items, run_options, tools, execute=None if run_options.dry_run else executor)
    return {record.outcome.input.source_path.name: record for record in run.records}


def dropped(record: pipeline.Normalised) -> set[str]:
    plan = record.outcome.metadata
    assert plan is not None
    return {status.key for status in plan.statuses if status.status == "dropped_format_limit"}


@real
@pytest.mark.parametrize("dry_run", [False, True])
def test_ic2_p1_non_tag_items_are_reported_dropped_and_audio_is_written(tmp_path: Path, dry_run: bool) -> None:
    folder = tmp_path / "in"
    expected = build_p1(folder)
    tone(folder / "plain.wav", "-c:a", "pcm_s24le")  # control: a clean file in the same batch
    keys: dict[str, set[str]] = {}
    for name in expected:
        source_tags = pipeline.read_source_tags(folder / name, {"wav": "wav", "aiff": "aiff", "flac": "flac",
                                                                "m4a": "mp4"}[name.rsplit(".", 1)[1]])
        assert source_tags.non_tag_items and not source_tags.native_inventory, name  # the probe's premise
        keys[name] = {item.identity for item in source_tags.non_tag_items}
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    run = run_normalise(tmp_path, folder, dry_run=dry_run)
    assert set(run) == {*expected, "plain.wav"}
    assert run["plain.wav"].outcome.status == "ok"
    for name in expected:
        record, outcome = run[name], run[name].outcome
        assert (outcome.status, outcome.reason_code, outcome.execution) == (
            "warn", "metadata_loss_warn", "plan" if dry_run else "committed"), (name, record.primary)
        assert dropped(record) == keys[name], name  # one roll-up per item, never silent (MD7)
        for status in outcome.metadata.statuses:
            if status.key in keys[name]:
                assert (status.loss_kind, status.detail) == ("whole_value", "not_carried_v1")
    published = sorted(path.name for path in (folder / "Normalised").glob("*")) if not dry_run else []
    assert published == ([] if dry_run else sorted({*(n.rsplit(".", 1)[0] + {"m4a": ".m4a", "flac": ".flac",
                                                                              "aiff": ".aiff", "wav": ".wav"}[
        n.rsplit(".", 1)[1]] for n in expected), "plain.wav"}))
    if not dry_run:  # the output carries no non-tag item (MD39 read-back already checked it)
        for name in expected:
            assert run[name].outcome.tag_write.matched
    assert {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()} == before
    assert sorted(tmp_path.rglob(".norma-*")) == []


@real
def test_non_tag_items_under_the_error_policy_fail_before_rendering(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    path = tone(folder / "bext.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"bext", b"Demo take 3".ljust(602, b"\0"))])
    tone(folder / "plain.wav", "-c:a", "pcm_s24le")
    run = run_normalise(tmp_path, folder, on_metadata_loss="error")
    record = run["bext.wav"]
    assert (record.outcome.status, record.outcome.reason_code, record.outcome.execution) == (
        "failed", "metadata_loss_error", "not_written")
    assert record.primary.stage == "metadata_planned" and "item:chunk:bext" in (record.primary.detail or "")
    assert record.outcome.output is None and record.outcome.verification is None
    assert run["plain.wav"].outcome.status == "ok"
    assert sorted(p.name for p in (folder / "Normalised").glob("*")) == ["plain.wav"]


@real
def test_d074_exempt_chunks_are_removed_by_design_and_only_the_rest_is_dropped(tmp_path: Path) -> None:
    """Real files: filler and ``PEAK`` alone normalise cleanly; beside ``bext`` only ``bext`` is a loss."""
    import numpy as np
    import soundfile

    folder = tmp_path / "in"
    peak = struct.pack("<II", 1, 0) + struct.pack("<fI", 0.06, 0) * 2  # version, time, 2 × (value, position)
    path = tone(folder / "filler.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"JUNK", b"\0" * 28), (b"junk", b"\0" * 4), (b"PAD ", b"\0" * 8),
                       (b"FLLR", b"\0" * 6), (b"PEAK", peak)])
    path = tone(folder / "filler.aiff", "-c:a", "pcm_s24be")
    riff_append(path, [(b"PEAK", struct.pack(">II", 1, 0) + struct.pack(">fI", 0.06, 0) * 2),
                       (b"FLLR", b"\0" * 6)], big=True)
    path = tone(folder / "mixed.wav", "-c:a", "pcm_s24le")
    riff_append(path, [(b"PEAK", peak), (b"JUNK", b"\0" * 28), (b"bext", b"Take 3".ljust(602, b"\0"))])
    t = np.arange(SECONDS * RATE) / RATE
    signal = (0.06 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    soundfile.write(folder / "libsndfile.wav", np.stack([signal, signal], axis=1), RATE, subtype="FLOAT")
    assert b"PEAK" in (folder / "libsndfile.wav").read_bytes()  # the premise: libsndfile writes PEAK
    for name, container in (("filler.wav", "wav"), ("filler.aiff", "aiff"), ("libsndfile.wav", "wav")):
        assert pipeline.read_source_tags(folder / name, container).non_tag_items, name
    run = run_normalise(tmp_path, folder)
    for name in ("filler.wav", "filler.aiff", "libsndfile.wav"):
        record = run[name]
        assert record.outcome.status == "ok", (name, record.primary, record.outcome.warnings)
        assert not dropped(record)
        assert {s.status for s in record.outcome.metadata.statuses if s.key.startswith("item:chunk:")} == {
            "removed_by_design"}
    record = run["mixed.wav"]
    assert (record.outcome.status, record.outcome.reason_code) == ("warn", "metadata_loss_warn")
    assert dropped(record) == {"item:chunk:bext"}
