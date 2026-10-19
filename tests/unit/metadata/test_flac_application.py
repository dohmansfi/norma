"""FLAC ``APPLICATION`` blocks are non-tag items (IC-2 F-1, CR-052 WS-06 part, D-072 (1), MD7).

Reader cases build the block by hand and compare with mutagen's independent block list. The
real-tool case runs ``pipeline.normalise_items`` on an FFmpeg-made FLAC with a block added, as
IC-2 probe P1 did, and expects ``failed/metadata_loss_error`` with no output.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from mutagen.flac import FLAC

from norma import batch, pipeline
from norma.adapters import ffmpeg
from norma.metadata.read import PAYLOAD_LIMIT, read_source_tags
from norma.models import RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")


@pytest.fixture
def evidence_folder_cleanup(monkeypatch: pytest.MonkeyPatch):
    """The evidence folder is a ``TemporaryDirectory``; see tests/unit/core/test_pipeline_doctor.py."""
    monkeypatch.setattr(shutil, "_use_fd_functions", False)
    yield
    assert ffmpeg._ACTIVE == {}, "a tool call outlived its test"


def application_body(app_id: bytes, data: bytes) -> bytes:
    return app_id + data


def test_application_blocks_are_inventoried_as_non_tag_items(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    first, second = application_body(b"riff", b"\x01\x02\x03"), application_body(b"aiff", b"\x00" * 9)
    build.write_flac_blocks(path, prefix, [blocks[0], (2, first), (2, second), *blocks[1:]], frames)
    tags = read_source_tags(path, "flac")
    items = tags.non_tag_items
    assert [item.identity for item in items] == ["item:chunk:APPLICATION#1", "item:chunk:APPLICATION#2"]
    assert [item.payload for item in items] == [first, second]  # the whole block body, application ID first
    assert [item.payload_size for item in items] == [len(first), len(second)]
    assert [item.payload_sha256 for item in items] == [hashlib.sha256(b).hexdigest() for b in (first, second)]
    assert {(item.kind, item.detail) for item in items} == {("chunk", "not_carried_v1")}
    # Independent oracle: mutagen sees the same two blocks, in order, and the locator is the block offset.
    mutagen_blocks = FLAC(path).metadata_blocks
    assert sum(1 for block in mutagen_blocks if block.code == 2) == 2
    raw = path.read_bytes()
    for item, body in zip(items, (first, second)):
        name, _, offset = item.locator.partition("@")
        assert name == "APPLICATION" and raw[int(offset)] & 0x7F == 2
        assert raw[int(offset) + 4:int(offset) + 4 + len(body)] == body
    assert tags.native_inventory == () and tags.fields == {}


def test_a_single_block_has_no_occurrence_suffix_and_oversize_keeps_only_the_hash(audio, build) -> None:
    path = audio("flac")
    prefix, blocks, frames = build.flac_blocks(path)
    big = application_body(b"norm", bytes(PAYLOAD_LIMIT + 1))
    build.write_flac_blocks(path, prefix, [blocks[0], (2, big), *blocks[1:]], frames)
    (item,) = read_source_tags(path, "flac").non_tag_items
    assert item.identity == "item:chunk:APPLICATION"
    assert (item.payload, item.payload_size, item.payload_sha256) == (None, len(big), hashlib.sha256(big).hexdigest())


def test_application_beside_tags_and_chapters_leaves_them_alone(audio, build) -> None:
    path = audio("flac")
    flac = FLAC(path)
    flac["TITLE"] = "Kept apart"
    flac.save()
    prefix, blocks, frames = build.flac_blocks(path)
    build.write_flac_blocks(path, prefix, [blocks[0], (2, b"abcdXYZ"), *blocks[1:]], frames)
    tags = read_source_tags(path, "flac")
    assert tags.fields["title"].value == "Kept apart"
    assert [item.identity for item in tags.non_tag_items] == ["item:chunk:APPLICATION"]
    assert [item.identity for item in tags.native_inventory] == ["store:vorbis:TITLE"]


def test_structure_only_flac_has_no_non_tag_item(audio) -> None:
    assert read_source_tags(audio("flac"), "flac").non_tag_items == ()  # STREAMINFO, PADDING, SEEKTABLE


# --- Real tools ---------------------------------------------------------------------------------


def tone(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=48000:duration=4",
                    "-af", "volume=-24dB,aformat=channel_layouts=stereo", "-c:a", "flac",
                    "-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1", str(path)],
                   check=True, timeout=60)
    return path


def insert_application(path: Path) -> None:
    raw = path.read_bytes()
    header, length = raw[4], int.from_bytes(raw[5:8], "big")
    end = 8 + length
    body = b"norm" + b"private data"
    block = bytes([2 | (header & 0x80)]) + len(body).to_bytes(3, "big") + body
    path.write_bytes(b"fLaC" + bytes([header & 0x7F]) + raw[5:end] + block + raw[end:])


def executor(request: pipeline.BatchRequest) -> batch.BatchResult:
    control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                 force_hook=request.force_hook)
    return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                               jobs=request.jobs, control=control, handle_sigint=False)


def test_normalise_drops_a_flac_application_block_and_accounts_for_it(tmp_path: Path, evidence_folder_cleanup) -> None:
    """Metadata § 3.5 (MD7, MD39) and § 6: the block is a non-tag item, ``dropped_format_limit`` /
    ``not_carried_v1``, so the file is WARN ``metadata_loss_warn`` (default policy) and still written, without it.
    """
    folder = tmp_path / "in"
    tone(folder / "plain.flac")
    marked = tone(folder / "marked.flac")
    insert_application(marked)
    assert [block.code for block in FLAC(marked).metadata_blocks].count(2) == 1  # independent: mutagen agrees
    before = marked.read_bytes()
    options = RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=(folder,), jobs=2)
    tools = pipeline.locate_normalise_tools(options)
    run = pipeline.normalise_items(pipeline.discover_inputs(options).items, options, tools, execute=executor)
    records = {record.outcome.input.source_path.name: record for record in run.records}
    plain = records["plain.flac"].outcome
    assert (plain.status, plain.reason_code) == ("ok", "ok")
    assert plain.metadata is not None and plain.metadata.non_tag_items == ()
    record = records["marked.flac"]
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("warn", "metadata_loss_warn", "committed")
    assert marked.read_bytes() == before  # the source is never touched
    # The accounting: one non-tag item, dropped, with its reason; nothing else lost.
    plan = outcome.metadata
    assert plan is not None and plan.loss_policy_result == "warn"
    assert [item.identity for item in plan.non_tag_items] == ["item:chunk:APPLICATION"]
    (rollup,) = [status for status in plan.statuses if status.key == "item:chunk:APPLICATION"]
    assert (rollup.status, rollup.detail, rollup.effective) == ("dropped_format_limit", "not_carried_v1", None)
    assert [status.key for status in plan.statuses if status.status.startswith("dropped")] == ["item:chunk:APPLICATION"]
    assert any(item.code == "metadata_loss_warn" for item in outcome.warnings)
    # The output exists, carries no APPLICATION block (MD39) and still has the Norma mark.
    assert outcome.output is not None
    produced = sorted(path.name for path in (folder / "Normalised").glob("*"))
    assert produced == ["marked.flac", "plain.flac"], produced
    output = folder / "Normalised" / "marked.flac"
    codes = [block.code for block in FLAC(output).metadata_blocks]
    assert 2 not in codes
    assert FLAC(output).tags is not None and "NORMA" in {key.upper() for key in FLAC(output).tags.keys()}
