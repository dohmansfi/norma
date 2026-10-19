"""TK-0428: real-tool staging and checked template orchestration (CR-084–089)."""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from mutagen.flac import FLAC, Picture

from norma import pipeline, safety
from norma.errors import FileFailure, UsageError
from norma.metadata import manifest
from tests.unit.core.test_pipeline_metadata import (
    REAL_FFMPEG, TAGS, by_name, fingerprint, generate, ledger, normalise, options, png, residue,
)

pytestmark = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch):
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def pictured_source(tmp_path):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={**TAGS, "composer": "Test Composer"})
    image = png(tmp_path / "images" / "cover.png").read_bytes()
    audio = FLAC(source)
    for kind in (3, 4):
        picture = Picture()
        picture.type, picture.mime, picture.data = kind, "image/png", image
        picture.width = picture.height = 8
        audio.add_picture(picture)
    audio.save()
    return source, image


@pytest.mark.parametrize("mark", [False, True])
def test_sidecar_exports_and_exact_space_total(tmp_path, monkeypatch, mark):
    source, image = pictured_source(tmp_path)
    before = fingerprint(source)
    estimates = []
    original = safety.estimate_space

    def estimate(*args, **kwargs):
        estimates.append(kwargs["sidecar_bytes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(safety, "estimate_space", estimate)
    record = by_name(normalise(options(tmp_path, source, format="mp3" if mark else "wav", wav_tags="info",
                                       on_metadata_loss="sidecar", no_source_tags=False)))[source.name]
    outcome = record.outcome
    assert outcome.execution == "committed", record.primary
    assert outcome.output is not None and outcome.metadata is not None
    nonaudio = [member for member in outcome.output.members if member.purpose != "audio"]
    assert nonaudio
    assert estimates == [sum(member.final_path.stat().st_size for member in nonaudio)]
    if not mark:
        assert {member.purpose for member in nonaudio} == {"tags_sidecar", "cover_export", "picture_export"}
        for member in nonaudio:
            if member.purpose != "tags_sidecar":
                assert member.final_path.read_bytes() == image
    assert set(outcome.published_paths) == {member.final_path for member in outcome.output.members}
    assert fingerprint(source) == before and residue(tmp_path) == []
    if mark:
        tags = pipeline.read_source_tags(outcome.output.audio_member.final_path, "mp3")
        assert pipeline.read_processing_mark(tags).validity == "valid"
        document = json.loads(nonaudio[0].final_path.read_text())
        assert document["plan_revision"] == outcome.metadata.plan_revision == 2


@pytest.mark.parametrize("purpose", ["tags_sidecar", "cover_export", "picture_export"])
def test_failed_member_staging_publishes_nothing(tmp_path, monkeypatch, purpose):
    source, _ = pictured_source(tmp_path)
    original = safety.RunRegistry.stage_member
    successful = []

    def stage(registry, index, member, data):
        if member.purpose == purpose:
            raise FileFailure("disk_full", "tags_written", "injected staging failure")
        result = original(registry, index, member, data)
        successful.append(member.member_id)
        return result

    monkeypatch.setattr(safety.RunRegistry, "stage_member", stage)
    record = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info",
                                       on_metadata_loss="sidecar")))[source.name]
    assert record.outcome.reason_code == "disk_full", record.primary
    assert record.outcome.published_paths == () and record.outcome.execution == "not_written"
    assert not any(member.final_path.exists() for member in record.outcome.output.members)
    assert residue(tmp_path) == []


def test_space_refusal_includes_exports_before_allocation(tmp_path, monkeypatch):
    source, _ = pictured_source(tmp_path)
    original = safety.RunRegistry.reserve_space
    captured = []

    def reserve(registry, index, folder, estimate, **kwargs):
        assert estimate.sidecar_bytes > 0
        captured.append(estimate)
        # Enough for the audio alone, not for the exact sidecar bytes.
        available = replace(estimate, sidecar_bytes=0).required_bytes
        registry._free_space = lambda path: available
        return original(registry, index, folder, estimate, **kwargs)

    monkeypatch.setattr(safety.RunRegistry, "reserve_space", reserve)
    monkeypatch.setattr(pipeline, "render", lambda *a, **k: pytest.fail("render after space refusal"))
    record = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info",
                                       on_metadata_loss="sidecar")))[source.name]
    assert record.outcome.reason_code == "disk_full", record.primary
    assert captured and record.outcome.published_paths == () and residue(tmp_path) == []
    assert not (source.parent / "Normalised").exists()


def test_bundle_publication_failure_rolls_back_exports_and_audio(tmp_path):
    source, _ = pictured_source(tmp_path)

    def fault(point, member_id):
        if point == "publish" and member_id == "audio":
            raise OSError("injected audio publication failure")

    record = by_name(normalise(options(tmp_path, source, format="wav", wav_tags="info",
                                       on_metadata_loss="sidecar"), fault_hook=fault))[source.name]
    assert record.outcome.execution == "not_written", record.primary
    assert record.outcome.published_paths == ()
    assert not any(member.final_path.exists() for member in record.outcome.output.members)
    assert residue(tmp_path) == []


def template(tmp_path, *inputs, callback=None, cancel=None, tools=None, **changes):
    opts = options(tmp_path, *inputs, command="tags_template", **changes)
    tools = tools or pipeline.locate_tools(opts, need_ffmpeg=False)
    return pipeline.template_tags_items(pipeline.discover_inputs(opts).items, opts, tools,
                                        on_inspection=callback, cancel_event=cancel)


@pytest.mark.parametrize("form", ["csv", "json"])
@pytest.mark.parametrize("mode, expected", [("fill", "Source title"), ("override", "Filename title")])
def test_template_round_trip_uses_source_and_optional_inference_only(tmp_path, form, mode, expected):
    source = generate(tmp_path / "in" / "01 Filename title.flac", "-c:a", "flac",
                      tags={**TAGS, "title": "Source title"})
    output = tmp_path / f"metadata.{form}"
    rows = []
    run = template(tmp_path, source, template_output=output, infer_from_filename="{track} {title}",
                   infer_mode=mode, artist="Ignored CLI", manifest=tmp_path / "missing.csv",
                   callback=lambda row: rows.append((row, output.exists())))
    assert run.path == output and not run.run_errors and run.text == output.read_text()
    assert len(rows) == 1 and rows[0][1] and run.outcomes[0].status in ("ok", "warn")
    loaded = manifest.load_manifest(output)
    opts = options(tmp_path, source)
    match = manifest.match_manifest(loaded, pipeline.discover_inputs(opts).items).inputs[0]
    assert match.row is not None and not match.failures
    cells = {**loaded.defaults, **match.row.cells}
    assert cells["title"].value == expected and cells["track"].value == 1
    assert cells["artist"].value == TAGS["artist"] and cells["date"].value == TAGS["date"]
    assert residue(tmp_path) == []


def test_template_stdout_is_csv_and_has_no_write_api_calls(tmp_path, monkeypatch):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    before = ledger(tmp_path)
    for name in ("preflight", "publish_run_output"):
        monkeypatch.setattr(safety, name, lambda *a, **k: pytest.fail("stdout must not invoke writes"))
    monkeypatch.setattr(pipeline, "measure", lambda *a, **k: pytest.fail("template must not measure"))
    run = template(tmp_path, source, template_output="-")
    assert run.path is None and not run.run_errors
    assert run.text.startswith("file,title,artist,") and "in/take.flac" in run.text
    assert ledger(tmp_path) == before


@pytest.mark.parametrize("kind", ["input", "config", "tool", "symlink", "directory", "missing_parent", "exists"])
def test_template_unsafe_destination_fails_before_rows_or_writes(tmp_path, kind):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    destination = tmp_path / "unsafe.csv"
    extra = {"overwrite": True}
    if kind == "input":
        destination.hardlink_to(source)
    elif kind == "config":
        destination.write_text("config source")
        extra["config"] = destination
    elif kind == "tool":
        shutil.copy2(REAL_FFMPEG.parent / "ffprobe", destination)
        located = pipeline.locate_tools(options(tmp_path, source), need_ffmpeg=False)
        extra["tools"] = replace(located, prober=replace(located.prober, path=destination))
    elif kind == "symlink":
        destination.symlink_to(source)
    elif kind == "directory":
        destination.mkdir()
    elif kind == "missing_parent":
        destination = tmp_path / "missing" / "unsafe.csv"
    else:
        destination.write_text("hours of edits")
        extra["overwrite"] = False
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError):
        template(tmp_path, source, template_output=destination, callback=rows.append, **extra)
    assert rows == [] and ledger(tmp_path) == before


def test_template_marked_sources_remain_readable_and_bad_input_is_per_file(tmp_path):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={**TAGS, "NORMA": "v=1;broken"})
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    run = template(tmp_path, source, bad, template_output=tmp_path / "metadata.json")
    assert len(run.outcomes) == 2 and any(outcome.status == "failed" for outcome in run.outcomes)
    assert run.path is not None and "take.flac" in run.text and "broken.wav" not in run.text


def test_template_cancellation_writes_nothing(tmp_path):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    before = ledger(tmp_path)
    cancel = threading.Event()
    cancel.set()
    run = template(tmp_path, source, template_output=tmp_path / "metadata.csv", cancel=cancel)
    assert run.cancelled and run.not_started == 1 and run.text == "" and run.path is None
    assert ledger(tmp_path) == before


def test_template_publication_error_preserves_outcomes_and_previous_file(tmp_path, monkeypatch):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags=TAGS)
    destination = tmp_path / "metadata.csv"
    destination.write_text("hours of edits")

    def fail(*args, **kwargs):
        raise pipeline.RunOutputError("injected publication failure")

    monkeypatch.setattr(safety, "publish_run_output", fail)
    run = template(tmp_path, source, template_output=destination, overwrite=True)
    assert len(run.outcomes) == 1 and run.path is None and run.run_errors
    assert destination.read_text() == "hours of edits" and residue(tmp_path) == []
