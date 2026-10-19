"""TK-0431: artwork dependency failures and no-store plan/run policy parity."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from norma import pipeline
from norma.errors import FileFailure
from tests.unit.core.test_pipeline_metadata import (
    by_name, generate, ledger, normalise, options, plan_tags, renders_forbidden,
)


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize("command", ["normalise", "tags_plan"])
@pytest.mark.parametrize("layer", ["row", "defaults"])
def test_missing_manifest_artwork_is_a_per_file_preparation_failure(
        tmp_path, monkeypatch, command, layer):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    manifest = tmp_path / "art.json"
    document = {"version": 1, "files": {"in/take.wav": {}}}
    if layer == "row":
        document["files"]["in/take.wav"] = {"artwork": "missing.jpg"}
    else:
        document["defaults"] = {"artwork": "missing.jpg"}
    manifest.write_text(json.dumps(document))
    calls = renders_forbidden(monkeypatch)
    before = ledger(tmp_path)
    if command == "normalise":
        record = by_name(normalise(options(tmp_path, source, manifest=manifest)))[source.name]
    else:
        record, = plan_tags(tmp_path, source, manifest=manifest).inspections
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code) == ("failed", "artwork_invalid")
    assert record.primary.stage == "metadata_prepared"
    assert str(tmp_path / "missing.jpg") in record.primary.message
    assert outcome.measurement_in is None and outcome.published_paths == ()
    assert calls == [] and ledger(tmp_path) == before


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize("container,codec", [("caf", "pcm_s24le"), ("adts", "aac")])
@pytest.mark.parametrize("policy", ["error", "warn", "sidecar"])
def test_no_store_plan_applies_loss_policy_before_any_metadata_or_audio_work(
        tmp_path, monkeypatch, container, codec, policy):
    source = generate(tmp_path / "in" / ("take." + container), "-c:a", codec)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("unexpected work")
        raise AssertionError("no-store tags plan must stop at tags_read")

    for name in ("read_source_tags", "provisional_format", "measure", "render"):
        monkeypatch.setattr(pipeline, name, forbidden)
    before = ledger(tmp_path)
    record, = plan_tags(tmp_path, source, on_metadata_loss=policy).inspections
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == (
        "failed" if policy == "error" else "warn",
        "metadata_loss_error" if policy == "error" else "metadata_loss_warn", "plan")
    warning, = outcome.warnings
    assert warning == pipeline.unreadable_tags(container)
    if policy == "error":
        assert record.primary.stage == "tags_read" and record.primary.detail == warning.message
    else:
        assert record.primary is None
    assert outcome.metadata is outcome.output is outcome.measurement_in is None
    assert outcome.conditional_plans == () and outcome.published_paths == ()
    assert record.source_tags is None and calls == [] and ledger(tmp_path) == before


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, NotADirectoryError])
def test_artwork_dependency_stat_errors_are_typed_and_do_not_start_preparation(
        tmp_path, monkeypatch, error):
    from norma.metadata.artwork import ExplicitArtwork
    from norma.models import InputItem

    source = tmp_path / "take.wav"
    source.write_bytes(b"not read by this test")
    info = source.stat()
    item = InputItem(source, tmp_path, PurePosixPath(source.name), (info.st_dev, info.st_ino), 0, False)
    meta = pipeline.MetadataInputs((item,), options(tmp_path, source))
    opinions = replace(meta.opinions(item), artwork=ExplicitArtwork("set", tmp_path / "cover.jpg", "manifest"),
                       artwork_layer="manifest")

    def refused(*args):
        raise error("cannot stat artwork")

    monkeypatch.setattr(pipeline.safety, "protect", refused)
    monkeypatch.setattr(meta, "ensure_preparer", lambda: pytest.fail("preparation must not start"))
    try:
        with pytest.raises(FileFailure) as failure:
            meta.resolve(item, opinions, pipeline._EMPTY_TAGS, "wav")
        assert (failure.value.reason, failure.value.stage) == ("artwork_invalid", "metadata_prepared")
        assert str(tmp_path / "cover.jpg") in failure.value.detail
    finally:
        meta.close()
