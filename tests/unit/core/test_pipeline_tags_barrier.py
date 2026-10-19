"""TK-0429: U-18 protects every tags-plan read and documentary bundle before rows."""
from __future__ import annotations

import json
import shutil
import threading
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

from norma import pipeline
from norma.errors import UsageError
from tests.unit.core.test_pipeline_metadata import generate, options, png, residue
from tests.unit.core.test_pipeline_report import ledger as full_ledger

pytestmark = pytest.mark.real_ffmpeg


def ledger(root):
    # Artwork preparation creates and removes private scratch below sandbox HOME.
    # Its parent mtime is not a user-file write; all files and other folders stay checked.
    return {name: entry for name, entry in full_ledger(root).items() if name != "home/tmp"}


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch):
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def run_plan(opts, *, report=None, callback=None, cancel=None):
    tools = pipeline.locate_tools(opts, need_ffmpeg=False)
    return pipeline.plan_tags_items(pipeline.discover_inputs(opts).items, opts, tools,
                                    report=pipeline.report_request(report) if report is not None else None,
                                    on_inspection=callback, cancel_event=cancel)


def all_plans(run):
    return [plan for row in run.outcomes for plan in
            ([row.output] if row.output is not None else []) +
            [branch.output for branch in row.conditional_plans if branch.output is not None]]


def alias(path, kind, root):
    if kind == "exact":
        return path
    if kind == "case":
        return path.with_name(path.name.upper())
    if kind == "unicode":
        return path.with_name(unicodedata.normalize("NFD", path.name))
    link = root / "alias.csv"
    if kind == "hardlink":
        link.hardlink_to(path)
    else:
        link.symlink_to(path)
    return link


@pytest.mark.parametrize("kind", ["manifest", "artwork", "config", "source", "tool"])
@pytest.mark.parametrize("name", ["exact", "case", "unicode", "hardlink", "symlink"])
def test_report_refuses_every_read_alias_before_callbacks(tmp_path, kind, name):
    source = generate(tmp_path / "in" / "café.wav", "-c:a", "pcm_s16le")
    opts = options(tmp_path, source, command="tags_plan", overwrite=True)
    tools = None
    if kind == "manifest":
        target = tmp_path / "café.csv"
        target.write_text("file,title\nin/café.wav,Demo\n", encoding="utf-8")
        opts = replace(opts, manifest=target)
    elif kind == "artwork":
        target = png(tmp_path / "café.png")
        opts = replace(opts, artwork=target)
    elif kind == "config":
        target = tmp_path / "café.toml"
        target.write_text("target = -16\n")
        opts = replace(opts, config=target)
    elif kind == "tool":
        # Use a private, valid prober so alias creation never touches the installed tool.
        folder = tmp_path / "tools"
        folder.mkdir()
        shutil.copy2(Path("/usr/local/bin/ffprobe"), folder / "café")
        located = pipeline.locate_tools(opts, need_ffmpeg=False)
        tools = replace(located, prober=replace(located.prober, path=folder / "café"))
        target = folder / "café"
    else:
        target = source
    destination = alias(target, name, tmp_path)
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError) as caught:
        if kind == "tool":
            assert tools is not None
            pipeline.plan_tags_items(pipeline.discover_inputs(opts).items, opts, tools,
                                     report=pipeline.report_request(destination), on_inspection=rows.append)
        else:
            run_plan(opts, report=destination, callback=rows.append)
    assert caught.value.exit_code == 2
    assert rows == [] and ledger(tmp_path) == before and residue(tmp_path) == []


@pytest.mark.parametrize("layer", ["row", "default", "folder"])
def test_artwork_read_from_manifest_or_folder_is_protected(tmp_path, layer):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    cover = png(source.parent / "cover.png")
    opts = options(tmp_path, source, command="tags_plan", overwrite=True)
    if layer == "folder":
        opts = replace(opts, folder_artwork=True)
    else:
        document = {"version": 1, "files": {"in/take.wav": {}}}
        if layer == "row":
            document["files"]["in/take.wav"]["artwork"] = "in/cover.png"
        else:
            document["defaults"] = {"artwork": "in/cover.png"}
        manifest = tmp_path / "metadata.json"
        manifest.write_text(json.dumps(document))
        opts = replace(opts, manifest=manifest)
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError, match="(metadata|artwork).*reads"):
        run_plan(opts, report=cover, callback=rows.append)
    assert rows == [] and ledger(tmp_path) == before and residue(tmp_path) == []


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("name", ["exact", "case", "unicode"])
@pytest.mark.parametrize("purpose", ["audio", "tags_sidecar"])
def test_report_refuses_all_planned_bundle_names(tmp_path, conditional, name, purpose):
    source = generate(tmp_path / "in" / ("café.wav" if conditional else "café.flac"),
                      "-c:a", "pcm_s16le" if conditional else "flac")
    output = tmp_path / "out"
    output.mkdir()
    opts = options(tmp_path, source, command="tags_plan", overwrite=True, out_dir=output,
                   format="wav", wav_tags="info", on_metadata_loss="sidecar", composer="Test Composer")
    reference = run_plan(opts)
    plans = all_plans(reference)
    assert plans and bool(reference.outcomes[0].conditional_plans) is conditional
    members = [member for plan in plans for member in plan.members if member.purpose == purpose]
    if not members:
        pytest.fail(f"fixture has no {purpose} member: {plans}")
    destination = alias(members[0].final_path, name, tmp_path)
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError, match="also planned"):
        run_plan(opts, report=destination, callback=rows.append)
    assert rows == [] and ledger(tmp_path) == before and residue(tmp_path) == []


def test_safe_report_retains_all_reservations_and_is_the_only_publication(tmp_path):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    opts = options(tmp_path, source, command="tags_plan", wav_tags="info", on_metadata_loss="sidecar",
                   composer="Test Composer")
    destination = tmp_path / "plan.csv"
    before = ledger(tmp_path)
    rows = []
    run = run_plan(opts, report=destination, callback=rows.append)
    assert len(rows) == len(run.inspections) == 1 and run.outputs is not None
    reserved = {d.final_path for d in run.outputs.write_set.destinations if d.input_index is not None}
    assert reserved == {m.final_path for plan in all_plans(run) for m in plan.members}
    assert ledger(tmp_path) == before and not (source.parent / "Normalised").exists()
    assert run.outputs.publish("report", b"checked report\n") == destination
    assert destination.read_bytes() == b"checked report\n"
    assert set(ledger(tmp_path)) - set(before) == {"plan.csv"}
    assert residue(tmp_path) == []


def test_metadata_reads_stay_protected_when_resolution_fails(tmp_path):
    source = generate(tmp_path / "in" / "take.flac", "-c:a", "flac", tags={"artist": "Source"})
    cover = png(tmp_path / "cover.png")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("file,artwork\nin/take.flac,cover.png\n")
    opts = options(tmp_path, source, command="tags_plan", manifest=manifest,
                   artist="Conflict", strict_conflicts=True, overwrite=True)
    reference = run_plan(opts)
    assert reference.outcomes[0].reason_code == "metadata_conflict_strict"
    rows = []
    before = ledger(tmp_path)
    with pytest.raises(UsageError, match="metadata.*reads"):
        run_plan(opts, report=cover, callback=rows.append)
    assert rows == [] and ledger(tmp_path) == before


def test_cancelled_plan_still_checks_manifest_before_returning_report(tmp_path):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("file,title\nin/take.wav,Demo\n")
    opts = options(tmp_path, source, command="tags_plan", manifest=manifest, overwrite=True)
    cancel = threading.Event()
    cancel.set()
    before = ledger(tmp_path)
    with pytest.raises(UsageError, match="metadata.*reads"):
        run_plan(opts, report=manifest, cancel=cancel)
    run = run_plan(opts, report=tmp_path / "plan.csv", cancel=cancel)
    assert run.cancelled and run.not_started == 1 and not run.inspections and run.outputs is not None
    assert ledger(tmp_path) == before


@pytest.mark.parametrize("kind", ["source", "config"])
@pytest.mark.parametrize("name", ["hardlink", "symlink"])
def test_template_uses_the_same_read_alias_barrier(tmp_path, kind, name):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    config = tmp_path / "config.toml"
    config.write_text("target = -16\n")
    destination = alias(source if kind == "source" else config, name, tmp_path)
    opts = options(tmp_path, source, command="tags_template", config=config,
                   template_output=destination, overwrite=True)
    rows = []
    before = ledger(tmp_path)
    with pytest.raises(UsageError):
        pipeline.template_tags_items(pipeline.discover_inputs(opts).items, opts,
                                     pipeline.locate_tools(opts, need_ffmpeg=False), on_inspection=rows.append)
    assert rows == [] and ledger(tmp_path) == before


def test_later_input_read_is_checked_before_even_the_first_callback(tmp_path):
    first = generate(tmp_path / "in" / "first.wav", "-c:a", "pcm_s16le")
    second = generate(tmp_path / "in" / "second.wav", "-c:a", "pcm_s16le")
    cover = png(tmp_path / "second.png")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("file,artwork\nin/second.wav,second.png\n")
    opts = options(tmp_path, first, second, command="tags_plan", manifest=manifest, overwrite=True)
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError):
        run_plan(opts, report=cover, callback=rows.append)
    assert rows == [] and ledger(tmp_path) == before


@pytest.mark.parametrize("purpose", ["cover_export", "picture_export"])
def test_report_cannot_take_a_planned_picture_export(tmp_path, purpose):
    from tests.unit.core.test_pipeline_switches import pictured_source

    source, _ = pictured_source(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    opts = options(tmp_path, source, command="tags_plan", format="wav", wav_tags="info",
                   out_dir=output, on_metadata_loss="sidecar", overwrite=True)
    reference = run_plan(opts)
    member = next(m for plan in all_plans(reference) for m in plan.members if m.purpose == purpose)
    before = ledger(tmp_path)
    rows = []
    with pytest.raises(UsageError, match="also planned"):
        run_plan(opts, report=member.final_path, callback=rows.append)
    assert rows == [] and ledger(tmp_path) == before


def test_publication_rechecks_a_report_that_becomes_a_read_alias(tmp_path, allow_linked_names):
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("file,title\nin/take.wav,Demo\n")
    opts = options(tmp_path, source, command="tags_plan", manifest=manifest, overwrite=True)
    destination = tmp_path / "report.csv"
    run = run_plan(opts, report=destination)
    assert run.outputs is not None
    destination.hardlink_to(manifest)
    before = ledger(tmp_path)
    with pytest.raises(pipeline.RunOutputError):
        run.outputs.publish("report", b"must not replace metadata\n")
    assert ledger(tmp_path) == before and residue(tmp_path) == []
