"""TK-0432: hermetic read-only verify orchestration and executor seams."""
from __future__ import annotations

import threading
import json
import hashlib
import math
import wave
from array import array
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from norma import pipeline
from norma import batch
from norma.errors import Cancelled, FileFailure, SkipFile, UsageError, EnvironmentError_
from norma.metadata import artwork
from norma.models import Artwork, InputItem, LoudnessMeasurement, ProbeResult, RawNativeItem, RunOptions, SourceTags, TagValue


def measurement(integrated=-16.0, peak=-6.0, **changes):
    return replace(LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=2.0, threshold_lufs=None,
        integrated_valid=True, peak_valid=True, lra_valid=True, threshold_valid=False,
        integrated_resolution_lu=.001, peak_resolution_linear=.001, dual_mono=False,
        meter="ebur128", value_source="metadata", ffmpeg_version="synthetic",
        decoded_duration_s=20.0, decoded_frames=960000, sample_rate_hz=48000,
        decoded_nonzero=True, terminal_peak_covered=True), **changes)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    paths = [tmp_path / f"take-{i}.wav" for i in range(4)]
    for path in paths:
        path.write_bytes(b"synthetic audio identity")
    items = tuple(InputItem(p, tmp_path, PurePosixPath(p.name),
                            (p.stat().st_dev, p.stat().st_ino), i, True)
                  for i, p in enumerate(paths))
    probe = ProbeResult(
        codec="pcm_s24le", codec_class="lossless", container="wav", audio_stream_index=0,
        ignored_streams=0, sample_rate_hz=48000, bit_depth=24, sample_float=False,
        sample_format="s32", channels=2, channel_layout="stereo", duration_s=20.0,
        duration_basis="stream", bitrate_bps=2304000, has_artwork=False, chapters=0,
        ffprobe_tags={}, ffprobe_version="synthetic")
    tags = SourceTags(fields={}, extras={}, artwork=(), non_tag_items=(), internal_conflicts=(),
                      native_inventory=(), stores={})
    tool = tmp_path / "tool"
    tool.write_bytes(b"synthetic executable identity")
    tools = pipeline.ReadOnlyTools(SimpleNamespace(path=tool), SimpleNamespace(path=tool))
    opts = RunOptions(command="verify", ffmpeg=tool, cwd=tmp_path, jobs=1)
    calls = []
    def classify(item, prober, **kwargs):
        calls.append(("classify", item.input_index))
        return probe
    monkeypatch.setattr(pipeline, "classify", classify)
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *args: tags)
    monkeypatch.setattr(pipeline, "measure", lambda *args, **kwargs: measurement())
    monkeypatch.setattr(pipeline, "measure_candidate", lambda *args, **kwargs: measurement())
    return SimpleNamespace(items=items, options=opts, tools=tools, tags=tags, calls=calls)


def test_verify_measures_output_without_planning_or_writing(rig, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("verify must not prepare, plan, render or allocate")
    for name in ("plan_gain", "plan_dynamic_preview", "render", "collect_evidence", "MetadataInputs"):
        monkeypatch.setattr(pipeline, name, forbidden)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in rig.options.cwd.rglob("*") if p.is_file()}
    run = pipeline.verify_items(rig.items[:1], rig.options, rig.tools)
    assert not run.cancelled and run.not_started == 0 and run.outputs is None
    inspection, = run.inspections
    outcome = inspection.outcome
    assert (outcome.status, outcome.reason_code) == ("ok", "ok")
    assert outcome.measurement_out == measurement() and outcome.measurement_in is None
    assert outcome.verification_reference.target_origin == "default"
    assert outcome.verification.loudness_ok and outcome.verification.peak_ok
    assert outcome.execution == "inspect" and outcome.published_paths == ()
    assert outcome.output is None and outcome.gain is None and outcome.target is None
    assert outcome.metadata is None and outcome.tag_write is None
    assert inspection.mark.validity == "absent" and inspection.source_tags == rig.tags
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in rig.options.cwd.rglob("*") if p.is_file()}


@pytest.mark.parametrize("peak,loudness,primary", [(-.5, -19, "verification_peak"),
                                                  (-6, -19, "verification_loudness"),
                                                  (-6, -16, "verification_tags")])
def test_tags_mismatch_and_invalid_mark_survive_audio_precedence(rig, monkeypatch, peak, loudness, primary):
    tags = replace(rig.tags, native_inventory=(RawNativeItem(
        "store:vorbis:NORMA", "vorbis", "NORMA", "VORBIS_COMMENT", None,
        "dn-raw-1", b"broken", "synthetic"),), stores={"vorbis": ("store:vorbis:NORMA",)})
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *args: tags)
    monkeypatch.setattr(pipeline, "measure_candidate", lambda *args, **kwargs: measurement(loudness, peak))
    opts = replace(rig.options, check_tags=True, title="expected", explicit_fields={"title"},
                   value_sources={"title": "cli"})
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    assert record.outcome.reason_code == primary and record.outcome.status == "failed"
    assert "verification_tags" in [d.code for d in record.diagnostics]
    assert "processing_mark_invalid" in [d.code for d in record.outcome.warnings]
    assert record.tag_mismatches == ("field:title",)
    ref, = record.tag_references
    assert (ref.name, ref.cell.value, ref.origin, ref.locator) == ("title", "expected", "cli", "--title")


def test_reference_layers_ignore_render_config_inference_and_merge_extras(rig):
    manifest = rig.options.cwd / "refs.json"
    manifest.write_text(json.dumps({"version": 1, "defaults": {
        "album": "default", "track_total": 12, "extras": {"Mood": "calm", "Other": "kept"}},
        "files": {rig.items[0].source_path.name: {"album": None,
                   "track": 3, "extras": {"mOOD": "row"}}}}))
    opts = replace(rig.options, check_tags=True, manifest=manifest, title="CLI", artist="config",
                   explicit_fields={"title"}, value_sources={"title": "cli", "artist": "config"},
                   tag=(("Ignore", "me"),), clear_tag=("album",), number_tracks=True,
                   infer_from_filename="{title}", folder_artwork=True)
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    refs = {ref.name: ref for ref in record.tag_references}
    assert set(refs) == {"album", "track_total", "track", "mOOD", "Other", "title"}
    assert refs["album"].cell.operation == "clear"
    assert refs["track"].cell.value == 3 and refs["track_total"].cell.value == 12
    assert refs["mOOD"].cell.value == "row" and refs["mOOD"].origin == "manifest"
    assert refs["Other"].origin == "manifest_default"
    assert refs["title"].origin == "cli"


def test_missing_mark_is_only_a_failure_with_check_tags(rig):
    record, = pipeline.verify_items(rig.items[:1], replace(rig.options, check_tags=True), rig.tools).inspections
    assert record.outcome.reason_code == "verification_tags"
    assert record.primary.message == "processing mark missing"


def artwork_manifest(rig, value="cover.png"):
    path = rig.options.cwd / "refs.json"
    path.write_text(json.dumps({"version": 1, "files": {
        item.source_path.name: {"artwork": value} for item in rig.items}}))
    return replace(rig.options, manifest=path, check_tags=True)


def fake_artwork_reader(monkeypatch, rig):
    path = rig.options.cwd / "cover.png"
    path.write_bytes(b"synthetic image evidence")
    calls = []
    monkeypatch.setattr(pipeline, "locate_sips_tool", lambda: rig.tools.ffmpeg)
    def read(path, tool, **kwargs):
        calls.append((path, tool, kwargs))
        path = path.resolve()
        info = path.stat()
        return artwork.ArtworkReference(path.read_bytes(), artwork.SourceStamp(
            path, (info.st_dev, info.st_ino), info.st_size, info.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(artwork, "read_artwork_reference", read)
    return path, calls


def test_artwork_reference_is_absolute_validated_once_protected_and_rechecked(rig, monkeypatch):
    path, calls = fake_artwork_reader(monkeypatch, rig)
    opts = artwork_manifest(rig)
    rechecked = []
    original = artwork.recheck_artwork_reference
    def recheck(reference):
        rechecked.append(reference.stamp.path)
        original(reference)
    monkeypatch.setattr(artwork, "recheck_artwork_reference", recheck)
    event = threading.Event()
    run = pipeline.verify_items(rig.items[:2], opts, rig.tools, cancel_event=event,
                                report=pipeline.report_request(opts.cwd / "report.json"))
    assert len(calls) == 1 and calls[0][0] == path
    assert calls[0][2]["cancel_event"] is event
    assert rechecked == [path, path]
    assert all(r.tag_references[0].cell.value == path for r in run.inspections)
    assert any(read.path == path for read in run.outputs.write_set.protected_reads)
    assert not (opts.cwd / "report.json").exists()


def test_report_aliases_artwork_even_with_failed_reference_before_any_row(rig, monkeypatch):
    path, _ = fake_artwork_reader(monkeypatch, rig)
    opts = replace(artwork_manifest(rig), overwrite=True)
    monkeypatch.setattr(artwork, "read_artwork_reference", lambda *a, **k: (_ for _ in ()).throw(
        FileFailure("artwork_invalid", "content_verified", "bad image")))
    rows = []
    with pytest.raises(UsageError):
        pipeline.verify_items(rig.items, opts, rig.tools, report=pipeline.report_request(path),
                              on_inspection=rows.append)
    assert not rows and not rig.calls and path.read_bytes() == b"synthetic image evidence"


def test_changed_artwork_cached_evidence_fails_file(rig, monkeypatch):
    path, _ = fake_artwork_reader(monkeypatch, rig)
    opts = artwork_manifest(rig)
    def change(*args, **kwargs):
        path.write_bytes(b"changed reference")
        return measurement()
    monkeypatch.setattr(pipeline, "measure_candidate", change)
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    assert record.outcome.reason_code == "source_changed"


@pytest.mark.parametrize("value", [None, "missing.png"])
def test_clear_needs_no_sips_and_missing_reference_is_file_local(rig, monkeypatch, value):
    opts = artwork_manifest(rig, value)
    if value is None:
        monkeypatch.setattr(pipeline, "locate_sips_tool", lambda: pytest.fail("clear needs no sips"))
    else:
        monkeypatch.setattr(pipeline, "locate_sips_tool", lambda: rig.tools.ffmpeg)
    run = pipeline.verify_items(rig.items[:2], opts, rig.tools)
    assert len(run.inspections) == 2
    assert {r.outcome.reason_code for r in run.inspections} == {
        "verification_tags" if value is None else "artwork_invalid"}


def execute(request):
    control = batch.BatchControl(commit_lock=request.commit_lock,
                                 cancel_event=request.cancel_event, force_hook=request.force_hook)
    return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                               jobs=request.jobs, control=control, on_outcome=request.on_outcome,
                               handle_sigint=False)


@pytest.mark.parametrize("command", ["analyse", "verify"])
def test_injected_batch_uses_jobs_completion_callbacks_and_input_order(rig, monkeypatch, command):
    rendezvous = threading.Barrier(2, timeout=5)
    second_finished = threading.Event()
    rows = []
    active = 0
    maximum = 0
    lock = threading.Lock()
    def measure(path, *args, **kwargs):
        nonlocal active, maximum
        assert kwargs["cancel_event"] is event
        with lock:
            active += 1
            maximum = max(maximum, active)
        rendezvous.wait()
        if path == rig.items[0].source_path:
            assert second_finished.wait(5)
        with lock:
            active -= 1
        return measurement()
    def callback(record):
        assert threading.current_thread() is threading.main_thread()
        rows.append(record)
        if record.outcome.input.input_index == 1:
            second_finished.set()
    monkeypatch.setattr(pipeline, "measure", measure)
    monkeypatch.setattr(pipeline, "measure_candidate", measure)
    event = threading.Event()
    opts = replace(rig.options, command=command, jobs=2)
    run = getattr(pipeline, command + "_items")(
        rig.items[:2], opts, rig.tools, execute=execute, cancel_event=event, on_inspection=callback)
    assert maximum == 2
    assert [r.outcome.input.input_index for r in rows] == [1, 0]
    assert [r.outcome.input.input_index for r in run.inspections] == [0, 1]
    assert rows[0] is run.inspections[1] and rows[1] is run.inspections[0]


@pytest.mark.parametrize("command", ["analyse", "verify"])
def test_batch_cancellation_counts_only_started_and_keeps_partial_evidence(rig, monkeypatch, command):
    event = threading.Event()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(pipeline, "measure", interrupt)
    monkeypatch.setattr(pipeline, "measure_candidate", interrupt)
    opts = replace(rig.options, command=command, jobs=1)
    run = getattr(pipeline, command + "_items")(
        rig.items, opts, rig.tools, execute=execute, cancel_event=event)
    assert run.cancelled and run.not_started == 3
    record, = run.inspections
    assert record.outcome.reason_code == "cancelled" and record.outcome.probe is not None
    assert record.outcome.execution == "inspect"


def marked(tags, text):
    native = RawNativeItem("store:vorbis:NORMA", "vorbis", "NORMA", "VORBIS_COMMENT", None,
                           "dn-raw-1", text.encode(), "synthetic")
    return replace(tags, native_inventory=(native,), stores={"vorbis": (native.identity,)})


MARK = ("v=1;tool=norma-0.1.0;target=-18;tp=-2;mode=linear;gain=+0.00;out_i=-18.00;"
        "out_tp=-2.00;origin=flac;ffmpeg=N-synthetic;date=2026-10-05")


@pytest.mark.parametrize("text,source,target,origin", [
    (MARK, "config", -18, "mark"), (MARK, "cli", -16, "cli"),
    ("broken", "config", -16, "config"), ("v=2", "default", -16, "default")])
def test_reference_policy_and_invalid_mark_warning_without_tag_check(rig, monkeypatch, text, source, target, origin):
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *a: marked(rig.tags, text))
    monkeypatch.setattr(pipeline, "measure_candidate", lambda *a, **k: measurement(target))
    opts = replace(rig.options, value_sources={"target": source, "true_peak": "cli", "tolerance": "config"})
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    ref = record.outcome.verification_reference
    assert (ref.target_lufs, ref.target_origin) == (target, origin)
    assert (ref.ceiling_dbtp, ref.ceiling_origin, ref.tolerance_origin) == (-1, "cli", "config")
    assert record.outcome.reason_code == ("ok" if text == MARK else "processing_mark_invalid")
    assert record.outcome.origin_codec == ("flac" if text == MARK else None)


def test_marked_reduce_uses_fresh_shortfall_without_gain_or_prediction(rig, monkeypatch):
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *a: marked(rig.tags, MARK.replace("linear", "reduce")))
    monkeypatch.setattr(pipeline, "measure_candidate", lambda *a, **k: measurement(-23))
    outcome, = pipeline.verify_items(rig.items[:1], rig.options, rig.tools).outcomes
    assert outcome.reason_code == "reduce_shortfall" and outcome.status == "warn"
    assert outcome.verification.measured_shortfall_lu == 5
    assert outcome.verification.expected_lufs is None and outcome.gain is None


@pytest.mark.parametrize("kind,reason", [("measurement_error", "verification_loudness"),
    ("decode_error", "decode_error"), ("timeout", "timeout"), ("permission_denied", "permission_denied"),
    ("source_changed", "source_changed"), ("silent", "verification_loudness")])
def test_measurement_translation_and_mark_diagnostics_survive_failures(rig, monkeypatch, kind, reason):
    from norma.audio import verify
    def fail(*args, **kwargs):
        if kind == "silent":
            raise SkipFile(kind, "measure", "no signal")
        raise FileFailure(kind, "measure", "exact tool detail")
    monkeypatch.setattr(pipeline, "measure_candidate", verify.measure_candidate)
    monkeypatch.setattr(verify.meter, "measure", fail)
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *a: marked(rig.tags, "broken"))
    record, = pipeline.verify_items(rig.items[:1], rig.options, rig.tools).inspections
    assert record.outcome.reason_code == reason and record.primary.stage == "verify"
    assert "processing_mark_invalid" in [d.code for d in record.diagnostics]
    assert "processing_mark_invalid" in [d.code for d in record.outcome.warnings]


@pytest.mark.parametrize("target", ["input", "manifest", "config", "tool", "artwork", "missing_artwork"])
@pytest.mark.parametrize("injected", [False, True])
def test_entire_report_barrier_precedes_executor_callbacks_and_files(rig, monkeypatch, target, injected):
    image, _ = fake_artwork_reader(monkeypatch, rig)
    opts = artwork_manifest(rig, "missing.png" if target == "missing_artwork" else "cover.png")
    config = opts.cwd / "settings.toml"
    config.write_text("# synthetic config")
    opts = replace(opts, overwrite=True, config=config)
    paths = {"input": rig.items[-1].source_path, "manifest": opts.manifest, "config": config,
             "tool": rig.tools.ffmpeg.path, "artwork": image, "missing_artwork": opts.cwd / "missing.png"}
    def no_execute(request):
        pytest.fail("barrier must precede executor")
    rows = []
    with pytest.raises(UsageError):
        pipeline.verify_items(rig.items, opts, rig.tools, report=pipeline.report_request(paths[target]),
                              execute=no_execute if injected else None, on_inspection=rows.append)
    assert not rig.calls and not rows


def test_report_is_only_written_by_explicit_publish(rig):
    path = rig.options.cwd / "report.json"
    run = pipeline.verify_items(rig.items[:1], rig.options, rig.tools,
                                report=pipeline.report_request(path))
    assert not path.exists()
    assert run.outputs.publish("report", b'{"verified":true}\n') == path
    assert path.read_bytes() == b'{"verified":true}\n'


def test_sips_missing_is_environment_failure_before_rows(rig, monkeypatch):
    opts = artwork_manifest(rig)
    def missing():
        raise EnvironmentError_("sips missing")
    monkeypatch.setattr(pipeline, "locate_sips_tool", missing)
    with pytest.raises(EnvironmentError_, match="sips missing"):
        pipeline.verify_items(rig.items, opts, rig.tools)
    assert not rig.calls


def test_manifest_whole_error_vs_matched_row_failure(rig):
    path = rig.options.cwd / "bad.csv"
    path.write_text("not,a,manifest\n")
    opts = replace(rig.options, check_tags=True, manifest=path)
    with pytest.raises(UsageError):
        pipeline.verify_items(rig.items, opts, rig.tools)
    assert not rig.calls
    path.write_text("file,title,track\ntake-0.wav,x,not-a-number\ntake-1.wav,right,2\n")
    run = pipeline.verify_items(rig.items[:2], opts, rig.tools)
    assert [r.outcome.reason_code for r in run.inspections] == ["manifest_row_invalid", "verification_tags"]


def test_changed_source_during_measurement_keeps_measurement_but_fails(rig, monkeypatch):
    def change(path, *args, **kwargs):
        path.write_bytes(b"changed source")
        return measurement()
    monkeypatch.setattr(pipeline, "measure_candidate", change)
    record, = pipeline.verify_items(rig.items[:1], rig.options, rig.tools).inspections
    assert record.outcome.reason_code == "source_changed" and record.outcome.measurement_out == measurement()


@pytest.mark.parametrize("injected", [False, True])
def test_cancel_before_start_has_no_fabricated_rows(rig, injected):
    event = threading.Event()
    event.set()
    run = pipeline.verify_items(rig.items, rig.options, rig.tools, cancel_event=event,
                                execute=execute if injected else None)
    assert run.cancelled and run.not_started == len(rig.items) and run.inspections == ()
    assert not rig.calls


def test_cancel_during_reference_preflight_starts_no_audio(rig, monkeypatch):
    opts = artwork_manifest(rig)
    event = threading.Event()
    monkeypatch.setattr(pipeline, "locate_sips_tool", lambda: rig.tools.ffmpeg)
    def cancel(*args, **kwargs):
        assert kwargs["cancel_event"] is event
        raise Cancelled()
    monkeypatch.setattr(artwork, "read_artwork_reference", cancel)
    run = pipeline.verify_items(rig.items, opts, rig.tools, cancel_event=event)
    assert run.cancelled and run.not_started == len(rig.items) and run.inspections == ()
    assert not rig.calls


def test_real_synthetic_wav_verify_is_read_only(tmp_path):
    """Actual adapters/meter/readers, no library or real-demo access."""
    ffmpeg = Path("/usr/local/bin/ffmpeg")
    if not ffmpeg.is_file() or not ffmpeg.with_name("ffprobe").is_file():
        pytest.skip("local FFmpeg pair unavailable")
    path = tmp_path / "synthetic.wav"
    samples = array("h", (round(4000 * math.sin(2 * math.pi * 997 * i / 48000))
                          for i in range(48000 * 4)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        writer.writeframes(samples.tobytes())
    opts = RunOptions(command="analyse", ffmpeg=ffmpeg, cwd=tmp_path, inputs=(path,), jobs=1)
    tools = pipeline.locate_tools(opts, need_ffmpeg=True)
    found = pipeline.discover_inputs(opts)
    analysed, = pipeline.analyse_items(found.items, opts, tools).inspections
    target = analysed.outcome.measurement_in.integrated_lufs
    opts = replace(opts, command="verify", target=target, value_sources={"target": "cli"})
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if p.is_file()}
    run = pipeline.verify_items(found.items, opts, tools, execute=execute)
    record, = run.inspections
    assert record.outcome.status == "ok", record.primary
    assert record.outcome.measurement_out.integrated_lufs == target
    assert record.outcome.verification_reference.target_origin == "cli"
    assert record.outcome.execution == "inspect" and record.outcome.published_paths == ()
    assert before == {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                      for p in tmp_path.rglob("*") if p.is_file()}


def test_manifest_artwork_parent_components_match_validated_absolute_cell(rig, monkeypatch):
    path, _ = fake_artwork_reader(monkeypatch, rig)
    folder = rig.options.cwd / "references"
    folder.mkdir()
    manifest = folder / "refs.json"
    manifest.write_text(json.dumps({"version": 1, "files": {
        "../" + rig.items[0].source_path.name: {"artwork": "../cover.png"}}}))
    opts = replace(rig.options, check_tags=True, manifest=manifest)
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    assert record.outcome.reason_code == "verification_tags", record.primary
    assert record.tag_references[0].cell.value == path


@pytest.mark.parametrize("command", ["analyse", "verify"])
def test_cancel_after_last_callback_is_truthful_without_rewriting_success(rig, command):
    event = threading.Event()
    run = getattr(pipeline, command + "_items")(
        rig.items[:1], replace(rig.options, command=command), rig.tools, cancel_event=event,
        on_inspection=lambda record: event.set())
    assert run.cancelled and run.not_started == 0
    assert run.inspections[0].outcome.status == "ok"


@pytest.mark.parametrize("command", ["analyse", "verify"])
def test_adapter_cancellation_propagates_to_batch_admission(rig, monkeypatch, command):
    def cancel(*args, **kwargs):
        raise Cancelled()
    monkeypatch.setattr(pipeline, "measure", cancel)
    monkeypatch.setattr(pipeline, "measure_candidate", cancel)
    run = getattr(pipeline, command + "_items")(
        rig.items, replace(rig.options, command=command), rig.tools, execute=execute)
    assert run.cancelled and run.not_started == 3
    record, = run.inspections
    assert record.outcome.reason_code == "cancelled"


@pytest.mark.parametrize("fronts,other,expected", [(1, True, "ok"), (0, True, "verification_tags"),
                                                 (2, False, "verification_tags")])
def test_pipeline_passes_exact_artwork_evidence_to_pure_checker(rig, monkeypatch, fronts, other, expected):
    path, _ = fake_artwork_reader(monkeypatch, rig)
    pictures = [Artwork(path.read_bytes(), "image/png", 1, 1, 3, "") for _ in range(fronts)]
    if other:
        pictures.append(Artwork(b"other picture", "image/png", 1, 1, 4, ""))
    tags = marked(replace(rig.tags, artwork=tuple(TagValue(p, "vorbis", "pic") for p in pictures)), MARK)
    monkeypatch.setattr(pipeline, "read_source_tags", lambda *a: tags)
    monkeypatch.setattr(pipeline, "measure_candidate", lambda *a, **k: measurement(-18))
    record, = pipeline.verify_items(rig.items[:1], artwork_manifest(rig), rig.tools).inspections
    assert record.outcome.reason_code == expected


def test_no_tag_check_does_not_load_unused_manifest_or_prepare_artwork(rig, monkeypatch):
    opts = replace(rig.options, manifest=rig.options.cwd / "absent.json", artwork=rig.options.cwd / "absent.png",
                   artist="not explicit", value_sources={"artist": "config"})
    monkeypatch.setattr(pipeline, "locate_sips_tool", lambda: pytest.fail("no effective artwork"))
    monkeypatch.setattr(pipeline, "_write_edge", lambda: pytest.fail("no tag check"))
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    assert record.outcome.status == "ok" and not record.tag_references


def test_cli_pair_components_stay_typed_and_point_to_actual_option(rig):
    opts = replace(rig.options, check_tags=True, track=3, track_total=12, disc=1, disc_total=2,
                   explicit_fields={"track", "disc"},
                   value_sources={"track": "cli", "track_total": "cli", "disc": "cli", "disc_total": "cli"})
    record, = pipeline.verify_items(rig.items[:1], opts, rig.tools).inspections
    refs = {r.name: r for r in record.tag_references}
    assert refs["track_total"].cell.value == 12 and refs["track_total"].locator == "--track"
    assert refs["disc_total"].cell.value == 2 and refs["disc_total"].locator == "--disc"
