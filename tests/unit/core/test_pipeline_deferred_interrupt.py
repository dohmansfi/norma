"""CR-102: real deferred SIGINT across pipeline caching and safety settlement.

Planning, the audio ladder and tag writing are synthetic, not FFmpeg evidence.
The pipeline worker/execution, batch signal handler and safety namespace operations
are real; every file stays in the hermetic tmp_path sandbox.
"""

from __future__ import annotations

import errno
import json
import signal
import threading
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from norma import batch, pipeline, report, safety
from norma.errors import FileFailure
from norma.models import (
    BundleMember,
    Diagnostic,
    GainPlan,
    InputItem,
    LoudnessMeasurement,
    OutputPlan,
    ProbeResult,
    RenderResult,
    RunOptions,
    TargetFormat,
    Verification,
)


@pytest.fixture
def planned(tmp_path, monkeypatch, allow_linked_names):
    """One preplanned bundle at the audio/tag edge, with real safety reservations."""
    root = tmp_path.resolve()
    source = root / "take.wav"
    source.write_bytes(b"synthetic unchanged source, not audio")
    info = source.stat()
    item = InputItem(source, root, PurePosixPath(source.name),
                     (info.st_dev, info.st_ino), 0, True)
    options = RunOptions(command="normalise", ffmpeg=root / "unused-ffmpeg", cwd=root)
    state = pipeline._NormaliseRun(
        [item], options, None, None, threading.Event(), "deferred", None)
    job = state.jobs[0]
    job.snapshot = safety.snapshot_source(item)
    job.probe = ProbeResult(
        codec="pcm_s16le", codec_class="lossless", container="wav", audio_stream_index=0,
        ignored_streams=0, sample_rate_hz=48000, bit_depth=16, sample_float=False,
        sample_format="s16", channels=2, channel_layout="stereo", duration_s=4.0,
        duration_basis="stream", bitrate_bps=1536000, has_artwork=False, chapters=0,
        ffprobe_tags={}, ffprobe_version="synthetic")
    job.measurement = LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=-16.0, true_peak_dbtp=-2.0,
        true_peak_linear=0.79, lra_lu=1.0, threshold_lufs=None, integrated_valid=True,
        peak_valid=True, lra_valid=True, threshold_valid=False, integrated_resolution_lu=0.1,
        peak_resolution_linear=1e-6, dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic", decoded_duration_s=4.0, decoded_frames=192000,
        sample_rate_hz=48000, decoded_nonzero=True, terminal_peak_covered=True)
    job.gain = GainPlan(
        action="linear", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0,
        tolerance_lu=0.5, max_limit_db=3.0, gain_db=1.0, predicted_true_peak_dbtp=-2.0,
        limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
        expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None,
        loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0, reason_code="ok")
    job.target = TargetFormat(
        format="wav", extension=".wav", encoder="pcm_s16le", bit_depth=16, sample_float=False,
        sample_rate_hz=48000, channels=2, channel_layout="stereo", dither="none",
        bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
        vbr_quality=None, bitrate_source="not_applicable", capability_basis="synthetic",
        stream_copy=False, decision="as_requested", notes=("target note",))
    job.meta_notes = ("metadata note",)
    warning = Diagnostic("metadata_loss_warn", "synthetic metadata warning", "metadata_planned")
    job.meta_warnings.append(warning)
    folder = root / "output"
    folder.mkdir()
    members = tuple(BundleMember(name, purpose, None, folder / filename, None, str(folder / filename))
                    for name, purpose, filename in (
                        ("tags", "tags_sidecar", "take.wav.tags.json"),
                        ("audio", "audio", "take.wav")))
    old = {member.final_path: f"old {member.member_id}".encode() for member in members}
    for path, data in old.items():
        path.write_bytes(data)
    job.output = OutputPlan(1, members, "none", item.identity, True)
    state.protected = safety.protect_inputs([item])
    state.checked = safety.preflight({0: job.output}, (), state.protected, overwrite=True)
    state.registry = safety.RunRegistry("deferred", umask=0o022)
    job.allocated = state.registry.allocate_plan(0, job.output)
    payloads = {}
    for member in job.allocated.members:
        payloads[member.final_path] = f"complete synthetic {member.member_id}".encode()
        member.temporary_path.write_bytes(payloads[member.final_path])
    rendered = RenderResult(
        job.allocated.audio_member.temporary_path, 1, job.gain, job.target, item.identity,
        0.0, "synthetic render tail", None, True, None, None)
    verified = Verification(job.measurement, -16.0, 0.0, 0.0, 0.0, True, True, True)

    def ladder(measurement, gain, attempt, before_retry, **kwargs):
        assert measurement is job.measurement and gain is job.gain
        return SimpleNamespace(plan=gain, result=rendered, verification=verified, warnings=())

    def tags(current, cancel):
        assert current is job and not cancel.is_set()
        return ("tags",)

    monkeypatch.setattr(pipeline, "run_ladder", ladder)
    monkeypatch.setattr(state, "_write_tags", tags)
    # Deterministic publication note; no claim about the host volume's durability.
    monkeypatch.setattr(safety, "_fsync_directory", lambda path: "synthetic publication note")
    outputs = pipeline.RunOutputs(state.checked, state.protected, state.registry, overwrite=True, requests=())
    return SimpleNamespace(state=state, job=job, outputs=outputs, old=old, payloads=payloads,
                           warning=warning, verified=verified, rendered=rendered)


@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.parametrize("phase", ["cleanup", "rollback", "success"])
@pytest.mark.parametrize("interrupt", [True, False], ids=["sigint", "uninterrupted"])
@pytest.mark.parametrize("warn", [True, False], ids=["warning", "no-warning"])
def test_deferred_interrupt_settlement_is_cached_before_callback(
        planned, monkeypatch, jobs, phase, interrupt, warn):
    bundle = planned
    state, job = bundle.state, bundle.job
    state.options = replace(state.options, jobs=jobs)
    if not warn:
        job.meta_warnings.clear()
    reached = threading.Event()
    release = threading.Event()
    captured = []
    seen = []
    retained = {}
    control = batch.BatchControl(commit_lock=state.registry.commit_lock, cancel_event=state.cancel_event)
    coordinator = threading.get_ident()
    previous_handler = signal.getsignal(signal.SIGINT)
    commit = safety.commit_bundle

    def capture(*args, **kwargs):
        assert job.record is None
        assert kwargs["cancel_event"] is control.cancel_event
        assert callable(kwargs["interruption_requested"])
        result = commit(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(safety, "commit_bundle", capture)

    def settlement_fault(at, member):
        if phase == "rollback" and (at, member) == ("publish", "audio"):
            raise FileFailure("commit_error", "committing", "specific publication refused")
        if (phase == "rollback" and (at, member) == ("rollback", "tags")) or (
                phase in ("cleanup", "success") and at == "cleanup"):
            reached.set()
            assert release.wait(5), "coordinator did not deliver SIGINT"
            assert control.interrupted == interrupt and not control.cancel_event.is_set()
            if phase == "cleanup":
                raise OSError(errno.EACCES, "specific cleanup refused")
            if phase == "rollback":
                raise OSError(errno.EACCES, "specific restore refused")

    state.registry.fault_hook = settlement_fault

    def progress(event):
        if event.kind == "started":
            try:
                assert reached.wait(5), "worker did not enter settlement"
                if interrupt:
                    signal.raise_signal(signal.SIGINT)
                assert control.interrupted == interrupt and not control.cancel_event.is_set()
            finally:
                release.set()

    def execute(request):
        def observe(outcome):
            assert threading.get_ident() == coordinator
            record = request.record_for(outcome)
            assert record is job.record and record.outcome is outcome
            seen.append(record)
            retained.update({path: path.read_bytes() for path in captured[0].retained_paths})
        return batch.execute_batch(
            request.entries, request.worker, settle_error=request.settle_error, jobs=request.jobs,
            control=control, on_progress=progress, on_outcome=observe, handle_sigint=True)

    run = state._execute(execute, [job], ("run note",), bundle.outputs)
    assert signal.getsignal(signal.SIGINT) == previous_handler
    assert run.cancelled == interrupt and control.cancel_event.is_set() == interrupt
    assert run.not_started == 0 and run.run_errors == ()
    assert len(run.records) == len(seen) == len(captured) == 1
    record = run.records[0]
    result = captured[0]
    assert record is seen[0] is job.record is state._record_for(record.outcome)
    success = ("warn", "metadata_loss_warn", "committed") if warn else ("ok", "ok", "committed")
    expected = (success if phase == "success" else
                ("failed", "commit_error" if phase == "rollback" else "cleanup_error", "publication_failed"))
    assert (record.outcome.status, record.outcome.reason_code, record.outcome.execution) == expected
    codes = (["backup_error"] if phase == "rollback" else []) + (
        ["cancelled"] if interrupt and phase != "success" else [])
    assert [d.code for d in record.diagnostics] == codes
    if phase == "success":
        assert record.primary is None and result.diagnostics == ()
        assert result.retained_paths == ()
    else:
        assert record.primary == result.diagnostics[0]
        assert record.diagnostics == result.diagnostics[1:]
        if phase == "cleanup":
            assert "specific cleanup refused" in record.primary.message
        else:
            assert record.primary.detail == "specific publication refused"
            assert "specific restore refused" in record.diagnostics[0].detail
        assert result.retained_paths and set(retained) == set(result.retained_paths)
        receipt = json.loads(next(data for path, data in retained.items() if "-receipt-" in path.name))
        assert receipt["source"] == str(job.item.source_path) and receipt["plan_revision"] == 1
        for member in receipt["members"]:
            preimage = Path(member["preimage"])
            if preimage in retained:
                assert retained[preimage] == bundle.old[Path(member["final"])]
    assert record.outcome.warnings == ((bundle.warning,) if warn else ())
    assert record.notes == ("target note", "metadata note", *result.notes)
    assert result.notes == (() if phase == "rollback" else ("synthetic publication note",))
    assert record.outcome.published_paths == result.published_paths
    published = ({job.allocated.members[0].final_path} if phase == "rollback" else set(bundle.payloads))
    assert set(result.published_paths) == published
    assert result.retained_failed_path is None and record.outcome.retained_failed_path is None
    assert record.outcome.verification is bundle.verified
    assert record.outcome.ffmpeg_stderr_tail == bundle.rendered.stderr_tail
    assert run.notes == ("run note",)
    # Public outcome projection stays truthful; the callback's complete local
    # record projection also carries primary/secondary diagnostics and notes.
    row = report.project_outcome(record.outcome, state.options)
    assert (row["status"], row["reason"], row["execution"]) == expected
    assert row["published_paths"] == [str(p) for p in result.published_paths]
    rendered = report.render_tsv(record.outcome, 1, primary=record.primary, notes=record.notes)
    assert rendered.startswith({"ok": "OK", "warn": "WARN", "failed": "FAIL"}[expected[0]] + "\t1/1\t")
    if phase == "rollback":
        assert "specific publication refused" in rendered
    projected = report.portable(dict(primary=record.primary, diagnostics=record.diagnostics, notes=record.notes))
    assert [d["code"] for d in projected["diagnostics"]] == codes
    assert projected["primary"] == report.portable(record.primary)
    assert projected["notes"] == list(record.notes)
    for path, data in bundle.payloads.items():
        assert path.read_bytes() == (data if path in published else bundle.old[path])
    safety.recheck_source(job.item, job.snapshot, rehash=True)
