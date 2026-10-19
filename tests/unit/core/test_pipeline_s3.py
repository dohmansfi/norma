"""TK-0419: the S3 ``normalise`` wiring (ARCHITECTURE § 4; audio §§ 4–5; formats F14; D-078–D-080).

Real-tool tests run the whole chain through ``pipeline.normalise_items`` and
``batch.execute_batch`` and check each published output independently: the
output is decoded by FFmpeg to 32-bit float and measured with the TESTING § 2
oracles (pyloudnorm integrated loudness; SciPy 4x Kaiser β12 true peak), apart
from the project's own meter. Fixtures F-05/le and F-05/hc come from the WS-07
catalogue and are copied into each test's own folder, so outputs never land
beside the session-shared fixture. Fault-injection tests replace only the
verification verdict (the real candidate is still rendered, probed and
measured), the C-04 finalisation or the candidate probe.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from norma import batch, pipeline
from norma.adapters import ffmpeg, ffprobe
from norma.audio.measure import conservative_peak_dbtp
from norma.audio.verify import MAX_RENDERS, assess
from norma.errors import FileFailure
from norma.models import RunOptions

REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
REAL_FFPROBE = Path("/usr/local/bin/ffprobe")
BITEXACT = ("-fflags", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1")
PEAK_LIMIT_DBTP = -0.9
"""c + 0.1 for the default ceiling −1 dBTP (audio T1)."""

real = pytest.mark.real_ffmpeg

_spec = importlib.util.spec_from_file_location(
    "norma_core_s3_fixtures", Path(__file__).parents[2] / "fixtures" / "make_fixtures.py")
assert _spec is not None and _spec.loader is not None
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
fixture_path = _fixtures.fixture_path  # the WS-07 catalogue, consumed without changing a shared conftest


@pytest.fixture(autouse=True)
def path_based_rmtree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evidence folder is a ``TemporaryDirectory``; see test_pipeline_doctor.py for why."""
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


@pytest.fixture(autouse=True)
def no_process_left() -> None:
    yield
    assert ffmpeg._ACTIVE == {} and ffprobe._ACTIVE == {}, "a tool call outlived its test"


# --- Helpers ----------------------------------------------------------------------------------


def generate(path: Path, *args: str, rate: int = 44100, seconds: float = 4, level_db: float = -25,
             channels: int = 2, source: str = "sine=frequency=997:sample_rate={rate}:duration={seconds}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", source.format(rate=rate, seconds=seconds), "-af", f"volume={level_db}dB",
                    "-ac", str(channels), *BITEXACT, *args, str(path)], check=True, timeout=60)
    return path


def copied(fixture: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(fixture, folder / fixture.name))


def with_stereo_layout(fixture: Path, folder: Path) -> Path:
    """The fixture's exact samples in a WAV that declares its stereo layout.

    The catalogue's F-05 WAVs carry no channel mask, so ffprobe reports no layout, and an
    unknown layout never borrows stereo lossy evidence (C-04): lossy targets need a
    declared layout. Only the header changes; the 24-bit samples are copied unchanged.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fixture.name
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(fixture),
                    "-map", "0:a:0", "-af", "aformat=channel_layouts=stereo", "-c:a", "pcm_s24le", *BITEXACT,
                    str(path)], check=True, timeout=60)
    return path


def options(tmp_path: Path, *inputs: Path, **changes) -> RunOptions:
    return replace(RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=tmp_path, inputs=tuple(inputs),
                              jobs=2), **changes)


def executor(controls: list | None = None):
    def execute(request: pipeline.BatchRequest) -> batch.BatchResult:
        control = batch.BatchControl(commit_lock=request.commit_lock, cancel_event=request.cancel_event,
                                     force_hook=request.force_hook)
        if controls is not None:
            controls.append(control)
        return batch.execute_batch(request.entries, request.worker, settle_error=request.settle_error,
                                   jobs=request.jobs, control=control, handle_sigint=False)

    return execute


def normalise(run_options: RunOptions, *, execute=None, **kwargs) -> pipeline.NormaliseRun:
    tools = pipeline.locate_normalise_tools(run_options)
    items = pipeline.discover_inputs(run_options).items
    if execute is None and not run_options.dry_run:
        execute = executor()
    return pipeline.normalise_items(items, run_options, tools, execute=execute, **kwargs)


def only(run: pipeline.NormaliseRun) -> pipeline.Normalised:
    assert len(run.records) == 1 and not run.cancelled and run.run_errors == (), run
    return run.records[0]


def residue(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(".norma-*"))


def counting_renders(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Wrap the pipeline's render: record each call's source, candidate and keyword options."""
    calls: list[dict] = []
    real_render = pipeline.render

    def wrapped(source, identity, probe_result, plan, target, candidate, env, evidence, **kwargs):
        calls.append({"source": source, "candidate": candidate, "plan": plan, "target": target, **kwargs})
        return real_render(source, identity, probe_result, plan, target, candidate, env, evidence, **kwargs)

    monkeypatch.setattr(pipeline, "render", wrapped)
    return calls


def oracle(path: Path, work: Path) -> tuple[float, float]:
    """Decode with FFmpeg to float WAV, then pyloudnorm I and SciPy 4x (Kaiser β12) true peak."""
    work.mkdir(parents=True, exist_ok=True)
    decoded = work / f"{path.stem}-decoded.wav"
    subprocess.run([str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                    "-map", "0:a:0", "-c:a", "pcm_f32le", str(decoded)], check=True, timeout=120)
    samples, rate = sf.read(decoded, dtype="float64", always_2d=True)
    decoded.unlink()
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.), padtype="constant")))
    return float(integrated), float(20 * np.log10(peak))


def assert_peak_safe(record: pipeline.Normalised, work: Path) -> tuple[float, float]:
    """The project's M7 bound and the independent oracle both at or below c + 0.1 dBTP."""
    outcome = record.outcome
    out = outcome.measurement_out
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= PEAK_LIMIT_DBTP
    integrated, peak = oracle(outcome.published_paths[0], work)
    assert peak <= PEAK_LIMIT_DBTP, peak
    return integrated, peak


# --- Pure wiring ------------------------------------------------------------------------------


def test_filters_follow_the_peak_policy() -> None:
    """ARCHITECTURE § 9: limit needs alimiter, dynamic loudnorm, reduce neither."""
    assert "alimiter" in pipeline.normalise_filters("limit")
    assert "loudnorm" not in pipeline.normalise_filters("limit")
    assert "loudnorm" in pipeline.normalise_filters("dynamic")
    assert "alimiter" not in pipeline.normalise_filters("dynamic")
    assert pipeline.normalise_filters("reduce") == pipeline.NORMALISE_FILTERS
    for mode in ("limit", "reduce", "dynamic"):
        assert set(pipeline.NORMALISE_FILTERS) <= set(pipeline.normalise_filters(mode))


@pytest.mark.parametrize("mode, options_wanted", [("limit", ("level", "latency")), ("reduce", ()),
                                                  ("dynamic", ())])
def test_tool_preflight_checks_the_policy_facilities(tmp_path: Path, monkeypatch, mode: str,
                                                     options_wanted: tuple) -> None:
    seen: dict = {}
    monkeypatch.setattr(pipeline, "locate_ffprobe", lambda path: "prober")
    monkeypatch.setattr(pipeline, "locate_ffmpeg", lambda path: "ffmpeg")
    monkeypatch.setattr(pipeline, "probe_environment", lambda executable: "environment")
    monkeypatch.setattr(pipeline, "check_requirements", lambda env, **kwargs: seen.update(kwargs))
    pipeline.locate_normalise_tools(options(tmp_path, peak_mode=mode))
    assert seen["filters"] == pipeline.normalise_filters(mode)
    assert tuple(seen["alimiter_options"]) == options_wanted


@real
@pytest.mark.parametrize("command", ["analyse", "normalise"])
@pytest.mark.parametrize("signal", ["quiet", "within", "silent", "unmeasurable"])
def test_dynamic_preview_adds_no_source_scan_or_files(tmp_path: Path, monkeypatch,
                                                     command: str, signal: str) -> None:
    """D-080/V8: only the normal meter scans; no loudnorm, render, candidate or output folder."""
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le", rate=48000,
                      source=("anullsrc=r={rate}:cl=stereo:d={seconds}" if signal == "silent"
                              else "sine=frequency=997:sample_rate={rate}:duration={seconds}"),
                      seconds=4 if signal != "unmeasurable" else 0.1)

    run_options = options(tmp_path, source, command=command, dry_run=command == "normalise",
                          peak_mode="dynamic", no_dual_mono=True)
    tools = (pipeline.locate_tools(run_options, need_ffmpeg=True) if command == "analyse"
             else pipeline.locate_normalise_tools(run_options))
    items = pipeline.discover_inputs(run_options).items
    executable = tools.ffmpeg if isinstance(tools, pipeline.ReadOnlyTools) else tools.executable
    assert executable is not None
    probed = pipeline.classify(items[0], tools.prober)
    scans: list[tuple] = []
    real_process = ffmpeg.run_process

    def counted(argv, **kwargs):
        if str(source) in argv:
            scans.append(tuple(argv))
        return real_process(argv, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("a read-only dynamic plan must not calibrate, render or execute")

    monkeypatch.setattr(ffmpeg, "run_process", counted)
    baseline = pipeline.measure(source, probed, executable, no_dual_mono=True)
    baseline_scans = tuple(scans)
    scans.clear()
    if signal == "within":
        assert baseline.integrated_lufs is not None
        run_options = replace(run_options, target=baseline.integrated_lufs)
    monkeypatch.setattr(pipeline, "calibrate_dynamic", forbidden)
    monkeypatch.setattr(pipeline, "render", forbidden)

    def ledger():
        # C-04 evidence may change the temporary directory's mtime, but leaves
        # no files; source files must retain their bytes and timestamps.
        return {p.relative_to(tmp_path).as_posix():
                (p.stat().st_mode, p.stat().st_mtime_ns if p.is_file() else None,
                 hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
                for p in tmp_path.rglob("*")}

    before = ledger()
    if command == "analyse":
        assert isinstance(tools, pipeline.ReadOnlyTools)
        run = pipeline.analyse_items(items, run_options, tools)
        assert len(run.inspections) == 1 and not run.cancelled
        outcome = run.inspections[0].outcome
        assert outcome.execution == "inspect" and outcome.output is None
    else:
        assert isinstance(tools, pipeline.NormaliseTools)
        outcome = only(pipeline.normalise_items(items, run_options, tools, execute=forbidden)).outcome
        assert outcome.execution == "plan"
    assert tuple(scans) == baseline_scans
    assert not any("loudnorm" in arg for argv in scans for arg in argv)
    assert ledger() == before
    action = {"quiet": "dynamic", "within": "copy", "silent": "skip", "unmeasurable": "skip"}[signal]
    plan = outcome.gain
    assert plan is not None
    assert plan.action == action and plan.dynamic_parameters is None
    assert plan.dynamic_preview == (action == "dynamic")
    assert outcome.measurement_out is None and outcome.verification is None and outcome.published_paths == ()
    assert not any(w.code == "reduce_shortfall" for w in outcome.warnings)
    if action == "dynamic":
        assert baseline.integrated_lufs is not None
        assert outcome.status == "ok" and outcome.reason_code == "ok"
        assert plan.gain_db == pytest.approx(run_options.target - baseline.integrated_lufs)
        assert plan.expected_integrated_lufs == run_options.target and plan.predicted_true_peak_dbtp is None
        assert plan.limiter_reduction_db == plan.effective_limiter_bound_db == plan.shortfall_lu == 0
        assert plan.limiter_threshold_dbtp == run_options.true_peak
    else:
        assert outcome.reason_code == {"within": "already_within_tolerance", "silent": "silent",
                                       "unmeasurable": "unmeasurable"}[signal]


# --- Real tools: limit, reduce, dynamic (TST-A-05, TST-A-09) ----------------------------------


@real
def test_limit_to_minus_14_on_f05_le_reaches_target_or_warns_the_fallback(fixture_path, tmp_path: Path,
                                                                          monkeypatch) -> None:
    source = copied(fixture_path("F-05", "le"), tmp_path / "in")
    renders = counting_renders(monkeypatch)
    record = only(normalise(options(tmp_path, source, target=-14.0)))
    outcome = record.outcome
    assert outcome.execution == "committed" and outcome.published_paths
    assert 1 <= len(renders) <= MAX_RENDERS and all(call["source"] == source for call in renders)
    assert [call["attempt"] for call in renders] == list(range(1, len(renders) + 1))
    assert renders[0]["plan"].action == "limit"
    integrated, _ = assert_peak_safe(record, tmp_path / "oracle")
    if outcome.gain.action == "limit":
        assert (outcome.status, outcome.reason_code) == ("ok", "ok")
        assert abs(outcome.measurement_out.integrated_lufs + 14.0) <= 0.5 and abs(integrated + 14.0) <= 0.5
    else:  # the verified WARN fallback (V3 step 6)
        assert outcome.gain.action == "reduce" and outcome.gain.fallback_from is not None
        assert (outcome.status, outcome.reason_code) == ("warn", "reduce_shortfall")
    assert outcome.measurement_out.decoded_frames == outcome.measurement_in.decoded_frames
    assert residue(tmp_path) == []


@real
def test_reduce_on_f05_hc_reports_the_measured_shortfall(fixture_path, tmp_path: Path, monkeypatch) -> None:
    source = copied(fixture_path("F-05", "hc"), tmp_path / "in")
    renders = counting_renders(monkeypatch)
    record = only(normalise(options(tmp_path, source, target=-14.0, peak_mode="reduce")))
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("warn", "reduce_shortfall", "committed")
    assert len(renders) == 1 and outcome.gain.action == "reduce" and outcome.gain.fallback_from is None
    shortfall = outcome.verification.measured_shortfall_lu
    assert shortfall == pytest.approx(-14.0 - outcome.measurement_out.integrated_lufs) and shortfall > 0
    integrated, _ = assert_peak_safe(record, tmp_path / "oracle")
    assert abs(shortfall - (-14.0 - integrated)) <= 0.2
    # V10: a committed file reports only the measured shortfall, once.
    found = [w for w in outcome.warnings if w.code == "reduce_shortfall"]
    assert len(found) == 1 and found[0].stage == "verify" and f"{shortfall:.3f}" in found[0].message
    assert residue(tmp_path) == []


@real
def test_explicit_dynamic_is_calibrated_rendered_and_verified(tmp_path: Path, monkeypatch) -> None:
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le", rate=48000, level_db=-25,
                      source="anoisesrc=color=pink:sample_rate={rate}:duration={seconds}:seed=7")
    calibrations: list[float] = []
    real_calibrate = pipeline.calibrate_dynamic

    def counted(*args, **kwargs):
        calibrations.append(kwargs["ceiling_dbtp"])
        return real_calibrate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "calibrate_dynamic", counted)
    renders = counting_renders(monkeypatch)
    record = only(normalise(options(tmp_path, source, target=-16.0, peak_mode="dynamic")))
    outcome = record.outcome
    assert outcome.execution == "committed" and outcome.gain.action == "dynamic", record
    assert outcome.gain.dynamic_parameters is not None and not outcome.gain.dynamic_preview
    assert calibrations[0] == -1.0 and len(calibrations) == len(renders)  # one per render (V6)
    assert all(call["plan"].action == "dynamic" for call in renders)
    integrated, _ = assert_peak_safe(record, tmp_path / "oracle")
    assert abs(outcome.measurement_out.integrated_lufs + 16.0) <= 0.5 and abs(integrated + 16.0) <= 0.5
    assert residue(tmp_path) == []


@real
def test_no_dual_mono_reaches_every_render(tmp_path: Path, monkeypatch) -> None:
    source = generate(tmp_path / "in" / "mono.wav", "-c:a", "pcm_s16le", channels=1)
    renders = counting_renders(monkeypatch)
    record = only(normalise(options(tmp_path, source, no_dual_mono=True)))
    assert record.outcome.execution == "committed"
    assert renders and all(call["no_dual_mono"] is True for call in renders)


# --- Real tools: lossy outputs and the guard (formats F7–F16) ---------------------------------


@real
@pytest.mark.parametrize("fmt, changes, codec, suffix", [
    ("mp3", {}, "mp3", ".mp3"),
    ("aac", {"bitrate": "256k"}, "aac", ".m4a"),
])
def test_lossy_outputs_are_verified_after_decoding(fixture_path, tmp_path: Path, monkeypatch, fmt: str,
                                                   changes: dict, codec: str, suffix: str) -> None:
    source = with_stereo_layout(fixture_path("F-05", "le"), tmp_path / "in")
    renders = counting_renders(monkeypatch)
    run = normalise(options(tmp_path, source, target=-14.0, format=fmt, **changes))
    record = only(run)
    outcome = record.outcome
    assert outcome.execution == "committed", record
    published = outcome.published_paths[0]
    assert published.suffix == suffix and published.parent == source.parent / "Normalised"
    written = ffprobe.probe(ffprobe.locate_ffprobe(REAL_FFPROBE), published)
    assert (written.codec, written.sample_rate_hz, written.channels) == (codec, 48000, 2)
    target = outcome.target
    if fmt == "mp3":  # automatic MP3 is evidenced V0 (F15)
        assert (target.encoder, target.bitrate_mode, target.vbr_quality) == ("libmp3lame", "vbr", 0)
    else:  # AAC 256k CBR from TK-0418's run-local evidence (F12)
        assert (target.encoder, target.bitrate_mode, target.effective_bitrate_bps) == ("aac_at", "cbr", 256000)
        assert target.capability_basis.startswith("ev:aac_at:cbr:48000:2:stereo")
    assert 1 <= len(renders) <= MAX_RENDERS and all(call["source"] == source for call in renders)
    integrated, _ = assert_peak_safe(record, tmp_path / "oracle")
    if outcome.gain.action != "reduce":
        assert abs(integrated + 14.0) <= 0.5
    assert residue(tmp_path) == []


@real
def test_lossy_source_follows_the_transcode_guard(tmp_path: Path) -> None:
    """F14: MP3 → AAC is blocked (kept as MP3, one more generation); allowed, it becomes AAC."""
    source = generate(tmp_path / "in" / "take.mp3", "-c:a", "libmp3lame", "-q:a", "2", "-write_xing", "1",
                      rate=44100, level_db=-25)
    blocked = only(normalise(options(tmp_path, source, format="aac")))
    outcome = blocked.outcome
    assert outcome.execution == "committed", blocked
    assert (outcome.target.format, outcome.target.decision) == ("mp3", "kept_source_codec")
    assert outcome.published_paths[0].suffix == ".mp3"
    codes = [w.code for w in outcome.warnings]
    assert (outcome.status, outcome.reason_code) == ("warn", "lossy_transcode_blocked")
    assert "lossy_reencoded" in codes and codes.count("lossy_reencoded") == 1
    reencoded = next(w for w in outcome.warnings if w.code == "lossy_reencoded")
    assert reencoded.message.startswith("re-encoded once as mp3 (V0)")
    assert_peak_safe(blocked, tmp_path / "oracle")

    allowed = only(normalise(options(tmp_path, source, format="aac", allow_lossy_transcode=True,
                                     out_dir=tmp_path / "allowed")))
    outcome = allowed.outcome
    assert outcome.execution == "committed", allowed
    assert (outcome.target.format, outcome.target.decision) == ("aac", "as_requested")
    assert outcome.published_paths[0].suffix == ".m4a"
    assert not any(w.code in ("lossy_transcode_blocked", "lossy_reencoded") for w in outcome.warnings)
    assert_peak_safe(allowed, tmp_path / "oracle")
    assert residue(tmp_path) == []


# --- Real tools: the bounded ladder (ADR-0014, TST-A-07) --------------------------------------


def inject(monkeypatch: pytest.MonkeyPatch, faults: dict[int, dict]) -> list[int]:
    """Replace the verdict of attempt n with ``assess`` of the real measurement changed by ``faults[n]``."""
    attempts: list[int] = []
    real_verify = pipeline.verify_candidate

    def wrapped(result, *args, **kwargs):
        verification = real_verify(result, *args, **kwargs)
        attempts.append(result.attempt)
        if result.attempt in faults:
            verification = assess(replace(verification.measurement, **faults[result.attempt]), result.plan)
        return verification

    monkeypatch.setattr(pipeline, "verify_candidate", wrapped)
    return attempts


HOT = {"true_peak_linear": 10 ** (-0.5 / 20), "true_peak_dbtp": -0.5}
MISS = {"integrated_lufs": -16.6}
EVERY_BUDGET = {1: HOT, 2: MISS, 3: MISS, 4: MISS, 5: {"true_peak_linear": 10 ** (-0.6 / 20), "true_peak_dbtp": -0.6}}
"""Peak correction, two loudness corrections, the reduce fallback, then an unsafe reduce: five renders."""


@real
@pytest.mark.parametrize("keep_failed", [False, True])
def test_retries_stop_at_five_renders_and_leave_no_candidates(tmp_path: Path, monkeypatch,
                                                              keep_failed: bool) -> None:
    source = generate(tmp_path / "in" / "quiet.wav", "-c:a", "pcm_s24le", level_db=-30)
    renders = counting_renders(monkeypatch)
    attempts = inject(monkeypatch, EVERY_BUDGET)
    finalised: list[str] = []
    real_finalise = pipeline.finalise_target

    def counted(probe_result, run_options, *args, action, **kwargs):
        finalised.append(action)
        return real_finalise(probe_result, run_options, *args, action=action, **kwargs)

    monkeypatch.setattr(pipeline, "finalise_target", counted)
    record = only(normalise(options(tmp_path, source, keep_failed=keep_failed)))
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("failed", "verification_peak", "not_written")
    assert len(renders) == MAX_RENDERS and attempts == [1, 2, 3, 4, 5]
    assert [call["plan"].action for call in renders] == ["linear", "limit", "limit", "limit", "reduce"]
    assert [call["attempt"] for call in renders] == [1, 2, 3, 4, 5]
    assert all(call["source"] == source for call in renders)  # every render reads the original (V3)
    assert finalised == ["linear", "limit", "limit", "limit", "reduce"]  # initial C-04, then V3a per retry
    assert outcome.gain.reduce_fallbacks == 1 and outcome.gain.loudness_corrections == 2
    assert outcome.gain.peak_corrections == 1
    candidates = [call["candidate"] for call in renders]
    assert len(set(candidates)) == MAX_RENDERS  # a fresh candidate per attempt
    assert not any(path.exists() for path in candidates[:-1])  # rejected candidates were disposed of
    if keep_failed:
        assert outcome.retained_failed_path == candidates[-1] and candidates[-1].exists()
        assert residue(tmp_path) == [candidates[-1]]
    else:
        assert outcome.retained_failed_path is None and residue(tmp_path) == []
    assert not (source.parent / "Normalised" / source.name).exists()


@real
def test_a_retry_passes_v3a_planning_and_its_failure_costs_no_render(tmp_path: Path, monkeypatch) -> None:
    source = generate(tmp_path / "in" / "quiet.wav", "-c:a", "pcm_s24le", level_db=-30)
    renders = counting_renders(monkeypatch)
    inject(monkeypatch, {1: MISS})
    real_finalise = pipeline.finalise_target
    calls: list[str] = []

    def failing(*args, action, **kwargs):
        calls.append(action)
        if len(calls) > 1:
            raise FileFailure("unsupported_target", "format", "injected at gain_planned")
        return real_finalise(*args, action=action, **kwargs)

    monkeypatch.setattr(pipeline, "finalise_target", failing)
    record = only(normalise(options(tmp_path, source)))
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("failed", "unsupported_target",
                                                                        "not_written")
    assert len(renders) == 1 and calls == ["linear", "linear"]
    assert not renders[0]["candidate"].exists() and residue(tmp_path) == []


@real
def test_a_corrected_file_is_published_once(tmp_path: Path, monkeypatch) -> None:
    source = generate(tmp_path / "in" / "quiet.wav", "-c:a", "pcm_s24le", level_db=-30)
    renders = counting_renders(monkeypatch)
    # An injected miss makes the correction overshoot the real output, so attempt 2 reads on target.
    inject(monkeypatch, {1: MISS, 2: {"integrated_lufs": -16.0}})
    record = only(normalise(options(tmp_path, source)))
    outcome = record.outcome
    assert (outcome.status, outcome.reason_code, outcome.execution) == ("ok", "ok", "committed")
    assert len(renders) == 2 and outcome.gain.loudness_corrections == 1
    assert outcome.published_paths == (source.parent / "Normalised" / source.name,)
    assert not renders[0]["candidate"].exists() and residue(tmp_path) == []
    assert math.isclose(renders[1]["plan"].gain_db, renders[0]["plan"].gain_db + 0.6, abs_tol=1e-9)


def retry_as_flac(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry whose final C-04 changes the bundle: a WAV source's retries resolve to FLAC.

    FLAC evidence is collected for every input so the changed target is evidenced (C-04).
    """
    real_finalise, real_candidates = pipeline.finalise_target, pipeline._candidate_formats
    seen: dict[str, int] = {}

    def changing(probe_result, run_options, *args, source_suffix=None, **kwargs):
        seen[source_suffix] = seen.get(source_suffix, 0) + 1
        if source_suffix == ".wav" and seen[source_suffix] > 1:
            run_options = replace(run_options, format="flac")
        return real_finalise(probe_result, run_options, *args, source_suffix=source_suffix, **kwargs)

    monkeypatch.setattr(pipeline, "finalise_target", changing)
    monkeypatch.setattr(pipeline, "_candidate_formats", lambda probe_result, run_options:
                        real_candidates(probe_result, run_options) | {"flac"})


@real
def test_a_retry_that_changes_the_bundle_reserves_and_publishes_the_new_member(tmp_path: Path,
                                                                              monkeypatch) -> None:
    """V3a ``writes_reserved``: a changed C-08 bundle is a new plan revision, reserved before allocation."""
    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s24le", level_db=-30)
    renders = counting_renders(monkeypatch)
    inject(monkeypatch, {1: MISS, 2: {"integrated_lufs": -16.0}})
    retry_as_flac(monkeypatch)
    record = only(normalise(options(tmp_path, source)))
    outcome = record.outcome
    assert outcome.execution == "committed", record
    folder = source.parent / "Normalised"
    assert outcome.published_paths == (folder / "take.flac",) and not (folder / "take.wav").exists()
    # S4 (TK-0427): C-07 revision 1 at planning, 2 for the retry's new target (MD34), 3 for the
    # final mark (mark_finalised); the C-08 bundle is bound to the final revision.
    assert outcome.output.plan_revision == 3 and outcome.target.format == "flac"
    assert outcome.metadata.plan_revision == 3 and outcome.metadata.mark_state == "final"
    assert [call["target"].format for call in renders] == ["wav", "flac"]
    assert renders[1]["candidate"].suffix == ".flac" and not renders[0]["candidate"].exists()
    assert residue(tmp_path) == []


@real
def test_a_retry_bundle_that_clashes_is_refused_without_a_render(tmp_path: Path, monkeypatch) -> None:
    """safety § 5.3: a retry's new member already reserved for another input refuses only this file."""
    folder = tmp_path / "in"
    first = generate(folder / "take.wav", "-c:a", "pcm_s24le", level_db=-30)
    generate(folder / "take.flac", "-c:a", "flac", level_db=-30, seconds=3)
    renders = counting_renders(monkeypatch)
    inject(monkeypatch, {1: MISS})
    retry_as_flac(monkeypatch)
    run = normalise(options(tmp_path, folder, jobs=1))
    rows = {record.outcome.input.source_path.name: record for record in run.records}
    wav = rows["take.wav"].outcome
    assert (wav.status, wav.reason_code, wav.execution) == ("failed", "name_clash", "not_written")
    assert rows["take.wav"].primary.stage == "writes_reserved"
    assert rows["take.flac"].outcome.execution == "committed"
    assert [call["source"] for call in renders].count(first) == 1  # the refused retry rendered nothing
    assert sorted(p.name for p in (folder / "Normalised").iterdir()) == ["take.flac"]
    assert residue(tmp_path) == []


# --- Real tools: the second interrupt kills a running ffprobe (CR-047) -------------------------

STUBBORN_PROBE = """
import signal
if '-version' in sys.argv:
    print('ffprobe version stubborn-1 Copyright')
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
"""


@real
def test_second_interrupt_kills_a_running_ffprobe(tmp_path: Path, make_tool, monkeypatch) -> None:
    folder = tmp_path / "in"
    for name in ("a.wav", "b.wav"):
        generate(folder / name, "-c:a", "pcm_s16le", seconds=2)
    stubborn = ffprobe.locate_ffprobe(make_tool("stubborn-ffprobe", STUBBORN_PROBE))
    controls: list = []
    seen: dict = {}
    real_probe = pipeline.probe_file

    def probe_candidate(prober, candidate, **kwargs):
        if not (candidate.name.startswith(".norma-") and "-0-cand-" in candidate.name):  # a.wav's candidate
            return real_probe(prober, candidate, **kwargs)
        seen["candidate"] = candidate

        def interrupt() -> None:
            end = time.monotonic() + 10
            while not ffprobe._ACTIVE and time.monotonic() < end:
                time.sleep(0.01)
            seen["at"] = time.monotonic()
            controls[0].interrupt()
            controls[0].interrupt()

        thread = threading.Thread(target=interrupt)
        thread.start()
        try:
            return real_probe(stubborn, candidate, **kwargs)
        finally:
            seen["ended"] = time.monotonic()
            thread.join()

    monkeypatch.setattr(pipeline, "probe_file", probe_candidate)
    run = normalise(options(tmp_path, folder, jobs=1), execute=executor(controls))
    rows = {record.outcome.input.source_path.name: record.outcome for record in run.records}
    a = rows["a.wav"]
    assert (a.status, a.reason_code, a.execution) == ("skipped", "cancelled", "not_written")
    assert run.cancelled and run.not_started == 1 and controls[0].force_event.is_set()
    assert seen["ended"] - seen["at"] < ffprobe.TERMINATE_GRACE_S  # SIGTERM-immune ffprobe killed at once
    assert ffprobe._ACTIVE == {} and not seen["candidate"].exists() and residue(tmp_path) == []


# --- Evidence is collected at the effective layout (D-081, TK-0420) -----------------------------


@real
@pytest.mark.parametrize("fmt, codec, suffix, encoder, mode", [
    ("aac", "aac", ".m4a", "aac_at", "cbr"),
    ("mp3", "mp3", ".mp3", "libmp3lame", "vbr"),
])
def test_unmasked_stereo_wav_normalises_to_lossy_with_the_assumed_layout_note(
        tmp_path: Path, fmt: str, codec: str, suffix: str, encoder: str, mode: str) -> None:
    """The collectors query ``stereo`` for a WAV that states no layout, so AAC and V0 resolve."""
    source = generate(tmp_path / "in" / "unmasked.wav", "-c:a", "pcm_s16le", rate=48000, level_db=-25)
    probed = ffprobe.probe(ffprobe.locate_ffprobe(REAL_FFPROBE), source)
    assert (probed.channels, probed.channel_layout) == (2, None)  # unmasked, as the brief requires
    record = only(normalise(options(tmp_path, source, target=-14.0, format=fmt)))
    outcome = record.outcome
    assert outcome.execution == "committed", record
    published = outcome.published_paths[0]
    assert published.suffix == suffix
    written = ffprobe.probe(ffprobe.locate_ffprobe(REAL_FFPROBE), published)
    assert (written.codec, written.channels) == (codec, 2)
    target = outcome.target
    assert (target.format, target.encoder, target.bitrate_mode) == (fmt, encoder, mode)
    assert target.capability_basis.startswith(f"ev:{encoder}:{mode}:48000:2:stereo")
    assert "channel layout not stated; assumed stereo" in target.notes
    assert residue(tmp_path) == []


def test_collect_evidence_queries_the_effective_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unmasked mono/stereo is queried as mono/stereo; stated and other layouts are queried as probed."""
    from norma.models import ProbeResult

    def probe_of(channels: int, layout: str | None) -> ProbeResult:
        return ProbeResult("pcm_s16le", "lossless", "wav", 0, 0, 48000, 16, False, "s16", channels, layout,
                           1.0, "stream", 1536000, False, 0, {}, "test")

    calls: list[tuple[str, int, int, str | None]] = []
    monkeypatch.setattr(pipeline, "_candidate_formats", lambda probe, run_options: {"wav", "mp3", "aac"})
    monkeypatch.setattr(pipeline, "collect_lossless_evidence",
                        lambda env, encoder, rate, channels, layout, work, *, cancel_event=None: calls.append(("lossless", rate, channels, layout)) or "L")
    probes_made: list[object] = []
    monkeypatch.setattr(pipeline, "probe_mp3_encoder_stats", lambda env, *, cancel_event=None: probes_made.append(env) or True)
    monkeypatch.setattr(pipeline, "collect_mp3_evidence",
                        lambda env, rate, channels, layout, work, *, encoder_stats, cancel_event=None: (
                            calls.append(("mp3", rate, channels, layout)), [encoder_stats and "M"])[1])
    monkeypatch.setattr(pipeline, "collect_aac_evidence",
                        lambda env, rate, channels, layout, work, *, cancel_event=None: calls.append(("aac", rate, channels, layout)) or "A")
    environment = SimpleNamespace(encoders=frozenset({"pcm_s16le", "pcm_s24le", "pcm_f32le", "libmp3lame", "aac_at"}))
    probes = (probe_of(1, None), probe_of(2, None), probe_of(2, "stereo"), probe_of(6, None), probe_of(3, "3.0"))
    pipeline.collect_evidence(environment, probes, RunOptions(command="normalise", ffmpeg=REAL_FFMPEG, cwd=Path.cwd()))
    seen = {(kind, channels, layout) for kind, _, channels, layout in calls}
    for kind in ("lossless", "mp3", "aac"):
        assert (kind, 1, "mono") in seen and (kind, 2, "stereo") in seen
        assert (kind, 6, None) in seen and (kind, 3, "3.0") in seen
        assert (kind, 1, None) not in seen and (kind, 2, None) not in seen
    assert [c for c in calls if c[0] == "aac"].count(("aac", 48000, 2, "stereo")) == 1  # one shared query
    assert len(probes_made) == 1  # CR-067: encoder statistics probed once, not per rate and layout


# --- CR-073: render receives the measured source frame count ------------------------------------


@real
def test_mp3_render_receives_the_measured_frames_and_does_not_decode_the_source_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TK-0426: pass 1's ``decoded_frames`` reaches ``render``; the MP3 reconciliation recounts nothing."""
    from norma.audio import render as render_module

    source = generate(tmp_path / "in" / "take.wav", "-c:a", "pcm_s16le", rate=48000, seconds=3, level_db=-25)
    calls = counting_renders(monkeypatch)

    def no_recount(*args, **kwargs):
        raise AssertionError("render decoded the source a second time")

    monkeypatch.setattr(render_module, "count_decoded_frames", no_recount)
    outcome = only(normalise(options(tmp_path, source, target=-14.0, format="mp3", jobs=1))).outcome
    assert outcome.execution == "committed", outcome
    assert outcome.target.format == "mp3" and outcome.published_paths[0].suffix == ".mp3"
    assert outcome.measurement_in.decoded_frames == 3 * 48000
    assert calls and all(call["source_frames"] == outcome.measurement_in.decoded_frames for call in calls)
