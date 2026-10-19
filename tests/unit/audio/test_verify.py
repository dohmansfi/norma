"""Candidate verification (TK-0108): V1/M7 acceptance, copy identity and the null helper.

Pure ``assess`` cases use injected C-03 records; the M7 quantiser regression,
copy identity and null tests run the installed FFmpeg under tmp_path only.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from norma.adapters import ffmpeg, ffprobe
from norma.audio import measure as meter
from norma.audio import verify as verify_module
from norma.audio.render import render
from norma.audio.verify import (
    PEAK_ALLOWANCE_DB, NullResidual, assess, check_copy_identity, decoded_digest, measure_candidate,
    null_residual, null_residual_argv, verification_failure, verify_candidate,
)
from norma.errors import FileFailure, SkipFile, ToolTimeout
from norma.formats import capabilities as cap
from norma.models import GainPlan, LoudnessMeasurement, TargetFormat


def measurement(integrated=-16.0, linear=.5, *, resolution=.001, state="measurable", **changes):
    peak = None if linear is None else (20 * math.log10(linear) if linear > 0 else -math.inf)
    if state != "measurable":
        integrated = None
    return replace(LoudnessMeasurement(
        signal_state=state, integrated_lufs=integrated, true_peak_dbtp=peak, true_peak_linear=linear,
        lra_lu=None, threshold_lufs=None, integrated_valid=integrated is not None, peak_valid=peak is not None,
        lra_valid=False, threshold_valid=False, integrated_resolution_lu=.001, peak_resolution_linear=resolution,
        dual_mono=False, meter="ebur128", value_source="metadata", ffmpeg_version="synthetic",
        decoded_duration_s=30.0, decoded_frames=1440000, sample_rate_hz=48000,
        decoded_nonzero=state != "silent", terminal_peak_covered=True,
    ), **changes)


def plan(action="linear", *, target=-16.0, ceiling=-1.0, tolerance=.5, gain=6.0, expected=None):
    if action == "copy":
        gain = 0.0
    return GainPlan(action=action, requested_peak_mode="reduce" if action == "reduce" else "limit",
                    target_lufs=target, ceiling_dbtp=ceiling, tolerance_lu=tolerance, max_limit_db=3.0,
                    gain_db=gain, predicted_true_peak_dbtp=ceiling - 1, limiter_reduction_db=0.0,
                    effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
                    expected_integrated_lufs=target if expected is None else expected,
                    shortfall_lu=0.0 if expected is None else max(0.0, target - expected), fallback_from=None,
                    loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0,
                    reason_code="already_within_tolerance" if action == "copy" else "ok")


def amplitude(dbtp):
    """Printed amplitude whose M7 upper bound sits exactly ``dbtp`` (q = 0.001)."""
    return 10 ** (dbtp / 20) - .0005


# --- Pure acceptance (V1, M7, T1, T2) ---------------------------------------------------------------


@pytest.mark.parametrize("integrated,ok", [(-16.5, True), (-15.5, True), (-16.0, True), (-16.501, False),
                                           (-15.499, False)])
def test_loudness_tolerance_boundaries(integrated, ok):
    verification = assess(measurement(integrated), plan())
    assert verification.loudness_ok is ok and verification.peak_ok and verification.measurable
    assert verification.loudness_error_lu == pytest.approx(integrated + 16)
    assert verification.measured_shortfall_lu == pytest.approx(max(0.0, -16 - integrated))
    failure = verification_failure(verification)
    assert (failure is None) is ok
    if not ok:
        assert (failure.reason, failure.stage) == ("verification_loudness", "verify")


@pytest.mark.parametrize("offset,ok", [(-1e-6, True), (1e-6, False)])
def test_peak_bound_uses_conservative_level(offset, ok):
    bound = -1.0 + PEAK_ALLOWANCE_DB + offset
    verification = assess(measurement(linear=amplitude(bound)), plan())
    assert verification.peak_ok is ok
    assert verification.peak_excess_db == pytest.approx(20 * math.log10(amplitude(bound)) + 1)


def test_m7_worked_example_printed_pass_is_not_peak_safe():
    """§ 5.1: a = 0.716 at c = −3 prints −2.9017 dBTP (central pass) but TP_up = −2.8957 fails."""
    verification = assess(measurement(linear=.716), plan(ceiling=-3.0))
    assert verification.peak_excess_db == pytest.approx(20 * math.log10(.716) + 3)
    assert verification.peak_excess_db <= .1 and not verification.peak_ok
    assert verification.measurement.true_peak_linear == .716  # the report keeps the printed value
    failure = verification_failure(verification)
    assert failure.reason == "verification_peak" and "-2.8957" in failure.detail


@pytest.mark.parametrize("resolution", [None])
def test_unknown_resolution_is_never_peak_safe(resolution):
    verification = assess(measurement(linear=.1, resolution=resolution), plan())
    assert not verification.peak_ok and verification.loudness_ok
    assert verification_failure(verification).reason == "verification_peak"


def test_peak_failure_precedes_loudness_failure():
    verification = assess(measurement(-20.0, linear=1.0), plan())
    assert not verification.peak_ok and not verification.loudness_ok
    assert verification_failure(verification).reason == "verification_peak"
    assert {d.code for d in verification.diagnostics} == {"verification_peak", "verification_loudness"}


@pytest.mark.parametrize("state,linear", [("unmeasurable", .2), ("unmeasurable", 1.0), ("silent", 0.0)])
def test_unmeasurable_candidate_fails_loudness_first(state, linear):
    record = measurement(linear=linear, state=state)
    if state == "silent":
        record = replace(record, decoded_nonzero=False)
    verification = assess(record, plan())
    assert not verification.measurable and not verification.loudness_ok
    assert verification.loudness_error_lu is None and verification.measured_shortfall_lu is None
    assert verification_failure(verification).reason == "verification_loudness"
    if state == "silent":
        assert verification.peak_excess_db is None


def test_reduce_accepts_measurable_safe_candidate_with_measured_shortfall():
    verification = assess(measurement(-19.25, linear=.5), plan("reduce", gain=3.0, expected=-19.0))
    assert verification.loudness_ok and verification_failure(verification) is None
    assert verification.measured_shortfall_lu == pytest.approx(3.25)
    assert verification.loudness_error_lu == pytest.approx(-.25)
    assert verification.expected_lufs == -19.0


def test_copy_uses_target_tolerance_and_skip_has_no_candidate():
    verification = assess(measurement(-16.2), plan("copy", expected=-16.1))
    assert verification.loudness_ok and verification.loudness_error_lu == pytest.approx(-.1)
    skip = replace(plan("copy"), action="skip", gain_db=None, predicted_true_peak_dbtp=None,
                   expected_integrated_lufs=None, shortfall_lu=None, reason_code="silent")
    with pytest.raises(ValueError):
        assess(measurement(), skip)


# --- Candidate measurement errors keep C-13 reasons at stage verify -------------------------------------


@pytest.mark.parametrize("raised,reason", [
    (FileFailure("measurement_error", "measure", "bad meter output"), "verification_loudness"),
    (FileFailure("decode_error", "measure", "corrupt packet"), "decode_error"),
    (ToolTimeout("ffmpeg exceeded its 60 s deadline", deadline_s=60.0).to_failure("decode_error", "measure"),
     "timeout"),
    (SkipFile("unsupported_channels", "measure", "6 channels"), "verification_loudness"),
])
def test_measure_candidate_maps_failures(monkeypatch, raised, reason):
    def fake(*args, **kwargs):
        raise raised
    monkeypatch.setattr(meter, "measure", fake)
    with pytest.raises(FileFailure) as caught:
        measure_candidate(Path("/unused/c.wav"), None, None)
    assert (caught.value.reason, caught.value.stage) == (reason, "verify")


# --- Real tools ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


@pytest.fixture(scope="module")
def env(executable):
    return ffmpeg.probe_environment(executable)


def tone(folder, name, *, seconds=2.0, level=-12.0, subtype="PCM_24", rate=48000):
    n = np.arange(int(seconds * rate), dtype=np.float64)
    x = 10 ** (level / 20) * np.sin(2 * np.pi * 997 * n / rate)
    path = folder / name
    sf.write(path, np.column_stack((x, x)), rate, subtype=subtype, format="WAV")
    return path


def identity(path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


def lossless(env, probe, fmt, folder, depth, dither="none"):
    encoder = {"wav": f"pcm_s{depth}le", "aiff": f"pcm_s{depth}be"}.get(fmt, fmt)
    record = ffmpeg.collect_lossless_evidence(env, encoder, probe.sample_rate_hz, probe.channels,
                                              probe.channel_layout, folder)
    target = TargetFormat(fmt, cap.OUTPUT_EXTENSIONS[fmt], encoder, depth, False, probe.sample_rate_hz,
                          probe.channels, probe.channel_layout, dither, "none", None, None, None,
                          "not_applicable", record.evidence_id, False, "as_requested")
    return target, record


def test_m7_quantiser_bin_regression_on_real_signal(tmp_path, executable, prober):
    """§ 8 P1 recipe: printed 0.716 on both scans at c = −3 is not peak-safe; report keeps 0.716."""
    rate, n = 48000, 4 * 48000
    x = .71625 * np.sin(2 * np.pi * 1000 * np.arange(n) / rate)
    ramp = int(.05 * rate)
    window = .5 - .5 * np.cos(np.pi * np.arange(ramp) / ramp)
    x[:ramp] *= window
    x[-ramp:] *= window[::-1]
    path = tmp_path / "p1.wav"
    sf.write(path, x, rate, subtype="DOUBLE", format="WAV")
    record = measure_candidate(path, ffprobe.probe(prober, path), executable, no_dual_mono=True)
    assert record.true_peak_linear == .716 and record.peak_resolution_linear == .001
    verification = assess(record, plan(ceiling=-3.0, target=record.integrated_lufs))
    assert verification.peak_excess_db <= .1 and not verification.peak_ok
    assert verification_failure(verification).reason == "verification_peak"


def test_copy_identity_accepts_lossless_transcode_and_rejects_changed_samples(tmp_path, env, executable, prober):
    source = tone(tmp_path, "source.wav")
    probe = ffprobe.probe(prober, source)
    target, record = lossless(env, probe, "flac", tmp_path / "ev", 24)
    candidate = tmp_path / "c.flac"
    candidate.write_bytes(b"")
    result = render(source, identity(source), probe, plan("copy", expected=-16.0), target, candidate, env, (record,))
    assert check_copy_identity(source, candidate, executable) == decoded_digest(source, executable)
    other = tone(tmp_path, "other.wav", level=-12.01)
    with pytest.raises(FileFailure) as caught:
        check_copy_identity(source, other, executable)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")
    verification = verify_candidate(result, source, probe, ffprobe.probe(prober, candidate), executable)
    assert verification.measurement.decoded_frames == 96000 and verification.measurable


@pytest.mark.parametrize("depth,expect_identity", [(24, True), (16, False)])
def test_verify_candidate_checks_identity_only_for_sample_preserving_copies(
        tmp_path, env, executable, prober, monkeypatch, depth, expect_identity):
    source = tone(tmp_path, "source.wav")
    probe = ffprobe.probe(prober, source)
    target, record = lossless(env, probe, "flac", tmp_path / "ev", depth,
                              dither="triangular" if depth == 16 else "none")
    candidate = tmp_path / "c.flac"
    candidate.write_bytes(b"")
    result = render(source, identity(source), probe, plan("copy"), target, candidate, env, (record,))
    calls = []
    monkeypatch.setattr(verify_module, "check_copy_identity", lambda *a, **k: calls.append(a))
    verify_candidate(result, source, probe, ffprobe.probe(prober, candidate), executable)
    assert bool(calls) is expect_identity


def test_null_residual_argv_is_double_precision_subtraction():
    argv = null_residual_argv(Path("/unused/ffmpeg"), Path("/s.wav"), Path("/c.wav"), -2.5, 2)
    graph = argv[argv.index("-filter_complex") + 1]
    assert "volume=-2.5dB:precision=double" in graph
    assert "aeval=exprs=val(0)-val(2)|val(1)-val(3):c=stereo" in graph
    assert argv.index("/c.wav") < argv.index("/s.wav")  # candidate is input 0
    mono = null_residual_argv(Path("/unused/ffmpeg"), Path("/s.wav"), Path("/c.wav"), 0.0, 1)
    assert "aeval=exprs=val(0)-val(1):c=mono" in mono[mono.index("-filter_complex") + 1]
    with pytest.raises(ValueError):
        null_residual_argv(Path("/unused/ffmpeg"), Path("/s.wav"), Path("/c.wav"), 0.0, 6)


def test_null_residual_detects_exact_null_and_wrong_gain(tmp_path, executable):
    source = tone(tmp_path, "source.wav")
    exact = null_residual(source, source, 0.0, executable, channels=2)
    assert exact.frames == 96000 and exact.peak_dbfs == -math.inf
    wrong = null_residual(source, source, .1, executable, channels=2)
    assert wrong.peak_dbfs > -60
    short = tone(tmp_path, "short.wav", seconds=1.5)
    assert null_residual(source, short, 0.0, executable, channels=2).frames == 72000


@pytest.mark.parametrize("residual,depth,sample_float,frames,ok", [
    (NullResidual(100, -95.0, -120.0), 24, False, 100, True),
    (NullResidual(100, -89.0, -120.0), 24, False, 100, False),
    (NullResidual(100, -89.0, -120.0), 32, True, 100, False),
    (NullResidual(100, -70.0, -84.0), 16, False, 100, True),
    (NullResidual(100, -70.0, -83.9), 16, False, 100, False),
    (NullResidual(99, -150.0, -150.0), 24, False, 100, False),
])
def test_null_limits_follow_testing_section_2(residual, depth, sample_float, frames, ok):
    target = TargetFormat("wav", ".wav", f"pcm_{'f' if sample_float else 's'}{depth}le", depth, sample_float,
                          48000, 2, "stereo", "none", "none", None, None, None, "not_applicable", "ev:x", False,
                          "as_requested")
    assert residual.passes(target, expected_frames=frames) is ok
