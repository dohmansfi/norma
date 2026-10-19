"""S2 linear render (TK-0108): argv builders, R7 checks and real-tool renders.

Real-tool tests write only under tmp_path; sources come from the WS-07
catalogue or are generated here. TST-A-04 (±0.5 LU at −14/−16), TST-A-10
(linear null) and TST-F-02 (zero-gain lossless chain) run the installed FFmpeg.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from norma.adapters import ffmpeg, ffprobe
from norma.audio.measure import conservative_peak_dbtp, measure
from norma.audio.plan import plan_gain
from norma.audio.render import (
    METADATA_FLAGS, SWR_OPTIONS, confirm_written, encoder_args, filter_chain,
    preserves_samples, render, render_argv, revalidate_basis,
)
from norma.audio.verify import decoded_digest, null_residual, verification_failure, verify_candidate
from norma.errors import FileFailure, ToolTimeout
from norma.formats import capabilities as cap
from norma.formats.resolve import finalise_target
from norma.models import GainPlan, ProbeResult, RunOptions, TargetFormat

FAKE_FFMPEG = Path("/unused/ffmpeg")
SOURCE = Path("/unused/source.wav")
CANDIDATE = Path("/unused/out/candidate.flac")


# --- Synthetic records -------------------------------------------------------------------------


def probe_of(*, codec="pcm_s24le", codec_class="lossless", rate=48000, depth=24, sample_float=False,
             channels=2, layout="stereo", duration=30.0, container="wav"):
    return ProbeResult(codec, codec_class, container, 0, 0, rate, depth, sample_float, "s32", channels, layout,
                       duration, "stream", None, False, 0, {}, "synthetic")


def target_of(fmt="flac", *, depth=24, sample_float=False, rate=48000, channels=2, layout="stereo",
              dither="none", encoder=None, stream_copy=False, basis="ev:synthetic"):
    if encoder is None:
        encoder = {"wav": f"pcm_{'f' if sample_float else 's'}{depth}le", "aiff": f"pcm_s{depth}be"}.get(fmt, fmt)
    if fmt in ("mp3", "aac"):
        return TargetFormat(fmt, ".mp3" if fmt == "mp3" else ".m4a", encoder, None, False, rate, channels, layout,
                            "none", "cbr", 256000, 256000, None, "auto", basis, False, "as_requested")
    if stream_copy:
        encoder, basis, dither = "stream_copy", "not_applicable", "none"
    return TargetFormat(fmt, cap.OUTPUT_EXTENSIONS[fmt], encoder, depth, sample_float, rate, channels, layout,
                        dither, "none", None, None, None, "not_applicable", basis, stream_copy, "as_requested")


def plan_of(action="linear", gain=6.5, *, target=-16.0, ceiling=-1.0, tolerance=.5, expected=None, **extra):
    if action == "copy":
        gain = 0.0
    reason = "already_within_tolerance" if action == "copy" else "ok"
    return GainPlan(action=action, requested_peak_mode=extra.pop("mode", "limit"), target_lufs=target,
                    ceiling_dbtp=ceiling, tolerance_lu=tolerance, max_limit_db=3.0, gain_db=gain,
                    predicted_true_peak_dbtp=-3.0, limiter_reduction_db=extra.pop("policy", 0.0),
                    effective_limiter_bound_db=extra.pop("effective", 0.0),
                    limiter_threshold_dbtp=extra.pop("threshold", None),
                    expected_integrated_lufs=target if expected is None else expected, shortfall_lu=0.0,
                    fallback_from=None, loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0,
                    reason_code=reason, **extra)


def options(tmp_path, **changes):
    return RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=tmp_path, jobs=1, **changes)


def af(argv):
    return argv[argv.index("-af") + 1] if "-af" in argv else None


# --- Pure argument building ----------------------------------------------------------------------


def test_stream_copy_argv_r8_flags_and_single_stream():
    argv = render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, plan_of("copy"), probe_of(),
                       target_of("wav", stream_copy=True))
    assert argv[:3] == (str(FAKE_FFMPEG), "-nostdin", "-hide_banner")
    assert "-xerror" in argv and "-af" not in argv
    assert argv[argv.index("-i") - 2:argv.index("-i") + 2] == ("-protocol_whitelist", "file", "-i", str(SOURCE))
    assert argv[argv.index("-map") + 1] == "0:a:0" and argv.count("-map") == 1
    assert argv[argv.index("-c:a"):argv.index("-c:a") + 2] == ("-c:a", "copy")
    flags = argv.index("-map_metadata")
    assert argv[flags:flags + len(METADATA_FLAGS)] == METADATA_FLAGS
    assert flags > argv.index("-i") and argv[-3:] == ("-f", "wav", str(CANDIDATE))


def test_linear_argv_is_float_gain_without_limiter():
    argv = render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, plan_of("linear", -4.25), probe_of(), target_of("flac"))
    assert af(argv) == "volume=-4.25dB:precision=float"
    assert not any(word in " ".join(argv) for word in ("alimiter", "loudnorm", "aresample", "dither"))
    assert argv[argv.index("-c:a"):argv.index("-f")] == (
        "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-compression_level", "8",
        *METADATA_FLAGS)


def test_resampling_uses_r2_settings_before_gain():
    chain = filter_chain(plan_of("linear", 3.0), probe_of(rate=96000), target_of("flac", rate=48000))
    assert chain == f"aformat=sample_fmts=fltp|flt,aresample=48000:{SWR_OPTIONS},volume=3.0dB:precision=float"
    assert SWR_OPTIONS == "resampler=swr:filter_size=64:phase_shift=10:exact_rational=1:cutoff=0.97"
    assert filter_chain(plan_of("copy"), probe_of(rate=48000), target_of("flac", rate=48000)) is None


DITHER = "aresample=osr=48000:osf=s16:dither_method=triangular:output_sample_bits=16"


@pytest.mark.parametrize("action,source_depth,target_kwargs,expected", [
    ("linear", 24, dict(depth=16, dither="triangular"), f"volume=2.0dB:precision=float,{DITHER}"),
    ("linear", 16, dict(depth=16, dither="triangular"), f"volume=2.0dB:precision=float,{DITHER}"),
    ("copy", 16, dict(depth=16, dither="triangular"), None),             # unchanged samples: no dither
    ("copy", 24, dict(depth=16, dither="triangular"), DITHER),           # narrowing changes samples
    ("linear", 24, dict(depth=16, dither="none"), "volume=2.0dB:precision=float"),  # explicit override
    ("linear", 24, dict(depth=24, dither="none"), "volume=2.0dB:precision=float"),
    ("linear", 24, dict(fmt="wav", depth=32, sample_float=True), "volume=2.0dB:precision=float"),
])
def test_dither_only_for_changed_16_bit_samples(action, source_depth, target_kwargs, expected):
    kwargs = dict(target_kwargs)
    target = target_of(kwargs.pop("fmt", "flac"), **kwargs)
    assert filter_chain(plan_of(action, 2.0), probe_of(depth=source_depth), target) == expected


def test_lossy_source_to_16_bit_lossless_is_dithered():
    probe = probe_of(codec="mp3", codec_class="lossy", depth=None, container="mp3")
    assert filter_chain(plan_of("copy"), probe, target_of("flac", depth=16, dither="triangular")) == DITHER


def test_encoder_args_native_alac_and_pcm():
    assert encoder_args(target_of("alac")) == ("-c:a", "alac", "-sample_fmt", "s32p", "-bits_per_raw_sample", "24")
    assert encoder_args(target_of("alac", depth=16)) == ("-c:a", "alac", "-sample_fmt", "s16p")
    assert encoder_args(target_of("flac", depth=16), flac_level=5) == (
        "-c:a", "flac", "-sample_fmt", "s16", "-compression_level", "5")
    assert encoder_args(target_of("aiff", depth=24)) == ("-c:a", "pcm_s24be")
    assert encoder_args(target_of("wav", depth=32, sample_float=True)) == ("-c:a", "pcm_f32le")
    with pytest.raises(ValueError):
        encoder_args(target_of("flac"), flac_level=9)
    with pytest.raises(ValueError):
        encoder_args(target_of("wav", depth=24, encoder="pcm_s16le"))
    with pytest.raises(ValueError, match="alac_at"):
        target_of("alac", encoder="alac_at")


def test_wav_info_pairs_are_copied_verbatim():
    pairs = (("INAM", "Demo = take 2"), ("IART", "Test Artist"))
    argv = render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, plan_of("linear"), probe_of(), target_of("wav"),
                       wav_info=pairs)
    assert argv[-7:-3] == ("-metadata", "INAM=Demo = take 2", "-metadata", "IART=Test Artist")
    with pytest.raises(ValueError):
        render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, plan_of("linear"), probe_of(), target_of("flac"), wav_info=pairs)


def test_inconsistent_records_are_programming_errors():
    with pytest.raises(ValueError):
        render_argv(FAKE_FFMPEG, SOURCE, Path("relative.flac"), plan_of(), probe_of(), target_of())
    with pytest.raises(ValueError):
        filter_chain(plan_of("linear"), probe_of(), target_of("wav", stream_copy=True))
    with pytest.raises(ValueError):
        filter_chain(plan_of("linear"), probe_of(channels=1, layout="mono"), target_of())
    skip = replace(plan_of("copy"), action="skip", gain_db=None, predicted_true_peak_dbtp=None,
                   expected_integrated_lufs=None, shortfall_lu=None, reason_code="silent")
    with pytest.raises(ValueError):
        filter_chain(skip, probe_of(), target_of())


@pytest.mark.parametrize("probe,target,expected", [
    (probe_of(), target_of("wav", stream_copy=True), True),
    (probe_of(depth=24), target_of("alac", depth=24), True),
    (probe_of(depth=16), target_of("flac", depth=24), True),
    (probe_of(depth=24), target_of("flac", depth=16), False),
    (probe_of(depth=24), target_of("wav", depth=32, sample_float=True), True),
    (probe_of(depth=32), target_of("wav", depth=32, sample_float=True), False),
    (probe_of(depth=32, sample_float=True), target_of("flac", depth=24), False),
    (probe_of(depth=32, sample_float=True), target_of("wav", depth=32, sample_float=True), True),
    (probe_of(depth=64, sample_float=True), target_of("wav", depth=32, sample_float=True), False),
    (probe_of(rate=96000), target_of("flac", rate=48000), False),
    (probe_of(depth=None), target_of("flac"), False),
    (probe_of(codec="mp3", codec_class="lossy", depth=None), target_of("flac", depth=16), False),
])
def test_preserves_samples(probe, target, expected):
    assert preserves_samples(probe, target) is expected


# --- S3 refusals and R7 checks before any process ----------------------------------------------------


def refuse_processes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no process may start")
    monkeypatch.setattr(ffmpeg, "run_process", refuse)


@pytest.fixture
def no_process(monkeypatch):
    refuse_processes(monkeypatch)


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


@pytest.fixture(scope="module")
def env(executable):
    return ffmpeg.probe_environment(executable)


def s3_plans(tmp_path):
    measured = measure_record(-23.0, -5.0)
    limit = plan_gain(measured, options(tmp_path))
    reduce = plan_gain(measured, options(tmp_path, peak_mode="reduce"))
    dynamic = replace(plan_of("linear"), action="dynamic", requested_peak_mode="dynamic",
                      limiter_threshold_dbtp=-1.0, dynamic_parameters={
                          "input_i": -23.0, "input_tp": -5.0, "input_lra": 3.0, "input_thresh": -33.0,
                          "target_offset": 0.1})
    assert (limit.action, reduce.action) == ("limit", "reduce")
    return {"limit": limit, "reduce": reduce, "dynamic": dynamic}


def measure_record(integrated, peak):
    from norma.models import LoudnessMeasurement
    return LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=0.0, threshold_lufs=None, integrated_valid=True,
        peak_valid=True, lra_valid=True, threshold_valid=False, integrated_resolution_lu=.001,
        peak_resolution_linear=.001, dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic", decoded_duration_s=30.0, decoded_frames=1440000, sample_rate_hz=48000,
        decoded_nonzero=True, terminal_peak_covered=True)


@pytest.mark.parametrize("change", [{"input_i": 3.0}, {"target_offset": 120.0}, {"input_lra": -1.0}])
def test_dynamic_values_outside_loudnorm_ranges_fail_before_any_process(tmp_path, env, no_process, change):
    """R5/V6: an out-of-range calibration value is a render_error, never clamped; no process starts."""
    plan = s3_plans(tmp_path)["dynamic"]
    plan = replace(plan, dynamic_parameters={**plan.dynamic_parameters, **change})
    source = tmp_path / "source.wav"
    source.write_bytes(b"")
    with pytest.raises(FileFailure) as caught:
        render(source, (0, 0), probe_of(), plan, target_of(), tmp_path / "c.flac", env, ())
    assert (caught.value.reason, caught.value.stage) == ("render_error", "render")
    assert "not clamped" in caught.value.detail


@pytest.mark.parametrize("fmt", ["mp3", "aac"])
def test_lossy_encoding_without_evidence_is_refused_before_any_process(tmp_path, env, no_process, fmt):
    source = tmp_path / "s.wav"
    source.write_bytes(b"")
    with pytest.raises(FileFailure) as caught:
        render(source, (0, 0), probe_of(), plan_of("linear"),
               target_of(fmt, encoder={"mp3": "libmp3lame", "aac": "aac_at"}[fmt]), tmp_path / "c.out", env, ())
    assert (caught.value.reason, caught.value.stage) == ("unsupported_target", "render")


def evidence(env, encoder, probe, folder, rate=None):
    return ffmpeg.collect_lossless_evidence(env, encoder, rate or probe.sample_rate_hz, probe.channels,
                                            probe.channel_layout, folder)


def test_r7_missing_or_changed_basis_is_unsupported_target(tmp_path, env, monkeypatch):
    probe = probe_of()
    record = evidence(env, "flac", probe, tmp_path / "ev")
    refuse_processes(monkeypatch)
    target = target_of("flac", basis=record.evidence_id)
    assert revalidate_basis(env, target, (record,)) is record
    assert revalidate_basis(env, target_of("wav", stream_copy=True), ()) is None
    cases = {
        "missing": (),
        "ambiguous": (record, record),
        "changed executable identity": (replace(record, executable_identity=(1, 2)),),
        "changed version": (replace(record, ffmpeg_version="other"),),
        "other encoder": (replace(record, encoder="alac"),),
        "other rate": (replace(record, sample_rate_hz=44100),),
        "unevidenced depth": (replace(record, supported_bit_depths=(16,)),),
    }
    for name, records in cases.items():
        with pytest.raises(FileFailure) as caught:
            revalidate_basis(env, target, records)
        assert (caught.value.reason, caught.value.stage) == ("unsupported_target", "render"), name
    source = tmp_path / "source.wav"
    source.write_bytes(b"")
    with pytest.raises(FileFailure, match="unsupported_target"):
        render(source, (0, 0), probe, plan_of("linear"), target, tmp_path / "c.flac", env, ())


def test_r7_lost_executable_is_render_error(tmp_path, env, no_process):
    moved = replace(env, executable=replace(env.executable, identity=(0, 0)))
    with pytest.raises(FileFailure) as caught:
        revalidate_basis(moved, target_of(), ())
    assert caught.value.reason == "render_error"


def test_source_identity_is_checked_before_rendering(tmp_path, env, no_process):
    source = tmp_path / "source.wav"
    source.write_bytes(b"x")
    target = target_of("wav", stream_copy=True)
    with pytest.raises(FileFailure) as caught:
        render(source, (1, 1), probe_of(), plan_of("copy"), target, tmp_path / "c.wav", env, ())
    assert (caught.value.reason, caught.value.stage) == ("source_changed", "render")
    with pytest.raises(FileFailure, match="source_changed"):
        render(tmp_path / "gone.wav", (1, 1), probe_of(), plan_of("copy"), target, tmp_path / "c.wav", env, ())
    with pytest.raises(ValueError):
        render(source, (1, 1), probe_of(), plan_of("copy"), target, tmp_path / "c.wav", env, (), attempt=6)


def stat_identity(path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


@pytest.mark.parametrize("outcome,reason", [
    (ToolTimeout("ffmpeg exceeded its 60 s deadline", deadline_s=60.0), "timeout"),
    (ffmpeg.ProcessResult(("x",), 1, b"", ("out.flac: No space left on device",), .1), "disk_full"),
    (ffmpeg.ProcessResult(("x",), 183, b"", ("Error while decoding stream",), .1), "render_error"),
])
def test_tool_outcomes_map_to_c13(tmp_path, env, monkeypatch, outcome, reason):
    source = tmp_path / "source.wav"
    source.write_bytes(b"x")

    def fake(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(ffmpeg, "run_process", fake)
    with pytest.raises(FileFailure) as caught:
        render(source, stat_identity(source), probe_of(), plan_of("copy"), target_of("wav", stream_copy=True),
               tmp_path / "c.wav", env, ())
    assert (caught.value.reason, caught.value.stage) == (reason, "render")


# --- Real renders ---------------------------------------------------------------------------------


def write_source(folder, name, *, rate=48000, seconds=2.0, subtype="PCM_24", channels=2, level=-12.0):
    n = np.arange(int(rate * seconds), dtype=np.float64)
    x = 10 ** (level / 20) * np.sin(2 * np.pi * 997 * n / rate) + 10 ** ((level - 20) / 20) * np.sin(
        2 * np.pi * 3001 * n / rate)
    data = np.column_stack([x] * channels) if channels > 1 else x
    path = folder / name
    sf.write(path, data, rate, subtype=subtype, format="WAV")
    return path


def lossless_target(env, probe, fmt, folder, *, depth=None, sample_float=False, rate=None, dither="none"):
    depth = depth or probe.bit_depth
    encoder = {"wav": f"pcm_{'f' if sample_float else 's'}{depth}le", "aiff": f"pcm_s{depth}be"}.get(fmt, fmt)
    record = evidence(env, encoder, probe, folder, rate)
    target = TargetFormat(fmt, cap.OUTPUT_EXTENSIONS[fmt], encoder, depth, sample_float, rate or probe.sample_rate_hz,
                          probe.channels, probe.channel_layout, dither, "none", None, None, None, "not_applicable",
                          record.evidence_id, False, "as_requested")
    return target, record


def candidate_for(folder, target, name="candidate"):
    path = folder / f"{name}{target.extension}"
    path.write_bytes(b"")
    return path


def test_render_overwrites_candidate_in_place_with_only_requested_metadata(tmp_path, env, executable, prober):
    plain = write_source(tmp_path, "plain.wav")
    source = tmp_path / "tagged.wav"
    ffmpeg.check_result(ffmpeg.run_process(ffmpeg.ffmpeg_argv(
        executable, "-y", *ffmpeg.input_args(plain), "-c:a", "copy", "-metadata", "title=SRC", "-metadata",
        "artist=Source", str(source)), deadline_s=60))
    probe = ffprobe.probe(prober, source)
    assert probe.ffprobe_tags.get("title") == "SRC"
    target, record = lossless_target(env, probe, "wav", tmp_path / "ev")
    candidate = candidate_for(tmp_path, target)
    before = stat_identity(candidate)
    result = render(source, stat_identity(source), probe, plan_of("linear", -3.0), target, candidate, env,
                    (record,), attempt=2, wav_info=(("INAM", "Kept title"),))
    assert stat_identity(candidate) == before and result.candidate_path == candidate
    assert result.attempt == 2 and result.decoded_source_identity == stat_identity(source)
    assert result.encoder_settings_valid and result.encoder_effective_bitrate_bps is None
    assert "24-bit 48000 Hz 2 ch" in result.encoder_setting_evidence and result.normalization_type is None
    tags = ffprobe.probe(prober, candidate).ffprobe_tags
    assert {key.lower(): value for key, value in tags.items()} == {"title": "Kept title"}


def test_written_header_mismatch_is_encoder_settings_mismatch(tmp_path, env, prober):
    source = write_source(tmp_path, "s16.wav", subtype="PCM_16")
    probe = ffprobe.probe(prober, source)
    target, record = lossless_target(env, probe, "flac", tmp_path / "ev", depth=16)
    candidate = candidate_for(tmp_path, target)
    render(source, stat_identity(source), probe, plan_of("copy"), target, candidate, env, (record,))
    assert confirm_written(candidate, target).startswith("written header 16-bit 48000 Hz 2 ch")
    with pytest.raises(FileFailure) as caught:
        confirm_written(candidate, replace(target, bit_depth=24))
    assert (caught.value.reason, caught.value.stage) == ("encoder_settings_mismatch", "render")
    with pytest.raises(FileFailure, match="render_error"):
        confirm_written(candidate, replace(target, format="wav", encoder="pcm_s16le", extension=".wav"))


def test_resampled_copy_keeps_length_and_rate(tmp_path, env, prober, executable):
    source = write_source(tmp_path, "s48.wav", seconds=3.0)
    probe = ffprobe.probe(prober, source)
    target, record = lossless_target(env, probe, "flac", tmp_path / "ev", rate=44100)
    candidate = candidate_for(tmp_path, target)
    result = render(source, stat_identity(source), probe, plan_of("copy"), target, candidate, env, (record,))
    assert "44100 Hz" in result.encoder_setting_evidence
    frames = sf.info(candidate).frames
    assert abs(frames - 3 * 44100) <= 3  # TESTING § 2: ±1 sample per second


@pytest.mark.parametrize("rate,subtype,depth", [(44100, "PCM_16", 16), (96000, "PCM_24", 24)])
def test_f02_zero_gain_lossless_chain_is_bit_identical(tmp_path, env, prober, executable, rate, subtype, depth):
    """TST-F-02: WAV → FLAC → native ALAC → AIFF, decoded samples exactly equal."""
    current = write_source(tmp_path, "chain.wav", rate=rate, subtype=subtype)
    digests = [decoded_digest(current, executable)]
    for step, fmt in enumerate(("flac", "alac", "aiff")):
        probe = ffprobe.probe(prober, current)
        assert probe.bit_depth == depth and probe.sample_rate_hz == rate
        target, record = lossless_target(env, probe, fmt, tmp_path / f"ev{step}")
        candidate = candidate_for(tmp_path, target, f"step{step}")
        result = render(current, stat_identity(current), probe, plan_of("copy"), target, candidate, env, (record,))
        assert af(render_argv(executable, current, candidate, plan_of("copy"), probe, target)) is None
        assert result.encoder_setting_evidence.startswith(f"written header {depth}-bit {rate} Hz 2 ch")
        if fmt == "alac":
            assert result.target.encoder == "alac" and ffmpeg.read_alac_header(candidate.read_bytes()).bit_depth == depth
        digests.append(decoded_digest(candidate, executable))
        current = candidate
    assert len(set(digests)) == 1


def independent(path, dual_mono=False):
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if dual_mono and samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.), padtype="constant")))
    return integrated, 20 * np.log10(peak)


def basis(executable, env):
    return cap.ToolBasis(executable.path, executable.identity, executable.version, env.platform_basis)


def normalise_linear(fixture, tmp_path, env, executable, prober, target_lufs, **option_changes):
    """measure → plan → finalise C-04 → render → verify, as the S2 pipeline sequences them."""
    probe = ffprobe.probe(prober, fixture)
    run = options(tmp_path, target=target_lufs, **option_changes)
    source_measurement = measure(fixture, probe, executable)
    plan = plan_gain(source_measurement, run)
    assert plan.action == "linear"
    records = []
    for encoder in ("pcm_s16le", "pcm_s24le", "pcm_f32le"):
        records.append(evidence(env, encoder, probe, tmp_path / "ev"))
    target = finalise_target(probe, run, records, basis(executable, env), action=plan.action)
    candidate = candidate_for(tmp_path, target)
    result = render(fixture, stat_identity(fixture), probe, plan, target, candidate, env, records)
    verification = verify_candidate(result, fixture, probe, ffprobe.probe(prober, candidate), executable,
                                    no_dual_mono=run.no_dual_mono)
    return probe, plan, target, result, verification, source_measurement


@pytest.mark.parametrize("fixture_id,target_lufs", [("F-04", -14.0), ("F-04", -16.0), ("F-06", -14.0),
                                                    ("F-06", -16.0)])
def test_a04_linear_outputs_meet_target(fixture_path, tmp_path, env, executable, prober, fixture_id, target_lufs):
    """TST-A-04: both meters within target ±0.5 LU, both TP oracles ≤ −0.9 dBTP."""
    fixture = fixture_path(fixture_id)
    _, plan, target, result, verification, _ = normalise_linear(
        fixture, tmp_path, env, executable, prober, target_lufs)
    argv = render_argv(executable, fixture, result.candidate_path, plan, probe_of(), target)
    assert af(argv) == f"volume={plan.gain_db!r}dB:precision=float"
    assert target.format == "wav" and not target.stream_copy
    assert verification_failure(verification) is None
    out = verification.measurement
    assert abs(out.integrated_lufs - target_lufs) <= .5
    oracle_i, oracle_tp = independent(result.candidate_path)
    assert abs(oracle_i - target_lufs) <= .5
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9
    assert oracle_tp <= -.9
    assert verification.loudness_error_lu == pytest.approx(out.integrated_lufs - target_lufs)


@pytest.mark.parametrize("fixture_id,changes,limit", [
    ("F-04", {}, "peak"), ("F-06", {}, "peak"), ("F-04", {"bit_depth": "16"}, "rms"),
])
def test_a10_linear_null(fixture_path, tmp_path, env, executable, prober, fixture_id, changes, limit):
    """TST-A-10: residual of output − source × gain; 24-bit/float < −90 dBFS peak, 16-bit ≤ −84 dBFS RMS."""
    fixture = fixture_path(fixture_id)
    probe, plan, target, result, verification, source = normalise_linear(
        fixture, tmp_path, env, executable, prober, -16.0, **changes)
    argv = render_argv(executable, fixture, result.candidate_path, plan, probe, target)
    assert "alimiter" not in " ".join(argv)
    if limit == "rms":
        assert target.bit_depth == 16 and target.dither == "triangular" and af(argv).endswith(DITHER)
    residual = null_residual(fixture, result.candidate_path, plan.gain_db, executable, channels=2)
    assert verification.measurement.decoded_frames == source.decoded_frames == residual.frames
    assert residual.passes(target, expected_frames=source.decoded_frames)
    if limit == "peak":
        assert residual.peak_dbfs < -90
    else:
        assert residual.rms_dbfs <= -84


def test_a10_endpoint_impulses_keep_positions(fixture_path, tmp_path, env, executable, prober):
    fixture = fixture_path("F-03", "endpoint")
    probe = ffprobe.probe(prober, fixture)
    target, record = lossless_target(env, probe, "wav", tmp_path / "ev", depth=32, sample_float=True)
    candidate = candidate_for(tmp_path, target)
    render(fixture, stat_identity(fixture), probe, plan_of("linear", 6.0), target, candidate, env, (record,))
    source, _ = sf.read(fixture, dtype="float64", always_2d=True)
    output, _ = sf.read(candidate, dtype="float64", always_2d=True)
    assert output.shape == source.shape
    assert np.array_equal(np.flatnonzero(output[:, 0]), np.flatnonzero(source[:, 0]))
    assert np.allclose(output, source * 10 ** (6 / 20), rtol=1e-6, atol=0)
    assert math.isclose(np.max(np.abs(output)), .03 * 10 ** .3, rel_tol=1e-6)
