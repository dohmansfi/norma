"""TK-0111: explicit dynamic mode, MP3/AAC encoding and the lossy length rule (CR-059).

Pure tests use synthetic records; real-tool tests run the installed FFmpeg and
write only under tmp_path, with sources from the WS-07 catalogue
(``fixture_path``). Acceptance destinations (test plan):

- TST-A-09 (dynamic part) — explicit ``dynamic`` on F-05: the loudnorm
  calibration is parsed into C-05 ``dynamic_parameters``, the calibrated
  render reaches the target with safe peaks on ebur128 and the independent
  oracles, and no other policy ever selects ``dynamic`` (R5, R6, V6).
- TST-A-06 — MP3 V0 and AAC 256k from F-04/F-05, measured after decoding:
  conservative ebur128 and SciPy 4× true peaks ≤ −0.9 dBTP (R7, V1, V5).
- CR-059 / audio E3 — the length rule accepts real lossy outputs (MP3 exact,
  AAC trailing padding < one frame, no priming) and still fails a truncated one.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
from scipy.signal import resample_poly

from norma.adapters import ffmpeg, ffprobe
from norma.audio.measure import (
    LoudnormJson,
    calibrate_dynamic,
    calibration_argv,
    conservative_peak_dbtp,
    measure,
    parse_calibration,
)
from norma.audio.plan import (
    DynamicCalibration,
    dynamic_peak_ceiling,
    needs_dynamic_calibration,
    plan_gain,
    replan_dynamic_peak,
)
from norma.audio.render import (
    SWR_OPTIONS,
    confirm_aac,
    encoder_args,
    filter_chain,
    lossy_encoder_args,
    render,
    render_argv,
    revalidate_basis,
)
from norma.audio.verify import (
    MAX_RENDERS,
    Retry,
    assess,
    check_candidate_length,
    check_encoded_stream,
    next_step,
    run_ladder,
    trailing_padding_allowance,
    verify_candidate,
)
from norma.errors import FileFailure
from norma.models import GainPlan, LoudnessMeasurement, ProbeResult, RunOptions, TargetFormat

FAKE_FFMPEG = Path("/unused/ffmpeg")
SOURCE = Path("/unused/source.wav")
CANDIDATE = Path("/unused/out/candidate.m4a")
PARAMS = {"input_i": -23.55, "input_tp": -1.01, "input_lra": 1.0, "input_thresh": -33.55, "target_offset": 0.62}


# --- Synthetic records ---------------------------------------------------------------------------


def measurement(integrated=-23.0, peak=-5.0, *, frames=1440000, rate=48000, **changes) -> LoudnessMeasurement:
    return replace(LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=1.0, threshold_lufs=None, integrated_valid=True,
        peak_valid=True, lra_valid=True, threshold_valid=False, integrated_resolution_lu=.001,
        peak_resolution_linear=.001, dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic", decoded_duration_s=frames / rate, decoded_frames=frames, sample_rate_hz=rate,
        decoded_nonzero=True, terminal_peak_covered=True), **changes)


def options(tmp_path=Path("/unused"), **changes) -> RunOptions:
    return RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=tmp_path, jobs=1, **changes)


def calibration(ceiling=-1.0, target=-16.0, **values) -> DynamicCalibration:
    return DynamicCalibration({**PARAMS, **values}, "dynamic", target, ceiling)


def dynamic_plan(**changes) -> GainPlan:
    plan = plan_gain(measurement(), options(peak_mode="dynamic"), calibration=calibration())
    return replace(plan, **changes)


def probe_of(*, codec="pcm_s24le", codec_class="lossless", rate=48000, depth=24, channels=2, layout="stereo",
             container="wav", duration=30.0) -> ProbeResult:
    return ProbeResult(codec, codec_class, container, 0, 0, rate, depth, False, "s32", channels, layout,
                       duration, "stream", None, False, 0, {}, "synthetic")


def lossy_target(fmt="mp3", *, mode=None, bps=None, quality=None, rate=48000, channels=2, layout="stereo",
                 basis="ev:synthetic", encoder=None) -> TargetFormat:
    mode = mode or ("vbr" if fmt == "mp3" else "cbr")
    if mode == "vbr":
        quality = 0 if quality is None else quality
    else:
        bps = bps or 256000
    return TargetFormat(fmt, ".mp3" if fmt == "mp3" else ".m4a", encoder or {"mp3": "libmp3lame", "aac": "aac_at"}[fmt],
                        None, False, rate, channels, layout, "none", mode, bps, bps, quality, "auto", basis, False,
                        "as_requested")


def wav24(rate=48000, channels=2) -> TargetFormat:
    return TargetFormat("wav", ".wav", "pcm_s24le", 24, False, rate, channels, "stereo" if channels == 2 else "mono",
                        "none", "none", None, None, None, "not_applicable", "ev:synthetic", False, "as_requested")


def af(argv):
    return argv[argv.index("-af") + 1] if "-af" in argv else None


# --- Calibration parsing (R5) --------------------------------------------------------------------

LOUDNORM_STDERR = """[Parsed_loudnorm_0 @ 0x600001dace70] 
{
\t"input_i" : "-23.55",
\t"input_tp" : "-1.01",
\t"input_lra" : "1.00",
\t"input_thresh" : "-33.55",
\t"output_i" : "-16.62",
\t"output_tp" : "-1.00",
\t"output_lra" : "0.10",
\t"output_thresh" : "-26.62",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.62"
}
[out#0/null @ 0x600001fac480] video:0KiB audio:22500KiB""".splitlines()


def test_loudnorm_json_is_parsed_into_the_five_c05_values():
    parser = LoudnormJson()
    for line in LOUDNORM_STDERR:
        parser.feed(line)
    result = parse_calibration(parser.last(), target_lufs=-16.0, ceiling_dbtp=-1.0)
    assert dict(result.parameters) == PARAMS
    assert result.normalization_type == "dynamic"
    assert (result.target_lufs, result.ceiling_dbtp) == (-16.0, -1.0)


@pytest.mark.parametrize("change", [{"input_i": "-inf"}, {"input_thresh": "nan"}, {"target_offset": "x"},
                                    {"normalization_type": "none"}, {"input_lra": None}])
def test_invalid_calibration_is_measurement_error_never_zero(change):
    fields = {"input_i": "-23.55", "input_tp": "-1.01", "input_lra": "1.00", "input_thresh": "-33.55",
              "target_offset": "0.62", "normalization_type": "dynamic"}
    fields.update(change)
    fields = {k: v for k, v in fields.items() if v is not None}
    with pytest.raises(FileFailure) as caught:
        parse_calibration(fields, target_lufs=-16.0, ceiling_dbtp=-1.0)
    assert (caught.value.reason, caught.value.stage) == ("measurement_error", "measure")


def test_missing_or_broken_json_is_measurement_error():
    for lines in ([], ["{", '"input_i" : "-23"']):  # nothing, and an unterminated object
        parser = LoudnormJson()
        for line in lines:
            parser.feed(line)
        with pytest.raises(FileFailure, match="no complete JSON"):
            parser.last()


def test_calibration_argv_is_r5_analysis_of_the_original_source():
    argv = calibration_argv(ffmpeg.Executable(FAKE_FFMPEG, (1, 2), "synthetic"), SOURCE, target_lufs=-16.0,
                            ceiling_dbtp=-1.35, dual_mono=True)
    assert af(argv) == "loudnorm=I=-16.0:TP=-1.35:LRA=50:dual_mono=true:print_format=json"
    assert argv[argv.index("-i") + 1] == str(SOURCE) and argv[-3:] == ("-f", "null", "-")
    assert "-xerror" in argv and "-nostats" in argv
    with pytest.raises(FileFailure, match="not clamped"):
        calibration_argv(ffmpeg.Executable(FAKE_FFMPEG, (1, 2), "synthetic"), SOURCE, target_lufs=-16.0,
                         ceiling_dbtp=-9.5, dual_mono=False)


# --- Planning (P2 row 2, R6) ---------------------------------------------------------------------


def test_dynamic_plan_carries_calibration_and_no_limiter_bounds():
    plan = dynamic_plan()
    assert plan.action == "dynamic" and plan.requested_peak_mode == "dynamic"
    assert dict(plan.dynamic_parameters) == PARAMS
    assert plan.limiter_threshold_dbtp == -1.0  # R6: the effective dynamic TP setting
    assert plan.limiter_reduction_db == plan.effective_limiter_bound_db == 0.0
    assert plan.gain_db == pytest.approx(7.0) and plan.expected_integrated_lufs == -16.0
    assert plan.predicted_true_peak_dbtp is None and plan.shortfall_lu == 0.0


def test_calibration_must_match_policy_target_and_ceiling():
    with pytest.raises(ValueError):
        plan_gain(measurement(), options(peak_mode="limit"), calibration=calibration())
    with pytest.raises(ValueError):
        plan_gain(measurement(), options(peak_mode="dynamic"), calibration=calibration(ceiling=-2.0))
    with pytest.raises(ValueError):
        plan_gain(measurement(), options(peak_mode="dynamic", target=-14.0), calibration=calibration())


@pytest.mark.parametrize("mode", ["limit", "reduce"])
@pytest.mark.parametrize("integrated,peak,cap", [(-23, -10, 3), (-23, -5, 3), (-23, -2, 3), (-23, -5, 0),
                                                 (-16, -0.5, 3), (-40, -1, 3)])
def test_dynamic_is_never_selected_implicitly(mode, integrated, peak, cap):
    source = measurement(integrated, peak)
    selected = options(peak_mode=mode, max_limit=cap)
    assert not needs_dynamic_calibration(source, selected)
    assert plan_gain(source, selected).action != "dynamic"


def test_v6_dynamic_ceiling_lowering_and_range():
    plan = dynamic_plan()
    bound = -0.85
    assert dynamic_peak_ceiling(plan, bound) == pytest.approx(-1.0 - (0.15 + 0.2))
    assert dynamic_peak_ceiling(replace(plan, limiter_threshold_dbtp=-8.9), bound) is None  # below TP −9: no clamp
    lowered = replan_dynamic_peak(plan, calibration(ceiling=-1.35, target_offset=0.4))
    assert lowered.limiter_threshold_dbtp == -1.35 and lowered.peak_corrections == 1
    assert lowered.dynamic_parameters["target_offset"] == 0.4
    assert lowered.loudness_corrections == lowered.reduce_fallbacks == 0
    with pytest.raises(ValueError):
        replan_dynamic_peak(lowered, calibration(ceiling=-1.5))  # the one peak budget is spent
    with pytest.raises(ValueError):
        replan_dynamic_peak(plan, calibration(ceiling=-0.5))  # never raises the ceiling


# --- next_step for dynamic (V3 steps 2 and 7, V6) ------------------------------------------------


def out(integrated=-16.0, peak=-1.2):
    return measurement(integrated, peak)


def test_dynamic_peak_failure_recalibrates_once_then_fails():
    source, plan = measurement(), dynamic_plan()
    asked = []

    def calibrate(ceiling):
        asked.append(ceiling)
        return calibration(ceiling=ceiling, target_offset=0.5)

    step = next_step(source, assess(out(peak=-0.8), plan), plan, calibrate=calibrate)
    assert isinstance(step, Retry) and step.kind == "peak_correction"
    assert asked == [pytest.approx(step.plan.limiter_threshold_dbtp)] and step.plan.limiter_threshold_dbtp < -1.2
    again = next_step(source, assess(out(peak=-0.8), step.plan), step.plan, calibrate=calibrate)
    assert isinstance(again, FileFailure) and again.reason == "verification_peak" and len(asked) == 1


def test_dynamic_loudness_miss_fails_without_calibration_or_mode_switch():
    source, plan = measurement(), dynamic_plan()

    def calibrate(ceiling):
        raise AssertionError("no calibration for a loudness miss")

    step = next_step(source, assess(out(integrated=-17.0), plan), plan, calibrate=calibrate)
    assert isinstance(step, FileFailure) and step.reason == "verification_loudness"


def test_dynamic_ceiling_out_of_range_fails_verification_peak_without_calibration():
    source, plan = measurement(), dynamic_plan(limiter_threshold_dbtp=-8.9)

    def calibrate(ceiling):
        raise AssertionError("no calibration outside loudnorm's TP range")

    step = next_step(source, assess(out(peak=-0.8), plan), plan, calibrate=calibrate)
    assert isinstance(step, FileFailure) and step.reason == "verification_peak" and "not clamped" in step.detail


def test_dynamic_peak_correction_needs_a_calibrate_callback():
    with pytest.raises(ValueError, match="calibrate"):
        next_step(measurement(), assess(out(peak=-0.8), dynamic_plan()), dynamic_plan())


# --- Dynamic render arguments (R5, R6) -----------------------------------------------------------


def test_dynamic_chain_is_r5_loudnorm_then_swr_to_the_target_rate():
    chain = filter_chain(dynamic_plan(), probe_of(rate=44100), wav24())
    assert chain == ("loudnorm=I=-16.0:TP=-1.0:LRA=50:measured_I=-23.55:measured_TP=-1.01:measured_LRA=1.0:"
                     "measured_thresh=-33.55:offset=0.62:linear=false:dual_mono=false:print_format=json,"
                     f"aresample=48000:{SWR_OPTIONS}")
    assert "volume=" not in chain and "alimiter" not in chain


def test_dynamic_chain_mono_policy_and_16_bit_dither_last():
    target = replace(wav24(channels=1), encoder="pcm_s16le", bit_depth=16, dither="triangular")
    chain = filter_chain(dynamic_plan(), probe_of(channels=1, layout="mono"), target)
    assert "dual_mono=true" in chain
    assert chain.endswith("aresample=osr=48000:osf=s16:dither_method=triangular:output_sample_bits=16")
    assert "dual_mono=false" in filter_chain(dynamic_plan(), probe_of(channels=1, layout="mono"), target,
                                             no_dual_mono=True)


# --- Lossy encoder arguments (R7) ----------------------------------------------------------------


def test_lossy_encoder_args_come_only_from_the_c04_record():
    assert encoder_args(lossy_target("mp3")) == ("-c:a", "libmp3lame", "-q:a", "0")
    assert encoder_args(lossy_target("mp3", quality=2)) == ("-c:a", "libmp3lame", "-q:a", "2")
    assert encoder_args(lossy_target("mp3", mode="cbr", bps=192000)) == ("-c:a", "libmp3lame", "-b:a", "192000")
    assert encoder_args(lossy_target("aac")) == ("-c:a", "aac_at", "-aac_at_mode", "cbr", "-b:a", "256000")
    argv = render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, dynamic_plan(), probe_of(), lossy_target("aac"))
    assert argv[-3:] == ("-f", "ipod", str(CANDIDATE)) and "-write_xing" not in argv
    for bad in (lossy_target("aac", encoder="aac"), lossy_target("mp3", encoder="lame"),
                replace(lossy_target("mp3"), dither="triangular")):
        with pytest.raises(ValueError):
            lossy_encoder_args(bad)


def test_r7_lossy_basis_must_list_the_applied_setting(tmp_path, env):
    try:
        record = ffmpeg.collect_aac_evidence(env, 48000, 2, "stereo", tmp_path / "ev")
    except Exception:
        pytest.skip("build lacks aac_at")
    if 256000 not in record.nominal_bitrates_bps:
        pytest.skip("AAC 256k is not evidenced on this build")
    target = lossy_target("aac", basis=record.evidence_id)
    assert revalidate_basis(env, target, (record,)) is record
    for bad in (replace(target, effective_bitrate_bps=384000, requested_bitrate_bps=384000),
                replace(target, sample_rate_hz=44100)):
        with pytest.raises(FileFailure) as caught:
            revalidate_basis(env, bad, (record,))
        assert caught.value.reason == "unsupported_target"


def test_aac_diagnostics_other_than_the_flush_note_are_a_mismatch():
    target = lossy_target("aac")
    evidence, bitrate = confirm_aac(target, ["Trying to remove 1024 samples, but the queue is empty"])
    assert bitrate == 256000 and "no aac_at setting diagnostic" in evidence
    with pytest.raises(FileFailure) as caught:
        confirm_aac(target, ["Bitrate 384000 not allowed; changing to 320000"])
    assert (caught.value.reason, caught.value.stage) == ("encoder_settings_mismatch", "render")


def test_lossy_probe_must_show_the_c04_codec_rate_and_channels():
    check_encoded_stream(probe_of(codec="aac", codec_class="lossy", container="mov", depth=None), lossy_target("aac"))
    with pytest.raises(FileFailure) as caught:
        check_encoded_stream(probe_of(codec="aac", codec_class="lossy", rate=44100, depth=None), lossy_target("aac"))
    assert caught.value.reason == "encoder_settings_mismatch"
    check_encoded_stream(probe_of(codec="weird"), wav24())  # lossless targets are header-confirmed at render


# --- CR-059 length rule (pure) -------------------------------------------------------------------


def frames_of(frames, rate=48000):
    return measurement(frames=frames, rate=rate)


def test_padding_allowance_by_target():
    assert trailing_padding_allowance(lossy_target("aac")) == 1023
    assert trailing_padding_allowance(lossy_target("mp3")) == 0
    assert trailing_padding_allowance(wav24()) == 0
    copied = replace(lossy_target("aac"), encoder="stream_copy", capability_basis="not_applicable", stream_copy=True,
                     bitrate_mode="none", requested_bitrate_bps=None, effective_bitrate_bps=None,
                     bitrate_source="not_applicable")
    assert trailing_padding_allowance(copied) == 0


@pytest.mark.parametrize("frames,padding,ok", [
    (1440000, 0, True), (1440001, 0, False), (1439999, 0, False),            # the lossless/MP3 rule is unchanged
    (1440000, 1023, True), (1440448, 1023, True), (1441023, 1023, True),
    (1441024, 1023, False),                                                   # untrimmed priming or a second frame
    (1439999, 1023, False), (1438976, 1023, False),                           # a truncated encode never passes
])
def test_same_rate_length_rule_with_trailing_padding(frames, padding, ok):
    if ok:
        check_candidate_length(1440000, 48000, frames_of(frames), trailing_padding=padding)
    else:
        with pytest.raises(FileFailure) as caught:
            check_candidate_length(1440000, 48000, frames_of(frames), trailing_padding=padding)
        assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")


def test_resampled_length_rule_with_trailing_padding():
    # 30 s at 44.1 kHz → 1,440,000 at 48 kHz, ±30 frames, plus AAC padding above only.
    check_candidate_length(1323000, 44100, frames_of(1440000 + 30 + 1023), trailing_padding=1023)
    check_candidate_length(1323000, 44100, frames_of(1440000 - 30), trailing_padding=1023)
    for frames in (1440000 + 31 + 1023, 1440000 - 31):
        with pytest.raises(FileFailure):
            check_candidate_length(1323000, 44100, frames_of(frames), trailing_padding=1023)


# --- Real-tool helpers ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


@pytest.fixture(scope="module")
def env(executable):
    return ffmpeg.probe_environment(executable)


def stat_identity(path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


def decode(path: Path, executable) -> tuple[np.ndarray, int]:
    """The first audio stream decoded by FFmpeg to float64, for the independent oracles."""
    probe = subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                            "-f", "f64le", "-"], capture_output=True, check=True)
    info = ffprobe.probe(ffprobe.locate_ffprobe(), path)
    return np.frombuffer(probe.stdout, dtype="<f8").reshape(-1, info.channels), info.sample_rate_hz


def independent(path: Path, executable) -> tuple[float, float]:
    """pyloudnorm I and SciPy 4× (Kaiser β12) true peak of the decoded file: the TESTING § 2 oracles."""
    samples, rate = decode(path, executable)
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.), padtype="constant")))
    return integrated, 20 * np.log10(peak)


def lossless_records(env, probe, folder):
    return tuple(ffmpeg.collect_lossless_evidence(env, encoder, probe.sample_rate_hz, probe.channels,
                                                  probe.channel_layout, folder) for encoder in ("pcm_s24le",))


class Session:
    """measure → [calibrate] → plan → (render → probe → verify) per attempt, as the pipeline will sequence them."""

    def __init__(self, fixture, tmp_path, env, executable, prober, output: TargetFormat | None = None,
                 records=(), **changes):
        self.fixture, self.tmp, self.env, self.executable, self.prober = fixture, tmp_path, env, executable, prober
        self.options = options(tmp_path, **changes)
        self.probe = ffprobe.probe(prober, fixture)
        self.source = measure(fixture, self.probe, executable)
        self.calibrations: list[DynamicCalibration] = []
        initial = None
        if needs_dynamic_calibration(self.source, self.options):
            initial = self.calibrate(self.options.true_peak)
        self.plan = plan_gain(self.source, self.options, calibration=initial)
        if output is None:
            records = lossless_records(env, self.probe, tmp_path / "ev")
            output = replace(wav24(self.probe.sample_rate_hz, self.probe.channels),
                             channel_layout=self.probe.channel_layout, capability_basis=records[0].evidence_id)
        self.target, self.records = output, tuple(records)
        self.argvs: list[tuple[str, ...]] = []
        self.results = []
        self.inject: dict[int, dict] = {}

    def calibrate(self, ceiling):
        result = calibrate_dynamic(self.fixture, self.probe, self.executable, target_lufs=self.options.target,
                                   ceiling_dbtp=ceiling)
        self.calibrations.append(result)
        return result

    def attempt(self, plan, number):
        candidate = self.tmp / f"attempt{number}{self.target.extension}"
        candidate.write_bytes(b"")
        self.argvs.append(render_argv(self.executable, self.fixture, candidate, plan, self.probe, self.target))
        result = render(self.fixture, stat_identity(self.fixture), self.probe, plan, self.target, candidate,
                        self.env, self.records, attempt=number)
        verification = verify_candidate(result, self.fixture, self.probe, ffprobe.probe(self.prober, candidate),
                                        self.executable, source_frames=self.source.decoded_frames)
        self.results.append((result, verification))
        if number in self.inject:  # fault injection on the real measurement
            verification = assess(replace(verification.measurement, **self.inject[number]), plan)
        return result, verification

    def run(self):
        return run_ladder(self.source, self.plan, self.attempt, calibrate=self.calibrate)


def input_of(argv):
    return Path(argv[argv.index("-i") + 1])


# --- TST-A-09 dynamic (real tools) ---------------------------------------------------------------


def test_a09_dynamic_on_f05_reaches_target_with_safe_peaks(fixture_path, tmp_path, env, executable, prober):
    fixture = fixture_path("F-05")
    session = Session(fixture, tmp_path, env, executable, prober, peak_mode="dynamic")
    (calibrated,) = session.calibrations
    plan = session.plan
    assert plan.action == "dynamic" and dict(plan.dynamic_parameters) == dict(calibrated.parameters)
    assert calibrated.normalization_type == "dynamic"
    # The calibration is the renderer's input, not the planning meter: ebur128 stays the oracle (M1, R5).
    assert abs(calibrated.parameters["input_i"] - session.source.integrated_lufs) < 1.0
    outcome = session.run()
    assert len(outcome.renders) == 1 and outcome.plan.action == "dynamic"
    chain = af(session.argvs[0])
    assert chain.startswith(f"loudnorm=I=-16.0:TP=-1.0:LRA=50:measured_I={calibrated.parameters['input_i']!r}:")
    assert "linear=false" in chain and chain.endswith(f"aresample=48000:{SWR_OPTIONS}")
    assert outcome.result.normalization_type == "dynamic"
    out = outcome.verification.measurement
    oracle_i, oracle_tp = independent(outcome.result.candidate_path, executable)
    assert abs(out.integrated_lufs + 16.0) <= .5 and abs(oracle_i + 16.0) <= .5
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9 and oracle_tp <= -.9
    assert out.decoded_frames == session.source.decoded_frames  # the lossless length rule, unchanged


def test_a09_no_implicit_dynamic_on_f05_under_the_default_policy(fixture_path, tmp_path, env, executable, prober):
    session = Session(fixture_path("F-05"), tmp_path, env, executable, prober)
    assert session.calibrations == [] and session.plan.action != "dynamic"
    outcome = session.run()
    assert all(r.plan.action != "dynamic" for r in outcome.renders)
    assert all("loudnorm" not in " ".join(argv) for argv in session.argvs)


def test_a09_dynamic_peak_retry_recalibrates_the_original_source(fixture_path, tmp_path, env, executable, prober):
    """V6: an (injected) unsafe first candidate lowers the ceiling, recalibrates and renders once more."""
    fixture = fixture_path("F-05")
    session = Session(fixture, tmp_path, env, executable, prober, peak_mode="dynamic")
    session.inject = {1: {"true_peak_linear": 10 ** (-.85 / 20), "true_peak_dbtp": -.85}}
    outcome = session.run()
    assert len(outcome.renders) == 2 <= MAX_RENDERS and all(input_of(argv) == fixture for argv in session.argvs)
    lowered = outcome.plan.limiter_threshold_dbtp
    assert lowered == pytest.approx(-1.0 - (conservative_peak_dbtp(10 ** (-.85 / 20), .001) + 1.0 + .2))
    assert [c.ceiling_dbtp for c in session.calibrations] == [-1.0, lowered]
    assert f"TP={lowered!r}:" in af(session.argvs[1]) and outcome.plan.peak_corrections == 1
    assert dict(outcome.plan.dynamic_parameters) == dict(session.calibrations[1].parameters)
    out = outcome.verification.measurement
    assert abs(out.integrated_lufs + 16.0) <= .5
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9


# --- TST-A-06 lossy encodes measured after decoding (real tools) ---------------------------------


@pytest.fixture(scope="module")
def lossy_records(env, tmp_path_factory):
    if not {"libmp3lame", "aac_at"} <= env.encoders:
        pytest.skip("build lacks aac_at or libmp3lame")
    folder = tmp_path_factory.mktemp("lossy-evidence")
    mp3 = ffmpeg.collect_mp3_v0_evidence(env, 48000, 2, "stereo", folder)
    aac = ffmpeg.collect_aac_evidence(env, 48000, 2, "stereo", folder)
    if 0 not in mp3.vbr_qualities or 256000 not in aac.nominal_bitrates_bps:
        pytest.skip("MP3 V0 or AAC 256k is not evidenced for 48 kHz stereo on this build")
    return {"mp3": mp3, "aac": aac}


def lossy_session(fixture, tmp_path, env, executable, prober, records, fmt, **changes):
    record = records[fmt]
    target = lossy_target(fmt, basis=record.evidence_id)
    return Session(fixture, tmp_path, env, executable, prober, output=target, records=(record,), **changes)


@pytest.mark.parametrize("fmt", ["mp3", "aac"])
@pytest.mark.parametrize("fixture_id", ["F-04", "F-05"])
def test_a06_lossy_outputs_are_peak_safe_after_decoding(fixture_path, tmp_path, env, executable, prober,
                                                        lossy_records, fmt, fixture_id):
    fixture = fixture_path(fixture_id)
    session = lossy_session(fixture, tmp_path, env, executable, prober, lossy_records, fmt)
    outcome = session.run()
    assert len(outcome.renders) <= MAX_RENDERS and all(input_of(argv) == fixture for argv in session.argvs)
    result = outcome.result
    assert result.encoder_settings_valid
    if fmt == "mp3":
        assert "-q:a" in session.argvs[0] and result.encoder_effective_bitrate_bps is None
        assert "Xing VBR tag" in result.encoder_setting_evidence
    else:
        assert result.encoder_effective_bitrate_bps == 256000
        assert session.argvs[0][session.argvs[0].index("-b:a") + 1] == "256000"
    out = outcome.verification.measurement
    oracle_i, oracle_tp = independent(result.candidate_path, executable)
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9 and oracle_tp <= -.9
    if outcome.plan.action == "reduce":
        assert [w.code for w in outcome.warnings] == ["reduce_shortfall"]
    else:
        assert abs(out.integrated_lufs + 16.0) <= .5 and abs(oracle_i + 16.0) <= .5
    extra = out.decoded_frames - session.source.decoded_frames
    assert extra == 0 if fmt == "mp3" else 0 <= extra <= 1023  # CR-059, measured on the real output
    lag = _lag(decode(fixture, executable)[0][:, 0], decode(result.candidate_path, executable)[0][:, 0])
    assert lag == 0  # priming removed: the encode starts where the source starts


@pytest.mark.parametrize("fmt", ["mp3", "aac"])
@pytest.mark.parametrize("fixture_id,variant", [("F-04", None), ("F-05", "le")])
def test_a06_ceiling_only_linear_encodes_are_made_peak_safe(fixture_path, tmp_path, env, executable, prober,
                                                             lossy_records, fmt, fixture_id, variant):
    """C7's recipe: a linear plan whose predicted peak sits just under c; the decoded encode decides.

    F-04 → MP3 V0 overshoots (CP1 R-8: −0.7377 dBTP SciPy), so V5's peak
    correction must run and the published candidate is safe; every other
    combination must be safe, by whatever bounded path the ladder takes.
    """
    fixture = fixture_path(fixture_id, variant)
    source = measure(fixture, ffprobe.probe(prober, fixture), executable)
    target_lufs = math.floor((source.integrated_lufs - 1.0 - source.true_peak_dbtp) * 1000) / 1000
    session = lossy_session(fixture, tmp_path, env, executable, prober, lossy_records, fmt, target=target_lufs)
    assert session.plan.action == "linear"
    outcome = session.run()
    assert len(outcome.renders) <= MAX_RENDERS and all(input_of(argv) == fixture for argv in session.argvs)
    first = session.results[0][1].measurement
    if (fixture_id, fmt) == ("F-04", "mp3"):
        assert conservative_peak_dbtp(first.true_peak_linear, first.peak_resolution_linear) > -.9
        assert outcome.renders[1].plan.action == "limit" and outcome.renders[1].plan.peak_corrections == 1
    out = outcome.verification.measurement
    oracle_i, oracle_tp = independent(outcome.result.candidate_path, executable)
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9 and oracle_tp <= -.9
    if outcome.plan.action != "reduce":
        assert abs(out.integrated_lufs - target_lufs) <= .5 and abs(oracle_i - target_lufs) <= .5


def _lag(reference, output, start=2 * 48000 - 2000, length=4480, window=2500):
    """Best-correlating shift over a span holding F-05's first burst (F-04 is noise throughout)."""
    ref = reference[start:start + length]
    scores = {k: float(np.dot(ref, output[start + k:start + k + length])) for k in range(-window, window + 1)}
    return max(scores, key=scores.get)


def test_a06_dynamic_to_aac_combines_both(fixture_path, tmp_path, env, executable, prober, lossy_records):
    session = lossy_session(fixture_path("F-05"), tmp_path, env, executable, prober, lossy_records, "aac",
                            peak_mode="dynamic")
    outcome = session.run()
    assert outcome.plan.action == "dynamic" and outcome.result.normalization_type == "dynamic"
    out = outcome.verification.measurement
    assert abs(out.integrated_lufs + 16.0) <= .5
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9


# --- CR-059 on real outputs: a truncated encode still fails --------------------------------------


@pytest.mark.parametrize("fmt", ["mp3", "aac"])
def test_length_rule_fails_a_truncated_real_encode(fixture_path, tmp_path, env, executable, prober,
                                                   lossy_records, fmt):
    """An encode that lost its last 1,152 source frames fails even with the AAC padding allowance."""
    fixture = fixture_path("F-04")
    probe = ffprobe.probe(prober, fixture)
    source = measure(fixture, probe, executable)
    target = lossy_target(fmt, basis=lossy_records[fmt].evidence_id)
    truncated = tmp_path / f"truncated{target.extension}"
    keep = source.decoded_frames - 1152
    subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-y", "-i", str(fixture), "-map", "0:a:0",
                    "-af", f"atrim=end_sample={keep}", *lossy_encoder_args(target), "-f",
                    "mp3" if fmt == "mp3" else "ipod", str(truncated)], check=True)
    candidate = measure(truncated, ffprobe.probe(prober, truncated), executable)
    assert keep <= candidate.decoded_frames <= keep + trailing_padding_allowance(target)
    with pytest.raises(FileFailure) as caught:
        check_candidate_length(source.decoded_frames, probe.sample_rate_hz, candidate,
                               trailing_padding=trailing_padding_allowance(target))
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")


def test_aac_setting_change_reported_by_the_encoder_is_a_mismatch(fixture_path, tmp_path, env, executable, prober,
                                                                  lossy_records):
    """A doctored record that 'evidences' 384k: aac_at clamps to 320k and says so; the render fails (R7)."""
    record = replace(lossy_records["aac"], nominal_bitrates_bps=(384000,))
    target = lossy_target("aac", bps=384000, basis=record.evidence_id)
    fixture = fixture_path("F-05", "le")
    probe = ffprobe.probe(prober, fixture)
    candidate = tmp_path / "clamped.m4a"
    candidate.write_bytes(b"")
    plan = plan_gain(measure(fixture, probe, executable), options(tmp_path))
    with pytest.raises(FileFailure) as caught:
        render(fixture, stat_identity(fixture), probe, plan, target, candidate, env, (record,))
    assert (caught.value.reason, caught.value.stage) == ("encoder_settings_mismatch", "render")
    assert "changing to 320000" in caught.value.detail
