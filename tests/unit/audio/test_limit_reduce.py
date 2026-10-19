"""TK-0110: limit and reduce renders with the bounded retry ladder, against the installed FFmpeg.

Real-tool tests write only under tmp_path; sources come from the WS-07
catalogue (``fixture_path``). Acceptance destinations (test plan):

- TST-A-05 — F-05/le at −14 and −12.5 LUFS: initial ``limit``, target ±0.5 LU
  on ebur128 and pyloudnorm, or the verified WARN reduce fallback; both true
  peak meters ≤ −0.9 dBTP; policy bound ≤ 3 dB, effective bound separate.
  F-05/c6 at −21.5 keeps the cap/meter counterexample within the budget.
- TST-A-09 — F-05/hc ``reduce`` at −14 and −12.5: no limiter, safe peak,
  measured shortfall within 0.2 LU of an independent measurement.
- No timing shift after limiting (R4 ``latency=1``): equal frame counts and
  impulse/burst positions; TST-A-12 determinism of a limited render.
- TST-A-07 — injected misses drive real renders through the ladder: at most
  five, every one reading the original source, budgets never reset.
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
    ALIMITER_SETTINGS, RENDER_ACTIONS, S2_ACTIONS, SWR_OPTIONS, filter_chain, render,
    render_argv,
)
from norma.audio.verify import MAX_RENDERS, assess, decoded_digest, run_ladder, verify_candidate
from norma.errors import FileFailure
from norma.formats import capabilities as cap
from norma.formats.resolve import finalise_target
from norma.models import GainPlan, ProbeResult, RunOptions, TargetFormat

FAKE_FFMPEG = Path("/unused/ffmpeg")
SOURCE = Path("/unused/source.wav")
CANDIDATE = Path("/unused/out/candidate.wav")


# --- Synthetic records ---------------------------------------------------------------------------


def probe_of(*, rate=48000, depth=24, channels=2, layout="stereo", duration=30.0):
    return ProbeResult("pcm_s24le", "lossless", "wav", 0, 0, rate, depth, False, "s32", channels, layout,
                       duration, "stream", None, False, 0, {}, "synthetic")


def target_of(fmt="wav", *, depth=24, rate=48000, dither="none"):
    encoder = {"wav": f"pcm_s{depth}le", "aiff": f"pcm_s{depth}be"}.get(fmt, fmt)
    return TargetFormat(fmt, cap.OUTPUT_EXTENSIONS[fmt], encoder, depth, False, rate, 2, "stereo", dither, "none",
                        None, None, None, "not_applicable", "ev:synthetic", False, "as_requested")


def limit_plan(gain=6.0, threshold=-1.3, **changes) -> GainPlan:
    return GainPlan(action="limit", requested_peak_mode="limit", target_lufs=-14.0, ceiling_dbtp=-1.0,
                    tolerance_lu=.5, max_limit_db=3.0, gain_db=gain, predicted_true_peak_dbtp=0.5,
                    limiter_reduction_db=1.5, effective_limiter_bound_db=1.8, limiter_threshold_dbtp=threshold,
                    expected_integrated_lufs=-14.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok", **changes)


def reduce_plan(gain=2.5) -> GainPlan:
    return replace(limit_plan(), action="reduce", requested_peak_mode="reduce", gain_db=gain,
                   limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
                   expected_integrated_lufs=-20.0, shortfall_lu=6.0, reason_code="reduce_shortfall")


def af(argv):
    return argv[argv.index("-af") + 1] if "-af" in argv else None


LIMITER = (f"aresample=192000:{SWR_OPTIONS},alimiter=limit={10 ** (-1.3 / 20)!r}:{ALIMITER_SETTINGS},"
           f"aresample=48000:{SWR_OPTIONS}")


# --- Pure argument building ----------------------------------------------------------------------


def test_render_actions_and_the_unchanged_pipeline_gate():
    assert RENDER_ACTIONS == {"copy", "linear", "limit", "reduce", "dynamic"}  # dynamic since TK-0111
    assert S2_ACTIONS == {"copy", "linear"}  # the pipeline's own gate moves at TK-0419
    assert ALIMITER_SETTINGS == "attack=5:release=50:level_in=1:level_out=1:asc=0:level=0:latency=1"


def test_limit_chain_is_r4_oversampled_limiter_after_float_gain():
    chain = filter_chain(limit_plan(), probe_of(), target_of())
    assert chain == f"aformat=sample_fmts=fltp|flt,volume=6.0dB:precision=float,{LIMITER}"
    argv = render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, limit_plan(), probe_of(), target_of())
    assert argv.count("-af") == 1 and "loudnorm" not in " ".join(argv)


def test_limit_chain_resamples_to_target_first_and_dithers_last():
    chain = filter_chain(limit_plan(), probe_of(rate=96000), target_of(depth=16, dither="triangular"))
    assert chain == (f"aformat=sample_fmts=fltp|flt,aresample=48000:{SWR_OPTIONS},volume=6.0dB:precision=float,"
                     f"{LIMITER},aresample=osr=48000:osf=s16:dither_method=triangular:output_sample_bits=16")


def test_limit_at_zero_gain_still_processes_in_float():
    chain = filter_chain(limit_plan(gain=0.0), probe_of(), target_of())
    assert chain == f"aformat=sample_fmts=fltp|flt,{LIMITER}"


def test_reduce_is_plain_gain_without_limiter():
    chain = filter_chain(reduce_plan(), probe_of(), target_of())
    assert chain == "volume=2.5dB:precision=float"
    assert "alimiter" not in " ".join(render_argv(FAKE_FFMPEG, SOURCE, CANDIDATE, reduce_plan(), probe_of(),
                                                  target_of()))


def test_out_of_range_threshold_fails_before_any_process(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("no process may start")
    monkeypatch.setattr(ffmpeg, "run_process", refuse)
    for threshold in (-24.2, 0.5):
        with pytest.raises(FileFailure) as caught:
            filter_chain(limit_plan(threshold=threshold), probe_of(), target_of())
        assert (caught.value.reason, caught.value.stage) == ("render_error", "render")
        assert "not clamped" in caught.value.detail
    with pytest.raises(ValueError):
        filter_chain(replace(reduce_plan(), action="limit", requested_peak_mode="limit"), probe_of(), target_of())


def test_limit_chain_to_a_lossy_target_has_no_dither():
    """Lossy encodes since TK-0111: the same limiter chain, no dither or depth step (R3, R7)."""
    target = TargetFormat("mp3", ".mp3", "libmp3lame", None, False, 48000, 2, "stereo", "none", "vbr", None, None, 0,
                          "auto", "ev:synthetic", False, "as_requested")
    assert filter_chain(limit_plan(), probe_of(), target) == (
        f"aformat=sample_fmts=fltp|flt,volume=6.0dB:precision=float,{LIMITER}")


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


def evidence(env, encoder, probe, folder):
    return ffmpeg.collect_lossless_evidence(env, encoder, probe.sample_rate_hz, probe.channels,
                                            probe.channel_layout, folder)


def run_options(tmp_path, **changes):
    return RunOptions(command="normalise", ffmpeg=Path("/usr/local/bin/ffmpeg"), cwd=tmp_path, jobs=1, **changes)


def independent(path):
    """pyloudnorm I and SciPy 4× (Kaiser β12) true peak: the TESTING § 2 oracles."""
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.), padtype="constant")))
    return integrated, 20 * np.log10(peak)


class Session:
    """measure → plan → (finalise C-04 → render → verify) per attempt, as the pipeline will sequence them."""

    def __init__(self, fixture, tmp_path, env, executable, prober, **changes):
        self.fixture, self.tmp, self.env, self.executable, self.prober = fixture, tmp_path, env, executable, prober
        self.options = run_options(tmp_path, **changes)
        self.probe = ffprobe.probe(prober, fixture)
        self.source = measure(fixture, self.probe, executable)
        self.plan = plan_gain(self.source, self.options)
        self.records = tuple(evidence(env, encoder, self.probe, tmp_path / "ev")
                             for encoder in ("pcm_s16le", "pcm_s24le", "pcm_f32le"))
        self.basis = cap.ToolBasis(executable.path, executable.identity, executable.version, env.platform_basis)
        self.argvs: list[tuple[str, ...]] = []
        self.inject: dict[int, dict] = {}

    def attempt(self, plan, number):
        target = finalise_target(self.probe, self.options, self.records, self.basis, action=plan.action)
        candidate = self.tmp / f"attempt{number}{target.extension}"
        candidate.write_bytes(b"")
        self.argvs.append(render_argv(self.executable, self.fixture, candidate, plan, self.probe, target))
        result = render(self.fixture, stat_identity(self.fixture), self.probe, plan, target, candidate, self.env,
                        self.records, attempt=number)
        verification = verify_candidate(result, self.fixture, self.probe, ffprobe.probe(self.prober, candidate),
                                        self.executable, source_frames=self.source.decoded_frames)
        if number in self.inject:  # TST-A-07 fault injection on the real measurement
            verification = assess(replace(verification.measurement, **self.inject[number]), plan)
        return result, verification

    def run(self, before_retry=None):
        return run_ladder(self.source, self.plan, self.attempt, before_retry)


def input_of(argv):
    return Path(argv[argv.index("-i") + 1])


# --- TST-A-05 limit --------------------------------------------------------------------------------


@pytest.mark.parametrize("target_lufs", [-14.0, -12.5])
def test_a05_f05_le_limit_reaches_target_or_verified_fallback(fixture_path, tmp_path, env, executable, prober,
                                                               target_lufs):
    fixture = fixture_path("F-05", "le")
    session = Session(fixture, tmp_path, env, executable, prober, target=target_lufs)
    initial = session.plan
    assert initial.action == "limit" and initial.limiter_threshold_dbtp == pytest.approx(-1.3)
    assert 0 < initial.limiter_reduction_db <= 3.0
    assert initial.effective_limiter_bound_db == pytest.approx(initial.limiter_reduction_db + .3)
    outcome = session.run()
    assert len(outcome.renders) <= MAX_RENDERS
    assert all(input_of(argv) == fixture for argv in session.argvs)
    assert "alimiter=limit=" in af(session.argvs[0])
    out = outcome.verification.measurement
    oracle_i, oracle_tp = independent(outcome.result.candidate_path)
    if outcome.plan.action == "limit":
        assert abs(out.integrated_lufs - target_lufs) <= .5 and abs(oracle_i - target_lufs) <= .5
        assert outcome.warnings == ()
    else:
        assert outcome.plan.action == "reduce" and outcome.plan.fallback_from is not None
        assert [w.code for w in outcome.warnings] == ["reduce_shortfall"]
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9
    assert oracle_tp <= -.9
    assert out.decoded_frames == session.source.decoded_frames


def test_a05_f05_c6_counterexample_stays_within_budget(fixture_path, tmp_path, env, executable, prober):
    """C6: the cap-safe −21.5 plan; any correction is cap-checked, and the result is verified, not assumed."""
    fixture = fixture_path("F-05", "c6")
    session = Session(fixture, tmp_path, env, executable, prober, target=-21.5)
    assert session.plan.action in ("linear", "limit") and session.plan.limiter_reduction_db <= 3.0
    retries = []
    outcome = session.run(lambda old, new, kind: retries.append((kind, new.limiter_reduction_db)))
    assert len(outcome.renders) <= MAX_RENDERS and all(input_of(argv) == fixture for argv in session.argvs)
    assert all(policy <= 3.0 for _, policy in retries)
    out = outcome.verification.measurement
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9
    if outcome.plan.action == "reduce":
        assert outcome.plan.fallback_from in ("linear", "limit") and outcome.warnings[0].code == "reduce_shortfall"
    else:
        assert abs(out.integrated_lufs + 21.5) <= .5


# --- TST-A-09 reduce -------------------------------------------------------------------------------


@pytest.mark.parametrize("target_lufs", [-14.0, -12.5])
@pytest.mark.parametrize("peak_mode", ["reduce", "limit"])
def test_a09_f05_hc_reduce_measured_shortfall(fixture_path, tmp_path, env, executable, prober, target_lufs,
                                              peak_mode):
    """Explicit reduce, and the default policy whose cap F-05/hc exceeds (P2 last row)."""
    fixture = fixture_path("F-05", "hc")
    session = Session(fixture, tmp_path, env, executable, prober, target=target_lufs, peak_mode=peak_mode)
    assert session.plan.action == "reduce" and session.plan.fallback_from is None
    assert session.plan.gain_db == pytest.approx(-1.0 - session.source.true_peak_dbtp)
    outcome = session.run()
    assert len(outcome.renders) == 1 and outcome.plan.action == "reduce"
    assert "alimiter" not in " ".join(session.argvs[0])
    out = outcome.verification.measurement
    oracle_i, oracle_tp = independent(outcome.result.candidate_path)
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9 and oracle_tp <= -.9
    shortfall = outcome.verification.measured_shortfall_lu
    assert shortfall == pytest.approx(max(0.0, target_lufs - out.integrated_lufs))
    assert shortfall > 0 and abs(shortfall - max(0.0, target_lufs - oracle_i)) <= .2
    assert [w.code for w in outcome.warnings] == ["reduce_shortfall"]


# --- No timing shift after limiting (R4 latency=1); determinism -------------------------------------


def lossless_target(env, probe, folder, *, sample_float=False):
    encoder = "pcm_f32le" if sample_float else "pcm_s24le"
    record = evidence(env, encoder, probe, folder)
    target = TargetFormat("wav", ".wav", encoder, 32 if sample_float else 24, sample_float, probe.sample_rate_hz,
                          probe.channels, probe.channel_layout, "none", "none", None, None, None, "not_applicable",
                          record.evidence_id, False, "as_requested")
    return target, record


def limited_render(fixture, folder, env, prober, plan, name, *, sample_float=True):
    probe = ffprobe.probe(prober, fixture)
    target, record = lossless_target(env, probe, folder / f"ev-{name}", sample_float=sample_float)
    candidate = folder / f"{name}.wav"
    candidate.write_bytes(b"")
    render(fixture, stat_identity(fixture), probe, plan, target, candidate, env, (record,))
    return candidate


def lag(reference, output, window=40):
    """The integer shift maximising the correlation of ``output`` against ``reference``."""
    scores = {k: float(np.dot(reference[window:-window], output[window + k:len(output) - window + k]))
              for k in range(-window, window + 1)}
    return max(scores, key=scores.get)


def test_limiting_keeps_impulse_positions_and_frame_count(fixture_path, tmp_path, env, prober):
    """F-03/endpoint at +20 dB under a −12 dBTP threshold: the 0.3 impulse is limited, nothing moves."""
    fixture = fixture_path("F-03", "endpoint")
    plan = replace(limit_plan(gain=20.0, threshold=-12.0), ceiling_dbtp=-9.0, limiter_reduction_db=0.0,
                   effective_limiter_bound_db=0.0)
    candidate = limited_render(fixture, tmp_path, env, prober, plan, "endpoint")
    source, _ = sf.read(fixture, dtype="float64", always_2d=True)
    output, _ = sf.read(candidate, dtype="float64", always_2d=True)
    assert output.shape == source.shape
    for position in (1000,):
        window = output[position - 20:position + 21, 0]
        assert position - 20 + int(np.argmax(np.abs(window))) == position
    assert np.max(np.abs(output[980:1021, 0])) < .3 * 10 ** (20 / 20) * .5      # it was limited
    assert lag(source[:, 0], output[:, 0]) == 0


def test_limiting_f05_le_keeps_burst_alignment(fixture_path, tmp_path, env, prober):
    fixture = fixture_path("F-05", "le")
    plan = limit_plan(gain=11.4)
    candidate = limited_render(fixture, tmp_path, env, prober, plan, "le", sample_float=False)
    source, rate = sf.read(fixture, dtype="float64", always_2d=True)
    output, _ = sf.read(candidate, dtype="float64", always_2d=True)
    assert output.shape == source.shape and rate == 48000
    for start in (2, 7, 12, 17, 22, 27):
        span = slice(start * rate - 2000, start * rate + 2480)
        assert lag(source[span, 0], output[span, 0]) == 0


def test_a12_limited_render_is_deterministic(fixture_path, tmp_path, env, executable, prober):
    fixture = fixture_path("F-05", "le")
    first = limited_render(fixture, tmp_path, env, prober, limit_plan(gain=11.4), "one", sample_float=False)
    second = limited_render(fixture, tmp_path, env, prober, limit_plan(gain=11.4), "two", sample_float=False)
    assert decoded_digest(first, executable) == decoded_digest(second, executable)


# --- TST-A-07 retry caps with real renders -----------------------------------------------------------


def test_a07_injected_misses_exhaust_budgets_with_original_source_renders(fixture_path, tmp_path, env, executable,
                                                                          prober):
    """Attempt 1: injected unsafe peak; 2–4: injected loudness misses; the fallback is measured for real."""
    fixture = fixture_path("F-05", "le")
    session = Session(fixture, tmp_path, env, executable, prober, target=-14.0)
    # Sizes keep every correction inside the 3 dB cap: policy 1.33 + adjustment 0.36 + 0.55 + 0.55 ≈ 2.8.
    hot = 10 ** (-.85 / 20)
    session.inject = {1: {"true_peak_linear": hot, "true_peak_dbtp": -.85},
                      2: {"integrated_lufs": -14.55}, 3: {"integrated_lufs": -14.55}, 4: {"integrated_lufs": -14.55}}
    kinds = []
    outcome = session.run(lambda old, new, kind: kinds.append(kind))
    assert kinds == ["peak_correction", "loudness_correction", "loudness_correction", "reduce_fallback"]
    assert len(outcome.renders) == MAX_RENDERS == len(session.argvs)
    assert all(input_of(argv) == fixture for argv in session.argvs)
    assert [r.attempt for r in outcome.renders] == [1, 2, 3, 4, 5]
    counters = [(r.plan.loudness_corrections, r.plan.peak_corrections, r.plan.reduce_fallbacks)
                for r in outcome.renders]
    assert counters == [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1)]
    lowered = outcome.renders[1].plan.limiter_threshold_dbtp
    assert lowered < -1.3 and f"alimiter=limit={10 ** (lowered / 20)!r}:" in af(session.argvs[1])
    assert all(r.plan.limiter_threshold_dbtp == lowered for r in outcome.renders[1:4])
    assert "alimiter" not in af(session.argvs[4]) and outcome.plan.fallback_from == "limit"
    out = outcome.verification.measurement
    assert conservative_peak_dbtp(out.true_peak_linear, out.peak_resolution_linear) <= -.9
    assert [w.code for w in outcome.warnings] == ["reduce_shortfall"]
    assert all(r.candidate_path.exists() for r in outcome.renders)  # caller owns candidate disposal


def test_a07_unsafe_reduce_fallback_fails_without_a_sixth_render(fixture_path, tmp_path, env, executable, prober):
    fixture = fixture_path("F-05", "le")
    session = Session(fixture, tmp_path, env, executable, prober, target=-14.0)
    session.inject = {1: {"integrated_lufs": -14.55}, 2: {"integrated_lufs": -14.55}, 3: {"integrated_lufs": -14.55},
                      4: {"true_peak_linear": 10 ** (-.6 / 20), "true_peak_dbtp": -.6}}
    with pytest.raises(FileFailure) as caught:
        session.run()
    assert (caught.value.reason, caught.value.stage) == ("verification_peak", "verify")
    assert len(session.argvs) == 4 and "alimiter" not in af(session.argvs[3])
