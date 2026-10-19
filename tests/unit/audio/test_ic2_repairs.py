"""IC-2 repairs (TK-0109): long-file measurement (F-2, CR-053) and candidate length (F-3, CR-054).

F-2: FFmpeg progress text ends in a bare CR and, without ``-nostats``, joins the
next metadata line once a scan lasts about half a second. The real-file cases
reproduce Appendix A P4 (a file longer than four minutes, generated under
tmp_path at a modest rate) through ``measure`` and through ``norma analyse``.

F-3: a render that exits 0 but writes too little audio must fail
``render_error`` at stage verify. The real cases reproduce Appendix A P7 (e)
through ``verify_candidate`` and through ``norma normalise``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from norma import cli, pipeline
from norma.adapters import ffmpeg, ffprobe
from norma.audio import measure as meter
from norma.audio.measure import MeterParser, measure, measurement_argv, sample_count_argv
from norma.audio.render import render
from norma.audio.verify import check_candidate_length, verify_candidate
from norma.errors import FileFailure
from norma.formats import capabilities as cap
from norma.models import GainPlan, LoudnessMeasurement, TargetFormat

LONG_S = 300
"""Longer than four minutes; P4 failed at 180 s and above on the development Mac."""
LONG_RATE = 48000


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


@pytest.fixture(scope="module")
def env(executable):
    return ffmpeg.probe_environment(executable)


@pytest.fixture
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory cleanup.
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def long_tone(folder: Path, executable) -> Path:
    """A 300 s stereo 16-bit 48 kHz sine (58 MB) made by FFmpeg: lavfi peak 1/8, +12 dB, −3 dB mono-to-stereo upmix.

    [verified] Before the fix each meter scan of this file took about 0.8–1 s on the development Mac
    (arm64, FFmpeg N-125892, 2026-10-11) and FFmpeg's first progress line, at 0.5 s, joined a
    metadata line. A much faster machine may finish a scan before 0.5 s and so not reproduce the
    old failure; the argv and parser tests above guard the repair on any machine.
    """
    path = folder / "long.wav"
    subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate={LONG_RATE}:duration={LONG_S}", "-af", "volume=12dB", "-ac", "2",
                    "-c:a", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact", str(path)], check=True)
    return path


def tone(folder: Path, name: str, *, seconds=4.0, level=-20.0, rate=48000) -> Path:
    n = np.arange(int(seconds * rate), dtype=np.float64)
    x = 10 ** (level / 20) * np.sin(2 * np.pi * 997 * n / rate)
    path = folder / name
    sf.write(path, np.column_stack((x, x)), rate, subtype="PCM_24", format="WAV")
    return path


def halve(candidate: Path) -> None:
    """Rewrite a WAV candidate in place as a valid WAV with its first half only (P7 (e))."""
    data, rate = sf.read(candidate, dtype="int32", always_2d=True)
    sf.write(candidate.with_suffix(".half.wav"), data[: len(data) // 2], rate, subtype="PCM_24", format="WAV")
    half = candidate.with_suffix(".half.wav").read_bytes()
    candidate.with_suffix(".half.wav").unlink()
    with open(candidate, "r+b") as handle:  # same inode, as a misbehaving encoder would leave it
        handle.write(half)
        handle.truncate()


# --- F-2: progress text never reaches the parser -----------------------------------------------------


def test_every_meter_scan_passes_nostats(executable):
    path = Path("/unused/in.wav")
    for argv in (sample_count_argv(executable, path), measurement_argv(executable, path, dual_mono=False),
                 measurement_argv(executable, path, dual_mono=True, padded=True)):
        assert "-nostats" in argv and argv.index("-nostats") < argv.index("-i")
        assert argv.index("-xerror") < argv.index("-i")


def test_measure_runs_exactly_three_quiet_scans(tmp_path, executable, prober, monkeypatch):
    path = tone(tmp_path, "short.wav")
    seen = []
    real = ffmpeg.run_process

    def spy(argv, **kwargs):
        seen.append(tuple(argv))
        return real(argv, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_process", spy)
    result = measure(path, ffprobe.probe(prober, path), executable)
    assert result.decoded_frames == 4 * 48000 and len(seen) == 3
    assert all("-nostats" in argv for argv in seen)


def meter_lines(index: int, *, peak=0.25) -> list[str]:
    stamp = f"pts:{index * 4800} pts_time:{index / 10:g}"
    return [f"[Parsed_ametadata_1 @ 0x6000] frame:{index} {stamp}",
            "[Parsed_ametadata_1 @ 0x6000] lavfi.r128.M=-20.000",
            "[Parsed_ametadata_1 @ 0x6000] lavfi.r128.I=-20.000",
            "[Parsed_ametadata_1 @ 0x6000] lavfi.r128.LRA=0.000",
            f"[Parsed_ametadata_1 @ 0x6000] lavfi.r128.true_peaks_ch0={peak}"]


@pytest.mark.parametrize("where", [0, 1, 4])
def test_parser_keeps_the_line_after_a_progress_cr(where):
    """IC-2 raw capture: ``size=… elapsed=…    \\r[Parsed_ametadata_2 @ …] frame:…`` on one LF line."""
    progress = "size=N/A time=00:09:03.74 bitrate=N/A speed=1.07e+03x elapsed=0:00:00.51    \r"
    lines = meter_lines(0) + meter_lines(1, peak=0.5)
    lines[5 + where] = progress + lines[5 + where]
    lines[2] += "\r"  # a CRLF-terminated line keeps its content
    parser = MeterParser(1)
    for line in lines:
        parser.feed(line)
    parser.finish()
    assert parser.records == 2 and parser.peak == .5 and parser.integrated_lufs == -20.0


def test_parser_still_rejects_a_genuine_duplicate_after_cr():
    parser = MeterParser(1)
    for line in meter_lines(0) + ["junk\r[Parsed_ametadata_1 @ 0x6000] lavfi.r128.M=-21.000"]:
        try:
            parser.feed(line)
        except FileFailure as exc:
            assert "duplicate metadata field: lavfi.r128.M" in exc.detail
            return
    pytest.fail("a duplicate field after the CR must still fail")


def test_p4_long_file_measures(tmp_path, executable, prober):
    path = long_tone(tmp_path, executable)
    result = measure(path, ffprobe.probe(prober, path), executable)
    assert result.signal_state == "measurable" and result.decoded_frames == LONG_S * LONG_RATE
    assert result.decoded_duration_s == LONG_S and not result.dual_mono
    # Peak 0.5/√2 (−9.03 dBFS), RMS −12.04 dBFS per channel; two channels add 3.01 dB; K-weighting at 440 Hz.
    assert -10.5 < result.integrated_lufs < -9.0
    assert result.true_peak_dbtp == pytest.approx(-9.03, abs=.1)


def test_p4_long_file_analyse(tmp_path, executable, capsys):
    path = long_tone(tmp_path, executable)
    report = tmp_path / "analysis.json"
    assert cli.main(["analyse", str(path), "--report", str(report), "--config", "/dev/null"]) == 0
    capsys.readouterr()
    row = json.loads(report.read_text())["outcomes"][0]
    assert row["status"] != "failed" and row["reason"] != "measurement_error"
    assert row["in_signal_state"] == "measurable" and -10.5 < row["in_lufs"] < -9.0


# --- F-3: candidate length ---------------------------------------------------------------------------


def record(frames: int | None, rate: int) -> LoudnessMeasurement:
    return LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=-16.0, true_peak_dbtp=-6.0, true_peak_linear=.5,
        lra_lu=None, threshold_lufs=None, integrated_valid=True, peak_valid=True, lra_valid=False,
        threshold_valid=False, integrated_resolution_lu=.001, peak_resolution_linear=.001, dual_mono=False,
        meter="ebur128", value_source="metadata", ffmpeg_version="synthetic",
        decoded_duration_s=(frames or 0) / rate, decoded_frames=frames, sample_rate_hz=rate,
        decoded_nonzero=True, terminal_peak_covered=True,
    )


@pytest.mark.parametrize("source,source_rate,frames,rate,ok", [
    (960000, 48000, 960000, 48000, True),
    (960000, 48000, 959999, 48000, False),       # one frame short at the same rate
    (960000, 48000, 960001, 48000, False),       # one frame long at the same rate
    (960000, 48000, 480000, 48000, False),       # P7 (e): half the audio
    (0, 48000, 0, 48000, True),
    (960000, 96000, 441000, 44100, True),        # F-14: 10 s, 96 → 44.1 kHz
    (960000, 96000, 441010, 44100, True),        # +1 sample per second
    (960000, 96000, 440990, 44100, True),        # −1 sample per second
    (960000, 96000, 441011, 44100, False),
    (960000, 96000, 440989, 44100, False),
    (24000, 48000, 22051, 44100, True),          # 0.5 s: never tighter than one sample
    (24000, 48000, 22052, 44100, False),
    (960000, 96000, 220500, 44100, False),       # half the audio after resampling
])
def test_length_rule(source, source_rate, frames, rate, ok):
    if ok:
        check_candidate_length(source, source_rate, record(frames, rate))
        return
    with pytest.raises(FileFailure) as caught:
        check_candidate_length(source, source_rate, record(frames, rate))
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")
    assert str(frames) in caught.value.detail


def test_unknown_candidate_length_is_not_accepted():
    with pytest.raises(FileFailure) as caught:
        check_candidate_length(960000, 48000, replace(record(960000, 48000), decoded_frames=None))
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")
    with pytest.raises(ValueError):
        check_candidate_length(-1, 48000, record(0, 48000))


def linear_plan(gain_db: float) -> GainPlan:
    return GainPlan(action="linear", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0,
                    tolerance_lu=.5, max_limit_db=3.0, gain_db=gain_db, predicted_true_peak_dbtp=-2.0,
                    limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
                    expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok")


def wav_target(env, probe, folder: Path, *, rate: int | None = None):
    rate = rate or probe.sample_rate_hz
    record_ = ffmpeg.collect_lossless_evidence(env, "pcm_s24le", rate, probe.channels, probe.channel_layout, folder)
    target = TargetFormat("wav", cap.OUTPUT_EXTENSIONS["wav"], "pcm_s24le", 24, False, rate, probe.channels,
                          probe.channel_layout, "none", "none", None, None, None, "not_applicable",
                          record_.evidence_id, False, "as_requested")
    return target, record_


def rendered(tmp_path, env, prober, *, rate=None):
    source = tone(tmp_path, "source.wav")
    probe = ffprobe.probe(prober, source)
    measured = measure(source, probe, ffmpeg.locate_ffmpeg())
    target, evidence = wav_target(env, probe, tmp_path / "ev", rate=rate)
    candidate = tmp_path / "candidate.wav"
    candidate.write_bytes(b"")
    info = os.stat(source)
    result = render(source, (info.st_dev, info.st_ino), probe, linear_plan(-16.0 - measured.integrated_lufs),
                    target, candidate, env, (evidence,))
    return source, probe, measured, result, candidate


@pytest.mark.parametrize("rate", [None, 44100])
def test_whole_candidate_is_accepted(tmp_path, env, executable, prober, rate):
    source, probe, measured, result, candidate = rendered(tmp_path, env, prober, rate=rate)
    for frames in (None, measured.decoded_frames):  # recounted, or passed from pass 1
        verification = verify_candidate(result, source, probe, ffprobe.probe(prober, candidate), executable,
                                        source_frames=frames)
        assert verification.loudness_ok and verification.peak_ok
    expected = 4 * (rate or 48000)
    assert abs(verification.measurement.decoded_frames - expected) <= 4


@pytest.mark.parametrize("rate", [None, 44100])
def test_p7e_short_candidate_fails_render_error(tmp_path, env, executable, prober, rate):
    source, probe, _, result, candidate = rendered(tmp_path, env, prober, rate=rate)
    halve(candidate)
    with pytest.raises(FileFailure) as caught:
        verify_candidate(result, source, probe, ffprobe.probe(prober, candidate), executable)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")
    assert f"candidate decodes to {2 * (rate or 48000)} frames" in caught.value.detail


def test_source_recount_failure_is_restated_at_verify(tmp_path, env, executable, prober, monkeypatch):
    source, probe, _, result, candidate = rendered(tmp_path, env, prober)

    def broken(*args, **kwargs):
        raise FileFailure("measurement_error", "measure", "no decoded sample records")

    monkeypatch.setattr(meter, "count_decoded_frames", broken)
    with pytest.raises(FileFailure) as caught:
        verify_candidate(result, source, probe, ffprobe.probe(prober, candidate), executable)
    assert (caught.value.reason, caught.value.stage) == ("decode_error", "verify")


def test_p7e_normalise_publishes_nothing(tmp_path, monkeypatch, capsys, path_based_evidence_cleanup):
    """Appendix A P7 (e) through the real CLI: a render that drops half the audio and exits 0."""
    (tmp_path / "src").mkdir()
    source = tone(tmp_path / "src", "a.wav", seconds=20)
    original = source.read_bytes()
    real_render = pipeline.render

    def short_render(*args, **kwargs):
        result = real_render(*args, **kwargs)
        halve(result.candidate_path)
        return result

    monkeypatch.setattr(pipeline, "render", short_render)
    out, report = tmp_path / "out", tmp_path / "r.json"
    code = cli.main(["normalise", str(source), "-o", str(out), "--report", str(report), "--config", "/dev/null"])
    capsys.readouterr()
    row = json.loads(report.read_text())["outcomes"][0]
    assert code == 1
    assert (row["status"], row["reason"]) == ("failed", "render_error")
    assert row["published_paths"] == [] and not (out / "a.wav").exists()
    assert not out.exists() or not any(p.name.startswith(".norma-") for p in out.rglob("*"))
    assert source.read_bytes() == original
    assert math.isfinite(row["in_lufs"])
