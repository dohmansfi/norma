"""TK-0118 (CR-077): a dynamic render at the source's own rate has exactly the source's frame count.

loudnorm works at 192 kHz internally; the round trip back to the source rate could leave one frame
more than the source (22050 Hz 661637 -> 661638, 44.1 kHz 1323138 -> 1323139), which verification
rightly refuses. The graph now ends in ``atrim=end_sample=<source frames>`` at the same rate only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from norma import cli
from norma.adapters import ffmpeg
from norma.audio.render import filter_chain, needs_source_frames, render_argv
from norma.models import GainPlan, ProbeResult, TargetFormat

PARAMS = {"input_i": -24.78, "input_tp": -16.41, "input_lra": 0.0, "input_thresh": -34.78, "target_offset": 0.0}


# --- Pure graph -----------------------------------------------------------------------------------


def dynamic_plan() -> GainPlan:
    return GainPlan(action="dynamic", requested_peak_mode="dynamic", target_lufs=-16.0, ceiling_dbtp=-1.0,
                    tolerance_lu=.5, max_limit_db=3.0, gain_db=0.0, predicted_true_peak_dbtp=-1.0,
                    limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=-1.0,
                    expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok", dynamic_parameters=PARAMS)


def probe_of(rate: int, channels: int = 1) -> ProbeResult:
    return ProbeResult("pcm_s24le", "lossless", "wav", 0, 0, rate, 24, False, "s32", channels,
                       "stereo" if channels == 2 else "mono", 30.0, "stream", None, False, 0, {}, "synthetic")


def wav_target(rate: int, channels: int = 1, *, dither="none") -> TargetFormat:
    return TargetFormat("wav", ".wav", "pcm_s24le", 24, False, rate, channels, "stereo" if channels == 2 else "mono",
                        dither, "none", None, None, None, "not_applicable", "ev:x", False, "as_requested")


def test_the_same_rate_dynamic_graph_ends_in_an_exact_trim_after_the_resampler():
    chain = filter_chain(dynamic_plan(), probe_of(22050), wav_target(22050), source_frames=661637)
    assert chain.endswith("cutoff=0.97,atrim=end_sample=661637")
    assert chain.index("loudnorm") < chain.index("aresample=22050") < chain.index("atrim")


def test_the_trim_comes_before_the_dither_so_the_quantiser_sees_the_final_length():
    target = TargetFormat("wav", ".wav", "pcm_s16le", 16, False, 22050, 1, "mono", "triangular", "none", None, None,
                          None, "not_applicable", "ev:x", False, "as_requested")
    chain = filter_chain(dynamic_plan(), probe_of(22050), target, source_frames=661637)
    assert chain.index("atrim=end_sample=661637") < chain.index("osf=s16")


def test_a_resampled_dynamic_graph_is_left_exactly_as_it_was():
    plain = filter_chain(dynamic_plan(), probe_of(32000), wav_target(44100))
    assert filter_chain(dynamic_plan(), probe_of(32000), wav_target(44100), source_frames=960137) == plain
    assert "atrim" not in plain and not needs_source_frames(dynamic_plan(), probe_of(32000), wav_target(44100))
    argv = render_argv(Path("/unused/ffmpeg"), Path("/unused/s.wav"), Path("/unused/c.wav"), dynamic_plan(),
                       probe_of(32000), wav_target(44100), source_frames=960137)
    assert "atrim" not in argv[argv.index("-af") + 1]


def test_only_a_dynamic_same_rate_render_is_trimmed():
    probe, target = probe_of(22050), wav_target(22050)
    assert needs_source_frames(dynamic_plan(), probe, target)
    from dataclasses import replace
    linear = replace(dynamic_plan(), action="linear", requested_peak_mode="limit", gain_db=-2.0,
                     limiter_threshold_dbtp=None, dynamic_parameters=None)
    chain = filter_chain(linear, probe, target, source_frames=661637)
    assert chain is not None and "atrim" not in chain and not needs_source_frames(linear, probe, target)


@pytest.mark.parametrize("bad", [0, -5, 1.5, "7"])
def test_a_trim_count_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="source_frames"):
        filter_chain(dynamic_plan(), probe_of(22050), wav_target(22050), source_frames=bad)


# --- Real FFmpeg, through the CLI -----------------------------------------------------------------


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def env(executable):
    return ffmpeg.probe_environment(executable)


@pytest.fixture
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory cleanup.
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def noise(folder: Path, rate: int, frames: int, channels: int = 1) -> Path:
    samples = .03 * np.random.default_rng(1118).normal(size=(frames, channels))
    path = folder / f"noise-{rate}-{frames}-{channels}.wav"
    sf.write(path, samples if channels > 1 else samples[:, 0], rate, subtype="PCM_24")
    return path


def normalise_dynamic(tmp_path: Path, source: Path, fmt: str, *extra: str):
    """One real ``normalise --peak-mode dynamic``; returns (exit code, report row, render argv)."""
    real_run = ffmpeg.run_process
    seen: dict = {}

    def run(argv, **kwargs):
        args = [str(arg) for arg in argv]
        if "-af" in args and "loudnorm" in args[args.index("-af") + 1] and "measured_I" in args[args.index("-af") + 1]:
            seen["argv"] = args
        return real_run(argv, **kwargs)

    out, report = tmp_path / "out", tmp_path / "report.json"
    argv = ["normalise", str(source), "-f", fmt, "--peak-mode", "dynamic", "--config", "/dev/null",
            "-o", str(out), "--report", str(report), *extra]
    with patch.object(ffmpeg, "run_process", run):
        code = cli.main(argv)
    return code, json.loads(report.read_text())["outcomes"][0], seen.get("argv")


def decode(path: Path, executable, channels: int) -> np.ndarray:
    raw = subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                          "-f", "f64le", "-c:a", "pcm_f64le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f8").reshape(-1, channels)


def lag(reference: np.ndarray, output: np.ndarray, start: int, length=8000, window=200) -> int:
    ref = reference[start:start + length]
    scores = {k: float(np.dot(ref, output[start + k:start + k + length])) for k in range(-window, window + 1)}
    return max(scores, key=scores.get)


def assert_exact_and_aligned(source: Path, published: Path, executable, frames: int, channels: int) -> None:
    reference = decode(source, executable, channels)[:, 0]
    result = decode(published, executable, channels)
    assert reference.shape[0] == frames and result.shape[0] == frames  # the frame count, not within a tolerance
    assert lag(reference, result[:, 0], 1000) == 0
    assert lag(reference, result[:, 0], frames - 9000) == 0


WAV_CASES = [
    (22050, 661637, 1),    # CR-077's first example (+1 before the repair)
    (44100, 1323138, 2),   # CR-077's second example
    (48000, 5 * 48000 + 137, 1),
    (96000, 3 * 96000 + 511, 1),
    (22050, 5 * 22050 + 1, 2),
    (44100, 4 * 44100 + 333, 1),
    (48000, 3 * 48000 + 1, 2),
]


@pytest.mark.parametrize("rate,frames,channels", WAV_CASES)
def test_a_same_rate_dynamic_wav_has_exactly_the_source_frames_with_zero_lag(tmp_path, executable, capsys, rate,
                                                                              frames, channels):
    (tmp_path / "src").mkdir()
    source = noise(tmp_path / "src", rate, frames, channels)
    code, row, argv = normalise_dynamic(tmp_path, source, "wav")
    capsys.readouterr()
    assert code == 0 and row["status"] in ("ok", "warn") and len(row["published_paths"]) == 1, row
    assert f"atrim=end_sample={frames}" in argv[argv.index("-af") + 1]
    assert_exact_and_aligned(source, Path(row["published_paths"][0]), executable, frames, channels)


@pytest.fixture
def v0_records(env, tmp_path):
    folder = tmp_path / "mp3-evidence"
    cache = {}

    def record(rate, channels):
        if (rate, channels) not in cache:
            if "libmp3lame" not in env.encoders:
                pytest.skip("build lacks libmp3lame")
            cache[rate, channels] = ffmpeg.collect_mp3_v0_evidence(
                env, rate, channels, "stereo" if channels == 2 else "mono", folder)
            if 0 not in cache[rate, channels].vbr_qualities:
                pytest.skip(f"MP3 V0 is not evidenced at {rate} Hz, {channels} ch on this build")
        return cache[rate, channels]
    return record


MP3_CASES = [
    (22050, 661637, 1),
    (44100, 1323138, 1),
    (44100, 2 * 44100 + 777, 2),
    (48000, 3 * 48000 + 137, 1),
]


@pytest.mark.parametrize("rate,frames,channels", MP3_CASES)
def test_a_same_rate_dynamic_mp3_has_exactly_the_source_frames_with_zero_lag(tmp_path, executable, env, v0_records,
                                                                              capsys, path_based_evidence_cleanup,
                                                                              rate, frames, channels):
    v0_records(rate, channels)
    (tmp_path / "src").mkdir()
    source = noise(tmp_path / "src", rate, frames, channels)
    code, row, argv = normalise_dynamic(tmp_path, source, "mp3", "--sample-rate", "same")
    capsys.readouterr()
    assert code == 0 and row["status"] in ("ok", "warn") and len(row["published_paths"]) == 1, row
    assert f"atrim=end_sample={frames}" in argv[argv.index("-af") + 1]
    assert_exact_and_aligned(source, Path(row["published_paths"][0]), executable, frames, channels)


def test_a_resampled_dynamic_wav_is_not_trimmed_and_still_publishes(tmp_path, executable, capsys):
    (tmp_path / "src").mkdir()
    source = noise(tmp_path / "src", 32000, 4 * 32000 + 137)
    code, row, argv = normalise_dynamic(tmp_path, source, "wav", "--sample-rate", "44100")
    capsys.readouterr()
    assert code == 0 and len(row["published_paths"]) == 1, row
    assert "atrim" not in argv[argv.index("-af") + 1]
    decoded = decode(Path(row["published_paths"][0]), executable, 1)
    assert abs(decoded.shape[0] * 32000 - (4 * 32000 + 137) * 44100) <= 32000  # the unchanged length rule
