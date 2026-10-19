"""Catalogue-backed meter acceptance with independent pyloudnorm/SciPy oracles."""

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from norma.adapters import ffmpeg, ffprobe
from norma.audio.measure import conservative_peak_dbtp, measure
from norma.errors import FileFailure, SkipFile, ToolFailed, ToolMissing, ToolTimeout, exit_contribution
from norma.models import ProbeResult


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


def measured(path, executable, prober, **kwargs):
    return measure(path, ffprobe.probe(prober, path), executable, **kwargs)


def independent(path, dual_mono=False):
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if dual_mono and samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.), padtype="constant")))
    return integrated, 20 * np.log10(peak)


@pytest.mark.parametrize("id,expected", [("F-01", -23), ("F-02", -33)])
def test_a01_catalogue_tone_precision(fixture_path, executable, prober, id, expected):
    path = fixture_path(id)
    digest = hashlib.sha256(path.read_bytes()).digest()
    result = measured(path, executable, prober)
    oracle_i, _ = independent(path)
    assert abs(result.integrated_lufs - expected) <= .1
    assert abs(result.integrated_lufs - oracle_i) <= .05
    assert result.decoded_frames == 960000
    assert result.threshold_lufs is None and not result.threshold_valid
    assert result.integrated_resolution_lu == .001 and result.peak_resolution_linear == .001
    assert result.terminal_peak_covered and result.ffmpeg_version == executable.version
    assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize("variant,expected", [("lra1", 10), ("lra2", 5), ("lra3", 20), ("lra4", 15)])
def test_a01_lra_unpadded(fixture_path, executable, prober, variant, expected):
    result = measured(fixture_path("F-02", variant), executable, prober)
    assert abs(result.lra_lu - expected) <= 1


def test_a02_intersample_peak(fixture_path, executable, prober):
    path = fixture_path("F-03")
    result = measured(path, executable, prober)
    _, oracle = independent(path)
    assert abs(result.true_peak_dbtp - (-2.989700043)) <= .3
    uncertainty = 20 * math.log10((result.true_peak_linear + .0005) / result.true_peak_linear)
    assert abs(result.true_peak_dbtp - oracle) <= .05 + uncertainty


@pytest.mark.parametrize("id,variant", [("F-04", None), ("F-05", None), ("F-05", "le"),
                                       ("F-05", "c6"), ("F-06", None), ("F-07", None)])
def test_a03_independent_loudness(fixture_path, executable, prober, id, variant):
    path = fixture_path(id, variant)
    result = measured(path, executable, prober)
    oracle, _ = independent(path, result.dual_mono)
    assert abs(result.integrated_lufs - oracle) <= .2


def test_a03_mono_policy(fixture_path, executable, prober):
    path = fixture_path("F-07")
    dual = measured(path, executable, prober)
    single = measured(path, executable, prober, no_dual_mono=True)
    oracle, _ = independent(path)
    assert abs(single.integrated_lufs - oracle) <= .2
    assert abs(dual.integrated_lufs - single.integrated_lufs - 3.010299957) <= .1
    assert dual.true_peak_linear == single.true_peak_linear


@pytest.mark.parametrize("id,variant,state,reason", [
    ("F-08", None, "silent", "silent"),
    ("F-09", None, "measurable", "too_short"),
    ("F-09", "100ms", "unmeasurable", "unmeasurable"),
    ("F-09", "300ms", "unmeasurable", "unmeasurable"),
    ("F-09", "450ms", "measurable", "too_short"),
    ("F-09", "quiet", "unmeasurable", "unmeasurable"),
])
def test_a03_classification(fixture_path, executable, prober, id, variant, state, reason):
    path = fixture_path(id, variant)
    result = measured(path, executable, prober)
    assert result.signal_state == state
    assert [d.code for d in result.diagnostics] == [reason]
    assert result.decoded_duration_s == sf.info(path).duration
    if state == "silent":
        assert result.true_peak_dbtp == -math.inf and result.decoded_nonzero is False
    elif variant == "quiet":
        assert result.true_peak_linear == 0 and result.true_peak_dbtp is None
        assert result.decoded_nonzero and not result.peak_valid


@pytest.mark.parametrize("variant", ["endpoint", "tail4801", "tail5279"])
def test_a02_terminal_peaks(fixture_path, executable, prober, variant):
    path = fixture_path("F-03", variant)
    result = measured(path, executable, prober)
    samples, _ = sf.read(path, always_2d=True)
    assert result.terminal_peak_covered
    assert result.true_peak_linear + .0005 >= np.max(np.abs(samples))
    assert result.decoded_frames == len(samples)
    if variant.startswith("tail"):
        assert result.integrated_lufs is None and result.signal_state == "unmeasurable"


def test_a02_p1_real_quantiser_bin(tmp_path, executable, prober):
    n = np.arange(4 * 48000)
    samples = .71625 * np.sin(2 * np.pi * 1000 * n / 48000)
    ramp = .5 - .5 * np.cos(np.pi * np.arange(2400) / 2399)
    samples[:2400] *= ramp
    samples[-2400:] *= ramp[::-1]
    path = tmp_path / "p1.wav"
    sf.write(path, samples, 48000, subtype="DOUBLE")
    result = measured(path, executable, prober, no_dual_mono=True)
    assert result.true_peak_linear == .716
    upper = conservative_peak_dbtp(result.true_peak_linear, result.peak_resolution_linear)
    assert result.true_peak_dbtp < -2.9 < upper
    assert 20 * np.log10(np.max(np.abs(samples))) > -2.9


@pytest.mark.parametrize("channels,layout", [
    (3, "3.0"), (4, "quad"), (4, "4.0"), (5, "5.0"),
    (6, "5.1"), (8, "7.1"), (6, None),
])
@pytest.mark.parametrize("no_dual_mono", [False, True])
def test_a03_multichannel_refused_before_any_scan(tmp_path, monkeypatch, channels, layout, no_dual_mono):
    # ADR-0020: only the probed channel count decides; even an absent layout
    # and --no-dual-mono cannot enable multichannel measurement.
    probe = ProbeResult(
        codec="pcm_f32le", codec_class="lossless", container="wav",
        audio_stream_index=0, ignored_streams=0, sample_rate_hz=48000,
        bit_depth=None, sample_float=True, sample_format="flt",
        channels=channels, channel_layout=layout, duration_s=1,
        duration_basis="stream", bitrate_bps=None, has_artwork=False,
        chapters=0, ffprobe_tags={}, ffprobe_version="test",
    )
    executable = ffmpeg.Executable(tmp_path / "absent-ffmpeg", (1, 2), "test")

    def forbidden(*args, **kwargs):
        pytest.fail("multichannel refusal must precede argv construction and every scan")

    monkeypatch.setattr(ffmpeg, "ffmpeg_argv", forbidden)
    monkeypatch.setattr(ffmpeg, "run_process", forbidden)
    with pytest.raises(SkipFile) as caught:
        measure(tmp_path / "absent-surround.wav", probe, executable, no_dual_mono=no_dual_mono)
    assert caught.value.reason == "unsupported_channels"
    assert caught.value.stage == "measure" and caught.value.status == "skipped"
    assert exit_contribution(caught.value.status, caught.value.reason) == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_audio_fails(tmp_path, executable, prober, value):
    samples = np.zeros((48000, 2))
    samples[24000, 1] = value
    path = tmp_path / "invalid.wav"
    sf.write(path, samples, 48000, subtype="DOUBLE")
    with pytest.raises(FileFailure, match="measurement_error|decode_error"):
        measured(path, executable, prober)


def test_under_100ms_is_not_missing_metadata(tmp_path, executable, prober):
    path = tmp_path / "tiny.wav"
    sf.write(path, .1 * np.sin(np.arange(48)), 48000, subtype="DOUBLE")
    result = measured(path, executable, prober)
    assert result.signal_state == "unmeasurable" and result.decoded_frames == 48
    assert result.lra_lu is None and result.terminal_peak_covered


@pytest.mark.parametrize("kind,reason", [("timeout", "timeout"), ("missing", "decode_error"), ("failed", "decode_error")])
def test_tool_failures_map_to_c13(tmp_path, executable, prober, monkeypatch, kind, reason):
    path = tmp_path / "tone.wav"
    sf.write(path, np.sin(np.arange(48000)) * .1, 48000)
    probe = ffprobe.probe(prober, path)
    def fail(*args, **kwargs):
        if kind == "timeout":
            raise ToolTimeout("deadline", deadline_s=60)
        if kind == "missing":
            raise ToolMissing("tool vanished")
        raise ToolFailed("decoder failed", returncode=1)
    monkeypatch.setattr(ffmpeg, "run_process", fail)
    with pytest.raises(FileFailure) as caught:
        measure(path, probe, executable)
    assert caught.value.reason == reason


def test_duration_shortfall_precedes_skip(tmp_path, executable, prober):
    path = tmp_path / "zero.wav"
    sf.write(path, np.zeros(48000), 48000)
    probe = replace(ffprobe.probe(prober, path), duration_s=2, duration_basis="stream")
    with pytest.raises(FileFailure, match="decode_error"):
        measure(path, probe, executable)
