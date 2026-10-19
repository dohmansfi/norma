"""Long dynamic-mode regressions (TK-0713, CR-076; IC-3 recheck 3 F-5 and CR-077).

Only the real child-process CLI drives Norma; no product module is imported or mocked. The
sources are generated seeded Gaussian noise (σ 0.03, PCM24), 30 seconds or longer, because the
loudnorm lookahead flush that broke the MP3 encoder account (F-5) and the same-rate extra frame
(CR-077) only appear on programmes of that size; the 2 s cases of ``test_low_rate_lossy.py``
could not see either. FFmpeg decodes the outputs and ``soundfile`` reads the untouched sources,
so the judge is independent of Norma's own verifier and encoder account.

What each test asserts about a published ``--peak-mode dynamic`` render:

- exit 0, one committed row, mode ``dynamic``, the requested container and rate;
- decoded length: exactly the source frames at the same rate (MP3, WAV); within one sample of
  the rational rate-scaled count when resampled (D-086); 0–1023 extra frames, never short, for
  AAC (the E3 trailer rule);
- integer cross-correlation lag zero both at the start and near the end of the programme, so a
  tail timestamp jump or a dropped or added block shows as a shifted end;
- integrated loudness within ±0.5 LU of the −16 LUFS default on FFmpeg ``ebur128`` (mono
  outputs after the 3.01 LU dual-mono offset);
- the source tree is unchanged by SHA-256, size and mtime.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SEARCH = 600  # frames either side of zero (the F-1 priming defect was +529; F-5's jumps ~4,000)
WINDOW = 8000
HEAD = 1000  # edge frames excluded from each correlation window (codec and filter edges)
TARGET = -16.0
DUAL_MONO_OFFSET_LU = 10 * np.log10(2)  # 3.01 LU
AAC_TRAILER = 1023  # one AAC frame less one sample (rule E3)

# (rate, frames, channels) — the lengths are chosen, not round: each has a non-zero residue.
MP3_CASES = [
    pytest.param(32000, 960137, 1, "44100", False, id="mp3-32000-mono-to-44100-review-repro"),
    pytest.param(192000, 5760137, 1, None, True, id="mp3-192000-mono-auto-48000-bandlimited"),
    pytest.param(44100, 1323137, 2, None, False, id="mp3-44100-stereo-same-rate"),
    pytest.param(44100, 1323138, 2, None, False, id="mp3-44100-stereo-same-rate-cr077-length"),
    pytest.param(22050, 661637, 1, None, False, id="mp3-22050-mono-same-rate-cr077-length"),
    pytest.param(48000, 1440137, 2, None, False, id="mp3-48000-stereo-same-rate-30s"),
]
SAME_RATE_CASES = [
    pytest.param(22050, 661637, 1, id="22050-mono-661637"),
    pytest.param(44100, 1323138, 2, id="44100-stereo-1323138"),
    pytest.param(48000, 1440137, 2, id="48000-stereo-1440137"),
]


def noise_wav(path: Path, rate: int, frames: int, channels: int, *, seed: int = 1023,
              bandlimit_hz: float | None = None) -> Path:
    """Untagged PCM24 Gaussian noise (σ 0.03); optionally low-passed below ``bandlimit_hz``."""
    samples = 0.03 * np.random.default_rng(seed).normal(size=(frames, channels))
    if bandlimit_hz is not None:
        samples = sosfiltfilt(butter(8, bandlimit_hz, fs=rate, output="sos"), samples, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, subtype="PCM_24")
    return path


def decoded(path: Path, tools, fixture_module) -> tuple[np.ndarray, int]:
    """External FFmpeg decode of the whole stream: no resampling, no padding subtraction."""
    facts = tools.facts(path)
    rate, channels = int(facts["sample_rate"]), int(facts["channels"])
    return fixture_module.decode(path).reshape(-1, channels), rate


def reference_at(original: np.ndarray, rate: int, output_rate: int) -> np.ndarray:
    """The source's first channel at the output rate (identity when the rates match)."""
    first = original[:, 0]
    if rate == output_rate:
        return first
    divisor = gcd(rate, output_rate)
    return resample_poly(first, output_rate // divisor, rate // divisor)


def best_lag(reference: np.ndarray, output: np.ndarray, start: int) -> int:
    """Integer lag (output relative to reference) maximising the correlation of one window."""
    stop = start + WINDOW
    assert start - SEARCH >= 0 and stop + SEARCH <= len(output), (start, stop, len(output))
    window = reference[start:stop]
    scores = [float(np.dot(window, output[start + k:stop + k])) for k in range(-SEARCH, SEARCH + 1)]
    return int(np.argmax(scores)) - SEARCH


def assert_zero_lag_at_both_ends(reference: np.ndarray, output: np.ndarray) -> None:
    start = HEAD
    end = len(reference) - HEAD - WINDOW
    assert best_lag(reference, output, start) == 0, "start lag"
    assert best_lag(reference, output, end) == 0, "end lag"


def run_dynamic(norma, tools, source: Path, tmp_path: Path, *options):
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "--report", report,
                "--peak-mode", "dynamic", *options)
    assert run.code == 0, run.stdout + run.stderr
    (row,), (record,) = run.files, tools.report(report)
    assert record["execution"] == "committed", record
    output = Path(row[9])
    assert output.is_file() and output == Path(record["out_path"])
    assert row[6] == record["mode"] == "dynamic", row
    return run, row, record, output


def check_loudness(tools, output: Path, channels: int) -> None:
    """``ebur128`` reads a mono file as one front channel, 3.01 LU below the dual-mono
    convention that Norma's target uses (a stereo file with equal channels), so a mono output
    is compared after adding that known offset."""
    integrated, _ = tools.ebur128(output)
    if channels == 1:
        integrated += DUAL_MONO_OFFSET_LU
    assert abs(integrated - TARGET) <= 0.5, integrated


def make_source(tmp_path: Path, rate: int, frames: int, channels: int, **kwargs):
    source = noise_wav(tmp_path / "in" / "input.wav", rate, frames, channels, **kwargs)
    original, _ = sf.read(source, dtype="float64", always_2d=True)
    assert original.shape == (frames, channels)
    return source, original


# --- MP3: the F-5 regression --------------------------------------------------------------


@pytest.mark.parametrize(("rate", "frames", "channels", "requested", "bandlimited"), MP3_CASES)
def test_f5_dynamic_mp3_publishes_with_the_encoder_length_and_zero_lag(
        norma, tools, fixture_module, tmp_path, rate, frames, channels, requested, bandlimited):
    # The 192 kHz source is low-passed at 20 kHz: see the white-noise strict xfail below.
    source, original = make_source(tmp_path, rate, frames, channels,
                                   bandlimit_hz=20000 if bandlimited else None)
    before = tools.snapshot(source.parent)
    options = ["-f", "mp3"] + (["--sample-rate", requested] if requested else [])
    _, row, record, output = run_dynamic(norma, tools, source, tmp_path, *options)
    assert row[0] in {"OK", "WARN"} and record["out_format"] == "mp3"
    assert record["bitrate_mode"] == "vbr" and record["vbr_quality"] == "0"
    after, output_rate = decoded(output, tools, fixture_module)
    expected_rate = int(requested) if requested else (48000 if rate > 96000 else rate)
    assert output_rate == expected_rate and after.shape[1] == channels
    if output_rate == rate:
        # Same rate: the decoded length is the source length, exactly (CR-077 lengths included).
        assert len(after) == frames, (len(after), frames)
    else:
        scaled = frames * output_rate / rate  # rational count; D-086 allows ±1 sample
        assert abs(len(after) - scaled) <= 1, (len(after), scaled)
    assert_zero_lag_at_both_ends(reference_at(original, rate, output_rate), after[:, 0])
    check_loudness(tools, output, channels)
    assert tools.snapshot(source.parent) == before


@pytest.mark.xfail(strict=True, reason=(
    "B-009 (IC-3 recheck 3 reproduction; TK-0117 observation): dynamic mode on white noise at "
    "192 kHz is refused verification_loudness (about −22.1 LUFS): ultrasonic energy is metered "
    "at the source rate and loudnorm's internal path gets no correction. Safe refusal, not F-5."))
def test_cr_review_white_noise_192_khz_mono_dynamic_mp3_publishes(
        norma, tools, fixture_module, tmp_path):
    source, _ = make_source(tmp_path, 192000, 5760137, 1)
    _, row, record, output = run_dynamic(norma, tools, source, tmp_path, "-f", "mp3")
    assert row[0] in {"OK", "WARN"} and Path(output).is_file()


# --- WAV and AAC at the same rate: the CR-077 regression -----------------------------------


@pytest.mark.parametrize(("rate", "frames", "channels"), SAME_RATE_CASES)
def test_cr077_dynamic_wav_at_the_same_rate_keeps_the_exact_frame_count(
        norma, tools, fixture_module, tmp_path, rate, frames, channels):
    source, original = make_source(tmp_path, rate, frames, channels)
    before = tools.snapshot(source.parent)
    _, row, record, output = run_dynamic(norma, tools, source, tmp_path, "-f", "wav")
    assert row[0] in {"OK", "WARN"} and record["out_format"] == "wav"
    after, output_rate = decoded(output, tools, fixture_module)
    assert output_rate == rate and after.shape == (frames, channels), (after.shape, frames)
    assert_zero_lag_at_both_ends(original[:, 0], after[:, 0])
    check_loudness(tools, output, channels)
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize(("rate", "frames", "channels"), SAME_RATE_CASES)
def test_dynamic_aac_at_the_same_rate_follows_the_trailer_rule_with_zero_lag(
        norma, tools, fixture_module, tmp_path, rate, frames, channels):
    source, original = make_source(tmp_path, rate, frames, channels)
    before = tools.snapshot(source.parent)
    _, row, record, output = run_dynamic(norma, tools, source, tmp_path, "-f", "aac")
    assert row[0] in {"OK", "WARN"} and record["out_format"] == "aac"
    facts = tools.facts(output)
    assert facts["codec_name"] == "aac" and int(facts["sample_rate"]) == rate
    after, output_rate = decoded(output, tools, fixture_module)
    assert output_rate == rate and after.shape[1] == channels
    extra = len(after) - frames
    assert 0 <= extra <= AAC_TRAILER, extra  # trailing padding only, never short
    assert_zero_lag_at_both_ends(original[:, 0], after[:, 0])
    check_loudness(tools, output, channels)
    assert tools.snapshot(source.parent) == before
