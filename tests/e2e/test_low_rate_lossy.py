"""Low-rate lossy regressions (TK-0712, CR-066; IC-3 findings F-1 and F-2).

Only the real child-process CLI drives Norma; no product module is imported or mocked. The
inputs are generated seeded noise (a trim or priming error shows as a length or lag change,
never as a periodic-tone ambiguity). FFmpeg decodes the outputs and ``soundfile`` reads the
untouched sources, so the judge is independent of Norma's own verifier.

F-1: MP3 V0 at the MPEG-2 rates (22.05, 24 and 32 kHz) with frame residues 0, 1, 137 and 511
must decode to exactly the source length at integer lag zero. F-2: automatic AAC from an
8 kHz stereo WAV must settle on 48 kbit/s with ``aac_bitrate_adjusted``, keep 8 kHz and decode
within the AAC length rule (0–1023 extra frames, never short).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SECONDS = 2  # short on purpose: the residue, not the duration, is what is being varied
SEARCH = 600  # frames either side of zero; the F-1 priming defect was +529
SPAN = slice(1000, 9000)


def noise_wav(path: Path, rate: int, frames: int, *, seed: int, sine: bool = False) -> Path:
    """Untagged stereo PCM24 Gaussian noise (σ 0.03), optionally with a 997 Hz sine."""
    samples = 0.03 * np.random.default_rng(seed).normal(size=(frames, 2))
    if sine:
        samples += 0.05 * np.sin(2 * np.pi * 997 * np.arange(frames) / rate)[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, subtype="PCM_24")
    return path


def decoded(path: Path, tools, fixture_module) -> tuple[np.ndarray, int]:
    """External FFmpeg decode of the whole stream: no resampling, no padding subtraction."""
    facts = tools.facts(path)
    rate, channels = int(facts["sample_rate"]), int(facts["channels"])
    return fixture_module.decode(path).reshape(-1, channels), rate


def best_lag(reference: np.ndarray, output: np.ndarray) -> int:
    """Integer lag (output relative to source) with the highest correlation over ``SPAN``."""
    scores = [float(np.dot(reference[SPAN], output[SPAN.start + k:SPAN.stop + k]))
              for k in range(-SEARCH, SEARCH + 1)]
    return int(np.argmax(scores)) - SEARCH


def run_file(norma, tools, source: Path, tmp_path: Path, *options):
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "--report", report, *options)
    assert run.code == 0, run.stdout + run.stderr
    (row,), (record,) = run.files, tools.report(report)
    assert record["execution"] == "committed", record
    output = Path(row[9])
    assert output.is_file() and output == Path(record["out_path"])
    return run, row, record, output


@pytest.mark.parametrize("residue", [0, 1, 137, 511])
@pytest.mark.parametrize("rate", [22050, 24000, 32000])
def test_f1_mp3_v0_at_mpeg2_rates_keeps_the_exact_length_and_zero_lag(
        norma, tools, fixture_module, tmp_path, rate, residue):
    frames = SECONDS * rate + residue
    source = noise_wav(tmp_path / "in" / "input.wav", rate, frames, seed=1020 + residue)
    original, _ = sf.read(source, dtype="float64", always_2d=True)
    before = tools.snapshot(source.parent)
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-f", "mp3")
    assert row[0] in {"OK", "WARN"} and record["out_format"] == "mp3"
    assert record["bitrate_mode"] == "vbr" and record["vbr_quality"] == "0"
    after, output_rate = decoded(output, tools, fixture_module)
    assert output_rate == rate and after.shape == original.shape == (frames, 2), (
        after.shape, original.shape)
    assert best_lag(original[:, 0], after[:, 0]) == 0
    assert tools.snapshot(source.parent) == before


def test_f2_automatic_aac_from_8_khz_stereo_settles_on_48_kbit_s_without_resampling(
        norma, tools, fixture_module, tmp_path):
    frames = 5 * 8000 + 137
    source = noise_wav(tmp_path / "in" / "input.wav", 8000, frames, seed=1020, sine=True)
    original, _ = sf.read(source, dtype="float64", always_2d=True)
    before = tools.snapshot(source.parent)
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-f", "aac")
    assert row[0] == "WARN" and record["reason"] == "aac_bitrate_adjusted"
    assert "aac_bitrate_adjusted" in row[10]
    assert record["out_format"] == "aac" and record["encoder"] == "aac_at"
    assert record["bitrate_mode"] == "cbr" and record["effective_bitrate_bps"] == "48000"
    assert int(record["requested_bitrate_bps"]) > 48000
    facts = tools.facts(output)
    assert facts["codec_name"] == "aac" and facts["sample_rate"] == "8000"
    after, rate = decoded(output, tools, fixture_module)
    assert rate == 8000 and after.shape[1] == 2
    extra = len(after) - len(original)
    assert 0 <= extra <= 1023, extra
    assert best_lag(original[:, 0], after[:, 0]) == 0
    assert tools.snapshot(source.parent) == before
