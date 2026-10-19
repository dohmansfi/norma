"""S2 acceptance, audio: loudness, linear null and lossless chain (TST-A-04, A-10, F-02).

Every test drives ``python -m norma normalise`` on an untagged generated input and judges the
file it wrote with meters outside Norma: ``pyloudnorm`` and a 4x polyphase true-peak estimate, and
FFmpeg's ``ebur128`` (loudness test plan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

KINDS = ["wav24", "float", "wav16", "flac24", "aiff24", "alac24"]
CEILING = -1.0


def normalise(norma, demos, untagged, kind, *options):
    """Copy the untagged input into a fresh folder, normalise it, return (run, output, report)."""
    source = untagged(kind)
    folder = demos({source.name: source}, folder=f"in-{kind}")
    report = folder.parent / f"report-{kind}.csv"
    run = norma("normalise", folder, "-o", folder.parent / f"out-{kind}", "--report", report, *options)
    return run, folder, report


@pytest.mark.parametrize("target", [-16, -14])
@pytest.mark.parametrize("kind", KINDS)
def test_a04_untagged_inputs_reach_the_target_on_independent_meters(norma, tools, demos, untagged, kind, target):
    """TST-A-04: linear to −16 and −14; both integrated meters within ±0.5 LU, both peaks ≤ −0.9 dBTP."""
    run, _, report = normalise(norma, demos, untagged, kind, "-t", target)
    assert run.code == 0, run.stderr
    assert run.summary["ok"] == "1" and run.summary["failed"] == "0"
    (record,) = tools.report(report)
    assert (record["status"], record["mode"]) == ("ok", "linear")
    assert float(record["target_lufs"]) == target and record["limited_db"] in {"", "0.0", "0"}
    output = next(iter(run.files))[9]
    samples, rate = tools.decode(Path(output))
    py_i, py_tp = tools.pyloudnorm(samples, rate)
    ff_i, ff_tp = tools.ebur128(Path(output))
    assert abs(py_i - target) <= 0.5 and abs(ff_i - target) <= 0.5, (py_i, ff_i)
    assert abs(py_i - ff_i) <= 0.2
    assert py_tp <= CEILING + 0.1 and ff_tp <= CEILING + 0.1, (py_tp, ff_tp)


def test_a04_default_target_is_exactly_minus_sixteen(norma, tools, demos, untagged):
    run, _, report = normalise(norma, demos, untagged, "wav24")
    assert run.code == 0, run.stderr
    (record,) = tools.report(report)
    assert float(record["target_lufs"]) == -16.0 and record["target_origin"] == "default"
    assert float(record["gain_db"]) == pytest.approx(-16.0 - float(record["in_lufs"]), abs=1e-6)


def test_a04_a_selected_tolerance_is_honoured(norma, tools, demos, untagged):
    """A tight --tolerance 0.1 is met by the written file, and the report names its origin."""
    run, _, report = normalise(norma, demos, untagged, "float", "-t", "-14", "--tolerance", "0.1", "--skip-within", "0.05")
    assert run.code == 0, run.stderr
    (record,) = tools.report(report)
    assert float(record["tolerance_lu"]) == 0.1 and record["tolerance_origin"] == "cli"
    samples, rate = tools.decode(Path(run.files[0][9]))
    assert abs(tools.pyloudnorm(samples, rate)[0] + 14) <= 0.1
    assert abs(tools.ebur128(Path(run.files[0][9]))[0] + 14) <= 0.1


@pytest.mark.parametrize(("kind", "target"), [("wav24", -14), ("float", -16), ("wav16", -14), ("flac24", -14)])
def test_a10_output_is_the_source_times_the_reported_gain(norma, tools, demos, untagged, kind, target):
    """TST-A-10: no dynamics processing — the decoded output nulls against source × gain.

    24-bit and float: peak residual below −90 dBFS. 16-bit: RMS at or below −84 dBFS (dither level).
    The frame count is unchanged for a same-rate chain.
    """
    run, folder, report = normalise(norma, demos, untagged, kind, "-t", target)
    assert run.code == 0, run.stderr
    (record,) = tools.report(report)
    assert record["mode"] == "linear" and record["limited_db"] in {"", "0.0", "0"}
    source, source_rate = tools.decode(next(folder.iterdir()))
    output, output_rate = tools.decode(Path(run.files[0][9]))
    assert (output.shape, output_rate) == (source.shape, source_rate)
    residual = output - source * 10 ** (float(record["gain_db"]) / 20)
    if record["out_bit_depth"] == "16":
        rms = float(np.sqrt(np.mean(residual ** 2)))
        assert 20 * np.log10(rms) <= -84
    else:
        assert 20 * np.log10(max(float(np.max(np.abs(residual))), 1e-30)) < -90


def test_a10_endpoint_impulses_keep_their_positions(norma, tools, tmp_path):
    """TST-A-10: impulses on the first and last frame stay on those frames; the length is unchanged.

    A 6 s tone at −10 dBFS carries ±0.5 impulses at both ends, so −16 LUFS is a pure attenuation
    (the plan stays linear, no limiter or reduce path) and the impulses remain the largest samples.
    """
    source = tools.tone(tmp_path / "in" / "impulse.wav", level=-10, seconds=6)
    samples, rate = tools.decode(source)
    samples[0, :], samples[-1, :] = 0.5, -0.5
    sf.write(source, samples, rate, subtype="PCM_24")
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "-t", "-16", "--report", report)
    assert run.code == 0, run.stdout + run.stderr
    (record,) = tools.report(report)
    assert record["mode"] == "linear"
    before, _ = tools.decode(source)
    after, _ = tools.decode(Path(run.files[0][9]))
    assert after.shape == before.shape
    for channel in range(before.shape[1]):
        assert np.flatnonzero(np.abs(before[:, channel]) > 0.4).tolist() == [0, before.shape[0] - 1]
        assert np.flatnonzero(np.abs(after[:, channel]) > 0.2).tolist() == [0, before.shape[0] - 1]
        assert np.sign(after[0, channel]) == 1 and np.sign(after[-1, channel]) == -1
    gain = 10 ** (float(record["gain_db"]) / 20)
    assert after[0, 0] == pytest.approx(0.5 * gain, abs=1e-6) and after[-1, 0] == pytest.approx(-0.5 * gain, abs=1e-6)


@pytest.mark.parametrize(("rate", "subtype", "depth"), [(44100, "PCM_16", "16"), (96000, "PCM_24", "24")])
def test_f02_zero_gain_lossless_chain_is_bit_identical(norma, tools, tmp_path, rate, subtype, depth):
    """TST-F-02: WAV → FLAC → native ALAC → AIFF at zero gain keeps every decoded sample.

    The tone sits at −20 LUFS and the target is −20, so each step is a within-tolerance copy
    (mode ``copy``, gain 0). ALAC must be the native encoder at the true depth, never ``alac_at``.
    Each step feeds the previous output back in. Outputs now carry the processing mark (B-002,
    metadata spec 10), so every step passes ``--force`` (``--no-mark-processed`` is specified but
    not yet accepted by ``normalise``: "not used by normalise", → CR in the TK-0714 report). The mark
    is metadata and never touches a decoded sample.
    """
    current = tools.tone(tmp_path / "chain" / f"step0-{rate}.wav", level=-20, seconds=6, rate=rate, subtype=subtype)
    original, _ = tools.decode(current)
    for index, target_format in enumerate(["flac", "alac", "aiff"], start=1):
        folder = tmp_path / f"stage-{rate}-{index}"
        folder.mkdir()
        shutil_copy = folder / current.name
        shutil_copy.write_bytes(current.read_bytes())
        report = tmp_path / f"report-{rate}-{index}.csv"
        run = norma("normalise", shutil_copy, "-t", "-20", "-f", target_format, "-o", folder / "out",
                    "--report", report, "--force")
        assert run.code == 0, run.stdout + run.stderr
        (record,) = tools.report(report)
        assert (record["status"], record["mode"], float(record["gain_db"])) == ("ok", "copy", 0.0)
        assert record["out_format"] == target_format and record["out_bit_depth"] == depth
        assert "alac_at" not in record["encoder"]
        current = Path(run.files[0][9])
        decoded, _ = tools.decode(current)
        assert decoded.shape == original.shape and np.array_equal(decoded, original)
        if target_format == "alac":
            facts = tools.facts(current)
            assert facts["codec_name"] == "alac" and facts["bits_per_raw_sample"] == depth


def test_f02_a_marked_output_fed_back_without_force_is_already_processed(norma, tools, tmp_path):
    """B-002 / metadata spec 10 / formats F17: the mark is detected and the file is skipped.

    A first run writes a marked FLAC. Feeding it straight back is ``SKIP already_processed`` with
    exit 0 (a skip-only run is safe to repeat, cli spec 7) and no second output. ``--force`` then
    bypasses the guard: converting the marked FLAC to ALAC (a same-format copy would need no output)
    writes an output whose decoded samples still equal the original.
    """
    source = tools.tone(tmp_path / "in" / "step0.wav", level=-20, seconds=6)
    original, _ = tools.decode(source)
    first = norma("normalise", source, "-t", "-20", "-f", "flac", "-o", tmp_path / "first")
    assert first.code == 0, first.stdout + first.stderr
    marked = Path(first.files[0][9])
    assert marked.suffix == ".flac"

    again = norma("normalise", marked, "-t", "-20", "-f", "flac", "-o", tmp_path / "again")
    assert again.code == 0, again.stdout + again.stderr
    (row,) = again.files
    assert row[0] == "SKIP" and row[10].startswith("already_processed: ")
    assert (again.summary["ok"], again.summary["skipped"], again.summary["failed"]) == ("0", "1", "0")
    assert tools.audio_files(tmp_path / "again") == []

    forced = norma("normalise", marked, "-t", "-20", "-f", "alac", "-o", tmp_path / "forced", "--force")
    assert forced.code == 0, forced.stdout + forced.stderr
    decoded, _ = tools.decode(Path(forced.files[0][9]))
    assert forced.files[0][0] == "OK" and np.array_equal(decoded, original)
