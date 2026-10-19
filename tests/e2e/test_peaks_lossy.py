"""S3 command-line acceptance (TK-0709): peaks, decoded lossy audio and safeguards.

Only the real child-process CLI drives Norma. Catalogue signals, external FFmpeg
fixture encodes, pyloudnorm/SciPy and the sandbox ledger are independent judges;
no product module is imported here. Retry counters are report evidence, not a
claim to observe every internal transition (the unit ladder suite does that).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

TARGET = -14.0
CEILING = -1.0
ENCODERS = {
    "flac": (".flac", ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-q:a", "0", "-write_xing", "1"]),
    "aac": (".m4a", ["-c:a", "aac_at", "-b:a", "256k"]),
}


def encode(source, destination, kind, environment):
    """Generate an untagged input; keep MP3's length-restoring Xing/LAME frame."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [environment["ffmpeg_path"], "-nostdin", "-v", "error", "-y", "-i", str(source),
         "-map", "0:a:0", "-map_metadata", "-1", "-map_chapters", "-1",
         "-fflags", "+bitexact", "-flags:a", "+bitexact", *ENCODERS[kind][1], str(destination)],
        capture_output=True, text=True, timeout=240, check=False,
    )
    assert result.returncode == 0, result.stderr
    return destination


def decoded(path, tools, fixture_module):
    """Use the external decoder for both lossy containers; no resampling/trim."""
    facts = tools.facts(path)
    rate, channels = int(facts["sample_rate"]), int(facts["channels"])
    return fixture_module.decode(path).reshape(-1, channels), rate


def run_file(norma, tools, source, tmp_path, *options):
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "--report", report, *options)
    assert run.code == 0, run.stdout + run.stderr
    (row,), (record,) = run.files, tools.report(report)
    assert record["execution"] == "committed", record
    output = Path(row[9])
    assert output.is_file() and output == Path(record["out_path"])
    return run, row, record, output


def independent(path, tools, fixture_module, *, target: float | None = TARGET):
    samples, rate = decoded(path, tools, fixture_module)
    py_i, py_tp = tools.pyloudnorm(samples, rate)
    ff_i, ff_tp = tools.ebur128(path)
    if target is not None:
        assert abs(py_i - target) <= 0.5 and abs(ff_i - target) <= 0.5, (py_i, ff_i)
    assert abs(py_i - ff_i) <= 0.2, (py_i, ff_i)
    assert py_tp <= CEILING + 0.1 and ff_tp <= CEILING + 0.1, (py_tp, ff_tp)
    return samples, rate, py_i, ff_i


def lag(reference, output, window=40):
    """Integer lag around zero; crop equal edges so every score has equal support."""
    scores = [np.dot(reference[window:-window], output[window + k:len(output) - window + k])
              for k in range(-window, window + 1)]
    return int(np.argmax(scores)) - window


@pytest.mark.parametrize("target", [-14.0, -12.5])
def test_a05_limiter_reaches_target_and_keeps_burst_alignment(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, target):
    source = demos({"hot.wav": fixture_path("F-05", "le")}) / "hot.wav"
    before = tools.snapshot(source.parent)
    run, row, record, output = run_file(norma, tools, source, tmp_path, "-t", target)
    assert (row[0], row[6], record["mode"]) == ("OK", "limit", "limit")
    assert 0 < float(record["limited_db"]) <= 3.0
    assert float(record["effective_limiter_bound_db"]) == pytest.approx(
        float(record["limited_db"]) + 0.3, abs=1e-6)
    after, rate, _, _ = independent(output, tools, fixture_module, target=target)
    original, original_rate = tools.decode(source)
    assert (after.shape, rate) == (original.shape, original_rate)
    for seconds in (2, 7, 12, 17, 22, 27):
        span = slice(seconds * rate - 2000, seconds * rate + 2480)
        assert lag(original[span, 0], after[span, 0]) == 0
    assert tools.snapshot(source.parent) == before
    assert run.summary["ok"] == "1" and run.summary["failed"] == "0"


def test_a05_minus_sixteen_is_a_linear_control(norma, tools, fixture_path, demos, tmp_path):
    source = demos({"control.wav": fixture_path("F-05", "le")}) / "control.wav"
    _, row, record, _ = run_file(norma, tools, source, tmp_path, "-t", "-16")
    assert row[6] == record["mode"] == "linear" and float(record["limited_db"]) == 0


@pytest.mark.parametrize("target", [-14.0, -12.5])
@pytest.mark.parametrize("policy", ["reduce", "limit"])
def test_a09_reduce_is_plain_gain_and_reports_measured_shortfall(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, target, policy):
    source = demos({"crest.wav": fixture_path("F-05", "hc")}) / "crest.wav"
    before = tools.snapshot(source.parent)
    run, row, record, output = run_file(norma, tools, source, tmp_path, "-t", target, "--peak-mode", policy)
    assert (row[0], row[6], record["reason"]) == ("WARN", "reduce", "reduce_shortfall")
    assert "measures " in row[10] and "LU below" in row[10]
    assert float(record["limited_db"]) == float(record["effective_limiter_bound_db"]) == 0
    after, rate, py_i, ff_i = independent(output, tools, fixture_module, target=None)
    measured = float(record["measured_shortfall_lu"])
    assert measured > 0 and abs(measured - max(0, target - py_i)) <= 0.2
    assert abs(measured - max(0, target - ff_i)) <= 0.2
    original, original_rate = tools.decode(source)
    assert (after.shape, rate) == (original.shape, original_rate)
    # A dynamics processor would not null against one constant gain.
    residual = after - original * 10 ** (float(record["gain_db"]) / 20)
    assert 20 * np.log10(max(float(np.max(np.abs(residual))), 1e-30)) < -90
    assert tools.snapshot(source.parent) == before and run.summary["warn"] == "1"


def test_a09_zero_cap_selects_reduce(norma, tools, fixture_path, demos, tmp_path):
    source = demos({"hot.wav": fixture_path("F-05", "le")}) / "hot.wav"
    _, row, record, _ = run_file(norma, tools, source, tmp_path, "-t", TARGET, "--max-limit", "0")
    assert row[0] == "WARN" and row[6] == "reduce"
    assert float(record["limited_db"]) == 0 and float(record["measured_shortfall_lu"]) > 0


@pytest.mark.parametrize("variant", ["le", "hc"])
def test_a09_explicit_dynamic_reaches_target_and_ceiling(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, variant):
    source = demos({"dynamic.wav": fixture_path("F-05", variant)}) / "dynamic.wav"
    before = tools.snapshot(source.parent)
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", TARGET, "--peak-mode", "dynamic")
    assert row[6] == record["mode"] == "dynamic"
    assert float(record["limited_db"]) == float(record["effective_limiter_bound_db"]) == 0
    samples, rate, _, _ = independent(output, tools, fixture_module)
    original, original_rate = tools.decode(source)
    assert (samples.shape, rate) == (original.shape, original_rate)  # not loudnorm's internal 192 kHz
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize("command", [["analyse"], ["normalise", "--dry-run"]], ids=["analyse", "dry-run"])
@pytest.mark.parametrize("report_requested", [False, True], ids=["no-writes", "report-only"])
def test_c03_a09_dynamic_preview_is_uncalibrated_and_sources_are_unchanged(
        norma, tools, fixture_path, demos, tmp_path, ledger, command, report_requested):
    source = demos({"preview.wav": fixture_path("F-05", "hc")}) / "preview.wav"
    report = tmp_path / "preview.csv"
    before = ledger()
    options = ["--report", report] if report_requested else []
    run = norma(*command, source, "-t", TARGET, "--peak-mode", "dynamic", *options)
    assert run.code == 0, run.stdout + run.stderr
    (row,) = run.files
    assert row[6] == "dynamic (calibrated at render)" and row[7:10] == ["", "", ""]
    assert run.summary["report"] == (str(report) if report_requested else "")
    delta = tools.difference(before, ledger())
    assert delta == {"added": {"preview.csv"} if report_requested else set(),
                     "removed": set(), "changed": set()}
    assert not list((tmp_path / "home" / "tmp").iterdir())
    if report_requested:
        (record,) = tools.report(report)
        assert record["mode"] == "dynamic"
        assert record["out_lufs"] == record["out_dbtp"] == record["measured_shortfall_lu"] == ""
        # Predicted peak and calibration are not columns in the fixed C-09 report.
        assert float(record["predicted_shortfall_lu"]) == 0
        assert float(record["gain_db"]) == pytest.approx(TARGET - float(record["in_lufs"]))
        assert float(record["expected_lufs"]) == TARGET
        assert all(record[name] == "0" for name in ("loudness_corrections", "peak_corrections", "reduce_fallbacks"))
        assert record["execution"] == ("inspect" if command[0] == "analyse" else "plan")


@pytest.mark.parametrize("signal", ["pink", "hot"])
@pytest.mark.parametrize("source_kind", ["wav", "flac"])
@pytest.mark.parametrize("target_kind", ["mp3", "aac"])
def test_a06_lossless_to_lossy_is_measured_after_decode_with_exact_length_and_no_priming(
        norma, tools, fixture_path, fixture_module, demos, tmp_path, norma_environment,
        signal, source_kind, target_kind):
    cached = fixture_path("F-04") if signal == "pink" else fixture_path("F-05", "le")
    source = demos({"input.wav": cached}) / "input.wav"
    if source_kind == "flac":
        source = encode(source, tmp_path / "flac" / "input.flac", "flac", norma_environment)
    before = tools.snapshot(source.parent)
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", TARGET, "-f", target_kind)
    assert row[0] == "OK" and record["out_format"] == target_kind
    assert record["encoder"] == {"mp3": "libmp3lame", "aac": "aac_at"}[target_kind]
    if target_kind == "mp3":
        assert record["bitrate_mode"] == "vbr" and record["vbr_quality"] == "0"
        assert record["requested_bitrate_bps"] == record["effective_bitrate_bps"] == ""
    else:
        assert record["bitrate_mode"] == "cbr" and record["vbr_quality"] == ""
        assert record["requested_bitrate_bps"] == record["effective_bitrate_bps"] == "256000"
        assert "aac_bitrate_adjusted" not in row[10]  # automatic lossless request — B-006
    after, rate, _, _ = independent(output, tools, fixture_module)
    original, original_rate = decoded(source, tools, fixture_module)
    assert rate == original_rate == 48000 and after.shape[1] == original.shape[1] == 2
    extra = len(after) - len(original)
    assert extra == 0 if target_kind == "mp3" else 0 <= extra <= 1023
    # Seeded noise / burst transient makes this a trim check, not a periodic-tone ambiguity.
    start = 0 if signal == "pink" else 2 * rate - 2000
    assert lag(original[start:start + 8000, 0], after[start:start + 8000, 0]) == 0
    assert tools.snapshot(source.parent) == before


def test_d081_unmasked_stereo_wav_reaches_automatic_aac_with_the_note(
        norma, tools, tmp_path, norma_environment):
    source = tools.tone(tmp_path / "in" / "unmasked.wav", level=-20)
    probe = subprocess.run(
        [str(Path(norma_environment["ffmpeg_path"]).with_name("ffprobe")), "-v", "error",
         "-select_streams", "a:0", "-show_entries", "stream=channel_layout", "-of", "default=nw=1", str(source)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    assert probe.stdout.strip() in {"", "channel_layout=unknown"}
    run, row, record, output = run_file(norma, tools, source, tmp_path, "-f", "aac")
    assert row[0] == "OK" and "channel layout not stated; assumed stereo" in run.stderr
    assert record["out_format"] == "aac" and tools.facts(output)["codec_name"] == "aac"
    assert record["requested_bitrate_bps"] == record["effective_bitrate_bps"] == "256000"


@pytest.fixture
def lossy_source(tools, tmp_path, norma_environment):
    """Five-second tone; tag-free encoded source, safely below the ceiling."""
    def make(kind, *, level=-20):
        wav = tools.tone(tmp_path / "original" / "tone.wav", level=level)
        return encode(wav, tmp_path / "in" / ("bounce" + ENCODERS[kind][0]), kind, norma_environment)
    return make


@pytest.mark.parametrize("source_kind,target_kind", [("mp3", "aac"), ("aac", "mp3")])
@pytest.mark.parametrize("allowed", [False, True], ids=["guarded", "allowed"])
def test_f04_lossy_codec_switch_is_guarded_unless_explicitly_allowed(
        norma, tools, fixture_module, tmp_path, lossy_source, source_kind, target_kind, allowed):
    source = lossy_source(source_kind)
    before = tools.snapshot(source.parent)
    options = ["--allow-lossy-transcode"] if allowed else []
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", TARGET, "-f", target_kind, *options)
    effective = target_kind if allowed else source_kind
    assert record["out_format"] == effective and tools.facts(output)["codec_name"] == effective
    warnings = {item["code"] for item in json.loads(record["warnings"])}
    if allowed:
        assert row[0] == "OK" and warnings == set()  # F14 warns for same-codec generations only
    else:
        assert row[0] == "WARN" and "lossy_transcode_blocked" in warnings
        assert "lossy_reencoded" in warnings
        # An AAC source's average bitrate can request the next discrete nominal setting;
        # that warning has primary precedence without hiding the guard in the report.
        assert record["reason"] == ("aac_bitrate_adjusted" if "aac_bitrate_adjusted" in warnings
                                    else "lossy_transcode_blocked")
        assert record["reason"] in row[10]
    assert record["source_codec"] == source_kind
    independent(output, tools, fixture_module)
    assert tools.snapshot(source.parent) == before


def test_f04_retained_aac_rejects_v0_before_any_write(norma, lossy_source, ledger, tmp_path):
    source = lossy_source("aac")
    before = ledger()
    run = norma("normalise", source, "-f", "mp3", "--bitrate", "V0", "-o", tmp_path / "out")
    assert run.code == 2 and run.files == [] and "V0" in run.stderr
    assert ledger() == before


@pytest.mark.parametrize("source_kind", ["mp3", "aac"])
@pytest.mark.parametrize("preset", ["same", "apple"])
@pytest.mark.parametrize("explicit_policy", [False, True], ids=["default", "reencode"])
def test_f04_same_codec_changes_reencode_with_a_warning(
        norma, tools, tmp_path, lossy_source, source_kind, preset, explicit_policy):
    source = lossy_source(source_kind)
    before = tools.snapshot(source.parent)
    options = ["--lossy-sources", "reencode"] if explicit_policy else []
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", TARGET, "-f", preset, *options)
    assert row[0] == "WARN" and record["out_format"] == source_kind
    warnings = {item["code"] for item in json.loads(record["warnings"])}
    assert "lossy_reencoded" in warnings
    assert record["reason"] == ("aac_bitrate_adjusted" if "aac_bitrate_adjusted" in warnings
                                else "lossy_reencoded")
    assert record["encoder"] != "stream_copy" and tools.facts(output)["codec_name"] == source_kind
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize("source_kind", ["mp3", "aac"])
def test_f04_lossy_skip_leaves_sources_untouched_and_lossless_neighbour_is_processed(
        norma, tools, tmp_path, lossy_source, source_kind):
    source = lossy_source(source_kind)
    neighbour = tools.tone(source.parent / "clean.wav", level=-20)
    before = tools.snapshot(source.parent)
    report = tmp_path / "report.csv"
    run = norma("normalise", source.parent, "--lossy-sources", "skip", "-o", tmp_path / "out", "--report", report)
    assert run.code == 0, run.stdout + run.stderr
    row = run.row(source.name)
    assert row[0] == "SKIP" and "lossy_source_skipped" in row[10] and row[9] == ""
    assert run.row(neighbour.name)[0] == "OK"
    assert [path.name for path in tools.audio_files(tmp_path / "out")] == ["clean.wav"]
    assert tools.snapshot(source.parent) == before
    assert run.summary["skipped"] == "1" and run.summary["ok"] == "1"


@pytest.mark.parametrize("source_kind", ["mp3", "aac"])
@pytest.mark.parametrize("requested,effective", [
    ("same", "flac"), ("mp3", "flac"), ("aac", "flac"), ("apple", "alac"),
    ("wav", "wav"), ("flac", "flac"), ("alac", "alac"), ("aiff", "aiff"),
])
def test_f04_lossless_policy_uses_the_format_family_and_warns_about_origin(
        norma, tools, tmp_path, lossy_source, source_kind, requested, effective):
    source = lossy_source(source_kind)
    before = tools.snapshot(source.parent)
    run, row, record, output = run_file(norma, tools, source, tmp_path, "--lossy-sources", "lossless", "-f", requested)
    assert (row[0], record["reason"], record["out_format"]) == ("WARN", "lossy_origin", effective)
    assert record["source_codec"] == source_kind and record["out_bit_depth"] == "16"
    assert record["bitrate_mode"] == "none"
    assert record["requested_bitrate_bps"] == record["effective_bitrate_bps"] == record["vbr_quality"] == ""
    assert ("FLAC is not among the formats Apple documents" in run.stderr) == (effective == "flac")
    assert tools.facts(output)["codec_name"] == {"wav": "pcm_s16le", "aiff": "pcm_s16be"}.get(effective, effective)
    assert tools.snapshot(source.parent) == before


@pytest.mark.parametrize("source_kind,bitrate", [("mp3", "V0"), ("aac", "128k")])
@pytest.mark.parametrize("explicit_bitrate", [False, True], ids=["automatic", "explicit"])
def test_f04_within_tolerance_lossy_input_skips_without_an_encoder_setting_or_copy_note(
        norma, tools, fixture_module, tmp_path, lossy_source, source_kind, bitrate, explicit_bitrate):
    source = lossy_source(source_kind)
    original, rate = decoded(source, tools, fixture_module)
    before = tools.snapshot(source.parent)
    options = ["--bitrate", bitrate] if explicit_bitrate else []
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "--report", report, "-t", "-20", *options)
    assert run.code == 0, run.stdout + run.stderr
    (row,), (record,) = run.files, tools.report(report)
    assert (row[0], record["reason"], record["execution"]) == ("SKIP", "already_within_tolerance", "not_written")
    assert row[6] == record["mode"] == "copy"
    assert record["encoder"] == "stream_copy" and record["bitrate_mode"] == "none"
    assert record["requested_bitrate_bps"] == record["effective_bitrate_bps"] == record["vbr_quality"] == ""
    assert row[9] == record["out_path"] == "" and not tools.audio_files(tmp_path / "out")
    after, output_rate = decoded(source, tools, fixture_module)
    assert rate == output_rate and np.array_equal(original, after)
    phrase = f"--bitrate {bitrate} not applied"
    assert phrase not in run.stderr  # D-036 / CR-025 ruling 9: no-output P4 skip has no copy note
    assert phrase not in row[10] and "lossy_reencoded" not in row[10]
    assert tools.snapshot(source.parent) == before


def test_a07_real_c6_cap_rejected_correction_finishes_as_verified_reduce(
        norma, tools, fixture_path, fixture_module, demos, tmp_path):
    source = demos({"c6.wav": fixture_path("F-05", "c6")}) / "c6.wav"
    before = tools.snapshot(source.parent)
    preview = norma("normalise", source, "--dry-run", "-t", "-21.5")
    assert preview.code == 0 and preview.files[0][6] == "limit", preview.stdout + preview.stderr
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", "-21.5")
    assert (row[0], row[6], record["reason"]) == ("WARN", "reduce", "reduce_shortfall")
    assert record["reduce_fallbacks"] == "1"  # fallback_from is not a C-09 report column
    assert 0 <= int(record["loudness_corrections"]) <= 2 and 0 <= int(record["peak_corrections"]) <= 1
    _, _, py_i, _ = independent(output, tools, fixture_module, target=None)
    assert abs(float(record["measured_shortfall_lu"]) - max(0, -21.5 - py_i)) <= 0.2
    assert "limit could not reach" in row[10]
    assert tools.snapshot(source.parent) == before


def test_a07_real_endpoint_loudness_correction_reaches_the_target(
        norma, tools, fixture_path, demos, tmp_path):
    source = demos({"endpoint.wav": fixture_path("F-03", "endpoint")}) / "endpoint.wav"
    _, row, record, output = run_file(norma, tools, source, tmp_path, "-t", "-40")
    assert row[0] == "WARN" and "too_short" in row[10]
    assert 1 <= int(record["loudness_corrections"]) <= 2
    assert record["reduce_fallbacks"] == "0" and record["peak_corrections"] == "0"
    assert abs(float(record["out_lufs"]) + 40) <= 0.5
    original, rate = tools.decode(source)
    after, output_rate = tools.decode(output)
    assert original.shape == after.shape and rate == output_rate
    assert lag(original[:, 0], after[:, 0]) == 0


@pytest.mark.parametrize("source_kind,bitrate", [("mp3", "V9"), ("aac", "300k")])
def test_f04_unevidenced_bitrate_refuses_even_a_prospective_copy_before_any_write(
        norma, lossy_source, ledger, tmp_path, source_kind, bitrate):
    """F7/F13: do not bypass preflight just because the source already meets −20."""
    source = lossy_source(source_kind)
    before = ledger()
    run = norma("normalise", source, "-t", "-20", "--bitrate", bitrate, "-o", tmp_path / "out")
    assert run.code == 2 and run.files == [] and bitrate in run.stderr
    assert "not evidenced" in run.stderr and ledger() == before


@pytest.mark.parametrize("source_kind", ["mp3", "aac"])
def test_f04_lossless_policy_does_not_change_lossless_neighbours(
        norma, tools, lossy_source, tmp_path, source_kind):
    source = lossy_source(source_kind)
    neighbour = tools.tone(source.parent / "clean.wav", level=-20)
    before = tools.snapshot(source.parent)
    report = tmp_path / "report.csv"
    run = norma("normalise", source.parent, "--lossy-sources", "lossless", "-f", "mp3",
                "-o", tmp_path / "out", "--report", report)
    assert run.code == 0, run.stdout + run.stderr
    records = {Path(record["input"]).name: record for record in tools.report(report)}
    assert run.row(source.name)[0] == "WARN" and records[source.name]["out_format"] == "flac"
    assert records[source.name]["reason"] == "lossy_origin"
    assert run.row(neighbour.name)[0] == "OK" and records[neighbour.name]["out_format"] == "mp3"
    assert records[neighbour.name]["vbr_quality"] == "0"
    assert sorted(path.suffix for path in tools.audio_files(tmp_path / "out")) == [".flac", ".mp3"]
    assert tools.snapshot(source.parent) == before
