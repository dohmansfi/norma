"""S1 acceptance: ``norma analyse`` on generated fixtures (TST-A-01 to A-03, S-05, C-07 in part)."""

from __future__ import annotations

import csv
import json
import shutil
import unicodedata

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.signal import resample_poly

F03_PEAK_DBTP = -2.989700043  # TESTING § 4.3 analytic oracle


def independent(path, *, dual_mono=False):
    """Oracle outside Norma: pyloudnorm K-weighted I and a 4x polyphase true-peak estimate."""
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if dual_mono and samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    integrated = pyln.Meter(rate, filter_class="K-weighting", block_size=0.400).integrated_loudness(samples)
    peak = np.max(np.abs(resample_poly(samples, 4, 1, axis=0, window=("kaiser", 12.0), padtype="constant")))
    return float(integrated), float(20 * np.log10(peak))


def only_row(run):
    assert len(run.files) == 1, run.stdout
    return run.files[0]


def numbers(row):
    """Integrated loudness, true peak and gain columns, as printed (one decimal)."""
    return float(row[3]), float(row[4]), float(row[5])


# C-09 flat report fields, in order (contract C-09).
REPORT_COLUMNS = (
    "input", "status", "reason", "out_path", "out_format", "out_bit_depth", "out_rate",
    "in_lufs", "in_dbtp", "in_lra", "gain_db", "mode", "limited_db", "out_lufs", "out_dbtp",
    "tag_sources", "conflicts", "dropped", "sidecar", "elapsed_s",
    "execution", "input_index", "in_signal_state", "out_signal_state",
    "target_lufs", "ceiling_dbtp", "expected_lufs", "predicted_shortfall_lu",
    "measured_shortfall_lu", "loudness_corrections", "peak_corrections", "reduce_fallbacks",
    "limiter_threshold_dbtp", "effective_limiter_bound_db", "requested_bitrate_bps", "effective_bitrate_bps",
    "bitrate_mode", "vbr_quality", "encoder", "ffmpeg_version", "warnings",
    "retained_failed_path", "published_paths", "in_peak_resolution_linear",
    "out_peak_resolution_linear", "target_origin", "ceiling_origin", "tolerance_lu",
    "tolerance_origin", "mark_validity", "source_codec", "origin_codec", "conditional_plans",
)
STATUS_LABEL = {"ok": "OK", "warn": "WARN", "skipped": "SKIP", "failed": "FAIL"}
MEASURED_COLUMNS = ("in_lufs", "in_dbtp", "in_lra", "gain_db", "out_lufs", "out_dbtp")
BOUNDED_JOBS = pytest.mark.parametrize("jobs", [("-j", "2"), ("-j", "4")], ids=["jobs2", "jobs4"])


def assert_stdout_rows(run, expected):
    """Stdout rows arrive in completion order, so compare them against the natural-order ``expected``.

    ``expected`` is the independently written natural discovery order: ``(path, label, reason)``.
    Every file must appear exactly once with the ``n/N`` counter of its natural index, its status
    and its reason code; nothing may be missing, duplicated or invented.
    """
    total = len(expected)
    paths = [row[2] for row in run.files]
    assert len(paths) == total, run.stdout
    assert len(set(paths)) == total, f"duplicate rows: {paths}"
    assert sorted(paths) == sorted(path for path, _, _ in expected)
    assert sorted(row[1] for row in run.files) == sorted(f"{n}/{total}" for n in range(1, total + 1))
    by_path = {row[2]: row for row in run.files}  # safe: uniqueness was asserted above
    for number, (path, label, reason) in enumerate(expected, 1):
        row = by_path[path]
        assert row[1] == f"{number}/{total}", (path, row[1])
        assert row[0] == label, (path, row)
        assert row[10].split(":", 1)[0] == reason, (path, row[10])
        if label == "FAIL":
            assert row[3:10] == [""] * 7, (path, row)


def read_report(path):
    """The report's per-file rows exactly as written, plus its JSON envelope (None for CSV)."""
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == REPORT_COLUMNS
            return list(reader), None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    rows = envelope["outcomes"]
    assert all(tuple(row) == REPORT_COLUMNS for row in rows)
    return rows, envelope


def assert_report_in_input_order(report, expected, run):
    """The final report is in input_index order, whatever order stdout completed in."""
    total = len(expected)
    rows, envelope = read_report(report)
    assert [row["input"] for row in rows] == [path for path, _, _ in expected]
    assert [int(row["input_index"]) for row in rows] == list(range(total))
    for row, (_, label, reason) in zip(rows, expected, strict=True):
        assert STATUS_LABEL[row["status"]] == label, row["input"]
        assert (row["reason"] or "") == (reason or "ok"), row["input"]  # an OK file's reason code is ok
        if label == "FAIL":
            assert all(row[column] in ("", None) for column in MEASURED_COLUMNS), row["input"]
    # Stdout and report describe the same files with the same outcomes.
    stdout = {row[2]: row[0] for row in run.files}
    assert stdout == {row["input"]: STATUS_LABEL[row["status"]] for row in rows}
    counts = {key: sum(1 for _, label, _ in expected if label == name)
              for key, name in STATUS_LABEL.items()}
    assert {key: int(run.summary[key]) for key in counts} == counts
    assert run.summary["report"] == str(report)
    if envelope is not None:
        assert {key: envelope["counts"][key] for key in counts} == counts
        assert envelope["report_path"] == str(report)


@pytest.mark.parametrize(("id", "expected"), [("F-01", -23.0), ("F-02", -33.0)])
def test_a01_integrated_loudness_and_gain_for_tones(norma, fixture_path, id, expected):
    """TST-A-01: I within ±0.1 LU of −23/−33 and of the independent meter; default target −16."""
    path = fixture_path(id)
    run = norma("analyse", path)
    row = only_row(run)
    loudness, peak, gain = numbers(row)
    oracle_i, oracle_tp = independent(path)
    assert run.code == 0
    assert row[:3] == ["OK", "1/1", str(path)]
    assert abs(loudness - expected) <= 0.1
    assert abs(loudness - oracle_i) <= 0.1
    assert abs(peak - oracle_tp) <= 0.3
    assert (row[6], row[10]) == ("linear", "")
    assert gain == pytest.approx(-16.0 - expected, abs=0.1)
    assert run.summary["ok"] == "1" and run.summary["failed"] == "0"
    assert run.stderr == ""


def test_a01_target_option_changes_only_the_predicted_gain(norma, fixture_path):
    path = fixture_path("F-02")
    default, louder = only_row(norma("analyse", path)), only_row(norma("analyse", "-t", "-14", path))
    assert default[3:5] == louder[3:5]
    assert float(default[5]) == pytest.approx(17.0, abs=0.1)
    assert float(louder[5]) == pytest.approx(19.0, abs=0.1)


def test_a02_true_peak_of_the_inter_sample_fixture(norma, fixture_path):
    """TST-A-02: F-03 true peak within ±0.3 dB of the analytic −2.9897 dBTP."""
    path = fixture_path("F-03")
    row = only_row(norma("analyse", path))
    _, peak, _ = numbers(row)
    assert row[0] == "OK"
    assert abs(peak - F03_PEAK_DBTP) <= 0.3
    assert abs(peak - independent(path)[1]) <= 0.3


@pytest.mark.parametrize("id", ["F-04", "F-06"])
def test_a03_programme_material_agrees_with_an_independent_meter(norma, fixture_path, id):
    """TST-A-03: pink-noise programme (and the clipped float fixture) within 0.2 LU of pyloudnorm."""
    path = fixture_path(id)
    row = only_row(norma("analyse", path))
    loudness, peak, _ = numbers(row)
    oracle_i, oracle_tp = independent(path)
    assert row[0] == "OK"
    assert abs(loudness - oracle_i) <= 0.2
    assert abs(peak - oracle_tp) <= 0.3


def test_a03_mono_policy_changes_loudness_by_3_01_lu_and_not_the_peak(norma, fixture_path):
    """TST-A-03: dual-mono is the default; --no-dual-mono differs by 3.010 ±0.1 LU, TP unchanged."""
    path = fixture_path("F-07")
    dual, single = only_row(norma("analyse", path)), only_row(norma("analyse", "--no-dual-mono", path))
    dual_i, dual_tp, _ = numbers(dual)
    single_i, single_tp, _ = numbers(single)
    assert dual_i - single_i == pytest.approx(3.010299957, abs=0.1)
    assert dual_tp == single_tp
    assert abs(dual_i - independent(path, dual_mono=True)[0]) <= 0.2
    assert abs(single_i - independent(path)[0]) <= 0.2


def ten_file_batch(fixture_path, demos):
    good = fixture_path("F-01")
    members = {f"take-{number:02d}.wav": good for number in range(1, 10)}
    members["take-10-truncated.wav"] = fixture_path("F-10")
    root = demos(members)
    # Independent natural order (numbers compare numerically): take-01 … take-09, then take-10.
    expected = [(str(root / f"take-{number:02d}.wav"), "OK", "") for number in range(1, 10)]
    expected.append((str(root / "take-10-truncated.wav"), "FAIL", "truncated"))
    return root, expected


@BOUNDED_JOBS
def test_s05_ten_file_batch_with_one_truncated_file_reports_every_file(norma, fixture_path, demos, jobs):
    """TST-S-05: nine OK, one structurally truncated FAIL, exit 1, ten rows (stdout in completion order)."""
    root, expected = ten_file_batch(fixture_path, demos)
    run = norma("analyse", *jobs, root)
    assert run.code == 1
    assert_stdout_rows(run, expected)
    assert {key: run.summary[key] for key in ("ok", "warn", "skipped", "failed", "not_started", "run_errors")} == {
        "ok": "9", "warn": "0", "skipped": "0", "failed": "1", "not_started": "0", "run_errors": "0"}
    assert run.summary["report"] == ""  # no report was asked for, so none is named


@BOUNDED_JOBS
@pytest.mark.parametrize("extension", ["csv", "json"])
def test_s05_requested_report_keeps_input_index_order_while_stdout_completes_in_any_order(
        norma, fixture_path, demos, tmp_path, jobs, extension):
    """C-09: stdout rows are in completion order; the final CSV/JSON report is in input_index order."""
    root, expected = ten_file_batch(fixture_path, demos)
    report = tmp_path / f"batch-report.{extension}"
    run = norma("analyse", *jobs, "--report", report, root)
    assert run.code == 1
    assert_stdout_rows(run, expected)
    assert_report_in_input_order(report, expected, run)
    assert sorted(path.name for path in tmp_path.glob("batch-report.*")) == [report.name]


def test_s05_report_order_is_stable_across_repeated_concurrent_runs(norma, fixture_path, demos, tmp_path):
    """Several bounded-concurrency runs may complete differently but must report identically."""
    root, expected = ten_file_batch(fixture_path, demos)
    orders = set()
    for attempt in range(4):
        report = tmp_path / f"repeat-{attempt}.csv"
        run = norma("analyse", "-j", "4", "--report", report, root)
        assert run.code == 1
        assert_stdout_rows(run, expected)
        assert_report_in_input_order(report, expected, run)
        orders.add(tuple(row[2] for row in run.files))
    assert all(len(order) == 10 for order in orders)


@BOUNDED_JOBS
@pytest.mark.parametrize("extension", [None, "json"])
def test_s05_every_kind_of_bad_input_is_reported_beside_the_good_one(
        norma, fixture_path, demos, tmp_path, jobs, extension):
    """Skips and failures keep distinct reason codes; the good file is still measured; exit 1."""
    root = demos({"a-good.wav": fixture_path("F-01"), "b-not-audio.wav": fixture_path("F-11"),
                  "c-truncated.wav": fixture_path("F-10"), "d-truncated.aif": fixture_path("F-10", "aiff")})
    (root / "e-empty.wav").write_bytes(b"")
    sf.write(root / "f-silence.wav", np.zeros((96000, 2)), 48000, subtype="PCM_24")
    sf.write(root / "g-three-channels.wav", np.full((48000, 3), 0.05), 48000, subtype="PCM_24")
    (root / "notes.txt").write_text("not audio by name", encoding="utf-8")
    # Independent natural order and expectation; notes.txt is not audio by name and is only counted as ignored.
    expected = [(str(root / name), label, reason) for name, label, reason in (
        ("a-good.wav", "OK", ""), ("b-not-audio.wav", "SKIP", "not_audio"),
        ("c-truncated.wav", "FAIL", "truncated"), ("d-truncated.aif", "FAIL", "truncated"),
        ("e-empty.wav", "SKIP", "not_audio"), ("f-silence.wav", "SKIP", "silent"),
        ("g-three-channels.wav", "SKIP", "unsupported_channels"))]
    report = tmp_path / f"bad-inputs.{extension}" if extension else None
    run = norma("analyse", *jobs, *(("--report", report) if report else ()), root)
    assert run.code == 1
    assert_stdout_rows(run, expected)
    assert str(root / "notes.txt") not in {row[2] for row in run.files}
    assert run.summary["ok"] == "1" and run.summary["skipped"] == "4" and run.summary["failed"] == "2"
    assert run.summary["ignored"] == "1"
    if report:
        assert_report_in_input_order(report, expected, run)
    else:
        assert run.summary["report"] == ""


def test_s05_skips_alone_do_not_fail_the_run(norma, fixture_path, demos):
    """Exit 0 allows warnings and skips (CLI exit rules); only a file failure gives 1."""
    run = norma("analyse", demos({"b-not-audio.wav": fixture_path("F-11")}))
    assert run.code == 0
    assert [row[0] for row in run.files] == ["SKIP"]


@BOUNDED_JOBS
def test_s01_folder_walk_is_recursive_only_on_request(norma, fixture_path, tmp_path, jobs):
    """The F-15 path tree: -r reaches subfolders and drops hidden, sidecar, own-output and alias files."""
    tree = fixture_path("F-15")
    nfd = unicodedata.normalize("NFD", "Perché")
    # Independent expectation (CLI § 2.4): per folder, case-folded natural order with sub-folders in place.
    # alias.wav (a symlink), hardlink.wav and space name.wav are one inode: the first one met, alias.wav,
    # keeps its place in the order and is reported under its real name, space name.wav; the others are
    # dropped. Hidden, AppleDouble, Normalised/ and .norma-originals/ entries are excluded.
    top = ["space name.wav", "Perché.wav", "Test Artist's demo.wav", "東京 🎵.wav"]
    deep_order = ["a/Song.wav", "space name.wav", "Artist/Album/03 - Perché.wav", "Artist/Album/004_Not_a_match.wav",
                  "b/song.WAV", "Perché.wav", "Test Artist's demo.wav", f"unicode/{nfd}.wav", "東京 🎵.wav"]
    flat = norma("analyse", *jobs, tree)
    deep = norma("analyse", *jobs, "-r", tree)
    assert flat.code == deep.code == 0
    assert_stdout_rows(flat, [(str(tree / name), "OK", "") for name in top])
    assert_stdout_rows(deep, [(str(tree / name), "OK", "") for name in deep_order])
    names = [row[2][len(str(tree)) + 1:] for row in deep.files]
    for name in names:
        parts = name.split("/")
        assert not any(part.startswith(".") for part in parts), name
        assert parts[0] != "Normalised", name
    assert "alias.wav" not in names and "hardlink.wav" not in names
    assert "duplicate file(s) (same device and inode) were dropped" in deep.stderr
    assert int(deep.summary["ignored"]) > 0
    assert int(flat.summary["ignored"]) >= 0 and flat.summary["report"] == deep.summary["report"] == ""
    # The recursive report lists the same nine files in input_index order.
    report = tmp_path / "deep.csv"
    reported = norma("analyse", *jobs, "-r", "--report", report, tree)
    assert reported.code == 0
    assert_report_in_input_order(report, [(str(tree / name), "OK", "") for name in deep_order], reported)
    # Quiet hides the stderr note and nothing else.
    assert norma("analyse", "-r", "-q", tree).stderr == ""


def test_c07_american_spelling_is_an_alias(norma, fixture_path):
    path = fixture_path("F-01")
    british, american = norma("analyse", path), norma("analyze", path)
    assert british.code == american.code == 0
    assert british.files == american.files


def test_c08_dynamic_analyse_succeeds_with_a_read_only_preview(norma, fixture_path, demos, ledger):
    """S3 (CR-062): dynamic analyse shows the preview label without filesystem changes."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    before = ledger()
    run = norma("analyse", "--peak-mode", "dynamic", path)
    row = only_row(run)
    assert run.code == 0, run.stderr
    assert row[:3] == ["OK", "1/1", str(path)]
    assert row[6] == "dynamic (calibrated at render)"
    assert row[7:10] == [""] * 3
    assert run.summary["ok"] == "1" and run.summary["failed"] == "0"
    assert ledger() == before


@pytest.mark.parametrize(("arguments", "message"), [
    (("analyse", "--report", "{cwd}/no-such-folder/out.csv", "{f}"), "does not exist"),
    (("analyse", "--report", "{cwd}/out.txt", "{f}"), "--report must name a .csv or .json file"),
    (("analyse",), "no input given"),
    (("normalise", "--dry-run", "--report", "{cwd}/out.xml", "{f}"), "--report must name a .csv or .json file"),
    (("verify", "--skip-within", "0.2", "{f}"), "--skip-within is not used by verify"),
    (("tags", "template", "{f}"), "tags template requires -o FILE.csv|FILE.json|-"),
    (("analyse", "--in-place", "{f}"), "--in-place is not available in this version"),
    (("analyse", "--no-backup", "{f}"), "--no-backup is not available in this version"),
    (("analyse", "--max-limti", "3", "{f}"), "unknown option"),
])
def test_c08_unavailable_and_unsafe_requests_exit_2_before_any_work(norma, fixture_path, tmp_path, arguments, message):
    path = fixture_path("F-01")
    run = norma(*(str(path) if argument == "{f}" else argument.replace("{cwd}", str(tmp_path)) for argument in arguments))
    assert run.code == 2
    assert run.stdout == ""
    assert message in run.stderr.splitlines()[0]
    assert run.stderr.splitlines()[0].startswith("norma: error: ")


def test_c08_tags_template_requires_output_before_any_work(norma, fixture_path, demos, tmp_path, ledger):
    """S4 (CR-093), CLI § 2/U-18: missing -o is a usage error before tools or writes."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        tool = tool_dir / name
        tool.write_text('#!/bin/sh\nprintf invoked > "$0.called"\nexit 77\n', encoding="utf-8")
        tool.chmod(0o755)
    before = ledger()
    run = norma("tags", "template", path, "--ffmpeg", tool_dir / "ffmpeg", path=str(tool_dir))
    assert run.code == 2
    assert run.stdout == ""
    assert run.stderr.splitlines()[0] == "norma: error: tags template requires -o FILE.csv|FILE.json|-"
    assert ledger() == before


def test_c08_tags_plan_never_invokes_ffmpeg_or_fills_audio_cells(norma, fixture_path, demos, tmp_path, ledger):
    """S4 (CR-091), CLI §§ 2/7.1: ffprobe-only planning, no measurement, render or writes."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    ffmpeg = tool_dir / "ffmpeg"
    ffmpeg.write_text('#!/bin/sh\nprintf invoked > "$0.called"\nexit 77\n', encoding="utf-8")
    ffmpeg.chmod(0o755)
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    (tool_dir / "ffprobe").symlink_to(ffprobe)
    before = ledger()
    run = norma("tags", "plan", path, "--ffmpeg", ffmpeg, "--target", "-14", "--log", tmp_path / "run.log")
    row = only_row(run)
    assert run.code == 0, run.stderr
    assert row[:3] == ["OK", "1/1", str(path)]
    assert row[3:10] == [""] * 7
    assert "reduce_shortfall" not in run.stdout
    assert run.summary["ok"] == "1" and run.summary["report"] == ""
    assert "no audio is processed by tags plan" in run.stderr
    assert "writes no log file" in run.stderr
    assert ledger() == before


def test_c08_report_refusals_exit_2_and_write_nothing(norma, fixture_path, tmp_path):
    """S2 (CR-050): an existing report needs --overwrite and a symlink is never followed; neither is touched."""
    path = fixture_path("F-01")
    existing = tmp_path / "existing.csv"
    existing.write_text("keep me\n")
    link = tmp_path / "link.csv"
    link.symlink_to(existing)
    for report, message in ((existing, "already exists; use --overwrite"), (link, "exists as a symlink")):
        run = norma("analyse", "--report", str(report), path)
        assert run.code == 2, run.stderr
        assert run.stdout == ""
        assert message in run.stderr.splitlines()[0]
    assert existing.read_text() == "keep me\n"
    assert link.is_symlink()
    assert sorted(entry.name for entry in tmp_path.glob("*.csv")) == ["existing.csv", "link.csv"]
