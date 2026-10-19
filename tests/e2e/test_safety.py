"""S2 acceptance, file safety and batches (TST-S-02 to S-05, CR-046, D-069).

Every test drives the real ``norma`` command on generated untagged inputs inside the hermetic
sandbox. What a run left on disk is judged by the identity ledger of the whole sandbox, not by what
the run reported.
"""

from __future__ import annotations

import os
import re
import signal
import stat
import subprocess
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

REPORT_NAME = re.compile(r"norma-report-\d{4}-\d\d-\d\dT\d{6}(-\d+)?\.csv")


def tone_folder(tools, folder: Path, names, *, seconds: float = 4.0) -> Path:
    """A folder of short untagged tones, each at a distinct level so no two files are alike."""
    for index, name in enumerate(names):
        tools.tone(folder / name, level=-30 + 2 * (index % 8), seconds=seconds, frequency=440 + 110 * index)
    return folder


def outputs_in(folder: Path, tools) -> list[str]:
    return [str(path.relative_to(folder)) for path in tools.audio_files(folder)]


# --- TST-S-02: naming modes and collisions ------------------------------------------------


def test_s02_default_output_goes_into_normalised_beside_each_source(norma, tools, tmp_path):
    """Recursive default: ``root/Album/song.wav`` → ``root/Album/Normalised/song.wav``, no mirrored tree."""
    root = tone_folder(tools, tmp_path / "demos", ["song.wav", "album/track.wav"])
    run = norma("normalise", root, "-r", "-t", "-16")
    assert run.code == 0, run.stderr
    assert outputs_in(root, tools) == [
        "Normalised/song.wav", "album/Normalised/track.wav", "album/track.wav", "song.wav",
    ]
    assert {Path(row[9]) for row in run.files} == {root / "Normalised" / "song.wav",
                                                    root / "album" / "Normalised" / "track.wav"}


def test_s02_out_dir_mirrors_the_parent_relative_to_each_retained_root(norma, tools, tmp_path):
    root = tone_folder(tools, tmp_path / "demos", ["a.wav", "sub/b.wav"])
    destination = tmp_path / "elsewhere" / "tree"
    run = norma("normalise", root, "-r", "-o", destination)
    assert run.code == 0, run.stderr
    assert outputs_in(destination, tools) == ["a.wav", "sub/b.wav"]
    assert not (root / "Normalised").exists() and not (root / "sub" / "Normalised").exists()


def test_s02_explicit_files_contribute_their_basename_to_a_flat_out_dir(norma, tools, tmp_path):
    tone_folder(tools, tmp_path / "demos", ["a.wav", "sub/b.wav"])
    destination = tmp_path / "flat"
    run = norma("normalise", tmp_path / "demos" / "a.wav", tmp_path / "demos" / "sub" / "b.wav", "-o", destination)
    assert run.code == 0, run.stderr
    assert outputs_in(destination, tools) == ["a.wav", "b.wav"]


def test_s02_suffix_writes_beside_the_source_with_the_target_expanded(norma, tools, tmp_path):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    run = norma("normalise", root, "-t", "-14", "--suffix= [{target}]")
    assert run.code == 0, run.stderr
    assert outputs_in(root, tools) == ["song [-14].wav", "song.wav"]
    assert not (root / "Normalised").exists()


def test_s02_out_dir_and_suffix_exclude_each_other(norma, tools, tmp_path, ledger):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    before = ledger()
    run = norma("normalise", root, "-o", tmp_path / "out", "-s", "-x")
    assert run.code == 2 and run.stdout == ""
    assert ledger() == before


@pytest.mark.parametrize(("options", "name"), [
    ([], "song.wav"),
    (["--title", "Perche\u0301 / demo: take", "--track", "3", "--suffix= [{target}]"],
     "03 Perché - demo- take [-16].wav"),
])
def test_s02_rename_uses_resolved_tags_and_changes_only_the_copy_name(norma, tools, tmp_path, ledger, options, name):
    """S4 (CR-091), CLI § 3.2: fallback/explicit tags, cleaning, NFC and suffix placement."""
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    before = ledger()
    pattern = "{track:02} {title}" if options else "{title}"
    run = norma("normalise", root, "--rename", pattern, *options)
    output = root / name if options else root / "Normalised" / name
    assert run.code == 0, run.stderr
    assert len(run.files) == 1 and run.files[0][0] == "OK", run.stdout
    assert run.files[0][9] == str(output)
    assert output.is_file() and tools.decode(output)[0].shape == tools.decode(root / "song.wav")[0].shape
    after = ledger()
    assert {key: after[key] for key in before} == before
    assert {path for path in tools.audio_files(root)} == {root / "song.wav", output}


def test_s02_rename_with_an_empty_placeholder_fails_the_file_without_writes(norma, tools, tmp_path, ledger):
    """CLI § 3.2: a missing artist is unsafe_output, not a usage refusal; dry run writes nothing."""
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    before = ledger()
    run = norma("normalise", root, "--dry-run", "--rename", "{artist} {title}")
    assert run.code == 1, run.stderr
    assert len(run.files) == 1 and run.files[0][0] == "FAIL", run.stdout
    assert run.files[0][10].startswith("unsafe_output:")
    assert run.summary["failed"] == "1" and run.summary["report"] == ""
    assert ledger() == before


def test_s02_names_that_collide_fail_every_participant_and_clobber_nothing(norma, tools, tmp_path):
    """Case-folded and NFC/NFD collisions in a flat -o folder fail all initial participants."""
    root = tmp_path / "demos"
    composed, decomposed = "Perché.wav", unicodedata.normalize("NFD", "Perché.wav")
    assert composed != decomposed
    members = ["A/Song.wav", "B/song.wav", f"C/{composed}", f"D/{decomposed}", "E/Unique.wav"]
    tone_folder(tools, root, members)
    out = tmp_path / "out"
    run = norma("normalise", *(root / member for member in members), "-o", out)
    by_name = {Path(row[2]).name: row for row in run.files}
    assert run.code == 1
    for name in ("Song.wav", "song.wav", composed, decomposed):
        assert by_name[name][0] == "FAIL" and "name_clash" in by_name[name][10], by_name[name]
    assert by_name["Unique.wav"][0] == "OK"
    assert outputs_in(out, tools) == ["Unique.wav"]


def test_s02_accents_are_never_stripped_when_comparing_names(norma, tools, tmp_path):
    members = ["A/Perché.wav", "B/Perche.wav"]
    tone_folder(tools, tmp_path / "demos", members)
    out = tmp_path / "out"
    run = norma("normalise", *(tmp_path / "demos" / member for member in members), "-o", out)
    assert run.code == 0, run.stderr
    assert {unicodedata.normalize("NFC", name) for name in outputs_in(out, tools)} == {"Perché.wav", "Perche.wav"}


def test_s02_an_existing_output_is_kept_unless_overwrite_is_given(norma, tools, tmp_path):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    assert norma("normalise", root).code == 0
    target = root / "Normalised" / "song.wav"
    target.write_bytes(b"sentinel")
    second = norma("normalise", root)
    assert second.code == 0 and [row[0] for row in second.files] == ["SKIP"]
    assert "output_exists" in second.files[0][10] and second.summary["skipped"] == "1"
    assert target.read_bytes() == b"sentinel"
    third = norma("normalise", root, "--overwrite")
    assert third.code == 0, third.stderr
    assert target.read_bytes() != b"sentinel" and target.stat().st_size > 1000


# --- TST-S-03: sources never change --------------------------------------------------------


def test_s03_sources_are_unchanged_by_checksum_and_mtime(norma, tools, tmp_path):
    root = tone_folder(tools, tmp_path / "demos", ["a.wav", "b.wav", "sub/c.wav"])
    sources = {path: (path.stat().st_mtime_ns, tools.sha256(path)) for path in tools.audio_files(root)}
    for options in ([], ["-o", str(tmp_path / "out")], ["--suffix=-copy"], ["--dry-run"], ["--overwrite"]):
        assert norma("normalise", root, "-r", *options).code == 0
        assert {path: (path.stat().st_mtime_ns, tools.sha256(path)) for path in sources} == sources, options


@pytest.mark.parametrize("flag", ["--in-place", "--no-backup"])
@pytest.mark.parametrize("modifier", [[], ["--dry-run"], ["--yes"], ["--force"], ["--overwrite"]])
@pytest.mark.parametrize("command", ["normalise", "analyse"])
def test_s03_in_place_and_no_backup_are_refused_before_any_row_or_write(
        norma, tools, tmp_path, ledger, command, modifier, flag):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    before = ledger()
    run = norma(command, root, flag, *modifier)
    assert run.code == 2 and run.stdout == "" and run.stderr.strip()
    assert ledger() == before


# --- TST-S-04: SIGINT ----------------------------------------------------------------------


@pytest.mark.parametrize("interrupts", [1, 2])
def test_s04_sigint_stops_cleanly_without_a_partial_output(start, tools, tmp_path, interrupts):
    """TST-S-04: exit 130, a final summary, only complete outputs, no orphaned worker."""
    root = tmp_path / "demos"
    source = tools.tone(root / "seed.wav", level=-24, seconds=60)
    count = 24
    for index in range(count):
        (root / f"take-{index:02d}.wav").write_bytes(source.read_bytes())
    source.unlink()
    process = start("normalise", root, "-j", "1", "-t", "-16")
    out = root / "Normalised"
    assert tools.wait_until(lambda: out.is_dir() and bool(tools.audio_files(out)), timeout=60)
    for index in range(interrupts):
        process.send_signal(signal.SIGINT)
        if index + 1 < interrupts:
            time.sleep(0.3)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 130, (stdout, stderr)
    summary = dict(cell.split("=", 1) for line in stdout.splitlines() if line.startswith("SUMMARY")
                   for cell in line.split("\t")[1:])
    assert summary, stdout
    assert "Interrupted" in stderr or "interrupted" in stderr
    assert int(summary["not_started"]) > 0
    rows = [line.split("\t") for line in stdout.splitlines() if line.split("\t")[0] in {"OK", "WARN", "SKIP", "FAIL"}]
    assert len(rows) + int(summary["not_started"]) <= count
    finished = [row for row in rows if row[0] in {"OK", "WARN"}]
    produced = tools.audio_files(out)
    # Every published file is complete and was reported; nothing else (no temp or partial file) remains.
    assert sorted(path.name for path in produced) == sorted(Path(row[9]).name for row in finished)
    leftovers = [p.name for p in out.iterdir() if p not in produced and not REPORT_NAME.fullmatch(p.name)]
    assert leftovers == []
    frames = tools.decode(root / "take-00.wav")[0].shape
    for path in produced:
        assert tools.decode(path)[0].shape == frames
    others = subprocess.run(["pgrep", "-f", str(tmp_path)], capture_output=True, text=True)
    assert others.stdout.strip() == ""


# --- TST-S-05: a control batch with one truncated file -------------------------------------


def test_s05_ten_file_batch_with_one_truncated_input(norma, tools, tmp_path, fixture_path):
    """Nine OK, one structurally truncated FAIL, exit 1, ten ordered report rows."""
    root = tmp_path / "batch"
    names = [f"take-{index:02d}.wav" for index in range(1, 11)]
    tone_folder(tools, root, names[:4] + names[5:])
    truncated = fixture_path("F-10")
    (root / names[4]).write_bytes(truncated.read_bytes())
    report = tmp_path / "report.csv"
    run = norma("normalise", root, "--report", report, "-o", tmp_path / "out")
    assert run.code == 1, run.stderr
    assert len(run.files) == 10
    assert [Path(row[2]).name for row in sorted(run.files, key=lambda row: int(row[1].split("/")[0]))] == names
    assert {row[0] for row in run.files if Path(row[2]).name != names[4]} == {"OK"}
    assert run.row(names[4])[0] == "FAIL"
    assert (run.summary["ok"], run.summary["failed"], run.summary["run_errors"]) == ("9", "1", "0")
    rows = tools.report(report)
    assert [Path(row["input"]).name for row in rows] == names
    assert [int(row["input_index"]) for row in rows] == list(range(10))
    assert [row["status"] for row in rows].count("ok") == 9 and rows[4]["status"] == "failed"
    assert sorted(outputs_in(tmp_path / "out", tools)) == sorted(set(names) - {names[4]})
    assert (tmp_path / "report.csv").is_file() and not any(tmp_path.joinpath("out").glob("norma-report-*"))


# --- CR-046: output mode and timestamp ------------------------------------------------------


@pytest.mark.parametrize(("mask", "mode"), [(0o022, 0o644), (0o077, 0o600)])
def test_cr046_output_takes_the_umask_mode_and_keeps_the_source_mtime(norma, tools, tmp_path, mask, mode):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    source = root / "song.wav"
    old_ns = 1_614_834_367_123_456_789  # 2021-03-04, with a nanosecond part to prove it is kept
    os.utime(source, ns=(old_ns, old_ns))
    run = norma("normalise", root, "-t", "-16", umask=mask)
    assert run.code == 0, run.stderr
    output = root / "Normalised" / "song.wav"
    assert stat.S_IMODE(output.stat().st_mode) == mode
    assert output.stat().st_mtime_ns == source.stat().st_mtime_ns == old_ns
    report = next((root / "Normalised").glob("norma-report-*.csv"))
    assert stat.S_IMODE(report.stat().st_mode) == mode


# --- D-069: automatic report collisions -----------------------------------------------------


@pytest.mark.parametrize("overwrite", [False, True])
def test_d069_automatic_report_collision_takes_an_ordinal_and_never_replaces(norma, tools, tmp_path, overwrite):
    root = tone_folder(tools, tmp_path / "demos", ["song.wav"])
    out = root / "Normalised"
    out.mkdir()
    now = datetime.now()
    sentinels = {}
    for offset in range(-2, 25):  # every second the run could name its report after
        path = out / f"norma-report-{(now + timedelta(seconds=offset)):%Y-%m-%dT%H%M%S}.csv"
        path.write_text(f"sentinel {offset}\n", encoding="utf-8")
        sentinels[path] = tools.sha256(path)
    started = time.monotonic()
    run = norma("normalise", root, *(["--overwrite"] if overwrite else []))
    assert time.monotonic() - started < 20
    assert run.code == 0, run.stderr
    published = Path(run.summary["report"])
    assert re.fullmatch(r"norma-report-\d{4}-\d\d-\d\dT\d{6}-2\.csv", published.name), published
    assert published.parent == out and published.read_text(encoding="utf-8").startswith("input,status")
    assert {path: tools.sha256(path) for path in sentinels} == sentinels
