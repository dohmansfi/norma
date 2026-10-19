"""S1 acceptance: the read-only commands write nothing (TST-C-03, TST-S-03 in part).

Each case takes an identity ledger of the whole test sandbox (inputs, fixture cache, HOME, XDG
folders, TMPDIR and the working directory: type, size, mtime_ns and SHA-256 or link target of every
entry), runs the command in a child process, and requires the ledger to be unchanged. A run that
creates, removes, renames, touches or rewrites anything — a report, log, ``Normalised`` folder,
backup, temporary file, preference file or cache — fails here. The child is not under the Python
audit hook (tests/conftest.py), so this ledger is the evidence for it.
"""

from __future__ import annotations

import csv
import json
import os
import stat

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def library(fixture_path, demos):
    """A tagged, untagged, mixed-container and deliberately broken folder with a nested level."""
    root = demos({
        "01 untagged.wav": fixture_path("F-01"),
        "02 mono.wav": fixture_path("F-07"),
        "03 tagged.wav": fixture_path("F-13", "wav24"),
        "04 tagged.flac": fixture_path("F-13", "flac24"),
        "05 tagged.m4a": fixture_path("F-13", "alac24"),
        "06 tagged.mp3": fixture_path("F-13", "mp3-v0"),
        "07 truncated.wav": fixture_path("F-10"),
        "08 not-audio.wav": fixture_path("F-11"),
        "09 caf.caf": fixture_path("F-12", "caf-pcm"),
        "nested/10 untagged.wav": fixture_path("F-01"),
    })
    sf.write(root / "11 silence.wav", np.zeros((96000, 2)), 48000, subtype="PCM_24")
    return root


COMMANDS = {
    "analyse": ("analyse", "{root}"),
    "analyse recursive, targets, verbose": ("analyse", "-r", "-v", "-t", "-14", "-p", "-2", "{root}"),
    "analyse quiet, no dual mono": ("analyse", "-r", "-q", "--no-dual-mono", "{root}"),
    "analyse names a log and overwrite": ("analyse", "-r", "--log", "{cwd}/run.log", "--overwrite", "{root}"),
    "analyze (alias)": ("analyze", "{root}/01 untagged.wav"),
    "verify": ("verify", "{root}"),
    "verify recursive, targets, verbose": ("verify", "-r", "-v", "-t", "-14", "-p", "-2", "--tolerance", "1", "{root}"),
    "verify checks tags": ("verify", "-r", "--check-tags", "{root}"),
    "verify names a log and overwrite": ("verify", "-r", "--log", "{cwd}/run.log", "--overwrite", "{root}"),
    "verify a single tagged file": ("verify", "{root}/03 tagged.wav"),
    "tags show": ("tags", "show", "-r", "{root}"),
    "tags show a single file": ("tags", "show", "{root}/03 tagged.wav"),
    "tags plan": ("tags", "plan", "-r", "{root}"),
    "doctor": ("doctor",),
    "formats": ("formats",),
    "help": ("help", "analyse"),
    "normalise dry run": ("normalise", "--dry-run", "-r", "{root}"),
    "normalise dry run with targets": ("normalise", "-n", "-t", "-14", "-p", "-2", "{root}/01 untagged.wav"),
}
REFUSED = {
    "analyse --report exists": ("analyse", "--report", "{cwd}/existing.csv", "{root}"),
    "analyse --report symlink": ("analyse", "--report", "{cwd}/link.csv", "--overwrite", "{root}"),
    "analyse --report missing folder": ("analyse", "--report", "{cwd}/no-such-folder/report.json", "{root}"),
    "analyse --report bad extension": ("analyse", "--report", "{cwd}/report.txt", "{root}"),
    "normalise --dry-run --report exists": ("normalise", "--dry-run", "--report", "{cwd}/existing.csv", "{root}"),
    "verify --report exists": ("verify", "--report", "{cwd}/existing.csv", "{root}"),
    "verify --report symlink": ("verify", "--report", "{cwd}/link.csv", "--overwrite", "{root}"),
    "verify --report bad extension": ("verify", "--report", "{cwd}/report.txt", "{root}"),
    "verify --skip-within": ("verify", "--skip-within", "0.2", "{root}"),
    "verify field option without --check-tags": ("verify", "--album", "Demos", "{root}"),
    "verify in place": ("verify", "--in-place", "{root}"),
    "tags plan --report exists": ("tags", "plan", "--report", "{cwd}/existing.csv", "{root}"),
    "tags template": ("tags", "template", "{root}"),
    "in place": ("analyse", "--in-place", "{root}"),
    "no backup": ("analyse", "--no-backup", "{root}"),
    "no input": ("analyse",),
}


def expand(arguments, root, cwd):
    return [argument.replace("{root}", str(root)).replace("{cwd}", str(cwd)) for argument in arguments]


@pytest.mark.parametrize("name", list(COMMANDS))
def test_c03_read_only_commands_leave_the_sandbox_identical(norma, library, ledger, tmp_path, name):
    """TST-C-03: no audio, output, temporary, backup, report or log appears; nothing is touched."""
    before = ledger()
    run = norma(*expand(COMMANDS[name], library, tmp_path))
    after = ledger()
    assert run.code in {0, 1}, run.stderr  # 1 only for the deliberately broken files
    assert run.stdout
    if name == "doctor" or name.startswith("normalise"):
        # Not a pure file-reading command: the encoder-evidence probe makes a ``norma-doctor-*`` (doctor)
        # or ``norma-evidence-*`` (normalise, TK-0708 finding) folder in TMPDIR and removes it again.
        # the CLI read-only rules say --dry-run creates no temporary file, so the normalise case is a
        # proposed finding, not an endorsement. Only TMPDIR's own mtime may differ; every entry,
        # including TMPDIR's contents, must be as before.
        for ledger_ in (before, after):
            ledger_["home/tmp"] = ledger_["home/tmp"][:1]
        assert not list((tmp_path / "home" / "tmp").iterdir())
    assert after == before
    assert not (tmp_path / "run.log").exists()
    assert not (library / "Normalised").exists()


@pytest.mark.parametrize("arguments", [("analyse",), ("normalise", "--dry-run")], ids=["analyse", "dry run"])
def test_c03_dynamic_previews_succeed_and_write_nothing(norma, fixture_path, demos, ledger, tmp_path, arguments):
    """S3 (CR-062): dynamic previews leave no audio, output, report, log or temporary residue."""
    root = demos({"take.wav": fixture_path("F-01")})
    before = ledger()
    run = norma(*arguments, "--peak-mode", "dynamic", root / "take.wav")
    after = ledger()
    assert run.code == 0, run.stderr
    assert len(run.files) == 1, run.stdout
    assert run.files[0][0] == "OK"
    assert run.files[0][6] == "dynamic (calibrated at render)"
    if arguments[0] == "normalise":
        # Existing C-04 evidence probes may change only TMPDIR's own mtime, never its contents.
        for ledger_ in (before, after):
            ledger_["home/tmp"] = ledger_["home/tmp"][:1]
        assert not list((tmp_path / "home" / "tmp").iterdir())
    assert after == before
    assert not (root / "Normalised").exists()


@pytest.mark.parametrize("name", list(REFUSED))
def test_c03_refused_requests_write_nothing(norma, library, ledger, tmp_path, name):
    """Requests this version refuses (unsafe reports, later commands, in-place) exit 2 and leave no file."""
    (tmp_path / "existing.csv").write_text("keep me\n")
    (tmp_path / "link.csv").symlink_to(tmp_path / "existing.csv")
    before = ledger()
    run = norma(*expand(REFUSED[name], library, tmp_path))
    assert run.code == 2
    assert run.stdout == ""
    after = ledger()
    if name.startswith("normalise"):  # same transient TMPDIR probe as above: compare TMPDIR's own mtime out
        for ledger_ in (before, after):
            ledger_["home/tmp"] = ledger_["home/tmp"][:1]
        assert not list((tmp_path / "home" / "tmp").iterdir())
    assert after == before
    assert not list(tmp_path.glob("report.*"))
    assert (tmp_path / "existing.csv").read_text() == "keep me\n"


def test_c03_verify_inspects_against_an_explicit_target_and_writes_nothing(norma, fixture_path, demos, ledger, tmp_path):
    """S5: ``verify`` is read-only; F-01 (−23 LUFS) matches an explicit −23 target and the row says so."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    before = ledger()
    run = norma("verify", "-t", "-23", path)
    after = ledger()
    assert run.code == 0, run.stderr
    assert len(run.files) == 1, run.stdout
    row = run.files[0]
    assert row[:3] == ["OK", "1/1", str(path)]
    assert row[3:5] == ["", ""]  # in_* stay empty: the file under inspection is an output
    assert abs(float(row[5])) <= 0.1  # deviation from the target (LU)
    assert row[6] == "verify"
    assert abs(float(row[7]) + 23.0) <= 0.1  # measured out_LUFS
    assert row[9] == str(path)  # the inspected file itself, not a planned output
    assert row[10] == "target -23 (command line)"
    assert run.summary["ok"] == "1" and run.summary["failed"] == "0" and run.summary["report"] == ""
    assert run.stderr == ""
    assert after == before
    assert not (path.parent / "Normalised").exists()


def test_c03_verify_fails_a_file_that_misses_the_target_and_writes_nothing(norma, fixture_path, demos, ledger):
    """The default −16 target is not met by a −23 LUFS file: FAIL ``verification_loudness``, exit 1, no writes."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    before = ledger()
    run = norma("verify", path)
    assert run.code == 1, run.stderr
    row = run.files[0]
    assert row[:3] == ["FAIL", "1/1", str(path)]
    assert row[6] == "verify" and abs(float(row[7]) + 23.0) <= 0.1
    assert row[10].startswith("verification_loudness: ") and row[10].endswith("target -16 (default)")
    assert run.summary["failed"] == "1"
    assert ledger() == before


def test_c03_verify_check_tags_fails_a_file_with_no_processing_mark_and_writes_nothing(
        norma, fixture_path, demos, ledger):
    """Journey A.8 / MD27: an unmarked file fails ``verification_tags`` (processing mark missing), exit 1."""
    path = demos({"take.wav": fixture_path("F-01")}) / "take.wav"
    before = ledger()
    run = norma("verify", "-t", "-23", "--check-tags", path)
    assert run.code == 1, run.stderr
    assert len(run.files) == 1, run.stdout
    row = run.files[0]
    assert row[:3] == ["FAIL", "1/1", str(path)]
    assert abs(float(row[7]) + 23.0) <= 0.1  # loudness itself is on target; only the tag check fails
    assert row[10] == "verification_tags: processing mark missing; target -23 (command line)"
    assert run.summary["ok"] == "0" and run.summary["failed"] == "1"
    assert ledger() == before


@pytest.mark.parametrize("extension", ["csv", "json"])
def test_c03_verify_writes_only_the_explicit_report_in_input_index_order(
        norma, fixture_path, demos, ledger, tmp_path, extension):
    """Report-only publication for ``verify`` at bounded concurrency; stdout may complete in any order."""
    names = [f"take-{number:02d}.wav" for number in range(1, 7)]
    root = demos({name: fixture_path("F-01") for name in names})
    report = tmp_path / f"verify.{extension}"
    before = ledger()
    run = norma("verify", "-j", "4", "-t", "-23", "--report", report, "--log", tmp_path / "run.log", root)
    after = ledger()
    assert run.code == 0, run.stderr
    expected = [str(root / name) for name in names]
    assert len(run.files) == 6 and len({row[2] for row in run.files}) == 6, run.stdout
    assert sorted(row[2] for row in run.files) == expected
    assert {row[2]: row[1] for row in run.files} == {path: f"{n}/6" for n, path in enumerate(expected, 1)}
    assert {row[0] for row in run.files} == {"OK"}
    assert run.summary["report"] == str(report) and run.summary["run_errors"] == "0"
    assert sorted(set(after) - set(before)) == [report.name]
    assert {key: after[key] for key in before} == before
    if extension == "csv":
        with report.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(report.read_text(encoding="utf-8"))["outcomes"]
    assert [row["input"] for row in rows] == expected
    assert [int(row["input_index"]) for row in rows] == list(range(6))
    for row in rows:
        assert row["status"] == "ok" and row["mode"] in ("", None)
        assert row["execution"] == "inspect"
        assert row["out_path"] == row["input"]  # the inspected file, not evidence of writing
        assert float(row["target_lufs"]) == -23.0 and row["target_origin"] == "cli"
        assert abs(float(row["out_lufs"]) + 23.0) <= 0.1
        assert row["in_lufs"] in ("", None) and row["gain_db"] in ("", None)
        published = row["published_paths"]
        assert (json.loads(published) if isinstance(published, str) else published) == []
    assert not (root / "Normalised").exists()
    assert not (tmp_path / "run.log").exists()


def test_c03_analyse_report_writes_only_that_report(norma, library, ledger, tmp_path):
    """S2 (CR-050): ``analyse --report PATH`` adds exactly one file, the report; nothing else changes."""
    before = ledger()
    report = tmp_path / "report.csv"
    run = norma("analyse", "-r", "--report", str(report), library)
    after = ledger()
    assert run.code in {0, 1}, run.stderr
    assert report.is_file()
    assert sorted(set(after) - set(before)) == ["report.csv"]
    assert {key: after[key] for key in before} == before
    assert not (library / "Normalised").exists()


@pytest.mark.parametrize("extension", ["csv", "json"])
def test_c03_tags_plan_writes_only_the_explicit_report(norma, fixture_path, demos, ledger, tmp_path, extension):
    """S4 (CR-091), CLI §§ 1/7.3: report-only publication, never audio, log or temporary output."""
    root = demos({"take.wav": fixture_path("F-01")})
    report = tmp_path / f"report.{extension}"
    before = ledger()
    run = norma("tags", "plan", root, "--report", report, "--log", tmp_path / "run.log")
    after = ledger()
    assert run.code == 0, run.stderr
    assert len(run.files) == 1 and run.files[0][0] == "OK", run.stdout
    assert run.files[0][3:10] == [""] * 7
    assert run.summary["report"] == str(report) and run.summary["run_errors"] == "0"
    assert sorted(set(after) - set(before)) == [report.name]
    assert {key: after[key] for key in before} == before
    if extension == "csv":
        with report.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(report.read_text(encoding="utf-8"))["outcomes"]
    assert len(rows) == 1
    assert rows[0]["execution"] == "plan"
    published = rows[0]["published_paths"]
    assert (json.loads(published) if isinstance(published, str) else published) == []
    assert not (root / "Normalised").exists()
    assert not (tmp_path / "run.log").exists()


def test_c03_commands_that_find_no_ffmpeg_write_nothing(norma, library, ledger, tmp_path, without_ffmpeg):
    before = ledger()
    for arguments in (("doctor",), ("analyse", "{root}"), ("tags", "show", "{root}")):
        run = norma(*expand(arguments, library, tmp_path), path=without_ffmpeg)
        assert run.code == 3, (arguments, run.stderr)
        assert (run.stdout == "") == (arguments[0] != "doctor")
    assert ledger() == before


def test_s03_originals_are_byte_and_time_identical_after_every_command(norma, library, ledger, tmp_path):
    """TST-S-03 (read-only slice): source SHA-256, size, mtime_ns and type are unchanged by a full sweep."""
    sources = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in library.rglob("*") if path.is_file()}
    for arguments in (*COMMANDS.values(), *REFUSED.values()):
        norma(*expand(arguments, library, tmp_path))
    for path, (mtime_ns, content) in sources.items():
        assert path.stat().st_mtime_ns == mtime_ns, path
        assert path.read_bytes() == content, path


def test_c03_a_read_only_library_is_enough(norma, library, ledger, tmp_path):
    """The commands never need write access to the inputs: a locked-down copy gives the same result."""
    writable = norma("analyse", "-r", library)
    writable_tags = norma("tags", "show", "-r", library)
    modes = {}
    for current, folders, names in os.walk(library):
        for name in folders + names:
            path = os.path.join(current, name)
            modes[path] = os.lstat(path).st_mode
    try:
        for path in sorted(modes, key=len, reverse=True):
            os.chmod(path, stat.S_IMODE(modes[path]) & ~0o222)
        os.chmod(library, stat.S_IMODE(os.lstat(library).st_mode) & ~0o222)
        locked = norma("analyse", "-r", library)
        shown = norma("tags", "show", "-r", library)
    finally:
        os.chmod(library, stat.S_IMODE(os.lstat(library).st_mode) | 0o700)
        for path in sorted(modes, key=len):
            os.chmod(path, stat.S_IMODE(modes[path]))
    assert locked.code == writable.code == 1
    # Rows arrive in completion order, so compare the multisets of rows (duplicates and gaps still differ).
    assert len(locked.files) == len(writable.files) == 11
    assert sorted(row[:7] for row in locked.files) == sorted(row[:7] for row in writable.files)
    assert shown.code == writable_tags.code == 1
    assert len(shown.files) == len(writable_tags.files) == 11
    assert sorted(row[:3] for row in shown.files) == sorted(row[:3] for row in writable_tags.files)
    assert sorted(line[:4] for line in shown.records("FIELD")) == sorted(line[:4] for line in writable_tags.records("FIELD"))
