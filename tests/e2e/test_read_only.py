"""S2/S4 acceptance, read-only modes, tagged inputs and depth rules (TST-C-03, D-066 lifted by CR-088, D-064).

The identity ledger of the whole sandbox (tests/e2e/conftest.py ``snapshot``) is the judge of what a
command wrote: it records every file, folder and link under the test's temporary folder, including
the redirected system temporary folder (``home/tmp``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import soundfile as sf
import numpy as np

TAGGED = ["wav24", "flac24", "aiff24", "alac24"]


def mixed_folder(tools, untagged, demos):
    return demos({"one.wav": untagged("wav24"), "two.flac": untagged("flac24"),
                  "three.aif": untagged("aiff24"), "four.m4a": untagged("alac24")}, folder="mixed")


# --- TST-C-03: nothing is written -----------------------------------------------------------


def test_c03_dry_run_writes_nothing_anywhere(norma, tools, demos, untagged, ledger, tmp_path):
    """No audio, folder, temporary file, backup, automatic report or log — and the system temp is empty.

    Capability-evidence encodes in a system temporary folder removed before return are allowed
    (D-059, D-067 (3)); the ledger of the redirected ``home/tmp`` proves it was removed.
    """
    root = mixed_folder(tools, untagged, demos)
    before = ledger()
    run = norma("normalise", root, "--dry-run", "-t", "-14", "--log", tmp_path / "run.log")
    assert run.code == 0, run.stderr
    assert len(run.files) == 4 and all(row[0] in {"OK", "WARN"} for row in run.files)
    assert run.summary["report"] == ""
    assert ledger() == before
    assert list((tmp_path / "home" / "tmp").iterdir()) == []
    assert not (tmp_path / "run.log").exists() and "--log ignored" in run.stderr


def test_c03_dry_run_prediction_matches_the_real_run(norma, tools, demos, untagged):
    root = mixed_folder(tools, untagged, demos)
    planned = norma("normalise", root, "-n", "-t", "-14")
    real = norma("normalise", root, "-t", "-14", "-o", root.parent / "out")
    assert planned.code == real.code == 0
    key = lambda run: sorted((Path(row[2]).name, row[0], row[5], row[6]) for row in run.files)  # noqa: E731
    assert key(planned) == key(real)


def test_c03_analyse_writes_nothing(norma, tools, demos, untagged, ledger):
    root = mixed_folder(tools, untagged, demos)
    before = ledger()
    run = norma("analyse", root)
    assert run.code == 0, run.stderr
    assert len(run.files) == 4 and ledger() == before
    assert list((root.parent / "home" / "tmp").iterdir()) == []


@pytest.mark.parametrize("command", [["analyse"], ["normalise", "--dry-run"]])
def test_c03_an_explicit_report_is_the_only_write(norma, tools, demos, untagged, ledger, tmp_path, command):
    root = mixed_folder(tools, untagged, demos)
    before = ledger()
    run = norma(*command, root, "--report", tmp_path / "report.csv")
    assert run.code == 0, run.stderr
    delta = tools.difference(before, ledger())
    assert delta == {"added": {"report.csv"}, "removed": set(), "changed": set()}
    assert len(tools.report(tmp_path / "report.csv")) == 4


def test_c03_a_dry_run_that_would_fail_exits_1_and_still_writes_nothing(norma, tools, demos, fixture_path, ledger):
    """A truncated file (F-10) still fails in a dry run — the tagged F-13 fixture no longer does (CR-088)."""
    root = demos({"truncated.wav": fixture_path("F-10")}, folder="truncated")
    before = ledger()
    run = norma("normalise", root, "--dry-run")
    assert run.code == 1 and [row[0] for row in run.files] == ["FAIL"]
    assert run.files[0][10].startswith("truncated: ")
    assert ledger() == before


# --- D-066 lifted (S4, CR-088): tagged inputs normalise with their tags carried --------------


def tag_view(run, path: Path) -> dict:
    """What ``tags show`` listed for one file: canonical fields, extras (without the mark), pictures, mark."""
    rows = [line for line in run.lines if len(line) == 4 and line[1] == str(path)]

    def kind(name):
        return {line[2]: json.loads(line[3]) for line in rows if line[0] == name}

    fields = {key: item["value"] for key, item in kind("FIELD").items()}
    extras = {key: item["value"] for key, item in kind("EXTRA").items() if key.upper() != "NORMA"}
    pictures = {key: (item["value"]["mime"], item["value"]["picture_type"]) for key, item in kind("ARTWORK").items()}
    return {"fields": fields, "extras": extras, "pictures": pictures, "mark": sorted(kind("MARK"))}


@pytest.mark.parametrize("variant", TAGGED)
@pytest.mark.parametrize("extra", [[], ["--force"]], ids=["plain", "force"])
def test_d066_lifted_tagged_input_is_normalised_with_every_tag_carried(
        norma, tools, demos, fixture_path, ledger, variant, extra):
    """The F-13 tags and artwork read back identically from the output; the source is untouched.

    Same-format outputs lose nothing (metadata spec 6): the row is OK with no note, and the output
    holds the same canonical fields, the extra and the picture as the source, plus a valid mark
    (B-002). ``--force`` changes nothing for a source that carries no mark.
    """
    source = fixture_path("F-13", variant)
    root = demos({source.name: source}, folder="tagged")
    before = ledger()
    run = norma("normalise", root, *extra)
    assert run.code == 0, run.stdout + run.stderr
    (row,) = run.files
    assert row[0] == "OK" and row[10] == "" and "metadata_loss" not in run.stdout
    output = Path(row[9])
    assert output == root / "Normalised" / source.name and output.is_file()
    delta = tools.difference(before, ledger())
    assert delta["removed"] == set() and delta["changed"] == set(), delta
    assert tools.audio_files(root) == sorted([root / source.name, output])

    shown_source = tag_view(norma("tags", "show", root / source.name), root / source.name)
    shown_output = tag_view(norma("tags", "show", output), output)
    assert shown_source["mark"] == ["absent"] and shown_output["mark"] == ["valid"]
    assert shown_source["fields"] and shown_source["extras"] == {"DEMO_NOTE": "seedless"}
    assert shown_output["fields"] == shown_source["fields"]
    assert shown_output["extras"] == shown_source["extras"]
    assert shown_output["pictures"] == shown_source["pictures"] == {"1": ("image/png", 3)}


@pytest.mark.parametrize("variant", TAGGED)
def test_d066_lifted_a_dry_run_plans_a_tagged_input_and_writes_nothing(norma, tools, demos, fixture_path, ledger, variant):
    source = fixture_path("F-13", variant)
    root = demos({source.name: source}, folder="tagged")
    before = ledger()
    run = norma("normalise", root, "--dry-run")
    assert run.code == 0, run.stdout + run.stderr
    (row,) = run.files
    assert row[0] == "OK" and row[10] == "" and row[9] == ""
    assert ledger() == before


def test_d066_lifted_a_tagged_file_is_processed_beside_the_untagged_files(
        norma, tools, demos, untagged, fixture_path, fixture_module):
    root = demos({"clean.wav": untagged("wav24"), "tagged.wav": fixture_path("F-13", "wav24")}, folder="pair")
    run = norma("normalise", root, "-o", root.parent / "out")
    assert run.code == 0, run.stdout + run.stderr
    assert {Path(row[2]).name: row[0] for row in run.files} == {"clean.wav": "OK", "tagged.wav": "OK"}
    assert [path.name for path in tools.audio_files(root.parent / "out")] == ["clean.wav", "tagged.wav"]
    clean, tagged = (tag_view(norma("tags", "show", root.parent / "out" / name), root.parent / "out" / name)
                     for name in ("clean.wav", "tagged.wav"))
    assert tagged["fields"]["title"] == fixture_module.TITLE and tagged["extras"] == {"DEMO_NOTE": "seedless"}
    assert tagged["pictures"] == {"1": ("image/png", 3)}
    assert clean["pictures"] == {} and clean["extras"] == {}  # nothing of the tagged file leaks across


# --- D-064: automatic depth ------------------------------------------------------------------


def eight_bit(tools, folder: Path) -> Path:
    path = tools.tone(folder / "eight.wav", level=-20, seconds=6)
    samples, rate = tools.decode(path)
    sf.write(path, samples, rate, subtype="PCM_U8")
    assert tools.facts(path)["codec_name"] == "pcm_u8"
    return path


def test_d064_eight_bit_is_rounded_up_to_sixteen_with_a_visible_note(norma, tools, tmp_path):
    source = eight_bit(tools, tmp_path / "in")
    report = tmp_path / "report.csv"
    run = norma("normalise", source, "-o", tmp_path / "out", "-t", "-16", "--report", report)
    assert run.code == 0, run.stderr
    (row,) = run.files
    assert row[0] == "WARN" and "format_adjusted" in row[10] and "8-bit to 16-bit" in row[10]
    (record,) = tools.report(report)
    assert (record["status"], record["reason"], record["out_bit_depth"]) == ("warn", "format_adjusted", "16")
    assert tools.facts(Path(row[9]))["codec_name"] == "pcm_s16le"
    samples, rate = tools.decode(Path(row[9]))
    assert abs(tools.pyloudnorm(samples, rate)[0] + 16) <= 0.5


def test_d064_an_explicit_bit_depth_overrides_the_automatic_one(norma, tools, tmp_path):
    source = eight_bit(tools, tmp_path / "in")
    run = norma("normalise", source, "-o", tmp_path / "out", "-t", "-16", "--bit-depth", "24")
    assert run.code == 0, run.stderr
    assert tools.facts(Path(run.files[0][9]))["codec_name"] == "pcm_s24le"


def test_d064_twenty_bit_pcm_reaches_the_product_as_24_bit_and_stays_24(norma, tools, tmp_path):
    """20-bit PCM cannot be exercised as 20-bit through this FFmpeg (N-125892).

    The WAV-extensible valid-bits field and a 20-bit AIFF are both reported by ffprobe as
    ``bits_per_raw_sample=24``, so the product sees an ordinary 24-bit file: it is written as 24-bit
    PCM with no depth note. The 20 → 24 rule itself is covered by the unit tests of the format
    resolver; this test pins what the command line can honestly show.
    """
    path = tools.tone(tmp_path / "in" / "twenty.wav", level=-20, seconds=6)
    samples, rate = tools.decode(path)
    sf.write(path, np.round(samples * 2 ** 19) / 2 ** 19, rate, subtype="PCM_24")  # 20 significant bits
    run = norma("normalise", path, "-o", tmp_path / "out", "-t", "-16")
    assert run.code == 0, run.stderr
    assert run.files[0][0] == "OK" and run.files[0][10] == ""
    assert tools.facts(Path(run.files[0][9]))["bits_per_raw_sample"] == "24"


# --- Regression for CR-051 (fixed by TK-0510) --------------------------------------------------------------


# macOS refuses a path of 1,024 bytes or more (``PC_PATH_MAX``). Norma's longest private name in the
# output folder is the publication recovery receipt, ``.norma-<run id>-<n>-receipt-<8 characters>.json``:
# 55 characters for a 25-character run id and a one-digit index. With the source folder at 957
# characters, ``<folder>/Normalised`` is 968 and the receipt path 1,024: the receipt cannot be written,
# so the file fails ``commit_error`` after planning, allocation, render and verification. The
# folder allocation, the 34-character report name and its private name all still fit. Three lengths
# (956 to 958) fail this way; 959 also defeats the report (run_errors 1), 955 succeeds
# (FFmpeg N-125892, the development Mac arm64, 2026-10-12).
DEEP_SOURCE_FOLDER = 957


def deep_folder(tmp_path: Path, length: int) -> str:
    """A folder name, relative to ``tmp_path``, that makes the folder's absolute path ``length`` long."""
    remaining = length - len(os.path.realpath(tmp_path))
    parts = []
    while remaining > 0:
        parts.append("d" * min(200, remaining - 1))  # one separator before each part
        remaining -= len(parts[-1]) + 1
    assert remaining == 0
    return "/".join(parts)


@pytest.mark.skipif(os.pathconf("/", "PC_PATH_MAX") != 1024, reason="needs the macOS path limit")
def test_cr_the_automatic_report_survives_a_file_that_fails_after_planning(norma, fixture_path, demos, tmp_path):
    """A failure S3 cannot correct: the file's private names no longer fit in the output folder (D-081).

    The original trigger, ``-t -40`` on F-03/endpoint, is now corrected by the S3 retry ladder.
    This one is the operating system's path limit, so no correction or retry applies; it is a
    per-file I/O problem found after preflight, which is FAIL and exit 1 (CLI § 8).
    """
    folder = deep_folder(tmp_path, DEEP_SOURCE_FOLDER)
    root = demos({"short.wav": fixture_path("F-03", "endpoint")}, folder=folder)
    assert len(os.path.realpath(root)) == DEEP_SOURCE_FOLDER
    run = norma("normalise", root, "-t", "-14")
    assert run.code == 1 and run.files[0][0] == "FAIL"
    assert "commit_error" in run.files[0][10] and "File name too long" in run.files[0][10]
    assert run.summary["run_errors"] == "0"
    report = Path(run.summary["report"])
    assert report.is_file() and report.parent == root / "Normalised"
    assert not (root / "Normalised" / "short.wav").exists()
