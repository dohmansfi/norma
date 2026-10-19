"""S5 acceptance of the real ``verify`` command (TST-A-08, C-04 to C-08, B-002/B-003; TK-0719).

The unit suites (``tests/unit/cli/test_s5_*.py``) already pin option applicability, executor
hand-off and the verify rows with injected tools, and ``test_s5_real.py`` runs a single demo
in-process. This module adds what only the child-process command can show on a batch:

- good and bad outputs made by the real ``normalise``, judged in the documented precedence,
  with the exit rule (any FAIL is exit 1) and the identity ledger of the whole sandbox;
- the 53 C-09 columns, CSV and JSON equal field by field, rows in input order with the natural
  ``input_index`` while stdout is a completion-order stream of ``n/N`` rows (never sorted here);
- hostile file names through the tab-separated stream, the CSV and the JSON;
- ``-q`` and ``-v``, the exit codes 0, 1 and 2, and config/CLI precedence for the real command;
- the network guard in two layers: every FFmpeg/ffprobe argv is audited from a logging shim
  (``-protocol_whitelist file``, absolute local paths only, no shell expansion) and the Python
  child runs with ``socket`` connect refused by a ``sitecustomize`` guard;
- the closed B-002 (processing mark: verify target and ``already_processed`` rerun) and B-003
  (tagged input carries its tags and ``--check-tags`` confirms them) behaviours.

Not claimed: the whole C-13 taxonomy (45 active entries) and every option matrix. Those stay with
the unit suites; the trace is kept in the test plan.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mutagen.flac import FLAC

COLUMNS = (
    "input status reason out_path out_format out_bit_depth out_rate in_lufs in_dbtp in_lra gain_db mode "
    "limited_db out_lufs out_dbtp tag_sources conflicts dropped sidecar elapsed_s execution input_index "
    "in_signal_state out_signal_state target_lufs ceiling_dbtp expected_lufs predicted_shortfall_lu "
    "measured_shortfall_lu loudness_corrections peak_corrections reduce_fallbacks limiter_threshold_dbtp "
    "effective_limiter_bound_db requested_bitrate_bps effective_bitrate_bps bitrate_mode vbr_quality "
    "encoder ffmpeg_version warnings retained_failed_path published_paths in_peak_resolution_linear "
    "out_peak_resolution_linear target_origin ceiling_origin tolerance_lu tolerance_origin mark_validity "
    "source_codec origin_codec conditional_plans").split()
RUN_SPECIFIC = {"elapsed_s"}


def tone_set(tools, folder: Path, names, level: float = -30.0) -> list[Path]:
    return [tools.tone(folder / name, level=level + index, seconds=3, frequency=440 + 70 * index)
            for index, name in enumerate(names)]


def level_of(norma, path: Path, tmp_path: Path) -> float:
    """The file's own integrated loudness, to 0.1 LU, from the real ``analyse`` (a verify target that passes)."""
    report = tmp_path / f"level-{path.stem}.json"
    assert norma("analyse", path, "--report", report).code == 0
    return round(json.loads(report.read_text())["outcomes"][0]["in_lufs"], 1)


def normalised(norma, sources, destination: Path, *options: object) -> list[Path]:
    run = norma("normalise", *sources, "-o", destination, *options)
    assert run.code == 0, run.stdout + run.stderr
    return [Path(row[9]) for row in run.files]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_csv_equals_json(cells: dict[str, str], typed: dict) -> None:
    assert list(cells) == COLUMNS and list(typed) == COLUMNS
    for column in COLUMNS:
        if column in RUN_SPECIFIC:
            continue
        cell, value = cells[column], typed[column]
        if value is None:
            assert cell == "", (column, cell)
        elif isinstance(value, bool):
            assert cell.lower() == str(value).lower(), (column, cell, value)
        elif isinstance(value, (list, dict)):
            assert json.loads(cell) == value, (column, cell, value)
        elif isinstance(value, (int, float)):
            assert float(cell) == pytest.approx(value, rel=1e-12, abs=1e-12), (column, cell, value)
        else:
            assert cell == value, (column, cell, value)


# --- TST-A-08: good and bad outputs, precedence, exit rule, no writes ---------------------------


def test_a08_batch_of_good_and_bad_outputs_exit_rule_precedence_and_no_writes(norma, tools, tmp_path, ledger):
    sources = tone_set(tools, tmp_path / "src", ["a.wav", "b.wav", "c.wav", "d.wav"])
    good_a, good_b = normalised(norma, sources[:2], tmp_path / "good", "-t", "-16")
    (loud,) = normalised(norma, sources[2:3], tmp_path / "other", "-t", "-12")
    (unmarked,) = normalised(norma, sources[3:4], tmp_path / "bare", "-t", "-16", "--no-mark-processed")
    before = ledger()

    run = norma("verify", good_a, good_b, loud, unmarked, "-t", "-16", "--check-tags")
    assert run.code == 1, run.stdout + run.stderr
    assert len(run.files) == 4
    assert {row[2] for row in run.files} == {str(good_a), str(good_b), str(loud), str(unmarked)}
    status = {Path(row[2]).name: (row[0], row[-1]) for row in run.files}
    assert status["a.wav"][0] == status["b.wav"][0] == "OK"
    assert status["c.wav"][0] == "FAIL" and "verification_loudness" in status["c.wav"][1]
    assert status["d.wav"][0] == "FAIL" and "processing mark missing" in status["d.wav"][1]
    assert "(command line)" in status["a.wav"][1]  # an explicit -t outranks the mark
    assert run.summary["failed"] == "2" and run.summary["ok"] == "2" and run.summary["report"] == ""
    assert ledger() == before

    # Without -t each file's own valid mark supplies its target: the -12 output now passes (B-002).
    run = norma("verify", good_a, loud)
    assert run.code == 0 and {row[0] for row in run.files} == {"OK"}
    assert all("(processing mark)" in row[-1] for row in run.files)
    # No tag check: an output without a mark is judged on loudness alone.
    run = norma("verify", unmarked, "-t", "-16")
    assert run.code == 0 and run.files[0][0] == "OK"
    # The wrong target is a loudness FAIL with the measured shortfall, never a repair.
    run = norma("verify", good_a, "-t", "-20")
    assert run.code == 1 and "verification_loudness" in run.files[0][-1]
    assert ledger() == before


# --- B-002 / B-003: the closed behaviours, once more through verify -----------------------------


def test_b002_b003_tags_and_mark_are_carried_and_verify_confirms_them(norma, tools, tmp_path, ledger):
    source = tools.tone(tmp_path / "src" / "song.flac", level=-26, seconds=3, frequency=500)
    tagged = FLAC(source)
    tagged.update(title="Tagged Song", artist="Ana Ünï", genre="Dub", date="2019-04-07", tracknumber="3")
    tagged.save()
    (output,) = normalised(norma, [source], tmp_path / "out", "-t", "-16")
    before = ledger()
    # B-003: the tags of the tagged input are in the output and every explicit reference matches.
    run = norma("verify", output, "--check-tags", "--title", "Tagged Song", "--artist", "Ana Ünï",
                "--genre", "Dub", "--date", "2019-04-07", "--track", "3")
    assert run.code == 0 and run.files[0][0] == "OK", run.stdout + run.stderr
    run = norma("verify", output, "--check-tags", "--title", "Other")
    assert run.code == 1 and "verification_tags" in run.files[0][-1]
    assert 'expected "Other" from the command line (--title), found "Tagged Song"' in run.files[0][-1]
    assert ledger() == before
    # B-002: the output is marked; running normalise on it again skips it without --force.
    rerun = norma("normalise", output, "-o", tmp_path / "again", "-t", "-16")
    assert rerun.code == 0 and rerun.files[0][0] == "SKIP" and "already_processed" in rerun.files[0][-1]
    assert tools.audio_files(tmp_path / "again") == []  # nothing but the automatic report may appear there


# --- TST-C-04: report columns, typed equivalence, ordering, no hidden rows -----------------------


def test_c04_report_53_columns_csv_json_equivalent_and_input_ordered_under_parallel_jobs(
        norma, tools, tmp_path, ledger):
    names = ["take-1.wav", "take-2.wav", "take-10.wav", "take-11.wav", "take-3.wav", "take-20.wav",
             "take-21.wav", "take-4.wav", "take-5.wav", "take-6.wav", "take-7.wav", "take-8.wav"]
    folder = tmp_path / "demos"
    for index, name in enumerate(names):  # distinct levels so no two rows are alike
        tools.tone(folder / name, level=-30 + (index % 7) * 2, seconds=2, frequency=300 + 40 * index)
    before = ledger()
    runs = {}
    for kind in ("csv", "json"):
        runs[kind] = norma("verify", folder, "-t", "-16", "-j", "4", "--report", tmp_path / f"report.{kind}")
        assert runs[kind].code == 1  # tones not at -16 fail loudness: an honest exit, not a crash
    after = ledger()
    assert set(after) - set(before) == {"report.csv", "report.json"}
    assert {name for name in before if before[name] != after.get(name)} == set()

    json_document = json.loads((tmp_path / "report.json").read_text())
    typed = json_document["outcomes"]
    cells = csv_rows(tmp_path / "report.csv")
    assert len(typed) == len(cells) == len(names) == int(runs["json"].summary["failed"])
    for csv_row, json_row in zip(cells, typed, strict=True):
        assert_csv_equals_json(csv_row, json_row)

    # Natural order and input identity: the report row k is the stdout row "k+1/12", whatever order
    # stdout completed in; each index appears exactly once on stdout and in each report.
    expected_order = ["take-1.wav", "take-2.wav", "take-3.wav", "take-4.wav", "take-5.wav", "take-6.wav",
                      "take-7.wav", "take-8.wav", "take-10.wav", "take-11.wav", "take-20.wav", "take-21.wav"]
    for kind, rows in (("csv", cells), ("json", typed)):
        assert [Path(row["input"]).name for row in rows] == expected_order, kind
        assert [int(row["input_index"]) for row in rows] == list(range(len(names))), kind
        stream = runs[kind].files
        assert sorted(int(row[1].split("/")[0]) for row in stream) == list(range(1, len(names) + 1))
        assert {row[1].split("/")[1] for row in stream} == {str(len(names))}
        by_index = {int(row[1].split("/")[0]) - 1: row[2] for row in stream}
        assert [by_index[index] for index in range(len(names))] == [row["input"] for row in rows]
        assert all(row["execution"] == "inspect" and row["published_paths"] in ("[]", []) for row in rows)
        assert all(row["reason"] == "verification_loudness" and row["status"] == "failed" for row in rows)
    assert json_document["counts"] == {"ok": 0, "warn": 0, "skipped": 0, "failed": len(names)}
    assert json_document["not_started"] == 0 and json_document["report_path"] == str(tmp_path / "report.json")
    assert json_document["options"]["command"] == "verify" and json_document["options"]["jobs"] == 4
    assert json_document["options"]["value_sources"]["jobs"] == "cli"
    assert runs["csv"].summary["report"] == str(tmp_path / "report.csv")


def test_c04_report_refusals_leave_the_sandbox_untouched(norma, tools, tmp_path, ledger):
    (path,) = tone_set(tools, tmp_path / "demos", ["one.wav"])
    level = level_of(norma, path, tmp_path)
    (tmp_path / "taken.json").write_text("{}")
    before = ledger()
    for report in (tmp_path / "taken.json", tmp_path / "missing-folder" / "r.csv", tmp_path / "r.txt"):
        run = norma("verify", path, "-t", level, "--report", report)
        assert run.code == 2 and run.stdout == "" and run.stderr.startswith("norma: error:"), (report, run)
        assert ledger() == before
    # --overwrite acts only on an explicit report; it never changes an input.
    run = norma("verify", path, "-t", level, "--report", tmp_path / "taken.json", "--overwrite")
    assert run.code == 0 and json.loads((tmp_path / "taken.json").read_text())["outcomes"][0]["status"] == "ok"


# --- TST-C-08: escaping, quiet, verbose, exit rules ----------------------------------------------

HOSTILE = ["tab\there.wav", "new\nline.wav", "back\\slash.wav", 'quo"te,comma;semi.wav',
           "ünï–çode ñ.wav", "a space.wav", "$(touch pwned).wav", ";touch pwned;.wav",
           "concat:one.wav|two.wav", "http:%2F%2F127.0.0.1:9%2Fx.wav", "-i.wav", "~tilde.wav"]


def test_c08_hostile_names_escape_in_the_stream_and_round_trip_in_both_reports(norma, tools, tmp_path):
    folder = tmp_path / "hostile"
    base = tools.tone(tmp_path / "seed.wav", level=-16, seconds=2)
    folder.mkdir()
    for name in HOSTILE:
        shutil.copyfile(base, folder / name)
    reports = {kind: tmp_path / f"r.{kind}" for kind in ("csv", "json")}
    level = level_of(norma, base, tmp_path)
    run = norma("verify", folder, "-t", level, "--report", reports["csv"])
    assert run.code == 0, run.stdout + run.stderr
    other = norma("verify", folder, "-t", level, "--report", reports["json"])
    assert other.code == 0
    assert not (tmp_path / "pwned").exists() and not (folder / "pwned").exists()
    # The stream is strictly one record per line with escaped tab/CR/LF/backslash in cells.
    lines = run.stdout.split("\n")
    assert len(lines) == len(HOSTILE) + 2 and lines[-1] == ""  # rows + SUMMARY, no stray line from "\n"
    raw_cells = [line.split("\t") for line in lines[:-1]]
    assert all(len(cells) == 11 for cells in raw_cells[:-1]), raw_cells
    assert {unescaped[2] for unescaped in run.files} == {str(folder / name) for name in HOSTILE}
    tabbed = [cells[2] for cells in raw_cells if "tab" in cells[2]]
    assert tabbed == [f"{folder}/tab\\there.wav"]
    # Both reports carry the exact names, unescaped, one row each and in natural input order.
    expected = sorted(str(folder / name) for name in HOSTILE)
    assert sorted(row["input"] for row in csv_rows(reports["csv"])) == expected
    assert sorted(row["input"] for row in json.loads(reports["json"].read_text())["outcomes"]) == expected
    assert [row["input_index"] for row in csv_rows(reports["csv"])] == [str(i) for i in range(len(HOSTILE))]
    assert len(csv_rows(reports["csv"])[0]) == 53
    # The CSV is also valid under a strict parse of its own text (no truncated multiline field).
    text = reports["csv"].read_text(encoding="utf-8")
    assert len(list(csv.reader(io.StringIO(text, newline="")))) == len(HOSTILE) + 1


def test_c08_quiet_hides_stderr_but_never_rows_summary_or_errors(norma, tools, tmp_path):
    source = tools.tone(tmp_path / "src" / "q.wav", level=-26, seconds=3)
    loud = norma("normalise", source.parent, "-o", tmp_path / "loud", "-t", "-16")
    quiet = norma("normalise", source.parent, "-o", tmp_path / "quiet", "-t", "-16", "-q")
    assert loud.code == quiet.code == 0
    assert "note:" in loud.stderr and quiet.stderr == ""
    assert [row[0] for row in quiet.files] == [row[0] for row in loud.files] and quiet.summary["ok"] == "1"
    # verify -q: failing rows, summary and exit 1 remain; a usage error still reaches stderr.
    failing = norma("verify", source, "-t", "-16", "-q")
    assert failing.code == 1 and failing.files[0][0] == "FAIL" and failing.summary["failed"] == "1"
    assert failing.stderr == ""
    bad = norma("verify", tmp_path / "nowhere", "-q")
    assert bad.code == 2 and bad.stdout == "" and "no such file or folder" in bad.stderr
    both = norma("verify", source, "-q", "-v")
    assert both.code == 2 and both.stdout == "" and "not allowed with argument" in both.stderr
    assert not (tmp_path / "nowhere").exists()


def test_c08_exit_codes_for_pass_fail_usage_and_missing_tool(norma, tools, tmp_path, ledger):
    (path,) = tone_set(tools, tmp_path / "demos", ["one.wav"], level=-16)
    level = level_of(norma, path, tmp_path)
    empty = tmp_path / "no-tools"
    empty.mkdir()
    before = ledger()
    assert norma("verify", path, "-t", level).code == 0
    assert norma("verify", path, "-t", level - 4).code == 1
    for arguments in (("verify",), ("verify", path, "--tolerance", "9"), ("verify", path, "--format", "mp3"),
                      ("verify", path, "--title", "x"), ("verify", path, "--manifest", "m.csv")):
        run = norma(*arguments)
        assert run.code == 2 and run.stdout == "", (arguments, run)
    environment = {**os.environ, "PATH": str(empty), "NO_COLOR": "1"}
    missing = subprocess.run([sys.executable, "-m", "norma", "verify", str(path)], capture_output=True,
                             text=True, cwd=tmp_path, env=environment, timeout=60, check=False)
    assert missing.returncode == 3 and missing.stdout == "" and "FFmpeg not found" in missing.stderr
    assert ledger() == before


# --- TST-C-06: configuration for the real command ------------------------------------------------


def test_c06_config_cli_default_precedence_and_rejections_for_verify(norma, tools, tmp_path, ledger):
    (path,) = tone_set(tools, tmp_path / "demos", ["one.wav"], level=-14)
    level = level_of(norma, path, tmp_path)
    config = tmp_path / "prefs.toml"
    config.write_text(f'target = {level}\ntolerance = 0.2\njobs = 2\nformat = "mp3"\nout_dir = "ignored"\n')
    report = tmp_path / "options.json"
    run = norma("verify", path, "--config", config, "-j", "1", "--report", report)
    assert run.code == 0 and run.files[0][0] == "OK", run.stdout + run.stderr
    options = json.loads(report.read_text())["options"]
    assert (options["target"], options["value_sources"]["target"]) == (level, "config")
    assert (options["tolerance"], options["value_sources"]["tolerance"]) == (0.2, "config")
    assert (options["jobs"], options["value_sources"]["jobs"]) == (1, "cli")
    assert options["true_peak"] == -1.0 and options["value_sources"]["true_peak"] == "default"
    assert options["format"] == "same"  # a non-applicable config value is silently ignored by verify
    assert f"target {level:g} (config)" in run.files[0][-1]
    # The command line beats the config.
    run = norma("verify", path, "--config", config, "-t", "-16")
    assert run.code == 1 and "(command line)" in run.files[0][-1]
    # Every rejected config exits 2 with no output and no write.
    cases = {'bogus = 1': "unknown config key", 'target = "loud"': "not valid", 'report = "x.csv"': "not a config option",
             'in_place = true': "not a config option", 'quiet = true': "not a config option",
             'jobs = 0': "not valid", 'tolerance = true': "not valid"}
    for body, message in cases.items():
        config.write_text(body + "\n")
        before = ledger()
        run = norma("verify", path, "--config", config)
        assert run.code == 2 and run.stdout == "" and message in run.stderr, (body, run.stderr)
        assert ledger() == before


# --- TST-C-05: no network, no shell, local files only --------------------------------------------

GUARD = """\
import os, socket
_log = os.environ["NORMA_TEST_NET_LOG"]
def _refuse(*args, **kwargs):
    with open(_log, "a") as handle:
        handle.write("network attempt\\n")
    raise OSError("network refused by the test guard")
for _name in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, _name, _refuse)
socket.create_connection = _refuse
socket.getaddrinfo = _refuse
"""


def logging_tools(folder: Path, log: Path) -> None:
    """ffmpeg/ffprobe shims that record one NUL-terminated argv file per call, then run the real tool."""
    folder.mkdir()
    log.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        real = shutil.which(name)
        assert real is not None
        shim = folder / name
        shim.write_text(
            "#!/bin/sh\n"
            f"record=$(mktemp {shlex.quote(str(log))}/call.XXXXXX)\n"
            f'{{ printf "%s\\0" "{name}"; for a in "$@"; do printf "%s\\0" "$a"; done; }} > "$record"\n'
            f'exec {shlex.quote(real)} "$@"\n')
        shim.chmod(0o755)


def argv_records(log: Path) -> list[list[str]]:
    """One argv per recorded call (calls run concurrently, so one file each; no interleaving)."""
    data = [path.read_bytes().decode("utf-8", "surrogateescape") for path in sorted(log.iterdir())]
    return [record.split("\0")[:-1] for record in data]


def test_c05_no_network_connection_and_every_tool_call_is_local_and_whitelisted(norma, tools, tmp_path, ledger):
    base = tools.tone(tmp_path / "seed.wav", level=-26, seconds=3)
    folder = tmp_path / "demos"
    folder.mkdir()
    names = ["http:%2F%2F127.0.0.1:9%2Fx.wav", "$(touch pwned).wav", "concat:one.wav|two.wav", "-i.wav",
             "file:colon.wav", "nested ; name.wav"]
    for name in names:
        shutil.copyfile(base, folder / name)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "files": {
        names[0]: {"title": "http://127.0.0.1:9/title", "comment": "ftp://example.invalid/x"}}}))
    guard, shims, net_log, call_log = tmp_path / "guard", tmp_path / "shims", tmp_path / "net.log", tmp_path / "calls"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(GUARD)
    logging_tools(shims, call_log)
    environment = {**os.environ, "NO_COLOR": "1", "NORMA_TEST_NET_LOG": str(net_log),
                   "PATH": f"{shims}{os.pathsep}{os.environ['PATH']}",
                   "PYTHONPATH": f"{guard}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                   "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9"}

    def run(*arguments: object) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "norma", *map(str, arguments)], capture_output=True,
                              text=True, cwd=tmp_path, env=environment, timeout=240, check=False)

    # Positive control: the guard really refuses a connection in a child, and logs it.
    control = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9), 1)"],
        capture_output=True, text=True, cwd=tmp_path, env=environment, timeout=30, check=False)
    assert control.returncode != 0 and "refused by the test guard" in control.stderr and net_log.read_text()
    net_log.unlink()
    before = ledger()
    runs = [run("analyse", folder),
            run("verify", folder, "-t", "-26", "--check-tags", "--manifest", manifest),
            run("normalise", folder, "-o", tmp_path / "out", "-t", "-16", "--manifest", manifest),
            run("tags", "show", folder / names[0]),
            run("normalise", folder, "--artwork", "http://127.0.0.1:9/a.png", "--dry-run"),
            run("normalise", folder, "--artwork", "file:///nonexistent.png", "--dry-run")]
    assert [item.returncode for item in runs[:4]] == [0, 1, 0, 0], [(item.returncode, item.stderr) for item in runs]
    assert runs[4].returncode == 2 and "not a local path" in runs[4].stderr
    assert runs[5].returncode == 2
    assert "Traceback" not in "".join(item.stderr for item in runs)
    # The URL in the manifest is data: it is compared, never fetched; the mismatch is reported as text.
    assert "http://127.0.0.1:9/title" in runs[1].stdout.replace("\\", "") or "expected" in runs[1].stdout
    assert not net_log.exists(), net_log.read_text()
    assert not (tmp_path / "pwned").exists() and not (folder / "pwned").exists()
    records = argv_records(call_log)
    assert records, "the audit saw no tool call"
    probes = 0
    for record in records:
        arguments = record[1:]
        # A URL may appear only as the VALUE of a -metadata tag (manifest text written as data), never as an input.
        for position, argument in enumerate(arguments):
            if "://" in argument:
                assert arguments[position - 1] == "-metadata" and argument.split("=", 1)[0] in ("title", "comment"), \
                    repr(record)
        previous = 0
        for index, argument in enumerate(arguments):
            if argument != "-i":
                continue
            probes += 1
            # Each input is preceded, since the previous input, by its own file-only whitelist.
            window = arguments[previous:index]
            assert "-protocol_whitelist" in window, repr(record)
            assert window[window.index("-protocol_whitelist") + 1] == "file", repr(record)
            previous = index + 2
            if arguments[index - 2:index] == ["-f", "lavfi"]:  # a generated capability-evidence signal, no file
                assert arguments[index + 1].startswith("sine="), repr(record)
                continue
            assert len(arguments) > index + 1 and arguments[index + 1].startswith("/"), repr(record)
    assert probes >= len(names) * 3
    assert len(tools.audio_files(tmp_path / "out")) == len(names)
    assert not [name for name in set(ledger()) - set(before) if Path(name).name == "pwned"]
