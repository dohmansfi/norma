"""S5 acceptance, parallel SIGINT in every publication phase (TST-S-04, TK-0719).

The older ``test_safety.py::test_s04_*`` interrupts a ``-j 1`` run during rendering. These tests
add what it states as a limit: ``-j`` above 1, each phase between render and publication, and the
read-only commands. The real ``python -m norma`` runs as a child process.

Determinism without a product hook: a gate shim stands in for ``ffmpeg`` (``ffprobe`` is
forwarded unchanged). It runs the real tool for every call except the first call of the chosen
phase, where it writes a ``started.<pid>`` marker (argv inside) and blocks until the test
releases it. The test therefore knows, from the marker count and not from a sleep, that
exactly ``-j`` workers are inside the phase before it signals. Only the child the test started
is signalled; the shim processes are reaped by Norma (the test checks they are gone and
releases the gate only in cleanup). A bounded wait guards every observation.

What is asserted is the TST-S-04 contract: exit 130, one final summary, one outcome per
started file, the rest counted ``not_started``, only complete audio published, no temporary
candidate left, sources untouched, no surviving worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

JOBS = 3
FILES = 10
OUTCOME_STATUSES = {"OK", "WARN", "SKIP", "FAIL"}
DEADLINE = 60.0

# Shell conditions per phase. ``$last`` is the final argument (the output), ``$*`` the argv.
PHASES = {
    # the render of a file: an overwrite-enabled call writing into the output folder
    "render": '*" -y "*) case "$last" in "$OUT/"*) hold=1;; esac;;',
    # the candidate's own measurement before the verdict
    "candidate_measure": '*" -i $OUT/"*ebur128*) hold=1;;',
    # the content hash that proves the candidate decodes to the planned samples
    "candidate_hash": '*" -f hash "*) hold=1;;',
    # verify: the measurement of an inspected file (the null muxer)
    "verify_measure": '*" -f null -"*) hold=1;;',
    # normalise: the single capability-evidence encode, written outside the output folder
    "evidence": '*" -y "*) case "$last" in "$OUT/"*) ;; *) hold=1;; esac;;',
}


class Gate:
    """A PATH folder holding the gate shim and the marker folder."""

    def __init__(self, root: Path, phase: str, out: Path, *, stubborn: bool = False) -> None:
        self.markers = root / "markers"
        self.folder = root / "shim"
        self.markers.mkdir(parents=True)
        self.folder.mkdir()
        real = shutil.which("ffmpeg")
        probe = shutil.which("ffprobe")
        assert real is not None and probe is not None, "FFmpeg is required"
        shim = self.folder / "ffmpeg"
        shim.write_text(
            "#!/bin/sh\n"
            f"OUT={shlex.quote(str(out))}\n"
            'for last; do :; done\n'
            'hold=\n'
            'case " $* " in\n'
            f"{PHASES[phase]}\n"
            'esac\n'
            'if [ -n "$hold" ]; then\n'
            + (f"  trap 'printf acknowledged > {shlex.quote(str(self.markers))}/terminated.$$' TERM\n"
               if stubborn else "")
            + f'  printf "%s\\n" "$*" > {shlex.quote(str(self.markers))}/started.$$\n'
            f'  while [ ! -e {shlex.quote(str(self.markers))}/release ]; do sleep 0.02; done\n'
            'fi\n'
            f'exec {shlex.quote(real)} "$@"\n')
        shim.chmod(0o755)
        forward = self.folder / "ffprobe"
        forward.write_text(f'#!/bin/sh\nexec {shlex.quote(probe)} "$@"\n')
        forward.chmod(0o755)

    def held(self) -> list[Path]:
        return sorted(self.markers.glob("started.*"))

    def inputs(self) -> set[str]:
        """The ``-i`` argument of every held call."""
        found = set()
        for marker in self.held():
            words = marker.read_text().split(" -i ", 1)
            if len(words) == 2:
                found.add(words[1].split(" -map", 1)[0].strip())
        return found

    def pids(self) -> list[int]:
        return [int(marker.name.split(".", 1)[1]) for marker in self.held()]

    def acknowledged(self) -> list[Path]:
        """Markers of held calls that received and survived SIGTERM (``stubborn`` gates only)."""
        return sorted(self.markers.glob("terminated.*"))

    def release(self) -> None:
        (self.markers / "release").write_text("released by the test cleanup\n")


def wait_until(predicate, timeout: float, step: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


@pytest.fixture
def gated(tmp_path):
    """``start(phase, *argv)`` -> (process, gate); the child and every shim are reaped afterwards."""
    started: list[tuple[subprocess.Popen, Gate]] = []

    def start(phase: str, *arguments: object, out: Path | None = None, stubborn: bool = False,
              tmpdir: Path | None = None):
        gate = Gate(tmp_path / f"gate-{len(started)}", phase, out or tmp_path / "out", stubborn=stubborn)
        environment = dict(os.environ)
        environment["NO_COLOR"] = "1"
        environment["PATH"] = f"{gate.folder}{os.pathsep}{environment['PATH']}"
        if tmpdir is not None:
            environment["TMPDIR"] = str(tmpdir)
        process = subprocess.Popen(
            [sys.executable, "-m", "norma", *map(str, arguments)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=environment)
        started.append((process, gate))
        return process, gate

    yield start
    for process, gate in started:
        gate.release()
        if process.poll() is None:
            process.kill()
        process.communicate()


def demo_folder(tools, folder: Path, count: int = FILES) -> Path:
    for index in range(count):
        tools.tone(folder / f"take-{index:02d}.wav", level=-30 + index, seconds=3, frequency=400 + 50 * index)
    return folder


def parse(stdout: str):
    rows = [line.split("\t") for line in stdout.splitlines() if line.split("\t")[0] in OUTCOME_STATUSES]
    summaries = [line for line in stdout.splitlines() if line.startswith("SUMMARY\t")]
    assert len(summaries) == 1, stdout
    summary = dict(cell.split("=", 1) for cell in summaries[0].split("\t")[1:])
    return rows, summary


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # existence probe only; delivers no signal
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else: certainly not reaped
        return True
    return True


def assert_workers_gone(gate: Gate) -> None:
    assert wait_until(lambda: not any(pid_alive(pid) for pid in gate.pids()), 10), \
        f"a gated worker survived Norma: {gate.pids()}"


@pytest.mark.parametrize("phase", ["render", "candidate_measure", "candidate_hash"])
@pytest.mark.parametrize("interrupts", [1, 2])
def test_s04_parallel_normalise_interrupt_in_each_phase(gated, tools, tmp_path, phase, interrupts):
    root = demo_folder(tools, tmp_path / "demos")
    before = tools.snapshot(root)
    out = tmp_path / "out"
    process, gate = gated(phase, "normalise", root, "-o", out, "-t", "-16", "-j", JOBS, out=out)
    assert wait_until(lambda: len(gate.held()) >= JOBS or process.poll() is not None, DEADLINE), \
        "readiness: the workers never reached the phase"
    assert process.poll() is None and len(gate.held()) == JOBS, (phase, gate.held())
    for number in range(interrupts):
        process.send_signal(signal.SIGINT)
        if number + 1 < interrupts:
            time.sleep(0.2)
    stdout, stderr = process.communicate(timeout=DEADLINE)
    assert process.returncode == 130, (stdout, stderr)
    assert "Traceback" not in stderr, stderr
    rows, summary = parse(stdout)
    names = [Path(row[2]).name for row in rows]
    assert len(names) == len(set(names)), "a file reported twice"
    started = int(summary["ok"]) + int(summary["warn"]) + int(summary["skipped"]) + int(summary["failed"])
    assert started == len(rows) >= JOBS and int(summary["not_started"]) == FILES - len(rows) > 0
    if phase == "render":  # the held inputs are sources: each started file has its one row
        assert {Path(path).name for path in gate.inputs()} <= set(names), (gate.inputs(), names)
    # Only complete audio exists, and exactly the files reported as written.
    produced = tools.audio_files(out) if out.is_dir() else []
    written = {Path(row[9]).name for row in rows if row[0] in {"OK", "WARN"}}
    assert {path.name for path in produced} == written
    frames = tools.decode(root / "take-00.wav")[0].shape
    for path in produced:
        assert tools.decode(path)[0].shape == frames
    leftovers = [entry.name for entry in out.iterdir() if entry not in produced
                 and not entry.name.startswith("norma-report-")] if out.is_dir() else []
    assert leftovers == [], leftovers
    for row in rows:
        if row[0] not in {"OK", "WARN"}:
            assert row[0] == "SKIP" and "cancelled" in row[-1], row
            assert not (out / Path(row[2]).name).exists()
    assert tools.snapshot(root) == before
    assert_workers_gone(gate)


@pytest.mark.parametrize("report", ["csv", "json"])
def test_s04_parallel_verify_interrupt_writes_only_the_explicit_report(gated, tools, tmp_path, report):
    root = demo_folder(tools, tmp_path / "demos")
    before = tools.snapshot(tmp_path / "demos")
    target = tmp_path / f"interrupted.{report}"
    # Verify only measures; the measurement of the input is a call to the null muxer.
    process, gate = gated("verify_measure", "verify", root, "-t", "-16", "-j", JOBS, "--report", target)
    assert wait_until(lambda: len(gate.held()) >= JOBS or process.poll() is not None, DEADLINE)
    assert process.poll() is None and len(gate.held()) == JOBS
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=DEADLINE)
    assert process.returncode == 130, (stdout, stderr)
    assert "Traceback" not in stderr, stderr
    rows, summary = parse(stdout)
    assert int(summary["not_started"]) == FILES - len(rows) > 0
    assert len({row[2] for row in rows}) == len(rows)
    assert all(row[0] == "SKIP" for row in rows)
    assert tools.snapshot(tmp_path / "demos") == before
    # The explicit report is the only write: one cancelled row per started file, in input order.
    assert summary["report"] == str(target)
    if report == "json":
        document = json.loads(target.read_text())
        assert document["not_started"] == FILES - len(rows)
        written = document["outcomes"]
    else:
        written = tools.report(target)
    assert [row["reason"] for row in written] == ["cancelled"] * len(rows)
    indices = [int(row["input_index"]) for row in written]
    assert indices == sorted(set(indices)) and len(indices) == len(rows)
    assert {Path(row["input"]).name for row in written} == {Path(row[2]).name for row in rows}
    assert {path.name for path in tmp_path.iterdir() if not path.name.startswith("gate-")} - {"home"} == {
        "demos", target.name}
    assert not [path for path in tmp_path.rglob(".norma-*")]
    assert_workers_gone(gate)


# --- A defect found by the second interrupt in the capability-evidence phase -------------------


def fingerprints(root: Path) -> dict[str, tuple[str, int]]:
    """Content hash and mtime of every file under ``root``."""
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_s04_second_interrupt_during_capability_evidence_is_still_an_interruption(gated, tools, tmp_path):
    """CR-109 (repaired by TK-0435, D-113): the forced interrupt while evidence is collected.

    The held evidence encode traps SIGTERM and refuses to exit, so the second SIGINT is the only
    way out: the first SIGINT must reach it as SIGTERM (observed through its acknowledgement
    marker, not a sleep) before the second is sent, and only the forced kill can end it.
    """
    root = demo_folder(tools, tmp_path / "demos", count=3)
    before = fingerprints(root)
    out = tmp_path / "out"
    scratch = tmp_path / "evidence-tmp"
    scratch.mkdir()
    log = tmp_path / "run.log"
    process, gate = gated("evidence", "normalise", root, "-o", out, "-t", "-16", "-j", 1,
                          "--log", log, out=out, stubborn=True, tmpdir=scratch)
    assert wait_until(lambda: len(gate.held()) >= 1 or process.poll() is not None, DEADLINE)
    assert process.poll() is None and len(gate.held()) == 1
    (worker,) = gate.pids()
    process.send_signal(signal.SIGINT)
    assert wait_until(lambda: len(gate.acknowledged()) == 1, DEADLINE), \
        "the first interrupt never reached the evidence encode as SIGTERM"
    assert process.poll() is None, "the encode refuses SIGTERM, so the first interrupt cannot finish the run"
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=DEADLINE)
    assert process.returncode == 130, (process.returncode, stderr[-300:])
    assert "Traceback" not in stderr, stderr
    rows, summary = parse(stdout)
    assert rows == [] and summary["not_started"] == "3", (rows, summary)
    assert summary["ok"] == summary["warn"] == summary["skipped"] == summary["failed"] == "0", summary
    assert summary["report"] == "", summary
    assert fingerprints(root) == before  # bytes and mtimes of the sources
    assert not out.exists() and not log.exists()
    assert list(scratch.iterdir()) == [], list(scratch.iterdir())
    assert not list(tmp_path.rglob(".norma-*")) and not list(tmp_path.rglob("norma-report-*"))
    assert {path.name for path in tmp_path.iterdir() if not path.name.startswith("gate-")} - {"home"} == {
        "demos", "evidence-tmp"}
    assert wait_until(lambda: not pid_alive(worker), 10), "the encode that refused SIGTERM was not reaped"
