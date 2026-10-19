"""Process-group lifecycle shared by both runners (IC-0 finding 1; safety § 6, ADR-0018).

Each case runs against the FFmpeg runner and ffprobe's private runner, so the
two stay behaviourally identical. Fake tools are Python scripts under tmp_path;
every descendant they start is recorded in a pid file and killed in teardown
if a test fails, so no fixture outlives its test.
"""

from __future__ import annotations

import inspect
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.adapters import ffprobe as p
from norma.errors import Cancelled, ToolTimeout

PY = sys.executable
GRACE = f.TERMINATE_GRACE_S
SLACK_S = 1.5
"""Allowed scheduling overhead on top of deadline + grace."""

IGNORE_TERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


Runner = Callable[..., tuple[int, bytes]]


def run_ffmpeg(argv: tuple[str, ...], deadline_s: float, cancel: threading.Event | None = None) -> tuple[int, bytes]:
    result = f.run_process(argv, deadline_s=deadline_s, cancel_event=cancel)
    return result.returncode, result.stdout


def run_ffprobe(argv: tuple[str, ...], deadline_s: float, cancel: threading.Event | None = None) -> tuple[int, bytes]:
    returncode, stdout, _tail = p._run(argv, deadline_s, cancel)
    return returncode, stdout


RUNNERS = {"ffmpeg": (run_ffmpeg, ToolTimeout), "ffprobe": (run_ffprobe, p._Timeout)}


@pytest.fixture(params=sorted(RUNNERS))
def runner(request: pytest.FixtureRequest) -> tuple[Runner, type[BaseException]]:
    return RUNNERS[request.param]


@pytest.fixture
def leader(tmp_path: Path):
    """Write a leader script that starts one descendant and records its pid."""
    pidfile = tmp_path / "child.pid"

    def factory(child_code: str, *, child_pipes: str = "inherit", after: str = "time.sleep(0.08)") -> tuple[str, ...]:
        redirect = ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" if child_pipes == "devnull" else ""
        path = tmp_path / "leader.py"
        path.write_text(
            "import subprocess, sys, time\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]{redirect})\n"
            f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
            "print('leader exiting', flush=True)\n"
            f"{after}\n",
            encoding="utf-8",
        )
        return (PY, str(path))

    factory.pidfile = pidfile  # type: ignore[attr-defined]
    yield factory
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def child_pid(leader) -> int:
    return int(leader.pidfile.read_text())


def test_early_leader_exit_with_pipe_holding_child_times_out(runner, leader) -> None:
    """The IC-0 reproduction: success after 6.1 s with a live child is now a bounded timeout."""
    run, timeout_error = runner
    deadline = 0.2
    start = time.monotonic()
    with pytest.raises(timeout_error):
        run(leader(IGNORE_TERM), deadline)
    elapsed = time.monotonic() - start
    assert elapsed < deadline + GRACE + SLACK_S, elapsed
    assert not alive(child_pid(leader)), "pipe-holding descendant survived the deadline"


def test_cancellation_after_early_leader_exit_kills_the_child(runner, leader) -> None:
    """The ffprobe ``_kill_group`` variant: the group, not just the leader, is torn down."""
    run, _ = runner
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            run(leader(IGNORE_TERM), 30.0, cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - start < 0.3 + GRACE + SLACK_S
    assert not alive(child_pid(leader))


def test_finished_leader_does_not_return_while_a_child_still_writes(runner, leader) -> None:
    """Buffers are complete: output a child writes after the leader exits is kept."""
    run, _ = runner
    child = "import sys, time; time.sleep(0.4); print('child done', flush=True)"
    returncode, stdout = run(leader(child), 10.0)
    assert returncode == 0
    assert stdout == b"leader exiting\nchild done\n"
    assert not alive(child_pid(leader))


def test_straggler_without_pipes_is_terminated_before_returning(runner, leader) -> None:
    """A descendant that let go of both pipes is still a group member; no call returns while it lives."""
    run, _ = runner
    start = time.monotonic()
    returncode, stdout = run(leader(IGNORE_TERM, child_pipes="devnull"), 10.0)
    assert returncode == 0 and stdout == b"leader exiting\n"
    assert not alive(child_pid(leader))
    assert time.monotonic() - start < GRACE + SLACK_S + 0.5


def test_leader_blocked_on_a_child_is_bounded_by_the_deadline(runner, leader) -> None:
    run, timeout_error = runner
    start = time.monotonic()
    with pytest.raises(timeout_error):
        run(leader(IGNORE_TERM, after="child.wait()"), 0.5)
    assert time.monotonic() - start < 0.5 + GRACE + SLACK_S
    assert not alive(child_pid(leader))


def test_no_reader_threads_outlive_a_call(runner, leader) -> None:
    run, timeout_error = runner
    before = threading.active_count()
    run((PY, "-c", "print('x')"), 10.0)
    with pytest.raises(timeout_error):
        run(leader(IGNORE_TERM), 0.2)
    assert threading.active_count() == before


def test_both_adapters_share_one_supervision_implementation() -> None:
    """ADR-0018 rule 4 forbids adapter-to-adapter imports, so the code is mirrored; it must not drift."""
    for name in ("_Stream", "_Supervisor"):
        assert inspect.getsource(getattr(f, name)) == inspect.getsource(getattr(p, name)), name
    for name in ("TERMINATE_GRACE_S", "_READ_CHUNK", "_MAX_LINE_BYTES", "_SETTLE_POLL_S"):
        assert getattr(f, name) == getattr(p, name), name
    assert not hasattr(p, "_kill_group")


def test_unterminated_and_overlong_lines_are_bounded(tmp_path: Path) -> None:
    body = ("import sys\nsys.stderr.write('a' * (3 * 1024 * 1024 + 5))\nsys.stderr.write('\\nlast')\n"
            "sys.stdout.write('no newline')\n")
    path = tmp_path / "s.py"
    path.write_text(body, encoding="utf-8")
    lines: list[str] = []
    result = f.run_process((PY, str(path)), deadline_s=30, on_stdout_line=lines.append)
    assert result.stdout == b"no newline" and lines == ["no newline"]
    assert result.stderr_tail[-1] == "last"
    assert [len(line) for line in result.stderr_tail[:-1]] == [f._MAX_LINE_BYTES] * 3 + [5]
