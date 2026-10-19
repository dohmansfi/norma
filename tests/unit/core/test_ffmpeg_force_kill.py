"""CR-045 (D-065): the FFmpeg adapter's public force-kill path for a repeated interrupt.

Fake tools are Python scripts that ignore SIGTERM, so only SIGKILL ends them: a call that
returns well inside the two-second grace period proves the kill was immediate.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from norma.adapters import ffmpeg
from norma.errors import Cancelled, ToolTimeout

STUBBORN = """
import signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = __import__("subprocess").Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
open(sys.argv[1], "w").write(f"{os.getpid()} {child.pid}")
sys.stdout.write("started\\n"); sys.stdout.flush()
time.sleep(60)
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True  # a zombie still answers signal 0 until it is reaped


def wait_until(condition, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def start_call(argv: tuple[str, ...], **kwargs) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = ffmpeg.run_process(argv, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            box["error"] = exc
        box["ended"] = time.monotonic()

    thread = threading.Thread(target=target)
    thread.start()
    return thread, box


def pids(path: Path) -> tuple[int, int]:
    wait_until(lambda: path.is_file() and path.read_text().strip() != "")
    leader, child = path.read_text().split()
    return int(leader), int(child)


def test_force_kill_with_nothing_running_signals_nothing() -> None:
    assert ffmpeg._ACTIVE == {}
    assert ffmpeg.force_kill() == 0


def test_force_kill_ends_a_running_group_at_once_and_the_call_is_cancelled(make_tool, tmp_path: Path) -> None:
    tool = make_tool("stubborn", STUBBORN)
    marker = tmp_path / "pids"
    thread, box = start_call((str(tool), str(marker)), deadline_s=60.0)
    leader, child = pids(marker)
    wait_until(lambda: len(ffmpeg._ACTIVE) == 1)
    forced_at = time.monotonic()
    assert ffmpeg.force_kill() == 1
    thread.join(10)
    assert not thread.is_alive()
    assert isinstance(box.get("error"), Cancelled)
    assert box["ended"] - forced_at < ffmpeg.TERMINATE_GRACE_S  # no SIGTERM grace was waited
    wait_until(lambda: not alive(child))
    assert ffmpeg._ACTIVE == {}
    del leader


def test_force_kill_cuts_a_graceful_teardown_short(make_tool, tmp_path: Path) -> None:
    tool = make_tool("stubborn", STUBBORN)
    marker = tmp_path / "pids"
    cancel = threading.Event()
    thread, box = start_call((str(tool), str(marker)), deadline_s=60.0, cancel_event=cancel)
    _leader, child = pids(marker)
    cancel.set()  # first interrupt: SIGTERM, which this group ignores, then a two-second grace
    began = time.monotonic()
    time.sleep(0.3)
    assert thread.is_alive(), "the graceful teardown should still be waiting"
    ffmpeg.force_kill()  # second interrupt
    thread.join(10)
    assert isinstance(box.get("error"), Cancelled)
    assert box["ended"] - began < ffmpeg.TERMINATE_GRACE_S
    wait_until(lambda: not alive(child))


def test_force_during_a_timeout_teardown_is_an_interruption(make_tool, tmp_path: Path) -> None:
    tool = make_tool("stubborn", STUBBORN)
    marker = tmp_path / "pids"
    thread, box = start_call((str(tool), str(marker)), deadline_s=0.5)
    pids(marker)
    time.sleep(0.8)  # the deadline has expired; teardown is in its grace period
    assert thread.is_alive()
    ffmpeg.force_kill()
    thread.join(10)
    assert isinstance(box.get("error"), Cancelled) and not isinstance(box.get("error"), ToolTimeout)


def test_force_kill_is_one_shot(make_tool, tmp_path: Path) -> None:
    ffmpeg.force_kill()
    tool = make_tool("quick", "print('done')")
    result = ffmpeg.run_process((str(tool),), deadline_s=10.0)
    assert result.returncode == 0 and result.stdout == b"done\n"


def test_force_kill_only_reaches_calls_in_progress(make_tool, tmp_path: Path) -> None:
    tool = make_tool("stubborn", STUBBORN)
    first, second = tmp_path / "one", tmp_path / "two"
    calls = [start_call((str(tool), str(marker)), deadline_s=60.0) for marker in (first, second)]
    pids(first), pids(second)
    wait_until(lambda: len(ffmpeg._ACTIVE) == 2)
    assert ffmpeg.force_kill() == 2
    for thread, box in calls:
        thread.join(10)
        assert isinstance(box.get("error"), Cancelled)
    assert ffmpeg._ACTIVE == {}


@pytest.mark.parametrize("name", ["_Stream", "_Supervisor"])
def test_supervision_code_stays_mirrored(name: str) -> None:
    """The force path lives outside ``_Supervisor``, which ffprobe.py mirrors byte for byte."""
    import inspect

    from norma.adapters import ffprobe

    assert inspect.getsource(getattr(ffmpeg, name)) == inspect.getsource(getattr(ffprobe, name))
