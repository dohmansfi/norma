"""CR-047: probe escalation mirrors FFmpeg, including deadline teardown."""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path

import pytest

from norma.adapters import ffmpeg, ffprobe
from norma.errors import Cancelled


def wait_for(predicate) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("probe did not reach the expected state")
        time.sleep(0.01)


@pytest.mark.parametrize("teardown", ["running", "cancelled", "timeout"])
def test_force_kill_ends_hung_probe_in_under_two_seconds(make_tool, tmp_path: Path, teardown: str) -> None:
    marker = tmp_path / "ready"
    tool = make_tool("hung-probe", """
import signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], 'w').write('ready')
time.sleep(60)
""")
    cancel = threading.Event()
    results = {}

    def run() -> None:
        try:
            ffprobe._run((str(tool), str(marker)), 0.4 if teardown == "timeout" else 60, cancel)
        except BaseException as exc:
            results["error"] = exc
        finally:
            results["ended"] = time.monotonic()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        wait_for(lambda: marker.exists() and bool(ffprobe._ACTIVE))
        if teardown == "cancelled":
            cancel.set()
        if teardown != "running":
            time.sleep(0.6)  # enter the two-second graceful teardown
            assert thread.is_alive()
        started = time.monotonic()
        assert ffprobe.force_kill() == 1
        thread.join(5)
        assert not thread.is_alive()
        assert isinstance(results["error"], Cancelled)
        assert results["ended"] - started < 2
        assert ffprobe._ACTIVE == {}
    finally:
        ffprobe.force_kill()
        thread.join(5)


def test_probe_force_kill_is_one_shot(make_tool) -> None:
    assert ffprobe.force_kill() == 0
    tool = make_tool("quick-probe", "print('done')")
    code, output, _ = ffprobe._run((str(tool),), 5)
    assert code == 0 and output == b"done\n"
    assert ffprobe._ACTIVE == {}


def test_pre_cancelled_probe_never_starts(make_tool, tmp_path: Path) -> None:
    marker = tmp_path / "started"
    tool = make_tool("probe", f"open({str(marker)!r}, 'w').write('bad')")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        ffprobe._run((str(tool),), 5, cancel)
    assert not marker.exists()


@pytest.mark.parametrize("name", ["_Stream", "_Supervisor"])
def test_supervision_remains_source_identical(name: str) -> None:
    assert inspect.getsource(getattr(ffmpeg, name)) == inspect.getsource(getattr(ffprobe, name))


def test_public_probe_force_kill_reaps_leader_and_descendant(make_tool, tmp_path):
    from .test_ffmpeg_force_kill import alive, pids

    marker = tmp_path / "pids"
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    tool = make_tool("hung-probe", f"""
import signal, subprocess
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c',
    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
open({str(marker)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')
time.sleep(60)
""")
    info = tool.stat()
    prober = ffprobe.Prober(tool, (info.st_dev, info.st_ino), "N-test")
    box = {}

    def run():
        try:
            ffprobe.probe(prober, source)
        except BaseException as exc:
            box["error"] = exc
        finally:
            box["ended"] = time.monotonic()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        leader, child = pids(marker)
        wait_for(lambda: bool(ffprobe._ACTIVE))
        started = time.monotonic()
        assert ffprobe.force_kill() == 1
        thread.join(5)
        assert not thread.is_alive()
        assert isinstance(box.get("error"), Cancelled)
        assert box["ended"] - started < 2
        wait_for(lambda: not alive(leader) and not alive(child))
        assert ffprobe._ACTIVE == {}
    finally:
        ffprobe.force_kill()
        thread.join(5)
