"""FFmpeg adapter runner: argv rules, deadlines, draining, cancellation, version/features.

Fake tools are small Python scripts; nothing here needs the real FFmpeg.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.errors import Cancelled, EnvironmentError_, ToolFailed, ToolMissing, ToolTimeout

PY = sys.executable


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_tool_deadline_policy() -> None:
    assert f.tool_deadline(None) == 3600.0
    assert f.tool_deadline(0.25) == 60.0
    assert f.tool_deadline(15.0) == 60.0
    assert f.tool_deadline(240.0) == 960.0
    assert f.PROBE_DEADLINE_S == 60.0 and f.TERMINATE_GRACE_S == 2.0
    with pytest.raises(ValueError):
        f.tool_deadline(-1.0)


def test_ffmpeg_argv_and_inputs() -> None:
    argv = f.ffmpeg_argv(Path("/usr/local/bin/ffmpeg"), "-i", "x")
    assert argv[:3] == ("/usr/local/bin/ffmpeg", "-nostdin", "-hide_banner")
    assert f.input_args(Path("/demo/a b;rm -rf.wav")) == (
        "-protocol_whitelist", "file", "-i", "/demo/a b;rm -rf.wav")
    with pytest.raises(ValueError):
        f.input_args(Path("relative.wav"))
    with pytest.raises(ValueError):
        f.input_args("http://example.com/a.wav")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        f.ffmpeg_argv(Path("/x"), 1)  # type: ignore[arg-type]


def test_run_process_rejects_bad_argv() -> None:
    with pytest.raises(ValueError):
        f.run_process(("python3", "-V"), deadline_s=5)
    with pytest.raises(TypeError):
        f.run_process((PY, 3), deadline_s=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        f.run_process((PY, "-V"), deadline_s=0)
    with pytest.raises(ToolMissing):
        f.run_process(("/nonexistent/ffmpeg",), deadline_s=5)


def test_run_process_captures_and_returns_exit_status(tmp_path: Path) -> None:
    body = "import sys\nprint('out-1'); print('out-2')\nprint('err', file=sys.stderr)\nsys.exit(3)\n"
    result = f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=30)
    assert result.returncode == 3
    assert result.stdout == b"out-1\nout-2\n"
    assert result.stderr_tail == ("err",)
    with pytest.raises(ToolFailed) as caught:
        f.check_result(result)
    assert caught.value.returncode == 3 and caught.value.stderr_tail == ("err",)


def test_large_stderr_is_drained_and_tail_bounded(tmp_path: Path) -> None:
    body = ("import sys\nfor i in range(200000):\n    sys.stderr.write(f'line {i} ' + 'x' * 40 + '\\n')\n"
            "sys.stdout.write('done\\n')\n")
    seen: list[str] = []
    result = f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=60,
                           on_stderr_line=lambda line: seen.append(line) if len(seen) < 3 else None)
    assert result.returncode == 0 and result.stdout == b"done\n"
    assert len(result.stderr_tail) == f.STDERR_TAIL_LINES
    assert result.stderr_tail[-1].startswith("line 199999 ")
    assert seen[0].startswith("line 0 ")


def test_stdout_limit_and_line_callback(tmp_path: Path) -> None:
    body = "for i in range(1000):\n    print(i)\n"
    path = str(script(tmp_path, "s.py", body))
    limited = f.run_process((PY, path), deadline_s=30, stdout_limit=10)
    assert limited.stdout == b"0\n1\n2\n3\n4\n" and limited.stdout_truncated
    lines: list[str] = []
    streamed = f.run_process((PY, path), deadline_s=30, capture_stdout=False, on_stdout_line=lines.append)
    assert streamed.stdout == b"" and lines[-1] == "999" and len(lines) == 1000


def test_callback_error_terminates_and_propagates(tmp_path: Path) -> None:
    body = "import time\nprint('first', flush=True)\ntime.sleep(30)\n"

    def boom(line: str) -> None:
        raise RuntimeError(f"parser failed on {line}")

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="parser failed on first"):
        f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=30, on_stdout_line=boom)
    assert time.monotonic() - start < 10


def test_deadline_kills_whole_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    body = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(ToolTimeout) as caught:
        f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=1.0)
    assert time.monotonic() - start < 8
    assert caught.value.deadline_s == 1.0
    assert caught.value.to_failure("render_error", "render").reason == "timeout"
    assert wait_dead(int(pidfile.read_text())), "grandchild survived process-group teardown"


def test_sigterm_ignoring_tool_is_killed_after_grace(tmp_path: Path) -> None:
    body = ("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\ntime.sleep(60)\n")
    ready = threading.Event()
    start = time.monotonic()
    with pytest.raises(ToolTimeout):
        f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=1.0,
                      on_stdout_line=lambda _line: ready.set())
    elapsed = time.monotonic() - start
    assert ready.is_set()
    assert 1.0 + f.TERMINATE_GRACE_S - 0.5 <= elapsed < 10


def test_cancel_event_terminates_and_raises_cancelled(tmp_path: Path) -> None:
    body = "import time\ntime.sleep(60)\n"
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    start = time.monotonic()
    with pytest.raises(Cancelled):
        f.run_process((PY, str(script(tmp_path, "s.py", body))), deadline_s=30, cancel_event=cancel)
    assert time.monotonic() - start < 8
    already = threading.Event()
    already.set()
    with pytest.raises(Cancelled):
        f.run_process((PY, "-V"), deadline_s=30, cancel_event=already)


def test_executable_revalidation(tmp_path: Path, make_tool) -> None:
    tool = make_tool("ffmpeg", "print('ffmpeg version N-1-gabc Copyright')")
    exe = f.Executable(tool, f.executable_identity(tool), "N-1-gabc")
    exe.revalidate()
    with pytest.raises(ToolMissing):
        f.Executable(tool, (0, 0), "N-1").revalidate()
    with pytest.raises(ToolMissing):
        f.run_process((str(tool),), deadline_s=5, executable=f.Executable(tool, (0, 0), "N-1"))
    tool.unlink()
    with pytest.raises(ToolMissing) as caught:
        exe.revalidate()
    assert caught.value.to_failure("render_error", "probe").reason == "probe_error"


def test_resolve_and_locate(tmp_path: Path, make_tool, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = make_tool("ffmpeg", "print('ffmpeg version N-125892-g406c5a37aa-x Copyright (c) 2000-2026')")
    assert f.resolve_executable(tool) == tool
    assert f.resolve_executable("ffmpeg", search_path=str(tool.parent)) == tool
    with pytest.raises(EnvironmentError_):
        f.resolve_executable("ffmpeg", search_path=str(tmp_path / "empty"))
    with pytest.raises(EnvironmentError_):
        f.resolve_executable("bin/ffmpeg")
    exe = f.locate_ffmpeg(tool)
    assert exe.version == "N-125892-g406c5a37aa-x" and exe.identity == f.executable_identity(tool)
    broken = make_tool("ffmpeg-broken", "import sys\nsys.exit(1)")
    with pytest.raises(EnvironmentError_):
        f.locate_ffmpeg(broken)


def test_parse_version() -> None:
    assert f.parse_version("ffmpeg version N-125892-g406c5a37aa-https://x Copyright\n") == \
        "N-125892-g406c5a37aa-https://x"
    assert f.parse_version("ffprobe version 7.1 Copyright") == "7.1"
    with pytest.raises(ValueError):
        f.parse_version("")


ENCODERS_TEXT = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D a64multi             Multicolor charset for Commodore 64 (codec a64_multi)
 A....D alac                 ALAC (Apple Lossless Audio Codec)
 A..... aac_at               aac (AudioToolbox) (codec aac)
 A....D flac                 FLAC (Free Lossless Audio Codec)
"""
FILTERS_TEXT = """Filters:
  T.. = Timeline support
  ------
 T. alimiter          A->A       Audio lookahead limiter.
 .. ebur128           A->N       EBU R128 scanner.
"""
ALIMITER_TEXT = """Filter alimiter
alimiter AVOptions:
   level_in          <double>     ..F.A....T. set input level (from 0.015625 to 64) (default 1)
   level             <boolean>    ..F.A....T. auto level (default true)
   latency           <boolean>    ..F.A....T. compensate delay (default false)
"""


def test_parse_feature_listings() -> None:
    assert {"alac", "aac_at", "flac", "a64multi"} == f.parse_encoders(ENCODERS_TEXT)
    assert f.parse_filters(FILTERS_TEXT) == {"alimiter", "ebur128"}
    assert f.parse_filter_options(ALIMITER_TEXT) == {"level_in", "level", "latency"}


def test_environment_and_requirements(make_tool) -> None:
    body = (
        f"enc = {ENCODERS_TEXT!r}\nfil = {FILTERS_TEXT!r}\nlim = {ALIMITER_TEXT!r}\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['-nostdin', '-hide_banner'], args\n"
        "if '-version' in args: print('ffmpeg version N-9 Copyright')\n"
        "elif '-encoders' in args: print(enc)\n"
        "elif '-filters' in args: print(fil)\n"
        "elif 'filter=alimiter' in args: print(lim)\n"
        "else: sys.exit(2)\n"
    )
    exe = f.locate_ffmpeg(make_tool("ffmpeg", body))
    env = f.probe_environment(exe)
    assert "flac" in env.encoders and "ebur128" in env.filters and env.platform_basis
    f.check_requirements(env, encoders=("flac", "alac"), filters=("ebur128", "alimiter"),
                         alimiter_options=f.ALIMITER_OPTIONS)
    with pytest.raises(EnvironmentError_) as caught:
        f.check_requirements(env, encoders=("libmp3lame",), filters=("loudnorm",))
    assert "encoder libmp3lame" in str(caught.value) and "filter loudnorm" in str(caught.value)
    with pytest.raises(ValueError):
        f.check_requirements(env, encoders=("alac_at",))
