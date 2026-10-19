"""TK-0415: the FFmpeg stream reader ends a line at CR, LF or CR LF (IC-2 F-2, D-072 (2)).

Without ``-nostats`` FFmpeg writes ``size=… speed=…`` progress lines that end in a bare CR.
A reader that splits on LF only glues that progress text to the front of the next metadata
line, which the meter then misreads. These tests feed ``_Stream`` directly (every way the
bytes can be cut into reads) and run real processes, including FFmpeg with forced progress.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

from norma.adapters import ffmpeg

PY = sys.executable
REAL_FFMPEG = Path("/usr/local/bin/ffmpeg")
real = pytest.mark.real_ffmpeg


class _Pipe:
    """Just enough of a pipe for ``_Stream.__init__``."""

    def fileno(self) -> int:
        return -1


def lines_of(chunks: list[bytes], *, tail_lines: int = 0) -> tuple[list[str], ffmpeg._Stream]:
    seen: list[str] = []
    stream = ffmpeg._Stream(_Pipe(), keep_bytes=0, tail_lines=tail_lines, on_line=seen.append)
    for chunk in chunks:
        stream.feed(chunk, callbacks=True)
    stream.finish(callbacks=True)
    return seen, stream


def every_cut(data: bytes, pieces: int):
    """Every way to cut ``data`` into ``pieces`` reads (empty reads included)."""
    for cuts in itertools.combinations_with_replacement(range(len(data) + 1), pieces - 1):
        bounds = (0, *cuts, len(data))
        yield [data[a:b] for a, b in itertools.pairwise(bounds)]


@pytest.mark.parametrize("data, expected", [
    (b"a\nb\n", ["a", "b"]),
    (b"a\rb\r", ["a", "b"]),
    (b"a\r\nb\r\n", ["a", "b"]),
    (b"a\r\n\r\nb", ["a", "", "b"]),
    (b"a\r\rb", ["a", "", "b"]),
    (b"a\n\rb", ["a", "", "b"]),
    (b"a\rb\nc\r\nd", ["a", "b", "c", "d"]),
    (b"no end", ["no end"]),
    (b"tail\r", ["tail"]),
    (b"", []),
])
def test_cr_lf_and_crlf_each_end_one_line(data: bytes, expected: list[str]) -> None:
    assert lines_of([data])[0] == expected


def test_the_ic2_progress_line_no_longer_prefixes_metadata() -> None:
    """The exact joined line IC-2 captured (Appendix A, P4)."""
    data = (b"size=N/A time=00:09:03.74 bitrate=N/A speed=1.07e+03x elapsed=0:00:00.51    \r"
            b"[Parsed_ametadata_2 @ 0x1] frame:6373 pts:26103808 pts_time:543.829333\n"
            b"[Parsed_ametadata_2 @ 0x1] lavfi.astats.Overall.Peak_level=-24.000000\n")
    seen, _ = lines_of([data])
    assert seen == ["size=N/A time=00:09:03.74 bitrate=N/A speed=1.07e+03x elapsed=0:00:00.51    ",
                    "[Parsed_ametadata_2 @ 0x1] frame:6373 pts:26103808 pts_time:543.829333",
                    "[Parsed_ametadata_2 @ 0x1] lavfi.astats.Overall.Peak_level=-24.000000"]


@pytest.mark.parametrize("data", [b"p1\rm1\r\nm2\nx\r\ry\r", b"\r\n\r\r\n\n", b"ab\r\ncd\ref\n"])
def test_line_ends_are_independent_of_how_reads_cut_the_bytes(data: bytes) -> None:
    """A CR LF pair split across two reads is still one line end; no read boundary adds a line."""
    whole = lines_of([data])[0]
    for pieces in (2, 3):
        for chunks in every_cut(data, pieces):
            assert lines_of(chunks)[0] == whole, chunks


def test_the_bounded_tail_holds_split_lines() -> None:
    _, stream = lines_of([b"size=1\rframe:1\nsize=2\r", b"frame:2\r\n"], tail_lines=8)
    assert list(stream.tail) == ["size=1", "frame:1", "size=2", "frame:2"]


def test_overlong_cr_terminated_lines_stay_bounded() -> None:
    big = b"x" * (ffmpeg._MAX_LINE_BYTES + 3)
    seen, _ = lines_of([big[:10], big[10:] + b"\r", b"\nnext\r"])
    assert [len(line) for line in seen] == [ffmpeg._MAX_LINE_BYTES, 3, 4] and seen[-1] == "next"


def test_run_process_splits_a_child_s_cr_output(tmp_path: Path) -> None:
    body = ("import sys, time\n"
            "for n in range(3):\n"
            "    sys.stderr.write(f'progress {n}\\r'); sys.stderr.flush(); time.sleep(0.01)\n"
            "    sys.stderr.write(f'meta {n}\\r'); sys.stderr.flush(); time.sleep(0.01)\n"
            "    sys.stderr.write('\\n' if n == 1 else ''); sys.stderr.flush()\n"
            "sys.stdout.write('out a\\rout b\\r\\n'); sys.stdout.flush()\n")
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    errors: list[str] = []
    outputs: list[str] = []
    result = ffmpeg.run_process((PY, str(script)), deadline_s=30, on_stderr_line=errors.append,
                                on_stdout_line=outputs.append)
    assert result.returncode == 0
    assert errors == ["progress 0", "meta 0", "progress 1", "meta 1", "progress 2", "meta 2"]  # CR, LF: one end
    assert outputs == ["out a", "out b"]
    assert list(result.stderr_tail)[-2:] == ["progress 2", "meta 2"]


@real
def test_real_ffmpeg_progress_lines_are_lines_of_their_own() -> None:
    """FFmpeg with progress forced every millisecond: every metadata frame header and every
    value keeps its own line, and no progress text is glued to either (hermetic: lavfi in,
    null out)."""
    lines: list[str] = []
    args = (str(REAL_FFMPEG), "-nostdin", "-hide_banner", "-stats", "-stats_period", "0.001", "-f", "lavfi",
            "-i", "sine=frequency=997:sample_rate=48000:duration=120",
            "-af", "ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.M", "-f", "null", "-")
    result = ffmpeg.run_process(args, deadline_s=120, on_stderr_line=lines.append)
    assert result.returncode == 0
    progress = [line for line in lines if line.startswith("size=")]
    frames = [line for line in lines if "] frame:" in line]
    values = [line for line in lines if "lavfi.r128.M=" in line]
    assert progress, "FFmpeg wrote no progress line, so the test proved nothing"
    assert not any("speed=" in line and "[Parsed_" in line for line in lines)
    assert all(line.startswith("[Parsed_ametadata_") for line in frames + values)
    assert len(frames) == len(values) == 1200
