"""Real-tool check of the F-4 note: a two-stream M4A is normalised and the note reaches the CLI's verbose output and log."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from norma import cli

pytestmark = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory cleanup.
    monkeypatch.setattr(shutil, '_use_fd_functions', False)


def test_two_stream_m4a_note_is_reported(tmp_path, capsys):
    source = tmp_path / 'two.m4a'
    subprocess.run(['/usr/local/bin/ffmpeg', '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'sine=f=300:d=4', '-f', 'lavfi', '-i', 'sine=f=900:d=4',
                    '-map', '0:a', '-map', '1:a', '-c:a', 'alac', '-map_metadata', '-1', '-fflags', '+bitexact',
                    str(source)], check=True)
    original = source.read_bytes()
    log = tmp_path / 'run.log'
    assert cli.main(['normalise', str(source), '-o', str(tmp_path / 'out'), '-v', '--log', str(log),
                     '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0].startswith('OK\t')
    # An OK row has an empty note cell (report.py, WS-03); the plan note surfaces in -v output and the log.
    expected = 'note: 1 extra audio stream not carried over; only the first audio stream is processed'
    assert expected in captured.err
    assert expected in log.read_text()
    assert source.read_bytes() == original
