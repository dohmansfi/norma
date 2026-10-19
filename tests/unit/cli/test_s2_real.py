"""Small real-tool S2 runs, confined to the shared hermetic temporary sandbox."""
import csv
import io
import json
import math
import shutil
import struct
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import cli

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='installed FFmpeg required')


@pytest.fixture(autouse=True)
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory
    # cleanup. Absolute-path removal stays checked; production is unchanged.
    monkeypatch.setattr(shutil, '_use_fd_functions', False)


@pytest.fixture
def demo(tmp_path):
    def make(name='demo.wav', tagged=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = 48000
        with wave.open(str(path), 'wb') as writer:
            writer.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
            writer.writeframes(b''.join(struct.pack('<hh', sample, sample)
                for i in range(rate * 4)
                for sample in [int(2000 * math.sin(2 * math.pi * 997 * i / rate))]))
        if tagged:
            value = b'Demo title\0'
            info = b'INFOINAM' + struct.pack('<I', len(value)) + value
            if len(value) % 2:
                info += b'\0'
            chunk = b'LIST' + struct.pack('<I', len(info)) + info
            data = path.read_bytes() + chunk
            path.write_bytes(data[:4] + struct.pack('<I', len(data) - 8) + data[8:])
        return path
    return make


def files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def summary(text):
    return dict(cell.split('=', 1) for cell in text.splitlines()[-1].split('\t')[1:])


def test_real_normalise_automatic_ordinal_never_clobbers_with_overwrite(demo, tmp_path, capsys, monkeypatch):
    source = demo()
    original = source.read_bytes()
    monkeypatch.setattr(cli, 'time', SimpleNamespace(monotonic=time.monotonic,
        strftime=lambda fmt: 'norma-report-2026-10-10T120000.csv'))
    assert cli.main(['normalise', str(source), '--config', '/dev/null']) == 0
    first = capsys.readouterr()
    base = Path(summary(first.out)['report'])
    assert base.parent == tmp_path / 'Normalised'
    rows = list(csv.DictReader(io.StringIO(base.read_text())))
    assert rows[0]['execution'] == 'committed'
    assert abs(float(rows[0]['out_lufs']) + 16) <= 0.5
    output = Path(rows[0]['out_path'])
    assert output.is_file() and source.read_bytes() == original
    sentinel = base.read_bytes()
    assert cli.main(['normalize', str(source), '--config', '/dev/null']) == 0
    second = capsys.readouterr()
    assert Path(summary(second.out)['report']).name == base.stem + '-2.csv'
    assert second.out.startswith('SKIP\t') and base.read_bytes() == sentinel
    other = demo('other.wav')
    assert cli.main(['normalise', str(other), '--config', '/dev/null', '--overwrite']) == 0
    third = capsys.readouterr()
    assert Path(summary(third.out)['report']).name == base.stem + '-3.csv'
    assert base.read_bytes() == sentinel and source.read_bytes() == original
    assert not any('.norma-' in p.name for p in tmp_path.rglob('*'))


def test_real_dry_run_no_writes_then_explicit_report_no_clobber(demo, tmp_path, capsys):
    source = demo()
    before = files(tmp_path)
    assert cli.main(['normalise', str(source), '-n', '--log', str(tmp_path / 'ignored.log'),
                     '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    assert files(tmp_path) == before
    assert captured.out.splitlines()[0].split('\t')[7:10] == ['', '', '']
    assert summary(captured.out)['report'] == ''
    path = tmp_path / 'preview.json'
    args = ['normalise', str(source), '-n', '--report', str(path), '--config', '/dev/null']
    assert cli.main(args) == 0
    capsys.readouterr()
    document = json.loads(path.read_text())
    assert document['outcomes'][0]['execution'] == 'plan'
    assert document['outcomes'][0]['out_lufs'] is None
    assert document['outcomes'][0]['expected_lufs'] is not None
    assert document['outcomes'][0]['published_paths'] == []
    assert not (tmp_path / 'Normalised').exists()
    report_bytes = path.read_bytes()
    assert cli.main(args) == 2
    assert capsys.readouterr().out == '' and path.read_bytes() == report_bytes
    assert cli.main(args + ['--overwrite']) == 0
    capsys.readouterr()
    assert source.read_bytes() == before[source.relative_to(tmp_path)]
    assert set(files(tmp_path)) == set(before) | {Path('preview.json')}


def test_real_tagged_input_normalised_with_tags_and_mark(demo, tmp_path, capsys):
    source = demo(tagged=True)
    original = source.read_bytes()
    assert cli.main(['normalise', str(source), '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('OK\t') and 'tagged inputs are refused' not in captured.err
    assert source.read_bytes() == original
    path = Path(summary(captured.out)['report'])
    assert path.parent == tmp_path / 'Normalised'
    row = next(csv.DictReader(io.StringIO(path.read_text())))
    assert row['execution'] == 'committed' and json.loads(row['published_paths']) == [row['out_path']]
    assert cli.main(['tags', 'show', row['out_path'], '--config', '/dev/null']) == 0
    tags = capsys.readouterr().out
    assert 'Demo title' in tags and 'MARK\t' in tags and '\tvalid\t' in tags


@pytest.mark.parametrize('command', ['analyse', 'normalise'])
@pytest.mark.parametrize('kind', ['existing', 'missing_parent', 'source_alias', 'log_alias'])
def test_real_report_log_barrier_no_rows_no_writes(demo, tmp_path, capsys, command, kind):
    source = demo()
    report_path = tmp_path / 'report.csv'
    args = [command, str(source), '--config', '/dev/null']
    if kind == 'existing':
        report_path.write_text('old report\n')
        args += ['--report', str(report_path)]
    elif kind == 'missing_parent':
        args += ['--report', str(tmp_path / 'absent/report.json')]
    elif kind == 'source_alias':
        report_path.symlink_to(source)
        args += ['--report', str(report_path), '--overwrite']
    elif command == 'normalise':
        args += ['--log', str(source), '--overwrite']
    else:
        pytest.skip('analyse ignores logs by design')
    before = files(tmp_path)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'norma: error:' in captured.err
    assert files(tmp_path) == before


def test_real_analyse_explicit_json_is_the_only_write(demo, tmp_path, capsys):
    source = demo()
    before = files(tmp_path)
    path = tmp_path / 'analysis.json'
    assert cli.main(['analyse', str(source), '--report', str(path), '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    document = json.loads(path.read_text())
    assert document['report_path'] == str(path)
    assert summary(captured.out)['run_id'] == document['run_id']
    assert document['outcomes'][0]['execution'] == 'inspect'
    assert document['outcomes'][0]['out_path'] is None
    assert document['outcomes'][0]['out_lufs'] is None
    assert set(files(tmp_path)) == set(before) | {Path('analysis.json')}
    assert source.read_bytes() == before[source.relative_to(tmp_path)]


def test_real_two_file_batch_log_append_and_explicit_sorted_report(demo, tmp_path, capsys):
    first, second = demo('one.wav'), demo('two.wav')
    originals = (first.read_bytes(), second.read_bytes())
    path, log = tmp_path / 'run.json', tmp_path / 'run.log'
    log.write_text('previous run\n')
    assert cli.main(['normalise', str(first), '-j', '2', str(second), '--report', str(path),
                     '--log', str(log), '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    document = json.loads(path.read_text())
    assert [row['input_index'] for row in document['outcomes']] == [0, 1]
    assert all(row['execution'] == 'committed' for row in document['outcomes'])
    assert document['counts']['ok'] == 2
    lines = log.read_text().splitlines()
    assert lines[0] == 'previous run'
    assert json.loads(lines[1])['run_id'] == document['run_id']
    assert summary(captured.out)['run_id'] == document['run_id']
    assert lines[-1].startswith('SUMMARY\t')
    assert (first.read_bytes(), second.read_bytes()) == originals
    assert not list(tmp_path.rglob('norma-report-*.csv'))


def test_real_two_stream_m4a_note_reaches_stderr_without_verbose(tmp_path, capsys):
    import subprocess
    source = tmp_path / 'two.m4a'
    subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', 'sine=f=300:d=4', '-f', 'lavfi', '-i', 'sine=f=900:d=4',
                    '-map', '0:a', '-map', '1:a', '-c:a', 'alac', '-map_metadata', '-1', '-fflags', '+bitexact', str(source)], check=True)
    assert cli.main(['normalise', str(source), '-o', str(tmp_path / 'out'), '--config', '/dev/null']) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0].startswith('OK\t')
    assert 'note: 1 extra audio stream not carried over' in captured.err
