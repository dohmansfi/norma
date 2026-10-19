"""Small real-tool S3 runs through the CLI, confined to the hermetic temporary sandbox (TK-0309)."""
import csv
import io
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from norma import cli

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='installed FFmpeg required')


@pytest.fixture(autouse=True)
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory cleanup.
    monkeypatch.setattr(shutil, '_use_fd_functions', False)


@pytest.fixture
def demo(tmp_path):
    def make(name='demo.wav', amplitude=2000, spike=None, rate=44100):
        path = tmp_path / name
        with wave.open(str(path), 'wb') as writer:
            writer.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
            writer.writeframes(b''.join(struct.pack('<hh', sample, sample)
                for i in range(rate * 5)
                for sample in [spike if spike is not None and i % rate == 100
                               else int(amplitude * math.sin(2 * math.pi * 997 * i / rate))]))
        return path
    return make


@pytest.fixture
def mp3_of(tmp_path):
    def make(source, name='bounce.mp3'):
        path = tmp_path / name
        subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-y', '-i', str(source), '-map_metadata', '-1',
                        '-fflags', '+bitexact', '-c:a', 'libmp3lame', '-q:a', '0', '-write_xing', '0',
                        '-id3v2_version', '0', str(path)], check=True)
        return path
    return make


def run(capsys, *args):
    code = cli.main(['normalise', *map(str, args), '--config', '/dev/null'])
    captured = capsys.readouterr()
    return code, captured


def rows_path(captured):
    return dict(cell.split('=', 1) for cell in captured.out.splitlines()[-1].split('\t')[1:])['report']


def rows(captured, tmp_path):
    path = Path(rows_path(captured))
    return list(csv.DictReader(io.StringIO(path.read_text())))


def test_real_mp3_v0_output(demo, tmp_path, capsys):
    source = demo()
    code, captured = run(capsys, source, '-f', 'mp3', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('OK\t')
    row = rows(captured, tmp_path)[0]
    output = Path(row['out_path'])
    assert output.suffix == '.mp3' and output.is_file()
    assert row['out_format'] == 'mp3' and row['bitrate_mode'] == 'vbr' and row['vbr_quality'] == '0'
    assert row['execution'] == 'committed' and abs(float(row['out_lufs']) + 16) <= 0.5


def test_real_aac_256k_output(demo, tmp_path, capsys):
    source = demo()
    code, captured = run(capsys, source, '-f', 'aac', '--bitrate', '256k', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('OK\t')
    row = rows(captured, tmp_path)[0]
    assert Path(row['out_path']).suffix == '.m4a'
    assert row['out_format'] == 'aac' and row['bitrate_mode'] == 'cbr'
    assert row['requested_bitrate_bps'] == row['effective_bitrate_bps'] == '256000'


def test_real_unsupported_explicit_mp3_setting_is_a_usage_error_before_any_write(demo, tmp_path, capsys):
    source = demo()
    before = sorted(p.name for p in tmp_path.rglob('*'))
    code, captured = run(capsys, source, '-f', 'mp3', '--bitrate', '999k', '-o', tmp_path / 'out')
    assert code == 2 and captured.out == '' and 'norma: error:' in captured.err
    assert sorted(p.name for p in tmp_path.rglob('*')) == before


def test_real_limit_within_the_cap_is_ok(demo, tmp_path, capsys):
    source = demo(spike=13000)
    code, captured = run(capsys, source, '-o', tmp_path / 'out')
    assert code == 0
    row = rows(captured, tmp_path)[0]
    assert row['mode'] == 'limit' and row['status'] == 'ok' and float(row['limited_db']) > 0
    assert captured.out.split('\t')[6] == 'limit'


def test_real_reduce_shortfall_is_a_warn_with_the_measured_wording(demo, tmp_path, capsys):
    source = demo(spike=26000)
    code, captured = run(capsys, source, '-t', '-14', '--peak-mode', 'reduce', '--max-limit', '2',
                         '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('WARN\t')
    assert 'reduce_shortfall: reduced so peaks stay under -1 dBTP; measures ' in captured.out
    assert 'LU below -14' in captured.out
    assert 'note: normalise ignores --max-limit' in captured.err
    row = rows(captured, tmp_path)[0]
    assert row['mode'] == 'reduce' and float(row['measured_shortfall_lu']) > 0


def test_real_limit_that_needs_more_than_the_cap_falls_back_to_reduce(demo, tmp_path, capsys):
    source = demo(spike=26000)
    code, captured = run(capsys, source, '-t', '-14', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('WARN\t')
    assert 'reduce_shortfall: limit could not reach the target (cap 3 dB)' in captured.out


def test_real_dynamic_normalise_is_accepted(demo, tmp_path, capsys):
    source = demo(spike=26000)
    code, captured = run(capsys, source, '-t', '-14', '--peak-mode', 'dynamic', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith(('OK\t', 'WARN\t'))
    assert captured.out.split('\t')[6] == 'dynamic'
    row = rows(captured, tmp_path)[0]
    assert row['mode'] == 'dynamic' and float(row['out_dbtp']) <= -1 + 0.1


@pytest.mark.parametrize('command', ['analyse', 'normalise'])
@pytest.mark.parametrize('explicit_report', [False, True])
def test_real_dynamic_preview_label_without_audio_or_automatic_writes(command, explicit_report, demo, tmp_path, capsys):
    # Depends on TK-0421's pipeline preview wiring (D-080); run after both lane-5 outputs land.
    source = demo(spike=26000)
    original = source.read_bytes()
    before = set(tmp_path.rglob('*'))
    args = [command, str(source), '--peak-mode', 'dynamic', '--config', '/dev/null']
    if command == 'normalise':
        args += ['--dry-run', '-o', str(tmp_path / 'uncreated')]
    report_path = tmp_path / 'preview.json'
    if explicit_report:
        args += ['--report', str(report_path)]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('OK\t')
    assert captured.out.splitlines()[0].split('\t')[6] == 'dynamic (calibrated at render)'
    expected = before | {report_path} if explicit_report else before
    assert source.read_bytes() == original and set(tmp_path.rglob('*')) == expected
    if explicit_report:
        row = json.loads(report_path.read_text())['outcomes'][0]
        assert row['mode'] == 'dynamic' and row['out_lufs'] is None and row['out_dbtp'] is None
        assert row['execution'] == ('plan' if command == 'normalise' else 'inspect')
        assert row['published_paths'] == []
    else:
        assert 'report=\t' in captured.out.splitlines()[-1]


def test_real_lossy_sources_lossless_stores_flac_with_warn_and_the_flac_note(demo, mp3_of, tmp_path, capsys):
    source = mp3_of(demo())
    code, captured = run(capsys, source, '--lossy-sources', 'lossless', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('WARN\t') and 'lossy_origin:' in captured.out
    row = rows(captured, tmp_path)[0]
    assert Path(row['out_path']).suffix == '.flac' and row['reason'] == 'lossy_origin'
    assert row['source_codec'] == 'mp3'
    assert captured.err.count('FLAC is not among the formats Apple documents') == 1
    code, captured = run(capsys, source, '--lossy-sources', 'lossless', '-f', 'apple', '-o', tmp_path / 'apple')
    assert code == 0 and 'FLAC is not among' not in captured.err
    assert Path(rows(captured, tmp_path)[0]['out_path']).suffix == '.m4a'


def test_real_lossy_sources_skip_leaves_the_mp3_alone(demo, mp3_of, tmp_path, capsys):
    source = mp3_of(demo())
    original = source.read_bytes()
    code, captured = run(capsys, source, '--lossy-sources', 'skip', '-o', tmp_path / 'out')
    assert code == 0 and captured.out.startswith('SKIP\t') and 'lossy_source_skipped' in captured.out
    assert source.read_bytes() == original and not list((tmp_path / 'out').glob('*.mp3'))


def test_real_automatic_report_sits_in_the_folder_the_pipeline_chooses(demo, tmp_path, capsys):
    one, two = demo('one.wav'), demo('two.wav')
    code, captured = run(capsys, one, two, '-o', tmp_path / 'out')
    assert code == 0
    assert Path(rows_path(captured)).parent == tmp_path / 'out'
