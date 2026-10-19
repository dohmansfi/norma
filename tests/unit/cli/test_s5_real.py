"""Real-tool verify smoke; synthetic audio only, entirely inside tmp_path."""
import csv
import hashlib
import io
import json
import math
import shutil
import struct
import wave

import pytest

from norma import cli

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='installed FFmpeg required')


@pytest.fixture
def demo(tmp_path, capsys):
    source = tmp_path / 'demo.wav'
    rate = 48000
    with wave.open(str(source), 'wb') as writer:
        writer.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(b''.join(struct.pack('<hh', value, value) for i in range(rate * 4)
            for value in [int(2000 * math.sin(2 * math.pi * 997 * i / rate))]))
    path = tmp_path / 'baseline.json'
    assert cli.main(['analyse', str(source), '--config', '/dev/null', '--report', str(path)]) == 0
    capsys.readouterr()
    level = json.loads(path.read_text())['outcomes'][0]['in_lufs']
    return source, round(level, 2)


def snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def mark(source, text):
    from mutagen.wave import WAVE
    from mutagen.id3 import TXXX
    audio = WAVE(source)
    audio.add_tags()
    audio.tags.add(TXXX(encoding=3, desc='NORMA', text=[text]))
    audio.save()


def summary(text):
    return dict(cell.split('=', 1) for cell in text.splitlines()[-1].split('\t')[1:])


@pytest.mark.parametrize('state', ['absent', 'valid', 'reduce', 'malformed', 'unsupported'])
def test_real_verify_mark_states_report_only_write_and_truthful_origins(demo, state, tmp_path, capsys):
    source, level = demo
    target = level + 2 if state == 'reduce' else level
    if state != 'absent':
        text = (f'v=1;tool=norma-0.1.0;target={target};tp=-1;mode={"reduce" if state == "reduce" else "linear"};'
                f'gain=+0.00;out_i={level:.2f};out_tp=-10.00;origin=flac;ffmpeg=N-synthetic;date=2026-10-05')
        mark(source, 'broken' if state == 'malformed' else 'v=2' if state == 'unsupported' else text)
    prefs = tmp_path / 'prefs.toml'
    prefs.write_text(f'target={level}\ntolerance=0.1\nskip_within=3\nlog="ignored.log"\n')
    before = snapshot(tmp_path)
    report = tmp_path / 'verify.json'
    args = ['verify', str(source), '--config', str(prefs), '--report', str(report), '-j', '2']
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    document = json.loads(report.read_text())
    row, = document['outcomes']
    assert row['mark_validity'] == ('valid' if state == 'reduce' else state)
    assert row['target_origin'] == ('mark' if state in ('valid', 'reduce') else 'config')
    assert row['target_lufs'] == target
    assert row['mode'] is None and row['gain_db'] is None and row['in_lufs'] is None
    assert row['out_path'] == str(source) and row['execution'] == 'inspect' and row['published_paths'] == []
    assert document['options']['skip_within'] == .3 and document['options']['tolerance'] == .1
    assert document['options']['value_sources']['skip_within'] == 'default'
    assert row['out_lufs'] == pytest.approx(level, abs=.02)
    if state == 'reduce':
        assert row['reason'] == 'reduce_shortfall'
        assert row['measured_shortfall_lu'] == pytest.approx(2, abs=.02)
        assert row['expected_lufs'] is None and row['predicted_shortfall_lu'] is None
        assert 'reduced output, peaks safe' in captured.out
    elif state in ('malformed', 'unsupported'):
        assert row['reason'] == 'processing_mark_invalid' and row['status'] == 'warn'
    else:
        assert row['reason'] == 'ok'
    assert summary(captured.out)['report'] == str(report)
    after = snapshot(tmp_path)
    assert {key: value for key, value in after.items() if key != 'verify.json'} == before
    assert set(after) == set(before) | {'verify.json'}
    # No report, no log, no output folder, no tag repair or mark insertion.
    before = after
    assert cli.main(['verify', str(source), '--config', str(prefs), '--check-tags']) == (1 if state == 'absent' else 0)
    text = capsys.readouterr()
    assert ('processing mark missing' in text.out) == (state == 'absent')
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('extension', ['csv', 'json'])
def test_real_manifest_mismatches_precise_report_barrier_and_sole_write(demo, tmp_path, capsys, extension):
    source, level = demo
    manifest = tmp_path / 'refs.json'
    manifest.write_text(json.dumps({'version': 1, 'defaults': {'compilation': False, 'track_total': 7},
        'files': {source.name: {'title': 'Expected', 'album': None, 'extras': {'Mood': 'calm'}}}}))
    before = snapshot(tmp_path)
    output = tmp_path / f'report.{extension}'
    args = ['verify', str(source), '--config', '/dev/null', '-t', str(level), '--check-tags', '--manifest', str(manifest)]
    assert cli.main(args + ['--report', str(output)]) == 1
    captured = capsys.readouterr()
    for part in ('processing mark missing', 'title: expected "Expected" from the manifest',
                 'compilation: expected false from the manifest defaults', 'track_total: expected 7',
                 'Mood: expected "calm"'):
        assert part in captured.out
    rows = (json.loads(output.read_text())['outcomes'] if extension == 'json' else
            list(csv.DictReader(io.StringIO(output.read_text()))))
    assert len(rows) == 1 and len(rows[0]) == 53
    assert rows[0]['reason'] == 'verification_tags'
    after = snapshot(tmp_path)
    assert {key: value for key, value in after.items() if key != output.name} == before
    assert cli.main(args + ['--report', str(manifest), '--overwrite']) == 2
    assert capsys.readouterr().out == '' and snapshot(tmp_path) == after
    assert cli.main(args + ['--report', str(output)]) == 2
    assert capsys.readouterr().out == '' and snapshot(tmp_path) == after


def test_real_missing_ffmpeg_and_usage_produce_no_writes(demo, tmp_path, monkeypatch, capsys):
    source, _ = demo
    before = snapshot(tmp_path)
    monkeypatch.setenv('PATH', '')
    assert cli.main(['verify', str(source), '--config', '/dev/null']) == 3
    captured = capsys.readouterr()
    assert captured.out == '' and 'FFmpeg not found' in captured.err
    assert cli.main(['verify', str(source), '--config', '/dev/null', '--skip-within', '.1']) == 2
    assert capsys.readouterr().out == '' and snapshot(tmp_path) == before
