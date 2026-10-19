"""S4 CLI metadata: validation, checked read-only planning and real tagged copies."""
import csv
import io
import json
import math
import shutil
import struct
import wave
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import cli, config, pipeline, report
from norma.models import Diagnostic


@pytest.mark.parametrize('command', [('normalise',), ('tags', 'plan')])
@pytest.mark.parametrize('arguments', [
    ['--tag', 'NORMA=x'], ['--tag', 'rEpLaYgAiN_TRACK_GAIN=x'], ['--tag', 'ITUNNORM=x'],
    ['--tag', 'ENCODER=x'], ['--tag', 'ITUNSMPB=x'], ['--tag', 'R128_TRACK_GAIN=x'],
    ['--tag', 'title=x'], ['--tag', '1bad=x'], ['--tag', 'DEMO='], ['--tag', 'DEMO=   '],
    ['--tag', 'compilation=1'], ['--tag', 'missing_equals'], ['--track', '0'],
    ['--track', '4/3'], ['--disc', '/3'], ['--date', '2026-02-30'],
    ['--artist', '\n'], ['--artist', 'line\nbreak'], ['--comment', 'x' * 4001],
    ['--track', '1', '--number-tracks'], ['--title', 'x', '--clear-tag', 'title'],
    ['--number-tracks', '--clear-tag', 'track'],
    ['--number-tracks', '--clear-tag', 'track_total'],
    ['--number-tracks', '--clear-tag', 'TRACK_TOTAL'],
    ['--tag', 'DEMO=x', '--clear-tag', 'demo'], ['--clear-tag', 'artwork'],
    ['--artwork', 'absent.png'], ['--manifest', 'absent.csv'],
    ['--artwork-max-size', '299'], ['--id3-version', '2.5'], ['--prefer', 'invalid'],
    ['--infer-mode', 'fill'], ['--infer-underscores'], ['--infer-from-filename', 'literal'],
    ['--infer-from-filename', '{invalid}'], ['--infer-from-filename', '{title'],
    ['--prefer', 'source', '--no-source-tags'], ['--artwork', 'x', '--remove-artwork'],
    ['-o', 'out', '-s', 'copy'], ['--on-metadata-loss', 'invalid'], ['--wav-tags', 'bad'],
])
def test_usage_validation_precedes_tools(command, arguments, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'locate_normalise_tools', lambda *a: pytest.fail('no tools'))
    monkeypatch.setattr(pipeline, 'locate_tools', lambda *a, **k: pytest.fail('no tools'))
    assert cli.main([*command, 'take.wav', '--config', '/dev/null', *arguments]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'norma: error:' in captured.err


def test_c10_metadata_shapes_and_provenance(tmp_path):
    prefs = tmp_path / 'prefs.toml'
    prefs.write_text('artist="Configured"\nalbum_artist="Band"\ngenre="Demo"\ncomposer="Me"\n'
                     'folder_artwork=true\nno_mark_processed=true\nid3_version=2.4\n')
    args = dict(inputs=['take.wav'], track='02/10', disc='1/2', artist='  Cafe\u0301  ',
                tag=['DEMO_NOTE=one=two', 'compilation=false'], clear_tag=['genre'], remove_artwork=True)
    options = config.build_options('normalise', args, config.load_config(prefs), Path('/tool/ffmpeg'))
    assert (options.track, options.track_total, options.disc, options.disc_total) == (2, 10, 1, 2)
    assert options.tag == (('DEMO_NOTE', 'one=two'), ('compilation', 'false'))
    assert options.clear_tag == ('genre',) and options.artist == 'Café'
    assert options.id3_version == '2.4' and options.no_mark_processed
    assert options.value_sources['artist'] == options.value_sources['track_total'] == 'cli'
    assert options.value_sources['album_artist'] == 'config'
    assert options.dropped_config_fields == ('folder_artwork',) and not options.folder_artwork


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / '01 Demo.wav'
    rate = 48000
    with wave.open(str(path), 'wb') as writer:
        writer.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(b''.join(struct.pack('<hh', value, value)
            for index in range(rate * 3)
            for value in [int(2000 * math.sin(2 * math.pi * 997 * index / rate))]))
    return path


def run(capsys, command, source, *args):
    code = cli.main([*command, str(source), '--config', '/dev/null', *map(str, args)])
    captured = capsys.readouterr()
    return code, captured


def ledger(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def png(path):
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress((b'\0' + b'\x80\x40\x20' * 8) * 8)) + chunk(b'IEND', b''))
    return path


real_tools = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='installed FFmpeg required')


@real_tools
def test_tags_plan_zero_ffmpeg_and_only_explicit_report(demo, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pipeline, 'locate_normalise_tools', lambda *a: pytest.fail('no encoding evidence'))
    monkeypatch.setattr(pipeline, 'analyse_items', lambda *a, **k: pytest.fail('no measurement'))
    # A tags plan must never execute FFmpeg, including capability-evidence encodes.
    import subprocess
    original = subprocess.Popen
    def launch(args, *a, **k):
        assert Path(args[0]).name != 'ffmpeg', args
        return original(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    before = ledger(tmp_path)
    code, captured = run(capsys, ('tags', 'plan'), demo, '--artist', 'Artist', '--tag', 'DEMO_NOTE=extra',
                         '--dry-run', '--peak-mode', 'dynamic', '--log', tmp_path / 'ignored.log', '-v')
    assert code == 0 and ledger(tmp_path) == before
    first = captured.out.splitlines()[0].split('\t')
    assert first[3:10] == [''] * 7
    assert 'if stream-copied' in captured.out and 'if encoded' in captured.out
    assert 'same in both alternatives' in captured.out and 'NATIVE_FATE\t' in captured.out
    assert '"rank":"E1"' in captured.out and '"mark":"pending"' in captured.out
    assert 'no audio is processed by tags plan' in captured.err and 'writes no log file' in captured.err
    destination = tmp_path / 'plan.json'
    code, captured = run(capsys, ('tags', 'plan'), demo, '--report', destination, '--tag', 'compilation=false')
    assert code == 0
    document = json.loads(destination.read_text())
    row = document['outcomes'][0]
    assert len(row['conditional_plans']) == 2 and row['published_paths'] == []
    assert row['execution'] == 'plan' and row['out_path'] is not None
    assert row['in_lufs'] is None and row['out_lufs'] is None and row['mode'] is None
    assert all(branch['metadata']['plan_revision'] == branch['output']['plan_revision'] for branch in row['conditional_plans'])
    assert all(branch['metadata']['loss_policy_result'] == 'ok' for branch in row['conditional_plans'])
    assert set(ledger(tmp_path)) == set(before) | {'plan.json'}
    prior = destination.read_bytes()
    code, captured = run(capsys, ('tags', 'plan'), demo, '--report', destination)
    assert code == 2 and captured.out == '' and destination.read_bytes() == prior


@real_tools
@pytest.mark.parametrize('command', [('normalise',), ('tags', 'plan')])
def test_single_file_metadata_is_counted_after_discovery(command, demo, tmp_path, capsys):
    other = tmp_path / 'other.wav'
    other.write_bytes(demo.read_bytes())
    before = ledger(tmp_path)
    code, captured = run(capsys, command, tmp_path, '--title', 'Single')
    assert code == 2 and captured.out == '' and '2 were found' in captured.err
    assert ledger(tmp_path) == before


@pytest.mark.parametrize('arguments', [[], ['-o', 'bad.txt'], ['-o', '-', '--infer-mode', 'fill'],
    ['-o', '-', '--infer-underscores'], ['-o', '-', '--infer-from-filename', '{unknown}'],
    ['-o', '-', '--artist', 'Artist'], ['-o', '-', '--manifest', 'x.csv'], ['-o', '-', '--report', 'x.csv']])
def test_template_usage_precedes_tools(arguments, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'locate_tools', lambda *a, **k: pytest.fail('no tools'))
    assert cli.main(['tags', 'template', 'take.wav', '--config', '/dev/null', *arguments]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'norma: error:' in captured.err


def test_template_config_matrix_and_destination(tmp_path):
    prefs = tmp_path / 'prefs.toml'
    prefs.write_text('artist="Ignored"\nfolder_artwork=true\nno_mark_processed=true\n'
                     'out_dir="elsewhere"\nsuffix=" copy"\nprefer="source"\nno_source_tags=true\n'
                     'recursive=true\ninfer_mode="override"\ninfer_underscores=true\n')
    options = config.build_options('tags_template', dict(template_output='metadata.csv'),
                                   config.load_config(prefs), Path('/tool/ffmpeg'))
    assert options.template_output == Path.cwd() / 'metadata.csv'
    assert options.recursive and options.infer_mode == 'override' and options.infer_underscores
    assert options.artist is None and not options.folder_artwork and not options.no_mark_processed
    assert options.out_dir is None and options.suffix is None and options.prefer == 'explicit'
    assert not options.no_source_tags
    assert {key for key, value in options.value_sources.items() if value == 'config'} == {
        'recursive', 'infer_mode', 'infer_underscores'}


@pytest.mark.parametrize('command', [('normalise',), ('tags', 'plan')])
def test_metadata_help_groups_and_enums(command, capsys):
    assert cli.main([*command, '--config', '/dev/null', '--help']) == 0
    text = capsys.readouterr().out
    assert [text.index(title + ':') for title in ('Loudness', 'Output', 'Format', 'Metadata', 'Run')] == sorted(
        text.index(title + ':') for title in ('Loudness', 'Output', 'Format', 'Metadata', 'Run'))
    for enumeration in ('explicit | source', 'fill | override', 'warn | sidecar | error',
                        'info+id3 | info | none', '2.3 | 2.4'):
        assert enumeration in ' '.join(text.split())
    assert '--no-mark-processed' in text and 'writes no tags' not in text


@real_tools
@pytest.mark.parametrize('terminal', [False, True])
def test_template_stdout_stays_csv_on_terminal(terminal, demo, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    before = ledger(tmp_path)
    code, captured = run(capsys, ('tags', 'template'), demo, '-o', '-', '-v')
    assert code == 0 and captured.out.startswith('file,') and ledger(tmp_path) == before
    assert 'SUMMARY' not in captured.out and 'STATUS' not in captured.out and '\x1b' not in captured.out
    assert ('STATUS' if terminal else 'SUMMARY\t') in captured.err


@real_tools
def test_template_ignores_metadata_config_and_has_no_title_fallback(demo, tmp_path, capsys):
    prefs = tmp_path / 'prefs.toml'
    prefs.write_text('artist="Not source tags"\nfolder_artwork=true\nlog="never.log"\n')
    before = ledger(tmp_path)
    assert cli.main(['tags', 'template', str(demo), '--config', str(prefs), '-o', '-']) == 0
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert all(not row['artist'] and not row['title'] for row in rows)
    assert 'writes no log' not in captured.err and ledger(tmp_path) == before


@real_tools
@pytest.mark.parametrize('form', ['csv', 'json', '-'])
def test_template_round_trip_no_ffmpeg(form, demo, tmp_path, capsys, monkeypatch):
    from mutagen.id3 import TIT2, TPE1, TRCK, TPOS, TDRC
    from mutagen.wave import WAVE
    source = WAVE(demo)
    source.add_tags()
    for frame in (TIT2(text='Source title'), TPE1(text='Source artist'), TRCK(text='2/8'),
                  TPOS(text='1/2'), TDRC(text='2026-10')):
        source.tags.add(frame)
    source.save()
    import subprocess
    original = subprocess.Popen
    def launch(args, *a, **k):
        assert Path(args[0]).name != 'ffmpeg', args
        return original(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    monkeypatch.chdir(tmp_path)
    destination = '-' if form == '-' else tmp_path / f'metadata.{form}'
    before = ledger(tmp_path)
    code, captured = run(capsys, ('tags', 'template'), demo, '-o', destination, '-q')
    assert code == 0
    if form == '-':
        assert ledger(tmp_path) == before
        assert captured.out.startswith('file,') and 'SUMMARY' not in captured.out and '\x1b' not in captured.out
        assert 'SUMMARY\t' in captured.err
        exported = tmp_path / 'stdout.csv'
        exported.write_text(captured.out)
    else:
        exported = destination
        assert set(ledger(tmp_path)) == set(before) | {f'metadata.{form}'}
        assert 'SUMMARY\t' in captured.out and 'report=' in captured.out
    # Re-load through the public CLI rather than bypassing manifest orchestration.
    code, captured = run(capsys, ('tags', 'plan'), demo, '--manifest', exported, '-f', 'flac', '-q')
    assert code == 0 and 'Source title' in captured.out and 'Source artist' in captured.out
    assert '"provenance":"manifest_default"' in captured.out
    assert '"track"' in captured.out and '"value":2' in captured.out and '"value":8' in captured.out


@real_tools
@pytest.mark.parametrize('mode', ['fill', 'override'])
def test_template_inference_and_source_precedence(mode, demo, tmp_path, capsys):
    from mutagen.id3 import TIT2
    from mutagen.wave import WAVE
    source = WAVE(demo)
    source.add_tags()
    source.tags.add(TIT2(text='Source title'))
    source.save()
    destination = tmp_path / 'inferred.csv'
    code, captured = run(capsys, ('tags', 'template'), demo, '-o', destination,
                         '--infer-from-filename', '{track} {title}', '--infer-mode', mode)
    assert code == 0
    rows = list(csv.DictReader(io.StringIO(destination.read_text())))
    assert rows[0]['title'] == ('Demo' if mode == 'override' else 'Source title')
    assert rows[0]['track'] == '1'


@real_tools
@pytest.mark.parametrize('kind', ['source', 'config', 'symlink', 'hardlink', 'directory', 'missing_parent', 'existing'])
def test_template_barrier_zero_rows_zero_writes(kind, demo, tmp_path, capsys):
    destination = tmp_path / 'metadata.csv'
    prefs = '/dev/null'
    overwrite = ['--overwrite'] if kind != 'existing' else []
    if kind == 'source':
        source = tmp_path / 'source.csv'
        source.write_bytes(demo.read_bytes())
        demo = destination = source
    elif kind == 'config':
        destination.write_text('artist="Config"\n')
        prefs = str(destination)
    elif kind == 'symlink':
        destination.symlink_to(demo)
    elif kind == 'hardlink':
        destination.hardlink_to(demo)
    elif kind == 'directory':
        destination.mkdir()
    elif kind == 'missing_parent':
        destination = tmp_path / 'absent' / 'metadata.csv'
    else:
        destination.write_text('hours of edits')
    before = ledger(tmp_path)
    assert cli.main(['tags', 'template', str(demo), '--config', prefs, '-o', str(destination), *overwrite]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'norma: error:' in captured.err and ledger(tmp_path) == before


@real_tools
def test_template_overwrite_and_publication_failure(demo, tmp_path, capsys, monkeypatch):
    destination = tmp_path / 'metadata.csv'
    destination.write_text('hours of edits')
    code, captured = run(capsys, ('tags', 'template'), demo, '-o', destination, '--overwrite')
    assert code == 0 and destination.read_text().startswith('file,')
    before = ledger(tmp_path)
    def fail(*a, **k):
        raise pipeline.RunOutputError('simulated permission failure')
    monkeypatch.setattr(pipeline.RunOutputs, 'publish', fail)
    code, captured = run(capsys, ('tags', 'template'), demo, '-o', destination, '--overwrite', '-q')
    assert code == 1 and 'could not write the template' in captured.err
    assert 'run_errors=1' in captured.out and ledger(tmp_path) == before
    assert 'note: template:' not in captured.err


@real_tools
def test_template_failed_file_does_not_contaminate_csv(demo, tmp_path, capsys, monkeypatch):
    broken = tmp_path / 'broken.wav'
    broken.write_bytes(demo.read_bytes())
    original = pipeline.read_source_tags
    def read(path, container):
        if path == broken:
            raise OSError('simulated unreadable tags')
        return original(path, container)
    monkeypatch.setattr(pipeline, 'read_source_tags', read)
    code, captured = run(capsys, ('tags', 'template'), tmp_path, '-o', '-', '-q')
    assert code == 1 and captured.out.startswith('file,')
    assert 'FAIL\t' in captured.err and 'FAIL' not in captured.out and 'SUMMARY' not in captured.out


@real_tools
@pytest.mark.parametrize('command', [('tags', 'plan'), ('normalise',)])
def test_numbering_inference_clears_and_single_branch(command, demo, tmp_path, capsys):
    other = tmp_path / '02 Other.wav'
    other.write_bytes(demo.read_bytes())
    destination = tmp_path / 'numbered.json'
    code, captured = run(capsys, command, tmp_path, '-f', 'flac', '--number-tracks',
                         '--infer-from-filename', '{track} {title}', '--clear-tag', 'genre', '--report', destination)
    assert code == 0
    rows = json.loads(destination.read_text())['outcomes']
    assert len(rows) == 2 and all(not row['conditional_plans'] for row in rows)
    for index, row in enumerate(rows, 1):
        assert row['tag_sources']['track'] == 'generated' and row['tag_sources']['title'] == 'filename'
    if command == ('normalise',):
        from mutagen.flac import FLAC
        for index, row in enumerate(rows, 1):
            output = FLAC(row['out_path'])
            assert output['tracknumber'] == [str(index)] and 'genre' not in output


@real_tools
def test_sidecar_publication_and_report_collision(demo, tmp_path, capsys):
    destination = tmp_path / 'sidecars.json'
    code, captured = run(capsys, ('normalise',), demo, '-f', 'aiff', '--date', '2026-10',
                         '--on-metadata-loss', 'sidecar', '--report', destination)
    assert code == 0
    row = json.loads(destination.read_text())['outcomes'][0]
    assert row['execution'] == 'committed' and row['sidecar'] in row['published_paths']
    sidecar = Path(row['sidecar'])
    assert json.loads(sidecar.read_text())['manifest']
    before = ledger(tmp_path)
    code, captured = run(capsys, ('tags', 'plan'), demo, '-f', 'aiff', '--date', '2026-10',
                         '--on-metadata-loss', 'sidecar', '--report', sidecar, '--overwrite')
    assert code == 2 and captured.out == '' and ledger(tmp_path) == before


@pytest.mark.parametrize('terminal', [False, True])
def test_conditional_refusal_visible_under_quiet(terminal, outcome, monkeypatch, capsys):
    item = pipeline.Inspection(outcome(measurement_in=None, execution='plan'),
        diagnostics=(Diagnostic('metadata_loss_error', 'encode: would lose metadata', 'metadata_planned'),))
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    monkeypatch.setattr(cli.shutil, 'which', lambda *a: '/tool/ffmpeg')
    monkeypatch.setattr(pipeline, 'locate_tools', lambda options, **k: SimpleNamespace(ffmpeg=None, prober=SimpleNamespace(version='fake')))
    monkeypatch.setattr(pipeline, 'discover_inputs', lambda options: SimpleNamespace(items=(item.outcome.input,), hints=(),
        excluded={}, duplicates=0, ignored_links=0, directory_links=0))
    def plan(items, options, tools, **kwargs):
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 0, False)
    monkeypatch.setattr(pipeline, 'plan_tags_items', plan)
    assert cli.main(['tags', 'plan', 'take.wav', '--config', '/dev/null', '-q']) == 0
    captured = capsys.readouterr()
    refusal_rows = [line.split('\t') for line in captured.out.splitlines() if line.startswith('REFUSAL\t')]
    assert len(refusal_rows) == 1 and refusal_rows[0][2] == 'planned'
    assert json.loads(refusal_rows[0][3]) == dict(reason_code='metadata_loss_error',
        stage='metadata_planned', detail=None)
    assert captured.err == ''


@real_tools
@pytest.mark.parametrize('refused', [('encode',), ('stream_copy',), ('stream_copy', 'encode')])
@pytest.mark.parametrize('form', ['json', 'csv'])
@pytest.mark.parametrize('terminal,quiet', [(False, True), (True, True), (True, False)])
def test_tags_plan_report_keeps_conditional_refusal(demo, tmp_path, capsys, monkeypatch, refused, form, terminal, quiet):
    """C-09 v1.4: explicitly simulated branch failures survive real CLI projection."""
    from norma.errors import FileFailure
    original = pipeline.plan_metadata
    def plan(*args, **kwargs):
        if kwargs['audio_basis'] in refused:
            raise FileFailure('metadata_loss_error', 'metadata_planned', f"simulated {kwargs['audio_basis']} loss")
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'plan_metadata', plan)
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    destination = tmp_path / f'conditional.{form}'
    before = demo.read_bytes()
    code, captured = run(capsys, ('tags', 'plan'), demo, '--report', destination, *(['-q'] if quiet else []))
    assert code == (1 if len(refused) == 2 else 0)
    if form == 'json':
        row: dict = json.loads(destination.read_text())['outcomes'][0]
    else:
        row = next(csv.DictReader(io.StringIO(destination.read_text())))
        for key in ('conditional_plans', 'published_paths'):
            row[key] = json.loads(row[key])
    assert row['status'] == ('failed' if len(refused) == 2 else 'ok')
    assert row['published_paths'] == [] and row['execution'] == 'plan'
    assert {branch['assumption'] for branch in row['conditional_plans']} == {'stream_copy', 'encode'}
    refusal_rows = [line.split('\t') for line in captured.out.splitlines() if line.startswith('REFUSAL\t')]
    assert len(refusal_rows) == len(refused)  # Primary diagnostics must not duplicate public refusals.
    for branch in row['conditional_plans']:
        basis = branch['assumption']
        if basis in refused:
            expected = dict(reason_code='metadata_loss_error', stage='metadata_planned', detail=f'simulated {basis} loss')
            assert branch['refusal'] == expected
            assert branch['metadata'] is branch['output'] is None
            label = {'stream_copy': 'if stream-copied', 'encode': 'if encoded'}[basis]
            assert json.loads(next(cells[3] for cells in refusal_rows if cells[2] == label)) == expected
        else:
            assert 'refusal' not in branch and branch['metadata'] is not None and branch['output'] is not None
            assert row['out_path'] == next(member['final_path'] for member in branch['output']['members']
                                           if member['purpose'] == 'audio')
    if len(refused) == 2:
        assert not row['out_path'] and 'PLAN\t' not in captured.out
    else:
        assert captured.out.count('PLAN\t') == 1 and 'same in both alternatives' not in captured.out
    if not quiet:
        assert ('1 FAIL' if len(refused) == 2 else '1 OK') in captured.out
    assert demo.read_bytes() == before
    assert not (tmp_path / 'Normalised').exists()


@real_tools
@pytest.mark.parametrize('form', ['json', 'csv'])
@pytest.mark.parametrize('terminal', [False, True])
def test_definitive_refusal_uses_contract_payload(demo, tmp_path, capsys, monkeypatch, form, terminal):
    destination = tmp_path / f'definitive.{form}'
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    before = demo.read_bytes()
    code, captured = run(capsys, ('tags', 'plan'), demo, '-f', 'aiff', '--date', '2026-10',
                         '--on-metadata-loss', 'error', '--report', destination, '-q')
    assert code == 1
    if form == 'json':
        row: dict = json.loads(destination.read_text())['outcomes'][0]
    else:
        row = next(csv.DictReader(io.StringIO(destination.read_text())))
        for key in ('conditional_plans', 'published_paths'):
            row[key] = json.loads(row[key])
    assert row['status'] == 'failed' and row['reason'] == 'metadata_loss_error'
    assert row['conditional_plans'] == [] and row['published_paths'] == []
    refusal_rows = [line.split('\t') for line in captured.out.splitlines() if line.startswith('REFUSAL\t')]
    assert len(refusal_rows) == 1 and refusal_rows[0][2] == 'planned'
    assert json.loads(refusal_rows[0][3]) == dict(reason_code='metadata_loss_error',
        stage='metadata_planned', detail='field:date')
    assert demo.read_bytes() == before and not (tmp_path / 'Normalised').exists()


@pytest.mark.parametrize('detail', [None, 'precise failure detail'])
def test_diagnostic_refusal_preserves_detail_not_display_message(outcome, detail):
    diagnostic = Diagnostic('metadata_loss_error', 'encode: human-facing message', 'metadata_planned', detail)
    cells = report.render_tag_plan(outcome(execution='plan'), diagnostics=(diagnostic,)).split('\t')
    assert cells[2] == 'planned'
    assert json.loads(cells[3]) == dict(reason_code=diagnostic.code, stage=diagnostic.stage, detail=detail)


@real_tools
def test_real_manifest_tag_artwork_and_no_mark_then_force(demo, tmp_path, capsys):
    cover = png(tmp_path / 'cover.png')
    manifest = tmp_path / 'metadata.csv'
    manifest.write_text('file,artist,album,date\n*,Manifest artist,Release,2026-10\n')
    destination = tmp_path / 'result.json'
    code, captured = run(capsys, ('normalise',), demo, '-f', 'flac', '--manifest', manifest,
                         '--artist', 'CLI artist', '--tag', 'DEMO_NOTE=Demo extra', '--tag', 'compilation=false',
                         '--track', '2/8', '--disc', '1/2', '--artwork', cover, '--no-mark-processed',
                         '--report', destination)
    assert code == 0
    row = json.loads(destination.read_text())['outcomes'][0]
    assert row['execution'] == 'committed' and row['tag_sources']['artist'] == 'cli'
    assert row['tag_sources']['album'] == 'manifest_default'
    output = Path(row['out_path'])
    code, captured = run(capsys, ('tags', 'show'), output)
    assert code == 0 and 'CLI artist' in captured.out and 'Release' in captured.out
    assert 'Demo extra' in captured.out and 'ARTWORK\t' in captured.out and '\tabsent\t' in captured.out
    # Omitting the mark does not require --force on a later format conversion.
    code, captured = run(capsys, ('normalise',), output, '-f', 'alac', '-o', tmp_path / 'marked')
    assert code == 0
    marked = tmp_path / 'marked' / output.with_suffix('.m4a').name
    code, captured = run(capsys, ('normalise',), marked, '-f', 'flac', '-o', tmp_path / 'skipped')
    assert code == 0 and captured.out.startswith('SKIP\t') and 'already_processed:' in captured.out
    code, captured = run(capsys, ('normalise',), marked, '--force', '-f', 'flac', '-o', tmp_path / 'forced')
    assert code == 0 and captured.out.startswith('OK\t')
    assert (tmp_path / 'forced' / output.name).is_file()


@real_tools
@pytest.mark.parametrize('strict', [False, True])
def test_real_conflict_exit_and_loss_fates(demo, tmp_path, capsys, strict):
    first_report = tmp_path / 'first.json'
    code, captured = run(capsys, ('normalise',), demo, '-f', 'flac', '--artist', 'Source artist',
                         '--no-mark-processed', '--report', first_report)
    assert code == 0
    source = Path(json.loads(first_report.read_text())['outcomes'][0]['out_path'])
    destination = tmp_path / 'conflicts.json'
    args = ['-f', 'aiff', '--artist', 'Other', '--date', '2026-10', '--report', destination]
    if strict:
        args += ['--strict-conflicts']
    code, captured = run(capsys, ('normalise',), source, *args)
    assert code == int(strict)
    row = json.loads(destination.read_text())['outcomes'][0]
    if strict:
        assert row['execution'] == 'not_written' and row['published_paths'] == []
        assert 'metadata_conflict_strict:' in captured.out
    else:
        assert row['execution'] == 'committed' and row['conflicts']
        assert any(loss['loss_kind'] == 'precision' for loss in row['dropped'])


@real_tools
@pytest.mark.parametrize('policy,code', [('warn', 0), ('error', 1)])
def test_real_loss_policy_pre_render(policy, code, demo, tmp_path, capsys):
    destination = tmp_path / 'loss.json'
    actual, captured = run(capsys, ('normalise',), demo, '-f', 'aiff', '--date', '2026-10',
                           '--on-metadata-loss', policy, '--report', destination)
    assert actual == code
    row = json.loads(destination.read_text())['outcomes'][0]
    if policy == 'error':
        assert row['execution'] == 'not_written' and not row['published_paths']
        assert 'metadata_loss_error:' in captured.out
    else:
        assert row['execution'] == 'committed' and row['status'] == 'warn' and row['dropped']


def test_tags_plan_faked_dispatch_preserves_failure_exit(outcome, monkeypatch, capsys):
    item = pipeline.Inspection(outcome(measurement_in=None, execution='plan', status='failed',
        reason_code='metadata_loss_error'), primary=Diagnostic('metadata_loss_error', 'would lose tags', 'metadata_planned'))
    monkeypatch.setattr(cli.shutil, 'which', lambda *a: '/tool/ffmpeg')
    monkeypatch.setattr(pipeline, 'locate_tools', lambda options, **k: SimpleNamespace(ffmpeg=None, prober=SimpleNamespace(version='fake')))
    monkeypatch.setattr(pipeline, 'discover_inputs', lambda options: SimpleNamespace(items=(item.outcome.input,), hints=(),
        excluded={}, duplicates=0, ignored_links=0, directory_links=0))
    def plan(items, options, tools, **kwargs):
        assert options.command == 'tags_plan' and options.no_mark_processed
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 0, False)
    monkeypatch.setattr(pipeline, 'plan_tags_items', plan)
    assert cli.main(['tags', 'plan', 'take.wav', '--config', '/dev/null', '--no-mark-processed']) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith('FAIL\t') and '\n\n' not in captured.out


@real_tools
def test_tags_plan_report_must_not_replace_manifest(demo, tmp_path, capsys):
    manifest = tmp_path / 'metadata.csv'
    manifest.write_text('file,artist\n*,Manifest artist\n')
    before = ledger(tmp_path)
    code, captured = run(capsys, ('tags', 'plan'), demo, '--manifest', manifest,
                         '--report', manifest, '--overwrite')
    assert code == 2, captured.out
    assert captured.out == '' and ledger(tmp_path) == before
