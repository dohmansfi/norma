"""S1 parsing, application-only dispatch, streams, refusals and environment failures."""
import importlib
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import __version__, cli, pipeline
from norma.models import Diagnostic, ProcessingMark, RawNativeItem, SourceTags, TagValue


@pytest.mark.parametrize('args', [[], ['tags'], ['help'], ['help', 'analyse'], ['tags', 'show', '--help']])
def test_command_lists_and_help(args, capsys):
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert 'usage:' in captured.out and captured.err == ''


@pytest.mark.parametrize('args', [['--version'], ['analyse', '--version'], ['tags', 'show', '--version']])
def test_version_is_only_distribution_version(args, capsys):
    assert cli.main(args) == 0
    assert capsys.readouterr().out == f'norma {__version__}\n'


@pytest.mark.parametrize('command', ['analyse', 'analyze', 'normalise', 'normalize', 'verify', 'formats', 'doctor', 'tags show', 'tags plan', 'tags template'])
@pytest.mark.parametrize('flag', ['--in-place', '--no-backup'])
def test_safety_refusal_before_help_or_any_work(command, flag, capsys, monkeypatch):
    monkeypatch.setattr(cli, '_pipeline', lambda: pytest.fail('preflight must not run'))
    assert cli.main(command.split() + ['x.wav', flag, '--help']) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'never replaces your originals' in captured.err


@pytest.mark.parametrize('args,expected', [(['normalise', 'take.wav'], 'no such file or folder'),
    (['normalize', 'take.wav'], 'no such file or folder'), (['verify', 'take.wav'], 'no such file or folder'),
    (['tags', 'plan', 'take.wav'], 'no such file or folder'), (['tags', 'template', 'take.wav'], 'requires -o'),
    (['analyse', '--report', 'result.csv', 'take.wav'], 'no such file or folder'),
    (['analyse', '--peak-mode', 'dynamic', 'take.wav'], 'no such file or folder'),
    (['analyse', '-t', 'nan', 'take.wav'], 'finite'), (['analyse', '-j', '33', 'take.wav'], '1 to 32'),
    (['analyse', '--targte', '-14'], 'did you mean "--target"'), (['normalice'], 'did you mean "normalise"'),
    (['analyse'], 'no input given'), (['formats', 'take.wav'], 'unknown option'),
    (['analyse', '-v', '-q'], 'not allowed')])
def test_usage_messages_and_empty_stdout(args, expected, capsys):
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and expected in captured.err


def test_help_config_default(tmp_path, capsys):
    config = tmp_path / 'prefs.toml'
    config.write_text('target = -14\n')
    assert cli.main(['analyse', '--config', str(config), '--help']) == 0
    assert 'your config: -14' in capsys.readouterr().out


@pytest.mark.parametrize('args', [['--unknown'], ['tags', '--unknown'], ['analyse', '--format', 'wav'],
                               ['--ffmpeg', '/tool/ffmpeg', 'formats']])
def test_options_cannot_escape_the_command_matrix(args, capsys):
    assert cli.main(args) == 2
    assert capsys.readouterr().out == ''


def test_dynamic_dry_run_still_checks_the_input(capsys):
    assert cli.main(['normalise', 'x.wav', '--peak-mode', 'dynamic', '--dry-run']) == 2
    assert 'no such file or folder' in capsys.readouterr().err


def test_quiet_formats_hides_only_stderr(capsys):
    assert cli.main(['formats', '-q']) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('OK\t') and captured.err == ''


def test_formats_requires_no_pipeline_or_ffmpeg(monkeypatch, capsys):
    monkeypatch.setattr(cli, '_pipeline', lambda: pytest.fail('formats is a pure policy table'))
    monkeypatch.setenv('PATH', '')
    assert cli.main(['formats']) == 0
    captured = capsys.readouterr()
    rows = captured.out.splitlines()
    assert len(rows) == 6
    assert {row.split('\t')[1] for row in rows} == {'wav', 'aiff', 'flac', 'alac', 'mp3', 'aac'}
    assert 'not established' in captured.out and 'policy only' in captured.out
    assert 'alac_at' not in captured.out


def test_missing_mutagen_becomes_environment_error(monkeypatch, capsys):
    real_import = importlib.import_module
    def missing(name):
        if name == 'norma.pipeline':
            raise ModuleNotFoundError('No module named mutagen', name='mutagen')
        return real_import(name)
    monkeypatch.setattr(cli.importlib, 'import_module', missing)
    assert cli.main(['doctor']) == 3
    assert 'mutagen is not installed' in capsys.readouterr().err


def test_doctor_without_ffmpeg_runs_all_rows(monkeypatch, capsys):
    monkeypatch.setenv('PATH', '')
    assert cli.main(['doctor']) == 3
    captured = capsys.readouterr()
    assert 'FAIL\tffmpeg\tFFmpeg not found' in captured.out
    assert '\tconfig\t' in captured.out and '\tmutagen\t' in captured.out
    assert 'Traceback' not in captured.out + captured.err


def snapshot(*, missing=False):
    info = pipeline.Check(pipeline.ToolInfo(Path('/tool/ffmpeg'), 'N-125892'))
    problem = pipeline.Check(problem=pipeline.Problem('not_found', 'missing'))
    facilities = pipeline.Check(pipeline.Facilities(frozenset(('alac',)), ('alac',)))
    return pipeline.EnvironmentInspection(pipeline.RuntimeInfo('synthetic', '3.12', Path('/python')),
        pipeline.PlatformInfo('Darwin', '15.6', 'arm64', 'synthetic'), problem if missing else info,
        info, facilities, facilities, facilities, pipeline.Check('synthetic'), problem,
        pipeline.Check(()))


@pytest.mark.parametrize('missing,expected', [(False, 2), (True, 3)])
def test_doctor_bad_config_and_environment_precedence(tmp_path, monkeypatch, capsys, missing, expected):
    config = tmp_path / 'bad.toml'
    config.write_text('target = "oops"\n')
    monkeypatch.setattr(pipeline, 'inspect_environment', lambda *a: snapshot(missing=missing))
    assert cli.main(['doctor', '--config', str(config)]) == expected
    captured = capsys.readouterr()
    assert 'FAIL\tconfig\t' in captured.out
    assert 'WARN\tsips\t' in captured.out


@pytest.fixture
def fake_run(monkeypatch, outcome):
    captured = []
    monkeypatch.setattr(cli.shutil, 'which', lambda value: '/tool/ffmpeg')
    monkeypatch.setattr(pipeline, 'locate_tools', lambda options, **kwargs:
        SimpleNamespace(prober=SimpleNamespace(version='synthetic'), ffmpeg=SimpleNamespace(version='synthetic')))
    def discover(options):
        captured.append(options)
        return SimpleNamespace(items=(outcome().input,), hints=(), excluded={}, duplicates=0,
                               ignored_links=0, directory_links=0, unreadable=())
    monkeypatch.setattr(pipeline, 'discover_inputs', discover)
    return captured


@pytest.mark.parametrize('alias', ['analyse', 'analyze'])
def test_analyse_uses_application_and_piped_summary(alias, fake_run, monkeypatch, outcome, capsys):
    item = pipeline.Inspection(outcome())
    def analyse(items, options, tools, **kwargs):
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 0, False)
    monkeypatch.setattr(pipeline, 'analyse_items', analyse)
    assert cli.main([alias, 'take.wav', '-t', '-14', '--log', 'ignored.log']) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('OK\t1/1\t') and 'SUMMARY\trun_id=' in captured.out
    assert len(captured.out.splitlines()[0].split('\t')) == 11
    assert 'writes no log' in captured.err
    assert fake_run[0].command == 'analyse' and fake_run[0].target == -14
    assert not Path('ignored.log').exists()


def test_paths_can_be_interleaved_with_options(fake_run, monkeypatch, outcome, capsys):
    monkeypatch.setattr(pipeline, 'analyse_items', lambda *a, **kw: pipeline.ReadOnlyRun((), 0, False))
    assert cli.main(['analyse', 'a.wav', '-t', '-14', 'b.wav', '-p', '-2', 'c.wav']) == 0
    assert tuple(path.name for path in fake_run[0].inputs) == ('a.wav', 'b.wav', 'c.wav')


def test_per_file_failure_does_not_abort_and_verbose_tail(fake_run, monkeypatch, outcome, capsys):
    items = (pipeline.Inspection(outcome(status='failed', reason_code='internal'),
                 Diagnostic('internal', 'unexpected failure', 'probe', '\n'.join(str(i) for i in range(30)))),
             pipeline.Inspection(outcome(1, 'next.wav')))
    def analyse(*a, **kwargs):
        for item in items:
            kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun(items, 0, False)
    monkeypatch.setattr(pipeline, 'analyse_items', analyse)
    assert cli.main(['analyse', 'take.wav', '-v']) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith('FAIL\t') and '\nOK\t' in captured.out
    assert 'failed=1' in captured.out and 'ok=1' in captured.out
    assert captured.err.splitlines()[:2] == ['10', '11']


def test_cancelled_summary_and_handler_restored(fake_run, monkeypatch, outcome, capsys):
    old_handler = signal.getsignal(signal.SIGINT)
    item = pipeline.Inspection(outcome(status='skipped', reason_code='cancelled'))
    def cancelled(*a, **kwargs):
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert kwargs['cancel_event'].is_set()
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 2, True)
    monkeypatch.setattr(pipeline, 'analyse_items', cancelled)
    assert cli.main(['analyse', 'take.wav']) == 130
    assert 'not_started=2' in capsys.readouterr().out
    assert signal.getsignal(signal.SIGINT) is old_handler


def test_tags_show_all_stores_conflicts_and_mark(fake_run, monkeypatch, outcome, capsys):
    native = RawNativeItem('native:one', 'ID3', 'TIT2', 'TIT2', '2.3', 'synthetic', b'title', 'ID3:TIT2')
    tags = SourceTags({'title': TagValue('Demo', 'ID3', 'TIT2')}, {}, (), (), (), (native,), {'ID3': ('native:one',), 'RIFF_INFO': ()})
    item = pipeline.Inspection(outcome(measurement_in=None), source_tags=tags,
                              mark=ProcessingMark('absent', None, None, None, None))
    def inspect(*a, **kwargs):
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 0, False)
    monkeypatch.setattr(pipeline, 'inspect_tags', inspect)
    assert cli.main(['tags', 'show', 'take.wav']) == 0
    captured = capsys.readouterr()
    assert '\nSTORE\t' in captured.out and 'RIFF_INFO' in captured.out
    assert '\nFIELD\t' in captured.out and 'Demo' in captured.out
    assert '\nNATIVE\t' in captured.out and '\nMARK\t' in captured.out
    assert captured.out.splitlines()[-1].startswith('SUMMARY\t')


@pytest.mark.parametrize('container', ['caf', 'adts'])
def test_unavailable_tag_reader_is_per_file_note(container, outcome):
    from norma.models import ProbeResult
    probe = ProbeResult('aac', 'lossy', container, 0, 0, 48000, None, False, 'fltp', 2, 'stereo',
                        10, 'stream', 256000, False, 0, {}, 'synthetic')
    diagnostic = pipeline.unreadable_tags(container)
    result = outcome(probe=probe, status='warn', reason_code='metadata_loss_warn', measurement_in=None,
                     warnings=(diagnostic,))
    text = cli.report.render_tsv(result, 1)
    assert text.startswith('WARN\t') and 'metadata_loss_warn:' in text and 'internal:' not in text


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='installed FFmpeg required')
def test_real_analyse_and_tags_are_read_only(tmp_path, capsys):
    source = tmp_path / 'demo.wav'
    rate = 48000
    with wave.open(str(source), 'wb') as writer:
        writer.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(b''.join(struct.pack('<hh', sample, sample)
            for i in range(rate * 4) for sample in [int(2000 * math.sin(2 * math.pi * 997 * i / rate))]))
    before = source.read_bytes()
    files = set(tmp_path.rglob('*'))
    for args in (['analyse', str(source)], ['tags', 'show', str(source)]):
        assert cli.main(args) == 0
        captured = capsys.readouterr()
        assert captured.out.splitlines()[0].startswith('OK\t')
        assert captured.out.splitlines()[-1].startswith('SUMMARY\t')
        assert source.read_bytes() == before and set(tmp_path.rglob('*')) == files
