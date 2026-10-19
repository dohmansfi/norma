"""S5 interface integration: explicit references, read-only execution and projections."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import batch, cli, pipeline


@pytest.fixture
def shell(monkeypatch, outcome):
    seen = SimpleNamespace(options=None, need_ffmpeg=None, execute=None, record=None)
    monkeypatch.setattr(cli.shutil, 'which', lambda value: '/tool/ffmpeg')
    def locate(options, *, need_ffmpeg):
        seen.need_ffmpeg = need_ffmpeg
        return SimpleNamespace(prober=SimpleNamespace(version='synthetic'),
                               ffmpeg=SimpleNamespace(version='synthetic'))
    monkeypatch.setattr(pipeline, 'locate_tools', locate)
    def discover(options):
        seen.options = options
        return SimpleNamespace(items=(outcome().input,), hints=(), excluded={}, duplicates=0,
                               ignored_links=0, directory_links=0)
    monkeypatch.setattr(pipeline, 'discover_inputs', discover)
    def verify(items, options, tools, **kwargs):
        seen.execute = kwargs.get('execute')
        record = seen.record or pipeline.Inspection(outcome(measurement_in=None))
        kwargs['on_inspection'](record)
        return pipeline.ReadOnlyRun((record,), 0, False)
    monkeypatch.setattr(pipeline, 'verify_items', verify)
    return seen


def test_verify_dispatch_requires_meter_and_supplies_batch_executor(shell, capsys):
    assert cli.main(['verify', 'take.wav', '--config', '/dev/null', '-j', '2',
                     '--check-tags', '--title', 'Expected', '--track', '2/7']) == 0
    assert shell.need_ffmpeg is True and callable(shell.execute)
    assert shell.options.command == 'verify' and shell.options.jobs == 2
    assert shell.options.title == 'Expected'
    assert (shell.options.track, shell.options.track_total) == (2, 7)
    assert shell.options.value_sources['track_total'] == 'cli'
    assert capsys.readouterr().out.splitlines()[-1].startswith('SUMMARY\t')


@pytest.mark.parametrize('flag,value', [('--title', 'Demo'), ('--artist', 'Someone'),
    ('--album-artist', 'Someone'), ('--genre', 'Rock'), ('--composer', 'Someone'),
    ('--album', 'Record'), ('--comment', ''), ('--date', '2026-10'),
    ('--track', '2/7'), ('--disc', '1/2'), ('--manifest', 'refs.json')])
def test_explicit_tag_references_require_check_tags_before_discovery(shell, capsys, flag, value):
    assert cli.main(['verify', 'take.wav', flag, value]) == 2
    captured = capsys.readouterr()
    assert not captured.out and '--check-tags' in captured.err
    assert shell.options is None


@pytest.mark.parametrize('arguments', [('--format', 'wav'), ('--peak-mode', 'reduce'),
    ('--max-limit', '2'), ('--skip-within', '.1'), ('--artwork', 'cover.png'),
    ('--remove-artwork',), ('--folder-artwork',), ('--tag', 'Mood=calm'),
    ('--clear-tag', 'album'), ('--infer-from-filename', '{title}'), ('--force',),
    ('--no-mark-processed',), ('-o', 'out'), ('--dry-run',), ('--keep-failed',)])
def test_verify_refuses_rendering_and_non_v_metadata_options(shell, capsys, arguments):
    assert cli.main(['verify', 'take.wav', '--check-tags', *arguments]) == 2
    assert capsys.readouterr().out == '' and shell.options is None


def test_verify_ignores_non_applicable_config_and_retains_audio_origins(shell, tmp_path, capsys):
    config = tmp_path / 'prefs.toml'
    config.write_text('artist="Not a reference"\npeak_mode="reduce"\n'
                      'skip_within=2\nprefer="source"\nno_source_tags=true\n'
                      'out_dir="out"\nsuffix="suffix"\ntarget=-14\ntrue_peak=-2\ntolerance=0.5\n')
    assert cli.main(['verify', 'take.wav', '--config', str(config), '--check-tags', '-v']) == 0
    options = shell.options
    assert options.artist is None and options.out_dir is None and options.suffix is None
    assert options.peak_mode == 'limit' and options.prefer == 'explicit' and not options.no_source_tags
    assert (options.target, options.true_peak, options.tolerance) == (-14, -2, .5)
    assert all(options.value_sources[key] == 'config' for key in ('target', 'true_peak', 'tolerance'))
    assert 'config artist' not in capsys.readouterr().err


@pytest.mark.parametrize('tolerance', ['0.05', '0.1', '0.299'])
def test_verify_accepts_documented_tolerance_without_inapplicable_skip_option(shell, capsys, tolerance):
    assert cli.main(['verify', 'take.wav', '--config', '/dev/null', '--tolerance', tolerance]) == 0
    assert shell.options.tolerance == float(tolerance)
    assert capsys.readouterr().out.splitlines()[-1].startswith('SUMMARY\t')


@pytest.mark.parametrize('flag', [[], ['-q'], ['-v']])
@pytest.mark.parametrize('primary', ['verification_peak', 'verification_loudness', 'verification_tags'])
def test_verify_keeps_precise_secondary_mismatch_and_target_in_all_modes(shell, outcome, capsys, flag, primary):
    from norma.models import Diagnostic, ManifestCell, ProcessingMark, SourceTags, TagValue, VerificationReference
    invalid = Diagnostic('processing_mark_invalid', 'processing mark is malformed', 'verify')
    mark = ProcessingMark('malformed', None, None, None, None, (invalid,))
    ref = VerificationReference(-14, 'cli', -1, 'default', .5, 'default', mark)
    failure = Diagnostic(primary, 'primary check failed', 'verify')
    loudness = Diagnostic('verification_loudness', 'secondary loudness failure', 'verify')
    record = pipeline.Inspection(outcome(status='failed', reason_code=primary, measurement_in=None,
        measurement_out=outcome().measurement_in, verification_reference=ref, warnings=(invalid, loudness)),
        primary=failure, source_tags=SourceTags({'title': TagValue('Actual', 'synthetic', '')}, {}, (), (), (), (), {}),
        diagnostics=(failure, loudness, invalid),
        tag_references=(pipeline.InspectionReference('title', ManifestCell('set', 'Expected'), 'cli', '--title'),),
        tag_mismatches=('field:title',))
    shell.record = record
    assert cli.main(['verify', 'take.wav', '--config', '/dev/null', '--check-tags', '-t', '-14', *flag]) == 1
    captured = capsys.readouterr()
    assert 'target -14 (command line)' in captured.out
    assert 'title: expected "Expected" from the command line (--title), found "Actual"' in captured.out + captured.err
    if flag == ['-q']:
        assert 'warning:' not in captured.err
    else:
        assert 'processing mark is malformed' in captured.err
    if flag == ['-v']:
        assert 'secondary loudness failure' in captured.err
    assert captured.out.splitlines()[-1].startswith('SUMMARY\t')
