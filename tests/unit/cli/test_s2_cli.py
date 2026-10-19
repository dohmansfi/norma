"""S2 interface seams with synthetic records; real tools have separate sandbox tests."""
import json
import os
import signal
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import batch, cli, pipeline, report
from norma.errors import EnvironmentError_, UsageError
from norma.models import Diagnostic


def doctor_snapshot(version):
    info = pipeline.Check(pipeline.ToolInfo(Path('/tool/ffmpeg'), version))
    facilities = pipeline.Check(pipeline.Facilities(frozenset(('alac',)), ('alac',)))
    return pipeline.EnvironmentInspection(pipeline.RuntimeInfo('synthetic', '3.12', Path('/python')),
        pipeline.PlatformInfo('Darwin', '15.6', 'arm64', 'synthetic'), info,
        info, facilities, facilities, facilities, pipeline.Check('synthetic'),
        pipeline.Check(problem=pipeline.Problem('not_found', 'missing optional sips')), pipeline.Check(()))


@pytest.fixture
def shell(monkeypatch, outcome, tmp_path):
    state = SimpleNamespace(records=(pipeline.Normalised(outcome(execution='committed')),),
                            plans={}, options=None, requests=(), publications=[],
                            report_error=False, log=None, execute_calls=0)
    monkeypatch.setattr(cli.shutil, 'which', lambda value: '/tool/ffmpeg')
    tools = SimpleNamespace(executable=SimpleNamespace(version='synthetic'),
                            prober=SimpleNamespace(version='synthetic'))
    monkeypatch.setattr(pipeline, 'locate_normalise_tools', lambda options: tools)
    monkeypatch.setattr(pipeline, 'discover_inputs', lambda options: SimpleNamespace(
        items=tuple(record.outcome.input for record in state.records), hints=(), excluded={},
        duplicates=0, ignored_links=0, directory_links=0))

    class Outputs:
        def destination(self, purpose):
            request = next((r for r in state.requests if r.purpose == purpose), None)
            return SimpleNamespace(final_path=request.path) if request else None

        def publish(self, purpose, data):
            if state.report_error:
                raise pipeline.RunOutputError(state.report_error if isinstance(state.report_error, str)
                                              else 'synthetic publication failure')
            path = self.destination(purpose).final_path
            state.publications.append((path, data))
            return path

        def open_log(self):
            path = self.destination('log').final_path
            state.log = path
            return os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)

    def normalise(items, options, tools, **kwargs):
        state.options = options
        state.requests = tuple(kwargs['run_outputs'](state.plans))
        outputs = Outputs()
        if not options.dry_run:
            state.execute_calls += 1
            lock = threading.RLock()
            records = {r.outcome.input.input_index: r for r in state.records}
            request = pipeline.BatchRequest(tuple(r.outcome for r in state.records),
                lambda *a: pytest.fail('preflight entries require no worker'),
                lambda *a: pytest.fail('preflight entries require no settler'), options.jobs,
                lock, kwargs['cancel_event'], lambda: None,
                lambda item: records[item.input.input_index], outputs)
            result = kwargs['execute'](request)
            assert result.outcomes == tuple(r.outcome for r in state.records)
        return pipeline.NormaliseRun(state.records, 0, False, options.dry_run, outputs=outputs)

    monkeypatch.setattr(pipeline, 'normalise_items', normalise)
    return state


@pytest.mark.parametrize('alias', ['normalise', 'normalize'])
def test_normalise_alias_batch_and_automatic_report(alias, shell, capsys, tmp_path):
    assert cli.main([alias, 'take.wav', '-j', '2']) == 0
    captured = capsys.readouterr()
    assert shell.options.command == 'normalise' and shell.options.jobs == 2
    assert shell.execute_calls == 1
    assert len(shell.requests) == 1 and not shell.requests[0].parent_must_exist
    path, data = shell.publications[0]
    assert path.parent == tmp_path and path.name.startswith('norma-report-')
    assert data.startswith(b'input,status,reason,')
    assert captured.out.startswith('OK\t1/1\t')
    assert captured.out.splitlines()[-1].startswith('SUMMARY\t')
    assert f'report={path}' in captured.out
    assert 'no tags or processing mark' not in captured.err


@pytest.mark.parametrize('folders', [('one',), ('one', 'two'), ()])
def test_automatic_report_location_uses_plans_not_success(shell, folders, tmp_path):
    shell.plans = {i: SimpleNamespace(audio_member=SimpleNamespace(final_path=tmp_path / folder / 'take.wav'))
                   for i, folder in enumerate(folders)}
    shell.records = (replace(shell.records[0], outcome=replace(shell.records[0].outcome,
                     status='skipped', reason_code='output_exists', execution='not_written')),)
    assert cli.main(['normalise', 'take.wav']) == 0
    expected = tmp_path / 'one' if len(folders) == 1 else tmp_path
    assert shell.requests[0].path.parent == expected


def test_automatic_report_without_planned_destination_uses_out_dir_else_start_folder(shell, tmp_path, monkeypatch):
    # D-072 (IC-2 F-6): a lone within-tolerance SKIP plans no audio destination.
    start = tmp_path / 'start'
    start.mkdir()
    monkeypatch.chdir(start)
    shell.records = (replace(shell.records[0], outcome=replace(shell.records[0].outcome,
                     status='skipped', reason_code='already_within_tolerance', execution='not_written')),)
    assert cli.main(['normalise', 'take.wav', '-o', str(tmp_path / 'dest')]) == 0
    assert shell.requests[0].path.parent == tmp_path / 'dest'
    assert cli.main(['normalise', 'take.wav']) == 0
    assert shell.requests[0].path.parent.resolve() == start.resolve()


def test_c04_notes_reach_stderr_and_log_but_not_the_row(shell, capsys, tmp_path):
    note = '1 extra audio stream not carried over; only the first audio stream is processed'
    shell.records = (replace(shell.records[0], notes=(note,)),)
    assert cli.main(['normalise', 'take.wav', '--log', str(tmp_path / 'run.log')]) == 0
    captured = capsys.readouterr()
    assert 'note: ' + note in captured.err
    assert note not in captured.out
    assert 'note: ' + note in (tmp_path / 'run.log').read_text()
    assert cli.main(['normalise', 'take.wav', '-q']) == 0
    assert note not in capsys.readouterr().err


@pytest.mark.parametrize('explicit', [False, True])
def test_dry_run_no_executor_auto_report_or_log(shell, explicit, capsys, tmp_path):
    shell.records = (replace(shell.records[0], outcome=replace(shell.records[0].outcome, execution='plan')),)
    args = ['normalise', 'take.wav', '-n', '--log', 'ignored.log', '--keep-failed', '--yes']
    if explicit:
        args += ['--report', 'preview.json']
    before = set(tmp_path.rglob('*'))
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert shell.execute_calls == 0 and shell.log is None
    assert set(tmp_path.rglob('*')) == before
    assert len(shell.publications) == int(explicit)
    assert len(shell.requests) == int(explicit)
    assert 'note: normalise --dry-run writes no log file; --log ignored' in captured.err
    assert 'note: normalise --dry-run ignores --keep-failed' in captured.err
    assert 'note: normalise ignores --yes — --yes has no effect: Norma asks no questions' in captured.err
    if explicit:
        document = json.loads(shell.publications[0][1])
        assert document['outcomes'][0]['execution'] == 'plan'
        assert document['outcomes'][0]['out_lufs'] is None
        assert document['outcomes'][0]['published_paths'] == []
        assert document['report_path'] == str(tmp_path / 'preview.json')


def test_batch_uses_shared_lock_cancellation_and_opt_out_handler(shell, monkeypatch):
    def execute(entries, worker, **kwargs):
        assert kwargs['jobs'] == 3 and kwargs['handle_sigint'] is False
        assert kwargs['control'].cancel_event is not None
        assert kwargs['control'].force_hook is not None
        for item in entries:
            kwargs['on_outcome'](item)
        return batch.BatchResult(tuple(entries), 0, False)
    monkeypatch.setattr(batch, 'execute_batch', execute)
    old = signal.getsignal(signal.SIGINT)
    assert cli.main(['normalise', 'take.wav', '-j', '3']) == 0
    assert signal.getsignal(signal.SIGINT) is old


def test_explicit_json_envelope_and_append_log_share_run_id(shell, capsys, tmp_path):
    path = tmp_path / 'run.log'
    path.write_text('earlier run\n')
    assert cli.main(['normalise', 'take.wav', '--report', 'run.json', '--log', str(path), '--overwrite']) == 0
    captured = capsys.readouterr()
    assert shell.options.overwrite
    assert all(r.parent_must_exist for r in shell.requests)
    document = json.loads(shell.publications[0][1])
    lines = path.read_text().splitlines()
    assert lines[0] == 'earlier run'
    header = json.loads(lines[1])
    assert header['run_id'] == document['run_id']
    assert f'run_id={document["run_id"]}' in captured.out
    assert lines[-1].startswith('SUMMARY\t')
    assert header['options']['value_sources']['overwrite'] == 'cli'


@pytest.mark.parametrize('quiet', [False, True])
def test_report_failure_is_run_error_without_extra_file_row(shell, quiet, capsys):
    shell.report_error = True
    assert cli.main(['normalise', 'take.wav', '--report', 'run.json'] + (['-q'] if quiet else [])) == 1
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert captured.out.startswith('OK\t')
    assert 'run_errors=1\treport=\t' in captured.out
    assert 'could not write the report' in captured.err
    assert 'the files above are as their rows say' in captured.err
    assert shell.publications == []


def test_report_racer_is_kept_and_named_as_n07(shell, capsys):
    shell.report_error = 'run.json appeared while the run was working; it was left untouched'
    assert cli.main(['normalise', 'take.wav', '--report', 'run.json']) == 1
    captured = capsys.readouterr()
    assert 'appeared while the run was working and was kept — the report was not written' in captured.err
    assert 'run_errors=1\treport=\t' in captured.out and shell.publications == []


def test_dry_run_log_from_config_is_not_noted(shell, tmp_path, capsys):
    shell.records = (replace(shell.records[0], outcome=replace(shell.records[0].outcome, execution='plan')),)
    path = tmp_path / 'prefs.toml'
    path.write_text(f'log = "{tmp_path / "configured.log"}"\n')
    assert cli.main(['normalise', 'take.wav', '-n', '--config', str(path)]) == 0
    captured = capsys.readouterr()
    assert shell.options.log == tmp_path / 'configured.log'
    assert shell.options.value_sources['log'] == 'config'
    assert '--log ignored' not in captured.err
    assert not (tmp_path / 'configured.log').exists()


def test_interruption_restores_handler_reports_correct_counts_and_force(shell, outcome, monkeypatch, capsys):
    forced = []
    monkeypatch.setattr(pipeline, 'force_kill_tools', lambda: forced.append(True))
    records = tuple(pipeline.Normalised(o) for o in (
        outcome(0, execution='committed'),
        outcome(1, status='skipped', reason_code='cancelled', execution='not_written'),
        outcome(2, status='failed', reason_code='cleanup_error', execution='publication_failed',
                published_paths=(Path.cwd() / 'survivor.wav',))))
    def interrupted(items, options, tools, **kwargs):
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        assert kwargs['cancel_event'].is_set()
        return pipeline.NormaliseRun(records, 4, True, False)
    monkeypatch.setattr(pipeline, 'normalise_items', interrupted)
    old = signal.getsignal(signal.SIGINT)
    assert cli.main(['normalise', 'take.wav', '-q']) == 130
    captured = capsys.readouterr()
    assert forced == [True] and signal.getsignal(signal.SIGINT) is old
    assert 'Interrupted: 1 finished, 1 cancelled, 1 failed, 4 not started.' in captured.err
    assert 'partly published:' in captured.err and 'survivor.wav' in captured.err
    assert 'not_started=4' in captured.out


@pytest.mark.parametrize('exception,code', [(UsageError('synthetic barrier refusal'), 2),
                                           (EnvironmentError_('synthetic missing tool'), 3)])
def test_preflight_failure_prints_no_rows_or_report(shell, monkeypatch, capsys, exception, code):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(pipeline, 'normalise_items', fail)
    assert cli.main(['normalise', 'take.wav']) == code
    captured = capsys.readouterr()
    assert captured.out == '' and 'synthetic' in captured.err
    assert shell.publications == []


@pytest.mark.parametrize('argument', ['wrong.txt', 'https://example.test/run.csv'])
def test_invalid_report_before_tool_discovery(argument, shell, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'locate_normalise_tools', lambda *a: pytest.fail('no tools before argument validation'))
    assert cli.main(['normalise', 'take.wav', '--report', argument]) == 2
    assert capsys.readouterr().out == ''


def test_configured_metadata_request_passed_to_pipeline(shell, tmp_path, capsys):
    path = tmp_path / 'prefs.toml'
    path.write_text('artist = "Demo artist"\n')
    assert cli.main(['normalise', 'take.wav', '--config', str(path)]) == 0
    assert 'needs metadata support' not in capsys.readouterr().err
    assert shell.options.artist == 'Demo artist'
    assert shell.options.value_sources['artist'] == 'config'


@pytest.mark.parametrize('version,expected', [('N-125892', 'OK'), ('N-125892-g406c5a37aa', 'OK'),
                                             ('N-125892-g406c5a37aa-https://www.martin-riedl.de', 'OK'),
                                             ('N-125893-gabcd', 'WARN'), ('N-125892oops', 'WARN'),
                                             ('N-1258920-gabcd', 'WARN'), ('7.1', 'WARN')])
def test_doctor_compares_build_token_without_suppressing_capability_checks(monkeypatch, capsys, version, expected):
    checked = doctor_snapshot(version)
    monkeypatch.setattr(pipeline, 'inspect_environment', lambda *a: checked)
    assert cli.main(['doctor']) == 0
    captured = capsys.readouterr()
    assert f'{expected}\ttested FFmpeg\t{version};' in captured.out
    assert 'OK\tfilters\t' in captured.out and 'OK\tencoders\t' in captured.out


@pytest.mark.parametrize('fail_at', [1, 3])
def test_log_error_even_during_final_summary_contributes_to_exit(shell, monkeypatch, capsys, fail_at):
    real_write = os.write
    calls = []
    def write(fd, data):
        calls.append(data)
        if len(calls) == fail_at:
            raise OSError('synthetic log full')
        return real_write(fd, data)
    monkeypatch.setattr(cli.os, 'write', write)
    assert cli.main(['normalise', 'take.wav', '--log', 'run.log', '-q']) == 1
    captured = capsys.readouterr()
    assert 'cannot append log' in captured.err
    assert 'run_errors=1' in captured.out and captured.out.startswith('OK\t')


def test_missing_ffmpeg_is_environment_exit_before_pipeline(shell, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, 'which', lambda value: None)
    assert cli.main(['normalise', 'take.wav']) == 3
    captured = capsys.readouterr()
    assert captured.out == '' and 'FFmpeg not found' in captured.err
    assert shell.options is None


def test_progress_is_stderr_verbose_only_when_piped(shell, capsys):
    assert cli.main(['normalise', 'take.wav', '-v']) == 0
    captured = capsys.readouterr()
    assert '0 started, 1/1 finished:' in captured.err
    assert '0 started, 1/1 finished:' not in captured.out


def test_doctor_build_match_never_overrides_missing_filter(monkeypatch, capsys):
    checked = replace(doctor_snapshot('N-125892-g406c5a37aa'),
        filters=pipeline.Check(problem=pipeline.Problem('missing', 'ebur128 is absent')))
    monkeypatch.setattr(pipeline, 'inspect_environment', lambda *a: checked)
    assert cli.main(['doctor']) == 3
    captured = capsys.readouterr()
    assert 'OK\ttested FFmpeg\t' in captured.out and 'FAIL\tfilters\t' in captured.out
