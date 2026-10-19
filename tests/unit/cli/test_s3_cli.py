"""S3 interface seams with synthetic records: dynamic, lossy options, labels and notes (TK-0309)."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma import cli, pipeline, report
from norma.audio.plan import plan_gain
from norma.models import Diagnostic, GainPlan, TargetFormat, Verification

from .test_s2_cli import shell  # noqa: F401  (fixture reuse: a fake pipeline around the real CLI)
from .test_cli import fake_run  # noqa: F401  (read-only pipeline seam)


def preview_gain(**changes):
    values = dict(action='dynamic', requested_peak_mode='dynamic', target_lufs=-16.0, ceiling_dbtp=-1.0,
                  tolerance_lu=0.5, max_limit_db=3.0, gain_db=8.0, predicted_true_peak_dbtp=None,
                  limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=-1.0,
                  expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                  peak_corrections=0, reduce_fallbacks=0, reason_code='ok')
    values.update(changes)
    return GainPlan(**values)


def test_normalise_accepts_dynamic_and_passes_it_to_the_pipeline(shell, capsys):  # noqa: F811
    assert cli.main(['normalise', 'take.wav', '--peak-mode', 'dynamic', '--config', '/dev/null']) == 0
    assert shell.options.peak_mode == 'dynamic'
    assert capsys.readouterr().out.startswith('OK\t')


def test_dynamic_from_the_config_reaches_a_real_normalise_run(shell, tmp_path, capsys):  # noqa: F811
    config = tmp_path / 'config.toml'
    config.write_text('peak_mode = "dynamic"\n')
    assert cli.main(['normalise', 'take.wav', '--config', str(config)]) == 0
    assert shell.options.peak_mode == 'dynamic' and shell.options.value_sources['peak_mode'] == 'config'


@pytest.mark.parametrize('alias', ['analyse', 'analyze'])
@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_analyse_dynamic_preview_reaches_pipeline_and_labels_rows(
        alias, configured, terminal, fake_run, monkeypatch, outcome, tmp_path, capsys):  # noqa: F811
    config = tmp_path / 'dynamic.toml'
    config.write_text('peak_mode = "dynamic"\n')
    args = ['--config', str(config)] if configured else ['--config', '/dev/null', '--peak-mode', 'dynamic']
    item = pipeline.Inspection(outcome(gain=preview_gain()))

    def analyse(items, options, tools, **kwargs):
        assert options.peak_mode == 'dynamic'
        kwargs['on_inspection'](item)
        return pipeline.ReadOnlyRun((item,), 0, False)

    monkeypatch.setattr(pipeline, 'analyse_items', analyse)
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    assert cli.main([alias, 'take.wav', *args]) == 0
    assert fake_run[0].value_sources['peak_mode'] == ('config' if configured else 'cli')
    captured = capsys.readouterr()
    assert 'dynamic (calibrated at render)' in captured.out
    assert 'available in a later version' not in captured.err


@pytest.mark.parametrize('alias', ['normalise', 'normalize'])
@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_dynamic_dry_run_passes_policy_without_execution_or_automatic_writes(
        alias, configured, terminal, shell, monkeypatch, tmp_path, capsys):  # noqa: F811
    config = tmp_path / 'dynamic.toml'
    config.write_text('peak_mode = "dynamic"\n')
    args = ['--config', str(config)] if configured else ['--config', '/dev/null', '--peak-mode', 'dynamic']
    record = shell.records[0]
    shell.records = (replace(record, outcome=replace(record.outcome, execution='plan', gain=preview_gain())),)
    monkeypatch.setattr(cli.sys.stdout, 'isatty', lambda: terminal)
    before = set(tmp_path.rglob('*'))
    assert cli.main([alias, 'take.wav', '-n', '--log', str(tmp_path / 'ignored.log'), *args]) == 0
    assert shell.options.peak_mode == 'dynamic' and shell.options.dry_run
    assert shell.options.value_sources['peak_mode'] == ('config' if configured else 'cli')
    assert shell.execute_calls == 0 and shell.requests == () and shell.publications == [] and shell.log is None
    assert set(tmp_path.rglob('*')) == before
    captured = capsys.readouterr()
    assert 'dynamic (calibrated at render)' in captured.out
    assert 'writes no log file; --log ignored' in captured.err


@pytest.mark.parametrize('command', ['analyse', 'normalise'])
def test_help_no_longer_says_dynamic_or_lossy_arrive_later(command, capsys):
    assert cli.main([command, '--help']) == 0
    text = capsys.readouterr().out
    assert 'dynamic arrives later' not in text and 'lossy writes' not in text and 'limiter/lossy' not in text
    assert 'limit | reduce | dynamic' in text
    if command == 'normalise':
        assert 'mp3 | aac' in text


def test_dynamic_dry_run_explicit_report_keeps_plain_mode_and_unmeasured_output(shell, tmp_path, capsys):  # noqa: F811
    record = shell.records[0]
    shell.records = (replace(record, outcome=replace(record.outcome, execution='plan', gain=preview_gain())),)
    assert cli.main(['normalise', 'take.wav', '--dry-run', '--peak-mode', 'dynamic',
                     '--config', '/dev/null', '--report', str(tmp_path / 'preview.json')]) == 0
    assert shell.execute_calls == 0 and len(shell.publications) == 1
    row = json.loads(shell.publications[0][1])['outcomes'][0]
    assert row['mode'] == 'dynamic' and row['execution'] == 'plan'
    assert row['out_lufs'] is None and row['out_dbtp'] is None and row['published_paths'] == []
    assert 'dynamic (calibrated at render)' in capsys.readouterr().out


def test_mode_cell_labels_a_dynamic_plan_without_calibration(outcome):
    preview = outcome(execution='plan', gain=preview_gain())
    assert report.row_cells(preview, 1)[6] == 'dynamic (calibrated at render)'
    assert report.render_tsv(preview, 1).split('\t')[6] == 'dynamic (calibrated at render)'
    calibrated = replace(preview, gain=preview_gain(dynamic_parameters=dict(
        input_i=-24.0, input_tp=-10.0, input_lra=2.0, input_thresh=-34.0, target_offset=0.0)))
    assert report.row_cells(calibrated, 1)[6] == 'dynamic'
    assert report.row_cells(replace(preview, gain=preview_gain(action='linear', requested_peak_mode='limit',
        limiter_threshold_dbtp=None)), 1)[6] == 'linear'


def test_report_mode_column_keeps_the_plain_action(outcome, options):
    row = report.project_outcome(outcome(execution='plan', gain=preview_gain()), options)
    assert row['mode'] == 'dynamic'


def limit_failed(outcome, options, **gain_changes):
    original = outcome()
    measurement = replace(original.measurement_in, integrated_lufs=-24.0, true_peak_dbtp=-2.0, true_peak_linear=0.794)
    gain = plan_gain(measurement, replace(options, peak_mode='reduce', target=-16.0))
    gain = replace(gain, requested_peak_mode='limit', max_limit_db=3.0, fallback_from='limit', **gain_changes)
    return replace(original, status='warn', reason_code='reduce_shortfall', gain=gain, measurement_in=measurement)


def test_limit_fallback_notes_give_the_cap_reason_and_the_measured_shortfall(outcome, options):
    result = limit_failed(outcome, options)
    out = replace(result.measurement_in, integrated_lufs=-19.2, true_peak_dbtp=-1.0)
    verified = Verification(out, -16.0, None, 3.2, 0, True, True, True)
    done = replace(result, measurement_out=out, verification=verified, execution='committed')
    note = report.diagnostic_note(done)
    assert note == ('reduce_shortfall: limit could not reach the target (cap 3 dB) — reduced to keep peaks at -1.0 dBTP; '
                    'measures -19.2 LUFS, 3.20 LU below -16')
    on_target = replace(done, verification=replace(verified, measured_shortfall_lu=0.0))
    assert report.diagnostic_note(on_target).endswith('measures -19.2 LUFS, on target, but by reduction')
    # An excess within the cap with no spent corrections (F-05/c6 shape: a cap-rejected correction) gets no reason.
    within = replace(done, measurement_in=replace(done.measurement_in, true_peak_dbtp=-10.0))
    note = report.diagnostic_note(within)
    assert 'corrections used up' not in note and '(cap' not in note
    assert note.startswith('reduce_shortfall: limit could not reach the target — reduced to keep peaks at -1.0 dBTP;')
    # Only a partly spent budget cannot name the cause either; both spent is genuinely exhausted.
    partly = replace(within, gain=replace(within.gain, loudness_corrections=2))
    assert '(' not in report.diagnostic_note(partly).split('—')[0]
    spent = replace(within, gain=replace(within.gain, loudness_corrections=2, peak_corrections=1))
    assert 'target (corrections used up) —' in report.diagnostic_note(spent)


def test_limit_fallback_prediction_says_how_much_limiting_it_would_need(outcome, options):
    preview = replace(limit_failed(outcome, options), execution='plan')
    note = report.diagnostic_note(preview)
    assert note.startswith('reduce_shortfall: would need 7.0 dB of limiting (cap 3) — would reduce instead, to about ')
    assert note.endswith('LU below -16')
    assert 'measured' not in note and 'measures' not in note


@pytest.mark.parametrize('mode,typed,shown', [('reduce', True, True), ('dynamic', True, True),
                                             ('reduce', False, False), ('limit', True, False)])
def test_typed_max_limit_with_a_capless_mode_gets_an_n01_note(shell, capsys, tmp_path, mode, typed, shown):  # noqa: F811
    config = tmp_path / 'config.toml'
    config.write_text(f'peak_mode = "{mode}"\nmax_limit = 2\n')
    args = ['normalise', 'take.wav', '--config', str(config)]
    if typed:
        args = ['normalise', 'take.wav', '--config', '/dev/null', '--peak-mode', mode, '--max-limit', '2']
        if mode == 'limit':
            args = args[:-2] + ['--max-limit', '2']
    assert cli.main(args) == 0
    err = capsys.readouterr().err
    assert (f'note: normalise ignores --max-limit — --max-limit has no effect with --peak-mode {mode}' in err) is shown


def test_max_limit_note_is_hidden_by_quiet(shell, capsys):  # noqa: F811
    assert cli.main(['normalise', 'take.wav', '--config', '/dev/null', '--peak-mode', 'reduce', '--max-limit', '2', '-q']) == 0
    assert 'ignores --max-limit' not in capsys.readouterr().err


def flac_record(shell_state, target_format):  # noqa: F811
    outcome = shell_state.records[0].outcome
    target = TargetFormat(target_format, '.' + target_format, 'synthetic', None, False, 48000, 2, 'stereo',
                          'none', 'none', None, None, None, 'not_applicable', 'synthetic', False, 'as_requested')
    output = SimpleNamespace(audio_member=SimpleNamespace(final_path=Path.cwd() / f'take.{target_format}'))
    return replace(shell_state.records[0], outcome=replace(outcome, target=target, output=output))


def test_flac_note_once_per_run_on_stderr_and_in_the_log(shell, capsys, tmp_path):  # noqa: F811
    first = flac_record(shell, 'flac')
    second = replace(first, outcome=replace(first.outcome, input=replace(first.outcome.input, input_index=1)))
    shell.records = (first, second)
    log = tmp_path / 'run.log'
    assert cli.main(['normalise', 'a.wav', 'b.wav', '--config', '/dev/null', '--log', str(log)]) == 0
    captured = capsys.readouterr()
    assert captured.err.count('note: ' + cli.FLAC_NOTE) == 1
    assert log.read_text().count('note: ' + cli.FLAC_NOTE) == 1
    assert cli.main(['normalise', 'a.wav', 'b.wav', '--config', '/dev/null', '-q']) == 0
    assert cli.FLAC_NOTE not in capsys.readouterr().err


def test_no_flac_note_without_a_flac_output(shell, capsys):  # noqa: F811
    shell.records = (flac_record(shell, 'wav'),)
    assert cli.main(['normalise', 'take.wav', '--config', '/dev/null']) == 0
    assert 'FLAC is not among' not in capsys.readouterr().err


def test_flac_note_ignores_a_flac_file_that_was_never_planned_an_output(shell, capsys):  # noqa: F811
    record = flac_record(shell, 'flac')
    shell.records = (replace(record, outcome=replace(record.outcome, status='skipped', reason_code='output_exists', output=None, execution='not_written')),)
    assert cli.main(['normalise', 'take.wav', '--config', '/dev/null']) == 0
    assert 'FLAC is not among' not in capsys.readouterr().err


def test_terminal_summary_counts_lossy_origin_files(outcome, envelope):
    origin = Diagnostic('lossy_origin', 'lossy mp3 source stored losslessly', 'format')
    rows = [outcome(0, status='warn', reason_code='lossy_origin', warnings=(origin,)), outcome(1)]
    text = report.render_summary(envelope(rows), terminal=True)
    assert text.startswith('1 OK, 1 WARN, 0 SKIP, 0 FAIL; 1 lossy origin; ')
    assert 'lossy origin' not in report.render_summary(envelope([outcome()]), terminal=True)
    # The piped SUMMARY keys are fixed (C-09, D-033): no lossy-origin key appears.
    assert 'lossy' not in report.render_summary(envelope(rows))


def test_cli_uses_the_pipeline_automatic_report_folder(shell, monkeypatch, tmp_path):  # noqa: F811
    calls = []

    def folder(plans, options):
        calls.append((plans, options.command))
        return tmp_path / 'chosen'

    monkeypatch.setattr(pipeline, 'automatic_report_folder', folder)
    shell.plans = {0: SimpleNamespace(audio_member=SimpleNamespace(final_path=tmp_path / 'one' / 'take.wav'))}
    assert cli.main(['normalise', 'take.wav', '--config', '/dev/null']) == 0
    assert calls == [(shell.plans, 'normalise')]
    assert shell.requests[0].path.parent == tmp_path / 'chosen'
    assert shell.requests[0].path.name.startswith('norma-report-')


def test_cli_keeps_no_copy_of_the_report_folder_rule():
    source = Path(cli.__file__).read_text()
    assert 'automatic_report_folder' in source
    assert 'folders = {plan.audio_member' not in source
