"""Read-only C-09 projections and script-safe C-13 output."""
import csv
import io
import json
import math
from dataclasses import replace

import pytest

from norma import report
from norma.audio.plan import plan_gain
from norma.models import Diagnostic, REPORT_COLUMNS, SUMMARY_KEYS


def test_53_columns_and_sorted_csv_json(outcome, envelope):
    run = envelope([outcome(1, 'b.wav'), outcome(0, 'a.wav')])
    rows = list(csv.DictReader(io.StringIO(report.render_csv(run))))
    decoded = json.loads(report.render_json(run))
    assert tuple(rows[0]) == REPORT_COLUMNS and len(rows[0]) == 53
    assert [row['input_index'] for row in rows] == ['0', '1']
    assert [row['input_index'] for row in decoded['outcomes']] == [0, 1]
    assert decoded['outcomes'][0]['out_path'] is None
    assert decoded['outcomes'][0]['out_lufs'] is None
    assert decoded['outcomes'][0]['execution'] == 'inspect'
    assert decoded['outcomes'][0]['published_paths'] == []
    assert decoded['options']['value_sources']['target'] == 'config'
    assert decoded['run_id'] == 'synthetic-run'


def test_prediction_never_fills_measured_output(outcome, options):
    original = outcome()
    gain = plan_gain(original.measurement_in, options)
    row = report.project_outcome(replace(original, gain=gain), options)
    assert row['gain_db'] == 7
    assert row['expected_lufs'] == -16
    assert row['out_lufs'] is None and row['out_dbtp'] is None
    assert row['target_origin'] == 'config'
    assert row['measured_shortfall_lu'] is None


def test_reduce_shortfall_is_predicted(outcome, options):
    original = outcome()
    measurement = replace(original.measurement_in, true_peak_dbtp=-2, true_peak_linear=0.7943282347242815)
    gain = plan_gain(measurement, replace(options, peak_mode='reduce'))
    result = replace(original, gain=gain, status='warn', reason_code='reduce_shortfall', warnings=gain.diagnostics)
    note = report.diagnostic_note(result)
    assert note == 'reduce_shortfall: would reduce so peaks stay under -1 dBTP — about -22.0 LUFS, 6.00 LU below -16'
    row = report.project_outcome(result, options)
    assert row['predicted_shortfall_lu'] == 6
    assert row['measured_shortfall_lu'] is None


def test_escaped_tsv_has_eleven_cells_one_line(outcome):
    original = outcome(name='tab\tline\nslash\\.wav', status='failed', reason_code='internal')
    diagnostic = Diagnostic('internal', 'bad\tline\nslash\\', 'tags_read')
    text = report.render_tsv(original, 1, primary=diagnostic)
    assert len(text.split('\t')) == 11 and len(text.splitlines()) == 1
    assert '\\t' in text and '\\n' in text and '\\\\' in text
    assert '\x1b' not in text
    assert text.startswith('FAIL\t1/1\t')


def test_summary_exact_order_counts_run_errors_actual_path(outcome, envelope):
    run = envelope([outcome(), outcome(1, status='failed', reason_code='internal')], not_started=2,
                   ignored=3, run_errors=('one', 'two'))
    line = report.render_summary(run)
    cells = line.split('\t')
    assert cells[0] == 'SUMMARY'
    assert tuple(cell.split('=')[0] for cell in cells[1:]) == SUMMARY_KEYS
    assert 'ok=1' in cells and 'failed=1' in cells and 'run_errors=2' in cells and 'report=' in cells


def test_silence_json_has_no_nonstandard_infinity(outcome, envelope):
    original = outcome()
    silent = replace(original.measurement_in, signal_state='silent', integrated_lufs=None, integrated_valid=False,
                     true_peak_dbtp=-math.inf, true_peak_linear=0, decoded_nonzero=False)
    decoded = json.loads(report.render_json(envelope([replace(original, measurement_in=silent)])))
    assert decoded['outcomes'][0]['in_dbtp'] is None
    assert decoded['outcomes'][0]['in_signal_state'] == 'silent'


def test_terminal_header_relative_path_and_no_colour(outcome, monkeypatch):
    original = outcome(name='x' * 100 + '.wav')
    monkeypatch.setenv('NO_COLOR', '')
    text = report.render_table([original], 1, original.input.root, colour=True)
    assert text.splitlines()[0].startswith('STATUS')
    assert '…' in text and '\x1b' not in text and str(original.input.root) not in text


def test_no_binary_payload_or_arbitrary_object():
    with pytest.raises(TypeError, match='binary'):
        report.portable(b'private bytes')
    with pytest.raises(TypeError, match='unsupported'):
        report.portable(object())
