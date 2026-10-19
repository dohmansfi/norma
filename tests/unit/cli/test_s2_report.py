"""S2 publication and measured-versus-predicted presentation, using immutable synthetic records."""
from dataclasses import replace
from pathlib import Path

import pytest

from norma import report
from norma.audio.plan import plan_gain
from norma.models import BundleMember, OutputPlan, Verification


@pytest.fixture
def planned(outcome):
    original = outcome()
    destination = Path.cwd() / 'Normalised/take.wav'
    member = BundleMember('audio', 'audio', None, destination, None, 'synthetic-key')
    return replace(original, output=OutputPlan(1, (member,), 'none', original.input.identity, False))


@pytest.mark.parametrize('execution,visible', [('plan', False), ('not_written', False), ('committed', True)])
def test_only_actual_audio_publication_fills_piped_output(planned, options, execution, visible):
    path = planned.output.audio_member.final_path
    result = replace(planned, execution=execution, published_paths=(path,) if visible else ())
    assert report.row_cells(result, 1)[9] == (str(path) if visible else '')
    assert report.project_outcome(result, options)['out_path'] == str(path)
    if execution == 'plan':
        assert report.row_cells(result, 1, terminal=True)[9] == str(path)


def test_partial_publication_lists_survivors_never_claims_nothing_written(planned):
    surviving = Path.cwd() / 'Normalised/take.wav.tags.json'
    result = replace(planned, status='failed', reason_code='cleanup_error',
                     execution='publication_failed', published_paths=(surviving,))
    note = report.diagnostic_note(result)
    assert 'partly published:' in note and str(surviving) in note
    assert 'nothing written' not in note and 'no output written' not in note
    assert report.row_cells(result, 1)[9] == str(surviving)


def test_retained_failed_and_restored_output_clauses(planned):
    result = replace(planned, status='failed', reason_code='verification_peak', execution='not_written',
                     retained_failed_path=Path.cwd() / 'Normalised/failed.wav')
    assert 'failed candidate kept at' in report.diagnostic_note(result)
    restored = replace(result, retained_failed_path=None)
    note = 'rolled back; the earlier output /earlier.wav was restored'
    assert note in report.diagnostic_note(restored, notes=(note,))


def test_reduce_measured_shortfall_not_replaced_by_prediction(planned, options):
    source = replace(planned.measurement_in, true_peak_dbtp=-2, true_peak_linear=0.7943282347242815)
    gain = plan_gain(source, replace(options, peak_mode='reduce'))
    measured = replace(source, integrated_lufs=-21.75)
    verified = Verification(measured, gain.expected_integrated_lufs, None, 5.75, 0, True, True, True)
    result = replace(planned, status='warn', reason_code='reduce_shortfall', gain=gain,
                     measurement_in=source, measurement_out=measured, verification=verified,
                     execution='committed', published_paths=(planned.output.audio_member.final_path,))
    row = report.project_outcome(result, options)
    assert row['predicted_shortfall_lu'] == 6 and row['measured_shortfall_lu'] == 5.75
    assert row['out_lufs'] == -21.75
    assert '5.75 LU below -16' in report.diagnostic_note(result) and 'measures -21.8 LUFS' in report.diagnostic_note(result)
    assert 'would' not in report.diagnostic_note(result)
    preview = replace(result, execution='plan', measurement_out=None, verification=None, published_paths=())
    assert report.row_cells(preview, 1)[7:10] == ['', '', '']
    assert report.row_cells(preview, 1, terminal=True)[7].startswith('~')
    assert 'would reduce' in report.diagnostic_note(preview) and '6.00 LU below -16' in report.diagnostic_note(preview)
