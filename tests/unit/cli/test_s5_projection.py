"""S5 read-only report and presentation contracts, using synthetic measurements."""
import csv
import io
import json
from dataclasses import replace

import pytest

from norma import pipeline, report
from norma.models import REPORT_COLUMNS, SUMMARY_KEYS, Diagnostic, ProcessingMark, VerificationReference


@pytest.fixture
def inspected(outcome):
    def make(origin='mark', validity='valid', mode='linear', level=-16.2, reason='ok', status='ok'):
        mark = ProcessingMark(validity, -16 if validity == 'valid' else None,
                              mode if validity == 'valid' else None, None, None)
        reference = VerificationReference(-16, origin, -1, 'cli', .1, 'config', mark)
        measured = replace(outcome().measurement_in, integrated_lufs=level)
        verification = pipeline.assess_inspection(measured, reference)
        return outcome(measurement_in=None, measurement_out=measured, verification=verification,
                       verification_reference=reference, status=status, reason_code=reason)
    return make


@pytest.mark.parametrize('origin,validity', [('cli', 'malformed'), ('mark', 'valid'),
                                           ('config', 'unsupported'), ('default', 'absent')])
def test_verify_projection_uses_per_file_reference_not_run_defaults(inspected, options, origin, validity):
    item = inspected(origin, validity)
    row = report.project_outcome(item, options)
    assert tuple(row) == REPORT_COLUMNS and len(row) == 53
    assert row['out_path'] == str(item.input.source_path)
    assert (row['target_lufs'], row['target_origin']) == (-16, origin)
    assert (row['ceiling_dbtp'], row['ceiling_origin']) == (-1, 'cli')
    assert (row['tolerance_lu'], row['tolerance_origin']) == (.1, 'config')
    assert row['mark_validity'] == validity
    assert row['out_lufs'] == -16.2 and row['execution'] == 'inspect'
    assert row['published_paths'] == []
    for key in ('in_lufs', 'in_dbtp', 'in_lra', 'gain_db', 'mode', 'limited_db',
                'expected_lufs', 'predicted_shortfall_lu', 'loudness_corrections',
                'peak_corrections', 'reduce_fallbacks', 'encoder', 'out_format'):
        assert row[key] is None, key


@pytest.mark.parametrize('origin,label', [('cli', 'command line'), ('mark', 'processing mark'),
                                        ('config', 'config'), ('default', 'default')])
def test_verify_row_shows_signed_deviation_inspected_path_and_target_even_when_ok(inspected, origin, label):
    item = inspected(origin)
    cells = report.row_cells(item, 1)
    assert cells[3:7] == ['', '', '-0.2', 'verify']
    assert cells[7:10] == ['-16.2', '-10.0', str(item.input.source_path)]
    assert f'target -16 ({label})' in cells[-1]
    assert 'file not processed' not in cells[-1]


def test_reduce_has_actual_deviation_and_shortfall_without_prediction(inspected, options):
    item = inspected(mode='reduce', level=-23, reason='reduce_shortfall', status='warn')
    assert item.verification.expected_lufs is None and item.verification.loudness_error_lu is None
    row = report.project_outcome(item, options)
    assert row['expected_lufs'] is None and row['predicted_shortfall_lu'] is None
    assert row['measured_shortfall_lu'] == 7
    cells = report.row_cells(item, 1)
    assert cells[5:7] == ['-7.0', 'verify']
    assert 'reduced output, peaks safe; measures -23.0 LUFS, 7.00 LU below -16' in cells[-1]
    assert 'processing mark' in cells[-1]


def test_verify_csv_json_escape_null_and_summary_shapes(inspected, envelope):
    item = inspected()
    item = replace(item, input=replace(item.input, source_path=item.input.source_path.parent / 'a\t"b\\c\n.wav'))
    run = envelope((item,))
    csv_row, = csv.DictReader(io.StringIO(report.render_csv(run)))
    json_row, = json.loads(report.render_json(run))['outcomes']
    assert len(csv_row) == len(json_row) == 53
    assert csv_row['out_path'] == json_row['out_path'] == str(item.input.source_path)
    assert csv_row['gain_db'] == '' and json_row['gain_db'] is None
    assert csv_row['published_paths'] == '[]' and json_row['published_paths'] == []
    assert tuple(cell.split('=')[0] for cell in report.render_summary(run).split('\t')[1:]) == SUMMARY_KEYS
    assert len(report.render_tsv(item, 1).splitlines()) == 1
    assert len(report.render_table((item,), 1, item.input.root).splitlines()) == 2


def test_precise_mismatches_use_typed_cells_and_actuals_without_comparing_again(inspected):
    from norma.models import Artwork, ManifestCell, SourceTags, TagValue
    refs = tuple(pipeline.InspectionReference(name, ManifestCell(operation, value), origin, locator)
        for name, operation, value, origin, locator in [
            ('title', 'set', 'Expected\tTitle', 'cli', '--title'),
            ('compilation', 'set', False, 'manifest', 'refs.json:take.wav'),
            ('track_total', 'set', 12, 'manifest_default', 'refs.json:defaults'),
            ('album', 'clear', None, 'manifest', 'refs.json:take.wav'),
            ('Mood', 'set', 'calm', 'manifest', 'refs.json:take.wav'),
            ('artwork', 'set', inspected().input.source_path.parent / 'cover.png', 'manifest', 'refs.json:take.wav')])
    picture = Artwork(b'never print embedded payload', 'image/png', 1, 1, 3, '')
    tags = SourceTags(fields={name: TagValue(value, 'synthetic', name) for name, value in
        [('title', 'Actual'), ('compilation', True), ('track_total', 9), ('album', 'Album')]},
        extras={'mOOD': TagValue('loud', 'synthetic', 'mOOD')},
        artwork=(TagValue(picture, 'synthetic', 'cover'),), non_tag_items=(), internal_conflicts=(),
        native_inventory=(), stores={})
    record = pipeline.Inspection(inspected(reason='verification_tags', status='failed'), source_tags=tags,
        tag_references=refs, tag_mismatches=('field:title', 'field:compilation', 'field:track_total',
        'field:album', 'extra:MOOD', 'field:artwork', 'extra:NORMA'))
    details = report.verification_details(record)
    assert len(details) == 7
    text = '\n'.join(details)
    for wanted in ('expected "Expected\\tTitle" from the command line (--title), found "Actual"',
                   'expected false from the manifest (refs.json:take.wav), found true',
                   'track_total: expected 12 from the manifest defaults', 'album: expected absent',
                   'Mood: expected "calm"', 'found "loud"', 'byte-exact front cover',
                   'SHA-256', 'processing mark missing'):
        assert wanted in text
    assert 'never print embedded payload' not in text and "b'" not in text
    assert 'target -16 (processing mark)' in report.diagnostic_note(record.outcome, notes=details)
