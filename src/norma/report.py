"""Pure C-09/C-13 projections; this module never prints or opens a file."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

from norma.models import REPORT_COLUMNS, SUMMARY_KEYS, Artwork, FileOutcome, InventoryRef, RunOptions, RunReport, Status

STATUS_WORDS: dict[Status, str] = {'ok': 'OK', 'warn': 'WARN', 'skipped': 'SKIP', 'failed': 'FAIL'}
ROW_HEADERS = ('STATUS', '#', 'file', 'in_LUFS', 'in_dBTP', 'gain', 'mode', 'out_LUFS', 'out_dBTP', 'output', 'note')


def escape_cell(value: object) -> str:
    return str(value).replace('\\', '\\\\').replace('\t', '\\t').replace('\r', '\\r').replace('\n', '\\n')


def portable(value):
    """Keep canonical types; never stringify a library object or serialise binary payloads."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        raise TypeError('binary payloads do not belong in a report')
    if isinstance(value, Mapping):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        ordered = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [portable(v) for v in ordered]
    if isinstance(value, Artwork):
        return {f.name: portable(getattr(value, f.name)) for f in fields(value) if f.name != 'data'}
    if is_dataclass(value):
        return {f.name: portable(getattr(value, f.name)) for f in fields(value)}
    raise TypeError(f'unsupported report value: {type(value).__name__}')


def _rollup(value, plan):
    if not isinstance(value, InventoryRef):
        return portable(value)
    inventory = getattr(plan, value.scope)
    indexed = {item.identity: item for item in inventory}
    items = []
    for identity in value.identities:
        item = indexed[identity]  # unresolved IDs are a contract violation, never a guessed descriptor
        if value.scope == 'non_tag_items':
            descriptor = {key: getattr(item, key) for key in ('identity', 'kind', 'locator', 'detail', 'payload_size', 'payload_sha256')}
        else:
            descriptor = {key: getattr(item, key) for key in ('identity', 'store', 'key', 'native_type', 'native_version', 'payload_encoding')}
            descriptor.update(payload_size=len(item.payload), payload_sha256=hashlib.sha256(item.payload).hexdigest())
        items.append(descriptor)
    return {'kind': 'inventory_ref', 'scope': value.scope, 'identities': list(value.identities), 'items': items}


def _losses(plan):
    if plan is None:
        return []
    return [{'key': item.key, 'loss_kind': item.loss_kind, 'requested': _rollup(item.requested, plan),
             'effective': _rollup(item.effective, plan), 'detail': item.detail,
             'source_native_ids': list(item.source_native_ids), 'expected_native_ids': list(item.expected_native_ids)}
            for item in plan.statuses if item.status == 'dropped_format_limit']


def project_metadata(plan, *, verbose=False) -> dict:
    """Portable plan evidence, dereferenced only within this immutable revision."""
    result = dict(plan_revision=plan.plan_revision, audio_basis=plan.audio_basis,
                  mark_state=plan.mark_state, loss_policy_result=plan.loss_policy_result,
                  resolved={key: portable(value) for key, value in plan.resolved.items()},
                  conflicts=portable(plan.conflicts), dropped=_losses(plan),
                  statuses=[dict(key=item.key, status=item.status, requested=_rollup(item.requested, plan),
                                 effective=_rollup(item.effective, plan), provenance=item.provenance,
                                 loss_kind=item.loss_kind, detail=item.detail,
                                 source_native_ids=list(item.source_native_ids),
                                 expected_native_ids=list(item.expected_native_ids)) for item in plan.statuses])
    if verbose:
        result['native_statuses'] = [dict(portable(item),
            source_items=_rollup(InventoryRef('source_native', item.source_ids), plan)['items'] if item.source_ids else [],
            expected_items=_rollup(InventoryRef('expected_native', item.expected_ids), plan)['items'] if item.expected_ids else [])
            for item in plan.native_statuses]
    return result


def render_tag_plan(outcome, *, verbose=False, options=None, diagnostics=()):
    """Documentary detail follows the ordinary row; no status rows or publications invented."""
    branches = [(branch.assumption, branch.metadata, branch.output) for branch in outcome.conditional_plans
                if branch.refusal is None]
    if outcome.metadata is not None:
        branches = [('planned', outcome.metadata, outcome.output)]
    lines = []
    projected = [(label, project_metadata(plan, verbose=verbose), output) for label, plan, output in branches]
    common = []
    if len(projected) > 1:
        common = [row for row in projected[0][1]['statuses']
                  if all(row in branch[1]['statuses'] for branch in projected[1:])]
    def emit(kind, branch, value):
        lines.append('\t'.join(escape_cell(v) for v in (kind, outcome.input.source_path, branch,
            json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))))
    for row in common:
        emit('TAG', 'same in both alternatives', row)
    refusals = {(branch.assumption, branch.refusal.reason_code, branch.refusal.stage, branch.refusal.detail)
                for branch in outcome.conditional_plans if branch.refusal is not None}
    for branch in outcome.conditional_plans:
        if branch.refusal is not None:
            label = {'stream_copy': 'if stream-copied', 'encode': 'if encoded'}[branch.assumption]
            emit('REFUSAL', label, portable(branch.refusal))
    for label, plan, output in projected:
        branch = {'stream_copy': 'if stream-copied', 'encode': 'if encoded'}.get(label, label)
        ranks = {'cli': 'E1', 'generated': 'E1', 'manifest': 'E2', 'manifest_default': 'E3',
                 'config': 'E4', 'source': 'source', 'filename': 'filename', 'fallback': 'last resort'}
        resolved = {key: dict(value, rank=('E4' if key == 'artwork' and options is not None
                    and options.folder_artwork and value['provenance'] in ('cli', 'config')
                    else ranks[value['provenance']])) for key, value in plan['resolved'].items()}
        emit('PLAN', branch, dict(plan_revision=plan['plan_revision'], mark=plan['mark_state'],
                                 resolved=resolved, output=portable(output)))
        for row in plan['statuses']:
            if row not in common:
                emit('TAG', branch, row)
        for conflict in plan['conflicts']:
            emit('CONFLICT', branch, conflict)
        for row in plan.get('native_statuses', ()):
            emit('NATIVE_FATE', branch, row)
    for diagnostic in diagnostics:
        basis = diagnostic.message.partition(': ')[0]
        if (basis, diagnostic.code, diagnostic.stage, diagnostic.detail) in refusals:
            continue  # The public branch record already carries this primary refusal.
        branch = ({'stream_copy': 'if stream-copied', 'encode': 'if encoded'}.get(basis, 'conditional')
                  if outcome.conditional_plans else 'planned')
        emit('REFUSAL', branch, dict(reason_code=diagnostic.code, stage=diagnostic.stage, detail=diagnostic.detail))
    return '\n'.join(lines)


def verification_details(inspection) -> tuple[str, ...]:
    """Format pipeline-supplied mismatches, never recompare tags or read references.

    Inspection is a pipeline-local record; keep this pure projection independent
    of pipeline imports. Artwork evidence is size/hash only, never image bytes.
    """
    references = {ref.name.casefold(): ref for ref in inspection.tag_references}
    tags = inspection.source_tags
    lines = []
    def value_text(value):
        return 'absent' if value is None else json.dumps(portable(value), ensure_ascii=False, separators=(',', ':'))
    for mismatch in inspection.tag_mismatches:
        if mismatch == 'extra:NORMA':
            lines.append('processing mark missing')
            continue
        kind, _, name = mismatch.partition(':')
        ref = references.get(name.casefold())
        if ref is None or tags is None:
            lines.append(mismatch + ': differs from explicit reference')
            continue
        origin = {'cli': 'the command line', 'manifest': 'the manifest',
                  'manifest_default': 'the manifest defaults'}[ref.origin]
        source = origin + (f' ({ref.locator})' if ref.locator else '')
        expected = 'absent' if ref.cell.operation == 'clear' else value_text(ref.cell.value)
        if kind == 'field' and name == 'artwork':
            if ref.cell.operation == 'set':
                expected = 'byte-exact front cover from ' + str(ref.cell.value)
            pictures = []
            for tag in tags.artwork:
                picture = tag.value
                if isinstance(picture, Artwork):
                    pictures.append(f'type {picture.picture_type}, {len(picture.data)} bytes, SHA-256 {hashlib.sha256(picture.data).hexdigest()}')
            actual = '; '.join(pictures) or 'absent'
        else:
            values = tags.fields if kind == 'field' else tags.extras
            tag = next((tag for key, tag in values.items() if key.casefold() == name.casefold()), None)
            actual = value_text(tag.value if tag is not None else None)
        lines.append(f'{ref.name}: expected {expected} from {source}, found {actual}')
    return tuple(lines)


def project_outcome(outcome: FileOutcome, options: RunOptions) -> dict:
    """The 53 stable columns, leaving unavailable measurements and destinations null."""
    row = dict.fromkeys(REPORT_COLUMNS)
    row.update(input=str(outcome.input.source_path), status=outcome.status, reason=outcome.reason_code,
               elapsed_s=outcome.elapsed_s, execution=outcome.execution, input_index=outcome.input.input_index,
               tag_sources={}, conflicts=[], dropped=[], warnings=portable(outcome.warnings),
               published_paths=portable(outcome.published_paths), conditional_plans=[],
               retained_failed_path=portable(outcome.retained_failed_path),
               source_codec=outcome.source_codec, origin_codec=outcome.origin_codec)
    for prefix, measurement in (('in', outcome.measurement_in), ('out', outcome.measurement_out)):
        if measurement is not None:
            row.update({f'{prefix}_lufs': measurement.integrated_lufs,
                        f'{prefix}_dbtp': portable(measurement.true_peak_dbtp),
                        f'{prefix}_signal_state': measurement.signal_state,
                        f'{prefix}_peak_resolution_linear': measurement.peak_resolution_linear})
            if prefix == 'in':
                row['in_lra'] = measurement.lra_lu
            row['ffmpeg_version'] = measurement.ffmpeg_version
    gain = outcome.gain
    if gain is not None:
        row.update(target_lufs=gain.target_lufs, ceiling_dbtp=gain.ceiling_dbtp,
                   tolerance_lu=gain.tolerance_lu, gain_db=gain.gain_db,
                   mode=None if gain.action == 'skip' else gain.action,
                   limited_db=gain.limiter_reduction_db,
                   effective_limiter_bound_db=gain.effective_limiter_bound_db,
                   limiter_threshold_dbtp=gain.limiter_threshold_dbtp,
                   expected_lufs=gain.expected_integrated_lufs,
                   predicted_shortfall_lu=gain.shortfall_lu,
                   loudness_corrections=gain.loudness_corrections,
                   peak_corrections=gain.peak_corrections, reduce_fallbacks=gain.reduce_fallbacks,
                   target_origin=options.value_sources.get('target', 'default'),
                   ceiling_origin=options.value_sources.get('true_peak', 'default'),
                   tolerance_origin=options.value_sources.get('tolerance', 'default'))
    if outcome.target is not None:
        target = outcome.target
        row.update(out_format=target.format, out_bit_depth=target.bit_depth, out_rate=target.sample_rate_hz,
                   requested_bitrate_bps=target.requested_bitrate_bps,
                   effective_bitrate_bps=target.effective_bitrate_bps, bitrate_mode=target.bitrate_mode,
                   vbr_quality=target.vbr_quality, encoder=target.encoder)
    if outcome.output is not None:
        row['out_path'] = str(outcome.output.audio_member.final_path)
        row['sidecar'] = next((str(member.final_path) for member in getattr(outcome.output, 'members', ())
                               if member.purpose == 'tags_sidecar' and member.final_path in outcome.published_paths), None)
    if outcome.conditional_plans:
        output = next((branch.output for branch in outcome.conditional_plans if branch.output is not None), None)
        row['out_path'] = str(output.audio_member.final_path) if output is not None else None
        conditional = []
        for branch in outcome.conditional_plans:
            entry = dict(assumption=branch.assumption,
                metadata=project_metadata(branch.metadata, verbose=True) if branch.metadata is not None else None,
                output=portable(branch.output))
            if branch.refusal is not None:
                entry['refusal'] = portable(branch.refusal)
            conditional.append(entry)
        row['conditional_plans'] = conditional
    if outcome.verification_reference is not None:
        reference = outcome.verification_reference
        row.update(target_lufs=reference.target_lufs, target_origin=reference.target_origin,
                   ceiling_dbtp=reference.ceiling_dbtp, ceiling_origin=reference.ceiling_origin,
                   tolerance_lu=reference.tolerance_lu, tolerance_origin=reference.tolerance_origin,
                   mark_validity=reference.mark.validity)
    if options.command == 'verify' or outcome.verification_reference is not None:
        row['out_path'] = str(outcome.input.source_path)

    if outcome.verification is not None:
        row['measured_shortfall_lu'] = outcome.verification.measured_shortfall_lu
    if outcome.metadata is not None:
        plan = outcome.metadata
        row.update(tag_sources={k: v.provenance for k, v in plan.resolved.items()},
                   conflicts=portable(plan.conflicts), dropped=_losses(plan))
    return row


def render_csv(report: RunReport) -> str:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, REPORT_COLUMNS, lineterminator='\n')
    writer.writeheader()
    for outcome in sorted(report.outcomes, key=lambda item: item.input.input_index):
        row = project_outcome(outcome, report.options)
        writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                         if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue()


def render_json(report: RunReport) -> str:
    envelope = {f.name: portable(getattr(report, f.name)) for f in fields(report) if f.name != 'outcomes'}
    envelope['outcomes'] = [project_outcome(item, report.options) for item in sorted(report.outcomes, key=lambda item: item.input.input_index)]
    return json.dumps(envelope, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'


def render_summary(report: RunReport, *, terminal: bool = False) -> str:
    if terminal:
        result = ', '.join(f'{report.counts[status]} {STATUS_WORDS[status]}' for status in STATUS_WORDS)
        for key in ('not_started', 'ignored'):
            if getattr(report, key):
                result += f', {getattr(report, key)} {key.replace("_", " ")}'
        levels = [o.measurement_in.integrated_lufs for o in report.outcomes if o.measurement_in is not None
                  and o.measurement_in.integrated_lufs is not None]
        origin = sum(any(d.code == 'lossy_origin' for d in o.warnings) or o.reason_code == 'lossy_origin' for o in report.outcomes)
        if origin:
            result += f'; {origin} lossy origin'
        plans = [[o.metadata] if o.metadata is not None else [branch.metadata for branch in o.conditional_plans
                                                           if branch.metadata is not None]
                 for o in report.outcomes]
        if any(plans):
            tagged = sum(o.tag_write is not None and o.execution == 'committed' for o in report.outcomes)
            conflicts = sum(len({json.dumps(portable(conflict), sort_keys=True) for plan in alternatives
                                 for conflict in plan.conflicts}) for alternatives in plans)
            dropped = sum(len({(row.key, row.loss_kind) for plan in alternatives for row in plan.statuses
                               if row.status == 'dropped_format_limit'}) for alternatives in plans)
            result += f'; {tagged} tagged, {conflicts} conflicts, {dropped} dropped'
        if report.options.command == 'analyse' and levels:
            result += f'; input spread {max(levels) - min(levels):.1f} LU'
        if report.options.dry_run:
            result = 'Dry run — ' + result
        if report.report_path is not None:
            result += '; report: ' + escape_cell(report.report_path)
        if report.run_errors:
            result += f'; {len(report.run_errors)} run errors'
        return result + f'; {report.elapsed_s:.2f} s'
    values = dict(run_id=report.run_id, **report.counts, not_started=report.not_started, ignored=report.ignored,
                  run_errors=len(report.run_errors), report=report.report_path or '', elapsed_s=f'{report.elapsed_s:.3f}')
    return 'SUMMARY\t' + '\t'.join(f'{key}={escape_cell(values[key])}' for key in SUMMARY_KEYS)


def _excess_db(outcome: FileOutcome) -> float | None:
    """Predicted peak excess over the ceiling at the nominal gain — what ``limit`` would have had to remove."""
    measurement, gain = outcome.measurement_in, outcome.gain
    if measurement is None or gain is None or measurement.integrated_lufs is None or measurement.true_peak_dbtp is None:
        return None
    return measurement.true_peak_dbtp + (gain.target_lufs - measurement.integrated_lufs) - gain.ceiling_dbtp


def _fallback_reason(gain, excess: float | None) -> str:
    """The parenthesised cause of a limit fallback, or '' when the diagnostics cannot recover it (CR-065).

    A rejected correction consumes no budget, so counters below the maxima cannot tell a cap rejection
    from anything else: only a nominal excess over the cap, or both budgets fully spent, is stated.
    """
    if excess is not None and excess > gain.max_limit_db:
        return f' (cap {gain.max_limit_db:g} dB)'
    if gain.loudness_corrections >= 2 and gain.peak_corrections >= 1:
        return ' (corrections used up)'
    return ''


def _shortfall_message(outcome: FileOutcome, fallback: str) -> str:
    """The § 10.2 `reduce_shortfall` wordings: predicted (nothing measured) and measured."""
    gain, verification = outcome.gain, outcome.verification
    target = f'{gain.target_lufs:g}'
    excess = _excess_db(outcome)
    measured = verification.measured_shortfall_lu if verification is not None else None
    if measured is not None:
        out = outcome.measurement_out or verification.measurement
        lufs, peak = f'{out.integrated_lufs:.1f}', f'{out.true_peak_dbtp:.1f}'
        if gain.requested_peak_mode == 'reduce':
            return f'reduced so peaks stay under {gain.ceiling_dbtp:g} dBTP; measures {lufs} LUFS, {measured:.2f} LU below {target}'
        tail = 'on target, but by reduction' if measured < 0.005 else f'{measured:.2f} LU below {target}'
        return f'limit could not reach the target{_fallback_reason(gain, excess)} — reduced to keep peaks at {peak} dBTP; measures {lufs} LUFS, {tail}'
    if gain.shortfall_lu is None or gain.expected_integrated_lufs is None:
        return fallback
    about = f'about {gain.expected_integrated_lufs:.1f} LUFS, {gain.shortfall_lu:.2f} LU below {target}'
    if gain.requested_peak_mode == 'reduce':
        return f'would reduce so peaks stay under {gain.ceiling_dbtp:g} dBTP — {about}'
    need = f'{excess:.1f} dB' if excess is not None else 'more'
    return f'would need {need} of limiting (cap {gain.max_limit_db:g}) — would reduce instead, to {about}'


def diagnostic_note(outcome: FileOutcome, primary=None, *, notes=()) -> str:
    if outcome.verification_reference is not None:
        reference = outcome.verification_reference
        origin = {'cli': 'command line', 'mark': 'processing mark', 'config': 'config', 'default': 'default'}[reference.target_origin]
        target = f'target {reference.target_lufs:g} ({origin})'
        diagnostic = primary or next((d for d in outcome.warnings if d.code == outcome.reason_code), None)
        message = diagnostic.message if diagnostic is not None else outcome.reason_code.replace('_', ' ')
        if outcome.reason_code == 'ok':
            return target
        if outcome.reason_code == 'verification_tags' and notes:
            message = '; '.join(notes)
        elif outcome.reason_code == 'reduce_shortfall' and outcome.verification is not None:
            measured = outcome.measurement_out
            shortfall = outcome.verification.measured_shortfall_lu
            if measured is not None and measured.integrated_lufs is not None and shortfall is not None:
                message = (f'reduced output, peaks safe; measures {measured.integrated_lufs:.1f} LUFS, '
                           f'{shortfall:.2f} LU below {reference.target_lufs:g} ({origin})')
        elif outcome.reason_code == 'processing_mark_invalid':
            message += f' — not used; {target}'
        # Inspection never had a candidate or a publication to undo.
        return f'{outcome.reason_code}: {message}; {target}' if target not in message else f'{outcome.reason_code}: {message}'
    if outcome.reason_code == 'ok':
        return ''
    diagnostic = primary or next((d for d in outcome.warnings if d.code == outcome.reason_code), None)
    message = diagnostic.message if diagnostic is not None else outcome.reason_code.replace('_', ' ')
    if outcome.reason_code == 'reduce_shortfall' and outcome.gain is not None:
        message = _shortfall_message(outcome, message)

    if outcome.status == 'failed' or outcome.reason_code == 'cancelled':
        if outcome.execution == 'publication_failed':
            publication = 'partly published: ' + (', '.join(map(str, outcome.published_paths)) or 'no final paths') + ' remain; see the report'
        elif outcome.retained_failed_path is not None:
            publication = f'failed candidate kept at {outcome.retained_failed_path} (--keep-failed); no output written'
        elif notes:
            publication = next((note for note in notes if 'rolled back' in note), '')
            if not publication:
                publication = 'file not processed' if outcome.output is None else 'no output written'
        else:
            publication = 'file not processed' if outcome.output is None else 'no output written'
        if publication not in message:
            message += ' — ' + publication
    return f'{outcome.reason_code}: {message}'


def mode_label(gain) -> str:
    """The row's `mode` cell: the C-05 action; a dynamic preview plan (no calibration) says it is not calibrated yet (D-080)."""
    if gain is None or gain.action == 'skip':
        return ''
    if gain.action == 'dynamic' and gain.dynamic_parameters is None:
        return 'dynamic (calibrated at render)'
    return gain.action


def _number(value) -> str:
    return '' if value is None else f'{value:.1f}'


def row_cells(outcome: FileOutcome, total: int, *, primary=None, notes=(), terminal=False) -> list[str]:
    measurement, gain = outcome.measurement_in, outcome.gain
    measured = outcome.measurement_out
    output = ''
    if outcome.execution == 'committed' and outcome.output is not None:
        audio = outcome.output.audio_member.final_path
        if audio in outcome.published_paths:
            output = str(audio)
    elif outcome.execution == 'publication_failed':
        output = '; '.join(map(str, outcome.published_paths))
    elif terminal and outcome.execution == 'plan' and outcome.output is not None:
        output = str(outcome.output.audio_member.final_path)
    elif terminal and outcome.conditional_plans:
        planned = next((branch.output for branch in outcome.conditional_plans if branch.output is not None), None)
        if planned is not None:
            output = str(planned.audio_member.final_path) + f' ({len(outcome.conditional_plans)} alternatives)'
    out_lufs = _number(measured.integrated_lufs) if measured else ''
    out_peak = _number(measured.true_peak_dbtp) if measured else ''
    if terminal and outcome.execution == 'plan' and gain is not None:
        out_lufs = '~' + _number(gain.expected_integrated_lufs) if gain.expected_integrated_lufs is not None else ''
        out_peak = '~' + _number(gain.predicted_true_peak_dbtp) if gain.predicted_true_peak_dbtp is not None else ''
    deviation, mode = _number(gain.gain_db) if gain else '', mode_label(gain)
    if outcome.verification_reference is not None:
        output = str(outcome.input.source_path)
        mode = 'verify'
        deviation = (_number(measured.integrated_lufs - outcome.verification_reference.target_lufs)
                     if measured is not None and measured.integrated_lufs is not None else '')
    return [STATUS_WORDS[outcome.status], f'{outcome.input.input_index + 1}/{total}', str(outcome.input.source_path),
            _number(measurement.integrated_lufs) if measurement else '',
            _number(measurement.true_peak_dbtp) if measurement else '',
            deviation, mode,
            out_lufs, out_peak, output, diagnostic_note(outcome, primary, notes=notes)]


def render_tsv(outcome: FileOutcome, total: int, *, primary=None, notes=()) -> str:
    return '\t'.join(escape_cell(cell) for cell in row_cells(outcome, total, primary=primary, notes=notes))


def display_path(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        try:
            return '~/' + str(path.relative_to(Path.home()))
        except ValueError:
            return str(path)


def shorten(text: str, width: int = 36) -> str:
    return text if len(text) <= width else text[:width // 2] + '…' + text[-(width - width // 2 - 1):]


def render_table(outcomes, total: int, cwd: Path, *, primaries: Mapping | None = None, notes: Mapping | None = None, colour: bool = False) -> str:
    rows = []
    for outcome in outcomes:
        row = row_cells(outcome, total, primary=(primaries or {}).get(outcome.input.input_index),
                        notes=(notes or {}).get(outcome.input.input_index, ()), terminal=True)
        row[2] = shorten(escape_cell(display_path(outcome.input.source_path, cwd)))
        row[9] = escape_cell(row[9])
        row[-1] = escape_cell(row[-1])
        rows.append(row)
    widths = [max(len(ROW_HEADERS[i]), *(len(row[i]) for row in rows)) for i in range(len(ROW_HEADERS))] if rows else [len(h) for h in ROW_HEADERS]
    lines = ['  '.join(h.ljust(widths[i]) for i, h in enumerate(ROW_HEADERS)).rstrip()]
    colours = {'OK': '32', 'WARN': '33', 'SKIP': '36', 'FAIL': '31'}
    for row in rows:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        if colour and 'NO_COLOR' not in os.environ:
            padded[0] = f'\x1b[{colours[row[0]]}m{row[0]}\x1b[0m' + ' ' * (widths[0] - len(row[0]))
        lines.append('  '.join(padded).rstrip())
    return '\n'.join(lines)
