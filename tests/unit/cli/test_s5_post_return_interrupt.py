"""TK-0316: CLI-owned signals after genuine operations, synthetic audio only."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest


POST_RETURN = '''
import json, signal, sys
from norma import cli, pipeline
case = json.loads(sys.argv[1])
prior = signal.getsignal(signal.SIGINT)
operation = pipeline.verify_items
seen = []
def after_return(*args, **kwargs):
    result = operation(*args, **kwargs)
    assert result.cancelled is False
    assert len(result.outcomes) == 1 and result.outcomes[0].status == 'failed'
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler) and handler is not prior
    seen.append(result)
    signal.raise_signal(signal.SIGINT)
    assert kwargs['cancel_event'].is_set()
    assert result.cancelled is False
    return result
pipeline.verify_items = after_return
code = cli.main(sys.argv[2:])
assert len(seen) == 1 and seen[0].cancelled is False
assert signal.getsignal(signal.SIGINT) is prior
raise SystemExit(code)
'''


def identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def source_audio(path):
    rate = 48000
    with wave.open(str(path), 'wb') as writer:
        writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(b''.join(struct.pack('<h', int(3000 * math.sin(2 * math.pi * 440 * i / rate)))
                                    for i in range(3 * rate)))


def test_real_verify_post_return_sigint_beats_real_failure(tmp_path, norma_environment):
    executable = Path(norma_environment['ffmpeg_path'])
    if not executable.is_file() or not executable.with_name('ffprobe').is_file():
        pytest.skip('explicit native FFmpeg/ffprobe pair required')
    source = tmp_path / 'source.wav'
    source_audio(source)
    before = identity(source)
    report = tmp_path / 'verify.json'
    child = subprocess.run([sys.executable, '-c', POST_RETURN, '{}', 'verify', str(source),
        '--config', os.devnull, '--ffmpeg', str(executable), '--report', str(report)],
        capture_output=True, text=True, timeout=40)
    assert 'Traceback' not in child.stderr, child.stderr
    assert identity(source) == before
    document = json.loads(report.read_text())
    row, = document['outcomes']
    assert row['status'] == 'failed' and row['reason'] == 'verification_loudness'
    assert row['execution'] == 'inspect' and row['published_paths'] == []
    assert document['not_started'] == 0 and document['run_errors'] == []
    assert len([line for line in child.stdout.splitlines() if line.startswith('FAIL\t')]) == 1
    assert len([line for line in child.stdout.splitlines() if line.startswith('SUMMARY\t')]) == 1
    assert child.returncode == 130, (child.stdout, child.stderr)
    assert child.stderr.count('Interrupted: 0 finished, 0 cancelled, 1 failed, 0 not started.') == 1


# All operations below are real. Only the timing, terminal and fault boundaries
# are instrumented, in an owned child; no success-shaped pipeline doubles.
NATIVE_BOUNDARY = '''
import contextlib, hashlib, io, json, os, signal, sys
from pathlib import Path
from norma import batch, cli, pipeline
case = json.loads(sys.argv[1])
if case.get('previous') == 'custom':
    def previous(signum, frame):
        raise RuntimeError('previous handler called')
    signal.signal(signal.SIGINT, previous)
prior = signal.getsignal(signal.SIGINT)
returned, forced, controls, sent, log_fds = [], [], [], [], []
publications = []
original_control = batch.BatchControl
class ObservedControl(original_control):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        controls.append(self)
batch.BatchControl = ObservedControl
pipeline.force_kill_tools = lambda: forced.append('force')

def identity(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns

def send(seam):
    if seam != case['seam'] or sent:
        return
    assert returned, 'operation must have returned before injection'
    handler = signal.getsignal(signal.SIGINT)
    if case.get('outside'):
        assert handler is prior, 'CLI widened its signal ownership'
        sent.append(seam)
        signal.raise_signal(signal.SIGINT)
        raise AssertionError('previous handler did not raise')
    assert callable(handler) and handler is not prior, 'signal outside CLI ownership'
    sent.append(seam)
    if case.get('mode') == 'event':
        controls[0].cancel_event.set()
    elif case.get('mode') == 'internal':
        controls[0].interrupt()
    elif case.get('mode') != 'none':
        for _ in range(case.get('signals', 1)):
            signal.raise_signal(signal.SIGINT)
    if case.get('mode') != 'none':
        assert controls[0].cancel_event.is_set()

name = {'analyse': 'analyse_items', 'verify': 'verify_items',
        'normalise': 'normalise_items', 'tags_show': 'inspect_tags',
        'tags_plan': 'plan_tags_items', 'tags_template': 'template_tags_items'}[case['command']]
operation = getattr(pipeline, name)
def after_return(*args, **kwargs):
    result = operation(*args, **kwargs)
    assert result.cancelled is False
    paths = tuple(p for o in result.outcomes for p in o.published_paths)
    returned.append((result, result.outcomes, repr(result.outcomes),
                     {str(p): identity(p) for p in paths}))
    send('return')
    return result
setattr(pipeline, name, after_return)

def wrap_render(name, seam):
    original = getattr(cli.report, name)
    def render(*args, **kwargs):
        text = original(*args, **kwargs)
        if returned:
            send(seam)
        return text
    setattr(cli.report, name, render)
wrap_render('render_tsv', 'row')
wrap_render('render_table', 'table')
wrap_render('render_summary', 'summary')

original_publish = pipeline.RunOutputs.publish
def publish(self, purpose, data, **kwargs):
    if returned and purpose == 'report':
        send('report')
        if case.get('report_error'):
            self.destination('report').final_path.write_bytes(b'report racer kept')
    path = original_publish(self, purpose, data, **kwargs)
    publications.append((str(path), identity(path)))
    if returned:
        send('published')
    return path
pipeline.RunOutputs.publish = publish
original_open_log = pipeline.RunOutputs.open_log
def open_log(self):
    fd = original_open_log(self)
    log_fds.append(fd)
    return fd
pipeline.RunOutputs.open_log = open_log
original_write, original_close = os.write, os.close
def write(fd, data):
    if returned and fd in log_fds:
        send('log_write')
        if case.get('log_error'):
            raise OSError('injected final log append failure')
    return original_write(fd, data)
def close(fd):
    if returned and fd in log_fds:
        send('log_close')
    value = original_close(fd)
    if returned and fd in log_fds and case.get('close_error'):
        raise RuntimeError('injected final log close failure')
    return value
os.write, os.close = write, close
original_signal = signal.signal
def install(signum, handler):
    if signum == signal.SIGINT and handler is prior and returned:
        send('restore')
    old = original_signal(signum, handler)
    if signum == signal.SIGINT and handler is prior and returned:
        send('restored')
    return old
signal.signal = install

class Output(io.StringIO):
    def isatty(self):
        return case.get('terminal', False)
stdout, stderr = Output(), io.StringIO()
error = None
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    try:
        code = cli.main(sys.argv[2:])
    except RuntimeError as exc:
        assert (case.get('close_error') and str(exc) == 'injected final log close failure' or
                case.get('outside') and str(exc) == 'previous handler called')
        error, code = str(exc), 87
assert signal.getsignal(signal.SIGINT) is prior
assert len(returned) == len(controls) == 1 and sent == [case['seam']]
result, outcomes, rows_before, paths_before = returned[0]
assert result.cancelled is False and len(result.outcomes) == len(outcomes)
assert all(actual is expected for actual, expected in zip(result.outcomes, outcomes))
assert repr(result.outcomes) == rows_before
assert {p: identity(p) for p in paths_before} == paths_before
assert all(identity(p) == pin for p, pin in publications)
expected_force = (case.get('mode', 'signal') == 'signal' and case.get('signals', 1) >= 2)
assert forced == (['force'] if expected_force else [])
for fd in log_fds:
    try:
        os.fstat(fd)
    except OSError:
        pass
    else:
        raise AssertionError('CLI leaked log descriptor')
print(json.dumps(dict(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
    statuses=[o.status for o in outcomes], reasons=[o.reason_code for o in outcomes],
    executions=[o.execution for o in outcomes], not_started=result.not_started,
    paths=list(paths_before), publications=[p for p, pin in publications], error=error)))
'''


def native_boundary(tmp_path, norma_environment, command, seam='return', **case):
    executable = Path(norma_environment['ffmpeg_path'])
    if not executable.is_file() or not executable.with_name('ffprobe').is_file():
        pytest.skip('explicit native FFmpeg/ffprobe pair required')
    source = tmp_path / 'source.wav'
    source_audio(source)
    before = identity(source)
    args = command.split('_') + [str(source), '--config', os.devnull, '--ffmpeg', str(executable)]
    existing_files = {p for p in tmp_path.rglob('*') if p.is_file()}
    report = tmp_path / 'report.json'
    if command == 'tags_template':
        args += ['-o', str(tmp_path / 'template.csv')]
    elif command != 'tags_show' and not case.get('automatic'):
        args += ['--report', str(report)]
    if command == 'normalise':
        args += ['-o', str(tmp_path / 'out'), '--format', 'wav']
        if case.get('dry_run'):
            args += ['--dry-run']
        if case.get('log'):
            args += ['--log', str(tmp_path / 'run.log')]
    if case.get('quiet'):
        args += ['--quiet']
    case.update(command=command, seam=seam)
    child = subprocess.run([sys.executable, '-c', NATIVE_BOUNDARY, json.dumps(case), *args],
        capture_output=True, text=True, timeout=45, env={**os.environ, 'NO_COLOR': ''})
    assert child.returncode == 0, (child.stdout, child.stderr)
    evidence = json.loads(child.stdout)
    assert identity(source) == before
    assert 'Traceback' not in evidence['stderr']
    assert evidence['not_started'] == 0 and len(evidence['statuses']) == 1
    assert 'cancelled' not in evidence['reasons']
    assert not list(tmp_path.rglob('.norma-*'))
    is_signal = case.get('mode', 'signal') == 'signal'
    assert evidence['code'] == (87 if case.get('close_error') or case.get('previous') == 'custom' else 130 if is_signal else
                                1 if command == 'verify' or case.get('report_error') or case.get('log_error') else 0), evidence
    expected_notice = (f'Interrupted: {int(command != "verify")} finished, 0 cancelled, '
                       f'{int(command == "verify")} failed, 0 not started.')
    notices = is_signal and not (case.get('close_error') or case.get('previous') == 'custom')
    assert evidence['stderr'].count('Interrupted') == notices, evidence
    if case.get('outside') and notices:
        diagnostic = ('note: channel layout not stated; assumed mono\n'
                      if command == 'normalise' and not case.get('quiet') else '')
        assert evidence['stderr'] == diagnostic + 'Interrupted before processing finished.\n'
    elif notices:
        assert expected_notice in evidence['stderr']
    if command != 'tags_template' and not case.get('outside'):
        if case.get('terminal'):
            lines = evidence['stdout'].splitlines()
            assert sum(line.startswith('STATUS ') for line in lines) == 1, lines
            assert sum('0 WARN, 0 SKIP,' in line for line in lines) == 1, lines
            assert sum(line.startswith(('OK ', 'FAIL ')) and '1/1' in line for line in lines) == 1, lines
        else:
            assert sum(line.startswith('SUMMARY\t') for line in evidence['stdout'].splitlines()) == 1
            rows = [line for line in evidence['stdout'].splitlines() if line.startswith(('OK\t', 'WARN\t', 'FAIL\t', 'SKIP\t'))]
            assert len(rows) == 1 and str(source) in rows[0]
    if report.exists() and not case.get('report_error'):
        document = json.loads(report.read_text())
        row, = document['outcomes']
        assert row['status'] == evidence['statuses'][0] and row['reason'] == evidence['reasons'][0]
        assert row['execution'] == evidence['executions'][0]
        assert row['published_paths'] == evidence['paths']
        assert document['not_started'] == 0
    if case.get('automatic'):
        automatic, = [Path(p) for p in evidence['publications']]
        assert automatic.parent == tmp_path / 'out' and automatic.name.startswith('norma-report-')
        with automatic.open(newline='') as reader:
            row, = csv.DictReader(reader)
        assert row['status'] == evidence['statuses'][0] and row['execution'] == 'committed'
        assert row['reason'] == evidence['reasons'][0]
    allowed = existing_files | {Path(p) for p in evidence['paths'] + evidence['publications']}
    if case.get('log'):
        allowed.add(tmp_path / 'run.log')
    if case.get('report_error'):
        allowed.add(report)
    assert {p for p in tmp_path.rglob('*') if p.is_file()} == allowed
    return evidence


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_show', 'tags_plan', 'tags_template', 'normalise'])
@pytest.mark.parametrize('mode', ['signal', 'none', 'event', 'internal'])
def test_shared_operations_keep_completed_results_and_signal_provenance(tmp_path, norma_environment, command, mode):
    evidence = native_boundary(tmp_path, norma_environment, command, mode=mode, quiet=True)
    if command == 'verify':
        assert evidence['statuses'] == ['failed']
        assert evidence['reasons'] == ['verification_loudness']
    if command == 'normalise':
        assert evidence['executions'] == ['committed']
        assert len(evidence['paths']) >= 1


@pytest.mark.parametrize('seam,settings', [
    ('return', dict(dry_run=True, quiet=True)),
    ('row', dict(dry_run=True)),
    ('table', dict(terminal=True)),
    ('summary', dict(quiet=True)),
    ('report', {}),
    ('published', {}),
    ('report', dict(automatic=True)),
    ('published', dict(automatic=True)),
    ('log_write', dict(log=True)),
    ('log_close', dict(log=True)),
    ('restore', dict(log=True)),
])
def test_normalise_post_return_presentation_publication_and_finalisation(tmp_path, norma_environment, seam, settings):
    evidence = native_boundary(tmp_path, norma_environment, 'normalise', seam, **settings)
    assert evidence['executions'] == (['plan'] if settings.get('dry_run') else ['committed'])


@pytest.mark.parametrize('command', ['verify', 'normalise'])
@pytest.mark.parametrize('signals', [2, 3])
def test_repeated_post_return_signals_force_exactly_once(tmp_path, norma_environment, command, signals):
    native_boundary(tmp_path, norma_environment, command, signals=signals)


@pytest.mark.parametrize('mode', ['none', 'signal'])
@pytest.mark.parametrize('error', ['report_error', 'log_error'])
def test_post_return_signal_retains_real_report_racer_or_log_error(tmp_path, norma_environment, mode, error):
    evidence = native_boundary(tmp_path, norma_environment, 'normalise',
        'report' if error == 'report_error' else 'log_write', mode=mode, log=True, **{error: True})
    assert evidence['executions'] == ['committed']
    if error == 'report_error':
        assert (tmp_path / 'report.json').read_bytes() == b'report racer kept'
        assert 'appeared while the run was working and was kept' in evidence['stderr']
    else:
        assert 'cannot append log' in evidence['stderr']
        assert 'injected final log append failure' in evidence['stderr']
    summary, = [line for line in evidence['stdout'].splitlines() if line.startswith('SUMMARY\t')]
    assert '\trun_errors=1\t' in summary


def test_unexpected_final_log_error_propagates_and_restores_handler(tmp_path, norma_environment):
    evidence = native_boundary(tmp_path, norma_environment, 'normalise', 'log_close', log=True, close_error=True)
    assert evidence['error'] == 'injected final log close failure'


@pytest.mark.parametrize('previous', ['default', 'custom'])
def test_inspection_finalised_signal_belongs_to_previous_handler(tmp_path, norma_environment, previous):
    # Safety § 6 owns presentation/report settlement; the prior handler resumes
    # only at actual restoration, after the selected report and summary settle.
    evidence = native_boundary(tmp_path, norma_environment, 'verify', 'restored', outside=True, previous=previous)
    assert evidence['stdout'].count('SUMMARY\t') == 1
    assert 'verification' in evidence['stdout']


@pytest.mark.parametrize('mode', ['none', 'event', 'internal'])
def test_dry_run_independent_stop_after_result_is_not_sigint(tmp_path, norma_environment, mode):
    evidence = native_boundary(tmp_path, norma_environment, 'normalise', mode=mode, dry_run=True)
    assert evidence['executions'] == ['plan'] and evidence['paths'] == []


def test_terminal_verify_keeps_failure_diagnostic_under_signal(tmp_path, norma_environment):
    evidence = native_boundary(tmp_path, norma_environment, 'verify', terminal=True)
    assert evidence['stdout'].count('FAIL ') == 2
    assert 'verification' in evidence['stdout']


@pytest.mark.parametrize('command', ['verify', 'normalise'])
def test_signal_during_handler_restoration_is_not_lost(tmp_path, norma_environment, command):
    native_boundary(tmp_path, norma_environment, command, 'restore')


@pytest.mark.parametrize('previous', ['default', 'custom'])
def test_normalise_finalised_signal_belongs_to_previous_handler(tmp_path, norma_environment, previous):
    native_boundary(tmp_path, norma_environment, 'normalise', 'restored', outside=True, previous=previous, log=True)
