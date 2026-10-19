"""TK-0317: settle selected reports under real kernel SIGINT, synthetic audio only."""
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


# Instrument timing, not results: the operation, registry, publication primitive,
# BatchControl and native tools all run unchanged in an owned child.
REPORT_BOUNDARY = r'''
import contextlib, hashlib, io, json, os, signal, subprocess, sys, threading
from pathlib import Path
from norma import batch, cli, pipeline, safety
case = json.loads(sys.argv[1])
prior = signal.getsignal(signal.SIGINT)
controls, returned, sent, deliveries, forced, candidates, publications = [], [], [], [], [], {}, []
report_path = Path(case['report'])
processes = []
original_popen = subprocess.Popen
def popen(*args, **kwargs):
    process = original_popen(*args, **kwargs)
    processes.append(process)
    return process
subprocess.Popen = popen
original_control = batch.BatchControl
class ObservedControl(original_control):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        controls.append(self)
batch.BatchControl = ObservedControl
original_force = pipeline.force_kill_tools
def force():
    forced.append('force')
    original_force()
pipeline.force_kill_tools = force

def identity(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns

def send(seam):
    if seam != case['seam'] or sent:
        return
    assert len(returned) == len(controls) == 1
    owner = signal.getsignal(signal.SIGINT)
    sent.append(dict(seam=seam, owned=owner is not prior,
                     registered=any(p in registered for p in candidates.values())))
    if case.get('mode') == 'event':
        controls[0].cancel_event.set()
    elif case.get('mode') == 'internal':
        controls[0].interrupt()
    elif case.get('mode') != 'none':
        for _ in range(case.get('signals', 1)):
            before = len(deliveries)
            if case.get('thread'):
                thread = threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGINT))
                thread.start()
                thread.join(5)
                assert not thread.is_alive()
            else:
                os.kill(os.getpid(), signal.SIGINT)
            # Python dispatches the real kernel signal on the main thread.
            if owner is not prior:
                assert len(deliveries) == before + 1
    if owner is not prior and case.get('mode', 'signal') != 'none':
        assert controls[0].cancel_event.is_set()

original_signal = signal.signal
installed = []
def install(signum, handler):
    restoring = signum == signal.SIGINT and handler is prior and returned
    if restoring:
        send('restore')
    if signum == signal.SIGINT and handler is not prior:
        def observed(signum, frame):
            deliveries.append(signum)
            handler(signum, frame)
        installed.append(observed)
        return original_signal(signum, observed)
    return original_signal(signum, handler)
signal.signal = install

name = {'analyse': 'analyse_items', 'verify': 'verify_items',
        'tags_plan': 'plan_tags_items', 'normalise': 'normalise_items'}[case['command']]
operation = getattr(pipeline, name)
def operation_return(*args, **kwargs):
    result = operation(*args, **kwargs)
    assert not result.cancelled
    assert controls[0].cancel_event is kwargs['cancel_event']
    outcomes = result.outcomes
    returned.append((result, outcomes, repr(outcomes),
        {str(p): identity(p) for o in outcomes for p in o.published_paths}))
    return result
setattr(pipeline, name, operation_return)

original_private = safety.RunRegistry._private
registered = set()
def private(self, *args, **kwargs):
    fd, path = original_private(self, *args, **kwargs)
    if returned:
        candidates[fd] = path
        send('allocate')
    return fd, path
safety.RunRegistry._private = private
original_register = safety.RunRegistry._register
def register(self, record):
    value = original_register(self, record)
    if returned:
        registered.add(record.path)
        send('register')
    return value
safety.RunRegistry._register = register
original_write, original_fsync = os.write, os.fsync
def write(fd, data):
    if fd in candidates:
        send('write')
        if case.get('fault') == 'write':
            raise OSError(28, 'injected report disk full')
    return original_write(fd, data)
def fsync(fd):
    if fd in candidates:
        assert candidates[fd] in registered
        send('fsync')
    return original_fsync(fd)
os.write, os.fsync = write, fsync
original_no_clobber = safety.publish_no_clobber
def publish_no_clobber(candidate, final):
    if returned and Path(candidate) in candidates.values():
        send('publish')
        if case.get('fault') == 'racer':
            Path(final).write_bytes(b'report racer kept')
        elif case.get('fault') == 'protected':
            os.link(case['protected'], final)
    return original_no_clobber(candidate, final)
safety.publish_no_clobber = publish_no_clobber
original_publish = pipeline.RunOutputs.publish
def publish(self, purpose, data, **kwargs):
    if returned and purpose == 'report':
        send('report')
        if case.get('fault') == 'parent':
            report_path.parent.rmdir()
        elif case.get('fault') == 'unexpected':
            raise RuntimeError('unexpected finalisation failure')
    path = original_publish(self, purpose, data, **kwargs)
    publications.append((str(path), identity(path)))
    if returned:
        send('published')
    return path
pipeline.RunOutputs.publish = publish
original_summary = cli.report.render_summary
def summary(*args, **kwargs):
    text = original_summary(*args, **kwargs)
    if returned:
        send('summary')
    return text
cli.report.render_summary = summary
stdout, stderr = io.StringIO(), io.StringIO()
error = None
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    try:
        code = cli.main(sys.argv[2:])
    except RuntimeError as exc:
        assert case.get('fault') == 'unexpected' and str(exc) == 'unexpected finalisation failure'
        code, error = 87, str(exc)
assert signal.getsignal(signal.SIGINT) is prior
assert len(returned) == len(controls) == len(sent) == 1
result, outcomes, rows, paths = returned[0]
assert not result.cancelled and len(result.outcomes) == len(outcomes)
assert all(a is b for a, b in zip(result.outcomes, outcomes))
assert repr(result.outcomes) == rows
assert all(a is b.outcome for a, b in zip(outcomes, result.inspections)) if hasattr(result, 'inspections') else True
assert {p: identity(p) for p in paths} == paths
assert all(identity(p) == pin for p, pin in publications)
assert processes and all(p.poll() is not None for p in processes)
assert not [t for t in threading.enumerate() if t is not threading.main_thread()]
open_fds = []
for fd in candidates:
    try:
        os.fstat(fd)
    except OSError:
        pass
    else:
        open_fds.append(fd)
print(json.dumps(dict(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
    sent=sent, deliveries=deliveries, force=forced, open_fds=open_fds,
    statuses=[o.status for o in outcomes], reasons=[o.reason_code for o in outcomes],
    executions=[o.execution for o in outcomes], not_started=result.not_started,
    paths=paths, publications=publications, error=error)))
'''


def identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def source_audio(path):
    rate = 48000
    with wave.open(str(path), 'wb') as writer:
        writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(b''.join(struct.pack('<h', int(3000 * math.sin(2 * math.pi * 440 * i / rate)))
                                    for i in range(3 * rate)))


def report_boundary(tmp_path, norma_environment, command='verify', extension='json', seam='fsync', **case):
    executable = Path(norma_environment['ffmpeg_path'])
    assert executable.is_file() and executable.with_name('ffprobe').is_file()
    sources = [tmp_path / name for name in ('a.wav', 'b.wav', 'bad.wav')]
    for path in sources[:2]:
        source_audio(path)
    sources[2].write_bytes(b'RIFF\xff\xff\xff\xffWAVEfmt ')
    before = {p: identity(p) for p in sources}
    parent = tmp_path / 'reports'
    parent.mkdir()
    stale = tmp_path / '.norma-foreign-run-out-stale.json'
    stale.write_bytes(b'not owned by this run')
    stale_before = identity(stale)
    target = parent / ('selected.' + extension)
    args = command.split('_') + [str(p) for p in sources] + [
        '--config', os.devnull, '--ffmpeg', str(executable), '--report', str(target), '--quiet']
    if command in ('analyse', 'verify', 'normalise'):
        args += ['-j', '2']
    if command == 'verify':
        args += ['-t', '-21.5', '--tolerance', '1']
    if command == 'normalise':
        args += ['-o', str(tmp_path / 'out'), '--format', 'wav']
    if case.get('fault') == 'protected':
        args += ['--overwrite']
    case.update(command=command, report=str(target), seam=seam, protected=str(sources[0]))
    existing = {p for p in tmp_path.rglob('*') if p.is_file()}
    child = subprocess.run([sys.executable, '-c', REPORT_BOUNDARY, json.dumps(case), *args],
        capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, (child.stdout, child.stderr)
    evidence = json.loads(child.stdout)
    assert 'Traceback' not in child.stderr + evidence['stderr']
    assert {p: identity(p) for p in sources} == before
    if not case.get('fault'):
        assert target.is_file(), evidence
    assert evidence['statuses'] == ['ok', 'ok', 'failed'], evidence
    assert evidence['not_started'] == 0 and 'cancelled' not in evidence['reasons']
    is_signal = case.get('mode', 'signal') == 'signal'
    unexpected = case.get('fault') == 'unexpected'
    assert evidence['code'] == (87 if unexpected else 130 if is_signal else 1), evidence
    assert evidence['sent'] == [dict(seam=seam, owned=True,
        registered=seam not in ('report', 'allocate'))], evidence
    assert evidence['deliveries'] == ([2] * case.get('signals', 1) if is_signal else [])
    forced = is_signal and case.get('signals', 1) >= 2
    assert evidence['force'] == (['force'] if forced else [])
    assert evidence['open_fds'] == [], evidence
    assert list(tmp_path.rglob('.norma-*')) == [stale]
    assert identity(stale) == stale_before
    notice = 'Interrupted: 2 finished, 0 cancelled, 1 failed, 0 not started.'
    assert evidence['stderr'].count('Interrupted') == (is_signal and not unexpected), evidence
    if is_signal and not unexpected:
        assert notice in evidence['stderr']
    summaries = [line for line in evidence['stdout'].splitlines() if line.startswith('SUMMARY\t')]
    assert len(summaries) == (0 if unexpected else 1)
    assert len([line for line in evidence['stdout'].splitlines()
                if line.startswith(('OK\t', 'WARN\t', 'FAIL\t', 'SKIP\t'))]) == 3
    if not case.get('fault'):
        assert target.is_file(), evidence
        if extension == 'json':
            document = json.loads(target.read_text())
            rows = document['outcomes']
            assert document['not_started'] == 0 and document['run_errors'] == []
            assert document['report_path'] == str(target)
        else:
            with target.open(newline='') as reader:
                rows = list(csv.DictReader(reader))
        assert [r['status'] for r in rows] == evidence['statuses']
        assert [r['reason'] for r in rows] == evidence['reasons']
        assert [r['execution'] for r in rows] == evidence['executions']
        assert [r['input'] for r in rows] == [str(p) for p in sources]
        assert '\trun_errors=0\treport=' + str(target) + '\t' in summaries[0]
    elif not unexpected:
        assert '\trun_errors=1\treport=\telapsed_s=' in summaries[0]
        assert str(target) in evidence['stderr']
        if case['fault'] == 'racer':
            assert target.read_bytes() == b'report racer kept'
            assert 'appeared while the run was working and was kept' in evidence['stderr']
        elif case['fault'] == 'protected':
            assert identity(target) == before[sources[0]]
            assert target.stat().st_ino == sources[0].stat().st_ino
        else:
            assert not target.exists()
        if case['fault'] == 'write':
            assert 'injected report disk full' in evidence['stderr']
        if case['fault'] == 'parent':
            assert not parent.exists()
    allowed = existing | {Path(p) for p in evidence['paths']}
    allowed |= {Path(p) for p, pin in evidence['publications']}
    if case.get('fault') in ('racer', 'protected'):
        allowed.add(target)
    assert {p for p in tmp_path.rglob('*') if p.is_file()} == allowed
    return evidence


def test_first_kernel_sigint_at_registered_report_fsync(tmp_path, norma_environment):
    report_boundary(tmp_path, norma_environment, thread=True)


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_plan'])
@pytest.mark.parametrize('extension', ['csv', 'json'])
@pytest.mark.parametrize('seam', ['report', 'allocate', 'register', 'write', 'fsync',
                                  'publish', 'published', 'summary', 'restore'])
def test_inspection_report_lifetime(tmp_path, norma_environment, command, extension, seam):
    report_boundary(tmp_path, norma_environment, command, extension, seam)


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_plan'])
@pytest.mark.parametrize('signals', [2, 3])
def test_report_repeated_kernel_signals_force_once(tmp_path, norma_environment, command, signals):
    report_boundary(tmp_path, norma_environment, command, signals=signals, thread=True)


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_plan'])
@pytest.mark.parametrize('mode', ['none', 'event', 'internal'])
def test_report_independent_stop_is_not_user_sigint(tmp_path, norma_environment, command, mode):
    report_boundary(tmp_path, norma_environment, command, mode=mode)


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_plan'])
@pytest.mark.parametrize('extension', ['csv', 'json'])
@pytest.mark.parametrize('mode', ['none', 'signal'])
@pytest.mark.parametrize('fault,seam', [('racer', 'publish'), ('protected', 'publish'),
                                      ('write', 'write'), ('parent', 'report')])
def test_signal_preserves_genuine_report_failure(tmp_path, norma_environment, command, extension, mode, fault, seam):
    report_boundary(tmp_path, norma_environment, command, extension, seam, mode=mode, fault=fault)


@pytest.mark.parametrize('command', ['analyse', 'verify', 'tags_plan'])
@pytest.mark.parametrize('mode', ['none', 'signal'])
def test_unexpected_finalisation_error_propagates_and_restores(tmp_path, norma_environment, command, mode):
    evidence = report_boundary(tmp_path, norma_environment, command,
                               seam='report', mode=mode, fault='unexpected')
    assert evidence['error'] == 'unexpected finalisation failure'


@pytest.mark.parametrize('extension', ['csv', 'json'])
def test_normalise_committed_audio_and_report_control(tmp_path, norma_environment, extension):
    evidence = report_boundary(tmp_path, norma_environment, 'normalise', extension, thread=True)
    assert evidence['executions'] == ['committed', 'committed', 'not_written']
    assert len(evidence['paths']) == 2
