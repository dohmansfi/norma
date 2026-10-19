"""S3 AAC nominal query/encode/decode evidence and build-matched MP3 V0."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.errors import Cancelled, EnvironmentError_
from norma.formats.capabilities import ToolBasis, automatic_mp3_capability, lossy_capability
from norma.models import CapabilityEvidence


def adts(*, rate_index: int = 3, channels: int = 2, profile: int = 1) -> bytes:
    return bytes((255, 241, (profile << 6) | (rate_index << 2) | (channels >> 2),
                  ((channels & 3) << 6), 1, 31, 252, 0))


@pytest.fixture(autouse=True)
def _encoder_stats_supported(request, monkeypatch):
    '''CR-067: fake-FFmpeg tests assume the statistics options exist; the probe has its own tests.'''
    if 'real_ffmpeg' not in request.keywords:
        monkeypatch.setattr(f, 'probe_mp3_encoder_stats', lambda env, *, cancel_event=None: True)


@pytest.fixture
def env() -> f.FFmpegEnvironment:
    return f.FFmpegEnvironment(f.Executable(Path('/fake/ffmpeg'), (1, 2), 'N-test'),
                               frozenset({'aac_at', 'libmp3lame'}), frozenset(), 'test platform')


def install_aac_stub(monkeypatch, *, warning=False, decode=True, packets=None, encode_code=0):
    paths = []
    calls = []
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda rate, ch, candidates, *, cancel_event=None: tuple(candidates))
    monkeypatch.setattr(f, '_decoded_evidence_ok', lambda *args, cancel_event=None: decode)

    def run(argv, **kwargs):
        calls.append(argv)
        if '-b:a' in argv:
            path = Path(argv[-1])
            paths.append(path)
            path.write_bytes(b'fake encoded file')
            if warning:
                kwargs['on_stderr_line']('Bitrate 384000 not allowed; changing to 320000')
            output = b''
            code = encode_code
        else:
            output = packets if packets is not None else adts()
            code = 0
        return f.ProcessResult(tuple(argv), code, output, (), 0.01, False)

    monkeypatch.setattr(f, 'run_process', run)
    return paths, calls


def test_aac_evidence_is_nominal_not_an_average_and_cleans_temp(env, monkeypatch, tmp_path):
    paths, calls = install_aac_stub(monkeypatch)
    record = f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path,
                                    cbr_candidates_bps=(256000, 288000))
    assert record.nominal_bitrates_bps == (256000, 288000)
    assert record.nominal_confirmation == 'encoder_diagnostic'
    assert record.basis == 'installed_query'
    assert record.channel_layout == 'stereo'
    assert paths and all(not path.exists() for path in paths)
    assert not list(tmp_path.glob('norma-*'))
    encode = calls[0]
    assert encode[encode.index('-aac_at_mode') + 1] == 'cbr'
    assert encode[encode.index('-loglevel') + 1] == 'warning'
    assert encode[encode.index('-f', encode.index('-b:a')) + 1] == 'ipod'
    basis = ToolBasis(record.executable_path, record.executable_identity, record.ffmpeg_version,
                      record.platform_basis)
    cap = lossy_capability('aac', 'cbr', [record], basis, sample_rate_hz=48000,
                           channels=2, channel_layout='stereo')
    assert cap.supported and cap.evidence_for_setting(bitrate_bps=256000) == record.evidence_id


@pytest.mark.parametrize('failure', ['warning', 'decode', 'profile', 'rate', 'channels', 'empty', 'encode'])
def test_aac_unconfirmed_settings_are_typed_gaps(env, monkeypatch, tmp_path, failure):
    packets = {'profile': adts(profile=0), 'rate': adts(rate_index=4),
               'channels': adts(channels=1), 'empty': b''}.get(failure)
    paths, _ = install_aac_stub(monkeypatch, warning=failure == 'warning', decode=failure != 'decode',
                                packets=packets, encode_code=1 if failure == 'encode' else 0)
    record = f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path,
                                    cbr_candidates_bps=(256000,))
    assert record.nominal_bitrates_bps == ()
    assert record.nominal_confirmation == 'not_applicable'
    assert not record.supports_settings
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize('channels,layout', [(2, None), (1, 'stereo'), (6, '5.1')])
def test_aac_unknown_or_incompatible_layout_never_borrows_stereo(env, monkeypatch, tmp_path, channels, layout):
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda *args, cancel_event=None: pytest.fail('must not query this layout'))
    record = f.collect_aac_evidence(env, 48000, channels, layout, tmp_path)
    assert record.nominal_bitrates_bps == () and 'no unchanged' in record.source_locator


def test_aac_unavailable_nominal_query_keeps_gap(env, monkeypatch, tmp_path):
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda *args, cancel_event=None: ())
    record = f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path)
    assert record.nominal_confirmation == 'not_applicable'


def test_aac_cancellation_removes_partial_encode(env, monkeypatch, tmp_path):
    paths, _ = install_aac_stub(monkeypatch)

    def cancel(argv, **kwargs):
        path = Path(argv[-1])
        paths.append(path)
        path.write_bytes(b'partial')
        raise Cancelled()

    monkeypatch.setattr(f, 'run_process', cancel)
    with pytest.raises(Cancelled):
        f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path, cbr_candidates_bps=(256000,))
    assert all(not path.exists() for path in paths)
    assert not list(tmp_path.glob('norma-*'))


def test_missing_aac_encoder_retains_environment_error(env, tmp_path):
    with pytest.raises(EnvironmentError_):
        f.collect_aac_evidence(replace(env, encoders=frozenset()), 48000, 2, 'stereo', tmp_path)


@pytest.mark.parametrize('value', [0, -1, 1 << 32, True, 256000.0])
def test_bad_nominal_candidate_rejected_before_query(env, monkeypatch, tmp_path, value):
    monkeypatch.setattr(f, '_aac_nominal_settings', lambda *args, cancel_event=None: pytest.fail('must not query'))
    with pytest.raises(ValueError):
        f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path, cbr_candidates_bps=(value,))


@pytest.mark.parametrize('data', [b'', b'garbage', adts()[:-1], adts() + b'x'])
def test_truncated_or_malformed_adts_is_not_evidence(data):
    with pytest.raises(ValueError):
        f._check_adts(data, 48000, 2)


@pytest.mark.real_ffmpeg
@pytest.mark.parametrize('rate', [44100, 48000])
def test_real_aac_and_mp3_v0_records(rate, tmp_path):
    env = f.probe_environment(f.locate_ffmpeg('/usr/local/bin/ffmpeg'))
    if 'aac_at' not in env.encoders or 'libmp3lame' not in env.encoders:
        pytest.skip('build lacks aac_at or libmp3lame')
    aac = f.collect_aac_evidence(env, rate, 2, 'stereo', tmp_path)
    assert 256000 in aac.nominal_bitrates_bps, aac.source_locator
    assert aac.nominal_confirmation == 'encoder_diagnostic'
    assert set(aac.nominal_bitrates_bps) <= set(f.AAC_CBR_CANDIDATES_BPS)
    assert f.revalidate_evidence(env, aac)
    mp3 = f.collect_mp3_v0_evidence(env, rate, 2, 'stereo', tmp_path)
    if f._matched_lossy_build(env):
        assert mp3.vbr_qualities == (0,), mp3.source_locator
        assert mp3.basis == 'build_matched_evidence'
        basis = ToolBasis(env.executable.path, env.executable.identity, env.executable.version, env.platform_basis)
        assert automatic_mp3_capability([mp3], basis, sample_rate_hz=rate,
                                        channels=2, channel_layout='stereo').supported
    else:
        assert bool(mp3.vbr_qualities) == (mp3.nominal_confirmation != 'not_applicable')
    assert not list(tmp_path.glob('norma-*'))
    assert {field.name for field in fields(CapabilityEvidence)} == {
        'evidence_id', 'executable_path', 'executable_identity', 'ffmpeg_version', 'platform_basis',
        'encoder', 'sample_rate_hz', 'channels', 'channel_layout', 'bitrate_mode', 'supported_bit_depths',
        'supports_float', 'nominal_bitrates_bps', 'vbr_qualities', 'basis', 'source_locator', 'nominal_confirmation'}


@pytest.mark.real_ffmpeg
def test_real_aac_clamp_is_not_evidenced(tmp_path):
    env = f.probe_environment(f.locate_ffmpeg('/usr/local/bin/ffmpeg'))
    if 'aac_at' not in env.encoders:
        pytest.skip('build lacks aac_at')
    record = f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path,
                                   cbr_candidates_bps=(256000, 384000))
    assert record.nominal_bitrates_bps == (256000,), record.source_locator


@pytest.mark.real_ffmpeg
def test_unknown_build_does_not_borrow_v0_primary_source(monkeypatch, tmp_path):
    env = f.probe_environment(f.locate_ffmpeg('/usr/local/bin/ffmpeg'))
    if 'libmp3lame' not in env.encoders:
        pytest.skip('build lacks libmp3lame')
    monkeypatch.setattr(f, '_matched_lossy_build', lambda env: False)
    record = f.collect_mp3_v0_evidence(env, 44100, 2, 'stereo', tmp_path)
    assert record.vbr_qualities == () and record.nominal_confirmation == 'not_applicable'
    assert 'gap' in record.source_locator


@pytest.mark.parametrize('failure', ['none', 'new', 'mode_set', 'mode_get', 'wrong_mode',
                                      'bitrate_set', 'bitrate_get', 'clamp', 'size'])
def test_audiotoolbox_query_checks_status_readback_and_disposes(monkeypatch, failure):
    from types import SimpleNamespace

    disposed = []
    nominal = [0]
    brat = int.from_bytes(b'brat', 'big')

    def new(source, target, converter):
        converter._obj.value = 123
        assert source._obj.channels == target._obj.channels == 2
        assert source._obj.rate == target._obj.rate == 48000
        return 1 if failure == 'new' else 0

    def set_property(converter, prop, size, value):
        if prop == brat:
            nominal[0] = value._obj.value
            return int(failure == 'bitrate_set')
        return int(failure == 'mode_set')

    def get_property(converter, prop, size, value):
        if prop == brat:
            value._obj.value = nominal[0] - 8000 if failure == 'clamp' else nominal[0]
            if failure == 'size':
                size._obj.value = 0
            return int(failure == 'bitrate_get')
        value._obj.value = 1 if failure == 'wrong_mode' else 0
        return int(failure == 'mode_get')

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    lib = SimpleNamespace(AudioConverterNew=Function(new),
                          AudioConverterSetProperty=Function(set_property),
                          AudioConverterGetProperty=Function(get_property),
                          AudioConverterDispose=Function(lambda converter: disposed.append(converter.value) or 0))
    monkeypatch.setattr(f.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(f.ctypes, 'CDLL', lambda path: lib)
    assert f._aac_nominal_settings(48000, 2, (256000,)) == ((256000,) if failure == 'none' else ())
    assert disposed == [123]


def test_audiotoolbox_missing_framework_is_a_gap(monkeypatch):
    monkeypatch.setattr(f.platform, 'system', lambda: 'Darwin')

    def absent(path):
        raise OSError('framework unavailable')

    monkeypatch.setattr(f.ctypes, 'CDLL', absent)
    assert f._aac_nominal_settings(48000, 2, (256000,)) == ()


@pytest.mark.parametrize('match', [True, False])
def test_only_matched_build_flush_note_is_excepted(env, monkeypatch, tmp_path, match):
    paths, calls = install_aac_stub(monkeypatch)
    base = f.run_process

    def flush_note(argv, **kwargs):
        if '-b:a' in argv:
            kwargs['on_stderr_line']('[aac_at @ 0x1234] Trying to remove 1024 samples, but the queue is empty')
        return base(argv, **kwargs)

    monkeypatch.setattr(f, '_matched_lossy_build', lambda env: match)
    monkeypatch.setattr(f, 'run_process', flush_note)
    record = f.collect_aac_evidence(env, 48000, 2, 'stereo', tmp_path, cbr_candidates_bps=(256000,))
    assert record.nominal_bitrates_bps == ((256000,) if match else ())
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize('quality,decode', [(0, True), (0, False), (70, True)])
def test_v0_source_proof_does_not_override_decode_failure_or_conflicting_quality(env, monkeypatch, tmp_path,
                                                                              quality, decode):
    from .test_capabilities import mp3_file

    def encode(env, source, outputs, *, cancel_event=None):
        for path, args in outputs:
            path.write_bytes(mp3_file(44100, [128000, 192000], xing=(b'Xing', quality)))
        return {path: True for path, args in outputs}, ()

    monkeypatch.setattr(f, '_encode_outputs', encode)
    monkeypatch.setattr(f, '_matched_lossy_build', lambda env: True)
    monkeypatch.setattr(f, '_decoded_evidence_ok', lambda *args, cancel_event=None: decode)
    record = f.collect_mp3_v0_evidence(env, 44100, 2, 'stereo', tmp_path)
    assert record.vbr_qualities == ((0,) if quality == 0 and decode else ())
    assert not list(tmp_path.glob('norma-*'))
