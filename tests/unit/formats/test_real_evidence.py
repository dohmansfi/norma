"""Static policy ∩ real evidence from the installed FFmpeg (marker ``real_ffmpeg``; TST-F-01, TST-C-01 data).

The WS-04 adapter produces the evidence with tiny encodes under tmp_path; this
test only checks that the intersection claims exactly what that evidence
supports and reports the rest as conservative gaps. It records the installed
build's behaviour; it is not a complete capability matrix.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import norma.formats.capabilities as cap
from norma.adapters import ffmpeg as f, ffprobe
from norma.errors import FileFailure, UsageError
from norma.formats.resolve import effective_channel_layout, finalise_target, resolve_target
from norma.models import CapabilityEvidence, ProbeResult, RunOptions, TargetFormatName

pytestmark = pytest.mark.real_ffmpeg

FFMPEG = Path("/usr/local/bin/ffmpeg")


def test_d081_ffmpeg_unmasked_stereo_wav_resolves_to_aac(tmp_path: Path) -> None:
    """A real unmasked PCM WAV, not a fabricated C-02 or capability record."""
    source = tmp_path / 'unmasked.wav'
    subprocess.run([str(FFMPEG), '-v', 'error', '-nostdin',
                    '-f', 'lavfi', '-i', 'sine=f=440:r=48000:d=1',
                    '-ac', '2', '-c:a', 'pcm_s16le', '-map_metadata', '-1',
                    '-fflags', '+bitexact', str(source)], check=True, timeout=60)
    original = source.read_bytes()
    probe = ffprobe.probe(ffprobe.locate_ffprobe(FFMPEG.with_name('ffprobe')), source)
    assert (probe.channels, probe.channel_layout, probe.sample_rate_hz) == (2, None, 48000)
    layout, assumed = effective_channel_layout(probe.channels, probe.channel_layout)
    assert (layout, assumed) == ('stereo', True)
    env = f.probe_environment(f.locate_ffmpeg(FFMPEG))
    basis = cap.ToolBasis(env.executable.path, env.executable.identity, env.executable.version, env.platform_basis)
    # TK-0420 will wire this same helper into pipeline evidence collection.
    record = f.collect_aac_evidence(env, probe.sample_rate_hz, probe.channels, layout, tmp_path / 'aac')
    assert 256000 in record.nominal_bitrates_bps, record.source_locator
    opts = RunOptions(command='normalise', format='aac', bitrate='256k',
                      ffmpeg=env.executable.path, cwd=tmp_path)
    target = resolve_target(probe, opts, (record,), basis)
    assert (target.format, target.encoder, target.channel_layout) == ('aac', 'aac_at', 'stereo')
    assert target.capability_basis == record.evidence_id
    assert (target.bitrate_mode, target.effective_bitrate_bps) == ('cbr', 256000)
    assert target.notes == ('channel layout not stated; assumed stereo',)
    assert not target.diagnostics and not target.stream_copy
    assert probe.channel_layout is None and source.read_bytes() == original


def test_static_policy_against_installed_evidence(tmp_path: Path) -> None:
    env = f.probe_environment(f.locate_ffmpeg(FFMPEG))
    basis = cap.ToolBasis(env.executable.path, env.executable.identity, env.executable.version, env.platform_basis)

    mp3_cbr, mp3_vbr = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path / "mp3")
    records = [mp3_cbr, mp3_vbr]
    records.append(f.collect_lossless_evidence(env, "flac", 96000, 2, "stereo", tmp_path / "flac"))
    records.append(f.collect_lossless_evidence(env, "alac", 96000, 2, "stereo", tmp_path / "alac"))
    for encoder in ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le"):
        records.append(f.collect_lossless_evidence(env, encoder, 48000, 2, "stereo", tmp_path / encoder))
    assert all(path.is_relative_to(tmp_path) for path in tmp_path.rglob("*"))

    # MP3 CBR: exactly the adapter's validated choices, nothing added.
    cbr = cap.lossy_capability("mp3", "cbr", records, basis, sample_rate_hz=44100, channels=2,
                               channel_layout="stereo")
    assert cbr.supported and cbr.nominal_bitrates_bps == tuple(sorted(mp3_cbr.nominal_bitrates_bps))
    assert 128000 in cbr.nominal_bitrates_bps and cbr.evidence_id == mp3_cbr.evidence_id

    # Automatic MP3 (V0): claimed only if the record confirms V0, otherwise the N-3 conservative gap.
    automatic = cap.automatic_mp3_capability(records, basis, sample_rate_hz=44100, channels=2,
                                             channel_layout="stereo")
    v0_evidenced = mp3_vbr.nominal_confirmation != "not_applicable" and 0 in mp3_vbr.vbr_qualities
    assert automatic.supported == v0_evidenced
    if not v0_evidenced:
        assert automatic.reason == "unsupported_target" and automatic.vbr_qualities == ()

    # Missing AAC records still claim nothing; the S3 collector is checked below.
    aac = cap.lossy_capability("aac", "cbr", records, basis, sample_rate_hz=48000, channels=2,
                               channel_layout="stereo")
    assert not aac.supported and aac.reason == "unsupported_target"

    # Lossless: native FLAC/ALAC 16/24 at 96 kHz; WAV integer and float at 48 kHz.
    for target in ("flac", "alac"):
        result = cap.lossless_capability(target, records, basis, sample_rate_hz=96000, channels=2,
                                         channel_layout="stereo")
        assert [r.representation.bit_depth for r in result.representations] == [16, 24]
    wav = cap.lossless_capability("wav", records, basis, sample_rate_hz=48000, channels=2, channel_layout="stereo")
    assert wav.representation(24) is not None and wav.representation(32, True) is not None
    assert not cap.lossless_capability("aiff", records, basis, sample_rate_hz=48000, channels=2,
                                       channel_layout="stereo").supported

    # A changed platform basis is not reusable support.
    other = cap.ToolBasis(basis.executable_path, basis.executable_identity, basis.ffmpeg_version, "macOS other")
    assert not cap.lossy_capability("mp3", "cbr", records, other, sample_rate_hz=44100, channels=2,
                                    channel_layout="stereo").supported


@pytest.mark.parametrize('rate', [44100, 48000])
def test_s3_installed_lossy_evidence_resolves_and_finalises(tmp_path: Path, rate: int) -> None:
    """Real collector records, synthetic C-02 input; no user audio or library access."""
    env = f.probe_environment(f.locate_ffmpeg(FFMPEG))
    basis = cap.ToolBasis(env.executable.path, env.executable.identity, env.executable.version, env.platform_basis)
    aac = f.collect_aac_evidence(env, rate, 2, 'stereo', tmp_path / 'aac')
    mp3 = f.collect_mp3_v0_evidence(env, rate, 2, 'stereo', tmp_path / 'mp3')
    assert 256000 in aac.nominal_bitrates_bps, aac.source_locator
    assert aac.nominal_confirmation == 'encoder_diagnostic'
    assert mp3.vbr_qualities == (0,), mp3.source_locator
    assert mp3.nominal_confirmation != 'not_applicable'
    records = (aac, mp3)
    source = ProbeResult('pcm_s24le', 'lossless', 'wav', 0, 0, rate, 24, False, 's32',
                         2, 'stereo', 1.0, 'stream', None, False, 0, {}, env.executable.version)
    targets: tuple[tuple[TargetFormatName, CapabilityEvidence], ...] = (('aac', aac), ('mp3', mp3))
    for codec, record in targets:
        opts = RunOptions(command='normalise', format=codec, ffmpeg=env.executable.path, cwd=tmp_path)
        target = resolve_target(source, opts, records, basis)
        assert target.encoder == record.encoder and target.capability_basis == record.evidence_id
        assert not target.stream_copy and target.bitrate_source == 'auto'
        if codec == 'aac':
            assert (target.bitrate_mode, target.requested_bitrate_bps,
                    target.effective_bitrate_bps, target.vbr_quality) == ('cbr', 256000, 256000, None)
        else:
            assert (target.bitrate_mode, target.requested_bitrate_bps,
                    target.effective_bitrate_bps, target.vbr_quality) == ('vbr', None, None, 0)
        lossy_source = ProbeResult(codec, 'lossy', codec, 0, 0, rate, None, False, 'fltp',
                                   2, 'stereo', 1.0, 'stream', 101234, False, 0, {}, env.executable.version)
        explicit = RunOptions(command='normalise', bitrate='256k' if codec == 'aac' else 'V0',
                              value_sources={'bitrate': 'cli'}, ffmpeg=env.executable.path, cwd=tmp_path)
        copied = finalise_target(lossy_source, explicit, records, basis, action='copy')
        encoded = finalise_target(lossy_source, explicit, records, basis, action='linear')
        assert copied.stream_copy and copied.capability_basis == 'not_applicable'
        assert 'from the command line' in copied.notes[0]
        assert not encoded.stream_copy and encoded.capability_basis == record.evidence_id
        assert encoded.bitrate_source == 'explicit' and not encoded.notes
        # Real stereo records must not grant mono or unknown-layout support.
        for layout in ('mono', None):
            capability = cap.lossy_capability(codec, target.bitrate_mode, records, basis,
                                              sample_rate_hz=rate, channels=1, channel_layout=layout)
            assert not capability.supported
        other = cap.ToolBasis(basis.executable_path, basis.executable_identity,
                              basis.ffmpeg_version, 'changed platform')
        with pytest.raises(FileFailure, match='unsupported_target'):
            resolve_target(source, opts, records, other)
        with pytest.raises(UsageError):
            finalise_target(lossy_source, explicit, records, other, action='copy')
