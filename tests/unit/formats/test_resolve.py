"""Hermetic F2–F16/C-04 decisions using synthetic, injected evidence only."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from norma.errors import EnvironmentError_, FileFailure, SkipFile, UsageError
from norma.formats import capabilities as cap
from norma.formats.resolve import _choose, effective_channel_layout, finalise_target, preflight_targets, provisional_target, resolve_target
from norma.models import CapabilityEvidence, ProbeResult, RunOptions, TargetFormat

BASIS = cap.ToolBasis(Path('/synthetic/ffmpeg'), (1, 2), 'synthetic-test', 'synthetic-platform')


def probe(codec='pcm_s24le', container='wav', depth: int | None = 24, floating=False, rate=48000,
          channels=2, layout: str | None = 'stereo', bitrate=None):
    return ProbeResult(codec, 'lossy' if codec in ('mp3', 'aac') else 'lossless', container,
                       0, 0, rate, depth, floating, 'test', channels, layout, 10.0, 'stream',
                       bitrate, False, 0, {}, 'synthetic-test')


def options(**kwargs):
    return RunOptions(command='normalise', ffmpeg=BASIS.executable_path,
                      cwd=Path('/synthetic'), **kwargs)


def records(rate=48000, channels=2, layout: str | None = 'stereo', bitrates=(64000, 128000, 256000, 320000),
            qualities=tuple(range(10))):
    result = []
    for target, spec in cap.TARGETS.items():
        for encoder in spec.encoders:
            for mode in spec.bitrate_modes:
                result.append(CapabilityEvidence(
                    f'ev:{encoder}:{mode}:{rate}:{channels}:{layout}', BASIS.executable_path,
                    BASIS.executable_identity, BASIS.ffmpeg_version, BASIS.platform_basis,
                    encoder, rate, channels, layout, mode,
                    tuple(r.bit_depth for r in spec.representations if r.encoder == encoder and not r.sample_float),
                    any(r.sample_float for r in spec.representations if r.encoder == encoder),
                    bitrates if mode == 'cbr' else (), qualities if mode == 'vbr' else (),
                    'installed_query', 'synthetic unit test',
                    'not_applicable' if mode == 'none' else 'validated_setting'))
    return tuple(result)


def resolve(p=None, opts=None, evidence=None, **kwargs):
    p = p or probe()
    return resolve_target(p, opts or options(), records(p.sample_rate_hz) if evidence is None else evidence,
                          BASIS, **kwargs)


def codes(target):
    return {d.code for d in target.diagnostics}


@pytest.mark.parametrize('source', [
    probe(), probe('pcm_s24be', 'aiff'), probe('flac', 'flac'), probe('alac', 'mp4'),
    probe(container='mp4'), probe(container='mov'),
    probe('mp3', 'mp3', depth=None), probe('aac', 'adts', depth=None),
    replace(probe('opus'), codec_class='unsupported'),
    probe(channels=3, layout=None), probe(channels=6, layout='5.1'),
    replace(probe('opus', channels=6, layout='5.1'), codec_class='unsupported'),
])
@pytest.mark.parametrize('fmt', ['same', 'apple', 'wav', 'aiff', 'flac', 'alac', 'mp3', 'aac'])
@pytest.mark.parametrize('policy', ['reencode', 'skip', 'lossless'])
@pytest.mark.parametrize('allow', [False, True])
def test_provisional_target_matches_private_policy(source, fmt, policy, allow):
    opts = options(format=fmt, lossy_sources=policy, allow_lossy_transcode=allow)
    try:
        expected = _choose(source, opts)
    except (SkipFile, FileFailure) as error:
        with pytest.raises(type(error)) as public_error:
            provisional_target(source, opts)
        assert str(public_error.value) == str(error)
        assert public_error.value.reason == error.reason
        assert public_error.value.stage == error.stage
        assert public_error.value.status == error.status
    else:
        assert provisional_target(source, opts) == expected


@pytest.mark.parametrize('target', ['wav', 'aiff', 'flac', 'alac', 'mp3', 'aac'])
def test_provisional_choice_does_not_claim_capability_support(target):
    source, opts = probe(), options(format=target)
    assert provisional_target(source, opts) == (target, 'as_requested')
    with pytest.raises(FileFailure, match='unsupported_target'):
        resolve_target(source, opts, (), BASIS)


def test_provisional_choice_does_not_validate_encoding_options():
    source = probe('mp3', 'mp3', depth=None)
    opts = options(format='aac', lossy_sources='skip', bit_depth='float', bitrate='V0')
    assert provisional_target(source, opts) == ('mp3', 'skipped_lossy')
    opts = options(format='aac', bit_depth='float', bitrate='V0', allow_lossy_transcode=True)
    assert provisional_target(source, opts) == ('aac', 'as_requested')
    with pytest.raises(UsageError):
        resolve_target(source, opts, (), BASIS)


@pytest.mark.parametrize('target', ['wav', 'aiff', 'flac', 'alac'])
@pytest.mark.parametrize('depth,floating', [(8, False), (16, False), (20, False), (24, False),
                                            (32, False), (32, True), (None, False)])
def test_every_lossless_f4_row_and_d064(target, depth, floating):
    expected = (32, True) if floating and target == 'wav' else (
        (24, False) if floating else
        ({8: 16, 20: 24, None: 24}.get(depth, depth), False))
    if expected == (32, False) and target != 'wav':
        expected = (24, False)
    result = resolve(probe(depth=depth, floating=floating), options(format=target))
    assert (result.bit_depth, result.sample_float) == expected
    assert ('format_adjusted' in codes(result)) == (expected != (depth, floating))
    if expected != (depth, floating):
        assert any('representation adjusted' in n for n in result.notes)
    assert result.dither == ('triangular' if expected == (16, False) else 'none')
    assert not result.stream_copy
    assert result.capability_basis.startswith('ev:')
    assert (result.requested_bitrate_bps, result.effective_bitrate_bps, result.vbr_quality,
            result.bitrate_mode, result.bitrate_source) == (None, None, None, 'none', 'not_applicable')


@pytest.mark.parametrize('container', ['mp4', 'mov'])
@pytest.mark.parametrize('depth', [8, 20, None, 24, 32])
def test_pcm_in_mp4_mov_same_native_alac(container, depth):
    result = resolve(probe(container=container, depth=depth))
    assert (result.format, result.encoder, result.extension) == ('alac', 'alac', '.m4a')
    assert 'format_adjusted' in codes(result)
    assert any('D-064' in note for note in result.notes)


@pytest.mark.parametrize('container', ['mp4', 'mov'])
def test_explicit_format_and_depth_override_d064(container):
    result = resolve(probe(container=container, depth=8), options(format='wav', bit_depth='32'))
    assert (result.format, result.bit_depth) == ('wav', 32)
    assert 'format_adjusted' not in codes(result)


@pytest.mark.parametrize('codec,container,target,extension', [
    ('pcm_s24le', 'wav', 'wav', '.wav'), ('pcm_s24be', 'aiff', 'aiff', '.aiff'),
    ('pcm_s24le', 'caf', 'wav', '.wav'), ('flac', 'flac', 'flac', '.flac'),
    ('alac', 'caf', 'alac', '.m4a'), ('aac', 'adts', 'aac', '.m4a'),
    ('aac', 'caf', 'aac', '.m4a'), ('mp3', 'mp3', 'mp3', '.mp3'),
])
def test_same_codec_containers(codec, container, target, extension):
    assert (r := resolve(probe(codec, container), source_suffix='.aiff')).format == target
    assert r.extension == extension


@pytest.mark.parametrize('codec', ['mp3', 'aac'])
@pytest.mark.parametrize('fmt', ['same', 'apple', 'wav', 'aiff', 'flac', 'alac', 'mp3', 'aac'])
def test_d026_fallback_every_family(codec, fmt):
    result = resolve(probe(codec, codec, depth=None), options(format=fmt, lossy_sources='lossless'))
    expected = fmt if fmt in ('wav', 'aiff', 'flac', 'alac') else ('alac' if fmt == 'apple' else 'flac')
    assert (result.format, result.bit_depth, result.dither, result.decision) == (expected, 16, 'triangular', 'lossy_origin')
    assert codes(result) == {'lossy_origin'}


@pytest.mark.parametrize('codec,target', [('mp3', 'aac'), ('aac', 'mp3')])
def test_lossy_guard_and_allow(codec, target):
    p = probe(codec, codec)
    kept = resolve(p, options(format=target))
    assert (kept.format, kept.decision) == (codec, 'kept_source_codec')
    assert codes(kept) == {'lossy_transcode_blocked'}
    allowed = resolve(p, options(format=target, allow_lossy_transcode=True))
    assert (allowed.format, allowed.decision) == (target, 'as_requested')


@pytest.mark.parametrize('codec', ['mp3', 'aac'])
def test_apple_keeps_lossy_codec(codec):
    assert resolve(probe(codec, codec), options(format='apple')).format == codec


@pytest.mark.parametrize('target', ['mp3', 'aac'])
@pytest.mark.parametrize('source_rate,expected', [(88200, 44100), (176400, 44100), (96000, 48000), (192000, 48000)])
def test_f5_rate_families(target, source_rate, expected):
    result = resolve(probe(rate=source_rate), options(format=target), records(expected))
    assert result.sample_rate_hz == expected
    assert 'format_adjusted' in codes(result)


@pytest.mark.parametrize('target', ['mp3', 'aac'])
def test_unknown_high_rate_not_invented(target):
    with pytest.raises(FileFailure, match='unsupported_target'):
        resolve(probe(rate=64000), options(format=target), records(48000))


@pytest.mark.parametrize('target', ['mp3', 'aac'])
def test_explicit_rate_cap(target):
    with pytest.raises(UsageError, match='policy cap'):
        resolve(opts=options(format=target, sample_rate=96000), evidence=records(96000))


@pytest.mark.parametrize('target', ['flac', 'alac'])
def test_explicit_lossless_rate_cap(target):
    with pytest.raises(UsageError, match='policy cap'):
        resolve(opts=options(format=target, sample_rate=384000), evidence=records(384000))


def test_supported_low_aac_rate_never_upsampled_for_floor():
    r = resolve(probe(rate=22050), options(format='aac'), records(22050, bitrates=(96000,)))
    assert (r.sample_rate_hz, r.requested_bitrate_bps, r.effective_bitrate_bps) == (22050, 256000, 96000)
    assert codes(r) == {'aac_bitrate_adjusted'}


@pytest.mark.parametrize('source_bitrate,choices,requested,effective', [
    (None, (256000, 320000), 256000, 256000), (270000, (256000, 320000), 270000, 320000),
    (400000, (256000, 320000), 400000, 320000), (64000, (128000,), 256000, 128000),
])
def test_aac_discrete_automatic_choices(source_bitrate, choices, requested, effective):
    r = resolve(probe('aac', 'mp4', bitrate=source_bitrate), options(format='aac'), records(bitrates=choices))
    assert (r.requested_bitrate_bps, r.effective_bitrate_bps) == (requested, effective)
    assert ('aac_bitrate_adjusted' in codes(r)) == (requested != effective)


@pytest.mark.parametrize('codec,container,depth,floating,bitrate', [
    ('pcm_s16le', 'wav', 16, False, 1536000), ('pcm_s24le', 'wav', 24, False, 2304000),
    ('pcm_f32le', 'wav', 32, True, 3072000), ('flac', 'flac', 24, False, 1800000),
    ('alac', 'm4a', 16, False, 900000), ('pcm_s16be', 'aiff', 16, False, 1411200),
])
def test_b006_lossless_source_bitrate_never_drives_automatic_aac(codec, container, depth, floating, bitrate):
    source = probe(codec, container, depth, floating, bitrate=bitrate)
    assert source.codec_class == 'lossless'
    r = resolve(source, options(format='aac'), records())
    assert (r.requested_bitrate_bps, r.effective_bitrate_bps, r.bitrate_source) == (256000, 256000, 'auto')
    assert not r.diagnostics and not codes(r)
    assert not any('adjusted' in n for n in r.notes)
    assert resolve(source, options(format='aac'), records(bitrates=(128000, 256000))).effective_bitrate_bps == 256000


def test_b006_lossless_source_with_only_low_choices_still_uses_floor_rule():
    r = resolve(probe(bitrate=1536000, rate=22050), options(format='aac'), records(22050, bitrates=(96000,)))
    assert (r.requested_bitrate_bps, r.effective_bitrate_bps) == (256000, 96000)
    assert codes(r) == {'aac_bitrate_adjusted'}


def test_b006_lossy_source_bitrate_unchanged_for_automatic_aac():
    r = resolve(probe('aac', 'mp4', bitrate=270000), options(format='aac'), records(bitrates=(256000, 320000)))
    assert (r.requested_bitrate_bps, r.effective_bitrate_bps) == (270000, 320000)
    assert codes(r) == {'aac_bitrate_adjusted'}


def test_explicit_aac_below_floor_honoured_without_warning():
    r = resolve(opts=options(format='aac', bitrate='128k', value_sources={'bitrate': 'config'}))
    assert (r.requested_bitrate_bps, r.effective_bitrate_bps, r.bitrate_source) == (128000, 128000, 'explicit')
    assert not r.diagnostics
    assert any('from the config' in n and 'AAC floor' in n for n in r.notes)


@pytest.mark.parametrize('opts', [options(format='flac', bit_depth='float'),
                                  options(format='alac', bit_depth='32'),
                                  options(format='mp3', bit_depth='24'),
                                  options(format='aac', bitrate='V0'),
                                  options(format='aac', bitrate='300k')])
def test_invalid_explicit_combinations_usage(opts):
    with pytest.raises(UsageError):
        resolve(opts=opts)


@pytest.mark.parametrize('target', ['wav', 'flac', 'mp3', 'aac'])
def test_automatic_no_evidence_failure(target):
    with pytest.raises(FileFailure) as error:
        resolve(opts=options(format=target), evidence=())
    assert (error.value.reason, error.value.status) == ('unsupported_target', 'failed')


@pytest.mark.parametrize('kwargs', [{'bit_depth': '24'}, {'sample_rate': 48000},
                                    {'format': 'aac', 'bitrate': '256k'},
                                    {'format': 'mp3', 'bitrate': 'V0'}])
def test_explicit_no_evidence_usage(kwargs):
    with pytest.raises(UsageError):
        resolve(opts=options(**kwargs), evidence=())


@pytest.mark.parametrize('change', [{'ffmpeg_version': 'other'}, {'executable_identity': (9, 9)},
                                     {'platform_basis': 'other'}, {'executable_path': Path('/other/tool')},
                                     {'channel_layout': 'mono'}, {'channels': 1}, {'sample_rate_hz': 44100},
                                     {'nominal_confirmation': 'not_applicable'}, {'nominal_bitrates_bps': ()}])
def test_aac_nonmatching_or_unconfirmable_evidence(change):
    ev = next(r for r in records() if r.encoder == 'aac_at')
    ev = replace(ev, **change)
    with pytest.raises(FileFailure):
        resolve(opts=options(format='aac'), evidence=(ev,))
    with pytest.raises(UsageError):
        resolve(opts=options(format='aac', bitrate='256k'), evidence=(ev,))


@pytest.mark.parametrize('channels,layout', [(1, 'mono'), (2, 'stereo')])
@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav'])
def test_d081_unmasked_inputs_use_standard_layout_and_note(channels, layout, target):
    source = probe(channels=channels, layout=None)
    result = resolve(source, options(format=target), records(channels=channels, layout=layout))
    assert result.channel_layout == layout
    assert result.notes == (f'channel layout not stated; assumed {layout}',)
    assert not result.diagnostics
    assert source.channel_layout is None
    if target == 'mp3':
        assert result.vbr_quality == 0
    elif target == 'aac':
        assert result.effective_bitrate_bps == 256000


@pytest.mark.parametrize('channels,layout,expected', [
    (1, None, ('mono', True)), (2, None, ('stereo', True)), (3, None, (None, False)),
    (1, 'mono', ('mono', False)), (2, 'stereo', ('stereo', False)),
    (2, '2.0', ('2.0', False)), (3, '3.0', ('3.0', False)),
    (2, '', ('', False)),
])
def test_d081_helper_preserves_stated_and_other_unknown_layouts(channels, layout, expected):
    assert effective_channel_layout(channels, layout) == expected


@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav'])
def test_d081_three_channel_unmasked_input_keeps_refusal(target):
    with pytest.raises(SkipFile, match='unsupported_channels'):
        resolve(probe(channels=3, layout=None), options(format=target), records())


@pytest.mark.parametrize('channels,layout', [(1, 'mono'), (2, 'stereo'), (2, '2.0')])
@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav'])
def test_d081_stated_layouts_unchanged(channels, layout, target):
    result = resolve(probe(channels=channels, layout=layout), options(format=target),
                     records(channels=channels, layout=layout))
    assert result.channel_layout == layout and result.notes == ()
    assert not result.diagnostics


@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav'])
def test_d081_assumption_still_requires_matching_evidence(target):
    with pytest.raises(FileFailure, match='unsupported_target'):
        resolve(probe(layout=None), options(format=target), records(channels=1, layout='mono'))
    with pytest.raises(FileFailure, match='unsupported_target'):
        resolve(probe(layout='2.0'), options(format=target), records())


@pytest.mark.parametrize('codec,container', [('pcm_s24le', 'wav'), ('mp3', 'mp3'), ('aac', 'mp4')])
def test_d081_assumed_layout_does_not_force_unchanged_copy_to_encode(codec, container):
    source = probe(codec=codec, container=container, layout=None)
    copied = finalise_target(source, options(), records(), BASIS, action='copy')
    encoded = finalise_target(source, options(), records(), BASIS, action='linear')
    assert copied.stream_copy and not encoded.stream_copy
    assert copied.channel_layout == encoded.channel_layout == 'stereo'
    assert copied.notes == encoded.notes == ('channel layout not stated; assumed stereo',)


@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav'])
def test_d081_effective_layout_evidence_wins_over_own_unstated_layout(target):
    both = records(layout='stereo') + records(layout=None)
    result = resolve(probe(layout=None), options(format=target), both)
    assert result.channel_layout == 'stereo'
    assert result.capability_basis.endswith(':stereo')
    assert result.notes == ('channel layout not stated; assumed stereo',)


@pytest.mark.parametrize('channels', [1, 2])
@pytest.mark.parametrize('target', ['mp3', 'aac', 'wav', 'flac'])
def test_d081_own_unstated_layout_evidence_still_resolves_without_assumption(channels, target):
    """A record collected at the input's own (absent) layout is exact, not borrowed."""
    result = resolve(probe(channels=channels, layout=None), options(format=target),
                     records(channels=channels, layout=None))
    assert result.channel_layout is None
    assert result.capability_basis.endswith(':None')
    assert result.notes == ()


def test_d081_unconfirmed_own_layout_aac_record_does_not_rescue_the_assumption():
    """The real collector gives an unmasked AAC query no settings; nothing may be invented."""
    with pytest.raises(FileFailure, match='unsupported_target'):
        resolve(probe(layout=None), options(format='aac'), records(layout=None, bitrates=()))
    with pytest.raises(UsageError, match='stereo'):
        resolve(probe(layout=None), options(format='aac', bitrate='256k'), records(layout=None, bitrates=()))


def test_d081_note_order_and_copy_with_own_layout_evidence():
    source = probe(codec='mp3', container='mp3', layout=None)
    kept = resolve(source, options(format='aac'), records())
    assert kept.notes[0] == 'channel layout not stated; assumed stereo'
    assert kept.decision == 'kept_source_codec'
    copied = finalise_target(probe(layout=None), options(), records(layout=None), BASIS, action='copy')
    assert copied.stream_copy and copied.channel_layout is None and copied.notes == ()


def test_mp3_auto_requires_v0_no_cbr_fallback():
    with pytest.raises(FileFailure):
        resolve(opts=options(format='mp3'), evidence=records(qualities=(1, 2)))
    r = resolve(opts=options(format='mp3', bitrate='V2'), evidence=records(qualities=(1, 2)))
    assert (r.bitrate_mode, r.vbr_quality, r.requested_bitrate_bps) == ('vbr', 2, None)
    assert any('MP3 V0 quality' in n for n in r.notes)


def test_mp3_explicit_cbr_nominal_not_stream_average():
    r = resolve(probe('mp3', 'mp3', bitrate=101234), options(bitrate='320k'))
    assert (r.bitrate_mode, r.requested_bitrate_bps, r.effective_bitrate_bps, r.vbr_quality) == ('cbr', 320000, 320000, None)


@pytest.mark.parametrize('provenance', ['cli', 'config'])
def test_mixed_lossless_lossy_option_applicability(provenance):
    opts = options(format='apple', bit_depth='16', dither='triangular', bitrate='256k',
                   value_sources={k: provenance for k in ('bit_depth', 'dither', 'bitrate')})
    resolved, notes = preflight_targets((probe(), probe('mp3', 'mp3')), opts, records(), BASIS)
    lossless, lossy = resolved
    assert isinstance(lossless, TargetFormat) and isinstance(lossy, TargetFormat)
    assert (lossless.format, lossless.bit_depth) == ('alac', 16)
    assert any('--bitrate' in n and 'ignored' in n for n in lossless.notes)
    assert (lossy.format, lossy.effective_bitrate_bps) == ('mp3', 256000)
    assert any('--bit-depth' in n and 'ignored' in n for n in lossy.notes)
    assert any('--dither' in n and 'ignored' in n for n in lossy.notes)
    assert not notes
    assert not lossy.diagnostics


def test_explicit_depth_after_d026_effective_lossless_not_usage():
    r = resolve(probe('mp3', 'mp3'), options(format='mp3', lossy_sources='lossless', bit_depth='24'))
    assert (r.format, r.bit_depth) == ('flac', 24)


def test_batch_aggregates_only_applicable_invalid_files():
    opts = options(format='same', bitrate='V0', inputs=tuple(Path('/synthetic') / n for n in ('wav', 'aac1', 'mp3', 'aac2')))
    with pytest.raises(UsageError) as error:
        preflight_targets((probe(), probe('aac', 'mp4'), probe('mp3', 'mp3'), probe('aac', 'adts')),
                           opts, records(), BASIS, input_paths=opts.inputs)
    text = str(error.value)
    assert '/synthetic/aac1:' in text and '/synthetic/aac2:' in text
    assert '/synthetic/wav:' not in text and '/synthetic/mp3:' not in text


def test_automatic_failure_does_not_hide_later_usage_error():
    opts = options(bit_depth='32')
    with pytest.raises(UsageError, match=r'input\[1\]'):
        preflight_targets((probe('mp3', 'mp3'), probe('flac', 'flac')), opts, (), BASIS)


def test_automatic_failures_continue_and_return_in_order():
    result, notes = preflight_targets((None, probe(), probe('aac', 'mp4')), options(),
                                     tuple(r for r in records() if r.encoder != 'aac_at'), BASIS)
    assert result[0] is None
    assert isinstance(result[1], TargetFormat) and isinstance(result[2], FileFailure)
    assert result[1].format == 'wav'
    assert result[2].reason == 'unsupported_target'
    assert not notes


def test_lossy_skip_ignores_invalid_encoding_options_and_evidence():
    opts = options(format='aac', lossy_sources='skip', bit_depth='float', bitrate='V0')
    result, notes = preflight_targets((None, probe('mp3', 'mp3')), opts, (), BASIS,
                                     available_encoders=frozenset())
    assert isinstance(result[1], TargetFormat)
    assert result[1].decision == 'skipped_lossy'
    assert result[1].capability_basis == 'not_applicable'
    assert len(notes) == 1


def test_bitrate_n12_exactly_one_when_no_effective_lossy_target():
    result, notes = preflight_targets((probe(), probe('flac', 'flac')), options(bitrate='256k'), records(), BASIS)
    assert len(notes) == 1 and all(isinstance(r, TargetFormat) and r.bitrate_source == 'not_applicable' for r in result)
    assert preflight_targets((), options(), (), BASIS)[1] == ()


def test_missing_required_encoder_environment_not_capability_gap():
    with pytest.raises(EnvironmentError_) as error:
        preflight_targets((probe(),), options(), (), BASIS, available_encoders=frozenset())
    assert error.value.exit_code == 3
    assert 'pcm_s24le' in str(error.value)
    result, _ = preflight_targets((probe(),), options(), (), BASIS, available_encoders=frozenset({'pcm_s24le'}))
    assert isinstance(result[0], FileFailure)
    assert result[0].reason == 'unsupported_target'


@pytest.mark.parametrize('codec', ['opus', 'pcm_mulaw'])
def test_unsupported_input_skipped_before_options(codec):
    with pytest.raises(SkipFile, match='unsupported_codec'):
        resolve(replace(probe(codec), codec_class='unsupported'), options(format='aac', bit_depth='float'))


def test_multichannel_no_downmix():
    with pytest.raises(SkipFile, match='unsupported_channels'):
        resolve(probe(channels=6, layout='5.1'))


@pytest.mark.parametrize('codec,container,depth', [('pcm_s16le', 'wav', 16), ('alac', 'mp4', 24),
                                               ('flac', 'flac', 24), ('mp3', 'mp3', None), ('aac', 'mp4', None)])
def test_copy_records_apply_no_encoder_setting(codec, container, depth):
    p = probe(codec, container, depth=depth)
    opts = options(bitrate='128k', value_sources={'bitrate': 'config'})
    provisional = resolve(p, opts)
    final = finalise_target(p, opts, records(), BASIS, action='copy')
    assert not provisional.stream_copy and final.stream_copy
    assert (final.encoder, final.capability_basis, final.dither) == ('stream_copy', 'not_applicable', 'none')
    assert (final.bitrate_mode, final.requested_bitrate_bps, final.effective_bitrate_bps,
            final.vbr_quality, final.bitrate_source) == ('none', None, None, None, 'not_applicable')
    if codec in ('mp3', 'aac'):
        assert len([n for n in final.notes if 'not applied' in n]) == 1
        assert any('128k' in n and 'from the config' in n for n in final.notes)
        assert not any('overrides automatic' in n for n in final.notes)
    with pytest.raises(FrozenInstanceError):
        final.stream_copy = False


def test_copy_retry_reapplies_setting_without_stale_note():
    p, opts, evidence = probe('mp3', 'mp3'), options(bitrate='256k'), records()
    copied = finalise_target(p, opts, evidence, BASIS, action='copy')
    encoded = finalise_target(p, opts, evidence, BASIS, action='linear')
    assert copied.stream_copy and not encoded.stream_copy
    assert (encoded.encoder, encoded.effective_bitrate_bps, encoded.bitrate_source) == ('libmp3lame', 256000, 'explicit')
    assert not any('not applied' in n for n in encoded.notes)
    assert not encoded.capability_basis == 'not_applicable'


def test_automatic_aac_adjustment_removed_only_from_copy_record():
    p, opts, ev = probe('aac', 'mp4', bitrate=400000), options(), records()
    copied = finalise_target(p, opts, ev, BASIS, action='copy')
    encoded = finalise_target(p, opts, ev, BASIS, action='linear')
    assert not copied.diagnostics and not copied.notes
    assert 'aac_bitrate_adjusted' in codes(encoded)


@pytest.mark.parametrize('kwargs', [{'sample_rate': 44100}, {'bit_depth': '16'}, {'format': 'flac'}])
def test_requested_conversion_defeats_copy(kwargs):
    p, opts = probe(), options(**kwargs)
    rate = 44100 if kwargs.get('sample_rate') == 44100 else 48000
    assert not finalise_target(p, opts, records(rate), BASIS, action='copy').stream_copy


def test_lossless_encoder_endianness_change_not_copy():
    assert not finalise_target(probe('pcm_s24be', 'caf'), options(), records(), BASIS, action='copy').stream_copy


def test_skipped_audio_plan_has_no_copy_note():
    result = finalise_target(probe('mp3', 'mp3'), options(bitrate='256k'), records(), BASIS, action='skip')
    assert not result.stream_copy and not any('not applied' in n for n in result.notes)


def test_copy_cannot_bypass_explicit_or_auto_preflight():
    p = probe('mp3', 'mp3')
    with pytest.raises(UsageError):
        finalise_target(p, options(bitrate='256k'), (), BASIS, action='copy')
    with pytest.raises(FileFailure):
        finalise_target(p, options(), (), BASIS, action='copy')


def test_pure_domain_imports_and_inert_initialiser():
    # The project source is read only; no tools, subprocesses or audio files are used.
    import norma.formats.resolve as module
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('norma.'):
            assert node.module in ('norma.models', 'norma.errors', 'norma.formats')
        if isinstance(node, ast.Import):
            assert all(n.name not in ('subprocess', 'mutagen') for n in node.names)


@pytest.mark.parametrize('fmt,policy,bitrate,codecs,expected,n12', [
    ('same', 'reencode', '256k', ('pcm_s24le', 'flac'), ('wav', 'flac'), True),  # B1
    ('alac', 'reencode', '320k', ('pcm_s24le', 'mp3'), ('alac', 'alac'), True),  # B2
    ('apple', 'reencode', '256k', ('pcm_s24le', 'mp3'), ('alac', 'mp3'), False),  # B3
    ('same', 'reencode', '256k', ('pcm_s24le', 'aac'), ('wav', 'aac'), False),  # B4
    ('mp3', 'lossless', '320k', ('pcm_s24le', 'mp3', 'aac'), ('mp3', 'flac', 'flac'), False),  # B5
    ('mp3', 'lossless', '320k', ('mp3', 'aac'), ('flac', 'flac'), True),  # B6
    ('apple', 'lossless', '256k', ('pcm_s24le', 'aac'), ('alac', 'alac'), True),  # B7
    ('same', 'skip', '256k', ('pcm_s24le', 'aac'), ('wav', 'aac'), True),  # B8
])
@pytest.mark.parametrize('provenance', ['cli', 'config'])
def test_f8_worked_batches_b1_b8(fmt, policy, bitrate, codecs, expected, n12, provenance):
    ps = tuple(probe(codec, 'wav' if codec.startswith('pcm_') else codec) for codec in codecs)
    results, notes = preflight_targets(ps, options(format=fmt, lossy_sources=policy, bitrate=bitrate,
                                                  value_sources={'bitrate': provenance}), records(), BASIS)
    assert all(isinstance(r, TargetFormat) for r in results)
    assert tuple(r.format for r in results if isinstance(r, TargetFormat)) == expected
    assert len(notes) == int(n12)
    for result in results:
        assert isinstance(result, TargetFormat)
        if result.format in ('wav', 'flac', 'alac', 'aiff') or result.decision == 'skipped_lossy':
            assert (result.bitrate_mode, result.requested_bitrate_bps, result.effective_bitrate_bps,
                    result.vbr_quality, result.bitrate_source) == ('none', None, None, None, 'not_applicable')
        else:
            nominal = int(bitrate[:-1]) * 1000
            assert (result.bitrate_mode, result.requested_bitrate_bps, result.effective_bitrate_bps,
                    result.vbr_quality, result.bitrate_source) == ('cbr', nominal, nominal, None, 'explicit')
    if policy == 'skip':
        assert isinstance(results[1], TargetFormat) and results[1].decision == 'skipped_lossy'


def test_b9_guard_revalidates_vbr_against_retained_aac():
    opts = options(format='mp3', bitrate='V0', inputs=(Path('/synthetic/wav'), Path('/synthetic/aac')))
    with pytest.raises(UsageError) as error:
        preflight_targets((probe(), probe('aac', 'mp4')), opts, records(), BASIS, input_paths=opts.inputs)
    assert '/synthetic/aac:' in str(error.value) and '/synthetic/wav:' not in str(error.value)


def test_b10_explicit_aac_never_rounded_or_clamped():
    with pytest.raises(UsageError):
        resolve(opts=options(format='aac', bitrate='300k'))


@pytest.mark.parametrize('target,token,depth,floating', [
    ('wav', '16', 16, False), ('wav', '24', 24, False), ('wav', '32', 32, False),
    ('wav', 'float', 32, True), ('aiff', '32', 32, False), ('flac', '16', 16, False),
    ('alac', '24', 24, False),
])
def test_supported_explicit_representations_override_auto(target, token, depth, floating):
    r = resolve(probe(depth=8), options(format=target, bit_depth=token))
    assert (r.bit_depth, r.sample_float) == (depth, floating)
    assert 'format_adjusted' not in codes(r)


@pytest.mark.parametrize('target', ['wav', 'aiff', 'flac', 'alac', 'mp3', 'aac'])
def test_explicit_rate_validated_at_requested_rate_for_each_target(target):
    r = resolve(opts=options(format=target, sample_rate=44100), evidence=records(44100))
    assert r.sample_rate_hz == 44100
    with pytest.raises(UsageError):
        resolve(opts=options(format=target, sample_rate=44100), evidence=records(48000))


@pytest.mark.parametrize('dither', ['none', 'triangular'])
def test_dither_explicit_provenance_and_float(dither):
    opts = options(bit_depth='16', dither=dither, value_sources={'dither': 'config'})
    integer = resolve(opts=opts)
    assert integer.dither == dither
    assert any('from the config' in n and '--dither' in n for n in integer.notes)
    floating = resolve(opts=replace(opts, bit_depth='float'))
    assert floating.dither == 'none'
    assert any('ignored for float' in n for n in floating.notes)


def test_duplicate_evidence_never_grants_support():
    ev = next(r for r in records() if r.encoder == 'pcm_s24le')
    duplicate = replace(ev, evidence_id='duplicate')
    with pytest.raises(FileFailure, match='ambiguous'):
        resolve(evidence=(ev, duplicate))
    with pytest.raises(UsageError, match='ambiguous'):
        resolve(opts=options(bit_depth='24'), evidence=(ev, duplicate))


def test_same_rate_auto_and_default_options_not_explicit_overrides():
    opts = options(bit_depth='same', sample_rate='same', dither='auto', bitrate='auto',
                   explicit_fields=frozenset({'bit_depth', 'sample_rate', 'dither', 'bitrate'}))
    with pytest.raises(FileFailure):
        resolve(opts=opts, evidence=())


def test_unsupported_extra_depth_conservatively_refused():
    with pytest.raises(FileFailure, match='representation policy'):
        resolve(probe(depth=12))


def test_copy_pcm_mp4_d064_still_requires_conversion():
    r = finalise_target(probe(container='mp4'), options(), records(), BASIS, action='copy')
    assert not r.stream_copy and r.format == 'alac'


def test_n12_exact_cli_wording_and_config_provenance():
    _, notes = preflight_targets((probe(),), options(bitrate='256k', value_sources={'bitrate': 'config'}), records(), BASIS)
    assert notes == ('--bitrate 256k (from the config) ignored — no file in this run uses a lossy-output bitrate setting',)


def test_available_encoder_check_uses_guarded_target_only():
    r, _ = preflight_targets((probe('aac', 'mp4'),), options(format='mp3'), records(), BASIS,
                             available_encoders=frozenset({'aac_at'}))
    assert isinstance(r[0], TargetFormat) and r[0].format == 'aac'


def test_preflight_aggregates_invalid_depth_with_environment_set():
    ps = (probe('flac', 'flac'), probe('alac', 'mp4'))
    with pytest.raises(UsageError) as error:
        preflight_targets(ps, options(bit_depth='float'), records(), BASIS, available_encoders=frozenset())
    assert 'input[0]' in str(error.value) and 'input[1]' in str(error.value)


def test_suffix_alignment_and_iterator_consumption():
    with pytest.raises(ValueError, match='one entry per probe'):
        preflight_targets((probe(),), options(), records(), BASIS, source_suffixes=())
    ps = (probe('pcm_s24be', 'aiff'), probe())
    result, _ = preflight_targets(iter(ps), options(), iter(records()), BASIS,
                                  source_suffixes=iter(('.aiff', '.wav')))
    assert isinstance(result[0], TargetFormat) and result[0].extension == '.aiff'
    assert isinstance(result[1], TargetFormat) and result[1].extension == '.wav'


def test_initialiser_has_no_executable_statements():
    import norma.formats as module
    tree = ast.parse(Path(module.__file__).read_text())
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)


def test_explicit_dither_copy_is_named_ignored_not_applied():
    result = finalise_target(probe('pcm_s16le', depth=16),
                             options(dither='triangular', value_sources={'dither': 'config'}),
                             records(), BASIS, action='copy')
    assert result.stream_copy and result.dither == 'none'
    assert len(result.notes) == 1
    assert '--dither triangular (from the config) ignored' in result.notes[0]
    assert not result.diagnostics


def test_discovery_paths_not_cli_folder_arguments_name_usage_errors():
    opts = options(format='aac', bitrate='300k', inputs=(Path('/synthetic/folder'),))
    paths = (Path('/synthetic/folder/a.wav'), Path('/synthetic/folder/b.wav'))
    with pytest.raises(UsageError) as error:
        preflight_targets((probe(), probe()), opts, records(), BASIS, input_paths=iter(paths))
    assert all(f'{path}:' in str(error.value) for path in paths)
    assert '/synthetic/folder:' not in str(error.value)
    with pytest.raises(ValueError, match='input_paths must have one entry'):
        preflight_targets((probe(), probe()), opts, records(), BASIS, input_paths=opts.inputs)


def test_without_discovered_paths_refusal_uses_honest_ordinal():
    opts = options(format='aac', bitrate='300k', inputs=(Path('/synthetic/folder'),))
    with pytest.raises(UsageError) as error:
        preflight_targets((probe(),), opts, records(), BASIS)
    assert 'input[0]:' in str(error.value) and '/synthetic/folder:' not in str(error.value)


def _extra_streams(count):
    return replace(probe(), ignored_streams=count)


@pytest.mark.parametrize('count,noun', [(1, 'audio stream'), (2, 'audio streams')])
def test_ignored_streams_note_names_the_count(count, noun):
    note = resolve(_extra_streams(count)).notes
    assert [n for n in note if 'not carried over' in n] == [
        f'{count} extra {noun} not carried over; only the first audio stream is processed']


def test_no_ignored_streams_means_no_note():
    assert not [n for n in resolve().notes if 'not carried over' in n]


def test_ignored_streams_note_adds_no_diagnostic_code():
    assert codes(resolve(_extra_streams(1))) == codes(resolve())


@pytest.mark.parametrize('action', ['copy', 'linear', 'limit', 'reduce', 'dynamic'])
def test_ignored_streams_note_survives_finalisation(action):
    p = _extra_streams(1)
    target = finalise_target(p, options(), records(p.sample_rate_hz), BASIS, action=action)
    assert any('1 extra audio stream not carried over' in n for n in target.notes)


def test_ignored_streams_note_dropped_when_no_output_is_planned():
    p = _extra_streams(1)
    target = finalise_target(p, options(), records(p.sample_rate_hz), BASIS, action='skip')
    assert not [n for n in target.notes if 'not carried over' in n]


@pytest.mark.parametrize('codec', ['mp3', 'aac'])
@pytest.mark.parametrize('provenance', ['cli', 'config'])
@pytest.mark.parametrize('bitrate', ['256k', 'auto'])
def test_b11_b12_copy_records_and_run_notes(codec, provenance, bitrate):
    p = probe(codec, 'mp3' if codec == 'mp3' else 'mp4', depth=None)
    opts = options(bitrate=bitrate, value_sources={'bitrate': provenance})
    provisional, run_notes = preflight_targets((p,), opts, records(), BASIS)
    copied = finalise_target(p, opts, records(), BASIS, action='copy')
    assert isinstance(provisional[0], TargetFormat)
    assert not provisional[0].stream_copy and copied.stream_copy
    assert not run_notes and not copied.diagnostics
    assert copied.decision == 'as_requested'
    assert (copied.encoder, copied.capability_basis, copied.dither) == ('stream_copy', 'not_applicable', 'none')
    assert (copied.bitrate_mode, copied.requested_bitrate_bps, copied.effective_bitrate_bps,
            copied.vbr_quality, copied.bitrate_source) == ('none', None, None, None, 'not_applicable')
    if bitrate == 'auto':
        assert copied.notes == ()
    else:
        source = 'from the command line' if provenance == 'cli' else 'from the config'
        assert len(copied.notes) == 1
        assert '256k' in copied.notes[0] and source in copied.notes[0]


@pytest.mark.parametrize('codec,bitrate', [('mp3', '256k'), ('mp3', 'V0'), ('mp3', 'V2'), ('aac', '128k')])
@pytest.mark.parametrize('provenance', ['cli', 'config'])
@pytest.mark.parametrize('action', ['linear', 'limit', 'reduce', 'dynamic'])
def test_b13_retry_from_original_inputs_supersedes_copy(codec, bitrate, provenance, action):
    p = probe(codec, 'mp3' if codec == 'mp3' else 'mp4', depth=None, bitrate=101234)
    opts = options(bitrate=bitrate, value_sources={'bitrate': provenance})
    preflight_targets((p,), opts, records(), BASIS)
    copied = finalise_target(p, opts, records(), BASIS, action='copy')
    encoded = finalise_target(p, opts, records(), BASIS, action=action)
    assert copied.stream_copy and copied.effective_bitrate_bps is None
    assert encoded is not copied and not encoded.stream_copy
    assert encoded.encoder == cap.TARGETS[codec].encoders[0]
    assert encoded.capability_basis.startswith('ev:') and encoded.bitrate_source == 'explicit'
    assert not any('not applied' in note for note in encoded.notes)
    assert sum('not applied' in note for note in copied.notes) == 1
    if bitrate.startswith('V'):
        assert (encoded.bitrate_mode, encoded.vbr_quality, encoded.requested_bitrate_bps,
                encoded.effective_bitrate_bps) == ('vbr', int(bitrate[1:]), None, None)
    else:
        assert (encoded.bitrate_mode, encoded.vbr_quality, encoded.requested_bitrate_bps,
                encoded.effective_bitrate_bps) == ('cbr', None, int(bitrate[:-1]) * 1000, int(bitrate[:-1]) * 1000)
    # lossy_reencoded describes completed rendering, never a resolver prediction.
    assert 'lossy_reencoded' not in codes(encoded)


@pytest.mark.parametrize('provenance', ['cli', 'config'])
@pytest.mark.parametrize('case', ['B3', 'B9', 'B10', 'B11'])
def test_invalid_bitrate_cases_name_only_applicable_files_before_copy(case, provenance):
    fmt, bitrate, codec = {
        'B3': ('apple', '256k', 'mp3'), 'B9': ('mp3', 'V0', 'aac'),
        'B10': ('aac', '300k', 'pcm_s24le'), 'B11': ('same', '256k', 'mp3'),
    }[case]
    p = probe(codec, 'wav' if codec.startswith('pcm_') else codec)
    opts = options(format=fmt, bitrate=bitrate, value_sources={'bitrate': provenance})
    evidence = records(bitrates=(128000, 320000)) if case in ('B3', 'B11') else records()
    paths = (Path('/synthetic/valid.wav'), Path('/synthetic/refused.audio'))
    ps = (probe(), p) if case in ('B3', 'B9', 'B11') else (p,)
    paths = paths if len(ps) == 2 else paths[1:]
    with pytest.raises(UsageError) as error:
        preflight_targets(ps, opts, evidence, BASIS, input_paths=paths)
    assert '/synthetic/refused.audio:' in str(error.value)
    assert '/synthetic/valid.wav:' not in str(error.value)
    with pytest.raises(UsageError):
        finalise_target(p, opts, evidence, BASIS, action='copy')


@pytest.mark.parametrize('codec,target', [('mp3', 'aac'), ('aac', 'mp3')])
@pytest.mark.parametrize('allow', [False, True])
@pytest.mark.parametrize('action', ['copy', 'linear'])
def test_f14_guard_finalises_effective_codec_not_requested_codec(codec, target, allow, action):
    p = probe(codec, codec, depth=None)
    opts = options(format=target, allow_lossy_transcode=allow)
    result = finalise_target(p, opts, records(), BASIS, action=action)
    assert result.format == (target if allow else codec)
    assert result.stream_copy == (action == 'copy' and not allow)
    assert ('lossy_transcode_blocked' in codes(result)) == (not allow)
    assert result.decision == ('as_requested' if allow else 'kept_source_codec')


@pytest.mark.parametrize('container', ['adts', 'caf', 'mp4'])
def test_aac_container_change_alone_does_not_force_encode(container):
    p = probe('aac', container, depth=None)
    result = finalise_target(p, options(), records(), BASIS, action='copy')
    assert result.stream_copy and result.extension == '.m4a'


@pytest.mark.parametrize('codec', ['mp3', 'aac'])
def test_lossy_rate_conversion_defeats_copy(codec):
    p = probe(codec, codec, rate=96000, depth=None)
    result = finalise_target(p, options(), records(48000), BASIS, action='copy')
    assert not result.stream_copy and result.sample_rate_hz == 48000
    assert codes(result) == {'format_adjusted'}
