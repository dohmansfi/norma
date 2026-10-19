"""C-04 capability evidence: header readers and evidence decisions (fake encodes)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from norma.adapters import ffmpeg as f
from norma.errors import EnvironmentError_

# --- Synthetic headers -------------------------------------------------------------------

_RATE_INDEX = {44100: ("1", 0), 48000: ("1", 1), 32000: ("1", 2), 22050: ("2", 0), 24000: ("2", 1),
               16000: ("2", 2), 11025: ("2.5", 0), 12000: ("2.5", 1), 8000: ("2.5", 2)}
_VERSION_BITS = {"1": 0b11, "2": 0b10, "2.5": 0b00}
_BITRATES = {"1": (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
             "2": (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)}


def mp3_frame(rate: int, bps: int, mode: int, xing: tuple[bytes, int | None] | None = None) -> bytes:
    version, rate_index = _RATE_INDEX[rate]
    table = _BITRATES["1" if version == "1" else "2"]
    header = (0x7FF << 21) | (_VERSION_BITS[version] << 19) | (0b01 << 17) | (1 << 16)
    header |= table.index(bps // 1000) << 12 | rate_index << 10 | mode << 6
    length = (144000 if version == "1" else 72000) * (bps // 1000) // rate
    frame = bytearray(length)
    frame[:4] = header.to_bytes(4, "big")
    if xing is not None:
        tag, quality = xing
        side = (17 if mode == 3 else 32) if version == "1" else (9 if mode == 3 else 17)
        flags = 0x8 if quality is not None else 0
        body = tag + flags.to_bytes(4, "big") + (quality.to_bytes(4, "big") if quality is not None else b"")
        frame[4 + side:4 + side + len(body)] = body
    return bytes(frame)


def mp3_file(rate: int, bitrates: list[int], mode: int = 1, xing: tuple[bytes, int | None] | None = None) -> bytes:
    frames = [mp3_frame(rate, bitrates[0], mode, xing)] + [mp3_frame(rate, b, mode) for b in bitrates[1:]]
    return b"".join(frames)


def wav_bytes(rate: int, channels: int, bits: int, fmt: int, extensible: bool = True, valid: int | None = None) -> bytes:
    if extensible:
        guid = fmt.to_bytes(2, "little") + bytes.fromhex("000000001000800000aa00389b71")
        body = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * channels * bits // 8, channels * bits // 8,
                           bits, 22, valid or bits, 3) + guid
    else:
        body = struct.pack("<HHIIHH", fmt, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    chunk = b"fmt " + len(body).to_bytes(4, "little") + body
    junk = b"JUNK" + (3).to_bytes(4, "little") + b"abc\x00"
    payload = b"WAVE" + junk + chunk + b"data" + (0).to_bytes(4, "little")
    return b"RIFF" + len(payload).to_bytes(4, "little") + payload


def extended(value: float) -> bytes:
    exponent, mantissa = 16383 + 63, int(value)
    while mantissa and mantissa < (1 << 63):
        mantissa <<= 1
        exponent -= 1
    return exponent.to_bytes(2, "big") + mantissa.to_bytes(8, "big")


def aiff_bytes(rate: int, channels: int, bits: int, compression: bytes | None) -> bytes:
    comm = struct.pack(">HIH", channels, 100, bits) + extended(rate)
    if compression is not None:
        comm += compression + b"\x00"
    form = b"AIFC" if compression is not None else b"AIFF"
    payload = form + b"COMM" + len(comm).to_bytes(4, "big") + comm + (b"\x00" if len(comm) & 1 else b"")
    return b"FORM" + len(payload).to_bytes(4, "big") + payload


def flac_bytes(rate: int, channels: int, bits: int) -> bytes:
    packed = (rate << 12) | ((channels - 1) << 9) | ((bits - 1) << 4)
    info = bytes(10) + packed.to_bytes(4, "big") + bytes(20)
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + info


def alac_bytes(rate: int, channels: int, bits: int) -> bytes:
    cookie = (b"\x00\x00\x00\x24alac" + bytes(4) + (4096).to_bytes(4, "big") + bytes([0, bits, 40, 10, 14, channels])
              + bytes(2) + bytes(4) + bytes(4) + rate.to_bytes(4, "big"))
    return b"\x00\x00\x00\x20ftypM4A " + bytes(24) + b"....stsd....alac" + bytes(20) + cookie


# --- Header readers ----------------------------------------------------------------------


def test_read_mp3_frames_and_xing() -> None:
    data = b"ID3\x03\x00\x00\x00\x00\x00\x02xx" + mp3_file(44100, [128000, 128000, 96000], xing=(b"Xing", 90))
    frames = f.read_mp3_frames(data)
    assert [frame.bitrate_bps for frame in frames] == [128000, 128000, 96000]
    assert all(frame.sample_rate_hz == 44100 and frame.version == "1" for frame in frames)
    xing = f.read_xing_tag(data, frames[0])
    assert xing is not None and xing.tag == b"Xing" and xing.quality == 90
    low = f.read_mp3_frames(mp3_file(8000, [64000], mode=3))
    assert low[0].version == "2.5" and low[0].channel_mode == 3
    assert f.read_xing_tag(mp3_file(8000, [64000], mode=3), low[0]) is None
    with pytest.raises(ValueError):
        f.read_mp3_frames(mp3_file(44100, [128000]) + b"garbage!")
    with pytest.raises(ValueError):
        f.read_mp3_frames(b"")


def test_read_pcm_headers() -> None:
    assert f.read_wav_header(wav_bytes(96000, 2, 32, 3)) == f.PcmHeader(2, 96000, 32, True)
    assert f.read_wav_header(wav_bytes(48000, 2, 32, 1, valid=24)) == f.PcmHeader(2, 48000, 24, False)
    assert f.read_wav_header(wav_bytes(44100, 1, 16, 1, extensible=False)) == f.PcmHeader(1, 44100, 16, False)
    assert f.read_aiff_header(aiff_bytes(44100, 2, 24, None)) == f.PcmHeader(2, 44100, 24, False)
    assert f.read_aiff_header(aiff_bytes(96000, 2, 32, b"fl32")) == f.PcmHeader(2, 96000, 32, True)
    assert f.read_flac_header(flac_bytes(96000, 2, 24)) == f.PcmHeader(2, 96000, 24, False)
    assert f.read_alac_header(alac_bytes(48000, 2, 24)) == f.PcmHeader(2, 48000, 24, False)
    for reader in (f.read_wav_header, f.read_aiff_header, f.read_flac_header, f.read_alac_header):
        with pytest.raises(ValueError):
            reader(b"nothing here at all, not audio")
    with pytest.raises(ValueError):
        f.read_aiff_header(aiff_bytes(44100, 2, 16, b"ulaw"))


# --- Evidence decisions with fake encodes ----------------------------------------------------


@pytest.fixture(autouse=True)
def _encoder_stats_supported(request, monkeypatch):
    """CR-067: fake-FFmpeg tests assume the statistics options exist; the probe has its own tests."""
    if "real_ffmpeg" not in request.keywords:
        monkeypatch.setattr(f, "probe_mp3_encoder_stats", lambda env, *, cancel_event=None: True)


@pytest.fixture
def env() -> f.FFmpegEnvironment:
    exe = f.Executable(Path("/usr/local/bin/ffmpeg"), (1, 2), "N-test")
    return f.FFmpegEnvironment(exe, frozenset({"libmp3lame", "flac", "alac", "pcm_s24le", "pcm_f32le", "pcm_s16be"}),
                               frozenset(), "macOS test arm64")


def fake_encoder(monkeypatch: pytest.MonkeyPatch, writer) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def encode(env, source, outputs, *, cancel_event=None):
        calls.append(source)
        written = {}
        for path, args in outputs:
            data = writer(path, args)
            if data is not None:
                path.write_bytes(data)
            written[path] = data is not None
        return written, ("fake stderr",)

    monkeypatch.setattr(f, "_encode_outputs", encode)
    return calls


def arg(args: tuple[str, ...], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def test_mp3_cbr_accepts_only_unchanged_frame_bitrates(env, monkeypatch, tmp_path) -> None:
    def writer(path, args):
        if arg(args, "-b:a"):
            bps = int(arg(args, "-b:a"))
            applied = bps if bps in (64000, 128000, 320000) else 96000  # silent clamp, as LAME does
            if bps == 256000:
                return mp3_file(48000, [256000])  # wrong rate: rejected
            if bps == 72000:
                return None  # encode failure
            return mp3_file(44100, [applied, applied])
        return mp3_file(44100, [128000, 192000], xing=(b"Xing", 0))

    calls = fake_encoder(monkeypatch, writer)
    cbr, vbr = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path / "ev")
    assert cbr.nominal_bitrates_bps == (64000, 96000, 128000, 320000)
    assert cbr.nominal_confirmation == "validated_setting" and cbr.basis == "installed_query"
    assert (cbr.encoder, cbr.bitrate_mode, cbr.sample_rate_hz, cbr.channels) == ("libmp3lame", "cbr", 44100, 2)
    assert cbr.executable_identity == (1, 2) and cbr.ffmpeg_version == "N-test"
    assert "-protocol_whitelist" in calls[0] and "lavfi" in calls[0]
    # Xing quality 0 does not confirm any requested V level: explicit conservative gap.
    assert vbr.vbr_qualities == () and vbr.nominal_confirmation == "not_applicable"
    assert not vbr.supports_settings and "gap" in vbr.source_locator
    assert cbr.evidence_id != vbr.evidence_id


def test_mp3_vbr_confirmed_only_by_matching_quality_indicator(env, monkeypatch, tmp_path) -> None:
    def writer(path, args):
        quality = arg(args, "-q:a")
        if quality is None:
            return mp3_file(44100, [128000])
        return mp3_file(44100, [128000, 160000], xing=(b"Xing", 100 - 10 * int(quality)))

    fake_encoder(monkeypatch, writer)
    _, vbr = f.collect_mp3_evidence(env, 44100, 2, "stereo", tmp_path / "ev", cbr_candidates_bps=(128000,))
    assert vbr.vbr_qualities == tuple(range(10)) and vbr.nominal_confirmation == "validated_setting"


def test_mp3_layout_mismatch_and_total_failure(env, monkeypatch, tmp_path) -> None:
    fake_encoder(monkeypatch, lambda path, args: mp3_file(44100, [128000], mode=1))
    cbr, _ = f.collect_mp3_evidence(env, 44100, 1, "mono", tmp_path / "a", cbr_candidates_bps=(128000,))
    assert cbr.nominal_bitrates_bps == () and cbr.nominal_confirmation == "not_applicable"
    # A 5.1 request that FFmpeg silently downmixed to stereo frames is never evidence.
    cbr, _ = f.collect_mp3_evidence(env, 44100, 6, "5.1", tmp_path / "c", cbr_candidates_bps=(128000,))
    assert cbr.nominal_bitrates_bps == ()
    fake_encoder(monkeypatch, lambda path, args: None)
    cbr, vbr = f.collect_mp3_evidence(env, 44100, 6, "5.1", tmp_path / "b")
    assert cbr.nominal_bitrates_bps == () and vbr.vbr_qualities == ()
    assert "fake stderr" in cbr.source_locator


def test_missing_encoder_is_environment_error(env, tmp_path) -> None:
    bare = f.FFmpegEnvironment(env.executable, frozenset(), frozenset(), "x")
    with pytest.raises(EnvironmentError_):
        f.collect_mp3_evidence(bare, 44100, 2, "stereo", tmp_path)
    with pytest.raises(ValueError):
        f.collect_lossless_evidence(env, "alac_at", 44100, 2, "stereo", tmp_path)
    with pytest.raises(ValueError):
        f.collect_lossless_evidence(env, "pcm_s24le", 44100, 2, "stereo;rm", tmp_path)


def test_lossless_depths_come_from_written_headers(env, monkeypatch, tmp_path) -> None:
    def writer(path, args):
        if arg(args, "-c:a") == "flac":
            # The 24-bit request really writes 24; a 16-bit request that came out 24 would not count.
            return flac_bytes(96000, 2, 24 if arg(args, "-sample_fmt") == "s32" else 16)
        if arg(args, "-c:a") == "alac":
            return alac_bytes(96000, 2, 16)  # both candidates write 16-bit: only 16 is evidenced
        if arg(args, "-c:a") == "pcm_f32le":
            return wav_bytes(96000, 2, 32, 3)
        if arg(args, "-c:a") == "pcm_s24le":
            return wav_bytes(96000, 2, 24, 1)
        return aiff_bytes(48000, 2, 16, None)  # wrong rate

    fake_encoder(monkeypatch, writer)
    flac = f.collect_lossless_evidence(env, "flac", 96000, 2, "stereo", tmp_path)
    assert flac.supported_bit_depths == (16, 24) and not flac.supports_float
    assert (flac.bitrate_mode, flac.nominal_confirmation) == ("none", "not_applicable")
    assert f.collect_lossless_evidence(env, "alac", 96000, 2, "stereo", tmp_path).supported_bit_depths == (16,)
    float_wav = f.collect_lossless_evidence(env, "pcm_f32le", 96000, 2, "stereo", tmp_path)
    assert float_wav.supports_float and float_wav.supported_bit_depths == ()
    assert f.collect_lossless_evidence(env, "pcm_s24le", 96000, 2, "stereo", tmp_path).supported_bit_depths == (24,)
    assert f.collect_lossless_evidence(env, "pcm_s16be", 96000, 2, "stereo", tmp_path).supported_bit_depths == ()


def test_revalidate_evidence(env, monkeypatch, tmp_path, make_tool) -> None:
    tool = make_tool("ffmpeg", "print('ffmpeg version N-1 x')")
    real_env = f.FFmpegEnvironment(f.Executable(tool, f.executable_identity(tool), "N-1"), env.encoders,
                                   frozenset(), "macOS test arm64")
    fake_encoder(monkeypatch, lambda path, args: flac_bytes(44100, 2, 16))
    record = f.collect_lossless_evidence(real_env, "flac", 44100, 2, "stereo", tmp_path)
    assert f.revalidate_evidence(real_env, record)
    other = f.FFmpegEnvironment(real_env.executable, env.encoders, frozenset(), "macOS other")
    assert not f.revalidate_evidence(other, record)
    tool.unlink()
    assert not f.revalidate_evidence(real_env, record)
