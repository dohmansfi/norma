"""TK-0113 / CR-063: exact MP3 length and zero priming at every rate and frame residue.

IC-3 F-1: FFmpeg's mp3 muxer takes the LAME tag's delay from the first
packet's skip side data only, so at MPEG-2 rates (576-sample frames) the
1,105-sample priming that spans two packets is stored as 47 instead of 576
(+529 leading frames), and at some 32 kHz residues the padding is 38 short.
The render now rewrites both fields from the encoder's own sample count.

Real-tool tests run the installed FFmpeg and write only under tmp_path, with
generated sources (seed-1020 Gaussian noise, as the IC-3 reproduction).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import norma.audio.render as render_module
from norma import cli
from norma.adapters import ffmpeg, ffprobe
from norma.audio.measure import measure
from norma.audio.render import (
    MP3_DECODER_DELAY,
    MP3_ENCODER_STATS,
    Mp3EncoderCount,
    _crc16_arc,
    render,
    render_argv,
    restore_mp3_gapless,
)
from norma.audio.verify import check_candidate_length, trailing_padding_allowance
from norma.errors import FileFailure
from norma.models import GainPlan, ProbeResult, TargetFormat

RATES = (22050, 24000, 32000, 44100, 48000)
RESIDUES = (0, 1, 137, 511)


# --- Synthetic records ---------------------------------------------------------------------------


def linear_plan(gain=-2.0) -> GainPlan:
    return GainPlan(action="linear", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0,
                    tolerance_lu=.5, max_limit_db=3.0, gain_db=gain, predicted_true_peak_dbtp=-3.0,
                    limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None,
                    expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok")


def mp3_target(rate=48000, channels=2, layout="stereo", *, mode="vbr", bps=None, basis="ev:synthetic"):
    quality = 0 if mode == "vbr" else None
    return TargetFormat("mp3", ".mp3", "libmp3lame", None, False, rate, channels, layout, "none", mode, bps, bps,
                        quality, "auto", basis, False, "as_requested")


def wav_target(rate, channels, layout, basis):
    return TargetFormat("wav", ".wav", "pcm_s24le", 24, False, rate, channels, layout, "none", "none", None, None,
                        None, "not_applicable", basis, False, "as_requested")


def probe_of(rate=48000, channels=2) -> ProbeResult:
    return ProbeResult("pcm_s24le", "lossless", "wav", 0, 0, rate, 24, False, "s32", channels,
                       "stereo" if channels == 2 else "mono", 30.0, "stream", None, False, 0, {}, "synthetic")


def stats(samples: list[int], packets: int, priming: int = 1105, rate: int = 22050) -> list[str]:
    tb = f"1/{rate}"
    lines, pts = [], 0
    for n, count in enumerate(samples):
        lines.append(f"pre {n} {pts} {tb} {count}")
        pts += count
    return lines + [f"post {n} {n * 576 - priming} {tb}" for n in range(packets)]


# --- Pure: argv, encoder account, CRC ------------------------------------------------------------


def test_mp3_renders_ask_the_encoder_for_its_account_and_nothing_else_does():
    source, candidate = Path("/unused/s.wav"), Path("/unused/c.mp3")
    argv = render_argv(Path("/unused/ffmpeg"), source, candidate, linear_plan(), probe_of(), mp3_target())
    start = argv.index("-stats_enc_pre")
    assert argv[start:start + len(MP3_ENCODER_STATS)] == MP3_ENCODER_STATS
    assert argv[-3:] == ("-f", "mp3", str(candidate))
    for fmt in ("aac", "wav"):
        target = (wav_target(48000, 2, "stereo", "ev:x") if fmt == "wav" else
                  TargetFormat("aac", ".m4a", "aac_at", None, False, 48000, 2, "stereo", "none", "cbr", 256000,
                               256000, None, "auto", "ev:x", False, "as_requested"))
        other = render_argv(Path("/unused/ffmpeg"), source, Path(f"/unused/c.{fmt}"), linear_plan(), probe_of(),
                            target)
        assert "-stats_enc_pre" not in other and "-stats_enc_post" not in other


def test_encoder_account_sums_samples_and_reads_the_priming():
    count = Mp3EncoderCount()
    for line in ["noise from elsewhere", *stats([576, 576, 371], 5)]:
        count.feed(line)
    assert count.totals(22050) == (1523, 1105, 5)


@pytest.mark.parametrize("lines,why", [
    ([], "no frames or packets"),
    (stats([576], 0), "no frames or packets"),
    (["pre 0 0 1/22050 576", "pre 2 576 1/22050 576", "post 0 -1105 1/22050"], "out of order"),
    (["pre 0 0 1/22050 x", "post 0 -1105 1/22050"], "unreadable"),
    (stats([576], 1, rate=44100), "time base"),
    (stats([576], 1, priming=528), "not the expected 576-sample encoder delay"),
])
def test_an_incomplete_encoder_account_is_a_render_error(lines, why):
    count = Mp3EncoderCount()
    for line in lines:
        count.feed(line)
    with pytest.raises(FileFailure) as caught:
        count.totals(22050)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "render") and why in caught.value.detail


def account(lines, rate=22050):
    count = Mp3EncoderCount()
    for line in lines:
        count.feed(line)
    return count.totals(rate)


def test_a_shorter_flushed_last_packet_is_legitimate():
    """The real encoder's last packet can sit less than a frame after the one before (seen at 32 kHz, +137)."""
    lines = [f"pre {n} {n * 1152} 1/32000 1152" for n in range(4)] + [
        "post 0 -1105 1/32000", "post 1 47 1/32000", "post 2 1199 1/32000", "post 3 2351 1/32000",
        "post 4 3494 1/32000"]
    assert account(lines, 32000) == (4608, 1105, 5)


@pytest.mark.parametrize("rate,frame", [(8000, 576), (22050, 576), (24000, 576), (32000, 1152), (48000, 1152)])
def test_a_legitimate_account_passes_at_every_frame_size(rate, frame):
    lines = [f"pre {n} {n * frame} 1/{rate} {frame}" for n in range(3)] + [f"pre 3 {3 * frame} 1/{rate} 100"]
    lines += [f"post {n} {n * frame - 1105} 1/{rate}" for n in range(6)]
    assert account(lines, rate) == (3 * frame + 100, 1105, 6)


def test_the_review_reproduction_with_a_shifted_first_post_pts_is_refused():
    """IC-3 recheck F-3: the first post timestamp moved by 100 samples, later lines real (32 kHz)."""
    lines = stats([1152] * 3, 4, rate=32000)
    lines = [line.replace("post 0 -1105", "post 0 -1205") for line in lines]
    for rate_lines in (lines, [line.replace("post 0 -1205", "post 0 -1005") for line in lines]):
        with pytest.raises(FileFailure) as caught:
            account(rate_lines, 32000)
        assert (caught.value.reason, caught.value.stage) == ("render_error", "render")
        assert "frame size" in caught.value.detail or "not the expected" in caught.value.detail


@pytest.mark.parametrize("lines,why", [
    (["pre 0 0 1/22050 576", "pre 1 bananas garbage 1152", "post 0 -1105 1/22050"], "unreadable"),
    (["pre 0 0 1/22050 576", "pre 1 576 1/22050 576", "post 0 -1105 1/22050", "post 1 bananas garbage"], "unreadable"),
    (["pre 0 0 1/22050 576", "pre 1 500 1/22050 576", "post 0 -1105 1/22050"], "before the previous frame ended"),
    (["pre 0 0 1/22050 576", "pre 1 576 1/44100 576", "post 0 -1105 1/22050"], "time base"),
    (["pre 0 0 1/22050 576", "post 0 -1105 1/22050", "post 1 -529 1/44100"], "time base"),
    (["pre 0 0 1/22050 576", "post 0 -1105 22050"], "time base"),
    (["pre 0 0 1/22050 576", "post 0 -1105 1/22050", "post 1 -529 1/22050", "post 2 100 1/22050",
      "post 3 676 1/22050"], "packet 2 starts at 100, not at 47 where the 576-sample frame size"),
    (["pre 0 0 1/22050 576", "post 0 -1105 1/22050", "post 1 -529 1/22050", "post 2 100 1/22050"],
     "last packet starts at 100, beyond 47"),
    (["pre 0 0 1/22050 576", "post 0 -1105 1/22050", "post 1 -1105 1/22050"], "advances by"),
    (["pre 0 0 1/22050 576", "post 0 -1105 1/22050", "post 1 47 1/22050"], "last packet starts at 47, beyond -529"),
    (["pre 0 0 1/22050 576", "post 0 -1000 1/22050"], "not the expected"),
    (["pre 0 0 1/22050 576", "pre 1 576 1/22050 576", "post 0 -1105 1/22050"], "cannot hold"),
])
def test_an_inconsistent_encoder_account_is_a_render_error(lines, why):
    with pytest.raises(FileFailure) as caught:
        account(lines)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "render") and why in caught.value.detail


def gapped_stats(frames: int, last: int, gap_at: int, gap: int, rate: int = 44100,
                 packet_shift=lambda k, pts, size: pts) -> list[str]:
    """A real-shaped account whose frame timestamps jump forward by ``gap`` at frame ``gap_at`` (F-5).

    Packets follow the frames' own timeline, as FFmpeg's audio frame queue
    stamps them; ``packet_shift`` alters one for the adversarial cases.
    """
    size = 1152 if rate >= 32000 else 576
    counts = [size] * (frames - 1) + [last]
    lines, starts, stamps, position = [], [], [], 0
    for n, count in enumerate(counts):
        pts = position + (gap if n >= gap_at else 0)
        lines.append(f"pre {n} {pts} 1/{rate} {count}")
        starts.append(position)
        stamps.append(pts)
        position += count

    def timeline(q):
        if q < 0:
            return q
        if q >= position:
            return stamps[-1] + counts[-1] + q - position
        i = max(n for n, start in enumerate(starts) if start <= q)
        return stamps[i] + q - starts[i]
    packets = -(-(position + 1105) // size)
    return lines + [f"post {k} {packet_shift(k, timeline(k * size - 1105), size)} 1/{rate}" for k in range(packets)]


def test_a_forward_jump_in_frame_timestamps_adds_no_samples():
    """IC-3 recheck 3 F-5, CR-074: the dynamic graph's 4,221-sample jump (32 to 44.1 kHz) is not audio."""
    count = Mp3EncoderCount()
    for line in gapped_stats(40, 693, gap_at=30, gap=4221):
        count.feed(line)
    assert count.totals(44100) == (39 * 1152 + 693, 1105, 41) and count.timestamp_gaps == 1
    # the packet that crosses the jump steps by a frame plus the jump, and only that one
    steps = [b - a for a, b in zip(count._packet_pts, count._packet_pts[1:])]
    assert sorted(set(steps[:-1])) == [1152, 1152 + 4221] and steps.count(1152 + 4221) == 1
    count = Mp3EncoderCount()
    for line in gapped_stats(40, 693, gap_at=30, gap=4221):
        count.feed(line)
    assert count.totals(44100, (39 * 1152 + 693, 44100))[0] == 39 * 1152 + 693


@pytest.mark.parametrize("rate", [22050, 44100])
@pytest.mark.parametrize("name,shift", [
    ("packets ignore the jump", lambda k, pts, size: pts - 4221 if pts > 30 * size else pts),
    ("one packet past the jump moved by a sample", lambda k, pts, size: pts + 1 if k == 33 else pts),
    ("one packet before the jump moved by the jump", lambda k, pts, size: pts + 4221 if k == 20 else pts),
    ("the first packet moved with the jump", lambda k, pts, size: pts + 4221 if k == 0 else pts),
])
def test_a_jump_the_packets_do_not_follow_is_refused(rate, name, shift):
    lines = gapped_stats(40, 693, gap_at=30, gap=4221, rate=rate, packet_shift=shift)
    with pytest.raises(FileFailure) as caught:
        account(lines, rate)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "render"), name
    assert "input timeline" in caught.value.detail or "not the expected" in caught.value.detail or (
        "advances by" in caught.value.detail), name


def test_a_frame_that_starts_before_the_previous_one_ended_is_refused():
    lines = gapped_stats(40, 693, gap_at=30, gap=4221)
    lines[31] = lines[31].replace(lines[31].split()[2], str(int(lines[31].split()[2]) - 4300))
    with pytest.raises(FileFailure, match="before the previous frame ended"):
        account(lines, 44100)


def source_account(samples, rate, source):
    lines = [f"pre {n} {n * 1152} 1/{rate} 1152" for n in range(samples // 1152)] + (
        [f"pre {samples // 1152} {samples // 1152 * 1152} 1/{rate} {samples % 1152}"] if samples % 1152 else [])
    lines += [f"post {n} {n * 1152 - 1105} 1/{rate}" for n in range(samples // 1152 + 2)]
    count = Mp3EncoderCount()
    for line in lines:
        count.feed(line)
    return count.totals(rate, source)


@pytest.mark.parametrize("samples,rate,source", [
    (4617, 32000, (4617, 32000)),        # same rate, exact
    (6362, 44100, (4617, 32000)),        # 4617 x 44100 / 32000 = 6362.8
    (6363, 44100, (4617, 32000)),
])
def test_the_encoder_input_count_may_equal_what_was_fed_it(samples, rate, source):
    assert source_account(samples, rate, source)[0] == samples


@pytest.mark.parametrize("samples,rate,source,why", [
    (4616, 32000, (4617, 32000), "received 4616 samples, the source has 4617"),       # final partial record 9 -> 8
    (4608, 32000, (4617, 32000), "received 4608 samples, the source has 4617"),       # final pre record dropped
    (4618, 32000, (4617, 32000), "received 4618 samples"),
    (6360, 44100, (4617, 32000), "expect 6362.8"),                                    # outside the resampled rule
    (6366, 44100, (4617, 32000), "expect 6362.8"),
    (6361, 44100, (4617, 32000), "expect 6362.8"),
    (6364, 44100, (4617, 32000), "expect 6362.8"),
])
def test_an_encoder_input_count_that_is_not_what_was_fed_it_is_refused(samples, rate, source, why):
    with pytest.raises(FileFailure) as caught:
        source_account(samples, rate, source)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "render") and why in caught.value.detail


def test_the_resampled_tolerance_is_one_sample():
    """3 s of 32 kHz audio scales to 132300 at 44100 Hz: one sample either side, not the verify rule's three."""
    source = (96000, 32000)
    for samples in (132299, 132300, 132301):
        assert source_account(samples, 44100, source)[0] == samples
    for samples in (132298, 132302):
        with pytest.raises(FileFailure, match="expect 132300.0"):
            source_account(samples, 44100, source)


def test_crc16_arc_check_value():
    assert _crc16_arc(b"123456789") == 0xBB3D  # the catalogued CRC-16/ARC check value


# --- Real tools ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def executable():
    return ffmpeg.locate_ffmpeg()


@pytest.fixture(scope="module")
def prober():
    return ffprobe.locate_ffprobe()


@pytest.fixture(scope="module")
def env(executable):
    environment = ffmpeg.probe_environment(executable)
    if "libmp3lame" not in environment.encoders:
        pytest.skip("build lacks libmp3lame")
    return environment


@pytest.fixture
def v0_records(env, tmp_path):
    """V0 evidence per rate and channel count, collected under this test's own tmp_path (hermetic guard)."""
    folder = tmp_path / "mp3-evidence"
    cache = {}

    def record(rate, channels):
        key = (rate, channels)
        if key not in cache:
            cache[key] = ffmpeg.collect_mp3_v0_evidence(env, rate, channels, "stereo" if channels == 2 else "mono",
                                                        folder)
            if 0 not in cache[key].vbr_qualities:
                pytest.skip(f"MP3 V0 is not evidenced at {rate} Hz, {channels} ch on this build")
        return cache[key]
    return record


def noise_source(folder: Path, rate: int, residue: int, channels: int = 2) -> Path:
    samples = .03 * np.random.default_rng(1020).normal(size=(5 * rate + residue, channels))
    path = folder / f"noise-{rate}-{residue}-{channels}.wav"
    sf.write(path, samples if channels > 1 else samples[:, 0], rate, subtype="PCM_24")
    return path


def identity(path: Path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


def decode(path: Path, executable, channels: int) -> np.ndarray:
    raw = subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                          "-f", "f64le", "-c:a", "pcm_f64le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f8").reshape(-1, channels)


def lag(reference: np.ndarray, output: np.ndarray, start=1000, length=8000, window=600) -> int:
    """IC-3's integer cross-correlation: the best shift of ``output`` against ``reference``."""
    ref = reference[start:start + length]
    scores = {k: float(np.dot(ref, output[start + k:start + k + length])) for k in range(-window, window + 1)}
    return max(scores, key=scores.get)


def render_mp3(tmp_path, env, prober, source, target, record, plan=None, name="candidate.mp3"):
    probe = ffprobe.probe(prober, source)
    candidate = tmp_path / name
    candidate.write_bytes(b"")
    before = identity(candidate)
    result = render(source, identity(source), probe, plan or linear_plan(), target, candidate, env, (record,))
    assert identity(candidate) == before  # rewritten in place, same inode
    return probe, result


def lame_fields(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    frame = ffmpeg.read_mp3_frames(data)[0]
    side = (17 if frame.channel_mode == 3 else 32) if frame.version == "1" else (9 if frame.channel_mode == 3 else 17)
    value = int.from_bytes(data[frame.offset + 4 + side + 141:frame.offset + 4 + side + 144], "big")
    return value >> 12, value & 0xFFF


def crc_verified(path: Path, prober) -> bool:
    """FFmpeg's reader exports the tag's encoder string only when the LAME tag CRC matches."""
    out = subprocess.run([str(prober.path), "-v", "error", "-show_entries", "stream_tags=encoder", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True).stdout
    return out.strip().startswith("Lav")


@pytest.mark.parametrize("residue", RESIDUES)
@pytest.mark.parametrize("rate", RATES)
def test_mp3_v0_keeps_every_sample_with_zero_lag(tmp_path, env, executable, prober, v0_records, rate, residue):
    """CR-063 acceptance: decoded length exact and cross-correlation lag 0 at every rate and residue."""
    record = v0_records(rate, 2)
    source = noise_source(tmp_path, rate, residue)
    probe, result = render_mp3(tmp_path, env, prober, source, mp3_target(rate, basis=record.evidence_id), record)
    frames = 5 * rate + residue
    out = measure(result.candidate_path, ffprobe.probe(prober, result.candidate_path), executable)
    assert out.decoded_frames == frames and out.sample_rate_hz == rate
    check_candidate_length(frames, probe.sample_rate_hz, out, trailing_padding=trailing_padding_allowance(
        result.target))
    decoded = decode(result.candidate_path, executable, 2)
    reference = sf.read(source)[0]
    assert lag(reference[:, 0], decoded[:, 0]) == 0 and lag(reference[:, 1], decoded[:, 1]) == 0
    delay, padding = lame_fields(result.candidate_path)
    assert delay == 576 and f"gapless delay {delay}, padding {padding}" in result.encoder_setting_evidence
    assert "Xing VBR tag" in result.encoder_setting_evidence
    assert crc_verified(result.candidate_path, prober)


@pytest.mark.parametrize("rate,residue", [(22050, 137), (32000, 137)])
def test_mp3_mono_and_cbr_are_exact_at_the_failing_rates(tmp_path, env, executable, prober, v0_records,
                                                         rate, residue):
    mono = v0_records(rate, 1)
    source = noise_source(tmp_path, rate, residue, channels=1)
    _, result = render_mp3(tmp_path, env, prober, source,
                           mp3_target(rate, 1, mono.channel_layout, basis=mono.evidence_id), mono)
    assert decode(result.candidate_path, executable, 1).shape[0] == 5 * rate + residue
    assert lag(sf.read(source)[0], decode(result.candidate_path, executable, 1)[:, 0]) == 0
    cbr, _ = ffmpeg.collect_mp3_evidence(env, rate, 2, "stereo", tmp_path / "cbr-ev", cbr_candidates_bps=(64000,))
    if 64000 not in cbr.nominal_bitrates_bps:
        pytest.skip(f"MP3 CBR 64k is not evidenced at {rate} Hz on this build")
    stereo = noise_source(tmp_path, rate, residue)
    _, result = render_mp3(tmp_path, env, prober, stereo,
                           mp3_target(rate, mode="cbr", bps=64000, basis=cbr.evidence_id), cbr, name="cbr.mp3")
    decoded = decode(result.candidate_path, executable, 2)
    assert decoded.shape[0] == 5 * rate + residue and lag(sf.read(stereo)[0][:, 0], decoded[:, 0]) == 0
    assert "Info tag" in result.encoder_setting_evidence and crc_verified(result.candidate_path, prober)


def test_a_resampled_mp3_decodes_to_exactly_what_the_encoder_received(tmp_path, env, executable, prober,
                                                                      v0_records):
    """48 kHz → 22.05 kHz: the MP3 holds the same frames as a lossless render of the same chain."""
    record = v0_records(22050, 2)
    source = noise_source(tmp_path, 48000, 137)
    _, result = render_mp3(tmp_path, env, prober, source, mp3_target(22050, basis=record.evidence_id), record)
    lossless = ffmpeg.collect_lossless_evidence(env, "pcm_s24le", 22050, 2, "stereo", tmp_path / "wav-ev")
    wav = tmp_path / "reference.wav"
    wav.write_bytes(b"")
    render(source, identity(source), ffprobe.probe(prober, source), linear_plan(),
           wav_target(22050, 2, "stereo", lossless.evidence_id), wav, env, (lossless,))
    reference = decode(wav, executable, 2)
    decoded = decode(result.candidate_path, executable, 2)
    assert decoded.shape == reference.shape and lag(reference[:, 0], decoded[:, 0]) == 0
    out = measure(result.candidate_path, ffprobe.probe(prober, result.candidate_path), executable)
    check_candidate_length(48000 * 5 + 137, 48000, out)


def plain_encode(executable, source: Path, target: Path) -> list[str]:
    """FFmpeg's own MP3 V0 encode with Norma's flags and encoder account, without the CR-063 rewrite."""
    out = subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-y", "-i", str(source),
                          "-map", "0:a:0", "-c:a", "libmp3lame", "-q:a", "0", "-map_metadata", "-1",
                          "-fflags", "+bitexact", "-flags:a", "+bitexact", *MP3_ENCODER_STATS, "-f", "mp3",
                          str(target)], capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def test_unrewritten_mpeg2_encodes_carry_the_ffmpeg_fields_and_the_rewrite_fixes_only_them(tmp_path, executable,
                                                                                         prober, env):
    """The counter-example itself: FFmpeg stores delay 47 at 22.05 kHz; the length rule refuses it."""
    source = noise_source(tmp_path, 22050, 137)
    mp3 = tmp_path / "plain.mp3"
    count = Mp3EncoderCount()
    for line in plain_encode(executable, source, mp3):
        count.feed(line)
    assert lame_fields(mp3) == (47, 781) and not crc_verified(mp3, prober)
    plain = measure(mp3, ffprobe.probe(prober, mp3), executable)
    assert plain.decoded_frames == 110387 + 529
    with pytest.raises(FileFailure) as caught:
        check_candidate_length(110387, 22050, plain)
    assert (caught.value.reason, caught.value.stage) == ("render_error", "verify")
    before = mp3.read_bytes()
    assert restore_mp3_gapless(mp3, *count.totals(22050)) == (576, 781)
    after = mp3.read_bytes()
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    frame = ffmpeg.read_mp3_frames(after)[0]
    lame = frame.offset + 4 + 17 + 120
    assert len(before) == len(after) and set(changed) <= {lame + 21, lame + 22, lame + 34, lame + 35}
    assert measure(mp3, ffprobe.probe(prober, mp3), executable).decoded_frames == 110387
    assert crc_verified(mp3, prober)


def test_the_rewrite_is_byte_identical_where_ffmpeg_was_already_right(tmp_path, executable):
    """At 44.1 kHz FFmpeg's fields and CRC are correct: recomputing them changes no byte."""
    source = noise_source(tmp_path, 44100, 137)
    mp3 = tmp_path / "plain.mp3"
    count = Mp3EncoderCount()
    for line in plain_encode(executable, source, mp3):
        count.feed(line)
    before = mp3.read_bytes()
    assert restore_mp3_gapless(mp3, *count.totals(44100)) == lame_fields(mp3) == (576, 1123)
    assert mp3.read_bytes() == before


def test_a_file_that_disagrees_with_the_encoder_account_is_refused(tmp_path, executable):
    source = noise_source(tmp_path, 32000, 137)
    mp3 = tmp_path / "plain.mp3"
    count = Mp3EncoderCount()
    for line in plain_encode(executable, source, mp3):
        count.feed(line)
    samples, priming, packets = count.totals(32000)
    before = mp3.read_bytes()
    cases = {
        "frames written": (samples, priming, packets + 1),
        "out of range": (samples + 4096, priming, packets),
        "decoder delay": (samples, MP3_DECODER_DELAY + 4096, packets),
    }
    for name, args in cases.items():
        with pytest.raises(FileFailure) as caught:
            restore_mp3_gapless(mp3, *args)
        assert (caught.value.reason, caught.value.stage) == ("render_error", "render"), name
    assert mp3.read_bytes() == before  # a refused rewrite writes nothing
    broken = tmp_path / "no-lame.mp3"
    data = bytearray(before)
    frame = ffmpeg.read_mp3_frames(before)[0]
    data[frame.offset + 4 + 32 + 120:frame.offset + 4 + 32 + 124] = b"XXXX"
    broken.write_bytes(bytes(data))
    with pytest.raises(FileFailure, match="no LAME extension"):
        restore_mp3_gapless(broken, samples, priming, packets)


@pytest.fixture
def path_based_evidence_cleanup(monkeypatch):
    # CR-041: the shared guard cannot assess fd-relative TemporaryDirectory cleanup.
    monkeypatch.setattr(shutil, "_use_fd_functions", False)


def test_a_shifted_first_post_timestamp_publishes_nothing_through_the_pipeline(tmp_path, env, executable, v0_records,
                                                                               monkeypatch, capsys,
                                                                               path_based_evidence_cleanup):
    """IC-3 recheck F-3 end to end: the real encoder, one injected shift, no tag rewrite and no output."""
    if v0_records(32000, 2) is None:
        pytest.skip("no MP3 evidence")
    (tmp_path / "src").mkdir()
    source = noise_source(tmp_path / "src", 32000, 137)
    original = source.read_bytes()
    real_feed = Mp3EncoderCount.feed
    seen = []

    def shifted(self, line):
        if line.startswith("post 0 "):
            fields = line.split()
            fields[2] = str(int(fields[2]) - 100)
            seen.append(line)
            line = " ".join(fields)
        return real_feed(self, line)

    out, report = tmp_path / "out", tmp_path / "report.json"
    with patch.object(Mp3EncoderCount, "feed", shifted):
        code = cli.main(["normalise", str(source), "-f", "mp3", "--config", "/dev/null", "-o", str(out),
                         "--report", str(report)])
    capsys.readouterr()
    row = json.loads(report.read_text())["outcomes"][0]
    assert seen and code == 1
    assert (row["status"], row["reason"]) == ("failed", "render_error")
    assert row["published_paths"] == [] and not (out / "noise-32000-137-2.mp3").exists()
    assert not out.exists() or not any(p.suffix == ".mp3" or p.name.startswith(".norma-") for p in out.rglob("*"))
    assert source.read_bytes() == original


@pytest.mark.parametrize("name,rate,sample_rate,change", [
    ("terminal count 657 to 653 on a 32 to 44.1 kHz render", 32000, 44100, -4),
    ("final pre record dropped", 32000, None, None),
    ("final pre count 9 to 8", 32000, None, -1),
])
def test_a_false_terminal_account_is_refused_before_the_tag_is_written(
        tmp_path, env, v0_records, monkeypatch, capsys, path_based_evidence_cleanup, name, rate, sample_rate, change):
    """IC-3 recheck 2 F-4, CR-071: real encodes, one injected terminal record, no tag rewrite and no output."""
    if v0_records(sample_rate or rate, 2) is None:
        pytest.skip("no MP3 evidence")
    (tmp_path / "src").mkdir()
    source = noise_source(tmp_path / "src", rate, 137)
    original = source.read_bytes()
    real_feed = Mp3EncoderCount.feed
    seen = []

    def corrupt(self, line):
        fields = line.split()
        if fields and fields[0] == "pre" and 0 < int(fields[4]) < 1152:
            seen.append(line)
            if change is None:
                return None
            fields[4] = str(int(fields[4]) + change)
            line = " ".join(fields)
        return real_feed(self, line)

    rewrites = []
    monkeypatch.setattr(render_module, "restore_mp3_gapless", lambda candidate, *args: rewrites.append(args))
    out, report = tmp_path / "out", tmp_path / "report.json"
    argv = ["normalise", str(source), "-f", "mp3", "--config", "/dev/null", "-o", str(out), "--report", str(report)]
    with patch.object(Mp3EncoderCount, "feed", corrupt):
        code = cli.main(argv + (["--sample-rate", str(sample_rate)] if sample_rate else []))
    capsys.readouterr()
    row = json.loads(report.read_text())["outcomes"][0]
    assert len(seen) == 1 and code == 1 and rewrites == []  # refused before the tag rewrite: it never ran
    assert (row["status"], row["reason"]) == ("failed", "render_error")
    assert row["published_paths"] == []
    assert not out.exists() or not any(p.suffix == ".mp3" or p.name.startswith(".norma-") for p in out.rglob("*"))
    assert source.read_bytes() == original


def dynamic_source(folder: Path, rate: int, frames: int, band_hz: int | None = None) -> Path:
    """IC-3 recheck 3's F-5 source: seed-1023 Gaussian noise, amplitude 0.03, mono PCM24.

    ``band_hz`` removes the noise above it first. White noise at 192 kHz is
    three-quarters ultrasonic: the 48 kHz output then measures about 6 LU
    under loudnorm's calibration, and dynamic mode, which gets no loudness
    correction, fails ``verification_loudness`` after its account passes.
    """
    noise = np.random.default_rng(1023).normal(size=frames)
    if band_hz is not None:
        spectrum = np.fft.rfft(noise)
        spectrum[np.fft.rfftfreq(frames, 1 / rate) > band_hz] = 0
        noise = np.fft.irfft(spectrum, frames)
        noise /= noise.std()
    path = folder / f"dynamic-{rate}.wav"
    sf.write(path, .03 * noise, rate, subtype="PCM_24")
    return path


def normalise_dynamic(tmp_path, source: Path, sample_rate: int | None, feed=None):
    """One real ``normalise --peak-mode dynamic`` to MP3; returns (exit, report row, render argv, account)."""
    real_run, real_feed, real_totals = ffmpeg.run_process, Mp3EncoderCount.feed, Mp3EncoderCount.totals
    seen: dict = {}

    def run(argv, **kwargs):
        if "-stats_enc_pre" in argv:
            seen["argv"] = [str(arg) for arg in argv]
        return real_run(argv, **kwargs)

    def totals(self, rate, source=None):
        seen["account"] = self
        try:
            return real_totals(self, rate, source)
        except FileFailure as exc:
            self.refusal = exc.detail
            raise

    out, report = tmp_path / "out", tmp_path / "report.json"
    argv = ["normalise", str(source), "-f", "mp3", "--peak-mode", "dynamic", "--config", "/dev/null",
            "-o", str(out), "--report", str(report)] + (["--sample-rate", str(sample_rate)] if sample_rate else [])
    with patch.object(ffmpeg, "run_process", run), patch.object(Mp3EncoderCount, "totals", totals), \
            patch.object(Mp3EncoderCount, "feed", feed or real_feed):
        code = cli.main(argv)
    return code, json.loads(report.read_text())["outcomes"][0], seen.get("argv"), seen.get("account")


@pytest.mark.parametrize("rate,frames,sample_rate,out_rate,band_hz", [
    (32000, 960137, 44100, 44100, None),     # IC-3 recheck 3 F-5 as reproduced: pre 1038 jumps 4,221 samples
    (192000, 5760137, None, 48000, 20000),   # automatic 48 kHz (F-5: pre 1130 jumps), audible band only
])
def test_a_dynamic_mp3_publishes_every_sample_with_zero_lag(tmp_path, env, executable, v0_records, capsys,
                                                             path_based_evidence_cleanup, rate, frames,
                                                             sample_rate, out_rate, band_hz):
    """CR-074: real 30-second dynamic renders publish, exact length and lag 0 against the graph's own PCM."""
    v0_records(out_rate, 1)
    (tmp_path / "src").mkdir()
    source = dynamic_source(tmp_path / "src", rate, frames, band_hz)
    original = source.read_bytes()
    code, row, argv, count = normalise_dynamic(tmp_path, source, sample_rate)
    capsys.readouterr()
    assert code == 0 and row["status"] in ("ok", "warn") and len(row["published_paths"]) == 1, row
    assert count.timestamp_gaps >= 1  # the natural jump F-5 found is still there and now accepted
    graph = argv[argv.index("-af") + 1]
    raw = subprocess.run([str(executable.path), "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
                          "-af", graph, "-f", "f64le", "-c:a", "pcm_f64le", "-"],
                         capture_output=True, check=True).stdout
    reference = np.frombuffer(raw, dtype="<f8")
    decoded = decode(Path(row["published_paths"][0]), executable, 1)[:, 0]
    assert decoded.shape[0] == reference.shape[0] == count.samples
    assert abs(count.samples * rate - frames * out_rate) <= rate  # D-086, unchanged
    tail = reference.shape[0] - 9000
    assert lag(reference, decoded) == 0 and lag(reference, decoded, start=tail) == 0
    assert source.read_bytes() == original


def test_a_dynamic_jump_the_packets_do_not_follow_publishes_nothing(tmp_path, env, v0_records, monkeypatch,
                                                                     capsys, path_based_evidence_cleanup):
    """CR-074 adversarial: the real jump kept in the frames, removed from the packets — refused before the tag."""
    v0_records(44100, 1)
    (tmp_path / "src").mkdir()
    source = dynamic_source(tmp_path / "src", 32000, 960137)
    original = source.read_bytes()
    real_feed = Mp3EncoderCount.feed

    def hide_jump(self, line):
        fields = line.split()
        if fields and fields[0] == "post" and self.timestamp_gaps and int(fields[1]) == self.packets:
            if self._packet_pts and int(fields[2]) - self._packet_pts[-1] > 1152:
                fields[2] = str(self._packet_pts[-1] + 1152)
                self.hidden = True
                line = " ".join(fields)
        return real_feed(self, line)

    rewrites = []
    monkeypatch.setattr(render_module, "restore_mp3_gapless", lambda candidate, *args: rewrites.append(args))
    code, row, _, count = normalise_dynamic(tmp_path, source, 44100, feed=hide_jump)
    capsys.readouterr()
    assert getattr(count, "hidden", False) and code == 1 and rewrites == []
    assert (row["status"], row["reason"], row["published_paths"]) == ("failed", "render_error", [])
    assert count.error is None and "on the encoder's input timeline" in count.refusal
    out = tmp_path / "out"
    assert not out.exists() or not any(p.suffix == ".mp3" or p.name.startswith(".norma-") for p in out.rglob("*"))
    assert source.read_bytes() == original


def test_the_review_192_khz_noise_passes_its_account_and_fails_only_loudness(tmp_path, env, v0_records, capsys,
                                                                             path_based_evidence_cleanup):
    """F-5's exact 192 kHz fixture: the account (jump included) is accepted; dynamic then misses loudness, safely."""
    v0_records(48000, 1)
    (tmp_path / "src").mkdir()
    source = dynamic_source(tmp_path / "src", 192000, 5760137)
    code, row, _, count = normalise_dynamic(tmp_path, source, None)
    capsys.readouterr()
    assert count.timestamp_gaps >= 1 and not hasattr(count, "refusal")
    assert count.totals(48000, (5760137, 192000))[0] == 1440034
    assert code == 1 and (row["status"], row["reason"], row["published_paths"]) == (
        "failed", "verification_loudness", [])
