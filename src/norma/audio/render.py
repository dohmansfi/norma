"""Render: FFmpeg arguments and one render into a caller-allocated candidate (C-05).

Rules (formats F4, F6, F9–F15):

- Every render reads the original source, maps only ``0:a:0`` and carries the
  R8 metadata flags among the output options; argv elements are separate
  values, never a shell string (R1, R8).
- Signal path: decode → [resample to the C-04 rate with SWR's R2 settings] →
  [``volume=<g>dB:precision=float``] → [oversampled limiter] → [triangular
  dither, 16-bit only] → encode. A step appears only when it changes
  something: an unchanged copy gets no gain, resampler or dither (R1–R3, F6).
- ``linear`` and ``reduce`` are plain float gain without a limiter (R3).
  ``limit`` runs R4's chain at the target rate r: up to ``4r`` with SWR,
  ``alimiter`` at ``10**(L/20)`` with explicit levels, ``asc=0``, ``level=0``
  and ``latency=1`` (delay compensation: no timing shift, same frame count),
  then back to r. A threshold outside alimiter's range fails before any
  process; it is never clamped.
- Explicit ``dynamic`` (R5, R6) runs R5's calibrated ``loudnorm`` from the
  plan's C-05 ``dynamic_parameters`` at its dynamic ceiling (C-05
  ``limiter_threshold_dbtp``), ``linear=false``, then SWR back to the C-04
  rate r: loudnorm's internal 192 kHz is not an output rate. Its input is
  the decoded source unchanged, exactly what the calibration analysed. A
  value outside loudnorm's range fails before any process; never clamped.
  ``normalization_type`` comes from the render's own loudnorm JSON.
- Encoders come from the finalised C-04 record: PCM WAV/AIFF, FLAC and
  native ``alac`` (never ``alac_at``); MP3 ``libmp3lame`` at the record's VBR
  quality (``-q:a``) or CBR nominal (``-b:a``); AAC ``aac_at`` CBR at the
  record's effective nominal (R7, ADR-0015). A ``stream_copy`` record copies
  the encoded stream bit-exactly (F13). WS-01 chooses no codec, setting,
  depth or rate.
- Before encoding, the C-04 ``capability_basis`` must name exactly one record
  of this run's evidence that still matches the invoked executable; otherwise
  the file fails ``unsupported_target`` with no process started (R7).
- After encoding, the written header must show the C-04 depth, float kind,
  rate and channels, or the file fails ``encoder_settings_mismatch`` (R7).
  MP3: every written frame header carries the C-04 rate and channel mode (and
  the CBR nominal), and the first frame carries the Xing/Info tag whose
  gapless delay and padding let decoding restore the exact length (CR-059).
- MP3 gapless fields (CR-063): FFmpeg's muxer derives the LAME tag's delay
  and padding from per-packet skip side data, which misstates them when the
  encoder's priming spans more than one packet (MPEG-2 rates, 576-sample
  frames: delay 47 instead of 576) and at some frame residues (32 kHz: the
  padding 38 short). The render therefore asks the encoder itself, through
  FFmpeg's encoder statistics on stdout, how many samples it received and
  its priming, and rewrites the two fields (and the tag CRC) in place from
  that count: ``delay = priming - 529`` and
  ``padding = packets x frame size - delay - samples``. A count or tag it
  cannot read fails ``render_error``; no sample is added or dropped.
  AAC (``encoder_diagnostic``): any ``aac_at`` diagnostic other than its
  1024-sample flush-queue note fails; a stream average is never evidence.
- Each render has its own adapter deadline (T3, ADR-0018); a timeout is
  ``timeout`` and an executable lost after preflight is ``render_error``.

Stage S3 renders every action (:data:`RENDER_ACTIONS`) to lossless and lossy
targets or stream copies. :data:`S2_ACTIONS` and :class:`RenderNotAvailable`
are S2 leftovers that the pipeline no longer uses since its S3 wiring
(TK-0419); they are kept for their existing tests.
"""

from __future__ import annotations

import bisect
import errno
import os
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from norma.adapters import ffmpeg
from norma.audio.measure import LoudnormJson, count_decoded_frames
from norma.audio.plan import (
    DYNAMIC_LRA,
    LOUDNORM_MEASURED_OPTIONS,
    LOUDNORM_TARGET_RANGE,
    LOUDNORM_TP_RANGE,
    in_range,
    limiter_threshold_supported,
)
from norma.errors import FileFailure, ToolError
from norma.models import (
    LOSSLESS_FORMATS,
    CapabilityEvidence,
    GainPlan,
    Identity,
    ProbeResult,
    RenderResult,
    TargetFormat,
)

STAGE = "render"
S2_ACTIONS: frozenset[str] = frozenset({"copy", "linear"})
"""The actions the S2 pipeline admits; unchanged until the pipeline's S3 wiring."""
RENDER_ACTIONS: frozenset[str] = frozenset({"copy", "linear", "limit", "reduce", "dynamic"})
"""Actions :func:`render` executes."""

SWR_OPTIONS = "resampler=swr:filter_size=64:phase_shift=10:exact_rational=1:cutoff=0.97"
"""R2's provisional resampler settings for every explicit rate conversion."""

LIMITER_OVERSAMPLING = 4
"""R4: the limiter runs at four times the effective target rate."""
ALIMITER_SETTINGS = "attack=5:release=50:level_in=1:level_out=1:asc=0:level=0:latency=1"
"""R4's fixed limiter options after ``limit``: no auto level, no adaptive release, delay compensated."""

METADATA_FLAGS: tuple[str, ...] = (
    "-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
)
"""R8 / MD30: no inherited, muxer or chapter metadata reaches a candidate."""

MUXERS: dict[str, str] = {"wav": "wav", "aiff": "aiff", "flac": "flac", "alac": "ipod", "mp3": "mp3", "aac": "ipod"}
"""Muxer per C-04 format; ``ipod`` is the .m4a muxer the capability evidence used."""

_PACKED_LOSSLESS: dict[tuple[str, int], tuple[str, ...]] = {
    # Same representations as the installed-query evidence (adapters/ffmpeg.py).
    ("flac", 16): ("-sample_fmt", "s16"),
    ("flac", 24): ("-sample_fmt", "s32", "-bits_per_raw_sample", "24"),
    ("alac", 16): ("-sample_fmt", "s16p"),
    ("alac", 24): ("-sample_fmt", "s32p", "-bits_per_raw_sample", "24"),
}
_PCM_SUFFIX = {"wav": "le", "aiff": "be"}
LOSSY_ENCODERS: dict[str, str] = {"mp3": "libmp3lame", "aac": "aac_at"}
"""R7: the only encoder per lossy format (ADR-0015, D-029)."""
_AAC_DIAGNOSTIC = re.compile(r"^\[aac_at @ 0x[0-9a-f]+\]\s*(.*)$")
_AAC_FLUSH_NOTE = re.compile(r"Trying to remove 1024 samples, but the queue is empty")
"""aac_at's end-of-stream queue bookkeeping, not a setting change (TK-0418 evidence)."""
FLAC_LEVELS = range(0, 9)
"""``--flac-level`` range; applied only to FLAC outputs."""
MP3_ENCODER_STATS: tuple[str, ...] = (
    "-stats_enc_pre", "pipe:1", "-stats_enc_pre_fmt", "pre {n} {pts} {tb} {samp}",
    "-stats_enc_post", "pipe:1", "-stats_enc_post_fmt", "post {n} {pts} {tb}",
)
"""CR-063: FFmpeg's encoder statistics on stdout, one line per frame sent and packet received.

Both share one output (FFmpeg opens a stats path once) and every line is
flushed whole from the encoder thread, so lines never interleave.
"""
MP3_DECODER_DELAY = 528 + 1
MP3_ENCODER_DELAY = 576
"""libmp3lame's encoder delay, constant for every frame size; with the decoder delay it is the packets' priming (1,105)."""
_TIME_BASE = re.compile(r"1/[1-9][0-9]*")
"""The LAME-tag convention's decoder delay: the tag's delay excludes it, its padding includes it."""
_LAME_TAG_VERSIONS = (b"LAME", b"Lavf", b"Lavc")
"""Encoder strings whose LAME-tag delay and padding decoders honour."""
_MP3_FIELD_MAX = (1 << 12) - 1


class RenderNotAvailable(FileFailure):
    """A render path not yet admitted by the pipeline's S2 gate; mapped as ``failed/internal``.

    :func:`render` itself no longer raises it (TK-0111); the pipeline raises
    it before any process until its S3 wiring (TK-0419), so no candidate is written.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("internal", STAGE, detail)


# --- Pure argument building ---------------------------------------------------------------


def preserves_samples(probe: ProbeResult, target: TargetFormat) -> bool:
    """True when a zero-gain render keeps every decoded sample value exactly.

    A stream copy does; so does a lossless-to-lossless conversion at the same
    rate and channel count into an equal or wider representation (an integer
    source of at most 24 bits fits 32-bit float exactly). A lossy source, a
    lossy target or any narrowing changes samples (formats F6, audio R3).
    """
    if target.stream_copy:
        return True
    if probe.codec_class != "lossless" or target.format not in LOSSLESS_FORMATS:
        return False
    if target.sample_rate_hz != probe.sample_rate_hz or target.channels != probe.channels:
        return False
    if probe.bit_depth is None:
        return False
    if probe.sample_float:
        return target.sample_float and probe.bit_depth <= 32
    if target.sample_float:
        return probe.bit_depth <= 24
    return target.bit_depth is not None and target.bit_depth >= probe.bit_depth


def _gain(plan: GainPlan) -> float:
    return 0.0 if plan.gain_db is None else plan.gain_db


def _number(value: float) -> str:
    """A float FFmpeg's expression parser reads back exactly (round-trip repr)."""
    text = repr(float(value))
    return "0" if text in ("0.0", "-0.0") else text


def _check_renderable(plan: GainPlan, target: TargetFormat, probe: ProbeResult) -> None:
    """Refuse later S3 work clearly, and inconsistent records as programming errors."""
    if plan.action == "skip":
        raise ValueError("a skip plan has no render")
    if target.decision == "skipped_lossy":
        raise ValueError("a skipped_lossy target has no render")
    if plan.action not in RENDER_ACTIONS:
        raise ValueError(f"unknown render action {plan.action!r}")
    if target.stream_copy and plan.action != "copy":
        raise ValueError("a stream-copy target requires a copy plan (C-04)")
    if plan.action == "copy" and _gain(plan) != 0:
        raise ValueError("a copy plan applies no gain")
    if plan.action in ("linear", "limit", "reduce") and plan.gain_db is None:
        raise ValueError(f"a {plan.action} plan needs a gain")
    if plan.action == "dynamic":
        dynamic_filter(plan, target.sample_rate_hz, dual_mono=False)  # range checks, before any process
    if plan.action == "limit":
        threshold = plan.limiter_threshold_dbtp
        if threshold is None:
            raise ValueError("a limit plan needs a limiter threshold (C-05)")
        if not limiter_threshold_supported(threshold):
            raise FileFailure("render_error", STAGE,
                              f"limiter threshold {threshold:.4f} dBTP is outside alimiter's supported range "
                              f"(R4); it is not clamped")
    if target.channels != probe.channels:
        raise ValueError("no implicit channel remapping or downmix (C-04)")


def limiter_filters(threshold_dbtp: float, rate: int) -> tuple[str, str, str]:
    """R4's oversampled limiter at effective target rate ``rate``: up, limit, down (pure)."""
    if not limiter_threshold_supported(threshold_dbtp):
        raise ValueError(f"limiter threshold {threshold_dbtp!r} dBTP is outside alimiter's range")
    return (f"aresample={LIMITER_OVERSAMPLING * rate}:{SWR_OPTIONS}",
            f"alimiter=limit={_number(10 ** (threshold_dbtp / 20))}:{ALIMITER_SETTINGS}",
            f"aresample={rate}:{SWR_OPTIONS}")


def dynamic_filter(plan: GainPlan, rate: int, *, dual_mono: bool) -> tuple[str, str]:
    """R5's calibrated loudnorm render and SWR back to ``rate`` (pure).

    The ceiling is the plan's dynamic ``TP`` (``limiter_threshold_dbtp``, V6).
    Every value must lie in the installed filter's range; a value outside it
    fails ``render_error`` here, before any process, and is never clamped.
    """
    if plan.action != "dynamic" or plan.dynamic_parameters is None or plan.limiter_threshold_dbtp is None:
        raise ValueError("a dynamic render needs a dynamic plan with its calibration and ceiling (C-05)")
    checks = [("I", plan.target_lufs, LOUDNORM_TARGET_RANGE), ("TP", plan.limiter_threshold_dbtp, LOUDNORM_TP_RANGE)]
    checks += [(option, plan.dynamic_parameters[key], bounds)
               for key, (option, bounds) in LOUDNORM_MEASURED_OPTIONS.items()]
    for option, value, bounds in checks:
        if not in_range(value, bounds):
            raise FileFailure("render_error", STAGE, f"loudnorm {option}={value!r} is outside its supported range "
                                                     f"{bounds} (R5, V6); it is not clamped")
    measured = ":".join(f"{option}={_number(plan.dynamic_parameters[key])}"
                        for key, (option, _) in LOUDNORM_MEASURED_OPTIONS.items())
    return (f"loudnorm=I={_number(plan.target_lufs)}:TP={_number(plan.limiter_threshold_dbtp)}:LRA={DYNAMIC_LRA}:"
            f"{measured}:linear=false:dual_mono={str(dual_mono).lower()}:print_format=json",
            f"aresample={rate}:{SWR_OPTIONS}")


def needs_source_frames(plan: GainPlan, probe: ProbeResult, target: TargetFormat) -> bool:
    """True when the graph ends in a sample-exact trim to the source frame count (CR-077)."""
    return plan.action == "dynamic" and not target.stream_copy and target.sample_rate_hz == probe.sample_rate_hz


def filter_chain(plan: GainPlan, probe: ProbeResult, target: TargetFormat, *,
                 no_dual_mono: bool = False, source_frames: int | None = None) -> str | None:
    """The ``-af`` graph for one render, or None when no filter applies.

    Order per R1: resample, gain, limiter, then dither and quantise.
    Processing starts from 32-bit float so no integer conversion precedes it.
    Triangular dither applies only to changed samples quantised to 16 bits
    (R3, F6); a limited or dynamic render always changes samples.

    ``dynamic`` (R5) feeds loudnorm the decoded source unchanged — the input
    its calibration analysed — and restores the C-04 rate afterwards; M3's
    mono policy sets ``dual_mono`` as for measurement. At the source's own
    rate the round trip through loudnorm's 192 kHz can leave one frame more
    than the source (CR-077); with ``source_frames`` the graph then ends in
    ``atrim=end_sample`` at exactly that count. The extra frame is at the end,
    so no programme sample moves. A resampled render is not trimmed.
    """
    _check_renderable(plan, target, probe)
    if target.stream_copy:
        return None
    rate = target.sample_rate_hz
    gain = _gain(plan)
    limited = plan.action == "limit"
    parts: list[str] = []
    if plan.action == "dynamic":
        parts += dynamic_filter(plan, rate, dual_mono=probe.channels == 1 and not no_dual_mono)
        if source_frames is not None and needs_source_frames(plan, probe, target):
            if not (isinstance(source_frames, int) and source_frames >= 1):
                raise ValueError(f"source_frames must be a positive integer, got {source_frames!r}")
            parts.append(f"atrim=end_sample={source_frames}")
    elif rate != probe.sample_rate_hz:
        parts += ["aformat=sample_fmts=fltp|flt", f"aresample={rate}:{SWR_OPTIONS}"]
    elif limited:
        parts.append("aformat=sample_fmts=fltp|flt")
    if gain != 0 and plan.action != "dynamic":
        parts.append(f"volume={_number(gain)}dB:precision=float")
    if limited:
        assert plan.limiter_threshold_dbtp is not None
        parts += limiter_filters(plan.limiter_threshold_dbtp, rate)
    changed = limited or plan.action == "dynamic" or gain != 0 or not preserves_samples(probe, target)
    if target.dither == "triangular" and not target.sample_float and changed:
        if target.bit_depth != 16:
            raise ValueError(f"no triangular dither recipe for {target.bit_depth}-bit output (R3)")
        parts.append(f"aresample=osr={rate}:osf=s16:dither_method=triangular:output_sample_bits=16")
    return ",".join(parts) or None


def encoder_args(target: TargetFormat, *, flac_level: int = 8) -> tuple[str, ...]:
    """Encoder options for the finalised C-04 record (R7); copy for a stream copy."""
    if target.stream_copy:
        return ("-c:a", "copy")
    if target.encoder in ffmpeg.NEVER_ENCODERS:
        raise ValueError("alac_at is never a Norma encoder; use native alac")
    if target.format not in LOSSLESS_FORMATS:
        return lossy_encoder_args(target)
    if target.format in _PCM_SUFFIX:
        kind = "f" if target.sample_float else "s"
        expected = f"pcm_{kind}{target.bit_depth}{_PCM_SUFFIX[target.format]}"
        if target.encoder != expected:
            raise ValueError(f"{target.format} {target.bit_depth}-bit target names encoder "
                             f"{target.encoder!r}, expected {expected!r}")
        return ("-c:a", target.encoder)
    if target.encoder != target.format or target.sample_float or target.bit_depth is None:
        raise ValueError(f"unexpected {target.format} record: encoder {target.encoder!r}, "
                         f"{target.bit_depth}-bit, float={target.sample_float}")
    try:
        representation = _PACKED_LOSSLESS[(target.format, target.bit_depth)]
    except KeyError:
        raise ValueError(f"{target.format} offers no {target.bit_depth}-bit representation") from None
    args = ("-c:a", target.encoder, *representation)
    if target.format == "flac":
        if not (isinstance(flac_level, int) and flac_level in FLAC_LEVELS):
            raise ValueError(f"flac_level must be 0..8, got {flac_level!r}")
        args += ("-compression_level", str(flac_level))
    return args


def lossy_encoder_args(target: TargetFormat) -> tuple[str, ...]:
    """R7: MP3/AAC options from the finalised C-04 record only; WS-01 picks no setting.

    MP3 VBR applies ``-q:a <vbr_quality>`` (V0 by default, F15) and CBR
    ``-b:a <effective nominal>``; AAC applies ``aac_at`` CBR at the effective
    nominal (ADR-0015). The MP3 muxer's default Xing/LAME tag is kept; its
    delay and padding fields are rewritten after the encode from the
    encoder's own sample count (:func:`restore_mp3_gapless`, CR-063).
    """
    expected = LOSSY_ENCODERS.get(target.format)
    if expected is None or target.encoder != expected:
        raise ValueError(f"{target.format} target names encoder {target.encoder!r}, expected {expected!r}")
    if target.bit_depth is not None or target.sample_float or target.dither != "none":
        raise ValueError(f"a {target.format} record has no depth, float kind or dither (C-04)")
    if target.format == "mp3" and target.bitrate_mode == "vbr":
        assert target.vbr_quality is not None
        return ("-c:a", expected, "-q:a", str(target.vbr_quality))
    if target.bitrate_mode == "cbr" and target.effective_bitrate_bps is not None:
        args: tuple[str, ...] = ("-c:a", expected)
        if target.format == "aac":
            args += ("-aac_at_mode", "cbr")
        return args + ("-b:a", str(target.effective_bitrate_bps))
    raise ValueError(f"no evidenced {target.format} setting in the C-04 record "
                     f"(mode {target.bitrate_mode!r}, quality {target.vbr_quality!r})")


def render_argv(
    executable: ffmpeg.Executable | Path, source: Path, candidate: Path, plan: GainPlan, probe: ProbeResult,
    target: TargetFormat, *, flac_level: int = 8, wav_info: Sequence[tuple[str, str]] = (),
    no_dual_mono: bool = False, source_frames: int | None = None,
) -> tuple[str, ...]:
    """The complete argv for one render; pure. ``wav_info`` is C-07's RIFF INFO list, verbatim (R8).

    ``source_frames`` is the source's decoded frame count, used by a same-rate dynamic render's trim (CR-077).
    """
    if not isinstance(candidate, Path) or not candidate.is_absolute():
        raise ValueError(f"candidate must be an absolute path, got {candidate!r}")
    chain = filter_chain(plan, probe, target, no_dual_mono=no_dual_mono, source_frames=source_frames)
    args: list[str] = ["-xerror", "-nostats", "-y", *ffmpeg.input_args(source), "-map", "0:a:0"]
    if chain is not None:
        args += ["-af", chain]
    args += encoder_args(target, flac_level=flac_level)
    args += METADATA_FLAGS
    if target.format == "mp3" and not target.stream_copy:
        args += MP3_ENCODER_STATS
    if wav_info:
        if target.format != "wav":
            raise ValueError("RIFF INFO pairs apply to WAV targets only (R8)")
        for key, value in wav_info:
            args += ["-metadata", f"{key}={value}"]
    args += ["-f", MUXERS[target.format], str(candidate)]
    return ffmpeg.ffmpeg_argv(executable, *args)


# --- Edge checks ----------------------------------------------------------------------------


def revalidate_basis(env: ffmpeg.FFmpegEnvironment, target: TargetFormat,
                     evidence: Iterable[CapabilityEvidence]) -> CapabilityEvidence | None:
    """R7: the encode's evidence record exists in this run and still matches the executable.

    Returns the record, or None for a stream copy (which applies no setting).
    A lost or changed executable is ``render_error`` (C-13 stage mapping); a
    missing, ambiguous or changed evidence record is ``unsupported_target``.
    """
    if target.stream_copy:
        return None
    try:
        env.executable.revalidate()
    except ToolError as exc:
        raise exc.to_failure("render_error", STAGE) from None
    matches = [record for record in evidence if record.evidence_id == target.capability_basis]
    if len(matches) != 1:
        state = "missing from" if not matches else "ambiguous in"
        raise FileFailure("unsupported_target", STAGE,
                          f"capability basis {target.capability_basis!r} is {state} this run's evidence")
    record = matches[0]
    if target.format in LOSSLESS_FORMATS:
        representation_ok = (record.supports_float if target.sample_float
                             else target.bit_depth in record.supported_bit_depths)
    else:  # F9/F12: the applied setting is one of the record's confirmed choices
        representation_ok = (record.supports_settings and record.bitrate_mode == target.bitrate_mode
                             and (target.effective_bitrate_bps in record.nominal_bitrates_bps
                                  if target.bitrate_mode == "cbr" else target.vbr_quality in record.vbr_qualities))
    if not ffmpeg.revalidate_evidence(env, record):
        raise FileFailure("unsupported_target", STAGE,
                          f"{record.evidence_id} no longer matches the invoked FFmpeg executable")
    if (record.encoder != target.encoder or record.sample_rate_hz != target.sample_rate_hz
            or record.channels != target.channels or record.channel_layout != target.channel_layout
            or not representation_ok):
        setting = (f"{target.bit_depth}-bit{' float' if target.sample_float else ''}"
                   if target.format in LOSSLESS_FORMATS else
                   f"{target.bitrate_mode} {target.effective_bitrate_bps or f'V{target.vbr_quality}'}")
        raise FileFailure("unsupported_target", STAGE,
                          f"{record.evidence_id} does not evidence {target.encoder} {setting} at "
                          f"{target.sample_rate_hz} Hz, {target.channels} ch / {target.channel_layout}")
    return record


def _source_identity(source: Path, expected: Identity) -> Identity:
    try:
        info = os.stat(source)
    except FileNotFoundError:
        raise FileFailure("source_changed", STAGE, f"source disappeared: {source}") from None
    except PermissionError:
        raise FileFailure("permission_denied", STAGE, f"source unreadable: {source}") from None
    identity = (info.st_dev, info.st_ino)
    if identity != tuple(expected):
        raise FileFailure("source_changed", STAGE, f"source identity changed: {source}")
    return identity


_HEADER_READERS = {
    "wav": ffmpeg.read_wav_header, "aiff": ffmpeg.read_aiff_header,
    "flac": ffmpeg.read_flac_header, "alac": ffmpeg.read_alac_header,
}
_HEADER_BYTES = 1024 * 1024


def _mp3_layout_ok(channel_mode: int, channels: int) -> bool:
    return channel_mode == 3 if channels == 1 else channels == 2 and channel_mode in (0, 1)


class Mp3EncoderCount:
    """CR-063, CR-069: the encoder's own account of one MP3 encode, read from :data:`MP3_ENCODER_STATS` lines.

    ``pre`` lines are the frames sent to the encoder (their sample counts sum
    to the programme length it received); ``post`` lines are the packets it
    returned. The first packet's timestamp sits ``priming`` samples before
    the first frame's: libmp3lame's encoder delay plus the 529-sample decoder
    delay. Every record is validated, not just the first: timestamps are
    integers in one time base, no frame starts before the previous one
    ended, and the packets move forward.

    CR-074: the sample counts are authoritative, not the frame timestamps. A
    frame may start *after* the previous one ended — loudnorm's dynamic
    graph moves its lookahead flush forward by a few thousand samples
    (IC-3 recheck 3 F-5) — but the encoder receives only the samples the
    frames carry, never the gap, so the gap adds nothing to the programme.
    The packets carry the input timeline's timestamps (FFmpeg's audio frame
    queue), so :meth:`totals` cross-checks each packet against that
    timeline: packet ``k`` starts at the timestamp of input sample
    ``k x frame size - priming``. For contiguous frames this is the old
    one-frame step; across a gap the step is a frame plus the gap, and only
    there. The flushed last packet may sit earlier (but after the one
    before). With the priming checked against the encoder's known delay, an
    inconsistent account is never turned into LAME-tag fields. Other stdout
    lines are ignored; a malformed stats line is kept as an error for
    :meth:`totals`, never raised inside the process callback.
    """

    def __init__(self) -> None:
        self.samples = 0
        self.frames_sent = 0
        self.packets = 0
        self.first_frame: tuple[int, str] | None = None
        self.first_packet: tuple[int, str] | None = None
        self.error: str | None = None
        self.timestamp_gaps = 0
        """Forward jumps between frames (CR-074): counted, never added to :attr:`samples`."""
        self._base: str | None = None
        self._next_frame_pts = 0
        self._frame_starts: list[int] = []
        self._frame_pts: list[int] = []
        self._packet_pts: list[int] = []

    def _check_base(self, base: str) -> None:
        if not _TIME_BASE.fullmatch(base):
            raise ValueError(f"unreadable time base {base!r}")
        if self._base is None:
            self._base = base
        elif base != self._base:
            raise ValueError(f"time base {base} differs from the earlier {self._base}")

    def feed(self, line: str) -> None:
        fields = line.split()
        if not fields or fields[0] not in ("pre", "post") or self.error is not None:
            return
        try:
            if fields[0] == "pre":
                _, index, pts_text, base, samples_text = fields
                index, pts, samples = int(index), int(pts_text), int(samples_text)
                if index != self.frames_sent or samples < 0:
                    raise ValueError("frame numbers or sample counts out of order")
                self._check_base(base)
                if self.frames_sent == 0:
                    self.first_frame = (pts, base)
                elif pts < self._next_frame_pts:
                    raise ValueError(f"frame starts at {pts}, before the previous frame ended ({self._next_frame_pts})")
                elif pts > self._next_frame_pts:
                    self.timestamp_gaps += 1
                if samples:
                    self._frame_starts.append(self.samples)
                    self._frame_pts.append(pts)
                self._next_frame_pts = pts + samples
                self.frames_sent += 1
                self.samples += samples
            else:
                _, index, pts_text, base = fields
                index, pts = int(index), int(pts_text)
                if index != self.packets:
                    raise ValueError("packet numbers out of order")
                self._check_base(base)
                if self.packets == 0:
                    self.first_packet = (pts, base)
                elif pts <= self._packet_pts[-1]:
                    raise ValueError(f"packet timestamp advances by {pts - self._packet_pts[-1]}")
                self._packet_pts.append(pts)
                self.packets += 1
        except ValueError as exc:
            self.error = f"unreadable encoder statistics line {line!r}: {exc}"

    def _input_pts(self, position: int) -> int:
        """The timestamp of input sample ``position`` on the frames' own timeline (CR-074).

        Before the first frame and after the last, the timeline runs on
        contiguously from that frame (priming and padding carry no frame).
        """
        assert self.first_frame is not None
        if position < 0 or not self._frame_starts:
            return self.first_frame[0] + position
        if position >= self.samples:
            return self._next_frame_pts + position - self.samples
        i = bisect.bisect_right(self._frame_starts, position) - 1
        return self._frame_pts[i] + position - self._frame_starts[i]

    def _packet_mismatch(self, frame_size: int, priming: int) -> str | None:
        """Why a packet does not sit where its frame of input puts it, or None (CR-069, CR-074)."""
        last = self.packets - 1
        for k in range(1, self.packets):
            pts, expected = self._packet_pts[k], self._input_pts(k * frame_size - priming)
            if k < last and pts != expected:
                return (f"packet {k} starts at {pts}, not at {expected} where the {frame_size}-sample frame size "
                        f"puts it on the encoder's input timeline")
            if k == last and pts > expected:
                return (f"the last packet starts at {pts}, beyond {expected} where the {frame_size}-sample frame "
                        f"size puts it on the encoder's input timeline")
        return None

    def totals(self, rate: int, source: tuple[int, int] | None = None) -> tuple[int, int, int]:
        """``(samples received, priming, packets)``; ``render_error`` when the account is incomplete or inconsistent.

        ``source`` is ``(frames, sample rate)`` of what Norma fed the encoder
        (CR-071, D-085). The encoder's total input samples must equal the
        source frames at the same rate, so a missing or altered terminal
        record is refused there before any tag write. For a resampled render
        they must sit within one sample of the rate-scaled count (D-086): a
        terminal record altered by less than that is not detectable, and is
        the trust boundary D-085 records (B-007).
        """
        why = self.error
        if why is None and (self.first_frame is None or self.first_packet is None or self.samples <= 0):
            why = "the encoder reported no frames or packets"
        if why is None:
            assert self.first_frame is not None and self.first_packet is not None
            base = f"1/{rate}"
            frame_size = 1152 if rate >= 32000 else 576
            priming = self.first_frame[0] - self.first_packet[0]
            if self._base != base:
                why = f"encoder statistics use time base {self._base}, not {base}"
            elif priming != MP3_ENCODER_DELAY + MP3_DECODER_DELAY:
                why = (f"encoder priming of {priming} samples is not the expected {MP3_ENCODER_DELAY}-sample "
                       f"encoder delay plus the {MP3_DECODER_DELAY}-sample decoder delay")
            elif why := self._packet_mismatch(frame_size, priming):
                pass
            elif self.packets * frame_size < priming + self.samples:
                why = f"{self.packets} packets of {frame_size} cannot hold {priming} priming and {self.samples} samples"
            elif source is not None and (why := _input_count_mismatch(self.samples, rate, *source)):
                pass
            else:
                return self.samples, priming, self.packets
        raise FileFailure("render_error", STAGE, f"cannot account for the MP3 encode's samples: {why}")


def _input_count_mismatch(samples: int, rate: int, source_frames: int, source_rate: int) -> str | None:
    """CR-071: why the encoder's input count differs from what was fed it, or None.

    Same rate: exact. Resampled: within one sample of ``frames x rate /
    source rate`` (integer arithmetic). D-086 fixes this at one sample,
    whatever the length: it is a check on the encoder's own input count,
    deliberately tighter than the verify length rule's ``max(1, frames /
    source rate)`` tolerance, which would admit the four-sample loss of
    IC-3 recheck 2 F-4 on a 5 s source.
    """
    if source_frames <= 0 or source_rate <= 0:
        return f"the source's {source_frames} frames at {source_rate} Hz cannot be reconciled with the encoder's count"
    if source_rate == rate:
        if samples != source_frames:
            return f"the encoder received {samples} samples, the source has {source_frames} frames at {rate} Hz"
        return None
    if abs(samples * source_rate - source_frames * rate) > source_rate:
        return (f"the encoder received {samples} samples at {rate} Hz, but {source_frames} source frames at "
                f"{source_rate} Hz expect {source_frames * rate / source_rate:.1f} ± 1")
    return None


def _crc16_arc(data: bytes) -> int:
    """CRC-16/ARC (reflected 0x8005, initial 0): the LAME tag CRC, as LAME and FFmpeg's reader compute it."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _lame_tag_offset(data: bytes, frame: ffmpeg.Mp3Frame, packets: int) -> int:
    """The byte offset of the first frame's LAME extension, after checking its Xing/Info frame count."""
    side = (17 if frame.channel_mode == 3 else 32) if frame.version == "1" else (9 if frame.channel_mode == 3 else 17)
    pos = frame.offset + 4 + side
    if data[pos:pos + 4] not in (b"Xing", b"Info"):
        raise ValueError("the first frame carries no Xing/Info tag")
    flags = int.from_bytes(data[pos + 4:pos + 8], "big")
    if not flags & 0x1:
        raise ValueError("the Xing/Info tag carries no frame count")
    tagged = int.from_bytes(data[pos + 8:pos + 12], "big")
    if tagged != packets:
        raise ValueError(f"the Xing/Info tag counts {tagged} frames, the encoder returned {packets}")
    lame = pos + 8 + 4 + (4 if flags & 0x2 else 0) + (100 if flags & 0x4 else 0) + (4 if flags & 0x8 else 0)
    if lame + 36 > frame.offset + frame.length or data[lame:lame + 4] not in _LAME_TAG_VERSIONS:
        raise ValueError("the first frame carries no LAME extension a decoder honours")
    return lame


def restore_mp3_gapless(candidate: Path, samples: int, priming: int, packets: int) -> tuple[int, int]:
    """CR-063: rewrite the LAME tag's delay and padding from the encoder's own count; return them.

    ``samples`` is the programme length the encoder received, ``priming`` its
    first packet's lead and ``packets`` the audio frames it returned
    (:class:`Mp3EncoderCount`). With frame size ``s`` (1,152 MPEG-1, 576
    MPEG-2/2.5), ``packets x s = priming + samples + trailing`` and the LAME
    convention stores ``delay = priming - 529`` and
    ``padding = packets x s - delay - samples``, so a decoder that skips
    ``delay + 529`` and drops ``padding - 529`` keeps exactly the
    programme. Only those three bytes and the tag CRC (CRC-16/ARC of every
    frame byte before it) change, in place on the same inode; nothing else
    is touched. A file whose frames, tag or fields do not fit fails
    ``render_error``.
    """
    try:
        data = candidate.read_bytes()
        frames = ffmpeg.read_mp3_frames(data)
        if len(frames) != packets + 1:
            raise ValueError(f"{len(frames)} frames written for {packets} encoded packets and the tag frame")
        first = frames[0]
        lame = _lame_tag_offset(data, first, packets)
    except (OSError, ValueError) as exc:
        raise FileFailure("render_error", STAGE, f"cannot restore the MP3 gapless fields: {exc}") from None
    frame_size = 1152 if first.version == "1" else 576
    delay = priming - MP3_DECODER_DELAY
    padding = packets * frame_size - delay - samples
    if not (0 <= delay <= _MP3_FIELD_MAX and MP3_DECODER_DELAY <= padding <= _MP3_FIELD_MAX):
        raise FileFailure("render_error", STAGE,
                          f"MP3 gapless fields out of range: delay {delay}, padding {padding} for {samples} "
                          f"samples in {packets} frames of {frame_size}")
    patched = bytearray(data[first.offset:lame + 36])
    field = lame + 21 - first.offset
    patched[field:field + 3] = ((delay << 12) | padding).to_bytes(3, "big")
    crc_at = lame + 34 - first.offset
    patched[crc_at:crc_at + 2] = _crc16_arc(bytes(patched[:crc_at])).to_bytes(2, "big")
    try:
        with open(candidate, "r+b") as handle:
            handle.seek(lame + 21)
            handle.write(patched[field:field + 3])
            handle.seek(lame + 34)
            handle.write(patched[crc_at:crc_at + 2])
    except OSError as exc:
        raise FileFailure("disk_full" if exc.errno == errno.ENOSPC else "render_error", STAGE,
                          f"cannot write the MP3 gapless fields: {exc}") from None
    return delay, padding


def confirm_mp3(candidate: Path, target: TargetFormat) -> tuple[str, int | None]:
    """R7 for MP3: every written frame header and the first frame's Xing/Info tag.

    Each frame must carry the C-04 rate and a channel mode for its channel
    count; CBR frames must all carry the nominal; VBR needs the ``Xing`` tag
    (VBR mode), CBR the ``Info`` tag. The tag's gapless fields are rewritten
    afterwards by :func:`restore_mp3_gapless` (CR-063). Returns the evidence
    locator and the confirmed nominal (None for VBR, which has none).

    CR-075: a V0 encode whose frames all carry one bitrate (seen at 12 kHz
    stereo: every frame 64 kbit/s, LAME's ceiling there) gets the ``Info``
    tag and stays refused. FFmpeg's muxer chooses ``Xing`` or ``Info`` from
    the frame bitrates alone and writes no VBR method or quality into the
    LAME extension, so such a file is structurally the same as a CBR 64k
    encode: nothing written confirms V0.
    """
    try:
        data = candidate.read_bytes()
        frames = ffmpeg.read_mp3_frames(data)
    except (OSError, ValueError) as exc:
        raise FileFailure("render_error", STAGE, f"cannot read the written MP3 frames: {exc}") from None
    tag = ffmpeg.read_xing_tag(data, frames[0])
    expected_tag = b"Xing" if target.bitrate_mode == "vbr" else b"Info"
    bad = [f for f in frames if f.sample_rate_hz != target.sample_rate_hz
           or not _mp3_layout_ok(f.channel_mode, target.channels)
           or (target.bitrate_mode == "cbr" and f.bitrate_bps != target.effective_bitrate_bps)]
    if bad or tag is None or tag.tag != expected_tag:
        why = (f"{len(bad)} of {len(frames)} frames differ from {target.sample_rate_hz} Hz, {target.channels} ch"
               + (f", {target.effective_bitrate_bps} bps" if target.bitrate_mode == "cbr" else "")
               if bad else f"first frame carries {tag.tag if tag else None!r}, expected {expected_tag!r}")
        raise FileFailure("encoder_settings_mismatch", STAGE, f"written MP3 does not confirm the C-04 setting: {why}")
    if target.bitrate_mode == "vbr":
        return (f"argv -q:a {target.vbr_quality} (validated setting, basis {target.capability_basis}); "
                f"{len(frames)} written frames at {target.sample_rate_hz} Hz {target.channels} ch; Xing VBR tag"), None
    return (f"argv -b:a {target.effective_bitrate_bps}; every one of {len(frames)} written frames carries "
            f"{target.effective_bitrate_bps} bps at {target.sample_rate_hz} Hz {target.channels} ch; Info tag; "
            f"basis {target.capability_basis}"), target.effective_bitrate_bps


def confirm_aac(target: TargetFormat, diagnostics: Sequence[str]) -> tuple[str, int | None]:
    """R7 for AAC (``encoder_diagnostic``): no aac_at diagnostic may report a changed setting.

    aac_at reports an unsupported nominal as a diagnostic while exiting 0
    (ADR-0015), so every diagnostic other than the flush-queue note fails
    ``encoder_settings_mismatch``. Rate and channels are rechecked on the
    candidate's probe at verification.
    """
    unexpected = [line for line in diagnostics if not _AAC_FLUSH_NOTE.fullmatch(line)]
    if unexpected:
        raise FileFailure("encoder_settings_mismatch", STAGE,
                          "aac_at reported: " + " | ".join(unexpected[-3:]))
    return (f"argv -aac_at_mode cbr -b:a {target.effective_bitrate_bps}; no aac_at setting diagnostic; "
            f"basis {target.capability_basis}"), target.effective_bitrate_bps


def confirm_written(candidate: Path, target: TargetFormat) -> str:
    """R7: read the written header back; return the setting evidence locator.

    The header must carry the C-04 depth, float kind, rate and channels;
    anything else is ``encoder_settings_mismatch``, never a silent success.
    Lossy targets use :func:`confirm_mp3` and :func:`confirm_aac` instead.
    """
    if target.stream_copy:
        return "stream copy: encoded audio copied unchanged; no encoder setting applied"
    if target.format not in LOSSLESS_FORMATS:
        raise ValueError("lossy targets are confirmed by confirm_mp3 / confirm_aac")
    reader = _HEADER_READERS[target.format]
    try:
        with open(candidate, "rb") as handle:
            # ALAC's sample description sits in the moov box, written last.
            data = handle.read() if target.format == "alac" else handle.read(_HEADER_BYTES)
        header = reader(data)
    except (OSError, ValueError) as exc:
        raise FileFailure("render_error", STAGE, f"cannot read the written {target.format} header: {exc}") from None
    expected = (target.bit_depth, target.sample_float, target.sample_rate_hz, target.channels)
    written = (header.bit_depth, header.sample_float, header.sample_rate_hz, header.channels)
    label = (f"{header.bit_depth}-bit{' float' if header.sample_float else ''} "
             f"{header.sample_rate_hz} Hz {header.channels} ch")
    if written != expected:
        raise FileFailure("encoder_settings_mismatch", STAGE,
                          f"written header {label} differs from the C-04 target {expected}")
    return f"written header {label}; encoder {target.encoder}; basis {target.capability_basis}"


def _process_failure(result: ffmpeg.ProcessResult) -> FileFailure:
    detail = f"ffmpeg exited with status {result.returncode}"
    if result.stderr_tail:
        detail += "\n" + "\n".join(result.stderr_tail)
    full = any(os.strerror(errno.ENOSPC) in line for line in result.stderr_tail)
    return FileFailure("disk_full" if full else "render_error", STAGE, detail)


# --- One render -----------------------------------------------------------------------------


def render(
    source: Path, source_identity: Identity, probe: ProbeResult, plan: GainPlan, target: TargetFormat,
    candidate: Path, env: ffmpeg.FFmpegEnvironment, evidence: Iterable[CapabilityEvidence], *,
    attempt: int = 1, flac_level: int = 8, wav_info: Sequence[tuple[str, str]] = (),
    no_dual_mono: bool = False, cancel_event: threading.Event | None = None,
    source_frames: int | None = None,
) -> RenderResult:
    """Render the original source once into ``candidate`` and return C-05 RenderResult.

    ``candidate`` is the private temporary WS-05 allocated; it is overwritten
    in place (same inode). ``source_identity`` is C-01's identity, checked
    before and after the render. The result claims no measured output: the
    caller verifies the candidate (``audio/verify.py``) before any use.
    ``no_dual_mono`` is the run's M3 option, used by a dynamic render.
    ``source_frames`` is the source's C-03 decoded frame count; an MP3 render
    reconciles it with the encoder's account (CR-071), and a same-rate dynamic
    render trims to it (CR-077); either counts the source itself when it is
    not supplied.
    """
    if not (isinstance(attempt, int) and 1 <= attempt <= 5):
        raise ValueError("attempt must be 1..5 (ADR-0014)")
    mp3 = target.format == "mp3" and not target.stream_copy
    if source_frames is None and (mp3 or needs_source_frames(plan, probe, target)):
        # check the render is possible before spending a decode on it
        filter_chain(plan, probe, target, no_dual_mono=no_dual_mono)
        revalidate_basis(env, target, tuple(evidence))
        try:
            source_frames = count_decoded_frames(source, probe, env.executable, cancel_event=cancel_event)
        except FileFailure as exc:
            raise FileFailure("render_error", STAGE, f"cannot count the source frames: {exc.detail}") from None
    argv = render_argv(env.executable, source, candidate, plan, probe, target,
                       flac_level=flac_level, wav_info=wav_info, no_dual_mono=no_dual_mono,
                       source_frames=source_frames)
    revalidate_basis(env, target, tuple(evidence))
    identity = _source_identity(source, source_identity)
    loudnorm = LoudnormJson()
    aac_diagnostics: list[str] = []
    mp3_count = Mp3EncoderCount() if target.format == "mp3" and not target.stream_copy else None

    def feed(line: str) -> None:
        loudnorm.feed(line)
        match = _AAC_DIAGNOSTIC.match(line.strip())
        if match:
            aac_diagnostics.append(match.group(1).strip())

    try:
        result = ffmpeg.run_process(argv, executable=env.executable, capture_stdout=False,
                                    deadline_s=ffmpeg.tool_deadline(probe.duration_s), on_stderr_line=feed,
                                    on_stdout_line=mp3_count.feed if mp3_count else None,
                                    cancel_event=cancel_event)
    except ToolError as exc:
        raise exc.to_failure("render_error", STAGE) from None
    if result.returncode != 0:
        raise _process_failure(result)
    _source_identity(source, source_identity)
    bitrate: int | None = None
    if target.stream_copy or target.format in LOSSLESS_FORMATS:
        setting = confirm_written(candidate, target)
    elif mp3_count is not None:
        setting, bitrate = confirm_mp3(candidate, target)
        assert source_frames is not None
        delay, padding = restore_mp3_gapless(
            candidate, *mp3_count.totals(target.sample_rate_hz, (source_frames, probe.sample_rate_hz)))
        setting += (f"; gapless delay {delay}, padding {padding} from the encoder's "
                    f"{mp3_count.samples} samples")
    else:
        setting, bitrate = confirm_aac(target, aac_diagnostics)
    normalization = None
    if plan.action == "dynamic":
        try:
            normalization = loudnorm.last().get("normalization_type")
        except FileFailure as exc:
            raise FileFailure("render_error", STAGE, f"dynamic render: {exc.detail}") from None
        if normalization not in ("dynamic", "linear"):
            raise FileFailure("render_error", STAGE,
                              f"dynamic render reported normalization_type={normalization!r}")
    return RenderResult(
        candidate_path=candidate, attempt=attempt, plan=plan, target=target,
        decoded_source_identity=identity, elapsed_s=result.elapsed_s,
        stderr_tail="\n".join(result.stderr_tail) or None,
        encoder_effective_bitrate_bps=bitrate, encoder_settings_valid=True,
        encoder_setting_evidence=setting, normalization_type=normalization,
    )
