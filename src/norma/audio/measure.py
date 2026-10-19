"""Streaming ebur128 measurement (C-03; audio rules M1–M7).

The two meter scans retain only complete frame records. A separate cumulative
astats scan counts decoded frames and distinguishes exact zero from a peak
rounded to zero; its floating conversion never precedes the loudness meter.
No audio or metadata file is written. Structural validation belongs to WS-05
and must precede this entry point; a probe alone does not establish integrity.

Every scan passes ``-nostats`` (IC-2 F-2, D-072 (2)): without it FFmpeg
writes ``size=… speed=…`` progress text ending in a bare CR once a scan lasts
about half a second, and that text joins the next metadata line. As defence
in depth the record parser also keeps only the text after the last CR of a
line, so a progress prefix can never hide a frame header or field.

:func:`calibrate_dynamic` is explicit ``dynamic`` mode's extra analysis of the
original source (audio R5, V6): one ``loudnorm`` pass whose final JSON object
supplies C-05 ``dynamic_parameters``. Its values calibrate the renderer only;
the ebur128 scans above remain the planning and acceptance meter (M1). An
invalid or missing calibration value is ``measurement_error``, never zero.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable
from pathlib import Path

from norma.adapters import ffmpeg
from norma.audio.plan import (
    DYNAMIC_LRA,
    LOUDNORM_TARGET_RANGE,
    LOUDNORM_TP_RANGE,
    DynamicCalibration,
    in_range,
)
from norma.errors import FileFailure, SkipFile, ToolError
from norma.models import DYNAMIC_CALIBRATION_KEYS, Diagnostic, LoudnessMeasurement, ProbeResult

_PREFIX = re.compile(r"^\[[^]]+\]\s*")
_HEADER = re.compile(r"^frame:(\d+)\s+pts:\S+\s+pts_time:(\S+)\s*$")
_ASTATS = (
    "aformat=sample_fmts=dbl,astats=metadata=1:reset=0:measure_perchannel=none:"
    "measure_overall=Peak_level+Number_of_samples+Number_of_NaNs+Number_of_Infs,"
    "ametadata=mode=print"
)
SCAN_FLAGS: tuple[str, ...] = ("-xerror", "-nostats")
"""Leading options of every meter scan: stop on decode errors, never print progress (F-2)."""


def _failure(detail: str) -> FileFailure:
    return FileFailure("measurement_error", "measure", detail)


class _Records:
    """Bounded metadata record assembly, ignoring summaries and tool chatter."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.fields: dict[str, str] | None = None
        self.records = 0
        self.last_index = -1

    def feed(self, line: str) -> None:
        # A bare CR ends FFmpeg progress text; only what follows the last one is a line (F-2).
        text = _PREFIX.sub("", line.rstrip("\r").rsplit("\r", 1)[-1]).strip()
        match = _HEADER.fullmatch(text)
        if match:
            self.flush()
            index = int(match[1])
            if index != self.last_index + 1:
                raise _failure("non-contiguous metadata frame sequence")
            self.last_index = index
            try:
                timestamp = float(match[2])
            except ValueError:
                raise _failure("invalid metadata timestamp") from None
            if not math.isfinite(timestamp):
                raise _failure("non-finite metadata timestamp")
            self.fields = {}
        elif text.startswith("frame:"):
            raise _failure("malformed metadata frame header")
        elif text.startswith(self.prefix):
            if self.fields is None or "=" not in text:
                raise _failure("metadata outside a complete frame record")
            key, value = text.split("=", 1)
            if key in self.fields:
                raise _failure(f"duplicate metadata field: {key}")
            self.fields[key] = value

    def flush(self) -> None:
        if self.fields is not None:
            self.consume(self.fields)
            self.records += 1
            self.fields = None

    def consume(self, fields: dict[str, str]) -> None:
        raise NotImplementedError

    @staticmethod
    def number(fields: dict[str, str], key: str) -> float:
        try:
            return float(fields[key])
        except (KeyError, ValueError):
            raise _failure(f"missing or invalid metadata field: {key}") from None


class MeterParser(_Records):
    """Last valid programme I/LRA, maximum channel peak over every frame.

    Feed captured stderr lines, then call finish after successful process EOF.
    No summary, aggregate true_peak or partial record is an alternative oracle.
    """

    def __init__(self, channels: int) -> None:
        if channels < 1:
            raise ValueError("channels must be positive")
        super().__init__("lavfi.r128.")
        self.channels = channels
        self.integrated_lufs: float | None = None
        self.lra_lu: float | None = None
        self.peak: float | None = None

    def consume(self, fields: dict[str, str]) -> None:
        integrated = self.number(fields, "lavfi.r128.I")
        lra = self.number(fields, "lavfi.r128.LRA")
        if math.isnan(integrated) or integrated == math.inf:
            raise _failure("invalid non-finite integrated loudness")
        if not math.isfinite(lra) or lra < 0:
            raise _failure("invalid loudness range")
        peaks = {f"lavfi.r128.true_peaks_ch{channel}" for channel in range(self.channels)}
        present = {key for key in fields if key.startswith("lavfi.r128.true_peaks_ch")}
        if present != peaks:
            raise _failure("missing or unexpected true-peak channel")
        values = [self.number(fields, key) for key in peaks]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise _failure("invalid non-finite or negative channel peak")
        self.peak = max(values + ([] if self.peak is None else [self.peak]))
        if math.isfinite(integrated) and integrated != -70:
            self.integrated_lufs = integrated
        self.lra_lu = lra

    def finish(self) -> None:
        self.flush()


class _SampleParser(_Records):
    def __init__(self) -> None:
        super().__init__("lavfi.astats.")
        self.frames = 0
        self.nonzero = False

    def consume(self, fields: dict[str, str]) -> None:
        count = self.number(fields, "lavfi.astats.Overall.Number_of_samples")
        peak = self.number(fields, "lavfi.astats.Overall.Peak_level")
        nans = self.number(fields, "lavfi.astats.Number of NaNs")
        infinities = self.number(fields, "lavfi.astats.Number of Infs")
        if not math.isfinite(count) or not count.is_integer() or count <= self.frames:
            raise _failure("invalid cumulative decoded sample count")
        if nans != 0 or infinities != 0 or math.isnan(peak) or peak == math.inf:
            raise _failure("decoded samples contain non-finite audio")
        self.frames = int(count)
        self.nonzero |= peak != -math.inf


def measurement_argv(
    executable: ffmpeg.Executable, path: Path, *, dual_mono: bool, padded: bool = False,
) -> tuple[str, ...]:
    """M1/M6 argv: first audio stream, no pre-meter sample transformation."""
    filters = (
        ("apad=pad_dur=0.1," if padded else "")
        + f"ebur128=peak=true:metadata=1:dualmono={str(dual_mono).lower()}:panlaw=-3.0103,"
        + "ametadata=mode=print"
    )
    return ffmpeg.ffmpeg_argv(
        executable, *SCAN_FLAGS, *ffmpeg.input_args(path), "-map", "0:a:0",
        "-af", filters, "-f", "null", "-",
    )


def sample_count_argv(executable: ffmpeg.Executable, path: Path) -> tuple[str, ...]:
    """The cumulative astats scan: decoded frame count, exact zero and finiteness (no meter)."""
    return ffmpeg.ffmpeg_argv(
        executable, *SCAN_FLAGS, *ffmpeg.input_args(path), "-map", "0:a:0",
        "-af", _ASTATS, "-f", "null", "-",
    )


def conservative_peak_dbtp(amplitude: float | None, resolution: float | None) -> float | None:
    """M7 decision bound, not a reported measurement; None cannot prove safety."""
    if (amplitude is None or resolution is None or not math.isfinite(amplitude)
            or amplitude < 0 or not math.isfinite(resolution) or resolution <= 0):
        return None
    upper = amplitude + resolution / 2
    return 20 * math.log10(upper) if math.isfinite(upper) else None


def _scan(
    executable: ffmpeg.Executable, argv: tuple[str, ...], parser: _Records,
    duration_s: float | None, cancel_event: threading.Event | None,
) -> None:
    try:
        ffmpeg.check_result(ffmpeg.run_process(
            argv, executable=executable, deadline_s=ffmpeg.tool_deadline(duration_s),
            capture_stdout=False, on_stderr_line=parser.feed, cancel_event=cancel_event,
        ))
    except ToolError as exc:
        raise exc.to_failure("decode_error", "measure") from None
    parser.flush()


def _count_samples(
    path: Path, probe: ProbeResult, executable: ffmpeg.Executable,
    cancel_event: threading.Event | None,
) -> _SampleParser:
    samples = _SampleParser()
    _scan(executable, sample_count_argv(executable, path), samples, probe.duration_s, cancel_event)
    if not samples.records:
        raise _failure("no decoded sample records")
    return samples


def count_decoded_frames(
    path: Path, probe: ProbeResult, executable: ffmpeg.Executable, *,
    cancel_event: threading.Event | None = None,
) -> int:
    """Decoded frames per channel of the first audio stream, by the same scan C-03 uses.

    Raises FileFailure at stage ``measure`` like :func:`measure`; callers at a
    later stage restate it. Used by candidate verification when the source's
    C-03 frame count is not supplied (IC-2 F-3).
    """
    return _count_samples(path, probe, executable, cancel_event).frames


def measure(
    path: Path, probe: ProbeResult, executable: ffmpeg.Executable, *,
    no_dual_mono: bool = False, cancel_event: threading.Event | None = None,
) -> LoudnessMeasurement:
    """Fully decode a structurally validated local input and return immutable C-03.

    Silent/unmeasurable states carry their C-13 diagnostic for the caller to
    settle as skips; tool, decode and parser failures raise FileFailure. Each
    of the three scans has its own duration-aware adapter deadline. The third
    scan supplies exact programme duration, never the padded meter duration.
    ADR-0020 limits v1 to mono/stereo: other channel counts raise SkipFile
    before constructing or running any scan, without producing C-03 values.
    """
    if probe.channels > 2:
        raise SkipFile("unsupported_channels", "measure", "v1 supports mono and stereo inputs only")
    dual_mono = probe.channels == 1 and not no_dual_mono
    programme, padded = MeterParser(probe.channels), MeterParser(probe.channels)
    # Establish finite, non-corrupt decoded audio before classifying meter values.
    samples = _count_samples(path, probe, executable, cancel_event)
    _scan(executable, measurement_argv(executable, path, dual_mono=dual_mono),
          programme, probe.duration_s, cancel_event)
    _scan(executable, measurement_argv(executable, path, dual_mono=dual_mono, padded=True),
          padded, probe.duration_s, cancel_event)
    duration = samples.frames / probe.sample_rate_hz
    if programme.records != samples.frames * 10 // probe.sample_rate_hz:
        raise _failure("programme metadata does not cover all complete 100 ms intervals")
    if padded.records <= programme.records or padded.peak is None:
        raise _failure("supplementary terminal peak coverage unavailable")
    if probe.duration_basis != "unknown" and probe.duration_s is not None:
        if probe.duration_s - duration > 0.5:
            raise FileFailure("decode_error", "measure", "decoded duration shortfall exceeds 0.5 s")
    peak = max(value for value in (programme.peak, padded.peak) if value is not None)
    integrated = programme.integrated_lufs
    diagnostics: tuple[Diagnostic, ...] = ()
    if not samples.nonzero:
        if peak != 0 or integrated is not None:
            raise _failure("meter disagrees with proven zero decoded samples")
        state, dbtp = "silent", -math.inf
        diagnostics = (Diagnostic("silent", "all decoded samples are zero", "measure"),)
    elif integrated is None:
        state, dbtp = "unmeasurable", 20 * math.log10(peak) if peak > 0 else None
        diagnostics = (Diagnostic("unmeasurable", "nonzero signal has no gated integrated loudness", "measure"),)
    else:
        state, dbtp = "measurable", 20 * math.log10(peak) if peak > 0 else None
        if dbtp is None:
            raise _failure("measurable signal has no finite printed peak")
        if duration < 3:
            diagnostics = (Diagnostic("too_short", "measurable programme is shorter than 3 s", "measure"),)
    return LoudnessMeasurement(
        signal_state=state, integrated_lufs=integrated, true_peak_dbtp=dbtp,
        true_peak_linear=peak, lra_lu=programme.lra_lu, threshold_lufs=None,
        integrated_valid=integrated is not None, peak_valid=dbtp is not None,
        lra_valid=programme.lra_lu is not None, threshold_valid=False,
        integrated_resolution_lu=0.001, peak_resolution_linear=0.001,
        dual_mono=dual_mono, meter="ebur128", value_source="metadata",
        ffmpeg_version=executable.version, decoded_duration_s=duration,
        decoded_frames=samples.frames, sample_rate_hz=probe.sample_rate_hz,
        decoded_nonzero=samples.nonzero, terminal_peak_covered=True, diagnostics=diagnostics,
    )


# --- Explicit dynamic mode: loudnorm calibration (audio R5, R6, V6) ------------------------------


def _calibration_failure(detail: str) -> FileFailure:
    return FileFailure("measurement_error", "measure", f"invalid loudnorm calibration: {detail}")


class LoudnormJson:
    """Collect the JSON objects ``loudnorm`` prints to stderr with ``print_format=json``.

    loudnorm writes a ``[Parsed_loudnorm_N @ 0x…]`` line, then the object one
    line per member between a bare ``{`` and ``}``. Only complete objects are
    kept; :meth:`last` returns the final one (R5: "parse the final JSON object").
    """

    def __init__(self) -> None:
        self._lines: list[str] | None = None
        self.objects: list[str] = []

    def feed(self, line: str) -> None:
        text = _PREFIX.sub("", line.rstrip("\r").rsplit("\r", 1)[-1]).strip()
        if text == "{":
            self._lines = [text]
        elif self._lines is not None:
            self._lines.append(text)
            if text == "}":
                self.objects.append("\n".join(self._lines))
                self._lines = None

    def last(self) -> dict[str, str]:
        if not self.objects:
            raise _calibration_failure("loudnorm printed no complete JSON object")
        try:
            value = json.loads(self.objects[-1])
        except ValueError:
            raise _calibration_failure("loudnorm JSON does not parse") from None
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                  for k, v in value.items()):
            raise _calibration_failure("loudnorm JSON is not an object of strings")
        return value


def parse_calibration(fields: dict[str, str], *, target_lufs: float, ceiling_dbtp: float) -> DynamicCalibration:
    """The five C-05 calibration values and ``normalization_type`` from one loudnorm object.

    Every value must be present and finite; loudnorm's ``-inf``/``inf`` or a
    missing key is rejected, never replaced by zero (R5).
    """
    values: dict[str, float] = {}
    for key in DYNAMIC_CALIBRATION_KEYS:
        if key not in fields:
            raise _calibration_failure(f"missing {key}")
        try:
            number = float(fields[key])
        except ValueError:
            raise _calibration_failure(f"{key}={fields[key]!r}") from None
        if not math.isfinite(number):
            raise _calibration_failure(f"{key}={fields[key]!r} is not finite")
        values[key] = number
    kind = fields.get("normalization_type")
    if kind not in ("dynamic", "linear"):
        raise _calibration_failure(f"normalization_type={kind!r}")
    return DynamicCalibration(parameters=values, normalization_type=kind, target_lufs=target_lufs,
                              ceiling_dbtp=ceiling_dbtp)


def calibration_filter(*, target_lufs: float, ceiling_dbtp: float, dual_mono: bool) -> str:
    """R5's analysis filter: ``loudnorm=I=<t>:TP=<ceiling>:LRA=50:dual_mono=<bool>:print_format=json``."""
    for name, value, bounds in (("I", target_lufs, LOUDNORM_TARGET_RANGE), ("TP", ceiling_dbtp, LOUDNORM_TP_RANGE)):
        if not in_range(value, bounds):
            raise _calibration_failure(f"{name}={value!r} is outside loudnorm's range {bounds}; it is not clamped")
    return (f"loudnorm=I={float(target_lufs)!r}:TP={float(ceiling_dbtp)!r}:LRA={DYNAMIC_LRA}:"
            f"dual_mono={str(dual_mono).lower()}:print_format=json")


def calibration_argv(executable: ffmpeg.Executable, path: Path, *, target_lufs: float, ceiling_dbtp: float,
                     dual_mono: bool) -> tuple[str, ...]:
    """The calibration pass over the original source: first audio stream, no sample transformation."""
    return ffmpeg.ffmpeg_argv(
        executable, *SCAN_FLAGS, *ffmpeg.input_args(path), "-map", "0:a:0",
        "-af", calibration_filter(target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp, dual_mono=dual_mono),
        "-f", "null", "-",
    )


def calibrate_dynamic(
    path: Path, probe: ProbeResult, executable: ffmpeg.Executable, *, target_lufs: float, ceiling_dbtp: float,
    no_dual_mono: bool = False, cancel_event: threading.Event | None = None,
) -> DynamicCalibration:
    """Run R5's loudnorm analysis of the original source at ``ceiling_dbtp`` (V6: the current dynamic ceiling).

    Not a render and not a measurement of record (V2, M1). Uses M3's mono
    policy, its own duration-aware deadline (T3) and ``-xerror``. Tool and
    decode failures keep their C-13 reasons; invalid JSON is ``measurement_error``.
    """
    if probe.channels > 2:
        raise SkipFile("unsupported_channels", "measure", "v1 supports mono and stereo inputs only")
    dual_mono = probe.channels == 1 and not no_dual_mono
    argv = calibration_argv(executable, path, target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp,
                            dual_mono=dual_mono)
    parser = LoudnormJson()
    _run_lines(executable, argv, parser.feed, probe.duration_s, cancel_event)
    return parse_calibration(parser.last(), target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp)


def _run_lines(executable: ffmpeg.Executable, argv: tuple[str, ...], feed: Callable[[str], None],
               duration_s: float | None, cancel_event: threading.Event | None) -> None:
    try:
        ffmpeg.check_result(ffmpeg.run_process(
            argv, executable=executable, deadline_s=ffmpeg.tool_deadline(duration_s),
            capture_stdout=False, on_stderr_line=feed, cancel_event=cancel_event,
        ))
    except ToolError as exc:
        raise exc.to_failure("decode_error", "measure") from None
