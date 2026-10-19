"""Candidate and read-only verification (C-05/C-09; audio V1, V9, M7, T1–T3).

Read-only :func:`select_reference` and :func:`assess_inspection` consume
resolved run options and an already-parsed mark, never files or mark text.
V9's marked reduce has an upper loudness bound; candidate :func:`assess`
retains V1's distinct pure-reduce semantics.

- The whole decoded candidate is measured with the source's meter and mono
  policy (``audio/measure.py``, M3). A measurement the meter cannot produce
  fails ``verification_loudness``; decode corruption, timeouts and tool loss
  keep their own C-13 reasons at stage ``verify``.
- Acceptance (V1): ordinary actions need ``abs(I_out - t) <= tolerance``; a
  pure ``reduce`` needs a valid measured I. Every action needs the M7
  conservative level ``20*log10(a + q/2) <= c + 0.1`` dBTP; an unknown ``q``
  is never peak-safe. ``peak_excess_db`` keeps the printed ``TP_out - c`` (T2).
- A copy that should keep every sample (``render.preserves_samples``) must
  decode to the source's exact samples, compared as SHA-256 of a float64
  decode (F13, TST-F-02).
- Every candidate must decode to the source's length (IC-2 F-3, D-072 (3);
  V1 "validate decode/structure first"; TESTING § 2): the same frame count at
  the same rate, otherwise within ±1 sample per second of audio of the
  rate-scaled count. Anything else is ``render_error`` at stage ``verify``:
  the render did not reproduce its whole input.
- Lossy encodes (CR-059, audio E3): priming is never allowed for — a
  decoded encode must start where the source starts, so leading priming
  would show as excess length. MP3 is held to the lossless rule exactly: the
  render rewrites the Xing/LAME tag's delay and padding from the encoder's
  own sample count (CR-063), so they restore the exact length. An
  ``aac_at`` M4A's edit list removes the priming, but up to one AAC frame
  minus one sample (1,023 frames) of trailing padding can remain; only that
  trailing excess is accepted, never a shortfall. Measured on this Mac,
  FFmpeg N-125892: MP3 V0 0 extra frames and zero lag at 22.05, 24, 32,
  44.1 and 48 kHz, frame residues 0/1/137/511 (2026-10-12, TK-0113; before
  CR-063 FFmpeg's own fields gave +529 at 22.05/24 kHz and +38 at some
  32 kHz residues); AAC 256k/128k 0–896 extra frames, zero lag in every
  case (2026-10-11).
- A lossy candidate's probe must show the C-04 codec, rate and channels,
  otherwise ``encoder_settings_mismatch`` (R7).
- :func:`null_residual` subtracts ``source x gain`` from a linear candidate in
  double precision inside FFmpeg and reports the residual's peak and RMS
  (TST-A-10); it needs no runtime dependency beyond FFmpeg.

- The bounded retry ladder (ADR-0014, audio § 5.2): :func:`next_step` applies
  V3's seven steps in order to one verified candidate and returns the
  acceptance, the one retry plan or the terminal failure; :func:`run_ladder`
  drives it with caller callbacks, at most five renders, every one from the
  original source. Budgets live in the immutable C-05 plans (two loudness
  corrections, one peak correction, one reduce fallback) and never reset.
  The V3a planning states before each retry belong to the caller.
- Explicit ``dynamic`` (V3 steps 2 and 7, V6): one peak correction lowers
  the dynamic ceiling by ``o + 0.2`` and needs a fresh calibration of the
  original source at that ceiling, supplied by the caller's ``calibrate``
  callback (not a render); a ceiling outside loudnorm's range fails
  ``verification_peak``. A dynamic loudness miss fails, never corrected or
  switched to another mode.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from norma.adapters import ffmpeg
from norma.audio import measure as meter
from norma.audio.plan import (
    FALLBACK_NOTE,
    ORDINARY_ACTIONS,
    DynamicCalibration,
    dynamic_peak_ceiling,
    reduce_fallback,
    replan_dynamic_peak,
    replan_loudness,
    replan_peak,
)
from norma.audio.render import preserves_samples
from norma.errors import FileFailure, SkipFile, ToolError
from norma.models import (
    LOSSLESS_FORMATS,
    Diagnostic,
    GainPlan,
    LoudnessMeasurement,
    ProbeResult,
    ProcessingMark,
    RenderResult,
    RunOptions,
    TargetFormat,
    Verification,
    VerificationReference,
)

STAGE = "verify"
PEAK_ALLOWANCE_DB = 0.1
"""Output true peak may reach ceiling + 0.1 dB, judged on M7's conservative level (T1)."""

NULL_PEAK_LIMIT_DBFS = -90.0
"""TESTING § 2: linear null residual for 24-bit and float outputs, < -90 dBFS peak."""
NULL_RMS_LIMIT_DBFS = -84.0
"""TESTING § 2: linear null residual for 16-bit outputs, <= -84 dBFS RMS (dither level)."""


# --- Read-only reference selection (V9) ---------------------------------------------------------


def select_reference(options: RunOptions, mark: ProcessingMark) -> VerificationReference:
    """Select V9's immutable reference from resolved C-10 values and a WS-06 parsed mark.

    Only a CLI target outranks a valid mark. Ceiling and tolerance keep their
    resolved values and origins; missing value-source entries mean default.
    The mark is retained verbatim, including invalid-mark diagnostics. This
    function neither parses tags nor changes the run options.
    """
    target = options.target
    origin: Literal["cli", "mark", "config", "default"] = options.value_sources.get("target", "default")
    if origin != "cli" and mark.validity == "valid":
        if mark.target_lufs is None:
            raise ValueError("a valid parsed mark must provide a target")
        target, origin = mark.target_lufs, "mark"
    return VerificationReference(
        target_lufs=target, target_origin=origin,
        ceiling_dbtp=options.true_peak, ceiling_origin=options.value_sources.get("true_peak", "default"),
        tolerance_lu=options.tolerance, tolerance_origin=options.value_sources.get("tolerance", "default"),
        mark=mark,
    )


def assess_inspection(measurement: LoudnessMeasurement, reference: VerificationReference) -> Verification:
    """Pure V9 verdict on unchanged audio; use ``verification_failure`` for the primary failure.

    Ordinary audio expects the reference target. A valid reduce mark instead
    permits any measured loudness up to target + tolerance; it has no predicted
    output loudness or prediction error. Every measurable row carries its
    fresh shortfall and printed peak excess. Accepted reduce shortfall warns;
    invalid-mark diagnostics survive even a CLI override. No retry is offered.
    """
    target, ceiling = reference.target_lufs, reference.ceiling_dbtp
    reduce = reference.mark.validity == "valid" and reference.mark.mode == "reduce"
    measurable = measurement.signal_state == "measurable"
    integrated = measurement.integrated_lufs if measurable else None
    bound = meter.conservative_peak_dbtp(measurement.true_peak_linear, measurement.peak_resolution_linear)
    peak_ok = bound is not None and bound <= ceiling + PEAK_ALLOWANCE_DB
    printed = measurement.true_peak_dbtp
    peak_excess = printed - ceiling if printed is not None and math.isfinite(printed) else None
    diagnostics = list(reference.mark.diagnostics)
    if (reference.mark.validity in ("malformed", "unsupported")
            and not any(d.code == "processing_mark_invalid" for d in diagnostics)):
        diagnostics.append(Diagnostic("processing_mark_invalid",
                                      f"processing mark is {reference.mark.validity}", STAGE))
    expected = None if reduce else target
    error = shortfall = None
    loudness_ok = False
    if integrated is None:
        diagnostics.append(Diagnostic("verification_loudness",
                                      f"input is {measurement.signal_state}: no valid integrated loudness", STAGE))
    else:
        shortfall = max(0.0, target - integrated)
        error = None if reduce else integrated - target
        loudness_ok = (integrated <= target + reference.tolerance_lu if reduce
                       else abs(integrated - target) <= reference.tolerance_lu)
        if not loudness_ok:
            limit = (f"above {target + reference.tolerance_lu:g} LUFS for marked reduce" if reduce
                     else f"outside {target:g} ± {reference.tolerance_lu:g} LU")
            diagnostics.append(Diagnostic("verification_loudness", f"input measures {integrated:.3f} LUFS, {limit}",
                                          STAGE))
    if not peak_ok:
        why = ("peak resolution unknown: the input cannot be shown peak-safe" if bound is None
               else f"conservative true peak {bound:.4f} dBTP exceeds {ceiling + PEAK_ALLOWANCE_DB:g} dBTP")
        diagnostics.append(Diagnostic("verification_peak", why, STAGE))
    if reduce and loudness_ok and peak_ok and shortfall is not None and shortfall > 0:
        diagnostics.append(Diagnostic("reduce_shortfall", f"reduce output measures {shortfall:.3f} LU below the target",
                                      STAGE))
    return Verification(
        measurement=measurement, expected_lufs=expected, loudness_error_lu=error,
        measured_shortfall_lu=shortfall, peak_excess_db=peak_excess, loudness_ok=loudness_ok,
        peak_ok=peak_ok, measurable=measurable, diagnostics=tuple(diagnostics),
    )


# --- Candidate measurement and acceptance -------------------------------------------------------


def measure_candidate(
    candidate: Path, probe: ProbeResult, executable: ffmpeg.Executable, *,
    no_dual_mono: bool = False, cancel_event: threading.Event | None = None,
) -> LoudnessMeasurement:
    """Measure the entire decoded candidate with § 2's meter and the run's mono policy (V1, M3).

    ``probe`` is the candidate's own C-02 probe. Invalid meter output becomes
    ``verification_loudness`` (V1); other failures keep their reason.
    """
    try:
        return meter.measure(candidate, probe, executable, no_dual_mono=no_dual_mono, cancel_event=cancel_event)
    except FileFailure as exc:
        reason = "verification_loudness" if exc.reason == "measurement_error" else exc.reason
        raise FileFailure(reason, STAGE, exc.detail, exc.diagnostics + (exc.diagnostic(),)) from None
    except SkipFile as exc:
        raise FileFailure("verification_loudness", STAGE, f"candidate cannot be measured: {exc}",
                          (exc.diagnostic(),)) from None


def assess(measurement: LoudnessMeasurement, plan: GainPlan) -> Verification:
    """Pure V1 verdict on one measured candidate against the plan it was rendered from."""
    if plan.action == "skip":
        raise ValueError("a skip plan has no candidate to verify")
    target, ceiling = plan.target_lufs, plan.ceiling_dbtp
    measurable = measurement.signal_state == "measurable"
    integrated = measurement.integrated_lufs if measurable else None
    bound = meter.conservative_peak_dbtp(measurement.true_peak_linear, measurement.peak_resolution_linear)
    peak_ok = bound is not None and bound <= ceiling + PEAK_ALLOWANCE_DB
    printed = measurement.true_peak_dbtp
    peak_excess = printed - ceiling if printed is not None and math.isfinite(printed) else None

    diagnostics: list[Diagnostic] = []
    error = shortfall = None
    loudness_ok = False
    if integrated is None:
        diagnostics.append(Diagnostic("verification_loudness",
                                      f"candidate is {measurement.signal_state}: no valid integrated loudness",
                                      STAGE))
    else:
        shortfall = max(0.0, target - integrated)
        if plan.expected_integrated_lufs is not None:
            error = integrated - plan.expected_integrated_lufs
        loudness_ok = plan.action == "reduce" or abs(integrated - target) <= plan.tolerance_lu
        if not loudness_ok:
            diagnostics.append(Diagnostic(
                "verification_loudness",
                f"candidate measures {integrated:.3f} LUFS, outside {target:g} ± {plan.tolerance_lu:g} LU", STAGE))
    if not peak_ok:
        why = ("peak resolution unknown: the candidate cannot be shown peak-safe" if bound is None
               else f"conservative true peak {bound:.4f} dBTP exceeds {ceiling + PEAK_ALLOWANCE_DB:g} dBTP")
        diagnostics.append(Diagnostic("verification_peak", why, STAGE))
    return Verification(
        measurement=measurement, expected_lufs=plan.expected_integrated_lufs, loudness_error_lu=error,
        measured_shortfall_lu=shortfall, peak_excess_db=peak_excess, loudness_ok=loudness_ok,
        peak_ok=peak_ok, measurable=measurable, diagnostics=tuple(diagnostics),
    )


def verification_failure(verification: Verification) -> FileFailure | None:
    """The terminal C-13 failure for a rejected candidate, or None when it is accepted.

    Order per V3: an unmeasurable candidate first, then peak, then loudness.
    """
    by_code = {d.code: d for d in verification.diagnostics}
    if not verification.measurable:
        return FileFailure("verification_loudness", STAGE, "candidate has no valid integrated loudness",
                           verification.diagnostics)
    if not verification.peak_ok:
        detail = by_code["verification_peak"].message if "verification_peak" in by_code else None
        return FileFailure("verification_peak", STAGE, detail, verification.diagnostics)
    if not verification.loudness_ok:
        detail = by_code["verification_loudness"].message if "verification_loudness" in by_code else None
        return FileFailure("verification_loudness", STAGE, detail, verification.diagnostics)
    return None


TRAILING_PADDING_FRAMES: dict[str, int] = {"aac": 1023}
"""CR-059: decoded trailing padding an encode may add, per lossy C-04 format.

AAC-LC codes 1,024-sample frames; the M4A edit list removes priming, and at
most one frame minus one sample of end padding can remain. MP3 is absent:
its Xing/LAME tag, rewritten by the render from the encoder's own count
(CR-063), makes the decoded length exact, so the lossless rule holds.
"""


def trailing_padding_allowance(target: TargetFormat) -> int:
    """CR-059: frames of trailing codec padding a decoded candidate of ``target`` may carry (0 unless encoded AAC)."""
    if target.stream_copy or target.format in LOSSLESS_FORMATS:
        return 0
    return TRAILING_PADDING_FRAMES.get(target.format, 0)


def check_candidate_length(source_frames: int, source_rate_hz: int, measurement: LoudnessMeasurement, *,
                           trailing_padding: int = 0) -> None:
    """Pure length rule (IC-2 F-3, CR-059): raise ``render_error`` unless the candidate kept the whole programme.

    Same rate: decoded frame counts must be equal. Different rate: the
    candidate may differ from ``source_frames x rate_out / rate_in`` by at most
    one sample per second of source audio (at least one sample), the TESTING
    § 2 resampled-length tolerance. Integer arithmetic, no rounding.

    ``trailing_padding`` (:func:`trailing_padding_allowance`) widens only the
    upper bound, by that many output frames, for a lossy encode's end
    padding: a short candidate is never accepted, and leading priming that an
    encode failed to remove exceeds it. Zero keeps the lossless rule exactly.
    """
    if source_frames < 0 or source_rate_hz <= 0 or trailing_padding < 0:
        raise ValueError("source frames and padding must be non-negative and the source rate positive")
    frames, rate = measurement.decoded_frames, measurement.sample_rate_hz
    if frames is None:
        raise FileFailure("render_error", STAGE,
                          "candidate's decoded frame count is unknown; its length cannot be shown")
    padding = f" plus up to {trailing_padding} frames of codec padding" if trailing_padding else ""
    if rate == source_rate_hz:
        if not 0 <= frames - source_frames <= trailing_padding:
            raise FileFailure("render_error", STAGE,
                              f"candidate decodes to {frames} frames, source to {source_frames}{padding} at {rate} Hz")
        return
    # -max(1, n/rate_in) <= frames - n*rate_out/rate_in <= max(1, n/rate_in) + padding, scaled by rate_in.
    tolerance = max(source_rate_hz, source_frames)
    difference = frames * source_rate_hz - source_frames * rate
    if not -tolerance <= difference <= tolerance + trailing_padding * source_rate_hz:
        expected = source_frames * rate / source_rate_hz
        raise FileFailure("render_error", STAGE,
                          f"candidate decodes to {frames} frames at {rate} Hz; {source_frames} source frames at "
                          f"{source_rate_hz} Hz expect {expected:.1f} ± 1 per second{padding}")


_PROBED_CODECS = {"mp3": "mp3", "aac": "aac"}


def check_encoded_stream(candidate_probe: ProbeResult, target: TargetFormat) -> None:
    """R7: a lossy encode's probe shows the C-04 codec, rate and channels, or ``encoder_settings_mismatch``."""
    if target.stream_copy or target.format in LOSSLESS_FORMATS:
        return
    expected = (_PROBED_CODECS[target.format], target.sample_rate_hz, target.channels)
    written = (candidate_probe.codec, candidate_probe.sample_rate_hz, candidate_probe.channels)
    if written != expected:
        raise FileFailure("encoder_settings_mismatch", STAGE,
                          f"encoded candidate probes as {written}, C-04 expects {expected}")


def verify_candidate(
    result: RenderResult, source: Path, source_probe: ProbeResult, candidate_probe: ProbeResult,
    executable: ffmpeg.Executable, *, no_dual_mono: bool = False, cancel_event: threading.Event | None = None,
    source_frames: int | None = None,
) -> Verification:
    """Measure and judge one rendered candidate: whole length, and exact samples for a preserving copy.

    Returns the C-05 Verification (call :func:`verification_failure` for the
    verdict). A candidate whose decoded length differs from the source's
    (:func:`check_candidate_length`), or a copy whose decoded samples differ,
    raises ``render_error``: the render did not reproduce its input.
    ``source_frames`` is the source's C-03 ``decoded_frames`` from pass 1;
    when omitted the source is counted again with the same scan.
    """
    check_encoded_stream(candidate_probe, result.target)
    measurement = measure_candidate(result.candidate_path, candidate_probe, executable,
                                    no_dual_mono=no_dual_mono, cancel_event=cancel_event)
    if source_frames is None:
        source_frames = _source_frames(source, source_probe, executable, cancel_event)
    check_candidate_length(source_frames, source_probe.sample_rate_hz, measurement,
                           trailing_padding=trailing_padding_allowance(result.target))
    verification = assess(measurement, result.plan)
    if result.plan.action == "copy" and preserves_samples(source_probe, result.target):
        check_copy_identity(source, result.candidate_path, executable,
                            duration_s=source_probe.duration_s, cancel_event=cancel_event)
    return verification


def _source_frames(source: Path, probe: ProbeResult, executable: ffmpeg.Executable,
                   cancel_event: threading.Event | None) -> int:
    try:
        return meter.count_decoded_frames(source, probe, executable, cancel_event=cancel_event)
    except FileFailure as exc:
        reason = "decode_error" if exc.reason == "measurement_error" else exc.reason
        raise FileFailure(reason, STAGE, f"source frames could not be recounted: {exc.detail}",
                          exc.diagnostics + (exc.diagnostic(),)) from None


# --- Bounded retry ladder (audio § 5.2, V2–V7) ---------------------------------------------------

MAX_RENDERS = 5
"""ADR-0014: initial + two loudness corrections + one reduce fallback + one peak correction."""

RetryKind = Literal["loudness_correction", "peak_correction", "reduce_fallback"]


@dataclass(frozen=True)
class Accepted:
    """V3 steps 3–4: the candidate rendered from ``plan`` is accepted; ``warnings`` join its outcome."""

    plan: GainPlan
    verification: Verification
    warnings: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class Retry:
    """One retry: render ``plan`` from the original source after the caller's V3a planning states."""

    plan: GainPlan
    kind: RetryKind


def _fail(reason: Literal["verification_peak", "verification_loudness"], verification: Verification,
          why: str) -> FileFailure:
    base = verification_failure(verification)
    detail = base.detail if base is not None and base.reason == reason and base.detail else None
    return FileFailure(reason, STAGE, f"{detail}; {why}" if detail else why, verification.diagnostics)


def _accept_warnings(plan: GainPlan, verification: Verification) -> tuple[Diagnostic, ...]:
    """V3 step 3 / V10: a fallback always warns; an explicit reduce warns on a nonzero measured shortfall."""
    if plan.action != "reduce":
        return ()
    shortfall = verification.measured_shortfall_lu
    assert shortfall is not None
    if plan.fallback_from is not None:
        return (Diagnostic("reduce_shortfall", f"{FALLBACK_NOTE}; measured shortfall {shortfall:.3f} LU", STAGE),)
    if shortfall > 0:
        return (Diagnostic("reduce_shortfall", f"reduce output measures {shortfall:.3f} LU below the target", STAGE),)
    return ()


Calibrate = Callable[[float], DynamicCalibration]
"""Caller callback for V6: analyse the original source again at the given dynamic ceiling (not a render)."""


def next_step(source: LoudnessMeasurement, verification: Verification, plan: GainPlan, *,
              calibrate: Calibrate | None = None) -> Accepted | Retry | FileFailure:
    """V3's ordered steps for one verified candidate rendered from ``plan``.

    ``source`` is the original source's C-03 record (the plan's ``TP_in`` and
    ``I_in``). Returns the acceptance, the single next retry, or the terminal
    failure (returned, not raised). Step 1's own failures — decode, length,
    deadline, I/O — are raised earlier by render and :func:`verify_candidate`.
    Pure except for ``calibrate``, which only a dynamic peak correction calls
    (V6); a calibration failure it raises ends the file.
    """
    if plan.action == "skip":
        raise ValueError("a skip plan has no candidate")
    ordinary = plan.action in ORDINARY_ACTIONS
    # Step 1: no correction is inferred from missing numbers.
    if not verification.measurable:
        failure = verification_failure(verification)
        assert failure is not None
        return failure
    # Step 2: peak first; never correct loudness while the peak is unresolved.
    if not verification.peak_ok:
        measurement = verification.measurement
        bound = meter.conservative_peak_dbtp(measurement.true_peak_linear, measurement.peak_resolution_linear)
        if bound is None:
            return _fail("verification_peak", verification, "a correction cannot make an unknown peak provably safe")
        if plan.action == "reduce":
            return _fail("verification_peak", verification, "a reduce candidate has no limiter to lower (V6)")
        if plan.peak_corrections >= 1:
            return _fail("verification_peak", verification, "the one peak correction is already used")
        if plan.action == "dynamic":
            ceiling = dynamic_peak_ceiling(plan, bound)
            if ceiling is None:
                return _fail("verification_peak", verification,
                             "the lowered dynamic ceiling is outside loudnorm's TP range; it is not clamped (V6)")
            if calibrate is None:
                raise ValueError("a dynamic peak correction needs a calibrate callback (V6)")
            calibration = calibrate(ceiling)
            if calibration.ceiling_dbtp != ceiling:
                raise ValueError("calibrate must analyse at the requested dynamic ceiling")
            return Retry(replan_dynamic_peak(plan, calibration), "peak_correction")
        if not ordinary:
            return _fail("verification_peak", verification, f"no peak correction for {plan.action}")
        corrected = replan_peak(source, plan, bound)
        if corrected is None:
            return _fail("verification_peak", verification,
                         "a peak correction is not eligible under the peak policy, cap or limiter range")
        return Retry(corrected, "peak_correction")
    # Step 3: pure reduce finishes with its measured shortfall.
    if plan.action == "reduce":
        return Accepted(plan, verification, _accept_warnings(plan, verification))
    # Step 4: ordinary acceptance.
    if verification.loudness_ok:
        return Accepted(plan, verification)
    # Step 7: explicit dynamic receives no ordinary correction or mode switch.
    if not ordinary:
        return _fail("verification_loudness", verification, f"{plan.action} receives no loudness correction")
    # Step 5: a cap-safe loudness correction while budget remains.
    integrated = verification.measurement.integrated_lufs
    assert integrated is not None
    if plan.loudness_corrections < 2:
        corrected = replan_loudness(source, plan, integrated)
        if corrected is not None:
            return Retry(corrected, "loudness_correction")
    # Step 6: the sole reduce fallback.
    if plan.reduce_fallbacks < 1:
        return Retry(reduce_fallback(source, plan), "reduce_fallback")
    return _fail("verification_loudness", verification, "the reduce fallback is already used")


Attempt = Callable[[GainPlan, int], "tuple[RenderResult, Verification]"]
"""Caller callback: allocate a fresh candidate, render ``plan`` from the original source as attempt n, verify it."""
BeforeRetry = Callable[[GainPlan, GainPlan, RetryKind], None]
"""Caller callback for V3a's ``gain_planned`` → ``metadata_planned`` → ``writes_reserved`` before a retry."""


@dataclass(frozen=True)
class LadderResult:
    """The accepted candidate and every attempt in order (``renders[-1]`` is the accepted one)."""

    plan: GainPlan
    result: RenderResult
    verification: Verification
    warnings: tuple[Diagnostic, ...]
    renders: tuple[RenderResult, ...]
    verifications: tuple[Verification, ...]


def run_ladder(source: LoudnessMeasurement, initial: GainPlan, attempt: Attempt,
               before_retry: BeforeRetry | None = None, *, calibrate: Calibrate | None = None) -> LadderResult:
    """Drive :func:`next_step` from ``initial`` to acceptance, or raise the terminal :class:`FileFailure`.

    ``attempt(plan, n)`` must render from the original source and return the
    RenderResult (attempt n, the same plan) with its Verification; a failure
    it raises ends the file unchanged (V3 step 1). ``before_retry`` runs the
    V3a planning states; a failure there ends the file with that state's
    reason and consumes no render. A failed candidate never becomes a source.
    ``calibrate`` serves an explicit dynamic plan's peak correction (V6).
    """
    plan = initial
    renders: list[RenderResult] = []
    checks: list[Verification] = []
    while True:
        number = len(renders) + 1
        if number > MAX_RENDERS:  # unreachable while every retry consumes a bounded counter
            raise RuntimeError(f"retry ladder exceeded {MAX_RENDERS} renders (ADR-0014)")
        result, verification = attempt(plan, number)
        if result.attempt != number or result.plan != plan:
            raise ValueError("attempt must return the render of the requested plan and attempt number")
        renders.append(result)
        checks.append(verification)
        step = next_step(source, verification, plan, calibrate=calibrate)
        if isinstance(step, FileFailure):
            raise step
        if isinstance(step, Accepted):
            return LadderResult(step.plan, result, verification, step.warnings, tuple(renders), tuple(checks))
        if before_retry is not None:
            before_retry(plan, step.plan, step.kind)
        plan = step.plan


# --- Decoded-sample identity for copies ---------------------------------------------------------

_DIGEST = re.compile(rb"^SHA256=([0-9a-f]{64})\s*$", re.MULTILINE)


def decoded_digest(path: Path, executable: ffmpeg.Executable, *, duration_s: float | None = None,
                   cancel_event: threading.Event | None = None) -> str:
    """SHA-256 of the first audio stream decoded to interleaved float64 (exact for <= 32-bit PCM and float)."""
    argv = ffmpeg.ffmpeg_argv(executable, "-xerror", "-nostats", *ffmpeg.input_args(path), "-map", "0:a:0",
                              "-c:a", "pcm_f64le", "-f", "hash", "-hash", "sha256", "pipe:1")
    try:
        result = ffmpeg.check_result(ffmpeg.run_process(
            argv, executable=executable, deadline_s=ffmpeg.tool_deadline(duration_s),
            stdout_limit=4096, cancel_event=cancel_event))
    except ToolError as exc:
        raise exc.to_failure("decode_error", STAGE) from None
    match = _DIGEST.search(result.stdout)
    if match is None:
        raise FileFailure("decode_error", STAGE, f"no decoded-sample digest for {path}")
    return match.group(1).decode("ascii")


def check_copy_identity(source: Path, candidate: Path, executable: ffmpeg.Executable, *,
                        duration_s: float | None = None, cancel_event: threading.Event | None = None) -> str:
    """Prove a copy kept every decoded sample (F13, TST-F-02); returns the shared digest."""
    expected = decoded_digest(source, executable, duration_s=duration_s, cancel_event=cancel_event)
    actual = decoded_digest(candidate, executable, duration_s=duration_s, cancel_event=cancel_event)
    if actual != expected:
        raise FileFailure("render_error", STAGE,
                          f"copied candidate's decoded samples differ from the source ({actual[:12]} != {expected[:12]})")
    return actual


# --- Linear null test (TST-A-10) ---------------------------------------------------------------

_PREFIX = re.compile(r"^\[[^]]+\]\s*")
_NULL_KEYS = {
    "lavfi.astats.Overall.Peak_level": "peak",
    "lavfi.astats.Overall.RMS_level": "rms",
    "lavfi.astats.Overall.Number_of_samples": "frames",
}


@dataclass(frozen=True)
class NullResidual:
    """Residual of ``candidate - source x gain``; dBFS values may be ``-inf`` for an exact null."""

    frames: int
    """Frames per channel compared; FFmpeg stops at the shorter input, so compare with both lengths."""
    peak_dbfs: float
    rms_dbfs: float

    def passes(self, target: TargetFormat, *, expected_frames: int) -> bool:
        """TESTING § 2 limits: 16-bit integer RMS <= -84 dBFS, otherwise peak < -90 dBFS."""
        if self.frames != expected_frames:
            return False
        if not target.sample_float and target.bit_depth is not None and target.bit_depth <= 16:
            return self.rms_dbfs <= NULL_RMS_LIMIT_DBFS
        return self.peak_dbfs < NULL_PEAK_LIMIT_DBFS


def null_residual_argv(executable: ffmpeg.Executable | Path, source: Path, candidate: Path, gain_db: float,
                       channels: int) -> tuple[str, ...]:
    """Pure argv: both inputs to float64, source scaled by the gain, subtracted per channel."""
    if channels not in (1, 2):
        raise ValueError("v1 null test supports mono and stereo only")
    if not math.isfinite(gain_db):
        raise ValueError("gain must be finite")
    exprs = "|".join(f"val({index})-val({index + channels})" for index in range(channels))
    layout = "mono" if channels == 1 else "stereo"
    graph = (
        "[0:a:0]aformat=sample_fmts=dbl[cand];"
        f"[1:a:0]aformat=sample_fmts=dbl,volume={float(gain_db)!r}dB:precision=double[ref];"
        f"[cand][ref]amerge=inputs=2,aeval=exprs={exprs}:c={layout},"
        "astats=metadata=1:reset=0:measure_perchannel=none:"
        "measure_overall=Peak_level+RMS_level+Number_of_samples,ametadata=mode=print"
    )
    return ffmpeg.ffmpeg_argv(executable, "-xerror", "-nostats", *ffmpeg.input_args(candidate),
                              *ffmpeg.input_args(source), "-filter_complex", graph, "-f", "null", "-")


def null_residual(source: Path, candidate: Path, gain_db: float, executable: ffmpeg.Executable, *,
                  channels: int, duration_s: float | None = None,
                  cancel_event: threading.Event | None = None) -> NullResidual:
    """Measure ``candidate - source x 10**(gain/20)`` without resampling (same rate required)."""
    values: dict[str, str] = {}

    def feed(line: str) -> None:
        text = _PREFIX.sub("", line).strip()
        key, _, value = text.partition("=")
        if key in _NULL_KEYS:
            values[_NULL_KEYS[key]] = value

    argv = null_residual_argv(executable, source, candidate, gain_db, channels)
    try:
        ffmpeg.check_result(ffmpeg.run_process(
            argv, executable=executable, deadline_s=ffmpeg.tool_deadline(duration_s), capture_stdout=False,
            on_stderr_line=feed, cancel_event=cancel_event))
    except ToolError as exc:
        raise exc.to_failure("decode_error", STAGE) from None
    try:
        return NullResidual(int(float(values["frames"])), float(values["peak"]), float(values["rms"]))
    except (KeyError, ValueError):
        raise FileFailure("decode_error", STAGE, "null residual metadata missing or invalid") from None
