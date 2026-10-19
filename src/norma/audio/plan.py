"""Pure gain planning and retry replanning (C-03/C-05/C-10; audio P1–P4, R4, V4–V6).

:func:`plan_gain` makes the initial plan. The three replanning functions
make the immutable successor plan for one retry of the bounded ladder
(ADR-0014, audio § 5.2); ``audio/verify.py`` decides which one applies:

- :func:`replan_loudness` — V3 step 5 with V4's cap check: ``g' = g + (t − I_out)``;
  an already required limiter and its lowered threshold are kept.
- :func:`replan_peak` — V5: lower the limiter threshold by ``o + 0.2`` dB with
  ``o = TP_out,up − c`` (M7's conservative level), at unchanged gain.
- :func:`reduce_fallback` — V3 step 6: pure ``reduce`` at gain ``c − TP_in``.

Peak terms in predictions and cap checks use the printed ``TP_in`` (P2 peak
terms); only the copy row uses M7's bound. The accumulated peak adjustment
is not a C-05 field: it is recovered from ``limiter_threshold_dbtp``
(``L = c − 0.3 − adjustment``, R4). Counters only ever increase, and a
rejected proposal returns None and consumes nothing (V4).

Explicit ``dynamic`` (P2 row 2, R5, R6, V6) is planned only with the source's
loudnorm calibration (:class:`DynamicCalibration`, made by
``audio/measure.calibrate_dynamic``); :func:`needs_dynamic_calibration` tells
the caller when one is required. It is never selected implicitly.
:func:`dynamic_peak_ceiling` and :func:`replan_dynamic_peak` make V6's one
dynamic peak correction.

The read-only commands (``analyse``, ``normalise --dry-run``) run no
calibration (audio V8); for an explicit ``dynamic`` policy the pipeline
shows :func:`plan_dynamic_preview`, C-05 v1.3's preview plan (D-080,
CR-037), which is never rendered.
The S2 preflight refusal of ``dynamic`` is retired (CR-062, CR-068).

A copy action is not a stream-copy decision or a terminal skip: the caller
resolves format/metadata/path work (P4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from norma.models import (
    DYNAMIC_CALIBRATION_KEYS,
    Diagnostic,
    GainAction,
    GainPlan,
    LoudnessMeasurement,
    ReasonCode,
    RunOptions,
)

LIMITER_MARGIN_DB = 0.3
"""R4's fixed initial safety margin below the ceiling; never counted against ``max_limit``."""
PEAK_CORRECTION_MARGIN_DB = 0.2
"""V5/V6: a peak correction lowers the threshold by the overshoot plus this margin."""
ALIMITER_LIMIT_RANGE: tuple[float, float] = (0.0625, 1.0)
"""The installed ``alimiter`` filter's supported linear ``limit`` range (audio § 4.2)."""
FALLBACK_NOTE = "limit could not reach target"
"""D-015 / V3 step 3: the note every ordinary reduce fallback carries."""

ORDINARY_ACTIONS: frozenset[str] = frozenset({"copy", "linear", "limit"})
"""Actions that receive loudness corrections, a peak correction and the reduce fallback (V3)."""

# --- loudnorm option ranges (installed ``ffmpeg -h filter=loudnorm``, N-125892, 2026-10-11) -----

LOUDNORM_TARGET_RANGE: tuple[float, float] = (-70.0, -5.0)
"""``I``: a target outside it fails; it is never clamped."""
LOUDNORM_TP_RANGE: tuple[float, float] = (-9.0, 0.0)
"""``TP``: the dynamic ceiling, initially c and lowered by V6; outside it fails ``verification_peak``."""
DYNAMIC_LRA = 50
"""R5: ``LRA=50`` so loudnorm's range target never compresses programme dynamics on its own."""
LOUDNORM_MEASURED_OPTIONS: Mapping[str, tuple[str, tuple[float, float]]] = {
    # C-05 key: (loudnorm render option, its supported range)
    "input_i": ("measured_I", (-99.0, 0.0)),
    "input_tp": ("measured_TP", (-99.0, 99.0)),
    "input_lra": ("measured_LRA", (0.0, 99.0)),
    "input_thresh": ("measured_thresh", (-99.0, 0.0)),
    "target_offset": ("offset", (-99.0, 99.0)),
}
"""R5: how each C-05 ``dynamic_parameters`` value reaches the calibrated render."""


def in_range(value: float, bounds: tuple[float, float]) -> bool:
    return math.isfinite(value) and bounds[0] <= value <= bounds[1]


@dataclass(frozen=True)
class DynamicCalibration:
    """One loudnorm analysis of the original source (R5): renderer calibration, not a measurement of record."""

    parameters: Mapping[str, float]
    """Exactly :data:`~norma.models.DYNAMIC_CALIBRATION_KEYS`, finite (C-05 ``dynamic_parameters``)."""
    normalization_type: Literal["dynamic", "linear"]
    target_lufs: float
    ceiling_dbtp: float
    """The ``TP`` the analysis ran at: the plan's dynamic ceiling."""

    def __post_init__(self) -> None:
        if set(self.parameters) != set(DYNAMIC_CALIBRATION_KEYS) or len(self.parameters) != len(DYNAMIC_CALIBRATION_KEYS):
            raise ValueError(f"calibration must hold exactly {DYNAMIC_CALIBRATION_KEYS}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.parameters.values()):
            raise ValueError("calibration values must be finite")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _already_peak_safe(measurement: LoudnessMeasurement, ceiling: float) -> bool:
    # Do not import measure.py: its adapter dependencies are outside the pure
    # domain. M7's upper bound is used only for the copy row, not predictions.
    amplitude = measurement.true_peak_linear
    resolution = measurement.peak_resolution_linear
    if (amplitude is None or resolution is None or not math.isfinite(resolution)
            or resolution <= 0):
        return False
    upper = amplitude + resolution / 2
    return (math.isfinite(upper) and upper > 0
            and 20 * math.log10(upper) <= ceiling)


def _copy_row(measurement: LoudnessMeasurement, options: RunOptions) -> bool:
    """P2's first row: within ``skip_within`` and already peak-safe under M7 (never with unknown q)."""
    assert measurement.integrated_lufs is not None
    gain = options.target - measurement.integrated_lufs
    return (options.skip_within > 0 and abs(gain) <= options.skip_within
            and _already_peak_safe(measurement, options.true_peak))


def needs_dynamic_calibration(measurement: LoudnessMeasurement, options: RunOptions) -> bool:
    """True when :func:`plan_gain` will select ``dynamic`` and so needs the source's calibration.

    Only an explicit ``dynamic`` policy, only for a measurable source, and
    not when P2's copy row applies first. The calibration runs at ceiling c.
    """
    return (options.peak_mode == "dynamic" and measurement.signal_state == "measurable"
            and not _copy_row(measurement, options))


def plan_gain(measurement: LoudnessMeasurement, options: RunOptions, *,
              calibration: DynamicCalibration | None = None) -> GainPlan:
    """Return the initial immutable plan using validated effective options.

    Loudness already embodies the meter's dual-mono setting; never apply a
    second mono adjustment here. The caller must measure with the same
    no_dual_mono option. Unknown peak resolution forbids an already-safe copy
    but does not replace central-value plan predictions with safety bounds.
    Expected loudness/peak and shortfall are predictions, not verified output.
    Silent/unmeasurable records yield skip plans with no invented numbers.

    Explicit ``dynamic`` (P2 row 2) needs ``calibration`` from the original
    source at this target and ceiling whenever :func:`needs_dynamic_calibration`
    is true; without it this raises ValueError (a caller error: the read-only
    commands use :func:`plan_dynamic_preview` instead, audio V8). A
    calibration for any other policy, target or ceiling is refused too.
    """
    action: GainAction = "skip"
    reason: ReasonCode = "silent" if measurement.signal_state == "silent" else "unmeasurable"
    gain = predicted_peak = expected = shortfall = threshold = None
    policy_bound = effective_bound = 0.0
    diagnostics = measurement.diagnostics

    if calibration is not None:
        if options.peak_mode != "dynamic":
            raise ValueError("a loudnorm calibration applies to explicit dynamic mode only (R5)")
        if (calibration.target_lufs, calibration.ceiling_dbtp) != (options.target, options.true_peak):
            raise ValueError("the calibration was made for another target or ceiling (R5)")

    if measurement.signal_state == "measurable":
        integrated = measurement.integrated_lufs
        peak = measurement.true_peak_dbtp
        # C-03 guarantees finite I/TP for measurable records.
        assert integrated is not None and peak is not None
        gain = options.target - integrated
        predicted_peak = peak + gain
        planned_reduction = max(0.0, predicted_peak - options.true_peak)
        reason = "ok"
        if _copy_row(measurement, options):
            action, reason, gain = "copy", "already_within_tolerance", 0.0
        elif options.peak_mode == "dynamic":
            if calibration is None:
                raise ValueError("an explicit dynamic plan needs the source's loudnorm calibration (R5)")
            return GainPlan(
                action="dynamic", requested_peak_mode="dynamic", target_lufs=options.target,
                ceiling_dbtp=options.true_peak, tolerance_lu=options.tolerance, max_limit_db=options.max_limit,
                gain_db=gain, predicted_true_peak_dbtp=None, limiter_reduction_db=0.0,
                effective_limiter_bound_db=0.0, limiter_threshold_dbtp=calibration.ceiling_dbtp,
                expected_integrated_lufs=options.target, shortfall_lu=0.0, fallback_from=None,
                loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0, reason_code="ok",
                diagnostics=diagnostics, dynamic_parameters=calibration.parameters,
            )
        elif predicted_peak <= options.true_peak:
            action = "linear"
        elif (options.peak_mode == "limit" and options.max_limit > 0
                and planned_reduction <= options.max_limit):
            action = "limit"
            threshold = options.true_peak - LIMITER_MARGIN_DB
            policy_bound = planned_reduction
            effective_bound = max(0.0, predicted_peak - threshold)
        else:
            action = "reduce"
            gain = options.true_peak - peak

        predicted_peak = peak + gain
        expected = integrated + gain
        shortfall = max(0.0, options.target - expected)
        if action == "reduce" and shortfall > 0:
            reason = "reduce_shortfall"
            diagnostics += (Diagnostic(
                "reduce_shortfall", "predicted reduce output is below the requested target", "plan",
            ),)

    return GainPlan(
        action=action, requested_peak_mode=options.peak_mode,
        target_lufs=options.target, ceiling_dbtp=options.true_peak,
        tolerance_lu=options.tolerance, max_limit_db=options.max_limit,
        gain_db=gain, predicted_true_peak_dbtp=predicted_peak,
        limiter_reduction_db=policy_bound, effective_limiter_bound_db=effective_bound,
        limiter_threshold_dbtp=threshold, expected_integrated_lufs=expected,
        shortfall_lu=shortfall, fallback_from=None, loudness_corrections=0,
        peak_corrections=0, reduce_fallbacks=0, reason_code=reason,
        diagnostics=diagnostics,
    )


def plan_dynamic_preview(measurement: LoudnessMeasurement, options: RunOptions) -> GainPlan:
    """The read-only plan an explicit ``dynamic`` policy shows in ``analyse`` and dry run (C-05 v1.3, D-080).

    It shows what ``normalise`` would plan without running its calibration
    (audio V8). Where ``normalise`` would need none — a silent or
    unmeasurable source (skip, no invented numbers) or P2's copy row, which
    precedes the dynamic row — it is exactly :func:`plan_gain`'s plan.
    Otherwise it is the preview: ``action=dynamic`` with
    ``dynamic_parameters=None``, nominal gain ``t − I_in``, expected loudness
    ``t``, predicted peak unknown, both limiter bounds 0, the dynamic ceiling
    ``c`` as ``limiter_threshold_dbtp``, shortfall 0 and no retry budget
    used. The interfaces label it "dynamic (calibrated at render)"; the
    gain and loudness are nominal, and loudnorm's actual result is known
    only after a calibrated render is verified.

    A preview is never rendered (C-05 ``RenderResult``, audio R5): a render
    needs :func:`plan_gain` with the source's calibration. Any other policy
    is a caller error (ValueError); use :func:`plan_gain` for it.
    """
    if options.peak_mode != "dynamic":
        raise ValueError("a dynamic preview applies to an explicit dynamic policy only (C-05 v1.3)")
    if not needs_dynamic_calibration(measurement, options):
        return plan_gain(measurement, options)
    integrated = measurement.integrated_lufs
    assert integrated is not None  # measurable: C-03 guarantees a finite I
    return GainPlan(
        action="dynamic", requested_peak_mode="dynamic", target_lufs=options.target,
        ceiling_dbtp=options.true_peak, tolerance_lu=options.tolerance, max_limit_db=options.max_limit,
        gain_db=options.target - integrated, predicted_true_peak_dbtp=None, limiter_reduction_db=0.0,
        effective_limiter_bound_db=0.0, limiter_threshold_dbtp=options.true_peak,
        expected_integrated_lufs=options.target, shortfall_lu=0.0, fallback_from=None,
        loudness_corrections=0, peak_corrections=0, reduce_fallbacks=0, reason_code="ok",
        diagnostics=measurement.diagnostics, dynamic_parameters=None,
    )


# --- Retry replanning (audio § 5.2, V3–V6) -----------------------------------------------------

RETRY_STAGE = "retry"
"""Stage of the notes the replanning functions add; a reduce fallback drops earlier ones."""


def peak_adjustment(plan: GainPlan) -> float:
    """The accumulated V5 threshold lowering of a ``limit`` plan; zero for every other action (R4)."""
    if plan.action != "limit" or plan.limiter_threshold_dbtp is None:
        return 0.0
    return max(0.0, plan.ceiling_dbtp - LIMITER_MARGIN_DB - plan.limiter_threshold_dbtp)


def limiter_threshold_supported(threshold_dbtp: float) -> bool:
    """True when ``10**(L/20)`` lies in alimiter's range; outside it a render fails, never clamps (R4)."""
    low, high = ALIMITER_LIMIT_RANGE
    return math.isfinite(threshold_dbtp) and low <= 10 ** (threshold_dbtp / 20) <= high


def _source_terms(source: LoudnessMeasurement) -> tuple[float, float]:
    if (source.signal_state != "measurable" or source.integrated_lufs is None
            or source.true_peak_dbtp is None or not math.isfinite(source.true_peak_dbtp)):
        raise ValueError("replanning needs the measurable source record the initial plan was made from")
    return source.integrated_lufs, source.true_peak_dbtp


def _require_ordinary(plan: GainPlan, what: str) -> None:
    if plan.action not in ORDINARY_ACTIONS:
        raise ValueError(f"a {plan.action} plan receives no {what} (V3)")
    if plan.gain_db is None:
        raise ValueError("an ordinary plan carries a gain")


def _ordinary_successor(
    plan: GainPlan, source: LoudnessMeasurement, gain: float, threshold: float, *, keep_limiter: bool,
    loudness_corrections: int, peak_corrections: int, note: Diagnostic | None = None,
) -> GainPlan | None:
    """V4: replan ``gain`` at limiter threshold ``threshold``; None when policy or cap forbid it.

    The threshold is passed, not recomputed, so a kept limiter keeps its
    exact lowered value; the adjustment is ``c − 0.3 − threshold`` (R4).
    """
    integrated, peak = _source_terms(source)
    ceiling = plan.ceiling_dbtp
    adjustment = max(0.0, ceiling - LIMITER_MARGIN_DB - threshold)
    predicted = peak + gain
    diagnostics = plan.diagnostics + ((note,) if note is not None else ())
    common = dict(
        gain_db=gain, predicted_true_peak_dbtp=predicted, expected_integrated_lufs=integrated + gain,
        shortfall_lu=max(0.0, plan.target_lufs - (integrated + gain)), fallback_from=None,
        loudness_corrections=loudness_corrections, peak_corrections=peak_corrections,
        reduce_fallbacks=plan.reduce_fallbacks, reason_code="ok", diagnostics=diagnostics,
    )
    if not keep_limiter and adjustment == 0 and predicted <= ceiling:
        return replace(plan, action="linear", limiter_reduction_db=0.0, effective_limiter_bound_db=0.0,
                       limiter_threshold_dbtp=None, **common)
    policy = max(0.0, predicted - ceiling + adjustment)
    if plan.requested_peak_mode != "limit" or plan.max_limit_db <= 0 or policy > plan.max_limit_db:
        return None  # V4: explicit reduce never gains a limiter, and the cap is never widened
    if not limiter_threshold_supported(threshold):
        return None
    return replace(plan, action="limit", limiter_reduction_db=policy,
                   effective_limiter_bound_db=max(0.0, predicted - threshold),
                   limiter_threshold_dbtp=threshold, **common)


def replan_loudness(source: LoudnessMeasurement, plan: GainPlan, measured_lufs: float) -> GainPlan | None:
    """V3 step 5: the next plan after a loudness miss, or None when V4 rejects the proposal.

    ``g' = g + (t − I_out)`` from the original source. A limiter the current
    plan already needs is kept with its lowered threshold; otherwise the plan
    is linear when its predicted peak fits, else limit when policy and cap
    allow. A rejected proposal consumes no correction (V4).
    """
    _require_ordinary(plan, "loudness correction")
    if plan.loudness_corrections >= 2:
        raise ValueError("both loudness corrections are already used (ADR-0014)")
    if not math.isfinite(measured_lufs):
        raise ValueError("a loudness correction needs a finite measured loudness")
    assert plan.gain_db is not None
    gain = plan.gain_db + (plan.target_lufs - measured_lufs)
    limited = plan.action == "limit" and plan.limiter_threshold_dbtp is not None
    threshold = plan.limiter_threshold_dbtp if limited else plan.ceiling_dbtp - LIMITER_MARGIN_DB
    assert threshold is not None
    return _ordinary_successor(plan, source, gain, threshold, keep_limiter=limited,
                               loudness_corrections=plan.loudness_corrections + 1,
                               peak_corrections=plan.peak_corrections)


def replan_peak(source: LoudnessMeasurement, plan: GainPlan, conservative_peak_dbtp: float) -> GainPlan | None:
    """V5: the next plan after an unsafe ordinary candidate, or None when no correction is eligible.

    ``conservative_peak_dbtp`` is the candidate's M7 level ``TP_out,up``. The
    threshold drops by ``o + 0.2`` dB, ``o = TP_out,up − c``, from R4's
    ``c − 0.3`` for a previously linear candidate. Gain is unchanged; the cap
    is rechecked with the new adjustment. Only policy ``limit`` has a limiter
    to lower; an out-of-range threshold is ineligible, never clamped.
    """
    _require_ordinary(plan, "peak correction")
    if plan.peak_corrections >= 1:
        raise ValueError("the peak correction is already used (ADR-0014)")
    if not math.isfinite(conservative_peak_dbtp):
        raise ValueError("a peak correction needs a finite conservative peak level")
    assert plan.gain_db is not None
    overshoot = conservative_peak_dbtp - plan.ceiling_dbtp
    base = (plan.limiter_threshold_dbtp if plan.action == "limit" and plan.limiter_threshold_dbtp is not None
            else plan.ceiling_dbtp - LIMITER_MARGIN_DB)
    threshold = base - (overshoot + PEAK_CORRECTION_MARGIN_DB)
    note = Diagnostic("ok", f"limiter threshold lowered by {overshoot + PEAK_CORRECTION_MARGIN_DB:.4f} dB "
                            f"for peak safety (conservative output peak {conservative_peak_dbtp:.4f} dBTP)",
                      RETRY_STAGE)
    return _ordinary_successor(plan, source, plan.gain_db, threshold, keep_limiter=True,
                               loudness_corrections=plan.loudness_corrections, peak_corrections=1, note=note)


def reduce_fallback(source: LoudnessMeasurement, plan: GainPlan) -> GainPlan:
    """V3 step 6: the sole pure-``reduce`` fallback, gain ``c − TP_in``, recording ``fallback_from``.

    It always carries ``reduce_shortfall`` with D-015's note, even when the
    predicted shortfall is zero, and never returns to target-seeking.
    """
    _require_ordinary(plan, "reduce fallback")
    if plan.reduce_fallbacks >= 1:
        raise ValueError("the reduce fallback is already used (ADR-0014)")
    integrated, peak = _source_terms(source)
    gain = plan.ceiling_dbtp - peak
    expected = integrated + gain
    kept = tuple(d for d in plan.diagnostics if d.stage != RETRY_STAGE)
    note = Diagnostic("reduce_shortfall", FALLBACK_NOTE, RETRY_STAGE)
    action: Literal["copy", "linear", "limit"] = plan.action  # type: ignore[assignment]
    return replace(
        plan, action="reduce", gain_db=gain, predicted_true_peak_dbtp=peak + gain, limiter_reduction_db=0.0,
        effective_limiter_bound_db=0.0, limiter_threshold_dbtp=None, expected_integrated_lufs=expected,
        shortfall_lu=max(0.0, plan.target_lufs - expected), fallback_from=action, reduce_fallbacks=1,
        reason_code="reduce_shortfall", diagnostics=kept + (note,),
    )


# --- Explicit dynamic: V6's one peak correction ------------------------------------------------


def dynamic_peak_ceiling(plan: GainPlan, conservative_peak_dbtp: float) -> float | None:
    """V6: the lowered dynamic ceiling, or None when it leaves loudnorm's ``TP`` range (no clamp).

    The ceiling (C-05 ``limiter_threshold_dbtp``, initially c) drops by
    ``o + 0.2`` dB with ``o = TP_out,up − c`` from M7's conservative level.
    """
    if plan.action != "dynamic" or plan.limiter_threshold_dbtp is None:
        raise ValueError("only a dynamic plan has a dynamic ceiling (R6)")
    if plan.dynamic_preview:
        raise ValueError("a dynamic preview is never rendered, so it receives no peak correction (C-05 v1.3)")
    if plan.peak_corrections >= 1:
        raise ValueError("the peak correction is already used (ADR-0014)")
    if not math.isfinite(conservative_peak_dbtp):
        raise ValueError("a peak correction needs a finite conservative peak level")
    overshoot = conservative_peak_dbtp - plan.ceiling_dbtp
    ceiling = plan.limiter_threshold_dbtp - (overshoot + PEAK_CORRECTION_MARGIN_DB)
    return ceiling if in_range(ceiling, LOUDNORM_TP_RANGE) else None


def replan_dynamic_peak(plan: GainPlan, calibration: DynamicCalibration) -> GainPlan:
    """V6: the successor dynamic plan rendered at the recalibrated, lowered ceiling.

    ``calibration`` is the original source's fresh analysis at the ceiling
    :func:`dynamic_peak_ceiling` returned; it consumes the one peak correction.
    """
    if plan.action != "dynamic" or plan.limiter_threshold_dbtp is None:
        raise ValueError("only a dynamic plan receives a dynamic peak correction (V6)")
    if plan.dynamic_preview:
        raise ValueError("a dynamic preview is never rendered, so it receives no peak correction (C-05 v1.3)")
    if plan.peak_corrections >= 1:
        raise ValueError("the peak correction is already used (ADR-0014)")
    if calibration.target_lufs != plan.target_lufs or not calibration.ceiling_dbtp < plan.limiter_threshold_dbtp:
        raise ValueError("the recalibration must use the plan's target and a lower ceiling (V6)")
    lowered = plan.limiter_threshold_dbtp - calibration.ceiling_dbtp
    note = Diagnostic("ok", f"dynamic ceiling lowered by {lowered:.4f} dB to {calibration.ceiling_dbtp:.4f} dBTP "
                            f"for peak safety", RETRY_STAGE)
    return replace(plan, limiter_threshold_dbtp=calibration.ceiling_dbtp, peak_corrections=1,
                   dynamic_parameters=calibration.parameters, diagnostics=plan.diagnostics + (note,))
