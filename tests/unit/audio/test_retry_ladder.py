"""TK-0110: pure retry replanning and the bounded ladder (audio §§ 4.2, 5; ADR-0014, ADR-0018, ADR-0019).

Synthetic C-03/C-05 records only; no process starts. Real-tool limit and
reduce renders are in ``test_limit_reduce.py``. TST-A-05 (caps, c6), TST-A-07
(≤ 5 renders, budgets 2/1/1 never reset, peak first), TST-A-09 (reduce).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from norma.audio.plan import (
    FALLBACK_NOTE, LIMITER_MARGIN_DB, limiter_threshold_supported, peak_adjustment, plan_gain, reduce_fallback,
    replan_loudness, replan_peak,
)
from norma.audio.verify import (
    MAX_RENDERS, Accepted, Retry, assess, next_step, run_ladder,
)
from norma.errors import FileFailure
from norma.models import LoudnessMeasurement, RenderResult, RunOptions, TargetFormat

CANDIDATE = Path("/unused/out/candidate.wav")
TARGET = TargetFormat("wav", ".wav", "pcm_s24le", 24, False, 48000, 2, "stereo", "none", "none", None, None, None,
                      "not_applicable", "ev:synthetic", False, "as_requested")


def measurement(integrated=-23.0, peak=-5.0, *, resolution=.001, linear=None, state="measurable"):
    linear = 10 ** (peak / 20) if linear is None else linear
    measurable = state == "measurable"
    return LoudnessMeasurement(
        signal_state=state, integrated_lufs=integrated if measurable else None,
        true_peak_dbtp=20 * math.log10(linear) if linear > 0 else -math.inf, true_peak_linear=linear,
        lra_lu=0.0 if measurable else None, threshold_lufs=None, integrated_valid=measurable, peak_valid=True,
        lra_valid=measurable, threshold_valid=False, integrated_resolution_lu=.001,
        peak_resolution_linear=resolution, dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic", decoded_duration_s=30.0, decoded_frames=1440000, sample_rate_hz=48000,
        decoded_nonzero=state != "silent", terminal_peak_covered=True)


def options(**changes):
    return RunOptions(command="normalise", ffmpeg=Path("/unused/ffmpeg"), cwd=Path("/unused"), jobs=1, **changes)


def out(integrated, peak_dbtp, *, resolution=.001):
    """A measured candidate; ``peak_dbtp`` is the printed central value, M7 adds ``q/2`` to its amplitude."""
    return measurement(integrated, peak_dbtp, resolution=resolution)


def step(source, plan, candidate):
    return next_step(source, assess(candidate, plan), plan)


# --- R4 bounds and V4 cap ----------------------------------------------------------------------------


def test_initial_limit_plan_reports_both_bounds():
    plan = plan_gain(measurement(-23.0, -5.0), options())
    assert plan.action == "limit" and plan.limiter_threshold_dbtp == pytest.approx(-1.3)
    assert plan.limiter_reduction_db == pytest.approx(3.0)            # policy bound, = max_limit, accepted
    assert plan.effective_limiter_bound_db == pytest.approx(3.3)      # includes the 0.3 dB margin
    assert peak_adjustment(plan) == 0.0 and LIMITER_MARGIN_DB == 0.3


def test_loudness_correction_cap_boundary():
    """V4: a policy bound equal to max_limit is accepted (effective = cap + 0.3); 0.001 dB above is not.

    Binary-exact values (−5.25, −16.25) keep the equality exact in floating point.
    """
    source = measurement(-23.0, -5.25)
    plan = plan_gain(source, options())
    assert plan.action == "limit" and plan.limiter_reduction_db == 2.75
    corrected = replan_loudness(source, plan, -16.25)
    assert corrected is not None and corrected.action == "limit"
    assert corrected.gain_db == 7.25 and corrected.limiter_reduction_db == 3.0 == corrected.max_limit_db
    assert corrected.effective_limiter_bound_db == pytest.approx(3.3)
    assert corrected.limiter_threshold_dbtp == plan.limiter_threshold_dbtp
    assert corrected.loudness_corrections == 1
    assert replan_loudness(source, plan, -16.251) is None             # 0.001 dB above the cap


def test_cap_check_counts_peak_adjustment():
    source = measurement(-23.0, -6.0)
    plan = plan_gain(source, options())                               # policy 2.0
    peaked = replan_peak(source, plan, -.85)                          # o = .15 → adjustment .35
    assert peaked is not None and peak_adjustment(peaked) == pytest.approx(.35)
    assert peaked.limiter_threshold_dbtp == pytest.approx(-1.65)
    assert peaked.limiter_reduction_db == pytest.approx(2.35) and peaked.gain_db == plan.gain_db
    assert peaked.effective_limiter_bound_db == pytest.approx(2.65)
    # Remaining cap: 3 − 2.35 = .65 dB more gain.
    assert replan_loudness(source, peaked, -16.64) is not None
    assert replan_loudness(source, peaked, -16.66) is None
    kept = replan_loudness(source, peaked, -16.2)
    assert kept.action == "limit" and kept.limiter_threshold_dbtp == peaked.limiter_threshold_dbtp
    assert kept.peak_corrections == 1 and kept.loudness_corrections == 1


def test_correction_that_fits_returns_to_linear_but_kept_limiter_stays():
    source = measurement(-23.0, -9.0)
    linear = plan_gain(source, options())
    assert linear.action == "linear"
    assert replan_loudness(source, linear, -16.4).action == "linear"        # +0.4: −1.6 fits
    assert replan_loudness(source, linear, -14.5).action == "linear"        # less gain still fits
    up_one = replan_loudness(source, linear, -17.5)                         # +1.5 → −0.5 dBTP
    assert up_one.action == "limit" and up_one.limiter_threshold_dbtp == pytest.approx(-1.3)
    limited = plan_gain(measurement(-23.0, -5.0), options())
    down = replan_loudness(measurement(-23.0, -5.0), limited, -14.0)        # g' = 5 → predicted −0, still limit
    assert down.action == "limit"
    quiet = replan_loudness(measurement(-23.0, -5.0), limited, -10.0)       # g' = 1 → fits, limiter kept
    assert quiet.action == "limit" and quiet.limiter_reduction_db == 0.0
    assert quiet.effective_limiter_bound_db == 0.0


def test_reduce_policy_and_zero_cap_never_gain_a_limiter():
    source = measurement(-23.0, -9.0)
    for run in (options(peak_mode="reduce"), options(max_limit=0.0)):
        plan = plan_gain(source, run)
        assert plan.action == "linear"
        assert replan_loudness(source, plan, -17.5) is None
        assert replan_peak(source, plan, -.8) is None


def test_limiter_threshold_range_is_not_clamped():
    assert limiter_threshold_supported(0.0) and limiter_threshold_supported(-24.08)
    assert not limiter_threshold_supported(-24.1) and not limiter_threshold_supported(.01)
    assert not limiter_threshold_supported(float("nan"))
    source = measurement(-40.0, -50.0)
    plan = plan_gain(source, options(true_peak=-9.0))
    assert plan.action == "linear"                                        # predicted −26 dBTP: cap unused
    lowered = replan_peak(source, plan, 5.0)                              # L = −9.3 − 14.2 = −23.5
    assert lowered is not None and lowered.limiter_threshold_dbtp == pytest.approx(-23.5)
    assert lowered.limiter_reduction_db == 0.0
    assert replan_peak(source, plan, 5.7) is None                         # L = −24.2: outside, not clamped


# --- V5 peak correction ------------------------------------------------------------------------------


def test_m7_worked_example_peak_correction():
    """§ 5.1: printed 0.716 at c = −3 → o = 0.1043, adjustment 0.3043; reduce fails at once."""
    source = measurement(-30.0, -12.0)
    plan = plan_gain(source, options(true_peak=-3.0, target=-21.0))
    assert plan.action == "linear"
    candidate = out(-21.0, 20 * math.log10(.716))
    assert candidate.true_peak_dbtp == pytest.approx(-2.9017, abs=1e-4)
    result = step(source, plan, candidate)
    assert isinstance(result, Retry) and result.kind == "peak_correction"
    assert peak_adjustment(result.plan) == pytest.approx(.3043, abs=1e-4)
    assert result.plan.action == "limit" and result.plan.peak_corrections == 1
    assert result.plan.limiter_threshold_dbtp == pytest.approx(-3.3 - .3043, abs=1e-4)
    assert any(d.stage == "retry" and "peak safety" in d.message for d in result.plan.diagnostics)
    reduce = replace(plan, action="reduce", requested_peak_mode="reduce")
    failure = step(source, reduce, candidate)
    assert isinstance(failure, FileFailure) and failure.reason == "verification_peak"
    assert reduce.peak_corrections == 0


def test_peak_first_even_when_loudness_also_misses():
    source = measurement(-23.0, -9.0)
    plan = plan_gain(source, options())
    result = step(source, plan, out(-18.0, -.5))
    assert isinstance(result, Retry) and result.kind == "peak_correction"
    assert result.plan.loudness_corrections == 0 and result.plan.gain_db == plan.gain_db


@pytest.mark.parametrize("case", ["used", "unknown_q", "cap", "reduce_policy"])
def test_peak_failure_without_eligible_correction(case):
    source = measurement(-23.0, -5.0)
    plan = plan_gain(source, options(peak_mode="reduce" if case == "reduce_policy" else "limit"))
    candidate = out(-16.0, -.5)
    if case == "used":
        plan = replace(plan, peak_corrections=1)
    elif case == "unknown_q":
        candidate = out(-16.0, -1.5, resolution=None)
    elif case == "cap":
        assert plan.limiter_reduction_db == pytest.approx(3.0)        # no cap left for any adjustment
    result = step(source, plan, candidate)
    assert isinstance(result, FileFailure) and (result.reason, result.stage) == ("verification_peak", "verify")


# --- V3 order: corrections, fallback, budgets ------------------------------------------------------


def test_reduce_fallback_records_origin_and_always_warns():
    source = measurement(-23.0, -5.0)
    plan = replace(plan_gain(source, options()), loudness_corrections=2, peak_corrections=1)
    fallback = reduce_fallback(source, plan)
    assert fallback.action == "reduce" and fallback.fallback_from == "limit"
    assert fallback.gain_db == pytest.approx(4.0) and fallback.expected_integrated_lufs == pytest.approx(-19.0)
    assert fallback.shortfall_lu == pytest.approx(3.0) and fallback.reason_code == "reduce_shortfall"
    assert (fallback.loudness_corrections, fallback.peak_corrections, fallback.reduce_fallbacks) == (2, 1, 1)
    assert fallback.limiter_threshold_dbtp is None and fallback.limiter_reduction_db == 0
    assert [d.message for d in fallback.diagnostics if d.code == "reduce_shortfall"] == [FALLBACK_NOTE]
    with pytest.raises(ValueError):
        reduce_fallback(source, fallback)
    # Accepted even at zero measured shortfall, still WARN with the note.
    accepted = step(source, fallback, out(-15.9, -1.2))
    assert isinstance(accepted, Accepted)
    assert accepted.verification.measured_shortfall_lu == 0.0
    assert [w.code for w in accepted.warnings] == ["reduce_shortfall"] and FALLBACK_NOTE in accepted.warnings[0].message


def test_explicit_reduce_warns_only_on_nonzero_measured_shortfall():
    source = measurement(-23.0, -5.0)
    plan = plan_gain(source, options(peak_mode="reduce"))
    assert plan.action == "reduce" and plan.fallback_from is None
    short = step(source, plan, out(-19.05, -1.05))
    assert isinstance(short, Accepted) and short.verification.measured_shortfall_lu == pytest.approx(3.05)
    assert [w.code for w in short.warnings] == ["reduce_shortfall"]
    met = step(source, plan, out(-15.9, -1.05))
    assert isinstance(met, Accepted) and met.warnings == ()


def test_cap_ineligible_correction_goes_straight_to_fallback():
    """F-05/c6 shape (audio § 8 C6): cap-safe initial plan, the first correction would need 4.77 dB."""
    source = measurement(-30.0, -6.72)
    plan = plan_gain(source, options(target=-21.5))
    assert plan.action == "limit" and plan.limiter_reduction_db == pytest.approx(2.78)
    assert replan_loudness(source, plan, -23.491) is None                 # policy 4.771 dB > 3
    result = step(source, plan, out(-23.491, -1.3))
    assert isinstance(result, Retry) and result.kind == "reduce_fallback"
    assert result.plan.loudness_corrections == 0 and result.plan.fallback_from == "limit"


def test_unmeasurable_candidate_fails_without_correction():
    source = measurement()
    plan = plan_gain(source, options())
    for state in ("unmeasurable", "silent"):
        candidate = measurement(state=state, linear=.001 if state == "unmeasurable" else 0.0)
        result = step(source, plan, candidate)
        assert isinstance(result, FileFailure) and result.reason == "verification_loudness"


def test_dynamic_receives_no_ordinary_correction():
    source = measurement()
    plan = replace(plan_gain(source, options()), action="dynamic", requested_peak_mode="dynamic",
                   limiter_reduction_db=0.0, effective_limiter_bound_db=0.0, dynamic_parameters={
                       "input_i": -23.0, "input_tp": -5.0, "input_lra": 1.0, "input_thresh": -33.0,
                       "target_offset": 0.0})
    result = step(source, plan, out(-18.0, -2.0))
    assert isinstance(result, FileFailure) and result.reason == "verification_loudness"
    with pytest.raises(ValueError):
        replan_loudness(source, plan, -18.0)


# --- run_ladder: ≤ 5 renders, counters never reset, V3a callbacks ------------------------------------


class Scripted:
    """Fake attempt: returns scripted measurements per render, recording every plan it renders."""

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.plans = []

    def __call__(self, plan, number):
        self.plans.append(plan)
        candidate = self.candidates.pop(0)
        result = RenderResult(CANDIDATE, number, plan, TARGET, (1, 2), .1, None, None, True, "synthetic", None)
        return result, assess(candidate, plan)


def test_worst_case_ladder_uses_all_five_renders_and_succeeds():
    source = measurement(-23.0, -9.0)
    initial = plan_gain(source, options())
    attempt = Scripted([out(-16.0, -.5),     # 1 linear: peak unsafe → peak correction
                        out(-17.0, -1.2),    # 2 limit: loudness miss → correction 1
                        out(-16.8, -1.2),    # 3 → correction 2
                        out(-16.7, -1.2),    # 4 → fallback
                        out(-15.0, -1.1)])   # 5 reduce: accepted, WARN
    retries = []
    outcome = run_ladder(source, initial, attempt, lambda old, new, kind: retries.append(kind))
    assert retries == ["peak_correction", "loudness_correction", "loudness_correction", "reduce_fallback"]
    assert len(outcome.renders) == MAX_RENDERS == 5 and [r.attempt for r in outcome.renders] == [1, 2, 3, 4, 5]
    counters = [(p.loudness_corrections, p.peak_corrections, p.reduce_fallbacks) for p in attempt.plans]
    assert counters == [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1)]
    assert [p.action for p in attempt.plans] == ["linear", "limit", "limit", "limit", "reduce"]
    assert outcome.plan.fallback_from == "limit" and [w.code for w in outcome.warnings] == ["reduce_shortfall"]
    # Every limit plan keeps the lowered threshold after the peak correction.
    assert len({p.limiter_threshold_dbtp for p in attempt.plans[1:4]}) == 1
    assert attempt.plans[1].limiter_threshold_dbtp < -1.3


def test_ladder_that_never_converges_fails_within_the_budget():
    source = measurement(-23.0, -9.0)
    attempt = Scripted([out(-16.0, -.5), out(-17.0, -1.2), out(-16.8, -1.2), out(-16.7, -1.2), out(-15.0, -.7)])
    with pytest.raises(FileFailure) as caught:
        run_ladder(source, plan_gain(source, options()), attempt)
    assert caught.value.reason == "verification_peak" and len(attempt.plans) == 5   # unsafe fallback: no 6th


def test_loudness_budget_exhausted_after_fallback_used_fails_loudness():
    source = measurement(-23.0, -9.0)
    plan = replace(plan_gain(source, options()), loudness_corrections=2, reduce_fallbacks=1)
    result = step(source, plan, out(-17.0, -2.0))
    assert isinstance(result, FileFailure) and result.reason == "verification_loudness"


def test_before_retry_failure_ends_the_file_without_a_render():
    source = measurement(-23.0, -9.0)
    attempt = Scripted([out(-17.0, -2.0)])

    def refuse(old, new, kind):
        assert (old.loudness_corrections, new.loudness_corrections, kind) == (0, 1, "loudness_correction")
        raise FileFailure("metadata_loss_error", "metadata_planned", "retry would drop a tag")
    with pytest.raises(FileFailure) as caught:
        run_ladder(source, plan_gain(source, options()), attempt, refuse)
    assert (caught.value.reason, caught.value.stage) == ("metadata_loss_error", "metadata_planned")
    assert len(attempt.plans) == 1


def test_attempt_must_render_the_requested_plan():
    source = measurement(-23.0, -9.0)
    plan = plan_gain(source, options())

    def wrong(p, n):
        result = RenderResult(CANDIDATE, n + 1, p, TARGET, (1, 2), .1, None, None, True, "synthetic", None)
        return result, assess(out(-16.0, -2.0), p)
    with pytest.raises(ValueError):
        run_ladder(source, plan, wrong)


def test_first_candidate_accepted_takes_one_render():
    source = measurement(-23.0, -9.0)
    attempt = Scripted([out(-16.1, -2.1)])
    outcome = run_ladder(source, plan_gain(source, options()), attempt)
    assert len(outcome.renders) == 1 and outcome.warnings == () and outcome.plan.action == "linear"


def test_replanning_is_immutable_and_refuses_spent_budgets():
    source = measurement(-23.0, -5.0)
    plan = plan_gain(source, options())
    before = replace(plan)
    replan_loudness(source, plan, -17.0)
    replan_peak(source, plan, -.5)
    reduce_fallback(source, plan)
    assert plan == before
    with pytest.raises(ValueError):
        replan_loudness(source, replace(plan, loudness_corrections=2), -17.0)
    with pytest.raises(ValueError):
        replan_peak(source, replace(plan, peak_corrections=1), -.5)
    with pytest.raises(ValueError):
        replan_loudness(measurement(state="unmeasurable", linear=.001), plan, -17.0)
