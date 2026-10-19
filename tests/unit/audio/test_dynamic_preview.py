"""TK-0112: the read-only dynamic preview plan (C-05 v1.3, D-080, CR-037; audio V8, V10).

Hermetic: synthetic C-03 records and C-10 options only. No file is opened,
no process runs and no calibration is made — the preview exists precisely so
that ``analyse`` and ``normalise --dry-run`` need none.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from norma.adapters import ffmpeg
from norma.audio.plan import (
    DynamicCalibration,
    dynamic_peak_ceiling,
    needs_dynamic_calibration,
    plan_dynamic_preview,
    plan_gain,
    replan_dynamic_peak,
)
from norma.audio.render import dynamic_filter, filter_chain, render, render_argv
from norma.models import Diagnostic, GainPlan, LoudnessMeasurement, ProbeResult, RenderResult, RunOptions, TargetFormat

UNUSED = Path("/unused")
PARAMS = {"input_i": -23.55, "input_tp": -1.01, "input_lra": 1.0, "input_thresh": -33.55, "target_offset": 0.62}


def measurement(integrated=-23.0, peak=-5.0, *, resolution=.001, **changes) -> LoudnessMeasurement:
    return replace(LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak,
        true_peak_linear=10 ** (peak / 20), lra_lu=1.0, threshold_lufs=None, integrated_valid=True,
        peak_valid=True, lra_valid=True, threshold_valid=False, integrated_resolution_lu=.001,
        peak_resolution_linear=resolution, dual_mono=False, meter="ebur128", value_source="metadata",
        ffmpeg_version="synthetic", decoded_duration_s=30.0, decoded_frames=1440000, sample_rate_hz=48000,
        decoded_nonzero=True, terminal_peak_covered=True), **changes)


def silent() -> LoudnessMeasurement:
    return replace(measurement(), signal_state="silent", integrated_lufs=None, integrated_valid=False,
                   true_peak_linear=0, true_peak_dbtp=-math.inf, decoded_nonzero=False)


def unmeasurable() -> LoudnessMeasurement:
    return replace(measurement(), signal_state="unmeasurable", integrated_lufs=None, integrated_valid=False)


def options(command="analyse", **changes) -> RunOptions:
    # Paths are immutable C-10 values only: these files are never accessed.
    return RunOptions(command=command, ffmpeg=UNUSED / "ffmpeg", cwd=UNUSED, jobs=1,
                      **{"peak_mode": "dynamic", **changes})


def preview(**changes) -> GainPlan:
    return plan_dynamic_preview(measurement(**changes), options())


def probe() -> ProbeResult:
    return ProbeResult("pcm_s24le", "lossless", "wav", 0, 0, 48000, 24, False, "s32", 2, "stereo",
                       30.0, "stream", None, False, 0, {}, "synthetic")


def wav24() -> TargetFormat:
    return TargetFormat("wav", ".wav", "pcm_s24le", 24, False, 48000, 2, "stereo", "none", "none", None, None,
                        None, "not_applicable", "ev:synthetic", False, "as_requested")


# --- The preview's fixed fields (C-05 v1.3) ------------------------------------------------------


@pytest.mark.parametrize("command", ["analyse", "normalise"])  # normalise: its --dry-run plan
@pytest.mark.parametrize("integrated,peak,target,ceiling", [
    (-23.0, -5.0, -16.0, -1.0),    # boost whose linear peak would exceed c
    (-23.0, -10.0, -16.0, -1.0),   # boost that would fit linearly: dynamic is still what was asked for
    (-9.0, 0.5, -14.0, -2.0),      # cut, source above 0 dBTP
    (-40.0, -20.0, -16.0, -9.0),   # far below target at the lowest ceiling
])
def test_preview_holds_the_fixed_v13_fields(command, integrated, peak, target, ceiling):
    selected = options(command, target=target, true_peak=ceiling)
    plan = plan_dynamic_preview(measurement(integrated, peak), selected)
    assert plan.dynamic_preview
    assert (plan.action, plan.requested_peak_mode, plan.dynamic_parameters) == ("dynamic", "dynamic", None)
    assert plan.gain_db == pytest.approx(target - integrated)  # nominal t − I_in
    assert plan.expected_integrated_lufs == target
    assert plan.predicted_true_peak_dbtp is None  # unknown until a calibrated render is measured
    assert plan.limiter_reduction_db == plan.effective_limiter_bound_db == 0.0
    assert plan.limiter_threshold_dbtp == plan.ceiling_dbtp == ceiling  # the dynamic ceiling c
    assert plan.shortfall_lu == 0.0  # V10: no predicted shortfall, and no reduce_shortfall warning
    assert plan.fallback_from is None
    assert plan.loudness_corrections == plan.peak_corrections == plan.reduce_fallbacks == 0
    assert plan.reason_code == "ok"
    assert (plan.target_lufs, plan.tolerance_lu, plan.max_limit_db) == (target, selected.tolerance, selected.max_limit)


def test_preview_satisfies_the_models_invariant_when_rebuilt():
    plan = preview()
    assert GainPlan(**{f: getattr(plan, f) for f in plan.__dataclass_fields__}) == plan  # __post_init__ passes


@pytest.mark.parametrize("change,rule", [
    ({"gain_db": None}, "nominal gain"),
    ({"expected_integrated_lufs": -17.0}, "target loudness"),
    ({"predicted_true_peak_dbtp": -1.5}, "predicts no true peak"),
    ({"limiter_reduction_db": 0.5}, "limiter bounds"),
    ({"effective_limiter_bound_db": 0.5}, "limiter bounds"),
    ({"limiter_threshold_dbtp": -1.5}, "dynamic ceiling"),
    ({"shortfall_lu": 1.0}, "no shortfall"),
    ({"peak_corrections": 1}, "no retry budget"),
    ({"loudness_corrections": 1}, "no retry budget"),
])
def test_models_refuses_any_other_uncalibrated_dynamic_plan(change, rule):
    with pytest.raises(ValueError, match=rule):
        replace(preview(), **change)


def test_preview_differs_from_the_calibrated_plan_only_in_its_calibration():
    source, selected = measurement(), options("normalise")
    calibrated = plan_gain(source, selected, calibration=DynamicCalibration(PARAMS, "dynamic", -16.0, -1.0))
    plan = plan_dynamic_preview(source, selected)
    assert not calibrated.dynamic_preview
    assert replace(calibrated, dynamic_parameters=None) == plan


def test_preview_keeps_the_measurement_diagnostics_and_adds_none():
    note = Diagnostic("ok", "synthetic measurement note", "measure")
    plan = plan_dynamic_preview(measurement(diagnostics=(note,)), options())
    assert plan.diagnostics == (note,)


# --- Where normalise would need no calibration, the preview is plan_gain's plan -----------------


@pytest.mark.parametrize("make,state", [(silent, "silent"), (unmeasurable, "unmeasurable")])
def test_silent_and_unmeasurable_inputs_skip_as_for_other_plans(make, state):
    source, selected = make(), options()
    plan = plan_dynamic_preview(source, selected)
    assert plan == plan_gain(source, selected)
    assert (plan.action, plan.reason_code, plan.dynamic_parameters) == ("skip", state, None)
    assert plan.gain_db is plan.expected_integrated_lufs is plan.predicted_true_peak_dbtp is plan.shortfall_lu is None
    assert plan.limiter_threshold_dbtp is None and not plan.dynamic_preview
    for mode in ("limit", "reduce"):  # the same skip as every other policy, bar the requested mode
        assert replace(plan_gain(source, options(peak_mode=mode)), requested_peak_mode="dynamic") == plan


def test_copy_row_precedes_the_dynamic_row_in_the_preview_too():
    within = measurement(-16.1, -5.0)
    assert not needs_dynamic_calibration(within, options())
    plan = plan_dynamic_preview(within, options())
    assert plan == plan_gain(within, options())
    assert (plan.action, plan.reason_code, plan.gain_db) == ("copy", "already_within_tolerance", 0.0)


def test_unknown_peak_resolution_forbids_the_copy_row_and_previews_dynamic():
    plan = plan_dynamic_preview(measurement(-16.1, -5.0, resolution=None), options())
    assert plan.dynamic_preview and plan.gain_db == pytest.approx(0.1)


@pytest.mark.parametrize("make", [measurement, silent, unmeasurable,
                                  lambda: measurement(-16.1, -5.0), lambda: measurement(-9.0, 0.5)])
def test_preview_is_dynamic_exactly_when_normalise_would_calibrate(make):
    source = make()
    assert plan_dynamic_preview(source, options()).dynamic_preview == needs_dynamic_calibration(source, options())


@pytest.mark.parametrize("mode", ["limit", "reduce"])
def test_other_policies_are_a_caller_error(mode):
    with pytest.raises(ValueError, match="explicit dynamic policy only"):
        plan_dynamic_preview(measurement(), options(peak_mode=mode))


def test_preview_is_pure_and_deterministic():
    source, selected = measurement(), options()
    before = (replace(source), replace(selected))
    assert plan_dynamic_preview(source, selected) == plan_dynamic_preview(source, selected)
    assert (source, selected) == before


# --- Never rendered (C-05 v1.3, R5) ---------------------------------------------------------------


def test_render_refuses_the_preview_before_any_process():
    plan = preview()
    with pytest.raises(ValueError, match="calibration"):
        dynamic_filter(plan, 48000, dual_mono=False)
    with pytest.raises(ValueError, match="calibration"):
        filter_chain(plan, probe(), wav24())
    executable = ffmpeg.Executable(UNUSED / "ffmpeg", (1, 2), "synthetic")
    with pytest.raises(ValueError, match="calibration"):
        render_argv(executable, UNUSED / "source.wav", UNUSED / "out/candidate.wav", plan, probe(), wav24())
    # render() builds its argv first: it fails before revalidating evidence, reading the source or running FFmpeg.
    env = SimpleNamespace(executable=executable)
    with pytest.raises(ValueError, match="calibration"):
        render(UNUSED / "source.wav", (1, 2), probe(), plan, wav24(), UNUSED / "out/candidate.wav", env, ())


def test_render_result_never_holds_a_preview():
    plan = preview()
    with pytest.raises(ValueError, match="never rendered"):
        RenderResult(UNUSED / "out/candidate.wav", 1, plan, wav24(), (1, 2), 0.0, None, None, True, None, "dynamic")


def test_the_preview_receives_no_v6_peak_correction():
    plan = preview()
    with pytest.raises(ValueError, match="never rendered"):
        dynamic_peak_ceiling(plan, -0.8)
    with pytest.raises(ValueError, match="never rendered"):
        replan_dynamic_peak(plan, DynamicCalibration(PARAMS, "dynamic", -16.0, -1.35))


def test_preview_runs_no_calibration_by_construction():
    """Audio V8: plan.py imports no adapter or measurement module, so the preview cannot calibrate."""
    tree = ast.parse(Path(__file__).parents[3].joinpath("src/norma/audio/plan.py").read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(name and (name.startswith("norma.adapters") or name == "norma.audio.measure") for name in imported)
