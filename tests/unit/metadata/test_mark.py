"""Processing mark text and finalisation owned by WS-06 (MD26, MD34, MD37; CR-086; TK-0613, S4).

The grammar cases follow the § 10 table. The equality cases compare ``model`` with the composition
TK-0427 put in ``pipeline.py`` (read only here): same text, same C-07 revision, on several plans.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from norma import pipeline
from norma.metadata import losscheck, mapping, model, write
from norma.models import GainPlan, LoudnessMeasurement, MetadataPlan, RawNativeItem, ResolvedValue

SPEC_EXAMPLE = (
    "v=1;tool=norma-0.1.0;target=-16;tp=-1;mode=limit;gain=+5.80;limited=2.60;out_i=-16.02;"
    "out_tp=-1.04;origin=pcm_s24le;ffmpeg=N-125892;date=2026-10-05"
)


def gain_plan(**changes) -> GainPlan:
    base = GainPlan(action="limit", requested_peak_mode="limit", target_lufs=-16.0, ceiling_dbtp=-1.0,
                    tolerance_lu=0.5, max_limit_db=3.0, gain_db=5.8, predicted_true_peak_dbtp=-1.0,
                    limiter_reduction_db=2.6, effective_limiter_bound_db=2.9, limiter_threshold_dbtp=-1.3,
                    expected_integrated_lufs=-16.0, shortfall_lu=0.0, fallback_from=None, loudness_corrections=0,
                    peak_corrections=0, reduce_fallbacks=0, reason_code="ok")
    action = changes.get("action", "limit")
    if action != "limit":  # C-05 invariants: no limiter bounds without a limiter
        base = replace(base, limiter_reduction_db=0.0, effective_limiter_bound_db=0.0,
                       limiter_threshold_dbtp=None if action != "dynamic" else -1.0)
    if action == "dynamic":  # the read-only preview shape (D-080): nominal gain, expects the target, no peak
        target = changes.get("target_lufs", base.target_lufs)
        base = replace(base, requested_peak_mode="dynamic", predicted_true_peak_dbtp=None,
                       expected_integrated_lufs=target, shortfall_lu=0.0, gain_db=changes.get("gain_db") or 4.0)
        changes = {**changes, "gain_db": base.gain_db}
    return replace(base, **changes)


def measurement(integrated: float | None = -16.02, peak: float | None = -1.04) -> LoudnessMeasurement:
    return LoudnessMeasurement(
        signal_state="measurable", integrated_lufs=integrated, true_peak_dbtp=peak, true_peak_linear=0.9,
        lra_lu=3.0, threshold_lufs=None, integrated_valid=True, peak_valid=True, lra_valid=True,
        threshold_valid=False, integrated_resolution_lu=0.1, peak_resolution_linear=1e-6, dual_mono=False,
        meter="ebur128", value_source="metadata", ffmpeg_version="N-125892-g406c5a37aa",
        decoded_duration_s=3.0, decoded_frames=144000, sample_rate_hz=48000, decoded_nonzero=True,
        terminal_peak_covered=True)


def mark_text(gain: GainPlan | None = None, measured: LoudnessMeasurement | None = None, **changes) -> str:
    arguments = {"origin": "pcm_s24le", "ffmpeg_version": "N-125892-g406c5a37aa", "run_date": "2026-10-05",
                 "tool_version": "0.1.0", **changes}
    return model.processing_mark_value(gain or gain_plan(), measured or measurement(), **arguments)


# --- Text: the § 10 table -----------------------------------------------------------------


def test_the_spec_example_is_reproduced_exactly() -> None:
    assert mark_text() == SPEC_EXAMPLE
    assert model.parse_processing_mark([("x", mark_text())]).validity == "valid"


def test_keys_come_in_table_order() -> None:
    keys = [pair.split("=", 1)[0] for pair in mark_text().split(";")]
    assert keys == ["v", "tool", "target", "tp", "mode", "gain", "limited", "out_i", "out_tp", "origin",
                    "ffmpeg", "date"]


@pytest.mark.parametrize("action", ["copy", "linear", "reduce", "dynamic"])
def test_limited_is_present_only_for_limit(action: str) -> None:
    text = mark_text(gain_plan(action=action, gain_db=-3.0))
    assert "limited=" not in text and f"mode={action};gain=-3.00;out_i=" in text
    parsed = model.parse_processing_mark([("x", text)])
    assert (parsed.validity, parsed.mode) == ("valid", action)


@pytest.mark.parametrize(("value", "signed", "strip", "expected"), [
    (-16.0, False, True, "-16"), (-14.5, False, True, "-14.5"), (-14.25, False, True, "-14.25"),
    (-16.125, False, False, "-16.12"), (-16.135, False, False, "-16.14"), (5.8, True, False, "+5.80"),
    (-0.0, True, False, "+0.00"), (-0.004, True, False, "+0.00"), (-3.0, True, False, "-3.00"),
    (0, True, False, "+0.00"), (0.0, False, False, "0.00"), (-0.0, False, True, "0"), (2.6, False, False, "2.60"),
    (-23.015, False, False, "-23.02"), (-0.005, False, False, "0.00"),
])
def test_number_rounding_is_half_even_on_the_shortest_decimal(value, signed, strip, expected) -> None:
    assert model.format_mark_number(value, signed=signed, strip=strip) == expected


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), True, "1.0"])
def test_a_non_finite_or_missing_number_is_a_bug_never_written(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        model.format_mark_number(value)
    with pytest.raises(ValueError, match="finite"):
        # A real LoudnessMeasurement cannot hold these values; a stand-in proves the mark does not trust it.
        mark_text(measured=SimpleNamespace(integrated_lufs=value, true_peak_dbtp=-1.0))  # type: ignore[arg-type]


def test_a_zero_gain_and_a_missing_gain_both_read_plus_zero() -> None:
    assert "gain=+0.00;" in mark_text(gain_plan(action="copy", gain_db=None))
    assert "gain=+0.00;" in mark_text(gain_plan(action="copy", gain_db=-0.0))


@pytest.mark.parametrize(("version", "token"), [
    ("N-125892-g406c5a37aa", "N-125892"), ("7.1", "7.1"), ("7.1.1", "7.1.1"), ("N-125892", "N-125892"),
    ("n7.1-3-gabcdef-tessus", "n7.1-3"),
])
def test_the_ffmpeg_token_stops_before_the_second_dash(version: str, token: str) -> None:
    assert model.ffmpeg_version_token(version) == token
    assert f";ffmpeg={token};" in mark_text(ffmpeg_version=version)


# --- Text: limits and refusals ------------------------------------------------------------


def test_the_longest_legal_mark_is_accepted_at_255_characters() -> None:
    base = mark_text(origin="a")
    padding = 255 - len(base)
    text = mark_text(origin="a" * (1 + padding))
    assert len(text) == 255
    assert model.parse_processing_mark([("x", text)]).validity == "valid"


def test_one_character_over_255_is_refused_and_would_read_back_malformed() -> None:
    base = mark_text(origin="a")
    text = base.replace("origin=a", "origin=" + "a" * (1 + 256 - len(base)))
    assert len(text) == 256
    parsed = model.parse_processing_mark([("x", text)])
    assert (parsed.validity, parsed.diagnostics[0].detail) == ("malformed", "malformed:too_long")
    with pytest.raises(ValueError, match="too_long"):
        mark_text(origin="a" * (1 + 256 - len(base)))


@pytest.mark.parametrize(("changes", "reason"), [
    ({"origin": "PCM S24"}, "not_printable_ascii"), ({"origin": "pcm;s24"}, "not_key_value_pairs"),
    ({"origin": "a=b"}, "invalid_value:origin"), ({"origin": "Flac"}, "invalid_value:origin"),
    ({"origin": ""}, "invalid_value:origin"), ({"tool_version": "0 1"}, "not_printable_ascii"),
    ({"tool_version": "é"}, "not_printable_ascii"), ({"run_date": "2026-02-30"}, "invalid_value:date"),
    ({"run_date": "5 Oct 2026"}, "not_printable_ascii"), ({"ffmpeg_version": "N 125892-g1"}, "not_printable_ascii"),
])
def test_a_value_that_would_not_read_back_valid_is_refused(changes: dict, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        mark_text(**changes)


def test_the_reserved_name_is_matched_case_insensitively_and_refused_as_a_user_extra() -> None:
    assert model.MARK_NAME == "NORMA"
    for name in ("NORMA", "norma", "Norma", "nOrMa"):
        assert model.is_reserved_name(name) and model.extra_name_problem(name) == "reserved_extra_name"
        assert model.is_mark_key("id3v2", f"TXXX:{name}") and model.is_mark_key("vorbis", name)
    assert model.extra_name_problem("NORMAL") is None and not model.is_reserved_name("NORMAL")


# --- Equality with the pipeline's current composition -------------------------------------


@pytest.mark.parametrize(("gain", "measured", "origin", "ffmpeg_version"), [
    (gain_plan(), measurement(), "pcm_s24le", "N-125892-g406c5a37aa"),
    (gain_plan(action="copy", gain_db=None), measurement(-14.0, -2.0), "flac", "7.1"),
    (gain_plan(action="linear", gain_db=-3.0, target_lufs=-14.5, ceiling_dbtp=-1.5), measurement(-14.51, -1.6),
     "mp3", "N-1-g2"),
    (gain_plan(action="reduce", gain_db=-8.125, target_lufs=-23.0), measurement(-23.135, -9.0), "alac", "6.0.1"),
    (gain_plan(action="dynamic", target_lufs=-16.125), measurement(-16.0049, -0.0), "aac", "n7-8-gabc"),
    (gain_plan(action="limit", gain_db=12.345, limiter_reduction_db=0.005, effective_limiter_bound_db=0.005), measurement(-16.135, -0.0049),
     "pcm_s16be", "N-99999-gabcdef"),
])
def test_the_text_equals_the_pipeline_text(gain, measured, origin, ffmpeg_version) -> None:
    ours = model.processing_mark_value(gain, measured, origin=origin, ffmpeg_version=ffmpeg_version,
                                       run_date="2026-10-13", tool_version="0.1.0")
    theirs = pipeline.processing_mark_value(gain, measured, origin=origin, ffmpeg_version=ffmpeg_version,
                                            run_date="2026-10-13", tool_version="0.1.0")
    assert ours == theirs


def _pending_plan(target: str, *, wav_tags: str = "info+id3", id3_version: str = "2.3", revision: int = 1,
                  with_title: bool = True) -> MetadataPlan:
    fields = {"title": "Night Bus", "artist": "Test Artist"} if with_title else {}
    items: list[RawNativeItem] = []
    for store in mapping.target_stores(target, wav_tags):
        recipes = mapping.native_recipes(store, fields, id3_version=id3_version)
        items += write.prepare_store(store, recipes, id3_version=id3_version).items
    expected = tuple(items)
    store = mapping.primary_store(target, wav_tags) if mapping.target_stores(target, wav_tags) else None
    effective = mapping.project_fields(store, fields, id3_version=id3_version) if store else {}
    return MetadataPlan(
        plan_revision=revision, audio_basis="encode", conditional=False, source_native=(), requested_native=(),
        expected_native=expected, resolved={name: ResolvedValue(value, "cli", None) for name, value in fields.items()},
        effective={name: ResolvedValue(value, "cli", None) for name, value in effective.items()}, extras={},
        effective_extras={}, artwork=None, effective_artwork=(), statuses=(),
        native_statuses=losscheck.native_statuses((), expected, (), audio_basis="encode"), conflicts=(),
        non_tag_items=(), sidecar_requests=(), mark_state="pending", loss_policy_result="ok")


def _mark_items(target: str, mark: str, wav_tags: str, id3_version: str) -> tuple[RawNativeItem, ...]:
    """What a caller of ``model.finalise_mark`` prepares: the mark's items from WS-06's recipes and stores."""
    items: list[RawNativeItem] = []
    for store in mapping.target_stores(target, wav_tags):
        if mapping.extra_key(store, model.MARK_NAME, processing_mark=True) is None:
            continue
        recipes = mapping.native_recipes(store, {}, mark=mark, id3_version=id3_version)
        items.extend(write.prepare_store(store, recipes, id3_version=id3_version).items)
    return tuple(items)


PLAN_CASES = [
    ("flac", "info+id3", "2.3", 1), ("mp3", "info+id3", "2.3", 1), ("mp3", "info+id3", "2.4", 2),
    ("aiff", "info+id3", "2.4", 1), ("wav", "info+id3", "2.3", 3), ("alac", "info+id3", "2.3", 1),
    ("aac", "info+id3", "2.3", 1),
]


@pytest.mark.parametrize(("target", "wav_tags", "id3_version", "revision"), PLAN_CASES)
@pytest.mark.parametrize("with_title", [True, False])
def test_finalisation_equals_the_pipeline_revision(target, wav_tags, id3_version, revision, with_title) -> None:
    plan = _pending_plan(target, wav_tags=wav_tags, id3_version=id3_version, revision=revision, with_title=with_title)
    mark = mark_text()
    items = _mark_items(target, mark, wav_tags, id3_version)
    ours = model.finalise_mark(plan, mark, items)
    theirs = pipeline.finalise_mark(plan, mark, target=target, wav_tags=wav_tags, id3_version=id3_version)
    assert ours == theirs
    assert ours.plan_revision == revision + 1 and ours.mark_state == "final"
    rollup = ours.statuses[-1]
    assert (rollup.key, rollup.status, rollup.requested, rollup.effective) == ("extra:NORMA", "added", mark, mark)
    assert len(ours.expected_native) == len(plan.expected_native) + len(items)
    assert [row.status for row in ours.native_statuses[len(plan.native_statuses):]] == ["added"] * len(items)


def test_wav_info_only_gives_no_items_and_the_plan_is_not_finalised() -> None:
    plan = _pending_plan("wav", wav_tags="info")
    with pytest.raises(ValueError, match="needs a store"):
        model.finalise_mark(plan, mark_text(), _mark_items("wav", mark_text(), "info", "2.3"))
    with pytest.raises(ValueError, match="needs a store"):
        pipeline.finalise_mark(plan, mark_text(), target="wav", wav_tags="info", id3_version="2.3")


@pytest.mark.parametrize("state", ["final", "omitted", "not_storable"])
def test_a_plan_whose_mark_is_not_pending_is_returned_unchanged(state: str) -> None:
    plan = replace(_pending_plan("flac"), mark_state=state)
    items = _mark_items("flac", mark_text(), "info+id3", "2.3")
    assert model.finalise_mark(plan, mark_text(), items) is plan
    assert pipeline.finalise_mark(plan, mark_text(), target="flac", wav_tags="info+id3", id3_version="2.3") is plan


def test_an_invalid_mark_or_a_clashing_inventory_is_refused() -> None:
    plan = _pending_plan("flac")
    items = _mark_items("flac", mark_text(), "info+id3", "2.3")
    with pytest.raises(ValueError, match="not valid"):
        model.finalise_mark(plan, "v=1;tool=norma-0.1.0", items)
    with pytest.raises(ValueError, match="not valid"):
        model.finalise_mark(plan, "", items)
    done = model.finalise_mark(plan, mark_text(), items)
    with pytest.raises(ValueError, match="already holds"):
        model.finalise_mark(replace(done, mark_state="pending"), mark_text(), items)
    with pytest.raises(ValueError, match="already holds"):
        model.finalise_mark(plan, mark_text(), items + items)


def test_the_finalised_items_read_back_as_the_same_valid_mark(audio) -> None:
    """End to end on a real FLAC: the items finalise_mark adds are what the writer stores and the reader parses."""
    from norma.metadata.read import read_processing_mark, read_source_tags

    plan = _pending_plan("flac", with_title=False)
    mark = mark_text()
    final = model.finalise_mark(plan, mark, _mark_items("flac", mark, "info+id3", "2.3"))
    path = audio("flac")
    assert write.write_tags(path, final, target="flac").result.matched
    parsed = read_processing_mark(read_source_tags(path, "flac"))
    assert (parsed.validity, parsed.target_lufs, parsed.mode, parsed.origin_codec) == ("valid", -16.0, "limit",
                                                                                      "pcm_s24le")


def test_model_stays_free_of_mutagen_and_of_other_norma_modules() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(model.__file__).read_text(encoding="utf-8"))
    roots = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    roots |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert {name for name in roots if name.startswith(("norma", "mutagen"))} == {"norma.models"}
