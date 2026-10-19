"""C-13 taxonomy, exit precedence and exception families (TST-C-08 foundation)."""

from __future__ import annotations

import pytest

from norma import errors as e
from norma.models import REASON_CODES, Diagnostic


def test_table_covers_every_reason_exactly() -> None:
    assert tuple(e.REASONS) == REASON_CODES
    assert len(e.REASONS) == 47
    assert [spec.code for spec in e.REASONS.values() if not spec.active] == ["music_media_protected"]


@pytest.mark.parametrize("code", REASON_CODES)
def test_each_row_is_consistent(code: str) -> None:
    spec = e.reason_spec(code)
    expected = {"ok": {"ok"}, "skip": {"skipped"}, "warning": {"warn"}, "failure": {"failed"}}[spec.kind]
    if code == "already_within_tolerance":
        expected = {"ok", "skipped"}
    assert set(spec.statuses) == expected
    assert spec.streams[0] == "stdout"
    assert ("stderr" in spec.streams) == (spec.kind == "warning")
    if spec.kind == "failure":
        assert spec.exit_code == e.EXIT_FAILED
    elif code == "cancelled":
        assert spec.exit_code == e.EXIT_INTERRUPTED
    else:
        assert spec.exit_code == e.EXIT_OK


def test_counts_by_kind() -> None:
    kinds = [spec.kind for spec in e.REASONS.values()]
    assert (kinds.count("ok"), kinds.count("skip"), kinds.count("warning"), kinds.count("failure")) == (2, 9, 11, 25)


def test_unsupported_channels_is_a_skip_row() -> None:
    """ADR-0020 (D-056): more than two channels is skipped, exit 0, beside unsupported_codec."""
    spec = e.reason_spec("unsupported_channels")
    assert (spec.kind, spec.statuses, spec.exit_code, spec.streams, spec.active) == (
        "skip", ("skipped",), e.EXIT_OK, ("stdout",), True)
    codes = tuple(e.REASONS)
    assert codes.index("unsupported_channels") == codes.index("unsupported_codec") + 1
    assert e.exit_contribution("skipped", "unsupported_channels") == 0
    skip = e.SkipFile("unsupported_channels", "probe", "6 channels")
    assert skip.status == "skipped" and skip.diagnostic().code == "unsupported_channels"
    with pytest.raises(ValueError):
        e.FileFailure("unsupported_channels", "probe")


def test_warning_precedence_is_every_warning_once() -> None:
    warnings = {spec.code for spec in e.REASONS.values() if spec.kind == "warning"}
    assert set(e.WARNING_PRECEDENCE) == warnings and len(e.WARNING_PRECEDENCE) == len(warnings)
    assert e.primary_warning(["too_short", "lossy_origin", "reduce_shortfall"]) == "reduce_shortfall"
    assert e.primary_warning(["ok"]) is None


def test_settle_primary_promotes_only_ok() -> None:
    warn = (Diagnostic("too_short", "short", "measure"), Diagnostic("lossy_origin", "lossy", "resolve"))
    assert e.settle_primary("ok", "ok", warn) == ("warn", "lossy_origin")
    assert e.settle_primary("ok", "already_within_tolerance", warn) == ("warn", "lossy_origin")
    assert e.settle_primary("ok", "ok", ()) == ("ok", "ok")
    assert e.settle_primary("failed", "verification_peak", warn) == ("failed", "verification_peak")
    assert e.settle_primary("skipped", "already_processed", warn) == ("skipped", "already_processed")
    with pytest.raises(ValueError):
        e.settle_primary("ok", "decode_error", ())


def test_exit_precedence() -> None:
    assert e.combine_exit_codes([]) == 0
    assert e.combine_exit_codes([0, 1, 2]) == 2
    assert e.combine_exit_codes([1, 3, 2]) == 3
    assert e.combine_exit_codes([3, 130, 1]) == 130
    with pytest.raises(ValueError):
        e.combine_exit_codes([4])
    assert e.exit_contribution("failed", "timeout") == 1
    assert e.exit_contribution("skipped", "cancelled") == 130
    assert e.exit_contribution("warn", "too_short") == 0
    with pytest.raises(ValueError):
        e.exit_contribution("ok", "timeout")


def test_exception_families_reject_wrong_status() -> None:
    skip = e.SkipFile("not_audio", "probe", "no audio stream")
    assert skip.status == "skipped" and skip.reason == "not_audio"
    assert skip.diagnostic().code == "not_audio"
    e.SkipFile("already_within_tolerance", "plan")
    with pytest.raises(ValueError):
        e.SkipFile("decode_error", "decode")
    with pytest.raises(ValueError):
        e.FileFailure("not_audio", "probe")
    with pytest.raises(ValueError):
        e.FileFailure("nonsense", "probe")  # type: ignore[arg-type]
    assert e.FileFailure("probe_error", "probe").status == "failed"
    assert e.UsageError.exit_code == 2 and e.EnvironmentError_.exit_code == 3
    assert e.DoctorError is e.EnvironmentError_


def test_cancelled_escapes_worker_catch_all() -> None:
    assert not issubclass(e.Cancelled, Exception)
    with pytest.raises(e.Cancelled):
        try:
            raise e.Cancelled()
        except Exception:  # noqa: BLE001 - mirrors a worker's catch-all
            pytest.fail("Cancelled must not be caught as an ordinary exception")


def test_tool_errors_map_by_stage() -> None:
    assert e.tool_loss_reason("probe") == "probe_error"
    assert e.tool_loss_reason("measure") == "decode_error"
    assert e.tool_loss_reason("render") == "render_error"
    with pytest.raises(ValueError):
        e.tool_loss_reason("publish")
    missing = e.ToolMissing("gone", stderr_tail=("x",))
    assert missing.to_failure("render_error", "decode").reason == "decode_error"
    timeout = e.ToolTimeout("slow", deadline_s=60.0)
    assert timeout.to_failure("render_error", "render").reason == "timeout"
    failed = e.ToolFailed("bad", returncode=1, stderr_tail=("boom",))
    failure = failed.to_failure("render_error", "render")
    assert failure.reason == "render_error" and "boom" in (failure.detail or "")
