"""CR-078: B-009 refusal reason and cleanup, asserted as the accepted limitation (TK-0719).

``test_dynamic_long.py::test_cr_review_white_noise_192_khz_mono_dynamic_mp3_publishes`` remains
the strict expected failure for the TARGET behaviour (a published, on-target file). It cannot say
whether the refusal that happens today is the safe one. This module pins that: dynamic mode on
the same 192 kHz white-noise source is refused ``verification_loudness`` (exit 1) with nothing
published, no candidate or temporary file left, the source untouched and the report truthful.

This is not a pass for B-009 and not a waiver: when the defect is fixed, the strict xfail starts
passing and fails the suite, and the lead removes it and this module together (backlog B-009).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

RATE, FRAMES = 192000, 5760137  # the xfail's repro: 30.0007 s, mono


def white_noise(path: Path) -> Path:
    samples = 0.03 * np.random.default_rng(1023).normal(size=(FRAMES, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, RATE, subtype="PCM_24")
    return path


def test_cr078_b009_dynamic_white_noise_is_refused_loudness_with_nothing_left_behind(norma, tools, tmp_path):
    source = white_noise(tmp_path / "in" / "noise.wav")
    before = tools.snapshot(tmp_path)
    report = tmp_path / "report.json"
    run = norma("normalise", source, "-o", tmp_path / "out", "--peak-mode", "dynamic", "-f", "mp3",
                "--report", report)
    assert run.code == 1, run.stdout + run.stderr
    (row,) = run.files
    assert row[0] == "FAIL" and row[6] == "dynamic", row
    assert "verification_loudness at verify" in row[-1] and "no output written" in row[-1], row
    assert row[9] == ""  # no output path is claimed
    summary = run.summary
    assert (summary["ok"], summary["warn"], summary["skipped"], summary["failed"]) == ("0", "0", "0", "1")
    assert summary["not_started"] == "0" and summary["run_errors"] == "0"
    # The report tells the same story, with nothing published or retained.
    (record,) = json.loads(report.read_text())["outcomes"]
    assert record["status"] == "failed" and record["reason"] == "verification_loudness"
    assert record["execution"] == "not_written" and record["mode"] == "dynamic"
    assert record["published_paths"] == [] and record["retained_failed_path"] is None
    assert record["out_lufs"] < -21.0, "the refused candidate really is outside -16 +/- 0.5 (B-009: about -22.1)"
    # Cleanup: the output folder was never created, no candidate or temporary file remains, and
    # the only addition anywhere in the sandbox is the report.
    after = tools.snapshot(tmp_path)
    changes = tools.difference(before, after)
    assert not changes["removed"] and not changes["changed"], changes
    assert changes["added"] == {"report.json"}, changes
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.rglob(".norma-*")) and list((tmp_path / "home" / "tmp").iterdir()) == []
