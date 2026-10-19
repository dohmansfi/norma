"""ADR-0021: command-scoped inequality, unchanged defaults and finite checks."""
from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import get_args

import pytest

from norma import models as m


COMMANDS = get_args(m.Command)
# Independent expected policy: do not derive the oracle from the exported set.
GUARDED_COMMANDS = frozenset({"normalise", "analyse", "tags_plan"})
DORMANT_COMMANDS = tuple(command for command in COMMANDS if command not in GUARDED_COMMANDS)
PAIRS = ((0.05, 0.3), (0.1, 0.3), (0.299, 0.3), (0.3, 0.3),
         (0.3, 0.3000001), (0.05, 0), (3, 3))
ABS = Path("/synthetic")


def test_skip_within_commands_is_one_immutable_public_policy() -> None:
    assert type(m.SKIP_WITHIN_COMMANDS) is frozenset
    assert m.SKIP_WITHIN_COMMANDS == GUARDED_COMMANDS
    assert m.SKIP_WITHIN_COMMANDS <= set(COMMANDS)
    assert "SKIP_WITHIN_COMMANDS" not in m.RUN_OPTION_FIELDS


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("tolerance,skip_within", PAIRS)
def test_skip_within_pair_applicability(command, tolerance, skip_within) -> None:
    if command in GUARDED_COMMANDS and skip_within > tolerance:
        with pytest.raises(ValueError, match="skip_within must not exceed tolerance"):
            m.RunOptions(command, ABS / "ffmpeg", ABS,
                         tolerance=tolerance, skip_within=skip_within)
    else:
        options = m.RunOptions(command, ABS / "ffmpeg", ABS,
                               tolerance=tolerance, skip_within=skip_within)
        assert options.tolerance == tolerance
        assert options.skip_within == skip_within


@pytest.mark.parametrize("command", DORMANT_COMMANDS)
@pytest.mark.parametrize("tolerance", (0.05, 0.1, 0.299))
def test_low_dormant_tolerance_preserves_default_and_provenance(command, tolerance) -> None:
    sources: dict[str, m.OptionSource] = {"tolerance": "cli", "skip_within": "default"}
    options = m.RunOptions(command, ABS / "ffmpeg", ABS, tolerance=tolerance,
                           value_sources=sources, explicit_fields=frozenset({"tolerance"}))
    assert options.skip_within == 0.3
    assert next(item for item in fields(m.RunOptions) if item.name == "skip_within").default == 0.3
    assert options.tolerance == tolerance
    assert isinstance(options.value_sources, m.FrozenMap)
    assert dict(options.value_sources) == sources
    assert options.explicit_fields == frozenset({"tolerance"})
    sources["skip_within"] = "config"
    assert options.value_sources["skip_within"] == "default"
    assert replace(options, ffmpeg=ABS / "other-ffmpeg").skip_within == 0.3
    with pytest.raises(FrozenInstanceError):
        options.skip_within = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("field", ("tolerance", "skip_within"))
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_pair_fields_remain_universally_finite(command, field, value) -> None:
    with pytest.raises(ValueError, match=field):
        m.RunOptions(command, ABS / "ffmpeg", ABS, **{field: value})


@pytest.mark.parametrize("tolerance", (0.05, 0.1, 0.299))
def test_direct_verify_construction_keeps_omitted_default(tolerance) -> None:
    options = m.RunOptions("verify", ABS / "ffmpeg", ABS, tolerance=tolerance)
    assert options.skip_within == 0.3
    assert options.value_sources == m.FrozenMap()
    assert options.explicit_fields == frozenset()
