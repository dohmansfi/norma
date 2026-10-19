"""Consume the WS-07 catalogue without changing the shared conftest."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "norma_audio_test_fixtures", Path(__file__).parents[2] / "fixtures" / "make_fixtures.py",
)
assert _spec is not None and _spec.loader is not None
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
fixture_path = fixtures.fixture_path
