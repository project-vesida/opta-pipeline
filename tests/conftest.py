"""Shared pytest fixtures for the opta-pipeline test suite.

Most of this suite renders synthetic frames through ``opta_pipeline.synth``,
which needs the optional ``opta-model`` dependency
(``pip install 'opta-pipeline[dev]'``).  Without it, the modules that import
opta-model or synth are not collected and the core-only suite still runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import opta_model  # noqa: F401

    HAVE_OPTA_MODEL = True
except ImportError:
    HAVE_OPTA_MODEL = False

_HERE = Path(__file__).parent
_NEEDS_MODEL = re.compile(r"opta_model|opta_pipeline\.synth|sensor_fixtures")

if not HAVE_OPTA_MODEL:
    collect_ignore = sorted(
        p.name
        for p in _HERE.glob("test_*.py")
        if _NEEDS_MODEL.search(p.read_text(encoding="utf-8"))
    )


def pytest_report_header(config: pytest.Config) -> str:
    if HAVE_OPTA_MODEL:
        return "opta-model: installed (full suite incl. synthetic frames)"
    return (
        f"opta-model: not installed; {len(collect_ignore)} synth-dependent test "
        "modules skipped (pip install 'opta-pipeline[dev]')"
    )


if HAVE_OPTA_MODEL:
    from opta_model.hardware import SensorConfig
    from sensor_fixtures import SENSOR_SMALL

    @pytest.fixture(name="sensor_small")
    def sensor_small_fixture() -> SensorConfig:
        """400×300-px test sensor with IMX585 pixel parameters."""
        return SENSOR_SMALL
