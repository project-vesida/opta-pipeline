"""Shared sensor constants for the opta-pipeline test suite.

Single source of truth for SENSOR_SMALL: a 400×300-pixel sensor that has the
same pixel pitch, QE, and noise parameters as IMX585_PRESET but generates
frames 20× faster in tests (AGENTS.md key gotcha on test sensor size).
"""

from __future__ import annotations

from dataclasses import replace

from opta_model.hardware import NodeConfig, SensorConfig
from opta_model.hardware_catalog import (
    build_node_from_profile,
    resolve_hardware_profile,
)

from opta_pipeline.config import PipelineConfig

PROFILE_PIPELINE_TOY = "pipeline_toy"


def get_test_node(profile: str | None = None) -> NodeConfig:
    """Return a :class:`NodeConfig` for pipeline tests and figure scripts."""
    if profile is not None:
        return build_node_from_profile(profile)
    return resolve_hardware_profile(PROFILE_PIPELINE_TOY)


_toy_node = resolve_hardware_profile(PROFILE_PIPELINE_TOY)

SENSOR_SMALL: SensorConfig = _toy_node.sensor
OPTICS_DEFAULT = _toy_node.optics


def per_frame_only_config() -> PipelineConfig:
    """Default config with the stacked track-before-detect search switched off.

    ``run_pipeline`` / ``PipelineStream.flush`` always run the blind stacked
    search when ``stacking.enabled`` — it is a parallel detection path, not a
    fallback for an empty per-frame result.  With the shipped defaults
    (``enabled`` + ``coarse_to_fine``) that blind search costs tens of seconds
    per invocation at test scale, which is pure waste for tests whose subject
    is the *per-frame* chain (linker association, partial passes, blind plate
    solving).  Those tests pass this config; tests that exercise the stacked
    path build their own.
    """
    cfg = PipelineConfig.default()
    return replace(cfg, stacking=replace(cfg.stacking, enabled=False))
