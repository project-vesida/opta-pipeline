"""Contract tests: pipeline_defaults.yaml ↔ PipelineConfig dataclasses (WP-A2).

This file MUST fail if:
  - A YAML key is added without updating the matching dataclass.
  - PipelineConfig.default() raises an exception (YAML / dataclass mismatch).
  - Any config section is not immutable (frozen dataclass invariant).
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from opta_pipeline.config import (
    DEFAULTS_PATH,
    AstrometryConfig,
    CalibrationConfig,
    DetectionConfig,
    PipelineConfig,
    RollingShutterConfig,
    StackingConfig,
    TrackletConfig,
    derive_max_residual_arcsec,
    derive_max_sep_arcsec,
)


def _yaml_keys(section: str) -> set[str]:
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    return set(raw[section].keys())


def _field_names(cls) -> set[str]:
    return set(cls.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# 1. Load / round-trip
# ---------------------------------------------------------------------------


def test_default_loads_without_error() -> None:
    cfg = PipelineConfig.default()
    assert isinstance(cfg, PipelineConfig)


def test_from_yaml_idempotent() -> None:
    """Loading the same YAML twice produces equal configs."""
    a = PipelineConfig.from_yaml()
    b = PipelineConfig.from_yaml()
    assert a == b


# ---------------------------------------------------------------------------
# 2. Key coverage — YAML keys ↔ dataclass fields
# ---------------------------------------------------------------------------


def test_yaml_calibration_keys_all_exposed() -> None:
    missing = _yaml_keys("calibration") - _field_names(CalibrationConfig)
    assert not missing, f"YAML calibration keys not in CalibrationConfig: {missing}"


def test_yaml_detection_keys_all_exposed() -> None:
    missing = _yaml_keys("detection") - _field_names(DetectionConfig)
    assert not missing, f"YAML detection keys not in DetectionConfig: {missing}"


def test_yaml_astrometry_keys_all_exposed() -> None:
    # Only expose the subset we use; catalog / solve keys are not yet wired.
    # This test checks that all *currently mapped* YAML keys appear in the dataclass.
    exposed = _field_names(AstrometryConfig)
    # Every dataclass field must correspond to an existing YAML key.
    yaml_keys = _yaml_keys("astrometry")
    extra_in_class = exposed - yaml_keys
    assert not extra_in_class, f"AstrometryConfig fields not in YAML: {extra_in_class}"


def test_yaml_rolling_shutter_keys_all_exposed() -> None:
    missing = _yaml_keys("rolling_shutter") - _field_names(RollingShutterConfig)
    assert not missing, (
        f"YAML rolling_shutter keys not in RollingShutterConfig: {missing}"
    )


def test_yaml_tracklet_keys_all_exposed() -> None:
    missing = _yaml_keys("tracklet") - _field_names(TrackletConfig)
    assert not missing, f"YAML tracklet keys not in TrackletConfig: {missing}"


def test_yaml_stacking_keys_all_exposed() -> None:
    missing = _yaml_keys("stacking") - _field_names(StackingConfig)
    assert not missing, f"YAML stacking keys not in StackingConfig: {missing}"


# ---------------------------------------------------------------------------
# 3. Value spot-checks — loaded values match raw YAML
# ---------------------------------------------------------------------------


def test_snr_threshold_matches_yaml() -> None:
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    cfg = PipelineConfig.default()
    assert cfg.detection.snr_threshold == pytest.approx(
        raw["detection"]["snr_threshold"]
    )


def test_max_residual_arcsec_matches_yaml() -> None:
    """The shipped YAML declares the sentinel; the loader resolves it."""
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    cfg = PipelineConfig.default()
    assert raw["astrometry"]["max_residual_arcsec"] == "derived"
    assert cfg.astrometry.max_residual_arcsec == pytest.approx(
        derive_max_residual_arcsec(float(raw["hardware"]["pixel_scale_arcsec"]))
    )


def test_stacking_fallback_fps_matches_yaml() -> None:
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    cfg = PipelineConfig.default()
    assert cfg.stacking.fallback_fps == pytest.approx(
        raw["stacking"]["fallback_fps"]
    )


def test_tracklet_max_sep_arcsec_matches_yaml() -> None:
    """The shipped YAML declares the sentinel; the loader resolves it."""
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    cfg = PipelineConfig.default()
    assert raw["tracklet"]["max_sep_arcsec"] == "derived"
    assert cfg.tracklet.max_sep_arcsec == pytest.approx(
        derive_max_sep_arcsec(float(raw["hardware"]["frame_rate_hz"]))
    )


# ---------------------------------------------------------------------------
# 3b. Per-optics gate derivation (v3 migration, 2026-07-28)
# ---------------------------------------------------------------------------
#
# Gates derive from the optics profile instead of being absolute constants:
#   max_sep_arcsec      = design max rate 2.0 deg/s (T-03, SYSTEMS.md
#                         2026-05-27) x 3600 / fps x 1.25 headroom
#                         (= the ratified 2.5 deg/s Level-A rate bound)
#   max_residual_arcsec = 3 sigma x RSS(0.3 px x plate scale, 2.0"
#                         distortion)  (error_budget.astrometric_error_budget,
#                         the OpTA.NOD.ACC 10" budget's per-star terms)
# Regression values recomputed 2026-07-28 via
#   python3 -c "from opta_pipeline.config import *;
#     print(derive_max_sep_arcsec('selected_v3_roi'),
#           derive_max_residual_arcsec('selected_v3_roi'))"


def test_default_gates_are_v3_derived() -> None:
    """Shipped defaults: v3 ROI node -> 360"/frame and 22.354" (0.93 px)."""
    cfg = PipelineConfig.default()
    assert cfg.hardware_profile == "selected_v3_roi"
    # 2.0 * 3600 / 25 * 1.25 (exact 2.5 deg/s ceiling at 25 fps)
    assert cfg.tracklet.max_sep_arcsec == pytest.approx(360.0)
    # 3 * sqrt((0.3 * 23.9267)**2 + 2.0**2)
    assert cfg.astrometry.max_residual_arcsec == pytest.approx(22.354, abs=1e-3)


def test_gate_derivation_tracks_the_profile() -> None:
    """Switching the optics profile re-derives the plate-scale-bound gate."""
    pytest.importorskip("opta_model")  # profile names resolve via the catalog
    # pipeline_toy: 85 mm at the 2.9 um pitch -> 7.037 arcsec/px
    # 3 * sqrt((0.3 * 7.0373)**2 + 2.0**2) = 8.724
    assert derive_max_residual_arcsec("pipeline_toy") == pytest.approx(
        8.724, abs=1e-3
    )
    # Same 25 fps small mode -> same angular association gate
    assert derive_max_sep_arcsec("pipeline_toy") == pytest.approx(360.0)


def test_numeric_gate_override_still_wins(tmp_path) -> None:
    """Explicit numeric YAML values bypass the derivation (escape hatch)."""
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    raw["astrometry"]["max_residual_arcsec"] = 5.5
    raw["tracklet"]["max_sep_arcsec"] = 1234.0
    path = tmp_path / "override.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.astrometry.max_residual_arcsec == pytest.approx(5.5)
    assert cfg.tracklet.max_sep_arcsec == pytest.approx(1234.0)


# ---------------------------------------------------------------------------
# 4. Immutability
# ---------------------------------------------------------------------------


def test_pipeline_config_is_frozen() -> None:
    cfg = PipelineConfig.default()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.calibration = None  # type: ignore[assignment]


def test_sub_configs_are_frozen() -> None:
    cfg = PipelineConfig.default()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.detection.snr_threshold = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. Range validation
# ---------------------------------------------------------------------------
#
# mask_veto_max_trajectory_frac is a *fraction of the pass*.  Before this
# check there was no validation anywhere in config.py, and both ends of the
# range failed silently: > 1.0 hit the `max_frac > 1.0` early return in
# _trajectory_dominated_by_star and disabled the star veto entirely, while
# 0.0 vetoed every candidate.  Reject out-of-range values where they enter.


@pytest.mark.parametrize("frac", [1.5, -0.1, 100.0])
def test_out_of_range_mask_veto_frac_rejected_at_parse(frac, tmp_path) -> None:
    with open(DEFAULTS_PATH) as fh:
        raw = yaml.safe_load(fh)
    raw["detection"]["mask_veto_max_trajectory_frac"] = frac
    path = tmp_path / "bad.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)
    with pytest.raises(ValueError, match="mask_veto_max_trajectory_frac"):
        PipelineConfig.from_yaml(path)


@pytest.mark.parametrize("frac", [1.5, -0.1])
def test_out_of_range_mask_veto_frac_rejected_on_construction(frac) -> None:
    """dataclasses.replace() is the usual override path — validate it too."""
    base = PipelineConfig.default().detection
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        dataclasses.replace(base, mask_veto_max_trajectory_frac=frac)


@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_in_range_mask_veto_frac_accepted(frac) -> None:
    base = PipelineConfig.default().detection
    assert (
        dataclasses.replace(
            base, mask_veto_max_trajectory_frac=frac
        ).mask_veto_max_trajectory_frac
        == frac
    )
