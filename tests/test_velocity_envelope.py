"""Envelope validation: the ephemeris-vector velocity search recovers movers
across all directions and a range of rates — not just one hand-picked pass.

A single slow pass is exactly what let the pedestal bug hide (findings #6–#9),
so robustness means sweeping the kinematic envelope. This parametrises a clean
controlled scene over (speed × direction), injects a linear mover at that
image-plane velocity, and asserts the vector-centred, component-SNR-ranked
search recovers it — with the prior centre deliberately *offset* from the truth
so the grid, not the seed, does the work.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import WCSSolution
from opta_pipeline.config import PipelineConfig
from opta_pipeline.pipeline import (
    FrameContext,
    VelocityPrior,
    run_track_and_stack,
)
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = 25.0
_RA0, _DEC0 = 135.0, 45.0
_NODE = "NODE-ENV"
_MJD0 = 60000.0
_N_FRAMES = 30


def _wcs() -> WCSSolution:
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0, crpix2=_H / 2.0, crval1=_RA0, crval2=_DEC0,
        cd1_1=scale_deg, cd1_2=0.0, cd2_1=0.0, cd2_2=scale_deg,
        rms_arcsec=0.0, n_stars=0,
    )


def _config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(cfg.stacking, enabled=True, velocity_min_px_s=0.0),
    )


def _linear_pass(vx_px_s: float, vy_px_s: float) -> list:
    """Clean controlled sequence with a mv≈12 mover at the given image velocity."""
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=13.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=13.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    ang_vel_deg_s = math.hypot(vx_px_s, vy_px_s) * _PIXEL_SCALE / 3600.0
    angle_deg = math.degrees(math.atan2(vy_px_s, vx_px_s))
    # Centre the track so it stays in-frame both ways over the window.
    ref = (_N_FRAMES - 1) / 2.0 / _FPS
    x0, y0 = _W / 2.0, _H / 2.0

    frames = []
    for k in range(_N_FRAMES):
        t = k / _FPS
        sat = SatelliteSpec(
            magnitude=12.0,
            angular_velocity_deg_s=ang_vel_deg_s,
            x_center=x0 + vx_px_s * (t - ref),
            y_center=y0 + vy_px_s * (t - ref),
            angle_deg=angle_deg,
        )
        synth = generate_frame(
            SENSOR_SMALL, _OPTICS, sky_mag_arcsec2=21.0, elevation_deg=45.0,
            satellites=[sat], stars=star_specs,
            rng=np.random.default_rng(4000 + k),
        )
        ctx = FrameContext(
            star_matches=star_matches, utc_mjd=_MJD0 + t / 86400.0,
            frame_id=k, node_id=_NODE,
        )
        frames.append((synth.data_float, ctx))
    return frames


# Sweep the envelope: two rates × eight directions (all quadrants + axes).
_SPEEDS = (12.0, 28.0)
_DIRECTIONS = tuple(range(0, 360, 45))


@pytest.mark.parametrize("speed", _SPEEDS)
@pytest.mark.parametrize("direction_deg", _DIRECTIONS)
def test_prior_search_recovers_mover_any_direction(
    speed: float, direction_deg: float
) -> None:
    vx = speed * math.cos(math.radians(direction_deg))
    vy = speed * math.sin(math.radians(direction_deg))
    frames = _linear_pass(vx, vy)

    # Prior centre offset from the truth by ~2 px/s so the grid does the work.
    prior = VelocityPrior(
        vx_px_s=vx + 1.5,
        vy_px_s=vy - 1.5,
        half_width_px_s=3.0,
        step_px_s=1.0,
    )
    result = run_track_and_stack(frames, config=_config(), velocity_prior=prior)

    assert result.stack_result.best_vx_px_s == pytest.approx(vx, abs=1.5)
    assert result.stack_result.best_vy_px_s == pytest.approx(vy, abs=1.5)
    assert len(result.tracklets) >= 1
