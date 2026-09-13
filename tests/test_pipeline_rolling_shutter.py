"""Closed-loop rolling-shutter correction in the stacking path (T-03).

The generator now injects the rolling-shutter readout bias (a moving source is
displaced by its per-row exposure timing); ``run_track_and_stack`` now removes
it (``apply_rolling_shutter_correction`` on the stacked peak).  These tests
close the loop: a fast target imaged **off the reference row** accrues a
constant along-track bias that pushes the recovered RA/Dec past the 10″
OpTA.NOD.ACC budget with the correction off, and back well under it with the
correction on.

A second test guards the architecture's subtlety: for a target imaged **on**
the reference row the shift-and-add velocity fit already absorbs the readout
bias, so the correction must be a no-op there (never harmful).

Truth WCS / star matches are supplied so the blind solve and distortion are out
of the loop — this isolates the rolling-shutter term.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL as _SENSOR

from opta_pipeline.astrometry import WCSSolution, pixels_to_radec
from opta_pipeline.config import PipelineConfig
from opta_pipeline.pipeline import FrameContext, run_track_and_stack
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at
from opta_pipeline.synth.rolling_shutter import RollingShutterReadout

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(_SENSOR.pixel_size_um, _OPTICS.focal_length_mm)
_W, _H = _SENSOR.resolution_h, _SENSOR.resolution_v
_FPS = _SENSOR.frame_rate_hz
_RA0, _DEC0 = 135.0, 45.0
_COS_DEC = math.cos(math.radians(_DEC0))
_MJD0 = 60000.0
_NODE = "NODE-RS"

_REF_ROW = _H / 2.0
_ROW_READOUT_US = 130.0
_N_FRAMES = 24
_VX_PX_S = 120.0  # on the stacking velocity grid
_NOD_ACC_ARCSEC = 10.0
# Off-reference row for the biased target (see offrow_frames fixture).
_OFFROW = 30.0


def _true_wcs() -> WCSSolution:
    ps = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0, crpix2=_H / 2.0, crval1=_RA0, crval2=_DEC0,
        cd1_1=ps, cd1_2=0.0, cd2_1=0.0, cd2_2=ps, rms_arcsec=0.0, n_stars=0,
    )


def _sky_err(ra: float, dec: float, ra_t: float, dec_t: float) -> float:
    return math.hypot((ra - ra_t) * _COS_DEC * 3600.0, (dec - dec_t) * 3600.0)


def _build_frames(cy0: float) -> list:
    """A horizontal fast target at constant row ``cy0``, with readout injected."""
    true = _true_wcs()
    star_matches = star_field_at(true, _W, _H, mag_limit=14.0)
    catalog = catalog_stars_in_fov(true, _W, _H, mag_limit=11.0)
    stars = StarSpec.from_catalog(true, catalog)
    rs = RollingShutterReadout(row_readout_us=_ROW_READOUT_US, reference_row=_REF_ROW)
    avd = _VX_PX_S * _PIXEL_SCALE / 3600.0
    reference_s = ((_N_FRAMES - 1) / 2.0) / _FPS
    frames = []
    for fid in range(_N_FRAMES):
        t_s = fid / _FPS
        cx = _W / 2.0 + _VX_PX_S * (t_s - reference_s)
        synth = generate_frame(
            _SENSOR, _OPTICS, sky_mag_arcsec2=21.0,
            satellites=[SatelliteSpec(magnitude=8.0, angular_velocity_deg_s=avd,
                                      x_center=cx, y_center=cy0, angle_deg=0.0)],
            stars=stars, rolling_shutter=rs, rng=np.random.default_rng(7000 + fid),
        )
        ctx = FrameContext(star_matches=star_matches, utc_mjd=_MJD0 + t_s / 86400.0,
                           frame_id=fid, node_id=_NODE)
        frames.append((synth.data_float, ctx))
    return frames


def _config(correction: bool) -> PipelineConfig:
    base = PipelineConfig.default()
    return replace(
        base,
        detection=replace(base.detection, min_streak_pixels=3),
        tracklet=replace(base.tracklet, linear_fit_residual_arcsec=30.0),
        stacking=replace(base.stacking, enabled=True, velocity_max_px_s=240.0,
                         velocity_step_px_s=120.0),
        rolling_shutter=replace(base.rolling_shutter, enabled=True,
                                row_readout_us=_ROW_READOUT_US, reference_row=_REF_ROW),
        astrometry=replace(base.astrometry, rolling_shutter_correction=correction),
    )


def _tracklet_errors(frames: list, cy0: float, correction: bool) -> list[float]:
    """Great-circle errors of the recovered tracklet vs the true (readout-free)
    sky track, at the given correction setting."""
    true = _true_wcs()
    reference_s = ((_N_FRAMES - 1) / 2.0) / _FPS
    result = run_track_and_stack(frames, config=_config(correction))
    tracklet = max(result.tracklets, key=lambda t: len(t.points))
    errs = []
    for pt in tracklet.points:
        t_s = (pt.utc_mjd - _MJD0) * 86400.0
        gx = _W / 2.0 + _VX_PX_S * (t_s - reference_s)
        ra_t, dec_t = pixels_to_radec(true, gx, cy0)
        errs.append(_sky_err(pt.ra_deg, pt.dec_deg, ra_t, dec_t))
    return errs


@pytest.fixture(scope="module")
def offrow_frames() -> list:
    # Row 30 is 120 rows above the reference row → a constant, budget-breaking
    # bias (≈ 1.9 px ≈ 13″ at the 2.9 µm datasheet pitch).  Was row 235
    # (85 rows off) under the stale 3.76 µm geometry (MA-002); at 7.04″/px
    # that offset only reached ≈ 9.5″ and no longer exceeded the 10″ budget.
    return _build_frames(cy0=_OFFROW)


@pytest.fixture(scope="module")
def centred_frames() -> list:
    return _build_frames(cy0=_REF_ROW)


class TestRollingShutterCorrectionInStack:
    def test_uncorrected_off_row_target_exceeds_budget(
        self, offrow_frames: list
    ) -> None:
        """Without the T-03 step the readout bias blows the 10″ budget."""
        errs = _tracklet_errors(offrow_frames, _OFFROW, correction=False)
        assert max(errs) > _NOD_ACC_ARCSEC

    def test_correction_recovers_within_budget(self, offrow_frames: list) -> None:
        """With it on, the same target is recovered to well within budget."""
        on = _tracklet_errors(offrow_frames, _OFFROW, correction=True)
        off = _tracklet_errors(offrow_frames, _OFFROW, correction=False)
        assert max(on) < _NOD_ACC_ARCSEC
        assert float(np.mean(on)) < 1.0
        # The correction is the reason: a large reduction in along-track error.
        assert float(np.mean(on)) < 0.25 * float(np.mean(off))

    def test_correction_is_harmless_on_reference_row(
        self, centred_frames: list
    ) -> None:
        """A target on the reference row is already unbiased (the velocity fit
        absorbs the readout term), so the correction must not degrade it."""
        on = _tracklet_errors(centred_frames, _REF_ROW, correction=True)
        off = _tracklet_errors(centred_frames, _REF_ROW, correction=False)
        assert float(np.mean(on)) < _NOD_ACC_ARCSEC
        assert float(np.mean(on)) == pytest.approx(float(np.mean(off)), abs=0.5)
