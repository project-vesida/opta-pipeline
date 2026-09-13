"""End-to-end tests for the likelihood (ψ/φ) scorer in run_track_and_stack.

This is the pipeline-level closure of the stacked-SNR emission bug
(TODO.md `pipeline.py:489`; docs/endgame-plan.md first-light prerequisite):
with the trials-corrected threshold on the calibrated SNR map, pure noise
— even with stars and hot pixels present — must produce **zero** tracklets,
while a faint mover is still recovered into I-02 tracklets.

Geometry mirrors test_pipeline_stacking.py (SENSOR_SMALL, supplied star
matches, mv≈13 source at 200 px/s).
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
from opta_pipeline.pipeline import FrameContext, run_track_and_stack
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = SENSOR_SMALL.frame_rate_hz
_N_FRAMES = 40
_VX_PX_S = 200.0
_NODE = "NODE-LSTACK"
_MJD0 = 60000.0


def _wcs() -> WCSSolution:
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=135.0,
        crval2=45.0,
        cd1_1=scale_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=scale_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )


def _config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=200.0,
            velocity_step_px_s=200.0,
            # Coarse-to-fine seeding depth (see test_pipeline_stacking).
            # Stage-1 seeds a track only if its per-window matched-filter SNR
            # clears the window noise floor sqrt(2 ln(W*H)) ~= 4.84 sigma.
            # This fast mv13 mover (200 px/s -> ~8 px within-exposure trail)
            # is a per-frame streak, so the synth-PSF-matched round filter
            # (psf_sigma_px ~= 0.85 = FWHM 2.0 / 2.355, the correct value)
            # captures less of the trail per window than the old too-wide
            # 1.5 px kernel; 9-frame (0.36 s) seeds drop below the floor
            # (all-zeros stack).  Deepen the seed window (stacking-depth lever,
            # design rule 10) to 15 frames (0.60 s) to restore ~1 sigma margin.
            seed_window_s=0.60,
        ),
    )


def _u_star(cfg: PipelineConfig) -> float:
    """The trials-corrected threshold the pipeline applies to the SNR map."""
    n_grid = len(
        np.arange(
            -cfg.stacking.velocity_max_px_s,
            cfg.stacking.velocity_max_px_s + cfg.stacking.velocity_step_px_s * 0.5,
            cfg.stacking.velocity_step_px_s,
        )
    )
    n_trials = _W * _H * n_grid * n_grid
    return math.sqrt(2.0 * math.log(n_trials)) + cfg.stacking.far_margin_sigma


def _frame_pairs(with_satellite: bool, seed0: int) -> list:
    """Star field (+ optional mv13 mover) frame sequence with contexts."""
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=14.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    angular_velocity_deg_s = _VX_PX_S * _PIXEL_SCALE / 3600.0
    reference_s = ((_N_FRAMES - 1) / 2.0) / _FPS

    pairs = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        sats = []
        if with_satellite:
            sats.append(
                SatelliteSpec(
                    magnitude=13.0,
                    angular_velocity_deg_s=angular_velocity_deg_s,
                    x_center=_W / 2.0 + _VX_PX_S * (t_s - reference_s),
                    y_center=_H / 2.0,
                    angle_deg=0.0,
                )
            )
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=sats,
            stars=star_specs,
            rng=np.random.default_rng(seed0 + frame_id),
        )
        ctx = FrameContext(
            star_matches=star_matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        pairs.append((synth.data_float, ctx))
    return pairs


@pytest.fixture(scope="module")
def mover_result():
    return run_track_and_stack(_frame_pairs(True, 9100), config=_config())


class TestMoverRecovery:
    """A faint mover survives the trials-corrected gate into tracklets."""

    def test_velocity_recovered(self, mover_result) -> None:
        # Coarse-to-fine resolves velocity to the full-pass criterion step
        # PSF_fwhm / T ≈ 2.3 px/s here — not to a grid node (there is none).
        fine_step = 2.355 * _config().stacking.psf_sigma_px * _FPS / (_N_FRAMES - 1)
        assert mover_result.stack_result.best_vx_px_s == pytest.approx(
            _VX_PX_S, abs=1.5 * fine_step
        )
        assert mover_result.stack_result.best_vy_px_s == pytest.approx(
            0.0, abs=1.5 * fine_step
        )

    def test_snr_map_semantics(self, mover_result) -> None:
        """Likelihood mode: stacked is the SNR map, noise_rms is 1.0."""
        assert mover_result.stack_result.noise_rms == 1.0
        assert mover_result.stack_result.peak_snr > _u_star(_config())

    def test_detections_above_calibrated_threshold(self, mover_result) -> None:
        u_star = _u_star(_config())
        assert mover_result.stacked_detections
        assert all(d.snr >= u_star for d in mover_result.stacked_detections)

    def test_tracklets_emitted(self, mover_result) -> None:
        assert len(mover_result.tracklets) >= 1
        best = max(mover_result.tracklets, key=lambda t: len(t.points))
        assert best.node_id == _NODE
        expected_rate = _VX_PX_S * _PIXEL_SCALE
        assert abs(best.ra_rate_arcsec_s) == pytest.approx(expected_rate, rel=0.10)


class TestNoiseOnlyEmission:
    """THE regression for the pipeline.py:489 fake-tracklet bug (F6)."""

    def test_noise_only_emits_no_tracklets(self) -> None:
        """Stars + sky + read noise, no mover: zero detections, zero
        tracklets.  Under the classic scorer this scene could mint an I-02
        tracklet from one coherent noise peak; the calibrated map plus the
        u* gate makes that statistically impossible at the configured FAR
        margin."""
        result = run_track_and_stack(_frame_pairs(False, 9500), config=_config())
        assert result.stacked_detections == ()
        assert result.tracklets == ()

    # Moved to the nightly full run 2026-07-29 (runtime triage): 177 s, ~10 %
    # of the fast suite's 1814 s wall (measured via `python3 -m pytest
    # opta-pipeline/tests/ -q -m "not slow" --durations=40`) for a second
    # 40-frame blind coarse-to-fine over the same source-free scene family the
    # sibling test above already searches.  Both halves of the property stay
    # under PR CI: the *mechanism* — temporal-median subtraction removes a
    # static scene while the mover survives — in
    # test_likelihood_stack.py::TestTemporalMedian (unit level, sub-second),
    # and the *no-spurious-tracklets-from-static-defects* guarantee end to end
    # in test_pipeline_realistic.py::TestRealisticEndToEnd
    # ::test_no_spurious_long_tracklets_from_static_defects
    # (hot pixels + bias FPN through the full chain).  The expensive stacked-FP
    # gate itself also stays fast via test_noise_only_emits_no_tracklets above;
    # only the hot-pixel *variant* of that scene runs nightly.
    @pytest.mark.slow
    def test_hot_pixel_does_not_form_tracklet(self) -> None:
        """A static hot pixel (the classic scorer's known failure, see
        test_pipeline_stacking.py refinement test) is removed by
        temporal-median subtraction before stacking — no detections."""
        pairs = _frame_pairs(False, 9700)
        contaminated = []
        for raw, ctx in pairs:
            data = np.array(raw, copy=True)
            data[151, 251] += 40.0  # hot pixel well above single-frame noise
            contaminated.append((data, ctx))
        result = run_track_and_stack(contaminated, config=_config())
        assert result.stacked_detections == ()
        assert result.tracklets == ()
