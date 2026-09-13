"""Forced per-frame photometry (F8): honest positions, SNR, and fluxes.

Closes the second half of the pipeline.py:489 bug (TODO.md): expanded
per-frame detections were model-generated positions carrying the repeated
stacked SNR, so the linker's linear-fit QC validated its own input.  With
forced measurement, every emitted detection is measured on its own frame:
matched-filter SNR + ML flux at the predicted position, sub-pixel centroid
whenever the local SNR clears the single-frame threshold.

Unit tests validate the measurement against analytic matched-filter
predictions; the pipeline test proves the circularity is gone (measured
positions scatter around the model line instead of lying on it exactly)
while tracklet QC still passes, and that per-frame fluxes form a usable
light curve.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import WCSSolution
from opta_pipeline.config import PipelineConfig
from opta_pipeline.likelihood import forced_measurement, gaussian_psf_kernel
from opta_pipeline.pipeline import FrameContext, run_track_and_stack
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at

_PSF_SIGMA = 1.2


def _gaussian_frame(
    h: int, w: int, xc: float, yc: float, amplitude: float, noise: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (xx - xc) ** 2 + (yy - yc) ** 2
    return amplitude * np.exp(-r2 / (2.0 * _PSF_SIGMA**2)) + rng.normal(
        0.0, noise, (h, w)
    )


class TestForcedMeasurementUnit:
    """Single-frame measurement against analytic predictions."""

    _KERNEL = gaussian_psf_kernel(_PSF_SIGMA)

    def test_bright_source_refined_centroid_and_flux(self) -> None:
        """Sub-pixel position within 0.15 px; flux within 15 % of F_tot;
        SNR at the matched-filter prediction within 20 %."""
        amp, noise = 30.0, 1.0
        x_true, y_true = 40.37, 25.81
        frame = _gaussian_frame(64, 64, x_true, y_true, amp, noise, seed=3)
        m = forced_measurement(
            frame, noise, self._KERNEL, x_true + 0.8, y_true - 0.6,
            search_radius_px=2.4, refine_snr_min=3.0,
        )
        assert m.refined
        assert m.x_px == pytest.approx(x_true, abs=0.15)
        assert m.y_px == pytest.approx(y_true, abs=0.15)
        flux_tot = amp * 2.0 * np.pi * _PSF_SIGMA**2
        assert m.flux_e == pytest.approx(flux_tot, rel=0.15)
        pred_snr = flux_tot * float(
            np.sqrt((self._KERNEL * self._KERNEL).sum())
        ) / noise
        assert m.snr == pytest.approx(pred_snr, rel=0.2)

    def test_faint_frame_passes_prediction_through(self) -> None:
        """Below the refine threshold: predicted position unchanged,
        honest low SNR — no fabricated astrometry."""
        rng = np.random.default_rng(11)
        frame = rng.normal(0.0, 1.0, (64, 64))
        m = forced_measurement(
            frame, 1.0, self._KERNEL, 30.25, 33.75,
            search_radius_px=2.4, refine_snr_min=3.0,
        )
        assert not m.refined
        assert m.x_px == 30.25 and m.y_px == 33.75
        assert abs(m.snr) < 3.0

    def test_out_of_frame_prediction_is_null_measurement(self) -> None:
        frame = np.zeros((32, 32))
        m = forced_measurement(frame, 1.0, self._KERNEL, -50.0, -50.0)
        assert not m.refined
        assert m.snr == 0.0 and m.flux_e == 0.0

    def test_marginal_snr_peak_not_refined_at_default_gate(self) -> None:
        """A ~4-5σ local peak is below the default refinement gate: the
        prediction is passed through instead of snapping to a centroid
        whose position is noise-dominated (findings #11/#12)."""
        amp, noise = 2.0, 1.0  # matched SNR ≈ 4, below the default gate
        x_true, y_true = 40.37, 25.81
        frame = _gaussian_frame(64, 64, x_true, y_true, amp, noise, seed=5)
        m = forced_measurement(
            frame, noise, self._KERNEL, x_true + 0.4, y_true - 0.3,
            search_radius_px=3.0,
        )
        assert not m.refined
        assert m.x_px == x_true + 0.4 and m.y_px == y_true - 0.3
        assert m.snr > 2.0  # honest forced photometry still reported


# ── Pipeline-level: circularity is gone, light curve exists ────────────────

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = SENSOR_SMALL.frame_rate_hz
_N_FRAMES = 30
_VX_PX_S = 200.0
_MAG = 10.5  # bright: per-frame matched SNR ≫ refine threshold everywhere


def _wcs() -> WCSSolution:
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0, crpix2=_H / 2.0, crval1=135.0, crval2=45.0,
        cd1_1=scale_deg, cd1_2=0.0, cd2_1=0.0, cd2_2=scale_deg,
        rms_arcsec=0.0, n_stars=0,
    )


def _config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            coarse_to_fine=False,  # expansion under test, not the search
            velocity_max_px_s=200.0,
            velocity_step_px_s=200.0,
        ),
    )


@pytest.fixture(scope="module")
def bright_result():
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)
    ang_v = _VX_PX_S * _PIXEL_SCALE / 3600.0
    ref_s = ((_N_FRAMES - 1) / 2.0) / _FPS
    pairs = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        sat = SatelliteSpec(
            magnitude=_MAG,
            angular_velocity_deg_s=ang_v,
            x_center=_W / 2.0 + _VX_PX_S * (t_s - ref_s),
            y_center=_H / 2.0,
            angle_deg=0.0,
        )
        synth = generate_frame(
            SENSOR_SMALL, _OPTICS, sky_mag_arcsec2=21.0, elevation_deg=45.0,
            satellites=[sat], stars=star_specs,
            rng=np.random.default_rng(4200 + frame_id),
        )
        pairs.append(
            (
                synth.data_float,
                FrameContext(
                    star_matches=star_matches,
                    utc_mjd=60000.0 + t_s / 86400.0,
                    frame_id=frame_id,
                    node_id="NODE-F8",
                ),
            )
        )
    return run_track_and_stack(pairs, config=_config())


class TestPipelineForcedPhotometry:
    """Emitted detections are per-frame measurements, not model echoes."""

    def test_per_frame_snr_varies_and_is_not_the_stacked_value(
        self, bright_result
    ) -> None:
        snrs = [
            d.detection.snr
            for fd in bright_result.frame_detections
            for d in fd.detections
        ]
        assert len(snrs) >= 20
        stacked = bright_result.stack_result.peak_snr
        # Old behaviour: every value == stacked SNR.  Now: per-frame values
        # with real noise scatter, each well below the coadded statistic.
        assert float(np.std(snrs)) > 0.1
        assert max(snrs) < stacked

    def test_positions_scatter_measurably_but_within_budget(
        self, bright_result
    ) -> None:
        """The circularity kill: residuals of a linear fit to the emitted
        sky positions are nonzero (measured centroids, not the model line)
        yet small (bright source ⇒ sub-pixel astrometric noise)."""
        fds = bright_result.frame_detections
        ts = np.array([fd.utc_mjd for fd in fds]) * 86400.0
        ra = np.array([fd.detections[0].ra_deg for fd in fds]) * 3600.0
        dec = np.array([fd.detections[0].dec_deg for fd in fds]) * 3600.0
        ts = ts - ts.mean()

        resid_ra = ra - np.polyval(np.polyfit(ts, ra, 1), ts)
        rms_ra = float(np.sqrt(np.mean(resid_ra**2)))
        # Along-track (RA here): measured, nonzero — the circularity kill.
        # The source trails 8 px/frame at 200 px/s, so the point-kernel
        # centroid wanders along the streak (assessment F4 residual; the
        # streak kernel is the open Phase 2 extension) — bounded by the
        # linker budget, which the tracklet must still clear.
        assert rms_ra > 1e-4
        assert rms_ra < _config().tracklet.linear_fit_residual_arcsec

        # Cross-track (Dec here): no trailing component — clean sub-half-
        # pixel astrometric scatter for a bright source.
        resid_dec = dec - np.polyval(np.polyfit(ts, dec, 1), ts)
        rms_dec = float(np.sqrt(np.mean(resid_dec**2)))
        assert 1e-4 < rms_dec < 0.5 * _PIXEL_SCALE

    def test_light_curve_fluxes_present_and_consistent(self, bright_result) -> None:
        fluxes = np.array(
            [d.detection.flux_e for fd in bright_result.frame_detections
             for d in fd.detections]
        )
        assert np.all(np.isfinite(fluxes))
        assert np.all(fluxes > 0.0)
        # Constant-magnitude source: flux scatter ≪ mean (light curve is flat).
        assert float(np.std(fluxes) / np.mean(fluxes)) < 0.35

    def test_tracklet_still_passes_qc(self, bright_result) -> None:
        assert len(bright_result.tracklets) >= 1
        best = max(bright_result.tracklets, key=lambda t: len(t.points))
        expected_rate = _VX_PX_S * _PIXEL_SCALE
        assert abs(best.ra_rate_arcsec_s) == pytest.approx(expected_rate, rel=0.10)


# ── Regression (e2e findings #11/#12): marginal per-frame SNR must not let
#    refinement snap onto noise peaks and destroy the track astrometry ──────

_MAG_FAINT = 13.0  # per-frame matched SNR ≈ 2, stacked ≈ 12σ over 30 frames


def _faint_config() -> PipelineConfig:
    """Config for the marginal-SNR regression scene.

    The linker gate is 10″ (~1.1 px at this fixture's 9.1″/px): the old
    behaviour (refine gate = detection.snr_threshold = 3σ) snaps per-frame
    centroids onto noise peaks inside the 3-px search disc, inflating the
    track scatter to ~1.4 px RMS ≈ 13″ — over the gate, so the linker
    silently drops a ~12σ stacked detection.  With the decoupled 5σ gate
    (stacking.refine_snr_min) faint frames fall back to the stack's velocity
    solution and the tracklet survives with clean astrometry.
    """
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=10.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            coarse_to_fine=False,  # expansion under test, not the search
            velocity_max_px_s=200.0,
            velocity_step_px_s=200.0,
        ),
    )


@pytest.fixture(scope="module")
def faint_result():
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)
    ang_v = _VX_PX_S * _PIXEL_SCALE / 3600.0
    ref_s = ((_N_FRAMES - 1) / 2.0) / _FPS
    pairs = []
    truth = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        xc = _W / 2.0 + _VX_PX_S * (t_s - ref_s)
        yc = _H / 2.0
        truth.append((xc, yc))
        sat = SatelliteSpec(
            magnitude=_MAG_FAINT,
            angular_velocity_deg_s=ang_v,
            x_center=xc,
            y_center=yc,
            angle_deg=0.0,
        )
        synth = generate_frame(
            SENSOR_SMALL, _OPTICS, sky_mag_arcsec2=21.0, elevation_deg=45.0,
            satellites=[sat], stars=star_specs,
            rng=np.random.default_rng(4200 + frame_id),
        )
        pairs.append(
            (
                synth.data_float,
                FrameContext(
                    star_matches=star_matches,
                    utc_mjd=60000.0 + t_s / 86400.0,
                    frame_id=frame_id,
                    node_id="NODE-F8-FAINT",
                ),
            )
        )
    return run_track_and_stack(pairs, config=_faint_config()), truth


class TestMarginalSnrRefinementRegression:
    """Findings #11/#12: refinement at the detection threshold corrupted
    marginal-target astrometry until the linker rejected solidly
    stack-detected objects.  These pin the decoupled gate + predicted-
    position fallback end-to-end (stack → F8 expansion → linker QC)."""

    def test_refine_gate_decoupled_and_above_detection_threshold(self) -> None:
        cfg = PipelineConfig.default()
        assert cfg.stacking.refine_snr_min >= 5.0
        assert cfg.stacking.refine_snr_min > cfg.detection.snr_threshold

    def test_scene_is_marginal_per_frame(self, faint_result) -> None:
        """Setup guard: the scene must sit in the low per-frame SNR regime
        (else this file no longer tests the regression) while the stacked
        detection is solid."""
        result, _truth = faint_result
        assert len(result.stacked_detections) >= 1
        assert result.stack_result.peak_snr > 8.0
        snrs = [
            d.detection.snr
            for fd in result.frame_detections
            for d in fd.detections
        ]
        assert 0.5 < float(np.median(snrs)) < 4.5

    def test_marginal_track_survives_linker_qc(self, faint_result) -> None:
        """The regression: a ~12σ stacked detection must not be silently
        dropped because per-frame refinement inflated the O-C RMS."""
        result, _truth = faint_result
        assert len(result.tracklets) >= 1
        best = max(result.tracklets, key=lambda t: len(t.points))
        assert len(best.points) >= 20

    def test_track_oc_rms_not_noise_inflated(self, faint_result) -> None:
        """Emitted track positions stay near truth: predicted-position
        fallback ⇒ residual ≈ the stack solution error (≲1 px), not the
        ~1.4 px noise-peak scatter of the old 3σ refinement."""
        result, truth = faint_result
        wcs = _wcs()
        best = max(result.tracklets, key=lambda t: len(t.points))
        t0 = 60000.0
        errs = []
        for pt in best.points:
            frame_id = int(round((pt.utc_mjd - t0) * 86400.0 * _FPS))
            xt, yt = truth[frame_id]
            from opta_pipeline.astrometry import radec_to_pixels

            x, y = radec_to_pixels(wcs, pt.ra_deg, pt.dec_deg)
            errs.append(float(np.hypot(x - xt, y - yt)))
        rms_px = float(np.sqrt(np.mean(np.square(errs))))
        assert rms_px < 1.0
