"""End-to-end pipeline test on *realistic* frames (detector signature + calibration).

The standard integration test (``test_pipeline_integration.py``) runs the clean
synth path — no bias, flat, dark structure, or hot pixels — so it never
exercises master-dark/-flat calibration or the pipeline's robustness to fixed
detector defects.  This module drives the full production chain

    realistic synth → run_frame(calibrate w/ masters → detect → astrometry) → tracklet

on frames carrying a full instrumental signature, calibrated with synthetic
master dark/flat frames built from the *same* DetectorModel.  It is the harness
intended for tuning the pipeline against as-real-as-possible data, and it guards
two things the clean path cannot:

* **Calibration efficacy** — the satellite is still detected and astrometrically
  accurate after bias/flat/dark are removed by the masters.
* **False-alarm robustness** — static defects (hot pixels, bias FPN) do not
  survive calibration to spawn spurious long tracklets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL as _SENSOR

from opta_pipeline.astrometry import StarMatch
from opta_pipeline.pipeline import FrameContext, run_frame
from opta_pipeline.synth import (
    PSFModel,
    SatelliteSpec,
    StarSpec,
    build_detector_model,
    generate_dark_frame,
    generate_flat_frame,
    generate_frame,
)
from opta_pipeline.tracklet import FrameDetections, link_detections

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(_SENSOR.pixel_size_um, _OPTICS.focal_length_mm)
_W, _H = _SENSOR.resolution_h, _SENSOR.resolution_v
_CX, _CY = float(_W) / 2.0, float(_H) / 2.0

_RA0, _DEC0 = 135.0, 45.0
_COS_DEC = math.cos(math.radians(_DEC0))

_SAT_MAG = 9.0
_SAT_VEL_DEG_S = 0.5
_DX_PER_FRAME = 8.0

_N_FRAMES = 10
_DT_S = 1.0 / _SENSOR.frame_rate_hz
_MJD0 = 60000.0
_DT_MJD = _DT_S / 86400.0
_NODE = "NODE-SYNTH-REAL"

_STAR_XS = [60.0, 140.0, 260.0, 340.0]
_STAR_YS = [60.0, 240.0]

# Thresholds (generous: streak centroiding in noise after calibration)
_MIN_DETECTED = 8
_E2E_POS_ARCSEC = 15.0
_MIN_POINTS = 5


def _px_to_sky(x: float, y: float) -> tuple[float, float]:
    """True WCS: pixel (x, y) → (RA, Dec) degrees."""
    dx, dy = x - _CX, y - _CY
    ra = _RA0 + (_PIXEL_SCALE * dx) / (_COS_DEC * 3600.0)
    dec = _DEC0 + (_PIXEL_SCALE * dy) / 3600.0
    return ra, dec


def _star_matches() -> list[StarMatch]:
    """Eight StarMatch objects on a 4×2 grid (simulated catalog cross-match)."""
    return [
        StarMatch(x_px=x, y_px=y, ra_deg=ra, dec_deg=dec)
        for x in _STAR_XS
        for y in _STAR_YS
        for ra, dec in [_px_to_sky(x, y)]
    ]


def _sat_center(i: int) -> tuple[float, float]:
    """Ground-truth satellite pixel center for frame i."""
    cx = _CX - (_N_FRAMES / 2.0) * _DX_PER_FRAME + i * _DX_PER_FRAME
    return cx, _CY


def _run_realistic_chain(psf: PSFModel | None = None) -> dict:
    """Build masters once, then run the full chain on 10 realistic frames.

    *psf* selects the PSF model used for the science frames (``None`` → uniform
    default).  Calibration frames carry no sources, so they are PSF-independent.
    """
    # Detector signature: pedestal, offset FPN, PRNU + vignetting, hot pixels.
    detector = build_detector_model(
        _SENSOR,
        seed=2024,
        prnu_pct=1.5,
        vignetting_corner_factor=0.8,
        bias_offset_e=300.0,
        bias_fpn_rms_e=5.0,
        hot_pixel_fraction=2e-4,
        hot_pixel_dark_e_s=50.0,
    )
    # Master calibration frames from the SAME detector (frame-stable structure).
    darks = [
        generate_dark_frame(_SENSOR, detector, rng=np.random.default_rng(500 + i)).data
        for i in range(7)
    ]
    flats = [
        generate_flat_frame(
            _SENSOR, detector, illumination_e=8000.0, rng=np.random.default_rng(600 + i)
        ).data
        for i in range(7)
    ]
    from opta_pipeline.calibrate import make_master_dark, make_master_flat

    master_dark = make_master_dark(darks)
    master_flat = make_master_flat(flats)

    matches = _star_matches()
    frame_dets: list[FrameDetections] = []
    pos_errors: list[float] = []
    detected = 0

    for i in range(_N_FRAMES):
        cx, cy = _sat_center(i)
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            satellites=[
                SatelliteSpec(
                    magnitude=_SAT_MAG,
                    angular_velocity_deg_s=_SAT_VEL_DEG_S,
                    x_center=cx,
                    y_center=cy,
                    angle_deg=0.0,
                )
            ],
            stars=[
                StarSpec(magnitude=9.5, x=x, y=y)
                for x in _STAR_XS
                for y in _STAR_YS
            ],
            detector=detector,
            psf=psf,
            rng=np.random.default_rng(3000 + i),
        )
        ctx = FrameContext(
            star_matches=matches,
            utc_mjd=_MJD0 + i * _DT_MJD,
            frame_id=i,
            node_id=_NODE,
            master_dark=master_dark,
            master_flat=master_flat,
        )
        fd = run_frame(synth.data, ctx)
        if fd is None:
            continue
        detected += 1
        frame_dets.append(fd)
        best = min(
            fd.detections,
            key=lambda ad: math.hypot(ad.detection.x - cx, ad.detection.y - cy),
        )
        gt_ra, gt_dec = _px_to_sky(cx, cy)
        pos_errors.append(
            math.hypot(
                (best.ra_deg - gt_ra) * _COS_DEC * 3600.0,
                (best.dec_deg - gt_dec) * 3600.0,
            )
        )

    tracklets = link_detections(
        frame_dets,
        max_sep_arcsec=300.0,
        min_points=3,
        max_rms_arcsec=50.0,
        max_gap_frames=2,
    )
    return {
        "detected": detected,
        "pos_errors": pos_errors,
        "tracklets": tracklets,
        "master_dark": master_dark,
        "master_flat": master_flat,
    }


@pytest.fixture(scope="module")
def realistic_results() -> dict:
    """Full chain with the default uniform PSF."""
    return _run_realistic_chain(psf=None)


@pytest.fixture(scope="module")
def aberrated_results() -> dict:
    """Full chain with a field-variable (aberrated) PSF: 2 px → 6 px to corner."""
    return _run_realistic_chain(psf=PSFModel(fwhm_center_px=2.0, fwhm_edge_px=6.0))


class TestRealisticEndToEnd:
    """The full chain works on instrumentally-realistic, calibrated frames."""

    def test_detection_completeness(self, realistic_results: dict) -> None:
        assert realistic_results["detected"] >= _MIN_DETECTED, (
            f"Detected in {realistic_results['detected']}/{_N_FRAMES} frames "
            f"(need ≥ {_MIN_DETECTED}) — calibration may not be clearing the "
            "detector signature"
        )

    def test_astrometric_accuracy_after_calibration(
        self, realistic_results: dict
    ) -> None:
        errs = realistic_results["pos_errors"]
        assert errs, "No position errors recorded"
        mean_err = float(np.mean(errs))
        assert mean_err < _E2E_POS_ARCSEC, (
            f"Mean astrometric error {mean_err:.2f}\" exceeds {_E2E_POS_ARCSEC}\""
        )

    def test_satellite_tracklet_formed(self, realistic_results: dict) -> None:
        tracklets = realistic_results["tracklets"]
        assert any(len(t.points) >= _MIN_POINTS for t in tracklets), (
            f"No tracklet with ≥ {_MIN_POINTS} points "
            f"(got lengths {[len(t.points) for t in tracklets]})"
        )

    def test_no_spurious_long_tracklets_from_static_defects(
        self, realistic_results: dict
    ) -> None:
        """Hot pixels / bias FPN are static; after calibration they must not
        survive to form a second long (≥5-point) tracklet."""
        long_tracklets = [
            t for t in realistic_results["tracklets"] if len(t.points) >= _MIN_POINTS
        ]
        assert len(long_tracklets) == 1, (
            f"Expected exactly one long tracklet (the satellite); got "
            f"{len(long_tracklets)} — static defects may be leaking through "
            "calibration as false tracks"
        )

    def test_masters_capture_pedestal(self, realistic_results: dict) -> None:
        """Sanity: the master dark holds the bias pedestal it must remove."""
        master_dark = realistic_results["master_dark"]
        assert float(np.median(master_dark)) == pytest.approx(300.0, abs=2.0)


class TestRealisticEndToEndAberrated:
    """Field-dependent PSF blur must not break detection or bias astrometry.

    The integrated PSF is symmetric, so broadening the spot does not shift its
    centroid — plate-solving accuracy should survive aberration even though the
    streak is fatter.  This guards the pipeline's centroider against the
    realistic off-axis blur the clean path never produced.
    """

    def test_detection_completeness_under_aberration(
        self, aberrated_results: dict
    ) -> None:
        assert aberrated_results["detected"] >= _MIN_DETECTED, (
            f"Aberrated: detected in {aberrated_results['detected']}/{_N_FRAMES}"
        )

    def test_astrometry_unbiased_under_aberration(
        self, aberrated_results: dict
    ) -> None:
        errs = aberrated_results["pos_errors"]
        assert errs, "No position errors recorded"
        assert float(np.mean(errs)) < _E2E_POS_ARCSEC, (
            f"Aberrated mean astrometric error {np.mean(errs):.2f}\" "
            f"exceeds {_E2E_POS_ARCSEC}\""
        )

    def test_satellite_tracklet_formed_under_aberration(
        self, aberrated_results: dict
    ) -> None:
        tracklets = aberrated_results["tracklets"]
        assert any(len(t.points) >= _MIN_POINTS for t in tracklets), (
            f"Aberrated: no tracklet with ≥ {_MIN_POINTS} points "
            f"(lengths {[len(t.points) for t in tracklets]})"
        )
