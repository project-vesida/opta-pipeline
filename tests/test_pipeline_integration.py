"""End-to-end pipeline integration tests (OT-025).

Validates the full synth → calibrate → detect → astrometry → tracklet chain on
_N_FRAMES synthetic frames with a moving satellite at known sky positions.

Accept criteria (OT-025):
  - Detection completeness: streak detected in ≥ _MIN_DETECTED of _N_FRAMES frames
  - Astrometric accuracy: mean position error < _E2E_POS_ARCSEC
  - Tracklet formation: at least one tracklet with ≥ _MIN_POINTS observation points
  - Velocity direction: RA rate positive (prograde motion in RA)
  - Linear fit RMS: < _MAX_TRACKLET_RMS arcsec

WCS is calibrated per frame from 8 synthetic stars at known pixel positions,
simulating a catalog cross-match (a separate pipeline step not yet implemented).
The WCS used to compute ground-truth coordinates is the same true WCS, so the
only position error in the end-to-end test comes from streak centroiding in noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL as _SENSOR
from sensor_fixtures import per_frame_only_config

from opta_pipeline.astrometry import (
    StarMatch,
    astrometrise_detections,
    fit_wcs,
    pixels_to_radec,
)
from opta_pipeline.calibrate import calibrate_frame
from opta_pipeline.detect import Detection, detect_sources
from opta_pipeline.pipeline import FrameContext, run_frame, run_pipeline
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.tracklet import FrameDetections, link_detections

# ---------------------------------------------------------------------------
# Hardware — small sensor for test speed (same pixel/optics as IMX585)
# ---------------------------------------------------------------------------

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(
    _SENSOR.pixel_size_um, _OPTICS.focal_length_mm
)  # arcsec/px
_W, _H = _SENSOR.resolution_h, _SENSOR.resolution_v  # 400, 300
_CX, _CY = float(_W) / 2.0, float(_H) / 2.0  # 200.0, 150.0

# Field pointing (true WCS)
_RA0, _DEC0 = 135.0, 45.0
_COS_DEC = math.cos(math.radians(_DEC0))

# Satellite: bright enough for detection, fast enough for elongated streak
_SAT_MAG = 9.0
_SAT_VEL_DEG_S = 0.5  # deg/s → trail ≈ 7.9 px, elongation ≈ 2.7
_SAT_ANGLE = 0.0  # horizontal streak (RA direction)
_DX_PER_FRAME = 8.0  # px/frame ≈ 0.5 × 3600 × 0.04 / 9.12

# Sequence
_N_FRAMES = 10
_DT_S = 1.0 / _SENSOR.frame_rate_hz  # 0.04 s per frame
_MJD0 = 60000.0
_DT_MJD = _DT_S / 86400.0

_NODE = "NODE-SYNTH"

# Star grid: 4 cols × 2 rows, kept clear of the satellite path at y ≈ 150
_STAR_XS = [60.0, 140.0, 260.0, 340.0]
_STAR_YS = [60.0, 240.0]

# Test thresholds
_MIN_DETECTED = 8  # completeness: ≥ 80 % of frames
_E2E_POS_ARCSEC = 15.0  # position accuracy (generous for streak centroiding in noise)
_MIN_POINTS = 5  # tracklet must have at least this many observations
_MAX_TRACKLET_RMS = 50.0  # arcsec — satellite moves linearly; noise drives residuals


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _px_to_sky(x: float, y: float) -> tuple[float, float]:
    """True WCS: pixel (x, y) → (RA, Dec) via exact gnomonic (TAN) inversion.

    Written out independently of astrometry.py (standard TAN deprojection,
    SLALIB dtp2s form) so the truth model stays decoupled from the code under
    test.  A linear ΔRA·cosδ/Δδ truth is wrong at second order (ξ·η·tanδ ≈ 3″
    across even this 0.8°×0.6° field at dec 45°) — the pre-2026-07-10 fitter
    only matched it because it shared the same wrong model.
    """
    xi = math.radians(_PIXEL_SCALE * (x - _CX) / 3600.0)
    eta = math.radians(_PIXEL_SCALE * (y - _CY) / 3600.0)
    dec0 = math.radians(_DEC0)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    denom = cos_d0 - eta * sin_d0
    ra = _RA0 + math.degrees(math.atan2(xi, denom))
    dec = math.degrees(math.atan2(sin_d0 + eta * cos_d0, math.hypot(xi, denom)))
    return ra, dec


def _star_matches() -> list[StarMatch]:
    """Eight StarMatch objects on a 4×2 grid (simulated catalog cross-match)."""
    out = []
    for x in _STAR_XS:
        for y in _STAR_YS:
            ra, dec = _px_to_sky(x, y)
            out.append(StarMatch(x_px=x, y_px=y, ra_deg=ra, dec_deg=dec))
    return out


def _sat_center(frame_idx: int) -> tuple[float, float]:
    """Ground-truth satellite pixel center for frame_idx (0-indexed)."""
    cx = _CX - (_N_FRAMES / 2.0) * _DX_PER_FRAME + frame_idx * _DX_PER_FRAME
    return cx, _CY


# ---------------------------------------------------------------------------
# Shared module-scoped fixture — runs the chain once for all e2e tests
# ---------------------------------------------------------------------------


def _generate_synth_frame(
    frame_idx: int,
    rng: np.random.Generator,
):
    """Build the synthetic input for one integration-test frame."""
    cx, cy = _sat_center(frame_idx)
    return generate_frame(
        sensor=_SENSOR,
        optics=_OPTICS,
        sky_mag_arcsec2=21.0,
        elevation_deg=45.0,
        satellites=[
            SatelliteSpec(
                magnitude=_SAT_MAG,
                angular_velocity_deg_s=_SAT_VEL_DEG_S,
                x_center=cx,
                y_center=cy,
                angle_deg=_SAT_ANGLE,
            )
        ],
        stars=[StarSpec(magnitude=9.5, x=x, y=y) for x in _STAR_XS for y in _STAR_YS],
        rng=rng,
    )


@pytest.fixture(scope="module")
def pipeline_results() -> dict:
    """Run the full 10-frame pipeline once; share results across the module.

    Uses run_frame from pipeline.py so the orchestration logic is tested
    through the production code path (closes WP-A1 / U1).
    """
    matches = _star_matches()
    frame_dets: list[FrameDetections] = []
    pos_errors: list[float] = []
    detected = 0

    for i in range(_N_FRAMES):
        rng = np.random.default_rng(2000 + i)
        cx, cy = _sat_center(i)
        synth = _generate_synth_frame(i, rng)
        ctx = FrameContext(
            star_matches=matches,
            utc_mjd=_MJD0 + i * _DT_MJD,
            frame_id=i,
            node_id=_NODE,
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
        max_rms_arcsec=_MAX_TRACKLET_RMS,
        max_gap_frames=2,
    )

    return {
        "detected": detected,
        "pos_errors": pos_errors,
        "frame_dets": frame_dets,
        "tracklets": tracklets,
    }


# ---------------------------------------------------------------------------
# 1. Calibration layer: synth → calibrate
# ---------------------------------------------------------------------------


class TestCalibrationLayer:
    """Background is removed; streak signal survives; flags are set correctly."""

    def test_background_subtracted_median_near_zero(self) -> None:
        """After background subtraction the global frame median must be near zero.

        Note: with 0.04 s exposure through a 61 mm aperture on a 21 mag/arcsec²
        sky only ~0.3 e⁻/px fall on the sensor, so the background is essentially
        integer-zero.  The residual median must still be within 2 ADU of zero.
        """
        rng = np.random.default_rng(99)
        cx, cy = _sat_center(0)
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
                )
            ],
            rng=rng,
        )
        cal = calibrate_frame(synth.data, subtract_background=True)
        median_val = float(np.median(cal.data))
        assert (
            abs(median_val) < 2.0
        )  # absolute criterion — rms may be zero for near-empty sky

    def test_background_rms_non_negative(self) -> None:
        """background_rms must be ≥ 0 (can be zero when sky flux rounds to 0 ADU/px)."""
        rng = np.random.default_rng(100)
        synth = generate_frame(sensor=_SENSOR, optics=_OPTICS, rng=rng)
        cal = calibrate_frame(synth.data, subtract_background=True)
        assert cal.background_rms >= 0.0

    def test_calibrated_flags(self) -> None:
        rng = np.random.default_rng(101)
        synth = generate_frame(sensor=_SENSOR, optics=_OPTICS, rng=rng)
        cal = calibrate_frame(synth.data, subtract_background=True)
        assert cal.background_subtracted is True
        assert cal.dark_subtracted is False
        assert cal.flat_corrected is False


# ---------------------------------------------------------------------------
# 2. Detection layer: calibrate → detect
# ---------------------------------------------------------------------------


class TestDetectionLayer:
    """Streak classified correctly; stars detected as compact sources."""

    def test_satellite_streak_detected(self) -> None:
        """Bright horizontal satellite streak must produce ≥1 is_streak detection."""
        rng = np.random.default_rng(200)
        cx, cy = _sat_center(5)
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
                    angle_deg=_SAT_ANGLE,
                )
            ],
            rng=rng,
        )
        cal = calibrate_frame(synth.data, subtract_background=True)
        dets = detect_sources(cal.data, noise_rms=max(cal.background_rms, 1.0))
        assert any(d.is_streak for d in dets)

    def test_streak_centroid_within_3px_of_truth(self) -> None:
        """Detected streak centroid must be within 3 px of the injected center."""
        rng = np.random.default_rng(201)
        cx, cy = _sat_center(5)
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
                    angle_deg=_SAT_ANGLE,
                )
            ],
            rng=rng,
        )
        cal = calibrate_frame(synth.data, subtract_background=True)
        dets = detect_sources(cal.data, noise_rms=max(cal.background_rms, 1.0))
        streaks = [d for d in dets if d.is_streak]
        assert len(streaks) >= 1
        best = min(streaks, key=lambda d: math.hypot(d.x - cx, d.y - cy))
        assert math.hypot(best.x - cx, best.y - cy) < 3.0

    def test_injected_star_detected_as_compact_source(self) -> None:
        """An injected bright star must be detected and not flagged as a streak."""
        rng = np.random.default_rng(202)
        sx, sy = 80.0, 80.0
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            stars=[StarSpec(magnitude=8.0, x=sx, y=sy)],
            rng=rng,
        )
        cal = calibrate_frame(synth.data, subtract_background=True)
        dets = detect_sources(cal.data, noise_rms=max(cal.background_rms, 1.0))
        nearby = [d for d in dets if math.hypot(d.x - sx, d.y - sy) < 3.0]
        assert len(nearby) >= 1
        assert all(not d.is_streak for d in nearby)


# ---------------------------------------------------------------------------
# 3. Astrometry layer: star matches → WCS → coordinate assignment
# ---------------------------------------------------------------------------


class TestAstrometryLayer:
    """Plate solution recovery and sky-coordinate assignment on a synthetic frame."""

    def test_wcs_from_exact_star_grid_near_zero_rms(self) -> None:
        """WCS from exact catalog positions must have near-zero residual."""
        matches = _star_matches()
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert wcs.rms_arcsec < 0.1
        assert wcs.n_stars >= 8

    def test_reference_pixel_at_field_centre(self) -> None:
        """pixels_to_radec at CRPIX must return the true pointing within 0.01 deg."""
        matches = _star_matches()
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        ra, dec = pixels_to_radec(wcs, wcs.crpix1, wcs.crpix2)
        assert abs(ra - _RA0) < 0.01
        assert abs(dec - _DEC0) < 0.01

    def test_pixel_scale_recovered(self) -> None:
        """CD matrix determinant must reproduce the true pixel scale within 1 %."""
        matches = _star_matches()
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
        recovered = math.sqrt(det) * 3600.0  # arcsec/px
        assert recovered == pytest.approx(_PIXEL_SCALE, rel=0.01)

    def test_satellite_position_within_accuracy_budget(self) -> None:
        """Pipeline-recovered satellite position within _E2E_POS_ARCSEC of truth."""
        rng = np.random.default_rng(300)
        cx, cy = _sat_center(5)
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
                    angle_deg=_SAT_ANGLE,
                )
            ],
            rng=rng,
        )
        cal = calibrate_frame(synth.data, subtract_background=True)
        dets = detect_sources(cal.data, noise_rms=max(cal.background_rms, 1.0))
        streaks = [d for d in dets if d.is_streak]
        assert len(streaks) >= 1

        matches = _star_matches()
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        best = min(streaks, key=lambda d: math.hypot(d.x - cx, d.y - cy))
        astro = astrometrise_detections([best], wcs)

        gt_ra, gt_dec = _px_to_sky(cx, cy)
        err = math.hypot(
            (astro[0].ra_deg - gt_ra) * _COS_DEC * 3600.0,
            (astro[0].dec_deg - gt_dec) * 3600.0,
        )
        assert err < _E2E_POS_ARCSEC

    def test_sigma_positive_from_wcs_rms(self) -> None:
        """Positional uncertainties from astrometrise_detections must be positive."""
        matches = _star_matches()
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        det = Detection(
            x=_CX,
            y=_CY,
            snr=50.0,
            elongation=4.0,
            angle_deg=0.0,
            n_pixels=20,
            flux_e=5000.0,
            is_streak=True,
            fwhm_px=3.0,
        )
        astro = astrometrise_detections([det], wcs)
        assert astro[0].sigma_ra_arcsec > 0.0
        assert astro[0].sigma_dec_arcsec > 0.0


# ---------------------------------------------------------------------------
# 4. End-to-end chain: 10-frame completeness, accuracy, and tracklet
# ---------------------------------------------------------------------------


class TestEndToEndChain:
    """Full synth → calibrate → detect → astrometry → tracklet over 10 frames."""

    def test_detection_completeness(self, pipeline_results: dict) -> None:
        """Satellite detected in ≥ _MIN_DETECTED of _N_FRAMES frames."""
        n = pipeline_results["detected"]
        assert n >= _MIN_DETECTED, (
            f"Only {n}/{_N_FRAMES} frames had a satellite streak detected"
        )

    def test_position_accuracy_mean(self, pipeline_results: dict) -> None:
        """Mean per-frame angular error must be within _E2E_POS_ARCSEC."""
        errs = pipeline_results["pos_errors"]
        assert len(errs) > 0, "No position errors recorded"
        mean_err = float(np.mean(errs))
        assert mean_err < _E2E_POS_ARCSEC, (
            f"Mean position error {mean_err:.1f} arcsec > {_E2E_POS_ARCSEC}"
        )

    def test_tracklet_formed(self, pipeline_results: dict) -> None:
        """At least one tracklet with ≥ _MIN_POINTS observations must be formed."""
        tracklets = pipeline_results["tracklets"]
        assert len(tracklets) >= 1, "No tracklets formed from the 10-frame sequence"
        max_pts = max(len(t.points) for t in tracklets)
        assert max_pts >= _MIN_POINTS, (
            f"Longest tracklet has {max_pts} points; need ≥ {_MIN_POINTS}"
        )

    def test_tracklet_ra_rate_positive(self, pipeline_results: dict) -> None:
        """Satellite moves +RA; dominant tracklet RA rate must be positive."""
        tracklets = pipeline_results["tracklets"]
        assert len(tracklets) >= 1
        best = max(tracklets, key=lambda t: len(t.points))
        assert best.ra_rate_arcsec_s > 0.0, (
            f"RA rate {best.ra_rate_arcsec_s:.2f} arcsec/s; expected positive"
        )

    def test_tracklet_linear_fit_rms(self, pipeline_results: dict) -> None:
        """Linear-fit RMS of the dominant tracklet must be within _MAX_TRACKLET_RMS."""
        tracklets = pipeline_results["tracklets"]
        assert len(tracklets) >= 1
        best = max(tracklets, key=lambda t: len(t.points))
        assert best.rms_arcsec < _MAX_TRACKLET_RMS, (
            f"Tracklet RMS {best.rms_arcsec:.1f} arcsec > {_MAX_TRACKLET_RMS}"
        )

    def test_i02_records_have_required_keys(self, pipeline_results: dict) -> None:
        """Every I-02 record from e2e tracklets must carry all required fields."""
        tracklets = pipeline_results["tracklets"]
        assert len(tracklets) >= 1
        required = {"ra", "dec", "utc", "sigma_ra", "sigma_dec", "object_id", "node_id"}
        for t in tracklets:
            for rec in t.to_records():
                assert required <= set(rec.keys())

    def test_node_id_preserved_through_chain(self, pipeline_results: dict) -> None:
        """Tracklet node_id must match the node that observed the frames."""
        tracklets = pipeline_results["tracklets"]
        assert len(tracklets) >= 1
        for t in tracklets:
            assert t.node_id == _NODE


# ---------------------------------------------------------------------------
# T-08 · Partial-pass robustness: satellite enters or exits mid-sequence
# ---------------------------------------------------------------------------


class TestPartialPassDetection:
    """Satellite visible only in a subset of frames must still produce a tracklet.

    Tests two scenarios:
    - Entry case: satellite appears at frame 10 (frames 0–9 have no satellite).
    - Exit case: satellite disappears at frame 20 (frames 20–29 have no satellite).

    Both cases use N_FRAMES_TOTAL=30 frames with the same horizontal satellite
    (magnitude 9.0, 0.5 deg/s) so the trajectory is well within sensor bounds
    throughout the visible window.
    """

    _N_FRAMES_TOTAL = 30

    def _make_frame_pair(
        self,
        frame_idx: int,
        sat_x: float | None,
        sat_y: float,
        rng_seed: int,
    ) -> tuple[np.ndarray, FrameContext]:
        """Generate one (raw, FrameContext) pair for run_pipeline."""
        satellites = []
        if sat_x is not None:
            satellites = [
                SatelliteSpec(
                    magnitude=_SAT_MAG,
                    angular_velocity_deg_s=_SAT_VEL_DEG_S,
                    x_center=sat_x,
                    y_center=sat_y,
                    angle_deg=_SAT_ANGLE,
                )
            ]
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=satellites,
            stars=[
                StarSpec(magnitude=9.5, x=x, y=y) for x in _STAR_XS for y in _STAR_YS
            ],
            rng=np.random.default_rng(rng_seed),
        )
        ctx = FrameContext(
            star_matches=_star_matches(),
            utc_mjd=_MJD0 + frame_idx * _DT_MJD,
            frame_id=frame_idx,
            node_id=_NODE,
        )
        return synth.data, ctx

    def test_entry_case_forms_tracklet(self) -> None:
        """Satellite visible only in frames 10–29 must still produce a tracklet."""
        frame_pairs = []
        for i in range(self._N_FRAMES_TOTAL):
            if i < 10:
                # No satellite — blank frame
                sat_x = None
            else:
                # Center trajectory so satellite stays inside sensor during frames 10–29
                # At frame 10: x = CX - 10 * DX, at frame 19: x = CX (center)
                sat_x = _CX - 10.0 * _DX_PER_FRAME + (i - 10) * _DX_PER_FRAME
            frame_pairs.append(self._make_frame_pair(i, sat_x, _CY, rng_seed=4000 + i))

        # Per-frame linker path is the subject here; the stacked search is a
        # parallel path that would add tens of seconds per call.
        tracklets = run_pipeline(frame_pairs, config=per_frame_only_config())

        assert len(tracklets) >= 1, "No tracklet formed from partial-pass (entry case)"
        best = max(tracklets, key=lambda t: len(t.points))
        assert len(best.points) >= 5, (
            f"Tracklet too short: {len(best.points)} points (entry case)"
        )

    def test_exit_case_forms_tracklet(self) -> None:
        """Satellite visible only in frames 0–19 must still produce a tracklet."""
        frame_pairs = []
        for i in range(self._N_FRAMES_TOTAL):
            if i >= 20:
                # Satellite has exited — blank frame
                sat_x = None
            else:
                # Satellite crosses from left to right during frames 0–19
                sat_x = _CX - 10.0 * _DX_PER_FRAME + i * _DX_PER_FRAME
            frame_pairs.append(self._make_frame_pair(i, sat_x, _CY, rng_seed=5000 + i))

        # Per-frame linker path is the subject here; the stacked search is a
        # parallel path that would add tens of seconds per call.
        tracklets = run_pipeline(frame_pairs, config=per_frame_only_config())

        assert len(tracklets) >= 1, "No tracklet formed from partial-pass (exit case)"
        best = max(tracklets, key=lambda t: len(t.points))
        assert len(best.points) >= 5, (
            f"Tracklet too short: {len(best.points)} points (exit case)"
        )


# ---------------------------------------------------------------------------
# T-09 · Two simultaneous objects moving in perpendicular directions
# ---------------------------------------------------------------------------


class TestTwoSimultaneousObjects:
    """Two perpendicular satellites must form two independent tracklets.

    Satellite 1 moves horizontally (+x) and satellite 2 vertically (+y).
    The Hungarian linker must resolve inter-frame assignment correctly —
    no detection should appear in both tracklets.
    """

    _N_FRAMES = 10

    def test_two_satellites_form_separate_tracklets(self) -> None:
        """Two perpendicular satellites must produce two independent tracklets."""
        frame_pairs = []
        for i in range(self._N_FRAMES):
            # Satellite 1: horizontal, moves +8 px/frame in x
            sat1_x = 50.0 + 8.0 * i
            sat1_y = 100.0
            # Satellite 2: vertical, moves +8 px/frame in y
            sat2_x = 300.0
            sat2_y = 20.0 + 8.0 * i

            synth = generate_frame(
                sensor=_SENSOR,
                optics=_OPTICS,
                sky_mag_arcsec2=21.0,
                elevation_deg=45.0,
                satellites=[
                    SatelliteSpec(
                        magnitude=8.0,
                        angular_velocity_deg_s=0.5,
                        x_center=sat1_x,
                        y_center=sat1_y,
                        angle_deg=0.0,
                    ),
                    SatelliteSpec(
                        magnitude=8.0,
                        angular_velocity_deg_s=0.5,
                        x_center=sat2_x,
                        y_center=sat2_y,
                        angle_deg=90.0,
                    ),
                ],
                stars=[
                    StarSpec(magnitude=9.5, x=x, y=y)
                    for x in _STAR_XS
                    for y in _STAR_YS
                ],
                rng=np.random.default_rng(3000 + i),
            )
            ctx = FrameContext(
                star_matches=_star_matches(),
                utc_mjd=_MJD0 + i * _DT_MJD,
                frame_id=i,
                node_id=_NODE,
            )
            frame_pairs.append((synth.data, ctx))

        # Per-frame linker path is the subject here; the stacked search is a
        # parallel path that would add tens of seconds per call.
        tracklets = run_pipeline(frame_pairs, config=per_frame_only_config())

        # Two independent satellites must each produce a separate tracklet
        assert len(tracklets) >= 2, (
            f"Expected >=2 tracklets for two perpendicular satellites, "
            f"got {len(tracklets)}"
        )

        points_counts = sorted([len(t.points) for t in tracklets], reverse=True)
        assert points_counts[0] >= 5, (
            f"Primary tracklet too short: {points_counts[0]} points"
        )
        assert points_counts[1] >= 3, (
            f"Secondary tracklet too short: {points_counts[1]} points"
        )

        # No observation should appear in two tracklets (unique by utc_mjd + ra_deg)
        all_point_ids = []
        for t in tracklets:
            all_point_ids.extend(
                (round(p.utc_mjd, 9), round(p.ra_deg, 6)) for p in t.points
            )
        assert len(all_point_ids) == len(set(all_point_ids)), (
            "Same detection appears in two tracklets"
        )
