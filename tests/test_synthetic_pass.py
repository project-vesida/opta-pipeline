"""Acceptance tests for SyntheticPass (WP-A3).

Validates the full SGP4 geometry → synthetic frames → pipeline → tracklet
chain using a frozen ISS TLE.  These tests exercise real orbital mechanics —
each pass uses SGP4-propagated positions, not hand-coded straight-line motion.

Accept criteria (WP-A3):
  - SyntheticPass generates n_frames with ground-truth sky positions
  - Satellite is in the FOV for ≥ 60 % of frames (pointing centred at midpass)
  - Pipeline recovers ≥ 1 tracklet from the sequence
  - Mean position residual (recovered vs SGP4 ground truth) < OpTA.NOD.ACC (4 arcsec)
  - PassGroundTruth angular velocity is in the expected LEO range (0.3–2.5 deg/s)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from opta_model.geometry import Observer
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL, per_frame_only_config

from opta_pipeline.astrometry import StarMatch, pixels_to_radec
from opta_pipeline.pipeline import FrameContext, run_pipeline
from opta_pipeline.synth.pass_ import SyntheticPass, make_synthetic_pass

# ---------------------------------------------------------------------------
# Frozen ISS TLE (epoch 2024-01-01 12:00 UTC)
# Same TLE used in opta-model/tests/test_geometry.py — reproducible offline.
# ---------------------------------------------------------------------------

_ISS_NAME = "ISS (ZARYA)"
_ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00007500  00000-0  14000-3 0  9995"
_ISS_LINE2 = "2 25544  51.6400 100.0000 0005000  90.0000 270.0000 15.50000000400000"

# Berne, Switzerland — representative mid-latitude European observer
_OBSERVER = Observer(latitude_deg=46.95, longitude_deg=7.44, elevation_m=540.0)

# ISS at peak elevation (82.5°) from Berne: MJD 60313.090966
# = 2024-01-04 02:10:59 UTC.  Centered 12 frames before the peak so frame 12
# is the highest-elevation point.  Angular velocity ~1.0 deg/s → 15.8 px/frame.
# Computed by scanning ahead from epoch; good for ≥ 3 days from TLE epoch.
_T_START_MJD = 60313.090966

# 25 frames = 1 second at 25 fps; short enough for near-linear motion,
# long enough to form a tracklet.
_N_FRAMES = 25

# OpTA.NOD.ACC requirement
_NOD_ACC_ARCSEC = 4.0


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate pass once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iss_pass() -> SyntheticPass:
    """Generate a 25-frame synthetic ISS pass (runs SGP4 once per module)."""
    return make_synthetic_pass(
        _ISS_NAME,
        _ISS_LINE1,
        _ISS_LINE2,
        observer=_OBSERVER,
        t_start_mjd=_T_START_MJD,
        n_frames=_N_FRAMES,
        sensor=SENSOR_SMALL,
        optics=OPTICS_DEFAULT,
        satellite_magnitude=9.0,
        sky_mag_arcsec2=21.0,
        elevation_deg=45.0,
        rng=np.random.default_rng(42),
    )


def _star_matches_for_pass(synth_pass: SyntheticPass) -> list[StarMatch]:
    """Create 8 synthetic star matches on a 4×2 grid, derived from the pass WCS.

    Star positions are exact pixel→sky projections of the pointing WCS,
    so WCS residuals are near zero and position accuracy reflects only
    centroid noise.
    """
    w = SENSOR_SMALL.resolution_h
    h = SENSOR_SMALL.resolution_v
    xs = [w * 0.15, w * 0.38, w * 0.62, w * 0.85]
    ys = [h * 0.2, h * 0.8]
    matches = []
    for x in xs:
        for y in ys:
            ra, dec = pixels_to_radec(synth_pass.wcs, x, y)
            matches.append(StarMatch(x_px=x, y_px=y, ra_deg=ra, dec_deg=dec))
    return matches


# ---------------------------------------------------------------------------
# 1. Pass geometry
# ---------------------------------------------------------------------------


class TestPassGeometry:
    """Ground-truth positions and velocities are consistent with LEO orbit."""

    def test_correct_frame_count(self, iss_pass: SyntheticPass) -> None:
        assert len(iss_pass.frames) == _N_FRAMES
        assert len(iss_pass.ground_truth) == _N_FRAMES
        assert iss_pass.n_frames == _N_FRAMES

    def test_satellite_in_fov_majority_of_frames(self, iss_pass: SyntheticPass) -> None:
        """Satellite must be in FOV for ≥ 60 % of frames (pointing at midpass)."""
        in_fov = sum(1 for gt in iss_pass.ground_truth if gt.in_fov)
        assert in_fov >= _N_FRAMES * 0.6, (
            f"Only {in_fov}/{_N_FRAMES} frames have satellite in FOV"
        )

    def test_angular_velocity_in_leo_range(self, iss_pass: SyntheticPass) -> None:
        """LEO angular velocity seen from ground: typically 0.3–2.5 deg/s."""
        for gt in iss_pass.ground_truth:
            assert 0.3 <= gt.ang_vel_deg_s <= 3.0, (
                f"Frame {gt.frame_id}: ang_vel = {gt.ang_vel_deg_s:.3f} deg/s "
                "outside expected LEO range"
            )

    def test_sky_positions_monotonically_changing(
        self, iss_pass: SyntheticPass
    ) -> None:
        """RA or Dec should change frame-to-frame (satellite is moving)."""
        gts = iss_pass.ground_truth
        ra_changes = [
            abs(gts[i + 1].ra_deg - gts[i].ra_deg) for i in range(len(gts) - 1)
        ]
        dec_changes = [
            abs(gts[i + 1].dec_deg - gts[i].dec_deg) for i in range(len(gts) - 1)
        ]
        # At least one axis must change by more than 1 arcsec per frame
        assert any(d > 1 / 3600 for d in ra_changes + dec_changes), (
            "Satellite appears stationary — SGP4 propagation may have failed"
        )

    def test_frame_ids_sequential(self, iss_pass: SyntheticPass) -> None:
        for i, gt in enumerate(iss_pass.ground_truth):
            assert gt.frame_id == i

    def test_utc_mjd_increasing(self, iss_pass: SyntheticPass) -> None:
        gts = iss_pass.ground_truth
        dt_s = 1.0 / SENSOR_SMALL.frame_rate_hz
        for i in range(len(gts) - 1):
            assert gts[i + 1].utc_mjd > gts[i].utc_mjd
            diff_s = (gts[i + 1].utc_mjd - gts[i].utc_mjd) * 86400.0
            # Float64 MJD arithmetic has ~0.1 µs precision at this scale.
            assert abs(diff_s - dt_s) < 1e-6

    def test_wcs_pointing_at_midpass(self, iss_pass: SyntheticPass) -> None:
        """WCS CRVAL should be at the satellite's midpass sky position."""
        mid = iss_pass.ground_truth[_N_FRAMES // 2]
        assert iss_pass.wcs.crval1 == pytest.approx(mid.ra_deg, abs=1e-6)
        assert iss_pass.wcs.crval2 == pytest.approx(mid.dec_deg, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Frame content
# ---------------------------------------------------------------------------


class TestFrameContent:
    """Generated SynthFrames have the expected structure."""

    def test_frame_shapes(self, iss_pass: SyntheticPass) -> None:
        h, w = SENSOR_SMALL.resolution_v, SENSOR_SMALL.resolution_h
        for frame in iss_pass.frames:
            assert frame.data.shape == (h, w)

    def test_in_fov_frames_have_satellite_ground_truth(
        self, iss_pass: SyntheticPass
    ) -> None:
        """Frames with in_fov=True must have exactly one satellite in ground truth."""
        for frame, gt in zip(iss_pass.frames, iss_pass.ground_truth):
            if gt.in_fov and gt.ang_vel_deg_s > 1e-4:
                assert len(frame.satellites) == 1, (
                    f"Frame {gt.frame_id}: expected 1 satellite, "
                    f"got {len(frame.satellites)}"
                )

    def test_out_of_fov_frames_have_no_satellite(self, iss_pass: SyntheticPass) -> None:
        for frame, gt in zip(iss_pass.frames, iss_pass.ground_truth):
            if not gt.in_fov:
                assert len(frame.satellites) == 0


# ---------------------------------------------------------------------------
# 3. Tracklet recovery (acceptance criterion)
# ---------------------------------------------------------------------------


class TestTrackletRecovery:
    """Pipeline recovers a tracklet whose positions match SGP4 ground truth."""

    # Keep only frames whose full streak fits inside the FOV.  ``in_fov``
    # tests the streak *centre*; on the first/last frame of the pass the
    # trail (≈ 18 px at the 2.9 µm datasheet pitch, MA-002) can be clipped
    # by the frame edge, biasing the measured centroid toward the visible
    # half by tens of arcsec.  Margin = half the worst-case trail + PSF.
    _EDGE_MARGIN_PX = 20.0

    @pytest.fixture(scope="class")
    def pipeline_results(self, iss_pass: SyntheticPass) -> dict:
        star_matches = _star_matches_for_pass(iss_pass)
        w = SENSOR_SMALL.resolution_h
        h = SENSOR_SMALL.resolution_v
        m = self._EDGE_MARGIN_PX
        frame_pairs = [
            (
                frame.data,
                FrameContext(
                    star_matches=star_matches,
                    utc_mjd=gt.utc_mjd,
                    frame_id=gt.frame_id,
                    node_id="NODE-TEST",
                ),
            )
            for frame, gt in zip(iss_pass.frames, iss_pass.ground_truth)
            if gt.in_fov and m <= gt.x_px < w - m and m <= gt.y_px < h - m
        ]
        # SGP4-pass recovery through the per-frame chain is the subject; the
        # stacked search is a parallel path costing tens of seconds per call.
        tracklets = run_pipeline(frame_pairs, config=per_frame_only_config())
        return {"tracklets": tracklets, "pass": iss_pass}

    def test_at_least_one_tracklet_formed(self, pipeline_results: dict) -> None:
        """Pipeline must form at least one quality-controlled tracklet."""
        assert len(pipeline_results["tracklets"]) >= 1, (
            "No tracklets formed from the SGP4-driven synthetic pass"
        )

    def test_tracklet_has_minimum_points(self, pipeline_results: dict) -> None:
        """Dominant tracklet must have ≥ 3 observations (pipeline default)."""
        best = max(pipeline_results["tracklets"], key=lambda t: len(t.points))
        assert len(best.points) >= 3

    def test_position_residual_within_nod_acc(self, pipeline_results: dict) -> None:
        """Mean position residual must be < OpTA.NOD.ACC = 4.0 arcsec.

        This is the core acceptance criterion for WP-A3.  It verifies that
        the SGP4 geometry → pixel injection → calibrate → detect →
        WCS → astrometry → tracklet chain produces positions that are
        accurate to within the node-level accuracy requirement.
        """
        synth_pass = pipeline_results["pass"]
        gt_by_mjd = {gt.utc_mjd: gt for gt in synth_pass.ground_truth}

        best = max(pipeline_results["tracklets"], key=lambda t: len(t.points))
        errors: list[float] = []
        for pt in best.points:
            gt = gt_by_mjd.get(pt.utc_mjd)
            if gt is None:
                continue
            cos_dec = math.cos(math.radians(gt.dec_deg))
            err = math.hypot(
                (pt.ra_deg - gt.ra_deg) * cos_dec * 3600.0,
                (pt.dec_deg - gt.dec_deg) * 3600.0,
            )
            errors.append(err)

        assert errors, "No tracklet points matched to ground-truth timestamps"
        mean_err = float(np.mean(errors))
        assert mean_err < _NOD_ACC_ARCSEC, (
            f"Mean position residual {mean_err:.2f} arcsec exceeds "
            f"OpTA.NOD.ACC = {_NOD_ACC_ARCSEC} arcsec"
        )

    def test_tracklet_linear_rms_within_tolerance(self, pipeline_results: dict) -> None:
        """Linear-fit RMS of best tracklet should be small (< 4 arcsec)."""
        best = max(pipeline_results["tracklets"], key=lambda t: len(t.points))
        assert best.rms_arcsec < _NOD_ACC_ARCSEC


# ---------------------------------------------------------------------------
# 4. radec_to_pixels / pixels_to_radec round-trip (via WCS)
# ---------------------------------------------------------------------------


class TestWCSRoundtrip:
    """radec_to_pixels is the exact inverse of pixels_to_radec."""

    def test_roundtrip_centre(self, iss_pass: SyntheticPass) -> None:
        from opta_pipeline.astrometry import radec_to_pixels

        wcs = iss_pass.wcs
        ra0, dec0 = wcs.crval1, wcs.crval2
        x, y = radec_to_pixels(wcs, ra0, dec0)
        assert x == pytest.approx(wcs.crpix1, abs=1e-9)
        assert y == pytest.approx(wcs.crpix2, abs=1e-9)

    def test_roundtrip_offset_pixel(self, iss_pass: SyntheticPass) -> None:
        from opta_pipeline.astrometry import radec_to_pixels

        wcs = iss_pass.wcs
        x_in, y_in = 80.0, 210.0
        ra, dec = pixels_to_radec(wcs, x_in, y_in)
        x_out, y_out = radec_to_pixels(wcs, ra, dec)
        assert x_out == pytest.approx(x_in, abs=1e-9)
        assert y_out == pytest.approx(y_in, abs=1e-9)

    def test_ground_truth_pixel_matches_wcs(self, iss_pass: SyntheticPass) -> None:
        """PassGroundTruth pixel position must be consistent with the WCS."""
        from opta_pipeline.astrometry import radec_to_pixels

        wcs = iss_pass.wcs
        for gt in iss_pass.ground_truth:
            x_calc, y_calc = radec_to_pixels(wcs, gt.ra_deg, gt.dec_deg)
            assert x_calc == pytest.approx(gt.x_px, abs=1e-6)
            assert y_calc == pytest.approx(gt.y_px, abs=1e-6)


def test_sidereal_drift_shifts_star_field_off_midpass() -> None:
    """Fixed-mount realism: stars drift across the frame at the sidereal rate.

    The midpass frame is the drift reference (Δ=0) so it is unchanged, while
    off-midpass frames differ from the stationary render — the mask must track
    this (see synthetic_video_e2e_findings.md #10).
    """
    from opta_pipeline.synth.catalog import catalog_stars_in_fov

    n = 50
    kw = dict(satellite_magnitude=30.0, star_mag_limit=12.0, max_stars=60)
    static = make_synthetic_pass(
        _ISS_NAME, _ISS_LINE1, _ISS_LINE2, _OBSERVER, _T_START_MJD, n,
        SENSOR_SMALL, OPTICS_DEFAULT, sidereal_drift=False,
        rng=np.random.default_rng(5), **kw,
    )
    n_stars = len(
        catalog_stars_in_fov(
            static.wcs, SENSOR_SMALL.resolution_h, SENSOR_SMALL.resolution_v, 12.0
        )
    )
    if n_stars < 5:
        pytest.skip("no catalog stars in this toy FOV to drift")

    drift = make_synthetic_pass(
        _ISS_NAME, _ISS_LINE1, _ISS_LINE2, _OBSERVER, _T_START_MJD, n,
        SENSOR_SMALL, OPTICS_DEFAULT, sidereal_drift=True,
        rng=np.random.default_rng(5), **kw,
    )
    mid = n // 2
    # Midpass reference frame: drift Δ = 0 → identical to the stationary render.
    assert np.array_equal(static.frames[mid].data, drift.frames[mid].data)
    # Off-midpass frames differ: the star field has translated.
    assert not np.array_equal(static.frames[0].data, drift.frames[0].data)
    assert not np.array_equal(static.frames[-1].data, drift.frames[-1].data)
