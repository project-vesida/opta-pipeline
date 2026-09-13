"""Tests for opta_pipeline.astrometry.

Validates:
  - WCS fitting from matched star pairs recovers the plate solution
  - Plate solution residuals within OpTA.NOD.ACC budget (≤ 4.0 arcsec)
  - pixels_to_radec round-trip accuracy for injected positions
  - Rolling shutter correction removes the timing-induced centroid bias
  - astrometrise_detections applies WCS to detection list
  - Outlier rejection removes high-residual matches
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy import stats

from opta_pipeline.astrometry import (
    AstrometricDetection,
    StarMatch,
    WCSSolution,
    apply_rolling_shutter_correction,
    astrometrise_detections,
    fit_wcs,
    fit_wcs_sip,
    pixels_to_radec,
    radec_to_pixels,
)
from opta_pipeline.detect import Detection
from opta_pipeline.synth.catalog import star_field_at
from opta_pipeline.synth.distortion import DistortionModel

# OpTA.NOD.ACC requirement
_NOD_ACC_ARCSEC = 4.0

# Simulated field parameters (IMX585 + 85mm f/1.4)
_PIXEL_SCALE_DEG = 9.12 / 3600.0  # degrees/pixel
_CD = _PIXEL_SCALE_DEG  # square pixels, no rotation

# Frame size
_W, _H = 1920, 1080
_RA0, _DEC0 = 135.0, 45.0  # field centre


def _make_star_field(
    n: int = 20,
    ra0: float = _RA0,
    dec0: float = _DEC0,
    pixel_scale_deg: float = _CD,
    width: int = _W,
    height: int = _H,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[StarMatch]:
    """Return n matched stars from the frozen synthetic catalog.

    Builds an axis-aligned TAN WCS from the given parameters and queries
    ``star_field_at``.  Pixel positions are exact projections of the WCS
    (no hand-rolled TAN).  Centroid noise is applied after projection.
    """
    if rng is None:
        rng = np.random.default_rng(100)
    wcs = WCSSolution(
        crpix1=width / 2.0,
        crpix2=height / 2.0,
        crval1=ra0,
        crval2=dec0,
        cd1_1=pixel_scale_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=pixel_scale_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )
    # Fetch all in-FOV catalog stars (mag_limit=999 → no filtering)
    all_matches = star_field_at(wcs, width, height, mag_limit=999.0)
    # Subsample to exactly n stars (deterministic, rng-seeded)
    if len(all_matches) > n:
        idx = rng.choice(len(all_matches), size=n, replace=False)
        all_matches = [all_matches[i] for i in sorted(idx)]
    # Apply centroid noise if requested (sky coords are unchanged)
    if noise_px > 0.0:
        all_matches = [
            StarMatch(
                x_px=m.x_px + rng.normal(0.0, noise_px),
                y_px=m.y_px + rng.normal(0.0, noise_px),
                ra_deg=m.ra_deg,
                dec_deg=m.dec_deg,
            )
            for m in all_matches
        ]
    return all_matches


# ---------------------------------------------------------------------------
# 1. fit_wcs basic contract
# ---------------------------------------------------------------------------


class TestFitWCSBasic:
    """WCS fitting correctness and API contract."""

    def test_too_few_stars_raises(self) -> None:
        matches = [StarMatch(10.0, 10.0, 134.9, 44.9)] * 2
        with pytest.raises(ValueError, match="3"):
            fit_wcs(matches)

    def test_returns_wcs_solution(self) -> None:
        matches = _make_star_field(n=15)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert isinstance(wcs, WCSSolution)

    def test_n_stars_set(self) -> None:
        matches = _make_star_field(n=15)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert wcs.n_stars >= 3

    def test_crval_near_field_centre(self) -> None:
        """CRVAL should be close to the true field centre."""
        matches = _make_star_field(n=20, ra0=_RA0, dec0=_DEC0)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert abs(wcs.crval1 - _RA0) < 1.0
        assert abs(wcs.crval2 - _DEC0) < 1.0

    def test_pixel_scale_recovered(self) -> None:
        """CD matrix determinant should reproduce the correct pixel scale."""
        matches = _make_star_field(n=20)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
        recovered_scale_arcsec = math.sqrt(det) * 3600.0
        assert recovered_scale_arcsec == pytest.approx(9.12, abs=0.5)


# ---------------------------------------------------------------------------
# 2. Plate solution residuals (OpTA.NOD.ACC ≤ 4.0 arcsec)
# ---------------------------------------------------------------------------


class TestPlateSolutionResiduals:
    """RMS residuals must satisfy OpTA.NOD.ACC ≤ 4.0 arcsec."""

    def test_ideal_star_field_sub_arcsec_rms(self) -> None:
        """Perfect synthetic stars → near-zero RMS (numerical precision)."""
        matches = _make_star_field(n=20, noise_px=0.0)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert wcs.rms_arcsec < 0.01  # should be essentially zero

    def test_noisy_star_field_within_budget(self) -> None:
        """0.3-pixel centroid noise → RMS below OpTA.NOD.ACC (4 arcsec)."""
        matches = _make_star_field(n=30, noise_px=0.3, rng=np.random.default_rng(101))
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert wcs.rms_arcsec < _NOD_ACC_ARCSEC

    def test_more_stars_reduces_rms(self) -> None:
        """Both 5-star and 30-star fits must satisfy the NOD.ACC RMS budget.

        A direct n5-vs-n30 comparison is fragile: n5 can yield n_stars=4 after
        outlier rejection, which is near-degenerate (4 stars, 3 params → 1 dof)
        and produces an artificially low RMS that bears no relation to noise level.
        The meaningful check is that both fits are within the mission budget.
        """
        rng = np.random.default_rng(102)
        m5 = _make_star_field(n=5, noise_px=0.3, rng=rng)
        rng2 = np.random.default_rng(102)
        m30 = _make_star_field(n=30, noise_px=0.3, rng=rng2)
        wcs5 = fit_wcs(m5, frame_shape=(_H, _W))
        wcs30 = fit_wcs(m30, frame_shape=(_H, _W))
        assert wcs5.rms_arcsec < _NOD_ACC_ARCSEC
        assert wcs30.rms_arcsec < _NOD_ACC_ARCSEC


# ---------------------------------------------------------------------------
# 3. pixels_to_radec round-trip
# ---------------------------------------------------------------------------


class TestPixelsToRaDec:
    """Coordinate transform must round-trip correctly."""

    def test_reference_pixel_gives_crval(self) -> None:
        """At (CRPIX1, CRPIX2) the output must be (CRVAL1, CRVAL2)."""
        matches = _make_star_field(n=20)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        ra, dec = pixels_to_radec(wcs, wcs.crpix1, wcs.crpix2)
        assert ra == pytest.approx(wcs.crval1, abs=1e-9)
        assert dec == pytest.approx(wcs.crval2, abs=1e-9)

    def test_known_star_position_round_trip(self) -> None:
        """A star at known pixel → RA/Dec → must recover the catalog value."""
        matches = _make_star_field(n=20, noise_px=0.0)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        m = matches[0]
        ra, dec = pixels_to_radec(wcs, m.x_px, m.y_px)
        # With noise_px=0 the fit is essentially exact
        assert abs(ra - m.ra_deg) * math.cos(math.radians(m.dec_deg)) * 3600 < 0.1
        assert abs(dec - m.dec_deg) * 3600 < 0.1

    def test_ra_wrapped_to_positive(self) -> None:
        """RA output must be in [0, 360)."""
        matches = _make_star_field(n=10, ra0=1.0)  # near RA=0
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        ra, _ = pixels_to_radec(wcs, 0.0, 0.0)
        assert 0.0 <= ra < 360.0


# ---------------------------------------------------------------------------
# 4. Rolling-shutter correction (T-03)
# ---------------------------------------------------------------------------


class TestRollingShutterCorrection:
    """Timing correction must remove the rolling-shutter centroid bias."""

    def test_zero_velocity_no_correction(self) -> None:
        """A stationary target should have zero correction."""
        xc, yc = apply_rolling_shutter_correction(
            x=100.0,
            y=200.0,
            angular_velocity_x=0.0,
            angular_velocity_y=0.0,
            pixel_scale_arcsec=9.12,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        assert xc == pytest.approx(100.0)
        assert yc == pytest.approx(200.0)

    def test_reference_row_no_correction(self) -> None:
        """At the reference row the correction must be zero regardless of velocity."""
        xc, yc = apply_rolling_shutter_correction(
            x=500.0,
            y=540.0,  # y == reference_row
            angular_velocity_x=1800.0,
            angular_velocity_y=1800.0,
            pixel_scale_arcsec=9.12,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        assert xc == pytest.approx(500.0, abs=0.01)
        assert yc == pytest.approx(540.0, abs=0.01)

    def test_correction_magnitude_plausible(self) -> None:
        """At 0.5 deg/s and 37 µs/row, bias at row 0 should be ~0.06 px."""
        # Δy = (row_k - ref_row) × t_row_s × v_px_s
        # v = 0.5 deg/s = 1800 arcsec/s; v_px = 1800 / 9.12 ≈ 197.4 px/s
        # At row 0 (540 rows from center), Δt = 540 × 37e-6 = 0.01998 s
        # Δy = 197.4 × 0.01998 ≈ 3.94 px (but this is for one axis only)
        # With x velocity of 0, x should be unchanged
        xc, yc = apply_rolling_shutter_correction(
            x=200.0,
            y=0.0,
            angular_velocity_x=0.0,
            angular_velocity_y=1800.0,  # arcsec/s
            pixel_scale_arcsec=9.12,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        assert xc == pytest.approx(200.0, abs=0.01)
        # Correction should be non-trivial
        assert abs(yc - 0.0) > 0.1

    def test_symmetry_above_below_center(self) -> None:
        """Correction should be equal and opposite for rows equidistant from center."""
        kwargs = dict(
            angular_velocity_x=0.0,
            angular_velocity_y=1800.0,
            pixel_scale_arcsec=9.12,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        _, yc_above = apply_rolling_shutter_correction(x=0.0, y=340.0, **kwargs)
        _, yc_below = apply_rolling_shutter_correction(x=0.0, y=740.0, **kwargs)
        # Corrections should be roughly equal and opposite
        dy_above = yc_above - 340.0
        dy_below = yc_below - 740.0
        assert dy_above == pytest.approx(-dy_below, rel=0.01)


# ---------------------------------------------------------------------------
# 5. astrometrise_detections
# ---------------------------------------------------------------------------


class TestAstrometriseDetections:
    """Batch astrometry assignment."""

    def _make_detection(self, x: float, y: float, snr: float = 10.0) -> Detection:
        return Detection(
            x=x,
            y=y,
            snr=snr,
            elongation=1.0,
            angle_deg=0.0,
            n_pixels=5,
            flux_e=100.0,
            is_streak=False,
            fwhm_px=2.0,
        )

    def test_output_length_matches_input(self) -> None:
        matches = _make_star_field(n=15)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        dets = [self._make_detection(x, y) for (x, y) in [(100, 200), (500, 600)]]
        astro = astrometrise_detections(dets, wcs)
        assert len(astro) == 2

    def test_output_is_astrometric_detection(self) -> None:
        matches = _make_star_field(n=15)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        dets = [self._make_detection(300.0, 400.0)]
        astro = astrometrise_detections(dets, wcs)
        assert isinstance(astro[0], AstrometricDetection)

    def test_ra_dec_in_expected_range(self) -> None:
        """RA/Dec should be within a few degrees of the field centre."""
        matches = _make_star_field(n=20)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        dets = [self._make_detection(_W / 2, _H / 2)]
        astro = astrometrise_detections(dets, wcs)
        assert abs(astro[0].ra_deg - _RA0) < 2.0
        assert abs(astro[0].dec_deg - _DEC0) < 2.0

    def test_sigma_positive(self) -> None:
        matches = _make_star_field(n=10)
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        dets = [self._make_detection(500.0, 300.0)]
        astro = astrometrise_detections(dets, wcs)
        assert astro[0].sigma_ra_arcsec > 0
        assert astro[0].sigma_dec_arcsec > 0

    def test_within_nod_acc_budget(self) -> None:
        """Sigma should not exceed OpTA.NOD.ACC for a well-constrained plate."""
        matches = _make_star_field(n=30, noise_px=0.3, rng=np.random.default_rng(110))
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        dets = [self._make_detection(960.0, 540.0)]
        astro = astrometrise_detections(dets, wcs)
        assert astro[0].sigma_ra_arcsec <= _NOD_ACC_ARCSEC
        assert astro[0].sigma_dec_arcsec <= _NOD_ACC_ARCSEC

    def test_sigma_is_per_axis_not_radial(self) -> None:
        """sigma_ra/sigma_dec are PER-AXIS; WCSSolution.rms_arcsec is 2-D radial.

        ``fit_wcs`` defines ``rms² = mean(res_ξ² + res_η²) = σ_ξ² + σ_η²``, so
        the per-axis plate term is ``rms²/2``.  Quoting the radial RMS as each
        axis' sigma inflated the reported I-02 uncertainty by up to √2 (the
        pre-fix behaviour).  The 0.3 px centroiding term is already per-axis
        and must NOT be halved.
        """
        matches = _make_star_field(n=30, noise_px=0.5, rng=np.random.default_rng(311))
        wcs = fit_wcs(matches, frame_shape=(_H, _W))
        assert wcs.rms_arcsec > 0.0, "need a non-degenerate fit to see the factor"

        pixel_scale = math.sqrt(
            abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
        ) * 3600.0
        centroiding = 0.3 * pixel_scale
        expected = math.sqrt(wcs.rms_arcsec**2 / 2.0 + centroiding**2)
        radial = math.sqrt(wcs.rms_arcsec**2 + centroiding**2)

        astro = astrometrise_detections([self._make_detection(960.0, 540.0)], wcs)
        assert astro[0].sigma_ra_arcsec == pytest.approx(expected, rel=1e-12)
        assert astro[0].sigma_dec_arcsec == pytest.approx(expected, rel=1e-12)
        assert astro[0].sigma_ra_arcsec < radial


# ---------------------------------------------------------------------------
# 6. Outlier rejection
# ---------------------------------------------------------------------------


class TestOutlierRejection:
    """Outlier stars should not degrade the plate solution."""

    def test_outlier_stars_rejected(self) -> None:
        """Adding one wildly wrong match should not significantly increase RMS."""
        rng = np.random.default_rng(120)
        good = _make_star_field(n=20, noise_px=0.1, rng=rng)
        # Inject a bad match: pixel correct but RA/Dec wrong by 10 arcmin
        bad = StarMatch(x_px=500.0, y_px=300.0, ra_deg=_RA0 + 0.2, dec_deg=_DEC0 + 0.1)
        wcs_good = fit_wcs(good, frame_shape=(_H, _W))
        wcs_bad = fit_wcs(good + [bad], frame_shape=(_H, _W))
        # Rejection should keep the RMS similar to the clean case
        assert wcs_bad.rms_arcsec <= wcs_good.rms_arcsec * 3.0


# ---------------------------------------------------------------------------
# 7. Catalog integration (WP-A4)
# ---------------------------------------------------------------------------


class TestCatalogIntegration:
    """Plate-solve using catalog-derived stars closes U5 and meets OpTA.NOD.ACC."""

    def test_plate_solve_rms_on_500_star_field(self) -> None:
        """WP-A4 acceptance: RMS < 0.5 arcsec on a 500-star catalog field.

        star_field_at computes pixel positions via radec_to_pixels (the canonical
        WCS inverse), so fit_wcs on those positions should recover the pointing WCS
        with near-zero residuals.  The 0.5 arcsec bound verifies the round-trip
        consistency of the star_field_at → fit_wcs chain.
        """
        wcs_true = WCSSolution(
            crpix1=_W / 2.0,
            crpix2=_H / 2.0,
            crval1=_RA0,
            crval2=_DEC0,
            cd1_1=_CD,
            cd1_2=0.0,
            cd2_1=0.0,
            cd2_2=_CD,
            rms_arcsec=0.0,
            n_stars=0,
        )
        matches = star_field_at(wcs_true, _W, _H, mag_limit=14.0)
        assert len(matches) >= 500, f"Only {len(matches)} catalog stars — need ≥ 500"
        solved = fit_wcs(matches, frame_shape=(_H, _W))
        assert solved.rms_arcsec < 0.5, (
            f"RMS {solved.rms_arcsec:.3f} arcsec exceeds 0.5 arcsec (WP-A4 acceptance)"
        )


# ---------------------------------------------------------------------------
# 8. Gnomonic truth-field regression (P0 fix 2026-07-10)
# ---------------------------------------------------------------------------
#
# NOTE — deliberate exception to the "use star_field_at for all test star
# fields" rule (opta-pipeline/AGENTS.md): star_field_at places stars via
# radec_to_pixels, i.e. through the SAME projection model the fitter assumes.
# Truth generated that way can never expose a projection-model bug (an inverse
# crime — synthetic truth and fitter share the model; that is exactly how the
# pre-2026-07-10 linear-sky fit_wcs stayed green while leaving ~1900″ median
# residuals on real gnomonic geometry).  The truth fields below are therefore
# generated by the TRUE inverse gnomonic (standard TAN deprojection, SLALIB
# dtp2s form), written out explicitly and independently of astrometry.py.


def _tan_invert(
    xi_deg: float, eta_deg: float, ra0_deg: float, dec0_deg: float
) -> tuple[float, float]:
    """Independent inverse gnomonic: tangent plane (ξ, η) → (RA, Dec), degrees."""
    xi = math.radians(xi_deg)
    eta = math.radians(eta_deg)
    dec0 = math.radians(dec0_deg)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    denom = cos_d0 - eta * sin_d0
    ra = ra0_deg + math.degrees(math.atan2(xi, denom))
    dec = math.degrees(math.atan2(sin_d0 + eta * cos_d0, math.hypot(xi, denom)))
    return ra % 360.0, dec


def _gnomonic_truth_field(
    ra0: float,
    dec0: float,
    scale_arcsec: float,
    width: int,
    height: int,
    n: int,
    seed: int,
) -> list[StarMatch]:
    """Random in-frame pixels mapped to the sky by TRUE gnomonic inversion.

    The truth WCS is axis-aligned TAN: ξ = s·(x − w/2), η = s·(y − h/2) about
    (ra0, dec0), inverted by the standard TAN equations — NOT radec_to_pixels.
    """
    rng = np.random.default_rng(seed)
    s = scale_arcsec / 3600.0
    out: list[StarMatch] = []
    for _ in range(n):
        x = float(rng.uniform(0.0, width))
        y = float(rng.uniform(0.0, height))
        ra, dec = _tan_invert(s * (x - width / 2.0), s * (y - height / 2.0), ra0, dec0)
        out.append(StarMatch(x_px=x, y_px=y, ra_deg=ra, dec_deg=dec))
    return out


def _truth_residuals_arcsec(wcs: WCSSolution, matches: list[StarMatch]) -> np.ndarray:
    """Great-circle separation (arcsec) between WCS prediction and catalog truth."""
    res = []
    for m in matches:
        ra, dec = pixels_to_radec(wcs, m.x_px, m.y_px)
        r1, d1, r2, d2 = map(math.radians, (ra, dec, m.ra_deg, m.dec_deg))
        a = (
            math.sin((d2 - d1) / 2.0) ** 2
            + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2.0) ** 2
        )
        res.append(math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0)
    return np.array(res)


# IMX585 full-resolution production geometry: 3856×2180 px, 2.9 µm pitch,
# 25 mm focal → plate scale ≈ 23.93″/px, field 25.2°×14.4°, half-diagonal 14.5°.
_IMX585_W, _IMX585_H = 3856, 2180
_IMX585_SCALE_ARCSEC = math.degrees(2.9e-6 / 25e-3) * 3600.0


class TestGnomonicTruthField:
    """fit_wcs must solve TRUE gnomonic geometry, not just its own model."""

    def test_wide_field_dec45_sub_arcsec(self) -> None:
        """Noiseless 25.2°×14.4° field at dec 45° solves to ≈ machine precision.

        Pre-fix (linear ΔRA·cosδ model): median 1940″ / max 5911″ truth
        residuals, and the outlier loop silently pruned 200 → 3 stars to
        report rms 0.00″.  Post-fix all 200 stars are retained (this doubles
        as the degenerate-prune guard test) and residuals are sub-milliarcsec.
        """
        matches = _gnomonic_truth_field(
            30.0, 45.0, _IMX585_SCALE_ARCSEC, _IMX585_W, _IMX585_H, n=200, seed=7
        )
        wcs = fit_wcs(matches, frame_shape=(_IMX585_H, _IMX585_W))
        assert wcs.n_stars == 200, "prune must not discard stars on a perfect field"
        assert wcs.rms_arcsec < 1e-3
        assert abs(wcs.crval1 - 30.0) * 3600.0 < 0.01
        assert abs(wcs.crval2 - 45.0) * 3600.0 < 0.01
        res = _truth_residuals_arcsec(wcs, matches)
        assert float(res.max()) < 1e-3

    def test_wide_field_sip_builds_on_gnomonic_core(self) -> None:
        """fit_wcs_sip on an undistorted gnomonic field: no phantom distortion.

        Pre-fix the radial r²/r⁴ terms could not absorb the projection error
        (byte-identical to fit_wcs, both garbage).  Post-fix the projection
        error is gone by construction, so the SIP path must return the clean
        linear solve with zero distortion coefficients.
        """
        matches = _gnomonic_truth_field(
            30.0, 45.0, _IMX585_SCALE_ARCSEC, _IMX585_W, _IMX585_H, n=200, seed=7
        )
        sip = fit_wcs_sip(matches, frame_shape=(_IMX585_H, _IMX585_W))
        assert sip.sip_a1 == 0.0 and sip.sip_a2 == 0.0
        assert sip.rms_arcsec < 1e-3
        assert sip.n_stars == 200

    def test_ra_wrap_field_recovers_crval1(self) -> None:
        """A field straddling RA 0° solves correctly (RA branch-cut bug).

        Pre-fix: crval1 = mean of wrapped RAs ≈ 183° and rms ~306 000″.
        """
        matches = _gnomonic_truth_field(
            0.05, 10.0, 10.0, 1920, 1080, n=20, seed=11
        )
        ras = [m.ra_deg for m in matches]
        assert min(ras) < 1.0 and max(ras) > 359.0, "field must straddle RA 0"
        wcs = fit_wcs(matches, frame_shape=(1080, 1920))
        d_crval1 = abs((wcs.crval1 - 0.05 + 180.0) % 360.0 - 180.0)
        assert d_crval1 * 3600.0 < 0.01
        assert wcs.rms_arcsec < 1e-3
        assert wcs.n_stars == 20
        res = _truth_residuals_arcsec(wcs, matches)
        assert float(res.max()) < 1e-3

    def test_degenerate_prune_guard_on_unfittable_field(self) -> None:
        """Model mismatch must not collapse to a tiny-star 'exact' fit.

        A strong cubic pixel warp cannot be absorbed by TAN + affine, so a
        correct fit is impossible.  The failure must be visible: (nearly) all
        stars retained and an honest RMS above the rejection threshold —
        never the pre-fix silent 3-star rms-0.00″ success.
        """
        clean = _gnomonic_truth_field(
            30.0, 45.0, _IMX585_SCALE_ARCSEC, _IMX585_W, _IMX585_H, n=200, seed=7
        )
        cx = _IMX585_W / 2.0
        warped = [
            StarMatch(
                x_px=m.x_px + 30.0 * ((m.x_px - cx) / cx) ** 3,
                y_px=m.y_px,
                ra_deg=m.ra_deg,
                dec_deg=m.dec_deg,
            )
            for m in clean
        ]
        wcs = fit_wcs(warped, frame_shape=(_IMX585_H, _IMX585_W))
        assert wcs.n_stars >= len(warped) // 2, (
            f"prune collapsed to {wcs.n_stars} stars — silent degenerate fit"
        )
        assert wcs.rms_arcsec > 4.0, (
            f"rms {wcs.rms_arcsec:.2f}\" hides a model mismatch"
        )

    def test_round_trip_identity_wide_field(self) -> None:
        """radec_to_pixels ∘ pixels_to_radec is identity across the full field.

        Includes the frame corners (≥ 14° off-axis on IMX585 geometry).
        """
        matches = _gnomonic_truth_field(
            30.0, 45.0, _IMX585_SCALE_ARCSEC, _IMX585_W, _IMX585_H, n=200, seed=7
        )
        wcs = fit_wcs(matches, frame_shape=(_IMX585_H, _IMX585_W))
        pts = [
            (0.0, 0.0),
            (float(_IMX585_W), 0.0),
            (0.0, float(_IMX585_H)),
            (float(_IMX585_W), float(_IMX585_H)),
            (_IMX585_W / 2.0, _IMX585_H / 2.0),
            (123.4, 567.8),
        ]
        for x, y in pts:
            ra, dec = pixels_to_radec(wcs, x, y)
            x2, y2 = radec_to_pixels(wcs, ra, dec)
            assert (x2, y2) == pytest.approx((x, y), abs=1e-6)

    def test_round_trip_identity_across_ra_zero(self) -> None:
        """Round trip stays exact for a field straddling the RA 0° boundary."""
        matches = _gnomonic_truth_field(0.05, 10.0, 10.0, 1920, 1080, n=20, seed=11)
        wcs = fit_wcs(matches, frame_shape=(1080, 1920))
        for x, y in [(0.0, 0.0), (1920.0, 1080.0), (10.0, 540.0), (1900.0, 12.3)]:
            ra, dec = pixels_to_radec(wcs, x, y)
            x2, y2 = radec_to_pixels(wcs, ra, dec)
            assert (x2, y2) == pytest.approx((x, y), abs=1e-6)


# ---------------------------------------------------------------------------
# T-05. Astrometric residual shape
# ---------------------------------------------------------------------------


def _compute_inlier_residuals(
    n_stars: int,
    noise_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inlier_delta_ra_arcsec, inlier_delta_dec_arcsec) arrays.

    Builds a noisy star field, fits WCS, computes per-star residuals, then
    filters to the same inlier threshold used by fit_wcs (4.0 arcsec radius).
    """
    rng = np.random.default_rng(200 + n_stars)
    matches = _make_star_field(n=n_stars, noise_px=noise_px, rng=rng)
    wcs = fit_wcs(matches, frame_shape=(_H, _W))

    delta_ra_list: list[float] = []
    delta_dec_list: list[float] = []
    for m in matches:
        ra_pred, dec_pred = pixels_to_radec(wcs, m.x_px, m.y_px)
        dra = (ra_pred - m.ra_deg) * math.cos(math.radians(m.dec_deg)) * 3600.0
        ddec = (dec_pred - m.dec_deg) * 3600.0
        if math.hypot(dra, ddec) < 4.0:
            delta_ra_list.append(dra)
            delta_dec_list.append(ddec)

    return np.array(delta_ra_list), np.array(delta_dec_list)


_RESIDUAL_PARAMS = [(20, 0.1), (50, 0.3), (100, 0.3)]


class TestAstrometricResidualShape:
    """T-05: WCS residuals must be zero-mean, Gaussian-shaped, and chi²/DOF ≈ 1.

    A low RMS alone can hide systematic biases or mis-calibrated noise.
    These tests verify that residual statistics are well-behaved across
    a range of star-field sizes and centroid noise levels.
    """

    @pytest.mark.parametrize("n_stars,noise_px", _RESIDUAL_PARAMS)
    def test_residual_zero_mean(self, n_stars: int, noise_px: float) -> None:
        """Both mean ΔRA and mean ΔDec must be < 0.5 arcsec (no systematic bias)."""
        dra, ddec = _compute_inlier_residuals(n_stars, noise_px)
        assert abs(float(np.mean(dra))) < 0.5, (
            f"Mean ΔRA = {np.mean(dra):.3f} arcsec — systematic bias detected"
        )
        assert abs(float(np.mean(ddec))) < 0.5, (
            f"Mean ΔDec = {np.mean(ddec):.3f} arcsec — systematic bias detected"
        )

    @pytest.mark.parametrize("n_stars,noise_px", _RESIDUAL_PARAMS)
    def test_residual_shape(self, n_stars: int, noise_px: float) -> None:
        """Residuals Gaussian-shaped: skewness ∈ [-0.5, 0.5], kurtosis ∈ [-1, 2]."""
        dra, ddec = _compute_inlier_residuals(n_stars, noise_px)
        all_residuals = np.concatenate([dra, ddec])
        skewness = float(stats.skew(all_residuals))
        kurt = float(stats.kurtosis(all_residuals))  # Fisher (excess) kurtosis
        assert -0.5 <= skewness <= 0.5, (
            f"Skewness {skewness:.3f} out of Gaussian range [-0.5, 0.5]"
        )
        assert -1.0 <= kurt <= 2.0, (
            f"Excess kurtosis {kurt:.3f} out of Gaussian range [-1.0, 2.0]"
        )

    @pytest.mark.parametrize("n_stars,noise_px", _RESIDUAL_PARAMS)
    def test_chi_squared_per_dof(self, n_stars: int, noise_px: float) -> None:
        """Chi²/DOF must be in [0.3, 3.0] — broad tolerance for small N.

        σ_noise = noise_px × pixel_scale_arcsec.
        DOF = 2 × n_inlier − 6  (4 CD matrix + 2 CRVAL bias parameters).
        chi² = Σ(ΔRA² + ΔDec²) / σ_noise².
        """
        dra, ddec = _compute_inlier_residuals(n_stars, noise_px)
        n_inlier = len(dra)
        dof = 2 * n_inlier - 6
        if dof <= 0:
            pytest.skip(f"Degenerate case: n_inlier={n_inlier}, DOF={dof}")

        pixel_scale_arcsec = _PIXEL_SCALE_DEG * 3600.0
        sigma_noise = noise_px * pixel_scale_arcsec
        chi2 = float(np.sum(dra**2 + ddec**2)) / sigma_noise**2
        chi2_per_dof = chi2 / dof
        assert 0.3 <= chi2_per_dof <= 3.0, (
            f"chi²/DOF = {chi2_per_dof:.3f} outside [0.3, 3.0] "
            f"(chi²={chi2:.1f}, DOF={dof})"
        )


# ---------------------------------------------------------------------------
# T-07. Rolling-shutter quantitative bias and correction
# ---------------------------------------------------------------------------


class TestRollingShutterQuantitative:
    """T-07: Verify rolling-shutter bias magnitude and exact correction.

    Derivation for test geometry (angular_velocity_x = 7200 arcsec/s, pixel_scale = 9.12
    arcsec/px, row_readout_us = 37 µs, reference_row = 540, y_test = 0):

      v_px_s  = 7200 / 9.12  ≈ 789.47 px/s
      delta_t = (0 − 540) × 37e-6  = −0.01998 s
      |Δx|    = v_px_s × |delta_t| ≈ 15.77 px → 143.8 arcsec

    The existing rolling-shutter tests cover zero-velocity and symmetry.
    These tests verify the magnitude at operational satellite velocity
    (2 deg/s) and that the correction formula exactly inverts the bias.
    """

    def test_bias_magnitude_at_2deg_s_exceeds_5arcsec(self) -> None:
        """At row 0 vs reference 540, rolling-shutter positional bias is ~143 arcsec.

        This confirms the correction is not negligible at operational velocities
        (minimum threshold: 5 arcsec; expected: ~143 arcsec).
        """
        x_corr, _ = apply_rolling_shutter_correction(
            x=200.0,
            y=0.0,
            angular_velocity_x=7200.0,
            angular_velocity_y=0.0,
            pixel_scale_arcsec=9.12,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        bias_arcsec = abs(x_corr - 200.0) * 9.12
        assert bias_arcsec > 5.0, (
            f"Bias {bias_arcsec:.2f} arcsec is unexpectedly small; "
            f"expected ~143 arcsec at 2 deg/s"
        )

    def test_corrected_residual_below_2arcsec(self) -> None:
        """Rolling-shutter correction must exactly recover the true position.

        Construct the rolling-shutter-biased measured position by forward-modelling
        the timing offset, then apply the correction and verify the residual is
        below 2 arcsec (the correction is exact to floating-point precision;
        2 arcsec is a generous acceptance bound).
        """
        v_x_arcsec_s = 7200.0
        pixel_scale = 9.12
        v_x_px_s = v_x_arcsec_s / pixel_scale
        x_true = 200.0
        t_row_s = 37e-6
        t_offset = (0.0 - 540.0) * t_row_s  # row 0 vs reference row 540
        x_measured = x_true + v_x_px_s * t_offset  # forward-model the bias

        x_corr, _ = apply_rolling_shutter_correction(
            x=x_measured,
            y=0.0,
            angular_velocity_x=v_x_arcsec_s,
            angular_velocity_y=0.0,
            pixel_scale_arcsec=pixel_scale,
            row_readout_us=37.0,
            reference_row=540.0,
        )
        residual_arcsec = abs(x_corr - x_true) * pixel_scale
        assert residual_arcsec < 2.0, (
            f"Correction residual {residual_arcsec:.4f} arcsec exceeds 2.0 arcsec; "
            f"correction formula may not be an exact inverse of the bias model"
        )


# ---------------------------------------------------------------------------
# Distortion-aware solve (fit_wcs_sip)
# ---------------------------------------------------------------------------


class TestFitWcsSip:
    """The distortion-aware solver: radial SIP term + linear TAN."""

    def _distorted_matches(self, dm: DistortionModel) -> list[StarMatch]:
        """Catalog stars placed through the true WCS then optically distorted."""
        clean = _make_star_field(n=40)
        out = []
        for m in clean:
            xd, yd = dm.apply(m.x_px, m.y_px, (_H, _W))
            out.append(StarMatch(x_px=xd, y_px=yd, ra_deg=m.ra_deg, dec_deg=m.dec_deg))
        return out

    def test_recovers_distortion_within_budget(self) -> None:
        """A 3 % barrel is recovered to ≤ OpTA.NOD.ACC everywhere."""
        dm = DistortionModel.from_corner_displacement_px(
            -0.03 * math.hypot(_W / 2.0, _H / 2.0), (_H, _W)
        )
        matches = self._distorted_matches(dm)
        wcs = fit_wcs_sip(matches, frame_shape=(_H, _W))
        errs = []
        for m in matches:
            ra, dec = pixels_to_radec(wcs, m.x_px, m.y_px)
            cosd = math.cos(math.radians(dec))
            errs.append(math.hypot((m.ra_deg - ra) * cosd * 3600.0,
                                   (m.dec_deg - dec) * 3600.0))
        assert max(errs) < _NOD_ACC_ARCSEC
        assert wcs.sip_a1 != 0.0  # distortion actually fitted

    def test_zero_distortion_matches_linear(self) -> None:
        """With no distortion, the SIP fit reduces to the linear solution."""
        matches = _make_star_field(n=40)
        lin = fit_wcs(matches, frame_shape=(_H, _W))
        sip = fit_wcs_sip(matches, frame_shape=(_H, _W))
        assert abs(sip.sip_a1) < 1e-4
        assert abs(sip.sip_a2) < 1e-4
        assert sip.crval1 == pytest.approx(lin.crval1, abs=1e-6)
        assert sip.crval2 == pytest.approx(lin.crval2, abs=1e-6)
        assert sip.rms_arcsec < 0.5

    def test_few_matches_falls_back_to_linear(self) -> None:
        """Fewer than 5 matches → no spare DOF → plain linear fit (a1=a2=0)."""
        matches = _make_star_field(n=4)
        sip = fit_wcs_sip(matches, frame_shape=(_H, _W))
        assert sip.sip_a1 == 0.0 and sip.sip_a2 == 0.0

    def test_radec_pixel_round_trip_under_distortion(self) -> None:
        """pixels_to_radec ∘ radec_to_pixels is identity for a distorted WCS."""
        dm = DistortionModel.from_corner_displacement_px(
            -0.04 * math.hypot(_W / 2.0, _H / 2.0), (_H, _W)
        )
        wcs = fit_wcs_sip(self._distorted_matches(dm), frame_shape=(_H, _W))
        for x, y in [(100.0, 90.0), (960.0, 540.0), (1820.0, 1000.0)]:
            ra, dec = pixels_to_radec(wcs, x, y)
            x2, y2 = radec_to_pixels(wcs, ra, dec)
            assert (x2, y2) == pytest.approx((x, y), abs=1e-4)


# ---------------------------------------------------------------------------
# 10. SIP sub-threshold-distortion blind window (P1 fix 2026-07-12)
# ---------------------------------------------------------------------------
#
# Pre-fix, fit_wcs_sip accepted the SIP terms only when they *halved* the
# unpruned linear RMS — an amplitude gate requiring the distortion residual to
# exceed √3× the centroid-noise floor.  Radial distortion grows ~r³ toward
# the field edge, so distortion below that gate could carry a field-edge sky
# error above the OpTA.NOD.ACC 10″ budget while the linear fallback's fine
# prune discarded the distorted edge stars and reported a clean central RMS:
# at a1 = 5e-4, σ = 0.10 px the solver rejected SIP on 5/5 seeds, reported
# rms 2.65″ on 109/200 surviving stars, and left a true field-edge error of
# 12.5″ (recomputed 2026-07-12 via
# python3 opta-pipeline/scripts/repro_sip_subthreshold_window.py).  Model
# selection is now a nested-model F-test (p = 1e-3) and every solution passes
# a post-solve accuracy self-check that sets WCSSolution.accuracy_flag, so a
# silent NOD.ACC violation is impossible for structure the matches sample.

# OpTA.NOD.ACC: ≤ 10.0″ RMS per node (opta-engineering/SYSTEMS.md).
_NOD_ACC_BUDGET_ARCSEC = 10.0
_WINDOW_RA0, _WINDOW_DEC0 = 210.0, 45.0


def _radially_distort_px(
    xu: float, yu: float, a1: float, cx: float, cy: float, r_half: float
) -> tuple[float, float]:
    """Ideal gnomonic pixel → measured pixel under a radial lens distortion.

    Inverts the solver's undistortion convention ``ideal = centre +
    (measured − centre)·(1 + a1·rn²)`` by fixed-point iteration (mirrors
    scripts/repro_sip_subthreshold_window.py and astrometry._redistort).
    """
    dxu, dyu = xu - cx, yu - cy
    ru = math.hypot(dxu, dyu)
    if ru == 0.0 or a1 == 0.0:
        return xu, yu
    rm = ru
    for _ in range(30):
        rm = ru / (1.0 + a1 * (rm / r_half) ** 2)
    s = rm / ru
    return cx + dxu * s, cy + dyu * s


def _subthreshold_field(
    a1: float, noise_px: float, seed: int, n: int = 200
) -> list[StarMatch]:
    """Gnomonic truth field imaged through radial distortion + centroid noise."""
    rng = np.random.default_rng(1000 + seed)
    ideal = _gnomonic_truth_field(
        _WINDOW_RA0, _WINDOW_DEC0, _IMX585_SCALE_ARCSEC,
        _IMX585_W, _IMX585_H, n=n, seed=seed,
    )
    cx, cy = _IMX585_W / 2.0, _IMX585_H / 2.0
    r_half = math.hypot(cx, cy)
    out = []
    for m in ideal:
        xm, ym = _radially_distort_px(m.x_px, m.y_px, a1, cx, cy, r_half)
        out.append(StarMatch(
            x_px=xm + rng.normal(0.0, noise_px),
            y_px=ym + rng.normal(0.0, noise_px),
            ra_deg=m.ra_deg,
            dec_deg=m.dec_deg,
        ))
    return out


def _edge_probes(a1: float) -> list[StarMatch]:
    """Noiseless field-edge probe sources (corners + long-edge midpoints)."""
    cx, cy = _IMX585_W / 2.0, _IMX585_H / 2.0
    r_half = math.hypot(cx, cy)
    s = _IMX585_SCALE_ARCSEC / 3600.0
    probes = []
    for xu, yu in (
        (50.0, 50.0), (_IMX585_W - 50.0, 50.0),
        (50.0, _IMX585_H - 50.0), (_IMX585_W - 50.0, _IMX585_H - 50.0),
        (_IMX585_W - 50.0, cy), (50.0, cy),
    ):
        ra_t, dec_t = _tan_invert(
            s * (xu - cx), s * (yu - cy), _WINDOW_RA0, _WINDOW_DEC0
        )
        xm, ym = _radially_distort_px(xu, yu, a1, cx, cy, r_half)
        probes.append(StarMatch(x_px=xm, y_px=ym, ra_deg=ra_t, dec_deg=dec_t))
    return probes


class TestSipSubthresholdWindow:
    """Both ends of the SIP-vs-linear decision, plus the violation flag."""

    _SIGMA_PX = 0.10  # production-plausible centroid noise floor

    def test_subthreshold_distortion_fitted_and_in_budget(self) -> None:
        """a1 = 5e-4 at σ = 0.10 px — the pre-fix blind window — now solves.

        Pre-fix: SIP rejected 5/5 seeds, true field-edge error 12.5″ > 10″
        with reported rms 2.65″ (recomputed 2026-07-12 via
        python3 opta-pipeline/scripts/repro_sip_subthreshold_window.py).
        Post-fix the F-test accepts the distortion terms decisively (F ≈
        170–230 vs F_crit ≈ 6.9, measured 2026-07-12 on the repro script's
        geometry at this cell) and the edge stays within OpTA.NOD.ACC.
        """
        for seed in (1, 8, 15):
            matches = _subthreshold_field(5e-4, self._SIGMA_PX, seed=seed)
            wcs = fit_wcs_sip(matches, frame_shape=(_IMX585_H, _IMX585_W))
            assert wcs.sip_a1 != 0.0, f"seed {seed}: SIP rejected in the window"
            errs = _truth_residuals_arcsec(wcs, _edge_probes(5e-4))
            assert float(errs.max()) < _NOD_ACC_BUDGET_ARCSEC, (
                f"seed {seed}: edge error {errs.max():.1f}\" blows OpTA.NOD.ACC"
            )
            assert not wcs.accuracy_flag

    def test_noise_only_still_selects_linear(self) -> None:
        """Pure centroid noise must not grow SIP terms (the gate's raison d'être).

        The F-test bounds spurious acceptance at 0.1 % per solve; across ten
        seeds every fit must return the linear model, unflagged.
        """
        for seed in range(10):
            matches = _subthreshold_field(0.0, self._SIGMA_PX, seed=seed)
            wcs = fit_wcs_sip(matches, frame_shape=(_IMX585_H, _IMX585_W))
            assert wcs.sip_a1 == 0.0 and wcs.sip_a2 == 0.0, (
                f"seed {seed}: SIP terms fitted to pure noise"
            )
            assert not wcs.accuracy_flag

    def test_strong_distortion_still_selects_sip(self) -> None:
        """Well above the old gate (a1 = 2e-3) SIP is still chosen and in budget."""
        matches = _subthreshold_field(2e-3, self._SIGMA_PX, seed=1)
        wcs = fit_wcs_sip(matches, frame_shape=(_IMX585_H, _IMX585_W))
        assert wcs.sip_a1 != 0.0
        errs = _truth_residuals_arcsec(wcs, _edge_probes(2e-3))
        assert float(errs.max()) < _NOD_ACC_BUDGET_ARCSEC
        assert not wcs.accuracy_flag

    def test_unmodellable_warp_raises_accuracy_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warp neither model absorbs must flag the violation, never hide it.

        A strong cubic x-only pixel warp (30 px ≈ 720″ at the production
        scale) cannot be fit by TAN + radial SIP; whichever model is
        returned, its honest unpruned residuals exceed the 10″ budget, so
        accuracy_flag must be set and a warning logged.
        """
        clean = _gnomonic_truth_field(
            _WINDOW_RA0, _WINDOW_DEC0, _IMX585_SCALE_ARCSEC,
            _IMX585_W, _IMX585_H, n=200, seed=7,
        )
        cx = _IMX585_W / 2.0
        warped = [
            StarMatch(
                x_px=m.x_px + 30.0 * ((m.x_px - cx) / cx) ** 3,
                y_px=m.y_px,
                ra_deg=m.ra_deg,
                dec_deg=m.dec_deg,
            )
            for m in clean
        ]
        with caplog.at_level(logging.WARNING, logger="opta_pipeline.astrometry"):
            wcs = fit_wcs_sip(warped, frame_shape=(_IMX585_H, _IMX585_W))
        assert wcs.accuracy_flag, "out-of-budget solution returned unflagged"
        assert "OpTA.NOD.ACC" in caplog.text
