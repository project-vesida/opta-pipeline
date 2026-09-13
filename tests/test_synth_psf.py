"""Tests for the Phase 2 PSF model: field-variable, sub-pixel-accurate rendering.

The two properties that matter for tuning the pipeline's astrometry /
plate-solving:

* **Sub-pixel centroid is unbiased** — the integrated (erf) PSF places the
  flux-weighted centroid exactly at the injected position for any sub-pixel
  phase, so any astrometric error measured downstream comes from noise, not the
  generator.
* **Aberration broadens the field edges** — the recovered FWHM grows with field
  radius, letting the detector/centroider be stressed with realistic blur.
"""

from __future__ import annotations

import numpy as np
import pytest
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.synth import (
    StarSpec,
    build_detector_model,
    generate_frame,
)
from opta_pipeline.synth.psf import (
    PSFModel,
    render_gaussian,
    render_source,
    render_streak,
)


def _moments(f: np.ndarray):
    """Flux-weighted centroid and second moments (sxx, syy, sxy)."""
    ys, xs = np.mgrid[0 : f.shape[0], 0 : f.shape[1]]
    tot = f.sum()
    mx = (f * xs).sum() / tot
    my = (f * ys).sum() / tot
    sxx = (f * (xs - mx) ** 2).sum() / tot
    syy = (f * (ys - my) ** 2).sum() / tot
    sxy = (f * (xs - mx) * (ys - my)).sum() / tot
    return mx, my, sxx, syy, sxy


def _fwhm_halfmax_row(img: np.ndarray, yc: int, xc: int) -> float:
    """FWHM along the central row via linearly-interpolated half-maximum
    crossings (profile-shape independent — unlike second moments, which inflate
    for heavy wings)."""
    row = img[yc].astype(float)
    half = row[xc] / 2.0
    left = xc
    while left > 0 and row[left] > half:
        left -= 1
    # Interpolate between `left` (≤ half) and `left+1` (> half).
    xl = left + (half - row[left]) / (row[left + 1] - row[left])
    right = xc
    while right < len(row) - 1 and row[right] > half:
        right += 1
    # Interpolate between `right-1` (> half) and `right` (≤ half).
    xr = (right - 1) + (row[right - 1] - half) / (row[right - 1] - row[right])
    return float(xr - xl)


def _radial_fwhm(img: np.ndarray, xc: int, yc: int, half: int = 8, bg: float = 0.0):
    """Background-subtracted second-moment FWHM in a window about (xc, yc)."""
    b = img[yc - half : yc + half + 1, xc - half : xc + half + 1].astype(float) - bg
    b = np.clip(b, 0.0, None)
    ys, xs = np.mgrid[0 : b.shape[0], 0 : b.shape[1]]
    tot = b.sum()
    mx = (b * xs).sum() / tot
    my = (b * ys).sum() / tot
    var = (b * ((xs - mx) ** 2 + (ys - my) ** 2)).sum() / (2.0 * tot)
    return float(np.sqrt(var) * 2.3548)


# ---------------------------------------------------------------------------
# PSFModel
# ---------------------------------------------------------------------------


class TestPSFModel:
    def test_uniform_when_edge_equals_center(self) -> None:
        p = PSFModel(fwhm_center_px=2.5, fwhm_edge_px=2.5)
        shape = (300, 400)
        assert p.fwhm_at(200, 150, shape) == 2.5
        assert p.fwhm_at(0, 0, shape) == 2.5

    def test_radial_growth(self) -> None:
        p = PSFModel(fwhm_center_px=2.0, fwhm_edge_px=5.0, radial_power=2.0)
        shape = (300, 400)
        center = p.fwhm_at(199.5, 149.5, shape)  # exact centre
        corner = p.fwhm_at(0, 0, shape)
        assert center == pytest.approx(2.0, abs=1e-6)
        assert corner == pytest.approx(5.0, abs=1e-6)
        # Monotonic: a mid-field point lies strictly between.
        mid = p.fwhm_at(100, 75, shape)
        assert center < mid < corner


# ---------------------------------------------------------------------------
# render_gaussian
# ---------------------------------------------------------------------------


class TestRenderGaussian:
    def test_flux_conserved(self) -> None:
        f = np.zeros((40, 40))
        render_gaussian(f, 20.0, 20.0, 1000.0, 2.5)
        assert f.sum() == pytest.approx(1000.0, rel=1e-4)

    def test_subpixel_centroid_unbiased(self) -> None:
        """The flux-weighted centroid equals the injected sub-pixel position."""
        for x0, y0 in [(20.0, 20.0), (20.37, 19.62), (20.5, 20.5), (19.9, 20.1)]:
            f = np.zeros((40, 40))
            render_gaussian(f, x0, y0, 1000.0, 2.0)
            ys, xs = np.mgrid[0:40, 0:40]
            tot = f.sum()
            cx = (f * xs).sum() / tot
            cy = (f * ys).sum() / tot
            assert cx == pytest.approx(x0, abs=5e-3)
            assert cy == pytest.approx(y0, abs=5e-3)

    def test_measured_fwhm_matches_input(self) -> None:
        f = np.zeros((60, 60))
        render_gaussian(f, 30.0, 30.0, 1e6, 4.0)
        assert _radial_fwhm(f, 30, 30, half=12) == pytest.approx(4.0, rel=0.03)

    def test_edge_clipping_conserves_only_in_frame_flux(self) -> None:
        """A source near the corner deposits a partial PSF without error."""
        f = np.zeros((40, 40))
        render_gaussian(f, 0.5, 0.5, 1000.0, 2.0)  # most of the wing is off-frame
        assert 0.0 < f.sum() < 1000.0

    def test_zero_sigma_is_noop(self) -> None:
        f = np.zeros((10, 10))
        render_gaussian(f, 5.0, 5.0, 100.0, 0.0)
        assert f.sum() == 0.0


# ---------------------------------------------------------------------------
# render_streak
# ---------------------------------------------------------------------------


class TestRenderStreak:
    def test_flux_conserved(self) -> None:
        f = np.zeros((60, 60))
        render_streak(f, 20.0, 30.0, 40.0, 30.0, 5000.0, PSFModel(2.0, 2.0))
        assert f.sum() == pytest.approx(5000.0, rel=1e-3)

    def test_centroid_at_line_midpoint(self) -> None:
        f = np.zeros((60, 60))
        render_streak(f, 20.0, 30.0, 40.0, 30.0, 5000.0, PSFModel(2.0, 2.0))
        ys, xs = np.mgrid[0:60, 0:60]
        tot = f.sum()
        assert (f * xs).sum() / tot == pytest.approx(30.0, abs=0.1)
        assert (f * ys).sum() / tot == pytest.approx(30.0, abs=0.1)

    def test_degenerate_streak_falls_back_to_point(self) -> None:
        f = np.zeros((40, 40))
        render_streak(f, 20.0, 20.0, 20.0, 20.0, 1000.0, PSFModel(2.0, 2.0))
        assert f.sum() == pytest.approx(1000.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Field-variable PSF in a rendered frame
# ---------------------------------------------------------------------------


class TestAberrationInFrame:
    def test_corner_star_broader_than_center(self) -> None:
        """With an aberrated PSF, the corner star is measurably broader."""
        # Clean-ish detector so the second moment reflects the PSF, not defects.
        det = build_detector_model(
            SENSOR_SMALL, seed=1, prnu_pct=0.0, bias_fpn_rms_e=0.0,
            bias_offset_e=0.0, hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
            vignetting_corner_factor=1.0,
        )
        psf = PSFModel(fwhm_center_px=2.0, fwhm_edge_px=6.0)
        frame = generate_frame(
            SENSOR_SMALL, OPTICS_DEFAULT,
            stars=[StarSpec(magnitude=6.0, x=200, y=150),  # centre
                   StarSpec(magnitude=6.0, x=10, y=10)],   # corner
            detector=det, psf=psf, rng=np.random.default_rng(0),
        )
        bg = float(np.median(frame.data_float))
        fwhm_center = _radial_fwhm(frame.data_float, 200, 150, half=10, bg=bg)
        fwhm_corner = _radial_fwhm(frame.data_float, 10, 10, half=10, bg=bg)
        assert fwhm_corner > fwhm_center + 1.0, (
            f"corner FWHM {fwhm_corner:.2f} not broader than center "
            f"{fwhm_center:.2f}"
        )

    def test_uniform_psf_default_is_isotropic_across_field(self) -> None:
        """Default (no PSF arg) renders a uniform PSF — no field dependence."""
        det = build_detector_model(
            SENSOR_SMALL, seed=2, prnu_pct=0.0, bias_fpn_rms_e=0.0,
            bias_offset_e=0.0, hot_pixel_fraction=0.0, vignetting_corner_factor=1.0,
        )
        frame = generate_frame(
            SENSOR_SMALL, OPTICS_DEFAULT,
            stars=[StarSpec(magnitude=6.0, x=200, y=150),
                   StarSpec(magnitude=6.0, x=10, y=10)],
            detector=det, rng=np.random.default_rng(0),
        )
        bg = float(np.median(frame.data_float))
        fwhm_center = _radial_fwhm(frame.data_float, 200, 150, half=10, bg=bg)
        fwhm_corner = _radial_fwhm(frame.data_float, 10, 10, half=10, bg=bg)
        assert abs(fwhm_corner - fwhm_center) < 0.5


# ---------------------------------------------------------------------------
# Phase 2b: Moffat profile + anisotropy (coma)
# ---------------------------------------------------------------------------


class TestRenderSourceBackwardCompat:
    def test_round_gaussian_matches_erf_path(self) -> None:
        """render_source with a round Gaussian must equal the exact erf form."""
        a = np.zeros((40, 40))
        render_gaussian(a, 20.3, 19.7, 1000.0, 3.0)
        b = np.zeros((40, 40))
        render_source(b, 20.3, 19.7, 1000.0, fwhm_px=3.0)
        assert np.allclose(a, b)


class TestMoffatProfile:
    def test_flux_conserved(self) -> None:
        f = np.zeros((81, 81))
        render_source(f, 40.0, 40.0, 1e6, fwhm_px=4.0, profile="moffat",
                      moffat_beta=3.5)
        assert f.sum() == pytest.approx(1e6, rel=2e-3)

    def test_heavier_wings_than_gaussian(self) -> None:
        """Moffat puts more flux in the wings than a same-FWHM Gaussian."""
        g = np.zeros((81, 81))
        render_source(g, 40.0, 40.0, 1e6, fwhm_px=4.0, profile="gaussian")
        m = np.zeros((81, 81))
        render_source(m, 40.0, 40.0, 1e6, fwhm_px=4.0, profile="moffat",
                      moffat_beta=2.5)
        r = np.hypot(*(np.mgrid[0:81, 0:81] - 40))
        wing = r > 8.0
        assert m[wing].sum() / m.sum() > 10.0 * (g[wing].sum() / g.sum() + 1e-9)

    def test_halfmax_fwhm_matches_input(self) -> None:
        """The half-maximum FWHM equals the requested value (the profile's
        defining width), even though heavy wings inflate the second moment."""
        f = np.zeros((121, 121))
        render_source(f, 60.0, 60.0, 1e6, fwhm_px=6.0, profile="moffat",
                      moffat_beta=3.0)
        assert _fwhm_halfmax_row(f, 60, 60) == pytest.approx(6.0, abs=1.0)


class TestAnisotropy:
    def test_ellipticity_radial_interpolation(self) -> None:
        p = PSFModel(ellipticity_center=0.0, ellipticity_edge=0.4, radial_power=1.0)
        shape = (300, 400)
        assert p.ellipticity_at(199.5, 149.5, shape) == pytest.approx(0.0, abs=1e-6)
        assert p.ellipticity_at(0, 0, shape) == pytest.approx(0.4, abs=1e-6)

    def test_orientation_is_radial(self) -> None:
        """Major axis points away from the optical-axis centre (coma-like)."""
        p = PSFModel()
        shape = (300, 400)  # centre ≈ (199.5, 149.5)
        # A point to the upper-right of centre → angle in the first quadrant.
        theta = p.orientation_at(300, 250, shape)
        assert 0.0 < theta < np.pi / 2

    def test_render_elongates_along_theta(self) -> None:
        """e>0 with θ=0 stretches along x; θ=90° stretches along y."""
        fx = np.zeros((81, 81))
        render_source(fx, 40.0, 40.0, 1e6, fwhm_px=4.0, ellipticity=0.5, theta=0.0)
        _, _, sxx, syy, _ = _moments(fx)
        assert sxx > syy

        fy = np.zeros((81, 81))
        render_source(fy, 40.0, 40.0, 1e6, fwhm_px=4.0, ellipticity=0.5,
                      theta=np.pi / 2)
        _, _, sxx2, syy2, _ = _moments(fy)
        assert syy2 > sxx2

    def test_axis_ratio_recovered(self) -> None:
        """Recovered minor/major second-moment ratio ≈ (1 − e)."""
        f = np.zeros((101, 101))
        render_source(f, 50.0, 50.0, 1e6, fwhm_px=4.0, ellipticity=0.5, theta=0.0)
        _, _, sxx, syy, _ = _moments(f)
        assert np.sqrt(syy / sxx) == pytest.approx(0.5, abs=0.05)

    def test_elliptical_centroid_unbiased(self) -> None:
        """Even an elliptical Moffat keeps the centroid at the injected point."""
        f = np.zeros((101, 101))
        render_source(f, 50.37, 49.62, 1e6, fwhm_px=4.0, profile="moffat",
                      ellipticity=0.4, theta=0.7)
        mx, my, *_ = _moments(f)
        assert mx == pytest.approx(50.37, abs=0.02)
        assert my == pytest.approx(49.62, abs=0.02)


class TestAnisotropyInFrame:
    def test_corner_star_elongated_radially(self) -> None:
        """With coma (ellipticity_edge>0), the corner star's major axis points
        along the radial direction toward the frame centre."""
        det = build_detector_model(
            SENSOR_SMALL, seed=3, prnu_pct=0.0, bias_fpn_rms_e=0.0,
            bias_offset_e=0.0, hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
            vignetting_corner_factor=1.0,
        )
        psf = PSFModel(fwhm_center_px=2.0, fwhm_edge_px=3.0,
                       ellipticity_center=0.0, ellipticity_edge=0.6)
        # Corner star at (10, 10); frame centre ≈ (200, 150) → radial axis is the
        # (10,10)->centre diagonal, i.e. sxy > 0 (elongated along +x,+y line).
        frame = generate_frame(
            SENSOR_SMALL, OPTICS_DEFAULT,
            stars=[StarSpec(magnitude=6.0, x=10, y=10)],
            detector=det, psf=psf, rng=np.random.default_rng(0),
        )
        bg = float(np.median(frame.data_float))
        stamp = frame.data_float[0:21, 0:21] - bg
        stamp = np.clip(stamp, 0.0, None)
        _, _, sxx, syy, sxy = _moments(stamp)
        # A diagonal (radial) elongation has a strong positive xy covariance.
        assert sxy > 0.3 * np.sqrt(sxx * syy)
