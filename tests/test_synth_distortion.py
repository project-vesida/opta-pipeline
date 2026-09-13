"""Tests for optical (radial lens) distortion injection.

Two layers:

1. **Model** — Brown–Conrady radial distortion: identity at zero, barrel vs
   pincushion sign, fixed centre, radial symmetry, forward∘inverse round-trip,
   and the corner-displacement constructor.

2. **Impact / scientific rigor** — the reason this feature exists: a *linear*
   (TAN) WCS cannot absorb radial distortion.  We inject a known barrel,
   fit a linear WCS to (true RA/Dec ↔ distorted pixel), and show the residual
   (a) grows with field radius — the signature of uncorrected radial distortion,
   (b) scales with distortion magnitude, and (c) at a realistic ≳3 % barrel the
   field-edge residual exceeds the 10″ OpTA.NOD.ACC budget.  This is what makes
   a distortion-aware solver (SIP/TPV) mandatory rather than optional.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL, get_test_node

from opta_pipeline.astrometry import (
    StarMatch,
    WCSSolution,
    fit_wcs,
    fit_wcs_sip,
    pixels_to_radec,
)
from opta_pipeline.synth import StarSpec, build_detector_model, generate_frame
from opta_pipeline.synth.distortion import DistortionModel

SENSOR = SENSOR_SMALL
OPTICS = OPTICS_DEFAULT
_SHAPE = (SENSOR.resolution_v, SENSOR.resolution_h)
_CX, _CY = (SENSOR.resolution_h - 1) / 2.0, (SENSOR.resolution_v - 1) / 2.0
_R_HALF = math.hypot(SENSOR.resolution_h / 2.0, SENSOR.resolution_v / 2.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestDistortionModel:
    def test_zero_is_identity(self) -> None:
        d = DistortionModel()
        assert d.apply(380.0, 20.0, _SHAPE) == (380.0, 20.0)

    def test_centre_is_fixed(self) -> None:
        d = DistortionModel(k1=-0.2)
        assert d.apply(_CX, _CY, _SHAPE) == pytest.approx((_CX, _CY))

    def test_barrel_pulls_inward(self) -> None:
        d = DistortionModel(k1=-0.15)
        xd, yd = d.apply(380.0, 20.0, _SHAPE)
        r_ideal = math.hypot(380.0 - _CX, 20.0 - _CY)
        r_dist = math.hypot(xd - _CX, yd - _CY)
        assert r_dist < r_ideal  # barrel: image radius shrinks

    def test_pincushion_pushes_outward(self) -> None:
        d = DistortionModel(k1=0.15)
        xd, yd = d.apply(380.0, 20.0, _SHAPE)
        r_dist = math.hypot(xd - _CX, yd - _CY)
        assert r_dist > math.hypot(380.0 - _CX, 20.0 - _CY)

    def test_radial_symmetry(self) -> None:
        """Two points at equal field radius are displaced by equal magnitude."""
        d = DistortionModel(k1=-0.1)
        p1 = d.apply(_CX + 100, _CY, _SHAPE)
        p2 = d.apply(_CX, _CY + 100, _SHAPE)
        disp1 = math.hypot(p1[0] - (_CX + 100), p1[1] - _CY)
        disp2 = math.hypot(p2[0] - _CX, p2[1] - (_CY + 100))
        assert disp1 == pytest.approx(disp2, rel=1e-9)

    def test_inverse_round_trip(self) -> None:
        d = DistortionModel(k1=-0.12, k2=0.03)
        for x, y in [(350.0, 40.0), (10.0, 290.0), (200.0, 150.0)]:
            xd, yd = d.apply(x, y, _SHAPE)
            xi, yi = d.invert(xd, yd, _SHAPE)
            assert (xi, yi) == pytest.approx((x, y), abs=1e-3)

    def test_from_corner_displacement(self) -> None:
        d = DistortionModel.from_corner_displacement_px(-6.0, _SHAPE)
        # A corner pixel moves ~6 px radially inward.
        corner = (SENSOR.resolution_h - 1.0, SENSOR.resolution_v - 1.0)
        xd, yd = d.apply(*corner, _SHAPE)
        disp = math.hypot(xd - corner[0], yd - corner[1])
        assert disp == pytest.approx(6.0, rel=0.05)


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------


class TestDistortionInFrame:
    def test_none_is_byte_identical(self) -> None:
        a = generate_frame(SENSOR, OPTICS, stars=[StarSpec(9.0, 300, 80)],
                           rng=np.random.default_rng(0))
        b = generate_frame(SENSOR, OPTICS, stars=[StarSpec(9.0, 300, 80)],
                           distortion=None, rng=np.random.default_rng(0))
        assert np.array_equal(a.data_float, b.data_float)

    def test_corner_star_displaced_toward_centre(self) -> None:
        dm = DistortionModel.from_corner_displacement_px(-6.0, _SHAPE)
        f = generate_frame(SENSOR, OPTICS, stars=[StarSpec(7.0, 380, 20)],
                           distortion=dm, rng=np.random.default_rng(1))
        gt = f.stars[0]
        assert math.hypot(gt.x - _CX, gt.y - _CY) < math.hypot(380 - _CX, 20 - _CY)

    def test_flux_conserved_under_distortion(self) -> None:
        """Distortion only moves a (mid-field) source; its flux is preserved."""
        dm = DistortionModel(k1=-0.1)
        plain = generate_frame(SENSOR, OPTICS, stars=[StarSpec(8.0, 260, 110)],
                              rng=np.random.default_rng(2))
        dist = generate_frame(SENSOR, OPTICS, stars=[StarSpec(8.0, 260, 110)],
                             distortion=dm, rng=np.random.default_rng(2))
        bg = plain.sky_e_per_pixel * plain.data_float.size
        assert (dist.data_float.sum() - bg) == pytest.approx(
            plain.data_float.sum() - bg, rel=1e-3
        )

    def test_realistic_path_distorts(self) -> None:
        det = build_detector_model(SENSOR, seed=1, bias_offset_e=0.0,
                                   hot_pixel_fraction=0.0)
        dm = DistortionModel.from_corner_displacement_px(-8.0, _SHAPE)
        f = generate_frame(SENSOR, OPTICS, stars=[StarSpec(7.0, 380, 20)],
                           detector=det, distortion=dm, rng=np.random.default_rng(0))
        assert (f.stars[0].x, f.stars[0].y) != (380.0, 20.0)


# ---------------------------------------------------------------------------
# Scientific rigor: a linear WCS cannot absorb radial distortion
# ---------------------------------------------------------------------------


def _linear_wcs_residuals(barrel_pct: float):
    """Fit a linear WCS to a distorted star grid; return (rms, max, inner, outer)
    residuals in arcsec, where inner/outer are mean residuals at field radius
    < 0.5 and > 0.8."""
    ps_as = get_test_node().pixel_scale_arcsec
    ps = ps_as / 3600.0
    true = WCSSolution(
        crpix1=SENSOR.resolution_h / 2.0, crpix2=SENSOR.resolution_v / 2.0,
        crval1=200.0, crval2=30.0, cd1_1=ps, cd1_2=0.0, cd2_1=0.0, cd2_2=ps,
        rms_arcsec=0.0, n_stars=0,
    )
    dm = DistortionModel.from_corner_displacement_px(
        -barrel_pct / 100.0 * _R_HALF, _SHAPE
    )
    matches, r_field = [], []
    for gx in np.linspace(15, SENSOR.resolution_h - 15, 8):
        for gy in np.linspace(15, SENSOR.resolution_v - 15, 6):
            ra, dec = pixels_to_radec(true, gx, gy)        # true sky
            xd, yd = dm.apply(gx, gy, _SHAPE)              # distorted pixel
            matches.append(StarMatch(x_px=xd, y_px=yd, ra_deg=ra, dec_deg=dec))
            r_field.append(math.hypot(gx - _CX, gy - _CY) / _R_HALF)
    w = fit_wcs(matches, frame_shape=_SHAPE, max_residual_arcsec=1e9)
    res = []
    for m in matches:
        ra_f, dec_f = pixels_to_radec(w, m.x_px, m.y_px)
        cosd = math.cos(math.radians(dec_f))
        res.append(math.hypot((m.ra_deg - ra_f) * cosd * 3600.0,
                              (m.dec_deg - dec_f) * 3600.0))
    res = np.array(res)
    r_field = np.array(r_field)
    return (float(res.mean()), float(res.max()),
            float(res[r_field < 0.5].mean()), float(res[r_field > 0.8].mean()))


class TestLinearWCSCannotAbsorbDistortion:
    def test_residual_grows_with_field_radius(self) -> None:
        """The uncorrected-distortion signature: edge residual ≫ centre."""
        _, _, inner, outer = _linear_wcs_residuals(2.0)
        assert outer > 1.3 * inner

    def test_residual_scales_with_magnitude(self) -> None:
        rms_1 = _linear_wcs_residuals(1.0)[0]
        rms_3 = _linear_wcs_residuals(3.0)[0]
        assert rms_3 > 2.0 * rms_1

    def test_realistic_barrel_exceeds_nod_acc_budget(self) -> None:
        """A realistic ≳3 % barrel blows the 10″ OpTA.NOD.ACC budget at the edge,
        so a distortion-aware WCS is mandatory; a mild 1 % barrel stays under."""
        assert _linear_wcs_residuals(3.0)[1] > 10.0   # field-edge max
        assert _linear_wcs_residuals(1.0)[1] < 10.0


# ---------------------------------------------------------------------------
# Distortion-aware solve (SIP) closes the field-radius regression
# ---------------------------------------------------------------------------


def _sip_wcs_residuals(barrel_pct: float):
    """As :func:`_linear_wcs_residuals` but with the distortion-aware solver.

    Builds the same distorted star grid, fits ``fit_wcs_sip`` (TAN + radial
    SIP), and returns ``(rms, max, inner, outer)`` residuals in arcsec.  The
    solver sees only (true RA/Dec ↔ distorted pixel); it does not know the
    injected ``k1``.
    """
    ps_as = get_test_node().pixel_scale_arcsec
    ps = ps_as / 3600.0
    true = WCSSolution(
        crpix1=SENSOR.resolution_h / 2.0, crpix2=SENSOR.resolution_v / 2.0,
        crval1=200.0, crval2=30.0, cd1_1=ps, cd1_2=0.0, cd2_1=0.0, cd2_2=ps,
        rms_arcsec=0.0, n_stars=0,
    )
    dm = DistortionModel.from_corner_displacement_px(
        -barrel_pct / 100.0 * _R_HALF, _SHAPE
    )
    matches, r_field = [], []
    for gx in np.linspace(15, SENSOR.resolution_h - 15, 8):
        for gy in np.linspace(15, SENSOR.resolution_v - 15, 6):
            ra, dec = pixels_to_radec(true, gx, gy)
            xd, yd = dm.apply(gx, gy, _SHAPE)
            matches.append(StarMatch(x_px=xd, y_px=yd, ra_deg=ra, dec_deg=dec))
            r_field.append(math.hypot(gx - _CX, gy - _CY) / _R_HALF)
    w = fit_wcs_sip(matches, frame_shape=_SHAPE, max_residual_arcsec=4.0)
    res = []
    for m in matches:
        ra_f, dec_f = pixels_to_radec(w, m.x_px, m.y_px)
        cosd = math.cos(math.radians(dec_f))
        res.append(math.hypot((m.ra_deg - ra_f) * cosd * 3600.0,
                              (m.dec_deg - dec_f) * 3600.0))
    res = np.array(res)
    r_field = np.array(r_field)
    return (float(res.mean()), float(res.max()),
            float(res[r_field < 0.5].mean()), float(res[r_field > 0.8].mean()))


class TestSIPSolveAbsorbsDistortion:
    def test_edge_residual_within_budget_where_linear_fails(self) -> None:
        """At a 3 % barrel the linear solve blows the 10″ edge budget; the
        distortion-aware solve keeps the field-edge residual well under it."""
        assert _linear_wcs_residuals(3.0)[1] > 10.0          # linear fails
        _, sip_max, _, sip_edge = _sip_wcs_residuals(3.0)
        assert sip_edge < 3.0                                # T-11 ≤ 3″
        assert sip_max < 10.0                                # OpTA.NOD.ACC

    def test_sip_beats_linear_by_large_margin(self) -> None:
        """The SIP edge residual is a small fraction of the linear one."""
        lin_edge = _linear_wcs_residuals(3.0)[3]
        sip_edge = _sip_wcs_residuals(3.0)[3]
        assert sip_edge < 0.1 * lin_edge

    def test_residual_no_longer_grows_with_radius(self) -> None:
        """With distortion absorbed, the edge-vs-centre signature is gone."""
        _, _, inner, outer = _sip_wcs_residuals(3.0)
        # Edge residual is no longer dramatically larger than the centre.
        assert outer < 5.0 * max(inner, 0.05)

    def test_severe_barrel_still_within_budget(self) -> None:
        """Even a severe 5 % barrel is recovered to within OpTA.NOD.ACC."""
        _, sip_max, _, sip_edge = _sip_wcs_residuals(5.0)
        assert sip_max < 10.0
        assert sip_edge < 3.0

    def test_zero_distortion_is_a_clean_linear_fit(self) -> None:
        """With no distortion the SIP solver recovers a sub-arcsec linear fit."""
        rms, mx, _, _ = _sip_wcs_residuals(0.0)
        assert mx < 0.5
