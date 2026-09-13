"""Tests for blind catalog cross-matching (opta_pipeline.match).

These exercise the keystone detect→match step in isolation: given detected
star centroids built from a *true* WCS (with unknown roll / scale error /
pointing offset) and an *approximate* pointing hint, ``match_catalog`` must
recover correct StarMatch pairs without any truth-derived correspondence, and
admit no false matches that survive a subsequent plate fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from opta_pipeline.astrometry import (
    WCSSolution,
    fit_wcs,
    pixels_to_radec,
    radec_to_pixels,
)
from opta_pipeline.detect import Detection
from opta_pipeline.match import PointingHint, match_catalog
from opta_pipeline.synth.catalog import cone_search

_W, _H = 640, 480
_RA0, _DEC0 = 135.0, 45.0
_SCALE = 31.0  # arcsec/px (OpTA wide field)


def _true_wcs(roll_deg: float) -> WCSSolution:
    """A true TAN WCS centred on (_RA0,_DEC0) with a camera roll."""
    ps = _SCALE / 3600.0
    r = math.radians(roll_deg)
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=_RA0,
        crval2=_DEC0,
        cd1_1=ps * math.cos(r),
        cd1_2=-ps * math.sin(r),
        cd2_1=ps * math.sin(r),
        cd2_2=ps * math.cos(r),
        rms_arcsec=0.0,
        n_stars=0,
    )


def _build_frame(
    roll_deg: float,
    mag_limit: float = 11.0,
    centroid_noise_px: float = 0.2,
    n_false: int = 10,
    seed: int = 0,
) -> tuple[list[Detection], list[tuple[float, float]], WCSSolution]:
    """Project the catalog through a true WCS into noisy star detections.

    Returns (detections, truth_radec_for_real_stars, true_wcs).
    """
    true = _true_wcs(roll_deg)
    cat = cone_search(_RA0, _DEC0, radius_deg=2.0, mag_limit=mag_limit)
    rng = np.random.default_rng(seed)
    dets: list[Detection] = []
    truth: list[tuple[float, float]] = []
    for s in cat:
        x, y = radec_to_pixels(true, s.ra_deg, s.dec_deg)
        if 0 <= x < _W and 0 <= y < _H:
            dets.append(
                Detection(
                    x=x + rng.normal(0.0, centroid_noise_px),
                    y=y + rng.normal(0.0, centroid_noise_px),
                    snr=100.0 - s.magnitude * 5.0,
                    elongation=1.0,
                    angle_deg=0.0,
                    n_pixels=9,
                    flux_e=1000.0,
                    is_streak=False,
                    fwhm_px=2.0,
                )
            )
            truth.append((s.ra_deg, s.dec_deg))
    for _ in range(n_false):
        dets.append(
            Detection(
                x=float(rng.uniform(0, _W)),
                y=float(rng.uniform(0, _H)),
                snr=20.0,
                elongation=1.0,
                angle_deg=0.0,
                n_pixels=6,
                flux_e=500.0,
                is_streak=False,
                fwhm_px=2.0,
            )
        )
    return dets, truth, true


def _is_correct(match, true: WCSSolution) -> bool:
    """A match is correct if its pixel position maps to its catalog RA/Dec."""
    xt, yt = radec_to_pixels(true, match.ra_deg, match.dec_deg)
    return math.hypot(xt - match.x_px, yt - match.y_px) < 2.0


# ---------------------------------------------------------------------------
# cone_search
# ---------------------------------------------------------------------------


class TestConeSearch:
    def test_returns_stars_within_radius_only(self) -> None:
        stars = cone_search(_RA0, _DEC0, radius_deg=1.0, mag_limit=14.0)
        assert stars, "cone search returned no stars"
        for s in stars:
            r1, d1 = math.radians(_RA0), math.radians(_DEC0)
            r2, d2 = math.radians(s.ra_deg), math.radians(s.dec_deg)
            sep = math.degrees(
                math.acos(
                    min(
                        1.0,
                        math.sin(d1) * math.sin(d2)
                        + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2),
                    )
                )
            )
            assert sep <= 1.0 + 1e-6

    def test_brightest_first_and_mag_limited(self) -> None:
        stars = cone_search(_RA0, _DEC0, radius_deg=1.0, mag_limit=10.0)
        mags = [s.magnitude for s in stars]
        assert mags == sorted(mags)
        assert max(mags) <= 10.0

    def test_deterministic(self) -> None:
        a = cone_search(_RA0, _DEC0, 1.0, 12.0)
        b = cone_search(_RA0, _DEC0, 1.0, 12.0)
        assert [(s.ra_deg, s.dec_deg) for s in a] == [
            (s.ra_deg, s.dec_deg) for s in b
        ]


# ---------------------------------------------------------------------------
# match_catalog
# ---------------------------------------------------------------------------


class TestMatchCatalog:
    def test_recovers_matches_with_unknown_roll(self) -> None:
        """The keystone: ≥3 correct matches with no truth correspondence."""
        dets, _, true = _build_frame(roll_deg=12.0)
        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        catalog = cone_search(hint.ra_deg, hint.dec_deg, 2.0, mag_limit=11.5)
        matches = match_catalog(dets, catalog, hint, frame_shape=(_H, _W))
        assert len(matches) >= 3
        correct = sum(_is_correct(m, true) for m in matches)
        assert correct >= 3
        # False matches must be a small minority (the WCS fit rejects them).
        assert correct >= 0.9 * len(matches)

    @pytest.mark.parametrize("roll", [0.0, 30.0, 90.0, 170.0, -45.0])
    def test_robust_across_camera_roll(self, roll: float) -> None:
        dets, _, true = _build_frame(roll_deg=roll, seed=abs(int(roll)) + 7)
        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        catalog = cone_search(hint.ra_deg, hint.dec_deg, 2.0, mag_limit=11.5)
        matches = match_catalog(dets, catalog, hint, frame_shape=(_H, _W))
        assert sum(_is_correct(m, true) for m in matches) >= 5

    def test_tolerates_pointing_offset_and_scale_error(self) -> None:
        dets, _, true = _build_frame(roll_deg=8.0)
        # Hint offset by ~0.15° and scale wrong by 2%.
        hint = PointingHint(_RA0 + 0.15, _DEC0 - 0.1, _SCALE * 1.02, radius_deg=2.0)
        catalog = cone_search(hint.ra_deg, hint.dec_deg, 2.0, mag_limit=11.5)
        matches = match_catalog(dets, catalog, hint, frame_shape=(_H, _W))
        assert sum(_is_correct(m, true) for m in matches) >= 5

    def test_no_false_matches_survive_the_fit(self) -> None:
        """After fit_wcs outlier rejection, the recovered WCS is accurate."""
        dets, _, true = _build_frame(roll_deg=15.0, n_false=20)
        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        catalog = cone_search(hint.ra_deg, hint.dec_deg, 2.0, mag_limit=11.5)
        matches = match_catalog(dets, catalog, hint, frame_shape=(_H, _W))
        wcs = fit_wcs(matches, frame_shape=(_H, _W), max_residual_arcsec=10.0)
        ra, dec = pixels_to_radec(wcs, _W / 2.0, _H / 2.0)
        err = math.hypot(
            (ra - _RA0) * math.cos(math.radians(_DEC0)) * 3600.0,
            (dec - _DEC0) * 3600.0,
        )
        assert err < 10.0, f"blind solve center error {err:.2f}\" exceeds 10\""

    def test_ransac_vote_counts_catalog_stars_not_detections(self) -> None:
        """The vote must weigh *distinct* catalog stars, not raw inliers.

        Two transforms compete here.  The correct one puts five detections on
        five different catalog stars.  The decoy puts nine detections — three
        tight blobs of three — on just three catalog stars.  Counting every
        detection within tol of *some* star (``np.isfinite(d).sum()``) scores
        the decoy 9 vs 5 and it wins the RANSAC; the harvest then dedups it
        back to 3 claims, below ``min_matches``, and the frame fails to solve.
        Counting unique catalog indices scores 5 vs 3 and the correct
        transform wins.  This is the dense-region pile-up bias in its
        smallest reproducible form.
        """

        @dataclass(frozen=True)
        class _Star:
            ra_deg: float
            dec_deg: float
            magnitude: float

        true = _true_wcs(0.0)
        # Five spread stars (the solvable configuration) + a compact triangle.
        spread = [(60.0, 60.0), (300.0, 50.0), (560.0, 110.0),
                  (120.0, 300.0), (420.0, 400.0)]
        triangle = [(180.0, 180.0), (300.0, 150.0), (250.0, 260.0)]
        catalog = [
            _Star(*pixels_to_radec(true, x, y), 8.0 + 0.05 * i)
            for i, (x, y) in enumerate(spread + triangle)
        ]

        def _det(x: float, y: float, snr: float) -> Detection:
            return Detection(
                x=float(x), y=float(y), snr=snr, elongation=1.0,
                angle_deg=0.0, n_pixels=9, flux_e=1000.0,
                is_streak=False, fwhm_px=2.0,
            )

        dets = [_det(x, y, 100.0 - i) for i, (x, y) in enumerate(spread)]
        # The decoy: the compact triangle repeated three times over, offset so
        # it is congruent to the catalog triangle under a *different* affine.
        dx, dy = 150.0, 120.0
        blob = [(-0.6, 0.4), (0.5, -0.5), (0.0, 0.6)]
        for j, (x, y) in enumerate(triangle):
            for k, (ox, oy) in enumerate(blob):
                dets.append(_det(x + dx + ox, y + dy + oy, 50.0 - j - 0.1 * k))

        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        matches = match_catalog(
            dets, catalog, hint, frame_shape=(_H, _W), min_matches=5
        )
        assert len(matches) == 5
        assert all(_is_correct(m, true) for m in matches)

    def test_returns_empty_when_too_few_detections(self) -> None:
        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        catalog = cone_search(_RA0, _DEC0, 2.0, mag_limit=11.0)
        assert match_catalog([], catalog, hint, frame_shape=(_H, _W)) == []

    def test_returns_empty_on_pure_noise(self) -> None:
        """Random detections with no real stars yield no consistent solution."""
        rng = np.random.default_rng(3)
        dets = [
            Detection(
                x=float(rng.uniform(0, _W)),
                y=float(rng.uniform(0, _H)),
                snr=10.0,
                elongation=1.0,
                angle_deg=0.0,
                n_pixels=6,
                flux_e=400.0,
                is_streak=False,
                fwhm_px=2.0,
            )
            for _ in range(15)
        ]
        hint = PointingHint(_RA0, _DEC0, _SCALE, radius_deg=2.0)
        catalog = cone_search(_RA0, _DEC0, 2.0, mag_limit=11.0)
        matches = match_catalog(dets, catalog, hint, frame_shape=(_H, _W))
        # A handful of accidental matches may appear, but never a full solution.
        assert len(matches) < 5
