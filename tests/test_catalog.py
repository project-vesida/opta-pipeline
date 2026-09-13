"""Tests for opta_pipeline.synth.catalog (WP-A4).

Acceptance criteria:
  - star_field_at returns only in-FOV stars (pixel bounds correct)
  - Magnitude limit filtering works
  - Catalog is deterministic: same WCS → same stars
  - Different pointings give different stars (sensitivity check)
  - RA 0°/360° wrap is handled without duplicates or gaps
  - Large FOV yields ≥ 500 stars (density check for 500-star acceptance test)
  - Plate-solve RMS on a 500-star field from catalog < 0.5 arcsec (WP-A4 acceptance)
  - StarSpec.from_catalog produces correct pixel positions
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opta_pipeline.astrometry import WCSSolution, fit_wcs
from opta_pipeline.synth import StarSpec
from opta_pipeline.synth.catalog import CatalogStar, star_field_at

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Full IMX585 + 85mm field: 1920×1080 @ 9.12 arcsec/px
_W, _H = 1920, 1080
_CD = 9.12 / 3600.0  # deg/px
_RA0, _DEC0 = 135.0, 45.0

# Compact test field: 400×300 @ 9.12 arcsec/px
_W_SM, _H_SM = 400, 300


def _wcs(
    ra0: float = _RA0, dec0: float = _DEC0, cd: float = _CD, w: int = _W, h: int = _H
) -> WCSSolution:
    """Axis-aligned TAN WCS centred on (ra0, dec0)."""
    return WCSSolution(
        crpix1=w / 2.0,
        crpix2=h / 2.0,
        crval1=ra0,
        crval2=dec0,
        cd1_1=cd,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=cd,
        rms_arcsec=0.0,
        n_stars=0,
    )


# ---------------------------------------------------------------------------
# 1. Basic FOV filtering
# ---------------------------------------------------------------------------


class TestStarFieldAt:
    """star_field_at returns correct in-FOV stars."""

    def test_all_returned_stars_in_fov(self) -> None:
        """Every returned star must lie within [0, width) × [0, height)."""
        matches = star_field_at(_wcs(), _W, _H)
        for m in matches:
            assert 0.0 <= m.x_px < _W, f"x={m.x_px} outside [0, {_W})"
            assert 0.0 <= m.y_px < _H, f"y={m.y_px} outside [0, {_H})"

    def test_returns_nonzero_stars(self) -> None:
        matches = star_field_at(_wcs(), _W, _H)
        assert len(matches) > 0

    def test_mag_limit_filters_faint_stars(self) -> None:
        """Stars fainter than mag_limit must not appear."""
        limit = 10.0
        matches = star_field_at(_wcs(), _W, _H, mag_limit=limit)
        # We can only check the RA/Dec values because StarMatch doesn't carry magnitude.
        # Verify that a tighter limit yields fewer (or equal) stars than a looser one.
        matches_loose = star_field_at(_wcs(), _W, _H, mag_limit=14.0)
        assert len(matches) <= len(matches_loose)

    def test_zero_mag_limit_returns_no_stars(self) -> None:
        """mag_limit below catalog minimum → empty list."""
        matches = star_field_at(_wcs(), _W, _H, mag_limit=0.0)
        assert matches == []

    def test_sky_coords_in_plausible_range(self) -> None:
        """Returned RA/Dec must lie within a few degrees of the pointing centre."""
        matches = star_field_at(_wcs(), _W, _H)
        for m in matches:
            assert 0.0 <= m.ra_deg < 360.0
            assert -90.0 <= m.dec_deg <= 90.0
            # Stars must be near the field centre (within one FOV width)
            cos_dec = math.cos(math.radians(_DEC0))
            dra = abs((m.ra_deg - _RA0 + 180.0) % 360.0 - 180.0) * cos_dec
            ddec = abs(m.dec_deg - _DEC0)
            assert dra < 5.0 and ddec < 5.0, (
                f"Star at ({m.ra_deg:.2f}, {m.dec_deg:.2f}) is far from field centre "
                f"({_RA0}, {_DEC0})"
            )


# ---------------------------------------------------------------------------
# 2. Determinism and uniqueness
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Catalog is reproducible and pointing-sensitive."""

    def test_same_wcs_returns_same_stars(self) -> None:
        """Two identical calls must produce the same list in the same order."""
        w = _wcs()
        a = star_field_at(w, _W, _H)
        b = star_field_at(w, _W, _H)
        assert len(a) == len(b)
        for m_a, m_b in zip(a, b):
            assert m_a.ra_deg == m_b.ra_deg
            assert m_a.dec_deg == m_b.dec_deg

    def test_different_pointing_different_stars(self) -> None:
        """Shifting the pointing by one FOV width must change the returned stars."""
        a = star_field_at(_wcs(ra0=10.0, dec0=20.0), _W, _H)
        b = star_field_at(_wcs(ra0=10.0, dec0=50.0), _W, _H)  # 30° away
        ra_set_a = {m.ra_deg for m in a}
        ra_set_b = {m.ra_deg for m in b}
        # Virtually no overlap expected between fields 30° apart
        overlap = ra_set_a & ra_set_b
        assert len(overlap) < min(len(a), len(b)) * 0.1


# ---------------------------------------------------------------------------
# 3. RA 0°/360° wrap
# ---------------------------------------------------------------------------


class TestRAWrap:
    """Stars near RA=0/360 are handled without gaps or duplicates."""

    def test_ra_near_zero_returns_stars(self) -> None:
        """Pointing near RA=1° must still return stars (crosses the 0/360 boundary)."""
        matches = star_field_at(_wcs(ra0=1.0, dec0=0.0), _W_SM, _H_SM)
        assert len(matches) >= 5, f"Too few stars ({len(matches)}) near RA=0"

    def test_ra_near_360_returns_stars(self) -> None:
        matches = star_field_at(_wcs(ra0=359.0, dec0=0.0), _W_SM, _H_SM)
        assert len(matches) >= 5

    def test_no_duplicate_ra_dec(self) -> None:
        """No two returned stars should have identical (RA, Dec)."""
        matches = star_field_at(_wcs(ra0=1.0, dec0=0.0), _W, _H)
        coords = [(m.ra_deg, m.dec_deg) for m in matches]
        assert len(coords) == len(set(coords)), "Duplicate stars returned"


# ---------------------------------------------------------------------------
# 4. Density and 500-star acceptance criterion
# ---------------------------------------------------------------------------


class TestCatalogDensity:
    """Catalog density satisfies the 500-star acceptance criterion."""

    def test_full_sensor_yields_500_stars(self) -> None:
        """Full IMX585 FOV (~13 sq.deg) must yield ≥ 500 stars at mag_limit 14."""
        matches = star_field_at(_wcs(), _W, _H, mag_limit=14.0)
        assert len(matches) >= 500, (
            f"Only {len(matches)} stars in full FOV; need ≥ 500 for acceptance test"
        )

    def test_pixel_positions_span_full_sensor(self) -> None:
        """Stars should be spread across the sensor, not clumped."""
        matches = star_field_at(_wcs(), _W, _H)
        xs = [m.x_px for m in matches]
        ys = [m.y_px for m in matches]
        # Mean should be near the centre ± 20%
        assert abs(np.mean(xs) - _W / 2.0) < _W * 0.2
        assert abs(np.mean(ys) - _H / 2.0) < _H * 0.2


# ---------------------------------------------------------------------------
# 5. Plate-solve acceptance test (WP-A4 primary acceptance criterion)
# ---------------------------------------------------------------------------


class TestPlateSolve500Stars:
    """Plate-solve RMS on a 500-star catalog-derived field must be < 0.5 arcsec."""

    def test_rms_below_half_arcsec(self) -> None:
        """WP-A4 acceptance: fit_wcs on a 500-star catalog field gives RMS < 0.5 arcsec.

        Catalog stars have exact sky→pixel positions (via radec_to_pixels), so
        the residual measures the internal consistency of the star_field_at →
        fit_wcs round-trip.
        """
        w = _wcs()
        matches = star_field_at(w, _W, _H, mag_limit=14.0)
        assert len(matches) >= 500, (
            f"Only {len(matches)} stars — need ≥ 500 for this test"
        )
        # Fit WCS on the catalog matches
        solved = fit_wcs(matches, frame_shape=(_H, _W))
        assert solved.rms_arcsec < 0.5, (
            f"Plate-solve RMS {solved.rms_arcsec:.3f} arcsec exceeds 0.5 arcsec "
            "acceptance criterion (WP-A4)"
        )

    def test_solved_crval_matches_pointing(self) -> None:
        """fit_wcs on catalog stars should recover the true pointing centre."""
        w = _wcs()
        matches = star_field_at(w, _W, _H)
        solved = fit_wcs(matches, frame_shape=(_H, _W))
        # Allow ≤ 0.01° (36 arcsec) tolerance
        assert abs(solved.crval1 - _RA0) < 0.01
        assert abs(solved.crval2 - _DEC0) < 0.01

    def test_solved_pixel_scale_matches_input(self) -> None:
        """Recovered pixel scale must match the input WCS within 0.1 arcsec/px."""
        w = _wcs()
        matches = star_field_at(w, _W, _H)
        solved = fit_wcs(matches, frame_shape=(_H, _W))
        det = abs(solved.cd1_1 * solved.cd2_2 - solved.cd1_2 * solved.cd2_1)
        recovered_arcsec_px = math.sqrt(det) * 3600.0
        assert abs(recovered_arcsec_px - 9.12) < 0.1


# ---------------------------------------------------------------------------
# 6. StarSpec.from_catalog factory
# ---------------------------------------------------------------------------


class TestStarSpecFromCatalog:
    """StarSpec.from_catalog creates correct StarSpec objects from CatalogStar list."""

    def test_from_catalog_returns_correct_count(self) -> None:
        w = _wcs()
        catalog_stars = [
            CatalogStar(ra_deg=_RA0, dec_deg=_DEC0, magnitude=10.0),
            CatalogStar(ra_deg=_RA0 + 0.1, dec_deg=_DEC0 + 0.1, magnitude=11.0),
        ]
        specs = StarSpec.from_catalog(w, catalog_stars)
        assert len(specs) == 2

    def test_from_catalog_magnitudes_preserved(self) -> None:
        w = _wcs()
        catalog_stars = [CatalogStar(ra_deg=_RA0, dec_deg=_DEC0, magnitude=9.5)]
        specs = StarSpec.from_catalog(w, catalog_stars)
        assert specs[0].magnitude == pytest.approx(9.5)

    def test_from_catalog_pixel_position_at_crpix(self) -> None:
        """A star at (CRVAL1, CRVAL2) must land at (CRPIX1, CRPIX2)."""
        w = _wcs()
        catalog_stars = [CatalogStar(ra_deg=_RA0, dec_deg=_DEC0, magnitude=10.0)]
        specs = StarSpec.from_catalog(w, catalog_stars)
        assert specs[0].x == pytest.approx(w.crpix1, abs=1e-9)
        assert specs[0].y == pytest.approx(w.crpix2, abs=1e-9)
