"""Pointing-aware synthetic star catalog for pipeline testing (WP-A4).

Provides a deterministic frozen catalog of synthetic stars that mimics
Gaia DR3 source density.  Stars are generated from a tile-hash seed so
the same sky pointing always returns the same stars — no file I/O, fully
offline.  The star density (100 stars/sq.deg) and magnitude range (8–14)
are calibrated to give ≥ 500 stars in a typical full-sensor FOV
(~13 sq.deg for IMX585 + 85 mm).

For production use, implement the same ``star_field_at`` interface backed
by a live Gaia DR3 cone-search behind a disk cache.

Usage
-----
    from opta_pipeline.synth.catalog import star_field_at, CatalogStar
    from opta_pipeline.astrometry import WCSSolution

    matches = star_field_at(wcs, width=1920, height=1080, mag_limit=12.0)
    # → list[StarMatch] with exact pixel positions and catalog sky coords
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from opta_pipeline.astrometry import StarMatch, WCSSolution, radec_to_pixels

__all__ = [
    "CatalogStar",
    "star_field_at",
    "catalog_stars_in_fov",
    "cone_search",
    "STARS_PER_SQ_DEG",
]

# Stars per 1°×1° tile — calibrated so a full IMX585 FOV yields ~1 300 stars.
STARS_PER_SQ_DEG: int = 100
_MAG_MIN: float = 8.0
_MAG_MAX: float = 14.0


@dataclass(frozen=True)
class CatalogStar:
    """A single entry from the synthetic frozen catalog.

    Attributes
    ----------
    ra_deg : float
        Right Ascension in degrees (J2000, [0, 360)).
    dec_deg : float
        Declination in degrees (J2000, [−90, 90]).
    magnitude : float
        Apparent V-band magnitude.
    """

    ra_deg: float
    dec_deg: float
    magnitude: float


# ---------------------------------------------------------------------------
# Tile generator
# ---------------------------------------------------------------------------


def _tile_stars(tile_ra: int, tile_dec: int) -> list[CatalogStar]:
    """Return the stars in a 1°×1° sky tile, deterministically.

    The tile origin is ``(tile_ra, tile_dec)`` (integer-degree floor).
    Stars are drawn from a hash-seeded RNG so the same tile always
    produces the same set regardless of query order.
    """
    key = f"{tile_ra % 360:03d}{tile_dec:+04d}".encode()
    seed = int(hashlib.md5(key, usedforsecurity=False).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    stars: list[CatalogStar] = []
    for _ in range(STARS_PER_SQ_DEG):
        ra = float((tile_ra + rng.uniform(0.0, 1.0)) % 360.0)
        dec = float(np.clip(tile_dec + rng.uniform(0.0, 1.0), -90.0, 90.0))
        mag = float(_MAG_MIN + rng.uniform(0.0, _MAG_MAX - _MAG_MIN))
        stars.append(CatalogStar(ra_deg=ra, dec_deg=dec, magnitude=mag))
    return stars


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def star_field_at(
    wcs: WCSSolution,
    width: int,
    height: int,
    mag_limit: float = _MAG_MAX,
) -> list[StarMatch]:
    """Return catalog stars within the sensor FOV as StarMatch objects.

    All returned stars lie within the pixel rectangle ``[0, width) × [0, height)``.
    Pixel positions are computed via ``radec_to_pixels`` (exact inverse of the
    input WCS) — no centroid noise is added.

    Parameters
    ----------
    wcs : WCSSolution
        Pointing WCS defining the field centre and pixel scale.  Typically
        produced by ``make_synthetic_pass`` or ``fit_wcs``.
    width, height : int
        Sensor width and height in pixels.
    mag_limit : float
        Faintest magnitude to include (default 14.0, matching the catalog max).

    Returns
    -------
    list[StarMatch]
        In-FOV stars with exact pixel positions and catalog RA/Dec.
    """
    # Sky half-extents of the FOV (degrees), with a 1° safety margin
    cd_det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
    pixel_scale_deg = math.sqrt(cd_det) if cd_det > 0.0 else abs(wcs.cd1_1)
    fov_half_ra_sky = (width / 2.0) * pixel_scale_deg + 1.0
    fov_half_dec = (height / 2.0) * pixel_scale_deg + 1.0

    cos_dec = max(math.cos(math.radians(wcs.crval2)), 1e-6)
    ra_span = fov_half_ra_sky / cos_dec  # un-projected RA degrees to span

    dec_lo = max(-90.0, wcs.crval2 - fov_half_dec)
    dec_hi = min(90.0, wcs.crval2 + fov_half_dec)
    dec_tile_lo = int(math.floor(dec_lo))
    dec_tile_hi = int(math.floor(dec_hi))

    # Enumerate 1°×1° RA tiles covering the span (handles RA 0/360 wrap).
    n_ra_tiles = int(math.ceil(ra_span * 2.0)) + 3
    ra_start = int(math.floor((wcs.crval1 - ra_span) % 360.0))

    seen: set[tuple[int, int]] = set()
    catalog: list[CatalogStar] = []
    for i in range(n_ra_tiles):
        tile_ra = int((ra_start + i) % 360)
        for tile_dec in range(dec_tile_lo, dec_tile_hi + 1):
            key = (tile_ra, tile_dec)
            if key not in seen:
                seen.add(key)
                catalog.extend(_tile_stars(tile_ra, tile_dec))

    # Filter magnitude and project onto sensor
    matches: list[StarMatch] = []
    for star in catalog:
        if star.magnitude > mag_limit:
            continue
        x, y = radec_to_pixels(wcs, star.ra_deg, star.dec_deg)
        if 0.0 <= x < width and 0.0 <= y < height:
            matches.append(
                StarMatch(x_px=x, y_px=y, ra_deg=star.ra_deg, dec_deg=star.dec_deg)
            )

    return matches


def _angular_sep_deg(
    ra1: float, dec1: float, ra2: float, dec2: float
) -> float:
    """Great-circle separation between two sky positions, in degrees."""
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    sin_d = math.sin((d2 - d1) / 2.0) ** 2
    sin_r = math.sin((r2 - r1) / 2.0) ** 2
    a = sin_d + math.cos(d1) * math.cos(d2) * sin_r
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))


def cone_search(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    mag_limit: float = _MAG_MAX,
) -> list[CatalogStar]:
    """Return catalog stars within ``radius_deg`` of a sky position.

    Pointing-only cone search — no WCS required.  This is the production
    catalog interface (``Gaia DR3 cone-search`` swaps in behind the same
    signature); here it is served from the procedural tile generator so the
    blind-solve matcher can be fed candidate stars from an *approximate*
    pointing alone, with no truth coupling.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Cone centre (J2000 degrees).
    radius_deg : float
        Cone radius in degrees (great-circle).
    mag_limit : float
        Faintest magnitude to include.

    Returns
    -------
    list[CatalogStar]
        Stars within the cone, brightest first.
    """
    if radius_deg <= 0.0:
        return []

    dec_lo = max(-90.0, dec_deg - radius_deg)
    dec_hi = min(90.0, dec_deg + radius_deg)
    dec_tile_lo = int(math.floor(dec_lo))
    dec_tile_hi = int(math.floor(dec_hi))

    # RA span widens toward the pole; clamp cos to avoid blow-up near ±90°.
    cos_dec = max(math.cos(math.radians(dec_deg)), 1e-6)
    ra_span = min(180.0, radius_deg / cos_dec)
    n_ra_tiles = int(math.ceil(ra_span * 2.0)) + 3
    ra_start = int(math.floor((ra_deg - ra_span) % 360.0))

    seen: set[tuple[int, int]] = set()
    result: list[CatalogStar] = []
    for i in range(n_ra_tiles):
        tile_ra = int((ra_start + i) % 360)
        for tile_dec in range(dec_tile_lo, dec_tile_hi + 1):
            key = (tile_ra, tile_dec)
            if key in seen:
                continue
            seen.add(key)
            for star in _tile_stars(tile_ra, tile_dec):
                if star.magnitude > mag_limit:
                    continue
                if (
                    _angular_sep_deg(ra_deg, dec_deg, star.ra_deg, star.dec_deg)
                    <= radius_deg
                ):
                    result.append(star)

    result.sort(key=lambda s: s.magnitude)
    return result


def catalog_stars_in_fov(
    wcs: WCSSolution,
    width: int,
    height: int,
    mag_limit: float = _MAG_MAX,
) -> list[CatalogStar]:
    """Return CatalogStar objects (with magnitude) within the sensor FOV.

    Same tile-generation logic as :func:`star_field_at` but returns
    :class:`CatalogStar` objects so callers that need magnitudes (e.g.
    :meth:`StarSpec.from_catalog`) can use them directly.

    Parameters
    ----------
    wcs : WCSSolution
        Pointing WCS.
    width, height : int
        Sensor dimensions in pixels.
    mag_limit : float
        Faintest magnitude to include.

    Returns
    -------
    list[CatalogStar]
        In-FOV stars with RA/Dec and magnitude.
    """
    cd_det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
    pixel_scale_deg = math.sqrt(cd_det) if cd_det > 0.0 else abs(wcs.cd1_1)
    fov_half_ra_sky = (width / 2.0) * pixel_scale_deg + 1.0
    fov_half_dec = (height / 2.0) * pixel_scale_deg + 1.0

    cos_dec = max(math.cos(math.radians(wcs.crval2)), 1e-6)
    ra_span = fov_half_ra_sky / cos_dec

    dec_lo = max(-90.0, wcs.crval2 - fov_half_dec)
    dec_hi = min(90.0, wcs.crval2 + fov_half_dec)
    dec_tile_lo = int(math.floor(dec_lo))
    dec_tile_hi = int(math.floor(dec_hi))

    n_ra_tiles = int(math.ceil(ra_span * 2.0)) + 3
    ra_start = int(math.floor((wcs.crval1 - ra_span) % 360.0))

    seen: set[tuple[int, int]] = set()
    catalog: list[CatalogStar] = []
    for i in range(n_ra_tiles):
        tile_ra = int((ra_start + i) % 360)
        for tile_dec in range(dec_tile_lo, dec_tile_hi + 1):
            key = (tile_ra, tile_dec)
            if key not in seen:
                seen.add(key)
                catalog.extend(_tile_stars(tile_ra, tile_dec))

    result: list[CatalogStar] = []
    for star in catalog:
        if star.magnitude > mag_limit:
            continue
        x, y = radec_to_pixels(wcs, star.ra_deg, star.dec_deg)
        if 0.0 <= x < width and 0.0 <= y < height:
            result.append(star)
    return result
