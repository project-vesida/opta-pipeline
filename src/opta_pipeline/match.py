"""Blind catalog cross-match for in-pipeline plate solving.

Turns *detected* star centroids plus an *approximate* pointing into
:class:`~opta_pipeline.astrometry.StarMatch` pairs — the input ``fit_wcs``
needs — without any truth-derived correspondence.  This is the keystone that
lets the pipeline plate-solve a frame from the image itself instead of being
handed faked matches.

Method (asterism / triangle matching + RANSAC affine vote)
----------------------------------------------------------
1. **Cone-project the catalog.** Catalog stars (RA/Dec) around the pointing
   hint are gnomonically projected to an *approximate* pixel frame using the
   hint's pixel scale and the frame centre.  The result is only correct up to
   an unknown 2-D similarity (roll, residual scale error, pointing offset).
2. **Triangles.** From the brightest detections and brightest catalog stars we
   form all triangles and describe each by a rotation- and scale-invariant
   descriptor (sorted side-length ratios).  Vertices are canonically ordered by
   opposite-side length so a descriptor match also fixes the 3 vertex
   correspondences.
3. **Vote.** Each matched triangle proposes a 2-D affine det→catalog transform
   (estimated from its 3 correspondences, which admits roll/scale/reflection).
   The transform with the most catalog inliers wins (RANSAC).
4. **Harvest.** The winning transform is applied to *every* detected star; each
   that lands within tolerance of a unique catalog star becomes a StarMatch
   carrying the detected pixel position and the catalog RA/Dec.

Because the correspondence is established geometrically, the matcher tolerates
an unknown camera roll, several-percent scale error, and a substantial pointing
offset — and rejects false detections (they simply fail to vote consistently).

Usage
-----
    from opta_pipeline.match import PointingHint, match_catalog

    hint = PointingHint(ra_deg=135.0, dec_deg=45.0, pixel_scale_arcsec=31.0)
    catalog = cone_search(hint.ra_deg, hint.dec_deg, radius_deg=hint.radius_deg)
    matches = match_catalog(star_dets, catalog, hint, frame_shape=(h, w))
    wcs = fit_wcs(matches, frame_shape=(h, w))
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import KDTree

from opta_pipeline.astrometry import StarMatch
from opta_pipeline.detect import Detection

__all__ = [
    "PointingHint",
    "CatalogEntry",
    "match_catalog",
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointingHint:
    """Approximate frame pointing used to seed the blind solve.

    The matcher uses the hint only to (a) bound the catalog cone-search and
    (b) set the projection scale.  It does **not** rely on the hint being
    accurate to roll, and tolerates several-percent scale error and a pointing
    offset of order the search radius — the triangle/RANSAC step recovers the
    true geometry.

    Attributes
    ----------
    ra_deg, dec_deg : float
        Approximate field-centre sky position (J2000 degrees).
    pixel_scale_arcsec : float
        Approximate plate scale (arcsec/pixel).
    radius_deg : float
        Catalog cone-search radius (degrees).  Should comfortably exceed half
        the field diagonal plus the pointing uncertainty.
    mag_limit : float
        Faintest catalog magnitude to query — set near the frame's detection
        depth so the reference stars exist in the image.
    """

    ra_deg: float
    dec_deg: float
    pixel_scale_arcsec: float
    radius_deg: float = 2.0
    mag_limit: float = 12.0


class CatalogEntry(Protocol):
    """Minimal catalog-star contract the matcher consumes (RA/Dec + brightness).

    Satisfied by :class:`opta_pipeline.procedural_catalog.CatalogStar`; a production
    Gaia backend returns the same shape.
    """

    @property
    def ra_deg(self) -> float: ...
    @property
    def dec_deg(self) -> float: ...
    @property
    def magnitude(self) -> float: ...


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_catalog(
    catalog: Sequence[CatalogEntry],
    hint: PointingHint,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gnomonically project catalog stars to an approximate pixel frame.

    Returns ``(xy, ra, dec, mag)`` for stars in front of the tangent point,
    where ``xy`` is an (N, 2) array of approximate pixel coordinates and the
    remaining arrays are the parallel catalog RA/Dec (degrees) and magnitude.
    """
    ra0 = math.radians(hint.ra_deg)
    dec0 = math.radians(hint.dec_deg)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    scale = hint.pixel_scale_arcsec

    xs: list[float] = []
    ys: list[float] = []
    ras: list[float] = []
    decs: list[float] = []
    mags: list[float] = []
    for star in catalog:
        ra = math.radians(star.ra_deg)
        dec = math.radians(star.dec_deg)
        d_ra = ra - ra0
        cos_dec = math.cos(dec)
        sin_dec = math.sin(dec)
        cosc = sin_d0 * sin_dec + cos_d0 * cos_dec * math.cos(d_ra)
        if cosc <= 1e-6:
            continue  # behind the tangent point
        xi = cos_dec * math.sin(d_ra) / cosc
        eta = (cos_d0 * sin_dec - sin_d0 * cos_dec * math.cos(d_ra)) / cosc
        xi_arcsec = math.degrees(xi) * 3600.0
        eta_arcsec = math.degrees(eta) * 3600.0
        xs.append(cx + xi_arcsec / scale)
        ys.append(cy + eta_arcsec / scale)
        ras.append(star.ra_deg)
        decs.append(star.dec_deg)
        mags.append(star.magnitude)

    if not xs:
        z = np.zeros(0, dtype=np.float64)
        return np.zeros((0, 2), dtype=np.float64), z, z, z
    return (
        np.column_stack([xs, ys]).astype(np.float64),
        np.asarray(ras, dtype=np.float64),
        np.asarray(decs, dtype=np.float64),
        np.asarray(mags, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Triangle descriptors
# ---------------------------------------------------------------------------


def _triangle_descriptors(
    points: np.ndarray,
    n_bright: int,
    min_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build invariant triangle descriptors for the brightest *n_bright* points.

    ``points`` must already be ordered brightest-first.  Returns
    ``(descriptors, vertex_indices)`` where ``descriptors`` is (T, 2) of
    ``(s_mid/s_max, s_min/s_max)`` and ``vertex_indices`` is (T, 3) of point
    indices ordered canonically by opposite-side length (smallest first), so a
    descriptor match also yields the 3 vertex correspondences.

    Near-degenerate (collinear) triangles are dropped via ``min_ratio``.
    """
    m = min(n_bright, len(points))
    descs: list[tuple[float, float]] = []
    verts: list[tuple[int, int, int]] = []
    for i, j, k in itertools.combinations(range(m), 3):
        p = points[[i, j, k]]
        # opposite-side length for each vertex
        s_i = float(np.hypot(*(p[1] - p[2])))  # opposite vertex i
        s_j = float(np.hypot(*(p[0] - p[2])))  # opposite vertex j
        s_k = float(np.hypot(*(p[0] - p[1])))  # opposite vertex k
        order = sorted(
            ((s_i, i), (s_j, j), (s_k, k)), key=lambda t: t[0]
        )
        s_min, s_mid, s_max = order[0][0], order[1][0], order[2][0]
        if s_max < 1e-6 or (s_min / s_max) < min_ratio:
            continue
        descs.append((s_mid / s_max, s_min / s_max))
        verts.append((order[0][1], order[1][1], order[2][1]))

    if not descs:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
        )
    return (
        np.asarray(descs, dtype=np.float64),
        np.asarray(verts, dtype=np.int64),
    )


def _similarity_from_pairs(
    src: np.ndarray,
    dst: np.ndarray,
    scale_lo: float,
    scale_hi: float,
) -> np.ndarray | None:
    """Least-squares 2-D similarity (scale·R + t) mapping src→dst (Umeyama).

    A similarity has only 4 DOF — uniform scale, rotation (reflection allowed),
    and translation — so, unlike a full affine, it cannot *collapse* or *shear*
    the plane to manufacture a high-inlier-count degenerate solution.  Returns a
    (2, 3) matrix ``[scale·R | t]`` such that ``dst ≈ M @ [x, y, 1]``, or None
    if the points are degenerate or the recovered scale is outside
    ``[scale_lo, scale_hi]`` (the catalog is projected at the hint scale, so the
    true det→catalog scale must be ≈ 1).
    """
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float((src_c**2).sum() / n)
    if var_s < 1e-9:
        return None
    cov = (dst_c.T @ src_c) / n  # (2, 2)
    u, s, vt = np.linalg.svd(cov)
    rot = u @ vt  # reflection permitted (handles CD-handedness flips)
    scale = float(s.sum() / var_s)
    if not (scale_lo <= scale <= scale_hi):
        return None
    m = np.empty((2, 3), dtype=np.float64)
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * (rot @ mu_s)
    return m


def _apply_affine(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a (2, 3) affine to an (N, 2) point array."""
    return points @ m[:, :2].T + m[:, 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_catalog(
    star_detections: Sequence[Detection],
    catalog: Sequence[CatalogEntry],
    hint: PointingHint,
    frame_shape: tuple[int, int],
    *,
    n_bright_det: int = 15,
    n_bright_cat: int = 50,
    descriptor_tol: float = 0.02,
    match_tol_px: float = 4.0,
    min_matches: int = 5,
    max_candidates: int = 600,
    scale_lo: float = 0.7,
    scale_hi: float = 1.4,
) -> list[StarMatch]:
    """Cross-match detected stars to a catalog and return StarMatch pairs.

    Parameters
    ----------
    star_detections : sequence of Detection
        Non-streak point sources from :func:`detect_sources` (any order; the
        brightest by SNR are used to seed triangle matching).
    catalog : sequence of CatalogEntry
        Candidate catalog stars near the pointing (e.g. from
        :func:`opta_pipeline.procedural_catalog.cone_search`), brightest-first
        preferred but not required.
    hint : PointingHint
        Approximate pointing used for projection and cone scaling.
    frame_shape : tuple[int, int]
        ``(height, width)`` of the frame; sets the projection centre.
    n_bright_det, n_bright_cat : int
        How many of the brightest detections / catalog stars seed the triangle
        search.
    descriptor_tol : float
        Max Euclidean distance between triangle descriptors to be a candidate.
    match_tol_px : float
        Inlier tolerance (approximate pixels) for the RANSAC vote and harvest.
    min_matches : int
        Minimum inliers required to accept a solution; below this, ``[]``.
    max_candidates : int
        Cap on triangle correspondences verified (bounds compute).
    scale_lo, scale_hi : float
        Accepted range for the recovered det→catalog similarity scale.  The
        catalog is projected at the hint scale, so the true scale is ≈ 1;
        bounding it rejects degenerate collapsing transforms.

    Returns
    -------
    list[StarMatch]
        Detected-pixel ↔ catalog-RA/Dec pairs.  Empty if no consistent
        solution with at least ``min_matches`` inliers is found.
    """
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0

    if len(star_detections) < 3 or len(catalog) < 3:
        return []

    # Detections brightest-first (SNR); all-star pixel array for the harvest.
    dets = sorted(star_detections, key=lambda d: d.snr, reverse=True)
    det_xy = np.array([[d.x, d.y] for d in dets], dtype=np.float64)

    # Project the whole cone, then keep the brightest catalog stars that fall
    # within the frame (plus a pointing-error margin).  Selecting bright stars
    # *frame-locally* — rather than across the whole cone, which may be larger
    # than the FOV — ensures the reference set are the stars actually imaged, so
    # they correspond to the bright detections.  Bright stars are also sparse,
    # which keeps the chance-match probability low.
    cat_xy, cat_ra, cat_dec, cat_mag = _project_catalog(catalog, hint, cx, cy)
    margin = 0.25 * max(h, w)
    in_frame = (
        (cat_xy[:, 0] > -margin)
        & (cat_xy[:, 0] < w + margin)
        & (cat_xy[:, 1] > -margin)
        & (cat_xy[:, 1] < h + margin)
    )
    idx = np.where(in_frame)[0]
    idx = idx[np.argsort(cat_mag[idx])][:n_bright_cat]
    cat_xy, cat_ra, cat_dec = cat_xy[idx], cat_ra[idx], cat_dec[idx]
    if len(cat_xy) < 3:
        return []

    det_desc, det_verts = _triangle_descriptors(det_xy, n_bright_det)
    cat_desc, cat_verts = _triangle_descriptors(cat_xy, len(cat_xy))
    if len(det_desc) == 0 or len(cat_desc) == 0:
        return []

    cat_desc_tree = KDTree(cat_desc)
    cat_xy_tree = KDTree(cat_xy)

    # Gather candidate triangle correspondences, nearest-descriptor first.
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for t in range(len(det_desc)):
        idxs = cat_desc_tree.query_ball_point(det_desc[t], r=descriptor_tol)
        for ci in idxs:
            dist = float(np.hypot(*(det_desc[t] - cat_desc[ci])))
            candidates.append((dist, det_verts[t], cat_verts[ci]))
    if not candidates:
        return []
    candidates.sort(key=lambda c: c[0])

    # RANSAC: verify each candidate affine by catalog-inlier count.
    best_inliers = -1
    best_affine: np.ndarray | None = None
    for _, dv, cv in candidates[:max_candidates]:
        m = _similarity_from_pairs(det_xy[dv], cat_xy[cv], scale_lo, scale_hi)
        if m is None:
            continue
        projected = _apply_affine(m, det_xy)
        d, ci = cat_xy_tree.query(projected, distance_upper_bound=match_tol_px)
        # Vote on distinct catalog stars, not on detections: several
        # detections landing inside tol of one star are one piece of evidence
        # for this transform, not N.  Counting them all biased selection
        # toward affines that pile the field onto a dense region.  The harvest
        # below already dedups the output; the vote had not.
        finite = np.isfinite(d)
        n_in = int(np.unique(np.asarray(ci)[finite]).size)
        if n_in > best_inliers:
            best_inliers = n_in
            best_affine = m

    if best_affine is None or best_inliers < min_matches:
        return []

    # Refine: re-estimate the affine by least squares over the winning inlier
    # set, then re-harvest.  This removes the noise of a single-triangle
    # estimate and tightens correspondences in a dense field, cutting false
    # matches before any reach the WCS fit.
    projected = _apply_affine(best_affine, det_xy)
    dist, cat_idx = (
        np.asarray(a)
        for a in cat_xy_tree.query(projected, distance_upper_bound=match_tol_px)
    )
    inlier = np.isfinite(dist)
    if int(inlier.sum()) >= 3:
        refined = _similarity_from_pairs(
            det_xy[inlier], cat_xy[cat_idx[inlier]], scale_lo, scale_hi
        )
        if refined is not None:
            best_affine = refined

    # Harvest: map every detection, accept unique nearest catalog stars.
    projected = _apply_affine(best_affine, det_xy)
    dist, cat_idx = (
        np.asarray(a)
        for a in cat_xy_tree.query(projected, distance_upper_bound=match_tol_px)
    )

    # Resolve conflicts: each catalog star claimed by its closest detection.
    claim: dict[int, tuple[float, int]] = {}
    for di in range(len(det_xy)):
        if not np.isfinite(dist[di]):
            continue
        ci = int(cat_idx[di])
        prev = claim.get(ci)
        if prev is None or dist[di] < prev[0]:
            claim[ci] = (float(dist[di]), di)

    matches = [
        StarMatch(
            x_px=float(det_xy[di, 0]),
            y_px=float(det_xy[di, 1]),
            ra_deg=float(cat_ra[ci]),
            dec_deg=float(cat_dec[ci]),
        )
        for ci, (_, di) in claim.items()
    ]
    if len(matches) < min_matches:
        return []
    return matches
