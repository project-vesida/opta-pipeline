"""Astrometry module for OpTA pipeline.

Computes the World Coordinate System (WCS) for a frame by fitting a TAN
(gnomonic) plate solution to matched star positions, then projects detection
centroids from pixel coordinates to (RA, Dec).

Architecture
------------
1. `fit_wcs(matches)` — least-squares WCS fit in the gnomonic tangent plane
   using matched :class:`StarMatch` pairs (pixel xy ↔ RA/Dec).  Produces a
   `WCSSolution`.
2. `pixels_to_radec(wcs, x, y)` — apply WCS to convert pixel positions to
   equatorial coordinates.
3. `apply_rolling_shutter_correction(detection, wcs, row_readout_us, ...)` —
   correct centroid position for rolling-shutter timing bias (T-03).
4. `astrometrise_detections(detections, wcs, ...)` — apply the WCS to a
   full list of detections, returning `AstrometricDetection` objects.

WCS model: Standard TAN (gnomonic) projection, 6-parameter affine in the
tangent plane (ξ, η) about the reference point (CRVAL1, CRVAL2):
    [ξ]   [CD1_1  CD1_2] [x - CRPIX1]
    [η] = [CD2_1  CD2_2] [y - CRPIX2]
followed by the exact inverse gnomonic mapping (ξ, η) → (RA, Dec).  A
rectilinear optic delivers pixel offsets proportional to (ξ, η); the mapping
to (ΔRA·cosδ, Δδ) is non-linear at ~θ³/3 (≈19′ at the 14.5° half-diagonal of
the production 25.2°×14.4° field), so fitting in the tangent plane — not in
sky-coordinate differences — is what keeps wide fields sub-arcsecond.

Usage
-----
    from opta_pipeline.astrometry import fit_wcs, astrometrise_detections

    wcs = fit_wcs(star_matches, frame_shape=(h, w))
    astro_dets = astrometrise_detections(detections, wcs)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as f_distribution

from opta_pipeline.detect import Detection

logger = logging.getLogger(__name__)

__all__ = [
    "StarMatch",
    "WCSSolution",
    "AstrometricDetection",
    "fit_wcs",
    "fit_wcs_sip",
    "pixels_to_radec",
    "radec_to_pixels",
    "apply_rolling_shutter_correction",
    "astrometrise_detections",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarMatch:
    """A matched star pair: pixel position ↔ catalog RA/Dec (J2000).

    Parameters
    ----------
    x_px, y_px : float
        Centroid pixel position (0-indexed).
    ra_deg, dec_deg : float
        Catalog RA/Dec in degrees (J2000).
    """

    x_px: float
    y_px: float
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class WCSSolution:
    """WCS plate solution (TAN gnomonic projection, 6-parameter affine).

    The CD matrix maps pixel offsets from CRPIX to gnomonic tangent-plane
    coordinates (ξ, η) in degrees about (CRVAL1, CRVAL2); the sky position
    follows from the exact inverse gnomonic projection (see
    :func:`pixels_to_radec`).

    Attributes
    ----------
    crpix1, crpix2 : float
        Reference pixel (1-indexed FITS convention).
    crval1, crval2 : float
        Sky coordinates at reference pixel (degrees, J2000).
    cd1_1, cd1_2, cd2_1, cd2_2 : float
        CD matrix elements (degrees/pixel).
    rms_arcsec : float
        RMS residual of the plate fit in arcseconds, **2-D radial**:
        ``rms² = mean(res_ξ² + res_η²) = σ_ξ² + σ_η²`` over the fitted stars
        (see :func:`fit_wcs`).  It is therefore ≈ √2 larger than the per-axis
        residual sigma of an isotropic residual field — consumers reporting a
        *per-axis* uncertainty (I-02 ``sigma_ra``/``sigma_dec``) must use
        ``rms_arcsec² / 2``, as :func:`astrometrise_detections` does.  The
        radial definition is load-bearing for the ``accuracy_flag``
        self-check and the pinned OpTA.NOD.ACC thresholds — do not change it.
    n_stars : int
        Number of matched stars used in the fit.
    sip_a1, sip_a2 : float
        Radial SIP **undistortion** coefficients.  A measured (as-imaged)
        pixel is undistorted to its ideal gnomonic position before the CD
        matrix is applied::

            f = 1 + sip_a1·rn² + sip_a2·rn⁴
            (xu, yu) = centre + (x − centre)·f

        where ``rn`` is the radius from ``(sip_cx, sip_cy)`` normalised by
        ``sip_rhalf``.  Both zero (the default) → identity → a plain linear
        TAN solve, so every existing call site is unaffected.  Fit by
        :func:`fit_wcs_sip`; the radial form mirrors the dominant term of a
        FITS-SIP / Brown–Conrady distortion (T-11).
    sip_cx, sip_cy : float
        Distortion centre in pixels (the CRPIX/frame centre for a centred
        prime).  Unused when ``sip_a1 == sip_a2 == 0``.
    sip_rhalf : float
        Half-diagonal radius (px) normalising the SIP polynomial.  Defaults
        to 1.0; only meaningful with non-zero coefficients.
    accuracy_flag : bool
        Post-solve OpTA.NOD.ACC self-check (set by :func:`fit_wcs_sip`).
        ``True`` when the *returned* solution's honest residuals — evaluated
        against **all** input matches, before any outlier pruning — exceed
        the accuracy budget either over the whole field or in the outer
        field annulus (where radial distortion residuals concentrate).  A
        flagged solution must not be trusted for tracklet astrometry
        without inspection; the fit also logs a warning.  Default ``False``
        (also for :func:`fit_wcs`, which predates the check).  Consumed by
        the orchestrator (``opta_pipeline.pipeline``): the flag propagates
        into ``FrameDetections.wcs_accuracy_flag`` and annotates every
        tracklet containing flagged frames
        (``Tracklet.wcs_accuracy_flagged``); with
        ``AstrometryConfig.skip_flagged_frames`` (default off —
        annotate-only) flagged frames are excluded from astrometry.
    """

    crpix1: float
    crpix2: float
    crval1: float
    crval2: float
    cd1_1: float
    cd1_2: float
    cd2_1: float
    cd2_2: float
    rms_arcsec: float
    n_stars: int
    sip_a1: float = 0.0
    sip_a2: float = 0.0
    sip_cx: float = 0.0
    sip_cy: float = 0.0
    sip_rhalf: float = 1.0
    accuracy_flag: bool = False


@dataclass(frozen=True)
class AstrometricDetection:
    """A pipeline detection with equatorial coordinates assigned.

    Attributes
    ----------
    detection : Detection
        Source detection (centroid, SNR, etc.).
    ra_deg : float
        Right Ascension in degrees (J2000).
    dec_deg : float
        Declination in degrees (J2000).
    sigma_ra_arcsec : float
        **Per-axis** RA uncertainty (arcsec), on the RA·cos(Dec) great-circle
        scale: the per-axis plate-solve residual ``WCSSolution.rms_arcsec²/2``
        (that RMS is 2-D radial) RSS'd with the per-axis centroiding error
        (0.3 px).  Not a radial/total position error.
    sigma_dec_arcsec : float
        Per-axis Dec uncertainty (arcsec); same construction as
        ``sigma_ra_arcsec`` and numerically equal to it — the plate fit's
        residual field is treated as isotropic, so no per-axis anisotropy is
        propagated.
    """

    detection: Detection
    ra_deg: float
    dec_deg: float
    sigma_ra_arcsec: float
    sigma_dec_arcsec: float


# ---------------------------------------------------------------------------
# Gnomonic (TAN) tangent-plane projection
# ---------------------------------------------------------------------------
#
# The same standard equations as ``match.py::_project_catalog`` (which must
# keep its own copy: match.py already imports from this module, so the shared
# implementation lives here on the dependency-safe side).

# Below this the tangent point is ≥ ~89.99997° away — treat as "behind the
# tangent plane" and let the projection blow up to far-out-of-frame pixels
# instead of mirroring antipodal points into the frame.
_COS_C_MIN = 1e-9

# Positive degrees of freedom for the 6-parameter fit needs 2·n − 6 > 0.
_MIN_PRUNED_STARS = 4

# SIP-vs-linear model selection (fit_wcs_sip): significance level of the
# nested-model F-test.  Bounds the probability of accepting SIP terms fitted
# to pure centroid noise at 0.1 % per solve, while distortion whose residual
# is a fraction of the noise floor — invisible to the old 0.5×-RMS amplitude
# gate — is still detected (see the model-selection block in fit_wcs_sip).
_SIP_FTEST_P = 1e-3

# When the linear model already fits below this residual (arcsec) the fit is
# at numerical precision (noiseless synthetic fields; physical centroid noise
# at production scale is ≥ ~0.02″): the F ratio would compare roundoff, so
# model selection keeps the linear solution.
_LINEAR_PERFECT_RMS_ARCSEC = 1e-6

# Post-solve OpTA.NOD.ACC self-check (_check_accuracy): the outer field
# annulus starts at this fraction of the half-diagonal radius (radial
# distortion residuals grow ~r³, so a field-averaged RMS can look clean while
# the edge is out of budget), and the annulus RMS is only meaningful with at
# least this many stars in it.
_EDGE_ANNULUS_RN = 0.7
_MIN_ANNULUS_STARS = 8


def _radec_to_tangent(
    ra_deg: np.ndarray | float,
    dec_deg: np.ndarray | float,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic projection sky → tangent plane (ξ, η), all in degrees.

    RA-wrap safe: only trigonometric functions of ΔRA are used, so fields
    straddling the 0°/360° branch cut project correctly.  Points at or
    behind the tangent-plane horizon are clamped (they land far outside any
    physical frame rather than mirroring through the tangent point).
    """
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)

    d_ra = ra - ra0
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    cos_dra = np.cos(d_ra)
    cos_c = np.maximum(sin_d0 * sin_dec + cos_d0 * cos_dec * cos_dra, _COS_C_MIN)
    xi = cos_dec * np.sin(d_ra) / cos_c
    eta = (cos_d0 * sin_dec - sin_d0 * cos_dec * cos_dra) / cos_c
    return np.degrees(xi), np.degrees(eta)


def _tangent_to_radec(
    xi_deg: np.ndarray | float,
    eta_deg: np.ndarray | float,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse gnomonic: tangent plane (ξ, η) → sky, all in degrees.

    Standard TAN deprojection (SLALIB ``dtp2s`` form).  RA is returned in
    [0, 360).
    """
    xi = np.radians(np.asarray(xi_deg, dtype=np.float64))
    eta = np.radians(np.asarray(eta_deg, dtype=np.float64))
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)

    denom = cos_d0 - eta * sin_d0
    ra = np.degrees(ra0 + np.arctan2(xi, denom)) % 360.0
    dec = np.degrees(np.arctan2(sin_d0 + eta * cos_d0, np.hypot(xi, denom)))
    return ra, dec


def _circular_mean_ra_deg(ra_deg: np.ndarray) -> float:
    """Circular mean of right ascensions in degrees, safe across RA 0/360."""
    ra = np.radians(ra_deg)
    return math.degrees(
        math.atan2(float(np.mean(np.sin(ra))), float(np.mean(np.cos(ra))))
    ) % 360.0


# ---------------------------------------------------------------------------
# WCS fitting
# ---------------------------------------------------------------------------


def fit_wcs(
    matches: list[StarMatch],
    frame_shape: tuple[int, int] | None = None,
    max_residual_arcsec: float = 4.0,
) -> WCSSolution:
    """Fit a TAN (gnomonic) WCS plate solution from matched star pairs.

    Catalog positions are projected onto the gnomonic tangent plane about the
    current reference point, and a 3-parameter least-squares fit per axis
    (two CD terms plus bias) solves

        ξ = CD1_1·dx + CD1_2·dy + bias_ξ
        η = CD2_1·dx + CD2_2·dy + bias_η

    The bias terms are absorbed into a refined CRVAL (by exact inverse
    gnomonic projection) and the fit is re-projected about the refined
    centre until it converges, so the returned WCSSolution uses
    CRPIX = frame center with the exact sky position — self-consistent with
    :func:`pixels_to_radec`.  The initial CRVAL1 estimate is a *circular*
    mean, and all ΔRA arithmetic goes through trig of ΔRA, so fields
    straddling the RA 0°/360° branch cut solve correctly.

    A coarse-then-fine outlier rejection step removes stars whose residuals
    exceed max_residual_arcsec before re-fitting on the clean set.  A prune
    is only accepted when the surviving set keeps positive degrees of
    freedom (≥ 4 stars) *and* at least half of the candidate stars —
    otherwise the un-pruned fit with its honest (large) RMS is returned, so
    a model mismatch can never masquerade as a near-exact tiny-star fit.

    Parameters
    ----------
    matches : list[StarMatch]
        Matched star pairs.  At least 3 required for a non-degenerate fit.
    frame_shape : tuple[int, int] | None
        (height, width) used to set CRPIX at the frame center.
        If None, the centroid of matched pixel positions is used.
    max_residual_arcsec : float
        Outlier rejection threshold in arcseconds.

    Returns
    -------
    WCSSolution
        Plate solution with RMS residual.

    Raises
    ------
    ValueError
        If fewer than 3 matches are provided or the fit is degenerate.
    """
    if len(matches) < 3:
        raise ValueError("At least 3 star matches required for WCS fit")

    # Reference pixel: frame center (FITS 0-indexed)
    if frame_shape is not None:
        h, w = frame_shape
        crpix1 = w / 2.0
        crpix2 = h / 2.0
    else:
        crpix1 = float(np.mean([m.x_px for m in matches]))
        crpix2 = float(np.mean([m.y_px for m in matches]))

    def _fit(
        match_list: list[StarMatch],
    ) -> tuple[float, float, float, float, float, float, float]:
        """Tangent-plane fit; returns (crval1, crval2, c11, c12, c21, c22, rms).

        Projects the catalog positions gnomonically about the current CRVAL
        estimate, solves the linear system, absorbs the fitted bias into a
        refined CRVAL by exact inverse projection, and iterates until the
        bias vanishes — so the CD matrix is expressed about the *returned*
        CRVAL, self-consistent with pixels_to_radec.
        """
        n = len(match_list)
        dx = np.array([m.x_px - crpix1 for m in match_list])
        dy = np.array([m.y_px - crpix2 for m in match_list])
        ra_arr = np.array([m.ra_deg for m in match_list])
        dec_arr = np.array([m.dec_deg for m in match_list])

        # Circular mean: safe when the field straddles RA 0°/360°.
        cv1 = _circular_mean_ra_deg(ra_arr)
        cv2 = float(np.mean(dec_arr))

        # Design matrix: [dx, dy, 1] — bias term captures CRVAL offset
        A = np.column_stack([dx, dy, np.ones(n)])
        if np.linalg.matrix_rank(A[:, :2]) < 2:
            raise ValueError("Degenerate star configuration — cannot solve WCS")

        # Iterate the tangent point: the fitted bias is the tangent-plane
        # position of CRPIX; deproject it to get the true CRVAL, then re-fit
        # about it.  Re-centring contracts the residual mis-projection by
        # ~θ_field² per pass (a gnomonic about the wrong centre is projective,
        # not affine, in the true tangent plane), so a handful of these cheap
        # 3-column lstsq passes reach machine precision even on the 25° field.
        for _ in range(10):
            xi, eta = _radec_to_tangent(ra_arr, dec_arr, cv1, cv2)
            p_xi, _, _, _ = np.linalg.lstsq(A, xi, rcond=None)
            p_eta, _, _, _ = np.linalg.lstsq(A, eta, rcond=None)
            bias = math.hypot(float(p_xi[2]), float(p_eta[2]))
            if bias < 1e-12:  # degrees — sub-microarcsecond
                break
            ra_c, dec_c = _tangent_to_radec(
                float(p_xi[2]), float(p_eta[2]), cv1, cv2
            )
            cv1, cv2 = float(ra_c), float(dec_c)

        c11, c12 = float(p_xi[0]), float(p_xi[1])
        c21, c22 = float(p_eta[0]), float(p_eta[1])

        # Residuals in the tangent plane about the converged CRVAL.
        res_xi = (xi - (c11 * dx + c12 * dy + float(p_xi[2]))) * 3600.0
        res_eta = (eta - (c21 * dx + c22 * dy + float(p_eta[2]))) * 3600.0
        rms = float(np.sqrt(np.mean(res_xi**2 + res_eta**2)))

        return cv1, cv2, c11, c12, c21, c22, rms

    def _residuals(
        match_list: list[StarMatch],
        cv1: float,
        cv2: float,
        cd11: float,
        cd12: float,
        cd21: float,
        cd22: float,
    ) -> list[float]:
        """Return per-match tangent-plane residuals in arcsec."""
        dx = np.array([m.x_px - crpix1 for m in match_list])
        dy = np.array([m.y_px - crpix2 for m in match_list])
        ra_arr = np.array([m.ra_deg for m in match_list])
        dec_arr = np.array([m.dec_deg for m in match_list])
        xi, eta = _radec_to_tangent(ra_arr, dec_arr, cv1, cv2)
        res_xi = (xi - (cd11 * dx + cd12 * dy)) * 3600.0
        res_eta = (eta - (cd21 * dx + cd22 * dy)) * 3600.0
        return list(np.hypot(res_xi, res_eta))

    def _accept_prune(pruned: list[StarMatch], current: list[StarMatch]) -> bool:
        """A prune must keep positive DOF and at least half the stars.

        Without this, a model mismatch (e.g. distortion the model cannot
        absorb) lets the rejection loop collapse 200 stars to a 3-star exact
        fit reporting rms 0.00″ — a silent success carrying a garbage WCS.
        Refusing the prune keeps the honest large RMS visible to the caller.
        """
        return (
            len(pruned) < len(current)
            and len(pruned) >= _MIN_PRUNED_STARS
            and 2 * len(pruned) >= len(current)
        )

    # --- Initial fit on all stars ---
    crval1, crval2, c11, c12, c21, c22, rms = _fit(matches)
    current = list(matches)

    # --- Coarse outlier pass (100× threshold): removes extreme outliers that
    #     would otherwise contaminate the fine-pass fit. ---
    coarse_thresh = max_residual_arcsec * 100.0
    resids = _residuals(current, crval1, crval2, c11, c12, c21, c22)
    coarse = [m for m, r in zip(current, resids) if r <= coarse_thresh]
    if _accept_prune(coarse, current):
        crval1, crval2, c11, c12, c21, c22, rms = _fit(coarse)
        current = coarse

    # --- Fine outlier pass (normal threshold) ---
    resids = _residuals(current, crval1, crval2, c11, c12, c21, c22)
    fine = [m for m, r in zip(current, resids) if r <= max_residual_arcsec]
    if _accept_prune(fine, current):
        crval1, crval2, c11, c12, c21, c22, rms = _fit(fine)
        n_final = len(fine)
    else:
        n_final = len(current)

    return WCSSolution(
        crpix1=crpix1,
        crpix2=crpix2,
        crval1=crval1,
        crval2=crval2,
        cd1_1=c11,
        cd1_2=c12,
        cd2_1=c21,
        cd2_2=c22,
        rms_arcsec=rms,
        n_stars=n_final,
    )


def _check_accuracy(
    wcs: WCSSolution,
    matches: list[StarMatch],
    frame_shape: tuple[int, int],
    budget_arcsec: float,
) -> WCSSolution:
    """Post-solve OpTA.NOD.ACC self-check; returns *wcs* (flagged if violated).

    Evaluates the returned solution's sky residuals against **all** input
    matches — deliberately including the ones the outlier loop pruned, so a
    fit that discarded a distorted field edge cannot certify itself with the
    survivors' small RMS.  Two statistics are compared to the budget:

    * global RMS over the full match set, and
    * RMS over the outer field annulus (``rn ≥ 0.7`` of the half-diagonal,
      when populated) — radial distortion residuals grow ~r³, so the edge
      can be far out of budget while the field average still looks clean.

    On violation the solution is returned with ``accuracy_flag=True`` and a
    warning is logged; the WCS itself is unchanged.  This makes a *silent*
    accuracy violation impossible for any systematic the star matches sample
    (residual structure below the star-match noise floor remains invisible —
    that is a measurement limit, not a gate defect).

    **Why the gate stays on the full (non-robust) RMS** (policy decision,
    2026-07-12): making the statistic outlier-robust (median / trimmed RMS)
    was considered for the case where a few catalog-mismatch outliers — the
    population the outlier prune exists for — dominate an otherwise-in-budget
    fit.  The one known trip of this flag (the validation smoke scene,
    unpruned RMS 15.14″) was investigated and is *not* that case: all six
    matches carry 7.6–17.7″ residuals, the median (17.7″) sits *above* the
    full RMS, and the pattern is deterministic across frames — broad model
    error (the harness's synthetic star grid is generated with a flat linear
    approximation that disagrees with the exact gnomonic projection by up to
    ~21″ at the smoke FOV corners), which a robust statistic would not excuse
    and must not hide.  The warning therefore logs the median residual
    *alongside* the full RMS as a diagnostic (median ≈ RMS ⇒ broad error;
    median ≪ RMS ⇒ outlier-dominated, inspect the match set), but
    certification gates on the honest full RMS.  The 10″ budget is
    OpTA.NOD.ACC and is not loosened here; consumption policy (annotate
    tracklets, optionally skip flagged frames) lives in
    ``opta_pipeline.pipeline`` / ``AstrometryConfig.skip_flagged_frames``.
    """
    if not matches:
        return wcs
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    r_half = math.hypot(cx, cy)
    res = np.empty(len(matches))
    rn = np.empty(len(matches))
    for i, m in enumerate(matches):
        ra, dec = pixels_to_radec(wcs, m.x_px, m.y_px)
        dxi, deta = _radec_to_tangent(ra, dec, m.ra_deg, m.dec_deg)
        res[i] = math.hypot(float(dxi), float(deta)) * 3600.0
        rn[i] = math.hypot(m.x_px - cx, m.y_px - cy) / r_half
    rms_all = float(np.sqrt(np.mean(res**2)))
    outer = res[rn >= _EDGE_ANNULUS_RN]
    rms_outer = (
        float(np.sqrt(np.mean(outer**2)))
        if len(outer) >= _MIN_ANNULUS_STARS
        else 0.0
    )
    if rms_all <= budget_arcsec and rms_outer <= budget_arcsec:
        return wcs
    # Median logged as a robust companion statistic for diagnosis only —
    # median ≈ RMS means broad model error, median ≪ RMS means a few
    # outliers dominate (see the policy note in the docstring).  The gate
    # itself stays on the honest full RMS.
    logger.warning(
        "plate solution violates the OpTA.NOD.ACC budget: unpruned residual "
        'RMS %.2f" (median %.2f", full field) / %.2f" (outer annulus, '
        '%d stars) vs budget %.1f" — accuracy_flag set',
        rms_all,
        float(np.median(res)),
        rms_outer,
        len(outer),
        budget_arcsec,
    )
    return replace(wcs, accuracy_flag=True)


def fit_wcs_sip(
    matches: list[StarMatch],
    frame_shape: tuple[int, int],
    max_residual_arcsec: float = 4.0,
    accuracy_budget_arcsec: float = 10.0,
) -> WCSSolution:
    """Fit a **distortion-aware** WCS: linear TAN + a radial SIP term.

    A linear CD matrix cannot absorb the several-percent radial distortion of
    a wide, fast prime; the residual grows with field radius and at a realistic
    ≥ 3 % barrel the field-edge error blows the 10″ OpTA.NOD.ACC budget (see
    ``test_synth_distortion``).  This solver adds two radial undistortion
    coefficients ``(a1, a2)`` and fits all eight parameters jointly by
    Levenberg–Marquardt least squares, seeded from the linear :func:`fit_wcs`.

    The distortion centre is fixed at the frame centre (CRPIX) and the
    polynomial is normalised by the frame half-diagonal — matching a centred
    prime and the generator's :class:`~opta_pipeline.synth.distortion.DistortionModel`.
    With no distortion present the coefficients drive to ≈ 0 and the result
    matches the linear solve, so this is a safe drop-in for the linear path.

    SIP-vs-linear model selection is a nested-model F-test on the fit
    residuals (see the model-selection block), and every returned solution
    — SIP, linear, or the few-star fallback — passes through a post-solve
    OpTA.NOD.ACC self-check (:func:`_check_accuracy`) that sets
    ``accuracy_flag`` and logs a warning when the honest unpruned residuals
    exceed ``accuracy_budget_arcsec``, so an out-of-budget solution can
    never be returned silently.

    Parameters
    ----------
    matches : list[StarMatch]
        Matched star pairs.  At least 8 required so the two extra distortion
        degrees of freedom are comfortably over-determined; with fewer this
        falls back to :func:`fit_wcs`.
    frame_shape : tuple[int, int]
        ``(height, width)`` — sets CRPIX (frame centre) and the normalising
        half-diagonal radius.
    max_residual_arcsec : float
        Outlier rejection threshold (arcsec), applied coarse-then-fine as in
        :func:`fit_wcs` — except that the fine pass runs *after* model
        selection (see the coarse-prune block for why).
    accuracy_budget_arcsec : float
        Post-solve accuracy self-check threshold (arcsec).  Default 10.0 =
        OpTA.NOD.ACC (≤ 10.0″ RMS per node, ``opta-engineering/SYSTEMS.md``).
        This does not alter the fit — it only controls ``accuracy_flag``.

    Returns
    -------
    WCSSolution
        Plate solution with ``sip_a1/sip_a2`` populated (zero when the
        F-test keeps the linear model) and ``accuracy_flag`` set by the
        post-solve budget check.
    """
    if len(matches) < 8:
        # Two extra DOF need a comfortable margin of constraints; fall back.
        return _check_accuracy(
            fit_wcs(matches, frame_shape, max_residual_arcsec),
            matches,
            frame_shape,
            accuracy_budget_arcsec,
        )

    h, w = frame_shape
    crpix1 = w / 2.0
    crpix2 = h / 2.0
    r_half = math.hypot(w / 2.0, h / 2.0)

    seed = fit_wcs(matches, frame_shape, max_residual_arcsec)
    p0 = np.array(
        [seed.crval1, seed.crval2, seed.cd1_1, seed.cd1_2,
         seed.cd2_1, seed.cd2_2, 0.0, 0.0]
    )

    def _solve(
        match_list: list[StarMatch], p_init: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joint LM fit on a match subset; returns (params, per-match resid as)."""
        x = np.array([m.x_px for m in match_list])
        y = np.array([m.y_px for m in match_list])
        ra = np.array([m.ra_deg for m in match_list])
        dec = np.array([m.dec_deg for m in match_list])

        def residuals(p: np.ndarray) -> np.ndarray:
            cv1, cv2, c11, c12, c21, c22, a1, a2 = p
            # Radial undistortion about the frame centre.
            dxc = x - crpix1
            dyc = y - crpix2
            rn2 = (dxc * dxc + dyc * dyc) / (r_half * r_half)
            f = 1.0 + a1 * rn2 + a2 * rn2 * rn2
            dx = dxc * f
            dy = dyc * f
            # Model vs catalog in the gnomonic tangent plane about (cv1, cv2):
            # the radial terms then absorb real lens distortion, not the
            # projection's own non-linearity.
            xi_cat, eta_cat = _radec_to_tangent(ra, dec, cv1, cv2)
            d_xi = (xi_cat - (c11 * dx + c12 * dy)) * 3600.0
            d_eta = (eta_cat - (c21 * dx + c22 * dy)) * 3600.0
            return np.concatenate([d_xi, d_eta])

        sol = least_squares(
            residuals, p_init, method="lm", x_scale="jac", max_nfev=400
        )
        res = sol.fun
        n = len(match_list)
        per_match = np.hypot(res[:n], res[n:])
        return sol.x, per_match

    # --- Initial joint fit on all matches ---
    p, per_match = _solve(matches, p0)
    current = list(matches)

    # --- Coarse outlier rejection only (100× threshold; degenerate-prune
    #     guard as in fit_wcs: keep ≥ 8 stars and ≥ half the candidates).
    #     The fine prune is deferred until *after* model selection: pruning
    #     at max_residual_arcsec by the SIP fit's own residuals truncates the
    #     noise tail of the comparison set and inflates the F-test null ~2×
    #     (fine-pruned set: mean F ≈ 1.8, spurious SIP acceptance 1/5 seeds
    #     at σ = 0.10 px noise-only; coarse-only set: mean F = 1.08, max
    #     4.07, 0/60 seeds accepted — measured 2026-07-12 on the repro
    #     script's noise-only field generator, cf.
    #     opta-pipeline/scripts/repro_sip_subthreshold_window.py a1 = 0
    #     rows). ---
    keep = [
        m for m, r in zip(current, per_match)
        if r <= max_residual_arcsec * 100.0
    ]
    if 8 <= len(keep) < len(current) and 2 * len(keep) >= len(current):
        p, per_match = _solve(keep, p)
        current = keep

    # --- Model selection: keep the two distortion DOF only when they explain
    # real structure beyond the centroid-noise floor.  Nested-model F-test on
    # the same inlier set: SSE_lin from a linear fit with no outlier rejection
    # (so uncorrected distortion shows up as a large linear residual rather
    # than being thrown away) vs SSE_sip from the joint fit, 2 extra
    # parameters against 2n − 8 residual DOF.
    #
    # The criterion must reject BOTH failure ends:
    #   * SIP terms fitted to pure noise (the reason a gate exists at all):
    #     under noise the F statistic follows F(2, 2n−8), so the p = 1e-3
    #     critical value bounds spurious acceptance at 0.1 % per solve.
    #   * The sub-threshold-distortion blind window (P1, 2026-07-11): the
    #     previous amplitude-style gate ``rms_sip < 0.5·rms_lin`` needed the
    #     distortion residual D to exceed √3× the noise floor before SIP was
    #     accepted, yet D grows ~r³ toward the field edge — at the production
    #     IMX585 scale a1 ≈ 5e-4 with σ = 0.10 px was rejected on every seed
    #     while the true field-edge error reached 12.5″ > the 10″ OpTA.NOD.ACC
    #     budget, silently (recomputed 2026-07-12 via
    #     opta-pipeline/scripts/repro_sip_subthreshold_window.py).  The same
    #     cell gives an SSE ratio ≈ 2 → F ≫ F_crit, decisively accepted now.
    #
    # Guard: when the linear model already fits to numerical precision
    # (noiseless synthetic gnomonic fields), the SSE ratio is roundoff
    # garbage — model selection is moot, keep the linear seed. ---
    linear_cmp = fit_wcs(current, frame_shape, max_residual_arcsec=1e9)
    n = len(current)
    dof = 2 * n - 8
    sse_sip = float(np.sum(per_match**2))
    sse_lin = n * linear_cmp.rms_arcsec**2
    accept_sip = False
    if linear_cmp.rms_arcsec > _LINEAR_PERFECT_RMS_ARCSEC and sse_sip > 0.0:
        f_stat = ((sse_lin - sse_sip) / 2.0) / (sse_sip / dof)
        accept_sip = f_stat > f_distribution.ppf(1.0 - _SIP_FTEST_P, 2, dof)
    if not accept_sip:
        return _check_accuracy(seed, matches, frame_shape, accuracy_budget_arcsec)

    # --- Fine outlier rejection *within* the accepted SIP model (same
    #     degenerate-prune guard) — safe now that the model choice is made,
    #     and it keeps moderate mismatches out of the reported solution. ---
    keep = [m for m, r in zip(current, per_match) if r <= max_residual_arcsec]
    if 8 <= len(keep) < len(current) and 2 * len(keep) >= len(current):
        p, per_match = _solve(keep, p)
        current = keep
    rms = float(np.sqrt(np.mean(per_match**2)))

    return _check_accuracy(
        WCSSolution(
            crpix1=crpix1,
            crpix2=crpix2,
            crval1=float(p[0]),
            crval2=float(p[1]),
            cd1_1=float(p[2]),
            cd1_2=float(p[3]),
            cd2_1=float(p[4]),
            cd2_2=float(p[5]),
            rms_arcsec=rms,
            n_stars=len(current),
            sip_a1=float(p[6]),
            sip_a2=float(p[7]),
            sip_cx=crpix1,
            sip_cy=crpix2,
            sip_rhalf=r_half,
        ),
        matches,
        frame_shape,
        accuracy_budget_arcsec,
    )


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


def _undistort(wcs: WCSSolution, x: float, y: float) -> tuple[float, float]:
    """Map a measured (distorted) pixel to its ideal gnomonic pixel.

    Applies the radial SIP undistortion polynomial; identity when the WCS
    carries no distortion (``sip_a1 == sip_a2 == 0``).
    """
    if wcs.sip_a1 == 0.0 and wcs.sip_a2 == 0.0:
        return x, y
    dx = x - wcs.sip_cx
    dy = y - wcs.sip_cy
    rn2 = (dx * dx + dy * dy) / (wcs.sip_rhalf * wcs.sip_rhalf)
    f = 1.0 + wcs.sip_a1 * rn2 + wcs.sip_a2 * rn2 * rn2
    return wcs.sip_cx + dx * f, wcs.sip_cy + dy * f


def _redistort(
    wcs: WCSSolution, xu: float, yu: float, *, n_iter: int = 12
) -> tuple[float, float]:
    """Map an ideal gnomonic pixel back to its measured (distorted) position.

    Fixed-point inverse of :func:`_undistort`; identity when undistorted.
    """
    if wcs.sip_a1 == 0.0 and wcs.sip_a2 == 0.0:
        return xu, yu
    dxu = xu - wcs.sip_cx
    dyu = yu - wcs.sip_cy
    ru = math.hypot(dxu, dyu)
    if ru == 0.0:
        return xu, yu
    rd = ru  # initial guess
    for _ in range(n_iter):
        rn2 = (rd / wcs.sip_rhalf) ** 2
        f = 1.0 + wcs.sip_a1 * rn2 + wcs.sip_a2 * rn2 * rn2
        rd = ru / f
    scale = rd / ru
    return wcs.sip_cx + dxu * scale, wcs.sip_cy + dyu * scale


def pixels_to_radec(wcs: WCSSolution, x: float, y: float) -> tuple[float, float]:
    """Convert a pixel position to (RA, Dec) using a fitted WCS.

    Parameters
    ----------
    wcs : WCSSolution
        Plate solution from fit_wcs.
    x, y : float
        Pixel coordinates (0-indexed).

    Returns
    -------
    (ra_deg, dec_deg) : tuple[float, float]
        Equatorial coordinates in degrees (J2000).
    """
    x, y = _undistort(wcs, x, y)
    dx = x - wcs.crpix1
    dy = y - wcs.crpix2
    # CD matrix → gnomonic tangent-plane coordinates (dimensionless tangents)
    xi = math.radians(wcs.cd1_1 * dx + wcs.cd1_2 * dy)
    eta = math.radians(wcs.cd2_1 * dx + wcs.cd2_2 * dy)
    # Exact inverse gnomonic projection about (CRVAL1, CRVAL2)
    dec0 = math.radians(wcs.crval2)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    denom = cos_d0 - eta * sin_d0
    ra = (wcs.crval1 + math.degrees(math.atan2(xi, denom))) % 360.0
    dec = math.degrees(math.atan2(sin_d0 + eta * cos_d0, math.hypot(xi, denom)))
    return ra, dec


def radec_to_pixels(wcs: WCSSolution, ra: float, dec: float) -> tuple[float, float]:
    """Convert (RA, Dec) to pixel coordinates using a fitted WCS.

    Exact inverse of pixels_to_radec for the same WCS.

    Parameters
    ----------
    wcs : WCSSolution
        Plate solution from fit_wcs or a pointing WCS.
    ra, dec : float
        Equatorial coordinates in degrees (J2000).

    Returns
    -------
    (x, y) : tuple[float, float]
        Pixel coordinates (0-indexed).  May be outside the frame bounds
        if the sky position is beyond the plate extent.

    Raises
    ------
    ValueError
        If the WCS CD matrix is degenerate.
    """
    # Forward gnomonic projection about (CRVAL1, CRVAL2).  Trig of ΔRA makes
    # this wrap-safe across the RA 0°/360° branch cut.
    dec0 = math.radians(wcs.crval2)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    d_ra = math.radians(ra - wcs.crval1)
    dec_r = math.radians(dec)
    sin_dec, cos_dec = math.sin(dec_r), math.cos(dec_r)
    cos_dra = math.cos(d_ra)
    # Clamp behind-horizon points so they land far outside any frame rather
    # than mirroring antipodal positions through the tangent point.
    cos_c = max(sin_d0 * sin_dec + cos_d0 * cos_dec * cos_dra, _COS_C_MIN)
    xi = math.degrees(cos_dec * math.sin(d_ra) / cos_c)
    eta = math.degrees((cos_d0 * sin_dec - sin_d0 * cos_dec * cos_dra) / cos_c)

    # Solve: [cd1_1  cd1_2][dx]   [xi ]
    #        [cd2_1  cd2_2][dy] = [eta]
    det = wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1
    if abs(det) < 1e-30:
        raise ValueError("Degenerate WCS CD matrix — cannot invert")
    dx = (wcs.cd2_2 * xi - wcs.cd1_2 * eta) / det
    dy = (wcs.cd1_1 * eta - wcs.cd2_1 * xi) / det
    # (dx, dy) is the ideal gnomonic pixel offset; re-apply the optic's
    # distortion so the result lands where the source is actually imaged.
    return _redistort(wcs, wcs.crpix1 + dx, wcs.crpix2 + dy)


def apply_rolling_shutter_correction(
    x: float,
    y: float,
    angular_velocity_x: float,
    angular_velocity_y: float,
    pixel_scale_arcsec: float,
    row_readout_us: float,
    reference_row: float,
) -> tuple[float, float]:
    """Apply rolling-shutter timing correction to a pixel centroid (T-03).

    The rolling shutter reads out row by row.  A moving target appears at a
    position displaced from its true geometric position because each row is
    exposed at a slightly different time.

    Correction formula (first-order, T-03):
        Δy_px = (row_k − row_center) × t_row_s × v_dec_px_s

    Parameters
    ----------
    x, y : float
        Pixel centroid (0-indexed).
    angular_velocity_x, angular_velocity_y : float
        Target angular velocity in arcsec/s (x = RA direction, y = Dec direction).
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel.
    row_readout_us : float
        Time per row readout in microseconds.
    reference_row : float
        Row with zero timing offset (typically frame center row).

    Returns
    -------
    (x_corr, y_corr) : tuple[float, float]
        Corrected pixel position.
    """
    t_row_s = row_readout_us * 1e-6
    v_x_px_s = angular_velocity_x / pixel_scale_arcsec
    v_y_px_s = angular_velocity_y / pixel_scale_arcsec

    delta_t = (y - reference_row) * t_row_s
    x_corr = x - v_x_px_s * delta_t
    y_corr = y - v_y_px_s * delta_t
    return x_corr, y_corr


# ---------------------------------------------------------------------------
# Batch astrometry
# ---------------------------------------------------------------------------


def astrometrise_detections(
    detections: list[Detection],
    wcs: WCSSolution,
    pixel_scale_arcsec: float | None = None,
) -> list[AstrometricDetection]:
    """Apply a WCS solution to a list of detections.

    Parameters
    ----------
    detections : list[Detection]
        Pixel-space detections from detect.detect_sources.
    wcs : WCSSolution
        Plate solution from fit_wcs.
    pixel_scale_arcsec : float | None
        Pixel scale (arcsec/px) for uncertainty propagation.  If None,
        derived from the CD matrix determinant.

    Returns
    -------
    list[AstrometricDetection]
        Detections with RA/Dec assigned.  Order mirrors the input list.
    """
    if pixel_scale_arcsec is None:
        cd_det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
        pixel_scale_arcsec = math.sqrt(cd_det) * 3600.0 if cd_det > 0 else 1.0

    # Positional uncertainty, PER AXIS (sigma_ra and sigma_dec are per-axis
    # quantities in I-02).  ``WCSSolution.rms_arcsec`` is the 2-D *radial*
    # residual RMS — rms² = σ_ξ² + σ_η² (see fit_wcs) — so the per-axis plate
    # term is rms²/2, not rms².  Reporting the radial RMS as each axis' sigma
    # inflated both — by √2 in the plate-dominated limit, less once the
    # centroiding floor is folded in.  The 0.3 px centroiding term is already
    # per-axis and is NOT halved.  Fixed here only: rms_arcsec keeps its
    # radial definition, which the accuracy_flag policy and ~15 pinned
    # thresholds depend on.
    centroiding_arcsec = 0.3 * pixel_scale_arcsec
    sigma_arcsec = math.sqrt(wcs.rms_arcsec**2 / 2.0 + centroiding_arcsec**2)

    results: list[AstrometricDetection] = []
    for det in detections:
        ra, dec = pixels_to_radec(wcs, det.x, det.y)
        results.append(
            AstrometricDetection(
                detection=det,
                ra_deg=ra,
                dec_deg=dec,
                sigma_ra_arcsec=sigma_arcsec,
                sigma_dec_arcsec=sigma_arcsec,
            )
        )
    return results
