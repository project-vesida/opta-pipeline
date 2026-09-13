"""Source detection module for OpTA pipeline.

Performs threshold-based source extraction on a calibrated frame:
  1. Threshold: pixels > background_rms × snr_threshold
  2. Connected-component labelling (scipy.ndimage.label)
  3. Per-component centroid and second-moment ellipse fitting
  4. SNR, elongation, and FWHM computation
  5. Size filtering and streak/point classification

Output: list of Detection objects with pixel centroid, SNR, elongation,
and a flag indicating whether the source looks like a streak (satellite)
or a compact point source (star).

Usage
-----
    from opta_pipeline.detect import detect_sources

    detections = detect_sources(
        cal.data,
        noise_rms=max(cal.background_rms, 1.0),
        snr_threshold=3.0,
    )
    streaks = [d for d in detections if d.is_streak]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import label

__all__ = [
    "Detection",
    "detect_sources",
]


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single detected source (star or satellite streak).

    Attributes
    ----------
    x : float
        Flux-weighted centroid x (pixel, 0-indexed).
    y : float
        Flux-weighted centroid y (pixel, 0-indexed).
    snr : float
        Signal-to-noise ratio (integrated signal / sqrt(n_pixels) / noise_rms).
    elongation : float
        Ratio of semi-major to semi-minor axis (a/b ≥ 1).
        High value (> elongation_threshold) indicates a streak.
    angle_deg : float
        Orientation of the semi-major axis in degrees, measured CCW from
        the positive x-axis.
    n_pixels : int
        Number of connected above-threshold pixels.
    flux_e : float
        Integrated pixel values within the component (above-background electrons).
    is_streak : bool
        True if elongation > elongation_threshold at detection time.
    fwhm_px : float
        Equivalent circular FWHM in pixels (geometric mean of semi-axes × 2.355).
    """

    x: float
    y: float
    snr: float
    elongation: float
    angle_deg: float
    n_pixels: int
    flux_e: float
    is_streak: bool
    fwhm_px: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ellipse_from_moments(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute ellipse parameters from second moments.

    Returns (x_cen, y_cen, a, b, angle_deg) where a ≥ b are the
    semi-axes and angle_deg is the CCW angle from +x to the semi-major axis.
    """
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return 0.0, 0.0, 1.0, 1.0, 0.0

    x_cen = float((weights * xs).sum() / w_sum)
    y_cen = float((weights * ys).sum() / w_sum)

    dx = xs - x_cen
    dy = ys - y_cen

    m_xx = float((weights * dx * dx).sum() / w_sum)
    m_yy = float((weights * dy * dy).sum() / w_sum)
    m_xy = float((weights * dx * dy).sum() / w_sum)

    # Eigenvalues of the moment matrix
    trace = m_xx + m_yy
    det = m_xx * m_yy - m_xy * m_xy
    discriminant = max(0.0, (trace / 2.0) ** 2 - det)
    sqrt_disc = math.sqrt(discriminant)

    lam_max = trace / 2.0 + sqrt_disc
    lam_min = trace / 2.0 - sqrt_disc

    a = math.sqrt(max(lam_max, 1e-9))  # semi-major
    b = math.sqrt(max(lam_min, 1e-9))  # semi-minor

    # Angle of major axis
    angle_rad = math.atan2(2.0 * m_xy, m_xx - m_yy) / 2.0
    angle_deg = math.degrees(angle_rad)

    return x_cen, y_cen, a, b, angle_deg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_sources(
    frame: np.ndarray,
    noise_rms: float,
    snr_threshold: float = 3.0,
    min_pixels: int = 5,
    max_pixels: int = 500,
    elongation_threshold: float = 2.0,
    star_mask: np.ndarray | None = None,
) -> list[Detection]:
    """Extract sources from a calibrated (background-subtracted) frame.

    Parameters
    ----------
    frame : np.ndarray
        Calibrated science frame, background subtracted (float).  Pixel
        values represent above-background signal in electrons.  Also
        accepts a :attr:`~opta_pipeline.stack.StackResult.stacked` array;
        pass the matching ``StackResult.noise_rms`` as ``noise_rms``.
    noise_rms : float
        Per-pixel noise estimate in electrons.  For single frames use
        ``max(CalibratedFrame.background_rms, 1.0)``; for a stacked frame
        use ``StackResult.noise_rms`` (= single-frame noise × √N).
    snr_threshold : float
        Minimum per-pixel SNR to include a pixel in the detection mask.
    min_pixels : int
        Minimum number of connected pixels for a valid detection.
    max_pixels : int
        Maximum connected pixels; larger components are rejected.
    elongation_threshold : float
        Semi-major / semi-minor axis ratio above which a source is
        classified as a streak rather than a point source.
    star_mask : np.ndarray | None
        Boolean mask (same shape as frame); True pixels are excluded from
        detection (use to suppress regions around bright catalog stars).

    Returns
    -------
    list[Detection]
        All detections passing size and SNR filters, sorted by SNR descending.
    """
    if noise_rms <= 0:
        raise ValueError("noise_rms must be > 0")

    # --- Detection threshold mask ---
    threshold = noise_rms * snr_threshold
    det_mask = frame > threshold

    if star_mask is not None:
        det_mask &= ~star_mask

    if not det_mask.any():
        return []

    # --- Connected-component labelling ---
    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    labeled, n_labels = cast(
        tuple[np.ndarray, int], label(det_mask, structure=structure)
    )
    if n_labels == 0:
        return []

    # --- Extract detections ---
    yy, xx = np.mgrid[: frame.shape[0], : frame.shape[1]]
    detections: list[Detection] = []

    for comp_id in range(1, n_labels + 1):
        mask = labeled == comp_id
        n_pix = int(mask.sum())

        # Size filter
        if n_pix < min_pixels or n_pix > max_pixels:
            continue

        xs = xx[mask].astype(np.float64)
        ys = yy[mask].astype(np.float64)
        vals = frame[mask].astype(np.float64)

        # Use only positive values as weights (avoid negative bias)
        weights = np.maximum(vals, 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(vals)

        # Moments → ellipse
        x_cen, y_cen, a, b, angle_deg = _ellipse_from_moments(xs, ys, weights)
        elongation = a / max(b, 1e-6)
        fwhm_px = 2.355 * math.sqrt(a * b)  # geometric mean FWHM

        # SNR: total signal / noise in extraction area
        flux_e = float(vals.sum())
        snr = flux_e / (math.sqrt(n_pix) * noise_rms) if noise_rms > 0 else 0.0

        is_streak = elongation > elongation_threshold

        detections.append(
            Detection(
                x=x_cen,
                y=y_cen,
                snr=snr,
                elongation=elongation,
                angle_deg=angle_deg,
                n_pixels=n_pix,
                flux_e=flux_e,
                is_streak=is_streak,
                fwhm_px=fwhm_px,
            )
        )

    # Sort by SNR descending
    detections.sort(key=lambda d: d.snr, reverse=True)
    return detections
