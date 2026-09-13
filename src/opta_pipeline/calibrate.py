"""Calibration module for OpTA pipeline.

Applies standard CCD calibration steps to raw frames:
  1. Master dark creation (sigma-clipped median stack)
  2. Master flat creation (sigma-clipped median stack, normalised)
  3. Dark subtraction
  4. Flat-field correction
  5. Mesh-based background estimation and subtraction

All calibration steps are applied in electrons (or ADU at unity gain).

Usage
-----
    from opta_pipeline.calibrate import (
        make_master_dark,
        make_master_flat,
        calibrate_frame,
    )

    dark = make_master_dark(dark_frames)
    flat = make_master_flat(flat_frames)
    cal  = calibrate_frame(raw, master_dark=dark, master_flat=flat)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import median_filter

__all__ = [
    "CalibratedFrame",
    "make_master_dark",
    "make_master_flat",
    "calibrate_frame",
    "estimate_background",
]

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratedFrame:
    """Output of calibrate_frame.

    Attributes
    ----------
    data : np.ndarray
        Calibrated science frame in electrons (float32).  Background has
        been subtracted if subtract_background=True was passed.
    background : np.ndarray
        Background model in electrons (float32, same shape as data).
        All zeros if background subtraction was not performed.
    background_rms : float
        Robust estimate of the per-pixel background noise (electrons).
        MAD-based (``astropy.stats.mad_std``) estimate of the
        background-subtracted residual.
    dark_subtracted : bool
    flat_corrected : bool
    background_subtracted : bool
    """

    data: np.ndarray = field(hash=False, compare=False)
    background: np.ndarray = field(hash=False, compare=False)
    background_rms: float
    dark_subtracted: bool
    flat_corrected: bool
    background_subtracted: bool


# ---------------------------------------------------------------------------
# Master frame creation
# ---------------------------------------------------------------------------


def _sigma_clip_median(
    stack: np.ndarray,
    sigma: float = 3.0,
    n_iter: int = 3,
) -> np.ndarray:
    """Compute a sigma-clipped median along axis 0 (the frame axis).

    Delegates to astropy.stats.sigma_clip so behaviour is consistent with
    astropy's implementation (WP-C1).  All-clipped pixels fall back to the
    unclipped median rather than NaN.

    Parameters
    ----------
    stack : np.ndarray
        Shape (n_frames, height, width).
    sigma : float
        Clipping threshold in units of the standard deviation.
    n_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    np.ndarray
        Master frame (height, width) float32.
    """
    arr = stack.astype(np.float32)
    clipped = sigma_clip(arr, sigma=sigma, maxiters=n_iter, axis=0)
    master = np.ma.median(clipped, axis=0)
    # Fall back to the unclipped median for fully-masked pixels.
    if np.ma.is_masked(master):
        fallback = np.nanmedian(arr, axis=0).astype(np.float32)
        fill_mask = np.ma.getmaskarray(master)
        result = np.where(fill_mask, fallback, np.ma.filled(master, 0.0))
    else:
        result = np.ma.filled(master, 0.0)
    return result.astype(np.float32)


def make_master_dark(
    dark_frames: list[np.ndarray],
    sigma: float = 3.0,
) -> np.ndarray:
    """Create a master dark frame from a list of raw dark exposures.

    Parameters
    ----------
    dark_frames : list[np.ndarray]
        Raw dark frames (same shape, exposure time, and gain).
        At least 3 frames required.
    sigma : float
        Sigma-clipping threshold.

    Returns
    -------
    np.ndarray
        Master dark frame (height, width) float32.

    Raises
    ------
    ValueError
        If fewer than 3 frames are provided.
    """
    if len(dark_frames) < 3:
        raise ValueError("At least 3 dark frames required for master dark")
    stack = np.stack([f.astype(np.float32) for f in dark_frames], axis=0)
    return _sigma_clip_median(stack, sigma=sigma)


def make_master_flat(
    flat_frames: list[np.ndarray],
    sigma: float = 3.0,
) -> np.ndarray:
    """Create a normalised master flat from a list of flat-field frames.

    The combined frame is normalised by its median so that pixels have
    values near 1.0 — the correction factor for each pixel.

    Parameters
    ----------
    flat_frames : list[np.ndarray]
        Raw flat-field frames (same shape).  At least 3 required.
    sigma : float
        Sigma-clipping threshold.

    Returns
    -------
    np.ndarray
        Normalised master flat (height, width) float32, median ≈ 1.0.
        Bad pixels (< 0.3 of median) are set to 1.0 (no correction).

    Raises
    ------
    ValueError
        If fewer than 3 frames are provided.
    """
    if len(flat_frames) < 3:
        raise ValueError("At least 3 flat frames required for master flat")
    stack = np.stack([f.astype(np.float32) for f in flat_frames], axis=0)
    combined = _sigma_clip_median(stack, sigma=sigma)
    med = float(np.median(combined))
    if med <= 0:
        raise ValueError("Master flat median is non-positive — check flat frames")
    normalised = combined / med
    # Protect against dead/hot pixels
    bad = (normalised < 0.3) | ~np.isfinite(normalised)
    normalised[bad] = 1.0
    return normalised


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------


def _box_edges(length: int, box_size: int) -> np.ndarray:
    """Ceil-partition ``[0, length)`` into boxes of *box_size* pixels.

    The trailing remainder (``length % box_size`` pixels) forms its own last
    box, so every pixel belongs to a sampled box — no strip of the frame is
    ever merely extrapolated.

    Returns the ``n_boxes + 1`` edge coordinates (last edge == *length*).
    """
    n_boxes = max(1, -(-length // box_size))  # ceil division
    edges = np.minimum(np.arange(n_boxes + 1) * box_size, length)
    return edges


def _interp_box_centres(
    bg_map: np.ndarray,
    centres: np.ndarray,
    length: int,
    axis: int,
) -> np.ndarray:
    """Linearly interpolate a mesh along *axis* from box centres to pixels.

    Box medians are statistics of the box *centre* — mapping them onto frame
    corners (as ``scipy.ndimage.zoom`` does) shifts the model by half a box.
    Between the outermost centres this is plain bilinear interpolation
    (SExtractor / photutils style); in the outer half-box margins the local
    gradient is extrapolated linearly, so a linear background (twilight
    gradient, smooth vignetting) is reproduced exactly out to the frame edge.
    """
    n = bg_map.shape[axis]
    pix = np.arange(length, dtype=np.float64)
    if n == 1:
        idx0 = np.zeros(length, dtype=np.intp)
        idx1 = idx0
        t = np.zeros(length, dtype=np.float64)
    else:
        # Interval index such that centres[i0] <= p < centres[i1], clipped to
        # the outermost interval; t runs outside [0, 1] there (extrapolation).
        idx1 = np.clip(np.searchsorted(centres, pix), 1, n - 1)
        idx0 = idx1 - 1
        t = (pix - centres[idx0]) / (centres[idx1] - centres[idx0])
    lo = np.take(bg_map, idx0, axis=axis)
    hi = np.take(bg_map, idx1, axis=axis)
    shape = [1, 1]
    shape[axis] = length
    t = t.reshape(shape)
    return lo + t * (hi - lo)


def estimate_background(
    frame: np.ndarray,
    box_size: int = 64,
    filter_size: int = 3,
) -> tuple[np.ndarray, float]:
    """Estimate a smooth 2-D background model from a science frame.

    Algorithm (SExtractor / photutils ``Background2D`` style):
      1. Ceil-partition the *full* frame into boxes of *box_size* pixels; the
         trailing ``h % box_size`` / ``w % box_size`` remainder strips form
         their own (smaller) last boxes, so every pixel is sampled.
      2. Compute the sigma-clipped median of each box (robust against
         sources).
      3. Median-filter the low-resolution map over *filter_size* boxes
         (separably, ``mode="nearest"``) to reject boxes contaminated by
         bright sources.  A separable median filter — unlike a mean filter —
         preserves linear gradients exactly, including at the map borders
         and corners.
      4. Interpolate box-centre values bilinearly onto pixel coordinates,
         extrapolating the local gradient linearly in the outer half-box
         margins.  (Corner-anchored ``zoom(order=1)`` would shift the whole
         model by half a box.)

    Parameters
    ----------
    frame : np.ndarray
        Input science frame (float).
    box_size : int
        Side length of each background estimation box in pixels.
    filter_size : int
        Width of the median smoothing filter in units of boxes.

    Returns
    -------
    background : np.ndarray
        Smooth background model, same shape as *frame*.
    background_rms : float
        MAD-based robust noise estimate of ``frame - background`` via
        ``astropy.stats.mad_std`` (consistency factor 1.4826 included).
        No sigma-clip bias correction is applied because no sigma clipping
        is involved: ``mad_std`` depends only on the central 50 % of the
        residual distribution and is unbiased on continuous Gaussian noise
        (measured +0.2 %, from the background model's fitted degrees of
        freedom).  Beware: on *integer-quantised* low-count data the MAD is
        biased by discreteness (measured −11 % at Poisson λ=25, +33 % at
        λ=5), which no clip-consistency factor fixes — see TODO.md
        (u* under-coverage item).
    """
    h, w = frame.shape
    edges_y = _box_edges(h, box_size)
    edges_x = _box_edges(w, box_size)
    n_boxes_y = len(edges_y) - 1
    n_boxes_x = len(edges_x) - 1

    bg_map = np.zeros((n_boxes_y, n_boxes_x), dtype=np.float64)

    for by in range(n_boxes_y):
        for bx in range(n_boxes_x):
            y0, y1 = edges_y[by], edges_y[by + 1]
            x0, x1 = edges_x[bx], edges_x[bx + 1]
            box = frame[y0:y1, x0:x1].astype(np.float64)
            # Sigma-clip: one pass
            med = float(np.median(box))
            std = float(np.std(box))
            if std < 1e-6:
                std = 1.0
            clipped = box[np.abs(box - med) < 3.0 * std]
            bg_map[by, bx] = float(np.median(clipped)) if len(clipped) > 0 else med

    # Reject source-contaminated boxes.  Median (not uniform/mean) filter:
    # a mean filter biases the outermost boxes of any gradient by ~step/3
    # regardless of boundary mode, while the median of a monotone ramp is
    # its centre value.  Applied separably (rows, then columns) because a
    # full 2-D median window biases the *corner* boxes of a diagonal
    # gradient by one box-step under any boundary mode; the separable
    # version preserves every bilinear plane a + s·y + t·x exactly while
    # still rejecting isolated outlier boxes.
    if filter_size > 1:
        bg_map = median_filter(bg_map, size=(filter_size, 1), mode="nearest")
        bg_map = median_filter(bg_map, size=(1, filter_size), mode="nearest")

    # Box-centre pixel coordinates: box covering [y0, y1) has its median
    # located at (y0 + y1 - 1) / 2 in pixel-centre coordinates.
    centres_y = (edges_y[:-1] + edges_y[1:] - 1) / 2.0
    centres_x = (edges_x[:-1] + edges_x[1:] - 1) / 2.0

    background = _interp_box_centres(bg_map, centres_y, h, axis=0)
    background = _interp_box_centres(background, centres_x, w, axis=1)
    background = background.astype(np.float32)

    # Background RMS: MAD-based robust estimator via astropy.stats.mad_std
    residual = frame.astype(np.float64) - background
    background_rms = float(mad_std(residual))

    return background, background_rms


# ---------------------------------------------------------------------------
# Full calibration pipeline
# ---------------------------------------------------------------------------


def calibrate_frame(
    raw: np.ndarray,
    master_dark: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
    subtract_background: bool = True,
    background_box_size: int = 64,
    background_filter_size: int = 3,
) -> CalibratedFrame:
    """Apply standard CCD calibration steps to a raw frame.

    Steps applied (each optional):
      1. Dark subtraction: ``science − dark``
      2. Flat correction: ``science / flat``
      3. Background estimation and subtraction (mesh-based)

    Parameters
    ----------
    raw : np.ndarray
        Raw science frame (integer or float ADU).
    master_dark : np.ndarray | None
        Master dark frame (same shape as raw).  If None, skipped.
    master_flat : np.ndarray | None
        Normalised master flat frame.  If None, skipped.
    subtract_background : bool
        Whether to estimate and subtract the residual background.  When False,
        ``background_rms`` is NaN (no noise estimate is produced).
    background_box_size : int
        Box size for background mesh estimation (pixels).
    background_filter_size : int
        Smoothing filter size in boxes.

    Returns
    -------
    CalibratedFrame
        Calibrated frame with metadata.
    """
    cal = raw.astype(np.float32)
    dark_sub = False
    flat_corr = False

    if master_dark is not None:
        cal -= master_dark.astype(np.float32)
        dark_sub = True

    if master_flat is not None:
        flat = master_flat.astype(np.float32)
        nonzero = flat > 0.1
        cal[nonzero] /= flat[nonzero]
        flat_corr = True

    if subtract_background:
        background, bg_rms = estimate_background(
            cal,
            background_box_size,
            background_filter_size,
        )
        cal -= background
        bg_sub = True
    else:
        background = np.zeros_like(cal)
        bg_rms = float("nan")
        bg_sub = False

    return CalibratedFrame(
        data=cal,
        background=background,
        background_rms=float(bg_rms),
        dark_subtracted=dark_sub,
        flat_corrected=flat_corr,
        background_subtracted=bg_sub,
    )
