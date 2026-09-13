"""Point-spread-function model and sub-pixel-accurate rendering (Phase 2).

Realism axes, in order of how much they matter for tuning the pipeline's
centroiding / plate-solving / photometry:

* **Sub-pixel accuracy.**  Pixel values come from integrating the profile over
  the pixel area, not point-sampling it, so total flux is conserved and the
  flux-weighted centroid is unbiased at any sub-pixel phase.  For the common
  isotropic-Gaussian case this uses the exact ``erf`` form; the general
  (Moffat / elliptical) case uses supersample-and-bin.

* **Field-variable size (defocus / field curvature).**  The FWHM grows radially
  from ``fwhm_center_px`` on-axis to ``fwhm_edge_px`` at the frame corner.

* **Profile shape.**  ``profile="moffat"`` uses the Moffat function
  ``I(r) ∝ (1 + (r/α)²)^(−β)`` — the de-facto standard atmospheric+optical PSF
  in source-extraction tools (SExtractor, PSFEx) because its power-law wings
  match real stars far better than a Gaussian's.  ``profile="gaussian"`` keeps
  the Gaussian.

* **Anisotropy (coma / astigmatism).**  Off-axis aberrations stretch the PSF; to
  first order the elongation points *radially* (away from the optical axis) and
  grows with field radius.  ``ellipticity_center``/``ellipticity_edge`` set the
  ellipticity ``e = 1 − b/a``; the major axis is oriented radially.

The pixel convention matches the rest of the pipeline: integer coordinate ``i``
denotes the pixel spanning ``[i−0.5, i+0.5]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

__all__ = [
    "PSFModel",
    "render_gaussian",
    "render_source",
    "render_streak",
    "_FWHM_TO_SIGMA",
]

# FWHM = 2·√(2·ln2)·σ  (Gaussian)
_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_SQRT2: float = math.sqrt(2.0)


@dataclass(frozen=True)
class PSFModel:
    """Field-dependent PSF (size, profile shape, and radial anisotropy).

    At field radius ``r`` (normalised so the frame corner is 1.0) the FWHM and
    ellipticity interpolate from their on-axis to their corner values as
    ``value(r) = center + (edge − center) · r**radial_power``.  The major axis is
    oriented radially (coma-like).

    Attributes
    ----------
    fwhm_center_px, fwhm_edge_px : float
        FWHM on-axis and at the frame corner (pixels).  Equal → uniform size.
    radial_power : float
        Exponent of the radial growth for both size and ellipticity.
    profile : {"gaussian", "moffat"}
        Radial profile shape.
    moffat_beta : float
        Moffat power-law index β (typical seeing ≈ 2.5–4.7).  Lower β → heavier
        wings.  Ignored for the Gaussian profile.
    ellipticity_center, ellipticity_edge : float
        Ellipticity ``e = 1 − b/a`` on-axis and at the corner, in [0, 1).
        0 → round.  The geometric-mean FWHM is preserved as the PSF elongates.
    oversample : int
        Supersampling factor used for the Moffat / elliptical render path.
    """

    fwhm_center_px: float = 2.0
    fwhm_edge_px: float = 2.0
    radial_power: float = 2.0
    profile: str = "gaussian"
    moffat_beta: float = 3.5
    ellipticity_center: float = 0.0
    ellipticity_edge: float = 0.0
    oversample: int = 5

    def _radial_frac(self, x: float, y: float, shape: tuple[int, int]) -> float:
        """Normalised field radius (0 at centre, 1 at corner), raised to power."""
        h, w = shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r_corner = math.hypot(cx, cy)
        if r_corner <= 0:
            return 0.0
        return (math.hypot(x - cx, y - cy) / r_corner) ** self.radial_power

    def fwhm_at(self, x: float, y: float, shape: tuple[int, int]) -> float:
        """Effective FWHM (pixels) at pixel position ``(x, y)``."""
        if self.fwhm_edge_px == self.fwhm_center_px:
            return self.fwhm_center_px
        frac = self._radial_frac(x, y, shape)
        return self.fwhm_center_px + (self.fwhm_edge_px - self.fwhm_center_px) * frac

    def ellipticity_at(self, x: float, y: float, shape: tuple[int, int]) -> float:
        """Ellipticity ``e = 1 − b/a`` at ``(x, y)``."""
        if self.ellipticity_edge == self.ellipticity_center:
            return self.ellipticity_center
        frac = self._radial_frac(x, y, shape)
        return (
            self.ellipticity_center
            + (self.ellipticity_edge - self.ellipticity_center) * frac
        )

    def orientation_at(self, x: float, y: float, shape: tuple[int, int]) -> float:
        """Major-axis angle (radians): radial, i.e. pointing away from centre."""
        h, w = shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        return math.atan2(y - cy, x - cx)

    def params_at(
        self, x: float, y: float, shape: tuple[int, int]
    ) -> tuple[float, float, float]:
        """Return ``(fwhm, ellipticity, theta)`` at ``(x, y)``."""
        return (
            self.fwhm_at(x, y, shape),
            self.ellipticity_at(x, y, shape),
            self.orientation_at(x, y, shape),
        )


def render_gaussian(
    frame: np.ndarray,
    x: float,
    y: float,
    total_signal_e: float,
    fwhm_px: float,
    *,
    n_sigma: float = 4.0,
) -> None:
    """Add a flux-conserving, sub-pixel-accurate **isotropic Gaussian** in-place.

    Each pixel receives the exact integral of the Gaussian over its area (a
    product of ``erf`` differences), so the deposited flux ≈ ``total_signal_e``
    (minus the < 0.01 % beyond ``n_sigma``) and the centroid equals ``(x, y)``
    to machine precision.  This is the fast path for the round-Gaussian case;
    :func:`render_source` handles Moffat / elliptical PSFs.
    """
    sigma = fwhm_px * _FWHM_TO_SIGMA
    if sigma <= 0.0 or total_signal_e == 0.0:
        return
    h, w = frame.shape
    radius = int(math.ceil(n_sigma * sigma)) + 1

    xi_min = max(0, int(math.floor(x)) - radius)
    xi_max = min(w, int(math.ceil(x)) + radius + 1)
    yi_min = max(0, int(math.floor(y)) - radius)
    yi_max = min(h, int(math.ceil(y)) + radius + 1)
    if xi_max <= xi_min or yi_max <= yi_min:
        return

    xs = np.arange(xi_min, xi_max)
    ys = np.arange(yi_min, yi_max)
    denom = sigma * _SQRT2
    fx = 0.5 * (erf((xs + 0.5 - x) / denom) - erf((xs - 0.5 - x) / denom))
    fy = 0.5 * (erf((ys + 0.5 - y) / denom) - erf((ys - 0.5 - y) / denom))
    frame[yi_min:yi_max, xi_min:xi_max] += total_signal_e * np.outer(fy, fx)


def render_source(
    frame: np.ndarray,
    x: float,
    y: float,
    total_signal_e: float,
    *,
    fwhm_px: float,
    profile: str = "gaussian",
    moffat_beta: float = 3.5,
    ellipticity: float = 0.0,
    theta: float = 0.0,
    oversample: int = 5,
) -> None:
    """Add a flux-conserving PSF of arbitrary profile/shape in-place.

    Delegates to the exact :func:`render_gaussian` for the isotropic-Gaussian
    case; otherwise supersamples the (possibly elliptical) Gaussian or Moffat
    profile on an ``oversample × oversample`` sub-pixel grid and bins down.  The
    profile is normalised by its analytic plane integral, so the in-frame flux
    is conserved (truncated wings are physically dropped).

    Parameters
    ----------
    fwhm_px : float
        Geometric-mean FWHM in pixels (preserved as the PSF elongates).
    profile : {"gaussian", "moffat"}
    moffat_beta : float
        Moffat β (ignored for Gaussian).
    ellipticity : float
        ``e = 1 − b/a`` in [0, 1).  0 → round.
    theta : float
        Major-axis angle in radians.
    oversample : int
        Sub-pixel grid factor.
    """
    if fwhm_px <= 0.0 or total_signal_e == 0.0:
        return
    if profile == "gaussian" and ellipticity < 1e-9:
        render_gaussian(frame, x, y, total_signal_e, fwhm_px)
        return

    q = max(1e-3, 1.0 - ellipticity)  # axis ratio b/a
    fwhm_a = fwhm_px / math.sqrt(q)  # major-axis FWHM (geo-mean preserved)
    fwhm_b = fwhm_px * math.sqrt(q)  # minor-axis FWHM

    h, w = frame.shape
    # Moffat wings are heavier than Gaussian → use a wider stamp.
    k = 3.0 if profile == "gaussian" else 7.0
    radius = int(math.ceil(k * fwhm_a)) + 1
    x_lo = max(0, int(math.floor(x)) - radius)
    x_hi = min(w, int(math.ceil(x)) + radius + 1)
    y_lo = max(0, int(math.floor(y)) - radius)
    y_hi = min(h, int(math.ceil(y)) + radius + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return

    osf = max(1, int(oversample))
    nx, ny = x_hi - x_lo, y_hi - y_lo
    # Sub-pixel sample centres.
    sx = x_lo - 0.5 + (np.arange(nx * osf) + 0.5) / osf
    sy = y_lo - 0.5 + (np.arange(ny * osf) + 0.5) / osf
    dx = sx[None, :] - x
    dy = sy[:, None] - y
    ct, st = math.cos(theta), math.sin(theta)
    xr = dx * ct + dy * st  # along major axis
    yr = -dx * st + dy * ct  # along minor axis

    if profile == "moffat":
        denom = 2.0 * math.sqrt(2.0 ** (1.0 / moffat_beta) - 1.0)
        alpha_a, alpha_b = fwhm_a / denom, fwhm_b / denom
        prof = (1.0 + (xr / alpha_a) ** 2 + (yr / alpha_b) ** 2) ** (-moffat_beta)
        plane_integral = math.pi * alpha_a * alpha_b / (moffat_beta - 1.0)
    else:  # gaussian (elliptical)
        sa, sb = fwhm_a * _FWHM_TO_SIGMA, fwhm_b * _FWHM_TO_SIGMA
        prof = np.exp(-0.5 * ((xr / sa) ** 2 + (yr / sb) ** 2))
        plane_integral = 2.0 * math.pi * sa * sb

    # Midpoint-rule integral per sub-pixel (area = 1/osf²), normalised to total.
    contrib = prof * (total_signal_e / (plane_integral * osf * osf))
    binned = contrib.reshape(ny, osf, nx, osf).sum(axis=(1, 3))
    frame[y_lo:y_hi, x_lo:x_hi] += binned


def render_streak(
    frame: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    total_signal_e: float,
    psf: PSFModel,
    *,
    samples_per_px: float = 2.0,
) -> None:
    """Add a flux-conserving streak (line ⊛ field-variable PSF) in-place.

    The line is sampled at ``samples_per_px`` points per pixel of length; each
    sample deposits a PSF whose size/shape/orientation are evaluated from *psf*
    at that point, so the cross-section broadens and rotates correctly as the
    streak crosses aberrated field regions.  Signal is divided evenly across
    samples, conserving the total.
    """
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        fwhm, ell, theta = psf.params_at(x0, y0, frame.shape)
        render_source(
            frame, x0, y0, total_signal_e, fwhm_px=fwhm, profile=psf.profile,
            moffat_beta=psf.moffat_beta, ellipticity=ell, theta=theta,
            oversample=psf.oversample,
        )
        return

    n_samples = max(int(length * samples_per_px) + 1, 2)
    per_sample = total_signal_e / n_samples
    for t in np.linspace(0.0, 1.0, n_samples):
        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)
        fwhm, ell, theta = psf.params_at(xs, ys, frame.shape)
        render_source(
            frame, xs, ys, per_sample, fwhm_px=fwhm, profile=psf.profile,
            moffat_beta=psf.moffat_beta, ellipticity=ell, theta=theta,
            oversample=psf.oversample,
        )
