"""Optical (radial lens) distortion for realistic plate-solve testing.

Wide, fast primes like the selected 7Artisans 25 mm f/0.95 image the sky with
several-percent geometric distortion toward the field edge.  A linear (TAN)
WCS cannot absorb that, so injecting it is the prerequisite for honestly
testing the catalog cross-match / plate-solve chain and the T-11 distortion
gate (post-calibration residual ≤ 3″ at the field edge).

Model
-----
Brown–Conrady **radial** distortion (Brown 1966; the OpenCV / photogrammetry
standard, equivalent in form to the FITS SIP radial term).  An ideal
(undistorted, gnomonic) pixel maps to the distorted pixel the optic actually
images:

    x_d = c_x + (x − c_x)·(1 + k1·r² + k2·r⁴)
    y_d = c_y + (y − c_y)·(1 + k1·r² + k2·r⁴)

where ``r`` is the radius from the distortion centre **normalised by the frame
half-diagonal**, so ``r = 1`` at the corner and ``k1, k2`` are dimensionless.
``k1 < 0`` is barrel distortion (typical of wide/fast lenses); ``k1 > 0`` is
pincushion.

Scope / simplifications (documented for rigor)
----------------------------------------------
* Radial only — tangential/decentering (Brown's p1, p2) is omitted; radial is
  the dominant term for a centred prime and is what a SIP/TPV solver must fit.
* Applied in the image plane to source positions, not via full optical ray
  tracing; chromatic and thermal variation are not modelled.
* The injected distortion is the *ground truth*; the pipeline's job is to
  recover it with a distortion-aware WCS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DistortionModel"]


@dataclass(frozen=True)
class DistortionModel:
    """Brown–Conrady radial distortion in normalised (half-diagonal) radius.

    Attributes
    ----------
    k1, k2 : float
        Dimensionless radial coefficients.  ``k1 < 0`` → barrel.
    center : tuple[float, float] | None
        Distortion centre (x, y) in pixels.  ``None`` → frame centre.
    """

    k1: float = 0.0
    k2: float = 0.0
    center: tuple[float, float] | None = None

    def _geometry(self, shape: tuple[int, int]) -> tuple[float, float, float]:
        """Return ``(cx, cy, r_halfdiag)`` for this frame shape."""
        h, w = shape
        if self.center is not None:
            cx, cy = self.center
        else:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        return cx, cy, math.hypot(w / 2.0, h / 2.0)

    def apply(self, x: float, y: float, shape: tuple[int, int]) -> tuple[float, float]:
        """Map an ideal pixel ``(x, y)`` to its distorted (as-imaged) position."""
        if self.k1 == 0.0 and self.k2 == 0.0:
            return x, y
        cx, cy, r_half = self._geometry(shape)
        dx, dy = x - cx, y - cy
        rn = math.hypot(dx, dy) / r_half
        f = 1.0 + self.k1 * rn**2 + self.k2 * rn**4
        return cx + dx * f, cy + dy * f

    def invert(
        self, x_d: float, y_d: float, shape: tuple[int, int], *, n_iter: int = 12
    ) -> tuple[float, float]:
        """Map a distorted pixel back to its ideal position (fixed-point solve).

        Inverse of :meth:`apply`; used to undistort measured centroids and for
        round-trip tests.  Converges for the small distortions modelled here.
        """
        if self.k1 == 0.0 and self.k2 == 0.0:
            return x_d, y_d
        cx, cy, r_half = self._geometry(shape)
        dxd, dyd = x_d - cx, y_d - cy
        rd = math.hypot(dxd, dyd)
        if rd == 0.0:
            return x_d, y_d
        ru = rd  # initial guess
        for _ in range(n_iter):
            rn = ru / r_half
            f = 1.0 + self.k1 * rn**2 + self.k2 * rn**4
            ru = rd / f
        scale = ru / rd
        return cx + dxd * scale, cy + dyd * scale

    @classmethod
    def from_corner_displacement_px(
        cls, displacement_px: float, shape: tuple[int, int]
    ) -> DistortionModel:
        """Build a pure-``k1`` model with a given signed corner displacement.

        ``displacement_px`` is how far a corner pixel moves radially (negative =
        inward = barrel).  At the corner ``r = 1`` so ``k1 = displacement /
        r_halfdiag``.
        """
        _, _, r_half = cls()._geometry(shape)
        return cls(k1=displacement_px / r_half)
