"""Rolling-shutter readout injection for realistic plate-solve testing.

A CMOS rolling shutter does not expose the whole frame at once: it reads the
sensor row by row, so each row integrates over a slightly different time
window.  A source that *moves* across the field is therefore imaged at a
row-dependent displaced position — the along-track centroid bias the T-03
correction exists to remove.  Until now only the *correction*
(:func:`opta_pipeline.astrometry.apply_rolling_shutter_correction`) existed;
this is its forward (injection) counterpart so the readout step can be tested
closed-loop.

Model
-----
A source imaged at row ``y`` is sampled at a time offset
``Δt = (y − reference_row) · t_row`` relative to the reference epoch (the frame
centre row, zero offset).  Moving at pixel velocity ``(vx, vy)`` it is then
displaced by::

    x_d = x + vx · Δt
    y_d = y + vy · Δt

This is exactly inverted (to first order) by
``apply_rolling_shutter_correction``.

Scope / simplifications (documented for rigor)
----------------------------------------------
* First-order: ``Δt`` is evaluated at the source's nominal row; the row itself
  shifts by ``vy·Δt`` (a second-order, ``t_row²`` effect that is negligible for
  realistic readout times).
* **Stationary sources are unaffected** — a star on a fixed mount has zero
  velocity, so only sky-tracked targets shift.  This matches the physics of a
  staring array and is why the correction is keyed to the *target's* rate.
* The injected displacement is the ground truth the pipeline must recover; the
  geometric (un-readout) position is what is recorded as truth.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RollingShutterReadout"]


@dataclass(frozen=True)
class RollingShutterReadout:
    """Rolling-shutter readout timing (forward / injection model, T-03).

    Attributes
    ----------
    row_readout_us : float
        Time between successive row read-outs in microseconds.
    reference_row : float | None
        Row with zero timing offset (the reference epoch).  ``None`` → frame
        centre row.
    """

    row_readout_us: float
    reference_row: float | None = None

    def _ref(self, shape: tuple[int, int]) -> float:
        if self.reference_row is not None:
            return self.reference_row
        return (shape[0] - 1) / 2.0

    def displace(
        self,
        x: float,
        y: float,
        vx_px_s: float,
        vy_px_s: float,
        shape: tuple[int, int],
    ) -> tuple[float, float]:
        """Displace an as-imaged pixel by the rolling-shutter readout bias.

        Parameters
        ----------
        x, y : float
            Ideal (geometric) as-imaged pixel position.
        vx_px_s, vy_px_s : float
            Source pixel velocity (px/s) along the image axes.
        shape : tuple[int, int]
            ``(height, width)`` of the frame (for the default reference row).

        Returns
        -------
        (x_d, y_d) : tuple[float, float]
            Readout-displaced position; identity for a stationary source or
            zero readout time.
        """
        if self.row_readout_us == 0.0 or (vx_px_s == 0.0 and vy_px_s == 0.0):
            return x, y
        dt = (y - self._ref(shape)) * self.row_readout_us * 1e-6
        return x + vx_px_s * dt, y + vy_px_s * dt
