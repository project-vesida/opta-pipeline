"""Scene-domain pointing jitter for synthetic frames.

Wind buffeting and mount vibration move the *sky* across a detector that stays
exactly where it is.  The instrumental signature (bias pedestal + offset FPN,
flat field / PRNU + vignetting, dark-current map with hot pixels, dead pixels,
full-well saturation) is bolted to the silicon and must **not** move with the
scene — otherwise calibration leaves jitter-proportional residuals that no real
system produces.

:class:`PointingJitter` is therefore a *scene* model, not a post-processing
filter.  It supplies a per-frame rigid offset in pixels that
:mod:`opta_pipeline.synth.render` applies to source coordinates **before** any
detector stage runs, so:

* sources are drawn analytically at their shifted sub-pixel positions — no
  resampling, no interpolation blur of the PSF;
* the detector signature is applied afterwards, at fixed detector pixels;
* shot and read noise are generated *after* the shift, so the noise field is
  never resampled and its per-pixel variance is untouched.

Contrast with the legacy post-render corruptor
``opta_validation.corruption.pointing_jitter``, which translates an
already-rendered, already-noisy frame with bilinear ``scipy.ndimage.shift``:
that moves the detector signature along with the sky *and* spatially correlates
the noise field (attenuating its variance).  Use this module instead.

Temporal model
--------------
Jitter is **white per frame**: offsets are independent across frames, with no
temporal correlation.  Ratified 2026-07-27 — published jitter-robustness
numbers use this white per-frame model and are documented as such; a
temporally correlated AR(1)/PSD-shaped wind model is future work, not a
blocker.

Reproducibility
---------------
Offsets are drawn from ``N(0, rms_px)`` independently per axis, from a stream
derived from ``(seed, frame_index)``.  The draw therefore depends only on the
frame *index*, never on call order or on how many frames were rendered before
it — rendering frame 7 alone gives the same offset as rendering frames 0..7 in
sequence.  The stream is private: it never touches the per-frame noise RNG, so
turning jitter on does not perturb the noise realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PointingJitter"]


@dataclass(frozen=True)
class PointingJitter:
    """Per-frame rigid scene offset (wind / mount vibration).

    Attributes
    ----------
    rms_px : float
        Per-axis RMS of the Gaussian offset, in pixels.  ``0.0`` is an exact
        no-op (every offset is ``(0.0, 0.0)``).
    seed : int
        Explicit seed for the private offset stream.

    Examples
    --------
    >>> PointingJitter(rms_px=0.0, seed=1).offset_px(3)
    (0.0, 0.0)
    """

    rms_px: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a negative RMS (a scale parameter, not a signed offset)."""
        if self.rms_px < 0.0:
            raise ValueError("rms_px must be >= 0")

    def offset_px(self, frame_index: int) -> tuple[float, float]:
        """Return this frame's rigid scene offset ``(dx_px, dy_px)``.

        The offset is a pure function of ``(self.seed, frame_index)``; see the
        module docstring for why that matters.
        """
        if self.rms_px == 0.0:
            return 0.0, 0.0
        rng = np.random.default_rng([int(self.seed), int(frame_index)])
        dx, dy = rng.normal(0.0, float(self.rms_px), size=2)
        return float(dx), float(dy)
