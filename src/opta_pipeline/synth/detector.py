"""Frame-stable detector signature for realistic synthetic frames (Phase 1).

A :class:`DetectorModel` bundles the *instrumental* signatures that a real
CMOS/CCD sensor imprints on every frame and that the calibration stage
(:mod:`opta_pipeline.calibrate`) is designed to remove:

============================  ====================================  =================
Signature                     Physical origin                       Removed by
============================  ====================================  =================
bias pedestal + offset FPN    readout electronics / sense node      master dark
dark-current map + hot pix    thermal generation, defects           master dark
flat field (PRNU × vignette)  pixel gain spread + cos⁴ falloff       master flat
dead pixels                   manufacturing defects                 master flat
full-well + nonlinearity      finite charge capacity, A/D response  (not removable)
============================  ====================================  =================

Crucially the maps are **frame-stable**: a model built once is reused across
every science, dark, and flat frame of a sequence, so master-dark/-flat
calibration can actually recover the structure.  This is what makes the
synthetic frames exercise the full ``calibrate → detect → astrometry`` chain
rather than the trivial clean-frame path.

The decomposition follows the canonical CCD/CMOS noise chain documented in
SatSim (Cabello & Fletcher 2022), the Astropy CCD reduction guide, and Konnik
& Welsh's photosensor simulation tutorial.

Design intent
-------------
This object is a *passive* data holder plus a deterministic factory.  All
application logic lives in :mod:`opta_pipeline.synth.render` so the rendering
pipeline stays decoupled and extensible (add a stage, not a method).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from opta_model.hardware import SensorConfig

__all__ = [
    "DetectorModel",
    "build_detector_model",
]


@dataclass(frozen=True)
class DetectorModel:
    """Frame-stable instrumental signature for one sensor.

    All maps share the sensor shape ``(height, width)`` and are stored as
    float32.  Electron-domain quantities are in electrons; the flat field is
    a dimensionless multiplicative response centred on 1.0.

    Attributes
    ----------
    shape : tuple[int, int]
        ``(height, width)`` in pixels.
    bias_offset_e : float
        Uniform electronic pedestal added to every pixel (electrons).
    bias_fpn_e : np.ndarray
        Zero-mean fixed-pattern offset non-uniformity (electrons).
    flat : np.ndarray
        Photometric response = PRNU × vignetting × (dead-pixel mask).
        Multiplies the collected photo-signal (sky + sources); mean ≈ 1.0.
    dark_current_e_s : np.ndarray
        Per-pixel dark-current rate (e⁻/s), including hot pixels.
    full_well_e : float
        Charge capacity per pixel; signal saturates here (electrons).
    nonlinearity : float
        Fractional A/D nonlinearity coefficient (0 = perfectly linear).
        Models the mild compressive response near full well.
    gain_e_adu : float
        Sensor gain in electrons per ADU used at digitisation.
    """

    shape: tuple[int, int]
    bias_offset_e: float
    bias_fpn_e: np.ndarray = field(hash=False, compare=False)
    flat: np.ndarray = field(hash=False, compare=False)
    dark_current_e_s: np.ndarray = field(hash=False, compare=False)
    full_well_e: float
    nonlinearity: float
    gain_e_adu: float


def build_detector_model(
    sensor: SensorConfig,
    *,
    seed: int = 0,
    prnu_pct: float = 1.0,
    vignetting_corner_factor: float = 0.85,
    bias_offset_e: float = 100.0,
    bias_fpn_rms_e: float = 2.0,
    hot_pixel_fraction: float = 1e-5,
    hot_pixel_dark_e_s: float = 50.0,
    dead_pixel_fraction: float = 1e-6,
    nonlinearity: float = 0.0,
    gain_e_adu: float = 1.0,
) -> DetectorModel:
    """Build a deterministic, frame-stable :class:`DetectorModel`.

    Every map is drawn from a private RNG seeded by *seed*, independent of the
    per-frame noise stream, so the same model reused across a sequence yields
    identical fixed patterns each frame (the precondition for master-frame
    calibration and for static-defect decorrelation under shift-and-add).

    Parameters
    ----------
    sensor : SensorConfig
        Supplies the frame shape, baseline dark current, and full-well depth.
    seed : int
        Seed for the fixed-pattern RNG.
    prnu_pct : float
        Photo-response non-uniformity, percent RMS (Gaussian about 1.0).
        Typical CMOS values are 0.5–2 %.
    vignetting_corner_factor : float
        Fraction of on-axis illumination retained at the frame corner.
        1.0 disables vignetting; 0.85 ≈ 0.18 mag corner loss.  A radial
        profile normalised to 1.0 at the centre is used (cos⁴-like).
    bias_offset_e : float
        Uniform bias pedestal (electrons).
    bias_fpn_rms_e : float
        RMS of the zero-mean offset fixed-pattern noise (electrons).
    hot_pixel_fraction : float
        Fraction of pixels with anomalously high dark current.
    hot_pixel_dark_e_s : float
        Dark-current rate assigned to hot pixels (e⁻/s).
    dead_pixel_fraction : float
        Fraction of pixels with near-zero response (flat → ~0).
    nonlinearity : float
        Fractional A/D nonlinearity coefficient (see :class:`DetectorModel`).
    gain_e_adu : float
        Sensor gain in electrons per ADU.

    Returns
    -------
    DetectorModel
        Frame-stable instrumental signature for the sensor.
    """
    if not 0.0 < vignetting_corner_factor <= 1.0:
        raise ValueError("vignetting_corner_factor must be in (0, 1]")

    rng = np.random.default_rng(seed)
    h, w = sensor.resolution_v, sensor.resolution_h
    shape = (h, w)

    # --- Photo-response non-uniformity (PRNU): per-pixel gain spread ---
    prnu = rng.normal(1.0, prnu_pct / 100.0, size=shape).astype(np.float32)

    # --- Vignetting: smooth radial falloff normalised to 1.0 at centre ---
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    r_corner = math.hypot(cy, cx)
    r_norm = (r / r_corner) if r_corner > 0 else np.zeros_like(r)
    # cos⁴-like profile: choose theta_max so cos⁴(theta_max) == corner factor.
    theta_max = math.acos(vignetting_corner_factor ** 0.25)
    vignette = np.cos(r_norm * theta_max) ** 4
    flat = (prnu * vignette).astype(np.float32)

    # --- Dead pixels: response collapses to ~0 (master flat will mask them) ---
    if dead_pixel_fraction > 0:
        n_dead = int(round(h * w * dead_pixel_fraction))
        if n_dead > 0:
            dy = rng.integers(0, h, n_dead)
            dx = rng.integers(0, w, n_dead)
            flat[dy, dx] = 1e-3

    # --- Offset fixed-pattern noise (bias structure) ---
    bias_fpn = rng.normal(0.0, bias_fpn_rms_e, size=shape).astype(np.float32)

    # --- Dark-current map with hot pixels ---
    dark_map = np.full(shape, float(sensor.dark_current_e_s), dtype=np.float32)
    if hot_pixel_fraction > 0:
        n_hot = int(round(h * w * hot_pixel_fraction))
        if n_hot > 0:
            hy = rng.integers(0, h, n_hot)
            hx = rng.integers(0, w, n_hot)
            # Spread hot-pixel rates over an order of magnitude for realism.
            dark_map[hy, hx] = hot_pixel_dark_e_s * rng.uniform(0.3, 1.0, n_hot)

    return DetectorModel(
        shape=shape,
        bias_offset_e=float(bias_offset_e),
        bias_fpn_e=bias_fpn,
        flat=flat,
        dark_current_e_s=dark_map,
        full_well_e=float(sensor.full_well_e),
        nonlinearity=float(nonlinearity),
        gain_e_adu=float(gain_e_adu),
    )
