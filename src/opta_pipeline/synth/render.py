"""Composable effects-chain renderer for realistic synthetic frames (Phase 1).

This module turns the flat ``generate_frame`` routine into a *decoupled,
extensible* pipeline.  A frame is built by running an ordered list of
**stages** over a mutable :class:`RenderContext`.  Each stage is a plain
callable ``(RenderContext) -> None`` that mutates the working electron image
and/or records ground truth.  The default ordering follows the canonical
CCD/CMOS signal chain (SatSim, Astropy CCD guide):

    pointing jitter → sky → stars → satellites → flat-field → dark
        → shot noise → read noise → bias → saturation

The leading *scene* stage (:func:`stage_pointing_jitter`) is the reason the
chain is split this way: everything upstream of ``flat-field`` describes the
sky, everything downstream describes the silicon.  A pointing offset therefore
moves the sources and leaves the instrumental signature — and the noise field,
generated later — exactly where it is.

Extending the generator means writing a stage and inserting it into the list
returned by :func:`default_pipeline` — no edits to the core renderer.  This is
the architectural seam intended for future scene effects (background
gradients, cosmic rays, field-variable PSF, …).

Relationship to ``generate_frame``
----------------------------------
``generate_frame`` keeps its original clean-frame behaviour when no
:class:`~opta_pipeline.synth.detector.DetectorModel` is supplied (legacy
numbers are byte-stable).  When a detector model *is* supplied it delegates to
:func:`render_frame` here, which runs the full instrumental chain and adopts
the physically-correct *observed-sky* brightness convention (sky surface
brightness is taken as measured at the detector and is **not** re-attenuated
by atmospheric extinction; only source flux is extincted).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from opta_model.hardware import OpticsConfig, SensorConfig, compute_pixel_scale
from opta_model.radiometry import (
    atmospheric_extinction,
    sky_background_electrons,
    trailing_loss,
)

from opta_pipeline.synth import (
    SatelliteGroundTruth,
    SatelliteSpec,
    StarGroundTruth,
    StarSpec,
    SynthFrame,
    _sat_streak_signal_e,
    _star_total_signal_e,
)
from opta_pipeline.synth.detector import DetectorModel
from opta_pipeline.synth.distortion import DistortionModel
from opta_pipeline.synth.jitter import PointingJitter
from opta_pipeline.synth.psf import PSFModel, render_source, render_streak
from opta_pipeline.synth.rolling_shutter import RollingShutterReadout

__all__ = [
    "RenderContext",
    "Stage",
    "default_pipeline",
    "render_frame",
    "generate_dark_frame",
    "generate_flat_frame",
    "stage_pointing_jitter",
    "stage_add_sky",
    "stage_add_stars",
    "stage_add_satellites",
    "stage_apply_flat",
    "stage_add_dark",
    "stage_shot_noise",
    "stage_add_read_noise",
    "stage_add_bias",
    "stage_saturate",
    "PSFModel",
]


# ---------------------------------------------------------------------------
# Render context
# ---------------------------------------------------------------------------


@dataclass
class RenderContext:
    """Mutable working state threaded through the stage pipeline.

    Stages read the scene/hardware fields and mutate :attr:`image_e` and the
    ground-truth lists.  ``image_e`` holds the working frame in electrons:
    it starts as the expected (noise-free) collected charge and is progressively
    converted into a noisy, instrumentally-corrupted electron image.
    """

    sensor: SensorConfig
    optics: OpticsConfig
    detector: DetectorModel
    rng: np.random.Generator

    # Derived scene/hardware scalars
    integration_time_s: float
    pixel_scale_arcsec: float
    aperture_m: float
    sky_mag_arcsec2: float
    elevation_deg: float
    psf: PSFModel

    # Inputs
    satellites: list[SatelliteSpec]
    stars: list[StarSpec]

    # Working image (electrons) and accumulated ground truth
    image_e: np.ndarray = field(hash=False, compare=False)
    sat_truths: list[SatelliteGroundTruth] = field(default_factory=list)
    star_truths: list[StarGroundTruth] = field(default_factory=list)
    sky_e_per_pixel: float = 0.0
    distortion: DistortionModel | None = None
    rolling_shutter: RollingShutterReadout | None = None
    pointing_jitter: PointingJitter | None = None
    frame_index: int = 0
    #: This frame's rigid scene offset ``(dx_px, dy_px)``, set by
    #: :func:`stage_pointing_jitter`.  ``(0.0, 0.0)`` = boresight on nominal.
    scene_offset_px: tuple[float, float] = (0.0, 0.0)

    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Map an ideal *scene* pixel to its as-imaged position.

        Two mappings compose here, in physical order: the rigid pointing
        offset moves the sky across the focal plane, then the lens distortion
        model maps that field position to a pixel.  Only source stages call
        this — detector stages index the array directly, which is exactly why
        the instrumental signature stays put while the scene moves.
        """
        dx, dy = self.scene_offset_px
        x += dx
        y += dy
        if self.distortion is None:
            return x, y
        return self.distortion.apply(x, y, self.image_e.shape)


Stage = Callable[[RenderContext], None]


# ---------------------------------------------------------------------------
# Built-in stages (canonical CCD/CMOS signal-chain order)
# ---------------------------------------------------------------------------


def stage_pointing_jitter(ctx: RenderContext) -> None:
    """Draw this frame's rigid scene offset (no-op without a jitter model).

    Runs **first**, before any source or detector stage, so the offset is in
    place when :meth:`RenderContext.distort` projects sources and is invisible
    to everything downstream of ``stage_apply_flat``.  The offset comes from
    :meth:`PointingJitter.offset_px` — a private stream keyed on
    ``(seed, ctx.frame_index)`` — so enabling jitter leaves ``ctx.rng`` and
    hence the noise realization untouched.
    """
    if ctx.pointing_jitter is None:
        return
    ctx.scene_offset_px = ctx.pointing_jitter.offset_px(ctx.frame_index)


def stage_add_sky(ctx: RenderContext) -> None:
    """Add the diffuse sky background (observed-sky convention).

    The sky surface brightness is treated as *already observed at the
    detector* and is therefore not re-attenuated by atmospheric extinction —
    this matches ``opta_model.radiometry.sky_background_electrons`` and
    resolves the long-standing synth/model convention mismatch.
    """
    sky_e = sky_background_electrons(
        ctx.sky_mag_arcsec2,
        ctx.pixel_scale_arcsec,
        ctx.aperture_m,
        ctx.integration_time_s,
        ctx.sensor.quantum_efficiency,
    )
    ctx.sky_e_per_pixel = sky_e
    ctx.image_e += sky_e


def stage_add_stars(ctx: RenderContext) -> None:
    """Inject stellar PSFs as expected collected electrons (extincted)."""
    ext = atmospheric_extinction(ctx.elevation_deg)
    for s in ctx.stars:
        sig_e = _star_total_signal_e(
            s, ctx.aperture_m, ctx.integration_time_s,
            ctx.sensor.quantum_efficiency, ctx.optics.transmission, ext,
        )
        xs, ys = ctx.distort(s.x, s.y)
        fwhm, ell, theta = ctx.psf.params_at(xs, ys, ctx.image_e.shape)
        render_source(
            ctx.image_e, xs, ys, sig_e, fwhm_px=fwhm, profile=ctx.psf.profile,
            moffat_beta=ctx.psf.moffat_beta, ellipticity=ell, theta=theta,
            oversample=ctx.psf.oversample,
        )
        ctx.star_truths.append(
            StarGroundTruth(
                x=xs, y=ys, signal_electrons=sig_e, magnitude=s.magnitude
            )
        )


def stage_add_satellites(ctx: RenderContext) -> None:
    """Inject satellite streaks as expected collected electrons (extincted)."""
    ext = atmospheric_extinction(ctx.elevation_deg)
    width = ctx.sensor.resolution_h
    height = ctx.sensor.resolution_v
    for sat in ctx.satellites:
        cx = sat.x_center if sat.x_center is not None else width / 2.0
        cy = sat.y_center if sat.y_center is not None else height / 2.0

        trail_arcsec = sat.angular_velocity_deg_s * 3600.0 * ctx.integration_time_s
        trail_px = trail_arcsec / ctx.pixel_scale_arcsec
        half = trail_px / 2.0
        angle_rad = math.radians(sat.angle_deg)
        dx = half * math.cos(angle_rad)
        dy = half * math.sin(angle_rad)
        # Distort endpoints/centre to their as-imaged positions.
        x0, y0 = ctx.distort(cx - dx, cy - dy)
        x1, y1 = ctx.distort(cx + dx, cy + dy)
        cxd, cyd = ctx.distort(cx, cy)

        # Rolling-shutter readout: displace the moving streak by the per-row
        # timing bias (stationary stars are untouched); truth stays geometric.
        if ctx.rolling_shutter is not None:
            v_px_s = sat.angular_velocity_deg_s * 3600.0 / ctx.pixel_scale_arcsec
            vx_px_s = v_px_s * math.cos(angle_rad)
            vy_px_s = v_px_s * math.sin(angle_rad)
            x0, y0 = ctx.rolling_shutter.displace(
                x0, y0, vx_px_s, vy_px_s, ctx.image_e.shape
            )
            x1, y1 = ctx.rolling_shutter.displace(
                x1, y1, vx_px_s, vy_px_s, ctx.image_e.shape
            )

        t_loss = trailing_loss(
            sat.angular_velocity_deg_s, ctx.pixel_scale_arcsec, ctx.integration_time_s
        )
        n_pixels = max(1.0, trail_px)
        total_streak_e, per_pixel_signal_e = _sat_streak_signal_e(
            sat, ctx.aperture_m, ctx.integration_time_s,
            ctx.sensor.quantum_efficiency, ctx.optics.transmission, ext,
            n_pixels, t_loss,
        )
        render_streak(ctx.image_e, x0, y0, x1, y1, total_streak_e, ctx.psf)
        ctx.sat_truths.append(
            SatelliteGroundTruth(
                x_start=x0,
                y_start=y0,
                x_end=x1,
                y_end=y1,
                x_center=cxd,
                y_center=cyd,
                signal_electrons=per_pixel_signal_e,
                magnitude=sat.magnitude,
                angular_velocity_deg_s=sat.angular_velocity_deg_s,
            )
        )


def stage_apply_flat(ctx: RenderContext) -> None:
    """Apply the photometric response (PRNU × vignetting × dead pixels).

    Multiplies the collected photo-signal (sky + sources).  Applied *before*
    shot noise and *before* dark/bias because it scales the optical/photonic
    response, not the additive electronic terms.
    """
    ctx.image_e *= ctx.detector.flat


def stage_add_dark(ctx: RenderContext) -> None:
    """Add per-pixel dark-current charge (including hot pixels)."""
    ctx.image_e += ctx.detector.dark_current_e_s * ctx.integration_time_s


def stage_shot_noise(ctx: RenderContext) -> None:
    """Apply Poisson shot noise to the total collected charge."""
    lam = np.clip(ctx.image_e, 0.0, None)
    ctx.image_e = ctx.rng.poisson(lam).astype(np.float64)


def stage_add_read_noise(ctx: RenderContext) -> None:
    """Add Gaussian read noise (electronic, not subject to flat/shot)."""
    ctx.image_e += ctx.rng.normal(
        0.0, ctx.sensor.readout_noise_e, size=ctx.image_e.shape
    )


def stage_add_bias(ctx: RenderContext) -> None:
    """Add the bias pedestal and offset fixed-pattern noise."""
    ctx.image_e += ctx.detector.bias_offset_e + ctx.detector.bias_fpn_e


def stage_saturate(ctx: RenderContext) -> None:
    """Apply full-well saturation and mild A/D nonlinearity near full well.

    The compressive nonlinearity follows ``x' = x·(1 − k·x/FW)`` for the
    fractional coefficient *k*; with ``k = 0`` the response is perfectly
    linear and only the hard full-well clip applies.
    """
    fw = ctx.detector.full_well_e
    k = ctx.detector.nonlinearity
    if k > 0:
        frac = np.clip(ctx.image_e / fw, 0.0, 1.0)
        ctx.image_e = ctx.image_e * (1.0 - k * frac)
    np.clip(ctx.image_e, 0.0, fw, out=ctx.image_e)


def default_pipeline() -> list[Stage]:
    """Return the built-in stages in canonical signal-chain order.

    Callers extend the generator by inserting/removing stages in the returned
    list (e.g. ``p = default_pipeline(); p.insert(3, my_gradient_stage)``).
    """
    return [
        stage_pointing_jitter,
        stage_add_sky,
        stage_add_stars,
        stage_add_satellites,
        stage_apply_flat,
        stage_add_dark,
        stage_shot_noise,
        stage_add_read_noise,
        stage_add_bias,
        stage_saturate,
    ]


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def _make_header(
    sensor: SensorConfig,
    detector: DetectorModel,
    integration_time_s: float,
    pixel_scale: float,
    sky_e: float,
    date_obs: str,
    node_id: str,
    sensor_temp_c: float,
    imagetyp: str = "LIGHT",
) -> dict[str, Any]:
    """Build a FITS header dict for a rendered frame."""
    return {
        "DATE-OBS": date_obs,
        "NODE-ID": node_id,
        "EXPTIME": integration_time_s,
        "INSTRUME": "SYNTH",
        "IMAGETYP": imagetyp,
        "GAIN": detector.gain_e_adu,
        "TEMP": sensor_temp_c,
        "NAXIS": 2,
        "NAXIS1": sensor.resolution_h,
        "NAXIS2": sensor.resolution_v,
        "BITPIX": 16,
        "BSCALE": 1.0,
        "BZERO": 0.0,
        "PIXSCALE": pixel_scale,
        "SKYADU": sky_e / detector.gain_e_adu,
    }


def _finalize(
    image_e: np.ndarray,
    detector: DetectorModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantise an electron image to uint16 ADU; return ``(adu, electrons)``."""
    stored_e = image_e.copy()
    adu = np.clip(image_e / detector.gain_e_adu, 0, 2**16 - 1).astype(np.uint16)
    return adu, stored_e


def render_frame(
    sensor: SensorConfig,
    optics: OpticsConfig,
    detector: DetectorModel,
    *,
    sky_mag_arcsec2: float = 21.0,
    elevation_deg: float = 45.0,
    satellites: list[SatelliteSpec] | None = None,
    stars: list[StarSpec] | None = None,
    psf_fwhm_px: float = 2.0,
    psf: PSFModel | None = None,
    distortion: DistortionModel | None = None,
    rolling_shutter: RollingShutterReadout | None = None,
    pointing_jitter: PointingJitter | None = None,
    frame_index: int = 0,
    pipeline: list[Stage] | None = None,
    date_obs: str | None = None,
    node_id: str = "SYNTH-01",
    sensor_temp_c: float = 20.0,
    rng: np.random.Generator | None = None,
) -> SynthFrame:
    """Render a realistic frame by running an effects-chain over a context.

    Unlike :func:`opta_pipeline.synth.generate_frame`'s clean path, this
    routine imprints the full instrumental signature carried by *detector*
    (bias, flat, dark/hot pixels, saturation) and applies shot noise to the
    sources as well as the background.  Sources are rendered with the
    sub-pixel-accurate, field-variable PSF in :mod:`opta_pipeline.synth.psf`.

    Parameters
    ----------
    sensor, optics : SensorConfig, OpticsConfig
        Hardware presets.
    detector : DetectorModel
        Frame-stable instrumental signature (see
        :func:`opta_pipeline.synth.detector.build_detector_model`).
    sky_mag_arcsec2 : float
        Observed sky surface brightness in mag/arcsec² (not re-extincted).
    elevation_deg : float
        Source elevation for atmospheric extinction of stars/satellites.
    satellites, stars : list | None
        Sources to inject.
    psf_fwhm_px : float
        Uniform PSF FWHM in pixels.  Ignored when *psf* is given.
    psf : PSFModel | None
        Field-variable PSF (optical aberration).  Defaults to a uniform model
        built from ``psf_fwhm_px``.
    pointing_jitter : PointingJitter | None
        Scene-domain per-frame rigid offset (wind / mount vibration).  ``None``
        (the default) is an exact no-op — output is byte-identical to callers
        that predate the stage.  The offset moves *sources only*: the detector
        signature stays at fixed detector pixels and the noise field, generated
        downstream, is never resampled.
    frame_index : int
        Index of this frame within its sequence.  Selects the jitter offset
        (see :meth:`PointingJitter.offset_px`); ignored without a jitter model.
    pipeline : list[Stage] | None
        Ordered stages to run.  Defaults to :func:`default_pipeline`.
    date_obs, node_id, sensor_temp_c : ...
        FITS header metadata.
    rng : np.random.Generator | None
        RNG for the per-frame stochastic stages (seeded 0 if None).

    Returns
    -------
    SynthFrame
        Rendered frame with uint16 ADU data, the pre-quantisation electron
        image (``data_float``), and ground truth.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if pipeline is None:
        pipeline = default_pipeline()
    if psf is None:
        psf = PSFModel(fwhm_center_px=psf_fwhm_px, fwhm_edge_px=psf_fwhm_px)

    integration_time_s = 1.0 / sensor.frame_rate_hz
    pixel_scale = compute_pixel_scale(sensor.pixel_size_um, optics.focal_length_mm)

    ctx = RenderContext(
        sensor=sensor,
        optics=optics,
        detector=detector,
        rng=rng,
        integration_time_s=integration_time_s,
        pixel_scale_arcsec=pixel_scale,
        aperture_m=optics.aperture_mm / 1000.0,
        sky_mag_arcsec2=sky_mag_arcsec2,
        elevation_deg=elevation_deg,
        psf=psf,
        satellites=list(satellites or []),
        stars=list(stars or []),
        image_e=np.zeros(detector.shape, dtype=np.float64),
        distortion=distortion,
        rolling_shutter=rolling_shutter,
        pointing_jitter=pointing_jitter,
        frame_index=frame_index,
    )

    for stage in pipeline:
        stage(ctx)

    adu, stored_e = _finalize(ctx.image_e, detector)
    header = _make_header(
        sensor,
        detector,
        integration_time_s,
        pixel_scale,
        ctx.sky_e_per_pixel,
        date_obs if date_obs is not None else "2000-01-01T00:00:00.000",
        node_id,
        sensor_temp_c,
    )
    return SynthFrame(
        data=adu,
        data_float=stored_e,
        header=header,
        satellites=tuple(ctx.sat_truths),
        stars=tuple(ctx.star_truths),
        sky_e_per_pixel=ctx.sky_e_per_pixel,
        readout_noise_e=sensor.readout_noise_e,
        pixel_scale_arcsec=pixel_scale,
    )


# ---------------------------------------------------------------------------
# Calibration-frame generation
# ---------------------------------------------------------------------------


def generate_dark_frame(
    sensor: SensorConfig,
    detector: DetectorModel,
    *,
    date_obs: str | None = None,
    node_id: str = "SYNTH-01",
    sensor_temp_c: float = 20.0,
    rng: np.random.Generator | None = None,
) -> SynthFrame:
    """Render a dark frame (shutter closed): bias + dark + noise, no light.

    A master built from several of these via
    :func:`opta_pipeline.calibrate.make_master_dark` captures the bias
    pedestal, offset FPN, and dark-current/hot-pixel structure — exactly what
    ``calibrate_frame`` subtracts from science frames.

    ``rng=None`` seeds from OS entropy, so **each call is an independent noise
    realization** — unlike :func:`render_frame`, whose ``None`` default is the
    reproducible seed 0.  A master is built by combining several of these, and
    N bit-identical inputs would survive the sigma clip intact and bake one
    read/shot-noise draw into the master as false fixed-pattern structure that
    ``calibrate_frame`` then subtracts from every science frame.  Pass an
    explicit ``rng`` (distinct per frame) when reproducibility is needed.
    """
    if rng is None:
        rng = np.random.default_rng()
    pipeline = [
        stage_add_dark,
        stage_shot_noise,
        stage_add_read_noise,
        stage_add_bias,
        stage_saturate,
    ]
    frame = render_frame(
        sensor,
        detector=detector,
        optics=_NULL_OPTICS,
        sky_mag_arcsec2=99.0,  # unused (no sky stage), kept explicit
        satellites=[],
        stars=[],
        pipeline=pipeline,
        date_obs=date_obs,
        node_id=node_id,
        sensor_temp_c=sensor_temp_c,
        rng=rng,
    )
    frame.header["IMAGETYP"] = "DARK"
    return frame


def generate_flat_frame(
    sensor: SensorConfig,
    detector: DetectorModel,
    *,
    illumination_e: float | None = None,
    date_obs: str | None = None,
    node_id: str = "SYNTH-01",
    sensor_temp_c: float = 20.0,
    rng: np.random.Generator | None = None,
) -> SynthFrame:
    """Render a flat frame: uniform illumination × flat field + dark + noise.

    The uniform pre-flat illumination level (electrons) must sit well above the
    read/dark floor and **below full well** — a saturated flat clips away the
    very response structure it is meant to capture, leaving ``master_flat ≈ 1``
    (a silent no-op).  When *illumination_e* is ``None`` it defaults to half the
    full-well depth, which is safe for any sensor; the brightest vignetting
    corner (response ~0.6–1.0) then stays comfortably unsaturated.

    A master built from several of these via
    :func:`opta_pipeline.calibrate.make_master_flat` recovers the normalised
    photometric response (PRNU × vignetting), and division by it flattens the
    science frame.

    ``rng=None`` seeds from OS entropy — independent noise per call, for the
    same reason as :func:`generate_dark_frame` (a master built from identical
    frames carries one noise draw as false PRNU).  Pass an explicit ``rng``
    when reproducibility is needed.
    """
    if rng is None:
        rng = np.random.default_rng()
    if illumination_e is None:
        illumination_e = 0.5 * detector.full_well_e

    def _stage_uniform_illumination(ctx: RenderContext) -> None:
        """Fill the frame with a uniform pre-flat illumination level."""
        ctx.image_e += illumination_e

    pipeline = [
        _stage_uniform_illumination,
        stage_apply_flat,
        stage_add_dark,
        stage_shot_noise,
        stage_add_read_noise,
        stage_add_bias,
        stage_saturate,
    ]
    frame = render_frame(
        sensor,
        detector=detector,
        optics=_NULL_OPTICS,
        satellites=[],
        stars=[],
        pipeline=pipeline,
        date_obs=date_obs,
        node_id=node_id,
        sensor_temp_c=sensor_temp_c,
        rng=rng,
    )
    frame.header["IMAGETYP"] = "FLAT"
    return frame


# A placeholder optics config for calibration frames, which collect no
# on-sky light and therefore do not depend on focal length / aperture except
# for the (unused) header pixel-scale term.
_NULL_OPTICS = OpticsConfig(focal_length_mm=85.0, aperture_mm=60.0, transmission=0.9)
