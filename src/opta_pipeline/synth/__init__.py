"""Synthetic frame generator for OpTA pipeline testing.

Produces FITS frames that exercise the full pipeline (calibrate → detect →
astrometry → tracklet) with known ground truth.  All signal levels are
physically derived from opta_model.radiometry (OpTA design rule #4).

Noise model:  Poisson(sky_e + dark_e) + Gaussian(0, readout_noise_e) per pixel.
PSF model:    2-D Gaussian with configurable FWHM.
Streak model: Line source with Gaussian cross-section.

Sub-modules (WP-A3 onward):
  synth.pass_    SyntheticPass: SGP4-driven multi-frame pass generator (WP-A3).
  synth.catalog  Pointing-aware star catalog, ``star_field_at`` (WP-A4).
  synth.jitter   PointingJitter: scene-domain per-frame rigid pointing offset.

Usage
-----
    from opta_pipeline.synth import generate_frame, satellite_signal_electrons

    frame = generate_frame(
        sensor=IMX585_PRESET,
        optics=VILTROX_85_F14_PRESET,
        sky_mag_arcsec2=21.0,
        elevation_deg=45.0,
        satellites=[SatelliteSpec(magnitude=11.0, angular_velocity_deg_s=0.5)],
        rng=np.random.default_rng(42),
    )
    write_fits(frame, "synth_001.fits")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from opta_model.hardware import OpticsConfig, SensorConfig, compute_pixel_scale
    from opta_model.radiometry import (
        atmospheric_extinction,
        signal_electrons,
        trailing_loss,
    )
except ImportError as exc:  # pragma: no cover - exercised without opta-model
    raise ImportError(
        "opta_pipeline.synth renders frames from opta-model physics; install "
        "the optional dependency with: pip install 'opta-pipeline[synth]'"
    ) from exc

from opta_pipeline.synth.detector import DetectorModel, build_detector_model
from opta_pipeline.synth.distortion import DistortionModel
from opta_pipeline.synth.jitter import PointingJitter
from opta_pipeline.synth.psf import PSFModel
from opta_pipeline.synth.rolling_shutter import RollingShutterReadout

__all__ = [
    "SatelliteSpec",
    "StarSpec",
    "SatelliteGroundTruth",
    "StarGroundTruth",
    "SynthFrame",
    "DetectorModel",
    "build_detector_model",
    "PSFModel",
    "DistortionModel",
    "PointingJitter",
    "RollingShutterReadout",
    "generate_frame",
    "render_frame",
    "generate_dark_frame",
    "generate_flat_frame",
    "satellite_signal_electrons",
    "sky_electrons_per_pixel",
    "write_fits",
]

# ---------------------------------------------------------------------------
# Input specs (what to inject)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SatelliteSpec:
    """Specification for a single synthetic satellite streak.

    Parameters
    ----------
    magnitude : float
        Apparent V-band magnitude (from apparent_magnitude or direct input).
    angular_velocity_deg_s : float
        Angular velocity in deg/s — determines streak length.
    x_center : float | None
        Streak midpoint x in pixels.  If None, placed at frame centre.
    y_center : float | None
        Streak midpoint y in pixels.  If None, placed at frame centre.
    angle_deg : float
        Streak orientation in degrees (0 = horizontal, CCW positive).
    signal_e : float | None
        **Free-injection** override: total collected electrons in the streak.
        When set, the source is injected at this exact flux and ``magnitude`` is
        *not* routed through ``opta_model.radiometry`` (it is only recorded in
        ground truth).  This is the non-physics-traceable path for pure pipeline
        stress-testing; leave ``None`` for the default physics-traceable mode.
    """

    magnitude: float
    angular_velocity_deg_s: float = 0.5
    x_center: float | None = None
    y_center: float | None = None
    angle_deg: float = 0.0
    signal_e: float | None = None


@dataclass(frozen=True)
class StarSpec:
    """Specification for a synthetic stellar point source.

    Parameters
    ----------
    magnitude : float
        Apparent V-band magnitude.
    x : float
        Pixel x position.
    y : float
        Pixel y position.
    signal_e : float | None
        **Free-injection** override: total collected electrons in the PSF.
        When set, the source is injected at this exact flux and ``magnitude`` is
        *not* routed through ``opta_model.radiometry`` (only recorded in ground
        truth).  Non-physics-traceable path for stress-testing; ``None`` keeps
        the default traceable mode.
    """

    magnitude: float
    x: float
    y: float
    signal_e: float | None = None

    @classmethod
    def from_catalog(cls, wcs, catalog_stars) -> list[StarSpec]:
        """Create a list of StarSpec objects from catalog stars.

        Pixel positions are computed via ``radec_to_pixels(wcs, ...)``.
        Stars that project outside [0, inf) are still included; callers
        that need FOV filtering should pre-filter with ``star_field_at``.

        Parameters
        ----------
        wcs : WCSSolution
            Pointing WCS for pixel-position projection.
        catalog_stars : iterable of CatalogStar
            Stars from ``opta_pipeline.synth.catalog``.

        Returns
        -------
        list[StarSpec]
            One entry per catalog star, with pixel (x, y) and magnitude.
        """
        from opta_pipeline.astrometry import radec_to_pixels  # lazy: avoids circular

        result: list[StarSpec] = []
        for star in catalog_stars:
            x, y = radec_to_pixels(wcs, star.ra_deg, star.dec_deg)
            result.append(cls(magnitude=star.magnitude, x=x, y=y))
        return result


# ---------------------------------------------------------------------------
# Ground-truth output types (what was actually injected)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SatelliteGroundTruth:
    """Ground truth for a single injected satellite streak."""

    x_start: float
    y_start: float
    x_end: float
    y_end: float
    x_center: float
    y_center: float
    signal_electrons: float
    magnitude: float
    angular_velocity_deg_s: float


@dataclass(frozen=True)
class StarGroundTruth:
    """Ground truth for a single injected stellar PSF."""

    x: float
    y: float
    signal_electrons: float
    magnitude: float


# ---------------------------------------------------------------------------
# SynthFrame output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthFrame:
    """A synthetic FITS frame with associated ground truth.

    Attributes
    ----------
    data : np.ndarray
        2-D uint16 array (height × width) in ADU.  At low sky flux
        (≲1 e⁻/px), uint16 truncation of near-zero values introduces
        a negative bias in the mean.  Use ``data_float`` for noise-level
        assertions that require the full electron-domain precision.
    data_float : np.ndarray
        2-D float64 array (height × width) in electrons, before ADU
        quantisation.  Use this for low-flux statistical tests (WP-C3).
    header : dict[str, Any]
        FITS header key-value pairs (complies with pipeline_defaults.yaml).
    satellites : tuple[SatelliteGroundTruth, ...]
        Ground truth for each injected satellite streak.
    stars : tuple[StarGroundTruth, ...]
        Ground truth for each injected star.
    sky_e_per_pixel : float
        Sky background in electrons/pixel (for SNR computation).
    readout_noise_e : float
        Readout noise in electrons/pixel.
    pixel_scale_arcsec : float
        Image scale in arcsec/pixel.
    """

    data: np.ndarray = field(hash=False, compare=False)
    data_float: np.ndarray = field(hash=False, compare=False)
    header: dict[str, Any] = field(hash=False, compare=False)
    satellites: tuple[SatelliteGroundTruth, ...]
    stars: tuple[StarGroundTruth, ...]
    sky_e_per_pixel: float
    readout_noise_e: float
    pixel_scale_arcsec: float


# ---------------------------------------------------------------------------
# Radiometric helpers (physics-traceable per AGENTS.md rule #4)
# ---------------------------------------------------------------------------


def sky_electrons_per_pixel(
    sky_mag_arcsec2: float,
    pixel_scale_arcsec: float,
    aperture_m: float,
    integration_time_s: float,
    quantum_efficiency: float,
    elevation_deg: float = 90.0,
    sky_is_observed: bool = False,
) -> float:
    """Compute sky background electrons per pixel.

    Converts sky surface brightness from mag/arcsec² to mag/pixel, then
    applies signal_electrons to get the expected background count.

    Sky-brightness convention
    -------------------------
    Two conventions exist for ``sky_mag_arcsec2`` and they must not be mixed:

    * **observed-at-detector** (``sky_is_observed=True``): the value is the sky
      brightness as actually measured through the atmosphere, so *no* further
      extinction is applied.  This matches
      :func:`opta_model.radiometry.sky_background_electrons` and is the
      physically correct convention used by the realistic renderer
      (:func:`opta_pipeline.synth.render.render_frame`).
    * **above-atmosphere** (``sky_is_observed=False``, the default): the value
      is attenuated by ``atmospheric_extinction(elevation_deg)``.  This is the
      historical behaviour, retained as the default so legacy pinned numbers
      (smoke ``peak_snr``, figures) stay byte-stable.  Prefer the observed
      convention for new work.

    Parameters
    ----------
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec².
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel.
    aperture_m : float
        Clear aperture diameter in metres.
    integration_time_s : float
        Exposure time in seconds.
    quantum_efficiency : float
        Detector QE (0, 1].
    elevation_deg : float
        Object elevation for extinction correction (default 90° = no ext.).
        Ignored when ``sky_is_observed=True``.
    sky_is_observed : bool
        Select the brightness convention (see above).
    """
    pixel_area_arcsec2 = pixel_scale_arcsec**2
    sky_per_pixel_mag = sky_mag_arcsec2 - 2.5 * math.log10(
        max(pixel_area_arcsec2, 1e-30)
    )
    ext = 0.0 if sky_is_observed else atmospheric_extinction(elevation_deg)
    return signal_electrons(
        sky_per_pixel_mag,
        aperture_m,
        integration_time_s,
        quantum_efficiency,
        extinction_mag=ext,
    )


def satellite_signal_electrons(
    magnitude: float,
    angular_velocity_deg_s: float,
    sensor: SensorConfig,
    optics: OpticsConfig,
    elevation_deg: float = 45.0,
) -> float:
    """Compute per-pixel signal electrons for a trailed satellite.

    The satellite's trail spans N pixels determined by trailing_loss.  The
    returned value is the *per-pixel* signal, as seen by the detector.

    Parameters
    ----------
    magnitude : float
        Apparent V-band magnitude of the satellite.
    angular_velocity_deg_s : float
        Angular velocity in deg/s.
    sensor : SensorConfig
        Sensor hardware parameters.
    optics : OpticsConfig
        Optics hardware parameters.
    elevation_deg : float
        Elevation for atmospheric extinction.

    Returns
    -------
    float
        Signal electrons per pixel (total signal × trailing_loss).
    """
    aperture_m = optics.aperture_mm / 1000.0
    integration_time_s = 1.0 / sensor.frame_rate_hz
    pixel_scale = compute_pixel_scale(sensor.pixel_size_um, optics.focal_length_mm)
    ext = atmospheric_extinction(elevation_deg)

    total_signal = (
        signal_electrons(
            magnitude,
            aperture_m,
            integration_time_s,
            sensor.quantum_efficiency,
            extinction_mag=ext,
        )
        * optics.transmission
    )
    t_loss = trailing_loss(angular_velocity_deg_s, pixel_scale, integration_time_s)
    return total_signal * t_loss


# ---------------------------------------------------------------------------
# Per-source signal (traceable magnitude vs free-injection electrons)
# ---------------------------------------------------------------------------


def _star_total_signal_e(
    spec: StarSpec,
    aperture_m: float,
    integration_time_s: float,
    quantum_efficiency: float,
    transmission: float,
    ext: float,
) -> float:
    """Total PSF electrons for a star: ``spec.signal_e`` if given, else radiometry."""
    if spec.signal_e is not None:
        return spec.signal_e
    return (
        signal_electrons(
            spec.magnitude,
            aperture_m,
            integration_time_s,
            quantum_efficiency,
            extinction_mag=ext,
        )
        * transmission
    )


def _sat_streak_signal_e(
    spec: SatelliteSpec,
    aperture_m: float,
    integration_time_s: float,
    quantum_efficiency: float,
    transmission: float,
    ext: float,
    n_pixels: float,
    t_loss: float,
) -> tuple[float, float]:
    """Return ``(total_streak_e, per_pixel_e)`` for a satellite streak.

    Free-injection (``spec.signal_e`` set) spreads the given total over the
    trail; otherwise the total derives from radiometry and the per-pixel value
    is the trailing-loss-diluted signal.
    """
    if spec.signal_e is not None:
        total_streak_e = spec.signal_e
        return total_streak_e, total_streak_e / n_pixels
    total_signal_e = (
        signal_electrons(
            spec.magnitude,
            aperture_m,
            integration_time_s,
            quantum_efficiency,
            extinction_mag=ext,
        )
        * transmission
    )
    per_pixel_e = total_signal_e * t_loss
    return per_pixel_e * n_pixels, per_pixel_e


# ---------------------------------------------------------------------------
# PSF / streak drawing primitives
# ---------------------------------------------------------------------------


def _inject_psf(
    frame: np.ndarray,
    x: float,
    y: float,
    total_signal_e: float,
    fwhm_px: float = 2.0,
) -> None:
    """Inject a 2-D Gaussian PSF in-place (electrons, floating-point frame)."""
    sigma = fwhm_px / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    radius = int(math.ceil(4.0 * sigma)) + 1
    h, w = frame.shape

    yi_min = max(0, int(y) - radius)
    yi_max = min(h, int(y) + radius + 2)
    xi_min = max(0, int(x) - radius)
    xi_max = min(w, int(x) + radius + 2)

    yy, xx = np.mgrid[yi_min:yi_max, xi_min:xi_max]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    g_sum = g.sum()
    if g_sum > 0:
        frame[yi_min:yi_max, xi_min:xi_max] += total_signal_e * g / g_sum


def _inject_streak(
    frame: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    total_signal_e: float,
    fwhm_px: float = 2.0,
) -> None:
    """Inject a linear streak (line source + Gaussian cross-section) in-place.

    Signal is uniformly distributed along the line, with Gaussian
    broadening perpendicular to the direction of motion.
    """
    sigma = fwhm_px / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        _inject_psf(frame, x0, y0, total_signal_e, fwhm_px)
        return

    h, w = frame.shape
    n_samples = max(int(length * 2) + 1, 2)
    t = np.linspace(0.0, 1.0, n_samples)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    pad = int(math.ceil(4.0 * sigma)) + 2
    x_min = int(max(0, min(xs) - pad))
    x_max = int(min(w, max(xs) + pad + 1))
    y_min = int(max(0, min(ys) - pad))
    y_max = int(min(h, max(ys) + pad + 1))

    if x_max <= x_min or y_max <= y_min:
        return

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    # For each pixel, accumulate signal from all sample points along the streak
    acc = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float64)
    for xi, yi in zip(xs, ys):
        d2 = (xx - xi) ** 2 + (yy - yi) ** 2
        acc += np.exp(-d2 / (2.0 * sigma**2))

    acc_sum = acc.sum()
    if acc_sum > 0:
        frame[y_min:y_max, x_min:x_max] += total_signal_e * acc / acc_sum


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------


def generate_frame(
    sensor: SensorConfig,
    optics: OpticsConfig,
    sky_mag_arcsec2: float = 21.0,
    elevation_deg: float = 45.0,
    satellites: list[SatelliteSpec] | None = None,
    stars: list[StarSpec] | None = None,
    psf_fwhm_px: float = 2.0,
    gain_e_adu: float = 1.0,
    date_obs: str | None = None,
    node_id: str = "SYNTH-01",
    sensor_temp_c: float = 20.0,
    rng: np.random.Generator | None = None,
    detector: DetectorModel | None = None,
    psf: PSFModel | None = None,
    distortion: DistortionModel | None = None,
    rolling_shutter: RollingShutterReadout | None = None,
    pointing_jitter: PointingJitter | None = None,
    frame_index: int = 0,
    sky_is_observed: bool = False,
) -> SynthFrame:
    """Generate a synthetic FITS-ready frame with injected sources.

    Two rendering paths are available:

    * **clean (legacy)** — when ``detector is None`` (the default), the frame
      carries only the simple noise floor (Poisson sky+dark, Gaussian read)
      with additive, noise-free sources.  Output is byte-stable with historical
      pinned numbers.
    * **realistic** — when a :class:`DetectorModel` is supplied, the call
      delegates to :func:`opta_pipeline.synth.render.render_frame`, which runs
      the full instrumental signal chain (bias + offset FPN, flat field /
      PRNU + vignetting, dark-current map with hot/dead pixels, shot noise on
      sources, full-well saturation) and adopts the physically-correct
      observed-sky brightness convention.  Use this path to exercise the
      calibration → detection → astrometry chain on realistic frames.

    Parameters
    ----------
    sensor : SensorConfig
        Camera sensor parameters (from hardware.py presets).
    optics : OpticsConfig
        Lens parameters.
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec² (default 21.0, dark sky).
    elevation_deg : float
        Observation elevation for atmospheric extinction.
    satellites : list[SatelliteSpec] | None
        Satellite streaks to inject.
    stars : list[StarSpec] | None
        Stellar point sources to inject.
    psf_fwhm_px : float
        PSF FWHM in pixels for both stars and streak cross-section.
    gain_e_adu : float
        Sensor gain in electrons/ADU (default 1.0 = unity gain).  Ignored on
        the realistic path, which uses ``detector.gain_e_adu``.
    date_obs : str | None
        ISO 8601 UTC observation timestamp.  Defaults to J2000 epoch.
    node_id : str
        Node identifier for FITS header.
    sensor_temp_c : float
        Sensor temperature for FITS header.
    rng : np.random.Generator | None
        NumPy random generator.  Created from a fixed seed if None.
    detector : DetectorModel | None
        Frame-stable instrumental signature.  ``None`` selects the clean path;
        a model selects the realistic path (see above).
    pointing_jitter : PointingJitter | None
        Scene-domain per-frame rigid offset (wind / mount vibration), honoured
        on **both** paths.  Sources are drawn at their offset sub-pixel
        positions; the noise floor and (realistic path) the whole detector
        signature stay bolted to the detector.  ``None`` is an exact no-op, so
        every pre-existing caller keeps byte-identical frames.
    frame_index : int
        Index of this frame in its sequence; selects the jitter offset
        (:meth:`PointingJitter.offset_px`).  Ignored without a jitter model.
    sky_is_observed : bool
        Sky-brightness convention forwarded to :func:`sky_electrons_per_pixel`
        on the **clean path only** (see that function's docstring).  Default
        ``False`` (above-atmosphere) preserves byte-stable legacy numbers.
        Ignored on the realistic path (``detector`` set): ``render_frame``
        always uses the physically-correct observed convention directly via
        :func:`opta_model.radiometry.sky_background_electrons`, independent of
        this flag.

    Returns
    -------
    SynthFrame
        Frame data (uint16 ADU) plus ground-truth metadata.
    """
    if detector is not None:
        from opta_pipeline.synth.render import render_frame  # lazy: avoids cycle

        return render_frame(
            sensor,
            optics,
            detector,
            sky_mag_arcsec2=sky_mag_arcsec2,
            elevation_deg=elevation_deg,
            satellites=satellites,
            stars=stars,
            psf_fwhm_px=psf_fwhm_px,
            psf=psf,
            distortion=distortion,
            rolling_shutter=rolling_shutter,
            pointing_jitter=pointing_jitter,
            frame_index=frame_index,
            date_obs=date_obs,
            node_id=node_id,
            sensor_temp_c=sensor_temp_c,
            rng=rng,
        )

    if rng is None:
        rng = np.random.default_rng(0)
    if satellites is None:
        satellites = []
    if stars is None:
        stars = []

    width = sensor.resolution_h
    height = sensor.resolution_v
    aperture_m = optics.aperture_mm / 1000.0
    integration_time_s = 1.0 / sensor.frame_rate_hz
    pixel_scale = compute_pixel_scale(sensor.pixel_size_um, optics.focal_length_mm)

    # --- Noise floor ---
    sky_e = sky_electrons_per_pixel(
        sky_mag_arcsec2,
        pixel_scale,
        aperture_m,
        integration_time_s,
        sensor.quantum_efficiency,
        elevation_deg,
        sky_is_observed=sky_is_observed,
    )
    dark_e = sensor.dark_current_e_s * integration_time_s
    rn_e = sensor.readout_noise_e

    background_e = sky_e + dark_e
    frame_e = rng.poisson(lam=max(background_e, 1e-6), size=(height, width)).astype(
        np.float64
    )
    frame_e += rng.normal(0.0, rn_e, size=(height, width))

    ext = atmospheric_extinction(elevation_deg)

    shape = (height, width)

    # Scene-domain pointing jitter: a rigid offset of the *sky*, applied to
    # source coordinates only.  The noise floor above is already drawn, so it
    # is never resampled; sources below are rendered analytically at their
    # offset sub-pixel positions, so there is no interpolation blur either.
    jitter_dx, jitter_dy = (
        pointing_jitter.offset_px(frame_index)
        if pointing_jitter is not None
        else (0.0, 0.0)
    )

    def _distort(px: float, py: float) -> tuple[float, float]:
        """Map an ideal scene pixel to its as-imaged (jittered, distorted) position."""
        px += jitter_dx
        py += jitter_dy
        return distortion.apply(px, py, shape) if distortion is not None else (px, py)

    # --- Inject stars (at their distorted, as-imaged positions) ---
    star_truths: list[StarGroundTruth] = []
    for s in stars:
        sig_e = _star_total_signal_e(
            s, aperture_m, integration_time_s, sensor.quantum_efficiency,
            optics.transmission, ext,
        )
        xs, ys = _distort(s.x, s.y)
        _inject_psf(frame_e, xs, ys, sig_e, psf_fwhm_px)
        star_truths.append(
            StarGroundTruth(x=xs, y=ys, signal_electrons=sig_e, magnitude=s.magnitude)
        )

    # --- Inject satellites ---
    sat_truths: list[SatelliteGroundTruth] = []
    for sat in satellites:
        cx = sat.x_center if sat.x_center is not None else width / 2.0
        cy = sat.y_center if sat.y_center is not None else height / 2.0

        # Streak length in pixels
        trail_arcsec = sat.angular_velocity_deg_s * 3600.0 * integration_time_s
        trail_px = trail_arcsec / pixel_scale

        half = trail_px / 2.0
        angle_rad = math.radians(sat.angle_deg)
        dx = half * math.cos(angle_rad)
        dy = half * math.sin(angle_rad)

        # Distort endpoints/centre to their as-imaged positions.
        x0, y0 = _distort(cx - dx, cy - dy)
        x1, y1 = _distort(cx + dx, cy + dy)
        cxd, cyd = _distort(cx, cy)

        # Rolling-shutter readout: displace the moving streak by the per-row
        # timing bias (stars, being stationary, are untouched).  Truth stays at
        # the geometric centre (cxd, cyd) — the position the T-03 correction
        # must recover.
        if rolling_shutter is not None:
            v_px_s = sat.angular_velocity_deg_s * 3600.0 / pixel_scale
            vx_px_s = v_px_s * math.cos(angle_rad)
            vy_px_s = v_px_s * math.sin(angle_rad)
            x0, y0 = rolling_shutter.displace(x0, y0, vx_px_s, vy_px_s, shape)
            x1, y1 = rolling_shutter.displace(x1, y1, vx_px_s, vy_px_s, shape)

        # Per-pixel signal (total × τ_opt × trailing_loss), or free-injection.
        t_loss = trailing_loss(
            sat.angular_velocity_deg_s, pixel_scale, integration_time_s
        )
        n_pixels = max(1.0, trail_px)
        total_streak_e, per_pixel_signal_e = _sat_streak_signal_e(
            sat, aperture_m, integration_time_s, sensor.quantum_efficiency,
            optics.transmission, ext, n_pixels, t_loss,
        )
        _inject_streak(frame_e, x0, y0, x1, y1, total_streak_e, psf_fwhm_px)

        sat_truths.append(
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

    # --- Convert to ADU (uint16) ---
    # Preserve the pre-quantisation electron array for low-flux tests (WP-C3).
    frame_e_stored = frame_e.copy()
    frame_adu = np.clip(frame_e / gain_e_adu, 0, 2**16 - 1).astype(np.uint16)

    # --- FITS header ---
    if date_obs is None:
        date_obs = "2000-01-01T00:00:00.000"
    header: dict[str, Any] = {
        "DATE-OBS": date_obs,
        "NODE-ID": node_id,
        "EXPTIME": integration_time_s,
        "INSTRUME": "SYNTH",
        "GAIN": gain_e_adu,
        "TEMP": sensor_temp_c,
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "BITPIX": 16,
        "BSCALE": 1.0,
        "BZERO": 0.0,
        "PIXSCALE": pixel_scale,
        "SKYADU": sky_e / gain_e_adu,
    }

    return SynthFrame(
        data=frame_adu,
        data_float=frame_e_stored,
        header=header,
        satellites=tuple(sat_truths),
        stars=tuple(star_truths),
        sky_e_per_pixel=sky_e,
        readout_noise_e=rn_e,
        pixel_scale_arcsec=pixel_scale,
    )


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------


def write_fits(frame: SynthFrame, path: str | Path) -> Path:
    """Write a SynthFrame to a FITS file.

    Parameters
    ----------
    frame : SynthFrame
        Synthetic frame to write.
    path : str | Path
        Output file path.  Suffix should be ``.fits`` or ``.fits.gz``.

    Returns
    -------
    Path
        Resolved output path.
    """
    from astropy.io import fits  # lazy import — astropy is large

    hdu = fits.PrimaryHDU(data=frame.data)
    for key, val in frame.header.items():
        # FITS keys are 1-8 uppercase chars; hyphens are valid (e.g. DATE-OBS, NODE-ID).
        fits_key = key[:8].upper()
        hdu.header[fits_key] = val

    output = Path(path)
    hdu.writeto(output, overwrite=True)
    return output


# ---------------------------------------------------------------------------
# Realistic-renderer re-exports
# ---------------------------------------------------------------------------
# Imported at the bottom so render.py (which imports primitives from this
# package) sees a fully-initialised module — avoids a circular import.
from opta_pipeline.synth.render import (  # noqa: E402  (intentional late import)
    generate_dark_frame,
    generate_flat_frame,
    render_frame,
)
