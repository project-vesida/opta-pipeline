"""SGP4-driven synthetic pass generator (WP-A3).

Generates a sequence of SynthFrame objects whose satellite positions are derived
from a real TLE via Skyfield/SGP4 propagation, so the pipeline can be exercised
with realistic non-linear motion rather than the hand-coded straight-line
trajectories used in the integration test.

The pointing WCS is automatically centred on the satellite's midpass sky
position, ensuring the satellite crosses the FOV for all generated frames.
By default no catalog stars are injected (``stars=[]``); pass ``star_mag_limit``
to render a *stationary* star field (identical pixel positions every frame) into
the pixel data, and ``detector``/``psf``/``distortion``/``rolling_shutter`` to
use the realistic effects-chain renderer instead of the clean additive path.

Usage
-----
    from opta_pipeline.synth.pass_ import make_synthetic_pass, SyntheticPass
    from opta_model.geometry import Observer

    observer = Observer(latitude_deg=46.95, longitude_deg=7.44, elevation_m=540.0)
    synth_pass = make_synthetic_pass(
        "ISS (ZARYA)",
        "1 25544U 98067A   24001.50000000 ...",
        "2 25544  51.6400 ...",
        observer=observer,
        t_start_mjd=60310.5,
        n_frames=25,
        sensor=SENSOR_SMALL,
        optics=VILTROX_85_F14_PRESET,
    )
    for frame, gt in zip(synth_pass.frames, synth_pass.ground_truth):
        if gt.in_fov:
            ...  # run pipeline
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from opta_model._paths import DE421_PATH
from opta_model.geometry import Observer, load_tle_satellites
from opta_model.hardware import OpticsConfig, SensorConfig, compute_pixel_scale
from skyfield.api import EarthSatellite, wgs84

from opta_pipeline.astrometry import WCSSolution, radec_to_pixels
from opta_pipeline.synth import SatelliteSpec, StarSpec, SynthFrame, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov

if TYPE_CHECKING:
    from opta_pipeline.synth.detector import DetectorModel
    from opta_pipeline.synth.distortion import DistortionModel
    from opta_pipeline.synth.jitter import PointingJitter
    from opta_pipeline.synth.psf import PSFModel
    from opta_pipeline.synth.rolling_shutter import RollingShutterReadout

# Skyfield 1.54+ does not expose the Timescale through EarthSatellite.ts.
# Load it once from the same directory as opta_model uses for consistency.
_TS_CACHE = None


def _get_ts():
    """Return the cached Skyfield timescale used for synthetic passes."""
    global _TS_CACHE  # noqa: PLW0603
    if _TS_CACHE is None:
        from skyfield.api import Loader

        _TS_CACHE = Loader(str(DE421_PATH.parent)).timescale()
    return _TS_CACHE


__all__ = [
    "PassGroundTruth",
    "SyntheticPass",
    "make_synthetic_pass",
]

_MJD_EPOCH = datetime.datetime(1858, 11, 17, tzinfo=datetime.UTC)

# Mean sidereal rate: Earth rotates 360.9856°/solar day relative to the stars.
# On a fixed (non-tracking) alt-az mount the star field drifts at this rate, so
# stars are sharp per short exposure but move frame-to-frame across a stack.
_SIDEREAL_RATE_DEG_S = 360.9856235 / 86400.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassGroundTruth:
    """Per-frame ground truth for a satellite pass.

    Attributes
    ----------
    frame_id : int
        Sequential frame index within the pass (0-indexed).
    utc_mjd : float
        Frame mid-exposure time (Modified Julian Date, UTC scale).
    ra_deg : float
        True satellite RA at this frame (degrees, GCRS).
    dec_deg : float
        True satellite Dec at this frame (degrees, GCRS).
    x_px : float
        Pixel x of the satellite centre (may be outside FOV).
    y_px : float
        Pixel y of the satellite centre (may be outside FOV).
    ang_vel_deg_s : float
        Satellite angular velocity on the sky (deg/s).
    angle_deg : float
        Streak orientation in pixel space (degrees, 0 = horizontal, CCW +).
    in_fov : bool
        True if the satellite centre is within the sensor frame bounds.
    """

    frame_id: int
    utc_mjd: float
    ra_deg: float
    dec_deg: float
    x_px: float
    y_px: float
    ang_vel_deg_s: float
    angle_deg: float
    in_fov: bool


@dataclass(frozen=True)
class SyntheticPass:
    """A time-ordered sequence of synthetic frames driven by SGP4 geometry.

    Attributes
    ----------
    frames : tuple[SynthFrame, ...]
        One SynthFrame per time step.  Frames where ``in_fov=False`` contain
        no satellite injection (sky background and noise only).
    ground_truth : tuple[PassGroundTruth, ...]
        Per-frame ground truth (same length as ``frames``).
    wcs : WCSSolution
        Pointing WCS shared by all frames.  Built analytically from the
        satellite midpass sky position; rms_arcsec = 0.0, n_stars = 0.
    n_frames : int
        Number of frames in the pass.
    satellite_name : str
        TLE name field of the satellite.
    """

    frames: tuple[SynthFrame, ...] = field(hash=False, compare=False)
    ground_truth: tuple[PassGroundTruth, ...]
    wcs: WCSSolution
    n_frames: int
    satellite_name: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mjd_to_sfTime(ts, mjd: float):
    """Convert UTC MJD to a Skyfield Time object."""
    dt = _MJD_EPOCH + datetime.timedelta(days=mjd)
    return ts.from_datetime(dt)


def _pointing_wcs(
    ra0: float,
    dec0: float,
    pixel_scale_arcsec: float,
    width: int,
    height: int,
) -> WCSSolution:
    """Build an aligned TAN WCS centred on (ra0, dec0).

    Uses the same CD-matrix sign convention as the integration test:
    increasing x → increasing RA, increasing y → increasing Dec.
    """
    ps_deg = pixel_scale_arcsec / 3600.0
    return WCSSolution(
        crpix1=width / 2.0,
        crpix2=height / 2.0,
        crval1=ra0,
        crval2=dec0,
        cd1_1=ps_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=ps_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )


def _satellite_radec(
    satellite: EarthSatellite, observer_sf, mjd: float
) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of satellite as seen from observer at UTC MJD."""
    t = _mjd_to_sfTime(_get_ts(), mjd)
    topo = (satellite - observer_sf).at(t)
    ra, dec, _ = topo.radec()
    return ra.hours * 15.0, dec.degrees


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_synthetic_pass(
    tle_name: str,
    tle_line1: str,
    tle_line2: str,
    observer: Observer,
    t_start_mjd: float,
    n_frames: int,
    sensor: SensorConfig,
    optics: OpticsConfig,
    satellite_magnitude: float = 9.0,
    sky_mag_arcsec2: float = 21.0,
    elevation_deg: float = 45.0,
    rng: np.random.Generator | None = None,
    star_mag_limit: float | None = None,
    max_stars: int | None = None,
    gain_e_adu: float = 1.0,
    sidereal_drift: bool = False,
    detector: DetectorModel | None = None,
    psf: PSFModel | None = None,
    distortion: DistortionModel | None = None,
    rolling_shutter: RollingShutterReadout | None = None,
    pointing_jitter: PointingJitter | None = None,
    sky_is_observed: bool = False,
) -> SyntheticPass:
    """Generate a synthetic pass driven by SGP4/TLE propagation.

    The field pointing is automatically centred on the satellite's midpass
    sky position so the satellite transits the FOV for all generated frames.

    Parameters
    ----------
    tle_name : str
        TLE name line (e.g. 'ISS (ZARYA)').
    tle_line1, tle_line2 : str
        Standard two-line element set lines 1 and 2.
    observer : Observer
        Ground-observer location (lat/lon/elevation).
    t_start_mjd : float
        UTC start time for the first frame (Modified Julian Date).
    n_frames : int
        Number of frames to generate.
    sensor : SensorConfig
        Camera sensor parameters.
    optics : OpticsConfig
        Optics parameters (focal length, aperture).
    satellite_magnitude : float
        Apparent V-band magnitude used for signal injection.
    sky_mag_arcsec2 : float
        Sky surface brightness in mag/arcsec².
    elevation_deg : float
        Mean observation elevation for atmospheric extinction.
    rng : np.random.Generator | None
        Master RNG; spawns per-frame child generators for reproducibility.
        Seeded from 0 if None.
    star_mag_limit : float | None
        When set, render a stationary star field (catalog stars brighter than
        this V-mag) into every frame at fixed pixel positions.  ``None`` keeps
        the historical star-free behaviour.
    max_stars : int | None
        Keep only the ``max_stars`` brightest catalog stars.  The procedural
        catalog is far denser than the real sky at bright magnitudes, so over a
        wide FOV an uncapped field both dominates render time and (once masked)
        erases the target; capping to a few hundred bright stars is enough for
        plate-solving while leaving the frame clear for the mover.
    gain_e_adu : float
        Sensor gain (e⁻/ADU) passed through to ``generate_frame`` for ADU
        quantisation and FITS-header consistency.
    sidereal_drift : bool
        When ``True``, model a fixed (non-tracking) alt-az mount: the star field
        drifts across the frame at the sidereal rate (~0.68 px/s at 22″/px, a few
        px over a multi-second stack) while each frame stays individually sharp.
        The satellite track is unchanged, so the mover's motion *relative to the
        stars* is its apparent rate minus sidereal.  ``False`` (default) keeps the
        historical stationary field (sidereal-tracking mount).
    detector, psf, distortion, rolling_shutter : model | None
        Optional realistic-renderer models.  When ``detector`` is supplied,
        ``generate_frame`` switches from the clean additive path to the full
        effects chain (shot/read noise on collected charge, PRNU/vignette/dark,
        field-variable PSF, distortion, rolling-shutter displacement of movers).
    pointing_jitter : PointingJitter | None
        Scene-domain per-frame rigid pointing offset (wind / mount vibration).
        Applied to stars *and* mover together — the same whole-field rule
        ``sidereal_drift`` follows — upstream of every detector effect, so the
        instrumental signature stays fixed on the detector and the noise field
        is never resampled.  Frame *i* uses ``offset_px(i)``.  ``None`` (the
        default) is an exact no-op.
    sky_is_observed : bool
        Sky-brightness convention forwarded to ``generate_frame`` (clean path
        only; see ``sky_electrons_per_pixel``).  Default ``False``
        (above-atmosphere) keeps the historical behaviour.  Set ``True`` when
        the pass's synthetic sky must agree with an analytic
        ``opta_model.radiometry`` prediction. Ignored when ``detector`` is set.

    Returns
    -------
    SyntheticPass
        Frames, ground truth, and pointing WCS for the full pass.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # --- Load TLE ---
    sats = load_tle_satellites([tle_name, tle_line1, tle_line2])
    satellite = sats[0]

    observer_sf = wgs84.latlon(
        observer.latitude_deg,
        observer.longitude_deg,
        observer.elevation_m,
    )

    dt_s = 1.0 / sensor.frame_rate_hz
    dt_mjd = dt_s / 86400.0
    pixel_scale = compute_pixel_scale(sensor.pixel_size_um, optics.focal_length_mm)
    w, h = sensor.resolution_h, sensor.resolution_v

    # --- Pre-compute (ra, dec) for all frames + one-step lookahead ---
    # The lookahead enables forward-difference velocity at the last frame.
    n_total = n_frames + 1
    mjds = [t_start_mjd + i * dt_mjd for i in range(n_total)]
    radec_seq = [_satellite_radec(satellite, observer_sf, mjd) for mjd in mjds]

    # --- Determine pointing from midpass position ---
    mid_idx = n_frames // 2
    ra0, dec0 = radec_seq[mid_idx]
    wcs = _pointing_wcs(ra0, dec0, pixel_scale, w, h)

    # --- Star field ---
    # ``sidereal_drift=False`` keeps the historical stationary field (a sidereal-
    # tracking mount).  ``True`` models the fixed alt-az mount: stars are fixed on
    # the sky while the pointing advances at the sidereal rate, so they drift
    # across the frame (rebuilt per frame in the loop below).
    catalog_stars: list = []
    star_specs: list[StarSpec] = []
    if star_mag_limit is not None:
        catalog_stars = catalog_stars_in_fov(wcs, w, h, star_mag_limit)
        catalog_stars.sort(key=lambda s: s.magnitude)  # brightest first
        if max_stars is not None:
            catalog_stars = catalog_stars[:max_stars]
        star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    # --- Build frames ---
    frames_list: list[SynthFrame] = []
    gt_list: list[PassGroundTruth] = []

    for i in range(n_frames):
        ra, dec = radec_seq[i]
        ra_next, dec_next = radec_seq[i + 1]

        # Fixed-mount sidereal drift is a rigid translation of the *whole* field,
        # so the object shares the star drift: project through the drifting frame
        # (equivalently, shift RA by −rate·Δt against the fixed WCS).  ra_deg in
        # ground truth stays the true topocentric value for O−C comparison; only
        # the rendered pixel position drifts, keeping object and stars consistent
        # with the per-frame (drifting) WCS the pipeline solves.
        dra_deg = (
            _SIDEREAL_RATE_DEG_S * (i - mid_idx) * dt_s if sidereal_drift else 0.0
        )

        # Pixel centre via inverse WCS
        x_px, y_px = radec_to_pixels(wcs, ra - dra_deg, dec)
        in_fov = (0.0 <= x_px < w) and (0.0 <= y_px < h)

        # Angular velocity via forward differencing
        cos_dec = math.cos(math.radians(dec))
        v_ra_arcsec_s = (ra_next - ra + 180.0) % 360.0 - 180.0  # wrap RA diff
        v_ra_arcsec_s = v_ra_arcsec_s * cos_dec * 3600.0 / dt_s
        v_dec_arcsec_s = (dec_next - dec) * 3600.0 / dt_s
        v_total_arcsec_s = math.hypot(v_ra_arcsec_s, v_dec_arcsec_s)
        ang_vel_deg_s = v_total_arcsec_s / 3600.0

        # Streak orientation in pixel space
        # For our WCS (cd1_1 = cd2_2 = ps_deg, no rotation):
        #   v_x_px ∝ v_ra_arcsec_s,  v_y_px ∝ v_dec_arcsec_s
        angle_deg = math.degrees(math.atan2(v_dec_arcsec_s, v_ra_arcsec_s))

        utc_mjd = t_start_mjd + i * dt_mjd

        gt = PassGroundTruth(
            frame_id=i,
            utc_mjd=utc_mjd,
            ra_deg=ra,
            dec_deg=dec,
            x_px=x_px,
            y_px=y_px,
            ang_vel_deg_s=ang_vel_deg_s,
            angle_deg=angle_deg,
            in_fov=in_fov,
        )
        gt_list.append(gt)

        # Inject satellite only when it's within the frame
        sat_specs: list[SatelliteSpec] = []
        if in_fov and ang_vel_deg_s > 1e-4:
            sat_specs = [
                SatelliteSpec(
                    magnitude=satellite_magnitude,
                    angular_velocity_deg_s=ang_vel_deg_s,
                    x_center=x_px,
                    y_center=y_px,
                    angle_deg=angle_deg,
                )
            ]

        # Fixed-mount sidereal drift: the pointing advances in RA at the
        # sidereal rate, so a fixed-sky star's position relative to the (fixed)
        # frame WCS shifts by −rate·Δt.  Re-project the catalog per frame.
        frame_stars = star_specs
        if sidereal_drift and catalog_stars:
            dra_deg = _SIDEREAL_RATE_DEG_S * (i - mid_idx) / sensor.frame_rate_hz
            frame_stars = StarSpec.from_catalog(
                wcs,
                [replace(s, ra_deg=s.ra_deg - dra_deg) for s in catalog_stars],
            )

        # Per-frame RNG derived reproducibly from the master RNG
        frame_seed = int(rng.integers(0, 2**31))
        frame = generate_frame(
            sensor=sensor,
            optics=optics,
            sky_mag_arcsec2=sky_mag_arcsec2,
            elevation_deg=elevation_deg,
            satellites=sat_specs,
            stars=frame_stars,
            gain_e_adu=gain_e_adu,
            rng=np.random.default_rng(frame_seed),
            detector=detector,
            psf=psf,
            distortion=distortion,
            rolling_shutter=rolling_shutter,
            pointing_jitter=pointing_jitter,
            frame_index=i,
            sky_is_observed=sky_is_observed,
        )
        frames_list.append(frame)

    return SyntheticPass(
        frames=tuple(frames_list),
        ground_truth=tuple(gt_list),
        wcs=wcs,
        n_frames=n_frames,
        satellite_name=tle_name,
    )
