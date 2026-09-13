"""OpTA.NOD.ACC ``accuracy_flag`` consumption-policy tests.

``fit_wcs_sip``'s post-solve self-check (commit c90efcc) sets
``WCSSolution.accuracy_flag`` when the returned solution's honest unpruned
residuals exceed the 10″ budget.  These tests pin the *consumption* wiring:

* the flag propagates into ``FrameDetections.wcs_accuracy_flag`` on both the
  per-frame (``run_frame``) and the stacked forced-photometry
  (``run_track_and_stack``) paths;
* ``link_detections`` annotates every tracklet containing a flagged frame
  (``Tracklet.wcs_accuracy_flagged``) — annotate-only, nothing is dropped;
* ``AstrometryConfig.skip_flagged_frames`` (default OFF) opts into excluding
  flagged frames from astrometry, without touching detection.

The flag is tripped honestly: star-match pixel positions are jittered by a
seeded ~2.5 px (≈ 18″ at this plate scale) so the fitted solution's unpruned
residual RMS genuinely exceeds the budget — no monkeypatching of the check.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL as _SENSOR

from opta_pipeline.astrometry import (
    AstrometricDetection,
    StarMatch,
    fit_wcs_sip,
)
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import Detection
from opta_pipeline.pipeline import FrameContext, run_frame, run_track_and_stack
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.tracklet import FrameDetections, link_detections

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(_SENSOR.pixel_size_um, _OPTICS.focal_length_mm)
_W, _H = _SENSOR.resolution_h, _SENSOR.resolution_v
_CX, _CY = _W / 2.0, _H / 2.0
_RA0, _DEC0 = 135.0, 45.0
_MJD0 = 60000.0
_NODE = "NODE-ACC"

# 4×3 star grid clear of the satellite path at y ≈ 150.
_STAR_XS = [60.0, 140.0, 260.0, 340.0]
_STAR_YS = [50.0, 100.0, 250.0]
_JITTER_PX = 2.5  # ≈ 18″ at 7.0″/px — comfortably beyond the 10″ budget


def _px_to_sky(x: float, y: float) -> tuple[float, float]:
    """Exact gnomonic (TAN) truth WCS, independent of astrometry.py."""
    xi = math.radians(_PIXEL_SCALE * (x - _CX) / 3600.0)
    eta = math.radians(_PIXEL_SCALE * (y - _CY) / 3600.0)
    dec0 = math.radians(_DEC0)
    sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
    denom = cos_d0 - eta * sin_d0
    ra = _RA0 + math.degrees(math.atan2(xi, denom))
    dec = math.degrees(math.atan2(sin_d0 + eta * cos_d0, math.hypot(xi, denom)))
    return ra, dec


def _star_matches(jitter_px: float = 0.0, seed: int = 42) -> list[StarMatch]:
    """Gnomonic-consistent star grid; optional seeded pixel jitter.

    With jitter the catalog (ra, dec) stay truthy but the measured pixel
    positions are displaced, so the fitted plate solution carries an honest
    unpruned residual RMS ≈ jitter × plate scale.
    """
    rng = np.random.default_rng(seed)
    out: list[StarMatch] = []
    for x in _STAR_XS:
        for y in _STAR_YS:
            ra, dec = _px_to_sky(x, y)
            jx = float(rng.normal(0.0, jitter_px)) if jitter_px > 0 else 0.0
            jy = float(rng.normal(0.0, jitter_px)) if jitter_px > 0 else 0.0
            out.append(
                StarMatch(x_px=x + jx, y_px=y + jy, ra_deg=ra, dec_deg=dec)
            )
    return out


def test_jittered_matches_trip_the_flag() -> None:
    """Precondition for everything below: the jittered grid is out of budget."""
    clean = fit_wcs_sip(_star_matches(), frame_shape=(_H, _W))
    assert not clean.accuracy_flag
    jittered = fit_wcs_sip(_star_matches(_JITTER_PX), frame_shape=(_H, _W))
    assert jittered.accuracy_flag


# ---------------------------------------------------------------------------
# link_detections annotation (pure linker level)
# ---------------------------------------------------------------------------


def _frame(
    i: int, ra_step_deg: float = 0.01, flagged: bool = False
) -> FrameDetections:
    det = Detection(
        x=100.0 + i,
        y=150.0,
        snr=20.0,
        elongation=3.0,
        angle_deg=0.0,
        n_pixels=10,
        flux_e=1000.0,
        is_streak=True,
        fwhm_px=2.0,
    )
    astro = AstrometricDetection(
        detection=det,
        ra_deg=_RA0 + i * ra_step_deg,
        dec_deg=_DEC0,
        sigma_ra_arcsec=1.0,
        sigma_dec_arcsec=1.0,
    )
    return FrameDetections(
        detections=(astro,),
        utc_mjd=_MJD0 + i * 0.04 / 86400.0,
        frame_id=i,
        node_id=_NODE,
        wcs_accuracy_flag=flagged,
    )


def test_linker_annotates_tracklet_containing_flagged_frame() -> None:
    frames = [_frame(0), _frame(1, flagged=True), _frame(2)]
    tracklets = link_detections(frames)
    assert len(tracklets) == 1
    assert tracklets[0].wcs_accuracy_flagged


def test_linker_leaves_clean_tracklet_unannotated() -> None:
    frames = [_frame(0), _frame(1), _frame(2)]
    tracklets = link_detections(frames)
    assert len(tracklets) == 1
    assert not tracklets[0].wcs_accuracy_flagged


def test_linker_annotates_flag_on_seed_frame() -> None:
    """The flag is recorded on the tracklet-seeding path too, not only when
    a detection is matched to an existing track."""
    frames = [_frame(0, flagged=True), _frame(1), _frame(2)]
    tracklets = link_detections(frames)
    assert len(tracklets) == 1
    assert tracklets[0].wcs_accuracy_flagged


def test_flag_identification_survives_timestamp_collision() -> None:
    """PR #135 P3b regression: flagged frames are identified by the frame in
    hand at point creation, not by float ``utc_mjd`` set membership.  A
    flagged frame that happens to share its timestamp with an unflagged
    frame (e.g. truncated header times) must not contaminate tracklets that
    contain no point from it — and conversely the annotation cannot be
    dropped by float timestamp round-trips."""
    clean = [_frame(0), _frame(1), _frame(2)]
    # Flagged frame sharing frame 2's exact utc_mjd; its lone detection is
    # ~5 deg away, so it can never associate with the clean track.
    collider = FrameDetections(
        detections=(
            AstrometricDetection(
                detection=Detection(
                    x=10.0,
                    y=10.0,
                    snr=20.0,
                    elongation=3.0,
                    angle_deg=0.0,
                    n_pixels=10,
                    flux_e=1000.0,
                    is_streak=True,
                    fwhm_px=2.0,
                ),
                ra_deg=_RA0 + 5.0,
                dec_deg=_DEC0,
                sigma_ra_arcsec=1.0,
                sigma_dec_arcsec=1.0,
            ),
        ),
        utc_mjd=clean[2].utc_mjd,  # exact float collision
        frame_id=3,
        node_id=_NODE,
        wcs_accuracy_flag=True,
    )
    tracklets = link_detections(clean + [collider])
    assert len(tracklets) == 1  # collider's 1-point track dies at min_points
    assert not tracklets[0].wcs_accuracy_flagged


# ---------------------------------------------------------------------------
# run_frame propagation + skip policy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streak_frame() -> np.ndarray:
    """One synthetic frame with a bright, elongated satellite streak."""
    synth = generate_frame(
        sensor=_SENSOR,
        optics=_OPTICS,
        sky_mag_arcsec2=21.0,
        elevation_deg=45.0,
        satellites=[
            SatelliteSpec(
                magnitude=9.0,
                angular_velocity_deg_s=0.5,
                x_center=_CX,
                y_center=_CY,
                angle_deg=0.0,
            )
        ],
        stars=[
            StarSpec(magnitude=9.5, x=x, y=y) for x in _STAR_XS for y in _STAR_YS
        ],
        rng=np.random.default_rng(1234),
    )
    return synth.data


def _ctx(matches: list[StarMatch], frame_id: int = 0) -> FrameContext:
    return FrameContext(
        star_matches=matches,
        utc_mjd=_MJD0 + frame_id * 0.04 / 86400.0,
        frame_id=frame_id,
        node_id=_NODE,
    )


def test_run_frame_propagates_flag_annotate_only(streak_frame: np.ndarray) -> None:
    """Default policy: flagged frame still emits astrometry, carrying the flag."""
    fd = run_frame(streak_frame, _ctx(_star_matches(_JITTER_PX)))
    assert fd is not None
    assert fd.wcs_accuracy_flag
    assert len(fd.detections) >= 1


def test_run_frame_clean_solution_unflagged(streak_frame: np.ndarray) -> None:
    fd = run_frame(streak_frame, _ctx(_star_matches()))
    assert fd is not None
    assert not fd.wcs_accuracy_flag


def test_run_frame_linear_path_runs_self_check(streak_frame: np.ndarray) -> None:
    """``sip_distortion: false`` must not skip the OpTA.NOD.ACC self-check.

    The linear path used to return the bare ``fit_wcs`` result, so an
    out-of-budget solution sailed through unflagged."""
    cfg = PipelineConfig.default()
    cfg = replace(cfg, astrometry=replace(cfg.astrometry, sip_distortion=False))
    fd = run_frame(streak_frame, _ctx(_star_matches(_JITTER_PX)), cfg)
    assert fd is not None
    assert fd.wcs_accuracy_flag
    # Clean solutions on the linear path stay unflagged.
    fd = run_frame(streak_frame, _ctx(_star_matches()), cfg)
    assert fd is not None and not fd.wcs_accuracy_flag


def test_run_frame_skip_flagged_frames_drops_frame(
    streak_frame: np.ndarray,
) -> None:
    cfg = PipelineConfig.default()
    cfg = replace(cfg, astrometry=replace(cfg.astrometry, skip_flagged_frames=True))
    assert run_frame(streak_frame, _ctx(_star_matches(_JITTER_PX)), cfg) is None
    # Clean solutions are untouched by the skip policy.
    fd = run_frame(streak_frame, _ctx(_star_matches()), cfg)
    assert fd is not None and not fd.wcs_accuracy_flag


# ---------------------------------------------------------------------------
# Stacked path (run_track_and_stack) propagation + skip policy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack_pass() -> list[tuple[np.ndarray, FrameContext]]:
    """12-frame pass with a slow bright mover for the stacked path."""
    vx_px_s = 10.0  # inside the ±20 px/s test search band
    fps = 25.0
    frames: list[tuple[np.ndarray, FrameContext]] = []
    for i in range(12):
        t_s = i / fps
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[
                SatelliteSpec(
                    magnitude=9.0,
                    angular_velocity_deg_s=vx_px_s * _PIXEL_SCALE / 3600.0,
                    x_center=180.0 + vx_px_s * t_s,
                    y_center=150.0,
                    angle_deg=0.0,
                )
            ],
            stars=[
                StarSpec(magnitude=9.5, x=x, y=y)
                for x in _STAR_XS
                for y in _STAR_YS
            ],
            rng=np.random.default_rng(5000 + i),
        )
        # ctx matches are attached per test below.
        frames.append((synth.data_float, i))  # type: ignore[arg-type]
    return frames


def _stack_config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=20.0,
            velocity_step_px_s=10.0,
        ),
    )


def _with_matches(
    stack_pass: list, matches: list[StarMatch]
) -> list[tuple[np.ndarray, FrameContext]]:
    return [
        (data, _ctx(matches, frame_id=i)) for data, i in stack_pass
    ]


def test_stacked_path_annotates_tracklets(stack_pass: list) -> None:
    result = run_track_and_stack(
        _with_matches(stack_pass, _star_matches(_JITTER_PX)),
        config=_stack_config(),
    )
    assert len(result.tracklets) >= 1
    assert all(t.wcs_accuracy_flagged for t in result.tracklets)


def test_stacked_path_skip_drops_astrometry_not_detection(
    stack_pass: list,
) -> None:
    cfg = _stack_config()
    cfg = replace(cfg, astrometry=replace(cfg.astrometry, skip_flagged_frames=True))
    result = run_track_and_stack(
        _with_matches(stack_pass, _star_matches(_JITTER_PX)), config=cfg
    )
    # Detection (pixel space) is unaffected by the astrometric skip…
    assert len(result.stacked_detections) >= 1
    assert result.stack_result.peak_snr > 0.0
    # …but every frame's astrometry was excluded, so no tracklets emit.
    assert result.frame_detections == ()
    assert result.tracklets == ()


def test_stacked_path_clean_solution_unflagged(stack_pass: list) -> None:
    result = run_track_and_stack(
        _with_matches(stack_pass, _star_matches()), config=_stack_config()
    )
    assert len(result.tracklets) >= 1
    assert not any(t.wcs_accuracy_flagged for t in result.tracklets)
