"""Blind plate-solve recover-truth harness (Phase 1, distortion-free).

This is the keystone end-to-end test: render frames with the generator using a
*real* catalog star field (placed through a known true WCS with an unknown
camera roll), hand the pipeline only an **approximate pointing hint** — *no*
``StarMatch`` inputs — and assert it recovers each frame's WCS from the image
itself (detect stars → catalog cross-match → fit) to within the OpTA.NOD.ACC
budget (≤ 10″).

It covers both orchestrator paths:

* the single-frame streak path (``run_frame`` / ``run_pipeline``), and
* the track-before-detect **stacking** path (``run_track_and_stack``),

since the blind solve must hold per-frame in both.  Distortion and rolling
shutter are added in later phases; here the field is distortion-free so a linear
TAN solve is expected to close comfortably.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL as _SENSOR
from sensor_fixtures import per_frame_only_config

from opta_pipeline.astrometry import WCSSolution, pixels_to_radec
from opta_pipeline.config import PipelineConfig
from opta_pipeline.match import PointingHint
from opta_pipeline.pipeline import (
    FrameContext,
    run_frame,
    run_pipeline,
    run_track_and_stack,
)
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov
from opta_pipeline.synth.distortion import DistortionModel

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(_SENSOR.pixel_size_um, _OPTICS.focal_length_mm)
_W, _H = _SENSOR.resolution_h, _SENSOR.resolution_v
_FPS = _SENSOR.frame_rate_hz
_RA0, _DEC0 = 135.0, 45.0
_COS_DEC = math.cos(math.radians(_DEC0))
_ROLL_DEG = 11.0  # unknown camera roll the blind solve must recover
_MJD0 = 60000.0
_NODE = "NODE-BLIND"

# Pointing hint deliberately imperfect: offset ~0.05° and 1 % scale error, no
# knowledge of the camera roll.  The blind solve must still recover truth.
_HINT = PointingHint(
    ra_deg=_RA0 + 0.05,
    dec_deg=_DEC0 - 0.03,
    pixel_scale_arcsec=_PIXEL_SCALE * 1.01,
    radius_deg=0.9,
    mag_limit=12.0,
)

# OpTA.NOD.ACC budget and a tighter regression gate for the distortion-free case.
_NOD_ACC_ARCSEC = 10.0
_REGRESSION_MEAN_ARCSEC = 3.0


def _true_wcs() -> WCSSolution:
    """True TAN WCS centred on (_RA0,_DEC0) with an unknown camera roll."""
    ps = _PIXEL_SCALE / 3600.0
    r = math.radians(_ROLL_DEG)
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=_RA0,
        crval2=_DEC0,
        cd1_1=ps * math.cos(r),
        cd1_2=-ps * math.sin(r),
        cd2_1=ps * math.sin(r),
        cd2_2=ps * math.cos(r),
        rms_arcsec=0.0,
        n_stars=0,
    )


def _star_specs(true: WCSSolution) -> list[StarSpec]:
    """Render the catalog star field through the true WCS (≤ mv 11.5)."""
    catalog = catalog_stars_in_fov(true, _W, _H, mag_limit=11.5)
    return StarSpec.from_catalog(true, catalog)


def _sky_err(ra: float, dec: float, ra_true: float, dec_true: float) -> float:
    """Great-circle error between recovered and true sky position (arcsec)."""
    return math.hypot(
        (ra - ra_true) * _COS_DEC * 3600.0,
        (dec - dec_true) * 3600.0,
    )


# ---------------------------------------------------------------------------
# Single-frame streak path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streak_sequence() -> dict:
    """8 frames, no StarMatch supplied, a clear satellite streak per frame."""
    true = _true_wcs()
    stars = _star_specs(true)
    n_frames = 8
    dx_per_frame = 8.0
    frame_pairs: list[tuple[np.ndarray, FrameContext]] = []
    sat_centers: list[tuple[float, float]] = []
    for i in range(n_frames):
        cx = _W / 2.0 - (n_frames / 2.0) * dx_per_frame + i * dx_per_frame
        cy = _H / 2.0
        sat_centers.append((cx, cy))
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            satellites=[
                SatelliteSpec(
                    magnitude=7.0,
                    angular_velocity_deg_s=1.2,  # clear streak at 25 fps
                    x_center=cx,
                    y_center=cy,
                    angle_deg=0.0,
                )
            ],
            stars=stars,
            rng=np.random.default_rng(4000 + i),
        )
        ctx = FrameContext(
            utc_mjd=_MJD0 + i / _FPS / 86400.0,
            frame_id=i,
            node_id=_NODE,
            pointing=_HINT,  # blind: no star_matches
        )
        frame_pairs.append((synth.data, ctx))

    return {"frames": frame_pairs, "true": true, "sat_centers": sat_centers}


class TestBlindSingleFrame:
    def test_no_star_matches_supplied(self, streak_sequence: dict) -> None:
        """Acceptance: the pipeline runs with ctx.star_matches empty."""
        _raw, ctx = streak_sequence["frames"][0]
        assert ctx.star_matches == []
        assert ctx.pointing is not None

    def test_blind_solve_recovers_satellite_radec(
        self, streak_sequence: dict
    ) -> None:
        """Blind detect→match→solve recovers the streak RA/Dec to ≤ 10″."""
        true = streak_sequence["true"]
        errs: list[float] = []
        detected = 0
        for (raw, ctx), (cx, cy) in zip(
            streak_sequence["frames"], streak_sequence["sat_centers"]
        ):
            fd = run_frame(raw, ctx)
            if fd is None:
                continue
            detected += 1
            best = min(
                fd.detections,
                key=lambda a: math.hypot(a.detection.x - cx, a.detection.y - cy),
            )
            ra_t, dec_t = pixels_to_radec(true, cx, cy)
            errs.append(_sky_err(best.ra_deg, best.dec_deg, ra_t, dec_t))

        assert detected >= 7, f"blind-solved only {detected}/8 frames"
        assert errs
        assert max(errs) < _NOD_ACC_ARCSEC, (
            f"max blind astrometric error {max(errs):.2f}\" exceeds OpTA.NOD.ACC"
        )
        assert float(np.mean(errs)) < _REGRESSION_MEAN_ARCSEC, (
            f"mean blind error {np.mean(errs):.2f}\" — distortion-free regression"
        )

    def test_blind_tracklet_formed(self, streak_sequence: dict) -> None:
        """The blind chain links a satellite tracklet with no supplied matches."""
        # The blind per-frame chain is the subject; the stacked search is a
        # parallel path costing tens of seconds per call.
        tracklets = run_pipeline(
            streak_sequence["frames"], config=per_frame_only_config()
        )
        assert any(len(t.points) >= 5 for t in tracklets), (
            f"no tracklet ≥5 points (lengths {[len(t.points) for t in tracklets]})"
        )


# ---------------------------------------------------------------------------
# Track-before-detect stacking path
# ---------------------------------------------------------------------------

_N_FRAMES_STACK = 24


def _stack_config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=120.0,
            velocity_step_px_s=120.0,
        ),
    )


@pytest.fixture(scope="module")
def stack_sequence() -> dict:
    """24-frame faint-dot sequence, blind pointing, for the stacking path."""
    true = _true_wcs()
    stars = _star_specs(true)
    n_frames = _N_FRAMES_STACK
    vx_px_s = 120.0
    angular_velocity_deg_s = vx_px_s * _PIXEL_SCALE / 3600.0
    reference_s = ((n_frames - 1) / 2.0) / _FPS

    frame_pairs: list[tuple[np.ndarray, FrameContext]] = []
    for frame_id in range(n_frames):
        t_s = frame_id / _FPS
        cx = _W / 2.0 + vx_px_s * (t_s - reference_s)
        cy = _H / 2.0
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            satellites=[
                SatelliteSpec(
                    magnitude=12.0,
                    angular_velocity_deg_s=angular_velocity_deg_s,
                    x_center=cx,
                    y_center=cy,
                    angle_deg=0.0,
                )
            ],
            stars=stars,
            rng=np.random.default_rng(9000 + frame_id),
        )
        ctx = FrameContext(
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
            pointing=_HINT,  # blind: no star_matches
        )
        frame_pairs.append((synth.data_float, ctx))

    cfg = _stack_config()
    result = run_track_and_stack(frame_pairs, config=cfg)
    return {"true": true, "vx_px_s": vx_px_s, "cfg": cfg, "result": result}


class TestBlindStackingPath:
    def test_blind_stack_recovers_velocity_and_tracklet(
        self, stack_sequence: dict
    ) -> None:
        """Per-frame blind solve + shift-and-add finds the source and links it."""
        result = stack_sequence["result"]

        assert result.stack_result.peak_snr >= 5.0
        # Coarse-to-fine resolves to the full-pass criterion step, not to
        # a grid node (assessment F1; Phase 2).
        assert result.stack_result.best_vx_px_s == pytest.approx(
            stack_sequence["vx_px_s"], abs=3.0
        )
        assert any(len(t.points) >= 5 for t in result.tracklets), (
            "blind stacking produced no linked tracklet"
        )

    def test_blind_stack_astrometry_accurate(self, stack_sequence: dict) -> None:
        """Tracklet sky positions from the blind per-frame WCS meet OpTA.NOD.ACC."""
        result = stack_sequence["result"]
        true = stack_sequence["true"]
        tracklet = max(result.tracklets, key=lambda t: len(t.points))

        errs: list[float] = []
        for pt in tracklet.points:
            x = _W / 2.0 + stack_sequence["vx_px_s"] * (
                (pt.utc_mjd - _MJD0) * 86400.0
                - ((_N_FRAMES_STACK - 1) / 2.0) / _FPS
            )
            ra_t, dec_t = pixels_to_radec(true, x, _H / 2.0)
            errs.append(_sky_err(pt.ra_deg, pt.dec_deg, ra_t, dec_t))

        assert float(np.mean(errs)) < _NOD_ACC_ARCSEC, (
            f"blind stacked tracklet mean error {np.mean(errs):.2f}\" exceeds budget"
        )


# ---------------------------------------------------------------------------
# Blind solve under optical distortion (Phase 2: SIP)
# ---------------------------------------------------------------------------

_R_HALF = math.hypot(_W / 2.0, _H / 2.0)
# A realistic 5 % barrel for the wide, fast prime — a linear TAN cannot absorb
# it at the field edge (see test_synth_distortion); the SIP solve must.
# Raised from 3 % with the 2.9 µm pitch unification (MA-002): the finer plate
# scale maps the same fractional barrel to fewer arcsec, and at 3 % the linear
# solve no longer exceeded the 10″ budget (max ≈ 7.5″), so the regression
# lost its discriminating power.  Cheap fast primes commonly measure 3–5 %.
_BARREL = DistortionModel.from_corner_displacement_px(-0.05 * _R_HALF, (_H, _W))


def _run_distorted_streaks(sip: bool) -> tuple[int, list[float]]:
    """Blind-solve an off-centre streak sequence imaged through a 3 % barrel.

    Renders the catalog star field *and* the satellite through the optic's
    radial distortion, then blind-solves each frame (no ``StarMatch``) with the
    distortion-aware solver toggled by ``sip``.  Returns ``(n_detected, errs)``
    where ``errs`` are great-circle errors of the recovered streak RA/Dec vs
    truth (the object's true sky is its *ideal*, undistorted pixel through the
    true linear WCS).  The track sits at field radius ≈ 0.63, where a linear
    solve's uncorrected distortion is large.
    """
    true = _true_wcs()
    stars = _star_specs(true)
    cfg = replace(
        PipelineConfig.default(),
        astrometry=replace(PipelineConfig.default().astrometry, sip_distortion=sip),
    )
    n_frames = 8
    dx_per_frame = 7.0
    errs: list[float] = []
    detected = 0
    for i in range(n_frames):
        cx = _W * 0.78 - (n_frames / 2.0) * dx_per_frame + i * dx_per_frame
        cy = _H / 2.0 + 110.0
        synth = generate_frame(
            sensor=_SENSOR,
            optics=_OPTICS,
            sky_mag_arcsec2=21.0,
            satellites=[
                SatelliteSpec(
                    magnitude=7.0,
                    angular_velocity_deg_s=1.2,
                    x_center=cx,
                    y_center=cy,
                    angle_deg=0.0,
                )
            ],
            stars=stars,
            distortion=_BARREL,
            rng=np.random.default_rng(4000 + i),
        )
        ctx = FrameContext(
            utc_mjd=_MJD0 + i / _FPS / 86400.0,
            frame_id=i,
            node_id=_NODE,
            pointing=_HINT,
        )
        fd = run_frame(synth.data, ctx, cfg)
        if fd is None:
            continue
        detected += 1
        ra_t, dec_t = pixels_to_radec(true, cx, cy)
        best = min(
            fd.detections,
            key=lambda a: math.hypot(a.detection.x - cx, a.detection.y - cy),
        )
        errs.append(_sky_err(best.ra_deg, best.dec_deg, ra_t, dec_t))
    return detected, errs


@pytest.fixture(scope="module")
def distorted_sip() -> tuple[int, list[float]]:
    return _run_distorted_streaks(sip=True)


@pytest.fixture(scope="module")
def distorted_linear() -> tuple[int, list[float]]:
    return _run_distorted_streaks(sip=False)


class TestBlindSolveUnderDistortion:
    def test_sip_recovers_within_budget(
        self, distorted_sip: tuple[int, list[float]]
    ) -> None:
        """Blind SIP solve meets OpTA.NOD.ACC under a 3 % barrel at the edge."""
        detected, errs = distorted_sip
        assert detected >= 7, f"distorted blind-solve only {detected}/8 frames"
        assert errs
        assert max(errs) < _NOD_ACC_ARCSEC, (
            f"SIP blind error {max(errs):.2f}\" exceeds OpTA.NOD.ACC under distortion"
        )
        assert float(np.mean(errs)) < _REGRESSION_MEAN_ARCSEC

    def test_linear_solve_fails_where_sip_succeeds(
        self,
        distorted_sip: tuple[int, list[float]],
        distorted_linear: tuple[int, list[float]],
    ) -> None:
        """The regression: at this field radius the linear solve blows the 10″
        budget on the very frames the SIP solve recovers — distortion-awareness
        is mandatory, not optional."""
        _, sip_errs = distorted_sip
        _, lin_errs = distorted_linear
        assert max(lin_errs) > _NOD_ACC_ARCSEC  # linear fails the budget
        # SIP is better by a wide margin (here ~40×).
        assert float(np.mean(sip_errs)) < 0.25 * float(np.mean(lin_errs))
