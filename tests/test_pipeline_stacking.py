"""Batch track-before-detect tests for opta_pipeline.pipeline.

These tests exercise the operational case that single-frame streak detection
cannot close: a faint LEO-rate source recovered by blind shift-and-add, then
expanded back into astrometric tracklets through the normal I-02 linker.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline import pipeline as pipeline_mod
from opta_pipeline.astrometry import WCSSolution, pixels_to_radec
from opta_pipeline.catalog import ProceduralCatalog
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import detect_sources
from opta_pipeline.pipeline import (
    FrameContext,
    PipelineStream,
    run_pipeline,
    run_track_and_stack,
)
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at
from opta_pipeline.tracklet import Tracklet

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = SENSOR_SMALL.frame_rate_hz
_RA0 = 135.0
_DEC0 = 45.0
_NODE = "NODE-STACK"
_MJD0 = 60000.0


def _wcs() -> WCSSolution:
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=_RA0,
        crval2=_DEC0,
        cd1_1=scale_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=scale_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )


def _stack_config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=200.0,
            velocity_step_px_s=200.0,
            # Coarse-to-fine seeding depth.  Stage-1 seeds a track only if
            # its per-window matched-filter SNR clears the window's own
            # extreme-value noise floor sqrt(2 ln(W*H)) ~= 4.84 sigma.  This
            # mv13 mover is fast (200 px/s -> ~8 px within-exposure trail per
            # frame), so its per-frame source is a streak, not the round
            # FWHM=2 px synth PSF; the synth-PSF-matched round filter
            # (psf_sigma_px ~= 0.85, = psf_fwhm 2.0 / 2.355; the correct
            # matched value for the completeness workload, PR #<psf>) captures
            # less of that trail per window than the old too-wide 1.5 px kernel
            # did, dropping the 9-frame (0.36 s) seed to ~4.75 sigma -- below
            # the floor -- so it seeded nothing (all-zeros stack, peak_snr=0).
            # The principled lever is stacking depth (root AGENTS.md design
            # rule 10), not a wider mismatched kernel: 15-frame (0.60 s)
            # windows restore a best-window seed ~5.86 sigma (~1.0 sigma over
            # the floor).  Grid cost scales as (v_max*T_win/PSF)^2, so keep the
            # window only as deep as needed.
            seed_window_s=0.60,
        ),
    )


@pytest.fixture(scope="module")
def faint_stack_sequence() -> dict:
    """Return a 50-frame mv=13 sequence with deterministic catalog stars."""
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=14.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    n_frames = 50
    vx_px_s = 200.0
    vy_px_s = 0.0
    angular_velocity_deg_s = vx_px_s * _PIXEL_SCALE / 3600.0
    reference_s = ((n_frames - 1) / 2.0) / _FPS
    x_ref = _W / 2.0
    y_ref = _H / 2.0

    frame_pairs = []
    for frame_id in range(n_frames):
        t_s = frame_id / _FPS
        sat = SatelliteSpec(
            magnitude=13.0,
            angular_velocity_deg_s=angular_velocity_deg_s,
            x_center=x_ref + vx_px_s * (t_s - reference_s),
            y_center=y_ref + vy_px_s * (t_s - reference_s),
            angle_deg=0.0,
        )
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[sat],
            stars=star_specs,
            rng=np.random.default_rng(9000 + frame_id),
        )
        ctx = FrameContext(
            star_matches=star_matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        frame_pairs.append((synth.data_float, ctx))

    return {
        "frames": frame_pairs,
        "vx_px_s": vx_px_s,
        "angular_velocity_deg_s": angular_velocity_deg_s,
    }


def test_single_frame_path_does_not_recover_mv13(
    faint_stack_sequence: dict,
) -> None:
    """The chosen source is below the single-frame connected-component gate."""
    raw, _ctx = faint_stack_sequence["frames"][25]
    dets = detect_sources(raw - np.median(raw), noise_rms=1.0, snr_threshold=3.0)
    streaks = [det for det in dets if det.is_streak]
    assert not streaks


def test_run_track_and_stack_recovers_faint_tracklet(
    faint_stack_sequence: dict,
) -> None:
    """Blind stacking finds the true velocity and returns I-02 tracklets."""
    cfg = _stack_config()
    result = run_track_and_stack(faint_stack_sequence["frames"], config=cfg)

    assert result.stack_result.peak_snr >= 5.0
    # Coarse-to-fine resolves velocity to the full-pass criterion step
    # (PSF_fwhm/T ≈ 1.8 px/s here), not to a grid node; response_grid is a
    # single-stage diagnostic and is None in coarse-to-fine mode.
    assert result.stack_result.best_vx_px_s == pytest.approx(
        faint_stack_sequence["vx_px_s"], abs=3.0
    )
    assert result.stack_result.best_vy_px_s == pytest.approx(0.0, abs=3.0)
    assert len(result.tracklets) >= 1

    best = max(result.tracklets, key=lambda t: len(t.points))
    assert len(best.points) >= cfg.tracklet.min_detections
    assert best.node_id == _NODE
    assert best.ra_rate_arcsec_s > 0.0
    assert abs(best.ra_rate_arcsec_s) == pytest.approx(
        faint_stack_sequence["angular_velocity_deg_s"] * 3600.0,
        rel=0.10,
    )


def _plumbing_config() -> PipelineConfig:
    """Single-stage config for the run_pipeline/stream plumbing tests.

    These tests verify the run_pipeline/stream path-union MECHANISM, not
    the blind search algorithm (covered by test_coarse_fine.py,
    test_run_track_and_stack_recovers_faint_tracklet, and
    test_pipeline_likelihood.py).  The blind coarse-to-fine costs tens of
    seconds per invocation at this fixture scale (stage-1 hot loop is
    pure Python — see TODO.md perf item), so the plumbing tests pin
    coarse_to_fine=False; the 200 px/s sources sit on the 3x3 single-stage
    grid node.
    """
    cfg = _stack_config()
    return replace(cfg, stacking=replace(cfg.stacking, coarse_to_fine=False))


def _per_frame_only_config() -> PipelineConfig:
    """`_plumbing_config` with the stacked path off — per-frame result alone."""
    cfg = _plumbing_config()
    return replace(cfg, stacking=replace(cfg.stacking, enabled=False))


def test_run_pipeline_recovers_faint_source_via_stacked_path(
    faint_stack_sequence: dict,
) -> None:
    """Batch entry point recovers faint tracks the per-frame linker misses."""
    tracklets = run_pipeline(faint_stack_sequence["frames"], config=_plumbing_config())
    assert len(tracklets) >= 1
    assert max(len(t.points) for t in tracklets) >= 3


def test_pipeline_stream_flush_can_stack_faint_sources(
    faint_stack_sequence: dict,
) -> None:
    """Streaming flush retains raw frames for stack-backed recovery."""
    stream = PipelineStream(config=_plumbing_config())
    for raw, ctx in faint_stack_sequence["frames"]:
        assert stream.push(raw, ctx) == []

    tracklets = stream.flush()
    assert len(tracklets) >= 1
    assert all(t.node_id == _NODE for t in tracklets)


# ---------------------------------------------------------------------------
# Union of the per-frame and stacked detection paths
#
# The stacked search is a PARALLEL path, not a fallback for an empty per-frame
# result.  Under the old "return early if the linker found anything" rule a
# single bright mover suppressed the faint-mover search for the whole pass.
# ---------------------------------------------------------------------------

#: Cross-track separation of the two movers in the mixed scene.  60 px is
#: ~422" at this plate scale — far outside the 30" duplicate-identity gate
#: (so the two objects can never dedup into one) and far enough that the
#: bright streak's wings do not sit on the faint one.
_MOVER_DY_PX = 60.0

#: Both movers share this along-track rate so they land on the same node of
#: the coarse 3x3 single-stage velocity grid (`_plumbing_config`).
_VX_PX_S = 200.0

#: Tolerance for matching a tracklet to a mover by declination.  Generous
#: against the arc-mean-vs-midpass TAN curvature (a few arcsec here) while
#: still ~7x smaller than the 422" mover separation it must discriminate.
_DEC_MATCH_ARCSEC = 60.0


def _render_movers(
    n_frames: int,
    movers: list[tuple[float, float]],
    seed0: int,
) -> list[tuple[np.ndarray, FrameContext]]:
    """Render ``n_frames`` with one +x mover per ``(magnitude, dy_px)`` entry."""
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=14.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    angular_velocity_deg_s = _VX_PX_S * _PIXEL_SCALE / 3600.0
    reference_s = ((n_frames - 1) / 2.0) / _FPS

    frame_pairs: list[tuple[np.ndarray, FrameContext]] = []
    for frame_id in range(n_frames):
        t_s = frame_id / _FPS
        sats = [
            SatelliteSpec(
                magnitude=mag,
                angular_velocity_deg_s=angular_velocity_deg_s,
                x_center=_W / 2.0 + _VX_PX_S * (t_s - reference_s),
                y_center=_H / 2.0 + dy_px,
                angle_deg=0.0,
            )
            for mag, dy_px in movers
        ]
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=sats,
            stars=star_specs,
            rng=np.random.default_rng(seed0 + frame_id),
        )
        ctx = FrameContext(
            star_matches=star_matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        frame_pairs.append((synth.data_float, ctx))
    return frame_pairs


def _mover_dec_deg(dy_px: float) -> float:
    """Truth declination of a mover ``dy_px`` off the frame centre row."""
    _ra, dec = pixels_to_radec(_wcs(), _W / 2.0, _H / 2.0 + dy_px)
    return dec


def _tracklets_near_dec(tracklets: list[Tracklet], dec_deg: float) -> list[Tracklet]:
    """Tracklets whose mean declination is within ``_DEC_MATCH_ARCSEC``."""
    return [
        t
        for t in tracklets
        if abs(float(np.mean([p.dec_deg for p in t.points])) - dec_deg) * 3600.0
        < _DEC_MATCH_ARCSEC
    ]


@pytest.fixture(scope="module")
def mixed_stack_sequence() -> dict:
    """50 frames carrying one bright per-frame mover and one mv13 faint mover."""
    return {
        "frames": _render_movers(50, [(8.0, _MOVER_DY_PX), (13.0, 0.0)], 11000),
        "bright_dec": _mover_dec_deg(_MOVER_DY_PX),
        "faint_dec": _mover_dec_deg(0.0),
    }


@pytest.fixture(scope="module")
def bright_only_sequence() -> dict:
    """12 frames with a single bright mover — recoverable by *both* paths."""
    return {"frames": _render_movers(12, [(8.0, 0.0)], 12000)}


def test_bright_mover_does_not_suppress_faint_stacked_search(
    mixed_stack_sequence: dict,
) -> None:
    """Both movers survive: the bright one must not cancel the faint search.

    Regression for the old ``if tracklets or not stacking.enabled: return``
    early exit — the per-frame linker closes on the mv8 mover, which under
    the old rule skipped the stacked search entirely and silently dropped
    the mv13 companion, i.e. exactly the target class OpTA.NOD.DET is about.
    """
    tracklets = run_pipeline(mixed_stack_sequence["frames"], config=_plumbing_config())

    bright = _tracklets_near_dec(tracklets, mixed_stack_sequence["bright_dec"])
    faint = _tracklets_near_dec(tracklets, mixed_stack_sequence["faint_dec"])
    assert bright, (
        "bright per-frame mover lost; tracklet decs "
        f"{[float(np.mean([p.dec_deg for p in t.points])) for t in tracklets]}"
    )
    assert faint, (
        "faint mv13 mover lost — the stacked search did not run or did not "
        "survive the cross-path dedup; tracklet decs "
        f"{[float(np.mean([p.dec_deg for p in t.points])) for t in tracklets]}"
    )
    # Provenance: the faint mover is below the single-frame detection gate
    # (test_single_frame_path_does_not_recover_mv13), so its arc can only have
    # come from the stacked path.
    assert any(t.object_id.startswith("STK-") for t in faint)


def test_object_seen_by_both_paths_is_deduplicated(
    bright_only_sequence: dict,
) -> None:
    """One object found twice yields one tracklet, not two."""
    frames = bright_only_sequence["frames"]

    per_frame = run_pipeline(frames, config=_per_frame_only_config())
    stacked = run_track_and_stack(frames, config=_plumbing_config())
    # Guard against a vacuous assertion below: the dedup is only exercised if
    # both paths independently recovered the same single object.
    assert len(per_frame) == 1, f"per-frame path: {len(per_frame)} tracklets"
    assert len(stacked.tracklets) == 1, f"stacked path: {len(stacked.tracklets)}"

    union = run_pipeline(frames, config=_plumbing_config())
    assert len(union) == 1, (
        "cross-path duplicate not merged; object_ids "
        f"{[t.object_id for t in union]}"
    )


def test_flush_forwards_catalog_backend_to_stacked_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PipelineStream.flush must hand its catalog backend to the stacked path.

    ``run_pipeline`` always forwarded it; ``flush`` dropped it, so the stacked
    path's per-frame blind solves silently fell back to the default procedural
    catalog instead of the configured one (Gaia DR3 in production, I-03).
    """
    seen: dict[str, object] = {}

    def _spy(
        frames,
        config=None,
        *,
        catalog_backend=None,
        velocity_prior=None,
    ):
        seen["catalog_backend"] = catalog_backend
        return SimpleNamespace(tracklets=())

    monkeypatch.setattr(pipeline_mod, "run_track_and_stack", _spy)

    backend = ProceduralCatalog()
    stream = PipelineStream(config=_plumbing_config(), catalog_backend=backend)
    assert stream.flush() == []
    assert seen["catalog_backend"] is backend
