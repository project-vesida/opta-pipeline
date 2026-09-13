"""Windowed track-and-stack tests for opta_pipeline.pipeline.

A full LEO transit curves across the FOV, so shift-and-add (which assumes
linear motion) is applied per *window*.  These tests exercise
``run_windowed_track_and_stack`` on a clean, controlled multi-window scene.
They do not assert on realistic or quantized frames; that regime requires
separate coverage.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import WCSSolution
from opta_pipeline.config import PipelineConfig
from opta_pipeline.pipeline import (
    FrameContext,
    VelocityPrior,
    run_track_and_stack,
    run_windowed_track_and_stack,
)
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = 25.0
_RA0 = 135.0
_DEC0 = 45.0
_NODE = "NODE-WIN"
_MJD0 = 60000.0

_N_FRAMES = 100
_VX_PX_S = 40.0  # stays inside the 400 px width over the full 4 s pass
_VY_PX_S = 0.0


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


def _windowed_config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=80.0,
            velocity_step_px_s=40.0,
            window_duration_s=2.0,  # 50 frames @ 25 fps -> 2 windows over 4 s
            window_overlap_s=0.0,
        ),
    )


@pytest.fixture(scope="module")
def windowed_pass() -> dict:
    """Return a 100-frame mv=13 clean pass that splits into two windows."""
    wcs = _wcs()
    star_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=14.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    angular_velocity_deg_s = _VX_PX_S * _PIXEL_SCALE / 3600.0
    x0 = 120.0  # keeps x in [120, 280] over the pass, well inside the frame
    y0 = _H / 2.0

    frame_pairs = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        sat = SatelliteSpec(
            magnitude=13.0,
            angular_velocity_deg_s=angular_velocity_deg_s,
            x_center=x0 + _VX_PX_S * t_s,
            y_center=y0 + _VY_PX_S * t_s,
            angle_deg=0.0,
        )
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[sat],
            stars=star_specs,
            rng=np.random.default_rng(7000 + frame_id),
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
        "vx_px_s": _VX_PX_S,
        "angular_velocity_deg_s": angular_velocity_deg_s,
    }


@pytest.fixture(scope="module")
def windowed_result(windowed_pass: dict):
    """``run_windowed_track_and_stack`` at the shipped ``_windowed_config()``.

    Four tests below assert on *this same* computation — the same
    module-scoped frames through the same (frozen, deterministic) config — so
    the two-window blind coarse-to-fine search used to run four times per
    session for one result.  Tests that vary the config
    (``window_duration_s = 0``, ``window_overlap_s = 1``) still build and run
    their own.
    """
    return run_windowed_track_and_stack(
        windowed_pass["frames"], config=_windowed_config()
    )


def test_windowed_stack_splits_into_two_windows(windowed_result) -> None:
    """A 4 s pass at 2 s windows yields exactly two contiguous windows."""
    result = windowed_result

    assert result.n_windows == 2
    assert result.fps == pytest.approx(_FPS, rel=0.05)
    assert result.window_bounds == ((0, 50), (50, 100))


def test_windowed_stack_recovers_velocity_in_each_window(
    windowed_pass: dict, windowed_result
) -> None:
    """Both windows recover the injected velocity and emit tracklets."""
    result = windowed_result

    for window in result.window_results:
        # Coarse-to-fine resolves to the per-window criterion step
        # (PSF_fwhm/T_window ≈ 3.5 px/s for these short windows), not to a
        # grid node.
        assert window.stack_result.best_vx_px_s == pytest.approx(
            windowed_pass["vx_px_s"], abs=5.0
        )
        assert window.stack_result.best_vy_px_s == pytest.approx(0.0, abs=5.0)
        assert len(window.tracklets) >= 1

    assert len(result.tracklets) >= 2


def test_windowed_tracklets_are_window_prefixed(windowed_result) -> None:
    """Concatenated tracklets carry a per-window ``Wnn-`` object_id prefix."""
    result = windowed_result

    prefixes = {t.object_id.split("-", 1)[0] for t in result.tracklets}
    assert prefixes <= {"W00", "W01"}
    assert "W00" in prefixes and "W01" in prefixes


def test_velocity_prior_recovers_offgrid_injected_velocity(
    windowed_pass: dict,
) -> None:
    """A prior centred *near* (not on) the true velocity recovers it via the
    vector-centred grid + component-SNR ranking."""
    cfg = _windowed_config()
    # Prior centre offset from the true (40, 0); the grid still brackets it.
    prior = VelocityPrior(
        vx_px_s=38.0, vy_px_s=-2.0, half_width_px_s=6.0, step_px_s=2.0
    )
    result = run_track_and_stack(
        windowed_pass["frames"], config=cfg, velocity_prior=prior
    )
    assert result.stack_result.best_vx_px_s == pytest.approx(
        windowed_pass["vx_px_s"], abs=2.0
    )
    assert result.stack_result.best_vy_px_s == pytest.approx(0.0, abs=2.0)
    assert len(result.tracklets) >= 1


# Moved to the nightly full run 2026-07-29 (runtime triage): at 361 s this was
# the single most expensive test in the fast suite (20 % of its 1814 s wall,
# measured via `pytest opta-pipeline/tests/ -q -m "not slow" --durations=40`).
# The cost is structural, not accidental — the assertion IS that two entry
# points agree, so the full-pass blind coarse-to-fine over all 100 frames at
# +/-80 px/s must run twice, and the depth-matched auto seed window (#210) puts
# ~6900 hypotheses in stage 1 at this coarse_step=40 geometry.  What stays in
# the fast suite: the windowing geometry and per-window recovery
# (test_windowed_stack_splits_into_two_windows,
# test_windowed_stack_recovers_velocity_in_each_window, both on the shared
# `windowed_result`), the cross-window dedup
# (test_overlapping_windows_dedup_the_shared_object), and the unwindowed
# run_track_and_stack blind path itself (test_pipeline_stacking.py,
# test_pipeline_likelihood.py, test_pipeline_blind_solve.py).  Only the
# *degenerate-configuration identity* of the two entry points moves to nightly.
@pytest.mark.slow
def test_zero_window_duration_matches_single_stack(windowed_pass: dict) -> None:
    """``window_duration_s <= 0`` is identical to one ``run_track_and_stack``."""
    cfg = replace(
        _windowed_config(),
        stacking=replace(_windowed_config().stacking, window_duration_s=0.0),
    )
    windowed = run_windowed_track_and_stack(windowed_pass["frames"], config=cfg)
    single = run_track_and_stack(windowed_pass["frames"], config=cfg)

    assert windowed.n_windows == 1
    assert windowed.window_bounds == ((0, _N_FRAMES),)
    assert windowed.window_results[0].stack_result.best_vx_px_s == pytest.approx(
        single.stack_result.best_vx_px_s, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Cross-window tracklet dedup (window_overlap_s > 0)
# ---------------------------------------------------------------------------
#
# With overlapping windows the same object is measured by every window whose
# span contains it, so verbatim concatenation double-counts objects and FPs
# (inflated det/hr and FAR).  _dedup_cross_window removes duplicates in
# trajectory space, reusing coarse_fine._tracks_overlap on the fitted sky
# tracks over the shared time span.  At window_overlap_s = 0 the dedup pass
# is skipped entirely (byte-identical concatenation — pinned below).

from math import cos, radians  # noqa: E402

from opta_pipeline.pipeline import (  # noqa: E402
    _cross_window_duplicate,
    _dedup_cross_window,
)
from opta_pipeline.tracklet import Tracklet, TrackletPoint  # noqa: E402


def _mk_tracklet(
    t0_mjd: float,
    n_points: int,
    dt_s: float,
    ra0_deg: float,
    dec0_deg: float,
    ra_rate_as_s: float,
    dec_rate_as_s: float,
    object_id: str,
    rms_arcsec: float = 1.0,
    flagged: bool = False,
) -> Tracklet:
    """Hand-built linear-motion tracklet (rates in projected arcsec/s)."""
    cos_dec = cos(radians(dec0_deg))
    pts = []
    for i in range(n_points):
        t_s = i * dt_s
        pts.append(
            TrackletPoint(
                ra_deg=ra0_deg + ra_rate_as_s * t_s / 3600.0 / cos_dec,
                dec_deg=dec0_deg + dec_rate_as_s * t_s / 3600.0,
                utc_mjd=t0_mjd + t_s / 86400.0,
                sigma_ra_arcsec=1.0,
                sigma_dec_arcsec=1.0,
            )
        )
    return Tracklet(
        points=tuple(pts),
        object_id=object_id,
        node_id="NODE-DDP",
        rms_arcsec=rms_arcsec,
        ra_rate_arcsec_s=ra_rate_as_s,
        dec_rate_arcsec_s=dec_rate_as_s,
        wcs_accuracy_flagged=flagged,
    )


_T0 = 60000.0
_DT = 0.04  # 25 fps
# The shipped duplicate-identity radius (StackingConfig.dedup_radius_arcsec):
# an astrometric-error scale (~3x the 10 arcsec OpTA.NOD.ACC budget), NOT the
# 300 arcsec/frame association gate.
_DEDUP_GATE = PipelineConfig.default().stacking.dedup_radius_arcsec


class TestCrossWindowDuplicate:
    def test_same_object_overlapping_windows_is_duplicate(self) -> None:
        """Two window arcs of ONE linear track sharing a time span dedup."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        # Window B starts 25 frames (1 s) later on the same trajectory.
        b = _mk_tracklet(
            _T0 + 25 * _DT / 86400.0,
            50,
            _DT,
            135.0 + 300.0 * 25 * _DT / 3600.0 / cos(radians(45.0)),
            45.0,
            300.0,
            0.0,
            "W01-SAT-0001",
        )
        assert _cross_window_duplicate(a, b, gate_arcsec=300.0)

    def test_distinct_objects_are_kept(self) -> None:
        """Co-temporal but 1800 arcsec apart: not duplicates at the 300 gate."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(_T0, 50, _DT, 135.0, 45.5, 300.0, 0.0, "W01-SAT-0001")
        assert not _cross_window_duplicate(a, b, gate_arcsec=300.0)

    def test_same_object_dedups_at_shipped_radius(self) -> None:
        """The same-object pair also dedups at the tight shipped radius."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(
            _T0 + 25 * _DT / 86400.0,
            50,
            _DT,
            135.0 + 300.0 * 25 * _DT / 3600.0 / cos(radians(45.0)),
            45.0,
            300.0,
            0.0,
            "W01-SAT-0001",
        )
        assert _cross_window_duplicate(a, b, gate_arcsec=_DEDUP_GATE)

    def test_comoving_pair_250_arcsec_apart_is_not_duplicate(self) -> None:
        """PR #135 P2 regression: a distinct co-moving object 250 arcsec away
        (deployment-cluster / satellite-train geometry) is NOT duplicate
        identity.  Under the old gate — the reused 300 arcsec/frame
        *association* gate — this pair merged and the single-window object
        was silently deleted."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(
            _T0, 50, _DT, 135.0, 45.0 + 250.0 / 3600.0, 300.0, 0.0,
            "W01-SAT-0001",
        )
        assert not _cross_window_duplicate(a, b, gate_arcsec=_DEDUP_GATE)
        # Sanity pin of the defect: the association gate would have merged it.
        assert _cross_window_duplicate(a, b, gate_arcsec=300.0)

    def test_crossing_tracks_with_different_velocities_survive(self) -> None:
        """Two tracks that intersect mid-overlap but move differently are not
        duplicates: identity requires agreement over the WHOLE shared span
        (velocity agreement), not co-location at a single instant."""
        cos45 = cos(radians(45.0))
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        # b crosses a's track at t = 1 s (a is then 300 arcsec east of start),
        # moving due north at 100 arcsec/s.
        b = _mk_tracklet(
            _T0,
            50,
            _DT,
            135.0 + 300.0 / 3600.0 / cos45,
            45.0 - 100.0 / 3600.0,
            0.0,
            100.0,
            "W01-SAT-0001",
        )
        assert not _cross_window_duplicate(a, b, gate_arcsec=_DEDUP_GATE)

    def test_disjoint_time_spans_never_dedup(self) -> None:
        """Same trajectory, non-overlapping windows: both arcs are kept."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(
            _T0 + 50 * _DT / 86400.0,
            50,
            _DT,
            135.0 + 300.0 * 50 * _DT / 3600.0 / cos(radians(45.0)),
            45.0,
            300.0,
            0.0,
            "W01-SAT-0001",
        )
        assert not _cross_window_duplicate(a, b, gate_arcsec=300.0)


class TestDedupCrossWindow:
    def test_keeps_best_measured_arc(self) -> None:
        long_arc = _mk_tracklet(
            _T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001"
        )
        short_arc = _mk_tracklet(
            _T0 + 25 * _DT / 86400.0,
            30,
            _DT,
            135.0 + 300.0 * 25 * _DT / 3600.0 / cos(radians(45.0)),
            45.0,
            300.0,
            0.0,
            "W01-SAT-0001",
        )
        kept = _dedup_cross_window(
            [long_arc, short_arc], window_of=[0, 1], gate_arcsec=300.0
        )
        assert kept == [long_arc]

    def test_same_window_tracklets_never_dedup_each_other(self) -> None:
        """The per-window linker already separated same-window tracklets."""
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0002")
        kept = _dedup_cross_window([a, b], window_of=[0, 0], gate_arcsec=300.0)
        assert kept == [a, b]

    def test_distinct_objects_survive(self) -> None:
        a = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        b = _mk_tracklet(_T0, 50, _DT, 135.0, 45.5, 300.0, 0.0, "W01-SAT-0001")
        kept = _dedup_cross_window([a, b], window_of=[0, 1], gate_arcsec=300.0)
        assert kept == [a, b]

    def test_comoving_neighbor_seen_in_one_window_survives(self) -> None:
        """PR #135 P2 regression at the dedup level: a distinct co-moving
        object 250 arcsec away, detected in only ONE window (fewer points,
        so it loses the keep-priority contest), must survive dedup at the
        shipped radius.  With the reused 300 arcsec association gate it was
        silently deleted as a 'duplicate' of its brighter neighbour."""
        primary = _mk_tracklet(
            _T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001"
        )
        # Co-moving companion 250 arcsec north, seen only by window 1.
        companion = _mk_tracklet(
            _T0 + 25 * _DT / 86400.0,
            25,
            _DT,
            135.0 + 300.0 * 25 * _DT / 3600.0 / cos(radians(45.0)),
            45.0 + 250.0 / 3600.0,
            300.0,
            0.0,
            "W01-SAT-0001",
        )
        kept = _dedup_cross_window(
            [primary, companion], window_of=[0, 1], gate_arcsec=_DEDUP_GATE
        )
        assert kept == [primary, companion]

    def test_unflagged_duplicate_preferred_over_flagged(self) -> None:
        """PR #135 P3: a tracklet with flagged astrometry must never displace
        an unflagged duplicate, even when it has more points and lower RMS."""
        flagged_long = _mk_tracklet(
            _T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001",
            rms_arcsec=1.0, flagged=True,
        )
        clean_short = _mk_tracklet(
            _T0 + 25 * _DT / 86400.0,
            30,
            _DT,
            135.0 + 300.0 * 25 * _DT / 3600.0 / cos(radians(45.0)),
            45.0,
            300.0,
            0.0,
            "W01-SAT-0001",
            rms_arcsec=2.0,
        )
        kept = _dedup_cross_window(
            [flagged_long, clean_short],
            window_of=[0, 1],
            gate_arcsec=_DEDUP_GATE,
        )
        assert kept == [clean_short]

    def test_flag_priority_does_not_outrank_distinctness(self) -> None:
        """The flag only breaks ties among true duplicates: a flagged
        tracklet of a DISTINCT object is still kept."""
        clean = _mk_tracklet(_T0, 50, _DT, 135.0, 45.0, 300.0, 0.0, "W00-SAT-0001")
        flagged_other = _mk_tracklet(
            _T0, 50, _DT, 135.0, 45.5, 300.0, 0.0, "W01-SAT-0001", flagged=True
        )
        kept = _dedup_cross_window(
            [clean, flagged_other], window_of=[0, 1], gate_arcsec=_DEDUP_GATE
        )
        assert kept == [clean, flagged_other]


def test_overlapping_windows_dedup_the_shared_object(windowed_pass: dict) -> None:
    """1 s overlap on 2 s windows: the single injected object is not
    double-counted across windows, while per-window diagnostics keep every
    window's full output."""
    cfg = _windowed_config()
    cfg = replace(
        cfg, stacking=replace(cfg.stacking, window_overlap_s=1.0)
    )
    result = run_windowed_track_and_stack(windowed_pass["frames"], config=cfg)

    # 50-frame windows advancing by 25 frames over 100 frames -> 3 windows.
    assert result.n_windows == 3
    raw = sum(len(w.tracklets) for w in result.window_results)
    assert raw >= 3  # every window measures the object -> duplicates exist
    assert len(result.tracklets) < raw  # dedup actually removed duplicates
    # One object: at most the two non-overlapping arc ends can survive.
    assert 1 <= len(result.tracklets) <= 2


def test_zero_overlap_concat_is_verbatim(windowed_result) -> None:
    """window_overlap_s = 0 output is byte-identical plain concatenation."""
    result = windowed_result  # _windowed_config() has window_overlap_s = 0.0

    expected = tuple(
        replace(t, object_id=f"W{wi:02d}-{t.object_id}")
        for wi, w in enumerate(result.window_results)
        for t in w.tracklets
    )
    assert result.tracklets == expected
