"""Tests for opta_pipeline.tracklet.

Validates:
  - Empty input → no tracklets
  - Single-frame input → no tracklets (below min_points)
  - Straight-line synthetic tracklet recovered exactly
  - Multiple simultaneous satellites tracked independently
  - Linear-fit QC rejects junk associations
  - Gap tolerance (missing frame) correctly handled
  - I-02 serialisation: JSON keys and values
  - object_id assigned sequentially
  - RA 0/360 branch cut: separation, linking, and linear-fit QC are wrap-aware
  - Gap-scaled association gate: fast movers survive a dropped frame; the
    consecutive-frame gate is unchanged; no false linking on noise scenes
  - Crossing tracks: the motion-model cost keeps crossing tracks on their
    own partners (no swap), incl. with a dropped frame at the crossing and
    across the RA 0/360 branch cut; co-moving close pairs stay separate
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from opta_pipeline.astrometry import AstrometricDetection
from opta_pipeline.detect import Detection
from opta_pipeline.tracklet import (
    FrameDetections,
    Tracklet,
    TrackletPoint,
    _angular_sep_arcsec,
    _linear_fit_rms,
    link_detections,
    tracklets_to_json,
    tracklets_to_records,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE = "NODE-01"
_MJD0 = 60000.0  # arbitrary reference MJD
_DT_MJD = 1.0 / 86400.0  # 1 second in MJD units


def _make_astro(ra: float, dec: float, sigma: float = 1.0) -> AstrometricDetection:
    det = Detection(
        x=100.0,
        y=100.0,
        snr=20.0,
        elongation=1.2,
        angle_deg=0.0,
        n_pixels=5,
        flux_e=500.0,
        is_streak=True,
        fwhm_px=2.0,
    )
    return AstrometricDetection(
        detection=det,
        ra_deg=ra,
        dec_deg=dec,
        sigma_ra_arcsec=sigma,
        sigma_dec_arcsec=sigma,
    )


def _make_frame(
    detections: list[AstrometricDetection],
    frame_idx: int,
    dt_s: float = 1.0,
) -> FrameDetections:
    return FrameDetections(
        detections=tuple(detections),
        utc_mjd=_MJD0 + frame_idx * dt_s * _DT_MJD,
        frame_id=frame_idx,
        node_id=_NODE,
    )


def _straight_line_frames(
    n_frames: int = 10,
    ra0: float = 135.0,
    dec0: float = 45.0,
    ra_rate_arcsec_s: float = 500.0,  # arcsec/s projected
    dec_rate_arcsec_s: float = 200.0,
    noise_arcsec: float = 0.0,
    rng: np.random.Generator | None = None,
    dt_s: float = 0.04,  # 25 fps — 500 arcsec/s × 0.04s = 20 arcsec/frame
) -> list[FrameDetections]:
    """Synthetic satellite: constant angular velocity, one detection per frame.

    RA is emitted in the canonical [0, 360) convention (as astrometry would
    produce), so a track started near RA 360 crosses the 0/360 branch cut.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cos_dec = math.cos(math.radians(dec0))
    frames = []
    for i in range(n_frames):
        t_s = i * dt_s
        ra = ra0 + (ra_rate_arcsec_s * t_s) / (cos_dec * 3600.0)
        dec = dec0 + dec_rate_arcsec_s * t_s / 3600.0
        if noise_arcsec > 0:
            ra += rng.normal(0, noise_arcsec / (cos_dec * 3600.0))
            dec += rng.normal(0, noise_arcsec / 3600.0)
        frames.append(_make_frame([_make_astro(ra % 360.0, dec)], i, dt_s=dt_s))
    return frames


# ---------------------------------------------------------------------------
# 1. Basic API contract
# ---------------------------------------------------------------------------


class TestLinkDetectionsAPI:
    def test_empty_input(self) -> None:
        assert link_detections([]) == []

    def test_single_frame_below_min_points(self) -> None:
        frame = _make_frame([_make_astro(135.0, 45.0)], 0)
        result = link_detections([frame], min_points=3)
        assert result == []

    def test_returns_list_of_tracklets(self) -> None:
        frames = _straight_line_frames(n_frames=5)
        result = link_detections(frames, min_points=3)
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, Tracklet)

    def test_no_detections_in_frame(self) -> None:
        frames = _straight_line_frames(n_frames=5)
        # Insert an empty frame in the middle
        empty = _make_frame([], 2)
        mixed = frames[:2] + [empty] + frames[2:]
        # Should not raise
        link_detections(mixed, min_points=3)


# ---------------------------------------------------------------------------
# 2. Straight-line tracklet recovery
# ---------------------------------------------------------------------------


class TestStraightLineRecovery:
    def test_single_satellite_recovered(self) -> None:
        frames = _straight_line_frames(n_frames=10)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=0.1)
        assert len(tracklets) == 1

    def test_ideal_rms_near_zero(self) -> None:
        """Noise-free straight line → linear fit RMS essentially zero."""
        frames = _straight_line_frames(n_frames=10, noise_arcsec=0.0)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        assert len(tracklets) >= 1
        assert tracklets[0].rms_arcsec < 0.01

    def test_noisy_tracklet_within_budget(self) -> None:
        """0.5-arcsec position noise → linear fit RMS well below 8 arcsec."""
        frames = _straight_line_frames(
            n_frames=15, noise_arcsec=0.5, rng=np.random.default_rng(10)
        )
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=8.0)
        assert len(tracklets) >= 1
        assert tracklets[0].rms_arcsec < 4.0

    def test_tracklet_has_correct_point_count(self) -> None:
        frames = _straight_line_frames(n_frames=8)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        assert len(tracklets) == 1
        assert len(tracklets[0].points) == 8

    def test_tracklet_velocity_sign_correct(self) -> None:
        """RA rate must be positive for a prograde pass."""
        frames = _straight_line_frames(
            n_frames=10, ra_rate_arcsec_s=500.0, dec_rate_arcsec_s=0.0
        )
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        assert len(tracklets) >= 1
        assert tracklets[0].ra_rate_arcsec_s > 0.0


# ---------------------------------------------------------------------------
# 3. Multiple simultaneous satellites
# ---------------------------------------------------------------------------


class TestMultipleSatellites:
    def test_two_satellites_tracked_independently(self) -> None:
        """Two non-crossing tracks → two separate tracklets."""
        frames_a = _straight_line_frames(n_frames=8, ra0=130.0, dec0=40.0)
        frames_b = _straight_line_frames(n_frames=8, ra0=145.0, dec0=50.0)
        # Merge: each frame has two detections
        merged = []
        for fa, fb in zip(frames_a, frames_b):
            merged.append(
                FrameDetections(
                    detections=fa.detections + fb.detections,
                    utc_mjd=fa.utc_mjd,
                    frame_id=fa.frame_id,
                    node_id=fa.node_id,
                )
            )
        tracklets = link_detections(merged, min_points=3, max_rms_arcsec=1.0)
        assert len(tracklets) == 2

    def test_object_ids_are_unique(self) -> None:
        frames_a = _straight_line_frames(n_frames=8, ra0=130.0, dec0=40.0)
        frames_b = _straight_line_frames(n_frames=8, ra0=145.0, dec0=50.0)
        merged = [
            FrameDetections(
                detections=fa.detections + fb.detections,
                utc_mjd=fa.utc_mjd,
                frame_id=fa.frame_id,
                node_id=fa.node_id,
            )
            for fa, fb in zip(frames_a, frames_b)
        ]
        tracklets = link_detections(merged, min_points=3, max_rms_arcsec=1.0)
        ids = [t.object_id for t in tracklets]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# 4. Quality control
# ---------------------------------------------------------------------------


class TestQualityControl:
    def test_too_short_tracklet_rejected(self) -> None:
        frames = _straight_line_frames(n_frames=2)
        tracklets = link_detections(frames, min_points=3)
        assert len(tracklets) == 0

    def test_nonlinear_path_rejected(self) -> None:
        """Detections with random positions → high linear-fit RMS → rejected."""
        rng = np.random.default_rng(42)
        frames = []
        for i in range(8):
            ra = 135.0 + rng.uniform(-2.0, 2.0)
            dec = 45.0 + rng.uniform(-2.0, 2.0)
            frames.append(_make_frame([_make_astro(ra, dec)], i))
        tracklets = link_detections(
            frames, min_points=3, max_rms_arcsec=8.0, max_sep_arcsec=36000.0
        )
        # Random walk won't produce linear RMS < 8 arcsec over ±2 deg range
        assert all(t.rms_arcsec <= 8.0 for t in tracklets)


# ---------------------------------------------------------------------------
# 5. Gap tolerance
# ---------------------------------------------------------------------------


class TestGapTolerance:
    def test_one_missing_frame_tolerated(self) -> None:
        """A tracklet with one gap frame should still be recovered."""
        frames = _straight_line_frames(n_frames=8)
        # Drop frame 4
        gapped = [f for f in frames if f.frame_id != 4]
        tracklets = link_detections(
            gapped, min_points=3, max_rms_arcsec=1.0, max_gap_frames=1
        )
        # Should recover two sub-tracklets or one longer one depending on implementation
        total_points = sum(len(t.points) for t in tracklets)
        assert total_points >= 6  # at least 6 of 7 remaining points recovered

    def test_large_gap_breaks_tracklet(self) -> None:
        """A gap larger than max_gap_frames splits the tracklet."""
        frames = _straight_line_frames(n_frames=10)
        # Drop frames 4, 5, 6 (3-frame gap)
        gapped = [f for f in frames if f.frame_id not in {4, 5, 6}]
        tracklets = link_detections(
            gapped, min_points=3, max_rms_arcsec=1.0, max_gap_frames=1
        )
        # Should produce two separate tracklets (before and after gap)
        assert len(tracklets) == 2


# ---------------------------------------------------------------------------
# 5b. Gap-scaled association gate (regression for the unscaled-gate defect)
# ---------------------------------------------------------------------------


class TestGapScaledGate:
    """The association gate scales with the elapsed frame gap.

    Pre-fix defect (reproduced 2026-07-12 via the sweep in
    test_dropped_frame_fast_object_stays_one_tracklet): the gate compared the
    step from the *last matched point* against a flat `max_sep_arcsec`, so
    `max_gap_frames=1` nominally tolerated a skipped frame but the unscaled
    300-arcsec gate rejected the 2-frame step of any object faster than
    300·25/(2·3600) ≈ 1.04 °/s (recomputed 2026-07-12 via
    `python3 -c "print(300*25/(2*3600))"`).  A 10-frame track at 25 fps with
    frame 5 dropped split into [5, 4] fragments at 1.1 and 1.5 °/s — exactly
    the closest, brightest LEO passes (≈1.1 °/s overhead at 400 km).
    """

    @pytest.mark.parametrize("v_deg_s", [0.5, 1.1, 1.5])
    def test_dropped_frame_fast_object_stays_one_tracklet(
        self, v_deg_s: float
    ) -> None:
        """10 frames at 25 fps, frame 5 dropped → ONE tracklet with all 9 points."""
        frames = _straight_line_frames(
            n_frames=10,
            ra_rate_arcsec_s=v_deg_s * 3600.0,
            dec_rate_arcsec_s=0.0,
        )
        gapped = [f for f in frames if f.frame_id != 5]
        tracklets = link_detections(
            gapped, min_points=3, max_rms_arcsec=0.1, max_gap_frames=1
        )
        assert len(tracklets) == 1
        assert len(tracklets[0].points) == 9
        # All nine surviving detections belong to the single tracklet, in order
        assert [p.utc_mjd for p in tracklets[0].points] == [
            f.utc_mjd for f in gapped
        ]

    def test_consecutive_frame_gate_not_widened(self) -> None:
        """For consecutive frames the gate is max_sep_arcsec × 1 — exactly the
        pre-fix behaviour (verified bit-identical against the pre-fix linker
        over 50 randomized multi-object scenes at v ≤ 1.04 °/s, 2026-07-12)."""
        # Per-frame step 310″ > 300″: consecutive-frame linking must still fail.
        frames = _straight_line_frames(
            n_frames=10, ra_rate_arcsec_s=310.0 / 0.04, dec_rate_arcsec_s=0.0
        )
        assert link_detections(frames, min_points=3, max_rms_arcsec=1.0) == []
        # Per-frame step 290″ < 300″: links into one tracklet, as before.
        frames = _straight_line_frames(
            n_frames=10, ra_rate_arcsec_s=290.0 / 0.04, dec_rate_arcsec_s=0.0
        )
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        assert len(tracklets) == 1
        assert len(tracklets[0].points) == 10

    def test_noise_scene_no_false_linking(self) -> None:
        """Seeded pure-noise scene: the gap-widened gate creates no accepted
        false tracklets, under both the code defaults and the production-yaml
        gates (max_gap_frames=5, rms 15″).

        Measured across 200 seeds at 5 noise dets/frame (2026-07-12, old vs
        new gate): accepted-FP totals 3→5 (defaults) and 33→6 (production
        gates) — the min_points + linear-fit RMS QC, not the association gate,
        controls the FP floor.
        """
        rng = np.random.default_rng(1234)
        frames = []
        for i in range(20):
            n = int(rng.poisson(5.0))
            dets = tuple(
                _make_astro(
                    10.0 + float(rng.uniform(0.0, 1.0)),
                    float(rng.uniform(-0.5, 0.5)),
                )
                for _ in range(n)
            )
            frames.append(
                FrameDetections(
                    detections=dets,
                    utc_mjd=_MJD0 + i * 0.04 * _DT_MJD,
                    frame_id=i,
                    node_id=_NODE,
                )
            )
        assert link_detections(frames) == []
        assert link_detections(frames, max_gap_frames=5, max_rms_arcsec=15.0) == []


# ---------------------------------------------------------------------------
# 5c. Crossing tracks — motion-model cost (regression for the partner-swap
#     defect, fixed 2026-07-27)
# ---------------------------------------------------------------------------


def _crossing_frames(
    n: int = 10,
    ra0: float = 10.0,
    drop_frames_a: frozenset[int] = frozenset(),
) -> list[FrameDetections]:
    """Two straight tracks moving +RA at 72″/frame that cross in Dec mid-pass.

    Track A rises through dec 0 at +40″/frame; track B falls through dec 0 at
    -40″/frame; they intersect at frame (n-1)/2.  RA is emitted canonical
    [0, 360), so ra0 near 360 makes the crossing straddle the branch cut.
    `drop_frames_a` removes track A's detection from the listed frames.
    """
    cross = (n - 1) / 2.0
    frames = []
    for i in range(n):
        ra = (ra0 + 72.0 * i / 3600.0) % 360.0
        dec_a = -40.0 * (cross - i) / 3600.0  # rising through dec 0
        dec_b = +40.0 * (cross - i) / 3600.0  # falling through dec 0
        dets = []
        if i not in drop_frames_a:
            dets.append(_make_astro(ra, dec_a))
        dets.append(_make_astro(ra, dec_b))
        frames.append(_make_frame(dets, i, dt_s=0.04))
    return frames


def _assert_monotone_dec(tracklets: list[Tracklet]) -> None:
    """Each tracklet's declination is strictly monotone (no crossing kink),
    and the pair covers one rising and one falling track."""
    directions = []
    for t in tracklets:
        decs = [p.dec_deg for p in t.points]
        steps = [b - a for a, b in zip(decs, decs[1:])]
        assert all(s > 0 for s in steps) or all(s < 0 for s in steps)
        directions.append(steps[0] > 0)
    assert sorted(directions) == [False, True]


class TestCrossingTracks:
    """Pre-fix defect (strict xfail until 2026-07-27): the assignment cost
    was pure distance from the tracklet's last point, so two crossing tracks
    swapped partners at the intersection (the swapped assignment has lower
    total distance there) and both kinked halves died on the linear-fit RMS
    QC (measured rms 56.6″ vs the 8″ gate, 2026-07-12) — silent double loss.
    Fixed by the motion-model cost: with >= 2 points the cost is the residual
    from the tracklet's own linear RA/Dec prediction at the frame epoch."""

    def test_crossing_tracks_do_not_swap_partners(self) -> None:
        """Two straight tracks crossing mid-pass yield two 10-point
        tracklets, each with monotone declination (no kink at the crossing)."""
        frames = _crossing_frames(n=10)
        tracklets = link_detections(
            frames, min_points=3, max_rms_arcsec=8.0, max_gap_frames=1
        )
        assert len(tracklets) == 2
        for t in tracklets:
            assert len(t.points) == 10
            assert t.rms_arcsec < 0.01  # each partner is exactly linear
        _assert_monotone_dec(tracklets)

    def test_comoving_close_pair_stays_two_tracklets(self) -> None:
        """Co-moving parallel pair 100″ apart (< max_sep_arcsec=300) stays
        TWO tracklets with no point-stealing (#142 semantics)."""
        offset_deg = 100.0 / 3600.0
        frames = []
        for i in range(10):
            ra = 10.0 + 72.0 * i / 3600.0  # both move +RA at 72″/frame
            frames.append(
                _make_frame(
                    [_make_astro(ra, 45.0), _make_astro(ra, 45.0 + offset_deg)],
                    i,
                    dt_s=0.04,
                )
            )
        tracklets = link_detections(
            frames, min_points=3, max_rms_arcsec=8.0, max_gap_frames=1
        )
        assert len(tracklets) == 2
        dec_sets = sorted({p.dec_deg for t in tracklets for p in t.points})
        assert dec_sets == [45.0, 45.0 + offset_deg]
        for t in tracklets:
            assert len(t.points) == 10
            # No stealing: every point of a tracklet is from ONE truth track
            assert len({p.dec_deg for p in t.points}) == 1

    def test_crossing_with_dropped_frame_near_intersection(self) -> None:
        """One track skips a frame right before the crossing: the gap-scaled
        gate and the motion prediction (over the actual elapsed time) keep
        both tracks on their own partners."""
        frames = _crossing_frames(n=10, drop_frames_a=frozenset({4}))
        tracklets = link_detections(
            frames, min_points=3, max_rms_arcsec=8.0, max_gap_frames=1
        )
        assert len(tracklets) == 2
        assert sorted(len(t.points) for t in tracklets) == [9, 10]
        for t in tracklets:
            assert t.rms_arcsec < 0.01
        _assert_monotone_dec(tracklets)

    def test_crossing_tracks_across_ra_branch_cut(self) -> None:
        """The same crossing scene straddling RA 0/360: prediction and
        separation stay wrap-safe, no swap, records stay canonical."""
        frames = _crossing_frames(n=10, ra0=359.95)
        # Sanity: the scene really straddles the branch cut
        ras = [d.ra_deg for f in frames for d in f.detections]
        assert any(ra > 350.0 for ra in ras) and any(ra < 10.0 for ra in ras)
        tracklets = link_detections(
            frames, min_points=3, max_rms_arcsec=8.0, max_gap_frames=1
        )
        assert len(tracklets) == 2
        for t in tracklets:
            assert len(t.points) == 10
            assert t.rms_arcsec < 0.01
            for p in t.points:
                assert 0.0 <= p.ra_deg < 360.0
        _assert_monotone_dec(tracklets)


# ---------------------------------------------------------------------------
# 6. I-02 serialisation
# ---------------------------------------------------------------------------


class TestI02Serialisation:
    def _get_tracklet(self) -> Tracklet:
        frames = _straight_line_frames(n_frames=5)
        return link_detections(frames, min_points=3, max_rms_arcsec=1.0)[0]

    def test_to_records_keys(self) -> None:
        t = self._get_tracklet()
        records = t.to_records()
        required = {"ra", "dec", "utc", "sigma_ra", "sigma_dec", "object_id", "node_id"}
        for rec in records:
            assert required <= set(rec.keys())

    def test_to_records_node_id(self) -> None:
        t = self._get_tracklet()
        for rec in t.to_records():
            assert rec["node_id"] == _NODE

    def test_tracklets_to_json_valid(self) -> None:
        frames = _straight_line_frames(n_frames=5)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        js = tracklets_to_json(tracklets)
        parsed = json.loads(js)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1

    def test_tracklets_to_records_length(self) -> None:
        frames = _straight_line_frames(n_frames=5)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        records = tracklets_to_records(tracklets)
        # 5 frames × 1 detection = 5 records in one tracklet
        assert len(records) == sum(len(t.points) for t in tracklets)

    def test_ra_dec_in_records_match_points(self) -> None:
        frames = _straight_line_frames(n_frames=5)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=1.0)
        t = tracklets[0]
        for rec, pt in zip(t.to_records(), t.points):
            assert rec["ra"] == pt.ra_deg
            assert rec["dec"] == pt.dec_deg


# ---------------------------------------------------------------------------
# 7. RA 0/360 branch cut (fixed staring mount sweeps RA 0 every sidereal day)
# ---------------------------------------------------------------------------


class TestRABranchCut:
    def test_angular_sep_across_cut_at_equator(self) -> None:
        """359.99 -> 0.01 deg at dec 0 is 0.02 deg = 72 arcsec, not ~360 deg."""
        sep = _angular_sep_arcsec(359.99, 0.0, 0.01, 0.0)
        assert math.isclose(sep, 72.0, rel_tol=1e-9, abs_tol=1e-6)

    def test_angular_sep_across_cut_at_moderate_dec(self) -> None:
        """Same RA step at dec 45 scales by cos(dec); symmetric in argument order."""
        expected = 72.0 * math.cos(math.radians(45.0))
        sep_fwd = _angular_sep_arcsec(359.99, 45.0, 0.01, 45.0)
        sep_rev = _angular_sep_arcsec(0.01, 45.0, 359.99, 45.0)
        assert math.isclose(sep_fwd, expected, rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(sep_rev, expected, rel_tol=1e-9, abs_tol=1e-6)

    def test_angular_sep_matches_non_crossing_equivalent(self) -> None:
        """A step across the cut equals the same step away from it."""
        crossing = _angular_sep_arcsec(359.99, 30.0, 0.01, 30.0)
        plain = _angular_sep_arcsec(179.99, 30.0, 180.01, 30.0)
        assert math.isclose(crossing, plain, rel_tol=1e-9, abs_tol=1e-6)

    def test_linear_fit_rms_small_across_cut(self) -> None:
        """A genuinely linear track crossing RA 0 has near-zero fit RMS."""
        # 14.4 arcsec/s eastward at dec 0, 1-s cadence, canonical [0, 360) RA
        ras = [359.990, 359.994, 359.998, 0.002, 0.006, 0.010]
        points = [
            TrackletPoint(
                ra_deg=ra,
                dec_deg=0.0,
                utc_mjd=_MJD0 + i * _DT_MJD,
                sigma_ra_arcsec=1.0,
                sigma_dec_arcsec=1.0,
            )
            for i, ra in enumerate(ras)
        ]
        rms, ra_rate, dec_rate = _linear_fit_rms(points)
        assert rms < 0.01
        assert math.isclose(ra_rate, 14.4, rel_tol=1e-6)
        assert abs(dec_rate) < 1e-9

    def test_crossing_tracklet_linked_as_one(self) -> None:
        """6-point linear track crossing RA 0 -> ONE tracklet, all 6 points, QC pass."""
        frames = _straight_line_frames(n_frames=6, ra0=359.98, dec0=45.0)
        # Sanity: the synthetic track really straddles the branch cut
        ras = [f.detections[0].ra_deg for f in frames]
        assert any(ra > 350.0 for ra in ras) and any(ra < 10.0 for ra in ras)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=0.1)
        assert len(tracklets) == 1
        assert len(tracklets[0].points) == 6
        assert tracklets[0].rms_arcsec < 0.01
        assert tracklets[0].ra_rate_arcsec_s > 0.0

    def test_crossing_tracklet_records_stay_canonical(self) -> None:
        """I-02 records keep RA in [0, 360) even for a cut-crossing tracklet."""
        frames = _straight_line_frames(n_frames=6, ra0=359.98, dec0=45.0)
        tracklets = link_detections(frames, min_points=3, max_rms_arcsec=0.1)
        for rec in tracklets_to_records(tracklets):
            ra = rec["ra"]
            assert isinstance(ra, float)
            assert 0.0 <= ra < 360.0
