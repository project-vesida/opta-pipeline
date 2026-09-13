"""Tracklet module for OpTA pipeline (I-02).

Links astrometric detections across frames into tracklets using optimal
bipartite assignment (Hungarian algorithm) with linear-motion QC.

Architecture
------------
1. `link_detections(frames, ...)` — accepts per-frame lists of
   AstrometricDetections with timestamps; returns accepted `Tracklet` objects.
2. `tracklets_to_records(tracklets)` — serialises to I-02 JSON records.

Algorithm
---------
Frames are processed in time order.  At each step an optimal bipartite
assignment (scipy.optimize.linear_sum_assignment) matches open tracklets to
detections in the next frame.  The assignment cost is the residual from each
tracklet's motion model: once a tracklet has >= 2 points, its position at
the candidate frame's epoch is predicted by a linear RA/Dec fit over its
points (RA unwrapped across the 0/360 branch cut, extrapolated over the
actual elapsed time), and the cost is the angular separation between the
detection and that prediction.  A single-point tracklet has no rate estimate
yet, so its cost falls back to the plain distance from its only point.
Either way the cost is gated at `max_sep_arcsec` per elapsed frame (the gate
scales with the frame-id gap, so an object that skipped a frame is allowed
the slack it accrued meanwhile); detections outside the gate are
unassignable (infinite cost).  The motion-model cost is what keeps two
crossing tracks on their own partners through the intersection — pure
distance-from-last-point preferred the swapped assignment there.  Unmatched
detections seed new tracklets; tracklets that go unmatched for more than
`max_gap_frames` frames are closed.

After all frames are consumed, each candidate tracklet is subjected to a
linear-motion QC: RA(t) and Dec(t) are fit as first-order polynomials and the
RMS residual must be below `max_rms_arcsec`.  Tracklets with fewer than
`min_points` observations are discarded.

I-02 record format (JSON)
--------------------------
    {"ra": float, "dec": float, "utc": float,
     "sigma_ra": float, "sigma_dec": float,
     "object_id": str, "node_id": str}

where utc is Modified Julian Date (MJD) and angles are in degrees/arcsec.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from opta_pipeline.astrometry import AstrometricDetection

__all__ = [
    "FrameDetections",
    "TrackletPoint",
    "Tracklet",
    "link_detections",
    "tracklets_to_records",
    "tracklets_to_json",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameDetections:
    """All astrometric detections from a single frame.

    Parameters
    ----------
    detections : tuple[AstrometricDetection, ...]
        Detections from `astrometrise_detections`.
    utc_mjd : float
        Frame mid-exposure time as Modified Julian Date.
    frame_id : int
        Sequential frame index within the observation run.
    node_id : str
        Node identifier string (e.g. 'NODE-01').
    wcs_accuracy_flag : bool
        The frame's plate solution failed the post-solve OpTA.NOD.ACC
        self-check (``WCSSolution.accuracy_flag``, see
        :func:`opta_pipeline.astrometry.fit_wcs_sip`).  The astrometry is
        still emitted (annotate-only default: first light collects data
        rather than dropping it) but tracklets containing such frames carry
        ``Tracklet.wcs_accuracy_flagged`` for downstream QC.
    """

    detections: tuple[AstrometricDetection, ...] = field(hash=False, compare=False)
    utc_mjd: float
    frame_id: int
    node_id: str
    wcs_accuracy_flag: bool = False


@dataclass(frozen=True)
class TrackletPoint:
    """Single observation within a tracklet (one I-02 record).

    Parameters
    ----------
    ra_deg, dec_deg : float
        Equatorial coordinates (J2000, degrees).
    utc_mjd : float
        Observation time (MJD).
    sigma_ra_arcsec, sigma_dec_arcsec : float
        Per-axis position uncertainty from the WCS solution (arcsec).
    """

    ra_deg: float
    dec_deg: float
    utc_mjd: float
    sigma_ra_arcsec: float
    sigma_dec_arcsec: float


@dataclass(frozen=True)
class Tracklet:
    """A quality-controlled tracklet: linked detections from a single pass.

    Attributes
    ----------
    points : tuple[TrackletPoint, ...]
        Observations in time order (at least `min_points`).
    object_id : str
        Assigned identifier string (e.g. 'SAT-0001').
    node_id : str
        Node that observed this tracklet.
    rms_arcsec : float
        RMS residual of the linear RA/Dec fit (arcsec).
    ra_rate_arcsec_s : float
        Fitted RA angular rate (arcsec/s, RA·cos(Dec) convention).
    dec_rate_arcsec_s : float
        Fitted Dec angular rate (arcsec/s).
    wcs_accuracy_flagged : bool
        Quality annotation: at least one contributing frame's plate solution
        failed the post-solve OpTA.NOD.ACC self-check
        (``FrameDetections.wcs_accuracy_flag``).  The tracklet's astrometry
        may carry systematic error above the 10″ budget and should be
        down-weighted or inspected downstream.  Deliberately an annotation,
        not a rejection: the shipped policy is annotate-only so first light
        collects data (``AstrometryConfig.skip_flagged_frames`` opts into
        dropping flagged frames instead).  Not part of the I-02 record
        format, which is an interface spec (:meth:`to_records` unchanged).
    """

    points: tuple[TrackletPoint, ...] = field(hash=False, compare=False)
    object_id: str
    node_id: str
    rms_arcsec: float
    ra_rate_arcsec_s: float
    dec_rate_arcsec_s: float
    wcs_accuracy_flagged: bool = False

    def to_records(self) -> list[dict[str, object]]:
        """Serialize to a list of I-02 JSON-compatible dicts."""
        return [
            {
                "ra": p.ra_deg,
                "dec": p.dec_deg,
                "utc": p.utc_mjd,
                "sigma_ra": p.sigma_ra_arcsec,
                "sigma_dec": p.sigma_dec_arcsec,
                "object_id": self.object_id,
                "node_id": self.node_id,
            }
            for p in self.points
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcsec (small-angle, TAN approximation).

    The RA difference is wrapped into (-180, 180] deg so that steps across
    the 0/360 branch cut (e.g. 359.99 -> 0.01) measure the short way round.
    """
    cos_dec = math.cos(math.radians((dec1 + dec2) / 2.0))
    d_ra = ((ra2 - ra1 + 180.0) % 360.0 - 180.0) * cos_dec
    d_dec = dec2 - dec1
    return math.hypot(d_ra, d_dec) * 3600.0


def _linear_fit_rms(
    points: list[TrackletPoint],
) -> tuple[float, float, float]:
    """Fit linear RA(t) and Dec(t); return (rms_arcsec, ra_rate, dec_rate).

    RA is scaled by cos(mean_dec) so the rate is in projected arcsec/s.
    Times are offset to the midpoint for numerical stability.

    The RA series is unwrapped (period 360 deg) before fitting so a track
    crossing the 0/360 branch cut stays continuous; residuals and rates are
    invariant to the unwrap offset.
    """
    t = np.array([p.utc_mjd * 86400.0 for p in points])  # seconds
    ra = np.unwrap(np.array([p.ra_deg for p in points]), period=360.0)
    dec = np.array([p.dec_deg for p in points])

    t0 = float(np.mean(t))
    dt = t - t0
    cos_dec = math.cos(math.radians(float(np.mean(dec))))

    # Projected RA residuals in arcsec
    p_ra = np.polyfit(dt, ra * cos_dec * 3600.0, 1)
    p_dec = np.polyfit(dt, dec * 3600.0, 1)

    res_ra = ra * cos_dec * 3600.0 - np.polyval(p_ra, dt)
    res_dec = dec * 3600.0 - np.polyval(p_dec, dt)
    rms = float(np.sqrt(np.mean(res_ra**2 + res_dec**2)))

    ra_rate = float(p_ra[0])  # arcsec/s in RA·cos(dec) direction
    dec_rate = float(p_dec[0])  # arcsec/s in Dec direction
    return rms, ra_rate, dec_rate


def _predict_position(
    points: list[TrackletPoint], t_s: float
) -> tuple[float, float]:
    """Predict (ra_deg, dec_deg) at time `t_s` (seconds) by linear motion.

    Fits RA(t) and Dec(t) as first-order polynomials over the tracklet's
    points and extrapolates to `t_s`, so the prediction covers the actual
    elapsed time since the last observation (including skipped frames).
    RA is unwrapped (period 360 deg) before fitting so a track straddling
    the 0/360 branch cut stays continuous; the returned RA may therefore
    lie outside [0, 360) — `_angular_sep_arcsec` wraps the difference, so
    no renormalisation is needed.  Requires >= 2 points (an exact
    interpolation at 2 points, a least-squares fit beyond).
    """
    t = np.asarray([p.utc_mjd * 86400.0 for p in points])  # seconds
    ra = np.unwrap(np.asarray([p.ra_deg for p in points]), period=360.0)
    dec = np.asarray([p.dec_deg for p in points])

    t0 = float(np.mean(t))  # midpoint offset for numerical stability
    p_ra = np.polyfit(t - t0, ra, 1)
    p_dec = np.polyfit(t - t0, dec, 1)
    dt = t_s - t0
    return float(np.polyval(p_ra, dt)), float(np.polyval(p_dec, dt))


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------


def link_detections(
    frames: list[FrameDetections],
    max_sep_arcsec: float = 300.0,
    min_points: int = 3,
    max_rms_arcsec: float = 8.0,
    max_gap_frames: int = 1,
    object_id_prefix: str = "SAT",
) -> list[Tracklet]:
    """Link astrometric detections across frames into quality-controlled tracklets.

    The frame-to-frame assignment cost is a motion-model residual: once a
    tracklet has >= 2 points, its position at the candidate frame's epoch is
    predicted from a linear RA/Dec fit over its points (extrapolated over
    the actual elapsed time, RA unwrap-safe across 0/360) and the cost is
    the angular separation between detection and prediction.  Single-point
    tracklets have no rate estimate yet and fall back to the distance from
    their only point.  Both costs are gated by `max_sep_arcsec` per elapsed
    frame — a detection outside the gate is unassignable.

    Parameters
    ----------
    frames : list[FrameDetections]
        Per-frame detections with timestamps.  Need not be sorted; this
        function sorts by utc_mjd internally.
    max_sep_arcsec : float
        Maximum allowed angular distance between an observation and the
        tracklet's reference position (motion-model prediction with >= 2
        points, last point otherwise), *per elapsed frame*: an observation
        `g` frame-ids after the tracklet's last point must lie within
        `max_sep_arcsec * g` of the reference.  For consecutive frames
        (g = 1) the gate is
        `max_sep_arcsec` exactly.  Default 300 arcsec: at 25 fps an object
        at 0.5 °/s moves 72 arcsec/frame (0.5·3600/25, recomputed 2026-07-12
        via `python3 -c "print(0.5*3600/25)"`), so the default carries ~4×
        headroom and admits angular rates up to ≈2.08 °/s
        (300·25/3600, recomputed 2026-07-12 via
        `python3 -c "print(300*25/3600)"`).
    min_points : int
        Minimum number of observations for an accepted tracklet.
    max_rms_arcsec : float
        Maximum allowed linear-fit RMS residual (arcsec).
    max_gap_frames : int
        A tracklet can skip this many frames before it is closed.  The
        separation gate scales with the actual frame gap (see
        `max_sep_arcsec`), so gap tolerance does not silently lower the
        maximum trackable angular rate.
    object_id_prefix : str
        Prefix for auto-generated object IDs.

    Returns
    -------
    list[Tracklet]
        Accepted tracklets in discovery order.
    """
    if not frames:
        return []

    sorted_frames = sorted(frames, key=lambda f: f.utc_mjd)

    # Each active tracklet stores its points, one per-point flag recording
    # whether the point's frame of origin failed the OpTA.NOD.ACC plate-solve
    # self-check (FrameDetections.wcs_accuracy_flag — recorded at point
    # creation, when the originating frame is in hand, so the annotation
    # cannot be lost to float timestamp round-trips or spurious utc_mjd
    # collisions), and the frame_id of the last matched frame.  Gap detection
    # uses frame_id differences, not sorted-list indices, so dropped frames
    # are handled correctly.
    # Structure: list of (points, point_flags, last_frame_id)
    active: list[tuple[list[TrackletPoint], list[bool], int]] = []
    closed: list[tuple[list[TrackletPoint], list[bool]]] = []

    for frame in sorted_frames:
        dets = frame.detections
        if not dets:
            # No detections: age out tracklets that exceed gap limit.
            # gap = frame_id difference - 1  (gap of 1 means one frame was skipped).
            still_active = []
            for pts, flags, last_fid in active:
                if frame.frame_id - last_fid - 1 <= max_gap_frames:
                    still_active.append((pts, flags, last_fid))
                else:
                    closed.append((pts, flags))
            active = still_active
            continue

        n_active = len(active)
        n_det = len(dets)

        if n_active == 0:
            # Seed all detections as new tracklets
            for d in dets:
                pt = TrackletPoint(
                    ra_deg=d.ra_deg,
                    dec_deg=d.dec_deg,
                    utc_mjd=frame.utc_mjd,
                    sigma_ra_arcsec=d.sigma_ra_arcsec,
                    sigma_dec_arcsec=d.sigma_dec_arcsec,
                )
                active.append(([pt], [frame.wcs_accuracy_flag], frame.frame_id))
            continue

        # Build cost matrix: active tracklets × current detections.
        # Cost is the motion-model residual: a tracklet with >= 2 points
        # predicts its position at this frame's epoch from a linear RA/Dec
        # fit over its points (actual elapsed time, so skipped frames are
        # extrapolated over), and the cost is the separation between the
        # detection and that prediction.  A single-point tracklet has no
        # rate estimate yet and falls back to distance from its only point.
        # Without the prediction, two crossing tracks swap partners at the
        # intersection (the swapped assignment has lower total distance
        # there) and both kinked halves die on the linear-fit RMS QC.
        INF = 1e12
        cost = np.full((n_active, n_det), INF)
        t_frame_s = frame.utc_mjd * 86400.0
        for i, (pts, _flags, last_fid) in enumerate(active):
            # Only consider tracklets within gap limit
            gap_frames = frame.frame_id - last_fid
            if gap_frames - 1 > max_gap_frames:
                continue
            # The object keeps moving during skipped frames, so the gate
            # scales with the elapsed frame gap.  For consecutive frames
            # (gap_frames == 1) this is exactly max_sep_arcsec — identical
            # to the unscaled gate.
            gate = max_sep_arcsec * gap_frames
            if len(pts) >= 2:
                ref_ra, ref_dec = _predict_position(pts, t_frame_s)
            else:
                ref_ra, ref_dec = pts[-1].ra_deg, pts[-1].dec_deg
            for j, d in enumerate(dets):
                sep = _angular_sep_arcsec(
                    ref_ra, ref_dec, d.ra_deg, d.dec_deg
                )
                if sep <= gate:
                    cost[i, j] = sep

        # Optimal assignment (Hungarian)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        new_active: list[tuple[list[TrackletPoint], list[bool], int]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= INF:
                continue
            pts, flags, _ = active[r]
            d = dets[c]
            pts.append(
                TrackletPoint(
                    ra_deg=d.ra_deg,
                    dec_deg=d.dec_deg,
                    utc_mjd=frame.utc_mjd,
                    sigma_ra_arcsec=d.sigma_ra_arcsec,
                    sigma_dec_arcsec=d.sigma_dec_arcsec,
                )
            )
            flags.append(frame.wcs_accuracy_flag)
            new_active.append((pts, flags, frame.frame_id))
            matched_det.add(c)
            matched_trk.add(r)

        # Tracklets not matched: check gap limit
        for i, (pts, flags, last_fid) in enumerate(active):
            if i in matched_trk:
                continue
            if frame.frame_id - last_fid - 1 <= max_gap_frames:
                new_active.append((pts, flags, last_fid))
            else:
                closed.append((pts, flags))

        # Unmatched detections seed new tracklets
        for j, d in enumerate(dets):
            if j in matched_det:
                continue
            pt = TrackletPoint(
                ra_deg=d.ra_deg,
                dec_deg=d.dec_deg,
                utc_mjd=frame.utc_mjd,
                sigma_ra_arcsec=d.sigma_ra_arcsec,
                sigma_dec_arcsec=d.sigma_dec_arcsec,
            )
            new_active.append(([pt], [frame.wcs_accuracy_flag], frame.frame_id))

        active = new_active

    # Close all remaining active tracklets
    for pts, flags, _ in active:
        closed.append((pts, flags))

    # Quality filter and build Tracklet objects
    accepted: list[Tracklet] = []
    obj_counter = 0
    for pts, flags in closed:
        if len(pts) < min_points:
            continue
        rms, ra_rate, dec_rate = _linear_fit_rms(pts)
        if rms > max_rms_arcsec:
            continue
        obj_counter += 1
        node_id = sorted_frames[0].node_id  # all frames share the same node
        accepted.append(
            Tracklet(
                points=tuple(pts),
                object_id=f"{object_id_prefix}-{obj_counter:04d}",
                node_id=node_id,
                rms_arcsec=rms,
                ra_rate_arcsec_s=ra_rate,
                dec_rate_arcsec_s=dec_rate,
                wcs_accuracy_flagged=any(flags),
            )
        )

    return accepted


# ---------------------------------------------------------------------------
# I-02 serialisation
# ---------------------------------------------------------------------------


def tracklets_to_records(tracklets: list[Tracklet]) -> list[dict[str, object]]:
    """Flatten all tracklets to a list of I-02 observation records."""
    records: list[dict[str, object]] = []
    for t in tracklets:
        records.extend(t.to_records())
    return records


def tracklets_to_json(tracklets: list[Tracklet], indent: int = 2) -> str:
    """Serialise tracklets to an I-02 JSON string."""
    return json.dumps(tracklets_to_records(tracklets), indent=indent)
