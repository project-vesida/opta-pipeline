"""Pipeline orchestrator.

Single entry point for the full frame-processing chain:
  calibrate → detect → fit_wcs → astrometrise → FrameDetections

Encapsulates the ``noise_rms = max(bg_rms, 1.0)`` floor documented in
AGENTS.md — callers do not need to know about it.

Entry points
------------
run_frame(raw, ctx, config)
    Process one frame.  Returns FrameDetections or None if no streaks found.

run_pipeline(frames, config)
    Batch-process a sequence of (raw, ctx) pairs; returns accepted tracklets.

PipelineStream(config)
    Streaming interface: push frames one at a time, flush at end-of-pass.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import KDTree

from opta_pipeline.astrometry import (
    StarMatch,
    WCSSolution,
    _check_accuracy,
    apply_rolling_shutter_correction,
    astrometrise_detections,
    fit_wcs,
    fit_wcs_sip,
    radec_to_pixels,
)
from opta_pipeline.calibrate import calibrate_frame, estimate_background
from opta_pipeline.catalog import CatalogBackend, ProceduralCatalog
from opta_pipeline.coarse_fine import (
    TrackCandidate,
    blind_coarse_fine_search,
)
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import Detection, detect_sources
from opta_pipeline.likelihood import (
    PsiPhiStacker,
    bound_scale_epsilon,
    forced_measurement,
    gaussian_psf_kernel,
    matched_streak_kernel,
    median_subtraction_safe,
    poisson_tail_threshold,
    quantisation_step_e,
    temporal_median_subtract,
)
from opta_pipeline.match import PointingHint, match_catalog
from opta_pipeline.stack import StackResult
from opta_pipeline.tracklet import FrameDetections, Tracklet, link_detections

logger = logging.getLogger(__name__)

# Default reference-catalog backend: the deterministic, file-free procedural
# catalog.  Production injects a GaiaDR3Catalog via the run_* catalog_backend
# argument; the swap touches no pipeline logic (I-03).
_DEFAULT_CATALOG: CatalogBackend = ProceduralCatalog()

__all__ = [
    "FrameContext",
    "VelocityPrior",
    "StackedPipelineResult",
    "WindowedStackResult",
    "run_frame",
    "run_track_and_stack",
    "run_windowed_track_and_stack",
    "run_pipeline",
    "PipelineStream",
]

_SECONDS_PER_DAY = 86400.0

# OpTA.NOD.ACC accuracy budget (≤ 10.0″ RMS per node, SYSTEMS.md) for the
# post-solve self-check on the linear astrometry path — the same value as
# fit_wcs_sip's accuracy_budget_arcsec default on the SIP path.
_ACCURACY_BUDGET_ARCSEC = 10.0


@dataclass(frozen=True)
class FrameContext:
    """Per-frame inputs required by the orchestrator.

    Provide **either** ``star_matches`` (the legacy, externally-supplied path)
    **or** ``pointing`` (blind in-pipeline plate solve).  If both are given,
    ``star_matches`` takes precedence.

    Attributes
    ----------
    star_matches : list[StarMatch]
        Catalog-matched stars for plate solving (≥ 3 required).  Optional; when
        empty and ``pointing`` is set, the orchestrator blind-solves the WCS
        from the image (detect stars → catalog cross-match → fit).
    utc_mjd : float
        Frame mid-exposure time as Modified Julian Date.
    frame_id : int
        Sequential frame index within the observation run.
    node_id : str
        Node identifier string (e.g. 'NODE-01').
    pointing : PointingHint | None
        Approximate pointing used for a blind plate solve when no
        ``star_matches`` are supplied.
    master_dark : np.ndarray | None
        Master dark frame (same shape as raw).  None skips dark subtraction.
    master_flat : np.ndarray | None
        Normalised master flat frame.  None skips flat correction.
    """

    utc_mjd: float
    frame_id: int
    node_id: str
    star_matches: list[StarMatch] = field(
        default_factory=list, hash=False, compare=False
    )
    pointing: PointingHint | None = field(default=None, hash=False, compare=False)
    master_dark: np.ndarray | None = field(default=None, hash=False, compare=False)
    master_flat: np.ndarray | None = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class VelocityPrior:
    """Ephemeris prior for the shift-and-stack velocity search (pixels/second).

    A scheduled pass has a predicted sky-track rate *and direction*, so the
    search is a small box centred on the predicted velocity *vector* rather than
    a direction-agnostic symmetric grid.  Fewer hypotheses lower the
    extreme-value detection floor (better sensitivity), and centring on the
    prediction guarantees a grid node near the true velocity.

    Attributes
    ----------
    vx_px_s, vy_px_s : float
        Predicted image-plane velocity vector (pixels/second).
    half_width_px_s : float
        Grid half-extent about the prediction — the prior uncertainty.
    step_px_s : float
        Grid resolution; should be ≈ PSF_px / T_window so the worst-case
        off-node registration smear stays sub-PSF.
    """

    vx_px_s: float
    vy_px_s: float
    half_width_px_s: float
    step_px_s: float


@dataclass(frozen=True)
class StackedPipelineResult:
    """Diagnostic output from a blind track-and-stack batch run.

    reference_mjd is the median frame time (midpass epoch) used for stack shifts.
    """

    stack_result: StackResult
    stacked_detections: tuple[Detection, ...]
    frame_detections: tuple[FrameDetections, ...] = field(hash=False, compare=False)
    tracklets: tuple[Tracklet, ...] = field(hash=False, compare=False)
    reference_mjd: float


@dataclass(frozen=True)
class WindowedStackResult:
    """Result of windowed track-and-stack over a long (curved) pass.

    Shift-and-add assumes linear motion, valid only over short spans, so a long
    pass is split into windows that are each stacked independently with
    :func:`run_track_and_stack`.  ``tracklets`` concatenates every window's
    accepted tracklets, with ``object_id`` prefixed by the window index
    (``W00-…``); with ``window_overlap_s > 0`` cross-window duplicates (the
    same object measured by two overlapping windows) are removed in
    trajectory space (:func:`_dedup_cross_window`) so downstream det/hr and
    FAR metrics are not inflated — ``window_results`` keeps each window's
    full output.  Stitching the survivors into one arc is a separate
    chaining step.
    """

    window_results: tuple[StackedPipelineResult, ...] = field(hash=False, compare=False)
    tracklets: tuple[Tracklet, ...] = field(default=(), hash=False, compare=False)
    window_bounds: tuple[tuple[int, int], ...] = ()
    fps: float = 0.0

    @property
    def n_windows(self) -> int:
        """Number of stacking windows processed."""
        return len(self.window_results)


@dataclass(frozen=True)
class _CalibratedBatchFrame:
    """One calibrated frame with noise floor and orchestrator context for stacking."""

    data: np.ndarray = field(hash=False, compare=False)
    noise_rms: float
    ctx: FrameContext = field(hash=False, compare=False)


def _star_mask(
    shape: tuple[int, int],
    matches: list[StarMatch],
    radius_px: int,
) -> np.ndarray:
    """Return a boolean mask around catalog-star positions.

    Discs are stamped locally (a ``(2r+1)²`` patch per star) rather than
    tested against full-frame coordinate grids, so building one mask per
    frame for a whole pass stays cheap.
    """
    mask = np.zeros(shape, dtype=bool)
    if radius_px <= 0 or not matches:
        return mask

    h, w = shape
    r = int(radius_px)
    r2 = float(radius_px) ** 2
    for star in matches:
        x = float(star.x_px)
        y = float(star.y_px)
        if x < -radius_px or x >= w + radius_px:
            continue
        if y < -radius_px or y >= h + radius_px:
            continue
        x0 = max(0, int(math.floor(x)) - r)
        x1 = min(w, int(math.ceil(x)) + r + 1)
        y0 = max(0, int(math.floor(y)) - r)
        y1 = min(h, int(math.ceil(y)) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= r2
    return mask


# Borrow cap (in frames) when the drift rate cannot be measured — fewer than
# two solved frames, or no consecutive pair linking enough stars.  One frame
# is the only distance the original "drift between adjacent frames ≪ the mask
# radius" argument covers, so that is the conservative fallback.
_BORROW_CAP_FRAMES_UNMEASURED = 1

# Matched stars a consecutive solved pair must contribute before its
# displacement is trusted as a drift sample.  One or two nearest-neighbour
# links are as likely to be a mis-pairing as a measurement.
_MIN_DRIFT_PAIRS = 3


def _star_drift_px_per_frame(
    matches_per_frame: list[list[StarMatch]],
    solved: list[int],
    radius_px: int,
) -> float | None:
    """Median per-frame star displacement measured over the solved frames.

    Stars are linked across a consecutive solved pair by *pixel* nearest
    neighbour, gated at ``radius_px``: a star that moved further than a mask
    radius between two solved frames cannot be borrowed across that gap
    anyway, so excluding it costs nothing and keeps a mis-pairing from
    inflating the estimate.  Each pair contributes its median displacement
    divided by the frame gap once at least ``_MIN_DRIFT_PAIRS`` stars link;
    the estimate is the median over pairs.  Returns None when no pair
    qualifies, i.e. the drift is unmeasurable.

    Catalog coordinates deliberately play no part.  They are *not* a stable
    identity across frames: the real-sky sidecar path advances every star's
    RA per frame by the sidereal rate
    (``opta_validation.harness._drift_matches``), so an (ra, dec) join would
    find zero shared stars on exactly the partially-solved real night this
    cap exists to protect.
    """
    rates: list[float] = []
    for a, b in zip(solved, solved[1:]):
        prev = matches_per_frame[a]
        cur = matches_per_frame[b]
        if not prev or not cur:
            continue
        tree = KDTree([[m.x_px, m.y_px] for m in prev])
        dist, _ = tree.query(
            [[m.x_px, m.y_px] for m in cur],
            distance_upper_bound=float(radius_px),
        )
        linked = np.asarray(dist)[np.isfinite(dist)]
        if linked.size >= _MIN_DRIFT_PAIRS:
            rates.append(float(np.median(linked)) / float(b - a))
    if not rates:
        return None
    return float(np.median(rates))


def _star_masks_per_frame(
    shape: tuple[int, int],
    matches_per_frame: list[list[StarMatch]],
    radius_px: int,
) -> list[np.ndarray] | None:
    """Per-frame drift-tracking star masks from per-frame astrometry.

    On a fixed (non-tracking) mount the stars drift across the detector at
    the sidereal rate, so a single frame-0 mask covers the wrong pixels for
    most of the pass — the root cause of the sidereal false-positive
    tracklets (e2e findings §12): temporal-median subtraction leaves ~30–50σ
    point-like positive residuals at each star's *current* (drifted)
    position, which stacked trajectories crossing them accumulate above the
    trials-corrected u* gate.  Each frame therefore gets a mask built from
    its *own* star positions — supplied sidecar matches or the frame's own
    plate solve — so the drift is taken from measured astrometry (WCS), never
    fitted to the scene (design rule 10).  On a stationary mount all frames'
    positions coincide and this degrades to the static mask.

    Frames whose plate solve failed (empty matches) borrow the nearest solved
    frame's positions, but only out to a *measured* distance: the drift rate
    is estimated from the solved frames themselves
    (:func:`_star_drift_px_per_frame`) and the borrow is capped at
    ``radius_px / drift_per_frame`` frames, the point at which the borrowed
    disc has walked a full mask radius off the star.  That bound is what the
    "drift between adjacent frames is ≪ the mask radius" argument actually
    buys — at the v3 plate scale the sidereal rate is a fraction of the
    radius per frame, so a run of failed solves lasting seconds would
    otherwise stamp discs a full radius away, masking blank sky while leaving
    the real residuals unweighted.  Frames beyond the cap are left unmasked
    (all-False) and a single warning is emitted, mirroring the no-solves case.
    When the drift cannot be measured (fewer than two solved frames, or no
    consecutive pair linking ``_MIN_DRIFT_PAIRS`` stars within ``radius_px``)
    the cap falls back to the conservative fixed
    ``_BORROW_CAP_FRAMES_UNMEASURED`` = 1 frame; a measured drift of exactly
    zero (stationary mount) leaves the borrow unbounded.

    Returns None when masking is disabled (``radius_px <= 0``) or no frame has
    star positions at all.
    """
    if radius_px <= 0 or not matches_per_frame:
        return None
    solved = [i for i, mm in enumerate(matches_per_frame) if mm]
    if not solved:
        # Masking was *requested* (radius > 0) but cannot be applied: without
        # star positions the ψ/φ stack keeps full weight on bright-star
        # residuals — the sidereal FP-tracklet mechanism (e2e findings §12).
        # Say so instead of degrading silently.
        logger.warning(
            "star masking enabled (radius %d px) but no frame has star "
            "matches — stacking proceeds with NO star masks; bright-star "
            "residuals may produce false-positive tracklets",
            radius_px,
        )
        return None
    n_frames = len(matches_per_frame)
    drift = _star_drift_px_per_frame(matches_per_frame, solved, radius_px)
    if drift is None:
        cap = _BORROW_CAP_FRAMES_UNMEASURED
    elif drift <= 0.0:
        cap = n_frames  # no measurable drift ⇒ any solved frame still applies
    else:
        cap = max(1, int(float(radius_px) / drift))

    masks: list[np.ndarray] = []
    cache: dict[int, np.ndarray] = {}
    unmasked: np.ndarray | None = None
    n_beyond_cap = 0
    for i in range(n_frames):
        if matches_per_frame[i]:
            src = i
        else:
            src = min(solved, key=lambda j: abs(j - i))
            if abs(src - i) > cap:
                n_beyond_cap += 1
                if unmasked is None:
                    unmasked = np.zeros(shape, dtype=bool)
                masks.append(unmasked)
                continue
        if src not in cache:
            cache[src] = _star_mask(shape, matches_per_frame[src], radius_px)
        masks.append(cache[src])
    if n_beyond_cap:
        # Same failure mode as the no-solves branch, restricted to a run of
        # frames: those frames keep full ψ/φ weight on bright-star residuals.
        logger.warning(
            "%d frame(s) are more than %d frame(s) from any plate-solved "
            "frame (drift %s px/frame vs mask radius %d px) — star masks "
            "would be a full radius off, so those frames are left UNMASKED; "
            "bright-star residuals there may produce false-positive tracklets",
            n_beyond_cap,
            cap,
            "unmeasured" if drift is None else f"{drift:.3f}",
            radius_px,
        )
    return masks


def _trajectory_dominated_by_star(
    x0: float,
    y0: float,
    vx_px_s: float,
    vy_px_s: float,
    frame_times_s: list[float],
    star_mask: np.ndarray | Sequence[np.ndarray] | None,
    max_frac: float,
) -> bool:
    """Velocity-aware star veto: True iff the candidate is a static-star artefact.

    The catalog-star mask flags pixels dominated by bright-star Poisson
    residuals that temporal-median subtraction does not fully whiten.  Vetoing a
    stacked candidate because its *reference-epoch* peak lands on such a pixel is
    a completeness ceiling for movers: a genuine mover only transits a given star
    for a small fraction of its pass, so its epoch position on a star is
    incidental (measured ~13% SNR-independent loss in star-dense pointings; a
    15.2σ candidate was vetoed — docs/pipeline-stack-assessment.md F5).

    Instead evaluate the candidate's whole trajectory ``(x0 + vx·t, y0 + vy·t)``
    over the pass epochs and veto only when it dwells on masked pixels for at
    least ``max_frac`` of the frames.  A static source (v≈0, or a hypothesis that
    locked onto a bright-star residual) sits on the mask for the entire pass and
    is still rejected; a mover that clips a star at its reference epoch survives.

    ``star_mask`` is either one static mask or a per-frame sequence aligned
    with ``frame_times_s`` (drift-tracking masks, sidereal mounts): frame k's
    trajectory sample is tested against frame k's mask, so a hypothesis locked
    onto a *drifting* star keeps dwelling on the mask instead of walking off
    the stale frame-0 discs once the drift exceeds the mask radius.

    The veto requires *actual* dwell (``n_on > 0``): ``max_frac = 0.0`` means
    "veto on any mask contact at all", not "veto everything" — with a bare
    ``n_on >= max_frac·n_frames`` the zero case held for every candidate,
    including trajectories that never touch a mask pixel.  Values outside
    [0, 1] are rejected at config parse time
    (:class:`~opta_pipeline.config.DetectionConfig`); the ``> 1.0`` guard below
    is defence in depth for hand-built configs.
    """
    if star_mask is None or max_frac > 1.0 or not frame_times_s:
        return False
    per_frame = not isinstance(star_mask, np.ndarray)
    if per_frame:
        masks = list(star_mask)
        if len(masks) != len(frame_times_s):
            raise ValueError(
                f"per-frame star masks ({len(masks)}) must align with "
                f"frame_times_s ({len(frame_times_s)})"
            )
        if not any(m.any() for m in masks):
            return False
        h, w = masks[0].shape
    else:
        if not star_mask.any():
            return False
        h, w = star_mask.shape
    n_on = 0
    for k, t in enumerate(frame_times_s):
        xi = int(round(x0 + vx_px_s * t))
        yi = int(round(y0 + vy_px_s * t))
        mask_k = masks[k] if per_frame else star_mask
        if 0 <= xi < w and 0 <= yi < h and mask_k[yi, xi]:
            n_on += 1
    return n_on > 0 and n_on >= max_frac * len(frame_times_s)


def _fit_wcs_for(
    matches: list[StarMatch],
    shape: tuple[int, int],
    config: PipelineConfig,
) -> WCSSolution:
    """Fit a WCS, distortion-aware when configured (T-11), else linear.

    ``fit_wcs_sip`` adds a radial SIP term for the wide prime's barrel and
    degrades to the linear solution when no distortion is present, so the
    same call covers both the clean and the distorted regimes.  Both paths
    run the post-solve OpTA.NOD.ACC self-check (``fit_wcs_sip`` internally;
    the bare ``fit_wcs`` result is wrapped here), so ``accuracy_flag`` is
    set consistently regardless of ``sip_distortion``.
    """
    if config.astrometry.sip_distortion:
        return fit_wcs_sip(
            matches,
            frame_shape=shape,
            max_residual_arcsec=config.astrometry.max_residual_arcsec,
        )
    return _check_accuracy(
        fit_wcs(
            matches,
            frame_shape=shape,
            max_residual_arcsec=config.astrometry.max_residual_arcsec,
        ),
        matches,
        shape,
        _ACCURACY_BUDGET_ARCSEC,
    )


def _project_in_frame(
    backend: CatalogBackend,
    wcs: WCSSolution,
    shape: tuple[int, int],
    mag_limit: float,
) -> list[StarMatch]:
    """Project the catalog over the solved FOV onto the frame (for masking).

    Cone-searches the backend around the solved field centre out to the frame
    half-diagonal, then keeps the stars that project inside the frame.  Driving
    the star mask from the same backend as the cross-match keeps the whole
    catalog dependency behind one interface.
    """
    h, w = shape
    cd_det = abs(wcs.cd1_1 * wcs.cd2_2 - wcs.cd1_2 * wcs.cd2_1)
    pixel_scale_deg = math.sqrt(cd_det) if cd_det > 0.0 else abs(wcs.cd1_1)
    radius_deg = pixel_scale_deg * math.hypot(w / 2.0, h / 2.0) + 1.0
    stars = backend.cone_search(
        wcs.crval1, wcs.crval2, radius_deg, mag_limit=mag_limit
    )
    matches: list[StarMatch] = []
    for s in stars:
        x, y = radec_to_pixels(wcs, s.ra_deg, s.dec_deg)
        if 0.0 <= x < w and 0.0 <= y < h:
            matches.append(
                StarMatch(x_px=x, y_px=y, ra_deg=s.ra_deg, dec_deg=s.dec_deg)
            )
    return matches


def _solve_frame_wcs(
    star_detections: list[Detection],
    ctx: FrameContext,
    config: PipelineConfig,
    shape: tuple[int, int],
    backend: CatalogBackend,
) -> tuple[WCSSolution | None, list[StarMatch]]:
    """Resolve a frame's WCS and the star positions to mask.

    Two paths, chosen per frame:

    * **Supplied matches** (legacy / external sidecar): fit directly from
      ``ctx.star_matches``.  The same matches are returned for masking.
    * **Blind solve** (``ctx.pointing`` set, no matches): cross-match the
      non-streak detections against a catalog cone-search seeded by the
      pointing hint, fit the WCS, then project the full catalog through the
      *solved* WCS to get complete star positions for masking.

    Returns ``(wcs, mask_matches)``; ``(None, [])`` if the frame cannot be
    solved (too few stars or no consistent cross-match) so the caller can skip
    it.  ``backend`` is the reference catalog (procedural by default, a cached
    Gaia DR3 cone-search in production — I-03 — behind the same interface).
    """
    if ctx.star_matches:
        wcs = _fit_wcs_for(ctx.star_matches, shape, config)
        return wcs, ctx.star_matches

    if ctx.pointing is None:
        raise ValueError(
            "FrameContext needs either star_matches or a pointing hint"
        )

    stars = [d for d in star_detections if not d.is_streak]
    if len(stars) < 3:
        return None, []

    catalog = backend.cone_search(
        ctx.pointing.ra_deg,
        ctx.pointing.dec_deg,
        ctx.pointing.radius_deg,
        mag_limit=ctx.pointing.mag_limit,
    )
    matches = match_catalog(stars, catalog, ctx.pointing, frame_shape=shape)
    if len(matches) < 3:
        return None, []

    wcs = _fit_wcs_for(matches, shape, config)
    mask_matches = _project_in_frame(backend, wcs, shape, ctx.pointing.mag_limit)
    return wcs, mask_matches


def _calibrate_batch(
    frames: Iterable[tuple[np.ndarray, FrameContext]],
    config: PipelineConfig,
) -> list[_CalibratedBatchFrame]:
    """Calibrate a frame sequence once for stack-backed batch processing."""
    calibrated: list[_CalibratedBatchFrame] = []
    expected_shape: tuple[int, int] | None = None
    for raw, ctx in frames:
        if expected_shape is None:
            expected_shape = raw.shape
        elif raw.shape != expected_shape:
            raise ValueError(
                f"All frames in a stack must share one shape; got {raw.shape} "
                f"after {expected_shape}"
            )

        cal = calibrate_frame(
            raw,
            master_dark=ctx.master_dark,
            master_flat=ctx.master_flat,
            subtract_background=True,
            background_box_size=config.calibration.background_box_size,
            background_filter_size=config.calibration.background_filter_size,
        )
        calibrated.append(
            _CalibratedBatchFrame(
                data=cal.data,
                noise_rms=max(cal.background_rms, 1.0),
                ctx=ctx,
            )
        )
    return calibrated


def _frame_times_relative_to_midpass(
    frames: list[_CalibratedBatchFrame],
) -> tuple[list[float], float]:
    """Return seconds from the midpass epoch and the reference MJD."""
    mjds = np.array([f.ctx.utc_mjd for f in frames], dtype=np.float64)
    reference_mjd = float(np.median(mjds))
    return [
        float((mjd - reference_mjd) * _SECONDS_PER_DAY) for mjd in mjds
    ], reference_mjd


def _link_frame_detections(
    frame_dets: list[FrameDetections],
    config: PipelineConfig,
) -> list[Tracklet]:
    """Apply tracklet linker QC using thresholds from PipelineConfig.tracklet."""
    return link_detections(
        frame_dets,
        max_sep_arcsec=config.tracklet.max_sep_arcsec,
        min_points=config.tracklet.min_detections,
        max_rms_arcsec=config.tracklet.linear_fit_residual_arcsec,
        max_gap_frames=config.tracklet.max_gap_frames,
    )


def run_frame(
    raw: np.ndarray,
    ctx: FrameContext,
    config: PipelineConfig | None = None,
    *,
    catalog_backend: CatalogBackend | None = None,
) -> FrameDetections | None:
    """Process one raw frame through the full detection chain.

    calibrate → detect → fit_wcs → astrometrise → FrameDetections

    At the operational dark-sky level (~0.3 e⁻/px) background_rms rounds to 0;
    the 1 e⁻ noise floor prevents detect_sources from raising ValueError and
    corresponds to the readout-noise regime where Poisson statistics break down.

    Parameters
    ----------
    raw : np.ndarray
        Raw sensor frame (2-D integer or float ADU).
    ctx : FrameContext
        Per-frame context: catalog star matches, timestamp, IDs, optional masters.
    config : PipelineConfig | None
        Pipeline parameters.  Loads pipeline_defaults.yaml if None.
    catalog_backend : CatalogBackend | None
        Reference catalog for the blind solve.  Defaults to the procedural
        catalog; pass a ``GaiaDR3Catalog`` for production (I-03).

    Returns
    -------
    FrameDetections | None
        Sky-coordinate detections for all streak candidates, or None if none found.

    Raises
    ------
    ValueError
        If ctx supplies neither star_matches nor a pointing hint, or the
        supplied star_matches are geometrically degenerate.
    """
    if config is None:
        config = PipelineConfig.default()
    backend = catalog_backend if catalog_backend is not None else _DEFAULT_CATALOG

    cal = calibrate_frame(
        raw,
        master_dark=ctx.master_dark,
        master_flat=ctx.master_flat,
        subtract_background=True,
        background_box_size=config.calibration.background_box_size,
        background_filter_size=config.calibration.background_filter_size,
    )

    noise_rms = max(cal.background_rms, 1.0)

    star_mask = _star_mask(
        cal.data.shape,
        ctx.star_matches,
        config.detection.mask_star_radius_px,
    )
    dets = detect_sources(
        cal.data,
        noise_rms=noise_rms,
        snr_threshold=config.detection.snr_threshold,
        min_pixels=config.detection.min_streak_pixels,
        max_pixels=config.detection.max_streak_pixels,
        elongation_threshold=config.detection.elongation_threshold,
        star_mask=star_mask,
    )

    streaks = [d for d in dets if d.is_streak]
    if not streaks:
        return None

    h, w = raw.shape
    # Blind solve reuses the non-streak detections already in `dets`; the
    # legacy path ignores them and fits the supplied matches.
    wcs, _ = _solve_frame_wcs(dets, ctx, config, (h, w), backend)
    if wcs is None:
        return None

    # OpTA.NOD.ACC accuracy_flag consumption (see fit_wcs_sip): default is
    # annotate-only — the flag rides on FrameDetections so the tracklet
    # linker can annotate downstream QC.  With skip_flagged_frames set the
    # frame is dropped like an unsolved one instead.
    if wcs.accuracy_flag and config.astrometry.skip_flagged_frames:
        logger.info(
            "frame %d: plate solution accuracy-flagged and "
            "astrometry.skip_flagged_frames is set — frame skipped",
            ctx.frame_id,
        )
        return None

    astro = astrometrise_detections(streaks, wcs)

    return FrameDetections(
        detections=tuple(astro),
        utc_mjd=ctx.utc_mjd,
        frame_id=ctx.frame_id,
        node_id=ctx.node_id,
        wcs_accuracy_flag=wcs.accuracy_flag,
    )


def _likelihood_stack_and_detect(
    calibrated: list[_CalibratedBatchFrame],
    frame_times_s: list[float],
    config: PipelineConfig,
    velocity_prior: VelocityPrior | None,
    star_masks: list[np.ndarray] | None,
    subtract_static: bool,
) -> tuple[StackResult, list[Detection], list[tuple[float, float]]]:
    """ψ/φ trajectory search + trials-corrected detection on the SNR map.

    Returns ``(stack_result, detections, velocities)`` with one velocity
    vector per detection: in blind coarse-to-fine mode each accepted
    candidate carries its *own* velocity (assessment F7 — multi-object);
    the prior/single-stage paths return the winning hypothesis for every
    detection.

    Blind mode (no prior, ``stacking.coarse_to_fine``) delegates to
    :func:`~opta_pipeline.coarse_fine.blind_coarse_fine_search`: windowed
    criterion-matched seeding, pyramid refinement, full-pass u* gate.
    Candidates whose trajectory dwells on the star masks are dropped
    (bright-star Poisson residuals are not fully whitened by
    temporal-median subtraction).

    ``star_masks`` (per-frame, aligned with ``calibrated``) enter the ψ/φ
    accumulation as invalid pixels (V = ∞ ⇒ φ = 0): catalog-star pixels
    contribute neither signal nor statistical weight, per frame, at the
    star's *drifted* position.  This is the fix for the sidereal
    false-positive tracklets (e2e findings §12): under sidereal drift the
    temporal median no longer models the stars and leaves ~30–50σ
    point-like residuals at each frame's current star positions (measured
    on e2e scene D: max residual 28–50 e⁻ against a 1.05 e⁻ off-star RMS);
    fast-mover trajectories crossing several such residuals accumulated
    above u* while dwelling on the mask for only 10–28% of the pass — far
    below any veto fraction that would still pass real movers in
    star-dense fields, so the veto layer *cannot* separate them and the
    residual power must be removed from the statistic itself.  The φ
    (coverage) formulation keeps the SNR map calibrated: a mover crossing
    a masked star loses only those frames' weight (SNR × ≈ √(1−dwell)),
    instead of being vetoed outright.  Stars below the catalog magnitude
    limit remain covered by temporal-median subtraction (stationary
    mounts) — their sidereal residuals are a documented open tail, not
    hidden by this mechanism.

    ``subtract_static`` is the caller's motion-safety decision
    (:func:`median_subtraction_safe`): when the whole search band sits in
    the median's slow-mover blind spot the subtraction is skipped — one
    decision shared by the search, the winner re-score, and forced
    photometry, so their statistics agree.  With subtraction skipped the
    per-frame masks are the *primary* star suppression (full star flux is
    still in the frames).

    Detection contract: the SNR map has mean 0 / variance 1 under noise,
    and the threshold is the Bernstein tail bound over all pixels ×
    hypotheses plus a configured margin
    (:func:`~opta_pipeline.likelihood.poisson_tail_threshold`, contract
    since 2026-07-28) — the bound scale ε is estimated from the batch's
    own frames and per-frame σ, so the low-count Poisson tail that the
    former Gaussian extreme-value bound measurably under-covered (blank
    SMOKE FAR ≈ 9 %/scene at ~0.3 e⁻/px sky vs ≪ 1 % predicted; Gaussian
    control 1/100) is now inside the false-alarm budget.  ε → 0 (high
    counts, many frames) reproduces the old bound exactly.
    ``min_pixels=1`` because the map is already matched-filtered: requiring
    a multi-pixel super-threshold cluster would double-penalize compact
    sources.  Detection ``flux_e`` is replaced by the ML flux estimate
    Ψ/Φ at the peak (unbiased under partial coverage).

    Returns a :class:`StackResult` adapter (``stacked`` = SNR map,
    ``noise_rms`` = 1.0) so every downstream consumer keeps working.
    """
    stk = config.stacking
    frame_data = [item.data for item in calibrated]
    frame_noise = [item.noise_rms for item in calibrated]
    # Streak-kernel exposure = frame period, derived from the actual batch
    # cadence (one source of truth; falls back to the config when unavailable).
    exposure_s = _streak_exposure_s(stk, _frame_cadence_s(frame_times_s))

    if velocity_prior is None and stk.coarse_to_fine:
        cf = blind_coarse_fine_search(
            frame_data,
            frame_noise,
            frame_times_s,
            v_max_px_s=stk.velocity_max_px_s,
            coarse_step_px_s=stk.velocity_step_px_s,
            psf_sigma_px=stk.psf_sigma_px,
            prethreshold_sigma=stk.coarse_prethreshold_sigma,
            far_margin_sigma=stk.far_margin_sigma,
            max_candidates=stk.max_fine_candidates,
            velocity_min_px_s=stk.velocity_min_px_s,
            seed_window_s=stk.seed_window_s,
            subtract_median=subtract_static,
            seed_binning=stk.seed_binning,
            exposure_time_s=exposure_s,
            streak_min_length_px=stk.streak_min_length_px,
            masks=star_masks,
        )
        shape = frame_data[0].shape
        # Velocity-aware star veto (assessment F5): each blind candidate carries
        # its own velocity, so test its whole trajectory against the per-frame
        # (drift-tracking) star masks rather than only its reference-epoch
        # pixel.  A mover that transits a star at t=0 survives; a hypothesis
        # pinned on a (possibly drifting) bright-star residual is still dropped.
        max_frac = config.detection.mask_veto_max_trajectory_frac
        candidates = [
            c
            for c in cf.candidates
            if not _trajectory_dominated_by_star(
                c.x_px,
                c.y_px,
                c.vx_px_s,
                c.vy_px_s,
                frame_times_s,
                star_masks,
                max_frac,
            )
        ]
        dets = [
            Detection(
                x=c.x_px,
                y=c.y_px,
                snr=c.snr,
                elongation=0.0,
                angle_deg=0.0,
                n_pixels=1,
                flux_e=c.flux_e,
                is_streak=False,
                fwhm_px=2.355 * stk.psf_sigma_px,
            )
            for c in candidates
        ]
        velocities = [(c.vx_px_s, c.vy_px_s) for c in candidates]

        # Winner's full-frame SNR map (single hypothesis — cheap) for the
        # StackResult adapter and downstream diagnostics/figures.
        if candidates:
            best = candidates[0]
            winner = PsiPhiStacker(
                np.array([best.vx_px_s]),
                np.array([best.vy_px_s]),
                psf_sigma_px=stk.psf_sigma_px,
                exposure_s=exposure_s,
                streak_min_length_px=stk.streak_min_length_px,
            ).stack(
                frame_data,
                frame_noise,
                frame_times_s,
                masks=star_masks,
                subtract_temporal_median=subtract_static,
            )
            snr_map = winner.snr_map
            best_vx, best_vy, peak = (
                best.vx_px_s,
                best.vy_px_s,
                best.snr,
            )
        else:
            snr_map = np.zeros(shape, dtype=np.float64)
            best_vx = best_vy = 0.0
            peak = 0.0
        adapter = StackResult(
            stacked=snr_map,
            noise_rms=1.0,
            n_frames=len(calibrated),
            best_vx_px_s=best_vx,
            best_vy_px_s=best_vy,
            peak_snr=peak,
            response_grid=None,
        )
        return adapter, dets, velocities

    if velocity_prior is not None:
        stacker = PsiPhiStacker.from_prior(
            velocity_prior.vx_px_s,
            velocity_prior.vy_px_s,
            velocity_prior.half_width_px_s,
            velocity_prior.step_px_s,
            psf_sigma_px=stk.psf_sigma_px,
            velocity_min_px_s=stk.velocity_min_px_s,
            exposure_s=exposure_s,
            streak_min_length_px=stk.streak_min_length_px,
        )
    else:
        stacker = PsiPhiStacker.from_config(config, psf_sigma_px=stk.psf_sigma_px)

    lres = stacker.stack(
        frame_data,
        frame_noise,
        frame_times_s,
        masks=star_masks,
        subtract_temporal_median=subtract_static,
    )

    n_hyp = int(stacker.vx_grid.size * stacker.vy_grid.size)
    n_trials = max(int(lres.snr_map.size) * n_hyp, 2)
    # Bernstein bound scale from this batch's own frames/σ (likelihood
    # module docstring, "Detection threshold").  The round search kernel
    # is the conservative choice for ε even when the winner was re-scored
    # with a streak kernel (flatter kernel ⇒ smaller P_max ⇒ smaller ε).
    bound_scale = bound_scale_epsilon(
        gaussian_psf_kernel(stk.psf_sigma_px),
        frame_noise,
        n_frames=len(frame_data),
        quant_step_e=quantisation_step_e(frame_data),
    )
    u_star = poisson_tail_threshold(
        n_trials, bound_scale, stk.far_margin_sigma
    )

    # Pedestal removal before thresholding — the likelihood twin of the
    # classic "stacked images carry a DC pedestal ∝ N" gotcha
    # (opta-pipeline/AGENTS.md).  For low-count Poisson sky the temporal
    # *median* sits below the *mean* (skew), so the median-subtracted
    # frames keep a small positive mean that ψ accumulates coherently into
    # a smooth ≈ +3σ offset of the SNR map (measured on SENSOR_SMALL synth
    # scenes).  Re-estimating the map's own large-scale background (box ≫
    # PSF, so compact sources survive) restores the ≈ N(0,1) zero level
    # that the u* threshold assumes.
    background, _ = estimate_background(
        lres.snr_map,
        box_size=config.calibration.background_box_size,
        filter_size=config.calibration.background_filter_size,
    )
    detection_map = lres.snr_map - background

    dets = detect_sources(
        detection_map,
        noise_rms=1.0,
        snr_threshold=u_star,
        min_pixels=1,
        max_pixels=config.detection.max_streak_pixels,
        elongation_threshold=config.detection.elongation_threshold,
    )
    # Velocity-aware star veto (assessment F5): the prior/single-stage winner
    # is one hypothesis, so every peak shares (best_vx, best_vy).  Veto a peak
    # only if that trajectory is dominated by the (per-frame) star masks — not
    # merely because its reference-epoch pixel lands on a star (the old
    # pixel-level ``detect_sources(star_mask=...)`` dropped movers
    # SNR-independently; a 15.2σ prior candidate was vetoed —
    # docs/pipeline-stack-assessment.md F5).
    max_frac = config.detection.mask_veto_max_trajectory_frac
    dets = [
        det
        for det in dets
        if not _trajectory_dominated_by_star(
            det.x,
            det.y,
            lres.best_vx_px_s,
            lres.best_vy_px_s,
            frame_times_s,
            star_masks,
            max_frac,
        )
    ]
    flux_corrected = [
        replace(
            det,
            flux_e=float(
                lres.flux_map[
                    min(max(int(round(det.y)), 0), lres.flux_map.shape[0] - 1),
                    min(max(int(round(det.x)), 0), lres.flux_map.shape[1] - 1),
                ]
            ),
        )
        for det in dets
    ]

    adapter = StackResult(
        stacked=lres.snr_map,
        noise_rms=1.0,
        n_frames=lres.n_frames,
        best_vx_px_s=lres.best_vx_px_s,
        best_vy_px_s=lres.best_vy_px_s,
        peak_snr=lres.peak_snr,
        response_grid=lres.response_grid,
    )
    velocities = [
        (lres.best_vx_px_s, lres.best_vy_px_s) for _ in flux_corrected
    ]
    return adapter, flux_corrected, velocities


def run_track_and_stack(
    frames: Iterable[tuple[np.ndarray, FrameContext]],
    config: PipelineConfig | None = None,
    *,
    catalog_backend: CatalogBackend | None = None,
    velocity_prior: VelocityPrior | None = None,
) -> StackedPipelineResult:
    """Run blind track-before-detect over a frame sequence.

    The batch path follows the SOTA fixed-mount workflow: calibrate each
    frame, search a velocity grid in px/s, detect peaks in the stacked
    statistic, then project each accepted stack peak back through every
    frame's WCS to produce I-02-compatible tracklets.

    Scoring is the ψ/φ matched-filter trajectory search
    (:class:`~opta_pipeline.likelihood.PsiPhiStacker`) on the calibrated
    frames after temporal-median static-scene subtraction.  Detection runs
    on the calibrated (mean-0/variance-1) SNR map against the
    trials-corrected Bernstein threshold
    ``u* = poisson_tail_threshold(N_pix·N_hyp, ε) + far_margin_sigma``
    (contract since 2026-07-28; ε estimated from the batch's frames and
    σ) — pure noise cannot form tracklets within the false-alarm budget,
    *including* the low-count Poisson tail that the former Gaussian
    extreme-value bound under-covered (measured blank SMOKE FAR
    ≈ 9 %/scene vs ≪ 1 % predicted, seeds 0–99; closed by this contract —
    see the likelihood module docstring, "Detection threshold").
    ``stack_result.stacked`` **is
    the SNR map** and ``stack_result.noise_rms == 1.0``.  (The legacy
    peak-pixel :class:`~opta_pipeline.stack.Stacker` remains available as a
    library class for diagnostics scripts; it is no longer a pipeline path.)

    ``catalog_backend`` selects the reference catalog for the per-frame blind
    solve (procedural by default; a ``GaiaDR3Catalog`` in production — I-03).

    ``velocity_prior`` supplies an ephemeris-predicted velocity *vector*: the
    search is then a small box centred on it (``from_prior``) instead of the
    direction-agnostic symmetric grid from ``config`` (blind survey mode).
    """
    if config is None:
        config = PipelineConfig.default()
    backend = catalog_backend if catalog_backend is not None else _DEFAULT_CATALOG

    calibrated = _calibrate_batch(frames, config)
    if not calibrated:
        empty_stack = StackResult(
            stacked=np.zeros((0, 0), dtype=np.float64),
            noise_rms=0.0,
            n_frames=0,
            best_vx_px_s=0.0,
            best_vy_px_s=0.0,
            peak_snr=0.0,
        )
        return StackedPipelineResult(
            stack_result=empty_stack,
            stacked_detections=(),
            frame_detections=(),
            tracklets=(),
            reference_mjd=0.0,
        )

    shape = calibrated[0].data.shape

    # Solve each frame's WCS *before* masking — stars are the astrometric
    # reference, so they must be detected on the un-masked calibrated frame.
    # Each frame gets its own blind (or supplied) solve; the matched star
    # positions then drive that frame's star mask for the stack.
    frame_wcs: list[WCSSolution | None] = []
    mask_matches_per_frame: list[list[StarMatch]] = []
    for item in calibrated:
        if item.ctx.star_matches:
            star_dets: list[Detection] = []
        else:
            star_dets = detect_sources(
                item.data,
                noise_rms=item.noise_rms,
                snr_threshold=config.detection.snr_threshold,
                min_pixels=config.detection.min_streak_pixels,
                max_pixels=config.detection.max_streak_pixels,
                elongation_threshold=config.detection.elongation_threshold,
                    )
        wcs, mm = _solve_frame_wcs(star_dets, item.ctx, config, shape, backend)
        # OpTA.NOD.ACC accuracy_flag consumption: with skip_flagged_frames a
        # flagged frame is treated like an unsolved one for *astrometry*
        # (its forced-photometry points are not emitted).  Its star matches
        # are kept for masking either way — the masks are pixel-level and
        # tolerant of arcsec-scale WCS error, and dropping them would
        # reintroduce the sidereal bright-star FP mechanism (findings §12).
        # Default (annotate-only): the flag rides on FrameDetections below.
        if (
            wcs is not None
            and wcs.accuracy_flag
            and config.astrometry.skip_flagged_frames
        ):
            logger.info(
                "frame %d: plate solution accuracy-flagged and "
                "astrometry.skip_flagged_frames is set — astrometry skipped",
                item.ctx.frame_id,
            )
            wcs = None
        frame_wcs.append(wcs)
        mask_matches_per_frame.append(mm)

    frame_times_s, reference_mjd = _frame_times_relative_to_midpass(calibrated)
    # Per-frame drift-tracking star masks (e2e findings §12): each frame's
    # mask is built from that frame's OWN astrometric star positions, so on a
    # fixed mount the masks follow the sidereal drift instead of freezing the
    # frame-0 geometry (the source of the sidereal FP tracklets — see
    # _star_masks_per_frame / _likelihood_stack_and_detect docstrings).
    star_masks = _star_masks_per_frame(
        shape,
        mask_matches_per_frame,
        config.detection.mask_star_radius_px,
    )

    # One motion-safety decision for temporal-median subtraction, shared by
    # the velocity search, the winner re-score, and forced photometry: skip
    # the subtraction when even the fastest searched velocity cannot escape
    # the median's slow-mover blind spot (median_subtraction_safe).
    if velocity_prior is not None:
        v_search_max = math.hypot(
            abs(velocity_prior.vx_px_s) + velocity_prior.half_width_px_s,
            abs(velocity_prior.vy_px_s) + velocity_prior.half_width_px_s,
        )
    else:
        v_search_max = math.hypot(
            config.stacking.velocity_max_px_s, config.stacking.velocity_max_px_s
        )
    t_span_s = (
        max(frame_times_s) - min(frame_times_s) if len(frame_times_s) > 1 else 0.0
    )
    subtract_static = median_subtraction_safe(
        v_search_max, t_span_s, config.stacking.psf_sigma_px
    )

    stack_result, stacked_dets, det_velocities = _likelihood_stack_and_detect(
        calibrated,
        frame_times_s,
        config,
        velocity_prior,
        star_masks,
        subtract_static,
    )

    h, w = shape

    # Rolling-shutter correction (T-03).  The stacked peak sits at the
    # reference epoch (t_s = 0); its readout bias is the *constant* component of
    # the row-timing error.  In the shift-and-add architecture the velocity fit
    # already absorbs the *rate* (linear-in-time) component, so correcting only
    # the peak — once, at its reference-epoch row, then propagating by the
    # fitted rate — removes the residual offset without re-correcting what the
    # stack already handled (re-correcting *predicted* per-frame positions
    # would over-correct it; *measured* forced-photometry centroids carry
    # their frame's raw readout bias and get their own correction in the
    # expansion loop below).  The
    # correction is exact in pixel space, so the WCS pixel scale cancels and a
    # placeholder of 1.0 makes the angular velocities act as px/s directly.  The
    # reference row falls back to the frame centre when the configured value is
    # outside the frame, so a centred target sees zero correction.
    rs_correct = (
        config.astrometry.rolling_shutter_correction
        and config.rolling_shutter.enabled
    )
    ref_row = config.rolling_shutter.reference_row
    if not (0.0 <= ref_row < h):
        ref_row = h / 2.0

    def _corrected_peak(det: Detection, vx: float, vy: float) -> tuple[float, float]:
        if not rs_correct:
            return det.x, det.y
        return apply_rolling_shutter_correction(
            det.x,
            det.y,
            angular_velocity_x=vx,
            angular_velocity_y=vy,
            pixel_scale_arcsec=1.0,
            row_readout_us=config.rolling_shutter.row_readout_us,
            reference_row=ref_row,
        )

    # Per-detection velocities (F7): in blind coarse-to-fine mode each
    # candidate propagates along its OWN velocity vector; single-winner
    # paths supply the same vector for every detection.
    peak_positions = [
        _corrected_peak(det, vx, vy)
        for det, (vx, vy) in zip(stacked_dets, det_velocities)
    ]

    # Forced per-frame photometry (F8): every
    # expanded detection is *measured* on its own frame — matched-filter
    # SNR and ML flux at the predicted position, with the position refined
    # to the measured sub-pixel centroid whenever the local SNR clears the
    # refinement gate.  This replaces the model-generated positions
    # + repeated stacked SNR that made the linker's linear-fit QC circular
    # and could promote one coherent artifact into a full I-02 tracklet
    # (TODO.md pipeline.py:489, second half; docs/endgame-plan.md).  Faint
    # frames keep the predicted position but carry their honest low
    # per-frame SNR and forced flux (usable as light-curve points).  The
    # refinement gate is stacking.refine_snr_min (default 7 sigma — ~5
    # nominal plus a local-noise margin, see StackingConfig), deliberately
    # NOT detection.snr_threshold (3 sigma): at ~3 sigma per-frame the max
    # over the search disc is noise-dominated, so refinement snapped
    # marginal targets onto noise peaks, inflated the track's O-C RMS
    # ~2 px, and the linear-fit QC silently rejected solidly stack-detected
    # objects (e2e findings #11/#12).  Fallback (predicted) points lie on
    # the stack's velocity solution by construction, so for them the
    # linear-fit QC is a consistency check, not independent evidence — the
    # detection claim rests on the trials-corrected stacked SNR, and the
    # measured/forced distinction stays auditable in each point's per-frame
    # SNR (downstream weighting treats all points equally; nothing
    # double-counts fallback points as measured astrometry).  Accepted
    # refinements are deliberately NOT gated on consistency with the
    # prediction: a bright interloper (star residual, blend) refining away
    # from the predicted line is honest evidence that the component is not
    # a clean linear mover, and the linear-fit QC uses exactly that
    # evidence to reject it — suppressing or dropping such points was
    # found to launder contaminated components into hundreds of
    # QC-passing tracklets (findings #12).
    if subtract_static:
        work_frames, _static = temporal_median_subtract(
            [item.data for item in calibrated]
        )
    else:
        work_frames = [
            np.asarray(item.data, dtype=np.float64) for item in calibrated
        ]
    psf_kernel = gaussian_psf_kernel(config.stacking.psf_sigma_px)
    # Per-detection matched kernels (F4): a streaking mover is measured with
    # a kernel matched to its own trail |v|*exposure; the round PSF is used
    # when the streak kernel is disabled or the trail is negligible.  Built
    # once per detection (velocity is frame-independent within a window).
    exposure_s = _streak_exposure_s(
        config.stacking, _frame_cadence_s(frame_times_s)
    )
    det_kernels = [
        matched_streak_kernel(
            config.stacking.psf_sigma_px, vx, vy, exposure_s,
            min_length_px=config.stacking.streak_min_length_px,
        )
        if exposure_s > 0.0
        else psf_kernel
        for (vx, vy) in det_velocities
    ]

    frame_dets: list[FrameDetections] = []
    for idx, (item, t_s, wcs) in enumerate(
        zip(calibrated, frame_times_s, frame_wcs)
    ):
        if wcs is None:
            continue  # frame could not be plate-solved; skip its astrometry
        pixel_dets: list[Detection] = []
        for j, (det, (peak_x, peak_y), (det_vx, det_vy)) in enumerate(
            zip(stacked_dets, peak_positions, det_velocities)
        ):
            x = peak_x + det_vx * t_s
            y = peak_y + det_vy * t_s
            if not (0.0 <= x < w and 0.0 <= y < h):
                continue
            # Search radius = predicted-position uncertainty, floored at 2 px.
            # It bounds how far the measured centroid may sit from the track
            # prediction, which is set by the velocity fit + integer shift
            # rounding (≤0.5 px/frame) and the stacked-peak localization
            # (~1 px) — NOT by the matched-filter kernel width.  Scaling it
            # purely as 2·σ let it collapse to 1.7 px at the synth-matched
            # σ ≈ 0.85 (from 3.0 px at the old 1.5), too tight to reach the
            # true centroid of a fast within-exposure-trailed mover (the RS
            # fast-mover test degraded to 3.5″); the 2 px floor decouples it.
            m = forced_measurement(
                work_frames[idx],
                item.noise_rms,
                det_kernels[j],
                x,
                y,
                search_radius_px=max(2.0 * config.stacking.psf_sigma_px, 2.0),
                refine_snr_min=config.stacking.refine_snr_min,
            )
            if m.refined:
                # A *measured* centroid carries the frame's raw readout
                # bias — the stacked-peak correction above never touched
                # it — so it needs its own T-03 correction at its own
                # row.  Predicted (un-refined) positions descend from
                # the corrected peak and must not be corrected twice.
                x, y = _corrected_peak(
                    replace(det, x=m.x_px, y=m.y_px), det_vx, det_vy
                )
            else:
                x, y = m.x_px, m.y_px
            snr, flux = m.snr, m.flux_e
            pixel_dets.append(
                Detection(
                    x=x,
                    y=y,
                    snr=snr,
                    elongation=max(
                        det.elongation, config.detection.elongation_threshold + 1.0
                    ),
                    angle_deg=float(
                        np.degrees(np.arctan2(det_vy, det_vx))
                    ),
                    n_pixels=det.n_pixels,
                    flux_e=flux,
                    is_streak=True,
                    fwhm_px=det.fwhm_px,
                )
            )

        if not pixel_dets:
            continue

        astro = astrometrise_detections(pixel_dets, wcs)
        frame_dets.append(
            FrameDetections(
                detections=tuple(astro),
                utc_mjd=item.ctx.utc_mjd,
                frame_id=item.ctx.frame_id,
                node_id=item.ctx.node_id,
                wcs_accuracy_flag=wcs.accuracy_flag,
            )
        )

    tracklets = _link_frame_detections(frame_dets, config)
    return StackedPipelineResult(
        stack_result=stack_result,
        stacked_detections=tuple(stacked_dets),
        frame_detections=tuple(frame_dets),
        tracklets=tuple(tracklets),
        reference_mjd=reference_mjd,
    )


def _streak_exposure_s(stk, cadence_s: float = 0.0) -> float:
    """Effective exposure (s) for streak matching: 0 unless ``streak_kernel``.

    Reads the opt-in ``StackingConfig.streak_kernel`` flag; every streak-aware
    call site (blind gate, winner re-score, forced photometry) shares one
    on/off decision.

    Exposure is the per-frame integration time = the frame *period* for a
    continuous-readout rolling shutter (shutter-open duty ≈ 1, OpTA.NOD.FRM).
    It is therefore derived from the **actual frame cadence** (median inter-
    frame Δt of the batch being processed, ``cadence_s``) — one source of
    truth that automatically tracks the real capture rate (21 fps full-res,
    25 fps ROI, …) instead of a hardcoded fps.  ``StackingConfig.exposure_time_s``
    is only a fallback for when the cadence is unavailable (< 2 timestamps).
    """
    if not bool(stk.streak_kernel):
        return 0.0
    if cadence_s > 0.0:
        return float(cadence_s)
    return float(stk.exposure_time_s)


def _frame_cadence_s(frame_times_s: list[float]) -> float:
    """Median inter-frame Δt (s) of a batch, or 0 when it cannot be inferred."""
    t = np.sort(np.asarray(frame_times_s, dtype=np.float64))
    if t.size < 2:
        return 0.0
    dts = np.diff(t)
    dts = dts[dts > 0.0]
    return float(np.median(dts)) if dts.size else 0.0


def _infer_fps(
    frame_list: list[tuple[np.ndarray, FrameContext]],
    config: PipelineConfig,
) -> float:
    """Infer capture fps from the median inter-frame time, with a config fallback."""
    times = np.asarray([ctx.utc_mjd for _, ctx in frame_list], dtype=np.float64)
    if times.size >= 2:
        dts = np.diff(np.sort(times)) * _SECONDS_PER_DAY
        dts = dts[dts > 0.0]
        if dts.size:
            return float(1.0 / np.median(dts))
    return config.stacking.fallback_fps


def _sky_track_candidate(
    t: Tracklet, t0_s: float, ra0_deg: float, cos_dec: float
) -> TrackCandidate:
    """Project a tracklet's linear sky fit into a flat arcsec plane at ``t0_s``.

    Coordinates are (ΔRA·cosδ, ΔDec) in arcsec about ``ra0_deg`` (RA wrapped
    the short way round), velocities the tracklet's fitted rates (arcsec/s,
    already RA·cosδ-projected).  The linear fit passes through the points'
    mean, so the state at ``t0_s`` is the mean position propagated by the
    fitted rate.  Packed as a :class:`~opta_pipeline.coarse_fine.TrackCandidate`
    (a plain linear-state container; the score fields are unused here) so
    :func:`_cross_window_duplicate` can compare two tracklets' predicted
    states over their shared time span.
    """
    times = np.array([p.utc_mjd * _SECONDS_PER_DAY for p in t.points])
    x = np.array(
        [
            ((p.ra_deg - ra0_deg + 180.0) % 360.0 - 180.0) * cos_dec * 3600.0
            for p in t.points
        ]
    )
    y = np.array([p.dec_deg * 3600.0 for p in t.points])
    t_mean = float(np.mean(times))
    return TrackCandidate(
        x_px=float(np.mean(x)) + t.ra_rate_arcsec_s * (t0_s - t_mean),
        y_px=float(np.mean(y)) + t.dec_rate_arcsec_s * (t0_s - t_mean),
        vx_px_s=t.ra_rate_arcsec_s,
        vy_px_s=t.dec_rate_arcsec_s,
        snr=0.0,
        flux_e=0.0,
        seed_snr=0.0,
    )


def _cross_window_duplicate(
    a: Tracklet, b: Tracklet, gate_arcsec: float
) -> bool:
    """True iff two window-tracklets describe the same object.

    Both tracklets' fitted linear sky tracks are evaluated at the epochs of
    their points inside the *temporal intersection* of the two arcs (the
    stacking-window overlap — the only span where both windows measured the
    same frames); they are duplicates only when the tracks stay within
    ``gate_arcsec`` at **every** shared epoch.  Duplicate *identity* is a
    much stronger statement than same-object *association*: two window fits
    of one object agree within astrometric error wherever they share data,
    so ``gate_arcsec`` must be an astrometric-error scale
    (``StackingConfig.dedup_radius_arcsec``, ~3 x the 10 arcsec OpTA.NOD.ACC
    budget), never the linker's 300 arcsec/frame association gate — reusing
    that gate silently deleted distinct co-moving neighbours 250 arcsec
    apart (deployment clusters / satellite trains).  Requiring agreement
    over the *whole* shared span (max, not min, separation) additionally
    enforces velocity agreement: two distinct tracks that merely cross
    inside the overlap touch at one instant but diverge across the span, so
    they are kept.  Disjoint spans (no window overlap) can never dedup.
    """
    ta = [p.utc_mjd * _SECONDS_PER_DAY for p in a.points]
    tb = [p.utc_mjd * _SECONDS_PER_DAY for p in b.points]
    lo = max(min(ta), min(tb))
    hi = min(max(ta), max(tb))
    if lo > hi:
        return False
    eval_times = np.array(
        sorted({t for t in (*ta, *tb) if lo <= t <= hi})
    )
    if eval_times.size == 0:
        return False
    ra0 = a.points[0].ra_deg
    decs = [p.dec_deg for p in (*a.points, *b.points)]
    cos_dec = math.cos(math.radians(float(np.mean(decs))))
    ca = _sky_track_candidate(a, lo, ra0, cos_dec)
    cb = _sky_track_candidate(b, lo, ra0, cos_dec)
    rel_t = eval_times - lo
    dx = (ca.x_px - cb.x_px) + (ca.vx_px_s - cb.vx_px_s) * rel_t
    dy = (ca.y_px - cb.y_px) + (ca.vy_px_s - cb.vy_px_s) * rel_t
    return bool(np.max(np.hypot(dx, dy)) < gate_arcsec)


def _dedup_cross_window(
    tracklets: list[Tracklet],
    window_of: list[int],
    gate_arcsec: float,
) -> list[Tracklet]:
    """Drop cross-window duplicates, keeping the best-measured arc.

    Greedy NMS in priority order (unflagged astrometry first —
    ``Tracklet.wcs_accuracy_flagged`` marks arcs containing a frame whose
    plate solve failed the OpTA.NOD.ACC self-check, so a flagged arc must
    never displace an unflagged duplicate — then more points, then lower
    linear-fit RMS): a tracklet is dropped when a kept tracklet from a
    *different* window duplicates it (:func:`_cross_window_duplicate`).
    Same-window tracklets are never tested against each other — the
    per-window linker already separated them.  Output preserves the input
    (window) order of the survivors.  ``gate_arcsec`` is the duplicate-
    identity radius (``StackingConfig.dedup_radius_arcsec``, an
    astrometric-error scale — see :func:`_cross_window_duplicate` for why
    the association gate must not be reused here).
    """
    order = sorted(
        range(len(tracklets)),
        key=lambda i: (
            tracklets[i].wcs_accuracy_flagged,
            -len(tracklets[i].points),
            tracklets[i].rms_arcsec,
        ),
    )
    kept_idx: list[int] = []
    for i in order:
        dup = any(
            window_of[k] != window_of[i]
            and _cross_window_duplicate(tracklets[k], tracklets[i], gate_arcsec)
            for k in kept_idx
        )
        if not dup:
            kept_idx.append(i)
    return [tracklets[i] for i in sorted(kept_idx)]


def run_windowed_track_and_stack(
    frames: Iterable[tuple[np.ndarray, FrameContext]],
    config: PipelineConfig | None = None,
    *,
    catalog_backend: CatalogBackend | None = None,
    velocity_prior: VelocityPrior | None = None,
) -> WindowedStackResult:
    """Track-and-stack a long pass by splitting it into locally-linear windows.

    Shift-and-add stacks every frame against a single (vx, vy) hypothesis, which
    only holds while the sky-track is straight.  A full ~1-minute LEO transit
    curves across a wide FOV, so this splits the time-sorted sequence into
    windows of ``config.stacking.window_duration_s`` (optionally overlapping by
    ``window_overlap_s``) and runs :func:`run_track_and_stack` on each.  Each
    window is short enough to be locally linear, so the per-window tracklets are
    valid on their own; merging them into a single arc is a separate step.

    With ``window_overlap_s > 0`` every object (and every false positive) in
    an overlap region is measured by *two* windows, so concatenating the
    per-window tracklets verbatim double-counts it — inflating detections/hr
    and FAR.  ``tracklets`` is therefore deduplicated across window
    boundaries in trajectory space (:func:`_dedup_cross_window`: the fitted
    sky tracks must agree within ``stacking.dedup_radius_arcsec`` — an
    astrometric-error scale, ~3 x the OpTA.NOD.ACC budget — over the whole
    shared time span), keeping the best-measured arc of each duplicate
    group.  ``window_results`` still
    carries every window's full (un-deduplicated) output for diagnostics.
    At the shipped ``window_overlap_s = 0.0`` the dedup pass is skipped
    entirely — behaviour is byte-identical to plain concatenation.

    Two accounting caveats of the dedup pass (#142 follow-ups, both by
    design until the correlator rework lands):

    * A *true* duplicate whose two window fits disagree by more than the
      identity radius anywhere on the shared span is **double-counted, not
      merged** — failing the identity gate means the pass cannot assert
      the two arcs are one object without reintroducing the association
      gate that silently deleted co-moving neighbours.  Detections/hr and
      FAR derived from overlapping-window runs carry that (upward) bias
      for poorly-fit arcs.
    * Surviving tracklets keep their original ``Wnn-``-prefixed ids
      (provenance: which window measured the kept arc), so id numbering is
      non-contiguous after dedup.  Ids are labels, not indices — never
      derive counts or ordering from them.

    ``window_duration_s <= 0`` processes the whole sequence as one window, i.e.
    it is identical to calling :func:`run_track_and_stack` directly.
    """
    if config is None:
        config = PipelineConfig.default()

    frame_list = sorted(frames, key=lambda fc: fc[1].utc_mjd)
    n = len(frame_list)
    if n == 0:
        return WindowedStackResult(window_results=(), tracklets=(), window_bounds=())

    fps = _infer_fps(frame_list, config)
    win_dur = config.stacking.window_duration_s
    if win_dur and win_dur > 0.0:
        win_frames = max(1, round(win_dur * fps))
        overlap_frames = round(config.stacking.window_overlap_s * fps)
        overlap_frames = max(0, min(win_frames - 1, overlap_frames))
    else:
        win_frames = n
        overlap_frames = 0
    step = max(1, win_frames - overlap_frames)

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + win_frames, n)
        bounds.append((start, end))
        if end >= n:
            break
        start += step

    window_results: list[StackedPipelineResult] = []
    all_tracklets: list[Tracklet] = []
    window_of: list[int] = []
    for wi, (s, e) in enumerate(bounds):
        res = run_track_and_stack(
            frame_list[s:e],
            config=config,
            catalog_backend=catalog_backend,
            velocity_prior=velocity_prior,
        )
        window_results.append(res)
        for t in res.tracklets:
            all_tracklets.append(replace(t, object_id=f"W{wi:02d}-{t.object_id}"))
            window_of.append(wi)

    # Cross-window dedup (see the docstring): only overlapping windows can
    # measure the same object twice, so the pass is gated on an actual
    # overlap — at the shipped window_overlap_s = 0.0 this branch never
    # runs and the output is byte-identical to plain concatenation.
    if overlap_frames > 0 and len(bounds) > 1:
        all_tracklets = _dedup_cross_window(
            all_tracklets, window_of, config.stacking.dedup_radius_arcsec
        )

    return WindowedStackResult(
        window_results=tuple(window_results),
        tracklets=tuple(all_tracklets),
        window_bounds=tuple(bounds),
        fps=fps,
    )


#: ``object_id`` prefix marking a tracklet that came from the stacked
#: track-before-detect path rather than the per-frame linker.  The two paths
#: number their tracklets independently (``SAT-0001``, …), so the union
#: emitted by :func:`run_pipeline` / :meth:`PipelineStream.flush` needs one
#: side relabelled or two *distinct* objects could share an I-02
#: ``object_id``.  Same convention (and same purpose — provenance) as the
#: ``Wnn-`` prefix in :func:`run_windowed_track_and_stack`.  Ids are labels,
#: not indices: never derive counts or ordering from them.
_STACKED_ID_PREFIX = "STK-"


def _merge_detection_paths(
    per_frame: list[Tracklet],
    stacked: list[Tracklet],
    config: PipelineConfig,
) -> list[Tracklet]:
    """Union the per-frame and stacked tracklets, dropping cross-path duplicates.

    A source bright enough for single-frame detection is normally found by
    *both* paths, so concatenating them verbatim double-counts it.  The two
    lists are therefore run through the same trajectory-space identity gate
    the overlapping-window merge uses (:func:`_dedup_cross_window` with the
    per-frame path as "window" 0 and the stacked path as "window" 1): the two
    fitted sky tracks must agree within ``stacking.dedup_radius_arcsec`` — an
    astrometric-error scale, ~3 x the OpTA.NOD.ACC budget — over their whole
    shared time span, and the better-measured arc survives.  Tracklets from
    the *same* path are never tested against each other (the linker,
    respectively the blind search's ghost NMS, already separated them).

    The gate is deliberately **not** ``TrackletConfig.max_sep_arcsec``: reusing
    the linker's 300 arcsec/frame *association* gate as an *identity* gate
    silently deleted distinct co-moving neighbours (deployment clusters,
    satellite trains) — see :func:`_cross_window_duplicate` and issue #142.
    The documented consequence carries over here: a true duplicate whose two
    fits disagree by more than the identity radius anywhere on the shared span
    is **double-counted, not merged**.
    """
    return _dedup_cross_window(
        per_frame + stacked,
        [0] * len(per_frame) + [1] * len(stacked),
        config.stacking.dedup_radius_arcsec,
    )


def run_pipeline(
    frames: Iterable[tuple[np.ndarray, FrameContext]],
    config: PipelineConfig | None = None,
    *,
    catalog_backend: CatalogBackend | None = None,
) -> list[Tracklet]:
    """Batch-process a sequence of frames and return quality-controlled tracklets.

    Runs the fast per-frame chain (``run_frame`` → Hungarian linker → linear-fit
    QC) and, whenever ``config.stacking.enabled``, **also** the blind stacked
    track-before-detect search (:func:`run_track_and_stack`), returning the
    deduplicated union of the two (:func:`_merge_detection_paths`).

    The stacked search is *not* a fallback for an empty per-frame result.  It
    used to be, and that made one bright mover (aircraft, flare, bright
    satellite) suppress the faint-mover search the array exists for — the
    per-frame path finding *anything* skipped the stack for the whole pass.
    Both paths now always run, at the cost of paying the blind search on every
    call; set ``stacking.enabled = False`` for a per-frame-only run.

    Stacked-path tracklets carry the ``STK-`` ``object_id`` prefix so their
    provenance (and id uniqueness across the union) is preserved.

    Parameters
    ----------
    frames : Iterable of (raw_array, FrameContext) pairs
        Frames in any order; sorted internally by utc_mjd inside link_detections.
    config : PipelineConfig | None
        Pipeline parameters.  Loads pipeline_defaults.yaml if None.
    catalog_backend : CatalogBackend | None
        Reference catalog for the blind solve (procedural by default; Gaia DR3
        in production — I-03).  Forwarded to both paths.

    Returns
    -------
    list[Tracklet]
        Accepted tracklets after Hungarian linking and linear-fit QC, unioned
        with the stacked search's tracklets and deduplicated across the two.
        Per-frame tracklets come first; order within each path is preserved.
    """
    if config is None:
        config = PipelineConfig.default()

    frame_list = list(frames)
    frame_dets: list[FrameDetections] = []
    for raw, ctx in frame_list:
        fd = run_frame(raw, ctx, config=config, catalog_backend=catalog_backend)
        if fd is not None:
            frame_dets.append(fd)

    tracklets = _link_frame_detections(frame_dets, config)
    if not config.stacking.enabled:
        return tracklets

    stacked = run_track_and_stack(
        frame_list, config=config, catalog_backend=catalog_backend
    )
    return _merge_detection_paths(
        tracklets,
        [
            replace(t, object_id=f"{_STACKED_ID_PREFIX}{t.object_id}")
            for t in stacked.tracklets
        ],
        config,
    )


class PipelineStream:
    """Streaming interface: push one frame at a time, flush at end-of-pass.

    Accumulates frame detections until flush() is called.  push() always
    returns an empty list (no tracklets can be closed mid-pass without
    bookkeeping; use flush() at end-of-pass).  A future incremental
    implementation would emit tracklets whose gap limit has been exceeded.

    Usage
    -----
        stream = PipelineStream(config)
        for raw, ctx in frame_source:
            stream.push(raw, ctx)
        tracklets = stream.flush()
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        catalog_backend: CatalogBackend | None = None,
    ) -> None:
        """Load defaults and initialise empty per-pass detection buffers."""
        self._config = config if config is not None else PipelineConfig.default()
        self._catalog_backend = catalog_backend
        self._frame_dets: list[FrameDetections] = []
        self._frames: list[tuple[np.ndarray, FrameContext]] = []

    def push(self, raw: np.ndarray, ctx: FrameContext) -> list[Tracklet]:
        """Accumulate one frame's detections.  Returns an empty list."""
        self._frames.append((raw, ctx))
        fd = run_frame(
            raw, ctx, config=self._config, catalog_backend=self._catalog_backend
        )
        if fd is not None:
            self._frame_dets.append(fd)
        return []

    def flush(self) -> list[Tracklet]:
        """Close all open tracks and return accepted tracklets. Resets state.

        End-of-pass counterpart of :func:`run_pipeline` and identical in
        semantics: the per-frame linker's tracklets are unioned with the blind
        stacked search's whenever ``stacking.enabled`` (deduplicated by
        :func:`_merge_detection_paths`; stacked ids carry the ``STK-``
        prefix), **not** used as a condition for skipping it.  The old
        "stack only if the linker found nothing" rule let one bright mover
        suppress the faint-mover search for the whole pass.

        The stream's ``catalog_backend`` is forwarded to the stacked search,
        exactly as ``push`` forwards it to ``run_frame`` — dropping it here
        silently sent the stacked path's per-frame blind solves to the default
        procedural catalog instead of the configured one (e.g. Gaia DR3).
        """
        result = _link_frame_detections(self._frame_dets, self._config)
        if self._config.stacking.enabled:
            stacked = run_track_and_stack(
                self._frames,
                config=self._config,
                catalog_backend=self._catalog_backend,
            )
            result = _merge_detection_paths(
                result,
                [
                    replace(t, object_id=f"{_STACKED_ID_PREFIX}{t.object_id}")
                    for t in stacked.tracklets
                ],
                self._config,
            )
        self._frame_dets.clear()
        self._frames.clear()
        return result
