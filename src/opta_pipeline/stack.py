"""Track-and-stack module for OpTA pipeline (WP-A5).

Implements blind shift-and-add stacking over a (vx, vy) velocity hypothesis
grid.  For each hypothesis the calibrated frames are shifted to register a
hypothesised linearly-moving object and summed.  The hypothesis yielding the
highest peak SNR in the stacked image is returned by default; callers may
optionally refine the strongest peak candidates with a downstream scorer such
as connected-component SNR.

Physics
-------
For N independently-exposed frames each with per-pixel noise σ:

    Stacked signal at (x₀, y₀) = N_vis × S₁     (N_vis ≤ N in-FOV frames)
    Stacked noise  at (x₀, y₀) = σ × √N_vis     (independent Poisson frames)
    Stacked SNR                 = SNR₁ × √N_vis

The ``noise_rms`` in :class:`StackResult` uses N (not N_vis) as a
conservative upper bound; when scipy fills out-of-bounds pixels with ``cval=0``
frames that shift entirely off-frame contribute no noise at the reference
position, so the actual noise is never worse than σ × √N.

This √N SNR boost closes **OpTA.NOD.DET** at the pipeline level: a 125-frame
pass at 25 fps gives a √125 ≈ 11× improvement, adding ≈ 2.5 limiting
magnitudes beyond the single-frame threshold (T-08 closed).

Usage
-----
    from opta_pipeline.stack import Stacker, StackResult
    import numpy as np

    # calibrated, background-subtracted frames
    frames = [cal.data for cal in calibrated_frames]
    noise_rms = max(calibrated_frames[0].background_rms, 1.0)

    fps = 25.0
    times_s = [k / fps for k in range(len(frames))]

    # velocity grid in pixels/second — ±400 px/s covers ±1 deg/s at
    # 9.12 arcsec/px (IMX585 + 85 mm) or ±2.5 deg/s at 22.2 arcsec/px
    # (35 mm baseline, pipeline_defaults.yaml)
    vx = np.arange(-400.0, 410.0, 20.0)
    vy = np.array([0.0])       # known horizontal motion; use 2-D for blind

    result = Stacker(vx, vy).stack(frames, noise_rms, times_s)

    # pass stacked frame directly to detect_sources
    from opta_pipeline.detect import detect_sources
    detections = detect_sources(result.stacked, noise_rms=result.noise_rms)

Notes
-----
- Shifts use ``scipy.ndimage.shift`` with explicit ``order=1`` (bilinear,
  ``cval=0``).  Out-of-bounds pixels are filled with 0 so off-frame regions do
  not inflate the noise at the satellite position.  Bilinear interpolation
  slightly correlates neighbouring pixels, so ``noise_rms = σ√N`` remains a
  (mild) upper bound on the true stacked noise.
- Pass ``frame_times_s`` with t=0 at the *midpass* frame (e.g., subtract the
  median frame MJD from all timestamps) to keep the satellite reference
  position inside the frame for all hypotheses.
- CPU NumPy / SciPy reference implementation; GPU optimisation is out of scope
  for WP-A5.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import shift as ndimage_shift

__all__ = [
    "StackResult",
    "Stacker",
    "stack_frames",
    "symmetric_velocity_axis",
]


def symmetric_velocity_axis(v_max: float, step: float) -> np.ndarray:
    """Velocity axis ``[−v_max, …, 0, …, +v_max]`` sampled at ``step``.

    Built by mirroring the half-axis ``arange(0, v_max + step/2, step)``, so
    ``0.0`` is always a node and the two arms are exact negatives of each
    other.  The direct ``arange(-v_max, v_max + step/2, step)`` only has those
    properties when ``v_max`` is an integer multiple of ``step``; otherwise it
    walks off-node — at ``v_max=162, step=20`` it runs [−162 … +158] with no
    zero node at all, dropping the static hypothesis that the ``velocity_min``
    cut and the from_prior offsets both assume exists.  For a ``v_max`` that is
    an exactly-representable multiple of ``step`` (every production and test
    grid in this repo) the returned nodes are bit-identical to the direct form,
    so adopting this helper cannot move a pinned detection number.

    ``v_max`` is a *rounded* bound, not a hard ceiling: the axis ends at the
    node nearest ``v_max``, so the extreme node satisfies
    ``|node| <= v_max + step/2`` and may exceed ``v_max`` by up to half a step
    when ``v_max`` falls past the midpoint between nodes (175/20 → ±180).
    Rounding outward is the safe direction — the search box then covers the
    requested band instead of stopping short of it — and it costs at most one
    extra node per arm.

    Shared by :class:`Stacker`, :class:`~opta_pipeline.likelihood.PsiPhiStacker`
    and :func:`~opta_pipeline.coarse_fine.blind_coarse_fine_search` — the grid
    convention is a search-wide invariant, not a per-module detail.
    """
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    half = np.arange(0.0, float(v_max) + step * 0.5, step)
    return np.concatenate([-half[:0:-1], half])


@dataclass(frozen=True)
class StackResult:
    """Result of a track-and-stack operation.

    Attributes
    ----------
    stacked : np.ndarray
        Float64 image: sum of all shifted frames for the best velocity
        hypothesis.  Background-subtracted; same shape as one input frame.
        Pass to :func:`~opta_pipeline.detect.detect_sources` with
        ``noise_rms=result.noise_rms``.
    noise_rms : float
        Conservative per-pixel noise estimate:
        ``single_frame_noise_rms × √n_frames``.  Use this as the
        ``noise_rms`` argument to ``detect_sources``; it never under-
        estimates the true noise.
    n_frames : int
        Total number of frames stacked (including frames with no satellite).
    best_vx_px_s : float
        x-velocity hypothesis (pixels/second) that maximised peak SNR.
    best_vy_px_s : float
        y-velocity hypothesis (pixels/second) that maximised peak SNR.
    peak_snr : float
        ``max(stacked) / noise_rms`` for the selected hypothesis.  With a
        refinement scorer this may not be the global maximum of
        ``response_grid``.
    response_grid : np.ndarray | None
        Optional 2-D array of peak SNR values with shape
        ``(len(vx_grid), len(vy_grid))``.  This records the blind-search
        velocity response for diagnostics and publication figures.
    """

    stacked: np.ndarray = field(hash=False, compare=False)
    noise_rms: float
    n_frames: int
    best_vx_px_s: float
    best_vy_px_s: float
    peak_snr: float
    response_grid: np.ndarray | None = field(default=None, hash=False, compare=False)


class Stacker:
    """Blind track-and-stack over a (vx, vy) velocity search grid.

    For each velocity hypothesis (vx, vy) all frames are shifted by
    ``(−vx·t_k, −vy·t_k)`` so a linearly-moving object at
    ``(x₀ + vx·t_k, y₀ + vy·t_k)`` registers to ``(x₀, y₀)``.
    The shifted frames are summed and scored by peak SNR.

    Parameters
    ----------
    vx_grid : array-like
        x-velocity hypotheses in **pixels/second**.
    vy_grid : array-like
        y-velocity hypotheses in **pixels/second**.
    """

    def __init__(
        self,
        vx_grid: np.ndarray,
        vy_grid: np.ndarray,
        velocity_min_px_s: float = 0.0,
    ) -> None:
        """Store velocity search grids in pixels/second.

        ``velocity_min_px_s`` excludes hypotheses with ``hypot(vx, vy)`` below it
        from the search; static sources only stack coherently near v=0, so this
        suppresses their false peaks.  ``0.0`` searches the full grid.
        """
        self.vx_grid = np.asarray(vx_grid, dtype=np.float64)
        self.vy_grid = np.asarray(vy_grid, dtype=np.float64)
        self.velocity_min_px_s = float(velocity_min_px_s)

    @classmethod
    def from_config(cls, config) -> Stacker:
        """Build a Stacker from a :class:`~opta_pipeline.config.PipelineConfig`.

        The velocity grid is symmetric:
        ``[−velocity_max, …, 0, …, +velocity_max]`` at ``velocity_step``
        resolution (both axes).

        Parameters
        ----------
        config : PipelineConfig | StackingConfig
            Object with ``velocity_max_px_s`` and ``velocity_step_px_s``
            attributes, or with a ``.stacking`` sub-config that has them.
        """
        cfg = getattr(config, "stacking", config)
        vmax = cfg.velocity_max_px_s
        vstep = cfg.velocity_step_px_s
        grid = symmetric_velocity_axis(vmax, vstep)
        vmin = float(getattr(cfg, "velocity_min_px_s", 0.0))
        return cls(vx_grid=grid, vy_grid=grid, velocity_min_px_s=vmin)

    @classmethod
    def from_prior(
        cls,
        vx0_px_s: float,
        vy0_px_s: float,
        half_width_px_s: float,
        step_px_s: float,
        velocity_min_px_s: float = 0.0,
    ) -> Stacker:
        """Build a Stacker whose grid brackets a predicted velocity *vector*.

        For a scheduled pass the ephemeris predicts the sky-track rate *and
        direction*, so the search is a small box centred on ``(vx0, vy0)`` of
        half-width ``half_width_px_s`` (the prior uncertainty) at ``step_px_s``
        resolution — not the direction-agnostic symmetric box of
        :meth:`from_config`.  Fewer hypotheses ⇒ a lower extreme-value detection
        floor ⇒ better sensitivity; ``step_px_s`` should be ≈ PSF_px / T_window
        so the worst-case off-node registration smear stays sub-PSF.
        """
        if step_px_s <= 0.0:
            raise ValueError(f"step_px_s must be > 0, got {step_px_s}")
        hw = max(0.0, float(half_width_px_s))
        offsets = symmetric_velocity_axis(hw, step_px_s)
        return cls(
            vx_grid=vx0_px_s + offsets,
            vy_grid=vy0_px_s + offsets,
            velocity_min_px_s=velocity_min_px_s,
        )

    def stack(
        self,
        frames: list[np.ndarray],
        noise_rms: float,
        frame_times_s: list[float] | None = None,
        *,
        refine_top_k: int = 1,
        candidate_scorer: Callable[[np.ndarray, float], float] | None = None,
    ) -> StackResult:
        """Shift-and-add all frames over the velocity grid; return best result.

        Parameters
        ----------
        frames : list of np.ndarray
            Calibrated, background-subtracted frames (float, any dtype).
            All must share the same ``(height, width)`` shape.
        noise_rms : float
            Per-pixel noise for a **single** frame — e.g.,
            ``max(CalibratedFrame.background_rms, 1.0)``.
        frame_times_s : list of float | None
            Time of each frame in seconds, relative to the chosen reference
            epoch (t = 0).  Set t = 0 at the **midpass** frame so the
            satellite registers inside the frame for all hypotheses.  If
            ``None``, assumes t_k = k (integer seconds from frame 0).
        refine_top_k : int
            Number of peak-SNR velocity candidates to pass through
            ``candidate_scorer``.  Ignored when ``candidate_scorer`` is None.
        candidate_scorer : Callable[[np.ndarray, float], float] | None
            Optional refinement scorer receiving ``(stacked, stacked_noise)``.
            Return a finite score to select by a downstream statistic, e.g.
            connected-component SNR.  If every candidate returns NaN/inf, the
            peak-SNR winner is retained.

        Returns
        -------
        StackResult
            Best-hypothesis stacked image plus diagnostic metrics.

        Raises
        ------
        ValueError
            If ``frames`` is empty or ``frame_times_s`` has the wrong length.
        """
        if not frames:
            raise ValueError("frames must not be empty")
        n = len(frames)
        shape = frames[0].shape

        if frame_times_s is None:
            frame_times_s = list(range(n))
        if len(frame_times_s) != n:
            raise ValueError(
                f"len(frame_times_s)={len(frame_times_s)} != len(frames)={n}"
            )

        t = np.asarray(frame_times_s, dtype=np.float64)
        stacked_noise = noise_rms * math.sqrt(n)

        # Pre-convert to float64 once — avoids repeated casting in the inner loop
        frames_f64 = [f.astype(np.float64) for f in frames]

        response_grid = np.zeros(
            (len(self.vx_grid), len(self.vy_grid)), dtype=np.float64
        )
        best_stack: np.ndarray | None = None
        best_vx = 0.0
        best_vy = 0.0
        best_snr = -1.0
        top_k = max(1, int(refine_top_k)) if candidate_scorer is not None else 1
        # Cache the stacked image alongside each top-K candidate so the refine
        # pass can re-score it without re-stacking (shift-and-add is the cost).
        top_candidates: list[tuple[float, float, float, np.ndarray]] = []

        # Velocity hypotheses to score, with near-zero (static) exclusion.  Cells
        # below velocity_min_px_s are left at response_grid=0 and never win the
        # peak, so residual stars/hot pixels cannot form a false static peak.
        vmin = self.velocity_min_px_s
        hypotheses = [
            (i, j, float(vx), float(vy))
            for i, vx in enumerate(self.vx_grid)
            for j, vy in enumerate(self.vy_grid)
        ]
        if vmin > 0.0:
            passed = [h for h in hypotheses if math.hypot(h[2], h[3]) >= vmin]
            if passed:  # keep full grid if exclusion would empty the search
                hypotheses = passed

        for i, j, vx, vy in hypotheses:
            stacked = _shift_and_add(frames_f64, t, vx, vy, shape)
            # Score the peak *above the stacked background*, not its absolute
            # value.  Shift-and-add sums N frames, so any residual per-frame
            # background accumulates ∝N into a DC pedestal (while noise grows
            # only ∝√N); ranking on the raw peak lets that pedestal — not the
            # source — drive the search.  The robust median is a cheap DC
            # estimate (≈0 for clean background-subtracted frames, so this is a
            # no-op there); the 2-D background is removed downstream before
            # detection.
            peak = float(stacked.max() - np.median(stacked))
            snr = peak / stacked_noise if stacked_noise > 0 else 0.0
            response_grid[i, j] = snr
            if candidate_scorer is not None:
                _record_top_candidate(top_candidates, top_k, snr, vx, vy, stacked)
            if snr > best_snr:
                best_snr = snr
                best_vx = vx
                best_vy = vy
                best_stack = stacked.copy()

        # best_stack is set on the first iteration; len(frames) >= 1 guarantees this
        assert best_stack is not None
        if candidate_scorer is not None:
            refined = _select_refined_candidate(
                stacked_noise,
                top_candidates,
                candidate_scorer,
            )
            if refined is not None:
                best_stack, best_vx, best_vy, best_snr = refined

        return StackResult(
            stacked=best_stack,
            noise_rms=stacked_noise,
            n_frames=n,
            best_vx_px_s=best_vx,
            best_vy_px_s=best_vy,
            peak_snr=best_snr,
            response_grid=response_grid,
        )


def stack_frames(
    frames: list[np.ndarray],
    noise_rms: float,
    frame_times_s: list[float] | None = None,
    vx_grid: np.ndarray | None = None,
    vy_grid: np.ndarray | None = None,
) -> StackResult:
    """Convenience wrapper: create a :class:`Stacker` and stack frames.

    Parameters
    ----------
    frames : list of np.ndarray
        Calibrated, background-subtracted frames.
    noise_rms : float
        Per-frame per-pixel noise estimate.
    frame_times_s : list of float | None
        Frame times (seconds) relative to t = 0.
    vx_grid, vy_grid : np.ndarray | None
        Velocity search grids (pixels/second).  Default: ±2 px/s in 0.5
        steps — suitable for near-stationary lab targets.  For LEO objects
        (hundreds of px/s: 1 deg/s ≈ 395 px/s at 9.12 arcsec/px / 85 mm,
        ≈ 162 px/s at 22.2 arcsec/px / 35 mm) build a grid explicitly.

    Returns
    -------
    StackResult
    """
    if vx_grid is None:
        vx_grid = symmetric_velocity_axis(2.0, 0.5)
    if vy_grid is None:
        vy_grid = symmetric_velocity_axis(2.0, 0.5)
    return Stacker(vx_grid=vx_grid, vy_grid=vy_grid).stack(
        frames, noise_rms, frame_times_s
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shift_and_add(
    frames: list[np.ndarray],
    times_s: np.ndarray,
    vx_px_s: float,
    vy_px_s: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Sum frames shifted to register the hypothesis (vx_px_s, vy_px_s).

    Frame k is shifted by ``(−vy·t_k, −vx·t_k)`` in (row, col) order,
    bringing a satellite at ``(x₀ + vx·t_k, y₀ + vy·t_k)`` to ``(x₀, y₀)``.
    scipy.ndimage.shift fills out-of-bounds pixels with ``cval=0``.

    ``order=1`` (bilinear) is explicit: scipy's default is a cubic spline
    (order=3), which is ~4.5× slower and rings around bright pixels —
    negative lobes that leak into the median pedestal estimate used for
    hypothesis scoring.
    """
    result = np.zeros(shape, dtype=np.float64)
    for frame, t in zip(frames, times_s):
        dx = vx_px_s * float(t)
        dy = vy_px_s * float(t)
        # scipy shift=(dr, dc): content at (r,c) → (r+dr, c+dc).
        # To move satellite from (y0+dy, x0+dx) to (y0, x0): shift=(-dy, -dx).
        shifted = ndimage_shift(
            frame, shift=(-dy, -dx), order=1, mode="constant", cval=0.0
        )
        result += shifted
    return result


def _record_top_candidate(
    candidates: list[tuple[float, float, float, np.ndarray]],
    top_k: int,
    snr: float,
    vx: float,
    vy: float,
    stacked: np.ndarray,
) -> None:
    """Keep the top-K peak-SNR candidates (with their stacked image) for refine."""
    candidates.append((snr, vx, vy, stacked.copy()))
    candidates.sort(key=lambda item: item[0], reverse=True)
    del candidates[top_k:]


def _select_refined_candidate(
    stacked_noise: float,
    candidates: list[tuple[float, float, float, np.ndarray]],
    candidate_scorer: Callable[[np.ndarray, float], float],
) -> tuple[np.ndarray, float, float, float] | None:
    """Return the best candidate by downstream score, or None if all fail.

    Re-scores the *cached* stacked images (no re-stacking) with a stronger
    statistic than the peak used for ranking — e.g. the component/aperture SNR,
    which separates a coherent point source from the peak-pixel noise floor
    that a single-pixel peak cannot (a matched-filter-like re-rank).
    """
    best: tuple[np.ndarray, float, float, float] | None = None
    best_score = -math.inf
    best_peak_snr = -math.inf

    for peak_snr, vx, vy, stacked in candidates:
        score = float(candidate_scorer(stacked, stacked_noise))
        if not math.isfinite(score):
            continue
        if score > best_score or (score == best_score and peak_snr > best_peak_snr):
            best_score = score
            best_peak_snr = peak_snr
            best = (stacked, vx, vy, peak_snr)

    return best
