"""Coarse-to-fine blind velocity search (Phase 2 of the stack rework).

Single-stage blind search at the correct velocity sampling (step ≈
PSF_fwhm / T_pass, assessment F1) costs (2·v_max·T/PSF)² hypotheses —
~2.6 M at full geometry (F2).  This module replaces it with a two-part
scheme built on the ψ/φ likelihood images:

**Stage 1 — seeding.**  The pass is split into short windows of duration
T₀; the stage-1 grid step is re-derived from the actual window duration
(step ≈ PSF_fwhm / T₀) so sampling is criterion-compliant within each
window by construction.  Every window is searched over the full ±v_max
box at that step; SNR-map peaks above a pre-threshold become track
candidates (position at the window-anchor epoch + velocity ± step).  The
auto T₀ (``seed_window_s = 0``) is **depth-matched by construction**:
T₀ = max(PSF_fwhm / coarse_step, 4 frames, T·(prethreshold/u*)²), where
the last term inverts the seeding floor prethreshold·√(T/T₀) ≤ u*(T)
(:func:`opta_pipeline.sensitivity.seeding_floor`), so the stage-1 depth
never truncates the u*-limited full-pass sensitivity.  Deeper windows
cost more (finer stage-1 step over the same box) — the documented price
of a seeding stage that keeps up with the gate.

**Stage 2 — pyramid refinement.**  Each candidate is refined by repeatedly
doubling the integration window around its seed while halving the velocity
step (always keeping step ≈ PSF_fwhm / T_window): a ~(2·parent/child + 1)²
≈ 25-hypothesis local search per level on a small ROI, until the window
covers the pass and the step reaches the full-pass criterion.  Cost per
candidate is *logarithmic* in the step ratio instead of quadratic.

**Final gate.**  Each refined candidate is re-scored over the full pass at
its final velocity and accepted only above the trials-corrected threshold
u* = poisson_tail_threshold(N_pix · N_hyp_fine_equivalent, ε) + margin —
the Bernstein bound of :mod:`opta_pipeline.likelihood` ("Detection
threshold" in its module docstring), with the bound scale ε estimated
from the frames/noise actually searched (never from a target scene) and
N_hyp_fine_equivalent the (2·v_max/fine_step)² grid the scheme is
equivalent to — a conservative trials budget, so the noise-only
false-alarm guarantee of the single-stage gate is preserved.  At ε = 0
this reduces exactly to the former Gaussian extreme-value bound
√(2 ln N); at the low counts of the shipped scenes ε > 0 covers the
sub-exponential Poisson tail the Gaussian bound measurably under-covered
(blank-scene FAR ≈ 9 %/scene measured vs ≪ 1 % predicted — closed by
this contract, 2026-07-28).

Sensitivity trade (documented, not hidden): stage 1 sees only T₀ of data,
so a candidate must reach ≈ prethreshold_sigma per window to be seeded.
The effective limiting full-pass SNR is therefore

    SNR_lim ≈ max(u*, prethreshold_sigma · √(T_pass / T₀))

instead of u* alone.  This is the classical multi-stage
track-before-detect trade (Stetzler et al. 2025 accept the same loss in
catalog space; KBMOD avoids it by GPU brute force). Tests characterize
both sides of the boundary; completeness studies should use this scheme.

Multi-object (assessment F7): all accepted candidates are returned, each
with its own velocity — not just the winner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opta_pipeline.likelihood import (
    DEFAULT_PSF_SIGMA_PX,
    _add_shifted,
    _is_scalar_noise_rms,
    _snr_from,
    bound_scale_epsilon,
    gaussian_psf_kernel,
    make_psi_phi,
    median_subtraction_safe,
    poisson_tail_threshold,
    quantisation_step_e,
    streak_length_px,
    streak_psf_kernel,
    temporal_median_subtract,
)
from opta_pipeline.stack import symmetric_velocity_axis

__all__ = [
    "CoarseFineResult",
    "TrackCandidate",
    "blind_coarse_fine_search",
]

_PHI_EPS = 1e-12
_MAX_SEEDS_PER_WINDOW = 50


@dataclass(frozen=True)
class TrackCandidate:
    """One accepted track hypothesis at the global reference epoch (t=0).

    ``snr`` is the full-pass calibrated matched-filter SNR (comparable to
    u*); ``flux_e`` the ML flux estimate Ψ/Φ; ``seed_snr`` the stage-1
    window SNR that seeded the candidate (diagnostic for the sensitivity
    trade).
    """

    x_px: float
    y_px: float
    vx_px_s: float
    vy_px_s: float
    snr: float
    flux_e: float
    seed_snr: float


@dataclass(frozen=True)
class CoarseFineResult:
    """Result of a blind coarse-to-fine search.

    ``candidates`` are the accepted tracks sorted by SNR (descending);
    empty when nothing crosses u*.  ``u_star`` is the applied final
    threshold; ``n_windows`` / ``window_frames`` describe the stage-1
    seeding geometry; ``fine_step_px_s`` the terminal velocity resolution.
    """

    candidates: tuple[TrackCandidate, ...]
    u_star: float
    n_windows: int
    window_frames: int
    coarse_step_px_s: float
    fine_step_px_s: float


def blind_coarse_fine_search(
    frames: list[np.ndarray],
    noise_rms: float | list[float],
    frame_times_s: list[float],
    *,
    v_max_px_s: float,
    coarse_step_px_s: float,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    prethreshold_sigma: float = 4.0,
    far_margin_sigma: float = 0.5,
    max_candidates: int = 20,
    velocity_min_px_s: float = 0.0,
    seed_window_s: float = 0.0,
    subtract_median: bool | None = None,
    seed_binning: int = 1,
    exposure_time_s: float = 0.0,
    streak_min_length_px: float = 0.0,
    masks: list[np.ndarray] | None = None,
) -> CoarseFineResult:
    """Run the blind two-stage search.  See module docstring.

    ``frame_times_s`` must be relative to the chosen reference epoch
    (t = 0 at midpass recommended); candidate positions are reported at
    that epoch.

    ``masks`` are optional per-frame True-=-invalid pixel masks (aligned
    with ``frames``), forwarded to :func:`make_psi_phi` for seeding,
    refinement, and the final streak re-score alike: invalid pixels carry
    neither signal nor statistical weight (V = ∞ ⇒ φ = 0).  The pipeline
    supplies per-frame drift-tracking catalog-star masks here, so the
    bright-star residuals that temporal-median subtraction leaves under
    sidereal drift can neither seed nor pass the final gate as candidates
    (e2e findings §12).

    ``subtract_median = None`` (default) removes the static scene by
    temporal median unless the whole ±v_max search band sits in the
    median's slow-mover blind spot
    (:func:`~opta_pipeline.likelihood.median_subtraction_safe`) — then
    subtraction would only destroy in-scope signal and is skipped.

    ``seed_binning`` > 1 runs stage-1 seeding on b×b sum-binned ψ/φ maps:
    every per-hypothesis cost drops ×b² (measured ≈ 3.4× wall time at
    b = 2), at three documented prices, all confined to *seeding* (the
    pyramid refinement and the final u* gate stay full-resolution):
    integer shifts quantize to b full pixels (still ≪ PSF for b = 2);
    seed positions are known to ±b/2 px (well inside the pyramid's ROI);
    and the binned SNR statistic is slightly over-dispersed because
    binning sums correlated ψ neighbours while φ sums only variances
    (measured std ≈ 1.2 at b = 2, σ_psf = 1.5 — the pre-threshold is
    applied to this map, so effective seeding depth shifts accordingly).

    ``exposure_time_s`` > 0 enables the within-exposure streak kernel (F4)
    at the final u* gate: seeding and pyramid refinement still use the round
    PSF (one kernel across the grid keeps seeding cheap), but each refined
    candidate — whose velocity is now known — is re-scored over the full
    pass with a kernel matched to its trail ``|v|·exposure`` before being
    gated.  Ψ/√Φ stays ~N(0,1) for that fixed kernel, so u* is unchanged.
    Candidates whose trail is below ``streak_min_length_px`` keep their
    round-PSF score (byte-identical to ``exposure_time_s = 0``).
    """
    if not frames:
        raise ValueError("frames must not be empty")
    n = len(frames)
    if len(frame_times_s) != n:
        raise ValueError("frame_times_s length mismatch")
    times = np.asarray(frame_times_s, dtype=np.float64)
    order = np.argsort(times)
    frames = [frames[i] for i in order]
    times = times[order]
    # Per-frame inputs must follow the time sort or they misalign.
    if masks is not None:
        if len(masks) != n:
            raise ValueError(f"len(masks)={len(masks)} != len(frames)={n}")
        masks = [masks[i] for i in order]
    if not _is_scalar_noise_rms(noise_rms):
        noise_rms = [float(noise_rms[i]) for i in order]

    work = [np.asarray(f, dtype=np.float64) for f in frames]
    # Bernstein bound scale of the detection contract — estimated from the
    # frames as ingested (pre median-subtraction: the quantisation ladder
    # is a property of the recorded values) and the same per-frame σ the
    # ψ/φ statistic is normalised with.
    quant_step = quantisation_step_e(work)
    if subtract_median is None:
        span = float(times[-1] - times[0]) if n > 1 else 0.0
        subtract_median = median_subtraction_safe(
            math.hypot(v_max_px_s, v_max_px_s), span, psf_sigma_px
        )
    if subtract_median:
        work, _ = temporal_median_subtract(work)
    kernel = gaussian_psf_kernel(psf_sigma_px)
    bound_scale = bound_scale_epsilon(
        kernel, noise_rms, n_frames=n, quant_step_e=quant_step
    )
    psi, phi = make_psi_phi(work, noise_rms, kernel, masks)

    b = max(1, int(seed_binning))
    if b > 1:
        psi_seed = [_bin_sum(m, b) for m in psi]
        phi_seed = [_bin_sum(m, b) for m in phi]
        # Σψ over a bin sums kernel-correlated neighbours while Σφ sums
        # only variances, so the naive binned SNR is over-dispersed by an
        # exact, kernel-computable factor (≈1.9 at b=2, σ_psf=1.5;
        # measured 1.92).  Dividing restores ~N(0,1) so prethreshold_sigma
        # keeps its meaning at any binning.
        snr_corr = _binned_snr_dispersion(kernel, b)
    else:
        psi_seed, phi_seed = psi, phi
        snr_corr = 1.0

    h, w = work[0].shape
    psf_fwhm = 2.355 * psf_sigma_px
    span = float(times[-1] - times[0]) if n > 1 else 0.0
    fine_step = psf_fwhm / span if span > 0 else coarse_step_px_s
    # Final full-pass gate (fine-equivalent trials budget + Bernstein
    # bound scale) — needed up front: the auto seed window is derived
    # from it so stage-1 depth matches the gate by construction.
    u_star = _u_star_fine(
        h, w, v_max_px_s, fine_step, far_margin_sigma, bound_scale
    )

    # ── Stage 1: seed in criterion-matched windows ─────────────────────
    # Window duration sets seeding DEPTH (per-window SNR ∝ √n_win must
    # clear the pre-threshold against the window map's own noise maximum
    # ≈ √(2 ln N_pix) ≈ 4.2σ); the stage-1 grid step is then re-derived
    # from the actual window duration so sampling stays criterion-matched
    # regardless of the configured coarse step.  ``seed_window_s = 0``
    # auto-selects max(PSF_fwhm/coarse_step, 4 frames, T·(pre/u*)²): the
    # last term inverts seeding_floor(T, T₀) = pre·√(T/T₀) ≤ u*(T), so
    # the auto depth cannot truncate the u*-limited sensitivity (the old
    # depth-agnostic rule silently did — TODO item closed 2026-07-28).
    dt = float(np.median(np.diff(times))) if n > 1 else 1.0
    depth_t0 = (
        span * (prethreshold_sigma / u_star) ** 2 if u_star > 0.0 else 0.0
    )
    t0_target = (
        seed_window_s
        if seed_window_s > 0.0
        else max(psf_fwhm / coarse_step_px_s, 4.0 * dt, depth_t0)
    )
    win_frames = max(1, int(round(t0_target / max(dt, 1e-9))))
    bounds = [(s, min(s + win_frames, n)) for s in range(0, n, win_frames)]
    t_win = win_frames * dt
    step1 = min(coarse_step_px_s, psf_fwhm / t_win) if t_win > 0 else coarse_step_px_s

    grid = symmetric_velocity_axis(v_max_px_s, step1)
    hypotheses = [
        (float(vx), float(vy))
        for vx in grid
        for vy in grid
        if velocity_min_px_s <= 0.0
        or math.hypot(float(vx), float(vy)) >= velocity_min_px_s
    ]

    # Seed tuples: (snr, x, y, vx, vy, v_unc, t_anchor, k_anchor).
    # Positions are anchored at the window-centre epoch t_anchor — stage-1
    # shifts are computed relative to it, so they stay ≤ v·T₀/2 instead of
    # v·t (which sweeps off-frame at pass edges), and a single-frame seed's
    # position is exact at its own epoch even with unknown velocity.
    seeds: list[tuple[float, float, float, float, float, float, float, int]] = []
    for s, e in bounds:
        n_win = e - s
        t_anchor = float(np.mean(times[s:e]))
        k_anchor = (s + e) // 2
        # Windows of 1 frame carry no velocity information: seed with the
        # full box as uncertainty (the pyramid then starts from ±v_max).
        win_hyps = hypotheses if n_win > 1 else [(0.0, 0.0)]
        v_unc = step1 if n_win > 1 else v_max_px_s
        # Fail-fast pedestal estimate for this window: the Poisson-median
        # DC offset is a property of the *frames* (skewed sky statistics),
        # not of the velocity hypothesis, so the v = 0 stack's median is a
        # good proxy for every hypothesis.  Hypotheses whose raw maximum
        # cannot reach the pre-threshold even after pedestal removal (with
        # a 0.5σ safety margin for hypothesis-to-hypothesis pedestal
        # variation from zero-filled borders) skip the exact median +
        # argpartition — the two dominant per-hypothesis costs after the
        # shift-adds.  Hypotheses that pass are scored exactly as before,
        # so accepted seeds are identical (verified: all coarse-fine tests
        # incl. the sensitivity-trade pins are unchanged).
        hb, wb = psi_seed[0].shape
        psi0 = np.zeros((hb, wb))
        phi0 = np.zeros((hb, wb))
        for k in range(s, e):
            psi0 += psi_seed[k]
            phi0 += phi_seed[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            snr0 = np.where(phi0 > _PHI_EPS, psi0 / np.sqrt(phi0), 0.0)
        pedestal0 = float(np.median(snr0)) / snr_corr
        for vx, vy in win_hyps:
            psi_sum = np.zeros((hb, wb))
            phi_sum = np.zeros((hb, wb))
            for k in range(s, e):
                dx = int(round(vx * (times[k] - t_anchor) / b))
                dy = int(round(vy * (times[k] - t_anchor) / b))
                _add_shifted(psi_sum, psi_seed[k], dy, dx)
                _add_shifted(phi_sum, phi_seed[k], dy, dx)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(phi_sum > _PHI_EPS,
                               psi_sum / np.sqrt(phi_sum), 0.0)
            if snr_corr != 1.0:
                snr = snr / snr_corr
            if float(snr.max()) - pedestal0 < prethreshold_sigma - 0.5:
                continue  # fail-fast: cannot seed even with pedestal slack
            snr = snr - float(np.median(snr))  # Poisson-median DC pedestal
            flat = np.argpartition(snr.ravel(), -_MAX_SEEDS_PER_WINDOW)[
                -_MAX_SEEDS_PER_WINDOW:
            ]
            for idx in flat:
                val = float(snr.ravel()[idx])
                if val < prethreshold_sigma:
                    continue
                yb, xb = divmod(int(idx), wb)
                # Binned index → full-resolution bin-centre coordinates.
                x_full = (xb + 0.5) * b - 0.5
                y_full = (yb + 0.5) * b - 0.5
                seeds.append(
                    (val, x_full, y_full, vx, vy, v_unc, t_anchor, k_anchor)
                )

    if not seeds:
        return CoarseFineResult(
            candidates=(),
            u_star=u_star,
            n_windows=len(bounds),
            window_frames=win_frames,
            coarse_step_px_s=coarse_step_px_s,
            fine_step_px_s=fine_step,
        )

    # ── Greedy NMS on (x, y, vx, vy) ───────────────────────────────────
    seeds.sort(key=lambda c: c[0], reverse=True)
    kept: list[tuple[float, float, float, float, float, float, float, int]] = []
    r_pos = 3.0 * psf_fwhm
    for cand in seeds:
        _, x, y, vx, vy, _, t_a, _k = cand
        x0, y0 = x - vx * t_a, y - vy * t_a  # propagate to t=0 for NMS
        dup = any(
            math.hypot(x0 - (k[1] - k[3] * k[6]), y0 - (k[2] - k[4] * k[6]))
            < r_pos + (k[5] + step1) * abs(t_a - k[6])
            and math.hypot(vx - k[3], vy - k[4]) <= k[5] + step1
            for k in kept
        )
        if not dup:
            kept.append(cand)
        if len(kept) >= max_candidates:
            break

    # ── Stage 2: pyramid refinement + final full-pass gate ────────────
    accepted: list[TrackCandidate] = []
    for seed_snr, x0, y0, vx, vy, v_unc, t_anchor, k_anchor in kept:
        refined = _pyramid_refine(
            psi, phi, times, x0, y0, vx, vy, v_unc,
            t_anchor=t_anchor, k_anchor=k_anchor,
            psf_fwhm=psf_fwhm, fine_step=fine_step,
            win_frames=win_frames, v_max=v_max_px_s,
        )
        if refined is None:
            continue
        x0, y0, vx, vy, snr, flux = refined
        # Honest final gate score.  The pyramid's ROI statistic is a *search*
        # metric, not a calibrated one: it is maximized over the ROI's own
        # per-frame integer-rounding realizations and centred on the ROI
        # median, which under-estimates the DC pedestal next to the selected
        # blob (measured on blank SMOKE scenes: +1.3σ rounding-selection and
        # +0.2σ pedestal bias — enough to lift 5.2σ noise over u* = 5.96).
        # Re-score every candidate over the full pass at its now-fixed
        # velocity with the full-map median as zero level — the statistic
        # family the u* trials budget actually describes.  Trailed
        # candidates use the matched streak kernel (F4); the rest reuse the
        # already-computed round-PSF ψ/φ.
        rescored = None
        if exposure_time_s > 0.0:
            rescored = _streak_rescore(
                work, noise_rms, times, x0, y0, vx, vy,
                psf_sigma_px=psf_sigma_px,
                exposure_s=exposure_time_s,
                min_length_px=streak_min_length_px,
                masks=masks,
            )
        if rescored is None:
            rescored = _score_trajectory(psi, phi, times, x0, y0, vx, vy)
        if rescored is None:
            continue
        snr, flux = rescored
        if snr < u_star:
            continue
        if velocity_min_px_s > 0.0 and math.hypot(vx, vy) < velocity_min_px_s:
            continue
        accepted.append(
            TrackCandidate(
                x_px=x0, y_px=y0, vx_px_s=vx, vy_px_s=vy,
                snr=snr, flux_e=flux, seed_snr=seed_snr,
            )
        )

    # Final NMS in *trajectory space*: a weaker candidate whose predicted
    # track passes within r_pos of a stronger one's at any frame time is a
    # trail-crossing ghost — its flux comes from the crossing segment of
    # the stronger track (measured: ghosts at ~10× lower SNR).  Known
    # limitation, documented: a genuinely distinct, much fainter object
    # whose path crosses a bright track is suppressed too; separating
    # crossing objects needs CLEAN-style flux subtraction (future work).
    accepted.sort(key=lambda c: c.snr, reverse=True)
    final: list[TrackCandidate] = []
    for cand in accepted:
        dup = any(_tracks_overlap(cand, k, times, r_pos) for k in final)
        if not dup:
            final.append(cand)

    return CoarseFineResult(
        candidates=tuple(final),
        u_star=u_star,
        n_windows=len(bounds),
        window_frames=win_frames,
        coarse_step_px_s=coarse_step_px_s,
        fine_step_px_s=fine_step,
    )


def _tracks_overlap(
    a: TrackCandidate, b: TrackCandidate, times: np.ndarray, r_pos: float
) -> bool:
    """True if the two predicted tracks come within r_pos at any frame time."""
    dx = (a.x_px - b.x_px) + (a.vx_px_s - b.vx_px_s) * times
    dy = (a.y_px - b.y_px) + (a.vy_px_s - b.vy_px_s) * times
    return bool(np.min(np.hypot(dx, dy)) < r_pos)


def _streak_rescore(
    work: list[np.ndarray],
    noise_rms: float | list[float],
    times: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    *,
    psf_sigma_px: float,
    exposure_s: float,
    min_length_px: float,
    masks: list[np.ndarray] | None = None,
) -> tuple[float, float] | None:
    """Re-score one refined candidate over the full pass with a streak kernel.

    Rebuilds ψ/φ from the (static-scene-subtracted) working frames with a
    kernel matched to the candidate's within-exposure trail |v|·exposure and
    scores its trajectory via :func:`_score_trajectory` (same statistic as
    the round-PSF gate path, so the u* comparison is fair).  Returns ``None``
    — fall back to the round-PSF full-pass re-score — when the trail is
    below ``min_length_px``.
    """
    trail = streak_length_px(math.hypot(vx, vy), exposure_s)
    if trail < max(0.0, min_length_px) or trail <= 0.0:
        return None
    kernel = streak_psf_kernel(psf_sigma_px, trail, math.atan2(vy, vx))
    psi, phi = make_psi_phi(work, noise_rms, kernel, masks)
    return _score_trajectory(psi, phi, times, x0, y0, vx, vy)


def _score_trajectory(
    psi: list[np.ndarray],
    phi: list[np.ndarray],
    times: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
) -> tuple[float, float] | None:
    """Full-pass calibrated score of one fixed trajectory on given ψ/φ.

    Shift-adds the per-frame likelihood images along x(t) = (x0, y0) + v·t,
    removes the full-map median (the Poisson-median DC pedestal — a global
    zero level, deliberately *not* a local one, so a candidate cannot profit
    from an under-estimated pedestal next to its own blob), and returns
    ``(snr, flux)`` at the candidate's reference-epoch pixel; a ±1 px
    neighbourhood max absorbs integer rounding.  This is the statistic the
    u* trials budget describes: one hypothesis, full frame, ~N(0,1) noise.
    """
    h, w = psi[0].shape
    psi_sum = np.zeros((h, w))
    phi_sum = np.zeros((h, w))
    for k in range(len(psi)):
        dx = int(round(vx * float(times[k])))
        dy = int(round(vy * float(times[k])))
        _add_shifted(psi_sum, psi[k], dy, dx)
        _add_shifted(phi_sum, phi[k], dy, dx)
    snr_map = _snr_from(psi_sum, phi_sum)
    snr_map = snr_map - float(np.median(snr_map))  # DC pedestal (global)
    xi, yi = int(round(x0)), int(round(y0))
    y_lo, y_hi = max(0, yi - 1), min(h, yi + 2)
    x_lo, x_hi = max(0, xi - 1), min(w, xi + 2)
    sub = snr_map[y_lo:y_hi, x_lo:x_hi]
    if sub.size == 0:
        return None
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    gy, gx = y_lo + iy, x_lo + ix
    snr = float(snr_map[gy, gx])
    flux = (
        float(psi_sum[gy, gx] / phi_sum[gy, gx])
        if phi_sum[gy, gx] > _PHI_EPS
        else 0.0
    )
    return snr, flux


def _u_star_fine(
    h: int,
    w: int,
    v_max: float,
    fine_step: float,
    margin: float,
    bound_scale: float = 0.0,
) -> float:
    """Trials-corrected threshold for the fine-equivalent full search.

    Bernstein/Poisson-aware since 2026-07-28
    (:func:`opta_pipeline.likelihood.poisson_tail_threshold`);
    ``bound_scale = 0`` reproduces the former Gaussian extreme-value
    bound √(2 ln N) + margin exactly.
    """
    n_axis = max(int(2.0 * v_max / max(fine_step, 1e-9)) + 1, 1)
    n_trials = max(h * w * n_axis * n_axis, 2)
    return poisson_tail_threshold(n_trials, bound_scale, margin)


def _pyramid_refine(
    psi: list[np.ndarray],
    phi: list[np.ndarray],
    times: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    v_unc: float,
    *,
    t_anchor: float,
    k_anchor: int,
    psf_fwhm: float,
    fine_step: float,
    win_frames: int,
    v_max: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Refine one candidate: double the window, halve the step, per level.

    Every level is a local (2·parent/child+1)²-hypothesis search over an
    ROI patch stack — criterion-matched by construction (step =
    PSF_fwhm / T_window at every level).  The candidate position stays
    anchored at its seed epoch ``t_anchor`` throughout (where it is best
    determined); the window grows around the seed frame ``k_anchor``, and
    only the returned state is converted to the global t = 0 epoch.
    Refined velocities are confined per-axis to the ±``v_max`` search box:
    the final u* gate budgets trials for the fine-equivalent grid *inside*
    the box, so letting the local search wander outside it (seed ± its
    uncertainty can reach v_max + half_width) both breaks that equivalence
    and hands noise extra out-of-contract hypotheses (measured on blank
    SMOKE scenes: every emitted false positive sat outside the box).
    Returns (x0, y0, vx, vy, snr, flux) at t = 0, or None if the
    candidate leaves the frame.
    """
    n = len(psi)
    h, w = psi[0].shape
    n_level = max(win_frames, 2)
    step = v_unc
    while True:
        n_level = min(n_level * 2, n)
        lo = max(0, k_anchor - n_level // 2)
        hi = min(n, lo + n_level)
        lo = max(0, hi - n_level)  # re-expand when clipped at the end
        t_span = float(times[hi - 1] - times[lo]) if hi - lo > 1 else 0.0
        step = max(fine_step, psf_fwhm / t_span if t_span > 0 else step)
        half = max(step, v_unc if n_level >= n else min(v_unc, step * 2.0))

        result = _local_search(
            psi, phi, times, lo, hi, x0, y0, vx, vy,
            t_anchor=t_anchor,
            half_width=half, step=step, psf_fwhm=psf_fwhm, v_max=v_max,
        )
        if result is None:
            return None
        x0, y0, vx, vy, snr, flux = result
        v_unc = step
        if hi - lo >= n and step <= fine_step * (1.0 + 1e-9):
            xg = x0 - vx * t_anchor
            yg = y0 - vy * t_anchor
            if not (0.0 <= xg < w and 0.0 <= yg < h):
                return None
            return xg, yg, vx, vy, snr, flux


def _local_search(
    psi: list[np.ndarray],
    phi: list[np.ndarray],
    times: np.ndarray,
    lo: int,
    hi: int,
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    *,
    t_anchor: float,
    half_width: float,
    step: float,
    psf_fwhm: float,
    v_max: float,
) -> tuple[float, float, float, float, float, float] | None:
    """ROI patch-stack search over v ∈ (vx0, vy0) ± half_width at ``step``.

    Positions are relative to the anchor epoch: x(t) = x0 + v·(t −
    t_anchor).  The ROI covers the velocity-mismatch drift
    (± half_width·|t − t_anchor|max) plus the position uncertainty
    (~PSF); patches are gathered per frame at the *hypothesis*
    trajectory, so the accumulator peak offset is directly the
    correction to (x0, y0) at the anchor epoch.  Hypotheses outside the
    per-axis ±``v_max`` search box are skipped (see ``_pyramid_refine``).
    """
    t_rel = times[lo:hi] - t_anchor
    t_max = float(np.max(np.abs(t_rel))) if hi > lo else 0.0
    pad = int(math.ceil(2.0 * psf_fwhm))
    r = pad + min(int(math.ceil(half_width * t_max)), 60)
    size = 2 * r + 1

    offsets = symmetric_velocity_axis(half_width, step)
    v_lim = v_max + 1e-9
    best: tuple[float, float, float, float, float, float] | None = None
    for dvx in offsets:
        for dvy in offsets:
            vx = vx0 + float(dvx)
            vy = vy0 + float(dvy)
            if abs(vx) > v_lim or abs(vy) > v_lim:
                continue
            psi_acc = np.zeros((size, size))
            phi_acc = np.zeros((size, size))
            for k in range(lo, hi):
                cx = int(round(x0 + vx * (times[k] - t_anchor)))
                cy = int(round(y0 + vy * (times[k] - t_anchor)))
                _add_patch(psi_acc, psi[k], cy - r, cx - r)
                _add_patch(phi_acc, phi[k], cy - r, cx - r)
            if phi_acc.max() <= _PHI_EPS:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(
                    phi_acc > _PHI_EPS, psi_acc / np.sqrt(phi_acc), 0.0
                )
            snr = snr - float(np.median(snr))
            iy, ix = np.unravel_index(int(np.argmax(snr)), snr.shape)
            val = float(snr[iy, ix])
            if best is None or val > best[4]:
                flux = float(
                    psi_acc[iy, ix] / phi_acc[iy, ix]
                    if phi_acc[iy, ix] > _PHI_EPS
                    else 0.0
                )
                best = (
                    x0 + (int(ix) - r),
                    y0 + (int(iy) - r),
                    vx,
                    vy,
                    val,
                    flux,
                )
    if best is None:
        return None
    return best


def _binned_snr_dispersion(kernel: np.ndarray, b: int) -> float:
    """Std of the naive b×b-binned SNR statistic under white noise.

    ψ pixels are correlated by the matched-filter kernel: for lag Δ the
    correlation is the normalized kernel autocorrelation c(Δ).  Summing a
    b×b bin gives var(Σψ) = φ·ΣΣ c(Δᵢⱼ) while the naive denominator uses
    Σφ = b²·φ, so the statistic's std is √(ΣΣ c / b²).  Exact for uniform
    variance away from edges (measured 1.92 vs computed 1.90 at b = 2,
    σ_psf = 1.5).
    """
    from scipy.signal import correlate2d

    auto = correlate2d(kernel, kernel, mode="full")
    c0 = float(auto[kernel.shape[0] - 1, kernel.shape[1] - 1])
    cy, cx = kernel.shape[0] - 1, kernel.shape[1] - 1
    total = 0.0
    for dy in range(-(b - 1), b):
        for dx in range(-(b - 1), b):
            n_pairs = (b - abs(dy)) * (b - abs(dx))
            total += n_pairs * float(auto[cy + dy, cx + dx]) / c0
    return math.sqrt(total / (b * b))


def _bin_sum(image: np.ndarray, b: int) -> np.ndarray:
    """b×b sum-binning (truncates edge rows/cols that do not fill a bin)."""
    h, w = image.shape
    hb, wb = h // b, w // b
    return (
        image[: hb * b, : wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3))
    )


def _add_patch(
    accumulator: np.ndarray, image: np.ndarray, top: int, left: int
) -> None:
    """accumulator += image[top:top+size, left:left+size], zero-padded."""
    size = accumulator.shape[0]
    h, w = image.shape
    src_y0, src_y1 = max(0, top), min(h, top + size)
    src_x0, src_x1 = max(0, left), min(w, left + size)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return
    dst_y0 = src_y0 - top
    dst_x0 = src_x0 - left
    accumulator[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] += image[src_y0:src_y1, src_x0:src_x1]
