"""Analytic blind-search sensitivity vs observation time (stack-rework).

Pure closed-form companions to the calibrated detection scheme in
:mod:`opta_pipeline.coarse_fine` and :mod:`opta_pipeline.pipeline`. Nothing
here re-derives physics: every formula delegates to (or mirrors, with a
test pinning the equivalence) the exact expression the pipeline applies,
so the figure cannot drift from the shipped thresholds.

The three quantities that decide whether a pass of duration T is blind-
detectable (coarse_fine module docstring):

* achieved full-pass matched-filter SNR — grows as √T
  (:func:`snr_full_pass`, anchored to a *measured* reference point per the
  numbers policy, never to a pinned constant);
* the trials-corrected final gate u*(T) — grows only logarithmically with
  T because the fine-equivalent velocity grid refines as PSF_fwhm / T
  (:func:`u_star_blind`, delegates to ``coarse_fine._u_star_fine``).
  Since 2026-07-28 the gate is the Bernstein/Poisson-aware bound
  (likelihood module docstring, "Detection threshold"): it takes the
  bound scale ε — compute it with :func:`bound_scale` from the scene's
  σ/frame count/quantisation, mirroring what the pipeline estimates from
  its own frames; ε = 0 is the pre-2026-07-28 Gaussian bound;
* the stage-1 seeding floor — a candidate must reach the pre-threshold
  inside one seed window T₀, so the required *full-pass* SNR grows as
  √(T/T₀) (:func:`seeding_floor`).  The auto T₀
  (:func:`seed_window_auto_s`) is depth-matched by construction since
  2026-07-28: given T and u*(T) it includes the T·(prethreshold/u*)²
  term that inverts seeding_floor(T, T₀) ≤ u*(T), so at auto depth the
  floor never exceeds the gate (the old fixed-T₀ rule let it cancel the
  √T signal gain).

The blind detection limit is the max of the last two
(:func:`blind_limit`); ephemeris-cued mode replaces both with the
T-independent :func:`u_star_prior` floor of its small hypothesis box.

Requirement traceability: OpTA.NOD.DET (mv ≤ 13.0, stacked SNR ≥ 5).
"""

from __future__ import annotations

import math

from opta_pipeline.coarse_fine import _u_star_fine
from opta_pipeline.likelihood import (
    DEFAULT_PSF_SIGMA_PX,
    bound_scale_epsilon,
    gaussian_psf_kernel,
    poisson_tail_threshold,
)

__all__ = [
    "DEFAULT_PSF_SIGMA_PX",
    "PSF_FWHM_FACTOR",
    "blind_limit",
    "bound_scale",
    "crossing_time_s",
    "seed_window_auto_s",
    "seeding_floor",
    "snr_full_pass",
    "u_star_blind",
    "u_star_prior",
]

# Gaussian FWHM = 2.355 σ — same constant coarse_fine.py and likelihood.py
# use to convert psf_sigma_px into the velocity-sampling criterion.
PSF_FWHM_FACTOR = 2.355


def _mag_scale(mag: float | None, mag_ref: float | None) -> float:
    """Pogson flux ratio 10^(−0.4·(mag − mag_ref)); 1.0 when mag is None."""
    if mag is None:
        return 1.0
    if mag_ref is None:
        raise ValueError("mag_ref is required when mag is given")
    return 10.0 ** (-0.4 * (mag - mag_ref))


def snr_full_pass(
    t_s: float,
    *,
    snr_ref: float,
    t_ref_s: float,
    mag: float | None = None,
    mag_ref: float | None = None,
) -> float:
    """Full-pass matched-filter SNR at duration ``t_s``.

    Scaled from a measured anchor ``snr_ref`` at ``(t_ref_s, mag_ref)``:
    signal integrates linearly and noise as √N over a background-limited
    stack, so SNR ∝ √T (stack.py / fig_23); flux scales as the Pogson
    ratio between magnitudes.  Valid while the scene stays background- or
    read-noise-limited (sky ≈ 0.3 e⁻/px at the CI geometry — it does).
    """
    if t_s <= 0.0 or t_ref_s <= 0.0:
        raise ValueError("durations must be > 0")
    return snr_ref * _mag_scale(mag, mag_ref) * math.sqrt(t_s / t_ref_s)


def bound_scale(
    *,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    noise_rms: float = 1.0,
    n_frames: int = 1,
    quant_step_e: float = 1.0,
) -> float:
    """Bernstein bound scale ε for a scene of N frames at scalar σ.

    Analytic mirror of what the pipeline estimates from its own batch
    (:func:`opta_pipeline.likelihood.bound_scale_epsilon` with the same
    kernel construction): ε = (P_max/‖P‖₂)·s/(σ√N).  Use it to feed
    :func:`u_star_blind` / :func:`u_star_prior` / :func:`blind_limit`
    with the shipped contract's ε instead of the Gaussian-limit 0.
    """
    return bound_scale_epsilon(
        gaussian_psf_kernel(psf_sigma_px),
        noise_rms,
        n_frames=n_frames,
        quant_step_e=quant_step_e,
    )


def u_star_blind(
    t_s: float,
    *,
    height_px: int,
    width_px: int,
    v_max_px_s: float,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    far_margin_sigma: float = 0.5,
    bound_scale_eps: float = 0.0,
) -> float:
    """Blind final-gate threshold u* for a pass of duration ``t_s``.

    The fine-equivalent velocity step the coarse-to-fine scheme terminates
    at is PSF_fwhm / T (coarse_fine.py, assessment F1), so the trials count
    — and with it u* — grows with pass duration.  Delegates to
    ``coarse_fine._u_star_fine`` so this is the applied threshold, not a
    reimplementation.  ``bound_scale_eps`` is the Bernstein bound scale of
    the shipped contract (compute via :func:`bound_scale`); 0 gives the
    Gaussian limit.
    """
    if t_s <= 0.0:
        raise ValueError("t_s must be > 0")
    fine_step = PSF_FWHM_FACTOR * psf_sigma_px / t_s
    return _u_star_fine(
        height_px,
        width_px,
        v_max_px_s,
        fine_step,
        far_margin_sigma,
        bound_scale_eps,
    )


def u_star_prior(
    *,
    height_px: int,
    width_px: int,
    n_hyp: int,
    far_margin_sigma: float = 0.5,
    bound_scale_eps: float = 0.0,
) -> float:
    """Ephemeris-cued threshold: u* of a fixed small hypothesis box.

    Mirrors the single-stage gate in ``pipeline._stack_and_detect``
    (u* = poisson_tail_threshold(N_pix·N_hyp, ε) + margin, the Bernstein
    contract since 2026-07-28) for a prior-centred grid of ``n_hyp``
    velocity hypotheses.  Independent of pass duration — the cued box
    does not refine with T.  ``bound_scale_eps = 0`` is the Gaussian
    limit √(2 ln(N_pix·N_hyp)) + margin.
    """
    n_trials = max(int(height_px) * int(width_px) * int(n_hyp), 2)
    return poisson_tail_threshold(n_trials, bound_scale_eps, far_margin_sigma)


def seed_window_auto_s(
    *,
    coarse_step_px_s: float,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    fps: float = 25.0,
    t_span_s: float | None = None,
    u_star: float | None = None,
    prethreshold_sigma: float = 4.0,
) -> float:
    """Auto seed-window duration T₀ used when ``seed_window_s = 0``.

    Mirrors coarse_fine.blind_coarse_fine_search (rule since 2026-07-28):
    T₀ = max(PSF_fwhm / coarse_step, 4 frames, T·(prethreshold/u*)²).
    The last term inverts :func:`seeding_floor`\\ (T, T₀) ≤ u*(T), so the
    auto depth cannot truncate the u*-limited sensitivity — pass the pass
    duration ``t_span_s`` and the gate ``u_star`` (from
    :func:`u_star_blind` at the same T) to include it; omitting either
    reproduces the pre-2026-07-28 depth-agnostic rule.
    """
    if coarse_step_px_s <= 0.0 or fps <= 0.0:
        raise ValueError("coarse_step_px_s and fps must be > 0")
    base = max(PSF_FWHM_FACTOR * psf_sigma_px / coarse_step_px_s, 4.0 / fps)
    if t_span_s is None or u_star is None or u_star <= 0.0:
        return base
    return max(base, t_span_s * (prethreshold_sigma / u_star) ** 2)


def seeding_floor(
    t_s: float,
    *,
    t0_s: float,
    prethreshold_sigma: float = 4.0,
) -> float:
    """Required full-pass SNR for stage-1 seeding at window duration T₀.

    A candidate is only refined if its per-window SNR reaches the
    pre-threshold; the window sees T₀/T of the pass, so the full-pass SNR
    must exceed prethreshold · √(T/T₀) (coarse_fine module docstring).
    This is the term that cancels the √T stacking gain at fixed T₀.
    """
    if t_s <= 0.0 or t0_s <= 0.0:
        raise ValueError("durations must be > 0")
    return prethreshold_sigma * math.sqrt(t_s / t0_s)


def blind_limit(
    t_s: float,
    *,
    height_px: int,
    width_px: int,
    v_max_px_s: float,
    t0_s: float,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    far_margin_sigma: float = 0.5,
    prethreshold_sigma: float = 4.0,
    bound_scale_eps: float = 0.0,
) -> float:
    """Limiting full-pass SNR for blind coarse-to-fine detection.

    SNR_lim ≈ max(u*(T), prethreshold · √(T/T₀)) — the documented
    sensitivity trade of the two-stage scheme.  At the auto
    (depth-matched) T₀ the second term never exceeds the first by
    construction; a fixed explicit ``t0_s`` can still truncate depth.
    """
    return max(
        u_star_blind(
            t_s,
            height_px=height_px,
            width_px=width_px,
            v_max_px_s=v_max_px_s,
            psf_sigma_px=psf_sigma_px,
            far_margin_sigma=far_margin_sigma,
            bound_scale_eps=bound_scale_eps,
        ),
        seeding_floor(t_s, t0_s=t0_s, prethreshold_sigma=prethreshold_sigma),
    )


def crossing_time_s(
    *,
    snr_ref: float,
    t_ref_s: float,
    threshold: float,
    mag: float | None = None,
    mag_ref: float | None = None,
) -> float:
    """Smallest duration whose full-pass SNR reaches a fixed threshold.

    Inverts :func:`snr_full_pass` for a T-independent threshold (the
    ephemeris-prior floor): t = t_ref · (threshold / SNR_ref)².  Not valid
    against :func:`blind_limit`, whose seeding term grows with T too.
    """
    if snr_ref <= 0.0 or threshold <= 0.0:
        raise ValueError("snr_ref and threshold must be > 0")
    snr0 = snr_ref * _mag_scale(mag, mag_ref)
    return t_ref_s * (threshold / snr0) ** 2
